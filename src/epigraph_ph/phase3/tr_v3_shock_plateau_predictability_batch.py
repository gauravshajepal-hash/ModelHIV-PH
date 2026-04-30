import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_grasp_falsification_batch as grasp
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, quarter_gap
from epigraph_ph.runtime import ensure_dir, write_json


def _result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _brier(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean(np.asarray([(float(row[key]) - float(row["label"])) ** 2 for row in rows], dtype=np.float64)))


def _recall_at_k(rows: list[dict[str, Any]], key: str) -> float | None:
    event_count = int(sum(int(row["label"]) for row in rows))
    if event_count <= 0:
        return None
    ranked = sorted(list(rows), key=lambda row: float(row[key]), reverse=True)
    top = ranked[:event_count]
    return float(sum(int(row["label"]) for row in top)) / float(event_count)


def _fmt(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _lagged_shared_shock_predictability(payload: dict[str, Any], *, contract_name: str) -> dict[str, Any]:
    ordered = list(payload.get("ordered_quarters") or [])
    if len(ordered) < 6:
        return {"contract": contract_name, "status": "insufficient_history", "event_count": 0, "rows": []}
    rows: list[dict[str, Any]] = []
    history_x: list[list[float]] = []
    history_y: list[float] = []
    ridge = 0.5
    for idx in range(1, len(ordered)):
        prev = dict(ordered[idx - 1])
        current = dict(ordered[idx])
        prev_metrics = dict(prev.get("metrics") or {})
        features = [
            1.0,
            1.0 if bool(prev.get("shared_burst")) else 0.0,
            1.0 if bool(prev.get("diagnosed_only_burst")) else 0.0,
            float(prev.get("mean_abs_z") or 0.0),
            float(len(list(prev.get("burst_metrics") or []))),
            float(dict(prev_metrics.get("diagnosed_plhiv") or {}).get("z_score") or 0.0),
            float(dict(prev_metrics.get("alive_on_art") or {}).get("z_score") or 0.0),
            float(dict(prev_metrics.get("new_diagnosed_cases_period") or {}).get("z_score") or 0.0),
        ]
        label = 1.0 if bool(current.get("shared_burst")) else 0.0
        prevalence = float(np.mean(np.asarray(history_y, dtype=np.float64))) if history_y else 0.0
        persistence = 1.0 if bool(prev.get("shared_burst")) else 0.0
        if len(history_y) >= 4 and sum(history_y) >= 1.0:
            x = np.asarray(history_x, dtype=np.float64)
            y = np.asarray(history_y, dtype=np.float64)
            penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            candidate = float(np.clip(float(np.dot(np.asarray(features, dtype=np.float64), beta)), 0.0, 1.0))
        else:
            candidate = float(prevalence)
        rows.append(
            {
                "quarter": str(current["quarter"]),
                "label": float(label),
                "candidate_probability": float(candidate),
                "prevalence_probability": float(prevalence),
                "persistence_probability": float(persistence),
            }
        )
        history_x.append(list(features))
        history_y.append(float(label))
    candidate_brier = _brier(rows, "candidate_probability")
    prevalence_brier = _brier(rows, "prevalence_probability")
    persistence_brier = _brier(rows, "persistence_probability")
    event_count = int(sum(int(row["label"]) for row in rows))
    if event_count < 3:
        status = "insufficient_events"
    elif candidate_brier is not None and prevalence_brier is not None and persistence_brier is not None and candidate_brier + 1e-6 < min(prevalence_brier, persistence_brier):
        status = "incremental_predictive_signal"
    else:
        status = "no_incremental_predictive_signal"
    return {
        "contract": contract_name,
        "status": status,
        "event_count": int(event_count),
        "rows": rows,
        "metrics": {
            "candidate_brier": candidate_brier,
            "prevalence_brier": prevalence_brier,
            "persistence_brier": persistence_brier,
            "candidate_recall_at_k": _recall_at_k(rows, "candidate_probability"),
            "persistence_recall_at_k": _recall_at_k(rows, "persistence_probability"),
        },
    }


def _plateau_prediction_rows(
    observation_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    allowed_tiers: set[str],
) -> tuple[list[dict[str, Any]], float]:
    supported_rows = [
        row
        for row in sorted(list(observation_rows), key=lambda item: suite.quarter_sort_key(str(item["quarter"])))
        if row.get(metric_name) is not None and suite._metric_tier(row, metric_name) in allowed_tiers
    ]
    if len(supported_rows) < 4:
        return [], 0.0
    quarter_values = [(str(row["quarter"]), float(row[metric_name])) for row in supported_rows]
    diffs: list[float] = []
    adjacency_flags: list[bool] = []
    for idx in range(1, len(quarter_values)):
        prev_quarter, prev_value = quarter_values[idx - 1]
        quarter, value = quarter_values[idx]
        if quarter_gap(prev_quarter, quarter) != 1:
            diffs.append(float("nan"))
            adjacency_flags.append(False)
            continue
        diffs.append(abs(float(value) - float(prev_value)))
        adjacency_flags.append(True)
    nonzero = [float(value) for value, ok in zip(diffs, adjacency_flags, strict=False) if ok and value > 1e-9]
    threshold = float(np.median(np.asarray(nonzero, dtype=np.float64))) * 0.25 if nonzero else 0.0
    plateau_active: list[bool] = []
    for diff, ok in zip(diffs, adjacency_flags, strict=False):
        plateau_active.append(bool(ok and diff <= threshold))
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    current_run = 0
    for idx in range(len(plateau_active) - 1):
        current_active = bool(plateau_active[idx])
        if current_active:
            current_run += 1
        else:
            current_run = 0
        next_active = 1.0 if bool(plateau_active[idx + 1]) else 0.0
        prevalence = float(np.mean(np.asarray([float(row["label"]) for row in history], dtype=np.float64))) if history else 0.0
        persistence = 1.0 if current_active else 0.0
        same_state = [row for row in history if bool(row["current_active"]) == current_active]
        same_bucket = [row for row in same_state if int(row["run_bucket"]) == min(int(current_run), 4)]
        if len(same_bucket) >= 2:
            candidate = float(np.mean(np.asarray([float(row["label"]) for row in same_bucket], dtype=np.float64)))
        elif len(same_state) >= 2:
            candidate = float(np.mean(np.asarray([float(row["label"]) for row in same_state], dtype=np.float64)))
        else:
            candidate = float(prevalence)
        row = {
            "quarter": str(quarter_values[idx + 1][0]),
            "label": float(next_active),
            "candidate_probability": float(candidate),
            "prevalence_probability": float(prevalence),
            "persistence_probability": float(persistence),
            "current_active": bool(current_active),
            "run_bucket": int(min(int(current_run), 4)),
        }
        rows.append(row)
        history.append(dict(row))
    return rows, float(threshold)


def _plateau_predictability_contract_payload(
    archive_run_id: str,
    *,
    contract_name: str,
) -> dict[str, Any]:
    observation_rows, allowed_tiers, _, _ = grasp._contract_setup(archive_run_id, contract_name)
    metrics: dict[str, Any] = {}
    for metric_name in PRIMARY_METRICS:
        rows, threshold = _plateau_prediction_rows(observation_rows, metric_name=metric_name, allowed_tiers=allowed_tiers)
        event_count = int(sum(int(row["label"]) for row in rows))
        candidate_brier = _brier(rows, "candidate_probability")
        prevalence_brier = _brier(rows, "prevalence_probability")
        persistence_brier = _brier(rows, "persistence_probability")
        if event_count < 3:
            status = "insufficient_events"
        elif candidate_brier is not None and prevalence_brier is not None and persistence_brier is not None and candidate_brier + 1e-6 < min(prevalence_brier, persistence_brier):
            status = "incremental_plateau_signal"
        else:
            status = "no_incremental_plateau_signal"
        metrics[metric_name] = {
            "status": status,
            "threshold": float(threshold),
            "event_count": int(event_count),
            "candidate_brier": candidate_brier,
            "prevalence_brier": prevalence_brier,
            "persistence_brier": persistence_brier,
            "candidate_recall_at_k": _recall_at_k(rows, "candidate_probability"),
            "persistence_recall_at_k": _recall_at_k(rows, "persistence_probability"),
            "rows": rows,
        }
    signal_metrics = [metric for metric, payload in metrics.items() if str(payload["status"]) == "incremental_plateau_signal"]
    return {
        "contract": contract_name,
        "metrics": metrics,
        "decision": {
            "status": "incremental_plateau_signal_present" if signal_metrics else "no_incremental_plateau_signal",
            "signal_metrics": signal_metrics,
        },
    }


def _save_shock_graph(payload: dict[str, Any], path: Path, *, title: str) -> None:
    rows = list(payload.get("rows") or [])
    if not rows:
        suite._plot_placeholder(path, title=title, body="No shock rows.")
        return
    labels = [str(row["quarter"]) for row in rows]
    candidate = [float(row["candidate_probability"]) for row in rows]
    prevalence = [float(row["prevalence_probability"]) for row in rows]
    persistence = [float(row["persistence_probability"]) for row in rows]
    events = [float(row["label"]) for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, candidate, label="Candidate", linewidth=2.0)
    ax.plot(x, prevalence, label="Prevalence", linestyle="--")
    ax.plot(x, persistence, label="Persistence", linestyle=":")
    ax.scatter(x, events, label="Observed burst", marker="x", color="black")
    step = max(len(labels) // 8, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probability / event")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_plateau_graph(payload: dict[str, Any], path: Path, *, title: str) -> None:
    metrics = dict(payload.get("metrics") or {})
    if not metrics:
        suite._plot_placeholder(path, title=title, body="No plateau metrics.")
        return
    names = list(metrics.keys())
    candidate = [float(dict(metrics[name]).get("candidate_brier") or 0.0) for name in names]
    prevalence = [float(dict(metrics[name]).get("prevalence_brier") or 0.0) for name in names]
    persistence = [float(dict(metrics[name]).get("persistence_brier") or 0.0) for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.22, prevalence, width=0.22, label="Prevalence")
    ax.bar(x, persistence, width=0.22, label="Persistence")
    ax.bar(x + 0.22, candidate, width=0.22, label="Candidate")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Brier score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    shock = dict(payload.get("shock_predictability") or {})
    plateau = dict(payload.get("plateau_predictability") or {})
    lines = [
        "# TR-V3 Shock And Plateau Predictability Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Exact experiment: `{payload['exact_experiment_id']}`",
        f"- Dense experiment: `{payload['dense_experiment_id']}`",
        "",
        "## Shock Predictability",
        "",
        "| Contract | Status | Event count | Candidate Brier | Prevalence Brier | Persistence Brier | Candidate recall@K | Persistence recall@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for contract_name in ("exact_only", "purged_dense"):
        row = dict(shock.get(contract_name) or {})
        metrics = dict(row.get("metrics") or {})
        lines.append(
            f"| {contract_name} | {row.get('status', '')} | {int(row.get('event_count') or 0)} | "
            f"{_fmt(metrics.get('candidate_brier'), digits=6)} | "
            f"{_fmt(metrics.get('prevalence_brier'), digits=6)} | "
            f"{_fmt(metrics.get('persistence_brier'), digits=6)} | "
            f"{_fmt(metrics.get('candidate_recall_at_k'), digits=3)} | "
            f"{_fmt(metrics.get('persistence_recall_at_k'), digits=3)} |"
        )
    lines.extend(
        [
            "",
            "## Plateau Predictability",
            "",
            "| Contract | Metric | Status | Event count | Candidate Brier | Prevalence Brier | Persistence Brier |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for contract_name in ("exact_only", "purged_dense"):
        contract_payload = dict(plateau.get(contract_name) or {})
        for metric_name, metric_payload in dict(contract_payload.get("metrics") or {}).items():
            lines.append(
                f"| {contract_name} | `{metric_name}` | {metric_payload.get('status', '')} | {int(metric_payload.get('event_count') or 0)} | "
                f"{_fmt(metric_payload.get('candidate_brier'), digits=6)} | "
                f"{_fmt(metric_payload.get('prevalence_brier'), digits=6)} | "
                f"{_fmt(metric_payload.get('persistence_brier'), digits=6)} |"
            )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Shock decision: `{payload['decision']['shock']}`",
            f"- Plateau decision: `{payload['decision']['plateau']}`",
            f"- Why: {payload['decision']['why']}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_tr_v3_shock_plateau_predictability_batch(
    *,
    run_id: str,
    archive_run_id: str | None = None,
    exact_experiment_id: str = "EXP-R10-M1-F1-C1",
    dense_experiment_id: str = "EXP-R10-DENSE-M1-C1-H1",
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or suite._latest_standard_archive_run())
    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")
    exact_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="exact_only",
        experiment_ids=[str(exact_experiment_id)],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_ids=[str(dense_experiment_id)],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    exact_result = _result_map(exact_payload)[str(exact_experiment_id)]
    dense_result = _result_map(dense_payload)[str(dense_experiment_id)]

    exact_concordance = grasp._shock_concordance_contract_payload(exact_result, contract_name="exact_only")
    dense_concordance = grasp._shock_concordance_contract_payload(dense_result, contract_name="purged_dense")
    shock_exact = _lagged_shared_shock_predictability(exact_concordance, contract_name="exact_only")
    shock_dense = _lagged_shared_shock_predictability(dense_concordance, contract_name="purged_dense")

    plateau_exact = _plateau_predictability_contract_payload(archive_run, contract_name="exact_only")
    plateau_dense = _plateau_predictability_contract_payload(archive_run, contract_name="purged_dense")

    if str(shock_dense.get("status")) == "incremental_predictive_signal":
        shock_decision = "pursue_minimal_shock_sidecar"
    else:
        shock_decision = "do_not_promote_shock_model"
    if str(dict(plateau_dense.get("decision") or {}).get("status")) == "incremental_plateau_signal_present":
        plateau_decision = "pursue_plateau_sidecar"
    else:
        plateau_decision = "do_not_promote_plateau_model"

    decision = {
        "shock": str(shock_decision),
        "plateau": str(plateau_decision),
        "why": (
            "Promote only the sidecar whose predictive probe beats both prevalence and persistence baselines. "
            "Do not claim exogenous shock predictability without incremental signal."
        ),
    }

    _save_shock_graph(shock_exact, analysis_dir / "shock_predictability_exact.png", title=f"Shock predictability ({exact_experiment_id}, exact_only)")
    _save_shock_graph(shock_dense, analysis_dir / "shock_predictability_dense.png", title=f"Shock predictability ({dense_experiment_id}, purged_dense)")
    _save_plateau_graph(plateau_exact, analysis_dir / "plateau_predictability_exact.png", title="Plateau predictability (exact_only)")
    _save_plateau_graph(plateau_dense, analysis_dir / "plateau_predictability_dense.png", title="Plateau predictability (purged_dense)")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "exact_experiment_id": str(exact_experiment_id),
        "dense_experiment_id": str(dense_experiment_id),
        "shock_predictability": {"exact_only": shock_exact, "purged_dense": shock_dense},
        "plateau_predictability": {"exact_only": plateau_exact, "purged_dense": plateau_dense},
        "decision": decision,
    }
    write_json(analysis_dir / "tr_v3_shock_plateau_predictability_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_shock_plateau_predictability_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TR-V3 shock and plateau predictability probes.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--exact-experiment-id", default="EXP-R10-M1-F1-C1")
    parser.add_argument("--dense-experiment-id", default="EXP-R10-DENSE-M1-C1-H1")
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    args = parser.parse_args()
    run_tr_v3_shock_plateau_predictability_batch(
        run_id=str(args.run_id),
        archive_run_id=args.archive_run_id,
        exact_experiment_id=str(args.exact_experiment_id),
        dense_experiment_id=str(args.dense_experiment_id),
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
    )


if __name__ == "__main__":
    main()
