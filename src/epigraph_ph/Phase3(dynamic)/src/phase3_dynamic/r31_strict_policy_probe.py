from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_ordinal, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import R10_COMPARABLE_METRICS, _finite_float, _generated_at
from .r28_r10_contract_lineage_audit import _collect_r10_records
from .r29_strict_ledger_matched_r10_gate import _is_program_lineage
from .runtime import ensure_dir, write_json


R31_SCHEMA_VERSION = "phase3_dynamic.r31_strict_policy_probe.v1"
R31_RUN_ID = "p3d-r31-strict-policy-probe-20260502-s00"
R31_POLICY_IDS: tuple[str, ...] = (
    "carry_forward",
    "positive_velocity",
    "median_velocity",
    "recent_log_linear",
)
R31_PROBE_METRICS: tuple[str, ...] = ("alive_on_art", "new_diagnosed_cases_period")


def _history_by_metric(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, int, float]]]:
    history: dict[str, list[tuple[int, int, float]]] = {metric_name: [] for metric_name in R10_COMPARABLE_METRICS}
    for row in rows:
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        for metric_name in R10_COMPARABLE_METRICS:
            value = _finite_float(row.get(metric_name))
            provenance = dict(row.get("metric_provenance") or {}).get(metric_name)
            if value is None or not provenance:
                continue
            history[metric_name].append((quarter_ordinal(quarter), quarter_year(quarter), float(value)))
    for metric_name in history:
        history[metric_name].sort()
    return history


def _positive_velocity(history: list[tuple[int, int, float]]) -> float | None:
    diffs = [
        (b[2] - a[2]) / (b[0] - a[0])
        for a, b in zip(history, history[1:])
        if b[0] > a[0] and b[2] >= a[2]
    ]
    if not diffs:
        return None
    return float(np.median(np.asarray(diffs[-min(8, len(diffs)) :], dtype=np.float64)))


def _median_velocity(history: list[tuple[int, int, float]]) -> float | None:
    diffs = [
        (b[2] - a[2]) / (b[0] - a[0])
        for a, b in zip(history, history[1:])
        if b[0] > a[0]
    ]
    if not diffs:
        return None
    return float(np.median(np.asarray(diffs[-min(8, len(diffs)) :], dtype=np.float64)))


def _policy_prediction(
    *,
    metric_name: str,
    train_end_year: int,
    target_quarter: str,
    policy_id: str,
    history_by_metric: dict[str, list[tuple[int, int, float]]],
) -> float | None:
    history = [
        row
        for row in list(history_by_metric.get(metric_name) or [])
        if int(row[1]) <= int(train_end_year)
    ]
    if not history:
        return None
    target_ordinal = quarter_ordinal(target_quarter)
    last_ordinal, _last_year, last_value = max(history, key=lambda row: row[0])
    if policy_id == "carry_forward":
        return float(last_value)
    if policy_id == "positive_velocity":
        velocity = _positive_velocity(history)
        if velocity is None:
            return float(last_value)
        return float(max(last_value + velocity * (target_ordinal - last_ordinal), 0.0))
    if policy_id == "median_velocity":
        velocity = _median_velocity(history)
        if velocity is None:
            return float(last_value)
        return float(max(last_value + velocity * (target_ordinal - last_ordinal), 0.0))
    if policy_id == "recent_log_linear":
        if len(history) < 2:
            return float(last_value)
        recent = history[-min(6, len(history)) :]
        x = np.asarray([row[0] for row in recent], dtype=np.float64)
        y = np.log1p(np.asarray([max(row[2], 0.0) for row in recent], dtype=np.float64))
        slope, intercept = np.polyfit(x - x[-1], y, 1)
        return float(max(math.expm1(float(intercept + slope * (target_ordinal - x[-1]))), 0.0))
    raise ValueError(f"Unknown R31 policy: {policy_id}")


def _scope_records(records: list[dict[str, Any]], *, scope: str, metric_name: str, horizon: int) -> list[dict[str, Any]]:
    output = []
    for record in records:
        if int(record.get("horizon_years") or 0) != int(horizon):
            continue
        if str(record.get("metric_name") or "") != metric_name:
            continue
        if bool(record.get("unknown_provenance")):
            continue
        if scope == "program_mapped" and not _is_program_lineage(record):
            continue
        output.append(record)
    return output


def _score_policy_records(
    *,
    records: list[dict[str, Any]],
    policy_id: str,
    history_by_metric: dict[str, list[tuple[int, int, float]]],
) -> dict[str, Any]:
    policy_errors: list[float] = []
    r10_errors: list[float] = []
    for record in records:
        prediction = _policy_prediction(
            metric_name=str(record["metric_name"]),
            train_end_year=int(record["train_end_year"]),
            target_quarter=str(record["quarter"]),
            policy_id=policy_id,
            history_by_metric=history_by_metric,
        )
        if prediction is None:
            continue
        target = float(record["target_value"])
        scale = max(float(record["scale"]), float(np.finfo(np.float32).eps))
        policy_errors.append(abs(float(prediction) - target) / scale)
        r10_errors.append(float(record["r10_norm_error"]))
    if not policy_errors:
        return {
            "entry_count": 0,
            "policy_mean_mae": None,
            "strict_r10_mean_mae": None,
            "policy_minus_strict_r10": None,
            "status": "not_evaluable",
        }
    policy_mean = float(np.mean(np.asarray(policy_errors, dtype=np.float64)))
    r10_mean = float(np.mean(np.asarray(r10_errors, dtype=np.float64)))
    return {
        "entry_count": len(policy_errors),
        "policy_mean_mae": policy_mean,
        "strict_r10_mean_mae": r10_mean,
        "policy_minus_strict_r10": float(policy_mean - r10_mean),
        "policy_worst_mae": float(np.max(np.asarray(policy_errors, dtype=np.float64))),
        "strict_r10_worst_mae": float(np.max(np.asarray(r10_errors, dtype=np.float64))),
        "status": "pass" if policy_mean < r10_mean else "fail",
    }


def _probe_rows(
    *,
    records: list[dict[str, Any]],
    history_by_metric: dict[str, list[tuple[int, int, float]]],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in ("all_mapped", "program_mapped"):
        for horizon in horizons:
            for metric_name in R31_PROBE_METRICS:
                scoped_records = _scope_records(records, scope=scope, metric_name=metric_name, horizon=int(horizon))
                for policy_id in R31_POLICY_IDS:
                    score = _score_policy_records(
                        records=scoped_records,
                        policy_id=policy_id,
                        history_by_metric=history_by_metric,
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "horizon_years": int(horizon),
                            "metric_name": metric_name,
                            "policy_id": policy_id,
                            **score,
                        }
                    )
    return rows


def _best_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _finite_float(row.get("policy_mean_mae")) is None:
            continue
        grouped[(str(row.get("scope") or ""), int(row.get("horizon_years") or 0), str(row.get("metric_name") or ""))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        output.append(
            min(
                values,
                key=lambda row: (
                    float(row.get("policy_mean_mae") or float("inf")),
                    str(row.get("policy_id") or ""),
                ),
            )
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R31 Strict Policy Probe",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Best Train-Only Policy By Cell",
        "",
        "| Scope | Horizon | Metric | Policy | Policy MAE | Strict R10 | Delta | Status |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in list(report.get("best_policy_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | `{row.get('metric_name')}` | "
            f"`{row.get('policy_id')}` | {_format_float(row.get('policy_mean_mae'))} | "
            f"{_format_float(row.get('strict_r10_mean_mae'))} | "
            f"{_format_float(row.get('policy_minus_strict_r10'))} | `{row.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R31 is a train-only policy probe, not a promoted model.",
            "- It tests simple non-handwritten time-series policies on the same strict mapped R10 target rows.",
            "- The purpose is to separate a diagnosis-flow repair opportunity from the harder ART trajectory blocker.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    selected = [
        row
        for row in rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    labels = [f"h{row.get('horizon_years')} {row.get('metric_name')}" for row in selected]
    delta = np.asarray([float(row.get("policy_minus_strict_r10") or np.nan) for row in selected], dtype=np.float64)
    x = np.arange(len(selected), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle("R31 Best Train-Only Policy vs Strict Program R10", fontsize=14, fontweight="bold")
    ax.bar(x, delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in delta])
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("best policy minus strict R10")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r31_strict_policy_probe(
    *,
    run_id: str = R31_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    source_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    records, _reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    history = _history_by_metric(source_rows)
    probe_rows = _probe_rows(records=records, history_by_metric=history, horizons=horizons)
    best_rows = _best_policy_rows(probe_rows)
    program_long = [
        row
        for row in best_rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    blockers = [
        f"h{row.get('horizon_years')}_{row.get('metric_name')}_best_train_policy_not_better_than_strict_r10"
        for row in program_long
        if str(row.get("status") or "") != "pass"
    ]
    verdict = (
        "R31 finds a train-only policy that beats strict program R10 for every probed long-horizon program cell."
        if not blockers
        else "R31 confirms diagnosis-flow is repairable by train-only velocity, but ART remains worse than strict program R10."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R31_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "policy_ids": list(R31_POLICY_IDS),
        "probe_metrics": list(R31_PROBE_METRICS),
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "verdict": verdict,
        "probe_rows": probe_rows,
        "best_policy_rows": best_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r31_strict_policy_probe.json"
    md_path = analysis_dir / "r31_strict_policy_probe.md"
    csv_path = analysis_dir / "r31_strict_policy_probe_rows.csv"
    best_csv_path = analysis_dir / "r31_best_policy_rows.csv"
    dashboard_path = analysis_dir / "r31_strict_policy_probe_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "probe_csv": csv_path.as_posix(),
        "best_policy_csv": best_csv_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, probe_rows)
    _write_csv(best_csv_path, best_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, best_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R31 strict mapped policy probe.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R31_RUN_ID)
    args = parser.parse_args()
    run_r31_strict_policy_probe(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
