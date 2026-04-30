import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.tr_v3_05_autoresearch import build_annual_anchor_rows
from epigraph_ph.runtime import ensure_dir, write_json


DENSE_EXPERIMENT_IDS: list[str] = [
    "EXP-R10-DENSE-M1-H1",
    "EXP-R10-DENSE-M1-B1-H1",
    "EXP-R10-DENSE-M1-C1-H1",
    "EXP-R1",
]


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _result_row(result: dict[str, Any], *, contract_name: str) -> dict[str, Any]:
    summary = dict(result.get("quarterly_summary") or {})
    endpoint_audit = dict(summary.get("endpoint_audit_summary") or {})
    candidate_audit = dict(endpoint_audit.get("candidate") or {})
    by_metric = dict(candidate_audit.get("by_metric") or {})
    diagnosed = dict(by_metric.get("diagnosed_plhiv") or {})
    art = dict(by_metric.get("alive_on_art") or {})
    flow = dict(by_metric.get("new_diagnosed_cases_period") or {})
    return {
        "contract": str(contract_name),
        "experiment_id": str(result["experiment_id"]),
        "quarterly_mean_mae": float(summary.get("candidate_mean_mae") or 0.0),
        "quarterly_baseline_mae": float(summary.get("carry_forward_mean_mae") or 0.0),
        "diagnosed_raw_mae": float(diagnosed.get("raw_mae") or 0.0),
        "art_raw_mae": float(art.get("raw_mae") or 0.0),
        "flow_raw_mae": float(flow.get("raw_mae") or 0.0),
        "suppression_honesty_flags": endpoint_audit.get("suppression_honesty_flags") or {},
    }


def _save_overview(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Dense calibration batch", body="No rows.")
        return
    labels = [f"{row['contract']}:{row['experiment_id']}" for row in rows]
    candidate = [float(row["quarterly_mean_mae"]) for row in rows]
    baseline = [float(row["quarterly_baseline_mae"]) for row in rows]
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4.0, len(labels) * 0.45)))
    ax.barh([idx - 0.18 for idx in y], baseline, height=0.35, label="Carry-forward")
    ax.barh([idx + 0.18 for idx in y], candidate, height=0.35, label="Candidate")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Quarterly normalized MAE")
    ax.set_title("TR-V3 dense calibration batch")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    lines = [
        "# TR-V3 Dense Calibration Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        "",
        "## Decision",
        "",
        f"- Winner: `{decision.get('winner_id', '')}`",
        f"- Status: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "| Contract | Experiment | Quarterly MAE | Baseline MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Suppression honesty |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("rows") or []):
        lines.append(
            f"| {row['contract']} | {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
            f"{float(row['quarterly_baseline_mae']):.6f} | {float(row['diagnosed_raw_mae']):.3f} | "
            f"{float(row['art_raw_mae']):.3f} | {float(row['flow_raw_mae']):.3f} | `{dict(row['suppression_honesty_flags'])}` |"
        )
    lines.extend(
        [
            "",
            "## Lockbox",
            "",
            "| Contract | Winner | MAE |",
            "|---|---|---:|",
        ]
    )
    for contract_name, contract_payload in dict(payload.get("lockbox") or {}).items():
        winner = dict(contract_payload.get("winner") or {})
        lines.append(f"| {contract_name} | {winner.get('experiment_id', '')} | {float(winner.get('quarterly_mean_mae') or 0.0):.6f} |")
    return "\n".join(lines) + "\n"


def run_tr_v3_dense_calibration_batch(
    *,
    run_id: str,
    archive_run_id: str | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
    lockbox_holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or suite._latest_standard_archive_run())
    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")

    legacy_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="legacy_dense",
        experiment_ids=list(DENSE_EXPERIMENT_IDS),
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    purged_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_ids=list(DENSE_EXPERIMENT_IDS),
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )

    legacy_map = _suite_result_map(legacy_payload)
    purged_map = _suite_result_map(purged_payload)
    rows = [
        _result_row(legacy_map["EXP-R10-DENSE-M1-H1"], contract_name="legacy_dense"),
        _result_row(legacy_map["EXP-R10-DENSE-M1-B1-H1"], contract_name="legacy_dense"),
        _result_row(legacy_map["EXP-R10-DENSE-M1-C1-H1"], contract_name="legacy_dense"),
        _result_row(purged_map["EXP-R10-DENSE-M1-H1"], contract_name="purged_dense"),
        _result_row(purged_map["EXP-R10-DENSE-M1-B1-H1"], contract_name="purged_dense"),
        _result_row(purged_map["EXP-R10-DENSE-M1-C1-H1"], contract_name="purged_dense"),
    ]

    holdout_years = sorted(set(int(year) for year in (lockbox_holdout_years or [2025])))
    annual_rows = build_annual_anchor_rows(archive_run)
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    lockbox_contracts: dict[str, Any] = {}
    for contract_name, result_map in (("legacy_dense", legacy_map), ("purged_dense", purged_map)):
        observation_rows = publish._observation_rows_for_lockbox(archive_run, contract_name=contract_name, holdout_years=holdout_years)
        scoring_tiers = publish._scoring_tiers(contract_name)
        lockbox_rows = []
        for experiment_id in ("EXP-R10-DENSE-M1-H1", "EXP-R10-DENSE-M1-B1-H1", "EXP-R10-DENSE-M1-C1-H1", "EXP-R1"):
            lockbox_rows.append(
                publish._evaluate_fixed_holdout_experiment(
                    spec_map[experiment_id],
                    observation_rows=observation_rows,
                    annual_rows=annual_rows,
                    holdout_years=holdout_years,
                    scoring_tiers=scoring_tiers,
                    frozen_config=dict(result_map[experiment_id].get("best_candidate") or {}),
                )
            )
        winner = min(lockbox_rows, key=lambda row: float(row["quarterly_mean_mae"]))
        lockbox_contracts[contract_name] = {
            "rows": lockbox_rows,
            "winner": {
                "experiment_id": str(winner["experiment_id"]),
                "quarterly_mean_mae": float(winner["quarterly_mean_mae"]),
            },
        }

    incumbent = next(row for row in rows if row["contract"] == "purged_dense" and row["experiment_id"] == "EXP-R10-DENSE-M1-H1")
    challenger = next(row for row in rows if row["contract"] == "purged_dense" and row["experiment_id"] == "EXP-R10-DENSE-M1-C1-H1")
    if float(challenger["quarterly_mean_mae"]) < float(incumbent["quarterly_mean_mae"]):
        decision = {
            "winner_id": "EXP-R10-DENSE-M1-C1-H1",
            "status": "promote",
            "why": "Crossfit diagnosed calibration improved the primary purged-dense MAE.",
        }
    elif (
        float(challenger["quarterly_mean_mae"]) <= float(incumbent["quarterly_mean_mae"]) + 0.001
        and float(challenger["diagnosed_raw_mae"]) < float(incumbent["diagnosed_raw_mae"])
    ):
        decision = {
            "winner_id": "EXP-R10-DENSE-M1-C1-H1",
            "status": "sensitivity_keep",
            "why": "Primary MAE is effectively tied and diagnosed-stock calibration improved.",
        }
    else:
        decision = {
            "winner_id": "EXP-R10-DENSE-M1-H1",
            "status": "revert_crossfit_calibration",
            "why": "Crossfit diagnosed calibration did not beat the incumbent dense winner on the primary purged-dense contract.",
        }

    overview_path = analysis_dir / "dense_calibration_overview.png"
    _save_overview(rows, overview_path)
    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "rows": rows,
        "lockbox": lockbox_contracts,
        "decision": decision,
        "artifacts": {"overview_graph": overview_path.name},
    }
    write_json(analysis_dir / "tr_v3_dense_calibration_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_dense_calibration_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted TR-V3 dense calibration comparisons.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
    args = parser.parse_args()
    run_tr_v3_dense_calibration_batch(
        run_id=str(args.run_id),
        archive_run_id=args.archive_run_id,
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
        lockbox_holdout_years=list(args.lockbox_holdout_years),
    )


if __name__ == "__main__":
    main()
