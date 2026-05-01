from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_current_champion_expanded_harp_compatibility_batch as compatibility
from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, write_json


DEFAULT_MERGED_ARCHIVE_RUN_ID = "tr-v3-current-champion-expanded-harp-compatibility-20260419-s00"
DEFAULT_BASELINE_ARCHIVE_RUN_ID = "harp-archive-wdi-standard-20260412-s19"
EXACT_NEIGHBORHOOD_IDS: list[str] = [
    "EXP-R10-EXACT-CHAMPION",
    "EXP-R10-M1",
    "EXP-R10-M1-B1",
    "EXP-R10-M1-F1",
    "EXP-R10-M1-F1-C1",
    "EXP-R10-M2",
    "EXP-R1",
]
DENSE_NEIGHBORHOOD_IDS: list[str] = [
    "EXP-R10-DENSE-CHAMPION",
    "EXP-R10-DENSE-H1",
    "EXP-R10-DENSE-M1",
    "EXP-R10-DENSE-M1-H1",
    "EXP-R10-DENSE-M1-C1-H1",
    "EXP-R10-DENSE-M1-B1-H1",
    "EXP-R10-DENSE-M1-F1-H1",
    "EXP-R10-DENSE-M2",
    "EXP-R1",
]
CONTRACT_CONFIGS: dict[str, dict[str, Any]] = {
    "exact_only": {
        "contract_name": "exact_only",
        "current_champion_id": "EXP-R10-EXACT-CHAMPION",
        "experiment_ids": EXACT_NEIGHBORHOOD_IDS,
        "allowed_tiers": {"exact_observed"},
    },
    "purged_dense": {
        "contract_name": "purged_dense",
        "current_champion_id": "EXP-R10-DENSE-CHAMPION",
        "experiment_ids": DENSE_NEIGHBORHOOD_IDS,
        "allowed_tiers": {"exact_observed", "bridge_observed"},
    },
}


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _result_row(
    result: dict[str, Any],
    *,
    contract_name: str,
    archive_variant: str,
    allowed_tiers: set[str],
) -> dict[str, Any]:
    quarterly_summary = dict(result.get("quarterly_summary") or {})
    annual_summary = dict(result.get("annual_summary") or {})
    residual_rows = compatibility._collect_absolute_residual_rows(
        list(result.get("quarterly_rows") or []),
        allowed_tiers=set(allowed_tiers),
        contract_name=str(contract_name),
        archive_variant=str(archive_variant),
    )
    residual_lookup = compatibility._residual_lookup(residual_rows)
    endpoint_summary = dict(quarterly_summary.get("endpoint_audit_summary") or {})
    support_rows = compatibility._support_rows(quarterly_summary, contract_name=str(contract_name), archive_variant=str(archive_variant))
    honesty_flags = {str(key): int(value) for key, value in dict(endpoint_summary.get("suppression_honesty_flags") or {}).items()}
    return {
        "contract": str(contract_name),
        "archive_variant": str(archive_variant),
        "experiment_id": str(result["experiment_id"]),
        "decision": str(result.get("decision") or ""),
        "quarterly_mean_mae": float(quarterly_summary.get("candidate_mean_mae") or float("inf")),
        "quarterly_baseline_mae": float(quarterly_summary.get("carry_forward_mean_mae") or float("inf")),
        "quarterly_worst_mae": float(quarterly_summary.get("candidate_worst_mae") or float("inf")),
        "annual_mean_incidence_error": float(annual_summary.get("candidate_mean_incidence_error") or float("inf")),
        "score_tuple": [float(value) for value in suite._score_experiment_result(result)],
        "residual_rows": residual_rows,
        "support_rows": support_rows,
        "honesty_flags": honesty_flags,
        "diagnosed_residual_p90": float(dict(residual_lookup.get(("diagnosed_plhiv", "overall")) or {}).get("abs_residual_p90") or 0.0),
        "art_residual_p90": float(dict(residual_lookup.get(("alive_on_art", "overall")) or {}).get("abs_residual_p90") or 0.0),
        "flow_residual_p90": float(dict(residual_lookup.get(("new_diagnosed_cases_period", "overall")) or {}).get("abs_residual_p90") or 0.0),
    }


def _winner_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            tuple(float(value) for value in list(row.get("score_tuple") or [])),
            float(row.get("quarterly_mean_mae") or float("inf")),
        ),
    )


def _compare_to_baseline(
    baseline_row: dict[str, Any],
    merged_row: dict[str, Any],
) -> dict[str, Any]:
    baseline_residual_lookup = compatibility._residual_lookup(list(baseline_row.get("residual_rows") or []))
    merged_residual_lookup = compatibility._residual_lookup(list(merged_row.get("residual_rows") or []))
    baseline_support_lookup = compatibility._support_lookup(list(baseline_row.get("support_rows") or []))
    merged_support_lookup = compatibility._support_lookup(list(merged_row.get("support_rows") or []))
    residual_ratios: list[float] = []
    exact_share_deltas: list[float] = []
    for metric_name in compatibility.METRIC_ORDER:
        baseline_metric_row = dict(baseline_residual_lookup.get((metric_name, "overall")) or {})
        merged_metric_row = dict(merged_residual_lookup.get((metric_name, "overall")) or {})
        baseline_p90 = float(baseline_metric_row.get("abs_residual_p90") or 0.0)
        merged_p90 = float(merged_metric_row.get("abs_residual_p90") or 0.0)
        if baseline_p90 > 1e-9:
            residual_ratios.append(float(merged_p90 / baseline_p90))
        baseline_support = dict(baseline_support_lookup.get(metric_name) or {})
        merged_support = dict(merged_support_lookup.get(metric_name) or {})
        exact_share_deltas.append(float(merged_support.get("exact_share") or 0.0) - float(baseline_support.get("exact_share") or 0.0))
    worsened_flags = {
        key: int(merged_row.get("honesty_flags", {}).get(key, 0)) - int(baseline_row.get("honesty_flags", {}).get(key, 0))
        for key in sorted(set(dict(baseline_row.get("honesty_flags") or {})) | set(dict(merged_row.get("honesty_flags") or {})))
        if int(merged_row.get("honesty_flags", {}).get(key, 0)) > int(baseline_row.get("honesty_flags", {}).get(key, 0))
    }
    mean_mae_ratio = (
        float(merged_row["quarterly_mean_mae"]) / float(baseline_row["quarterly_mean_mae"])
        if float(baseline_row["quarterly_mean_mae"]) > 1e-9
        else 1.0
    )
    worst_mae_ratio = (
        float(merged_row["quarterly_worst_mae"]) / float(baseline_row["quarterly_worst_mae"])
        if float(baseline_row["quarterly_worst_mae"]) > 1e-9
        else 1.0
    )
    residual_p90_ratio_mean = float(np.mean(np.asarray(residual_ratios, dtype=np.float64))) if residual_ratios else 1.0
    mean_exact_share_delta = float(np.mean(np.asarray(exact_share_deltas, dtype=np.float64))) if exact_share_deltas else 0.0
    decision = compatibility._contract_decision(
        mean_mae_ratio=mean_mae_ratio,
        worst_mae_ratio=worst_mae_ratio,
        residual_p90_ratio_mean=residual_p90_ratio_mean,
        honesty_flag_worsened_count=int(len(worsened_flags)),
        mean_exact_share_delta=mean_exact_share_delta,
    )
    return {
        "baseline_experiment_id": str(baseline_row["experiment_id"]),
        "merged_experiment_id": str(merged_row["experiment_id"]),
        "mean_mae_ratio": mean_mae_ratio,
        "worst_mae_ratio": worst_mae_ratio,
        "residual_p90_ratio_mean": residual_p90_ratio_mean,
        "mean_exact_share_delta": mean_exact_share_delta,
        "worsened_honesty_flags": worsened_flags,
        "honesty_flag_worsened_count": int(len(worsened_flags)),
        "decision": decision,
    }


def _overall_decision(contract_rows: list[dict[str, Any]]) -> str:
    by_contract = {str(row["contract"]): dict(row) for row in contract_rows}
    severe = [row for row in by_contract.values() if str(row["baseline_comparison"]["decision"]) == "severe_drift"]
    if severe:
        return "reopen_broader_model_family_exploration"
    current_champion_kept = all(bool(row.get("current_champion_kept")) for row in by_contract.values())
    stable = all(str(row["baseline_comparison"]["decision"]) == "stable" for row in by_contract.values())
    if current_champion_kept and stable:
        return "keep_current_champions_on_merged_archive"
    return "promote_narrow_r10_refresh"


def _run_contract_neighborhood(
    *,
    archive_run_id: str,
    baseline_archive_run_id: str,
    contract_name: str,
    experiment_ids: list[str],
    current_champion_id: str,
    allowed_tiers: set[str],
) -> dict[str, Any]:
    baseline_payload = hardening._run_selected_suite_contract(
        archive_run_id=str(baseline_archive_run_id),
        contract_name=str(contract_name),
        experiment_ids=[str(current_champion_id)],
        quarterly_start_year=2010,
        quarterly_end_year=2025,
        quarterly_min_train_years=3,
        annual_start_year=2010,
        annual_end_year=2024,
        annual_min_train_years=5,
        horizon_years=1,
    )
    baseline_row = _result_row(
        dict(list(baseline_payload.get("results") or [])[0] or {}),
        contract_name=str(contract_name),
        archive_variant="baseline",
        allowed_tiers=set(allowed_tiers),
    )

    merged_payload = hardening._run_selected_suite_contract(
        archive_run_id=str(archive_run_id),
        contract_name=str(contract_name),
        experiment_ids=list(experiment_ids),
        quarterly_start_year=2010,
        quarterly_end_year=2025,
        quarterly_min_train_years=3,
        annual_start_year=2010,
        annual_end_year=2024,
        annual_min_train_years=5,
        horizon_years=1,
    )
    merged_map = _suite_result_map(merged_payload)
    merged_rows = [
        _result_row(
            dict(merged_map[experiment_id]),
            contract_name=str(contract_name),
            archive_variant="merged",
            allowed_tiers=set(allowed_tiers),
        )
        for experiment_id in experiment_ids
        if experiment_id in merged_map
    ]
    winner = _winner_row(merged_rows)
    current_row = next(row for row in merged_rows if str(row["experiment_id"]) == str(current_champion_id))
    baseline_comparison = _compare_to_baseline(baseline_row, winner)
    return {
        "contract": str(contract_name),
        "current_champion_id": str(current_champion_id),
        "baseline_current_champion": baseline_row,
        "merged_rows": merged_rows,
        "merged_winner": winner,
        "merged_current_champion": current_row,
        "current_champion_rank": 1 + sum(
            1
            for row in merged_rows
            if tuple(float(value) for value in list(row.get("score_tuple") or []))
            < tuple(float(value) for value in list(current_row.get("score_tuple") or []))
        ),
        "current_champion_kept": bool(str(winner["experiment_id"]) == str(current_champion_id)),
        "baseline_comparison": baseline_comparison,
        "winner_improvement_vs_current_champion": {
            "mean_mae_delta": float(current_row["quarterly_mean_mae"]) - float(winner["quarterly_mean_mae"]),
            "worst_mae_delta": float(current_row["quarterly_worst_mae"]) - float(winner["quarterly_worst_mae"]),
            "diagnosed_p90_delta": float(current_row["diagnosed_residual_p90"]) - float(winner["diagnosed_residual_p90"]),
            "art_p90_delta": float(current_row["art_residual_p90"]) - float(winner["art_residual_p90"]),
            "flow_p90_delta": float(current_row["flow_residual_p90"]) - float(winner["flow_residual_p90"]),
        },
    }


def _plot_contract_mae(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    labels = [str(row["experiment_id"]) for row in rows]
    candidate = [float(row["quarterly_mean_mae"]) for row in rows]
    baseline = [float(row["quarterly_baseline_mae"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.45)))
    ax.barh(y - 0.18, baseline, height=0.35, label="carry-forward")
    ax.barh(y + 0.18, candidate, height=0.35, label="candidate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Quarterly normalized MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_winner_drift(contract_payloads: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["contract"]) for row in contract_payloads]
    mae_ratio = [float(dict(row["baseline_comparison"])["mean_mae_ratio"]) for row in contract_payloads]
    residual_ratio = [float(dict(row["baseline_comparison"])["residual_p90_ratio_mean"]) for row in contract_payloads]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2.0, mae_ratio, width=width, label="mean MAE ratio")
    ax.bar(x + width / 2.0, residual_ratio, width=width, label="residual p90 ratio")
    ax.axhline(1.15, color="#7a7a7a", linestyle="--", linewidth=1.0)
    ax.axhline(1.35, color="#c44e52", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Ratio vs baseline current champion")
    ax.set_title("Narrow R10 winner drift on merged archive")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Current Champion R10 Neighborhood Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline archive: `{payload['baseline_archive_run_id']}`",
        f"- Merged archive: `{payload['merged_archive_run_id']}`",
        f"- Overall decision: `{payload['overall_decision']}`",
        "",
    ]
    for contract_payload in list(payload.get("contracts") or []):
        improvement = dict(contract_payload.get("winner_improvement_vs_current_champion") or {})
        baseline_comparison = dict(contract_payload.get("baseline_comparison") or {})
        lines.extend(
            [
                f"## {contract_payload['contract']}",
                "",
                f"- Current champion: `{contract_payload['current_champion_id']}`",
                f"- Winner on merged archive: `{contract_payload['merged_winner']['experiment_id']}`",
                f"- Current champion kept: `{contract_payload['current_champion_kept']}`",
                f"- Current champion rank on merged archive: `{contract_payload['current_champion_rank']}`",
                f"- Winner MAE improvement vs current champion: `{float(improvement.get('mean_mae_delta') or 0.0):.6f}`",
                f"- Winner drift decision vs baseline current champion: `{baseline_comparison.get('decision', '')}`",
                "",
                "| Experiment | Quarterly MAE | Worst MAE | Annual error | Diagnosed p90 | ART p90 | Flow p90 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in list(contract_payload.get("merged_rows") or []):
            lines.append(
                f"| `{row['experiment_id']}` | `{float(row['quarterly_mean_mae']):.6f}` | "
                f"`{float(row['quarterly_worst_mae']):.6f}` | `{float(row['annual_mean_incidence_error']):.6f}` | "
                f"`{float(row['diagnosed_residual_p90']):.3f}` | `{float(row['art_residual_p90']):.3f}` | "
                f"`{float(row['flow_residual_p90']):.3f}` |"
            )
        lines.extend(
            [
                "",
                "| Baseline current champion | Merged winner | MAE ratio | Worst ratio | Residual p90 ratio | Worsened honesty flags |",
                "|---|---|---:|---:|---:|---|",
                f"| `{baseline_comparison.get('baseline_experiment_id', '')}` | `{baseline_comparison.get('merged_experiment_id', '')}` | "
                f"`{float(baseline_comparison.get('mean_mae_ratio') or 0.0):.3f}` | `{float(baseline_comparison.get('worst_mae_ratio') or 0.0):.3f}` | "
                f"`{float(baseline_comparison.get('residual_p90_ratio_mean') or 0.0):.3f}` | "
                f"`{dict(baseline_comparison.get('worsened_honesty_flags') or {})}` |",
                "",
            ]
        )
    lines.extend(
        [
            "## Graphs",
            "",
            "- `current_champion_r10_neighborhood_exact_only.png`",
            "- `current_champion_r10_neighborhood_purged_dense.png`",
            "- `current_champion_r10_neighborhood_winner_drift.png`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_tr_v3_current_champion_r10_neighborhood_batch(
    *,
    run_id: str,
    merged_archive_run_id: str = DEFAULT_MERGED_ARCHIVE_RUN_ID,
    baseline_archive_run_id: str = DEFAULT_BASELINE_ARCHIVE_RUN_ID,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    contract_payloads = [
        _run_contract_neighborhood(
            archive_run_id=str(merged_archive_run_id),
            baseline_archive_run_id=str(baseline_archive_run_id),
            contract_name=str(config["contract_name"]),
            experiment_ids=list(config["experiment_ids"]),
            current_champion_id=str(config["current_champion_id"]),
            allowed_tiers=set(config["allowed_tiers"]),
        )
        for _, config in CONTRACT_CONFIGS.items()
    ]
    _plot_contract_mae(
        list(dict(contract_payloads[0]).get("merged_rows") or []),
        analysis_dir / "current_champion_r10_neighborhood_exact_only.png",
        title="Exact R10 neighborhood on merged archive",
    )
    _plot_contract_mae(
        list(dict(contract_payloads[1]).get("merged_rows") or []),
        analysis_dir / "current_champion_r10_neighborhood_purged_dense.png",
        title="Dense R10 neighborhood on merged archive",
    )
    _plot_winner_drift(contract_payloads, analysis_dir / "current_champion_r10_neighborhood_winner_drift.png")
    payload = {
        "run_id": str(run_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_archive_run_id": str(baseline_archive_run_id),
        "merged_archive_run_id": str(merged_archive_run_id),
        "contracts": contract_payloads,
        "overall_decision": _overall_decision(contract_payloads),
    }
    write_json(analysis_dir / "tr_v3_current_champion_r10_neighborhood_batch_report.json", payload)
    (analysis_dir / "tr_v3_current_champion_r10_neighborhood_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a narrow R10 neighborhood rerun on the merged archive.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-archive-run-id", default=DEFAULT_MERGED_ARCHIVE_RUN_ID)
    parser.add_argument("--baseline-archive-run-id", default=DEFAULT_BASELINE_ARCHIVE_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_tr_v3_current_champion_r10_neighborhood_batch(
        run_id=args.run_id,
        merged_archive_run_id=args.merged_archive_run_id,
        baseline_archive_run_id=args.baseline_archive_run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
