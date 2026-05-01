from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_champion_forecast as champion_forecast
from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as monthly_lane
from epigraph_ph.phase3 import tr_v3_phase2_champion_equivalence_batch as champion_equivalence
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, write_json


DEFAULT_COVERAGE_ARCHIVE_RUN_ID = "harp-archive-hiv-data-coverage-20260419-s00"
FORECAST_HORIZON_QUARTERS = 4
INTERVAL_ALPHA = 0.1
METRIC_ORDER: tuple[str, ...] = tuple(champion_forecast.PRIMARY_METRICS)
TIER_ORDER: tuple[str, ...] = ("overall", "exact_observed", "bridge_observed")


def _current_winner_configs() -> dict[str, dict[str, Any]]:
    return champion_equivalence._current_winner_configs()


def _resolve_baseline_archive_run_id(archive_run_id: str | None) -> str:
    resolved = str(archive_run_id or "").strip()
    if resolved:
        return resolved
    return str(hardening._latest_standard_archive_run())


def _archive_run_dir(run_id: str) -> Path:
    return ROOT_DIR / "artifacts" / "runs" / str(run_id)


def _build_merged_archive(
    *,
    run_id: str,
    baseline_archive_run_id: str,
    coverage_archive_run_id: str,
) -> dict[str, Any]:
    run_dir = _archive_run_dir(run_id)
    ensure_dir(run_dir)
    summary = monthly_lane._copy_harp_archive(
        _archive_run_dir(baseline_archive_run_id),
        run_dir,
        coverage_run_dir=_archive_run_dir(coverage_archive_run_id),
    )
    return dict(summary)


def _collect_absolute_residual_rows(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
    contract_name: str,
    archive_variant: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = {
        (metric_name, tier_name): []
        for metric_name in METRIC_ORDER
        for tier_name in TIER_ORDER
    }
    for split_row in quarterly_rows:
        target_rows = list(split_row.get("holdout_target_rows") or [])
        prediction_rows = list(split_row.get("candidate_prediction_rows") or [])
        for target_row, prediction_row in zip(target_rows, prediction_rows, strict=False):
            for metric_name in METRIC_ORDER:
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                tier_name = str(hardening._metric_tier(target_row, metric_name))
                if tier_name not in allowed_tiers:
                    continue
                abs_residual = abs(float(prediction_value) - float(target_value))
                buckets[(metric_name, "overall")].append(abs_residual)
                buckets[(metric_name, tier_name)].append(abs_residual)
    rows: list[dict[str, Any]] = []
    for metric_name in METRIC_ORDER:
        for tier_name in TIER_ORDER:
            values = np.asarray(buckets[(metric_name, tier_name)], dtype=np.float64)
            rows.append(
                {
                    "contract": str(contract_name),
                    "archive_variant": str(archive_variant),
                    "metric": str(metric_name),
                    "tier": str(tier_name),
                    "count": int(values.size),
                    "abs_residual_mean": float(np.mean(values)) if values.size else 0.0,
                    "abs_residual_p90": float(np.quantile(values, 0.9)) if values.size else 0.0,
                }
            )
    return rows


def _support_rows(
    quarterly_summary: dict[str, Any],
    *,
    contract_name: str,
    archive_variant: str,
) -> list[dict[str, Any]]:
    endpoint_summary = dict(quarterly_summary.get("endpoint_audit_summary") or {})
    holdout_support = dict(endpoint_summary.get("holdout_support_counts") or {})
    rows: list[dict[str, Any]] = []
    for metric_name in METRIC_ORDER:
        metric_payload = dict(holdout_support.get(metric_name) or {})
        exact_count = int(metric_payload.get("exact_observed") or 0)
        bridge_count = int(metric_payload.get("bridge_observed") or 0)
        scored_count = int(metric_payload.get("scored") or 0)
        exact_share = float(exact_count / scored_count) if scored_count > 0 else 0.0
        rows.append(
            {
                "contract": str(contract_name),
                "archive_variant": str(archive_variant),
                "metric": str(metric_name),
                "exact_count": exact_count,
                "bridge_count": bridge_count,
                "scored_count": scored_count,
                "exact_share": exact_share,
            }
        )
    return rows


def _future_forecast_rows(
    *,
    archive_run_id: str,
    result: dict[str, Any],
    forecast_contract: str,
    winner_id: str,
    allowed_tiers: set[str],
    horizon_quarters: int,
) -> list[dict[str, Any]]:
    observation_rows, _, _ = champion_forecast._contract_payload(str(archive_run_id), str(forecast_contract))
    if not observation_rows:
        return []
    latest_quarter = max((str(row["quarter"]) for row in observation_rows), key=suite.quarter_sort_key)
    annual_rows = champion_forecast.build_annual_anchor_rows(str(archive_run_id))
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    forecast_candidate = champion_forecast._forecast_future_rows(
        observation_rows,
        annual_rows=annual_rows,
        spec=spec_map[str(winner_id)],
        best_candidate=dict(result["best_candidate"]),
        forecast_quarters=champion_forecast._future_quarters(latest_quarter, int(horizon_quarters)),
    )
    interval_payload = champion_forecast._residual_interval_payload(
        list(result.get("quarterly_rows") or []),
        allowed_tiers=set(allowed_tiers),
        alpha=float(INTERVAL_ALPHA),
    )
    return champion_forecast._apply_empirical_intervals(
        list(forecast_candidate.get("prediction_rows") or []),
        interval_payload,
    )


def _evaluate_archive_contract(
    *,
    archive_run_id: str,
    contract_name: str,
    archive_variant: str,
    winner_config: dict[str, Any],
    forecast_horizon_quarters: int,
) -> dict[str, Any]:
    payload = hardening._run_selected_suite_contract(
        archive_run_id=str(archive_run_id),
        contract_name=str(winner_config["suite_contract"]),
        experiment_ids=[str(winner_config["winner_id"])],
        quarterly_start_year=2010,
        quarterly_end_year=2025,
        quarterly_min_train_years=3,
        annual_start_year=2010,
        annual_end_year=2024,
        annual_min_train_years=5,
        horizon_years=1,
    )
    result = dict(list(payload.get("results") or [])[0] or {})
    quarterly_summary = dict(result.get("quarterly_summary") or {})
    endpoint_summary = dict(quarterly_summary.get("endpoint_audit_summary") or {})
    honesty_flags = {str(key): int(value) for key, value in dict(endpoint_summary.get("suppression_honesty_flags") or {}).items()}
    residual_rows = _collect_absolute_residual_rows(
        list(result.get("quarterly_rows") or []),
        allowed_tiers=set(winner_config["allowed_tiers"]),
        contract_name=str(contract_name),
        archive_variant=str(archive_variant),
    )
    support_rows = _support_rows(quarterly_summary, contract_name=str(contract_name), archive_variant=str(archive_variant))
    forecast_rows = _future_forecast_rows(
        archive_run_id=str(archive_run_id),
        result=result,
        forecast_contract=str(winner_config["forecast_contract"]),
        winner_id=str(winner_config["winner_id"]),
        allowed_tiers=set(winner_config["allowed_tiers"]),
        horizon_quarters=int(forecast_horizon_quarters),
    )
    return {
        "archive_run_id": str(archive_run_id),
        "archive_variant": str(archive_variant),
        "contract": str(contract_name),
        "winner_id": str(winner_config["winner_id"]),
        "suite_contract": str(winner_config["suite_contract"]),
        "forecast_contract": str(winner_config["forecast_contract"]),
        "allowed_tiers": sorted(str(name) for name in set(winner_config["allowed_tiers"])),
        "quarterly_summary": quarterly_summary,
        "annual_summary": dict(result.get("annual_summary") or {}),
        "honesty_flags": honesty_flags,
        "residual_rows": residual_rows,
        "support_rows": support_rows,
        "forecast_rows": forecast_rows,
    }


def _residual_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["metric"]), str(row["tier"])): dict(row) for row in rows}


def _support_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric"]): dict(row) for row in rows}


def _forecast_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        quarter = str(row.get("quarter") or "")
        for metric_name in METRIC_ORDER:
            value = row.get(metric_name)
            if value is None:
                continue
            lookup[(quarter, metric_name)] = float(value)
    return lookup


def _mean_numeric(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _contract_decision(
    *,
    mean_mae_ratio: float,
    worst_mae_ratio: float,
    residual_p90_ratio_mean: float,
    honesty_flag_worsened_count: int,
    mean_exact_share_delta: float,
) -> str:
    if (
        mean_mae_ratio > 1.35
        or worst_mae_ratio > 1.35
        or residual_p90_ratio_mean > 1.50
        or int(honesty_flag_worsened_count) > 0
    ):
        return "severe_drift"
    if (
        mean_mae_ratio > 1.15
        or worst_mae_ratio > 1.15
        or residual_p90_ratio_mean > 1.25
        or abs(float(mean_exact_share_delta)) > 0.05
    ):
        return "moderate_drift"
    return "stable"


def _comparison_rows(
    baseline_payloads: dict[str, dict[str, Any]],
    merged_payloads: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contract_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    for contract_name in sorted(set(baseline_payloads) | set(merged_payloads)):
        baseline_payload = dict(baseline_payloads.get(contract_name) or {})
        merged_payload = dict(merged_payloads.get(contract_name) or {})
        baseline_summary = dict(baseline_payload.get("quarterly_summary") or {})
        merged_summary = dict(merged_payload.get("quarterly_summary") or {})
        baseline_mean_mae = float(baseline_summary.get("candidate_mean_mae") or 0.0)
        merged_mean_mae = float(merged_summary.get("candidate_mean_mae") or 0.0)
        baseline_worst_mae = float(baseline_summary.get("candidate_worst_mae") or 0.0)
        merged_worst_mae = float(merged_summary.get("candidate_worst_mae") or 0.0)
        baseline_annual_error = float(dict(baseline_payload.get("annual_summary") or {}).get("candidate_mean_incidence_error") or 0.0)
        merged_annual_error = float(dict(merged_payload.get("annual_summary") or {}).get("candidate_mean_incidence_error") or 0.0)
        baseline_residual_lookup = _residual_lookup(list(baseline_payload.get("residual_rows") or []))
        merged_residual_lookup = _residual_lookup(list(merged_payload.get("residual_rows") or []))
        residual_p90_ratios: list[float] = []
        for metric_name in METRIC_ORDER:
            baseline_row = dict(baseline_residual_lookup.get((metric_name, "overall")) or {})
            merged_row = dict(merged_residual_lookup.get((metric_name, "overall")) or {})
            baseline_p90 = float(baseline_row.get("abs_residual_p90") or 0.0)
            merged_p90 = float(merged_row.get("abs_residual_p90") or 0.0)
            ratio = float(merged_p90 / baseline_p90) if baseline_p90 > 1e-9 else (1.0 if merged_p90 <= 1e-9 else float("inf"))
            residual_p90_ratios.append(ratio)
            residual_rows.append(
                {
                    "contract": str(contract_name),
                    "metric": str(metric_name),
                    "baseline_abs_residual_mean": float(baseline_row.get("abs_residual_mean") or 0.0),
                    "merged_abs_residual_mean": float(merged_row.get("abs_residual_mean") or 0.0),
                    "baseline_abs_residual_p90": baseline_p90,
                    "merged_abs_residual_p90": merged_p90,
                    "p90_ratio": ratio,
                }
            )
        baseline_support_lookup = _support_lookup(list(baseline_payload.get("support_rows") or []))
        merged_support_lookup = _support_lookup(list(merged_payload.get("support_rows") or []))
        exact_share_deltas: list[float] = []
        for metric_name in METRIC_ORDER:
            baseline_row = dict(baseline_support_lookup.get(metric_name) or {})
            merged_row = dict(merged_support_lookup.get(metric_name) or {})
            exact_share_delta = float(merged_row.get("exact_share") or 0.0) - float(baseline_row.get("exact_share") or 0.0)
            exact_share_deltas.append(exact_share_delta)
            support_rows.append(
                {
                    "contract": str(contract_name),
                    "metric": str(metric_name),
                    "baseline_exact_count": int(baseline_row.get("exact_count") or 0),
                    "baseline_bridge_count": int(baseline_row.get("bridge_count") or 0),
                    "merged_exact_count": int(merged_row.get("exact_count") or 0),
                    "merged_bridge_count": int(merged_row.get("bridge_count") or 0),
                    "baseline_exact_share": float(baseline_row.get("exact_share") or 0.0),
                    "merged_exact_share": float(merged_row.get("exact_share") or 0.0),
                    "exact_share_delta": exact_share_delta,
                }
            )
        baseline_flags = dict(baseline_payload.get("honesty_flags") or {})
        merged_flags = dict(merged_payload.get("honesty_flags") or {})
        worsened_flags = {
            key: int(merged_flags.get(key, 0)) - int(baseline_flags.get(key, 0))
            for key in sorted(set(baseline_flags) | set(merged_flags))
            if int(merged_flags.get(key, 0)) > int(baseline_flags.get(key, 0))
        }
        baseline_forecast_lookup = _forecast_lookup(list(baseline_payload.get("forecast_rows") or []))
        merged_forecast_lookup = _forecast_lookup(list(merged_payload.get("forecast_rows") or []))
        forecast_keys = sorted(set(baseline_forecast_lookup) | set(merged_forecast_lookup), key=lambda item: (suite.quarter_sort_key(item[0]), item[1]))
        for quarter, metric_name in forecast_keys:
            baseline_value = float(baseline_forecast_lookup.get((quarter, metric_name), 0.0))
            merged_value = float(merged_forecast_lookup.get((quarter, metric_name), 0.0))
            forecast_rows.append(
                {
                    "contract": str(contract_name),
                    "quarter": str(quarter),
                    "metric": str(metric_name),
                    "baseline_value": baseline_value,
                    "merged_value": merged_value,
                    "delta": float(merged_value - baseline_value),
                }
            )
        contract_rows.append(
            {
                "contract": str(contract_name),
                "winner_id": str(merged_payload.get("winner_id") or baseline_payload.get("winner_id") or ""),
                "baseline_mean_mae": baseline_mean_mae,
                "merged_mean_mae": merged_mean_mae,
                "mean_mae_delta": float(merged_mean_mae - baseline_mean_mae),
                "mean_mae_ratio": float(merged_mean_mae / baseline_mean_mae) if baseline_mean_mae > 1e-9 else 1.0,
                "baseline_worst_mae": baseline_worst_mae,
                "merged_worst_mae": merged_worst_mae,
                "worst_mae_delta": float(merged_worst_mae - baseline_worst_mae),
                "worst_mae_ratio": float(merged_worst_mae / baseline_worst_mae) if baseline_worst_mae > 1e-9 else 1.0,
                "baseline_annual_error": baseline_annual_error,
                "merged_annual_error": merged_annual_error,
                "annual_error_delta": float(merged_annual_error - baseline_annual_error),
                "residual_p90_ratio_mean": _mean_numeric([value for value in residual_p90_ratios if np.isfinite(value)]),
                "mean_exact_share_delta": _mean_numeric(exact_share_deltas),
                "baseline_honesty_flags": baseline_flags,
                "merged_honesty_flags": merged_flags,
                "worsened_honesty_flags": worsened_flags,
                "honesty_flag_worsened_count": int(len(worsened_flags)),
                "decision": _contract_decision(
                    mean_mae_ratio=float(merged_mean_mae / baseline_mean_mae) if baseline_mean_mae > 1e-9 else 1.0,
                    worst_mae_ratio=float(merged_worst_mae / baseline_worst_mae) if baseline_worst_mae > 1e-9 else 1.0,
                    residual_p90_ratio_mean=_mean_numeric([value for value in residual_p90_ratios if np.isfinite(value)]),
                    honesty_flag_worsened_count=int(len(worsened_flags)),
                    mean_exact_share_delta=_mean_numeric(exact_share_deltas),
                ),
            }
        )
    return contract_rows, residual_rows, support_rows, forecast_rows


def _overall_decision(contract_rows: list[dict[str, Any]]) -> str:
    decisions = [str(row.get("decision") or "") for row in contract_rows]
    severe_count = sum(1 for decision in decisions if decision == "severe_drift")
    moderate_count = sum(1 for decision in decisions if decision == "moderate_drift")
    if severe_count >= 2:
        return "reopen_family_exploration"
    if severe_count >= 1 or moderate_count >= 1:
        return "rerun_r10_neighborhood"
    return "keep_current_champions"


def _plot_mae_compare(contract_rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["contract"]) for row in contract_rows]
    baseline = [float(row["baseline_mean_mae"]) for row in contract_rows]
    merged = [float(row["merged_mean_mae"]) for row in contract_rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2.0, baseline, width=width, label="baseline")
    ax.bar(x + width / 2.0, merged, width=width, label="merged")
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Quarterly mean MAE")
    ax.set_title("Current champion compatibility on expanded HARP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_p90_compare(residual_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [dict(row) for row in residual_rows if str(row.get("metric") or "")]
    labels = [f"{row['contract']}:{row['metric']}" for row in rows]
    baseline = [float(row["baseline_abs_residual_p90"]) for row in rows]
    merged = [float(row["merged_abs_residual_p90"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.42)))
    ax.barh(y - width / 2.0, baseline, height=width, label="baseline")
    ax.barh(y + width / 2.0, merged, height=width, label="merged")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute residual p90")
    ax.set_title("Residual-scale comparison by contract and metric")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_support_mix_compare(support_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [dict(row) for row in support_rows if str(row.get("metric") or "")]
    labels = [f"{row['contract']}:{row['metric']}" for row in rows]
    baseline = [float(row["baseline_exact_share"]) for row in rows]
    merged = [float(row["merged_exact_share"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.42)))
    ax.barh(y - width / 2.0, baseline, height=width, label="baseline")
    ax.barh(y + width / 2.0, merged, height=width, label="merged")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Exact-share of scored holdout support")
    ax.set_title("Support-mix comparison by contract and metric")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_forecast_compare(
    forecast_rows: list[dict[str, Any]],
    *,
    contract_name: str,
    path: Path,
) -> None:
    rows = [dict(row) for row in forecast_rows if str(row.get("contract") or "") == str(contract_name)]
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No forecast rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    quarter_axis = sorted({str(row["quarter"]) for row in rows}, key=suite.quarter_sort_key)
    quarter_index = {quarter: idx for idx, quarter in enumerate(quarter_axis)}
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(15, 4.5), squeeze=False)
    for ax, metric_name in zip(axes[0], METRIC_ORDER, strict=False):
        metric_rows = [row for row in rows if str(row["metric"]) == metric_name]
        baseline_values = np.full(len(quarter_axis), np.nan, dtype=np.float64)
        merged_values = np.full(len(quarter_axis), np.nan, dtype=np.float64)
        for row in metric_rows:
            idx = quarter_index[str(row["quarter"])]
            baseline_values[idx] = float(row["baseline_value"])
            merged_values[idx] = float(row["merged_value"])
        ax.plot(quarter_axis, baseline_values, marker="o", linewidth=2.0, label="baseline")
        ax.plot(quarter_axis, merged_values, marker="o", linewidth=2.0, label="merged")
        ax.set_title(metric_name.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Future forecast comparison: {contract_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Current Champion Expanded-HARP Compatibility Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline archive: `{payload['baseline_archive_run_id']}`",
        f"- Coverage archive: `{payload['coverage_archive_run_id']}`",
        f"- Merged archive run: `{payload['merged_archive_run_id']}`",
        f"- Exact champion: `{payload['current_champions']['exact_only']}`",
        f"- Dense champion: `{payload['current_champions']['purged_dense']}`",
        f"- Overall decision: `{payload['overall_decision']}`",
        "",
        "## Merge summary",
        "",
        f"- Mode: `{payload['merge_summary'].get('mode', 'copy_only')}`",
        f"- Coverage multinational rows: `{payload['merge_summary'].get('coverage_multinational_row_count', 0)}`",
        f"- Preserved observed program panel: `{payload['merge_summary'].get('preserved_observed_program_panel', False)}`",
        f"- Preserved diagnosis flow points: `{payload['merge_summary'].get('preserved_diagnosis_flow_points', False)}`",
        "",
        "## Contract compatibility",
        "",
        "| Contract | Winner | Baseline mean MAE | Merged mean MAE | MAE ratio | Residual p90 ratio | Exact-share delta | Worsened honesty flags | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("contract_rows") or []):
        lines.append(
            f"| `{row['contract']}` | `{row['winner_id']}` | `{float(row['baseline_mean_mae']):.6f}` | "
            f"`{float(row['merged_mean_mae']):.6f}` | `{float(row['mean_mae_ratio']):.3f}` | "
            f"`{float(row['residual_p90_ratio_mean']):.3f}` | `{float(row['mean_exact_share_delta']):.3f}` | "
            f"`{int(row['honesty_flag_worsened_count'])}` | `{row['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Residual p90 by metric",
            "",
            "| Contract | Metric | Baseline p90 | Merged p90 | Ratio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in list(payload.get("residual_rows") or []):
        lines.append(
            f"| `{row['contract']}` | `{row['metric']}` | `{float(row['baseline_abs_residual_p90']):.3f}` | "
            f"`{float(row['merged_abs_residual_p90']):.3f}` | `{float(row['p90_ratio']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Terminal future-forecast deltas",
            "",
            "| Contract | Metric | Baseline terminal | Merged terminal | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    terminal_rows = [dict(row) for row in payload.get("forecast_rows") or []]
    if terminal_rows:
        last_quarter_by_contract = {
            contract_name: max(
                [str(row["quarter"]) for row in terminal_rows if str(row["contract"]) == contract_name],
                key=suite.quarter_sort_key,
            )
            for contract_name in sorted({str(row["contract"]) for row in terminal_rows})
        }
        for row in terminal_rows:
            if str(row["quarter"]) != str(last_quarter_by_contract[str(row["contract"])]):
                continue
            lines.append(
                f"| `{row['contract']}` | `{row['metric']}` | `{float(row['baseline_value']):.3f}` | "
                f"`{float(row['merged_value']):.3f}` | `{float(row['delta']):.3f}` |"
            )
    lines.extend(
        [
            "",
            "## Graphs",
            "",
            "- `current_champion_expanded_harp_mae_compare.png`",
            "- `current_champion_expanded_harp_residual_p90_compare.png`",
            "- `current_champion_expanded_harp_support_compare.png`",
            "- `current_champion_expanded_harp_exact_forecast_compare.png`",
            "- `current_champion_expanded_harp_dense_forecast_compare.png`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_tr_v3_current_champion_expanded_harp_compatibility_batch(
    *,
    run_id: str,
    baseline_archive_run_id: str | None = None,
    coverage_archive_run_id: str = DEFAULT_COVERAGE_ARCHIVE_RUN_ID,
    forecast_horizon_quarters: int = FORECAST_HORIZON_QUARTERS,
) -> dict[str, Any]:
    resolved_baseline_archive = _resolve_baseline_archive_run_id(baseline_archive_run_id)
    resolved_coverage_archive = str(coverage_archive_run_id)
    run_dir = _archive_run_dir(run_id)
    analysis_dir = run_dir / "analysis"
    ensure_dir(analysis_dir)

    merge_summary = _build_merged_archive(
        run_id=str(run_id),
        baseline_archive_run_id=str(resolved_baseline_archive),
        coverage_archive_run_id=str(resolved_coverage_archive),
    )
    winner_configs = _current_winner_configs()

    baseline_payloads: dict[str, dict[str, Any]] = {}
    merged_payloads: dict[str, dict[str, Any]] = {}
    for contract_name, winner_config in winner_configs.items():
        baseline_payloads[contract_name] = _evaluate_archive_contract(
            archive_run_id=str(resolved_baseline_archive),
            contract_name=str(contract_name),
            archive_variant="baseline",
            winner_config=dict(winner_config),
            forecast_horizon_quarters=int(forecast_horizon_quarters),
        )
        merged_payloads[contract_name] = _evaluate_archive_contract(
            archive_run_id=str(run_id),
            contract_name=str(contract_name),
            archive_variant="merged",
            winner_config=dict(winner_config),
            forecast_horizon_quarters=int(forecast_horizon_quarters),
        )

    contract_rows, residual_rows, support_rows, forecast_rows = _comparison_rows(baseline_payloads, merged_payloads)
    overall_decision = _overall_decision(contract_rows)

    _plot_mae_compare(contract_rows, analysis_dir / "current_champion_expanded_harp_mae_compare.png")
    _plot_residual_p90_compare(residual_rows, analysis_dir / "current_champion_expanded_harp_residual_p90_compare.png")
    _plot_support_mix_compare(support_rows, analysis_dir / "current_champion_expanded_harp_support_compare.png")
    _plot_forecast_compare(forecast_rows, contract_name="exact_only", path=analysis_dir / "current_champion_expanded_harp_exact_forecast_compare.png")
    _plot_forecast_compare(forecast_rows, contract_name="purged_dense", path=analysis_dir / "current_champion_expanded_harp_dense_forecast_compare.png")

    payload = {
        "run_id": str(run_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_archive_run_id": str(resolved_baseline_archive),
        "coverage_archive_run_id": str(resolved_coverage_archive),
        "merged_archive_run_id": str(run_id),
        "current_champions": {
            contract_name: str(config["winner_id"])
            for contract_name, config in winner_configs.items()
        },
        "merge_summary": merge_summary,
        "contract_rows": contract_rows,
        "residual_rows": residual_rows,
        "support_rows": support_rows,
        "forecast_rows": forecast_rows,
        "overall_decision": overall_decision,
    }
    write_json(analysis_dir / "tr_v3_current_champion_expanded_harp_compatibility_batch_report.json", payload)
    (analysis_dir / "tr_v3_current_champion_expanded_harp_compatibility_batch_report.md").write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen current-champion expanded-HARP compatibility batch.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-archive-run-id", default="")
    parser.add_argument("--coverage-archive-run-id", default=DEFAULT_COVERAGE_ARCHIVE_RUN_ID)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=FORECAST_HORIZON_QUARTERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_tr_v3_current_champion_expanded_harp_compatibility_batch(
        run_id=args.run_id,
        baseline_archive_run_id=args.baseline_archive_run_id or None,
        coverage_archive_run_id=args.coverage_archive_run_id,
        forecast_horizon_quarters=args.forecast_horizon_quarters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
