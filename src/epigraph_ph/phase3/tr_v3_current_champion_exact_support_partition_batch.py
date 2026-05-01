from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_current_champion_expanded_harp_compatibility_batch as compatibility
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_repair_search as repair_search
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, write_json


DEFAULT_MERGED_ARCHIVE_RUN_ID = "tr-v3-current-champion-expanded-harp-compatibility-20260419-s00"
DEFAULT_BASELINE_ARCHIVE_RUN_ID = "harp-archive-wdi-standard-20260412-s19"
CURRENT_EXACT_EXPERIMENT_ID = "EXP-R10-EXACT-CHAMPION"
SERIES_MODELS: tuple[str, ...] = ("level", "delta", "piecewise")
VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "base": {
        "transition_model": "direct_observation_repair",
        "repair_params": {},
    },
    "bias": {
        "transition_model": "direct_observation_bias_corrected",
        "repair_params": {
            "diagnosed_bias_weight": 1.0,
            "art_bias_weight": 1.0,
            "flow_bias_weight": 1.0,
        },
    },
    "crossfit": {
        "transition_model": "direct_observation_joint_consistency_crossfit_calibrated",
        "repair_params": {
            "diagnosed_crossfit_weight": 1.0,
            "diagnosed_crossfit_min_points": 6,
            "diagnosed_crossfit_recent_pool": 16,
            "diagnosed_crossfit_min_train_years": 3,
        },
    },
}
PARTITIONS: tuple[str, ...] = ("common", "new")
METRIC_ORDER: tuple[str, ...] = tuple(compatibility.METRIC_ORDER)


def _clone_spec(
    base: suite.ExperimentSpec,
    *,
    experiment_id: str,
    description: str,
    transition_model: str,
    repair_params: dict[str, Any],
) -> suite.ExperimentSpec:
    payload = asdict(base)
    payload["experiment_id"] = str(experiment_id)
    payload["description"] = str(description)
    payload["transition_model"] = str(transition_model)
    payload["repair_params"] = {**dict(base.repair_params), **dict(repair_params)}
    return suite.ExperimentSpec(**payload)


def _build_support_partition_specs() -> list[suite.ExperimentSpec]:
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    base = spec_map[CURRENT_EXACT_EXPERIMENT_ID]
    specs: list[suite.ExperimentSpec] = []
    for diagnosed_model in SERIES_MODELS:
        for art_model in SERIES_MODELS:
            for variant_name, variant in VARIANT_CONFIGS.items():
                experiment_id = f"EXP-R10-EXACT-SP01-d{diagnosed_model}-a{art_model}-{variant_name}"
                specs.append(
                    _clone_spec(
                        base,
                        experiment_id=experiment_id,
                        description=(
                            "Merged exact support-partition stock-head refresh with "
                            f"diagnosed_series_model={diagnosed_model}, "
                            f"art_series_model={art_model}, variant={variant_name}."
                        ),
                        transition_model=str(variant["transition_model"]),
                        repair_params={
                            "diagnosed_series_model": str(diagnosed_model),
                            "art_series_model": str(art_model),
                            "flow_series_model": str(base.repair_params.get("flow_series_model") or "delta"),
                            **dict(variant["repair_params"]),
                        },
                    )
                )
    return specs


def _evaluate_spec_list(
    *,
    archive_run_id: str,
    specs: list[suite.ExperimentSpec],
) -> list[dict[str, Any]]:
    observation_rows = suite.build_quarterly_observation_rows(str(archive_run_id))
    annual_rows = suite.build_annual_anchor_rows(str(archive_run_id))
    availability = suite._build_availability_payload(str(archive_run_id))
    scoring_tiers = {"exact_observed"}
    return [
        suite._evaluate_experiment_spec(
            spec,
            observation_rows=observation_rows,
            annual_rows=annual_rows,
            availability=availability,
            scoring_tiers=scoring_tiers,
            archive_run_id=str(archive_run_id),
            quarterly_start_year=2010,
            quarterly_end_year=2025,
            quarterly_min_train_years=3,
            annual_start_year=2010,
            annual_end_year=2024,
            annual_min_train_years=5,
            horizon_years=1,
            split_local_dense=False,
            full_dense_rows=None,
        )
        for spec in specs
    ]


def _exact_quarters_by_metric(result: dict[str, Any]) -> dict[str, set[str]]:
    quarter_sets = {metric_name: set() for metric_name in METRIC_ORDER}
    for split in list(result.get("quarterly_rows") or []):
        for target_row in list(split.get("holdout_target_rows") or []):
            quarter = str(target_row.get("quarter") or "")
            for metric_name in METRIC_ORDER:
                if compatibility.hardening._metric_tier(target_row, metric_name) != "exact_observed":
                    continue
                if target_row.get(metric_name) is None:
                    continue
                quarter_sets[metric_name].add(quarter)
    return quarter_sets


def _metric_scale_map(result: dict[str, Any]) -> dict[str, float]:
    values = {metric_name: [] for metric_name in METRIC_ORDER}
    for split in list(result.get("quarterly_rows") or []):
        for target_row in list(split.get("holdout_target_rows") or []):
            for metric_name in METRIC_ORDER:
                if compatibility.hardening._metric_tier(target_row, metric_name) != "exact_observed":
                    continue
                value = target_row.get(metric_name)
                if value is None:
                    continue
                values[metric_name].append(abs(float(value)))
    return {
        metric_name: max(float(np.median(np.asarray(metric_values, dtype=np.float64))), 1.0) if metric_values else 1.0
        for metric_name, metric_values in values.items()
    }


def _top_level_support_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    quarterly_summary = dict(result.get("quarterly_summary") or {})
    return compatibility._support_rows(quarterly_summary, contract_name="exact_only", archive_variant="merged")


def _top_level_honesty_flags(result: dict[str, Any]) -> dict[str, int]:
    quarterly_summary = dict(result.get("quarterly_summary") or {})
    endpoint_summary = dict(quarterly_summary.get("endpoint_audit_summary") or {})
    return {str(key): int(value) for key, value in dict(endpoint_summary.get("suppression_honesty_flags") or {}).items()}


def _empty_metric_summary() -> dict[str, Any]:
    return {"raw_mae": None, "residual_p90": None, "count": 0}


def _partition_summary(
    result: dict[str, Any],
    *,
    common_quarters: dict[str, set[str]],
    new_quarters: dict[str, set[str]],
    metric_scales: dict[str, float],
) -> dict[str, Any]:
    partition_values: dict[str, dict[str, list[float]]] = {
        partition: {metric_name: [] for metric_name in METRIC_ORDER}
        for partition in PARTITIONS
    }
    partition_norms: dict[str, list[float]] = {partition: [] for partition in PARTITIONS}
    overall_exact: dict[str, list[float]] = {metric_name: [] for metric_name in METRIC_ORDER}
    for split in list(result.get("quarterly_rows") or []):
        targets = list(split.get("holdout_target_rows") or [])
        predictions = list(split.get("candidate_prediction_rows") or [])
        for target_row, prediction_row in zip(targets, predictions, strict=False):
            quarter = str(target_row.get("quarter") or "")
            for metric_name in METRIC_ORDER:
                if compatibility.hardening._metric_tier(target_row, metric_name) != "exact_observed":
                    continue
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                raw_error = abs(float(prediction_value) - float(target_value))
                overall_exact[metric_name].append(raw_error)
                if quarter in common_quarters[metric_name]:
                    partition = "common"
                elif quarter in new_quarters[metric_name]:
                    partition = "new"
                else:
                    continue
                partition_values[partition][metric_name].append(raw_error)
                partition_norms[partition].append(raw_error / max(float(metric_scales.get(metric_name) or 1.0), 1.0))
    payload: dict[str, Any] = {}
    for partition in PARTITIONS:
        payload[partition] = {
            "quarterly_mean_mae": float(np.mean(np.asarray(partition_norms[partition], dtype=np.float64)))
            if partition_norms[partition]
            else None,
            "by_metric": {
                metric_name: {
                    "raw_mae": float(np.mean(np.asarray(values, dtype=np.float64))) if values else None,
                    "residual_p90": float(np.quantile(np.asarray(values, dtype=np.float64), 0.9)) if values else None,
                    "count": int(len(values)),
                }
                for metric_name, values in partition_values[partition].items()
            },
        }
    payload["overall_exact"] = {
        metric_name: {
            "raw_mae": float(np.mean(np.asarray(values, dtype=np.float64))) if values else None,
            "residual_p90": float(np.quantile(np.asarray(values, dtype=np.float64), 0.9)) if values else None,
            "count": int(len(values)),
        }
        for metric_name, values in overall_exact.items()
    }
    return payload


def _safe_ratio(candidate_value: float | None, reference_value: float | None) -> float | None:
    if candidate_value is None or reference_value is None or abs(float(reference_value)) <= 1e-9:
        return None
    return float(float(candidate_value) / float(reference_value))


def _safe_improvement(candidate_value: float | None, reference_value: float | None) -> float | None:
    ratio = _safe_ratio(candidate_value, reference_value)
    if ratio is None:
        return None
    return float(1.0 - ratio)


def _support_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["metric"]): dict(row) for row in rows}


def _worsened_flags(candidate_flags: dict[str, int], reference_flags: dict[str, int]) -> dict[str, int]:
    worsened: dict[str, int] = {}
    for key in sorted(set(reference_flags) | set(candidate_flags)):
        delta = int(candidate_flags.get(key, 0)) - int(reference_flags.get(key, 0))
        if delta > 0:
            worsened[str(key)] = int(delta)
    return worsened


def _support_loss_metrics(candidate_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> list[str]:
    candidate_lookup = _support_lookup(candidate_rows)
    reference_lookup = _support_lookup(reference_rows)
    metrics: list[str] = []
    for metric_name in METRIC_ORDER:
        if int(dict(candidate_lookup.get(metric_name) or {}).get("scored_count") or 0) < int(
            dict(reference_lookup.get(metric_name) or {}).get("scored_count") or 0
        ):
            metrics.append(metric_name)
    return metrics


def _candidate_row(
    result: dict[str, Any],
    *,
    current_partition: dict[str, Any],
    current_support_rows: list[dict[str, Any]],
    current_honesty_flags: dict[str, int],
    common_quarters: dict[str, set[str]],
    new_quarters: dict[str, set[str]],
    metric_scales: dict[str, float],
) -> dict[str, Any]:
    partition = _partition_summary(
        result,
        common_quarters=common_quarters,
        new_quarters=new_quarters,
        metric_scales=metric_scales,
    )
    support_rows = _top_level_support_rows(result)
    honesty_flags = _top_level_honesty_flags(result)
    support_loss = _support_loss_metrics(support_rows, current_support_rows)
    worsened = _worsened_flags(honesty_flags, current_honesty_flags)
    common_mae_ratio = _safe_ratio(
        partition["common"]["quarterly_mean_mae"],
        current_partition["common"]["quarterly_mean_mae"],
    )
    diagnosed_common_ratio = _safe_ratio(
        dict(partition["common"]["by_metric"].get("diagnosed_plhiv") or {}).get("raw_mae"),
        dict(current_partition["common"]["by_metric"].get("diagnosed_plhiv") or {}).get("raw_mae"),
    )
    flow_common_ratio = _safe_ratio(
        dict(partition["common"]["by_metric"].get("new_diagnosed_cases_period") or {}).get("raw_mae"),
        dict(current_partition["common"]["by_metric"].get("new_diagnosed_cases_period") or {}).get("raw_mae"),
    )
    diagnosed_new_improvement = _safe_improvement(
        dict(partition["new"]["by_metric"].get("diagnosed_plhiv") or {}).get("raw_mae"),
        dict(current_partition["new"]["by_metric"].get("diagnosed_plhiv") or {}).get("raw_mae"),
    )
    art_new_improvement = _safe_improvement(
        dict(partition["new"]["by_metric"].get("alive_on_art") or {}).get("raw_mae"),
        dict(current_partition["new"]["by_metric"].get("alive_on_art") or {}).get("raw_mae"),
    )
    diagnosed_tail_improvement = _safe_improvement(
        dict(partition["overall_exact"].get("diagnosed_plhiv") or {}).get("residual_p90"),
        dict(current_partition["overall_exact"].get("diagnosed_plhiv") or {}).get("residual_p90"),
    )
    art_tail_improvement = _safe_improvement(
        dict(partition["overall_exact"].get("alive_on_art") or {}).get("residual_p90"),
        dict(current_partition["overall_exact"].get("alive_on_art") or {}).get("residual_p90"),
    )
    annual_ratio = _safe_ratio(
        float(dict(result.get("annual_summary") or {}).get("candidate_mean_incidence_error") or float("inf")),
        float(dict(current_result_annual := current_partition.get("_annual_summary") or {}).get("candidate_mean_incidence_error") or float("inf")),
    )
    gates = {
        "common_support_preserved": common_mae_ratio is not None and float(common_mae_ratio) <= 1.05,
        "common_diagnosed_preserved": diagnosed_common_ratio is not None and float(diagnosed_common_ratio) <= 1.10,
        "common_flow_preserved": flow_common_ratio is not None and float(flow_common_ratio) <= 1.10,
        "new_diagnosed_repaired": diagnosed_new_improvement is not None and float(diagnosed_new_improvement) >= 0.25,
        "new_art_repaired": art_new_improvement is not None and float(art_new_improvement) >= 0.25,
        "diagnosed_tail_repaired": diagnosed_tail_improvement is not None and float(diagnosed_tail_improvement) >= 0.20,
        "art_tail_repaired": art_tail_improvement is not None and float(art_tail_improvement) >= 0.20,
        "annual_invariant": annual_ratio is not None and float(annual_ratio) <= 1.05,
        "contract_honest": not worsened and not support_loss,
    }
    return {
        "experiment_id": str(result["experiment_id"]),
        "transition_model": str(result.get("best_candidate", {}).get("transition_model") or ""),
        "quarterly_mean_mae": float(dict(result.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")),
        "quarterly_worst_mae": float(dict(result.get("quarterly_summary") or {}).get("candidate_worst_mae") or float("inf")),
        "annual_mean_incidence_error": float(dict(result.get("annual_summary") or {}).get("candidate_mean_incidence_error") or float("inf")),
        "score_tuple": [float(value) for value in suite._score_experiment_result(result)],
        "partition": partition,
        "support_rows": support_rows,
        "honesty_flags": honesty_flags,
        "support_loss_metrics": support_loss,
        "worsened_honesty_flags": worsened,
        "comparisons": {
            "common_mae_ratio": common_mae_ratio,
            "diagnosed_common_ratio": diagnosed_common_ratio,
            "flow_common_ratio": flow_common_ratio,
            "diagnosed_new_improvement": diagnosed_new_improvement,
            "art_new_improvement": art_new_improvement,
            "diagnosed_tail_improvement": diagnosed_tail_improvement,
            "art_tail_improvement": art_tail_improvement,
            "annual_ratio": annual_ratio,
        },
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in list(row.get("score_tuple") or []))


def _plot_gate_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    gate_names = [
        "common_support_preserved",
        "common_diagnosed_preserved",
        "common_flow_preserved",
        "new_diagnosed_repaired",
        "new_art_repaired",
        "diagnosed_tail_repaired",
        "art_tail_repaired",
        "annual_invariant",
        "contract_honest",
    ]
    ordered = sorted(rows, key=lambda row: (_candidate_sort_key(row), str(row["experiment_id"])))
    labels = [str(row["experiment_id"]).replace("EXP-R10-EXACT-SP01-", "") for row in ordered]
    matrix = np.asarray(
        [[1.0 if bool(dict(row.get("gates") or {}).get(gate_name)) else 0.0 for gate_name in gate_names] for row in ordered],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(12.0, max(5.0, len(labels) * 0.36)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(gate_names)))
    ax.set_xticklabels(gate_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Exact support-partition gate matrix")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_new_quarter_improvement(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (_candidate_sort_key(row), str(row["experiment_id"])))[:12]
    labels = [str(row["experiment_id"]).replace("EXP-R10-EXACT-SP01-", "") for row in ordered]
    diagnosed = [float(dict(row.get("comparisons") or {}).get("diagnosed_new_improvement") or 0.0) for row in ordered]
    art = [float(dict(row.get("comparisons") or {}).get("art_new_improvement") or 0.0) for row in ordered]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.0, max(4.5, len(labels) * 0.42)))
    ax.barh(y - 0.18, diagnosed, height=0.34, label="Diagnosed new-quarter improvement")
    ax.barh(y + 0.18, art, height=0.34, label="ART new-quarter improvement")
    ax.axvline(0.25, color="tab:red", linestyle="--", linewidth=1.2, label="25% gate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Improvement fraction vs merged exact champion")
    ax.set_title("Exact support-partition new-quarter repair")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_common_support_ratio(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (_candidate_sort_key(row), str(row["experiment_id"])))[:12]
    labels = [str(row["experiment_id"]).replace("EXP-R10-EXACT-SP01-", "") for row in ordered]
    common = [float(dict(row.get("comparisons") or {}).get("common_mae_ratio") or 0.0) for row in ordered]
    diagnosed = [float(dict(row.get("comparisons") or {}).get("diagnosed_common_ratio") or 0.0) for row in ordered]
    flow = [float(dict(row.get("comparisons") or {}).get("flow_common_ratio") or 0.0) for row in ordered]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.0, max(4.5, len(labels) * 0.42)))
    ax.barh(y - 0.24, common, height=0.22, label="Common normalized MAE ratio")
    ax.barh(y, diagnosed, height=0.22, label="Diagnosed common raw ratio")
    ax.barh(y + 0.24, flow, height=0.22, label="Flow common raw ratio")
    ax.axvline(1.05, color="tab:red", linestyle="--", linewidth=1.2, label="1.05 gate")
    ax.axvline(1.10, color="tab:orange", linestyle=":", linewidth=1.2, label="1.10 gate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ratio vs merged exact champion")
    ax.set_title("Exact support-partition common-support preservation")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    current = dict(payload.get("merged_current_champion") or {})
    lines = [
        "# TR-V3 Current Champion Exact Support Partition Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline archive: `{payload['baseline_archive_run_id']}`",
        f"- Merged archive: `{payload['merged_archive_run_id']}`",
        f"- Candidate count: `{len(list(payload.get('candidate_rows') or []))}`",
        "",
        "## AutoResearch",
        "",
        "- Variant: `benchmark-hardening-loop`",
        "- Mutation unit: exact stock-head representation on the merged exact contract",
        "- Evaluation harness: support-partitioned common-vs-new exact quarter scoring",
        "- Keep-or-revert: keep only if a local exact refresh clears all support-partition gates",
        "",
        "## Current Champion",
        "",
        f"- Experiment: `{current.get('experiment_id', '')}`",
        f"- Common normalized MAE: `{dict(current.get('partition', {}).get('common') or {}).get('quarterly_mean_mae')}`",
        f"- New diagnosed raw MAE: `{dict(dict(current.get('partition', {}).get('new') or {}).get('by_metric', {}).get('diagnosed_plhiv', {})).get('raw_mae')}`",
        f"- New ART raw MAE: `{dict(dict(current.get('partition', {}).get('new') or {}).get('by_metric', {}).get('alive_on_art', {})).get('raw_mae')}`",
        f"- Honesty flags: `{dict(current.get('honesty_flags') or {})}`",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status', '')}`",
        f"- Winner: `{decision.get('winner_id', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "| Experiment | Common MAE ratio | New diagnosed improve | New ART improve | Diagnosed p90 improve | ART p90 improve | Annual ratio | Gates pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("candidate_rows") or []):
        comparisons = dict(row.get("comparisons") or {})
        lines.append(
            f"| `{row['experiment_id']}` | "
            f"{'' if comparisons.get('common_mae_ratio') is None else f'{float(comparisons['common_mae_ratio']):.3f}'} | "
            f"{'' if comparisons.get('diagnosed_new_improvement') is None else f'{float(comparisons['diagnosed_new_improvement']):.3f}'} | "
            f"{'' if comparisons.get('art_new_improvement') is None else f'{float(comparisons['art_new_improvement']):.3f}'} | "
            f"{'' if comparisons.get('diagnosed_tail_improvement') is None else f'{float(comparisons['diagnosed_tail_improvement']):.3f}'} | "
            f"{'' if comparisons.get('art_tail_improvement') is None else f'{float(comparisons['art_tail_improvement']):.3f}'} | "
            f"{'' if comparisons.get('annual_ratio') is None else f'{float(comparisons['annual_ratio']):.3f}'} | "
            f"`{bool(row['all_gates_pass'])}` |"
        )
    return "\n".join(lines) + "\n"


def run_tr_v3_current_champion_exact_support_partition_batch(
    *,
    run_id: str,
    merged_archive_run_id: str = DEFAULT_MERGED_ARCHIVE_RUN_ID,
    baseline_archive_run_id: str = DEFAULT_BASELINE_ARCHIVE_RUN_ID,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    current_spec = spec_map[CURRENT_EXACT_EXPERIMENT_ID]

    baseline_current = _evaluate_spec_list(archive_run_id=str(baseline_archive_run_id), specs=[current_spec])[0]
    merged_current = _evaluate_spec_list(archive_run_id=str(merged_archive_run_id), specs=[current_spec])[0]
    baseline_quarters = _exact_quarters_by_metric(baseline_current)
    merged_quarters = _exact_quarters_by_metric(merged_current)
    common_quarters = {
        metric_name: set(baseline_quarters[metric_name]) & set(merged_quarters[metric_name])
        for metric_name in METRIC_ORDER
    }
    new_quarters = {
        metric_name: set(merged_quarters[metric_name]) - set(baseline_quarters[metric_name])
        for metric_name in METRIC_ORDER
    }
    metric_scales = _metric_scale_map(merged_current)
    merged_current_partition = _partition_summary(
        merged_current,
        common_quarters=common_quarters,
        new_quarters=new_quarters,
        metric_scales=metric_scales,
    )
    merged_current_partition["_annual_summary"] = dict(merged_current.get("annual_summary") or {})
    current_support_rows = _top_level_support_rows(merged_current)
    current_honesty_flags = _top_level_honesty_flags(merged_current)

    candidate_results = _evaluate_spec_list(
        archive_run_id=str(merged_archive_run_id),
        specs=_build_support_partition_specs(),
    )
    candidate_rows = [
        _candidate_row(
            result,
            current_partition=merged_current_partition,
            current_support_rows=current_support_rows,
            current_honesty_flags=current_honesty_flags,
            common_quarters=common_quarters,
            new_quarters=new_quarters,
            metric_scales=metric_scales,
        )
        for result in candidate_results
    ]
    candidate_rows.sort(key=lambda row: (_candidate_sort_key(row), str(row["experiment_id"])))
    passing = [row for row in candidate_rows if bool(row.get("all_gates_pass"))]
    if passing:
        winner = min(passing, key=lambda row: (_candidate_sort_key(row), str(row["experiment_id"])))
        decision = {
            "status": "keep_local_exact_refresh",
            "winner_id": str(winner["experiment_id"]),
            "why": "At least one stock-head refresh candidate repaired newly admitted exact quarters without breaking common-support behavior.",
        }
    else:
        winner = None
        decision = {
            "status": "reopen_broader_model_family_exploration",
            "winner_id": "",
            "why": "No bounded exact stock-head refresh candidate cleared the support-partition gates on the merged archive.",
        }

    gate_matrix_path = analysis_dir / "exact_support_partition_gate_matrix.png"
    new_quarter_path = analysis_dir / "exact_support_partition_new_quarter_improvement.png"
    common_ratio_path = analysis_dir / "exact_support_partition_common_support_ratio.png"
    _plot_gate_matrix(candidate_rows, gate_matrix_path)
    _plot_new_quarter_improvement(candidate_rows, new_quarter_path)
    _plot_common_support_ratio(candidate_rows, common_ratio_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_archive_run_id": str(baseline_archive_run_id),
        "merged_archive_run_id": str(merged_archive_run_id),
        "common_quarters_by_metric": {metric_name: sorted(values) for metric_name, values in common_quarters.items()},
        "new_quarters_by_metric": {metric_name: sorted(values) for metric_name, values in new_quarters.items()},
        "metric_scales": metric_scales,
        "merged_current_champion": {
            "experiment_id": CURRENT_EXACT_EXPERIMENT_ID,
            "partition": merged_current_partition,
            "support_rows": current_support_rows,
            "honesty_flags": current_honesty_flags,
            "quarterly_summary": dict(merged_current.get("quarterly_summary") or {}),
            "annual_summary": dict(merged_current.get("annual_summary") or {}),
        },
        "candidate_rows": candidate_rows,
        "decision": decision,
        "artifacts": {
            "gate_matrix": gate_matrix_path.name,
            "new_quarter_improvement": new_quarter_path.name,
            "common_support_ratio": common_ratio_path.name,
        },
    }
    write_json(analysis_dir / "tr_v3_current_champion_exact_support_partition_batch_report.json", payload)
    (analysis_dir / "tr_v3_current_champion_exact_support_partition_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run support-partitioned exact stock-head refresh on the merged archive.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--merged-archive-run-id", default=DEFAULT_MERGED_ARCHIVE_RUN_ID)
    parser.add_argument("--baseline-archive-run-id", default=DEFAULT_BASELINE_ARCHIVE_RUN_ID)
    args = parser.parse_args()
    run_tr_v3_current_champion_exact_support_partition_batch(
        run_id=str(args.run_id),
        merged_archive_run_id=str(args.merged_archive_run_id),
        baseline_archive_run_id=str(args.baseline_archive_run_id),
    )


if __name__ == "__main__":
    main()
