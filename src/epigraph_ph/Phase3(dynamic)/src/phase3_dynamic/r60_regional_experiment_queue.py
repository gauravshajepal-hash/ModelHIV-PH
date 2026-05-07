from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .r52_subnational_coherence_gate import (
    _coherence_rows,
    _coherence_summary_table,
    _combined_table,
    _regional_summary_table,
)
from .r54_national_total_regional_adapter import (
    R48_DEFAULT_REPORT,
    R49_DEFAULT_REPORT,
    R51_DEFAULT_REPORT,
    _base_prediction_rows,
    _load_report,
)
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
)
from .r58_pareto_simplex_regional_ensemble import (
    CARRY_FORWARD_FAMILY,
    REFERENCE_FAMILY,
    _fit_simplex_weights,
    _pareto_candidate_families,
    _prediction_index,
)
from .runtime import ensure_dir, read_json, write_json


R60_SCHEMA_VERSION = "phase3_dynamic.r60_regional_experiment_queue.v1"
R60_RUN_ID = "p3d-r60-regional-experiment-queue-20260503-s00"
R60_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R60_RUN_ID
    / "analysis"
    / "r60_regional_experiment_queue_report.json"
)
R54_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r54-national-total-regional-adapter-20260503-s00"
    / "analysis"
    / "r54_national_total_regional_adapter_report.json"
)
R59_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r59-anchor-stable-pareto-ensemble-20260503-s00"
    / "analysis"
    / "r59_anchor_stable_pareto_ensemble_report.json"
)
R60_OPTIONAL_ANCHOR_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
R60_CANDIDATE_SET_KINDS: tuple[str, ...] = (
    "pareto",
    "carry_nonregress_all",
    "r52_nonregress_regional_and_carry_mass_share",
    "top5",
    "top10",
    "all",
)


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _metric_subset_strings(metrics: tuple[str, ...]) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for count in range(len(metrics) + 1):
        subsets.extend(tuple(combo) for combo in itertools.combinations(metrics, count))
    return subsets


def _experiment_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    sequence = 1
    for candidate_set in R60_CANDIDATE_SET_KINDS:
        for optional_anchors in _metric_subset_strings(R60_OPTIONAL_ANCHOR_METRICS):
            anchor_metrics = ("estimated_plhiv", *optional_anchors)
            specs.append(
                {
                    "experiment_id": f"R60-{sequence:03d}",
                    "mode": "anchor_simplex",
                    "candidate_set": candidate_set,
                    "anchor_metrics": list(anchor_metrics),
                    "description": (
                        "Anchor selected stock/observability streams to module-local regional support; "
                        f"fit simplex ensemble over {candidate_set} for remaining streams."
                    ),
                }
            )
            sequence += 1
    for candidate_set in ("pareto", "carry_nonregress_all", "top10"):
        for anchor_metrics in (
            ("estimated_plhiv",),
            ("estimated_plhiv", "diagnosed_plhiv"),
            ("estimated_plhiv", "diagnosed_plhiv", "alive_on_art"),
        ):
            for rate_family in (REFERENCE_FAMILY, CARRY_FORWARD_FAMILY, "r54_promoted"):
                specs.append(
                    {
                        "experiment_id": f"R60-{sequence:03d}",
                        "mode": "conditional_backhalf_rate",
                        "candidate_set": candidate_set,
                        "anchor_metrics": list(anchor_metrics),
                        "rate_family": rate_family,
                        "description": (
                            "Fit anchor/simplex front-half predictions, then force VL-tested/ART and "
                            f"suppressed/VL-tested conditional rates from {rate_family}."
                        ),
                    }
                )
                sequence += 1
    for fallback_scope in ("all_metrics", "backhalf_only"):
        for fallback_family in (REFERENCE_FAMILY, CARRY_FORWARD_FAMILY):
            specs.append(
                {
                    "experiment_id": f"R60-{sequence:03d}",
                    "mode": "previous_failure_fallback",
                    "base_family": "anchor_stable_pareto_ensemble",
                    "fallback_scope": fallback_scope,
                    "fallback_family": fallback_family,
                    "description": (
                        "Use R59 unless the same metric failed the previous split-stability check; "
                        f"then fallback to {fallback_family} for {fallback_scope}."
                    ),
                }
            )
            sequence += 1
    return specs


def _periods(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")}, key=quarter_sort_key)


def _family_row(combined_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    return next((dict(row) for row in combined_rows if str(row.get("candidate_family") or "") == family), {})


def _candidate_set_families(
    combined_rows: list[dict[str, Any]],
    *,
    kind: str,
) -> list[str]:
    rows = [dict(row) for row in combined_rows if row.get("candidate_family")]
    by_family = {str(row.get("candidate_family") or ""): row for row in rows}
    if kind == "pareto":
        return _pareto_candidate_families(rows)
    if kind == "top5":
        return [str(row.get("candidate_family") or "") for row in rows[:5]]
    if kind == "top10":
        return [str(row.get("candidate_family") or "") for row in rows[:10]]
    if kind == "all":
        return [str(row.get("candidate_family") or "") for row in rows]
    carry = by_family.get(CARRY_FORWARD_FAMILY) or {}
    reference = by_family.get(REFERENCE_FAMILY) or {}
    if kind == "carry_nonregress_all":
        return [
            str(row.get("candidate_family") or "")
            for row in rows
            if _finite_float(row.get("mean_regional_normalized_absolute_error")) is not None
            and _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error")) is not None
            and _finite_float(row.get("mean_regional_share_half_l1_error")) is not None
            and float(row.get("mean_regional_normalized_absolute_error")) <= float(carry.get("mean_regional_normalized_absolute_error") or float("inf"))
            and float(row.get("mean_aggregate_mass_normalized_absolute_error")) <= float(carry.get("mean_aggregate_mass_normalized_absolute_error") or float("inf"))
            and float(row.get("mean_regional_share_half_l1_error")) <= float(carry.get("mean_regional_share_half_l1_error") or float("inf"))
        ]
    if kind == "r52_nonregress_regional_and_carry_mass_share":
        return [
            str(row.get("candidate_family") or "")
            for row in rows
            if _finite_float(row.get("mean_regional_normalized_absolute_error")) is not None
            and _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error")) is not None
            and _finite_float(row.get("mean_regional_share_half_l1_error")) is not None
            and float(row.get("mean_regional_normalized_absolute_error")) <= float(reference.get("mean_regional_normalized_absolute_error") or float("inf"))
            and float(row.get("mean_aggregate_mass_normalized_absolute_error")) <= float(carry.get("mean_aggregate_mass_normalized_absolute_error") or float("inf"))
            and float(row.get("mean_regional_share_half_l1_error")) <= float(carry.get("mean_regional_share_half_l1_error") or float("inf"))
        ]
    raise ValueError(f"unknown candidate set: {kind}")


def _default_weights(families: list[str], default_family: str) -> np.ndarray:
    weights = np.zeros(len(families), dtype=np.float64)
    if not families:
        return weights
    if default_family in families:
        weights[families.index(default_family)] = 1.0
    else:
        weights[0] = 1.0
    return weights


def _available_families(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    *,
    period: str,
    regions: list[str],
    families: list[str],
    metric: str,
) -> list[str]:
    return [
        family
        for family in families
        if all((family, period, region) in indexed and _finite_float(indexed[(family, period, region)].get(metric)) is not None for region in regions)
    ]


def _prior_design(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    prior_periods: list[str],
    regions: list[str],
    families: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    design_rows: list[list[float]] = []
    target_values: list[float] = []
    for period in prior_periods:
        for region in regions:
            target = rows_by_key.get((period, region))
            target_value = _finite_float((target or {}).get(metric))
            if target_value is None:
                continue
            values: list[float] = []
            complete = True
            for family in families:
                value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
                if value is None:
                    complete = False
                    break
                values.append(float(value))
            if complete:
                design_rows.append(values)
                target_values.append(float(target_value))
    return (
        np.asarray(design_rows, dtype=np.float64).reshape((len(design_rows), len(families))),
        np.asarray(target_values, dtype=np.float64),
    )


def _anchor_simplex_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    candidate_families: list[str],
    default_family: str,
    anchor_metrics: tuple[str, ...],
    candidate_family: str,
    anchor_family: str = REFERENCE_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    periods = _periods(prediction_rows)
    output_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if family == anchor_family and candidate_period == period})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        period_rows: dict[str, dict[str, Any]] = {
            region: {"candidate_family": candidate_family, "holdout_period": period, "region": region}
            for region in regions
        }
        prior_periods = [prior for prior in periods if quarter_sort_key(prior) < quarter_sort_key(period)]
        for metric in COUNT_METRICS:
            if metric in anchor_metrics:
                for region in regions:
                    value = _finite_float((indexed.get((anchor_family, period, region)) or {}).get(metric))
                    if value is not None:
                        period_rows[region][metric] = float(value)
                weight_rows.append(
                    {
                        "candidate_family": candidate_family,
                        "holdout_period": period,
                        "metric_name": metric,
                        "weight_status": "anchor_metric_reference_family",
                        "candidate_count": 1,
                        "prior_period_count": len(prior_periods),
                        "prior_design_row_count": None,
                        "weight_by_candidate_family": {anchor_family: 1.0},
                    }
                )
                continue
            available = _available_families(
                indexed,
                period=period,
                regions=regions,
                families=candidate_families,
                metric=metric,
            )
            weights = _default_weights(available, default_family)
            design, target = _prior_design(
                indexed,
                rows_by_key,
                prior_periods=prior_periods,
                regions=regions,
                families=available,
                metric=metric,
            )
            fitted = False
            if available and design.shape[0] >= design.shape[1] and design.shape[1] > 0:
                weights = _fit_simplex_weights(design, target)
                fitted = True
            weight_payload = {family: float(weights[index]) for index, family in enumerate(available)}
            weight_rows.append(
                {
                    "candidate_family": candidate_family,
                    "holdout_period": period,
                    "metric_name": metric,
                    "weight_status": "simplex_fitted" if fitted else "default_reference_weight",
                    "candidate_count": len(available),
                    "prior_period_count": len(prior_periods),
                    "prior_design_row_count": int(design.shape[0]),
                    "weight_by_candidate_family": weight_payload,
                }
            )
            for region in regions:
                value = 0.0
                for index, family in enumerate(available):
                    metric_value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
                    if metric_value is not None:
                        value += float(weights[index]) * float(metric_value)
                period_rows[region][metric] = float(value)
        for region in regions:
            output_rows.append(_projection_cascade(period_rows[region]))
    return output_rows, weight_rows


def _apply_conditional_backhalf_rates(
    prediction_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    *,
    rate_family: str,
    r54_promoted_family: str,
) -> list[dict[str, Any]]:
    indexed = _prediction_index(source_rows)
    resolved_rate_family = r54_promoted_family if rate_family == "r54_promoted" else rate_family
    rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        adjusted = dict(row)
        period = str(row.get("holdout_period") or "")
        region = str(row.get("region") or "")
        rate_row = indexed.get((resolved_rate_family, period, region)) or {}
        art = max(float(adjusted.get("alive_on_art") or 0.0), 0.0)
        rate_art = max(float(rate_row.get("alive_on_art") or 0.0), 0.0)
        rate_tested = max(float(rate_row.get("tested_for_viral_load") or 0.0), 0.0)
        rate_suppressed = max(float(rate_row.get("virally_suppressed") or 0.0), 0.0)
        testing_rate = rate_tested / rate_art if rate_art > 0.0 else 0.0
        suppression_rate = rate_suppressed / rate_tested if rate_tested > 0.0 else 0.0
        adjusted["tested_for_viral_load"] = float(art * min(max(testing_rate, 0.0), 1.0))
        adjusted["virally_suppressed"] = float(adjusted["tested_for_viral_load"] * min(max(suppression_rate, 0.0), 1.0))
        adjusted["backhalf_rate_source_family"] = resolved_rate_family
        rows.append(_projection_cascade(adjusted))
    return rows


def _previous_failure_fallback_rows(
    source_rows: list[dict[str, Any]],
    r59_report: dict[str, Any],
    *,
    candidate_family: str,
    fallback_scope: str,
    fallback_family: str,
) -> list[dict[str, Any]]:
    indexed = _prediction_index(source_rows)
    periods = sorted({period for _family, period, _region in indexed if period}, key=quarter_sort_key)
    checks = (
        "regional_nonregression_vs_carry_forward",
        "regional_nonregression_vs_reference",
        "mass_nonregression_vs_carry_forward",
        "mass_nonregression_vs_reference",
        "share_nonregression_vs_carry_forward",
        "share_nonregression_vs_reference",
    )
    prior_failed = {
        (str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): any(row.get(check) is not True for check in checks)
        for row in list(r59_report.get("split_comparison_rows") or [])
    }
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(periods):
        regions = sorted({region for family, candidate_period, region in indexed if family == "anchor_stable_pareto_ensemble" and candidate_period == period})
        for region in regions:
            output: dict[str, Any] = {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
            }
            for metric in COUNT_METRICS:
                family = "anchor_stable_pareto_ensemble"
                if period_index > 0:
                    previous_period = periods[period_index - 1]
                    failed = prior_failed.get((previous_period, metric), False)
                    in_scope = fallback_scope == "all_metrics" or metric in {"tested_for_viral_load", "virally_suppressed"}
                    if failed and in_scope:
                        family = fallback_family
                value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
                if value is not None:
                    output[metric] = float(value)
                    output[f"{metric}_source_family"] = family
            rows.append(_projection_cascade(output))
    return rows


def _prediction_rows_for_spec(
    spec: dict[str, Any],
    *,
    source_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    r54_promoted_family: str,
    r59_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family = str(spec["experiment_id"]) + "__" + str(spec["mode"])
    mode = str(spec.get("mode") or "")
    if mode in {"anchor_simplex", "conditional_backhalf_rate"}:
        candidate_families = _candidate_set_families(combined_rows, kind=str(spec.get("candidate_set") or "pareto"))
        prediction_rows, weight_rows = _anchor_simplex_prediction_rows(
            source_rows,
            rows_by_key,
            candidate_families=candidate_families,
            default_family=r54_promoted_family,
            anchor_metrics=tuple(str(metric) for metric in list(spec.get("anchor_metrics") or [])),
            candidate_family=family,
        )
        if mode == "conditional_backhalf_rate":
            prediction_rows = _apply_conditional_backhalf_rates(
                prediction_rows,
                source_rows,
                rate_family=str(spec.get("rate_family") or REFERENCE_FAMILY),
                r54_promoted_family=r54_promoted_family,
            )
            for row in weight_rows:
                row["conditional_backhalf_rate_family"] = spec.get("rate_family")
        return prediction_rows, weight_rows
    if mode == "previous_failure_fallback":
        return (
            _previous_failure_fallback_rows(
                source_rows,
                r59_report,
                candidate_family=family,
                fallback_scope=str(spec.get("fallback_scope") or "all_metrics"),
                fallback_family=str(spec.get("fallback_family") or REFERENCE_FAMILY),
            ),
            [],
        )
    raise ValueError(f"unknown experiment mode: {mode}")


def _score_candidate(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence_rows = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence_rows))
    family = str(prediction_rows[0].get("candidate_family") or "") if prediction_rows else ""
    return split_rows, coherence_rows, _family_row(combined, family)


def _candidate_mean_gate(
    row: dict[str, Any],
    *,
    carry: dict[str, Any],
    reference: dict[str, Any],
    r54_gate: dict[str, Any],
    r59_gate: dict[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    regional = _finite_float(row.get("mean_regional_normalized_absolute_error"))
    mass = _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error"))
    share = _finite_float(row.get("mean_regional_share_half_l1_error"))
    checks = [
        ("carry_forward_regional", regional, _finite_float(carry.get("mean_regional_normalized_absolute_error")), True),
        ("carry_forward_mass", mass, _finite_float(carry.get("mean_aggregate_mass_normalized_absolute_error")), False),
        ("carry_forward_share", share, _finite_float(carry.get("mean_regional_share_half_l1_error")), False),
        ("r52_regional", regional, _finite_float(reference.get("mean_regional_normalized_absolute_error")), False),
        ("r52_mass", mass, _finite_float(reference.get("mean_aggregate_mass_normalized_absolute_error")), False),
        ("r52_share", share, _finite_float(reference.get("mean_regional_share_half_l1_error")), False),
        ("r54_regional", regional, _finite_float(r54_gate.get("promoted_mean_regional_normalized_absolute_error")), False),
        ("r54_mass", mass, _finite_float(r54_gate.get("promoted_mean_aggregate_mass_normalized_absolute_error")), False),
        ("r54_share", share, _finite_float(r54_gate.get("promoted_mean_regional_share_half_l1_error")), False),
        ("r59_regional", regional, _finite_float(r59_gate.get("ensemble_mean_regional_normalized_absolute_error")), False),
        ("r59_mass", mass, _finite_float(r59_gate.get("ensemble_mean_aggregate_mass_normalized_absolute_error")), False),
        ("r59_share", share, _finite_float(r59_gate.get("ensemble_mean_regional_share_half_l1_error")), False),
    ]
    for name, left, right, strict in checks:
        if left is None or right is None:
            blockers.append(f"{name}_missing")
        elif strict and not float(left) < float(right):
            blockers.append(f"{name}_not_improved")
        elif not strict and float(left) > float(right):
            blockers.append(f"{name}_regressed")
    return not blockers, blockers


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("queue_gate") or {})
    lines = [
        "# Phase 3 R60 Regional Experiment Queue",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Experiment count: `{gate.get('experiment_count')}`",
        f"- Mean-pass count: `{gate.get('mean_pass_count')}`",
        f"- Strict-pass count: `{gate.get('strict_pass_count')}`",
        f"- Best candidate: `{gate.get('best_candidate_family')}`",
        f"- Best regional / mass / share errors: `{gate.get('best_mean_regional_normalized_absolute_error')}` / `{gate.get('best_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('best_mean_regional_share_half_l1_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Top Experiments",
        "",
        "| Rank | Experiment | Mode | Regional NAE | Mass NAE | Share half-L1 | Mean gate | Split gate |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(list(report.get("experiment_score_rows") or [])[:20], start=1):
        lines.append(
            f"| {index} | `{row.get('experiment_id')}` | `{row.get('mode')}` | "
            f"{float(row.get('mean_regional_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_aggregate_mass_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_regional_share_half_l1_error') or 0.0):.6f} | "
            f"`{row.get('mean_gate_status')}` | `{row.get('split_gate_status')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    top = list(rows)[:15]
    labels = [str(row.get("experiment_id") or "") for row in top]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in top]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in top]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in top]
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R60 queued regional experiments")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r60_regional_experiment_queue(
    *,
    run_id: str = R60_RUN_ID,
    r54_report_path: Path | None = None,
    r59_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r59_path = Path(r59_report_path) if r59_report_path is not None else R59_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r59 = _read_report(r59_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    source_rows = (
        _base_prediction_rows(r48, r49, r51)
        + [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
        + [dict(row) for row in list(r59.get("prediction_rows") or [])]
    )
    r54_promoted = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or "")
    specs = _experiment_specs()
    all_split_rows = [dict(row) for row in list(r54.get("split_metric_score_rows") or [])]
    all_coherence_rows = [dict(row) for row in list(r54.get("coherence_rows") or [])]
    experiment_score_rows: list[dict[str, Any]] = []
    experiment_spec_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    split_comparison_rows: list[dict[str, Any]] = []
    carry = _family_row(list(r54.get("combined_candidate_table") or []), CARRY_FORWARD_FAMILY)
    reference = _family_row(list(r54.get("combined_candidate_table") or []), REFERENCE_FAMILY)
    for spec in specs:
        experiment_spec_rows.append(dict(spec))
        prediction_rows, weights = _prediction_rows_for_spec(
            spec,
            source_rows=source_rows,
            rows_by_key=rows_by_key,
            combined_rows=[dict(row) for row in list(r54.get("combined_candidate_table") or [])],
            r54_promoted_family=r54_promoted,
            r59_report=r59,
        )
        split_rows, coherence_rows, family_score = _score_candidate(rows_by_key, prediction_rows)
        all_split_rows.extend(split_rows)
        all_coherence_rows.extend(coherence_rows)
        weight_rows.extend(weights)
        family = str(family_score.get("candidate_family") or "")
        comparison = _split_comparison_rows(
            regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + split_rows,
            coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])] + coherence_rows,
            promoted_family=family,
        )
        split_gate = _split_stability_gate(comparison)
        split_comparison_rows.extend(comparison)
        mean_pass, mean_blockers = _candidate_mean_gate(
            family_score,
            carry=carry,
            reference=reference,
            r54_gate=dict(r54.get("adapter_gate") or {}),
            r59_gate=dict(r59.get("ensemble_gate") or {}),
        )
        experiment_score_rows.append(
            {
                **dict(spec),
                "candidate_family": family,
                "mean_regional_normalized_absolute_error": family_score.get("mean_regional_normalized_absolute_error"),
                "mean_aggregate_mass_normalized_absolute_error": family_score.get("mean_aggregate_mass_normalized_absolute_error"),
                "mean_regional_share_half_l1_error": family_score.get("mean_regional_share_half_l1_error"),
                "worst_regional_normalized_absolute_error": family_score.get("worst_regional_normalized_absolute_error"),
                "mean_gate_status": "pass" if mean_pass else "fail",
                "mean_gate_blockers": mean_blockers,
                "split_gate_status": split_gate.get("status"),
                "split_gate_failure_counts": split_gate.get("failure_counts"),
                "split_metric_count": split_gate.get("split_metric_count"),
            }
        )
    experiment_score_rows.sort(
        key=lambda row: (
            0 if row.get("mean_gate_status") == "pass" else 1,
            float(row.get("mean_regional_normalized_absolute_error") or float("inf")),
            float(row.get("mean_aggregate_mass_normalized_absolute_error") or float("inf")),
            float(row.get("mean_regional_share_half_l1_error") or float("inf")),
            str(row.get("experiment_id") or ""),
        )
    )
    mean_pass_rows = [row for row in experiment_score_rows if row.get("mean_gate_status") == "pass"]
    strict_pass_rows = [
        row
        for row in mean_pass_rows
        if row.get("split_gate_status") == "strict_split_stable_adapter_promoted"
    ]
    best = strict_pass_rows[0] if strict_pass_rows else (mean_pass_rows[0] if mean_pass_rows else (experiment_score_rows[0] if experiment_score_rows else {}))
    blockers: list[str] = []
    if not strict_pass_rows:
        blockers.append("no_experiment_passed_strict_split_gate")
    if not mean_pass_rows:
        blockers.append("no_experiment_beat_r59_mean_contract")
    if strict_pass_rows:
        status = "regional_experiment_queue_strict_champion_promoted"
    elif mean_pass_rows:
        status = "regional_experiment_queue_mean_promoted_split_limited"
    else:
        status = "regional_experiment_queue_diagnostic_only"
    gate = {
        "status": status,
        "blockers": blockers,
        "experiment_count": len(experiment_score_rows),
        "mean_pass_count": len(mean_pass_rows),
        "strict_pass_count": len(strict_pass_rows),
        "best_experiment_id": best.get("experiment_id"),
        "best_candidate_family": best.get("candidate_family"),
        "best_mean_regional_normalized_absolute_error": best.get("mean_regional_normalized_absolute_error"),
        "best_mean_aggregate_mass_normalized_absolute_error": best.get("mean_aggregate_mass_normalized_absolute_error"),
        "best_mean_regional_share_half_l1_error": best.get("mean_regional_share_half_l1_error"),
        "best_split_gate_status": best.get("split_gate_status"),
        "contract": (
            "R60 executes a broad train-origin regional experiment queue. Candidates may promote only if they beat "
            "carry-forward, R52, R54, and the current R59 mean contract. Strict promotion additionally requires the "
            "R55-style split nonregression gate across period-metric splits."
        ),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r60_regional_experiment_queue_report.json"
    md_path = analysis_dir / "r60_regional_experiment_queue_report.md"
    spec_csv = analysis_dir / "r60_experiment_specs.csv"
    score_csv = analysis_dir / "r60_experiment_score_rows.csv"
    weight_csv = analysis_dir / "r60_weight_rows.csv"
    comparison_csv = analysis_dir / "r60_split_comparison_rows.csv"
    dashboard_path = analysis_dir / "r60_regional_experiment_queue_dashboard.png"
    report = {
        "schema_version": R60_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "verdict": (
            "R60 found a strict regional champion."
            if strict_pass_rows
            else (
                "R60 found mean-level improvements over R59, but no strict split-stable champion."
                if mean_pass_rows
                else "R60 found no candidate that improves the current R59 mean contract."
            )
        ),
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "r59_report_path": r59_path.as_posix(),
        "queue_gate": gate,
        "experiment_specs": experiment_spec_rows,
        "experiment_score_rows": experiment_score_rows,
        "weight_rows": weight_rows,
        "split_comparison_rows": split_comparison_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "experiment_specs_csv": spec_csv.as_posix(),
            "experiment_score_csv": score_csv.as_posix(),
            "weight_csv": weight_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(spec_csv, experiment_spec_rows)
    _write_csv(score_csv, experiment_score_rows)
    _write_csv(weight_csv, weight_rows)
    _write_csv(comparison_csv, split_comparison_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, experiment_score_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R60 regional experiment queue.")
    parser.add_argument("--run-id", default=R60_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r59-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r60_regional_experiment_queue(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r59_report_path=None if args.r59_report_path is None else Path(args.r59_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
