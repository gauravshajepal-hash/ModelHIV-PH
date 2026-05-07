from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
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
    R52_DEFAULT_REPORT,
    _base_prediction_rows,
    _load_report,
)
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
)
from .runtime import ensure_dir, read_json, write_json


R56_SCHEMA_VERSION = "phase3_dynamic.r56_split_guarded_regional_selector.v1"
R56_RUN_ID = "p3d-r56-split-guarded-regional-selector-20260503-s00"
SELECTOR_FAMILY = "split_guarded_regional_selector"
DEFAULT_REFERENCE_FAMILY = "module_local_selector"
CARRY_FORWARD_FAMILY = "regional_carry_forward"
R54_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r54-national-total-regional-adapter-20260503-s00"
    / "analysis"
    / "r54_national_total_regional_adapter_report.json"
)


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _regional_score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): dict(row)
        for row in rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("metric_name")
    }


def _coherence_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return _regional_score_index(rows)


def _nonregression(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return float(left) <= float(right) + sys.float_info.epsilon * max(1.0, abs(float(right)))


def _periods_from_predictions(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")},
        key=quarter_sort_key,
    )


def _candidate_families_for_regions(
    indexed_predictions: dict[tuple[str, str, str], dict[str, Any]],
    *,
    period: str,
    regions: list[str],
    candidate_families: list[str],
) -> list[str]:
    return [
        family
        for family in candidate_families
        if all((family, period, region) in indexed_predictions for region in regions)
    ]


def _error_values(
    *,
    regional_index: dict[tuple[str, str, str], dict[str, Any]],
    coherence: dict[tuple[str, str, str], dict[str, Any]],
    family: str,
    period: str,
    metric: str,
) -> tuple[float | None, float | None, float | None]:
    regional = _finite_float((regional_index.get((family, period, metric)) or {}).get("normalized_absolute_error"))
    mass = _finite_float((coherence.get((family, period, metric)) or {}).get("aggregate_mass_normalized_absolute_error"))
    share = _finite_float((coherence.get((family, period, metric)) or {}).get("regional_share_half_l1_error"))
    return regional, mass, share


def _prior_candidate_score_rows(
    *,
    candidate_families: list[str],
    regional_score_rows: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    holdout_period: str,
    metric: str,
    carry_family: str = CARRY_FORWARD_FAMILY,
    reference_family: str = DEFAULT_REFERENCE_FAMILY,
) -> list[dict[str, Any]]:
    periods = sorted(
        {
            str(row.get("holdout_period") or "")
            for row in regional_score_rows
            if row.get("holdout_period") and quarter_sort_key(str(row.get("holdout_period"))) < quarter_sort_key(holdout_period)
        },
        key=quarter_sort_key,
    )
    regional_index = _regional_score_index(regional_score_rows)
    coherence_index = _coherence_index(coherence_rows)
    rows: list[dict[str, Any]] = []
    for family in candidate_families:
        regional_values: list[float] = []
        mass_values: list[float] = []
        share_values: list[float] = []
        strict_pass_count = 0
        failed_check_count = 0
        scored_period_count = 0
        for period in periods:
            regional, mass, share = _error_values(
                regional_index=regional_index,
                coherence=coherence_index,
                family=family,
                period=period,
                metric=metric,
            )
            if regional is None or mass is None or share is None:
                continue
            carry_regional, carry_mass, carry_share = _error_values(
                regional_index=regional_index,
                coherence=coherence_index,
                family=carry_family,
                period=period,
                metric=metric,
            )
            reference_regional, reference_mass, reference_share = _error_values(
                regional_index=regional_index,
                coherence=coherence_index,
                family=reference_family,
                period=period,
                metric=metric,
            )
            checks = (
                _nonregression(regional, carry_regional),
                _nonregression(regional, reference_regional),
                _nonregression(mass, carry_mass),
                _nonregression(mass, reference_mass),
                _nonregression(share, carry_share),
                _nonregression(share, reference_share),
            )
            strict = all(checks)
            strict_pass_count += int(strict)
            failed_check_count += sum(1 for check in checks if not check)
            scored_period_count += 1
            regional_values.append(float(regional))
            mass_values.append(float(mass))
            share_values.append(float(share))
        if not scored_period_count:
            continue
        rows.append(
            {
                "candidate_family": family,
                "holdout_period": holdout_period,
                "metric_name": metric,
                "prior_scored_period_count": scored_period_count,
                "prior_strict_pass_count": strict_pass_count,
                "prior_failed_check_count": failed_check_count,
                "prior_mean_regional_normalized_absolute_error": float(np.mean(np.asarray(regional_values, dtype=np.float64))),
                "prior_mean_aggregate_mass_normalized_absolute_error": float(np.mean(np.asarray(mass_values, dtype=np.float64))),
                "prior_mean_regional_share_half_l1_error": float(np.mean(np.asarray(share_values, dtype=np.float64))),
            }
        )
    return rows


def _select_family_from_prior_scores(
    prior_rows: list[dict[str, Any]],
    *,
    default_family: str = DEFAULT_REFERENCE_FAMILY,
) -> tuple[str, str]:
    if not prior_rows:
        return default_family, "default_no_prior_split"
    ranked = sorted(
        prior_rows,
        key=lambda row: (
            0
            if int(row.get("prior_strict_pass_count") or 0) == int(row.get("prior_scored_period_count") or 0)
            else 1,
            int(row.get("prior_failed_check_count") or 0),
            float(row.get("prior_mean_regional_normalized_absolute_error") or float("inf")),
            float(row.get("prior_mean_aggregate_mass_normalized_absolute_error") or float("inf")),
            float(row.get("prior_mean_regional_share_half_l1_error") or float("inf")),
            str(row.get("candidate_family") or ""),
        ),
    )
    chosen = str(ranked[0].get("candidate_family") or default_family)
    reason = (
        "prior_all_splits_strict_nonregression"
        if int(ranked[0].get("prior_strict_pass_count") or 0) == int(ranked[0].get("prior_scored_period_count") or 0)
        else "prior_lowest_failure_then_lexicographic_error"
    )
    return chosen, reason


def _split_guarded_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    regional_score_rows: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    candidate_families: list[str],
    selector_family: str = SELECTOR_FAMILY,
    default_family: str = DEFAULT_REFERENCE_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    periods = _periods_from_predictions(prediction_rows)
    selection_rows: list[dict[str, Any]] = []
    prior_score_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if family == default_family and candidate_period == period})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        period_rows: dict[str, dict[str, Any]] = {
            region: {
                "candidate_family": selector_family,
                "holdout_period": period,
                "region": region,
            }
            for region in regions
        }
        available_families = _candidate_families_for_regions(
            indexed,
            period=period,
            regions=regions,
            candidate_families=candidate_families,
        )
        for metric in COUNT_METRICS:
            prior_rows = _prior_candidate_score_rows(
                candidate_families=available_families,
                regional_score_rows=regional_score_rows,
                coherence_rows=coherence_rows,
                holdout_period=period,
                metric=metric,
            )
            prior_score_rows.extend(prior_rows)
            selected_family, reason = _select_family_from_prior_scores(
                prior_rows,
                default_family=default_family,
            )
            if selected_family not in available_families:
                selected_family = default_family if default_family in available_families else available_families[0]
                reason = "fallback_selected_family_unavailable"
            selection_rows.append(
                {
                    "holdout_period": period,
                    "metric_name": metric,
                    "selected_candidate_family": selected_family,
                    "selection_reason": reason,
                    "available_candidate_count": len(available_families),
                    "prior_candidate_count": len(prior_rows),
                }
            )
            for region in regions:
                selected = indexed.get((selected_family, period, region))
                if selected is None:
                    continue
                value = _finite_float(selected.get(metric))
                if value is None:
                    continue
                period_rows[region][metric] = float(value)
                period_rows[region][f"{metric}_selected_candidate_family"] = selected_family
        for region in regions:
            raw = period_rows[region]
            projected = _projection_cascade(raw)
            projected["projection_adjusted"] = any(
                abs(float(projected.get(metric) or 0.0) - float(raw.get(metric) or 0.0)) > 1e-8
                for metric in COUNT_METRICS
            )
            output_rows.append(projected)
    return output_rows, selection_rows, prior_score_rows


def _score_selector_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence))
    return split_rows, coherence, combined


def _combined_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_family") or ""): dict(row) for row in rows if row.get("candidate_family")}


def _gate(
    combined_rows: list[dict[str, Any]],
    *,
    split_gate: dict[str, Any],
    r54_gate: dict[str, Any] | None = None,
    selector_family: str = SELECTOR_FAMILY,
) -> dict[str, Any]:
    by_family = _combined_index(combined_rows)
    selector = by_family.get(selector_family) or {}
    carry = by_family.get(CARRY_FORWARD_FAMILY) or {}
    reference = by_family.get(DEFAULT_REFERENCE_FAMILY) or {}
    r54 = dict(r54_gate or {})
    selector_regional = _finite_float(selector.get("mean_regional_normalized_absolute_error"))
    selector_mass = _finite_float(selector.get("mean_aggregate_mass_normalized_absolute_error"))
    selector_share = _finite_float(selector.get("mean_regional_share_half_l1_error"))
    carry_regional = _finite_float(carry.get("mean_regional_normalized_absolute_error"))
    carry_mass = _finite_float(carry.get("mean_aggregate_mass_normalized_absolute_error"))
    carry_share = _finite_float(carry.get("mean_regional_share_half_l1_error"))
    reference_regional = _finite_float(reference.get("mean_regional_normalized_absolute_error"))
    reference_mass = _finite_float(reference.get("mean_aggregate_mass_normalized_absolute_error"))
    reference_share = _finite_float(reference.get("mean_regional_share_half_l1_error"))
    r54_regional = _finite_float(r54.get("promoted_mean_regional_normalized_absolute_error"))
    r54_mass = _finite_float(r54.get("promoted_mean_aggregate_mass_normalized_absolute_error"))
    r54_share = _finite_float(r54.get("promoted_mean_regional_share_half_l1_error"))
    blockers: list[str] = []
    if selector_regional is None or selector_mass is None or selector_share is None:
        blockers.append("selector_not_scored")
    if carry_regional is not None and selector_regional is not None and not selector_regional < carry_regional:
        blockers.append("selector_does_not_beat_carry_forward_regional_error")
    if carry_mass is not None and selector_mass is not None and selector_mass > carry_mass:
        blockers.append("selector_worsens_carry_forward_mass_error")
    if carry_share is not None and selector_share is not None and selector_share > carry_share:
        blockers.append("selector_worsens_carry_forward_share_error")
    if reference_regional is not None and selector_regional is not None and selector_regional > reference_regional:
        blockers.append("selector_worsens_r52_reference_regional_error")
    if reference_mass is not None and selector_mass is not None and selector_mass > reference_mass:
        blockers.append("selector_worsens_r52_reference_mass_error")
    if reference_share is not None and selector_share is not None and selector_share > reference_share:
        blockers.append("selector_worsens_r52_reference_share_error")
    if r54_regional is not None and selector_regional is not None and selector_regional > r54_regional:
        blockers.append("selector_does_not_beat_r54_regional_error")
    if r54_mass is not None and selector_mass is not None and selector_mass > r54_mass:
        blockers.append("selector_does_not_beat_r54_mass_error")
    if r54_share is not None and selector_share is not None and selector_share > r54_share:
        blockers.append("selector_does_not_beat_r54_share_error")
    if split_gate.get("status") != "strict_split_stable_adapter_promoted":
        blockers.append("selector_not_strict_split_stable")
    status = "split_guarded_regional_selector_promoted" if not blockers else "split_guarded_regional_selector_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "selector_family": selector_family,
        "selector_mean_regional_normalized_absolute_error": selector_regional,
        "selector_mean_aggregate_mass_normalized_absolute_error": selector_mass,
        "selector_mean_regional_share_half_l1_error": selector_share,
        "carry_forward_mean_regional_normalized_absolute_error": carry_regional,
        "carry_forward_mean_aggregate_mass_normalized_absolute_error": carry_mass,
        "carry_forward_mean_regional_share_half_l1_error": carry_share,
        "r52_reference_mean_regional_normalized_absolute_error": reference_regional,
        "r52_reference_mean_aggregate_mass_normalized_absolute_error": reference_mass,
        "r52_reference_mean_regional_share_half_l1_error": reference_share,
        "r54_reference_mean_regional_normalized_absolute_error": r54_regional,
        "r54_reference_mean_aggregate_mass_normalized_absolute_error": r54_mass,
        "r54_reference_mean_regional_share_half_l1_error": r54_share,
        "split_stability_status": split_gate.get("status"),
        "contract": (
            "R56 may promote only if a prior-split selector improves carry-forward, does not regress against "
            "the R52 module-local reference or R54 national-total adapter on mean regional/mass/share errors, "
            "and passes the R55-style period-metric split nonregression gate. The selector may use only earlier "
            "holdout-period diagnostics for the same cascade metric."
        ),
    }


def _selection_summary(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selection_rows:
        grouped[str(row.get("metric_name") or "")][str(row.get("selected_candidate_family") or "")] += 1
    rows: list[dict[str, Any]] = []
    for metric, counter in sorted(grouped.items()):
        for family, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "metric_name": metric,
                    "selected_candidate_family": family,
                    "selected_period_count": int(count),
                }
            )
    return rows


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
    gate = dict(report.get("selector_gate") or {})
    split_gate = dict(report.get("split_stability_gate") or {})
    lines = [
        "# Phase 3 R56 Split-Guarded Regional Selector",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Selector regional / mass / share errors: `{gate.get('selector_mean_regional_normalized_absolute_error')}` / `{gate.get('selector_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('selector_mean_regional_share_half_l1_error')}`",
        f"- R54 regional / mass / share errors: `{gate.get('r54_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_regional_share_half_l1_error')}`",
        f"- R52 regional / mass / share errors: `{gate.get('r52_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r52_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r52_reference_mean_regional_share_half_l1_error')}`",
        f"- Split-stability status: `{split_gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Selection Summary",
        "",
        "| Metric | Selected family | Period count |",
        "|---|---|---:|",
    ]
    for row in list(report.get("selection_summary_rows") or []):
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('selected_candidate_family')}` | {row.get('selected_period_count')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, combined_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    rows = [
        row
        for row in combined_rows
        if str(row.get("candidate_family") or "")
        in {
            SELECTOR_FAMILY,
            CARRY_FORWARD_FAMILY,
            DEFAULT_REFERENCE_FAMILY,
        }
        or str(row.get("candidate_family") or "").startswith("national_total_adapter__total=train_only_region_metric_selector__share=similarity_proxy_log_delta")
    ]
    labels = [str(row.get("candidate_family") or "") for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R56 split-guarded regional selector")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r56_split_guarded_regional_selector(
    *,
    run_id: str = R56_RUN_ID,
    r54_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
    r52_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r52_path = Path(r52_report_path or r54.get("r52_report_path") or R52_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r52 = _load_report(r52_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    base_rows = _base_prediction_rows(r48, r49, r51)
    adapter_rows = [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
    all_prediction_rows = base_rows + adapter_rows
    promoted_r54_family = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or "")
    candidate_families = [
        family
        for family in (
            DEFAULT_REFERENCE_FAMILY,
            promoted_r54_family,
            CARRY_FORWARD_FAMILY,
        )
        if family
    ]
    selector_rows, selection_rows, prior_score_rows = _split_guarded_prediction_rows(
        all_prediction_rows,
        regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])],
        coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])],
        candidate_families=candidate_families,
    )
    selector_split, selector_coherence, selector_combined = _score_selector_rows(rows_by_key, selector_rows)
    comparison_split_rows = [dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + selector_split
    comparison_coherence_rows = [dict(row) for row in list(r54.get("coherence_rows") or [])] + selector_coherence
    split_comparison = _split_comparison_rows(
        regional_score_rows=comparison_split_rows,
        coherence_rows=comparison_coherence_rows,
        promoted_family=SELECTOR_FAMILY,
    )
    split_gate = _split_stability_gate(split_comparison)
    combined_rows = _combined_table(
        _regional_summary_table(comparison_split_rows),
        _coherence_summary_table(comparison_coherence_rows),
    )
    gate = _gate(
        combined_rows,
        split_gate=split_gate,
        r54_gate=dict(r54.get("adapter_gate") or {}),
    )
    verdict = (
        "R56 promoted a split-guarded regional selector."
        if gate["status"] == "split_guarded_regional_selector_promoted"
        else "R56 did not beat the current R54/R52 references under the strict mean and split-stability contract."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r56_split_guarded_regional_selector_report.json"
    md_path = analysis_dir / "r56_split_guarded_regional_selector_report.md"
    selection_csv = analysis_dir / "r56_selection_rows.csv"
    prior_csv = analysis_dir / "r56_prior_score_rows.csv"
    prediction_csv = analysis_dir / "r56_prediction_rows.csv"
    split_csv = analysis_dir / "r56_split_metric_scores.csv"
    coherence_csv = analysis_dir / "r56_coherence_rows.csv"
    comparison_csv = analysis_dir / "r56_split_comparison_rows.csv"
    combined_csv = analysis_dir / "r56_combined_candidate_table.csv"
    dashboard_path = analysis_dir / "r56_split_guarded_regional_selector_dashboard.png"
    report = {
        "schema_version": R56_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r52_report_path": r52_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "candidate_families": candidate_families,
            "selector_rule": "for each holdout period and cascade metric, select among current regional references using only earlier split diagnostics for that same metric",
            "target_leakage_status": "no same-period regional target error enters the selector decision",
            "claim_limit": "regional readout stability selector only; no determinant, province-level, or causal transition claim",
        },
        "selector_gate": gate,
        "split_stability_gate": split_gate,
        "selection_summary_rows": _selection_summary(selection_rows),
        "selection_rows": selection_rows,
        "prior_score_rows": prior_score_rows,
        "prediction_rows": selector_rows,
        "split_metric_score_rows": selector_split,
        "coherence_rows": selector_coherence,
        "split_comparison_rows": split_comparison,
        "combined_candidate_table": combined_rows,
        "r52_gate_snapshot": dict(r52.get("coherence_gate") or {}),
        "r54_gate_snapshot": dict(r54.get("adapter_gate") or {}),
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "selection_csv": selection_csv.as_posix(),
            "prior_score_csv": prior_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "split_metric_scores_csv": split_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(selection_csv, selection_rows)
    _write_csv(prior_csv, prior_score_rows)
    _write_csv(prediction_csv, selector_rows)
    _write_csv(split_csv, selector_split)
    _write_csv(coherence_csv, selector_coherence)
    _write_csv(comparison_csv, split_comparison)
    _write_csv(combined_csv, combined_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R56 split-guarded regional selector.")
    parser.add_argument("--run-id", default=R56_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    parser.add_argument("--r52-report-path", default=None)
    args = parser.parse_args()
    run_r56_split_guarded_regional_selector(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
        r52_report_path=None if args.r52_report_path is None else Path(args.r52_report_path),
    )


if __name__ == "__main__":
    _main()
