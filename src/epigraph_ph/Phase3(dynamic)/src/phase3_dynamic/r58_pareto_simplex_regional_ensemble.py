from __future__ import annotations

import argparse
import csv
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
from .runtime import ensure_dir, read_json, write_json


R58_SCHEMA_VERSION = "phase3_dynamic.r58_pareto_simplex_regional_ensemble.v1"
R58_RUN_ID = "p3d-r58-pareto-simplex-regional-ensemble-20260503-s00"
ENSEMBLE_FAMILY = "pareto_simplex_regional_ensemble"
CARRY_FORWARD_FAMILY = "regional_carry_forward"
REFERENCE_FAMILY = "module_local_selector"
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


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "mean_regional_normalized_absolute_error",
        "mean_aggregate_mass_normalized_absolute_error",
        "mean_regional_share_half_l1_error",
    )
    left_values = [_finite_float(left.get(key)) for key in keys]
    right_values = [_finite_float(right.get(key)) for key in keys]
    if any(value is None for value in left_values + right_values):
        return False
    return all(float(left_values[i]) <= float(right_values[i]) for i in range(len(keys))) and any(
        float(left_values[i]) < float(right_values[i]) for i in range(len(keys))
    )


def _pareto_candidate_families(combined_rows: list[dict[str, Any]]) -> list[str]:
    scored = [
        dict(row)
        for row in combined_rows
        if row.get("candidate_family")
        and _finite_float(row.get("mean_regional_normalized_absolute_error")) is not None
        and _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error")) is not None
        and _finite_float(row.get("mean_regional_share_half_l1_error")) is not None
    ]
    frontier: list[dict[str, Any]] = []
    for row in scored:
        if not any(_dominates(other, row) for other in scored):
            frontier.append(row)
    frontier.sort(
        key=lambda row: (
            float(row.get("mean_regional_normalized_absolute_error") or float("inf")),
            float(row.get("mean_aggregate_mass_normalized_absolute_error") or float("inf")),
            float(row.get("mean_regional_share_half_l1_error") or float("inf")),
            str(row.get("candidate_family") or ""),
        )
    )
    return [str(row.get("candidate_family") or "") for row in frontier]


def _fit_simplex_weights(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    if design.ndim != 2:
        raise ValueError("design must be a matrix")
    if target.ndim != 1:
        raise ValueError("target must be a vector")
    if design.shape[0] != target.shape[0]:
        raise ValueError("design and target row counts differ")
    if design.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    active = list(range(design.shape[1]))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    while active:
        active_design = design[:, active]
        count = len(active)
        gram = active_design.T @ active_design
        rhs = active_design.T @ target
        system = np.block(
            [
                [gram, np.ones((count, 1), dtype=np.float64)],
                [np.ones((1, count), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
            ]
        )
        solution = np.linalg.lstsq(system, np.concatenate([rhs, np.asarray([1.0], dtype=np.float64)]), rcond=None)[0][:count]
        if np.all(solution >= -float(np.finfo(np.float64).eps)):
            solution = np.maximum(solution, 0.0)
            total = float(np.sum(solution))
            if total <= 0.0:
                solution = np.ones(count, dtype=np.float64) / float(count)
            else:
                solution = solution / total
            for index, value in zip(active, solution):
                weights[index] = float(value)
            return weights
        active.pop(int(np.argmin(solution)))
    return weights


def _periods(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")}, key=quarter_sort_key)


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
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    design_rows: list[list[float]] = []
    target_values: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for period in prior_periods:
        for region in regions:
            target = rows_by_key.get((period, region))
            if target is None:
                continue
            target_value = _finite_float(target.get(metric))
            if target_value is None:
                continue
            row_values: list[float] = []
            complete = True
            for family in families:
                prediction = indexed.get((family, period, region))
                value = _finite_float((prediction or {}).get(metric))
                if value is None:
                    complete = False
                    break
                row_values.append(float(value))
            if not complete:
                continue
            design_rows.append(row_values)
            target_values.append(float(target_value))
            diagnostics.append({"period": period, "region": region})
    return (
        np.asarray(design_rows, dtype=np.float64).reshape((len(design_rows), len(families))),
        np.asarray(target_values, dtype=np.float64),
        diagnostics,
    )


def _default_weights(families: list[str], default_family: str) -> np.ndarray:
    weights = np.zeros(len(families), dtype=np.float64)
    if not families:
        return weights
    if default_family in families:
        weights[families.index(default_family)] = 1.0
    else:
        weights[0] = 1.0
    return weights


def _ensemble_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    pareto_families: list[str],
    default_family: str,
    ensemble_family: str = ENSEMBLE_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    periods = _periods(prediction_rows)
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if family == default_family and candidate_period == period})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        period_rows: dict[str, dict[str, Any]] = {
            region: {
                "candidate_family": ensemble_family,
                "holdout_period": period,
                "region": region,
            }
            for region in regions
        }
        prior_periods = [prior_period for prior_period in periods if quarter_sort_key(prior_period) < quarter_sort_key(period)]
        for metric in COUNT_METRICS:
            available = _available_families(
                indexed,
                period=period,
                regions=regions,
                families=pareto_families,
                metric=metric,
            )
            weights = _default_weights(available, default_family)
            design, target, design_diagnostics = _prior_design(
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
                    "holdout_period": period,
                    "metric_name": metric,
                    "available_candidate_families": available,
                    "weight_by_candidate_family": weight_payload,
                    "prior_period_count": len(prior_periods),
                    "prior_design_row_count": int(design.shape[0]),
                    "candidate_count": len(available),
                    "weight_status": "simplex_fitted" if fitted else "default_reference_weight",
                }
            )
            for region in regions:
                value = 0.0
                for index, family in enumerate(available):
                    prediction = indexed.get((family, period, region))
                    metric_value = _finite_float((prediction or {}).get(metric))
                    if metric_value is not None:
                        value += float(weights[index]) * float(metric_value)
                period_rows[region][metric] = float(value)
                period_rows[region][f"{metric}_weight_by_candidate_family"] = weight_payload
        for region in regions:
            raw = period_rows[region]
            projected = _projection_cascade(raw)
            projected["projection_adjusted"] = any(
                abs(float(projected.get(metric) or 0.0) - float(raw.get(metric) or 0.0)) > 1e-8
                for metric in COUNT_METRICS
            )
            rows.append(projected)
    return rows, weight_rows


def _score_ensemble_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence))
    return split_rows, coherence, combined


def _family_row(combined_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    return next((dict(row) for row in combined_rows if str(row.get("candidate_family") or "") == family), {})


def _gate(
    combined_rows: list[dict[str, Any]],
    *,
    split_gate: dict[str, Any],
    r54_gate: dict[str, Any] | None = None,
    ensemble_family: str = ENSEMBLE_FAMILY,
) -> dict[str, Any]:
    ensemble = _family_row(combined_rows, ensemble_family)
    carry = _family_row(combined_rows, CARRY_FORWARD_FAMILY)
    reference = _family_row(combined_rows, REFERENCE_FAMILY)
    r54 = dict(r54_gate or {})
    ensemble_regional = _finite_float(ensemble.get("mean_regional_normalized_absolute_error"))
    ensemble_mass = _finite_float(ensemble.get("mean_aggregate_mass_normalized_absolute_error"))
    ensemble_share = _finite_float(ensemble.get("mean_regional_share_half_l1_error"))
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
    if ensemble_regional is None or ensemble_mass is None or ensemble_share is None:
        blockers.append("ensemble_not_scored")
    if carry_regional is not None and ensemble_regional is not None and not ensemble_regional < carry_regional:
        blockers.append("ensemble_does_not_beat_carry_forward_regional_error")
    if carry_mass is not None and ensemble_mass is not None and ensemble_mass > carry_mass:
        blockers.append("ensemble_worsens_carry_forward_mass_error")
    if carry_share is not None and ensemble_share is not None and ensemble_share > carry_share:
        blockers.append("ensemble_worsens_carry_forward_share_error")
    if reference_regional is not None and ensemble_regional is not None and ensemble_regional > reference_regional:
        blockers.append("ensemble_worsens_r52_reference_regional_error")
    if reference_mass is not None and ensemble_mass is not None and ensemble_mass > reference_mass:
        blockers.append("ensemble_worsens_r52_reference_mass_error")
    if reference_share is not None and ensemble_share is not None and ensemble_share > reference_share:
        blockers.append("ensemble_worsens_r52_reference_share_error")
    if r54_regional is not None and ensemble_regional is not None and ensemble_regional > r54_regional:
        blockers.append("ensemble_does_not_beat_r54_regional_error")
    if r54_mass is not None and ensemble_mass is not None and ensemble_mass > r54_mass:
        blockers.append("ensemble_does_not_beat_r54_mass_error")
    if r54_share is not None and ensemble_share is not None and ensemble_share > r54_share:
        blockers.append("ensemble_does_not_beat_r54_share_error")
    mean_pass = not blockers
    strict_split = split_gate.get("status") == "strict_split_stable_adapter_promoted"
    if mean_pass and strict_split:
        status = "pareto_simplex_regional_ensemble_promoted"
    elif mean_pass:
        status = "pareto_simplex_regional_ensemble_mean_promoted_split_limited"
        blockers = ["ensemble_not_strict_split_stable"]
    else:
        status = "pareto_simplex_regional_ensemble_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "ensemble_family": ensemble_family,
        "ensemble_mean_regional_normalized_absolute_error": ensemble_regional,
        "ensemble_mean_aggregate_mass_normalized_absolute_error": ensemble_mass,
        "ensemble_mean_regional_share_half_l1_error": ensemble_share,
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
            "R58 ensembles only Pareto-nondominated R54 candidate families. For each holdout period and metric, "
            "nonnegative weights summing to one are fitted using earlier regional rows for that same metric. "
            "Promotion requires mean nonregression against carry-forward, R52, and R54; strict promotion additionally "
            "requires the R55-style split gate."
        ),
    }


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
    gate = dict(report.get("ensemble_gate") or {})
    split_gate = dict(report.get("split_stability_gate") or {})
    lines = [
        "# Phase 3 R58 Pareto Simplex Regional Ensemble",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Ensemble regional / mass / share errors: `{gate.get('ensemble_mean_regional_normalized_absolute_error')}` / `{gate.get('ensemble_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('ensemble_mean_regional_share_half_l1_error')}`",
        f"- R54 regional / mass / share errors: `{gate.get('r54_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_regional_share_half_l1_error')}`",
        f"- Split-stability status: `{split_gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Pareto Families",
        "",
    ]
    for family in list(report.get("pareto_candidate_families") or []):
        lines.append(f"- `{family}`")
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, combined_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    keep = {
        ENSEMBLE_FAMILY,
        CARRY_FORWARD_FAMILY,
        REFERENCE_FAMILY,
    }
    rows = [
        row
        for row in combined_rows
        if str(row.get("candidate_family") or "") in keep
        or str(row.get("candidate_family") or "").startswith(
            "national_total_adapter__total=train_only_region_metric_selector__share=similarity_proxy_log_delta"
        )
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
    ax.set_title("R58 Pareto simplex regional ensemble")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r58_pareto_simplex_regional_ensemble(
    *,
    run_id: str = R58_RUN_ID,
    r54_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    base_rows = _base_prediction_rows(r48, r49, r51)
    adapter_rows = [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
    prediction_rows = base_rows + adapter_rows
    pareto_families = _pareto_candidate_families([dict(row) for row in list(r54.get("combined_candidate_table") or [])])
    default_family = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or (pareto_families[0] if pareto_families else ""))
    ensemble_rows, weight_rows = _ensemble_prediction_rows(
        prediction_rows,
        rows_by_key,
        pareto_families=pareto_families,
        default_family=default_family,
    )
    ensemble_split, ensemble_coherence, _ensemble_combined = _score_ensemble_rows(rows_by_key, ensemble_rows)
    comparison_split_rows = [dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + ensemble_split
    comparison_coherence_rows = [dict(row) for row in list(r54.get("coherence_rows") or [])] + ensemble_coherence
    split_comparison = _split_comparison_rows(
        regional_score_rows=comparison_split_rows,
        coherence_rows=comparison_coherence_rows,
        promoted_family=ENSEMBLE_FAMILY,
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
        "R58 promoted a Pareto-simplex regional ensemble with strict split stability."
        if gate["status"] == "pareto_simplex_regional_ensemble_promoted"
        else (
            "R58 improved mean regional/mass/share errors but remains split-limited."
            if gate["status"] == "pareto_simplex_regional_ensemble_mean_promoted_split_limited"
            else "R58 did not beat the current regional references."
        )
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r58_pareto_simplex_regional_ensemble_report.json"
    md_path = analysis_dir / "r58_pareto_simplex_regional_ensemble_report.md"
    weight_csv = analysis_dir / "r58_weight_rows.csv"
    prediction_csv = analysis_dir / "r58_prediction_rows.csv"
    split_csv = analysis_dir / "r58_split_metric_scores.csv"
    coherence_csv = analysis_dir / "r58_coherence_rows.csv"
    comparison_csv = analysis_dir / "r58_split_comparison_rows.csv"
    combined_csv = analysis_dir / "r58_combined_candidate_table.csv"
    dashboard_path = analysis_dir / "r58_pareto_simplex_regional_ensemble_dashboard.png"
    report = {
        "schema_version": R58_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "pareto_rule": "keep only candidate families not dominated on mean regional error, aggregate mass error, and regional share error in the frozen R54 candidate table",
            "weight_rule": "fit nonnegative weights summing to one from earlier same-metric regional rows only",
            "target_leakage_status": "no same-period target error enters ensemble weights",
            "claim_limit": "regional readout ensemble only; no determinant, province-level, or causal transition claim",
        },
        "pareto_candidate_families": pareto_families,
        "ensemble_gate": gate,
        "split_stability_gate": split_gate,
        "weight_rows": weight_rows,
        "prediction_rows": ensemble_rows,
        "split_metric_score_rows": ensemble_split,
        "coherence_rows": ensemble_coherence,
        "split_comparison_rows": split_comparison,
        "combined_candidate_table": combined_rows,
        "r54_gate_snapshot": dict(r54.get("adapter_gate") or {}),
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "weight_csv": weight_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "split_metric_scores_csv": split_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(weight_csv, weight_rows)
    _write_csv(prediction_csv, ensemble_rows)
    _write_csv(split_csv, ensemble_split)
    _write_csv(coherence_csv, ensemble_coherence)
    _write_csv(comparison_csv, split_comparison)
    _write_csv(combined_csv, combined_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R58 Pareto simplex regional ensemble.")
    parser.add_argument("--run-id", default=R58_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r58_pareto_simplex_regional_ensemble(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
