from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
    _apply_official_annual_measurement_error_heads,
    _candidate_predictions,
    _finite_float,
    _fit_official_annual_measurement_error_heads,
    _generated_at,
    _sha256,
    _strip_official_annual_validation_targets,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import _default_evidence_root
from .runtime import ensure_dir, read_json, write_json


R75_SCHEMA_VERSION = "phase3_dynamic.r75_bulk_unaids_annual_challenge.v1"
R75_RUN_ID = "p3d-r75-bulk-unaids-annual-challenge-20260507-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
R75_REQUIRED_TARGET_SPECS: tuple[dict[str, str], ...] = (
    {
        "metric_name": "annual_new_infections",
        "module_target": "incidence_validation",
        "indicator_gid": "NEW_INFECTIONS",
        "indicator": "New HIV Infections",
        "unit": "Number",
    },
    {
        "metric_name": "annual_aids_deaths",
        "module_target": "mortality_reporting",
        "indicator_gid": "AIDS_DEATHS",
        "indicator": "AIDS-related deaths",
        "unit": "Number",
    },
    {
        "metric_name": "estimated_plhiv",
        "module_target": "plhiv_stock_validation",
        "indicator_gid": "PLWH",
        "indicator": "People living with HIV",
        "unit": "Number",
    },
)
R75_DEFAULT_CANDIDATE_FAMILIES: tuple[str, ...] = (
    "r41_monotone_growth_component_process",
    "multi_horizon_weighted_process",
    "r18_evidence_backed_art_process",
    "r19_joint_service_cascade_process",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _r69_annual_path(r69_report: dict[str, Any]) -> Path | None:
    path = dict(r69_report.get("artifact_paths") or {}).get("annual_external_challenge_csv")
    return None if not path else Path(str(path))


def _subgroup_kind(subgroup: str) -> str | None:
    normalized = " ".join(str(subgroup or "").lower().split())
    if normalized == "all ages estimate":
        return "estimate"
    if normalized == "all ages lower estimate":
        return "lower"
    if normalized == "all ages upper estimate":
        return "upper"
    return None


def _target_year(time_period: str) -> int | None:
    try:
        return quarter_year(str(time_period))
    except (TypeError, ValueError):
        return None


def _bulk_unaids_target_rows(annual_csv: Path, *, external_start_year: int = 2010) -> list[dict[str, Any]]:
    specs_by_gid = {spec["indicator_gid"]: dict(spec) for spec in R75_REQUIRED_TARGET_SPECS}
    grouped: dict[tuple[int, str], dict[str, Any]] = defaultdict(dict)
    provenance: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _read_csv(annual_csv):
        gid = str(row.get("indicator_gid") or "")
        spec = specs_by_gid.get(gid)
        if spec is None:
            continue
        if str(row.get("unit") or "") != str(spec["unit"]):
            continue
        kind = _subgroup_kind(str(row.get("subgroup") or ""))
        if kind is None:
            continue
        year = _target_year(str(row.get("time_period") or ""))
        value = _finite_float(row.get("value"))
        if year is None or int(year) < int(external_start_year) or value is None:
            continue
        metric = str(spec["metric_name"])
        grouped[(int(year), metric)][kind] = max(float(value), 0.0)
        provenance[(int(year), metric)] = {
            "source_id": str(row.get("source_id") or "unaids_estimates_2025"),
            "source_tier": "external_official_model_estimate",
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
            "support_partition": "bulk_unaids_estimates_2025",
            "measurement_semantics": "modeled_estimate_interval",
            "indicator_gid": gid,
            "indicator": str(row.get("indicator") or spec["indicator"]),
            "unit": str(row.get("unit") or spec["unit"]),
            "subgroup_scope": "All ages estimate/lower/upper",
        }
    rows: list[dict[str, Any]] = []
    for (year, metric), values in sorted(grouped.items()):
        estimate = _finite_float(values.get("estimate"))
        if estimate is None:
            continue
        lower = _finite_float(values.get("lower"))
        upper = _finite_float(values.get("upper"))
        if lower is not None and upper is not None and float(lower) > float(upper):
            lower, upper = upper, lower
        rows.append(
            {
                "year": int(year),
                "quarter": f"{int(year)}-Q4",
                "metric_name": metric,
                "target_value": float(estimate),
                "target_lower": None if lower is None else float(lower),
                "target_upper": None if upper is None else float(upper),
                "target_interval_available": lower is not None and upper is not None,
                "metric_provenance": provenance.get((int(year), metric), {}),
            }
        )
    return rows


def _merge_external_targets_into_observations(
    observation_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in observation_rows if row.get("quarter")}
    for target in target_rows:
        quarter = str(target["quarter"])
        metric = str(target["metric_name"])
        row = rows_by_quarter.setdefault(quarter, {"quarter": quarter})
        row[metric] = float(target["target_value"])
        row[f"{metric}_target_lower"] = target.get("target_lower")
        row[f"{metric}_target_upper"] = target.get("target_upper")
        row[f"{metric}_target_interval_available"] = bool(target.get("target_interval_available"))
        provenance = dict(row.get("metric_provenance") or {})
        provenance[metric] = dict(target.get("metric_provenance") or {})
        row["metric_provenance"] = provenance
    return sorted(rows_by_quarter.values(), key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))


def _annual_target_scale(train_rows: list[dict[str, Any]], metric_name: str) -> float:
    values = [
        abs(float(value))
        for value in (_finite_float(row.get(metric_name)) for row in train_rows)
        if value is not None
    ]
    return max(values) if values else float(np.finfo(np.float32).eps)


def _annual_carry_forward_prediction(raw_train_rows: list[dict[str, Any]], holdout_row: dict[str, Any], metric_name: str) -> float | None:
    eligible = [
        row
        for row in sorted(raw_train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _finite_float(row.get(metric_name)) is not None
    ]
    if not eligible:
        return None
    return _finite_float(eligible[-1].get(metric_name))


def _interval_distance(value: float | None, lower: float | None, upper: float | None) -> float | None:
    if value is None or lower is None or upper is None:
        return None
    if float(value) < float(lower):
        return float(lower) - float(value)
    if float(value) > float(upper):
        return float(value) - float(upper)
    return 0.0


def _annual_score_row(
    *,
    family: str,
    horizon: int,
    train_end_year: int,
    holdout_years: list[int],
    holdout_row: dict[str, Any],
    metric_name: str,
    candidate_value: float | None,
    carry_value: float | None,
    scale: float,
) -> dict[str, Any]:
    target_value = _finite_float(holdout_row.get(metric_name))
    lower = _finite_float(holdout_row.get(f"{metric_name}_target_lower"))
    upper = _finite_float(holdout_row.get(f"{metric_name}_target_upper"))
    candidate_interval_distance = _interval_distance(candidate_value, lower, upper)
    carry_interval_distance = _interval_distance(carry_value, lower, upper)
    provenance = dict(dict(holdout_row.get("metric_provenance") or {}).get(metric_name) or {})
    return {
        "candidate_family": family,
        "horizon_years": int(horizon),
        "train_end_year": int(train_end_year),
        "holdout_years": holdout_years,
        "quarter": str(holdout_row.get("quarter") or ""),
        "year": quarter_year(str(holdout_row.get("quarter") or "")),
        "metric_name": metric_name,
        "target_value": None if target_value is None else float(target_value),
        "target_lower": lower,
        "target_upper": upper,
        "target_interval_available": lower is not None and upper is not None,
        "candidate_value": candidate_value,
        "carry_forward_value": carry_value,
        "scale": float(scale),
        "candidate_norm_error": None
        if candidate_value is None or target_value is None
        else abs(float(candidate_value) - float(target_value)) / float(scale),
        "carry_forward_norm_error": None
        if carry_value is None or target_value is None
        else abs(float(carry_value) - float(target_value)) / float(scale),
        "candidate_minus_carry_forward_norm_error": None
        if candidate_value is None or carry_value is None or target_value is None
        else (
            abs(float(candidate_value) - float(target_value))
            - abs(float(carry_value) - float(target_value))
        )
        / float(scale),
        "candidate_interval_miss_norm": None
        if candidate_interval_distance is None
        else float(candidate_interval_distance) / float(scale),
        "carry_forward_interval_miss_norm": None
        if carry_interval_distance is None
        else float(carry_interval_distance) / float(scale),
        "candidate_interval_covered": None
        if candidate_interval_distance is None
        else bool(float(candidate_interval_distance) <= 0.0),
        "carry_forward_interval_covered": None
        if carry_interval_distance is None
        else bool(float(carry_interval_distance) <= 0.0),
        "prediction_status": "not_predicted" if candidate_value is None else "scored",
        "observation_role": str(provenance.get("observation_role") or ""),
        "allowed_use": str(provenance.get("allowed_use") or ""),
        "source_id": str(provenance.get("source_id") or ""),
        "measurement_semantics": str(provenance.get("measurement_semantics") or ""),
        "support_partition": str(provenance.get("support_partition") or ""),
        "training_use": "train_origin_weak_measurement_head_only",
    }


def _rolling_annual_splits(rows: list[dict[str, Any]], *, start_year: int, end_year: int, min_train_years: int, horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    available_years = sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in rows
            if str(row.get("quarter") or "").endswith("-Q4")
        }
    )
    splits: list[dict[str, Any]] = []
    for train_end in available_years:
        prior_years = [year for year in available_years if year <= train_end]
        if len(prior_years) < int(min_train_years):
            continue
        for horizon in horizons:
            holdout_years = [int(train_end) + offset for offset in range(1, int(horizon) + 1)]
            if holdout_years[0] < int(start_year) or holdout_years[-1] > int(end_year):
                continue
            if not all(year in available_years for year in holdout_years):
                continue
            splits.append(
                {
                    "train_end_year": int(train_end),
                    "horizon_years": int(horizon),
                    "holdout_years": holdout_years,
                }
            )
    return splits


def _score_family_against_bulk_annual(
    rows: list[dict[str, Any]],
    *,
    family: str,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    score_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        train_rows = [_strip_official_annual_validation_targets(row) for row in raw_train_rows]
        holdout_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_rows:
            continue
        candidate_predictions, candidate_summary = _candidate_predictions(train_rows, holdout_rows, family=family)
        annual_head_process = _fit_official_annual_measurement_error_heads(raw_train_rows)
        candidate_predictions, annual_head_rows = _apply_official_annual_measurement_error_heads(
            candidate_predictions,
            holdout_rows,
            annual_head_process,
        )
        candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_predictions}
        for holdout_row in holdout_rows:
            quarter = str(holdout_row.get("quarter") or "")
            candidate_row = candidate_by_quarter.get(quarter, {})
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                target = _finite_float(holdout_row.get(metric_name))
                if target is None:
                    continue
                scale = _annual_target_scale(raw_train_rows, metric_name)
                candidate_value = _finite_float(candidate_row.get(metric_name))
                carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, metric_name)
                score_rows.append(
                    _annual_score_row(
                        family=family,
                        horizon=horizon,
                        train_end_year=train_end_year,
                        holdout_years=holdout_years,
                        holdout_row=holdout_row,
                        metric_name=metric_name,
                        candidate_value=None if candidate_value is None else float(candidate_value),
                        carry_value=None if carry_value is None else float(carry_value),
                        scale=scale,
                    )
                )
        manifest_rows.append(
            {
                "candidate_family": family,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "annual_head_status": str(annual_head_process.get("status") or ""),
                "joint_conservation_status": str(annual_head_process.get("joint_conservation_status") or ""),
                "candidate_summary_family": str(candidate_summary.get("family") or ""),
                "annual_measurement_head_row_count": len(annual_head_rows),
            }
        )
    return score_rows, manifest_rows


def _score_summary_by_fields(rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(dict(row))
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        candidate_errors = [
            float(row["candidate_norm_error"])
            for row in values
            if _finite_float(row.get("candidate_norm_error")) is not None
        ]
        carry_errors = [
            float(row["carry_forward_norm_error"])
            for row in values
            if _finite_float(row.get("carry_forward_norm_error")) is not None
        ]
        candidate_interval = [
            float(row["candidate_interval_miss_norm"])
            for row in values
            if _finite_float(row.get("candidate_interval_miss_norm")) is not None
        ]
        carry_interval = [
            float(row["carry_forward_interval_miss_norm"])
            for row in values
            if _finite_float(row.get("carry_forward_interval_miss_norm")) is not None
        ]
        candidate_covered = [
            bool(row["candidate_interval_covered"])
            for row in values
            if row.get("candidate_interval_covered") is not None
        ]
        carry_covered = [
            bool(row["carry_forward_interval_covered"])
            for row in values
            if row.get("carry_forward_interval_covered") is not None
        ]
        summary = {field: key[index] for index, field in enumerate(group_fields)}
        candidate_mean = None if not candidate_errors else float(np.mean(np.asarray(candidate_errors, dtype=np.float64)))
        carry_mean = None if not carry_errors else float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
        candidate_p90 = None if not candidate_errors else float(np.quantile(np.asarray(candidate_errors, dtype=np.float64), 0.9))
        carry_p90 = None if not carry_errors else float(np.quantile(np.asarray(carry_errors, dtype=np.float64), 0.9))
        summary.update(
            {
                "entry_count": len(values),
                "scored_candidate_entry_count": len(candidate_errors),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "candidate_p90_norm_error": candidate_p90,
                "carry_forward_p90_norm_error": carry_p90,
                "candidate_interval_miss_mean_norm": None
                if not candidate_interval
                else float(np.mean(np.asarray(candidate_interval, dtype=np.float64))),
                "carry_forward_interval_miss_mean_norm": None
                if not carry_interval
                else float(np.mean(np.asarray(carry_interval, dtype=np.float64))),
                "candidate_interval_coverage": None
                if not candidate_covered
                else float(sum(candidate_covered) / len(candidate_covered)),
                "carry_forward_interval_coverage": None
                if not carry_covered
                else float(sum(carry_covered) / len(carry_covered)),
            }
        )
        output.append(summary)
    return output


def _gate(family_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not target_rows:
        blockers.append("no_bulk_unaids_required_targets")
    if not score_rows:
        blockers.append("no_bulk_unaids_score_rows")
    leakage_rows = [
        row
        for row in score_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("annual_validation_role_leakage")
    scored_metrics = {str(row.get("metric_name") or "") for row in score_rows if _finite_float(row.get("candidate_norm_error")) is not None}
    missing_metrics = [metric for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if metric not in scored_metrics]
    blockers.extend([f"{metric}_not_scored" for metric in missing_metrics])
    candidates = [
        row
        for row in family_rows
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    candidates.sort(key=lambda row: (float(row["candidate_mean_norm_error"]), str(row.get("candidate_family") or "")))
    best = candidates[0] if candidates else {}
    best_mean = _finite_float(best.get("candidate_mean_norm_error"))
    best_carry = _finite_float(best.get("carry_forward_mean_norm_error"))
    best_coverage = _finite_float(best.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(best.get("carry_forward_interval_coverage"))
    if best_mean is None or best_carry is None:
        blockers.append("best_candidate_or_carry_not_evaluable")
    elif best_mean >= best_carry:
        blockers.append("best_candidate_not_better_than_carry_forward")
    if best_coverage is not None and carry_coverage is not None and best_coverage < carry_coverage:
        blockers.append("best_candidate_interval_coverage_worse_than_carry_forward")
    return {
        "status": "bulk_unaids_annual_challenge_pass" if not blockers else "bulk_unaids_annual_challenge_diagnostic_only",
        "blockers": blockers,
        "target_row_count": len(target_rows),
        "score_row_count": len(score_rows),
        "candidate_family_count": len({str(row.get("candidate_family") or "") for row in family_rows}),
        "best_candidate_family": best.get("candidate_family"),
        "best_candidate_mean_norm_error": best_mean,
        "best_carry_forward_mean_norm_error": best_carry,
        "best_candidate_minus_carry_forward_mean_norm_error": None
        if best_mean is None or best_carry is None
        else float(best_mean - best_carry),
        "best_candidate_interval_coverage": best_coverage,
        "best_carry_forward_interval_coverage": carry_coverage,
        "contract": (
            "R75 scores bulk UNAIDS 2025 all-ages annual estimates as validation-only official-style targets. "
            "Annual incidence, AIDS deaths, and PLHIV are predicted by train-origin weak-measurement heads with "
            "joint mass-balance PLHIV conservation; no holdout annual target enters quarterly state training."
        ),
    }


def run_r75_bulk_unaids_annual_challenge(
    *,
    run_id: str = R75_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    candidate_families: tuple[str, ...] = R75_DEFAULT_CANDIDATE_FAMILIES,
    external_start_year: int = 2010,
    start_year: int = 2019,
    end_year: int = 2024,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    root = Path(epigraph_root) if epigraph_root is not None else _default_evidence_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(root, source_run_id=source_run_id, preferred=baseline_source_run_id)
    observation_rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    rows = _merge_external_targets_into_observations(observation_rows, target_rows)
    score_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for family in candidate_families:
        family_scores, family_manifests = _score_family_against_bulk_annual(
            rows,
            family=family,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
        )
        score_rows.extend(family_scores)
        manifest_rows.extend(family_manifests)
    metric_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family",))
    gate = _gate(family_rows, score_rows, target_rows)
    report_path = analysis_dir / "r75_bulk_unaids_annual_challenge_report.json"
    markdown_path = analysis_dir / "r75_bulk_unaids_annual_challenge_report.md"
    score_csv = analysis_dir / "r75_score_rows.csv"
    family_csv = analysis_dir / "r75_family_rows.csv"
    metric_csv = analysis_dir / "r75_metric_rows.csv"
    horizon_csv = analysis_dir / "r75_horizon_rows.csv"
    target_csv = analysis_dir / "r75_bulk_unaids_target_rows.csv"
    manifest_csv = analysis_dir / "r75_manifest_rows.csv"
    report = {
        "schema_version": R75_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "bulk_unaids_annual_gate": gate,
        "candidate_families": list(candidate_families),
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "external_start_year": int(external_start_year),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "target_rows": target_rows,
        "score_rows": score_rows,
        "family_rows": family_rows,
        "metric_rows": metric_rows,
        "horizon_rows": horizon_rows,
        "manifest_rows": manifest_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
            "epigraph_root": root.as_posix(),
            "source_run_id": source_run_id,
            "baseline_source_run_id": baseline_source_run_id,
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "score_rows_csv": score_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "target_rows_csv": target_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(target_csv, target_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("bulk_unaids_annual_gate") or {})
    lines = [
        "# Phase 3 R75 Bulk UNAIDS Annual Challenge",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Best candidate family: `{gate.get('best_candidate_family')}`",
        f"- Best candidate mean normalized error: `{gate.get('best_candidate_mean_norm_error')}`",
        f"- Carry-forward mean normalized error for best family rows: `{gate.get('best_carry_forward_mean_norm_error')}`",
        f"- Best candidate interval coverage: `{gate.get('best_candidate_interval_coverage')}`",
        f"- Carry-forward interval coverage: `{gate.get('best_carry_forward_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Family Scores",
        "",
        "| Family | Entries | Candidate mean | Carry mean | Delta | Candidate interval coverage | Carry interval coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("family_rows") or []:
        lines.append(
            f"| `{row.get('candidate_family')}` | {int(row.get('entry_count') or 0)} | "
            f"{float(row.get('candidate_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('carry_forward_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('candidate_minus_carry_forward_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('candidate_interval_coverage') or 0.0):.6f} | "
            f"{float(row.get('carry_forward_interval_coverage') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R75 bulk UNAIDS annual challenge.")
    parser.add_argument("--run-id", default=R75_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--candidate-family", action="append", default=None)
    parser.add_argument("--external-start-year", type=int, default=2010)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, action="append", default=None)
    args = parser.parse_args()
    run_r75_bulk_unaids_annual_challenge(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        candidate_families=tuple(args.candidate_family or R75_DEFAULT_CANDIDATE_FAMILIES),
        external_start_year=int(args.external_start_year),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_train_years=int(args.min_train_years),
        horizons=tuple(int(value) for value in (args.horizon or [1, 3, 5])),
    )


if __name__ == "__main__":
    _main()
