from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, _finite_float, _generated_at, _sha256
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import _default_evidence_root
from .r75_bulk_unaids_annual_challenge import (
    _annual_carry_forward_prediction,
    _annual_score_row,
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _rolling_annual_splits,
    _score_summary_by_fields,
    _write_csv,
)
from .r83_quarterly_emission_bridge_audit import _annualized_candidate_value
from .r85_annual_ledger_forecast_grid import _dynamic_forecast_grid_predictions
from .r86_annual_calibrated_forecast_grid_ledger import _distribute_annual_total_by_quarter_shape
from .runtime import ensure_dir, read_json, write_json


R87_SCHEMA_VERSION = "phase3_dynamic.r87_train_backtested_emission_process_calibration.v1"
R87_RUN_ID = "p3d-r87-train-backtested-emission-process-calibration-20260507-s00"
R87_FAMILY = "train_backtested_raw_emission_process_calibration"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)

EMISSION_PROCESS_FAMILIES: tuple[str, ...] = (
    "identity_raw_process",
    "median_ratio_process",
    "recent_median_ratio_process",
    "log_ratio_trend_process",
)
ANNUAL_TO_QUARTERLY_METRIC: dict[str, str] = {
    "annual_new_infections": "incident_infections_period",
    "annual_aids_deaths": "aids_deaths_period",
}


def _available_annual_years(rows: list[dict[str, Any]], metric_name: str) -> list[int]:
    return sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in rows
            if str(row.get("quarter") or "").endswith("-Q4")
            and _finite_float(row.get(metric_name)) is not None
        }
    )


def _annualized_raw_values_for_year(
    rows: list[dict[str, Any]],
    *,
    year: int,
    train_end_year: int,
) -> dict[str, float | None]:
    try:
        predictions, _summary = _dynamic_forecast_grid_predictions(rows, [int(year)], train_end_year=int(train_end_year))
    except ValueError:
        return {metric_name: None for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS}
    output: dict[str, float | None] = {}
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        annualized = _annualized_candidate_value(predictions, year=int(year), annual_metric=metric_name)
        value = _finite_float(annualized.get("candidate_value"))
        output[metric_name] = None if value is None else float(value)
    return output


def _train_backtest_pairs(
    rows: list[dict[str, Any]],
    *,
    train_end_year: int,
    min_train_years: int,
) -> list[dict[str, Any]]:
    q4_by_year = {
        quarter_year(str(row.get("quarter") or "")): dict(row)
        for row in rows
        if str(row.get("quarter") or "").endswith("-Q4")
    }
    pairs: list[dict[str, Any]] = []
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        available_years = [year for year in _available_annual_years(rows, metric_name) if int(year) <= int(train_end_year)]
        for year in available_years:
            prior_years = [prior for prior in available_years if int(prior) < int(year)]
            if len(prior_years) < int(min_train_years):
                continue
            raw_values = _annualized_raw_values_for_year(rows, year=int(year), train_end_year=int(year) - 1)
            raw_value = _finite_float(raw_values.get(metric_name))
            target_value = _finite_float(q4_by_year.get(int(year), {}).get(metric_name))
            if raw_value is None or target_value is None:
                continue
            prior_rows = [
                dict(row)
                for row in rows
                if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) < int(year)
            ]
            pairs.append(
                {
                    "year": int(year),
                    "metric_name": metric_name,
                    "raw_value": max(float(raw_value), 0.0),
                    "target_value": max(float(target_value), 0.0),
                    "scale": _annual_target_scale(prior_rows, metric_name),
                    "training_use": "internal_rolling_origin_process_bias_pair",
                }
            )
    return pairs


def _fit_emission_ratio_family(pairs: list[dict[str, Any]], *, family: str) -> dict[str, Any]:
    usable = [
        dict(row)
        for row in sorted(pairs, key=lambda item: int(item.get("year") or 0))
        if _finite_float(row.get("raw_value")) is not None
        and _finite_float(row.get("target_value")) is not None
        and float(row.get("raw_value") or 0.0) > 0.0
    ]
    if str(family) == "identity_raw_process":
        return {"status": "completed", "family": family, "ratio": 1.0, "row_count": len(usable)}
    if not usable:
        return {"status": "not_estimable", "family": family, "reason": "no_positive_raw_pairs"}
    ratios = [max(float(row["target_value"]) / max(float(row["raw_value"]), float(np.finfo(np.float32).eps)), 0.0) for row in usable]
    years = [int(row["year"]) for row in usable]
    if str(family) == "median_ratio_process":
        return {
            "status": "completed",
            "family": family,
            "ratio": float(median(ratios)),
            "row_count": len(usable),
            "first_year": min(years),
            "last_year": max(years),
        }
    if str(family) == "recent_median_ratio_process":
        recent_years = sorted(set(years))[-3:]
        recent_ratios = [ratio for ratio, year in zip(ratios, years) if year in set(recent_years)]
        return {
            "status": "completed",
            "family": family,
            "ratio": float(median(recent_ratios)),
            "row_count": len(recent_ratios),
            "first_year": min(recent_years),
            "last_year": max(recent_years),
        }
    if str(family) == "log_ratio_trend_process":
        if len(usable) < 2:
            return {
                "status": "completed",
                "family": family,
                "intercept": float(np.log(max(ratios[-1], float(np.finfo(np.float32).eps)))),
                "slope": 0.0,
                "center_year": int(years[-1]),
                "row_count": len(usable),
            }
        center = float(max(years))
        x = np.asarray([float(year) - center for year in years], dtype=np.float64)
        y = np.log(np.maximum(np.asarray(ratios, dtype=np.float64), float(np.finfo(np.float32).eps)))
        design = np.column_stack([np.ones_like(x), x])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return {
            "status": "completed",
            "family": family,
            "intercept": float(coeffs[0]),
            "slope": float(coeffs[1]),
            "center_year": int(center),
            "row_count": len(usable),
            "first_year": min(years),
            "last_year": max(years),
        }
    return {"status": "not_estimable", "family": family, "reason": "unknown_family"}


def _predict_emission_ratio(raw_value: float | None, *, year: int, model: dict[str, Any]) -> float | None:
    raw = _finite_float(raw_value)
    if raw is None:
        return None
    raw = max(float(raw), 0.0)
    if str(model.get("status") or "") != "completed":
        return raw
    family = str(model.get("family") or "identity_raw_process")
    if family in {"identity_raw_process", "median_ratio_process", "recent_median_ratio_process"}:
        ratio = _finite_float(model.get("ratio"))
        return raw if ratio is None else float(raw * max(float(ratio), 0.0))
    if family == "log_ratio_trend_process":
        intercept = float(model.get("intercept") or 0.0)
        slope = float(model.get("slope") or 0.0)
        center = int(model.get("center_year") or int(year))
        ratio = float(np.exp(intercept + slope * (int(year) - int(center))))
        return float(raw * max(ratio, 0.0))
    return raw


def _select_metric_process_family(
    pairs: list[dict[str, Any]],
    *,
    metric_name: str,
) -> dict[str, Any]:
    metric_pairs = [
        dict(row)
        for row in sorted(pairs, key=lambda item: int(item.get("year") or 0))
        if str(row.get("metric_name") or "") == str(metric_name)
    ]
    family_scores: list[dict[str, Any]] = []
    for family in EMISSION_PROCESS_FAMILIES:
        errors: list[float] = []
        eval_count = 0
        for index in range(1, len(metric_pairs)):
            prior_pairs = metric_pairs[:index]
            target_pair = metric_pairs[index]
            model = _fit_emission_ratio_family(prior_pairs, family=family)
            predicted = _predict_emission_ratio(
                _finite_float(target_pair.get("raw_value")),
                year=int(target_pair.get("year") or 0),
                model=model,
            )
            target = _finite_float(target_pair.get("target_value"))
            scale = max(float(_finite_float(target_pair.get("scale")) or 0.0), float(np.finfo(np.float32).eps))
            if predicted is None or target is None:
                continue
            errors.append(abs(float(predicted) - float(target)) / scale)
            eval_count += 1
        family_scores.append(
            {
                "metric_name": metric_name,
                "family": family,
                "internal_mean_norm_error": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
                "internal_eval_count": eval_count,
            }
        )
    evaluable = [row for row in family_scores if _finite_float(row.get("internal_mean_norm_error")) is not None]
    if not evaluable:
        selected_family = "identity_raw_process"
    else:
        selected_family = str(
            sorted(
                evaluable,
                key=lambda row: (
                    float(row.get("internal_mean_norm_error") or float("inf")),
                    EMISSION_PROCESS_FAMILIES.index(str(row.get("family") or "identity_raw_process")),
                ),
            )[0]["family"]
        )
    selected_model = _fit_emission_ratio_family(metric_pairs, family=selected_family)
    return {
        "metric_name": metric_name,
        "selected_family": selected_family,
        "selected_model": selected_model,
        "family_scores": family_scores,
        "pair_count": len(metric_pairs),
        "contract": "family selected by internal rolling-origin annualized raw-emission error inside the train window only",
    }


def _apply_process_calibration_to_predictions(
    predictions: list[dict[str, Any]],
    *,
    holdout_years: list[int],
    metric_models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in predictions if row.get("quarter")}
    calibration_rows: list[dict[str, Any]] = []
    for year in sorted({int(value) for value in holdout_years}):
        year_rows = [
            by_quarter[quarter]
            for quarter in sorted(by_quarter, key=quarter_sort_key)
            if quarter_year(quarter) == int(year)
        ]
        for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            raw_annual = _annualized_candidate_value(list(by_quarter.values()), year=int(year), annual_metric=metric_name)
            raw_value = _finite_float(raw_annual.get("candidate_value"))
            model = dict(metric_models.get(metric_name, {}).get("selected_model") or {"family": "identity_raw_process", "status": "completed", "ratio": 1.0})
            corrected_value = _predict_emission_ratio(raw_value, year=int(year), model=model)
            calibration_rows.append(
                {
                    "year": int(year),
                    "metric_name": metric_name,
                    "raw_annual_value": raw_value,
                    "corrected_annual_value": corrected_value,
                    "selected_family": metric_models.get(metric_name, {}).get("selected_family"),
                    "training_use": "train_backtested_raw_emission_process_calibration",
                }
            )
            if corrected_value is None:
                continue
            if metric_name in ANNUAL_TO_QUARTERLY_METRIC:
                quarterly_metric = ANNUAL_TO_QUARTERLY_METRIC[metric_name]
                distribution = _distribute_annual_total_by_quarter_shape(
                    year_rows,
                    annual_total=float(corrected_value),
                    quarterly_metric=quarterly_metric,
                )
                distribution.pop("_shape_status", None)
                for quarter, value in distribution.items():
                    by_quarter[str(quarter)][quarterly_metric] = float(value)
            elif metric_name == "estimated_plhiv":
                q4 = f"{int(year)}-Q4"
                by_quarter.setdefault(q4, {"quarter": q4})["estimated_plhiv"] = float(corrected_value)
    return [by_quarter[quarter] for quarter in sorted(by_quarter, key=quarter_sort_key)], calibration_rows


def _process_calibrated_forecast_grid_predictions(
    rows: list[dict[str, Any]],
    holdout_years: list[int],
    *,
    train_end_year: int,
    min_train_years: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base_predictions, base_summary = _dynamic_forecast_grid_predictions(rows, holdout_years, train_end_year=train_end_year)
    pairs = _train_backtest_pairs(rows, train_end_year=train_end_year, min_train_years=min_train_years)
    metric_models = {
        metric_name: _select_metric_process_family(pairs, metric_name=metric_name)
        for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
    }
    calibrated, calibration_rows = _apply_process_calibration_to_predictions(
        base_predictions,
        holdout_years=holdout_years,
        metric_models=metric_models,
    )
    model_rows: list[dict[str, Any]] = []
    for metric_name, model in metric_models.items():
        for score in model.get("family_scores") or []:
            model_rows.append(
                {
                    **dict(score),
                    "selected_family": model.get("selected_family"),
                    "pair_count": model.get("pair_count"),
                    "train_end_year": int(train_end_year),
                }
            )
    summary = {
        **base_summary,
        "process_calibration_contract": (
            "R87 calibrates raw quarterly ledger emissions using only internally backtested annualized raw-emission "
            "bias inside the training window. It does not use holdout annual targets and does not replace the raw "
            "process with a holdout-year annual head."
        ),
        "selected_families": {
            metric_name: model.get("selected_family")
            for metric_name, model in metric_models.items()
        },
    }
    return calibrated, summary, model_rows + calibration_rows


def _score_process_calibrated_forecast_grid(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    score_rows: list[dict[str, Any]] = []
    annualized_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_q4_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_q4_rows:
            continue
        predictions, summary, split_calibration_rows = _process_calibrated_forecast_grid_predictions(
            rows,
            holdout_years,
            train_end_year=train_end_year,
            min_train_years=min_train_years,
        )
        for row in split_calibration_rows:
            row = dict(row)
            row.update(
                {
                    "candidate_family": R87_FAMILY,
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "holdout_years": holdout_years,
                }
            )
            calibration_rows.append(row)
        annualized_by_year_metric: dict[tuple[int, str], dict[str, Any]] = {}
        for year in holdout_years:
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                annualized = _annualized_candidate_value(predictions, year=int(year), annual_metric=metric_name)
                annualized.update(
                    {
                        "candidate_family": R87_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "training_use": "train_backtested_raw_emission_process_calibration",
                    }
                )
                annualized_rows.append(annualized)
                coverage_rows.append(
                    {
                        "candidate_family": R87_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "year": int(year),
                        "annual_metric": metric_name,
                        "coverage_status": annualized.get("coverage_status"),
                        "present_quarter_count": annualized.get("present_quarter_count"),
                        "emitted_quarter_count": annualized.get("emitted_quarter_count"),
                        "missing_required_outputs": annualized.get("missing_required_outputs"),
                    }
                )
                annualized_by_year_metric[(int(year), metric_name)] = annualized
        for holdout_row in holdout_q4_rows:
            holdout_year = quarter_year(str(holdout_row.get("quarter") or ""))
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                target = _finite_float(holdout_row.get(metric_name))
                if target is None:
                    continue
                annualized = annualized_by_year_metric.get((int(holdout_year), metric_name), {})
                candidate_value = _finite_float(annualized.get("candidate_value"))
                carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, metric_name)
                score = _annual_score_row(
                    family=R87_FAMILY,
                    horizon=horizon,
                    train_end_year=train_end_year,
                    holdout_years=holdout_years,
                    holdout_row=holdout_row,
                    metric_name=metric_name,
                    candidate_value=None if candidate_value is None else float(candidate_value),
                    carry_value=None if carry_value is None else float(carry_value),
                    scale=_annual_target_scale(raw_train_rows, metric_name),
                )
                score["prediction_status"] = str(annualized.get("coverage_status") or "not_predicted")
                score["training_use"] = "train_backtested_raw_emission_process_calibration"
                score_rows.append(score)
        manifest_rows.append(
            {
                "candidate_family": R87_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "selected_families": summary.get("selected_families"),
                "forecast_grid_contract": summary.get("forecast_grid_contract"),
                "dynamic_mae": summary.get("mae"),
                "model_contract": summary.get("model_contract"),
                "hazard_semantics": summary.get("hazard_semantics"),
                "incidence_diagnostics": summary.get("incidence_diagnostics"),
                "contract": summary.get("process_calibration_contract"),
            }
        )
    return score_rows, annualized_rows, coverage_rows, calibration_rows, manifest_rows


def _gate(*, score_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not target_rows:
        blockers.append("no_public_annual_target_rows")
    leakage_rows = [
        row
        for row in score_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("annual_validation_role_leakage")
    target_counts_by_metric: dict[str, int] = {}
    scored_counts_by_metric: dict[str, int] = {}
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        metric_rows = [row for row in score_rows if str(row.get("metric_name") or "") == metric_name]
        target_counts_by_metric[metric_name] = len(metric_rows)
        scored_counts_by_metric[metric_name] = sum(1 for row in metric_rows if _finite_float(row.get("candidate_norm_error")) is not None)
        if not metric_rows:
            blockers.append(f"{metric_name}_not_scored")
        elif scored_counts_by_metric[metric_name] < len(metric_rows):
            blockers.append(f"{metric_name}_incomplete_target_coverage")
    best = dict(family_rows[0]) if family_rows else {}
    candidate_mean = _finite_float(best.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(best.get("carry_forward_mean_norm_error"))
    candidate_coverage = _finite_float(best.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(best.get("carry_forward_interval_coverage"))
    if candidate_mean is None or carry_mean is None:
        blockers.append("candidate_or_carry_not_evaluable")
    elif candidate_mean >= carry_mean:
        blockers.append("raw_emission_process_calibration_not_better_than_carry_forward")
    if candidate_coverage is not None and carry_coverage is not None and candidate_coverage < carry_coverage:
        blockers.append("raw_emission_process_calibration_interval_coverage_worse_than_carry_forward")
    return {
        "status": "train_backtested_emission_process_calibration_pass" if not blockers else "train_backtested_emission_process_calibration_diagnostic_only",
        "blockers": blockers,
        "required_metric_count": len(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "scored_metric_count": sum(1 for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if scored_counts_by_metric.get(metric, 0) > 0),
        "target_counts_by_metric": target_counts_by_metric,
        "scored_counts_by_metric": scored_counts_by_metric,
        "candidate_mean_norm_error": candidate_mean,
        "carry_forward_mean_norm_error": carry_mean,
        "candidate_minus_carry_forward_mean_norm_error": None
        if candidate_mean is None or carry_mean is None
        else float(candidate_mean - carry_mean),
        "candidate_interval_coverage": candidate_coverage,
        "carry_forward_interval_coverage": carry_coverage,
        "contract": (
            "R87 is a raw-emission process calibration gate. It may rescale raw quarterly annual-ledger emissions only "
            "through metric-specific families selected by internal train-window rolling-origin error; held-out annual "
            "targets remain validation-only."
        ),
    }


def run_r87_train_backtested_emission_process_calibration(
    *,
    run_id: str = R87_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
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
    score_rows, annualized_rows, coverage_rows, calibration_rows, manifest_rows = _score_process_calibrated_forecast_grid(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    metric_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family",))
    gate = _gate(score_rows=score_rows, family_rows=family_rows, target_rows=target_rows)
    report_path = analysis_dir / "r87_train_backtested_emission_process_calibration_report.json"
    markdown_path = analysis_dir / "r87_train_backtested_emission_process_calibration_report.md"
    score_csv = analysis_dir / "r87_score_rows.csv"
    annualized_csv = analysis_dir / "r87_annualized_rows.csv"
    coverage_csv = analysis_dir / "r87_emission_coverage_rows.csv"
    calibration_csv = analysis_dir / "r87_process_calibration_rows.csv"
    family_csv = analysis_dir / "r87_family_rows.csv"
    metric_csv = analysis_dir / "r87_metric_rows.csv"
    horizon_csv = analysis_dir / "r87_horizon_rows.csv"
    manifest_csv = analysis_dir / "r87_manifest_rows.csv"
    report = {
        "schema_version": R87_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "train_backtested_emission_process_calibration_gate": gate,
        "candidate_family": R87_FAMILY,
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "target_row_count": len(target_rows),
        "score_rows": score_rows,
        "annualized_rows": annualized_rows,
        "coverage_rows": coverage_rows,
        "calibration_rows": calibration_rows,
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
            "annualized_rows_csv": annualized_csv.as_posix(),
            "coverage_rows_csv": coverage_csv.as_posix(),
            "calibration_rows_csv": calibration_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(annualized_csv, annualized_rows)
    _write_csv(coverage_csv, coverage_rows)
    _write_csv(calibration_csv, calibration_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("train_backtested_emission_process_calibration_gate") or {})
    lines = [
        "# Phase 3 R87 Train-Backtested Raw Emission Process Calibration",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Candidate mean normalized error: `{gate.get('candidate_mean_norm_error')}`",
        f"- Carry-forward mean normalized error: `{gate.get('carry_forward_mean_norm_error')}`",
        f"- Candidate minus carry-forward: `{gate.get('candidate_minus_carry_forward_mean_norm_error')}`",
        f"- Candidate interval coverage: `{gate.get('candidate_interval_coverage')}`",
        f"- Carry-forward interval coverage: `{gate.get('carry_forward_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Metric Rows",
        "",
        "| Metric | Scored Entries | Candidate Mean Error | Carry Mean Error | Candidate Interval Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('scored_candidate_entry_count')}` | "
            f"`{row.get('candidate_mean_norm_error')}` | `{row.get('carry_forward_mean_norm_error')}` | "
            f"`{row.get('candidate_interval_coverage')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R87 train-backtested raw emission process calibration gate.")
    parser.add_argument("--run-id", default=R87_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r87_train_backtested_emission_process_calibration(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
