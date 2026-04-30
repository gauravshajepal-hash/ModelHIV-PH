from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    STATE_NAMES,
    build_blocked_time_dataset,
    build_observation_rows,
    build_source_row_path,
    canonical_state_values,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
    state_sum,
)
from .metrics import quarter_ordinal, quarter_sort_key, quarter_year
from .observation_ledger import (
    build_contract_row_hash,
    build_observation_contract_lookup,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .phase2 import (
    allowed_direct_edge_keys_from_robustness,
    load_phase2_structural_inputs,
    resolve_phase2_determinant_robustness_report,
    resolve_phase2_structural_source_run_id,
)
from .runtime import ensure_dir, read_json, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID


INCIDENCE_MODULE_SEARCH_SCHEMA_VERSION = "phase3_dynamic_incidence_module_search.v1"

INCIDENCE_MODULE_FAMILIES: tuple[str, ...] = (
    "carry_forward_s_eff_hazard",
    "s_eff_hazard_ar",
    "s_eff_hazard_ar_weak_annual_measurement",
    "s_eff_hazard_r10_mechanistic",
    "s_eff_hazard_r10_mechanistic_weak_annual_measurement",
    "s_eff_hazard_r10_mechanistic_annual_measurement_readout",
    "s_eff_hazard_r10_mechanistic_kp_pressure",
    "s_eff_hazard_r10_mechanistic_kp_pressure_weak_annual_measurement",
    "s_eff_hazard_r10_mechanistic_kp_pressure_annual_measurement_readout",
    "s_eff_hazard_r10_mechanistic_phase2_stable",
    "s_eff_hazard_r10_mechanistic_phase2_stable_weak_annual_measurement",
    "s_eff_hazard_r10_mechanistic_phase2_stable_annual_measurement_readout",
    "s_eff_hazard_r10_mechanistic_kp_phase2_stable",
    "s_eff_hazard_r10_mechanistic_kp_phase2_stable_weak_annual_measurement",
    "s_eff_hazard_r10_mechanistic_kp_phase2_stable_annual_measurement_readout",
    "s_eff_hazard_kp_pressure",
    "s_eff_hazard_kp_pressure_weak_annual_measurement",
    "s_eff_hazard_phase2_stable",
    "s_eff_hazard_phase2_stable_weak_annual_measurement",
    "s_eff_hazard_kp_phase2_stable",
    "s_eff_hazard_kp_phase2_stable_weak_annual_measurement",
)

KP_PRESSURE_METRIC_TOKENS: tuple[str, ...] = (
    "men_who_have_sex_with_men",
    "transgender_people",
    "sex_workers",
    "people_who_inject_drugs",
    "transactional_sex_cases_period",
    "youth_cases_15_24_period",
    "prep_people_receiving",
    "prep_newly_enrolled_period",
)

KP_ALLOWED_ROLES: frozenset[str] = frozenset({"prior_context", "auxiliary_likelihood"})


@dataclass(slots=True)
class IncidenceModuleHead:
    family: str
    feature_names: list[str]
    coefficients: list[float]
    train_row_count: int
    hazard_floor: float
    hazard_ceiling: float
    kp_feature_summary: dict[str, Any] | None = None
    phase2_feature_summary: dict[str, Any] | None = None
    training_objective: str = "train_stock_balance_s_eff_incidence_hazard_with_annual_incidence_validation_only"


@dataclass(slots=True)
class WeakAnnualMeasurementCalibrator:
    intercept: float
    slope: float
    residual_sigma_log: float | None
    train_annual_count: int
    train_years: list[int]
    train_residuals_log: list[float]
    calibration_family: str = "proportional_log_bias"
    training_objective: str = "weak_measurement_proportional_log_annual_incidence_calibration_train_years_only"


@dataclass(slots=True)
class AnnualMeasurementReadoutHead:
    train_years: list[int]
    train_log_values: list[float]
    residual_sigma_log: float | None
    train_residuals_log: list[float]
    forecast_log_floor: float
    forecast_log_ceiling: float
    readout_family: str = "r10_log_delta_unclipped_annual_measurement_readout"
    training_objective: str = "weak_measurement_r10_log_delta_annual_incidence_readout_train_years_only"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _quarter_from_time_label(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        year = int(text[:4])
    except ValueError:
        return None
    if "-Q" in text:
        try:
            quarter = int(text.split("-Q", 1)[1][:1])
        except ValueError:
            return None
        return f"{year:04d}-Q{quarter}"
    if "-" not in text:
        return f"{year:04d}-Q4"
    try:
        month = int(text.split("-", 1)[1][:2])
    except ValueError:
        return f"{year:04d}-Q4"
    return f"{year:04d}-Q{((month - 1) // 3) + 1}"


def _quarter_from_ordinal(index: int) -> str:
    year = int(index) // 4
    quarter = int(index) % 4 + 1
    return f"{year:04d}-Q{quarter}"


def _interval_quarter_gap(previous_quarter: str, quarter: str) -> int:
    previous = quarter_ordinal(str(previous_quarter))
    current = quarter_ordinal(str(quarter))
    return max(int(current - previous), 1)


def _interval_hazard_to_quarter_rate(interval_hazard: float, interval_quarters: int) -> float:
    eps = float(np.finfo(np.float64).eps)
    interval_probability = min(max(float(interval_hazard), 0.0), 1.0 - eps)
    return float(-np.log1p(-interval_probability) / max(int(interval_quarters), 1))


def _quarter_rate_to_interval_incidence(rate: float, susceptible_effective: float, interval_quarters: int) -> float:
    return float(max(float(susceptible_effective), 0.0) * (1.0 - np.exp(-max(float(rate), 0.0) * max(int(interval_quarters), 1))))


def _annual_incidence_targets(validation_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        quarter = str(row.get("quarter") or "")
        value = _finite_float(row.get("annual_new_infections"))
        if value is None or value <= 0.0 or not quarter.endswith("-Q4"):
            continue
        targets[quarter_year(quarter)] = dict(row)
    return targets


def _base_family_for_measurement_family(family: str) -> str:
    for suffix in ("_weak_annual_measurement", "_annual_measurement_readout"):
        if family.endswith(suffix):
            return family[: -len(suffix)]
    return family


def _uses_weak_annual_measurement(family: str) -> bool:
    return family.endswith("_weak_annual_measurement")


def _uses_annual_measurement_readout(family: str) -> bool:
    return family.endswith("_annual_measurement_readout")


def _is_r10_mechanistic_family(family: str) -> bool:
    return "r10_mechanistic" in _base_family_for_measurement_family(str(family))


def _is_determinant_bundle_family(family: str) -> bool:
    base_family = _base_family_for_measurement_family(str(family))
    return "kp_pressure" in base_family or "phase2_stable" in base_family


def _determinant_free_counterpart(family: str) -> str | None:
    if not _is_determinant_bundle_family(family):
        return None
    suffix = "_weak_annual_measurement" if _uses_weak_annual_measurement(family) else ""
    if _uses_annual_measurement_readout(family):
        suffix = "_annual_measurement_readout"
    base_family = _base_family_for_measurement_family(str(family))
    if "r10_mechanistic" in base_family:
        return f"s_eff_hazard_r10_mechanistic{suffix}"
    return f"s_eff_hazard_ar{suffix}"


def _complete_annual_from_quarters(values_by_quarter: dict[str, float], year: int) -> float | None:
    quarters = [_quarter_from_ordinal(quarter_ordinal(f"{year:04d}-Q1") + offset) for offset in range(4)]
    if not all(quarter in values_by_quarter for quarter in quarters):
        return None
    values = [_finite_float(values_by_quarter.get(quarter)) for quarter in quarters]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values if value is not None))


def _annual_from_mixed_frequency_map(values_by_quarter: dict[str, float], year: int) -> tuple[float | None, str]:
    complete_value = _complete_annual_from_quarters(values_by_quarter, int(year))
    if complete_value is not None:
        return float(complete_value), "sum_of_four_quarterly_incidence_predictions"
    year_quarters = sorted(
        [quarter for quarter in values_by_quarter if quarter_year(quarter) == int(year)],
        key=quarter_sort_key,
    )
    if year_quarters == [f"{year:04d}-Q4"]:
        return float(values_by_quarter[year_quarters[0]]), "single_q4_interval_prediction_covering_annual_gap"
    return None, "not_scored_partial_year_without_complete_quarters_or_annual_q4_anchor"


def _train_annual_stock_balance_incidence(dataset: Any) -> dict[int, float]:
    by_quarter: dict[str, float] = {}
    for row in list(dataset.train_transition_rows):
        quarter = str(row.get("quarter") or "")
        value = _finite_float((row.get("stock_balance") or {}).get("incidence_inflow"))
        if not quarter or value is None:
            continue
        by_quarter[quarter] = max(float(value), 0.0)
    annual: dict[int, float] = {}
    for year in sorted({quarter_year(quarter) for quarter in by_quarter}):
        value, _contract = _annual_from_mixed_frequency_map(by_quarter, int(year))
        if value is not None and value > 0.0:
            annual[int(year)] = float(value)
    return annual


def _fit_weak_annual_measurement_calibrator(
    *,
    dataset: Any,
    targets: dict[int, dict[str, Any]],
    train_end_year: int,
) -> WeakAnnualMeasurementCalibrator | None:
    raw_annual = _train_annual_stock_balance_incidence(dataset)
    points: list[tuple[int, float, float]] = []
    for year, raw_value in sorted(raw_annual.items()):
        if int(year) > int(train_end_year):
            continue
        target = targets.get(int(year))
        target_value = None if target is None else _finite_float(target.get("annual_new_infections"))
        if target_value is None or target_value <= 0.0 or raw_value <= 0.0:
            continue
        points.append((int(year), float(raw_value), float(target_value)))
    if len(points) <= 1:
        return None
    log_ratios = np.asarray(
        [float(np.log(target_value) - np.log(raw_value)) for _year, raw_value, target_value in points],
        dtype=np.float64,
    )
    intercept = float(np.mean(log_ratios))
    residuals = log_ratios - intercept
    degrees = int(len(points) - 1)
    sigma = None
    if degrees > 0:
        variance = float(np.sum(np.square(residuals)) / float(degrees))
        sigma = float(np.sqrt(max(variance, 0.0)))
    return WeakAnnualMeasurementCalibrator(
        intercept=intercept,
        slope=1.0,
        residual_sigma_log=sigma,
        train_annual_count=len(points),
        train_years=[int(year) for year, _raw, _target in points],
        train_residuals_log=[float(value) for value in residuals],
    )


def _fit_annual_measurement_readout_head(
    *,
    targets: dict[int, dict[str, Any]],
    train_end_year: int,
) -> AnnualMeasurementReadoutHead | None:
    points: list[tuple[int, float]] = []
    for year, target in sorted(targets.items()):
        if int(year) > int(train_end_year):
            continue
        target_value = _finite_float(target.get("annual_new_infections"))
        if target_value is None or target_value <= 0.0:
            continue
        points.append((int(year), float(np.log(target_value))))
    if len(points) < 3:
        return None
    log_values = [value for _year, value in points]
    residuals: list[float] = []
    for idx in range(2, len(log_values)):
        predicted = 2.0 * float(log_values[idx - 1]) - float(log_values[idx - 2])
        residuals.append(float(log_values[idx] - predicted))
    sigma = None
    if residuals:
        sigma = float(np.sqrt(float(np.mean(np.square(np.asarray(residuals, dtype=np.float64))))))
    return AnnualMeasurementReadoutHead(
        train_years=[year for year, _value in points],
        train_log_values=[float(value) for _year, value in points],
        residual_sigma_log=sigma,
        train_residuals_log=residuals,
        forecast_log_floor=float(min(log_values)),
        forecast_log_ceiling=float(max(log_values)),
    )


def _annual_measurement_readout_forecast(
    *,
    head: AnnualMeasurementReadoutHead,
    years: list[int],
) -> dict[int, float]:
    history = [float(value) for value in head.train_log_values]
    forecast: dict[int, float] = {}
    for year in sorted({int(year) for year in years if int(year) > max(head.train_years)}):
        if len(history) >= 2:
            predicted_log = 2.0 * float(history[-1]) - float(history[-2])
        else:
            predicted_log = float(history[-1])
        forecast[int(year)] = float(np.exp(predicted_log))
        history.append(predicted_log)
    return forecast


def _apply_annual_measurement_readout(
    *,
    incidence_map: dict[str, float],
    readout_head: AnnualMeasurementReadoutHead,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    calibrated = {quarter: float(value) for quarter, value in dict(incidence_map).items()}
    years = sorted({quarter_year(quarter) for quarter in incidence_map})
    annual_forecast = _annual_measurement_readout_forecast(head=readout_head, years=years)
    rows: list[dict[str, Any]] = []
    for year, forecast_annual in sorted(annual_forecast.items()):
        raw_annual, aggregation_contract = _annual_from_mixed_frequency_map(dict(incidence_map), int(year))
        if raw_annual is None or raw_annual <= 0.0 or forecast_annual <= 0.0:
            continue
        ratio = float(forecast_annual) / float(raw_annual)
        year_quarters = sorted(
            [quarter for quarter in incidence_map if quarter_year(quarter) == int(year)],
            key=quarter_sort_key,
        )
        for quarter in year_quarters:
            calibrated[quarter] = float(max(float(incidence_map[quarter]) * ratio, 0.0))
        rows.append(
            {
                "year": int(year),
                "latent_raw_annual_incidence": float(raw_annual),
                "annual_measurement_readout_incidence": float(forecast_annual),
                "annual_rescale_ratio": float(ratio),
                "aggregation_contract": aggregation_contract,
                "measurement_head": "recursive R10-style log annual incidence delta readout trained on pre-holdout annual measurements",
            }
        )
    return calibrated, rows


def _apply_weak_annual_measurement_calibration(
    *,
    incidence_map: dict[str, float],
    calibrator: WeakAnnualMeasurementCalibrator,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    calibrated = {quarter: float(value) for quarter, value in dict(incidence_map).items()}
    rows: list[dict[str, Any]] = []
    for year in sorted({quarter_year(quarter) for quarter in incidence_map}):
        raw_annual = _complete_annual_from_quarters(dict(incidence_map), int(year))
        if raw_annual is None or raw_annual <= 0.0:
            continue
        calibrated_annual = float(np.exp(float(calibrator.intercept) + float(calibrator.slope) * float(np.log(raw_annual))))
        ratio = calibrated_annual / float(raw_annual) if raw_annual > 0.0 else 1.0
        quarters = [_quarter_from_ordinal(quarter_ordinal(f"{year:04d}-Q1") + offset) for offset in range(4)]
        for quarter in quarters:
            calibrated[quarter] = float(max(float(incidence_map[quarter]) * ratio, 0.0))
        rows.append(
            {
                "year": int(year),
                "latent_raw_annual_incidence": float(raw_annual),
                "weak_measurement_annual_incidence": float(calibrated_annual),
                "annual_rescale_ratio": float(ratio),
                "measurement_head": "log(target_annual_incidence)=log(latent_annual_stock_balance_incidence)+bias+epsilon",
            }
        )
    return calibrated, rows


def _train_hazard_pairs(dataset: Any) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for row in sorted(list(dataset.train_transition_rows), key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        previous_quarter = str(row.get("previous_quarter") or "")
        interval_hazard = _finite_float((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff"))
        if interval_hazard is None:
            continue
        interval_quarters = _interval_quarter_gap(previous_quarter, quarter)
        pairs.append((quarter, _interval_hazard_to_quarter_rate(float(interval_hazard), interval_quarters)))
    return pairs


def _matrix_rank_enough(x: np.ndarray) -> bool:
    if x.size == 0:
        return False
    return int(x.shape[0]) > int(x.shape[1])


def _pinv_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if not _matrix_rank_enough(x):
        return None
    return np.linalg.pinv(x) @ y


def _latest_past_value(values_by_quarter: dict[str, float], quarter: str) -> float | None:
    candidates = [
        (candidate, value)
        for candidate, value in values_by_quarter.items()
        if quarter_sort_key(candidate) <= quarter_sort_key(quarter)
    ]
    if not candidates:
        return None
    return float(max(candidates, key=lambda item: quarter_sort_key(item[0]))[1])


def _metric_is_kp_pressure(metric_name: str) -> bool:
    lowered = str(metric_name or "").lower()
    if "new_hiv_infections" in lowered or "incidence" in lowered:
        return False
    return any(token in lowered for token in KP_PRESSURE_METRIC_TOKENS)


def _load_kp_metric_series(
    *,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    max_quarter: str,
) -> dict[str, dict[str, float]]:
    archive_path = Path(epigraph_root) / "artifacts" / "runs" / source_run_id / "harp_archive" / "historical_metric_rows.json"
    contract_lookup, _ledger = build_observation_contract_lookup(
        Path(epigraph_root),
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    series: dict[str, dict[str, float]] = {}
    for row_index, row in enumerate(list(read_json(archive_path, default=[]) or [])):
        if str(row.get("region") or "").lower() != "national":
            continue
        metric_name = str(row.get("metric_name") or "")
        if not _metric_is_kp_pressure(metric_name):
            continue
        value = _finite_float(row.get("value"))
        if value is None:
            continue
        quarter = _quarter_from_time_label(str(row.get("time") or row.get("period_end") or ""))
        if quarter is None or quarter_sort_key(quarter) > quarter_sort_key(max_quarter):
            continue
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = dict(contract_lookup.get(row_hash) or {})
        if str(contract.get("observation_role") or "") not in KP_ALLOWED_ROLES:
            continue
        series.setdefault(metric_name, {})[quarter] = max(float(value), 0.0)
    return series


def _kp_pressure_scores(
    *,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    train_quarters: list[str],
    forecast_quarters: list[str],
    target_log_hazards: list[float],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]] | None:
    if len(train_quarters) != len(target_log_hazards) or len(train_quarters) < 4:
        return None
    max_quarter = train_quarters[-1]
    series = _load_kp_metric_series(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        max_quarter=max_quarter,
    )
    usable: dict[str, dict[str, float]] = {}
    for metric_name, values in series.items():
        carried = [_latest_past_value(values, quarter) for quarter in train_quarters]
        finite = [float(value) for value in carried if value is not None]
        if len(finite) < 3:
            continue
        array = np.asarray([float(value if value is not None else finite[0]) for value in carried], dtype=np.float64)
        if float(np.std(array)) <= 0.0:
            continue
        usable[metric_name] = dict(values)
    if not usable:
        return None
    columns: list[np.ndarray] = []
    metric_names: list[str] = []
    stats: dict[str, dict[str, float]] = {}
    for metric_name, values in sorted(usable.items()):
        carried = [_latest_past_value(values, quarter) for quarter in train_quarters]
        fallback = next(float(value) for value in carried if value is not None)
        raw = np.asarray([float(value if value is not None else fallback) for value in carried], dtype=np.float64)
        center = float(np.mean(raw))
        scale = float(np.std(raw))
        if scale <= 0.0:
            continue
        columns.append((raw - center) / scale)
        metric_names.append(metric_name)
        stats[metric_name] = {"center": center, "scale": scale, "last_train_value": float(raw[-1])}
    if not columns:
        return None
    matrix = np.column_stack(columns)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    weights = np.asarray(vt[0], dtype=np.float64)
    train_scores_array = matrix @ weights
    target = np.asarray(target_log_hazards, dtype=np.float64)
    if float(np.std(train_scores_array)) > 0.0 and float(np.std(target)) > 0.0:
        corr = float(np.corrcoef(train_scores_array, target)[0, 1])
        if np.isfinite(corr) and corr < 0.0:
            weights = -weights
            train_scores_array = -train_scores_array
    forecast_scores: dict[str, float] = {}
    for quarter in forecast_quarters:
        raw_values = []
        for metric_name in metric_names:
            values = usable[metric_name]
            latest = _latest_past_value(values, max_quarter)
            if latest is None:
                latest = stats[metric_name]["last_train_value"]
            raw_values.append((float(latest) - stats[metric_name]["center"]) / stats[metric_name]["scale"])
        forecast_scores[quarter] = float(np.asarray(raw_values, dtype=np.float64) @ weights)
    train_scores = {quarter: float(value) for quarter, value in zip(train_quarters, train_scores_array)}
    summary = {
        "metric_count": len(metric_names),
        "metric_names": metric_names,
        "weight_by_metric": {metric: float(weight) for metric, weight in zip(metric_names, weights)},
        "score_contract": "train-window standardized one-component SVD, sign oriented by train incidence-hazard correlation, holdout values carried forward from forecast origin",
    }
    return train_scores, forecast_scores, summary


def _phase2_stable_scores(
    *,
    epigraph_root: Path,
    source_run_id: str,
    train_quarters: list[str],
    forecast_quarters: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, Any]] | None:
    try:
        phase2_source_run_id, source_resolution = resolve_phase2_structural_source_run_id(
            Path(epigraph_root),
            observation_source_run_id=source_run_id,
        )
        structural_inputs = load_phase2_structural_inputs(Path(epigraph_root), phase2_source_run_id)
        robustness_report, robustness_resolution = resolve_phase2_determinant_robustness_report(
            Path(epigraph_root),
            phase2_source_run_id,
        )
    except (FileNotFoundError, ValueError):
        return None
    allowed = allowed_direct_edge_keys_from_robustness(robustness_report)
    if not allowed:
        return None
    quarter_index = {quarter: idx for idx, quarter in enumerate(structural_inputs.quarter_axis)}
    block_index = {block: idx for idx, block in enumerate(structural_inputs.block_axis)}
    tensor = np.asarray(structural_inputs.national_quarter_tensor[0], dtype=np.float64)
    train_origin_idx = max([quarter_index[quarter] for quarter in train_quarters if quarter in quarter_index] or [-1])
    train_features: dict[str, dict[str, float]] = {quarter: {} for quarter in train_quarters}
    forecast_features: dict[str, dict[str, float]] = {quarter: {} for quarter in forecast_quarters}
    used_edges: list[str] = []
    for row in list(structural_inputs.direct_edge_rows or []):
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        lag = int(row.get("lag") or 0)
        edge_key = f"direct:{source}->{target}:lag{lag}"
        if edge_key not in allowed or source not in block_index:
            continue
        source_block_index = int(block_index[source])
        raw_train: list[float] = []
        for quarter in train_quarters:
            q_idx = quarter_index.get(quarter)
            source_idx = None if q_idx is None else q_idx - lag
            raw_train.append(float(tensor[source_idx, source_block_index]) if source_idx is not None and source_idx >= 0 else 0.0)
        center = float(np.mean(np.asarray(raw_train, dtype=np.float64))) if raw_train else 0.0
        scale = float(np.std(np.asarray(raw_train, dtype=np.float64))) if raw_train else 0.0
        if scale <= 0.0:
            continue
        for quarter, raw_value in zip(train_quarters, raw_train):
            train_features[quarter][edge_key] = float((float(raw_value) - center) / scale)
        for quarter in forecast_quarters:
            q_idx = quarter_index.get(quarter)
            source_idx = None if q_idx is None else q_idx - lag
            if source_idx is None:
                raw_value = 0.0
            else:
                safe_idx = min(max(int(source_idx), 0), int(train_origin_idx), int(tensor.shape[0]) - 1)
                raw_value = float(tensor[safe_idx, source_block_index])
            forecast_features[quarter][edge_key] = float((raw_value - center) / scale)
        used_edges.append(edge_key)
    if not used_edges:
        return None
    summary = {
        "phase2_source_run_id": phase2_source_run_id,
        "source_resolution": source_resolution,
        "robustness_resolution": robustness_resolution,
        "allowed_edge_count": len(allowed),
        "used_edge_count": len(used_edges),
        "used_edge_keys": used_edges,
        "feature_contract": "source-stable direct Phase2 edges only; forecast-origin safe carry-forward for future source tensor values",
    }
    return train_features, forecast_features, summary


def _feature_matrix(
    *,
    train_quarters: list[str],
    train_log_hazards: list[float],
    train_kp_scores: dict[str, float] | None,
    train_phase2_scores: dict[str, dict[str, float]] | None,
    use_kp: bool,
    use_phase2: bool,
    use_r10_mechanistic: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if len(train_quarters) < 2:
        return None
    phase2_names = sorted({name for values in dict(train_phase2_scores or {}).values() for name in values})
    if use_r10_mechanistic:
        feature_names = [
            "intercept",
            "lag_log_s_eff_rate",
            "lag_log_s_eff_rate_delta",
            "lag_log_s_eff_rate_curvature",
            "time_index",
        ]
        first_index = 2
    else:
        feature_names = ["intercept", "time_index", "lag_log_s_eff_hazard"]
        first_index = 1
    if use_kp:
        feature_names.append("kp_pressure_score")
    if use_phase2:
        feature_names.extend([f"phase2:{name}" for name in phase2_names])
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for idx in range(first_index, len(train_quarters)):
        quarter = train_quarters[idx]
        if use_r10_mechanistic:
            previous_delta = float(train_log_hazards[idx - 1]) - float(train_log_hazards[idx - 2])
            previous_previous_delta = (
                float(train_log_hazards[idx - 2]) - float(train_log_hazards[idx - 3])
                if idx >= 3
                else 0.0
            )
            row = [
                1.0,
                float(train_log_hazards[idx - 1]),
                previous_delta,
                previous_delta - previous_previous_delta,
                float(idx),
            ]
        else:
            row = [1.0, float(idx), float(train_log_hazards[idx - 1])]
        if use_kp:
            if train_kp_scores is None or quarter not in train_kp_scores:
                return None
            row.append(float(train_kp_scores[quarter]))
        if use_phase2:
            values = dict((train_phase2_scores or {}).get(quarter) or {})
            if not phase2_names:
                return None
            row.extend(float(values.get(name) or 0.0) for name in phase2_names)
        x_rows.append(row)
        y_rows.append(float(train_log_hazards[idx]))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    beta = _pinv_fit(x, y)
    if beta is None:
        return None
    return x, y, feature_names


def fit_incidence_module_head(
    *,
    family: str,
    dataset: Any,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
) -> IncidenceModuleHead | None:
    if family not in INCIDENCE_MODULE_FAMILIES or family == "carry_forward_s_eff_hazard":
        return None
    base_family = _base_family_for_measurement_family(family)
    pairs = _train_hazard_pairs(dataset)
    if len(pairs) < 3:
        return None
    train_quarters = [quarter for quarter, _hazard in pairs]
    train_hazards = [max(float(hazard), float(dataset.eps)) for _quarter, hazard in pairs]
    train_log_hazards = [float(np.log(value)) for value in train_hazards]
    forecast_quarters = [str(row.get("quarter") or "") for row in sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    use_kp = "kp_pressure" in base_family
    use_phase2 = "phase2_stable" in base_family
    use_r10_mechanistic = _is_r10_mechanistic_family(base_family)
    kp_train_scores: dict[str, float] | None = None
    kp_summary = None
    if use_kp:
        kp = _kp_pressure_scores(
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
            train_quarters=train_quarters,
            forecast_quarters=forecast_quarters,
            target_log_hazards=train_log_hazards,
        )
        if kp is None:
            return None
        kp_train_scores, _kp_forecast_scores, kp_summary = kp
    phase2_train_scores: dict[str, dict[str, float]] | None = None
    phase2_summary = None
    if use_phase2:
        phase2 = _phase2_stable_scores(
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            train_quarters=train_quarters,
            forecast_quarters=forecast_quarters,
        )
        if phase2 is None:
            return None
        phase2_train_scores, _phase2_forecast_scores, phase2_summary = phase2
    design = _feature_matrix(
        train_quarters=train_quarters,
        train_log_hazards=train_log_hazards,
        train_kp_scores=kp_train_scores,
        train_phase2_scores=phase2_train_scores,
        use_kp=use_kp,
        use_phase2=use_phase2,
        use_r10_mechanistic=use_r10_mechanistic,
    )
    if design is None:
        return None
    x, y, feature_names = design
    beta = np.linalg.pinv(x) @ y
    return IncidenceModuleHead(
        family=family,
        feature_names=feature_names,
        coefficients=[float(value) for value in beta],
        train_row_count=int(x.shape[0]),
        hazard_floor=float(min(train_hazards)),
        hazard_ceiling=float(max(train_hazards)),
        kp_feature_summary=kp_summary,
        phase2_feature_summary=phase2_summary,
    )


def _forecast_scores_for_head(
    *,
    head: IncidenceModuleHead,
    dataset: Any,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    pairs = _train_hazard_pairs(dataset)
    train_quarters = [quarter for quarter, _hazard in pairs]
    train_hazards = [max(float(hazard), float(dataset.eps)) for _quarter, hazard in pairs]
    train_log_hazards = [float(np.log(value)) for value in train_hazards]
    forecast_quarters = [str(row.get("quarter") or "") for row in sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    use_kp = "kp_pressure_score" in head.feature_names
    use_phase2 = any(name.startswith("phase2:") for name in head.feature_names)
    kp_forecast_scores: dict[str, float] | None = None
    kp_summary = head.kp_feature_summary
    if use_kp:
        kp = _kp_pressure_scores(
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
            train_quarters=train_quarters,
            forecast_quarters=forecast_quarters,
            target_log_hazards=train_log_hazards,
        )
        if kp is None:
            return None, {}
        _kp_train, kp_forecast_scores, kp_summary = kp
    phase2_forecast_scores: dict[str, dict[str, float]] | None = None
    phase2_summary = head.phase2_feature_summary
    if use_phase2:
        phase2 = _phase2_stable_scores(
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            train_quarters=train_quarters,
            forecast_quarters=forecast_quarters,
        )
        if phase2 is None:
            return None, {}
        _phase2_train, phase2_forecast_scores, phase2_summary = phase2
    beta = np.asarray(head.coefficients, dtype=np.float64)
    current_log_hazard = float(train_log_hazards[-1])
    log_history = [float(value) for value in train_log_hazards]
    forecast_hazards: dict[str, float] = {}
    phase2_feature_names = [name.replace("phase2:", "", 1) for name in head.feature_names if name.startswith("phase2:")]
    use_r10_mechanistic = "lag_log_s_eff_rate_delta" in head.feature_names
    for step, quarter in enumerate(forecast_quarters, start=1):
        if use_r10_mechanistic:
            previous_delta = float(log_history[-1]) - float(log_history[-2]) if len(log_history) >= 2 else 0.0
            previous_previous_delta = float(log_history[-2]) - float(log_history[-3]) if len(log_history) >= 3 else 0.0
            row = [
                1.0,
                current_log_hazard,
                previous_delta,
                previous_delta - previous_previous_delta,
                float(len(train_quarters) - 1 + step),
            ]
        else:
            row = [1.0, float(len(train_quarters) - 1 + step), current_log_hazard]
        if use_kp:
            row.append(float((kp_forecast_scores or {}).get(quarter) or 0.0))
        if use_phase2:
            values = dict((phase2_forecast_scores or {}).get(quarter) or {})
            row.extend(float(values.get(name) or 0.0) for name in phase2_feature_names)
        x = np.asarray(row, dtype=np.float64)
        if x.size != beta.size:
            return None, {}
        predicted_log = float(x @ beta)
        hazard = float(np.exp(predicted_log))
        hazard = float(np.clip(hazard, float(head.hazard_floor), float(head.hazard_ceiling)))
        forecast_hazards[quarter] = hazard
        current_log_hazard = float(np.log(max(hazard, float(dataset.eps))))
        log_history.append(current_log_hazard)
    diagnostics = {
        "kp_feature_summary": kp_summary,
        "phase2_feature_summary": phase2_summary,
        "forecast_hazard_bounds": {
            "floor": float(head.hazard_floor),
            "ceiling": float(head.hazard_ceiling),
            "contract": "bounded to train-window empirical S_eff per-quarter incidence-rate range",
        },
        "r10_mechanistic_head": {
            "enabled": bool(use_r10_mechanistic),
            "feature_contract": (
                "R10-like endpoint delta features applied to the mechanistic S_eff per-quarter incidence-rate backbone"
                if use_r10_mechanistic
                else "not_enabled"
            ),
        },
    }
    return forecast_hazards, diagnostics


def carry_forward_s_eff_hazard(dataset: Any) -> dict[str, float]:
    pairs = _train_hazard_pairs(dataset)
    if not pairs:
        return {}
    last_hazard = float(pairs[-1][1])
    return {
        str(row.get("quarter") or ""): last_hazard
        for row in sorted(list(dataset.holdout_rows), key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    }


def s_eff_incidence_to_u(
    *,
    dataset: Any,
    hazard_map: dict[str, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not holdout_rows or not dataset.train_state_rows:
        return {}, []
    previous_state = canonical_state_values(dict(dataset.train_state_rows[-1].get("state_values") or {}))
    previous_quarter = str(dataset.train_state_rows[-1].get("quarter") or "")
    previous_population = _finite_float(dataset.train_rows[-1].get("population_total")) if dataset.train_rows else None
    incidence_map: dict[str, float] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for row in holdout_rows:
        quarter = str(row.get("quarter") or "")
        population = _finite_float(row.get("population_total"))
        if population is None:
            population = previous_population
        if population is None:
            population = state_sum(previous_state, STATE_NAMES)
        s_eff = max(float(population) - state_sum(previous_state, STATE_NAMES), 0.0)
        hazard = max(float(hazard_map.get(quarter) or 0.0), 0.0)
        interval_quarters = _interval_quarter_gap(previous_quarter, quarter)
        incidence = _quarter_rate_to_interval_incidence(hazard, float(s_eff), interval_quarters)
        previous_state["U"] = max(float(previous_state["U"]) + incidence, 0.0)
        incidence_map[quarter] = float(incidence)
        trajectory_rows.append(
            {
                "quarter": quarter,
                "population_denominator": float(population),
                "susceptible_effective": float(s_eff),
                "incidence_hazard_per_s_eff": float(hazard),
                "interval_quarters": int(interval_quarters),
                "incidence_inflow": float(incidence),
                "state_values_after_incidence": {name: float(value) for name, value in previous_state.items()},
                "contract": "standalone interval-aware incidence module: S_eff -> incident infections -> U; downstream care states are not tuned here",
                "hazard_semantics": "continuous per-quarter incidence rate applied over observed transition interval",
            }
        )
        previous_population = float(population)
        previous_quarter = quarter
    return incidence_map, trajectory_rows


def _annual_score_entries(
    *,
    family: str,
    incidence_map: dict[str, float],
    carry_forward_map: dict[str, float],
    targets: dict[int, dict[str, Any]],
    train_end_year: int,
    horizon_years: int,
    latent_incidence_map: dict[str, float] | None = None,
    weak_measurement_calibrator: WeakAnnualMeasurementCalibrator | None = None,
    annual_measurement_readout_head: AnnualMeasurementReadoutHead | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    years = sorted({quarter_year(quarter) for quarter in incidence_map})
    for year in years:
        target = targets.get(year)
        if target is None:
            continue
        candidate_value, aggregation_contract = _annual_from_mixed_frequency_map(incidence_map, int(year))
        if candidate_value is None:
            continue
        target_value = _finite_float(target.get("annual_new_infections"))
        if target_value is None or target_value <= 0.0:
            continue
        carry_value, carry_contract = _annual_from_mixed_frequency_map(carry_forward_map, int(year))
        if carry_value is None:
            continue
        latent_value = None
        latent_contract = None
        if latent_incidence_map is not None:
            latent_value, latent_contract = _annual_from_mixed_frequency_map(latent_incidence_map, int(year))
        scale = max(float(target_value), 1.0)
        row = {
            "family": family,
            "train_end_year": int(train_end_year),
            "horizon_years": int(horizon_years),
            "year": int(year),
            "target_annual_new_infections": float(target_value),
            "candidate_annual_incidence": candidate_value,
            "latent_annual_incidence": latent_value,
            "carry_forward_annual_incidence": carry_value,
            "annual_incidence_aggregation_contract": aggregation_contract,
            "carry_forward_aggregation_contract": carry_contract,
            "latent_aggregation_contract": latent_contract,
            "candidate_norm_error": abs(candidate_value - float(target_value)) / scale,
            "carry_forward_norm_error": abs(carry_value - float(target_value)) / scale,
            "candidate_minus_carry_forward_norm_error": (
                abs(candidate_value - float(target_value)) - abs(carry_value - float(target_value))
            )
            / scale,
            "target_allowed_use": str(((target.get("metric_provenance") or {}).get("annual_new_infections") or {}).get("allowed_use") or ""),
            "annual_measurement_head_applied": bool(weak_measurement_calibrator is not None or annual_measurement_readout_head is not None),
            "annual_measurement_readout_applied": bool(annual_measurement_readout_head is not None),
            "candidate_value_contract": (
                "annual measurement readout output; quarterly latent incidence remains separate"
                if annual_measurement_readout_head is not None
                else
                "weak annual measurement head output; quarterly latent incidence remains separate"
                if weak_measurement_calibrator is not None
                else "latent quarterly incidence aggregated to annual validation scale"
            ),
        }
        if annual_measurement_readout_head is not None:
            sigma = annual_measurement_readout_head.residual_sigma_log
            row.update(
                {
                    "annual_measurement_readout_family": annual_measurement_readout_head.readout_family,
                    "annual_measurement_readout_train_years": ",".join(str(year_value) for year_value in annual_measurement_readout_head.train_years),
                    "annual_measurement_readout_sigma_log": sigma,
                    "annual_measurement_readout_training_objective": annual_measurement_readout_head.training_objective,
                }
            )
            if sigma is not None and sigma > 0.0 and candidate_value > 0.0 and target_value > 0.0:
                residual = float(np.log(candidate_value) - np.log(float(target_value)))
                row["annual_measurement_readout_log_residual"] = residual
                row["annual_measurement_readout_negative_log_score"] = float(
                    0.5 * (residual / float(sigma)) ** 2 + float(np.log(float(sigma))) + 0.5 * float(np.log(2.0 * np.pi))
                )
        if weak_measurement_calibrator is not None:
            sigma = weak_measurement_calibrator.residual_sigma_log
            row.update(
                {
                    "weak_measurement_train_annual_count": int(weak_measurement_calibrator.train_annual_count),
                    "weak_measurement_train_years": ",".join(str(year_value) for year_value in weak_measurement_calibrator.train_years),
                    "weak_measurement_sigma_log": sigma,
                    "weak_measurement_intercept": float(weak_measurement_calibrator.intercept),
                    "weak_measurement_slope": float(weak_measurement_calibrator.slope),
                    "weak_measurement_calibration_family": weak_measurement_calibrator.calibration_family,
                }
            )
            if sigma is not None and sigma > 0.0 and candidate_value > 0.0 and target_value > 0.0:
                residual = float(np.log(candidate_value) - np.log(float(target_value)))
                row["weak_measurement_log_residual"] = residual
                row["weak_measurement_negative_log_score"] = float(
                    0.5 * (residual / float(sigma)) ** 2 + float(np.log(float(sigma))) + 0.5 * float(np.log(2.0 * np.pi))
                )
        entries.append(row)
    return entries


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return None if not finite else float(np.mean(np.asarray(finite, dtype=np.float64)))


def _family_summary(
    *,
    family: str,
    split_rows: list[dict[str, Any]],
    r10_annual_error: float | None,
) -> dict[str, Any]:
    entries = [entry for row in split_rows for entry in list(row.get("annual_score_entries") or [])]
    candidate_mae = _mean([float(entry["candidate_norm_error"]) for entry in entries])
    carry_mae = _mean([float(entry["carry_forward_norm_error"]) for entry in entries])
    candidate_minus_carry = _mean([float(entry["candidate_minus_carry_forward_norm_error"]) for entry in entries])
    blockers: list[str] = []
    if not entries:
        blockers.append("no_complete_annual_incidence_validation_entries")
    if candidate_mae is not None and carry_mae is not None and candidate_mae >= carry_mae:
        blockers.append("annual_incidence_not_better_than_carry_forward_s_eff")
    if r10_annual_error is not None and candidate_mae is not None and candidate_mae >= float(r10_annual_error):
        blockers.append("annual_incidence_not_better_than_r10")
    if any(row.get("fit_status") != "completed" for row in split_rows):
        blockers.append("some_blocked_splits_failed_closed")
    return {
        "family": family,
        "split_count": len(split_rows),
        "completed_split_count": sum(1 for row in split_rows if row.get("fit_status") == "completed"),
        "annual_entry_count": len(entries),
        "candidate_norm_mae": candidate_mae,
        "carry_forward_norm_mae": carry_mae,
        "candidate_minus_carry_forward_norm_mae": candidate_minus_carry,
        "r10_annual_incidence_error": r10_annual_error,
        "promotion_eligible": bool(not blockers and family != "carry_forward_s_eff_hazard"),
        "blockers": blockers,
    }


def _apply_determinant_bundle_keep_rules(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family = {str(row.get("family") or ""): row for row in summaries}
    updated: list[dict[str, Any]] = []
    for row in summaries:
        next_row = dict(row)
        family = str(next_row.get("family") or "")
        counterpart = _determinant_free_counterpart(family)
        if counterpart:
            base = by_family.get(counterpart)
            candidate_mae = _finite_float(next_row.get("candidate_norm_mae"))
            base_mae = None if base is None else _finite_float(base.get("candidate_norm_mae"))
            next_row["determinant_bundle_counterpart"] = counterpart
            next_row["determinant_bundle_counterpart_norm_mae"] = base_mae
            blockers = list(next_row.get("blockers") or [])
            if base_mae is None:
                blockers.append("determinant_free_counterpart_unavailable")
            elif candidate_mae is None or candidate_mae >= float(base_mae) - float(np.finfo(np.float64).eps) * max(1.0, abs(float(base_mae))):
                blockers.append("determinant_bundle_not_better_than_mechanistic_counterpart")
            next_row["blockers"] = list(dict.fromkeys(blockers))
            next_row["promotion_eligible"] = bool(not next_row["blockers"] and family != "carry_forward_s_eff_hazard")
        updated.append(next_row)
    return updated


def _r10_source_report_annual_incidence_error(source_report: Path) -> float | None:
    payload = dict(read_json(source_report, default={}) or {}) if source_report.exists() else {}
    contracts = list(payload.get("contracts") or [])
    preferred = (
        ("purged_dense", "EXP-R10-DENSE-CHAMPION", "merged_current_champion"),
        ("purged_dense", "EXP-R10-DENSE-CHAMPION", "baseline_current_champion"),
        ("exact_only", "EXP-R10-EXACT-CHAMPION", "merged_current_champion"),
        ("exact_only", "EXP-R10-EXACT-CHAMPION", "baseline_current_champion"),
    )
    for contract_name, experiment_id, champion_key in preferred:
        for contract in contracts:
            champion = dict((contract or {}).get(champion_key) or {})
            if str(champion.get("contract") or (contract or {}).get("contract") or "") != contract_name:
                continue
            if str(champion.get("experiment_id") or "") != experiment_id:
                continue
            value = _finite_float(champion.get("annual_mean_incidence_error"))
            if value is not None:
                return value
    for contract in contracts:
        for champion_key in ("merged_current_champion", "baseline_current_champion"):
            champion = dict((contract or {}).get(champion_key) or {})
            value = _finite_float(champion.get("annual_mean_incidence_error"))
            if value is not None:
                return value
    return None


def _r10_annual_incidence_error(epigraph_root: Path) -> float | None:
    search_roots = [sandbox_repo_root(), Path(epigraph_root)]
    seen: set[str] = set()
    for root in search_roots:
        for path in sorted(root.glob("artifacts/runs/p3d-r10-baseline*/analysis/r10_baseline_report.json"), reverse=True):
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            payload = dict(read_json(path, default={}) or {})
            value = _finite_float(payload.get("reference_annual_incidence_error"))
            if value is not None:
                return value
        for path in sorted(root.glob("artifacts/runs/*/analysis/hybrid_champion_search_report.json"), reverse=True):
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            payload = dict(read_json(path, default={}) or {})
            reference = dict(payload.get("r10_reference") or {})
            value = _finite_float(reference.get("reference_annual_incidence_error"))
            if value is not None:
                return value
            source_report = Path(str((payload.get("r10_baseline") or {}).get("source_report") or ""))
            value = _r10_source_report_annual_incidence_error(source_report)
            if value is not None:
                return value
    canonical_source_report = (
        Path(epigraph_root)
        / "artifacts"
        / "runs"
        / "tr-v3-current-champion-r10-neighborhood-20260419-s00"
        / "analysis"
        / "tr_v3_current_champion_r10_neighborhood_batch_report.json"
    )
    return _r10_source_report_annual_incidence_error(canonical_source_report)


def _evaluate_split(
    *,
    family: str,
    dataset: Any,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    targets: dict[int, dict[str, Any]],
    train_end_year: int,
    horizon_years: int,
) -> dict[str, Any]:
    carry_hazard = carry_forward_s_eff_hazard(dataset)
    carry_incidence, carry_trajectory = s_eff_incidence_to_u(dataset=dataset, hazard_map=carry_hazard)
    head = None
    diagnostics: dict[str, Any] = {}
    if family == "carry_forward_s_eff_hazard":
        hazard_map = carry_hazard
        fit_status = "completed"
    else:
        head = fit_incidence_module_head(
            family=family,
            dataset=dataset,
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        )
        if head is None:
            return {
                "family": family,
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon_years),
                "fit_status": "failed_closed",
                "annual_score_entries": [],
                "fit_blocker": "insufficient_train_rows_or_required_covariate_support",
            }
        hazard_map, diagnostics = _forecast_scores_for_head(
            head=head,
            dataset=dataset,
            epigraph_root=epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        )
        if hazard_map is None:
            return {
                "family": family,
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon_years),
                "fit_status": "failed_closed",
                "annual_score_entries": [],
                "fit_blocker": "forecast_covariate_support_unavailable",
            }
        fit_status = "completed"
    incidence_map, trajectory_rows = s_eff_incidence_to_u(dataset=dataset, hazard_map=dict(hazard_map))
    scoring_incidence_map = dict(incidence_map)
    measurement_calibrator = None
    annual_readout_head = None
    measurement_rows: list[dict[str, Any]] = []
    if _uses_weak_annual_measurement(family):
        measurement_calibrator = _fit_weak_annual_measurement_calibrator(
            dataset=dataset,
            targets=targets,
            train_end_year=train_end_year,
        )
        if measurement_calibrator is None:
            return {
                "family": family,
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon_years),
                "fit_status": "failed_closed",
                "annual_score_entries": [],
                "fit_blocker": "insufficient_pre_holdout_annual_incidence_measurement_support",
            }
        scoring_incidence_map, measurement_rows = _apply_weak_annual_measurement_calibration(
            incidence_map=dict(incidence_map),
            calibrator=measurement_calibrator,
        )
        diagnostics["weak_annual_measurement"] = {
            "intercept": float(measurement_calibrator.intercept),
            "slope": float(measurement_calibrator.slope),
            "residual_sigma_log": measurement_calibrator.residual_sigma_log,
            "train_annual_count": int(measurement_calibrator.train_annual_count),
            "train_years": list(measurement_calibrator.train_years),
            "train_residuals_log": list(measurement_calibrator.train_residuals_log),
            "calibration_family": measurement_calibrator.calibration_family,
            "calibration_rows": measurement_rows,
            "contract": "annual incidence rows fit a proportional log-bias observation head on train years only; no quarterly diagnosis-flow incidence target is used",
        }
    if _uses_annual_measurement_readout(family):
        annual_readout_head = _fit_annual_measurement_readout_head(
            targets=targets,
            train_end_year=train_end_year,
        )
        if annual_readout_head is None:
            return {
                "family": family,
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon_years),
                "fit_status": "failed_closed",
                "annual_score_entries": [],
                "fit_blocker": "insufficient_pre_holdout_annual_incidence_measurement_readout_support",
            }
        scoring_incidence_map, measurement_rows = _apply_annual_measurement_readout(
            incidence_map=dict(incidence_map),
            readout_head=annual_readout_head,
        )
        diagnostics["annual_measurement_readout"] = {
            "readout_family": annual_readout_head.readout_family,
            "train_years": list(annual_readout_head.train_years),
            "train_log_values": list(annual_readout_head.train_log_values),
            "train_residuals_log": list(annual_readout_head.train_residuals_log),
            "residual_sigma_log": annual_readout_head.residual_sigma_log,
            "forecast_log_floor": float(annual_readout_head.forecast_log_floor),
            "forecast_log_ceiling": float(annual_readout_head.forecast_log_ceiling),
            "calibration_rows": measurement_rows,
            "contract": "annual incidence rows train a recursive R10-style log-delta measurement readout on train years only; no quarterly diagnosis-flow incidence target is used",
        }
    entries = _annual_score_entries(
        family=family,
        incidence_map=scoring_incidence_map,
        carry_forward_map=carry_incidence,
        targets=targets,
        train_end_year=train_end_year,
        horizon_years=horizon_years,
        latent_incidence_map=incidence_map if (measurement_calibrator is not None or annual_readout_head is not None) else None,
        weak_measurement_calibrator=measurement_calibrator,
        annual_measurement_readout_head=annual_readout_head,
    )
    return {
        "family": family,
        "train_end_year": int(train_end_year),
        "holdout_years": list(dataset.holdout_years),
        "horizon_years": int(horizon_years),
        "fit_status": fit_status,
        "head": None if head is None else {
            "family": head.family,
            "feature_names": list(head.feature_names),
            "coefficients": list(head.coefficients),
            "train_row_count": int(head.train_row_count),
            "hazard_floor": float(head.hazard_floor),
            "hazard_ceiling": float(head.hazard_ceiling),
            "training_objective": head.training_objective,
            "kp_feature_summary": head.kp_feature_summary,
            "phase2_feature_summary": head.phase2_feature_summary,
        },
        "diagnostics": diagnostics,
        "hazard_map": {quarter: float(value) for quarter, value in dict(hazard_map).items()},
        "incidence_map": incidence_map,
        "weak_measurement_incidence_map": scoring_incidence_map if measurement_calibrator is not None else None,
        "annual_measurement_readout_incidence_map": scoring_incidence_map if annual_readout_head is not None else None,
        "carry_forward_incidence_map": carry_incidence,
        "trajectory_rows": trajectory_rows,
        "carry_forward_trajectory_rows": carry_trajectory,
        "annual_score_entries": entries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    summaries = list(payload.get("family_summaries") or [])
    label_lookup = {
        "carry_forward_s_eff_hazard": "S_eff CF",
        "s_eff_hazard_ar": "AR hazard",
        "s_eff_hazard_ar_weak_annual_measurement": "AR + annual weak",
        "s_eff_hazard_r10_mechanistic": "R10 mech",
        "s_eff_hazard_r10_mechanistic_weak_annual_measurement": "R10 mech + annual weak",
        "s_eff_hazard_r10_mechanistic_annual_measurement_readout": "R10 mech + annual readout",
        "s_eff_hazard_r10_mechanistic_kp_pressure": "R10 mech + KP",
        "s_eff_hazard_r10_mechanistic_kp_pressure_weak_annual_measurement": "R10 mech + KP + annual weak",
        "s_eff_hazard_r10_mechanistic_kp_pressure_annual_measurement_readout": "R10 mech + KP + annual readout",
        "s_eff_hazard_r10_mechanistic_phase2_stable": "R10 mech + P2",
        "s_eff_hazard_r10_mechanistic_phase2_stable_weak_annual_measurement": "R10 mech + P2 + annual weak",
        "s_eff_hazard_r10_mechanistic_phase2_stable_annual_measurement_readout": "R10 mech + P2 + annual readout",
        "s_eff_hazard_r10_mechanistic_kp_phase2_stable": "R10 mech + KP+P2",
        "s_eff_hazard_r10_mechanistic_kp_phase2_stable_weak_annual_measurement": "R10 mech + KP+P2 + annual weak",
        "s_eff_hazard_r10_mechanistic_kp_phase2_stable_annual_measurement_readout": "R10 mech + KP+P2 + annual readout",
        "s_eff_hazard_kp_pressure": "KP pressure",
        "s_eff_hazard_kp_pressure_weak_annual_measurement": "KP + annual weak",
        "s_eff_hazard_phase2_stable": "P2 stable",
        "s_eff_hazard_phase2_stable_weak_annual_measurement": "P2 + annual weak",
        "s_eff_hazard_kp_phase2_stable": "KP+P2",
        "s_eff_hazard_kp_phase2_stable_weak_annual_measurement": "KP+P2 + annual weak",
    }
    labels = [label_lookup.get(str(row.get("family") or ""), str(row.get("family") or "")) for row in summaries]
    values = [
        np.nan if row.get("candidate_norm_mae") is None else float(row.get("candidate_norm_mae") or 0.0)
        for row in summaries
    ]
    carry = [
        np.nan if row.get("carry_forward_norm_mae") is None else float(row.get("carry_forward_norm_mae") or 0.0)
        for row in summaries
    ]
    r10 = _finite_float((payload.get("r10_reference") or {}).get("annual_incidence_error"))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, values, width=0.36, label="candidate", color="#2f6f73")
    axes[0].bar(x + 0.18, carry, width=0.36, label="S_eff carry-forward", color="#c58b3b")
    if r10 is not None:
        axes[0].axhline(float(r10), color="#111111", linestyle="--", linewidth=1.2, label="R10 annual incidence")
    for idx, row in enumerate(summaries):
        if row.get("candidate_norm_mae") is None:
            axes[0].text(idx, 0.03, "failed\nclosed", ha="center", va="bottom", fontsize=7, color="#b23a48")
    axes[0].set_xticks(x, labels, rotation=0, ha="center")
    axes[0].set_ylabel("validation-normalized annual incidence MAE")
    axes[0].set_title("A. Validation-only annual incidence")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    finite_values = [float(value) for value in [*values, *carry] if np.isfinite(float(value))]
    if finite_values:
        axes[0].set_ylim(0.0, max(finite_values) * 1.18)
    completed = [int(row.get("completed_split_count") or 0) for row in summaries]
    failed = [max(int(row.get("split_count") or 0) - int(row.get("completed_split_count") or 0), 0) for row in summaries]
    axes[1].bar(x, completed, color="#2f6f73", label="completed")
    axes[1].bar(x, failed, bottom=completed, color="#b23a48", label="failed closed")
    axes[1].set_xticks(x, labels, rotation=0, ha="center")
    axes[1].set_ylabel("blocked splits")
    axes[1].set_title("B. Covariate support gate")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Separate S_eff Incidence Module Search", fontweight="bold")
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    def format_optional(value: Any) -> str:
        numeric = _finite_float(value)
        return "NA" if numeric is None else f"{numeric:.6f}"

    lines = [
        "# Separate Incidence Module Search",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source run: `{payload.get('source_run_id')}`",
        f"- Contract: `{(payload.get('contract') or {}).get('incidence_equation')}`",
        "",
        "## Family Scores",
        "",
        "| family | promoted | annual incidence MAE | carry-forward MAE | entries | blockers |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in list(payload.get("family_summaries") or []):
        lines.append(
            "| {family} | `{promoted}` | {candidate} | {carry} | {entries} | `{blockers}` |".format(
                family=row.get("family"),
                promoted=bool(row.get("promotion_eligible")),
                candidate=format_optional(row.get("candidate_norm_mae")),
                carry=format_optional(row.get("carry_forward_norm_mae")),
                entries=int(row.get("annual_entry_count") or 0),
                blockers=list(row.get("blockers") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Annual incidence rows are validation-only in this pass.",
            "- Diagnosis-flow rows are not used as incidence targets.",
            "- KP pressure is a train-window SVD covariate and fails closed when support is insufficient.",
            "- Phase 2 determinants are source-stable direct covariates only; exploratory edges are not used.",
            "- R10/mechanistic incidence heads use endpoint-style lag/delta/curvature features on the interval-aware S_eff hazard backbone.",
            "- Determinant bundle variants are promotion-blocked unless they improve over their determinant-free counterpart.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_incidence_module_search(
    *,
    run_id: str = "p3d-incidence-module-search-20260429-s00",
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    families: tuple[str, ...] = INCIDENCE_MODULE_FAMILIES,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    active_source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    active_baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=active_source_run_id,
        preferred=str(baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
        include_validation_only=True,
    )
    targets = _annual_incidence_targets(validation_rows)
    family_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    for horizon_years in (1, 5):
        splits = rolling_origin_splits(
            observation_rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon_years),
        )
        for split in splits:
            dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
            for family in families:
                row = _evaluate_split(
                    family=family,
                    dataset=dataset,
                    epigraph_root=epigraph_root,
                    source_run_id=active_source_run_id,
                    baseline_source_run_id=active_baseline_source_run_id,
                    targets=targets,
                    train_end_year=int(split["train_end_year"]),
                    horizon_years=int(horizon_years),
                )
                family_rows[family].append(row)
    r10_annual_error = _r10_annual_incidence_error(epigraph_root)
    summaries = [
        _family_summary(
            family=family,
            split_rows=rows,
            r10_annual_error=r10_annual_error,
        )
        for family, rows in family_rows.items()
    ]
    summaries = _apply_determinant_bundle_keep_rules(summaries)
    promoted = [row for row in summaries if bool(row.get("promotion_eligible"))]
    if promoted:
        best = min(promoted, key=lambda row: (float(row.get("candidate_norm_mae") or float("inf")), str(row.get("family") or "")))
    else:
        best = min(summaries, key=lambda row: (len(list(row.get("blockers") or [])), float(row.get("candidate_norm_mae") or float("inf")), str(row.get("family") or "")))
    score_entries = [entry for rows in family_rows.values() for row in rows for entry in list(row.get("annual_score_entries") or [])]
    split_diagnostics = []
    for rows in family_rows.values():
        for row in rows:
            split_diagnostics.append(
                {
                    "family": row.get("family"),
                    "train_end_year": row.get("train_end_year"),
                    "horizon_years": row.get("horizon_years"),
                    "fit_status": row.get("fit_status"),
                    "fit_blocker": row.get("fit_blocker"),
                    "feature_names": None if not row.get("head") else list((row.get("head") or {}).get("feature_names") or []),
                    "kp_metric_count": None if not row.get("head") else ((row.get("head") or {}).get("kp_feature_summary") or {}).get("metric_count"),
                    "phase2_used_edge_count": None if not row.get("head") else ((row.get("head") or {}).get("phase2_feature_summary") or {}).get("used_edge_count"),
                }
            )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    payload = {
        "schema_version": INCIDENCE_MODULE_SEARCH_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": active_source_run_id,
        "baseline_source_run_id": active_baseline_source_run_id,
        "contract": {
            "incidence_equation": "S_eff(t)=max(population_total(t)-current_PLHIV_state_total(t-),0); I_interval=S_eff(t)*(1-exp(-lambda_per_quarter*delta_quarters)); I_interval enters U before downstream care transitions",
            "training_targets": "stock-balance derived interval incidence converted to continuous per-quarter S_eff incidence rate from train rows only",
            "validation_targets": "annual_new_infections are validation-only; no quarterly diagnosis-flow incidence readout is fitted",
            "weak_annual_measurement": "optional branch fits proportional log-bias annual incidence measurement calibration on train years only and scores held-out annual years",
            "annual_measurement_readout": "optional branch recursively forecasts held-out annual incidence measurement scale from pre-holdout annual measurements, then rescales the latent mechanistic quarterly incidence path",
            "r10_mechanistic_hybrid": "R10-like lag/delta/curvature endpoint-head features are applied to the mechanistic interval-aware S_eff incidence-rate backbone, not to diagnosis-flow-derived incidence",
            "kp_pressure": "train-window standardized one-component SVD over ledger-allowed KP/youth/prep covariate rows; forecast values carried forward from forecast origin",
            "phase2_determinants": "source-stable direct Phase2 edges only; missing or exploratory-only robustness reports fail closed",
            "hazard_bounds": "forecast per-quarter incidence rates are clipped to the empirical train-window S_eff incidence-rate range",
            "determinant_bundle_keep_rule": "determinant bundle variants must beat carry-forward, R10 annual incidence reference, and their determinant-free mechanistic counterpart before promotion",
        },
        "diagnosis_flow_reference": {
            "repair_family": "selective_monthly_nowcast_backlog",
            "role": "diagnosis_flow_repair_only",
            "incidence_policy": "do_not_use_diagnosis_flow_derived_incidence_readout",
            "source_audit": (
                sandbox_repo_root()
                / "artifacts"
                / "scientific_audits"
                / "phase3_diagnosis_flow_readout_decomposition_20260429.md"
            ).as_posix(),
        },
        "benchmark_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizons": [1, 5],
        },
        "validation_target_years": sorted(targets.keys()),
        "r10_reference": {"annual_incidence_error": r10_annual_error},
        "family_summaries": summaries,
        "best_family": best,
        "annual_score_entries": score_entries,
        "split_diagnostics": split_diagnostics,
        "artifact_paths": {
            "json": (analysis_dir / "incidence_module_search.json").as_posix(),
            "markdown": (analysis_dir / "incidence_module_search.md").as_posix(),
            "dashboard_png": (analysis_dir / "incidence_module_search_dashboard.png").as_posix(),
            "annual_score_entries_csv": (analysis_dir / "incidence_module_annual_score_entries.csv").as_posix(),
            "split_diagnostics_csv": (analysis_dir / "incidence_module_split_diagnostics.csv").as_posix(),
        },
    }
    write_json(Path(payload["artifact_paths"]["json"]), payload)
    Path(payload["artifact_paths"]["markdown"]).write_text(_markdown_report(payload), encoding="utf-8")
    _write_csv(Path(payload["artifact_paths"]["annual_score_entries_csv"]), score_entries)
    _write_csv(Path(payload["artifact_paths"]["split_diagnostics_csv"]), split_diagnostics)
    _write_dashboard(payload, Path(payload["artifact_paths"]["dashboard_png"]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separate S_eff incidence module search.")
    parser.add_argument("--run-id", default="p3d-incidence-module-search-20260429-s00")
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    payload = run_incidence_module_search(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"]["json"])


if __name__ == "__main__":
    main()
