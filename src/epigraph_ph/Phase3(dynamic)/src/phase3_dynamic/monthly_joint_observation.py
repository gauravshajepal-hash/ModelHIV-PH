from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import inv_logit, logit, quarter_ordinal, quarter_sort_key
from .monthly_shock import (
    MonthlyShockContext,
    _chosen_month_metric_values,
    _fit_linear_effect,
    _linear_effect,
    _month_label,
    _quarter_month_range,
    _tukey_upper_fence,
    load_monthly_signal_rows,
)


MONTHLY_JOINT_SCHEMA_VERSION = "phase3_dynamic_monthly_joint_observation_state.v1"
JOINT_MONTHLY_METRICS: tuple[str, ...] = (
    "new_diagnosed_cases_period",
    "new_diagnosed_cases_monthly",
    "diagnosed_plhiv",
    "alive_on_art",
    "deaths_reported_period",
    "median_cd4_at_enrollment",
)


@dataclass(frozen=True, slots=True)
class MonthlyJointContext:
    epigraph_root: Path
    source_run_id: str
    baseline_source_run_id: str | None = None


def _shock_context(context: MonthlyJointContext) -> MonthlyShockContext:
    return MonthlyShockContext(
        epigraph_root=Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )


def _month_to_quarter(month_index: int) -> str:
    year = int(month_index) // 12
    month = int(month_index) % 12 + 1
    quarter = ((month - 1) // 3) + 1
    return f"{year:04d}-Q{quarter}"


def _quarter_from_ordinal(index: int) -> str:
    year = int(index) // 4
    quarter = int(index) % 4 + 1
    return f"{year:04d}-Q{quarter}"


def _positive_autocorrelation(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    previous = array[:-1]
    current = array[1:]
    if float(np.std(previous)) <= 0.0 or float(np.std(current)) <= 0.0:
        return 0.0
    correlation = float(np.corrcoef(previous, current)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, 0.0, 1.0))


def _robust_standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    scale = float(np.median(np.abs(array - center)))
    if scale <= 0.0:
        scale = float(np.std(array))
    if scale <= 0.0:
        scale = max(float(np.max(np.abs(array - center))), 1.0)
    return [float((value - center) / scale) for value in array]


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log1p(max(float(numerator), 0.0)) - np.log1p(max(float(denominator), 0.0)))


def _monthly_metric_values(rows: list[dict[str, Any]], train_end_month: int) -> dict[int, dict[str, float]]:
    chosen = _chosen_month_metric_values([row for row in rows if int(row["_month_index"]) <= int(train_end_month)])
    values: dict[int, dict[str, float]] = {}
    for (month, metric_name), row in sorted(chosen.items()):
        if metric_name not in JOINT_MONTHLY_METRICS:
            continue
        values.setdefault(int(month), {})[metric_name] = float(row.get("value") or 0.0)
    return values


def _quarter_months(monthly_values: dict[int, dict[str, float]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for month in sorted(monthly_values):
        grouped.setdefault(_month_to_quarter(month), []).append(int(month))
    return grouped


def _quarterly_monthly_summary(months: list[int], monthly_values: dict[int, dict[str, float]]) -> dict[str, float]:
    ordered = sorted(months)
    diagnosis_sum = 0.0
    deaths_sum = 0.0
    diagnosed_snapshots: list[float] = []
    art_snapshots: list[float] = []
    cd4_values: list[float] = []
    for month in ordered:
        values = monthly_values.get(month) or {}
        diagnosis_sum += float(values.get("new_diagnosed_cases_period") or values.get("new_diagnosed_cases_monthly") or 0.0)
        deaths_sum += float(values.get("deaths_reported_period") or 0.0)
        if values.get("diagnosed_plhiv") is not None:
            diagnosed_snapshots.append(float(values["diagnosed_plhiv"]))
        if values.get("alive_on_art") is not None:
            art_snapshots.append(float(values["alive_on_art"]))
        if values.get("median_cd4_at_enrollment") is not None:
            cd4_values.append(float(values["median_cd4_at_enrollment"]))
    return {
        "monthly_diagnosis_sum": float(diagnosis_sum),
        "monthly_deaths_sum": float(deaths_sum),
        "monthly_diagnosed_last": float(diagnosed_snapshots[-1]) if diagnosed_snapshots else 0.0,
        "monthly_art_last": float(art_snapshots[-1]) if art_snapshots else 0.0,
        "monthly_cd4_mean": float(np.mean(np.asarray(cd4_values, dtype=np.float64))) if cd4_values else 0.0,
        "month_support_count": float(len(ordered)),
    }


def _train_quarter_rows(dataset: Any) -> dict[str, dict[str, Any]]:
    rows = {str(row.get("quarter") or ""): dict(row) for row in list(getattr(dataset, "train_rows", []) or [])}
    transitions = {str(row.get("quarter") or ""): dict(row) for row in list(getattr(dataset, "train_transition_rows", []) or [])}
    result: dict[str, dict[str, Any]] = {}
    for quarter, row in rows.items():
        transition = transitions.get(quarter) or {}
        result[quarter] = {
            "diagnosed_plhiv": row.get("diagnosed_plhiv"),
            "alive_on_art": row.get("alive_on_art"),
            "new_diagnosed_cases_period": row.get("new_diagnosed_cases_period"),
            "deaths_reported_period": row.get("deaths_reported_period"),
            "flow_U_to_D": (transition.get("flows") or {}).get("U_to_D"),
            "attrition_outflow": (transition.get("stock_balance") or {}).get("attrition_outflow"),
        }
    return result


def _quarter_alignment_features(
    *,
    dataset: Any,
    monthly_values: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    grouped = _quarter_months(monthly_values)
    train_quarters = _train_quarter_rows(dataset)
    raw_rows: dict[str, dict[str, float]] = {}
    cd4_values = [
        float(values["median_cd4_at_enrollment"])
        for values in monthly_values.values()
        if values.get("median_cd4_at_enrollment") is not None and float(values.get("median_cd4_at_enrollment") or 0.0) > 0.0
    ]
    cd4_center = float(np.median(np.asarray(cd4_values, dtype=np.float64))) if cd4_values else 0.0
    for quarter, months in grouped.items():
        summary = _quarterly_monthly_summary(months, monthly_values)
        train_row = train_quarters.get(quarter) or {}
        quarterly_flow = train_row.get("flow_U_to_D")
        if quarterly_flow is None:
            quarterly_flow = train_row.get("new_diagnosed_cases_period")
        quarterly_flow_value = max(float(quarterly_flow or 0.0), 0.0)
        diagnosed_value = max(float(train_row.get("diagnosed_plhiv") or 0.0), 0.0)
        art_value = max(float(train_row.get("alive_on_art") or 0.0), 0.0)
        attrition_value = max(float(train_row.get("attrition_outflow") or train_row.get("deaths_reported_period") or 0.0), 0.0)
        monthly_diagnosis = float(summary["monthly_diagnosis_sum"])
        monthly_diagnosed = float(summary["monthly_diagnosed_last"])
        monthly_art = float(summary["monthly_art_last"])
        monthly_deaths = float(summary["monthly_deaths_sum"])
        monthly_cd4 = float(summary["monthly_cd4_mean"])
        diagnosis_gap = _safe_log_ratio(monthly_diagnosis, quarterly_flow_value)
        diagnosis_backlog_gap = _safe_log_ratio(max(quarterly_flow_value - monthly_diagnosis, 0.0), quarterly_flow_value)
        diagnosed_stock_gap = _safe_log_ratio(monthly_diagnosed, diagnosed_value)
        art_stock_gap = _safe_log_ratio(monthly_art, art_value)
        death_pressure_gap = _safe_log_ratio(monthly_deaths, attrition_value)
        cd4_backlog_pressure = 0.0
        if monthly_cd4 > 0.0 and cd4_center > 0.0:
            cd4_backlog_pressure = _safe_log_ratio(cd4_center, monthly_cd4)
        raw_rows[quarter] = {
            "diagnosis_reporting_gap": diagnosis_gap,
            "diagnosis_backlog_gap": diagnosis_backlog_gap,
            "diagnosed_stock_gap": diagnosed_stock_gap,
            "art_service_gap": art_stock_gap,
            "death_service_pressure": death_pressure_gap,
            "cd4_backlog_pressure": cd4_backlog_pressure,
            "monthly_quarter_support": float(summary["month_support_count"]) / 3.0,
        }
    columns = [
        "diagnosis_reporting_gap",
        "diagnosis_backlog_gap",
        "diagnosed_stock_gap",
        "art_service_gap",
        "death_service_pressure",
        "cd4_backlog_pressure",
    ]
    standardized_columns = {
        column: _robust_standardize([float(raw_rows[quarter].get(column) or 0.0) for quarter in sorted(raw_rows, key=quarter_sort_key)])
        for column in columns
    }
    standardized: dict[str, dict[str, float]] = {}
    ordered_quarters = sorted(raw_rows, key=quarter_sort_key)
    for index, quarter in enumerate(ordered_quarters):
        row = dict(raw_rows[quarter])
        for column in columns:
            row[column] = float(standardized_columns[column][index])
        standardized[quarter] = row
    return standardized


def _feature_for_quarter(
    quarter_features: dict[str, dict[str, float]],
    quarter: str,
    *,
    train_end_quarter: str,
    phis: dict[str, float],
    feature_names: list[str],
) -> dict[str, float]:
    if quarter in quarter_features and quarter_sort_key(quarter) <= quarter_sort_key(train_end_quarter):
        return dict(quarter_features[quarter])
    if not quarter_features:
        return {name: 0.0 for name in feature_names}
    valid = [item for item in quarter_features if quarter_sort_key(item) <= quarter_sort_key(train_end_quarter)]
    if not valid:
        return {name: 0.0 for name in feature_names}
    last_quarter = max(valid, key=quarter_sort_key)
    last_index = quarter_sort_key(last_quarter)[0] * 4 + quarter_sort_key(last_quarter)[1]
    target_index = quarter_sort_key(quarter)[0] * 4 + quarter_sort_key(quarter)[1]
    step = max(int(target_index - last_index), 0)
    last = dict(quarter_features[last_quarter])
    return {
        name: float(last.get(name) or 0.0) * (float(phis.get(name) or 0.0) ** step)
        for name in feature_names
    }


def build_quarterly_joint_observation_features(
    context: MonthlyJointContext,
    *,
    dataset: Any,
    train_end_quarter: str,
    quarters: list[str],
) -> dict[str, Any]:
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    rows = load_monthly_signal_rows(_shock_context(context))
    monthly_values = _monthly_metric_values(rows, train_end_month)
    aligned = _quarter_alignment_features(dataset=dataset, monthly_values=monthly_values)
    feature_names = [
        "reporting_intensity",
        "backlog_pressure",
        "service_capacity",
        "service_pressure",
        "reporting_shock",
        "service_shock",
        "lagged_reporting_intensity",
        "joint_observation_support",
    ]
    base_by_quarter: dict[str, dict[str, float]] = {}
    reporting_raw = []
    service_raw = []
    ordered_train_quarters = sorted(aligned, key=quarter_sort_key)
    for quarter in ordered_train_quarters:
        row = dict(aligned[quarter])
        reporting = float(row.get("diagnosis_reporting_gap") or 0.0) + float(row.get("diagnosed_stock_gap") or 0.0)
        backlog = float(row.get("diagnosis_backlog_gap") or 0.0) + float(row.get("cd4_backlog_pressure") or 0.0)
        service_capacity = float(row.get("art_service_gap") or 0.0)
        service_pressure = float(row.get("death_service_pressure") or 0.0) - service_capacity
        reporting_raw.append(reporting)
        service_raw.append(service_pressure)
        base_by_quarter[quarter] = {
            "reporting_intensity": float(np.tanh(reporting)),
            "backlog_pressure": float(np.tanh(backlog)),
            "service_capacity": float(np.tanh(service_capacity)),
            "service_pressure": float(np.tanh(service_pressure)),
            "joint_observation_support": float(row.get("monthly_quarter_support") or 0.0),
        }
    reporting_threshold = _tukey_upper_fence([abs(value) for value in reporting_raw])
    service_threshold = _tukey_upper_fence([abs(value) for value in service_raw])
    for quarter, reporting, service in zip(ordered_train_quarters, reporting_raw, service_raw):
        base_by_quarter[quarter]["reporting_shock"] = 1.0 if reporting_threshold > 0.0 and abs(reporting) > reporting_threshold else 0.0
        base_by_quarter[quarter]["service_shock"] = 1.0 if service_threshold > 0.0 and abs(service) > service_threshold else 0.0
    reporting_values = [float((base_by_quarter.get(quarter) or {}).get("reporting_intensity") or 0.0) for quarter in ordered_train_quarters]
    service_values = [float((base_by_quarter.get(quarter) or {}).get("service_pressure") or 0.0) for quarter in ordered_train_quarters]
    phis = {
        "reporting_intensity": _positive_autocorrelation(reporting_values),
        "backlog_pressure": _positive_autocorrelation([float((base_by_quarter.get(quarter) or {}).get("backlog_pressure") or 0.0) for quarter in ordered_train_quarters]),
        "service_capacity": _positive_autocorrelation([float((base_by_quarter.get(quarter) or {}).get("service_capacity") or 0.0) for quarter in ordered_train_quarters]),
        "service_pressure": _positive_autocorrelation(service_values),
        "reporting_shock": _positive_autocorrelation([float((base_by_quarter.get(quarter) or {}).get("reporting_shock") or 0.0) for quarter in ordered_train_quarters]),
        "service_shock": _positive_autocorrelation([float((base_by_quarter.get(quarter) or {}).get("service_shock") or 0.0) for quarter in ordered_train_quarters]),
        "lagged_reporting_intensity": _positive_autocorrelation(reporting_values),
        "joint_observation_support": _positive_autocorrelation([float((base_by_quarter.get(quarter) or {}).get("joint_observation_support") or 0.0) for quarter in ordered_train_quarters]),
    }
    quarter_features: dict[str, dict[str, float]] = {}
    for quarter in quarters:
        features = _feature_for_quarter(
            base_by_quarter,
            quarter,
            train_end_quarter=train_end_quarter,
            phis=phis,
            feature_names=feature_names,
        )
        previous_quarter = _quarter_from_ordinal(quarter_ordinal(quarter) - 1)
        lagged = _feature_for_quarter(
            base_by_quarter,
            previous_quarter,
            train_end_quarter=train_end_quarter,
            phis=phis,
            feature_names=feature_names,
        )
        features["lagged_reporting_intensity"] = float(lagged.get("reporting_intensity") or 0.0)
        quarter_features[quarter] = {name: float(features.get(name) or 0.0) for name in feature_names}
    return {
        "schema_version": MONTHLY_JOINT_SCHEMA_VERSION,
        "feature_names": feature_names,
        "quarter_features": quarter_features,
        "train_end_quarter": train_end_quarter,
        "train_end_month": _month_label(train_end_month),
        "monthly_row_count": len(rows),
        "aligned_train_quarter_count": len(aligned),
        "feature_autocorrelation": phis,
        "contract": (
            "Monthly observations are jointly aligned with quarterly cascade stocks/flows before "
            "hazard fitting; holdout quarters use AR-decayed train-origin observation states."
        ),
    }


def apply_monthly_joint_observation_to_paths(
    *,
    context: MonthlyJointContext,
    dataset: Any,
    hazard_paths: dict[str, Any],
    incidence_paths: dict[str, Any],
) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_rows or not holdout_rows:
        return {"hazard_paths": hazard_paths, "incidence_paths": incidence_paths, "diagnostics": {"status": "not_evaluable"}}
    train_quarters = [str(row.get("quarter") or "") for row in train_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    features_payload = build_quarterly_joint_observation_features(
        context,
        dataset=dataset,
        train_end_quarter=train_quarters[-1],
        quarters=train_quarters + holdout_quarters,
    )
    features = dict(features_payload.get("quarter_features") or {})
    feature_names = list(features_payload.get("feature_names") or [])
    adjusted_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("holdout_hazard_map") or {}).items()
    }
    adjusted_train_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("train_hazard_map") or {}).items()
    }
    diagnostics: dict[str, Any] = {
        "monthly_joint_features": features_payload,
        "targets": {},
        "contract": "Joint monthly observation states alter mechanistic hazards before quarterly state simulation and endpoint readout.",
    }
    eps = float(getattr(dataset, "eps", np.finfo(np.float32).eps))
    for transition in ("U_to_D", "D_to_A", "A_to_T", "T_to_V", "A_to_L", "T_to_L", "V_to_L", "L_to_R"):
        residuals = {}
        for row in train_rows:
            quarter = str(row.get("quarter") or "")
            observed = float((row.get("hazards") or {}).get(transition) or 0.0)
            base = float((adjusted_train_hazards.get(quarter) or {}).get(transition) or 0.0)
            residuals[quarter] = logit(observed, eps=eps) - logit(base, eps=eps)
        fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
        diagnostics["targets"][transition] = fit
        for target_map in (adjusted_train_hazards, adjusted_hazards):
            for quarter, values in target_map.items():
                base = float(values.get(transition) or 0.0)
                effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
                values[transition] = float(inv_logit(logit(base, eps=eps) + effect))
    adjusted_incidence = dict(incidence_paths)
    train_incidence_hazard = {quarter: float(value) for quarter, value in dict(incidence_paths.get("train_incidence_hazard_map") or {}).items()}
    holdout_incidence_hazard = {quarter: float(value) for quarter, value in dict(incidence_paths.get("holdout_incidence_hazard_map") or {}).items()}
    residuals = {}
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        observed = float((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff") or 0.0)
        base = float(train_incidence_hazard.get(quarter) or 0.0)
        residuals[quarter] = float(np.log1p(max(observed, 0.0)) - np.log1p(max(base, 0.0)))
    fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
    diagnostics["targets"]["incidence_hazard_per_s_eff"] = fit
    for target_map in (train_incidence_hazard, holdout_incidence_hazard):
        for quarter, base in list(target_map.items()):
            effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
            target_map[quarter] = float(max(np.expm1(np.log1p(max(float(base), 0.0)) + effect), 0.0))
    adjusted_incidence["train_incidence_hazard_map"] = train_incidence_hazard
    adjusted_incidence["holdout_incidence_hazard_map"] = holdout_incidence_hazard
    adjusted_paths = dict(hazard_paths)
    adjusted_paths["train_hazard_map"] = adjusted_train_hazards
    adjusted_paths["holdout_hazard_map"] = adjusted_hazards
    return {
        "hazard_paths": adjusted_paths,
        "incidence_paths": adjusted_incidence,
        "diagnostics": diagnostics,
    }
