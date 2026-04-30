from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import inv_logit, logit, quarter_sort_key
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


MONTHLY_LATENT_SCHEMA_VERSION = "phase3_dynamic_monthly_latent_observation_state.v1"
REPORTING_STREAM_METRICS: tuple[str, ...] = (
    "new_diagnosed_cases_period",
    "new_diagnosed_cases_monthly",
    "diagnosed_plhiv",
    "median_cd4_at_enrollment",
)
SERVICE_STREAM_METRICS: tuple[str, ...] = (
    "alive_on_art",
    "newly_enrolled_to_treatment",
    "deaths_reported_period",
)


@dataclass(frozen=True, slots=True)
class MonthlyLatentContext:
    epigraph_root: Path
    source_run_id: str
    baseline_source_run_id: str | None = None


def _shock_context(context: MonthlyLatentContext) -> MonthlyShockContext:
    return MonthlyShockContext(
        epigraph_root=Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )


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


def _fit_metric_residuals(items: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(items)
    if len(ordered) < 2:
        return {int(month): 0.0 for month, _value in ordered}
    months = np.asarray([float(month - ordered[0][0]) for month, _value in ordered], dtype=np.float64)
    values = np.asarray([float(value) for _month, value in ordered], dtype=np.float64)
    residuals: dict[int, float] = {int(ordered[0][0]): 0.0}
    if len(ordered) < 3:
        deltas = np.diff(values)
        standardized = _robust_standardize([float(value) for value in deltas])
        for (month, _value), residual in zip(ordered[1:], standardized):
            residuals[int(month)] = float(residual)
        return residuals
    y = values[1:]
    x = np.column_stack([np.ones(len(y), dtype=np.float64), months[1:], values[:-1]])
    beta = np.linalg.pinv(x) @ y
    fitted = x @ beta
    raw_residuals = [float(value) for value in (y - fitted)]
    standardized = _robust_standardize(raw_residuals)
    for (month, _value), residual in zip(ordered[1:], standardized):
        residuals[int(month)] = float(residual)
    return residuals


def _stream_metric_series(
    rows: list[dict[str, Any]],
    *,
    metric_names: tuple[str, ...],
    train_end_month: int,
) -> dict[str, list[tuple[int, float]]]:
    chosen = _chosen_month_metric_values([row for row in rows if int(row["_month_index"]) <= int(train_end_month)])
    series: dict[str, list[tuple[int, float]]] = {name: [] for name in metric_names}
    for (month, metric_name), row in sorted(chosen.items()):
        if metric_name not in series:
            continue
        value = float(np.log1p(max(float(row.get("value") or 0.0), 0.0)))
        series[metric_name].append((int(month), value))
    return {name: values for name, values in series.items() if values}


def _build_stream_process(
    rows: list[dict[str, Any]],
    *,
    stream_name: str,
    metric_names: tuple[str, ...],
    anchor_metric: str,
    train_end_month: int,
) -> dict[str, Any]:
    series = _stream_metric_series(rows, metric_names=metric_names, train_end_month=train_end_month)
    residual_by_metric = {
        metric_name: _fit_metric_residuals(values)
        for metric_name, values in series.items()
        if len(values) >= 2
    }
    months = sorted({month for values in residual_by_metric.values() for month in values})
    if not months or not residual_by_metric:
        return {
            "stream_name": stream_name,
            "metric_names": list(metric_names),
            "active_metric_names": [],
            "anchor_metric": anchor_metric,
            "train_end_month": _month_label(train_end_month),
            "feature_autocorrelation": 0.0,
            "month_table": {},
            "contract": "no_monthly_stream_support_available_before_forecast_origin",
        }
    active_metrics = sorted(residual_by_metric)
    matrix = np.zeros((len(months), len(active_metrics)), dtype=np.float64)
    support = np.zeros_like(matrix)
    for column, metric_name in enumerate(active_metrics):
        metric_values = residual_by_metric[metric_name]
        for row_index, month in enumerate(months):
            if month not in metric_values:
                continue
            matrix[row_index, column] = float(metric_values[month])
            support[row_index, column] = 1.0
    if matrix.shape[1] == 1:
        component = matrix[:, 0]
    else:
        u, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
        component = u[:, 0] * singular_values[0]
        if anchor_metric in active_metrics:
            anchor = matrix[:, active_metrics.index(anchor_metric)]
            if float(np.dot(component, anchor)) < 0.0:
                component = -component
    standardized_component = np.asarray(_robust_standardize([float(value) for value in component]), dtype=np.float64)
    bounded = np.tanh(standardized_component)
    shock_threshold = _tukey_upper_fence([abs(float(value)) for value in standardized_component])
    month_table: dict[int, dict[str, float]] = {}
    denominator = max(float(len(active_metrics)), 1.0)
    for month, raw_value, bounded_value, row_support in zip(months, standardized_component, bounded, support):
        month_table[int(month)] = {
            f"{stream_name}_latent_raw": float(raw_value),
            f"{stream_name}_intensity": float(bounded_value),
            f"{stream_name}_shock": 1.0 if shock_threshold > 0.0 and abs(float(raw_value)) > shock_threshold else 0.0,
            f"{stream_name}_support": float(np.sum(row_support)) / denominator,
        }
    return {
        "stream_name": stream_name,
        "metric_names": list(metric_names),
        "active_metric_names": active_metrics,
        "anchor_metric": anchor_metric,
        "train_end_month": _month_label(train_end_month),
        "feature_autocorrelation": _positive_autocorrelation([float(value) for value in bounded]),
        "month_table": month_table,
        "contract": (
            "Monthly latent observation state is the first singular component of train-origin "
            "per-metric one-step residuals; sign is anchored to the stream anchor when available."
        ),
    }


def _stream_feature_for_month(process: dict[str, Any], month: int) -> dict[str, float]:
    stream_name = str(process.get("stream_name") or "")
    table = {int(key): dict(value) for key, value in dict(process.get("month_table") or {}).items()}
    empty = {
        f"{stream_name}_latent_raw": 0.0,
        f"{stream_name}_intensity": 0.0,
        f"{stream_name}_shock": 0.0,
        f"{stream_name}_support": 0.0,
    }
    if not table:
        return empty
    train_end_label = str(process.get("train_end_month") or "")
    train_end_month = max(table) if not train_end_label else int(train_end_label[:4]) * 12 + int(train_end_label[5:7]) - 1
    if int(month) in table and int(month) <= train_end_month:
        return dict(table[int(month)])
    valid_months = [value for value in table if value <= train_end_month]
    if not valid_months:
        return empty
    last_month = max(valid_months)
    last = dict(table[last_month])
    phi = float(process.get("feature_autocorrelation") or 0.0)
    step = max(int(month) - int(last_month), 0)
    decay = phi ** step
    return {
        f"{stream_name}_latent_raw": float(last.get(f"{stream_name}_latent_raw") or 0.0) * decay,
        f"{stream_name}_intensity": float(last.get(f"{stream_name}_intensity") or 0.0) * decay,
        f"{stream_name}_shock": float(last.get(f"{stream_name}_shock") or 0.0) * decay,
        f"{stream_name}_support": float(last.get(f"{stream_name}_support") or 0.0) * decay,
    }


def build_quarterly_monthly_latent_features(
    context: MonthlyLatentContext,
    *,
    train_end_quarter: str,
    quarters: list[str],
) -> dict[str, Any]:
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    rows = load_monthly_signal_rows(_shock_context(context))
    reporting = _build_stream_process(
        rows,
        stream_name="reporting",
        metric_names=REPORTING_STREAM_METRICS,
        anchor_metric="new_diagnosed_cases_period",
        train_end_month=train_end_month,
    )
    service = _build_stream_process(
        rows,
        stream_name="service",
        metric_names=SERVICE_STREAM_METRICS,
        anchor_metric="alive_on_art",
        train_end_month=train_end_month,
    )
    feature_names = [
        "reporting_intensity",
        "service_intensity",
        "reporting_shock",
        "service_shock",
        "reporting_service_interaction",
        "lagged_reporting_intensity",
        "reporting_support",
        "service_support",
    ]
    quarter_features: dict[str, dict[str, float]] = {}
    for quarter in quarters:
        start_month, end_month = _quarter_month_range(quarter)
        reporting_rows = [_stream_feature_for_month(reporting, month) for month in range(start_month, end_month + 1)]
        service_rows = [_stream_feature_for_month(service, month) for month in range(start_month, end_month + 1)]
        lagged_reporting_rows = [_stream_feature_for_month(reporting, month - 3) for month in range(start_month, end_month + 1)]
        reporting_intensity = float(np.mean([row["reporting_intensity"] for row in reporting_rows]))
        service_intensity = float(np.mean([row["service_intensity"] for row in service_rows]))
        quarter_features[quarter] = {
            "reporting_intensity": reporting_intensity,
            "service_intensity": service_intensity,
            "reporting_shock": float(np.mean([row["reporting_shock"] for row in reporting_rows])),
            "service_shock": float(np.mean([row["service_shock"] for row in service_rows])),
            "reporting_service_interaction": float(reporting_intensity * service_intensity),
            "lagged_reporting_intensity": float(np.mean([row["reporting_intensity"] for row in lagged_reporting_rows])),
            "reporting_support": float(np.mean([row["reporting_support"] for row in reporting_rows])),
            "service_support": float(np.mean([row["service_support"] for row in service_rows])),
        }
    return {
        "schema_version": MONTHLY_LATENT_SCHEMA_VERSION,
        "feature_names": feature_names,
        "quarter_features": quarter_features,
        "monthly_row_count": len(rows),
        "train_end_quarter": train_end_quarter,
        "train_end_month": _month_label(train_end_month),
        "streams": {
            "reporting": {key: value for key, value in reporting.items() if key != "month_table"},
            "service": {key: value for key, value in service.items() if key != "month_table"},
        },
        "stream_month_tables": {
            "reporting": dict(reporting.get("month_table") or {}),
            "service": dict(service.get("month_table") or {}),
        },
        "contract": (
            "Quarterly features are aggregated from latent monthly reporting and service states "
            "estimated only through the forecast-origin month; holdout months use AR-decayed "
            "last-observed latent states."
        ),
    }


def apply_monthly_latent_to_paths(
    *,
    context: MonthlyLatentContext,
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
    all_quarters = train_quarters + holdout_quarters
    features_payload = build_quarterly_monthly_latent_features(
        context,
        train_end_quarter=train_quarters[-1],
        quarters=all_quarters,
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
        "monthly_latent_features": features_payload,
        "targets": {},
        "contract": "Latent monthly observation states alter hazards before quarterly state simulation and before endpoint readout.",
    }
    eps = float(getattr(dataset, "eps", np.finfo(np.float32).eps))
    residuals = {}
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        observed = float((row.get("hazards") or {}).get("U_to_D") or 0.0)
        base = float((adjusted_train_hazards.get(quarter) or {}).get("U_to_D") or 0.0)
        residuals[quarter] = logit(observed, eps=eps) - logit(base, eps=eps)
    fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
    diagnostics["targets"]["U_to_D"] = fit
    for target_map in (adjusted_train_hazards, adjusted_hazards):
        for quarter, values in target_map.items():
            base = float(values.get("U_to_D") or 0.0)
            effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
            values["U_to_D"] = float(inv_logit(logit(base, eps=eps) + effect))
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
