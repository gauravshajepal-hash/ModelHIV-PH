from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backhalf_channels import BackHalfChannelContext, apply_backhalf_channel_adjustment_to_paths
from .data import (
    ART_STATE_NAMES,
    DIAGNOSED_STATE_NAMES,
    STATE_NAMES,
    VL_TESTED_STATE_NAMES,
    BlockedTimeDataset,
    build_blocked_time_dataset,
    build_observation_rows,
    default_epigraph_root,
    sandbox_repo_root,
    state_sum,
)
from .incidence import carry_forward_incidence_flow_paths
from .decomposition import apply_decomposition_controls
from .metrics import PRIMARY_METRICS, inv_logit, logit, normalized_mae, quarter_sort_key, quarter_year
from .model import (
    _apply_observation_model,
    _quarter_positions,
    _simulate_sequence,
    carry_forward_hazards,
    fit_dynamic_hazard_paths,
    fit_observation_model,
    simulate_holdout,
)
from .incidence import fit_incidence_flow_paths
from .monthly_joint_observation import MonthlyJointContext, apply_monthly_joint_observation_to_paths
from .monthly_latent_state import MonthlyLatentContext, apply_monthly_latent_to_paths
from .monthly_shock import MonthlyShockContext, apply_monthly_shock_to_paths
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .runtime import ensure_dir, read_json, write_json
from .scenario_lab import (
    DEFAULT_ACTIVE_SOURCE_RUN_ID,
    DEFAULT_BASELINE_SOURCE_RUN_ID,
    _apply_projection_constraints,
    _build_projection_constraints,
    _filter_rows_before_holdout,
    _forecast_constrained_reference,
    _future_quarters,
    _load_reference_config,
    _projection_dataset,
    _scale_state_attrition_to_aggregate,
)


HYBRID_CHAMPION_SCHEMA_VERSION = "phase3_dynamic_hybrid_champion_search.v1"
SHOCK_TRAJECTORY_GATE_SCHEMA_VERSION = "phase3_dynamic_shock_aware_lifted_trajectory_gate.v1"
OBSERVATION_HEAD_METRICS: tuple[str, ...] = PRIMARY_METRICS + ("virally_suppressed",)
FLOW_LIKE_METRICS = {"new_diagnosed_cases_period"}
KNOWN_EXTERNAL_SHOCK_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (2020, 2021, "known_external_shock:covid_service_disruption"),
)
HEAD_FAMILIES: tuple[str, ...] = (
    "mechanistic_bounded_only",
    "monthly_joint_mechanistic_bounded_only",
    "monthly_latent_mechanistic_bounded_only",
    "monthly_shock_mechanistic_bounded_only",
    "endpoint_raw_affine",
    "endpoint_raw_affine_train_blend",
    "endpoint_raw_plus_lag",
    "endpoint_raw_plus_lag_train_blend",
    "endpoint_r10_like_delta",
    "endpoint_r10_like_delta_train_blend",
    "state_multihorizon_observation",
    "state_multihorizon_observation_train_blend",
    "state_multihorizon_support_decay",
    "state_multihorizon_support_decay_train_blend",
    "state_multihorizon_conditional_rates",
    "state_multihorizon_shock_nowcast",
    "state_multihorizon_shock_nowcast_train_blend",
    "lifted_residual_state",
    "conditional_horizon_gated_endpoint_state",
    "horizon_gated_endpoint_state",
    "monthly_joint_endpoint_raw_affine",
    "monthly_joint_state_multihorizon_support_decay",
    "monthly_joint_state_multihorizon_conditional_rates",
    "monthly_joint_lifted_residual_state",
    "monthly_joint_conditional_horizon_gated_endpoint_state",
    "monthly_joint_horizon_gated_endpoint_state",
    "backhalf_channel_mechanistic_bounded_only",
    "backhalf_channel_endpoint_raw_affine",
    "backhalf_channel_state_multihorizon_support_decay",
    "backhalf_channel_horizon_gated_endpoint_state",
    "monthly_latent_endpoint_raw_affine",
    "monthly_latent_state_multihorizon_support_decay",
    "monthly_latent_horizon_gated_endpoint_state",
    "monthly_shock_endpoint_raw_affine",
    "monthly_shock_state_multihorizon_support_decay",
    "monthly_shock_horizon_gated_endpoint_state",
)


@dataclass(slots=True)
class EndpointHead:
    family: str
    metric_coefficients: dict[str, list[float]]
    train_row_count: dict[str, int]
    endpoint_weight: float = 1.0
    feature_centers: dict[str, list[float]] | None = None
    feature_scales: dict[str, list[float]] | None = None
    training_horizons: dict[str, int] | None = None
    horizon_step_weights: list[float] | None = None
    correction_bounds: dict[str, float] | None = None
    training_objective: str = "one_step_endpoint_residual"


def _base_head_family(family: str) -> str:
    text = str(family)
    return (
        text.removeprefix("backhalf_channel_")
        .removeprefix("monthly_joint_")
        .removeprefix("monthly_latent_")
        .removeprefix("monthly_shock_")
        .removesuffix("_train_blend")
    )


def _uses_monthly_shock_hazard(family: str) -> bool:
    return str(family).startswith("monthly_shock_")


def _uses_monthly_latent_hazard(family: str) -> bool:
    return str(family).startswith("monthly_latent_")


def _uses_monthly_joint_hazard(family: str) -> bool:
    return str(family).startswith("monthly_joint_")


def _uses_backhalf_channel_hazard(family: str) -> bool:
    return str(family).startswith("backhalf_channel_")


def _is_multihorizon_family(family: str) -> bool:
    base_family = _base_head_family(family)
    return base_family.startswith("state_multihorizon_") or base_family == "lifted_residual_state"


def _is_lifted_residual_family(family: str) -> bool:
    return _base_head_family(family) == "lifted_residual_state"


def _is_conditional_rate_family(family: str) -> bool:
    base_family = _base_head_family(family)
    return base_family == "state_multihorizon_conditional_rates"


def _is_shock_nowcast_family(family: str) -> bool:
    return "shock_nowcast" in _base_head_family(family)


def _delegate_family_for_horizon(family: str, *, holdout_year_count: int) -> str:
    base_family = _base_head_family(family)
    if base_family not in {"horizon_gated_endpoint_state", "conditional_horizon_gated_endpoint_state"}:
        return family
    if _uses_backhalf_channel_hazard(family):
        prefix = "backhalf_channel_"
    elif _uses_monthly_joint_hazard(family):
        prefix = "monthly_joint_"
    elif _uses_monthly_latent_hazard(family):
        prefix = "monthly_latent_"
    elif _uses_monthly_shock_hazard(family):
        prefix = "monthly_shock_"
    else:
        prefix = ""
    if int(holdout_year_count) <= 1:
        return f"{prefix}endpoint_raw_affine"
    if base_family == "conditional_horizon_gated_endpoint_state":
        return f"{prefix}state_multihorizon_conditional_rates"
    return f"{prefix}state_multihorizon_support_decay"


def _rolling_splits(
    observation_rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> list[dict[str, Any]]:
    available_years = sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in observation_rows
            if str(row.get("quarter") or "")
        }
    )
    bounded = [year for year in available_years if int(start_year) <= int(year) <= int(end_year)]
    splits: list[dict[str, Any]] = []
    for train_end_year in bounded:
        train_years = [year for year in bounded if year <= train_end_year]
        holdout_years = [year for year in range(int(train_end_year) + 1, int(train_end_year) + int(horizon_years) + 1)]
        if len(train_years) < int(min_train_years):
            continue
        if not all(year in bounded for year in holdout_years):
            continue
        splits.append(
            {
                "train_end_year": int(train_end_year),
                "train_years": train_years,
                "holdout_years": holdout_years,
            }
        )
    return splits


def _forecast_reference_for_family(
    *,
    family: str,
    dataset: BlockedTimeDataset,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
) -> dict[str, Any]:
    if (
        not _uses_monthly_shock_hazard(family)
        and not _uses_monthly_latent_hazard(family)
        and not _uses_monthly_joint_hazard(family)
        and not _uses_backhalf_channel_hazard(family)
    ):
        return _forecast_constrained_reference(
            dataset=dataset,
            reference_config=reference_config,
            constraint_rows=constraint_rows,
        )
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
    monthly_adjustment = {"diagnostics": {"status": "not_configured"}}
    monthly_latent_adjustment = {"diagnostics": {"status": "not_configured"}}
    monthly_joint_adjustment = {"diagnostics": {"status": "not_configured"}}
    backhalf_channel_adjustment = {"diagnostics": {"status": "not_configured"}}
    if _uses_backhalf_channel_hazard(family) and backhalf_context is not None:
        backhalf_channel_adjustment = apply_backhalf_channel_adjustment_to_paths(
            context=backhalf_context,
            dataset=dataset,
            hazard_paths=hazard_paths,
            incidence_paths=incidence_paths,
        )
        hazard_paths = dict(backhalf_channel_adjustment.get("hazard_paths") or hazard_paths)
    elif _uses_monthly_joint_hazard(family) and monthly_joint_context is not None:
        monthly_joint_adjustment = apply_monthly_joint_observation_to_paths(
            context=monthly_joint_context,
            dataset=dataset,
            hazard_paths=hazard_paths,
            incidence_paths=incidence_paths,
        )
        hazard_paths = dict(monthly_joint_adjustment.get("hazard_paths") or hazard_paths)
        incidence_paths = dict(monthly_joint_adjustment.get("incidence_paths") or incidence_paths)
    elif _uses_monthly_latent_hazard(family) and monthly_latent_context is not None:
        monthly_latent_adjustment = apply_monthly_latent_to_paths(
            context=monthly_latent_context,
            dataset=dataset,
            hazard_paths=hazard_paths,
            incidence_paths=incidence_paths,
        )
        hazard_paths = dict(monthly_latent_adjustment.get("hazard_paths") or hazard_paths)
        incidence_paths = dict(monthly_latent_adjustment.get("incidence_paths") or incidence_paths)
    elif _uses_monthly_shock_hazard(family) and monthly_context is not None:
        monthly_adjustment = apply_monthly_shock_to_paths(
            context=monthly_context,
            dataset=dataset,
            hazard_paths=hazard_paths,
            incidence_paths=incidence_paths,
        )
        hazard_paths = dict(monthly_adjustment.get("hazard_paths") or hazard_paths)
        incidence_paths = dict(monthly_adjustment.get("incidence_paths") or incidence_paths)
    if reference_config.get("decomposition_cfg") is not None:
        decomposition_adjustment = apply_decomposition_controls(
            dataset=dataset,
            paths=hazard_paths,
            incidence_paths=incidence_paths,
            cfg=reference_config["decomposition_cfg"],
        )
        hazard_paths = decomposition_adjustment["paths"]
        incidence_paths = decomposition_adjustment["incidence_paths"]
    constraints = _build_projection_constraints(
        dataset=dataset,
        constraint_rows=constraint_rows,
        future_quarters=holdout_quarters,
    )
    incidence_map, attrition_map, constraint_application = _apply_projection_constraints(
        incidence_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
        attrition_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
        constraints=constraints,
    )
    state_attrition_map = _scale_state_attrition_to_aggregate(
        dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
        attrition_map,
    )
    raw = _simulate_sequence(
        dict(dataset.train_state_rows[-1]["state_values"]),
        holdout_rows,
        dict(hazard_paths.get("holdout_hazard_map") or {}),
        incidence_inflow_map=incidence_map,
        incidence_hazard_map=dict(incidence_paths.get("holdout_incidence_hazard_map") or {}),
        incidence_cap_map={quarter: float(constraints.get("incidence_cap_per_quarter") or 0.0) for quarter in holdout_quarters},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=attrition_map,
        state_attrition_outflow_map=state_attrition_map,
    )
    observation_model = fit_observation_model(
        dataset,
        dict(hazard_paths.get("train_hazard_map") or {}),
        reference_config["observation_cfg"],
        incidence_paths=incidence_paths,
        damping_cfg=reference_config["damping_cfg"],
    )
    prediction_rows = _apply_observation_model(
        list(raw.get("prediction_rows") or []),
        observation_model,
        positions=_quarter_positions(dataset),
        damping_cfg=reference_config["damping_cfg"],
    )
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": list(raw.get("trajectory_rows") or []),
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
        "constraints": constraints,
        "constraint_application": constraint_application,
        "monthly_shock_adjustment": dict(monthly_adjustment.get("diagnostics") or {}),
        "monthly_latent_adjustment": dict(monthly_latent_adjustment.get("diagnostics") or {}),
        "monthly_joint_adjustment": dict(monthly_joint_adjustment.get("diagnostics") or {}),
        "backhalf_channel_adjustment": dict(backhalf_channel_adjustment.get("diagnostics") or {}),
    }


def _train_backbone_prediction_rows(
    *,
    family: str,
    dataset: BlockedTimeDataset,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
) -> list[dict[str, Any]]:
    train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if len(train_rows) <= 2:
        return []
    prediction_rows: list[dict[str, Any]] = []
    for index in range(2, len(train_rows) + 1):
        prefix_rows = train_rows[:index]
        holdout_year = quarter_year(str(prefix_rows[-1].get("quarter") or ""))
        train_prefix = [row for row in prefix_rows if row is not prefix_rows[-1]]
        if len(train_prefix) < 2:
            continue
        try:
            prefix_dataset = build_blocked_time_dataset(
                train_prefix + [prefix_rows[-1]],
                [holdout_year],
            )
        except ValueError:
            continue
        if not prefix_dataset.train_transition_rows or not prefix_dataset.holdout_rows:
            continue
        prefix_constraints = _filter_rows_before_holdout(constraint_rows, [holdout_year])
        forecast = _forecast_reference_for_family(
            family=family,
            dataset=prefix_dataset,
            reference_config=reference_config,
            constraint_rows=prefix_constraints,
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
        )
        prediction_rows.extend(list(forecast.get("prediction_rows") or []))
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in prediction_rows}
    return [by_quarter[str(row.get("quarter") or "")] for row in train_rows[1:] if str(row.get("quarter") or "") in by_quarter]


def _train_multihorizon_examples(
    *,
    family: str,
    train_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    min_train_years: int,
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    objective_horizon_years: tuple[int, ...] = (1, 5),
) -> list[dict[str, Any]]:
    sorted_train = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if len(sorted_train) < 3:
        return []
    available_years = sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in sorted_train
            if str(row.get("quarter") or "")
        }
    )
    if not available_years:
        return []
    examples: list[dict[str, Any]] = []
    for objective_years in objective_horizon_years:
        splits = _rolling_splits(
            sorted_train,
            start_year=min(available_years),
            end_year=max(available_years),
            min_train_years=min_train_years,
            horizon_years=int(objective_years),
        )
        for split in splits:
            try:
                split_dataset = build_blocked_time_dataset(sorted_train, list(split["holdout_years"]))
            except ValueError:
                continue
            if not split_dataset.train_transition_rows or not split_dataset.holdout_rows:
                continue
            split_constraints = _filter_rows_before_holdout(constraint_rows, list(split["holdout_years"]))
            forecast = _forecast_reference_for_family(
                family=family,
                dataset=split_dataset,
                reference_config=reference_config,
                constraint_rows=split_constraints,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
            )
            prediction_by_quarter = {
                str(row.get("quarter") or ""): dict(row)
                for row in list(forecast.get("prediction_rows") or [])
            }
            trajectory_by_quarter = {
                str(row.get("quarter") or ""): dict(row)
                for row in list(forecast.get("trajectory_rows") or [])
            }
            target_rows = sorted(split_dataset.holdout_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
            horizon_quarters = len(target_rows)
            for step_index, target_row in enumerate(target_rows):
                quarter = str(target_row.get("quarter") or "")
                backbone_row = prediction_by_quarter.get(quarter)
                trajectory_row = trajectory_by_quarter.get(quarter)
                if backbone_row is None or trajectory_row is None:
                    continue
                examples.append(
                    {
                        "train_end_year": int(split["train_end_year"]),
                        "objective_horizon_years": int(objective_years),
                        "horizon_quarters": int(horizon_quarters),
                        "step_index": int(step_index),
                        "target_row": dict(target_row),
                        "backbone_row": backbone_row,
                        "trajectory_row": trajectory_row,
                    }
                )
    return examples


def _trajectory_state_values(trajectory_row: dict[str, Any] | None) -> dict[str, float]:
    values = dict((trajectory_row or {}).get("state_values") or {})
    return {name: float(values.get(name) or 0.0) for name in STATE_NAMES}


def _metric_provenance_signature(row: dict[str, Any], metric_name: str) -> tuple[str, ...]:
    provenance = dict((row.get("metric_provenance") or {}).get(metric_name) or {})
    if not provenance:
        return ()
    return (
        str(provenance.get("tier") or ""),
        str(provenance.get("aggregation_mode") or ""),
        str(provenance.get("series_kind") or ""),
        str(provenance.get("support_partition") or ""),
        str(provenance.get("measurement_semantics") or ""),
    )


def _robust_bounded_latest_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    if deltas.size == 0:
        return 0.0
    reference = deltas[:-1] if deltas.size >= 2 else deltas
    center = float(np.median(reference))
    abs_deviation = np.abs(reference - center)
    scale = float(np.median(abs_deviation))
    if scale <= 0.0:
        scale = float(np.mean(abs_deviation)) if abs_deviation.size else 0.0
    if scale <= 0.0:
        scale = max(float(np.max(np.abs(reference))) if reference.size else 0.0, 1.0)
    return float(np.tanh((float(deltas[-1]) - center) / scale))


def _positive_delta_autocorrelation(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    if deltas.size < 3:
        return 0.0
    previous = deltas[:-1]
    current = deltas[1:]
    if float(np.std(previous)) <= 0.0 or float(np.std(current)) <= 0.0:
        return 0.0
    correlation = float(np.corrcoef(previous, current)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, 0.0, 1.0))


def _tukey_upper_fence(values: list[float]) -> float:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    if finite.size < 4:
        return float(np.max(finite))
    q1, q3 = np.percentile(finite, [25.0, 75.0])
    return float(q3 + 1.5 * float(q3 - q1))


def _shock_nowcast_feature_row(
    train_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    step_index: int,
) -> list[float]:
    sorted_train = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    metric_rows = [row for row in sorted_train if row.get(metric_name) is not None]
    values = [float(row.get(metric_name) or 0.0) for row in metric_rows]
    latest_deviation = _robust_bounded_latest_deviation(values)
    persistence = _positive_delta_autocorrelation(values)
    persisted_deviation = float(latest_deviation * (persistence ** int(step_index)))
    support_shift = 0.0
    monthly_bridge = 0.0
    if len(metric_rows) >= 2:
        support_shift = 1.0 if _metric_provenance_signature(metric_rows[-1], metric_name) != _metric_provenance_signature(metric_rows[-2], metric_name) else 0.0
    if metric_rows:
        provenance = dict((metric_rows[-1].get("metric_provenance") or {}).get(metric_name) or {})
        aggregation_mode = str(provenance.get("aggregation_mode") or "")
        series_kind = str(provenance.get("series_kind") or "")
        if aggregation_mode in {"monthly_to_quarter_sum", "intraquarter_snapshot_bridge"} or series_kind.startswith("monthly"):
            monthly_bridge = 1.0
    return [
        latest_deviation,
        persisted_deviation,
        support_shift,
        monthly_bridge * latest_deviation,
        support_shift * latest_deviation,
    ]


def _state_feature_row(
    *,
    metric_name: str,
    raw_row: dict[str, Any],
    trajectory_row: dict[str, Any] | None,
    previous_value: float,
    previous_previous_value: float,
    step_index: int,
    horizon_quarters: int,
    shock_features: list[float] | None = None,
) -> list[float]:
    state = _trajectory_state_values(trajectory_row)
    diagnosed_state = state_sum(state, DIAGNOSED_STATE_NAMES)
    art_state = state_sum(state, ART_STATE_NAMES)
    vl_tested_state = state_sum(state, VL_TESTED_STATE_NAMES)
    total_state = float(sum(state.values()))
    stock_balance = dict((trajectory_row or {}).get("stock_balance") or {})
    horizon_position = float(step_index + 1) / max(float(horizon_quarters), 1.0)
    if metric_name == "diagnosed_plhiv":
        state_anchor = diagnosed_state
    elif metric_name == "alive_on_art":
        state_anchor = art_state
    elif metric_name == "virally_suppressed":
        state_anchor = float(state["V"])
    elif metric_name == "tested_for_viral_load":
        state_anchor = vl_tested_state
    else:
        state_anchor = float(raw_row.get(metric_name) or 0.0)
    features = [
        1.0,
        float(raw_row.get(metric_name) or 0.0),
        float(previous_value),
        float(previous_value) - float(previous_previous_value),
        float(state_anchor),
        diagnosed_state,
        art_state,
        vl_tested_state,
        float(state["U"]),
        total_state,
        float(stock_balance.get("incidence_inflow") or raw_row.get("incident_infections_period") or 0.0),
        float(stock_balance.get("attrition_outflow") or raw_row.get("net_attrition_outflow_period") or 0.0),
        horizon_position,
    ]
    if shock_features:
        features.extend(float(np.clip(value, -1.0, 1.0)) for value in shock_features)
    return features


def _conditional_rate_target(
    *,
    metric_name: str,
    numerator_row: dict[str, Any],
    denominator_row: dict[str, Any],
    eps: float,
) -> float | None:
    if metric_name == "alive_on_art":
        denominator_metric = "diagnosed_plhiv"
    elif metric_name == "virally_suppressed":
        denominator_metric = "alive_on_art"
    else:
        return None
    numerator = _finite_float(numerator_row.get(metric_name))
    denominator = _finite_float(denominator_row.get(denominator_metric))
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    rate = float(np.clip(numerator / max(denominator, float(eps)), float(eps), 1.0 - float(eps)))
    return logit(rate, eps=float(eps))


def _standardize_design(x_rows: list[list[float]], *, eps: float) -> tuple[np.ndarray, list[float], list[float]]:
    x = np.asarray(x_rows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional design matrix")
    centers = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    centers[0] = 0.0
    scales[0] = 1.0
    safe_scales = np.where(scales <= float(eps), 1.0, scales)
    return (x - centers) / safe_scales, [float(value) for value in centers], [float(value) for value in safe_scales]


def _standardized_feature(feature: list[float], centers: list[float], scales: list[float]) -> np.ndarray:
    raw = np.asarray(feature, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if raw.size != center.size or raw.size != scale.size:
        return raw
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return (raw - center) / safe_scale


def _horizon_step_weights(examples: list[dict[str, Any]]) -> list[float]:
    step_counts: dict[int, int] = {}
    for example in examples:
        step_index = int(example.get("step_index") or 0)
        step_counts[step_index] = step_counts.get(step_index, 0) + 1
    if not step_counts:
        return []
    max_count = max(step_counts.values())
    max_step = max(step_counts)
    return [float(step_counts.get(step, 0)) / float(max_count) for step in range(max_step + 1)]


def _metric_feature_row(
    *,
    family: str,
    raw_value: float,
    previous_value: float,
    previous_previous_value: float,
) -> list[float]:
    family = _base_head_family(family)
    if family == "endpoint_raw_affine":
        return [1.0, float(raw_value)]
    if family == "endpoint_raw_plus_lag":
        return [1.0, float(raw_value), float(previous_value)]
    if family == "endpoint_r10_like_delta":
        return [
            1.0,
            float(previous_value),
            float(raw_value) - float(previous_value),
            float(previous_value) - float(previous_previous_value),
        ]
    raise ValueError(f"Unsupported endpoint head family: {family}")


def _fit_endpoint_head(
    *,
    family: str,
    train_rows: list[dict[str, Any]],
    train_backbone_rows: list[dict[str, Any]],
    reference_config: dict[str, Any] | None = None,
    constraint_rows: list[dict[str, Any]] | None = None,
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    min_train_years: int = 5,
) -> EndpointHead | None:
    if _base_head_family(family) == "mechanistic_bounded_only":
        return EndpointHead(
            family=family,
            metric_coefficients={},
            train_row_count={},
            endpoint_weight=0.0,
            training_objective="bounded_mechanistic_state_backbone",
        )
    if _is_multihorizon_family(family):
        if reference_config is None:
            return None
        return _fit_multihorizon_state_head(
            family=family,
            train_rows=train_rows,
            reference_config=reference_config,
            constraint_rows=list(constraint_rows or []),
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
            min_train_years=min_train_years,
        )
    if len(train_backbone_rows) < 2 or len(train_rows) < 3:
        return None
    base_family = _base_head_family(family)
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in train_backbone_rows}
    coefficients: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for metric_name in PRIMARY_METRICS:
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for index in range(1, len(train_rows)):
            current = train_rows[index]
            raw = by_quarter.get(str(current.get("quarter") or ""))
            if raw is None:
                continue
            target = current.get(metric_name)
            raw_value = raw.get(metric_name)
            previous_value = train_rows[index - 1].get(metric_name)
            previous_previous_value = train_rows[index - 2].get(metric_name) if index >= 2 else previous_value
            if target is None or raw_value is None or previous_value is None or previous_previous_value is None:
                continue
            x_rows.append(
                _metric_feature_row(
                    family=base_family,
                    raw_value=float(raw_value),
                    previous_value=float(previous_value),
                    previous_previous_value=float(previous_previous_value),
                )
            )
            y_rows.append(float(target))
        if len(x_rows) < 2:
            return None
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
        beta = np.linalg.pinv(x) @ y
        coefficients[metric_name] = [float(value) for value in beta]
        counts[metric_name] = len(x_rows)
    endpoint_weight = 1.0
    if str(family).endswith("_train_blend"):
        provisional = EndpointHead(
            family=base_family,
            metric_coefficients=coefficients,
            train_row_count=counts,
            endpoint_weight=1.0,
        )
        endpoint_rows = _apply_endpoint_head(
            head=provisional,
            train_rows=train_rows[:1],
            backbone_rows=train_backbone_rows,
        )
        endpoint_weight = _fit_endpoint_blend_weight(
            train_rows=train_rows,
            backbone_rows=train_backbone_rows,
            endpoint_rows=endpoint_rows,
        )
    return EndpointHead(
        family=family,
        metric_coefficients=coefficients,
        train_row_count=counts,
        endpoint_weight=endpoint_weight,
    )


def _fit_multihorizon_state_head(
    *,
    family: str,
    train_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    min_train_years: int,
) -> EndpointHead | None:
    sorted_train = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    examples = _train_multihorizon_examples(
        family=family,
        train_rows=sorted_train,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    if not examples:
        return None
    coefficients: dict[str, list[float]] = {}
    centers: dict[str, list[float]] = {}
    scales: dict[str, list[float]] = {}
    correction_bounds: dict[str, float] = {}
    counts: dict[str, int] = {}
    horizon_counts: dict[str, int] = {}
    step_weights = _horizon_step_weights(examples)
    initial_previous_values = {
        metric_name: float(sorted_train[-1].get(metric_name) or 0.0)
        for metric_name in OBSERVATION_HEAD_METRICS
    }
    for example in examples:
        horizon_key = f"{int(example['objective_horizon_years'])}y"
        horizon_counts[horizon_key] = horizon_counts.get(horizon_key, 0) + 1
    for metric_name in OBSERVATION_HEAD_METRICS:
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        residual_bound_rows: list[float] = []
        previous_by_origin: dict[tuple[int, int], dict[str, float]] = {}
        previous_previous_by_origin: dict[tuple[int, int], dict[str, float]] = {}
        for example in sorted(
            examples,
            key=lambda row: (
                int(row["train_end_year"]),
                int(row["objective_horizon_years"]),
                int(row["step_index"]),
            ),
            ):
            target_row = dict(example["target_row"])
            target_value = target_row.get(metric_name)
            if target_value is None:
                continue
            raw_value = float((example.get("backbone_row") or {}).get(metric_name) or 0.0)
            conditional_rate_metric = _is_conditional_rate_family(family) and metric_name in {"alive_on_art", "virally_suppressed"}
            target_model_value = float(target_value)
            raw_model_value = raw_value
            if conditional_rate_metric:
                target_logit = _conditional_rate_target(
                    metric_name=metric_name,
                    numerator_row=target_row,
                    denominator_row=target_row,
                    eps=float(np.finfo(np.float32).eps),
                )
                raw_logit = _conditional_rate_target(
                    metric_name=metric_name,
                    numerator_row=dict(example["backbone_row"]),
                    denominator_row=dict(example["backbone_row"]),
                    eps=float(np.finfo(np.float32).eps),
                )
                if target_logit is None or raw_logit is None:
                    continue
                target_model_value = target_logit
                raw_model_value = raw_logit
            origin_key = (int(example["train_end_year"]), int(example["objective_horizon_years"]))
            if origin_key not in previous_by_origin:
                origin_train = [
                    row
                    for row in sorted_train
                    if quarter_year(str(row.get("quarter") or "")) <= int(example["train_end_year"])
                    and row.get(metric_name) is not None
                ]
                if origin_train:
                    previous = float(origin_train[-1].get(metric_name) or 0.0)
                    previous_previous = float(origin_train[-2].get(metric_name) or previous) if len(origin_train) >= 2 else previous
                else:
                    previous = float(initial_previous_values.get(metric_name) or 0.0)
                    previous_previous = previous
                previous_by_origin[origin_key] = {metric_name: previous}
                previous_previous_by_origin[origin_key] = {metric_name: previous_previous}
            previous = previous_by_origin[origin_key][metric_name]
            previous_previous = previous_previous_by_origin[origin_key][metric_name]
            x_rows.append(
                _state_feature_row(
                    metric_name=metric_name,
                    raw_row=dict(example["backbone_row"]),
                    trajectory_row=dict(example["trajectory_row"]),
                    previous_value=previous,
                    previous_previous_value=previous_previous,
                    step_index=int(example["step_index"]),
                    horizon_quarters=int(example["horizon_quarters"]),
                    shock_features=(
                        _shock_nowcast_feature_row(
                            [
                                row
                                for row in sorted_train
                                if quarter_year(str(row.get("quarter") or "")) <= int(example["train_end_year"])
                            ],
                            metric_name=metric_name,
                            step_index=int(example["step_index"]),
                        )
                        if _is_shock_nowcast_family(family)
                        else None
                    ),
                )
            )
            y_rows.append(target_model_value - raw_model_value if _is_lifted_residual_family(family) else target_model_value)
            residual_bound_rows.append(abs(target_model_value - raw_model_value))
            previous_previous_by_origin[origin_key][metric_name] = previous
            previous_by_origin[origin_key][metric_name] = float(target_value)
        if len(x_rows) < 2:
            continue
        x, center, scale = _standardize_design(x_rows, eps=float(np.finfo(np.float32).eps))
        y = np.asarray(y_rows, dtype=np.float64)
        beta = np.linalg.pinv(x) @ y
        coefficients[metric_name] = [float(value) for value in beta]
        centers[metric_name] = center
        scales[metric_name] = scale
        counts[metric_name] = len(x_rows)
        correction_bounds[metric_name] = _tukey_upper_fence(residual_bound_rows)
    required_missing = [metric for metric in PRIMARY_METRICS if metric not in coefficients]
    if required_missing:
        return None
    endpoint_weight = 1.0
    if str(family).endswith("_train_blend"):
        endpoint_weight = _fit_multihorizon_blend_weight(
            head=EndpointHead(
                family=_base_head_family(family),
                metric_coefficients=coefficients,
                train_row_count=counts,
                feature_centers=centers,
                feature_scales=scales,
                training_horizons=horizon_counts,
                horizon_step_weights=step_weights,
                correction_bounds=correction_bounds,
                training_objective=(
                    "blocked_1y_and_5y_lifted_residual_state_loss"
                    if _is_lifted_residual_family(family)
                    else "blocked_1y_and_5y_conditional_cascade_rate_loss"
                    if _is_conditional_rate_family(family)
                    else "blocked_1y_and_5y_state_observation_loss"
                ),
            ),
            examples=examples,
            train_rows=sorted_train,
        )
    return EndpointHead(
        family=family,
        metric_coefficients=coefficients,
        train_row_count=counts,
        endpoint_weight=endpoint_weight,
        feature_centers=centers,
        feature_scales=scales,
        training_horizons=horizon_counts,
        horizon_step_weights=step_weights,
        correction_bounds=correction_bounds,
        training_objective=(
            "blocked_1y_and_5y_lifted_residual_state_loss"
            if _is_lifted_residual_family(family)
            else "blocked_1y_and_5y_conditional_cascade_rate_loss"
            if _is_conditional_rate_family(family)
            else "blocked_1y_and_5y_state_observation_loss"
        ),
    )


def _fit_endpoint_blend_weight(
    *,
    train_rows: list[dict[str, Any]],
    backbone_rows: list[dict[str, Any]],
    endpoint_rows: list[dict[str, Any]],
) -> float:
    aligned_targets = train_rows[1 : 1 + min(len(backbone_rows), len(endpoint_rows))]
    numerator = 0.0
    denominator = 0.0
    scales = {
        metric_name: max(
            [abs(float(row.get(metric_name) or 0.0)) for row in train_rows if row.get(metric_name) is not None]
            or [1.0]
        )
        for metric_name in PRIMARY_METRICS
    }
    for target, backbone, endpoint in zip(aligned_targets, backbone_rows, endpoint_rows):
        for metric_name in PRIMARY_METRICS:
            target_value = target.get(metric_name)
            backbone_value = backbone.get(metric_name)
            endpoint_value = endpoint.get(metric_name)
            if target_value is None or backbone_value is None or endpoint_value is None:
                continue
            scale = max(float(scales.get(metric_name) or 1.0), 1.0)
            delta = (float(endpoint_value) - float(backbone_value)) / scale
            residual = (float(target_value) - float(backbone_value)) / scale
            numerator += delta * residual
            denominator += delta * delta
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _fit_multihorizon_blend_weight(
    *,
    head: EndpointHead,
    examples: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> float:
    numerator = 0.0
    denominator = 0.0
    sorted_train = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    scales = {
        metric_name: max(
            [abs(float(row.get(metric_name) or 0.0)) for row in sorted_train if row.get(metric_name) is not None]
            or [1.0]
        )
        for metric_name in PRIMARY_METRICS
    }
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault((int(example["train_end_year"]), int(example["objective_horizon_years"])), []).append(example)
    for (train_end_year, objective_horizon_years), group in grouped.items():
        origin_train = [
            row
            for row in sorted_train
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        if not origin_train:
            continue
        ordered = sorted(group, key=lambda row: int(row["step_index"]))
        backbone_rows = [dict(row["backbone_row"]) for row in ordered]
        trajectory_rows = [dict(row["trajectory_row"]) for row in ordered]
        endpoint_rows = _apply_endpoint_head(
            head=head,
            train_rows=origin_train,
            backbone_rows=backbone_rows,
            trajectory_rows=trajectory_rows,
            horizon_quarters=max(1, len(backbone_rows)),
        )
        for example, backbone, endpoint in zip(ordered, backbone_rows, endpoint_rows):
            target = dict(example["target_row"])
            for metric_name in PRIMARY_METRICS:
                target_value = target.get(metric_name)
                backbone_value = backbone.get(metric_name)
                endpoint_value = endpoint.get(metric_name)
                if target_value is None or backbone_value is None or endpoint_value is None:
                    continue
                scale = max(float(scales.get(metric_name) or 1.0), 1.0)
                delta = (float(endpoint_value) - float(backbone_value)) / scale
                residual = (float(target_value) - float(backbone_value)) / scale
                numerator += delta * residual
                denominator += delta * delta
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _enforce_endpoint_cone(row: dict[str, Any]) -> dict[str, Any]:
    diagnosed = max(float(row.get("diagnosed_plhiv") or 0.0), 0.0)
    art = max(float(row.get("alive_on_art") or 0.0), 0.0)
    suppressed = max(float(row.get("virally_suppressed") or 0.0), 0.0)
    diagnosed = max(diagnosed, art, suppressed)
    art = min(art, diagnosed)
    suppressed = min(suppressed, art)
    flow = max(float(row.get("new_diagnosed_cases_period") or 0.0), 0.0)
    bounded = dict(row)
    bounded["diagnosed_plhiv"] = float(diagnosed)
    bounded["alive_on_art"] = float(art)
    bounded["virally_suppressed"] = float(suppressed)
    bounded["new_diagnosed_cases_period"] = float(flow)
    return bounded


def _apply_endpoint_head(
    *,
    head: EndpointHead,
    train_rows: list[dict[str, Any]],
    backbone_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]] | None = None,
    horizon_quarters: int | None = None,
) -> list[dict[str, Any]]:
    if _base_head_family(head.family) == "mechanistic_bounded_only":
        return [_enforce_endpoint_cone(dict(row)) for row in backbone_rows]
    if not train_rows:
        return [_enforce_endpoint_cone(dict(row)) for row in backbone_rows]
    previous_values = {
        metric_name: float(train_rows[-1].get(metric_name) or 0.0)
        for metric_name in OBSERVATION_HEAD_METRICS
    }
    previous_previous_values = {
        metric_name: float(train_rows[-2].get(metric_name) or previous_values[metric_name])
        if len(train_rows) >= 2
        else previous_values[metric_name]
        for metric_name in OBSERVATION_HEAD_METRICS
    }
    rows: list[dict[str, Any]] = []
    base_family = _base_head_family(head.family)
    endpoint_weight = float(np.clip(float(head.endpoint_weight), 0.0, 1.0))
    trajectory_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in list(trajectory_rows or [])
    }
    sorted_backbone = sorted(backbone_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    effective_horizon_quarters = int(horizon_quarters or len(sorted_backbone) or 1)
    for step_index, raw_row in enumerate(sorted_backbone):
        predicted = dict(raw_row)
        effective_endpoint_weight = endpoint_weight
        if "support_decay" in base_family:
            step_weights = list(head.horizon_step_weights or [])
            step_support = float(step_weights[step_index]) if step_index < len(step_weights) else 0.0
            effective_endpoint_weight = float(np.clip(endpoint_weight * step_support, 0.0, 1.0))
        metrics = OBSERVATION_HEAD_METRICS if _is_multihorizon_family(head.family) else PRIMARY_METRICS
        trajectory_row = trajectory_by_quarter.get(str(raw_row.get("quarter") or ""))
        for metric_name in metrics:
            beta = np.asarray(head.metric_coefficients.get(metric_name) or [], dtype=np.float64)
            if _is_multihorizon_family(head.family):
                feature = _standardized_feature(
                    _state_feature_row(
                        metric_name=metric_name,
                        raw_row=dict(raw_row),
                        trajectory_row=trajectory_row,
                        previous_value=float(previous_values[metric_name]),
                        previous_previous_value=float(previous_previous_values[metric_name]),
                        step_index=step_index,
                        horizon_quarters=effective_horizon_quarters,
                        shock_features=(
                            _shock_nowcast_feature_row(
                                train_rows,
                                metric_name=metric_name,
                                step_index=step_index,
                            )
                            if _is_shock_nowcast_family(head.family)
                            else None
                        ),
                    ),
                    list((head.feature_centers or {}).get(metric_name) or []),
                    list((head.feature_scales or {}).get(metric_name) or []),
                )
            else:
                feature = np.asarray(
                    _metric_feature_row(
                        family=base_family,
                        raw_value=float(raw_row.get(metric_name) or 0.0),
                        previous_value=float(previous_values[metric_name]),
                        previous_previous_value=float(previous_previous_values[metric_name]),
                    ),
                    dtype=np.float64,
                )
            if beta.size != feature.size:
                continue
            fitted_value = float(feature @ beta)
            if _is_conditional_rate_family(head.family) and metric_name in {"alive_on_art", "virally_suppressed"}:
                eps = float(np.finfo(np.float32).eps)
                denominator_metric = "diagnosed_plhiv" if metric_name == "alive_on_art" else "alive_on_art"
                denominator = float(predicted.get(denominator_metric) or 0.0)
                raw_logit = _conditional_rate_target(
                    metric_name=metric_name,
                    numerator_row=dict(raw_row),
                    denominator_row=dict(raw_row),
                    eps=eps,
                )
                if raw_logit is not None:
                    bound = float((head.correction_bounds or {}).get(metric_name) or 0.0)
                    if bound > 0.0:
                        fitted_value = raw_logit + float(np.clip(fitted_value - raw_logit, -bound, bound))
                fitted_value = denominator * inv_logit(fitted_value)
            elif _is_lifted_residual_family(head.family):
                raw_value = float(raw_row.get(metric_name) or 0.0)
                bound = float((head.correction_bounds or {}).get(metric_name) or 0.0)
                residual = fitted_value
                if bound > 0.0:
                    residual = float(np.clip(residual, -bound, bound))
                fitted_value = raw_value + residual
            elif _is_shock_nowcast_family(head.family):
                bound = float((head.correction_bounds or {}).get(metric_name) or 0.0)
                raw_value = float(raw_row.get(metric_name) or 0.0)
                if bound > 0.0:
                    fitted_value = raw_value + float(np.clip(fitted_value - raw_value, -bound, bound))
            predicted[metric_name] = float(max(fitted_value, 0.0))
        predicted = _enforce_endpoint_cone(predicted)
        if effective_endpoint_weight < 1.0:
            blended = dict(predicted)
            for metric_name in metrics:
                blended[metric_name] = float(
                    effective_endpoint_weight * float(predicted.get(metric_name) or 0.0)
                    + (1.0 - effective_endpoint_weight) * float(raw_row.get(metric_name) or 0.0)
                )
            if raw_row.get("virally_suppressed") is not None and predicted.get("virally_suppressed") is not None:
                blended["virally_suppressed"] = float(
                    effective_endpoint_weight * float(predicted.get("virally_suppressed") or 0.0)
                    + (1.0 - effective_endpoint_weight) * float(raw_row.get("virally_suppressed") or 0.0)
                )
            predicted = _enforce_endpoint_cone(blended)
        for metric_name in OBSERVATION_HEAD_METRICS:
            previous_previous_values[metric_name] = previous_values[metric_name]
            previous_values[metric_name] = float(predicted.get(metric_name) or 0.0)
        rows.append(predicted)
    return rows


def _carry_forward_result(dataset: BlockedTimeDataset) -> dict[str, Any]:
    incidence = carry_forward_incidence_flow_paths(dataset, mode="last_train")
    return simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=dict(incidence.get("incidence_inflow_map") or {}),
        incidence_hazard_map=dict(incidence.get("incidence_hazard_map") or {}),
        population_denominator_map=dict(incidence.get("population_denominator_map") or {}),
        attrition_outflow_map=dict(incidence.get("attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(incidence.get("state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict(incidence.get("exit_channel_state_outflow_map") or {}),
    )


def _evaluate_family_split(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    split: dict[str, Any],
    reference_config: dict[str, Any],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    min_train_years: int,
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    model_family = _delegate_family_for_horizon(family, holdout_year_count=len(list(split["holdout_years"])))
    train_constraint_rows = _filter_rows_before_holdout(constraint_rows, list(split["holdout_years"]))
    train_backbone_rows = _train_backbone_prediction_rows(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    head = _fit_endpoint_head(
        family=model_family,
        train_rows=sorted(dataset.train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))),
        train_backbone_rows=train_backbone_rows,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    if head is None:
        return None
    backbone = _forecast_reference_for_family(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    candidate_rows = _apply_endpoint_head(
        head=head,
        train_rows=sorted(dataset.train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))),
        backbone_rows=list(backbone.get("prediction_rows") or []),
        trajectory_rows=list(backbone.get("trajectory_rows") or []),
        horizon_quarters=len(list(backbone.get("prediction_rows") or [])),
    )
    carry = _carry_forward_result(dataset)
    candidate_mae = float(normalized_mae(candidate_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps))
    return {
        "family": family,
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "candidate_mae": candidate_mae,
        "carry_forward_mae": float(carry["mae"]),
        "candidate_smape": None,
        "carry_forward_smape": float(carry["smape"]),
        "endpoint_head": {
            "candidate_family": family,
            "delegate_family": model_family,
            "family": head.family,
            "metric_coefficients": head.metric_coefficients,
            "train_row_count": head.train_row_count,
            "endpoint_weight": float(head.endpoint_weight),
            "feature_centers": head.feature_centers or {},
            "feature_scales": head.feature_scales or {},
            "training_horizons": head.training_horizons or {},
            "horizon_step_weights": head.horizon_step_weights or [],
            "correction_bounds": head.correction_bounds or {},
            "training_objective": head.training_objective,
        },
        "monthly_shock_adjustment": dict(backbone.get("monthly_shock_adjustment") or {}),
        "monthly_latent_adjustment": dict(backbone.get("monthly_latent_adjustment") or {}),
        "monthly_joint_adjustment": dict(backbone.get("monthly_joint_adjustment") or {}),
        "backhalf_channel_adjustment": dict(backbone.get("backhalf_channel_adjustment") or {}),
        "prediction_rows": candidate_rows,
        "carry_forward_prediction_rows": list(carry.get("prediction_rows") or []),
        "holdout_rows": list(dataset.holdout_rows),
        "metric_scales": dict(dataset.metric_scales),
    }


def _score_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [float(row["candidate_mae"]) for row in rows]
    carry = [float(row["carry_forward_mae"]) for row in rows]
    if not candidate:
        return {
            "candidate_mean_mae": float("inf"),
            "carry_forward_mean_mae": float("inf"),
            "candidate_worst_mae": float("inf"),
            "carry_forward_worst_mae": float("inf"),
            "split_count": 0,
        }
    return {
        "candidate_mean_mae": float(np.mean(np.asarray(candidate, dtype=np.float64))),
        "carry_forward_mean_mae": float(np.mean(np.asarray(carry, dtype=np.float64))),
        "candidate_worst_mae": float(max(candidate)),
        "carry_forward_worst_mae": float(max(carry)),
        "split_count": len(rows),
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _annual_metric_series(observation_rows: list[dict[str, Any]], metric_name: str) -> list[tuple[int, float]]:
    values_by_year: dict[int, list[tuple[str, float]]] = {}
    for row in observation_rows:
        quarter = str(row.get("quarter") or "")
        value = _finite_float(row.get(metric_name))
        if not quarter or value is None:
            continue
        values_by_year.setdefault(quarter_year(quarter), []).append((quarter, value))
    series: list[tuple[int, float]] = []
    for year, values in sorted(values_by_year.items()):
        ordered = sorted(values, key=lambda item: quarter_sort_key(item[0]))
        if metric_name in FLOW_LIKE_METRICS:
            annual_value = float(sum(value for _, value in ordered))
        else:
            annual_value = float(ordered[-1][1])
        series.append((int(year), annual_value))
    return series


def _support_signature(row: dict[str, Any]) -> tuple[str, ...]:
    provenance = dict(row.get("metric_provenance") or {})
    signature: list[str] = []
    for metric_name in OBSERVATION_HEAD_METRICS:
        metric_provenance = dict(provenance.get(metric_name) or {})
        if not metric_provenance:
            continue
        signature.append(
            "|".join(
                [
                    metric_name,
                    str(metric_provenance.get("tier") or ""),
                    str(metric_provenance.get("aggregation_mode") or ""),
                    str(metric_provenance.get("measurement_semantics") or ""),
                    str(metric_provenance.get("support_partition") or ""),
                ]
            )
        )
    return tuple(sorted(signature))


def _detect_shock_catalog(observation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_rows: list[dict[str, Any]] = []
    shock_years: dict[int, list[str]] = {}
    for metric_name in OBSERVATION_HEAD_METRICS:
        series = _annual_metric_series(observation_rows, metric_name)
        if len(series) < 4:
            continue
        diffs = [
            {
                "from_year": int(series[index - 1][0]),
                "to_year": int(series[index][0]),
                "delta": float(series[index][1] - series[index - 1][1]),
                "abs_delta": abs(float(series[index][1] - series[index - 1][1])),
            }
            for index in range(1, len(series))
        ]
        abs_diffs = np.asarray([float(row["abs_delta"]) for row in diffs], dtype=np.float64)
        q1, q3 = np.percentile(abs_diffs, [25.0, 75.0])
        iqr = float(q3 - q1)
        threshold = float(q3 + 1.5 * iqr)
        if threshold <= 0.0:
            continue
        for row in diffs:
            if float(row["abs_delta"]) <= threshold:
                continue
            label = f"detected_signal_shock:{metric_name}"
            shock_years.setdefault(int(row["to_year"]), []).append(label)
            signal_rows.append(
                {
                    "metric": metric_name,
                    "from_year": int(row["from_year"]),
                    "to_year": int(row["to_year"]),
                    "delta": float(row["delta"]),
                    "abs_delta": float(row["abs_delta"]),
                    "threshold": threshold,
                    "severity_ratio": float(row["abs_delta"]) / max(threshold, float(np.finfo(np.float32).eps)),
                    "rule": "annual first-difference exceeds Tukey upper fence computed from that metric's own historical differences",
                }
            )
    support_rows: list[dict[str, Any]] = []
    rows_by_year: dict[int, list[dict[str, Any]]] = {}
    for row in observation_rows:
        quarter = str(row.get("quarter") or "")
        if quarter:
            rows_by_year.setdefault(quarter_year(quarter), []).append(row)
    previous_signature: tuple[str, ...] | None = None
    previous_year: int | None = None
    for year, rows in sorted(rows_by_year.items()):
        year_signature = tuple(sorted({item for row in rows for item in _support_signature(row)}))
        if previous_signature is not None and year_signature != previous_signature:
            label = "support_or_reporting_shift"
            shock_years.setdefault(int(year), []).append(label)
            support_rows.append(
                {
                    "from_year": previous_year,
                    "to_year": int(year),
                    "previous_signature_size": len(previous_signature),
                    "current_signature_size": len(year_signature),
                    "rule": "typed observation support signature changed between adjacent years",
                }
            )
        previous_signature = year_signature
        previous_year = int(year)
    known_rows: list[dict[str, Any]] = []
    for start_year, end_year, label in KNOWN_EXTERNAL_SHOCK_WINDOWS:
        for year in range(int(start_year), int(end_year) + 1):
            shock_years.setdefault(year, []).append(label)
        known_rows.append(
            {
                "start_year": int(start_year),
                "end_year": int(end_year),
                "label": label,
                "role": "external_annotation_not_the_only_gate",
            }
        )
    year_labels = {str(year): sorted(set(labels)) for year, labels in sorted(shock_years.items())}
    return {
        "schema_version": "phase3_dynamic_shock_catalog.v1",
        "year_labels": year_labels,
        "detected_signal_shocks": signal_rows,
        "support_reporting_shifts": support_rows,
        "known_external_annotations": known_rows,
        "contract": "Shock labels are data-adaptive annual signal discontinuities plus typed support/reporting shifts; known external events are annotations, not the sole gate.",
    }


def _shock_labels_for_year(year: int, shock_catalog: dict[str, Any]) -> list[str]:
    labels = list((shock_catalog.get("year_labels") or {}).get(str(int(year))) or [])
    previous_labels = list((shock_catalog.get("year_labels") or {}).get(str(int(year) - 1)) or [])
    if previous_labels:
        labels.append("post_shock_rebound")
    return sorted(set(labels)) or ["stable"]


def _row_normalized_error(
    prediction: dict[str, Any] | None,
    target: dict[str, Any],
    metric_scales: dict[str, Any],
) -> float | None:
    if prediction is None:
        return None
    errors: list[float] = []
    for metric_name in OBSERVATION_HEAD_METRICS:
        target_value = _finite_float(target.get(metric_name))
        prediction_value = _finite_float(prediction.get(metric_name))
        if target_value is None or prediction_value is None:
            continue
        scale = _finite_float(metric_scales.get(metric_name))
        if scale is None or scale <= 0.0:
            scale = max(abs(target_value), 1.0)
        errors.append(abs(prediction_value - target_value) / max(scale, 1.0))
    if not errors:
        return None
    return float(np.mean(np.asarray(errors, dtype=np.float64)))


def _lifted_trajectory_entries(
    evaluated_rows: list[dict[str, Any]],
    shock_catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for split in evaluated_rows:
        prediction_by_quarter = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("prediction_rows") or [])
        }
        carry_by_quarter = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("carry_forward_prediction_rows") or [])
        }
        holdout_rows = sorted(list(split.get("holdout_rows") or []), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
        metric_scales = dict(split.get("metric_scales") or {})
        if not holdout_rows:
            continue
        max_step = max(len(holdout_rows) - 1, 0)
        for step_index, target in enumerate(holdout_rows):
            quarter = str(target.get("quarter") or "")
            if not quarter:
                continue
            candidate_error = _row_normalized_error(prediction_by_quarter.get(quarter), dict(target), metric_scales)
            carry_error = _row_normalized_error(carry_by_quarter.get(quarter), dict(target), metric_scales)
            if candidate_error is None or carry_error is None:
                continue
            labels = _shock_labels_for_year(quarter_year(quarter), shock_catalog)
            entries.append(
                {
                    "family": split.get("family"),
                    "train_end_year": int(split["train_end_year"]),
                    "holdout_years": list(split["holdout_years"]),
                    "quarter": quarter,
                    "step_index": int(step_index),
                    "is_terminal_step": bool(step_index == max_step),
                    "regime_labels": labels,
                    "candidate_mae": candidate_error,
                    "carry_forward_mae": carry_error,
                    "candidate_minus_carry_forward_mae": candidate_error - carry_error,
                }
            )
    return entries


def _annotate_residual_shock_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) < 4:
        return [dict(row) for row in entries]
    carry_errors = np.asarray([float(row["carry_forward_mae"]) for row in entries], dtype=np.float64)
    q1, q3 = np.percentile(carry_errors, [25.0, 75.0])
    threshold = float(q3 + 1.5 * float(q3 - q1))
    if threshold <= 0.0 or not np.isfinite(threshold):
        return [dict(row) for row in entries]
    annotated: list[dict[str, Any]] = []
    for row in entries:
        next_row = dict(row)
        labels = list(next_row.get("regime_labels") or [])
        if float(next_row.get("carry_forward_mae") or 0.0) > threshold:
            labels.append("residual_shock:carry_forward_path_error_outlier")
            next_row["residual_shock_threshold"] = threshold
        next_row["regime_labels"] = sorted(set(labels))
        annotated.append(next_row)
    return annotated


def _entry_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "entry_count": 0,
            "candidate_mean_mae": None,
            "carry_forward_mean_mae": None,
            "candidate_worst_mae": None,
            "carry_forward_worst_mae": None,
            "candidate_minus_carry_forward_mean_mae": None,
            "candidate_minus_carry_forward_worst_mae": None,
        }
    candidate = np.asarray([float(row["candidate_mae"]) for row in entries], dtype=np.float64)
    carry = np.asarray([float(row["carry_forward_mae"]) for row in entries], dtype=np.float64)
    return {
        "entry_count": len(entries),
        "candidate_mean_mae": float(np.mean(candidate)),
        "carry_forward_mean_mae": float(np.mean(carry)),
        "candidate_worst_mae": float(np.max(candidate)),
        "carry_forward_worst_mae": float(np.max(carry)),
        "candidate_minus_carry_forward_mean_mae": float(np.mean(candidate) - np.mean(carry)),
        "candidate_minus_carry_forward_worst_mae": float(np.max(candidate) - np.max(carry)),
    }


def _regime_summaries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({label for row in entries for label in list(row.get("regime_labels") or [])})
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_entries = [row for row in entries if label in set(row.get("regime_labels") or [])]
        rows.append({"regime_label": label, **_entry_summary(label_entries)})
    return rows


def _r10_reference_scores(r10_baseline: dict[str, Any]) -> dict[str, Any]:
    score_rows = list(r10_baseline.get("score_rows") or [])
    finite_rows = [
        row
        for row in score_rows
        if _finite_float(row.get("quarterly_mean_mae")) is not None
    ]
    expanded_rows = [
        row
        for row in finite_rows
        if str(row.get("archive_variant") or "") == "merged"
    ]
    best_expanded = min(expanded_rows, key=lambda row: float(row["quarterly_mean_mae"])) if expanded_rows else None
    best_any = min(finite_rows, key=lambda row: float(row["quarterly_mean_mae"])) if finite_rows else None
    reference = best_expanded or best_any
    return {
        "available": reference is not None,
        "reference_scope": "expanded_harp_merged" if best_expanded else "best_available_legacy_or_expanded",
        "reference_quarterly_mean_mae": None if reference is None else float(reference["quarterly_mean_mae"]),
        "reference_experiment_id": None if reference is None else reference.get("experiment_id"),
        "reference_contract": None if reference is None else reference.get("contract"),
        "all_score_rows": score_rows,
    }


def _shock_aware_lifted_gate_from_evaluated(
    *,
    family: str,
    evaluated_rows: list[dict[str, Any]],
    shock_catalog: dict[str, Any],
    r10_baseline: dict[str, Any],
) -> dict[str, Any]:
    entries = _lifted_trajectory_entries(evaluated_rows, shock_catalog)
    entries = _annotate_residual_shock_entries(entries)
    overall = _entry_summary(entries)
    shock_entries = [
        row
        for row in entries
        if any(label != "stable" for label in list(row.get("regime_labels") or []))
    ]
    intermediate_entries = [row for row in entries if not bool(row.get("is_terminal_step"))]
    shock_summary = _entry_summary(shock_entries)
    intermediate_summary = _entry_summary(intermediate_entries)
    r10_reference = _r10_reference_scores(r10_baseline)
    blockers: list[str] = []
    if not entries:
        blockers.append("no_evaluable_lifted_trajectory_entries")
    if overall.get("candidate_minus_carry_forward_mean_mae") is None or float(overall["candidate_minus_carry_forward_mean_mae"]) >= 0.0:
        blockers.append("candidate_lifted_path_mean_not_better_than_carry_forward")
    if overall.get("candidate_minus_carry_forward_worst_mae") is None or float(overall["candidate_minus_carry_forward_worst_mae"]) > 0.0:
        blockers.append("candidate_lifted_path_worst_regresses_against_carry_forward")
    if shock_catalog.get("year_labels") and not shock_entries:
        blockers.append("no_evaluable_entries_in_detected_or_annotated_shock_regimes")
    if shock_entries:
        if shock_summary.get("candidate_minus_carry_forward_mean_mae") is None or float(shock_summary["candidate_minus_carry_forward_mean_mae"]) >= 0.0:
            blockers.append("candidate_shock_regime_mean_not_better_than_carry_forward")
        if shock_summary.get("candidate_minus_carry_forward_worst_mae") is None or float(shock_summary["candidate_minus_carry_forward_worst_mae"]) > 0.0:
            blockers.append("candidate_shock_regime_worst_regresses_against_carry_forward")
    if intermediate_entries:
        if intermediate_summary.get("candidate_minus_carry_forward_mean_mae") is None or float(intermediate_summary["candidate_minus_carry_forward_mean_mae"]) >= 0.0:
            blockers.append("candidate_intermediate_path_mean_not_better_than_carry_forward")
    else:
        blockers.append("no_intermediate_lifted_path_entries")
    r10_value = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))
    candidate_mean = _finite_float(overall.get("candidate_mean_mae"))
    if r10_value is None:
        blockers.append("r10_reference_score_unavailable")
    elif candidate_mean is None or candidate_mean >= r10_value:
        blockers.append("candidate_lifted_path_mean_not_better_than_r10_reference")
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": SHOCK_TRAJECTORY_GATE_SCHEMA_VERSION,
        "family": family,
        "status": "pass" if not blockers else "fail",
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "contract": (
            "Full hybrid trajectory promotion requires the lifted path, not only terminal endpoints, "
            "to beat carry-forward overall, on detected/annotated shock regimes, and on intermediate "
            "states; it must also beat the best available R10 reference score on comparable normalized "
            "quarterly path error."
        ),
        "overall": overall,
        "shock_regime": shock_summary,
        "intermediate_path": intermediate_summary,
        "regime_summaries": _regime_summaries(entries),
        "r10_reference": r10_reference,
        "shock_catalog": shock_catalog,
    }


def _shock_aware_lifted_trajectory_gate_for_family(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    start_year: int,
    end_year: int,
    min_train_years: int,
    r10_baseline: dict[str, Any],
) -> dict[str, Any]:
    shock_catalog = _detect_shock_catalog(observation_rows)
    evaluated: list[dict[str, Any]] = []
    for horizon_years in (1, 5):
        splits = _rolling_splits(
            observation_rows,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizon_years=horizon_years,
        )
        for split in splits:
            result = _evaluate_family_split(
                family=family,
                observation_rows=observation_rows,
                constraint_rows=constraint_rows,
                split=split,
                reference_config=reference_config,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
                min_train_years=min_train_years,
            )
            if result is not None:
                result["gate_horizon_years"] = int(horizon_years)
                evaluated.append(result)
    return _shock_aware_lifted_gate_from_evaluated(
        family=family,
        evaluated_rows=evaluated,
        shock_catalog=shock_catalog,
        r10_baseline=r10_baseline,
    )


def _evaluate_family(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    reference_config: dict[str, Any],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    min_train_years: int,
) -> dict[str, Any]:
    rows = [
        result
        for split in splits
        if (
            result := _evaluate_family_split(
                family=family,
                observation_rows=observation_rows,
                constraint_rows=constraint_rows,
                split=split,
                reference_config=reference_config,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
                min_train_years=min_train_years,
            )
        )
        is not None
    ]
    stored_rows = []
    for row in rows:
        stored = dict(row)
        stored.pop("prediction_rows", None)
        stored.pop("carry_forward_prediction_rows", None)
        stored.pop("holdout_rows", None)
        stored_rows.append(stored)
    return {
        "family": family,
        "score": _score_family(stored_rows),
        "rows": stored_rows,
    }


def _near_horizon_gate_for_family(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    start_year: int,
    end_year: int,
    min_train_years: int,
) -> dict[str, Any]:
    splits = _rolling_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=5,
    )
    evaluated = [
        result
        for split in splits
        if (
            result := _evaluate_family_split(
                family=family,
                observation_rows=observation_rows,
                constraint_rows=constraint_rows,
                split=split,
                reference_config=reference_config,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
                min_train_years=min_train_years,
            )
        )
        is not None
    ]
    score = _score_family(evaluated)
    passes = (
        score["split_count"] > 0
        and float(score["candidate_mean_mae"]) < float(score["carry_forward_mean_mae"])
        and float(score["candidate_worst_mae"]) <= float(score["carry_forward_worst_mae"])
    )
    rows = []
    for row in evaluated:
        rows.append(
            {
                "train_end_year": int(row["train_end_year"]),
                "holdout_years": list(row["holdout_years"]),
                "candidate_mae": float(row["candidate_mae"]),
                "carry_forward_mae": float(row["carry_forward_mae"]),
                "candidate_minus_carry_forward_mae": float(row["candidate_mae"]) - float(row["carry_forward_mae"]),
            }
        )
    return {
        "schema_version": "phase3_dynamic_hybrid_near_horizon_gate.v1",
        "family": family,
        "status": "pass" if passes else "fail",
        "trust_2035_projection": bool(passes),
        "gate_contract": "same endpoint head family must beat carry-forward on historical five-year blocked-time analogs",
        **score,
        "rows": rows,
    }


def _long_horizon_status_for_family(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    monthly_context: MonthlyShockContext | None = None,
    monthly_latent_context: MonthlyLatentContext | None = None,
    monthly_joint_context: MonthlyJointContext | None = None,
    backhalf_context: BackHalfChannelContext | None = None,
    scenario_start_year: int,
    scenario_end_year: int,
    min_train_years: int,
) -> dict[str, Any]:
    future_quarters = _future_quarters(scenario_start_year, scenario_end_year)
    dataset = _projection_dataset(observation_rows, future_quarters)
    model_family = _delegate_family_for_horizon(
        family,
        holdout_year_count=int(scenario_end_year) - int(scenario_start_year) + 1,
    )
    train_backbone_rows = _train_backbone_prediction_rows(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    head = _fit_endpoint_head(
        family=model_family,
        train_rows=sorted(dataset.train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))),
        train_backbone_rows=train_backbone_rows,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    if head is None:
        return {
            "status": "fail",
            "reason": "endpoint_head_not_estimable",
            "unstable_zero_stock_quarter_count": 0,
            "first_zero_stock_quarter": None,
        }
    backbone = _forecast_reference_for_family(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    predictions = _apply_endpoint_head(
        head=head,
        train_rows=sorted(dataset.train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))),
        backbone_rows=list(backbone.get("prediction_rows") or []),
        trajectory_rows=list(backbone.get("trajectory_rows") or []),
        horizon_quarters=len(list(backbone.get("prediction_rows") or [])),
    )
    zero_quarters = [
        str(row.get("quarter") or "")
        for row in predictions
        if float(row.get("diagnosed_plhiv") or 0.0) <= 0.0
        or float(row.get("alive_on_art") or 0.0) <= 0.0
    ]
    return {
        "status": "pass" if not zero_quarters else "fail",
        "family": family,
        "delegate_family": model_family,
        "unstable_zero_stock_quarter_count": len(zero_quarters),
        "first_zero_stock_quarter": zero_quarters[0] if zero_quarters else None,
        "final_prediction": predictions[-1] if predictions else {},
    }


def _promotion_gate(
    *,
    score: dict[str, Any],
    near_horizon_gate: dict[str, Any],
    long_horizon_status: dict[str, Any],
    shock_trajectory_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if int(score.get("split_count") or 0) <= 0:
        blockers.append("no_scored_blocked_time_splits")
    if float(score.get("candidate_mean_mae") or float("inf")) >= float(score.get("carry_forward_mean_mae") or float("inf")):
        blockers.append("fails_one_year_carry_forward_mean_gate")
    if float(score.get("candidate_worst_mae") or float("inf")) > float(score.get("carry_forward_worst_mae") or float("inf")):
        blockers.append("fails_one_year_carry_forward_worst_gate")
    if str(near_horizon_gate.get("status") or "") != "pass":
        blockers.append("fails_bounded_near_horizon_scenario_gate")
    if str(long_horizon_status.get("status") or "") != "pass":
        blockers.append("fails_bounded_long_horizon_positive_stock_gate")
    if shock_trajectory_gate is not None and str(shock_trajectory_gate.get("status") or "") != "pass":
        blockers.append("fails_shock_aware_lifted_trajectory_gate")
        blockers.extend(str(value) for value in list(shock_trajectory_gate.get("blockers") or []))
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "promote" if not blockers else "reject",
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "contract": "hybrid champion promotion requires one-year blocked-time superiority, five-year bounded scenario-gate superiority, positive bounded 2035 state readout, and shock-aware lifted trajectory superiority over carry-forward and R10 references.",
    }


def _load_r10_baseline(epigraph_root: Path) -> dict[str, Any]:
    path = (
        epigraph_root
        / "artifacts"
        / "runs"
        / "tr-v3-current-champion-r10-neighborhood-20260419-s00"
        / "analysis"
        / "tr_v3_current_champion_r10_neighborhood_batch_report.json"
    )
    payload = dict(read_json(path, default={}) or {})
    score_rows: list[dict[str, Any]] = []
    for contract in list(payload.get("contracts") or []):
        for key in ("baseline_current_champion", "merged_current_champion"):
            champion = dict((contract or {}).get(key) or {})
            if not champion:
                continue
            score_rows.append(
                {
                    "source_key": key,
                    "contract": champion.get("contract") or (contract or {}).get("contract"),
                    "archive_variant": champion.get("archive_variant"),
                    "experiment_id": champion.get("experiment_id"),
                    "quarterly_mean_mae": champion.get("quarterly_mean_mae"),
                    "quarterly_worst_mae": champion.get("quarterly_worst_mae"),
                    "quarterly_baseline_mae": champion.get("quarterly_baseline_mae"),
                    "decision": champion.get("decision"),
                }
            )
    return {
        "source_report": path.as_posix(),
        "available": bool(payload),
        "role": "baseline_only_not_candidate_on_expanded_harp",
        "reason": "latest expanded-HARP tests showed the frozen R10 family is not the active scientific fit; this search may use endpoint-head ideas but cannot promote frozen R10 directly.",
        "score_rows": score_rows,
    }


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    families = list(payload.get("families") or [])
    labels = [str(row.get("family") or "") for row in families]
    candidate = [float((row.get("score") or {}).get("candidate_mean_mae") or 0.0) for row in families]
    carry = [float((row.get("score") or {}).get("carry_forward_mean_mae") or 0.0) for row in families]
    near_candidate = [float((row.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0) for row in families]
    near_carry = [float((row.get("near_horizon_gate") or {}).get("carry_forward_mean_mae") or 0.0) for row in families]
    shock_candidate = [
        float(((row.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
        for row in families
    ]
    shock_carry = [
        float(((row.get("shock_trajectory_gate") or {}).get("overall") or {}).get("carry_forward_mean_mae") or 0.0)
        for row in families
    ]
    r10_reference = [
        float(((row.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae") or 0.0)
        for row in families
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.8), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(labels))
    flat[0].bar(x - 0.18, candidate, width=0.36, color="#b23a48", label="hybrid")
    flat[0].bar(x + 0.18, carry, width=0.36, color="#2f6f73", label="carry-forward")
    flat[0].set_title("A. One-year blocked-time MAE", loc="left", fontweight="bold")
    flat[0].set_xticks(x, labels, rotation=35, ha="right")
    flat[0].legend(frameon=False)
    flat[1].bar(x - 0.18, near_candidate, width=0.36, color="#b23a48", label="hybrid")
    flat[1].bar(x + 0.18, near_carry, width=0.36, color="#2f6f73", label="carry-forward")
    flat[1].set_title("B. Five-year bounded gate MAE", loc="left", fontweight="bold")
    flat[1].set_xticks(x, labels, rotation=35, ha="right")
    flat[1].legend(frameon=False)
    flat[2].bar(x - 0.18, shock_candidate, width=0.36, color="#b23a48", label="hybrid")
    flat[2].bar(x + 0.18, shock_carry, width=0.36, color="#2f6f73", label="carry-forward")
    if any(value > 0.0 for value in r10_reference):
        flat[2].plot(x, r10_reference, color="#111827", linewidth=1.2, marker="o", markersize=3.0, label="R10 reference")
    flat[2].set_title("C. Shock-Aware Lifted Path MAE", loc="left", fontweight="bold")
    flat[2].set_xticks(x, labels, rotation=35, ha="right")
    flat[2].legend(frameon=False)
    best = dict(payload.get("best_family") or {})
    promotion = dict(best.get("promotion_gate") or {})
    text = [
        "Hybrid champion search",
        f"Best family: {best.get('family')}",
        f"Promotion: {promotion.get('status')}",
        f"Shock gate: {(best.get('shock_trajectory_gate') or {}).get('status')}",
        "Blockers:",
    ]
    text.extend([f"- {value}" for value in list(promotion.get("blockers") or [])] or ["- none"])
    text.extend(
        [
            "",
            "Frozen R10 role:",
            str((payload.get("r10_baseline") or {}).get("role") or ""),
        ]
    )
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Hybrid Champion Search", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _write_monthly_shock_dashboard(payload: dict[str, Any], path: Path) -> None:
    families = list(payload.get("families") or [])
    by_family = {str(row.get("family") or ""): dict(row) for row in families}
    pairs = [
        ("mechanistic", "mechanistic_bounded_only", "monthly_shock_mechanistic_bounded_only"),
        ("endpoint", "endpoint_raw_affine", "monthly_shock_endpoint_raw_affine"),
        ("state", "state_multihorizon_support_decay", "monthly_shock_state_multihorizon_support_decay"),
        ("horizon", "horizon_gated_endpoint_state", "monthly_shock_horizon_gated_endpoint_state"),
    ]
    labels: list[str] = []
    one_year_delta: list[float] = []
    five_year_delta: list[float] = []
    lifted_delta: list[float] = []
    for label, base_name, monthly_name in pairs:
        base = by_family.get(base_name)
        monthly = by_family.get(monthly_name)
        if not base or not monthly:
            continue
        labels.append(label)
        one_year_delta.append(
            float((monthly.get("score") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("score") or {}).get("candidate_mean_mae") or 0.0)
        )
        five_year_delta.append(
            float((monthly.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
        )
        lifted_delta.append(
            float(((monthly.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
            - float(((base.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
        )
    monthly_rows = [row for row in families if str(row.get("family") or "").startswith("monthly_shock_")]
    feature_payload = {}
    target_payload = {}
    if monthly_rows:
        first_split = (monthly_rows[0].get("rows") or [{}])[0]
        adjustment = dict(first_split.get("monthly_shock_adjustment") or {})
        feature_payload = dict(adjustment.get("monthly_features") or {})
        target_payload = dict(adjustment.get("targets") or {})
    quarter_features = dict(feature_payload.get("quarter_features") or {})
    ordered_quarters = sorted(quarter_features, key=quarter_sort_key)
    feature_quarters = ordered_quarters[-24:] if len(ordered_quarters) > 24 else ordered_quarters
    monthly_deviation = [float((quarter_features.get(quarter) or {}).get("monthly_deviation") or 0.0) for quarter in feature_quarters]
    residual_shock = [float((quarter_features.get(quarter) or {}).get("residual_shock") or 0.0) for quarter in feature_quarters]
    coefficient_labels: list[str] = []
    coefficient_values: list[float] = []
    for target_name, fit in target_payload.items():
        for feature_name, coefficient in zip(list(feature_payload.get("feature_names") or []), list((fit or {}).get("coefficients") or [])):
            if abs(float(coefficient)) <= 0.0:
                continue
            coefficient_labels.append(f"{target_name}\n{feature_name}")
            coefficient_values.append(float(coefficient))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(labels))
    width = 0.26
    flat[0].bar(x - width, one_year_delta, width=width, color="#8c1d18", label="1y blocked")
    flat[0].bar(x, five_year_delta, width=width, color="#d97706", label="5y bounded")
    flat[0].bar(x + width, lifted_delta, width=width, color="#2563eb", label="lifted path")
    flat[0].axhline(0.0, color="#111827", linewidth=0.9)
    flat[0].set_title("A. Monthly-Hazard Delta Versus Matched Base", loc="left", fontweight="bold")
    flat[0].set_ylabel("MAE delta; below zero is better")
    flat[0].set_xticks(x, labels, rotation=0)
    flat[0].legend(frameon=False)
    if feature_quarters:
        fx = np.arange(len(feature_quarters))
        flat[1].plot(fx, monthly_deviation, color="#0f766e", linewidth=1.4, label="bounded monthly deviation")
        flat[1].bar(fx, residual_shock, color="#b91c1c", alpha=0.35, label="residual shock")
        flat[1].set_xticks(fx[:: max(1, len(fx) // 8)], feature_quarters[:: max(1, len(fx) // 8)], rotation=35, ha="right")
    flat[1].set_title("B. Train-Origin Monthly Signal", loc="left", fontweight="bold")
    flat[1].legend(frameon=False)
    if coefficient_labels:
        cx = np.arange(len(coefficient_labels))
        colors = ["#2563eb" if value >= 0.0 else "#b91c1c" for value in coefficient_values]
        flat[2].bar(cx, coefficient_values, color=colors)
        flat[2].axhline(0.0, color="#111827", linewidth=0.9)
        flat[2].set_xticks(cx, coefficient_labels, rotation=35, ha="right")
    flat[2].set_title("C. First-Split Monthly Hazard Coefficients", loc="left", fontweight="bold")
    best_monthly = min(
        monthly_rows,
        key=lambda row: float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
    ) if monthly_rows else {}
    r10 = (((best_monthly.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    text = [
        "Monthly-native shock layer",
        f"Best monthly family: {best_monthly.get('family')}",
        f"1y mean MAE: {((best_monthly.get('score') or {}).get('candidate_mean_mae'))}",
        f"Lifted MAE: {(((best_monthly.get('shock_trajectory_gate') or {}).get('overall') or {}).get('candidate_mean_mae'))}",
        f"R10 reference MAE: {r10}",
        f"Monthly row count: {feature_payload.get('monthly_row_count')}",
        f"Feature autocorrelation: {feature_payload.get('feature_autocorrelation')}",
        "Interpretation:",
        "hazard-level monthly shocks improve some gates but do not pass R10.",
    ]
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(str(value) for value in text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Monthly-Native Shock Hazard Diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _write_monthly_latent_dashboard(payload: dict[str, Any], path: Path) -> None:
    families = list(payload.get("families") or [])
    by_family = {str(row.get("family") or ""): dict(row) for row in families}
    pairs = [
        ("mechanistic", "mechanistic_bounded_only", "monthly_latent_mechanistic_bounded_only"),
        ("endpoint", "endpoint_raw_affine", "monthly_latent_endpoint_raw_affine"),
        ("state", "state_multihorizon_support_decay", "monthly_latent_state_multihorizon_support_decay"),
        ("horizon", "horizon_gated_endpoint_state", "monthly_latent_horizon_gated_endpoint_state"),
    ]
    labels: list[str] = []
    one_year_delta: list[float] = []
    five_year_delta: list[float] = []
    lifted_delta: list[float] = []
    for label, base_name, latent_name in pairs:
        base = by_family.get(base_name)
        latent = by_family.get(latent_name)
        if not base or not latent:
            continue
        labels.append(label)
        one_year_delta.append(
            float((latent.get("score") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("score") or {}).get("candidate_mean_mae") or 0.0)
        )
        five_year_delta.append(
            float((latent.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
        )
        lifted_delta.append(
            float(((latent.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
            - float(((base.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
        )
    latent_rows = [row for row in families if str(row.get("family") or "").startswith("monthly_latent_")]
    feature_payload = {}
    target_payload = {}
    if latent_rows:
        first_split = (latent_rows[0].get("rows") or [{}])[0]
        adjustment = dict(first_split.get("monthly_latent_adjustment") or {})
        feature_payload = dict(adjustment.get("monthly_latent_features") or {})
        target_payload = dict(adjustment.get("targets") or {})
    quarter_features = dict(feature_payload.get("quarter_features") or {})
    ordered_quarters = sorted(quarter_features, key=quarter_sort_key)
    feature_quarters = ordered_quarters[-24:] if len(ordered_quarters) > 24 else ordered_quarters
    reporting = [float((quarter_features.get(quarter) or {}).get("reporting_intensity") or 0.0) for quarter in feature_quarters]
    service = [float((quarter_features.get(quarter) or {}).get("service_intensity") or 0.0) for quarter in feature_quarters]
    reporting_shock = [float((quarter_features.get(quarter) or {}).get("reporting_shock") or 0.0) for quarter in feature_quarters]
    coefficient_labels: list[str] = []
    coefficient_values: list[float] = []
    for target_name, fit in target_payload.items():
        for feature_name, coefficient in zip(list(feature_payload.get("feature_names") or []), list((fit or {}).get("coefficients") or [])):
            if abs(float(coefficient)) <= 0.0:
                continue
            coefficient_labels.append(f"{target_name}\n{feature_name}")
            coefficient_values.append(float(coefficient))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(labels))
    width = 0.26
    flat[0].bar(x - width, one_year_delta, width=width, color="#7f1d1d", label="1y blocked")
    flat[0].bar(x, five_year_delta, width=width, color="#a16207", label="5y bounded")
    flat[0].bar(x + width, lifted_delta, width=width, color="#1d4ed8", label="lifted path")
    flat[0].axhline(0.0, color="#111827", linewidth=0.9)
    flat[0].set_title("A. Monthly-Latent Delta Versus Matched Base", loc="left", fontweight="bold")
    flat[0].set_ylabel("MAE delta; below zero is better")
    flat[0].set_xticks(x, labels, rotation=0)
    flat[0].legend(frameon=False)
    if feature_quarters:
        fx = np.arange(len(feature_quarters))
        flat[1].plot(fx, reporting, color="#0f766e", linewidth=1.4, label="reporting intensity")
        flat[1].plot(fx, service, color="#7c2d12", linewidth=1.4, label="service intensity")
        flat[1].bar(fx, reporting_shock, color="#b91c1c", alpha=0.25, label="reporting shock")
        flat[1].set_xticks(fx[:: max(1, len(fx) // 8)], feature_quarters[:: max(1, len(fx) // 8)], rotation=35, ha="right")
    flat[1].set_title("B. Latent Monthly Observation States", loc="left", fontweight="bold")
    flat[1].legend(frameon=False)
    if coefficient_labels:
        cx = np.arange(len(coefficient_labels))
        colors = ["#2563eb" if value >= 0.0 else "#b91c1c" for value in coefficient_values]
        flat[2].bar(cx, coefficient_values, color=colors)
        flat[2].axhline(0.0, color="#111827", linewidth=0.9)
        flat[2].set_xticks(cx, coefficient_labels, rotation=35, ha="right")
    flat[2].set_title("C. First-Split Latent Hazard Coefficients", loc="left", fontweight="bold")
    best_latent = min(
        latent_rows,
        key=lambda row: float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
    ) if latent_rows else {}
    r10 = (((best_latent.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    streams = dict(feature_payload.get("streams") or {})
    text = [
        "Monthly latent observation-state layer",
        f"Best latent family: {best_latent.get('family')}",
        f"1y mean MAE: {((best_latent.get('score') or {}).get('candidate_mean_mae'))}",
        f"Lifted MAE: {(((best_latent.get('shock_trajectory_gate') or {}).get('overall') or {}).get('candidate_mean_mae'))}",
        f"R10 reference MAE: {r10}",
        f"Monthly row count: {feature_payload.get('monthly_row_count')}",
        f"Reporting metrics: {((streams.get('reporting') or {}).get('active_metric_names'))}",
        f"Service metrics: {((streams.get('service') or {}).get('active_metric_names'))}",
        "Interpretation:",
        "latent monthly observation states are hazard drivers, not endpoint corrections.",
    ]
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(str(value) for value in text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Monthly Latent Observation-State Diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _write_monthly_joint_dashboard(payload: dict[str, Any], path: Path) -> None:
    families = list(payload.get("families") or [])
    by_family = {str(row.get("family") or ""): dict(row) for row in families}
    pairs = [
        ("mechanistic", "mechanistic_bounded_only", "monthly_joint_mechanistic_bounded_only"),
        ("endpoint", "endpoint_raw_affine", "monthly_joint_endpoint_raw_affine"),
        ("state", "state_multihorizon_support_decay", "monthly_joint_state_multihorizon_support_decay"),
        ("residual", "lifted_residual_state", "monthly_joint_lifted_residual_state"),
        ("cond-rate", "state_multihorizon_conditional_rates", "monthly_joint_state_multihorizon_conditional_rates"),
        ("cond-gate", "conditional_horizon_gated_endpoint_state", "monthly_joint_conditional_horizon_gated_endpoint_state"),
        ("horizon", "horizon_gated_endpoint_state", "monthly_joint_horizon_gated_endpoint_state"),
    ]
    labels: list[str] = []
    one_year_delta: list[float] = []
    five_year_delta: list[float] = []
    lifted_delta: list[float] = []
    for label, base_name, joint_name in pairs:
        base = by_family.get(base_name)
        joint = by_family.get(joint_name)
        if not base or not joint:
            continue
        labels.append(label)
        one_year_delta.append(
            float((joint.get("score") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("score") or {}).get("candidate_mean_mae") or 0.0)
        )
        five_year_delta.append(
            float((joint.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
        )
        lifted_delta.append(
            float(((joint.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
            - float(((base.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
        )
    joint_rows = [row for row in families if str(row.get("family") or "").startswith("monthly_joint_")]
    feature_payload = {}
    target_payload = {}
    if joint_rows:
        first_split = (joint_rows[0].get("rows") or [{}])[0]
        adjustment = dict(first_split.get("monthly_joint_adjustment") or {})
        feature_payload = dict(adjustment.get("monthly_joint_features") or {})
        target_payload = dict(adjustment.get("targets") or {})
    quarter_features = dict(feature_payload.get("quarter_features") or {})
    ordered_quarters = sorted(quarter_features, key=quarter_sort_key)
    feature_quarters = ordered_quarters[-24:] if len(ordered_quarters) > 24 else ordered_quarters
    reporting = [float((quarter_features.get(quarter) or {}).get("reporting_intensity") or 0.0) for quarter in feature_quarters]
    backlog = [float((quarter_features.get(quarter) or {}).get("backlog_pressure") or 0.0) for quarter in feature_quarters]
    service = [float((quarter_features.get(quarter) or {}).get("service_pressure") or 0.0) for quarter in feature_quarters]
    coefficient_labels: list[str] = []
    coefficient_values: list[float] = []
    for target_name, fit in target_payload.items():
        for feature_name, coefficient in zip(list(feature_payload.get("feature_names") or []), list((fit or {}).get("coefficients") or [])):
            if abs(float(coefficient)) <= 0.0:
                continue
            coefficient_labels.append(f"{target_name}\n{feature_name}")
            coefficient_values.append(float(coefficient))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(labels))
    width = 0.26
    flat[0].bar(x - width, one_year_delta, width=width, color="#7f1d1d", label="1y blocked")
    flat[0].bar(x, five_year_delta, width=width, color="#a16207", label="5y bounded")
    flat[0].bar(x + width, lifted_delta, width=width, color="#1d4ed8", label="lifted path")
    flat[0].axhline(0.0, color="#111827", linewidth=0.9)
    flat[0].set_title("A. Joint Monthly Observation Delta Versus Matched Base", loc="left", fontweight="bold")
    flat[0].set_ylabel("MAE delta; below zero is better")
    flat[0].set_xticks(x, labels, rotation=0)
    flat[0].legend(frameon=False)
    if feature_quarters:
        fx = np.arange(len(feature_quarters))
        flat[1].plot(fx, reporting, color="#0f766e", linewidth=1.4, label="reporting")
        flat[1].plot(fx, backlog, color="#7c2d12", linewidth=1.4, label="backlog")
        flat[1].plot(fx, service, color="#1d4ed8", linewidth=1.4, label="service pressure")
        flat[1].set_xticks(fx[:: max(1, len(fx) // 8)], feature_quarters[:: max(1, len(fx) // 8)], rotation=35, ha="right")
    flat[1].set_title("B. Monthly-Quarterly Joint Observation States", loc="left", fontweight="bold")
    flat[1].legend(frameon=False)
    if coefficient_labels:
        cx = np.arange(len(coefficient_labels))
        colors = ["#2563eb" if value >= 0.0 else "#b91c1c" for value in coefficient_values]
        flat[2].bar(cx, coefficient_values, color=colors)
        flat[2].axhline(0.0, color="#111827", linewidth=0.9)
        flat[2].set_xticks(cx, coefficient_labels, rotation=35, ha="right")
    flat[2].set_title("C. First-Split Joint Hazard Coefficients", loc="left", fontweight="bold")
    best_joint = min(
        joint_rows,
        key=lambda row: float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
    ) if joint_rows else {}
    r10 = (((best_joint.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    text = [
        "Joint monthly observation-state layer",
        f"Best joint family: {best_joint.get('family')}",
        f"1y mean MAE: {((best_joint.get('score') or {}).get('candidate_mean_mae'))}",
        f"Lifted MAE: {(((best_joint.get('shock_trajectory_gate') or {}).get('overall') or {}).get('candidate_mean_mae'))}",
        f"R10 reference MAE: {r10}",
        f"Monthly row count: {feature_payload.get('monthly_row_count')}",
        f"Aligned train quarters: {feature_payload.get('aligned_train_quarter_count')}",
        "Interpretation:",
        "monthly observations are aligned to quarterly cascade states before hazard fitting.",
    ]
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(str(value) for value in text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Joint Monthly Observation-State Diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _write_backhalf_channel_dashboard(payload: dict[str, Any], path: Path) -> None:
    families = list(payload.get("families") or [])
    by_family = {str(row.get("family") or ""): dict(row) for row in families}
    pairs = [
        ("mechanistic", "mechanistic_bounded_only", "backhalf_channel_mechanistic_bounded_only"),
        ("endpoint", "endpoint_raw_affine", "backhalf_channel_endpoint_raw_affine"),
        ("state", "state_multihorizon_support_decay", "backhalf_channel_state_multihorizon_support_decay"),
        ("horizon", "horizon_gated_endpoint_state", "backhalf_channel_horizon_gated_endpoint_state"),
    ]
    labels: list[str] = []
    one_year_delta: list[float] = []
    five_year_delta: list[float] = []
    lifted_delta: list[float] = []
    for label, base_name, channel_name in pairs:
        base = by_family.get(base_name)
        channel = by_family.get(channel_name)
        if not base or not channel:
            continue
        labels.append(label)
        one_year_delta.append(
            float((channel.get("score") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("score") or {}).get("candidate_mean_mae") or 0.0)
        )
        five_year_delta.append(
            float((channel.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
            - float((base.get("near_horizon_gate") or {}).get("candidate_mean_mae") or 0.0)
        )
        lifted_delta.append(
            float(((channel.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
            - float(((base.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae") or 0.0)
        )
    channel_rows = [row for row in families if str(row.get("family") or "").startswith("backhalf_channel_")]
    feature_payload = {}
    target_payload = {}
    if channel_rows:
        first_split = (channel_rows[0].get("rows") or [{}])[0]
        adjustment = dict(first_split.get("backhalf_channel_adjustment") or {})
        feature_payload = dict(adjustment.get("backhalf_channel_features") or {})
        target_payload = dict(adjustment.get("targets") or {})
    quarter_features = dict(feature_payload.get("quarter_features") or {})
    ordered_quarters = sorted(quarter_features, key=quarter_sort_key)
    feature_quarters = ordered_quarters[-24:] if len(ordered_quarters) > 24 else ordered_quarters
    art_ltfu = [float((quarter_features.get(quarter) or {}).get("art_ltfu_pressure") or 0.0) for quarter in feature_quarters]
    vl_loss = [float((quarter_features.get(quarter) or {}).get("vl_testing_loss_pressure") or 0.0) for quarter in feature_quarters]
    suppression = [float((quarter_features.get(quarter) or {}).get("suppression_capacity") or 0.0) for quarter in feature_quarters]
    coefficient_labels: list[str] = []
    coefficient_values: list[float] = []
    for target_name, fit in target_payload.items():
        for feature_name, coefficient in zip(list((fit or {}).get("feature_names") or []), list((fit or {}).get("coefficients") or [])):
            if abs(float(coefficient)) <= 0.0:
                continue
            coefficient_labels.append(f"{target_name}\n{feature_name}")
            coefficient_values.append(float(coefficient))
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(labels))
    width = 0.26
    flat[0].bar(x - width, one_year_delta, width=width, color="#7f1d1d", label="1y blocked")
    flat[0].bar(x, five_year_delta, width=width, color="#a16207", label="5y bounded")
    flat[0].bar(x + width, lifted_delta, width=width, color="#1d4ed8", label="lifted path")
    flat[0].axhline(0.0, color="#111827", linewidth=0.9)
    flat[0].set_title("A. Back-Half Channel Delta Versus Matched Base", loc="left", fontweight="bold")
    flat[0].set_ylabel("MAE delta; below zero is better")
    flat[0].set_xticks(x, labels, rotation=0)
    flat[0].legend(frameon=False)
    if feature_quarters:
        fx = np.arange(len(feature_quarters))
        flat[1].plot(fx, art_ltfu, color="#b91c1c", linewidth=1.4, label="ART LTFU pressure")
        flat[1].plot(fx, vl_loss, color="#7c2d12", linewidth=1.4, label="VL-testing loss pressure")
        flat[1].plot(fx, suppression, color="#0f766e", linewidth=1.4, label="suppression capacity")
        flat[1].set_xticks(fx[:: max(1, len(fx) // 8)], feature_quarters[:: max(1, len(fx) // 8)], rotation=35, ha="right")
    flat[1].set_title("B. Train-Origin Back-Half Channel States", loc="left", fontweight="bold")
    flat[1].legend(frameon=False)
    if coefficient_labels:
        cx = np.arange(len(coefficient_labels))
        colors = ["#2563eb" if value >= 0.0 else "#b91c1c" for value in coefficient_values]
        flat[2].bar(cx, coefficient_values, color=colors)
        flat[2].axhline(0.0, color="#111827", linewidth=0.9)
        flat[2].set_xticks(cx, coefficient_labels, rotation=35, ha="right")
    flat[2].set_title("C. First-Split Transition-Channel Coefficients", loc="left", fontweight="bold")
    best_channel = min(
        channel_rows,
        key=lambda row: float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
    ) if channel_rows else {}
    r10 = (((best_channel.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    text = [
        "Back-half transition-channel layer",
        f"Best channel family: {best_channel.get('family')}",
        f"1y mean MAE: {((best_channel.get('score') or {}).get('candidate_mean_mae'))}",
        f"Lifted MAE: {(((best_channel.get('shock_trajectory_gate') or {}).get('overall') or {}).get('candidate_mean_mae'))}",
        f"R10 reference MAE: {r10}",
        f"Monthly row count: {feature_payload.get('monthly_row_count')}",
        f"Aligned train quarters: {feature_payload.get('aligned_train_quarter_count')}",
        "Interpretation:",
        "ART initiation, LTFU, re-engagement, VL-testing loss, and suppression enter hazards before state simulation.",
    ]
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(str(value) for value in text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Back-Half Transition-Channel Diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _write_family_csv(families: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "family",
        "candidate_mean_mae",
        "carry_forward_mean_mae",
        "candidate_worst_mae",
        "carry_forward_worst_mae",
        "near_horizon_status",
        "long_horizon_status",
        "shock_trajectory_status",
        "shock_trajectory_candidate_mean_mae",
        "shock_trajectory_r10_reference_mae",
        "promotion_status",
        "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in families:
            score = dict(row.get("score") or {})
            promotion = dict(row.get("promotion_gate") or {})
            writer.writerow(
                {
                    "family": row.get("family"),
                    "candidate_mean_mae": score.get("candidate_mean_mae"),
                    "carry_forward_mean_mae": score.get("carry_forward_mean_mae"),
                    "candidate_worst_mae": score.get("candidate_worst_mae"),
                    "carry_forward_worst_mae": score.get("carry_forward_worst_mae"),
                    "near_horizon_status": (row.get("near_horizon_gate") or {}).get("status"),
                    "long_horizon_status": (row.get("long_horizon_status") or {}).get("status"),
                    "shock_trajectory_status": (row.get("shock_trajectory_gate") or {}).get("status"),
                    "shock_trajectory_candidate_mean_mae": ((row.get("shock_trajectory_gate") or {}).get("overall") or {}).get("candidate_mean_mae"),
                    "shock_trajectory_r10_reference_mae": ((row.get("shock_trajectory_gate") or {}).get("r10_reference") or {}).get("reference_quarterly_mean_mae"),
                    "promotion_status": promotion.get("status"),
                    "blockers": "|".join(str(value) for value in list(promotion.get("blockers") or [])),
                }
            )


def _markdown_report(payload: dict[str, Any]) -> str:
    best = dict(payload.get("best_family") or {})
    promotion = dict(best.get("promotion_gate") or {})
    lines = [
        "# Phase 3 Hybrid Champion Search",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source run: `{payload.get('source_run_id')}`",
        f"- Best family: `{best.get('family')}`",
        f"- Promotion: `{promotion.get('status')}`",
        f"- Blockers: `{', '.join(str(value) for value in promotion.get('blockers') or []) or 'none'}`",
        "",
        "## Contract",
        "",
        "- The conserved mechanistic model remains the state backbone.",
        "- Endpoint heads are observation/readout corrections trained only on strict observation-ledger rows.",
        "- Frozen R10 is baseline-only because expanded-HARP tests falsified it as the active family.",
        "- Promotion requires beating carry-forward, passing the bounded five-year scenario gate, and passing a shock-aware lifted trajectory gate.",
        "- The lifted trajectory gate detects arbitrary observation shocks/support perturbations and compares the full path against carry-forward plus R10 references.",
        "- The horizon-gated candidate is claim-conditional: one-year endpoint readout uses the endpoint head; multi-year scenario readout uses the support-decayed state head.",
        "",
        "## Families",
        "",
        "| Family | Mean MAE | Carry Mean MAE | Near Gate | Long Gate | Shock Gate | Promotion |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in list(payload.get("families") or []):
        score = dict(row.get("score") or {})
        lines.append(
            f"| `{row.get('family')}` | {float(score.get('candidate_mean_mae') or 0.0):.6f} | "
            f"{float(score.get('carry_forward_mean_mae') or 0.0):.6f} | "
            f"`{(row.get('near_horizon_gate') or {}).get('status')}` | "
            f"`{(row.get('long_horizon_status') or {}).get('status')}` | "
            f"`{(row.get('shock_trajectory_gate') or {}).get('status')}` | "
            f"`{(row.get('promotion_gate') or {}).get('status')}` |"
        )
    return "\n".join(lines) + "\n"


def run_hybrid_champion_search(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
    scenario_start_year: int = 2026,
    scenario_end_year: int = 2035,
    reengagement_sensitivity_mode: str = "public_stock_flow_proxy",
    head_families: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    constraint_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    reference_config = _load_reference_config(Path(reference_report_path) if reference_report_path else None)
    monthly_context = MonthlyShockContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    monthly_latent_context = MonthlyLatentContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    monthly_joint_context = MonthlyJointContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    backhalf_context = BackHalfChannelContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        reengagement_sensitivity_mode=reengagement_sensitivity_mode,
    )
    splits = _rolling_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    r10_baseline = _load_r10_baseline(epigraph_root)
    families: list[dict[str, Any]] = []
    active_head_families = tuple(head_families or HEAD_FAMILIES)
    for family in active_head_families:
        row = _evaluate_family(
            family=family,
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            splits=splits,
            reference_config=reference_config,
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
            min_train_years=min_train_years,
        )
        near_gate = _near_horizon_gate_for_family(
            family=family,
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            reference_config=reference_config,
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
        )
        long_status = _long_horizon_status_for_family(
            family=family,
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            reference_config=reference_config,
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
            scenario_start_year=scenario_start_year,
            scenario_end_year=scenario_end_year,
            min_train_years=min_train_years,
        )
        shock_gate = _shock_aware_lifted_trajectory_gate_for_family(
            family=family,
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            reference_config=reference_config,
            monthly_context=monthly_context,
            monthly_latent_context=monthly_latent_context,
            monthly_joint_context=monthly_joint_context,
            backhalf_context=backhalf_context,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            r10_baseline=r10_baseline,
        )
        promotion = _promotion_gate(
            score=dict(row.get("score") or {}),
            near_horizon_gate=near_gate,
            long_horizon_status=long_status,
            shock_trajectory_gate=shock_gate,
        )
        row["near_horizon_gate"] = near_gate
        row["long_horizon_status"] = long_status
        row["shock_trajectory_gate"] = shock_gate
        row["promotion_gate"] = promotion
        families.append(row)
    best = min(
        families,
        key=lambda row: (
            0 if bool((row.get("promotion_gate") or {}).get("promotion_eligible")) else 1,
            float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
            float((row.get("score") or {}).get("candidate_worst_mae") or float("inf")),
            str(row.get("family") or ""),
        ),
    )
    promoted = [row for row in families if bool((row.get("promotion_gate") or {}).get("promotion_eligible"))]
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "hybrid_champion_search_report.json"
    report_md_path = analysis_dir / "hybrid_champion_search_report.md"
    family_csv_path = analysis_dir / "hybrid_family_scores.csv"
    dashboard_path = analysis_dir / "hybrid_champion_dashboard.png"
    monthly_joint_dashboard_path = analysis_dir / "monthly_joint_observation_dashboard.png"
    backhalf_channel_dashboard_path = analysis_dir / "backhalf_transition_channel_dashboard.png"
    monthly_dashboard_path = analysis_dir / "monthly_shock_hazard_dashboard.png"
    monthly_latent_dashboard_path = analysis_dir / "monthly_latent_state_dashboard.png"
    payload = {
        "schema_version": HYBRID_CHAMPION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "reengagement_sensitivity_mode": reengagement_sensitivity_mode,
        "reference_report_path": str(reference_config.get("path") or ""),
        "loop_variant": "evidence-to-model-loop",
        "contract": {
            "state_backbone": "bounded_conserved_phase3_mechanistic_state_simulation",
            "endpoint_head": "strict_observation_ledger_train_only_readout",
            "horizon_gated_endpoint_state": {
                "short_horizon_delegate": "endpoint_raw_affine",
                "multi_year_delegate": "state_multihorizon_support_decay",
                "gate": "delegate selected by requested holdout horizon, not by observed target values",
            },
            "lifted_residual_state": {
                "families": ["lifted_residual_state", "monthly_joint_lifted_residual_state"],
                "training_objective": "blocked_1y_and_5y_lifted_residual_state_loss",
                "target": "target endpoint minus conserved-backbone endpoint, not the raw endpoint level",
                "bound": "residual corrections are clipped by a train-origin Tukey upper fence of observed backbone residual magnitudes",
                "role": "MJO-02 readout head that preserves the mechanistic trajectory and learns only bounded lifted-path residuals",
            },
            "conditional_cascade_rate_readout": {
                "families": [
                    "state_multihorizon_conditional_rates",
                    "conditional_horizon_gated_endpoint_state",
                    "monthly_joint_state_multihorizon_conditional_rates",
                    "monthly_joint_conditional_horizon_gated_endpoint_state",
                ],
                "training_objective": "blocked_1y_and_5y_conditional_cascade_rate_loss",
                "conditional_rates": {
                    "alive_on_art": "alive_on_art / diagnosed_plhiv",
                    "virally_suppressed": "virally_suppressed / alive_on_art",
                },
                "bound": "rate logits are clipped around the mechanistic backbone rate by train-origin Tukey residual fences",
                "role": "MJO-03 back-half readout that makes ART and suppression conditional on upstream cascade stocks instead of fitting independent counts",
            },
            "frozen_r10": "baseline_only_not_candidate",
            "shock_aware_lifted_trajectory_gate": {
                "role": "full-path gate for arbitrary detected shocks, support/reporting perturbations, and known external shock annotations",
                "known_external_annotations_are_not_sufficient": True,
                "r10_reference": "must beat best available expanded-HARP R10 quarterly mean MAE when available",
            },
            "shock_nowcast_layer": {
                "families": ["state_multihorizon_shock_nowcast", "state_multihorizon_shock_nowcast_train_blend"],
                "drivers": [
                    "robust_latest_metric_deviation",
                    "positive_autocorrelation_persisted_deviation",
                    "typed_support_or_reporting_shift",
                    "monthly_harp_bridge_deviation",
                    "support_shift_by_deviation_interaction",
                ],
                "bound": "endpoint corrections are clipped by a train-origin Tukey upper fence of observed backbone residual magnitudes",
                "role": "bounded readout nowcaster; rejected families must not be interpreted as hazard-mechanism improvements",
            },
            "monthly_native_shock_hazard_layer": {
                "families": [
                    "monthly_shock_mechanistic_bounded_only",
                    "monthly_shock_endpoint_raw_affine",
                    "monthly_shock_state_multihorizon_support_decay",
                    "monthly_shock_horizon_gated_endpoint_state",
                ],
                "drivers": [
                    "HARP monthly robust deviation",
                    "typed source/support shift",
                    "monthly residual shock label",
                    "support_shift_by_deviation_interaction",
                ],
                "targets": ["U_to_D hazard", "incidence_hazard_per_S_eff"],
                "forecast_origin_contract": "monthly features are estimated from rows at or before the training-origin quarter; holdout months use decayed last-observed monthly signal",
                "role": "hazard-level reporting/service-intensity process before simulation and observation readout",
            },
            "monthly_latent_observation_state_layer": {
                "families": [
                    "monthly_latent_mechanistic_bounded_only",
                    "monthly_latent_endpoint_raw_affine",
                    "monthly_latent_state_multihorizon_support_decay",
                    "monthly_latent_horizon_gated_endpoint_state",
                ],
                "streams": {
                    "reporting_intensity": [
                        "new_diagnosed_cases_period",
                        "new_diagnosed_cases_monthly",
                        "diagnosed_plhiv",
                        "median_cd4_at_enrollment",
                    ],
                    "service_intensity": [
                        "alive_on_art",
                        "newly_enrolled_to_treatment",
                        "deaths_reported_period",
                    ],
                },
                "estimator": "first_singular_component_of_train_origin_monthly_one_step_residual_matrix",
                "targets": ["U_to_D hazard", "incidence_hazard_per_S_eff"],
                "forecast_origin_contract": "latent streams are estimated only through the training-origin month and AR-decayed into holdout months",
                "role": "monthly observation-state process before quarterly hazard simulation",
            },
            "monthly_joint_observation_state_layer": {
                "families": [
                    "monthly_joint_mechanistic_bounded_only",
                    "monthly_joint_endpoint_raw_affine",
                    "monthly_joint_state_multihorizon_support_decay",
                    "monthly_joint_state_multihorizon_conditional_rates",
                    "monthly_joint_lifted_residual_state",
                    "monthly_joint_conditional_horizon_gated_endpoint_state",
                    "monthly_joint_horizon_gated_endpoint_state",
                ],
                "states": ["reporting_intensity", "backlog_pressure", "service_capacity", "service_pressure"],
                "estimator": "train_origin_monthly_HARP_observation_rows_aligned_to_quarterly_cascade_stocks_and_flows",
                "targets": ["U_to_D hazard", "D_to_A hazard", "A_to_T hazard", "T_to_V hazard", "A/T/V_to_L hazards", "L_to_R hazard", "incidence_hazard_per_S_eff"],
                "forecast_origin_contract": "monthly-quarterly observation states are estimated only through the training-origin quarter and AR-decayed into holdout quarters",
                "role": "joint monthly observation and quarterly cascade-state process before hazard simulation",
            },
            "backhalf_transition_channel_layer": {
                "families": [
                    "backhalf_channel_mechanistic_bounded_only",
                    "backhalf_channel_endpoint_raw_affine",
                    "backhalf_channel_state_multihorizon_support_decay",
                    "backhalf_channel_horizon_gated_endpoint_state",
                ],
                "reengagement_sensitivity_mode": reengagement_sensitivity_mode,
                "transition_channels": {
                    "D_to_A": "ART initiation",
                    "A_to_T": "VL testing among active ART without recent VL",
                    "T_to_V": "suppression after VL-tested unsuppressed state",
                    "A_to_L": "ART interruption before recent VL",
                    "T_to_L": "ART interruption after VL-tested unsuppressed state",
                    "V_to_L": "ART interruption after suppression",
                    "L_to_R": "re-engagement from interrupted ART",
                    "R_to_A": "recently re-engaged return to active ART without recent VL",
                },
                "monthly_support_states": [
                    "art_initiation_pressure",
                    "art_ltfu_pressure",
                    "reengagement_pressure",
                    "vl_testing_loss_pressure",
                    "suppression_capacity",
                ],
                "forecast_origin_contract": "back-half channel states are estimated only through the training-origin quarter and AR-decayed into holdout quarters",
                "role": "transition-level re-identification of the cascade back half before state simulation, not endpoint correction",
                "reengagement_claim_contract": (
                    "Re-engagement is direct only when restart/return-to-care rows exist. "
                    "Otherwise sensitivity modes zero/public_stock_flow_proxy/upper_bound_proxy are proxy-only and not publishable process claims."
                ),
            },
            "promotion_rule": "beat carry-forward on one-year blocked time, pass bounded five-year scenario gate, remain positive in long horizon, and pass shock-aware lifted trajectory gate versus carry-forward and R10",
        },
        "benchmark_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": int(horizon_years),
            "split_count": len(splits),
        },
        "scenario_gate_contract": {
            "scenario_start_year": int(scenario_start_year),
            "scenario_end_year": int(scenario_end_year),
        },
        "observation_role_ledger_summary": dict(build_observation_role_ledger(
            epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        ).get("summary") or {}),
        "r10_baseline": r10_baseline,
        "families": families,
        "best_family": best,
        "champion_by_claim_aware_promotion": None if not promoted else min(
            promoted,
            key=lambda row: (
                float((row.get("score") or {}).get("candidate_mean_mae") or float("inf")),
                float((row.get("score") or {}).get("candidate_worst_mae") or float("inf")),
                str(row.get("family") or ""),
            ),
        ),
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "report_markdown": report_md_path.as_posix(),
            "family_scores_csv": family_csv_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
            "monthly_joint_dashboard_png": monthly_joint_dashboard_path.as_posix(),
            "backhalf_channel_dashboard_png": backhalf_channel_dashboard_path.as_posix(),
            "monthly_shock_dashboard_png": monthly_dashboard_path.as_posix(),
            "monthly_latent_dashboard_png": monthly_latent_dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    report_md_path.write_text(_markdown_report(payload), encoding="utf-8")
    _write_family_csv(families, family_csv_path)
    _write_dashboard(payload, dashboard_path)
    _write_monthly_joint_dashboard(payload, monthly_joint_dashboard_path)
    _write_backhalf_channel_dashboard(payload, backhalf_channel_dashboard_path)
    _write_monthly_shock_dashboard(payload, monthly_dashboard_path)
    _write_monthly_latent_dashboard(payload, monthly_latent_dashboard_path)
    write_json(report_path, payload)
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase3 hybrid mechanistic plus endpoint-head champion search.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--scenario-start-year", type=int, default=2026)
    parser.add_argument("--scenario-end-year", type=int, default=2035)
    parser.add_argument(
        "--reengagement-sensitivity-mode",
        choices=("zero", "public_stock_flow_proxy", "upper_bound_proxy"),
        default="public_stock_flow_proxy",
    )
    args = parser.parse_args()
    payload = run_hybrid_champion_search(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        reference_report_path=args.reference_report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
        horizon_years=args.horizon_years,
        scenario_start_year=args.scenario_start_year,
        scenario_end_year=args.scenario_end_year,
        reengagement_sensitivity_mode=args.reengagement_sensitivity_mode,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
