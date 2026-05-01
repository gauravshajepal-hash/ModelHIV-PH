from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    STATE_NAMES,
    TRANSITION_NAMES,
    BlockedTimeDataset,
    build_blocked_time_dataset,
    build_observation_rows,
    build_state_rows,
    default_epigraph_root,
    sandbox_repo_root,
    summarize_observation_provenance,
    summarize_state_provenance,
    transition_rows,
)
from .decomposition import DecompositionControlConfig, apply_decomposition_controls
from .incidence import IncidenceFlowConfig, carry_forward_incidence_flow_paths, fit_incidence_flow_paths
from .metrics import PRIMARY_METRICS, inv_logit, logit, normalized_mae, quarter_sort_key, quarter_year
from .model import (
    DampingConfig,
    DynamicBaselineConfig,
    ObservationModelConfig,
    ShockConfig,
    _apply_observation_model,
    _quarter_positions,
    _simulate_sequence,
    carry_forward_hazards,
    fit_dynamic_hazard_paths,
    fit_observation_model,
    simulate_holdout,
)
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .phase2 import resolve_phase2_determinant_robustness_report
from .runtime import ensure_dir, read_json, write_json


SCENARIO_LAB_SCHEMA_VERSION = "phase3_dynamic_scenario_lab.v1"
DEFAULT_ACTIVE_SOURCE_RUN_ID = "tr-v3-current-champion-expanded-harp-compatibility-20260419-s00"
DEFAULT_BASELINE_SOURCE_RUN_ID = "harp-archive-wdi-standard-20260412-s19"


def _future_quarters(start_year: int, end_year: int) -> list[str]:
    if int(end_year) < int(start_year):
        raise ValueError("scenario end year must be greater than or equal to start year")
    return [f"{year}-Q{quarter}" for year in range(int(start_year), int(end_year) + 1) for quarter in range(1, 5)]


def _default_reference_report_path() -> Path:
    return (
        sandbox_repo_root()
        / "artifacts"
        / "runs"
        / "p3d-robust-determinant-gate-20260425-rev05"
        / "analysis"
        / "rev_05_report.json"
    )


def _dataclass_from_config(cls: type, config: dict[str, Any] | None, fallback: Any | None = None) -> Any | None:
    if config is None:
        return fallback
    return cls(**{key: value for key, value in dict(config).items() if key in cls.__dataclass_fields__})


def _load_reference_config(reference_report_path: Path | None) -> dict[str, Any]:
    path = Path(reference_report_path) if reference_report_path is not None else _default_reference_report_path()
    payload = dict(read_json(path, default={}) or {})
    config = dict((payload.get("best_candidate") or {}).get("config") or {})
    source = "reference_report" if config else "built_in_defaults"
    return {
        "source": source,
        "path": path.as_posix(),
        "dynamic_cfg": _dataclass_from_config(DynamicBaselineConfig, config.get("dynamic_cfg"), DynamicBaselineConfig()),
        "incidence_cfg": _dataclass_from_config(IncidenceFlowConfig, config.get("incidence_cfg"), IncidenceFlowConfig()),
        "observation_cfg": _dataclass_from_config(ObservationModelConfig, config.get("observation_cfg"), ObservationModelConfig()),
        "shock_cfg": _dataclass_from_config(ShockConfig, config.get("shock_cfg"), None),
        "damping_cfg": _dataclass_from_config(DampingConfig, config.get("damping_cfg"), DampingConfig()),
        "decomposition_cfg": _dataclass_from_config(DecompositionControlConfig, config.get("decomposition_cfg"), None),
        "raw_config": config,
    }


def _metric_scales(train_rows: list[dict[str, Any]], eps: float) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric_name in PRIMARY_METRICS:
        observed = [abs(float(row.get(metric_name) or 0.0)) for row in train_rows if row.get(metric_name) is not None]
        scales[metric_name] = max(observed) if observed else eps
    return scales


def _projection_dataset(observation_rows: list[dict[str, Any]], future_quarters: list[str]) -> BlockedTimeDataset:
    train_rows = sorted(
        [
            dict(row)
            for row in observation_rows
            if row.get("diagnosed_plhiv") is not None and row.get("alive_on_art") is not None
        ],
        key=lambda row: quarter_sort_key(str(row.get("quarter") or "")),
    )
    if len(train_rows) < 3:
        raise ValueError("Need at least three strict observation rows for scenario projection")
    train_state_rows = build_state_rows(train_rows)
    eps = float(np.finfo(np.float32).eps)
    train_transition_rows = transition_rows(train_state_rows, eps=eps)
    future_rows = [
        {
            "quarter": quarter,
            "diagnosed_plhiv": None,
            "alive_on_art": None,
            "new_diagnosed_cases_period": None,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": None,
            "metric_provenance": {},
            "row_provenance_tier": "scenario_future_not_observed",
        }
        for quarter in future_quarters
    ]
    return BlockedTimeDataset(
        holdout_years=sorted({int(quarter.split("-Q", 1)[0]) for quarter in future_quarters}),
        observation_rows=list(train_rows + future_rows),
        train_rows=train_rows,
        holdout_rows=future_rows,
        train_state_rows=train_state_rows,
        holdout_state_rows=[],
        train_transition_rows=train_transition_rows,
        metric_scales=_metric_scales(train_rows, eps),
        eps=eps,
        provenance_summary={
            "observation_rows": summarize_observation_provenance(train_rows),
            "train_observation_rows": summarize_observation_provenance(train_rows),
            "holdout_observation_rows": {"row_count": len(future_rows), "contract": "scenario_future_not_observed"},
            "train_state_rows": summarize_state_provenance(train_state_rows),
            "holdout_state_rows": {"row_count": 0, "contract": "future_states_generated_by_scenario_simulation"},
            "state_parameter_contract": "all_strict_history_through_forecast_origin",
            "metric_scale_contract": "all_strict_history_through_forecast_origin",
        },
    )


def _residual_sigma(values: list[float], predictions: list[float], *, transform: str) -> float:
    if not values or not predictions:
        return 0.0
    if transform == "logit":
        residuals = [
            logit(float(value), eps=float(np.finfo(np.float32).eps))
            - logit(float(prediction), eps=float(np.finfo(np.float32).eps))
            for value, prediction in zip(values, predictions)
        ]
    elif transform == "log1p":
        residuals = [
            float(np.log1p(max(float(value), 0.0))) - float(np.log1p(max(float(prediction), 0.0)))
            for value, prediction in zip(values, predictions)
        ]
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    sigma = float(np.std(np.asarray(residuals, dtype=np.float64)))
    if sigma > 0.0:
        return sigma
    transformed = np.asarray(
        [
            logit(float(value), eps=float(np.finfo(np.float32).eps)) if transform == "logit" else np.log1p(max(float(value), 0.0))
            for value in values
        ],
        dtype=np.float64,
    )
    if transformed.size <= 1:
        return 0.0
    return float(np.std(np.diff(transformed)))


def _empirical_uncertainty(dataset: BlockedTimeDataset, hazard_paths: dict[str, Any], incidence_paths: dict[str, Any]) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_hazard_map = dict(hazard_paths.get("train_hazard_map") or {})
    transition_sigma: dict[str, float] = {}
    for transition in TRANSITION_NAMES:
        values = [float((row.get("hazards") or {}).get(transition) or 0.0) for row in train_rows]
        predictions = [
            float((train_hazard_map.get(str(row.get("quarter") or ""), {}) or {}).get(transition) or 0.0)
            for row in train_rows
        ]
        diagnostic_sigma = float(((hazard_paths.get("diagnostics") or {}).get(transition) or {}).get("residual_sd") or 0.0)
        transition_sigma[transition] = max(diagnostic_sigma, _residual_sigma(values, predictions, transform="logit"))
    incidence_values = [float(((row.get("stock_balance") or {}).get("incidence_inflow")) or 0.0) for row in train_rows]
    incidence_predictions = [
        float((incidence_paths.get("train_incidence_inflow_map") or {}).get(str(row.get("quarter") or ""), 0.0))
        for row in train_rows
    ]
    attrition_values = [float(((row.get("stock_balance") or {}).get("attrition_outflow")) or 0.0) for row in train_rows]
    attrition_predictions = [
        float((incidence_paths.get("train_attrition_outflow_map") or {}).get(str(row.get("quarter") or ""), 0.0))
        for row in train_rows
    ]
    return {
        "transition_logit_sigma": transition_sigma,
        "incidence_log_sigma": _residual_sigma(incidence_values, incidence_predictions, transform="log1p"),
        "attrition_log_sigma": _residual_sigma(attrition_values, attrition_predictions, transform="log1p"),
        "contract": "one_scenario_unit_equals_one_empirical_train_residual_sd_on_the_module_scale",
    }


def _determinant_status(robustness_report: dict[str, Any] | None) -> dict[str, Any]:
    edge_rows = list((robustness_report or {}).get("edge_rows") or [])
    direct_rows = [dict(row) for row in edge_rows if str(row.get("edge_kind") or "") == "direct"]
    strict_allowed = [row for row in direct_rows if bool(row.get("phase3_default_allowed"))]
    exploratory = [row for row in direct_rows if bool(row.get("exploratory_covariate_only"))]
    touched_blocks = sorted(
        {
            str(row.get("source") or "")
            for row in direct_rows
            if str(row.get("source") or "")
        }
        | {
            str(row.get("target") or "")
            for row in direct_rows
            if str(row.get("target") or "")
        }
    )
    return {
        "direct_edge_count": len(direct_rows),
        "strict_allowed_direct_edge_count": len(strict_allowed),
        "exploratory_direct_edge_count": len(exploratory),
        "touched_blocks": touched_blocks,
        "strict_allowed_edge_keys": [str(row.get("edge_key") or "") for row in strict_allowed],
        "exploratory_edge_keys": [str(row.get("edge_key") or "") for row in exploratory],
    }


def _annual_to_quarter_values(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric_name)
        if value is None:
            continue
        values.append(max(float(value), 0.0) / 4.0)
    return values


def _observed_quarter_values(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric_name)
        if value is None:
            continue
        values.append(max(float(value), 0.0))
    return values


def _train_balance_values(dataset: BlockedTimeDataset, field_name: str) -> list[float]:
    return [
        max(float((row.get("stock_balance") or {}).get(field_name) or 0.0), 0.0)
        for row in list(dataset.train_transition_rows or [])
    ]


def _max_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    if not clean:
        return None
    return float(max(clean))


def _build_projection_constraints(
    *,
    dataset: BlockedTimeDataset,
    constraint_rows: list[dict[str, Any]],
    future_quarters: list[str],
) -> dict[str, Any]:
    annual_incidence_quarter_values = _annual_to_quarter_values(constraint_rows, "annual_new_infections")
    annual_aids_death_quarter_values = _annual_to_quarter_values(constraint_rows, "annual_aids_deaths")
    reported_death_values = _observed_quarter_values(constraint_rows, "deaths_reported_period")
    empirical_incidence_values = _train_balance_values(dataset, "incidence_inflow")
    empirical_removal_values = _train_balance_values(dataset, "attrition_outflow")
    incidence_cap = _max_or_none(annual_incidence_quarter_values) or _max_or_none(empirical_incidence_values)
    removal_cap = _max_or_none(annual_aids_death_quarter_values + reported_death_values) or _max_or_none(empirical_removal_values)
    if incidence_cap is None:
        raise ValueError("Cannot build scenario demographic bounds without incidence or stock-balance support")
    if removal_cap is None:
        raise ValueError("Cannot build scenario mortality/removal bounds without death or stock-balance support")
    last_total = float(sum(float(dataset.train_state_rows[-1]["state_values"][name]) for name in STATE_NAMES))
    lower_bound_by_quarter: dict[str, float] = {}
    upper_bound_by_quarter: dict[str, float] = {}
    for index, quarter in enumerate(future_quarters, start=1):
        lower_bound_by_quarter[quarter] = float(max(last_total - index * float(removal_cap), 0.0))
        upper_bound_by_quarter[quarter] = float(last_total + index * float(incidence_cap))
    return {
        "schema_version": "phase3_dynamic_projection_constraints.v1",
        "contract": {
            "incidence_cap": "max annual_new_infections/4 when available, otherwise max train stock-balance incidence",
            "mortality_removal_cap": "max annual_aids_deaths/4 or reported quarterly deaths when available, otherwise max train stock-balance attrition",
            "demographic_bounds": "cumulative lower/upper total PLHIV envelope from last observed total plus capped inflow/removal",
            "validation_only_use": "allowed_as_projection_bound_not_training_target",
        },
        "incidence_cap_per_quarter": float(incidence_cap),
        "mortality_removal_cap_per_quarter": float(removal_cap),
        "last_observed_total_plhiv_state": float(last_total),
        "source_support": {
            "annual_incidence_quarter_value_count": len(annual_incidence_quarter_values),
            "annual_aids_death_quarter_value_count": len(annual_aids_death_quarter_values),
            "reported_death_quarter_value_count": len(reported_death_values),
            "empirical_incidence_balance_count": len(empirical_incidence_values),
            "empirical_removal_balance_count": len(empirical_removal_values),
            "annual_incidence_quarter_cap": _max_or_none(annual_incidence_quarter_values),
            "mortality_quarter_cap_from_annual_aids_deaths": _max_or_none(annual_aids_death_quarter_values),
            "mortality_quarter_cap_from_reported_deaths": _max_or_none(reported_death_values),
            "empirical_incidence_balance_max": _max_or_none(empirical_incidence_values),
            "empirical_removal_balance_max": _max_or_none(empirical_removal_values),
        },
        "lower_bound_by_quarter": lower_bound_by_quarter,
        "upper_bound_by_quarter": upper_bound_by_quarter,
    }


def _apply_projection_constraints(
    *,
    incidence_map: dict[str, float],
    attrition_map: dict[str, float],
    constraints: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    incidence_cap = float(constraints.get("incidence_cap_per_quarter") or 0.0)
    removal_cap = float(constraints.get("mortality_removal_cap_per_quarter") or 0.0)
    bounded_incidence: dict[str, float] = {}
    bounded_attrition: dict[str, float] = {}
    incidence_clipped = 0
    attrition_clipped = 0
    max_incidence_excess = 0.0
    max_attrition_excess = 0.0
    for quarter, value in incidence_map.items():
        raw = max(float(value), 0.0)
        bounded = min(raw, incidence_cap)
        bounded_incidence[quarter] = float(bounded)
        if raw > bounded:
            incidence_clipped += 1
            max_incidence_excess = max(max_incidence_excess, raw - bounded)
    for quarter, value in attrition_map.items():
        raw = max(float(value), 0.0)
        bounded = min(raw, removal_cap)
        bounded_attrition[quarter] = float(bounded)
        if raw > bounded:
            attrition_clipped += 1
            max_attrition_excess = max(max_attrition_excess, raw - bounded)
    return (
        bounded_incidence,
        bounded_attrition,
        {
            "incidence_clipped_quarter_count": incidence_clipped,
            "attrition_clipped_quarter_count": attrition_clipped,
            "max_incidence_excess_before_cap": float(max_incidence_excess),
            "max_attrition_excess_before_cap": float(max_attrition_excess),
        },
    )


def _aggregate_state_attrition_map(state_map: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        quarter: float(sum(float((values or {}).get(state_name) or 0.0) for state_name in STATE_NAMES))
        for quarter, values in state_map.items()
    }


def _scale_state_attrition_to_aggregate(
    state_map: dict[str, dict[str, float]],
    aggregate_map: dict[str, float],
) -> dict[str, dict[str, float]]:
    scaled: dict[str, dict[str, float]] = {}
    for quarter, values in state_map.items():
        raw_total = float(sum(float((values or {}).get(state_name) or 0.0) for state_name in STATE_NAMES))
        target_total = max(float(aggregate_map.get(quarter) or 0.0), 0.0)
        scale = 0.0 if raw_total <= 0.0 else min(target_total / raw_total, 1.0)
        scaled[quarter] = {
            state_name: float((values or {}).get(state_name) or 0.0) * scale
            for state_name in STATE_NAMES
        }
    return scaled


def _scenario_specs(determinant_status: dict[str, Any]) -> list[dict[str, Any]]:
    touched = set(determinant_status.get("touched_blocks") or [])
    specs: list[dict[str, Any]] = [
        {
            "scenario_id": "strict_reference_status_quo",
            "label": "Strict reference",
            "allowed_use": "development_projection_reference",
            "transition_sigma_units": {},
            "incidence_sigma_units": 0.0,
            "attrition_sigma_units": 0.0,
            "source": "strict_phase3_reference_no_determinant_perturbation",
        }
    ]
    if "structural_barrier_pressure" in touched:
        specs.append(
            {
                "scenario_id": "exploratory_structural_barrier_relief",
                "label": "Barrier relief",
                "allowed_use": "exploratory_sensitivity_not_policy_effect",
                "transition_sigma_units": {"U_to_D": 1.0, "D_to_A": 1.0, "A_to_L": -1.0, "T_to_L": -1.0, "V_to_L": -1.0, "L_to_R": 1.0, "R_to_A": 1.0},
                "incidence_sigma_units": -1.0,
                "attrition_sigma_units": -1.0,
                "source": "phase2_exploratory_direct_edges_touch_structural_barrier_pressure",
            }
        )
    if "care_access_continuity" in touched:
        specs.append(
            {
                "scenario_id": "exploratory_care_access_continuity_push",
                "label": "Care continuity push",
                "allowed_use": "exploratory_sensitivity_not_policy_effect",
                "transition_sigma_units": {"D_to_A": 1.0, "A_to_T": 1.0, "T_to_V": 1.0, "A_to_L": -1.0, "T_to_L": -1.0, "V_to_L": -1.0, "L_to_R": 1.0, "R_to_A": 1.0},
                "incidence_sigma_units": 0.0,
                "attrition_sigma_units": -1.0,
                "source": "phase2_exploratory_direct_edges_touch_care_access_continuity",
            }
        )
    if "mobility_exposure_pressure" in touched:
        specs.append(
            {
                "scenario_id": "exploratory_exposure_pressure_worsens",
                "label": "Exposure pressure worsens",
                "allowed_use": "exploratory_sensitivity_not_policy_effect",
                "transition_sigma_units": {"U_to_D": -1.0, "A_to_L": 1.0, "T_to_L": 1.0, "V_to_L": 1.0, "L_to_R": -1.0, "R_to_A": -1.0},
                "incidence_sigma_units": 1.0,
                "attrition_sigma_units": 1.0,
                "source": "phase2_exploratory_direct_edges_touch_mobility_exposure_pressure",
            }
        )
    if int(determinant_status.get("exploratory_direct_edge_count") or 0) > 0:
        specs.append(
            {
                "scenario_id": "exploratory_combined_95_95_95_push",
                "label": "Combined 95 push",
                "allowed_use": "exploratory_sensitivity_not_policy_effect",
                "transition_sigma_units": {"U_to_D": 1.0, "D_to_A": 1.0, "A_to_T": 1.0, "T_to_V": 1.0, "A_to_L": -1.0, "T_to_L": -1.0, "V_to_L": -1.0, "L_to_R": 1.0, "R_to_A": 1.0},
                "incidence_sigma_units": -1.0,
                "attrition_sigma_units": -1.0,
                "source": "combined_favorable_module_perturbation_from_exploratory_phase2_blocks",
            }
        )
    if int(determinant_status.get("strict_allowed_direct_edge_count") or 0) > 0:
        specs.append(
            {
                "scenario_id": "source_stable_determinant_bundle",
                "label": "Source-stable determinant bundle",
                "allowed_use": "development_scenario_from_source_stable_phase2_edges",
                "transition_sigma_units": {"U_to_D": 1.0, "D_to_A": 1.0, "A_to_T": 1.0, "T_to_V": 1.0, "A_to_L": -1.0, "T_to_L": -1.0, "V_to_L": -1.0, "L_to_R": 1.0, "R_to_A": 1.0},
                "incidence_sigma_units": -1.0,
                "attrition_sigma_units": -1.0,
                "source": "strict_allowed_phase2_direct_edges",
            }
        )
    return specs


def _scenario_rng_seed(run_id: str, scenario_id: str, explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        digest = hashlib.sha256(f"{explicit_seed}:{scenario_id}".encode("utf-8")).hexdigest()
    else:
        digest = hashlib.sha256(f"{run_id}:{scenario_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _perturbed_hazard_map(
    base_hazard_map: dict[str, dict[str, float]],
    *,
    scenario_spec: dict[str, Any],
    uncertainty: dict[str, Any],
    rng: np.random.Generator | None,
) -> dict[str, dict[str, float]]:
    transition_units = dict(scenario_spec.get("transition_sigma_units") or {})
    transition_sigma = dict(uncertainty.get("transition_logit_sigma") or {})
    perturbed: dict[str, dict[str, float]] = {}
    for quarter, row in sorted(base_hazard_map.items(), key=lambda item: quarter_sort_key(str(item[0]))):
        perturbed[quarter] = {}
        for transition in TRANSITION_NAMES:
            base_value = float((row or {}).get(transition) or 0.0)
            sigma = float(transition_sigma.get(transition) or 0.0)
            eta_shift = float(transition_units.get(transition) or 0.0) * sigma
            if rng is not None and sigma > 0.0:
                eta_shift += float(rng.normal(0.0, sigma))
            perturbed[quarter][transition] = float(inv_logit(logit(base_value, eps=float(np.finfo(np.float32).eps)) + eta_shift))
    return perturbed


def _perturbed_positive_map(
    base_map: dict[str, float],
    *,
    sigma: float,
    sigma_units: float,
    rng: np.random.Generator | None,
) -> dict[str, float]:
    perturbed: dict[str, float] = {}
    for quarter, value in sorted(base_map.items(), key=lambda item: quarter_sort_key(str(item[0]))):
        shift = float(sigma_units) * float(sigma)
        if rng is not None and float(sigma) > 0.0:
            shift += float(rng.normal(0.0, float(sigma)))
        perturbed[quarter] = float(np.expm1(np.log1p(max(float(value), 0.0)) + shift))
    return perturbed


def _perturbed_positive_values(
    base_map: dict[str, float],
    *,
    sigma: float,
    sigma_units: float,
    rng: np.random.Generator | None,
) -> dict[str, float]:
    shift = float(sigma_units) * float(sigma)
    if rng is not None and float(sigma) > 0.0:
        shift += float(rng.normal(0.0, float(sigma)))
    return {
        str(key): float(np.expm1(np.log1p(max(float(value), 0.0)) + shift))
        for key, value in sorted(base_map.items())
    }


def _draw_count(default_basis_count: int, draw_count: int | None) -> int:
    if draw_count is not None and int(draw_count) > 0:
        return int(draw_count)
    return max(int(default_basis_count), 1)


def _scenario_draws(
    *,
    run_id: str,
    scenario_spec: dict[str, Any],
    dataset: BlockedTimeDataset,
    base_hazard_map: dict[str, dict[str, float]],
    base_incidence_map: dict[str, float],
    base_incidence_hazard_map: dict[str, float],
    base_population_denominator_map: dict[str, float],
    base_attrition_map: dict[str, float],
    base_state_attrition_map: dict[str, dict[str, float]],
    observation_model: dict[str, Any],
    damping_cfg: DampingConfig | None,
    uncertainty: dict[str, Any],
    constraints: dict[str, Any],
    draw_count: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    positions = _quarter_positions(dataset)
    initial_state = dict(dataset.train_state_rows[-1]["state_values"])
    for draw_index in range(int(draw_count)):
        rng = None
        if draw_count > 1:
            rng = np.random.default_rng(_scenario_rng_seed(run_id, str(scenario_spec["scenario_id"]), seed) + draw_index)
        hazard_map = _perturbed_hazard_map(
            base_hazard_map,
            scenario_spec=scenario_spec,
            uncertainty=uncertainty,
            rng=rng,
        )
        incidence_map = _perturbed_positive_map(
            base_incidence_map,
            sigma=float(uncertainty.get("incidence_log_sigma") or 0.0),
            sigma_units=float(scenario_spec.get("incidence_sigma_units") or 0.0),
            rng=rng,
        )
        incidence_hazard_map = _perturbed_positive_map(
            base_incidence_hazard_map,
            sigma=float(uncertainty.get("incidence_log_sigma") or 0.0),
            sigma_units=float(scenario_spec.get("incidence_sigma_units") or 0.0),
            rng=rng,
        )
        attrition_map = _perturbed_positive_map(
            base_attrition_map,
            sigma=float(uncertainty.get("attrition_log_sigma") or 0.0),
            sigma_units=float(scenario_spec.get("attrition_sigma_units") or 0.0),
            rng=rng,
        )
        state_attrition_map = {
            quarter: _perturbed_positive_values(
                dict(values or {}),
                sigma=float(uncertainty.get("attrition_log_sigma") or 0.0),
                sigma_units=float(scenario_spec.get("attrition_sigma_units") or 0.0),
                rng=rng,
            )
            for quarter, values in base_state_attrition_map.items()
        }
        if state_attrition_map:
            attrition_map = _aggregate_state_attrition_map(state_attrition_map)
        incidence_map, attrition_map, _constraint_application = _apply_projection_constraints(
            incidence_map=incidence_map,
            attrition_map=attrition_map,
            constraints=constraints,
        )
        state_attrition_map = _scale_state_attrition_to_aggregate(state_attrition_map, attrition_map)
        raw = _simulate_sequence(
            initial_state,
            target_rows,
            hazard_map,
            incidence_inflow_map=incidence_map,
            incidence_hazard_map=incidence_hazard_map,
            incidence_cap_map={quarter: float(constraints.get("incidence_cap_per_quarter") or 0.0) for quarter in incidence_map},
            population_denominator_map=base_population_denominator_map,
            attrition_outflow_map=attrition_map,
            state_attrition_outflow_map=state_attrition_map,
        )
        calibrated = _apply_observation_model(
            list(raw.get("prediction_rows") or []),
            observation_model,
            positions=positions,
            damping_cfg=damping_cfg,
        )
        for prediction, trajectory in zip(calibrated, list(raw.get("trajectory_rows") or [])):
            state_values = dict(trajectory.get("state_values") or {})
            total_plhiv = float(sum(float(state_values.get(name) or 0.0) for name in STATE_NAMES))
            diagnosed = float(prediction.get("diagnosed_plhiv") or 0.0)
            art = float(prediction.get("alive_on_art") or 0.0)
            suppressed = float(prediction.get("virally_suppressed") or 0.0)
            diagnosed_share = diagnosed / total_plhiv if total_plhiv > 0.0 else 0.0
            art_share = art / diagnosed if diagnosed > 0.0 else 0.0
            suppression_share = suppressed / art if art > 0.0 else 0.0
            rows.append(
                {
                    "scenario_id": str(scenario_spec["scenario_id"]),
                    "draw_index": int(draw_index),
                    "quarter": str(prediction.get("quarter") or ""),
                    "incident_infections_period": float((prediction.get("incident_infections_period") or (trajectory.get("stock_balance") or {}).get("incidence_inflow") or 0.0)),
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "virally_suppressed": suppressed,
                    "diagnosed_share": float(diagnosed_share),
                    "art_given_diagnosed_share": float(art_share),
                    "suppressed_given_art_share": float(suppression_share),
                    "first_95_gap": float(max(0.0, 0.95 - diagnosed_share)),
                    "second_95_gap": float(max(0.0, 0.95 - art_share)),
                    "third_95_gap": float(max(0.0, 0.95 - suppression_share)),
                }
            )
    return rows


def _summarize_draws(draw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "incident_infections_period",
        "diagnosed_plhiv",
        "alive_on_art",
        "virally_suppressed",
        "first_95_gap",
        "second_95_gap",
        "third_95_gap",
    )
    summary_rows: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["scenario_id"]), str(row["quarter"])) for row in draw_rows},
        key=lambda item: (item[0], quarter_sort_key(item[1])),
    )
    for scenario_id, quarter in keys:
        subset = [row for row in draw_rows if str(row["scenario_id"]) == scenario_id and str(row["quarter"]) == quarter]
        summary: dict[str, Any] = {
            "scenario_id": scenario_id,
            "quarter": quarter,
            "draw_count": len(subset),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            summary[f"{metric}_p10"] = float(np.percentile(values, 10))
            summary[f"{metric}_median"] = float(np.percentile(values, 50))
            summary[f"{metric}_p90"] = float(np.percentile(values, 90))
        summary_rows.append(summary)
    return summary_rows


def _write_summary_csv(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not summary_rows:
        output_path.write_text("", encoding="utf-8")
        return
    fields = list(summary_rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def _stability_diagnostics(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scenario_id in sorted({str(row.get("scenario_id") or "") for row in summary_rows}):
        scenario_rows = sorted(
            [row for row in summary_rows if str(row.get("scenario_id") or "") == scenario_id],
            key=lambda row: quarter_sort_key(str(row.get("quarter") or "")),
        )
        zero_stock_quarters = [
            str(row.get("quarter") or "")
            for row in scenario_rows
            if float(row.get("diagnosed_plhiv_median") or 0.0) <= 0.0
            or float(row.get("alive_on_art_median") or 0.0) <= 0.0
            or float(row.get("virally_suppressed_median") or 0.0) <= 0.0
        ]
        final_row = scenario_rows[-1] if scenario_rows else {}
        rows.append(
            {
                "scenario_id": scenario_id,
                "status": "unstable_zero_stock_reached" if zero_stock_quarters else "positive_median_trajectory",
                "zero_stock_quarter_count": len(zero_stock_quarters),
                "first_zero_stock_quarter": zero_stock_quarters[0] if zero_stock_quarters else None,
                "final_quarter": final_row.get("quarter"),
                "final_diagnosed_plhiv_median": final_row.get("diagnosed_plhiv_median"),
                "final_alive_on_art_median": final_row.get("alive_on_art_median"),
                "final_virally_suppressed_median": final_row.get("virally_suppressed_median"),
                "final_first_95_gap_median": final_row.get("first_95_gap_median"),
                "final_second_95_gap_median": final_row.get("second_95_gap_median"),
                "final_third_95_gap_median": final_row.get("third_95_gap_median"),
            }
        )
    return {
        "rows": rows,
        "unstable_scenario_count": sum(1 for row in rows if str(row.get("status")) == "unstable_zero_stock_reached"),
        "contract": "zero median diagnosed/ART/suppressed stock is an automatic long-horizon warning, not a tuned threshold",
    }


def _filter_rows_before_holdout(rows: list[dict[str, Any]], holdout_years: list[int]) -> list[dict[str, Any]]:
    first_holdout = min(int(year) for year in holdout_years)
    return [
        dict(row)
        for row in rows
        if str(row.get("quarter") or "") and quarter_year(str(row.get("quarter") or "")) < first_holdout
    ]


def _forecast_constrained_reference(
    *,
    dataset: BlockedTimeDataset,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
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
    }


def _near_horizon_backtest_gate(
    *,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    min_train_years: int = 5,
    horizon_years: int = 5,
) -> dict[str, Any]:
    available_years = sorted({quarter_year(str(row.get("quarter") or "")) for row in observation_rows if str(row.get("quarter") or "")})
    split_rows: list[dict[str, Any]] = []
    for train_end_year in available_years:
        train_years = [year for year in available_years if year <= train_end_year]
        holdout_years = [year for year in range(int(train_end_year) + 1, int(train_end_year) + int(horizon_years) + 1)]
        if len(train_years) < int(min_train_years):
            continue
        if not all(year in available_years for year in holdout_years):
            continue
        dataset = build_blocked_time_dataset(observation_rows, holdout_years)
        if not dataset.train_transition_rows or not dataset.holdout_rows:
            continue
        train_constraint_rows = _filter_rows_before_holdout(constraint_rows, holdout_years)
        candidate = _forecast_constrained_reference(
            dataset=dataset,
            reference_config=reference_config,
            constraint_rows=train_constraint_rows,
        )
        carry_forward_incidence = carry_forward_incidence_flow_paths(dataset, mode="last_train")
        carry_forward = simulate_holdout(
            dataset,
            carry_forward_hazards(dataset, mode="last_train"),
            incidence_inflow_map=dict(carry_forward_incidence.get("incidence_inflow_map") or {}),
        incidence_hazard_map=dict(carry_forward_incidence.get("incidence_hazard_map") or {}),
        population_denominator_map=dict(carry_forward_incidence.get("population_denominator_map") or {}),
        attrition_outflow_map=dict(carry_forward_incidence.get("attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(carry_forward_incidence.get("state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict(carry_forward_incidence.get("exit_channel_state_outflow_map") or {}),
    )
        split_rows.append(
            {
                "train_end_year": int(train_end_year),
                "holdout_years": holdout_years,
                "candidate_mae": float(candidate["mae"]),
                "carry_forward_mae": float(carry_forward["mae"]),
                "candidate_minus_carry_forward_mae": float(candidate["mae"]) - float(carry_forward["mae"]),
                "constraint_application": dict(candidate.get("constraint_application") or {}),
            }
        )
    if not split_rows:
        return {
            "schema_version": "phase3_dynamic_near_horizon_backtest_gate.v1",
            "status": "not_available",
            "reason": "no_complete_historical_five_year_holdout_windows",
            "trust_2035_projection": False,
            "rows": [],
        }
    candidate_values = [float(row["candidate_mae"]) for row in split_rows]
    carry_values = [float(row["carry_forward_mae"]) for row in split_rows]
    pass_gate = float(np.mean(candidate_values)) < float(np.mean(carry_values)) and max(candidate_values) <= max(carry_values)
    return {
        "schema_version": "phase3_dynamic_near_horizon_backtest_gate.v1",
        "status": "pass" if pass_gate else "fail",
        "gate_contract": "historical five-year blocked-time analog for the 2026-2030 near-horizon before trusting 2035 projections",
        "trust_2035_projection": bool(pass_gate),
        "candidate_mean_mae": float(np.mean(np.asarray(candidate_values, dtype=np.float64))),
        "carry_forward_mean_mae": float(np.mean(np.asarray(carry_values, dtype=np.float64))),
        "candidate_worst_mae": float(max(candidate_values)),
        "carry_forward_worst_mae": float(max(carry_values)),
        "split_count": len(split_rows),
        "rows": split_rows,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    determinant = dict(report.get("determinant_status") or {})
    stability = dict(report.get("long_horizon_stability") or {})
    backtest = dict(report.get("near_horizon_backtest_gate") or {})
    constraints = dict(report.get("projection_constraints") or {})
    admissibility = dict(report.get("forecast_admissibility") or {})
    lines = [
        "# Phase 3 Scenario Lab",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Horizon: `{(report.get('scenario_horizon') or {}).get('start_year')}-{(report.get('scenario_horizon') or {}).get('end_year')}`",
        f"- Draw count per scenario: `{report.get('draw_count')}`",
        f"- Strict source-stable direct determinant edges: `{determinant.get('strict_allowed_direct_edge_count')}`",
        f"- Exploratory determinant edges: `{determinant.get('exploratory_direct_edge_count')}`",
        f"- Incidence cap per quarter: `{float(constraints.get('incidence_cap_per_quarter') or 0.0):.3f}`",
        f"- Mortality/removal cap per quarter: `{float(constraints.get('mortality_removal_cap_per_quarter') or 0.0):.3f}`",
        f"- Near-horizon backtest gate: `{backtest.get('status')}`",
        f"- 2035 projection status: `{admissibility.get('status')}`",
        f"- Unstable long-horizon scenarios: `{stability.get('unstable_scenario_count')}`",
        "",
        "## Contract",
        "",
        "- This is a development scenario laboratory, not a policy-effect estimator.",
        "- One perturbation unit equals one empirical train residual SD on the relevant module scale.",
        "- Direct Phase 2 determinant edges are not used as strict priors unless they pass the source-stability gate.",
        "- 2035 projections remain diagnostic unless the five-year near-horizon historical analog gate beats carry-forward.",
        "",
        "## Scenario Stability",
        "",
        "| Scenario | Status | First Zero-Stock Quarter | Final Diagnosis Gap | Final ART Gap | Final Suppression Gap |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in list(stability.get("rows") or []):
        lines.append(
            f"| `{row.get('scenario_id')}` | `{row.get('status')}` | `{row.get('first_zero_stock_quarter')}` | "
            f"{float(row.get('final_first_95_gap_median') or 0.0):.4f} | "
            f"{float(row.get('final_second_95_gap_median') or 0.0):.4f} | "
            f"{float(row.get('final_third_95_gap_median') or 0.0):.4f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, path in sorted(dict(report.get("artifact_paths") or {}).items()):
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def _write_dashboard(
    summary_rows: list[dict[str, Any]],
    scenario_specs: list[dict[str, Any]],
    output_path: Path,
    *,
    constraints: dict[str, Any] | None = None,
    near_horizon_gate: dict[str, Any] | None = None,
    forecast_admissibility: dict[str, Any] | None = None,
) -> None:
    scenario_order = [str(spec["scenario_id"]) for spec in scenario_specs]
    labels = {str(spec["scenario_id"]): str(spec["label"]) for spec in scenario_specs}
    colors = {
        "strict_reference_status_quo": "#1f2937",
        "exploratory_structural_barrier_relief": "#2f6f73",
        "exploratory_care_access_continuity_push": "#3b82f6",
        "exploratory_exposure_pressure_worsens": "#b23a48",
        "exploratory_combined_95_95_95_push": "#7c5c2e",
        "source_stable_determinant_bundle": "#0f766e",
    }
    panel_metrics = [
        ("incident_infections_period", "Incident infections / quarter (thousands)", 1_000.0),
        ("diagnosed_plhiv", "Diagnosed PLHIV (millions)", 1_000_000.0),
        ("alive_on_art", "Alive on ART (millions)", 1_000_000.0),
        ("virally_suppressed", "Virally suppressed (millions)", 1_000_000.0),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.0), constrained_layout=True)
    axes_flat = list(axes.flat)
    for ax, (metric, title, scale) in zip(axes_flat[:4], panel_metrics):
        for scenario_id in scenario_order:
            rows = [row for row in summary_rows if str(row.get("scenario_id")) == scenario_id]
            if not rows:
                continue
            x = np.arange(len(rows), dtype=np.float64)
            median = np.asarray([float(row[f"{metric}_median"]) / float(scale) for row in rows], dtype=np.float64)
            p10 = np.asarray([float(row[f"{metric}_p10"]) / float(scale) for row in rows], dtype=np.float64)
            p90 = np.asarray([float(row[f"{metric}_p90"]) / float(scale) for row in rows], dtype=np.float64)
            color = colors.get(scenario_id, "#6b7280")
            ax.plot(x, median, color=color, linewidth=1.6, label=labels.get(scenario_id, scenario_id))
            ax.fill_between(x, p10, p90, color=color, alpha=0.12, linewidth=0.0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(len({str(row.get("quarter")) for row in summary_rows}) - 1, 1))
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    gap_ax = axes_flat[4]
    baseline = [row for row in summary_rows if str(row.get("scenario_id")) == "strict_reference_status_quo"]
    if baseline:
        x = np.arange(len(baseline), dtype=np.float64)
        for metric, label, color in (
            ("first_95_gap", "diagnosis gap", "#2f6f73"),
            ("second_95_gap", "ART gap", "#3b82f6"),
            ("third_95_gap", "suppression gap", "#b23a48"),
        ):
            gap_ax.plot(x, [float(row[f"{metric}_median"]) for row in baseline], color=color, linewidth=1.8, label=label)
            gap_ax.fill_between(
                x,
                [float(row[f"{metric}_p10"]) for row in baseline],
                [float(row[f"{metric}_p90"]) for row in baseline],
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
    gap_ax.set_title("95-95-95 gaps under strict reference", loc="left", fontsize=11, fontweight="bold")
    gap_ax.set_ylim(bottom=0.0)
    gap_ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    gap_ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        gap_ax.spines[spine].set_visible(False)
    text_ax = axes_flat[5]
    text_ax.axis("off")
    constraints = dict(constraints or {})
    near_horizon_gate = dict(near_horizon_gate or {})
    forecast_admissibility = dict(forecast_admissibility or {})
    text_lines = [
        "Scenario contract",
        "One perturbation unit = one empirical train residual SD.",
        "Phase 2 determinant edges are scenario knobs only unless source-stable.",
        "Current strict gate allows zero direct determinant priors.",
        f"Incidence cap/quarter = {float(constraints.get('incidence_cap_per_quarter') or 0.0):.0f}.",
        f"Mortality/removal cap/quarter = {float(constraints.get('mortality_removal_cap_per_quarter') or 0.0):.0f}.",
        f"Near-horizon gate = {near_horizon_gate.get('status')}.",
        f"Publication forecast status = {forecast_admissibility.get('status')}.",
        "",
        "Scenarios:",
    ]
    text_lines.extend([f"- {labels.get(scenario_id, scenario_id)}" for scenario_id in scenario_order])
    text_ax.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=10, family="monospace")
    for ax in axes_flat[:5]:
        quarters = sorted({str(row.get("quarter")) for row in summary_rows}, key=quarter_sort_key)
        tick_positions = [idx for idx, quarter in enumerate(quarters) if quarter.endswith("-Q1")]
        tick_labels = [quarter.split("-Q", 1)[0] for quarter in quarters if quarter.endswith("-Q1")]
        ax.set_xticks(tick_positions, tick_labels, rotation=0)
    axes_flat[0].legend(frameon=False, fontsize=7, ncols=2)
    fig.suptitle("Phase 3 Scenario Lab: Strict Reference With Determinant Perturbation Sensitivities", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def _json_safe_config(config: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in config.items():
        if hasattr(value, "__dataclass_fields__"):
            safe[key] = asdict(value)
        else:
            safe[key] = value
    return safe


def _reference_quality(reference_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(read_json(Path(str(reference_config.get("path") or "")), default={}) or {})
    score = dict((payload.get("best_candidate") or {}).get("score") or {})
    candidate = score.get("candidate_mean_mae")
    carry = score.get("carry_forward_mean_mae")
    if candidate is None or carry is None:
        return {
            "status": "unknown",
            "source_report": reference_config.get("path"),
            "beats_carry_forward": False,
        }
    return {
        "status": "scored",
        "source_report": reference_config.get("path"),
        "candidate_mean_mae": float(candidate),
        "carry_forward_mean_mae": float(carry),
        "beats_carry_forward": bool(float(candidate) < float(carry)),
    }


def _forecast_admissibility(
    *,
    near_horizon_gate: dict[str, Any],
    long_horizon_stability: dict[str, Any],
    reference_quality: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not bool(near_horizon_gate.get("trust_2035_projection")):
        blockers.append("fails_2026_2030_near_horizon_backtest_proxy")
    if int(long_horizon_stability.get("unstable_scenario_count") or 0) > 0:
        blockers.append("long_horizon_zero_stock_instability")
    if not bool(reference_quality.get("beats_carry_forward")):
        blockers.append("strict_reference_does_not_beat_carry_forward")
    return {
        "status": "admissible_development_forecast" if not blockers else "diagnostic_only_not_publication_forecast",
        "trust_2035_projection": not blockers,
        "blockers": blockers,
        "contract": "2035 fan charts require positive long-horizon stability, a passed five-year near-horizon backtest proxy, and a reference model that beats carry-forward.",
    }


def run_scenario_lab(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    phase2_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    scenario_start_year: int = 2026,
    scenario_end_year: int = 2035,
    draw_count: int | None = None,
    seed: int | None = None,
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
    future_quarters = _future_quarters(scenario_start_year, scenario_end_year)
    dataset = _projection_dataset(observation_rows, future_quarters)
    reference_config = _load_reference_config(Path(reference_report_path) if reference_report_path else None)
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
    if reference_config.get("decomposition_cfg") is not None:
        decomposition_adjustment = apply_decomposition_controls(
            dataset=dataset,
            paths=hazard_paths,
            incidence_paths=incidence_paths,
            cfg=reference_config["decomposition_cfg"],
        )
        hazard_paths = decomposition_adjustment["paths"]
        incidence_paths = decomposition_adjustment["incidence_paths"]
    observation_model = fit_observation_model(
        dataset,
        dict(hazard_paths.get("train_hazard_map") or {}),
        reference_config["observation_cfg"],
        incidence_paths=incidence_paths,
        damping_cfg=reference_config["damping_cfg"],
    )
    resolved_phase2_source_run_id = phase2_source_run_id
    if resolved_phase2_source_run_id is None:
        resolved_phase2_source_run_id = str(((read_json(Path(reference_config["path"]), default={}) or {}).get("phase2_context") or {}).get("phase2_source_run_id") or "")
    robustness_report, robustness_resolution = resolve_phase2_determinant_robustness_report(
        epigraph_root,
        resolved_phase2_source_run_id or "missing-phase2-source",
    )
    determinant_status = _determinant_status(robustness_report)
    scenario_specs = _scenario_specs(determinant_status)
    uncertainty = _empirical_uncertainty(dataset, hazard_paths, incidence_paths)
    projection_constraints = _build_projection_constraints(
        dataset=dataset,
        constraint_rows=constraint_rows,
        future_quarters=future_quarters,
    )
    # The default Monte Carlo budget is tied to the larger of historical transition
    # support and forecast length to avoid recreating the large-file/OOM failure mode.
    effective_draw_count = _draw_count(max(len(dataset.train_transition_rows), len(future_quarters)), draw_count)
    all_draw_rows: list[dict[str, Any]] = []
    for scenario_spec in scenario_specs:
        all_draw_rows.extend(
            _scenario_draws(
                run_id=run_id,
                scenario_spec=scenario_spec,
                dataset=dataset,
                base_hazard_map=dict(hazard_paths.get("holdout_hazard_map") or {}),
                base_incidence_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
                base_incidence_hazard_map=dict(incidence_paths.get("holdout_incidence_hazard_map") or {}),
                base_population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
                base_attrition_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
                base_state_attrition_map=dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
                observation_model=observation_model,
                damping_cfg=reference_config["damping_cfg"],
                uncertainty=uncertainty,
                constraints=projection_constraints,
                draw_count=effective_draw_count,
                seed=seed,
            )
        )
    summary_rows = _summarize_draws(all_draw_rows)
    stability = _stability_diagnostics(summary_rows)
    near_horizon_gate = _near_horizon_backtest_gate(
        observation_rows=observation_rows,
        constraint_rows=constraint_rows,
        reference_config=reference_config,
    )
    quality = _reference_quality(reference_config)
    admissibility = _forecast_admissibility(
        near_horizon_gate=near_horizon_gate,
        long_horizon_stability=stability,
        reference_quality=quality,
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "scenario_lab_report.json"
    report_md_path = analysis_dir / "scenario_lab_report.md"
    summary_path = analysis_dir / "scenario_quarter_summary.csv"
    draw_path = analysis_dir / "scenario_draw_rows.json"
    dashboard_path = analysis_dir / "scenario_lab_fan_chart.png"
    _write_summary_csv(summary_rows, summary_path)
    write_json(draw_path, all_draw_rows)
    _write_dashboard(
        summary_rows,
        scenario_specs,
        dashboard_path,
        constraints=projection_constraints,
        near_horizon_gate=near_horizon_gate,
        forecast_admissibility=admissibility,
    )
    report = {
        "schema_version": SCENARIO_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "phase2_source_run_id": resolved_phase2_source_run_id,
        "scenario_horizon": {
            "start_year": int(scenario_start_year),
            "end_year": int(scenario_end_year),
            "quarter_count": len(future_quarters),
        },
        "contract": {
            "model_reference": "strict_phase3_reference_fitted_on_role_filtered_history",
            "phase2_use": "determinant_scenario_perturbation_only_unless_source_stable",
            "phase2_direct_prior_use": "not_used_when_robustness_gate_has_zero_allowed_edges",
            "perturbation_scale": str(uncertainty.get("contract")),
            "projection_constraints": "explicit incidence and mortality/removal caps plus cumulative demographic total bounds",
            "policy_effect_claim": "disallowed",
            "causal_effect_claim": "disallowed",
            "allowed_claim": "development_sensitivity_projection_with_empirical_uncertainty",
        },
        "reference_config": _json_safe_config(reference_config),
        "reference_quality": quality,
        "determinant_robustness_resolution": robustness_resolution,
        "determinant_status": determinant_status,
        "uncertainty": uncertainty,
        "projection_constraints": projection_constraints,
        "draw_count": effective_draw_count,
        "scenario_specs": scenario_specs,
        "long_horizon_stability": stability,
        "near_horizon_backtest_gate": near_horizon_gate,
        "forecast_admissibility": admissibility,
        "observation_role_ledger_summary": dict(build_observation_role_ledger(
            epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        ).get("summary") or {}),
        "dataset_provenance": dict(dataset.provenance_summary),
        "summary_rows": summary_rows,
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "report_markdown": report_md_path.as_posix(),
            "summary_csv": summary_path.as_posix(),
            "draw_rows_json": draw_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, report)
    report_md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase3 dynamic scenario lab projections.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--phase2-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--scenario-start-year", type=int, default=2026)
    parser.add_argument("--scenario-end-year", type=int, default=2035)
    parser.add_argument("--draw-count", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    report = run_scenario_lab(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        phase2_source_run_id=args.phase2_source_run_id,
        reference_report_path=args.reference_report_path,
        scenario_start_year=args.scenario_start_year,
        scenario_end_year=args.scenario_end_year,
        draw_count=args.draw_count,
        seed=args.seed,
    )
    print(json.dumps(report.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
