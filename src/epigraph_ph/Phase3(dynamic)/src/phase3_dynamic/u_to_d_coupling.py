from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    ART_STATE_NAMES,
    DIAGNOSED_STATE_NAMES,
    STATE_NAMES,
    TRANSITION_NAMES,
    VL_TESTED_STATE_NAMES,
    build_blocked_time_dataset,
    build_observation_rows,
    canonical_state_values,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
    state_sum,
)
from .diagnosis_incidence_repair import (
    _default_source_report,
    _fit_monthly_reporting_nowcast_head,
    _selected_family,
    apply_selective_repair_gate,
)
from .hybrid_champion import (
    _detect_shock_catalog,
    _load_r10_baseline,
    _load_reference_config,
    _r10_reference_scores,
    _shock_aware_lifted_gate_from_evaluated,
)
from .incidence import apply_stock_balance, fit_incidence_flow_paths
from .incidence_module_search import (
    _annual_incidence_targets,
    _evaluate_split as _evaluate_incidence_split,
)
from .metrics import normalized_mae, quarter_sort_key
from .model import (
    _apply_observation_model,
    _quarter_positions,
    carry_forward_hazards,
    fit_dynamic_hazard_paths,
    fit_observation_model,
    simulate_holdout,
)
from .monthly_shock import MonthlyShockContext
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
    _scale_state_attrition_to_aggregate,
)


U_TO_D_COUPLING_SCHEMA_VERSION = "phase3_dynamic_u_to_d_state_readout_coupling.v1"
PROMOTED_INCIDENCE_FAMILY = "s_eff_hazard_r10_mechanistic_annual_measurement_readout"
SELECTIVE_DIAGNOSIS_REPAIR_FAMILY = "selective_monthly_nowcast_backlog"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _hazard_bounds(dataset: Any, transition: str) -> dict[str, Any]:
    values: list[float] = []
    for row in list(getattr(dataset, "train_transition_rows", []) or []):
        value = _finite_float((row.get("hazards") or {}).get(transition))
        if value is not None:
            values.append(max(float(value), 0.0))
    if not values:
        return {
            "floor": 0.0,
            "ceiling": 0.0,
            "train_count": 0,
            "contract": "no train hazards available; branch fails closed upstream",
        }
    return {
        "floor": float(min(values)),
        "ceiling": float(max(values)),
        "train_count": len(values),
        "contract": "empirical train-window transition-hazard support; no hand-tuned bound",
    }


def _bounded_u_to_d_hazard(
    *,
    target_diagnosis_flow: float,
    undiagnosed_state: float,
    floor: float,
    ceiling: float,
    eps: float,
) -> dict[str, float]:
    if undiagnosed_state <= eps:
        raw = 0.0
    else:
        raw = max(float(target_diagnosis_flow), 0.0) / max(float(undiagnosed_state), float(eps))
    bounded = float(np.clip(raw, max(float(floor), 0.0), max(float(ceiling), 0.0)))
    return {
        "raw_hazard": float(raw),
        "bounded_hazard": bounded,
        "implied_flow": float(bounded * max(float(undiagnosed_state), 0.0)),
    }


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [float(row["candidate_mae"]) for row in rows if _finite_float(row.get("candidate_mae")) is not None]
    carry = [float(row["carry_forward_mae"]) for row in rows if _finite_float(row.get("carry_forward_mae")) is not None]
    if not candidate or not carry:
        return {
            "split_count": 0,
            "candidate_mean_mae": None,
            "carry_forward_mean_mae": None,
            "candidate_worst_mae": None,
            "carry_forward_worst_mae": None,
            "candidate_minus_carry_forward_mean_mae": None,
            "candidate_minus_carry_forward_worst_mae": None,
        }
    candidate_array = np.asarray(candidate, dtype=np.float64)
    carry_array = np.asarray(carry, dtype=np.float64)
    return {
        "split_count": len(candidate),
        "candidate_mean_mae": float(np.mean(candidate_array)),
        "carry_forward_mean_mae": float(np.mean(carry_array)),
        "candidate_worst_mae": float(np.max(candidate_array)),
        "carry_forward_worst_mae": float(np.max(carry_array)),
        "candidate_minus_carry_forward_mean_mae": float(np.mean(candidate_array) - np.mean(carry_array)),
        "candidate_minus_carry_forward_worst_mae": float(np.max(candidate_array) - np.max(carry_array)),
    }


def _latent_incidence_paths(
    *,
    dataset: Any,
    reference_config: dict[str, Any],
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    validation_targets: dict[int, dict[str, Any]],
    train_end_year: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_paths = dict(fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"]))
    incidence_result = _evaluate_incidence_split(
        family=PROMOTED_INCIDENCE_FAMILY,
        dataset=dataset,
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        targets=validation_targets,
        train_end_year=int(train_end_year),
        horizon_years=len(list(getattr(dataset, "holdout_years", []) or [])),
    )
    if str(incidence_result.get("fit_status") or "") != "completed":
        return base_paths, incidence_result
    base_paths["holdout_incidence_inflow_map"] = {
        str(quarter): float(value)
        for quarter, value in dict(incidence_result.get("incidence_map") or {}).items()
    }
    base_paths["holdout_incidence_hazard_map"] = {}
    base_paths["holdout_incidence_cap_map"] = {}
    base_paths["state_incidence_contract"] = (
        "latent interval-aware S_eff incidence from the promoted incidence branch enters U; "
        "annual measurement readout is excluded from state transitions"
    )
    base_paths["excluded_measurement_readout_map"] = {
        str(quarter): float(value)
        for quarter, value in dict(incidence_result.get("annual_measurement_readout_incidence_map") or {}).items()
    }
    return base_paths, incidence_result


def _simulate_base_path(
    *,
    dataset: Any,
    hazard_paths: dict[str, Any],
    incidence_paths: dict[str, Any],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
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
    raw = _simulate_sequence_with_u_to_d_targets(
        initial_state=dict(dataset.train_state_rows[-1]["state_values"]),
        target_rows=holdout_rows,
        hazard_map=dict(hazard_paths.get("holdout_hazard_map") or {}),
        u_to_d_flow_targets={},
        u_to_d_bounds=_hazard_bounds(dataset, "U_to_D"),
        eps=float(dataset.eps),
        incidence_inflow_map=incidence_map,
        incidence_cap_map={quarter: float(constraints.get("incidence_cap_per_quarter") or 0.0) for quarter in holdout_quarters},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=attrition_map,
        state_attrition_outflow_map=state_attrition_map,
        exit_channel_state_outflow_map=dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {}),
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
        "raw_prediction_rows": list(raw.get("prediction_rows") or []),
        "trajectory_rows": list(raw.get("trajectory_rows") or []),
        "u_to_d_coupling_rows": list(raw.get("u_to_d_coupling_rows") or []),
        "constraints": constraints,
        "constraint_application": constraint_application,
        "observation_model": observation_model,
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
    }


def _selective_diagnosis_flow_targets(
    *,
    dataset: Any,
    monthly_context: MonthlyShockContext,
    base_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    nowcast_head = _fit_monthly_reporting_nowcast_head(
        family=SELECTIVE_DIAGNOSIS_REPAIR_FAMILY,
        dataset=dataset,
        monthly_context=monthly_context,
    )
    if nowcast_head is None:
        return {}, {"status": "failed_closed", "reason": "monthly_nowcast_head_not_estimable"}
    selected_rows, backlog_head, updated_nowcast_head, nowcast_diagnostics, gate_diagnostics = apply_selective_repair_gate(
        repair_family=SELECTIVE_DIAGNOSIS_REPAIR_FAMILY,
        dataset=dataset,
        monthly_context=monthly_context,
        base_rows=base_rows,
        monthly_nowcast_head=nowcast_head,
        update_nowcast_incidence=False,
    )
    if backlog_head is None or updated_nowcast_head is None:
        return {}, {"status": "failed_closed", "reason": "selective_repair_gate_not_estimable"}
    source_by_quarter = {
        str(row.get("quarter") or ""): str(row.get("selected_source") or "")
        for row in gate_diagnostics
    }
    targets = {
        str(row.get("quarter") or ""): max(float(row.get("new_diagnosed_cases_period") or 0.0), 0.0)
        for row in selected_rows
        if str(row.get("quarter") or "")
        and source_by_quarter.get(str(row.get("quarter") or "")) in {"monthly_reporting_nowcast", "backlog_late_emission"}
    }
    target_sources = [
        source_by_quarter[quarter]
        for quarter in targets
        if source_by_quarter.get(quarter) in {"monthly_reporting_nowcast", "backlog_late_emission"}
    ]
    return targets, {
        "status": "completed",
        "contract": (
            "selective diagnosis-flow repair provides U_to_D flow targets only; "
            "update_nowcast_incidence=False prevents diagnosis-flow-derived incidence injection"
        ),
        "monthly_reporting_nowcast_head": {
            "train_quarter_count": int(updated_nowcast_head.train_quarter_count),
            "direct_monthly_row_count": int(updated_nowcast_head.direct_monthly_row_count),
            "train_partial_quarter_count": int(updated_nowcast_head.train_partial_quarter_count),
            "holdout_nowcast_quarter_count": int(updated_nowcast_head.holdout_nowcast_quarter_count),
            "training_objective": updated_nowcast_head.training_objective,
        },
        "backlog_late_emission_head": {
            "train_month_count": int(backlog_head.train_month_count),
            "late_share_emission_month_count": int(backlog_head.late_share_emission_month_count),
            "advanced_count_emission_month_count": int(backlog_head.advanced_count_emission_month_count),
            "train_loss": float(backlog_head.train_loss),
            "training_objective": backlog_head.training_objective,
        },
        "nowcast_diagnostics": nowcast_diagnostics,
        "selective_gate_diagnostics": gate_diagnostics,
        "u_to_d_target_count": len(targets),
        "u_to_d_target_source_counts": {
            source: sum(1 for value in target_sources if value == source)
            for source in sorted(set(target_sources))
        },
        "selective_gate_source_counts": {
            source: sum(1 for value in source_by_quarter.values() if value == source)
            for source in sorted(set(source_by_quarter.values()))
        },
    }


def _simulate_sequence_with_u_to_d_targets(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    hazard_map: dict[str, dict[str, float]],
    *,
    u_to_d_flow_targets: dict[str, float],
    u_to_d_bounds: dict[str, Any],
    eps: float,
    incidence_inflow_map: dict[str, float] | None = None,
    incidence_cap_map: dict[str, float] | None = None,
    population_denominator_map: dict[str, float] | None = None,
    attrition_outflow_map: dict[str, float] | None = None,
    state_attrition_outflow_map: dict[str, dict[str, float]] | None = None,
    exit_channel_state_outflow_map: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    previous_state = canonical_state_values(initial_state)
    incidence_inflow_map = dict(incidence_inflow_map or {})
    incidence_cap_map = dict(incidence_cap_map or {})
    population_denominator_map = dict(population_denominator_map or {})
    attrition_outflow_map = dict(attrition_outflow_map or {})
    state_attrition_outflow_map = dict(state_attrition_outflow_map or {})
    exit_channel_state_outflow_map = dict(exit_channel_state_outflow_map or {})
    floor = max(float(u_to_d_bounds.get("floor") or 0.0), 0.0)
    ceiling = max(float(u_to_d_bounds.get("ceiling") or 0.0), 0.0)
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    for row in sorted(list(target_rows), key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        hazards = dict(hazard_map.get(quarter) or {})
        target_flow = _finite_float(u_to_d_flow_targets.get(quarter))
        coupling = None
        if target_flow is not None and ceiling > 0.0:
            coupling = _bounded_u_to_d_hazard(
                target_diagnosis_flow=float(target_flow),
                undiagnosed_state=float(previous_state["U"]),
                floor=floor,
                ceiling=ceiling,
                eps=eps,
            )
            hazards["U_to_D"] = float(coupling["bounded_hazard"])
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        d_to_a = float(hazards.get("D_to_A") or 0.0) * max(float(previous_state["D"]), 0.0)
        a_to_t = float(hazards.get("A_to_T") or 0.0) * max(float(previous_state["A"]), 0.0)
        t_to_v = float(hazards.get("T_to_V") or 0.0) * max(float(previous_state["T"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        t_to_l = float(hazards.get("T_to_L") or 0.0) * max(float(previous_state["T"]), 0.0)
        v_to_l = float(hazards.get("V_to_L") or 0.0) * max(float(previous_state["V"]), 0.0)
        l_to_r = float(hazards.get("L_to_R") or 0.0) * max(float(previous_state["L"]), 0.0)
        r_to_a = float(hazards.get("R_to_A") or 0.0) * max(float(previous_state["R"]), 0.0)
        pre_balance_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a + r_to_a - a_to_t - a_to_l, 0.0),
            "T": max(float(previous_state["T"]) + a_to_t - t_to_v - t_to_l, 0.0),
            "V": max(float(previous_state["V"]) + t_to_v - v_to_l, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l + t_to_l + v_to_l - l_to_r, 0.0),
            "R": max(float(previous_state["R"]) + l_to_r - r_to_a, 0.0),
        }
        stock_balance = apply_stock_balance(
            pre_balance_state,
            incidence_inflow=float(incidence_inflow_map.get(quarter) or 0.0),
            incidence_cap=None if quarter not in incidence_cap_map else float(incidence_cap_map.get(quarter) or 0.0),
            population_denominator=None if quarter not in population_denominator_map else float(population_denominator_map.get(quarter) or 0.0),
            attrition_outflow=float(attrition_outflow_map.get(quarter) or 0.0),
            state_attrition_outflow=dict(state_attrition_outflow_map.get(quarter) or {}),
            exit_channel_state_outflows=dict(exit_channel_state_outflow_map.get(quarter) or {}),
        )
        current_state = {
            name: float((stock_balance.get("state_values") or {}).get(name) or 0.0)
            for name in STATE_NAMES
        }
        prediction = {
            "quarter": quarter,
            "diagnosed_plhiv": state_sum(current_state, DIAGNOSED_STATE_NAMES),
            "alive_on_art": state_sum(current_state, ART_STATE_NAMES),
            "new_diagnosed_cases_period": float(u_to_d),
            "tested_for_viral_load": state_sum(current_state, VL_TESTED_STATE_NAMES),
            "virally_suppressed": float(current_state["V"]),
            "incident_infections_period": float((stock_balance.get("stock_balance") or {}).get("incidence_inflow") or 0.0),
            "net_attrition_outflow_period": float((stock_balance.get("stock_balance") or {}).get("attrition_outflow") or 0.0),
        }
        prediction_rows.append(prediction)
        trajectory_rows.append(
            {
                "quarter": quarter,
                "hazards": {name: float(hazards.get(name) or 0.0) for name in TRANSITION_NAMES},
                "state_values": {name: float(current_state[name]) for name in STATE_NAMES},
                "stock_balance": dict(stock_balance.get("stock_balance") or {}),
            }
        )
        if coupling is not None:
            coupling_rows.append(
                {
                    "quarter": quarter,
                    "target_diagnosis_flow": float(target_flow),
                    "previous_undiagnosed_state": float(previous_state["U"]),
                    "raw_u_to_d_hazard": float(coupling["raw_hazard"]),
                    "bounded_u_to_d_hazard": float(coupling["bounded_hazard"]),
                    "implied_u_to_d_flow": float(u_to_d),
                    "train_hazard_floor": floor,
                    "train_hazard_ceiling": ceiling,
                    "contract": "U_to_D target is bounded by empirical train-window U_to_D hazards",
                }
            )
        previous_state = current_state
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": trajectory_rows,
        "u_to_d_coupling_rows": coupling_rows,
    }


def _fit_coupled_candidate(
    *,
    dataset: Any,
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    validation_targets: dict[int, dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext,
    train_end_year: int,
) -> dict[str, Any]:
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths, incidence_result = _latent_incidence_paths(
        dataset=dataset,
        reference_config=reference_config,
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        validation_targets=validation_targets,
        train_end_year=int(train_end_year),
    )
    split_constraints = _filter_rows_before_holdout(constraint_rows, list(dataset.holdout_years))
    base = _simulate_base_path(
        dataset=dataset,
        hazard_paths=hazard_paths,
        incidence_paths=incidence_paths,
        reference_config=reference_config,
        constraint_rows=split_constraints,
    )
    flow_targets, repair_diagnostics = _selective_diagnosis_flow_targets(
        dataset=dataset,
        monthly_context=monthly_context,
        base_rows=list(base.get("raw_prediction_rows") or []),
    )
    if str(repair_diagnostics.get("status") or "") != "completed":
        return {
            "status": "failed_closed",
            "reason": str(repair_diagnostics.get("reason") or "selective_diagnosis_repair_failed"),
            "incidence_result": incidence_result,
            "repair_diagnostics": repair_diagnostics,
        }
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    constraints = _build_projection_constraints(
        dataset=dataset,
        constraint_rows=split_constraints,
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
    raw = _simulate_sequence_with_u_to_d_targets(
        initial_state=dict(dataset.train_state_rows[-1]["state_values"]),
        target_rows=holdout_rows,
        hazard_map=dict(hazard_paths.get("holdout_hazard_map") or {}),
        u_to_d_flow_targets=flow_targets,
        u_to_d_bounds=_hazard_bounds(dataset, "U_to_D"),
        eps=float(dataset.eps),
        incidence_inflow_map=incidence_map,
        incidence_cap_map={quarter: float(constraints.get("incidence_cap_per_quarter") or 0.0) for quarter in holdout_quarters},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=attrition_map,
        state_attrition_outflow_map=state_attrition_map,
        exit_channel_state_outflow_map=dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {}),
    )
    observation_model = fit_observation_model(
        dataset,
        dict(hazard_paths.get("train_hazard_map") or {}),
        reference_config["observation_cfg"],
        incidence_paths=incidence_paths,
        damping_cfg=reference_config["damping_cfg"],
    )
    calibrated = _apply_observation_model(
        list(raw.get("prediction_rows") or []),
        observation_model,
        positions=_quarter_positions(dataset),
        damping_cfg=reference_config["damping_cfg"],
    )
    raw_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in list(raw.get("prediction_rows") or [])
    }
    prediction_rows: list[dict[str, Any]] = []
    for row in sorted(calibrated, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        next_row = dict(row)
        quarter = str(next_row.get("quarter") or "")
        # Keep the diagnosis-flow readout equal to the state transition that actually moved U -> D.
        if quarter in raw_by_quarter:
            next_row["new_diagnosed_cases_period"] = float(raw_by_quarter[quarter].get("new_diagnosed_cases_period") or 0.0)
        prediction_rows.append(next_row)
    return {
        "status": "completed",
        "prediction_rows": prediction_rows,
        "raw_prediction_rows": list(raw.get("prediction_rows") or []),
        "trajectory_rows": list(raw.get("trajectory_rows") or []),
        "u_to_d_coupling_rows": list(raw.get("u_to_d_coupling_rows") or []),
        "base_prediction_rows": list(base.get("prediction_rows") or []),
        "base_raw_prediction_rows": list(base.get("raw_prediction_rows") or []),
        "base_trajectory_rows": list(base.get("trajectory_rows") or []),
        "flow_targets": flow_targets,
        "repair_diagnostics": repair_diagnostics,
        "incidence_result": incidence_result,
        "incidence_paths_contract": {
            "state_incidence_contract": incidence_paths.get("state_incidence_contract"),
            "annual_measurement_readout_excluded": bool(incidence_paths.get("excluded_measurement_readout_map")),
        },
        "constraint_application": constraint_application,
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
    }


def _carry_forward_result(dataset: Any, reference_config: dict[str, Any]) -> dict[str, Any]:
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
    return simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
        incidence_hazard_map={},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {}),
    )


def _evaluate_split(
    *,
    split: dict[str, Any],
    observation_rows: list[dict[str, Any]],
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    validation_targets: dict[int, dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext,
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    candidate = _fit_coupled_candidate(
        dataset=dataset,
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        validation_targets=validation_targets,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        monthly_context=monthly_context,
        train_end_year=int(split["train_end_year"]),
    )
    if str(candidate.get("status") or "") != "completed":
        return {
            "family": "u_to_d_selective_state_readout_coupling",
            "train_end_year": int(split["train_end_year"]),
            "train_years": list(split["train_years"]),
            "holdout_years": list(split["holdout_years"]),
            "candidate_mae": float("inf"),
            "carry_forward_mae": float("inf"),
            "fit_status": str(candidate.get("status") or "failed_closed"),
            "fit_blocker": str(candidate.get("reason") or "coupled_candidate_failed"),
            "prediction_rows": [],
            "carry_forward_prediction_rows": [],
            "holdout_rows": list(dataset.holdout_rows),
            "metric_scales": dict(dataset.metric_scales),
        }
    carry = _carry_forward_result(dataset, reference_config)
    return {
        "family": "u_to_d_selective_state_readout_coupling",
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "candidate_mae": float(candidate["mae"]),
        "carry_forward_mae": float(carry["mae"]),
        "candidate_minus_carry_forward_mae": float(candidate["mae"]) - float(carry["mae"]),
        "fit_status": "completed",
        "u_to_d_coupling_rows": list(candidate.get("u_to_d_coupling_rows") or []),
        "repair_diagnostics_summary": {
            "status": (candidate.get("repair_diagnostics") or {}).get("status"),
            "monthly_nowcast_head": (candidate.get("repair_diagnostics") or {}).get("monthly_reporting_nowcast_head"),
            "backlog_late_emission_head": (candidate.get("repair_diagnostics") or {}).get("backlog_late_emission_head"),
            "u_to_d_target_count": (candidate.get("repair_diagnostics") or {}).get("u_to_d_target_count"),
            "u_to_d_target_source_counts": (candidate.get("repair_diagnostics") or {}).get("u_to_d_target_source_counts"),
            "selective_gate_source_counts": (candidate.get("repair_diagnostics") or {}).get("selective_gate_source_counts"),
        },
        "incidence_paths_contract": dict(candidate.get("incidence_paths_contract") or {}),
        "constraint_application": dict(candidate.get("constraint_application") or {}),
        "prediction_rows": list(candidate.get("prediction_rows") or []),
        "carry_forward_prediction_rows": list(carry.get("prediction_rows") or []),
        "holdout_rows": list(dataset.holdout_rows),
        "metric_scales": dict(dataset.metric_scales),
    }


def _one_year_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = _score_rows(rows)
    blockers: list[str] = []
    if int(score.get("split_count") or 0) <= 0:
        blockers.append("no_scored_one_year_splits")
    candidate_mean = _finite_float(score.get("candidate_mean_mae"))
    carry_mean = _finite_float(score.get("carry_forward_mean_mae"))
    candidate_worst = _finite_float(score.get("candidate_worst_mae"))
    carry_worst = _finite_float(score.get("carry_forward_worst_mae"))
    if candidate_mean is None or carry_mean is None or candidate_mean >= carry_mean:
        blockers.append("candidate_one_year_mean_not_better_than_carry_forward")
    if candidate_worst is None or carry_worst is None or candidate_worst > carry_worst:
        blockers.append("candidate_one_year_worst_worse_than_carry_forward")
    return {
        "schema_version": "phase3_dynamic_u_to_d_one_year_gate.v1",
        "status": "pass" if not blockers else "fail",
        **score,
        "blockers": blockers,
        "contract": "one-year blocked-time full-cascade score after bounded U_to_D state/readout coupling",
    }


def _promotion_gate(*, one_year_gate: dict[str, Any], shock_gate: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if str(one_year_gate.get("status") or "") != "pass":
        blockers.append("fails_one_year_state_readout_coupling_gate")
        blockers.extend(str(value) for value in list(one_year_gate.get("blockers") or []))
    if str(shock_gate.get("status") or "") != "pass":
        blockers.append("fails_shock_aware_lifted_trajectory_gate")
        blockers.extend(str(value) for value in list(shock_gate.get("blockers") or []))
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "promote_short_horizon_coupling_claim" if not blockers else "reject_short_horizon_coupling_claim",
        "promotion_eligible": bool(not blockers),
        "blockers": blockers,
        "claim_boundary": (
            "Promotion would support a short-horizon diagnosis-state coupling claim only. "
            "It would not validate annual-incidence-derived quarterly truth or long-horizon scenario use."
        ),
    }


def _metric_anatomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period", "virally_suppressed")
    entries: list[dict[str, Any]] = []
    for split in rows:
        prediction_by_quarter = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("prediction_rows") or [])
        }
        carry_by_quarter = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("carry_forward_prediction_rows") or [])
        }
        scales = dict(split.get("metric_scales") or {})
        for target in list(split.get("holdout_rows") or []):
            quarter = str(target.get("quarter") or "")
            prediction = prediction_by_quarter.get(quarter)
            carry = carry_by_quarter.get(quarter)
            if prediction is None or carry is None:
                continue
            for metric_name in metrics:
                target_value = _finite_float(target.get(metric_name))
                candidate_value = _finite_float(prediction.get(metric_name))
                carry_value = _finite_float(carry.get(metric_name))
                if target_value is None or candidate_value is None or carry_value is None:
                    continue
                scale = _finite_float(scales.get(metric_name))
                if scale is None or scale <= 0.0:
                    scale = max(abs(float(target_value)), 1.0)
                candidate_error = abs(float(candidate_value) - float(target_value)) / max(float(scale), 1.0)
                carry_error = abs(float(carry_value) - float(target_value)) / max(float(scale), 1.0)
                entries.append(
                    {
                        "metric_name": metric_name,
                        "quarter": quarter,
                        "train_end_year": int(split["train_end_year"]),
                        "candidate_norm_error": float(candidate_error),
                        "carry_forward_norm_error": float(carry_error),
                        "candidate_minus_carry_forward_norm_error": float(candidate_error - carry_error),
                    }
                )
    summaries: list[dict[str, Any]] = []
    for metric_name in metrics:
        metric_entries = [row for row in entries if row["metric_name"] == metric_name]
        if not metric_entries:
            continue
        candidate = np.asarray([float(row["candidate_norm_error"]) for row in metric_entries], dtype=np.float64)
        carry = np.asarray([float(row["carry_forward_norm_error"]) for row in metric_entries], dtype=np.float64)
        summaries.append(
            {
                "metric_name": metric_name,
                "entry_count": len(metric_entries),
                "candidate_mean_norm_error": float(np.mean(candidate)),
                "carry_forward_mean_norm_error": float(np.mean(carry)),
                "candidate_minus_carry_forward_mean_norm_error": float(np.mean(candidate) - np.mean(carry)),
                "worst_candidate_minus_carry_forward_norm_error": float(np.max(candidate - carry)),
            }
        )
    return sorted(summaries, key=lambda row: float(row["candidate_minus_carry_forward_mean_norm_error"]), reverse=True)


def _repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [
        row
        for row in rows
        if str((row.get("repair_diagnostics_summary") or {}).get("status") or "") == "completed"
    ]
    target_source_counts: dict[str, int] = {}
    gate_source_counts: dict[str, int] = {}
    target_count = 0
    split_summaries: list[dict[str, Any]] = []
    for row in completed_rows:
        summary = dict(row.get("repair_diagnostics_summary") or {})
        target_count += int(summary.get("u_to_d_target_count") or 0)
        for source, count in dict(summary.get("u_to_d_target_source_counts") or {}).items():
            target_source_counts[str(source)] = target_source_counts.get(str(source), 0) + int(count or 0)
        for source, count in dict(summary.get("selective_gate_source_counts") or {}).items():
            gate_source_counts[str(source)] = gate_source_counts.get(str(source), 0) + int(count or 0)
        split_summaries.append(
            {
                "train_end_year": int(row.get("train_end_year") or 0),
                "holdout_years": list(row.get("holdout_years") or []),
                "u_to_d_target_count": int(summary.get("u_to_d_target_count") or 0),
                "u_to_d_target_source_counts": dict(summary.get("u_to_d_target_source_counts") or {}),
                "selective_gate_source_counts": dict(summary.get("selective_gate_source_counts") or {}),
            }
        )
    return {
        "status": "completed" if completed_rows else "no_completed_repair_splits",
        "completed_split_count": len(completed_rows),
        "total_u_to_d_target_count": int(target_count),
        "u_to_d_target_source_counts": dict(sorted(target_source_counts.items())),
        "selective_gate_source_counts": dict(sorted(gate_source_counts.items())),
        "split_summaries": split_summaries,
        "contract": (
            "u_to_d_target_source_counts includes only selected repair rows that were converted into bounded "
            "U_to_D transition targets; selective_gate_source_counts records all sources considered by the gate."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    stored_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "prediction_rows",
                "carry_forward_prediction_rows",
                "holdout_rows",
                "metric_scales",
            }
        }
        for row in rows
    ]
    if not stored_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in stored_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stored_rows:
            writer.writerow(row)


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    gate = dict(payload.get("one_year_gate") or {})
    shock = dict(payload.get("shock_aware_lifted_trajectory_gate") or {})
    promotion = dict(payload.get("promotion_gate") or {})
    anatomy = list(payload.get("metric_anatomy") or [])
    rows = list(payload.get("rows") or [])
    coupling_rows = [item for row in rows for item in list(row.get("u_to_d_coupling_rows") or [])]
    labels = [str(row.get("train_end_year") or "") for row in rows]
    deltas = [float(row.get("candidate_minus_carry_forward_mae") or 0.0) for row in rows]
    raw_h = [float(row.get("raw_u_to_d_hazard") or 0.0) for row in coupling_rows]
    bounded_h = [float(row.get("bounded_u_to_d_hazard") or 0.0) for row in coupling_rows]
    shock_overall = dict(shock.get("overall") or {})
    colors = {"candidate": "#9f1239", "carry": "#0f766e", "r10": "#111827"}
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "figure.titlesize": 13,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6), constrained_layout=True)
        flat = axes.ravel()
        x = np.arange(2)
        flat[0].bar(
            x,
            [float(gate.get("candidate_mean_mae") or 0.0), float(gate.get("carry_forward_mean_mae") or 0.0)],
            color=[colors["candidate"], colors["carry"]],
        )
        flat[0].set_xticks(x, ["candidate", "carry-forward"])
        flat[0].set_ylabel("Norm. MAE")
        flat[0].set_title("A. One-Year State/Readout Coupling Gate", loc="left", fontweight="bold")
        if labels:
            sx = np.arange(len(labels))
            flat[1].bar(sx, deltas, color=["#0f766e" if value < 0.0 else "#9f1239" for value in deltas])
            flat[1].axhline(0.0, color="#111827", linewidth=0.9)
            flat[1].set_xticks(sx, labels, rotation=35, ha="right")
        flat[1].set_ylabel("Candidate minus carry-forward")
        flat[1].set_title("B. Split-Level Delta", loc="left", fontweight="bold")
        if coupling_rows:
            hx = np.arange(len(coupling_rows))
            flat[2].plot(hx, raw_h, color="#64748b", linewidth=1.0, label="raw target/U")
            flat[2].plot(hx, bounded_h, color=colors["candidate"], linewidth=1.4, label="bounded U_to_D")
            flat[2].legend(frameon=False)
        flat[2].set_ylabel("Hazard")
        flat[2].set_title("C. U_to_D Coupling Bounds", loc="left", fontweight="bold")
        text = [
            "Short-horizon coupling interpretation",
            f"One-year gate: {gate.get('status')}",
            f"Shock path: {shock.get('status')}",
            f"Promotion: {promotion.get('status')}",
            "",
            "Lifted path mean:",
            f"candidate {shock_overall.get('candidate_mean_mae')}",
            f"carry-forward {shock_overall.get('carry_forward_mean_mae')}",
            f"R10 {(shock.get('r10_reference') or {}).get('reference_quarterly_mean_mae')}",
            "",
            "Top blockers:",
        ]
        text.extend([f"- {value}" for value in list(promotion.get("blockers") or [])[:7]] or ["- none"])
        text.extend(["", "Metric anatomy:"])
        for row in anatomy[:4]:
            text.append(
                f"- {row.get('metric_name')}: "
                f"{float(row.get('candidate_minus_carry_forward_mean_norm_error') or 0.0):+.3f}"
            )
        flat[3].axis("off")
        flat[3].text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=8.3)
        for ax in flat[:3]:
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.suptitle("Phase 3 U_to_D State/Readout Coupling Gate", fontweight="bold")
        fig.savefig(path, dpi=330, bbox_inches="tight")
        plt.close(fig)


def run_u_to_d_coupling(
    *,
    run_id: str = "p3d-u-to-d-coupling-20260429-s00",
    source_report_path: str | Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    family: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> dict[str, Any]:
    source_report = Path(source_report_path) if source_report_path is not None else _default_source_report()
    source_payload = dict(read_json(source_report, default={}) or {})
    epigraph_root = default_epigraph_root()
    active_source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(source_run_id or source_payload.get("source_run_id") or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    active_baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=active_source_run_id,
        preferred=str(baseline_source_run_id or source_payload.get("baseline_source_run_id") or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    base_family = _selected_family(source_payload, family)
    reference_config = _load_reference_config(
        Path(str(source_payload.get("reference_report_path") or ""))
        if source_payload.get("reference_report_path")
        else None
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
    validation_targets = _annual_incidence_targets(validation_rows)
    monthly_context = MonthlyShockContext(
        epigraph_root=epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
    )
    rows: list[dict[str, Any]] = []
    splits = rolling_origin_splits(
        observation_rows,
        start_year=int(start_year),
        end_year=int(end_year),
        min_train_years=int(min_train_years),
        horizon_years=1,
    )
    for split in splits:
        result = _evaluate_split(
            split=split,
            observation_rows=observation_rows,
            epigraph_root=epigraph_root,
            source_run_id=active_source_run_id,
            baseline_source_run_id=active_baseline_source_run_id,
            validation_targets=validation_targets,
            reference_config=reference_config,
            constraint_rows=validation_rows,
            monthly_context=monthly_context,
        )
        if result is not None:
            rows.append(result)
    r10_baseline = _load_r10_baseline(epigraph_root)
    one_year_gate = _one_year_gate(rows)
    shock_gate = _shock_aware_lifted_gate_from_evaluated(
        family="u_to_d_selective_state_readout_coupling",
        evaluated_rows=rows,
        shock_catalog=_detect_shock_catalog(observation_rows),
        r10_baseline=r10_baseline,
    )
    promotion = _promotion_gate(one_year_gate=one_year_gate, shock_gate=shock_gate)
    metric_anatomy = _metric_anatomy(rows)
    repair_summary = _repair_summary(rows)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "u_to_d_coupling_report.json"
    rows_csv_path = analysis_dir / "u_to_d_coupling_rows.csv"
    dashboard_path = analysis_dir / "u_to_d_coupling_dashboard.png"
    payload = {
        "schema_version": U_TO_D_COUPLING_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": active_source_run_id,
        "baseline_source_run_id": active_baseline_source_run_id,
        "source_report_path": source_report.as_posix(),
        "base_family": base_family,
        "family": "u_to_d_selective_state_readout_coupling",
        "candidate_family": "u_to_d_selective_state_readout_coupling",
        "loop_variant": "evidence-to-model-loop",
        "contract": {
            "state_coupling": "selective diagnosis-flow repair is converted to a bounded U_to_D hazard and therefore moves mass from U to diagnosed states",
            "readout_coupling": "new_diagnosed_cases_period after observation calibration is reset to the realized U_to_D state transition",
            "incidence_guard": "annual_new_infections is validation-only; diagnosis-flow-derived incidence readout is disabled; latent S_eff incidence enters U separately",
            "bound_rule": "U_to_D hazard is clipped to empirical train-window U_to_D hazard support for each split",
        },
        "benchmark_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": 1,
            "split_count": len(rows),
        },
        "observation_role_ledger_summary": dict(
            build_observation_role_ledger(
                epigraph_root,
                source_run_id=active_source_run_id,
                baseline_source_run_id=active_baseline_source_run_id,
            ).get("summary")
            or {}
        ),
        "r10_reference": _r10_reference_scores(r10_baseline),
        "one_year_gate": one_year_gate,
        "shock_aware_lifted_trajectory_gate": shock_gate,
        "metric_anatomy": metric_anatomy,
        "repair_diagnostics_summary": repair_summary,
        "promotion_gate": promotion,
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "prediction_rows",
                    "carry_forward_prediction_rows",
                    "holdout_rows",
                    "metric_scales",
                }
            }
            for row in rows
        ],
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "rows_csv": rows_csv_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_csv(rows_csv_path, rows)
    _write_dashboard(payload, dashboard_path)
    write_json(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase3 U_to_D state/readout coupling gate.")
    parser.add_argument("--run-id", default="p3d-u-to-d-coupling-20260429-s00")
    parser.add_argument("--source-report-path")
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--family")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    payload = run_u_to_d_coupling(
        run_id=args.run_id,
        source_report_path=args.source_report_path,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        family=args.family,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
