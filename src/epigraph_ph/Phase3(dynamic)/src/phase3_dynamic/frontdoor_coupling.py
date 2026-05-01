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
from .diagnosis_incidence_repair import _default_source_report
from .hybrid_champion import (
    _detect_shock_catalog,
    _load_r10_baseline,
    _load_reference_config,
    _r10_reference_scores,
    _shock_aware_lifted_gate_from_evaluated,
)
from .incidence import apply_stock_balance
from .incidence_module_search import _annual_incidence_targets
from .metrics import normalized_mae, quarter_sort_key
from .model import (
    _apply_observation_model,
    _quarter_positions,
    fit_dynamic_hazard_paths,
    fit_observation_model,
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
from .u_to_d_coupling import (
    _carry_forward_result,
    _finite_float,
    _hazard_bounds,
    _latent_incidence_paths,
    _metric_anatomy,
    _repair_summary,
    _score_rows,
    _selective_diagnosis_flow_targets,
)


FRONTDOOR_COUPLING_SCHEMA_VERSION = "phase3_dynamic_u_to_d_d_to_a_state_readout_coupling.v1"
FRONTDOOR_FAMILY = "u_to_d_d_to_a_selective_state_readout_coupling"


def _bounded_flow_hazard(
    *,
    target_flow: float,
    denominator: float,
    floor: float,
    ceiling: float,
    eps: float,
) -> dict[str, float]:
    if denominator <= eps:
        raw = 0.0
    else:
        raw = max(float(target_flow), 0.0) / max(float(denominator), float(eps))
    bounded = float(np.clip(raw, max(float(floor), 0.0), max(float(ceiling), 0.0)))
    return {
        "raw_hazard": float(raw),
        "bounded_hazard": bounded,
        "implied_flow": float(bounded * max(float(denominator), 0.0)),
    }


def _d_to_a_available_pool_bounds(dataset: Any) -> dict[str, Any]:
    values: list[float] = []
    for row in list(getattr(dataset, "train_transition_rows", []) or []):
        flows = dict(row.get("flows") or {})
        previous_state = dict(row.get("state_values_previous") or {})
        u_to_d = _finite_float(flows.get("U_to_D"))
        d_to_a = _finite_float(flows.get("D_to_A"))
        previous_d = _finite_float(previous_state.get("D"))
        if u_to_d is None or d_to_a is None or previous_d is None:
            continue
        denominator = max(float(previous_d), 0.0) + max(float(u_to_d), 0.0)
        if denominator <= float(getattr(dataset, "eps", np.finfo(np.float32).eps)):
            continue
        values.append(max(float(d_to_a), 0.0) / denominator)
    if not values:
        return {
            "floor": 0.0,
            "ceiling": 0.0,
            "train_count": 0,
            "contract": "no train D_to_A available-pool hazards available; branch fails closed upstream",
        }
    return {
        "floor": float(min(values)),
        "ceiling": float(max(values)),
        "train_count": len(values),
        "contract": (
            "empirical train-window D_to_A hazard using available diagnosed pool "
            "D_prev + U_to_D; no hand-tuned linkage bound"
        ),
    }


def _linkage_response_summary(dataset: Any) -> dict[str, Any]:
    ratios: list[float] = []
    for row in list(getattr(dataset, "train_transition_rows", []) or []):
        flows = dict(row.get("flows") or {})
        u_to_d = _finite_float(flows.get("U_to_D"))
        d_to_a = _finite_float(flows.get("D_to_A"))
        if u_to_d is None or d_to_a is None or float(u_to_d) <= float(getattr(dataset, "eps", np.finfo(np.float32).eps)):
            continue
        ratios.append(max(float(d_to_a), 0.0) / max(float(u_to_d), float(getattr(dataset, "eps", np.finfo(np.float32).eps))))
    if not ratios:
        return {
            "status": "not_estimable",
            "linkage_flow_ratio": None,
            "train_count": 0,
            "contract": "no positive train U_to_D flow for empirical D_to_A response",
        }
    array = np.asarray(ratios, dtype=np.float64)
    return {
        "status": "completed",
        "linkage_flow_ratio": float(np.median(array)),
        "train_count": int(array.size),
        "min_linkage_flow_ratio": float(np.min(array)),
        "max_linkage_flow_ratio": float(np.max(array)),
        "contract": "median train-window D_to_A / U_to_D flow ratio; no hand-tuned linkage multiplier",
    }


def _simulate_sequence_with_frontdoor_targets(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    hazard_map: dict[str, dict[str, float]],
    *,
    u_to_d_flow_targets: dict[str, float],
    u_to_d_bounds: dict[str, Any],
    d_to_a_available_bounds: dict[str, Any],
    linkage_response: dict[str, Any],
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
    u_floor = max(float(u_to_d_bounds.get("floor") or 0.0), 0.0)
    u_ceiling = max(float(u_to_d_bounds.get("ceiling") or 0.0), 0.0)
    d_floor = max(float(d_to_a_available_bounds.get("floor") or 0.0), 0.0)
    d_ceiling = max(float(d_to_a_available_bounds.get("ceiling") or 0.0), 0.0)
    linkage_ratio = _finite_float(linkage_response.get("linkage_flow_ratio"))
    linkage_ratio = 0.0 if linkage_ratio is None else max(float(linkage_ratio), 0.0)
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    for row in sorted(list(target_rows), key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        hazards = dict(hazard_map.get(quarter) or {})
        target_flow = _finite_float(u_to_d_flow_targets.get(quarter))
        baseline_u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        baseline_d_to_a = float(hazards.get("D_to_A") or 0.0) * max(float(previous_state["D"]), 0.0)
        u_coupling = None
        d_coupling = None
        if target_flow is not None and u_ceiling > 0.0:
            u_coupling = _bounded_flow_hazard(
                target_flow=float(target_flow),
                denominator=float(previous_state["U"]),
                floor=u_floor,
                ceiling=u_ceiling,
                eps=eps,
            )
            hazards["U_to_D"] = float(u_coupling["bounded_hazard"])
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        available_diagnosed_pool = max(float(previous_state["D"]) + float(u_to_d), 0.0)
        if target_flow is not None and d_ceiling > 0.0:
            diagnosis_pressure_delta = max(float(u_to_d) - float(baseline_u_to_d), 0.0)
            target_d_to_a = max(float(baseline_d_to_a), 0.0) + diagnosis_pressure_delta * linkage_ratio
            d_coupling = _bounded_flow_hazard(
                target_flow=target_d_to_a,
                denominator=available_diagnosed_pool,
                floor=d_floor,
                ceiling=d_ceiling,
                eps=eps,
            )
            d_to_a = min(float(d_coupling["implied_flow"]), available_diagnosed_pool)
            hazards["D_to_A"] = float(d_coupling["bounded_hazard"])
        else:
            diagnosis_pressure_delta = 0.0
            target_d_to_a = baseline_d_to_a
            d_to_a = min(max(float(baseline_d_to_a), 0.0), available_diagnosed_pool)
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
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": state_sum(current_state, DIAGNOSED_STATE_NAMES),
                "alive_on_art": state_sum(current_state, ART_STATE_NAMES),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": state_sum(current_state, VL_TESTED_STATE_NAMES),
                "virally_suppressed": float(current_state["V"]),
                "incident_infections_period": float((stock_balance.get("stock_balance") or {}).get("incidence_inflow") or 0.0),
                "net_attrition_outflow_period": float((stock_balance.get("stock_balance") or {}).get("attrition_outflow") or 0.0),
            }
        )
        trajectory_rows.append(
            {
                "quarter": quarter,
                "hazards": {name: float(hazards.get(name) or 0.0) for name in TRANSITION_NAMES},
                "state_values": {name: float(current_state[name]) for name in STATE_NAMES},
                "stock_balance": dict(stock_balance.get("stock_balance") or {}),
            }
        )
        if u_coupling is not None or d_coupling is not None:
            coupling_rows.append(
                {
                    "quarter": quarter,
                    "target_diagnosis_flow": None if target_flow is None else float(target_flow),
                    "baseline_u_to_d_flow": float(baseline_u_to_d),
                    "realized_u_to_d_flow": float(u_to_d),
                    "diagnosis_pressure_delta": float(diagnosis_pressure_delta),
                    "linkage_flow_ratio": float(linkage_ratio),
                    "baseline_d_to_a_flow": float(baseline_d_to_a),
                    "target_d_to_a_flow": float(target_d_to_a),
                    "realized_d_to_a_flow": float(d_to_a),
                    "previous_undiagnosed_state": float(previous_state["U"]),
                    "previous_diagnosed_not_art_state": float(previous_state["D"]),
                    "available_diagnosed_pool": float(available_diagnosed_pool),
                    "raw_u_to_d_hazard": None if u_coupling is None else float(u_coupling["raw_hazard"]),
                    "bounded_u_to_d_hazard": None if u_coupling is None else float(u_coupling["bounded_hazard"]),
                    "raw_d_to_a_available_hazard": None if d_coupling is None else float(d_coupling["raw_hazard"]),
                    "bounded_d_to_a_available_hazard": None if d_coupling is None else float(d_coupling["bounded_hazard"]),
                    "u_to_d_train_hazard_floor": float(u_floor),
                    "u_to_d_train_hazard_ceiling": float(u_ceiling),
                    "d_to_a_available_hazard_floor": float(d_floor),
                    "d_to_a_available_hazard_ceiling": float(d_ceiling),
                    "contract": (
                        "U_to_D target is bounded by empirical train-window U_to_D hazard; "
                        "D_to_A response is bounded by empirical train-window available-pool D_to_A hazard"
                    ),
                }
            )
        previous_state = current_state
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": trajectory_rows,
        "frontdoor_coupling_rows": coupling_rows,
    }


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
    raw = _simulate_sequence_with_frontdoor_targets(
        initial_state=dict(dataset.train_state_rows[-1]["state_values"]),
        target_rows=holdout_rows,
        hazard_map=dict(hazard_paths.get("holdout_hazard_map") or {}),
        u_to_d_flow_targets={},
        u_to_d_bounds=_hazard_bounds(dataset, "U_to_D"),
        d_to_a_available_bounds=_d_to_a_available_pool_bounds(dataset),
        linkage_response=_linkage_response_summary(dataset),
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
        "frontdoor_coupling_rows": list(raw.get("frontdoor_coupling_rows") or []),
        "constraints": constraints,
        "constraint_application": constraint_application,
        "observation_model": observation_model,
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
    }


def _fit_frontdoor_candidate(
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
    u_to_d_bounds = _hazard_bounds(dataset, "U_to_D")
    d_to_a_bounds = _d_to_a_available_pool_bounds(dataset)
    linkage_response = _linkage_response_summary(dataset)
    if int(u_to_d_bounds.get("train_count") or 0) <= 0:
        return {"status": "failed_closed", "reason": "u_to_d_bounds_not_estimable", "incidence_result": incidence_result}
    if int(d_to_a_bounds.get("train_count") or 0) <= 0:
        return {"status": "failed_closed", "reason": "d_to_a_available_bounds_not_estimable", "incidence_result": incidence_result}
    if str(linkage_response.get("status") or "") != "completed":
        return {"status": "failed_closed", "reason": "linkage_response_not_estimable", "incidence_result": incidence_result}
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
    raw = _simulate_sequence_with_frontdoor_targets(
        initial_state=dict(dataset.train_state_rows[-1]["state_values"]),
        target_rows=holdout_rows,
        hazard_map=dict(hazard_paths.get("holdout_hazard_map") or {}),
        u_to_d_flow_targets=flow_targets,
        u_to_d_bounds=u_to_d_bounds,
        d_to_a_available_bounds=d_to_a_bounds,
        linkage_response=linkage_response,
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
        if quarter in raw_by_quarter:
            next_row["new_diagnosed_cases_period"] = float(raw_by_quarter[quarter].get("new_diagnosed_cases_period") or 0.0)
        prediction_rows.append(next_row)
    return {
        "status": "completed",
        "prediction_rows": prediction_rows,
        "raw_prediction_rows": list(raw.get("prediction_rows") or []),
        "trajectory_rows": list(raw.get("trajectory_rows") or []),
        "frontdoor_coupling_rows": list(raw.get("frontdoor_coupling_rows") or []),
        "base_prediction_rows": list(base.get("prediction_rows") or []),
        "base_raw_prediction_rows": list(base.get("raw_prediction_rows") or []),
        "base_trajectory_rows": list(base.get("trajectory_rows") or []),
        "flow_targets": flow_targets,
        "repair_diagnostics": repair_diagnostics,
        "linkage_diagnostics": {
            "u_to_d_bounds": u_to_d_bounds,
            "d_to_a_available_bounds": d_to_a_bounds,
            "linkage_response": linkage_response,
        },
        "incidence_result": incidence_result,
        "incidence_paths_contract": {
            "state_incidence_contract": incidence_paths.get("state_incidence_contract"),
            "annual_measurement_readout_excluded": bool(incidence_paths.get("excluded_measurement_readout_map")),
        },
        "constraint_application": constraint_application,
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
    }


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
    candidate = _fit_frontdoor_candidate(
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
            "family": FRONTDOOR_FAMILY,
            "train_end_year": int(split["train_end_year"]),
            "train_years": list(split["train_years"]),
            "holdout_years": list(split["holdout_years"]),
            "candidate_mae": float("inf"),
            "carry_forward_mae": float("inf"),
            "fit_status": str(candidate.get("status") or "failed_closed"),
            "fit_blocker": str(candidate.get("reason") or "frontdoor_candidate_failed"),
            "prediction_rows": [],
            "carry_forward_prediction_rows": [],
            "holdout_rows": list(dataset.holdout_rows),
            "metric_scales": dict(dataset.metric_scales),
        }
    carry = _carry_forward_result(dataset, reference_config)
    return {
        "family": FRONTDOOR_FAMILY,
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "candidate_mae": float(candidate["mae"]),
        "carry_forward_mae": float(carry["mae"]),
        "candidate_minus_carry_forward_mae": float(candidate["mae"]) - float(carry["mae"]),
        "fit_status": "completed",
        "frontdoor_coupling_rows": list(candidate.get("frontdoor_coupling_rows") or []),
        "repair_diagnostics_summary": {
            "status": (candidate.get("repair_diagnostics") or {}).get("status"),
            "monthly_nowcast_head": (candidate.get("repair_diagnostics") or {}).get("monthly_reporting_nowcast_head"),
            "backlog_late_emission_head": (candidate.get("repair_diagnostics") or {}).get("backlog_late_emission_head"),
            "u_to_d_target_count": (candidate.get("repair_diagnostics") or {}).get("u_to_d_target_count"),
            "u_to_d_target_source_counts": (candidate.get("repair_diagnostics") or {}).get("u_to_d_target_source_counts"),
            "selective_gate_source_counts": (candidate.get("repair_diagnostics") or {}).get("selective_gate_source_counts"),
        },
        "linkage_diagnostics": dict(candidate.get("linkage_diagnostics") or {}),
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
        "schema_version": "phase3_dynamic_frontdoor_one_year_gate.v1",
        "status": "pass" if not blockers else "fail",
        **score,
        "blockers": blockers,
        "contract": "one-year blocked-time full-cascade score after bounded U_to_D + D_to_A state/readout coupling",
    }


def _promotion_gate(*, one_year_gate: dict[str, Any], shock_gate: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if str(one_year_gate.get("status") or "") != "pass":
        blockers.append("fails_one_year_frontdoor_coupling_gate")
        blockers.extend(str(value) for value in list(one_year_gate.get("blockers") or []))
    if str(shock_gate.get("status") or "") != "pass":
        blockers.append("fails_shock_aware_lifted_trajectory_gate")
        blockers.extend(str(value) for value in list(shock_gate.get("blockers") or []))
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "promote_frontdoor_coupling_claim" if not blockers else "reject_frontdoor_coupling_claim",
        "promotion_eligible": bool(not blockers),
        "blockers": blockers,
        "claim_boundary": (
            "Promotion would support a short-horizon front-door cascade coupling claim only. "
            "It would not validate annual-incidence-derived quarterly truth or long-horizon scenario use."
        ),
    }


def _linkage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coupling_rows = [item for row in rows for item in list(row.get("frontdoor_coupling_rows") or [])]
    if not coupling_rows:
        return {"status": "no_coupled_rows", "coupled_quarter_count": 0}
    pressure = np.asarray([float(row.get("diagnosis_pressure_delta") or 0.0) for row in coupling_rows], dtype=np.float64)
    d_increment = np.asarray(
        [
            max(float(row.get("realized_d_to_a_flow") or 0.0) - float(row.get("baseline_d_to_a_flow") or 0.0), 0.0)
            for row in coupling_rows
        ],
        dtype=np.float64,
    )
    ratios = [
        float(row.get("linkage_flow_ratio") or 0.0)
        for row in coupling_rows
        if _finite_float(row.get("linkage_flow_ratio")) is not None
    ]
    return {
        "status": "completed",
        "coupled_quarter_count": len(coupling_rows),
        "total_diagnosis_pressure_delta": float(np.sum(pressure)),
        "total_d_to_a_increment": float(np.sum(d_increment)),
        "mean_linkage_flow_ratio": float(np.mean(np.asarray(ratios, dtype=np.float64))) if ratios else None,
        "contract": "D_to_A increments are measured relative to the baseline D_to_A flow for quarters with active U_to_D targets.",
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
    linkage = dict(payload.get("linkage_summary") or {})
    rows = list(payload.get("rows") or [])
    coupling_rows = [item for row in rows for item in list(row.get("frontdoor_coupling_rows") or [])]
    labels = [str(row.get("train_end_year") or "") for row in rows]
    deltas = [float(row.get("candidate_minus_carry_forward_mae") or 0.0) for row in rows]
    u_h = [float(row.get("bounded_u_to_d_hazard") or 0.0) for row in coupling_rows]
    d_h = [float(row.get("bounded_d_to_a_available_hazard") or 0.0) for row in coupling_rows]
    pressure = [float(row.get("diagnosis_pressure_delta") or 0.0) for row in coupling_rows]
    d_inc = [
        max(float(row.get("realized_d_to_a_flow") or 0.0) - float(row.get("baseline_d_to_a_flow") or 0.0), 0.0)
        for row in coupling_rows
    ]
    shock_overall = dict(shock.get("overall") or {})
    colors = {"candidate": "#9f1239", "carry": "#0f766e", "r10": "#111827", "link": "#1d4ed8"}
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
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.7), constrained_layout=True)
        flat = axes.ravel()
        x = np.arange(2)
        flat[0].bar(
            x,
            [float(gate.get("candidate_mean_mae") or 0.0), float(gate.get("carry_forward_mean_mae") or 0.0)],
            color=[colors["candidate"], colors["carry"]],
        )
        flat[0].set_xticks(x, ["candidate", "carry-forward"])
        flat[0].set_ylabel("Norm. MAE")
        flat[0].set_title("A. One-Year Front-Door Gate", loc="left", fontweight="bold")
        if labels:
            sx = np.arange(len(labels))
            flat[1].bar(sx, deltas, color=["#0f766e" if value < 0.0 else "#9f1239" for value in deltas])
            flat[1].axhline(0.0, color="#111827", linewidth=0.9)
            flat[1].set_xticks(sx, labels, rotation=35, ha="right")
        flat[1].set_ylabel("Candidate minus carry-forward")
        flat[1].set_title("B. Split-Level Delta", loc="left", fontweight="bold")
        if coupling_rows:
            hx = np.arange(len(coupling_rows))
            flat[2].plot(hx, u_h, color=colors["candidate"], linewidth=1.4, label="U_to_D")
            flat[2].plot(hx, d_h, color=colors["link"], linewidth=1.4, label="D_to_A available-pool")
            ax2 = flat[2].twinx()
            ax2.bar(hx, pressure, color="#cbd5e1", alpha=0.55, label="diagnosis pressure")
            ax2.plot(hx, d_inc, color="#334155", linewidth=0.9, label="D_to_A increment")
            flat[2].legend(frameon=False, loc="upper left")
            ax2.legend(frameon=False, loc="upper right")
            ax2.set_ylabel("Flow")
        flat[2].set_ylabel("Hazard")
        flat[2].set_title("C. Coupled Hazard And Flow Pressure", loc="left", fontweight="bold")
        text = [
            "Front-door coupling interpretation",
            f"One-year gate: {gate.get('status')}",
            f"Shock path: {shock.get('status')}",
            f"Promotion: {promotion.get('status')}",
            "",
            "Lifted path mean:",
            f"candidate {shock_overall.get('candidate_mean_mae')}",
            f"carry-forward {shock_overall.get('carry_forward_mean_mae')}",
            f"R10 {(shock.get('r10_reference') or {}).get('reference_quarterly_mean_mae')}",
            "",
            "Linkage:",
            f"coupled quarters {linkage.get('coupled_quarter_count')}",
            f"pressure {linkage.get('total_diagnosis_pressure_delta')}",
            f"D_to_A increment {linkage.get('total_d_to_a_increment')}",
            "",
            "Metric anatomy:",
        ]
        for row in anatomy[:4]:
            text.append(
                f"- {row.get('metric_name')}: "
                f"{float(row.get('candidate_minus_carry_forward_mean_norm_error') or 0.0):+.3f}"
            )
        text.extend(["", "Top blockers:"])
        text.extend([f"- {value}" for value in list(promotion.get("blockers") or [])[:5]] or ["- none"])
        flat[3].axis("off")
        flat[3].text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=8.2)
        for ax in flat[:3]:
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.suptitle("Phase 3 U_to_D + D_to_A Front-Door Coupling Gate", fontweight="bold")
        fig.savefig(path, dpi=330, bbox_inches="tight")
        plt.close(fig)


def run_frontdoor_coupling(
    *,
    run_id: str = "p3d-frontdoor-coupling-20260429-s00",
    source_report_path: str | Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
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
        family=FRONTDOOR_FAMILY,
        evaluated_rows=rows,
        shock_catalog=_detect_shock_catalog(observation_rows),
        r10_baseline=r10_baseline,
    )
    promotion = _promotion_gate(one_year_gate=one_year_gate, shock_gate=shock_gate)
    metric_anatomy = _metric_anatomy(rows)
    repair_summary = _repair_summary(rows)
    linkage_summary = _linkage_summary(rows)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "frontdoor_coupling_report.json"
    rows_csv_path = analysis_dir / "frontdoor_coupling_rows.csv"
    dashboard_path = analysis_dir / "frontdoor_coupling_dashboard.png"
    payload = {
        "schema_version": FRONTDOOR_COUPLING_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": active_source_run_id,
        "baseline_source_run_id": active_baseline_source_run_id,
        "source_report_path": source_report.as_posix(),
        "family": FRONTDOOR_FAMILY,
        "candidate_family": FRONTDOOR_FAMILY,
        "loop_variant": "evidence-to-model-loop",
        "contract": {
            "state_coupling": "selective diagnosis-flow repair is converted to bounded U_to_D and D_to_A hazards",
            "linkage_response": "D_to_A receives empirical train-window linkage response to excess U_to_D flow",
            "d_to_a_denominator": "D_to_A coupling uses available diagnosed pool D_prev + U_to_D for same-quarter front-door linkage",
            "readout_coupling": "new_diagnosed_cases_period after observation calibration is reset to realized U_to_D",
            "incidence_guard": "annual_new_infections is validation-only; diagnosis-flow-derived incidence readout is disabled; latent S_eff incidence enters U separately",
            "bound_rule": "U_to_D and D_to_A are clipped to empirical train-window hazard support",
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
        "linkage_summary": linkage_summary,
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
    parser = argparse.ArgumentParser(description="Run Phase3 U_to_D + D_to_A front-door coupling gate.")
    parser.add_argument("--run-id", default="p3d-frontdoor-coupling-20260429-s00")
    parser.add_argument("--source-report-path")
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    payload = run_frontdoor_coupling(
        run_id=args.run_id,
        source_report_path=args.source_report_path,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
