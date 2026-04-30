from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import (
    ART_STATE_NAMES,
    DIAGNOSED_STATE_NAMES,
    STATE_NAMES,
    TRANSITION_NAMES,
    VL_TESTED_STATE_NAMES,
    BlockedTimeDataset,
    canonical_state_values,
    state_sum,
)
from .decomposition import DecompositionControlConfig, apply_decomposition_controls
from .incidence import IncidenceFlowConfig, apply_stock_balance, fit_incidence_flow_paths
from .metrics import inv_logit, logit, normalized_mae, quarter_sort_key, smape
from .phase2 import DirectPriorFeature, HiddenDriverFeature, Phase2StructuralInputs
from .scientific_contracts import build_hazard_semantics, build_model_contract


@dataclass(slots=True)
class DynamicBaselineConfig:
    ridge_penalty: float = 0.1
    rho_clip: float = 0.9
    trend_scale: float = 1.0


@dataclass(slots=True)
class ObservationModelConfig:
    calibration_ridge: float = 0.1
    share_ridge_penalty: float = 0.1
    share_rho_clip: float = 0.9
    share_trend_scale: float = 1.0


@dataclass(slots=True)
class ShockConfig:
    shock_phi: float = 0.5
    shock_scale: float = 1.0
    gate_z: float = 1.0


@dataclass(slots=True)
class DampingConfig:
    min_dynamic_weight: float = 0.25
    gain_weight: float = 0.75
    residual_weight: float = 0.25
    horizon_decay: float = 0.6
    calibration_floor: float = 0.3
    calibration_gain_weight: float = 0.7


@dataclass(slots=True)
class DirectPriorConfig:
    effect_scale: float = 1.0
    precision_scale: float = 1.0
    residual_ridge: float = 0.1
    max_effect: float = 0.35


@dataclass(slots=True)
class HiddenDriverConfig:
    precision_scale: float = 1.0
    residual_ridge: float = 0.1
    max_effect: float = 0.2
    rank_cap: int = 1
    min_effect_weight: float = 1.0
    gain_weight: float = 0.0
    shock_penalty_weight: float = 0.0
    horizon_decay: float = 1.0
    gate_gain_weight: float = 0.0
    gate_recent_penalty: float = 0.0
    gate_threshold: float = -1.0
    blocked_weight: float = 1.0
    transition_weights: dict[str, float] | None = None


def carry_forward_hazards(dataset: BlockedTimeDataset, *, mode: str = "last_train") -> dict[str, dict[str, float]]:
    train_transition_rows = list(dataset.train_transition_rows)
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_transition_rows or not holdout_rows:
        return {}
    if str(mode) == "mean_train":
        baseline_hazards = {
            transition: float(np.mean([float((row.get("hazards") or {}).get(transition) or 0.0) for row in train_transition_rows]))
            for transition in TRANSITION_NAMES
        }
    else:
        baseline_hazards = {
            transition: float((train_transition_rows[-1].get("hazards") or {}).get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
    return {
        str(row.get("quarter") or ""): {transition: float(baseline_hazards.get(transition) or 0.0) for transition in TRANSITION_NAMES}
        for row in holdout_rows
    }


def _quarter_positions(dataset: BlockedTimeDataset) -> dict[str, int]:
    quarters = sorted({str(row.get("quarter") or "") for row in dataset.train_transition_rows + dataset.holdout_rows}, key=quarter_sort_key)
    return {quarter: idx for idx, quarter in enumerate(quarters)}


def _dynamic_weight(gain: float, residual_signal: float, damping_cfg: DampingConfig | None) -> float:
    if damping_cfg is None:
        return 1.0
    candidate = float(damping_cfg.min_dynamic_weight) + float(damping_cfg.gain_weight) * float(gain) + float(damping_cfg.residual_weight) * float(residual_signal)
    return float(np.clip(candidate, float(damping_cfg.min_dynamic_weight), 1.0))


def _decayed_weight(base_weight: float, step: int, *, floor: float, decay: float) -> float:
    return float(floor + (float(base_weight) - float(floor)) * (float(decay) ** int(step)))


def _fit_logit_ar_trend(
    train_quarters: list[str],
    train_values: list[float],
    forecast_quarters: list[str],
    *,
    ridge_penalty: float,
    rho_clip: float,
    trend_scale: float,
    positions: dict[str, int],
    eps: float,
    shock_cfg: ShockConfig | None = None,
    damping_cfg: DampingConfig | None = None,
) -> dict[str, Any]:
    etas = [logit(value, eps=eps) for value in train_values]
    if len(etas) < 2:
        constant_eta = float(etas[-1]) if etas else 0.0
        return {
            "train_eta_predictions": {quarter: constant_eta for quarter in train_quarters},
            "forecast_eta_predictions": {quarter: constant_eta for quarter in forecast_quarters},
            "last_residual": 0.0,
            "residual_sd": 0.0,
            "dynamic_gain": 0.0,
            "residual_signal": 0.0,
            "base_dynamic_weight": 1.0,
            "parameters": {"alpha": constant_eta, "slope": 0.0, "rho": 0.0},
        }
    y = np.asarray(etas[1:], dtype=np.float64)
    x_rows = []
    for idx in range(1, len(etas)):
        quarter = train_quarters[idx]
        x_rows.append([1.0, positions[quarter] * float(trend_scale), etas[idx - 1]])
    x = np.asarray(x_rows, dtype=np.float64)
    ridge = np.diag([0.0, float(ridge_penalty), float(ridge_penalty)])
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    alpha = float(beta[0])
    slope = float(beta[1])
    rho = float(np.clip(beta[2], -float(rho_clip), float(rho_clip)))
    train_eta_predictions: dict[str, float] = {train_quarters[0]: float(etas[0])}
    residuals: list[float] = []
    carry_forward_residuals: list[float] = []
    for idx in range(1, len(etas)):
        quarter = train_quarters[idx]
        previous_eta = float(etas[idx - 1])
        predicted = alpha + slope * positions[quarter] * float(trend_scale) + rho * previous_eta
        train_eta_predictions[quarter] = float(predicted)
        residuals.append(float(etas[idx] - predicted))
        carry_forward_residuals.append(float(etas[idx] - previous_eta))
    residual_sd = float(np.std(np.asarray(residuals, dtype=np.float64))) if residuals else 0.0
    last_residual = float(residuals[-1]) if residuals else 0.0
    carry_forward_mae = float(np.mean(np.abs(np.asarray(carry_forward_residuals, dtype=np.float64)))) if carry_forward_residuals else 0.0
    dynamic_mae = float(np.mean(np.abs(np.asarray(residuals, dtype=np.float64)))) if residuals else 0.0
    dynamic_gain = max(carry_forward_mae - dynamic_mae, 0.0) / max(carry_forward_mae, eps) if carry_forward_mae > eps else 0.0
    residual_signal = min(abs(last_residual) / max(residual_sd, eps), 3.0) / 3.0
    base_dynamic_weight = _dynamic_weight(dynamic_gain, residual_signal, damping_cfg)
    shock = 0.0
    if shock_cfg is not None:
        scale = max(residual_sd, eps)
        z_value = abs(last_residual) / scale if scale > 0.0 else 0.0
        if z_value >= float(shock_cfg.gate_z):
            shock = float(shock_cfg.shock_scale) * last_residual
    forecast_eta_predictions: dict[str, float] = {}
    last_eta = float(etas[-1])
    anchor_eta = float(etas[-1])
    for step, quarter in enumerate(forecast_quarters):
        base = alpha + slope * positions[quarter] * float(trend_scale) + rho * last_eta
        dynamic_eta = float(base + shock)
        if damping_cfg is None:
            eta = dynamic_eta
        else:
            weight = _decayed_weight(base_dynamic_weight, step, floor=float(damping_cfg.min_dynamic_weight), decay=float(damping_cfg.horizon_decay))
            eta = float(weight * dynamic_eta + (1.0 - weight) * anchor_eta)
        forecast_eta_predictions[quarter] = eta
        last_eta = eta
        if shock_cfg is not None:
            shock = float(shock_cfg.shock_phi) * shock
    return {
        "train_eta_predictions": train_eta_predictions,
        "forecast_eta_predictions": forecast_eta_predictions,
        "last_residual": last_residual,
        "residual_sd": residual_sd,
        "dynamic_gain": dynamic_gain,
        "residual_signal": residual_signal,
        "base_dynamic_weight": base_dynamic_weight,
        "parameters": {"alpha": alpha, "slope": slope, "rho": rho},
    }


def fit_dynamic_hazard_paths(dataset: BlockedTimeDataset, cfg: DynamicBaselineConfig, *, shock_cfg: ShockConfig | None = None, damping_cfg: DampingConfig | None = None) -> dict[str, Any]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_transition_rows or not holdout_rows:
        return {"train_hazard_map": {}, "holdout_hazard_map": {}, "diagnostics": {}}
    positions = _quarter_positions(dataset)
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    train_hazard_map: dict[str, dict[str, float]] = {str(row.get("quarter") or ""): {} for row in train_transition_rows}
    holdout_hazard_map: dict[str, dict[str, float]] = {quarter: {} for quarter in holdout_quarters}
    diagnostics: dict[str, Any] = {}
    for transition in TRANSITION_NAMES:
        train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
        train_values = [float((row.get("hazards") or {}).get(transition) or 0.0) for row in train_transition_rows]
        fit = _fit_logit_ar_trend(train_quarters, train_values, holdout_quarters, ridge_penalty=cfg.ridge_penalty, rho_clip=cfg.rho_clip, trend_scale=cfg.trend_scale, positions=positions, eps=dataset.eps, shock_cfg=shock_cfg, damping_cfg=damping_cfg)
        diagnostics[transition] = fit
        for quarter, eta in fit["train_eta_predictions"].items():
            train_hazard_map[quarter][transition] = float(inv_logit(eta))
        for quarter, eta in fit["forecast_eta_predictions"].items():
            holdout_hazard_map[quarter][transition] = float(inv_logit(eta))
    return {"train_hazard_map": train_hazard_map, "holdout_hazard_map": holdout_hazard_map, "diagnostics": diagnostics}


def _simulate_sequence(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    hazard_map: dict[str, dict[str, float]],
    *,
    incidence_inflow_map: dict[str, float] | None = None,
    incidence_hazard_map: dict[str, float] | None = None,
    incidence_cap_map: dict[str, float] | None = None,
    population_denominator_map: dict[str, float] | None = None,
    attrition_outflow_map: dict[str, float] | None = None,
    state_attrition_outflow_map: dict[str, dict[str, float]] | None = None,
    exit_channel_state_outflow_map: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    previous_state = canonical_state_values(initial_state)
    incidence_inflow_map = dict(incidence_inflow_map or {})
    incidence_hazard_map = dict(incidence_hazard_map or {})
    incidence_cap_map = dict(incidence_cap_map or {})
    population_denominator_map = dict(population_denominator_map or {})
    attrition_outflow_map = dict(attrition_outflow_map or {})
    state_attrition_outflow_map = dict(state_attrition_outflow_map or {})
    exit_channel_state_outflow_map = dict(exit_channel_state_outflow_map or {})
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for row in target_rows:
        quarter = str(row.get("quarter") or "")
        hazards = dict(hazard_map.get(quarter) or {})
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
            incidence_inflow=None if quarter in incidence_hazard_map and quarter in population_denominator_map else float(incidence_inflow_map.get(quarter) or 0.0),
            incidence_hazard=None if quarter not in incidence_hazard_map else float(incidence_hazard_map.get(quarter) or 0.0),
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
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}

def _fit_linear_calibration(raw_values: list[float], observed_values: list[float], positions: list[float], ridge_penalty: float, *, eps: float) -> dict[str, float]:
    if len(raw_values) < 1 or len(observed_values) < 1:
        return {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0, "gain": 0.0}
    raw = np.maximum(np.asarray(raw_values, dtype=np.float64), 0.0)
    observed = np.maximum(np.asarray(observed_values, dtype=np.float64), 0.0)
    denominator = float(raw @ raw + float(ridge_penalty))
    raw_scale = 1.0 if denominator <= eps else max(float(raw @ observed) / denominator, 0.0)
    fitted = raw_scale * raw
    raw_mae = float(np.mean(np.abs(observed - raw)))
    calibrated_mae = float(np.mean(np.abs(observed - fitted)))
    gain = max(raw_mae - calibrated_mae, 0.0) / max(raw_mae, eps) if raw_mae > eps else 0.0
    return {"intercept": 0.0, "raw_scale": float(raw_scale), "time_slope": 0.0, "gain": float(gain)}


def _apply_linear_calibration(raw_value: float, position: float, coeffs: dict[str, float]) -> float:
    del position
    return float(max(coeffs["raw_scale"], 0.0) * max(raw_value, 0.0))


def _fit_share_model(rows: list[dict[str, Any]], *, numerator_key: str, denominator_key: str, cfg: ObservationModelConfig, positions: dict[str, int], holdout_quarters: list[str], eps: float) -> dict[str, Any]:
    usable = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value <= eps:
            continue
        usable.append((str(row.get("quarter") or ""), float(numerator) / denominator_value))
    if not usable:
        return {"forecast": {quarter: 0.0 for quarter in holdout_quarters}, "train": {}, "mean_share": 0.0, "gain": 0.0}
    train_quarters = [quarter for quarter, _share in usable]
    train_shares = [share for _quarter, share in usable]
    fit = _fit_logit_ar_trend(train_quarters, train_shares, holdout_quarters, ridge_penalty=cfg.share_ridge_penalty, rho_clip=cfg.share_rho_clip, trend_scale=cfg.share_trend_scale, positions=positions, eps=eps, shock_cfg=None, damping_cfg=None)
    train_pred = [float(inv_logit(fit["train_eta_predictions"][quarter])) for quarter in train_quarters]
    mean_share = float(np.mean(train_shares))
    baseline_mae = float(np.mean(np.abs(np.asarray(train_shares, dtype=np.float64) - mean_share)))
    fit_mae = float(np.mean(np.abs(np.asarray(train_shares, dtype=np.float64) - np.asarray(train_pred, dtype=np.float64))))
    gain = max(baseline_mae - fit_mae, 0.0) / max(baseline_mae, eps) if baseline_mae > eps else 0.0
    return {"forecast": {quarter: float(inv_logit(eta)) for quarter, eta in fit["forecast_eta_predictions"].items()}, "train": {quarter: float(inv_logit(eta)) for quarter, eta in fit["train_eta_predictions"].items()}, "mean_share": mean_share, "gain": gain}


def fit_observation_model(
    dataset: BlockedTimeDataset,
    train_hazard_map: dict[str, dict[str, float]],
    cfg: ObservationModelConfig,
    *,
    incidence_paths: dict[str, Any] | None = None,
    damping_cfg: DampingConfig | None = None,
) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if len(train_rows) < 3:
        holdout_quarters = [str(row.get("quarter") or "") for row in dataset.holdout_rows]
        return {"primary_coefficients": {}, "share_forecasts": {"tested_for_viral_load": {quarter: None for quarter in holdout_quarters}, "virally_suppressed": {quarter: None for quarter in holdout_quarters}}, "share_means": {"tested_for_viral_load": 0.0, "virally_suppressed": 0.0}}
    positions = _quarter_positions(dataset)
    initial_state = dict(dataset.train_state_rows[0]["state_values"])
    raw_train = _simulate_sequence(
        initial_state,
        train_rows[1:],
        train_hazard_map,
        incidence_inflow_map=dict((incidence_paths or {}).get("train_incidence_inflow_map") or {}),
        incidence_hazard_map=dict((incidence_paths or {}).get("train_incidence_hazard_map") or {}),
        incidence_cap_map=dict((incidence_paths or {}).get("train_incidence_cap_map") or {}),
        population_denominator_map=dict((incidence_paths or {}).get("train_population_denominator_map") or {}),
        attrition_outflow_map=dict((incidence_paths or {}).get("train_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict((incidence_paths or {}).get("train_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict((incidence_paths or {}).get("train_exit_channel_state_outflow_map") or {}),
    )["prediction_rows"]
    target_train = train_rows[1:]
    primary_coefficients: dict[str, dict[str, float]] = {}
    primary_weights: dict[str, float] = {}
    for metric_name in ("diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period"):
        raw_values = [float(row.get(metric_name) or 0.0) for row in raw_train]
        observed_values = [float(row.get(metric_name) or 0.0) for row in target_train]
        quarter_positions = [float(positions[str(row.get("quarter") or "")]) for row in target_train]
        coeffs = _fit_linear_calibration(raw_values, observed_values, quarter_positions, cfg.calibration_ridge, eps=dataset.eps)
        primary_coefficients[metric_name] = coeffs
        primary_weights[metric_name] = 1.0 if damping_cfg is None else float(np.clip(float(damping_cfg.calibration_floor) + float(damping_cfg.calibration_gain_weight) * float(coeffs.get("gain") or 0.0), float(damping_cfg.calibration_floor), 1.0))
    holdout_quarters = [str(row.get("quarter") or "") for row in dataset.holdout_rows]
    testing_share = _fit_share_model(train_rows, numerator_key="tested_for_viral_load", denominator_key="alive_on_art", cfg=cfg, positions=positions, holdout_quarters=holdout_quarters, eps=dataset.eps)
    suppression_share = _fit_share_model(train_rows, numerator_key="virally_suppressed", denominator_key="alive_on_art", cfg=cfg, positions=positions, holdout_quarters=holdout_quarters, eps=dataset.eps)
    share_weights = {}
    for metric_name, summary in (("tested_for_viral_load", testing_share), ("virally_suppressed", suppression_share)):
        share_weights[metric_name] = 1.0 if damping_cfg is None else float(np.clip(float(damping_cfg.calibration_floor) + float(damping_cfg.calibration_gain_weight) * float(summary.get("gain") or 0.0), float(damping_cfg.calibration_floor), 1.0))
    return {
        "primary_coefficients": primary_coefficients,
        "primary_weights": primary_weights,
        "share_forecasts": {"tested_for_viral_load": testing_share["forecast"], "virally_suppressed": suppression_share["forecast"]},
        "share_train": {"tested_for_viral_load": testing_share["train"], "virally_suppressed": suppression_share["train"]},
        "share_means": {"tested_for_viral_load": testing_share["mean_share"], "virally_suppressed": suppression_share["mean_share"]},
        "share_weights": share_weights,
    }


def _apply_observation_model(raw_prediction_rows: list[dict[str, float]], observation_model: dict[str, Any], *, positions: dict[str, int], damping_cfg: DampingConfig | None = None) -> list[dict[str, float]]:
    coeffs = dict(observation_model.get("primary_coefficients") or {})
    primary_weights = dict(observation_model.get("primary_weights") or {})
    share_forecasts = dict(observation_model.get("share_forecasts") or {})
    share_weights = dict(observation_model.get("share_weights") or {})
    share_means = dict(observation_model.get("share_means") or {})
    testing_share = dict(share_forecasts.get("tested_for_viral_load") or {})
    suppression_share = dict(share_forecasts.get("virally_suppressed") or {})
    calibrated_rows: list[dict[str, float]] = []
    for step, row in enumerate(raw_prediction_rows):
        quarter = str(row.get("quarter") or "")
        position = float(positions.get(quarter, 0))
        diagnosed_raw = float(row.get("diagnosed_plhiv") or 0.0)
        alive_raw = float(row.get("alive_on_art") or 0.0)
        new_diag_raw = float(row.get("new_diagnosed_cases_period") or 0.0)
        diagnosed_calibrated = _apply_linear_calibration(diagnosed_raw, position, coeffs.get("diagnosed_plhiv") or {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0})
        alive_calibrated = _apply_linear_calibration(alive_raw, position, coeffs.get("alive_on_art") or {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0})
        new_diag_calibrated = _apply_linear_calibration(new_diag_raw, position, coeffs.get("new_diagnosed_cases_period") or {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0})
        if damping_cfg is None:
            diagnosed_weight = alive_weight = new_diag_weight = 1.0
            share_test_weight = share_supp_weight = 1.0
        else:
            diagnosed_weight = _decayed_weight(float(primary_weights.get("diagnosed_plhiv") or 1.0), step, floor=float(damping_cfg.calibration_floor), decay=float(damping_cfg.horizon_decay))
            alive_weight = _decayed_weight(float(primary_weights.get("alive_on_art") or 1.0), step, floor=float(damping_cfg.calibration_floor), decay=float(damping_cfg.horizon_decay))
            new_diag_weight = _decayed_weight(float(primary_weights.get("new_diagnosed_cases_period") or 1.0), step, floor=float(damping_cfg.calibration_floor), decay=float(damping_cfg.horizon_decay))
            share_test_weight = _decayed_weight(float(share_weights.get("tested_for_viral_load") or 1.0), step, floor=float(damping_cfg.calibration_floor), decay=float(damping_cfg.horizon_decay))
            share_supp_weight = _decayed_weight(float(share_weights.get("virally_suppressed") or 1.0), step, floor=float(damping_cfg.calibration_floor), decay=float(damping_cfg.horizon_decay))
        diagnosed = diagnosed_weight * diagnosed_calibrated + (1.0 - diagnosed_weight) * diagnosed_raw
        alive = alive_weight * alive_calibrated + (1.0 - alive_weight) * alive_raw
        new_diag = new_diag_weight * new_diag_calibrated + (1.0 - new_diag_weight) * new_diag_raw
        alive = max(alive, 0.0)
        diagnosed = max(diagnosed, alive)
        new_diag = max(new_diag, 0.0)
        test_share_value = testing_share.get(quarter)
        supp_share_value = suppression_share.get(quarter)
        raw_test = row.get("tested_for_viral_load")
        raw_supp = row.get("virally_suppressed")
        blended_test_share = None
        blended_supp_share = None
        if test_share_value is not None:
            anchor = float(share_means.get("tested_for_viral_load") or 0.0)
            blended_test_share = share_test_weight * max(float(test_share_value), 0.0) + (1.0 - share_test_weight) * anchor
        if supp_share_value is not None:
            anchor = float(share_means.get("virally_suppressed") or 0.0)
            blended_supp_share = share_supp_weight * max(float(supp_share_value), 0.0) + (1.0 - share_supp_weight) * anchor
        tested = raw_test if blended_test_share is None else blended_test_share * alive
        suppressed = raw_supp if blended_supp_share is None else blended_supp_share * alive
        if tested is not None:
            tested = min(max(float(tested), 0.0), alive)
        if suppressed is not None:
            suppressed = min(max(float(suppressed), 0.0), alive)
        if tested is not None and suppressed is not None:
            tested = min(alive, max(float(tested), float(suppressed)))
        calibrated_rows.append({"quarter": quarter, "diagnosed_plhiv": diagnosed, "alive_on_art": alive, "new_diagnosed_cases_period": new_diag, "tested_for_viral_load": tested, "virally_suppressed": suppressed})
    return calibrated_rows

def _forecast_origin_index(quarter_index: dict[str, int], train_quarters: list[str]) -> int:
    train_indices = [int(quarter_index[quarter]) for quarter in train_quarters if quarter in quarter_index]
    return max(train_indices) if train_indices else -1


def _forecast_origin_tensor_value(
    tensor: np.ndarray,
    source_idx: int | None,
    value_idx: int,
    forecast_origin_idx: int,
) -> tuple[float, bool]:
    if source_idx is None or forecast_origin_idx < 0:
        return 0.0, False
    safe_idx = min(int(source_idx), int(forecast_origin_idx), int(tensor.shape[0]) - 1)
    if safe_idx < 0:
        return 0.0, False
    return float(tensor[safe_idx, value_idx]), int(source_idx) > int(forecast_origin_idx)


def _phase2_forecast_contract(
    *,
    forecast_origin_quarter: str | None,
    holdout_future_source_count: int,
    holdout_value_count: int,
) -> dict[str, Any]:
    return {
        "forecast_origin_contract": "train_history_only_with_carry_forward_for_future_phase2_values",
        "forecast_origin_quarter": forecast_origin_quarter,
        "holdout_future_source_count": int(holdout_future_source_count),
        "holdout_value_count": int(holdout_value_count),
    }


def _direct_prior_design(structural_inputs: Phase2StructuralInputs, features: list[DirectPriorFeature], train_quarters: list[str], holdout_quarters: list[str], *, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if not features:
        return np.zeros((len(train_quarters), 0)), np.zeros((len(holdout_quarters), 0)), np.zeros(0), np.zeros(0), [], _phase2_forecast_contract(forecast_origin_quarter=None, holdout_future_source_count=0, holdout_value_count=0)
    quarter_index = {quarter: idx for idx, quarter in enumerate(structural_inputs.quarter_axis)}
    tensor = np.asarray(structural_inputs.national_quarter_tensor[0], dtype=np.float64)
    forecast_origin_idx = _forecast_origin_index(quarter_index, train_quarters)
    forecast_origin_quarter = structural_inputs.quarter_axis[forecast_origin_idx] if forecast_origin_idx >= 0 else None
    holdout_future_source_count = 0
    holdout_value_count = 0
    train_columns = []
    holdout_columns = []
    prior_means = []
    prior_precisions = []
    names = []
    for feature in features:
        train_raw = []
        holdout_raw = []
        for quarter in train_quarters:
            q_idx = quarter_index.get(quarter)
            source_idx = q_idx - int(feature.lag) if q_idx is not None else None
            train_raw.append(float(tensor[source_idx, feature.tensor_index]) if source_idx is not None and source_idx >= 0 else 0.0)
        for quarter in holdout_quarters:
            q_idx = quarter_index.get(quarter)
            source_idx = q_idx - int(feature.lag) if q_idx is not None else None
            value, was_future = _forecast_origin_tensor_value(tensor, source_idx, feature.tensor_index, forecast_origin_idx)
            holdout_raw.append(value)
            holdout_future_source_count += int(was_future)
            holdout_value_count += 1
        mean_value = float(np.mean(train_raw)) if train_raw else 0.0
        std_value = float(np.std(train_raw)) if train_raw else 0.0
        if std_value <= eps:
            train_column = np.zeros(len(train_raw), dtype=np.float64)
            holdout_column = np.zeros(len(holdout_raw), dtype=np.float64)
        else:
            train_column = (np.asarray(train_raw, dtype=np.float64) - mean_value) / std_value
            holdout_column = (np.asarray(holdout_raw, dtype=np.float64) - mean_value) / std_value
        train_columns.append(train_column)
        holdout_columns.append(holdout_column)
        prior_means.append(float(feature.prior_scale) * float(feature.phase2_weight))
        prior_precisions.append(max(float(feature.stability), 0.05) * max(float(feature.support_count), 1.0))
        names.append(f"{feature.transition}:{feature.source}->{feature.target}@lag{feature.lag}")
    return (
        np.asarray(np.column_stack(train_columns), dtype=np.float64),
        np.asarray(np.column_stack(holdout_columns), dtype=np.float64),
        np.asarray(prior_means, dtype=np.float64),
        np.asarray(prior_precisions, dtype=np.float64),
        names,
        _phase2_forecast_contract(
            forecast_origin_quarter=forecast_origin_quarter,
            holdout_future_source_count=holdout_future_source_count,
            holdout_value_count=holdout_value_count,
        ),
    )


def _hidden_driver_design(structural_inputs: Phase2StructuralInputs, features: list[HiddenDriverFeature], train_quarters: list[str], holdout_quarters: list[str], *, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if not features:
        return np.zeros((len(train_quarters), 0)), np.zeros((len(holdout_quarters), 0)), np.zeros(0), [], _phase2_forecast_contract(forecast_origin_quarter=None, holdout_future_source_count=0, holdout_value_count=0)
    quarter_index = {quarter: idx for idx, quarter in enumerate(structural_inputs.quarter_axis)}
    tensor = np.asarray(structural_inputs.hidden_mode_quarter_tensor[0], dtype=np.float64)
    if tensor.ndim == 1:
        tensor = tensor[:, np.newaxis]
    forecast_origin_idx = _forecast_origin_index(quarter_index, train_quarters)
    forecast_origin_quarter = structural_inputs.quarter_axis[forecast_origin_idx] if forecast_origin_idx >= 0 else None
    holdout_future_source_count = 0
    holdout_value_count = 0
    train_columns = []
    holdout_columns = []
    prior_precisions = []
    names = []
    for feature in features:
        if int(feature.mode_index) < 0 or int(feature.mode_index) >= int(tensor.shape[1]):
            continue
        train_raw = []
        holdout_raw = []
        for quarter in train_quarters:
            q_idx = quarter_index.get(quarter)
            train_raw.append(float(tensor[q_idx, feature.mode_index]) if q_idx is not None and q_idx < tensor.shape[0] else 0.0)
        for quarter in holdout_quarters:
            q_idx = quarter_index.get(quarter)
            value, was_future = _forecast_origin_tensor_value(tensor, q_idx, feature.mode_index, forecast_origin_idx)
            holdout_raw.append(value)
            holdout_future_source_count += int(was_future)
            holdout_value_count += 1
        mean_value = float(np.mean(train_raw)) if train_raw else 0.0
        std_value = float(np.std(train_raw)) if train_raw else 0.0
        if std_value <= eps:
            train_column = np.zeros(len(train_raw), dtype=np.float64)
            holdout_column = np.zeros(len(holdout_raw), dtype=np.float64)
        else:
            train_column = (np.asarray(train_raw, dtype=np.float64) - mean_value) / std_value
            holdout_column = (np.asarray(holdout_raw, dtype=np.float64) - mean_value) / std_value
        train_columns.append(train_column)
        holdout_columns.append(holdout_column)
        prior_precisions.append(max(float(feature.stability), 0.05) * max(float(feature.support_scale), 0.05) * max(float(feature.support_count), 1.0))
        target_text = ','.join(feature.target_blocks)
        names.append(f"{feature.transition}:hidden_mode_{feature.mode_index}[{target_text}]")
    if not train_columns:
        return np.zeros((len(train_quarters), 0)), np.zeros((len(holdout_quarters), 0)), np.zeros(0), [], _phase2_forecast_contract(forecast_origin_quarter=forecast_origin_quarter, holdout_future_source_count=holdout_future_source_count, holdout_value_count=holdout_value_count)
    return (
        np.asarray(np.column_stack(train_columns), dtype=np.float64),
        np.asarray(np.column_stack(holdout_columns), dtype=np.float64),
        np.asarray(prior_precisions, dtype=np.float64),
        names,
        _phase2_forecast_contract(
            forecast_origin_quarter=forecast_origin_quarter,
            holdout_future_source_count=holdout_future_source_count,
            holdout_value_count=holdout_value_count,
        ),
    )


def _apply_direct_priors(dataset: BlockedTimeDataset, structural_inputs: Phase2StructuralInputs, direct_prior_features: dict[str, list[DirectPriorFeature]], prior_cfg: DirectPriorConfig, paths: dict[str, Any]) -> dict[str, Any]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    train_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("train_hazard_map") or {}).items()}
    holdout_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("holdout_hazard_map") or {}).items()}
    diagnostics: dict[str, Any] = {}
    for transition in TRANSITION_NAMES:
        features = list((direct_prior_features or {}).get(transition) or [])
        if not features:
            diagnostics[transition] = {"feature_count": 0}
            continue
        x_train, x_holdout, prior_means, prior_precisions, names, design_contract = _direct_prior_design(structural_inputs, features, train_quarters, holdout_quarters, eps=dataset.eps)
        if x_train.shape[1] == 0:
            diagnostics[transition] = {"feature_count": 0, **design_contract}
            continue
        y = np.asarray([
            logit(float((row.get("hazards") or {}).get(transition) or 0.0), eps=dataset.eps)
            - logit(float((train_hazard_map.get(str(row.get("quarter") or ""), {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            for row in train_transition_rows
        ], dtype=np.float64)
        diag_values = float(prior_cfg.residual_ridge) + float(prior_cfg.precision_scale) * prior_precisions
        lambda_diag = np.diag(diag_values)
        rhs = x_train.T @ y + diag_values * (float(prior_cfg.effect_scale) * prior_means)
        beta = np.linalg.solve(x_train.T @ x_train + lambda_diag, rhs)
        beta = np.clip(beta, -float(prior_cfg.max_effect), float(prior_cfg.max_effect))
        train_effect = np.asarray(x_train @ beta, dtype=np.float64)
        holdout_effect = np.asarray(x_holdout @ beta, dtype=np.float64)
        for quarter, effect in zip(train_quarters, train_effect):
            base_eta = logit(float((train_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            train_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + effect))
        for quarter, effect in zip(holdout_quarters, holdout_effect):
            base_eta = logit(float((holdout_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            holdout_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + effect))
        diagnostics[transition] = {"feature_count": len(features), "feature_names": names, "coefficients": {name: float(value) for name, value in zip(names, beta)}, "mean_abs_holdout_effect": float(np.mean(np.abs(holdout_effect))) if len(holdout_effect) else 0.0, **design_contract}
    return {"train_hazard_map": train_hazard_map, "holdout_hazard_map": holdout_hazard_map, "diagnostics": diagnostics}


def _apply_hidden_drivers(dataset: BlockedTimeDataset, structural_inputs: Phase2StructuralInputs, hidden_driver_features: dict[str, list[HiddenDriverFeature]], hidden_cfg: HiddenDriverConfig, paths: dict[str, Any]) -> dict[str, Any]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    train_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("train_hazard_map") or {}).items()}
    holdout_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("holdout_hazard_map") or {}).items()}
    diagnostics: dict[str, Any] = {}
    for transition in TRANSITION_NAMES:
        features = [feature for feature in list((hidden_driver_features or {}).get(transition) or []) if int(feature.mode_index) < int(hidden_cfg.rank_cap)]
        if not features:
            diagnostics[transition] = {"feature_count": 0}
            continue
        x_train, x_holdout, prior_precisions, names, design_contract = _hidden_driver_design(structural_inputs, features, train_quarters, holdout_quarters, eps=dataset.eps)
        if x_train.shape[1] == 0:
            diagnostics[transition] = {"feature_count": 0, **design_contract}
            continue
        y = np.asarray([
            logit(float((row.get("hazards") or {}).get(transition) or 0.0), eps=dataset.eps)
            - logit(float((train_hazard_map.get(str(row.get("quarter") or ""), {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            for row in train_transition_rows
        ], dtype=np.float64)
        diag_values = float(hidden_cfg.residual_ridge) + float(hidden_cfg.precision_scale) * prior_precisions
        lambda_diag = np.diag(diag_values)
        beta = np.linalg.solve(x_train.T @ x_train + lambda_diag, x_train.T @ y)
        beta = np.clip(beta, -float(hidden_cfg.max_effect), float(hidden_cfg.max_effect))
        raw_train_effect = np.asarray(x_train @ beta, dtype=np.float64)
        raw_holdout_effect = np.asarray(x_holdout @ beta, dtype=np.float64)
        residual_before = float(np.mean(np.abs(y))) if len(y) else 0.0
        residual_after = float(np.mean(np.abs(y - raw_train_effect))) if len(y) else 0.0
        hidden_gain = max(residual_before - residual_after, 0.0) / max(residual_before, dataset.eps) if residual_before > dataset.eps else 0.0
        effect_sd = float(np.std(raw_train_effect)) if len(raw_train_effect) else 0.0
        recent_signal = min(abs(float(raw_train_effect[-1])) / max(effect_sd, dataset.eps), 3.0) / 3.0 if len(raw_train_effect) else 0.0
        gate_score = float(hidden_cfg.gate_gain_weight) * float(hidden_gain) - float(hidden_cfg.gate_recent_penalty) * float(recent_signal)
        transition_weight = 1.0 if gate_score >= float(hidden_cfg.gate_threshold) else float(hidden_cfg.blocked_weight)
        transition_weight *= float((hidden_cfg.transition_weights or {}).get(transition, 1.0))
        base_weight = float(hidden_cfg.min_effect_weight) + float(hidden_cfg.gain_weight) * float(hidden_gain)
        base_weight = float(np.clip(base_weight, float(hidden_cfg.min_effect_weight), 1.0))
        effective_decay = float(np.clip(float(hidden_cfg.horizon_decay) * (1.0 - float(hidden_cfg.shock_penalty_weight) * float(recent_signal)), 0.05, 0.98))
        train_effect = np.asarray(transition_weight * base_weight * raw_train_effect, dtype=np.float64)
        holdout_effect = []
        for step, effect in enumerate(raw_holdout_effect):
            weight = transition_weight * _decayed_weight(base_weight, step, floor=float(hidden_cfg.min_effect_weight), decay=effective_decay)
            holdout_effect.append(float(weight) * float(effect))
        holdout_effect = np.asarray(holdout_effect, dtype=np.float64)
        for quarter, effect in zip(train_quarters, train_effect):
            base_eta = logit(float((train_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            train_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + effect))
        for quarter, effect in zip(holdout_quarters, holdout_effect):
            base_eta = logit(float((holdout_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            holdout_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + effect))
        diagnostics[transition] = {
            "feature_count": len(features),
            "feature_names": names,
            "coefficients": {name: float(value) for name, value in zip(names, beta)},
            "mean_abs_holdout_effect": float(np.mean(np.abs(holdout_effect))) if len(holdout_effect) else 0.0,
            "raw_mean_abs_holdout_effect": float(np.mean(np.abs(raw_holdout_effect))) if len(raw_holdout_effect) else 0.0,
            "hidden_gain": float(hidden_gain),
            "recent_signal": float(recent_signal),
            "gate_score": float(gate_score),
            "gate_passed": bool(gate_score >= float(hidden_cfg.gate_threshold)),
            "transition_weight": float(transition_weight),
            "base_weight": float(base_weight),
            "effective_decay": float(effective_decay),
            **design_contract,
        }
    return {"train_hazard_map": train_hazard_map, "holdout_hazard_map": holdout_hazard_map, "diagnostics": diagnostics}


def forecast_dynamic_baseline(
    dataset: BlockedTimeDataset,
    dynamic_cfg: DynamicBaselineConfig,
    *,
    incidence_cfg: IncidenceFlowConfig | None = None,
    observation_cfg: ObservationModelConfig | None = None,
    shock_cfg: ShockConfig | None = None,
    damping_cfg: DampingConfig | None = None,
    structural_inputs: Phase2StructuralInputs | None = None,
    direct_prior_features: dict[str, list[DirectPriorFeature]] | None = None,
    prior_cfg: DirectPriorConfig | None = None,
    hidden_driver_features: dict[str, list[HiddenDriverFeature]] | None = None,
    hidden_cfg: HiddenDriverConfig | None = None,
    decomposition_cfg: DecompositionControlConfig | None = None,
) -> dict[str, Any]:
    paths = fit_dynamic_hazard_paths(dataset, dynamic_cfg, shock_cfg=shock_cfg, damping_cfg=damping_cfg)
    incidence_paths = None if incidence_cfg is None else fit_incidence_flow_paths(dataset, incidence_cfg)
    prior_diagnostics = None
    hidden_diagnostics = None
    decomposition_diagnostics = None
    incidence_enabled = incidence_cfg is not None
    phase2_direct_enabled = (
        structural_inputs is not None
        and direct_prior_features is not None
        and prior_cfg is not None
        and any(len(rows) > 0 for rows in dict(direct_prior_features or {}).values())
    )
    phase2_hidden_enabled = (
        structural_inputs is not None
        and hidden_driver_features is not None
        and hidden_cfg is not None
        and any(len(rows) > 0 for rows in dict(hidden_driver_features or {}).values())
    )
    if phase2_direct_enabled:
        prior_adjustment = _apply_direct_priors(dataset, structural_inputs, direct_prior_features, prior_cfg, paths)
        paths["train_hazard_map"] = prior_adjustment["train_hazard_map"]
        paths["holdout_hazard_map"] = prior_adjustment["holdout_hazard_map"]
        prior_diagnostics = prior_adjustment["diagnostics"]
    if phase2_hidden_enabled:
        hidden_adjustment = _apply_hidden_drivers(dataset, structural_inputs, hidden_driver_features, hidden_cfg, paths)
        paths["train_hazard_map"] = hidden_adjustment["train_hazard_map"]
        paths["holdout_hazard_map"] = hidden_adjustment["holdout_hazard_map"]
        hidden_diagnostics = hidden_adjustment["diagnostics"]
    if decomposition_cfg is not None:
        decomposition_adjustment = apply_decomposition_controls(
            dataset=dataset,
            paths=paths,
            incidence_paths=incidence_paths,
            cfg=decomposition_cfg,
        )
        paths = decomposition_adjustment["paths"]
        incidence_paths = decomposition_adjustment["incidence_paths"]
        decomposition_diagnostics = decomposition_adjustment["diagnostics"]
    model_contract = build_model_contract(
        incidence_enabled=incidence_enabled,
        observation_calibration_enabled=observation_cfg is not None,
        phase2_direct_enabled=phase2_direct_enabled,
        phase2_hidden_enabled=phase2_hidden_enabled,
        decomposition_enabled=decomposition_cfg is not None,
    )
    hazard_semantics = build_hazard_semantics(
        incidence_enabled=incidence_enabled,
        observation_calibration_enabled=observation_cfg is not None,
        phase2_direct_enabled=phase2_direct_enabled,
        phase2_hidden_enabled=phase2_hidden_enabled,
        decomposition_enabled=decomposition_cfg is not None,
    )
    if not dataset.train_state_rows:
        return {
            "prediction_rows": [],
            "mae": float("inf"),
            "smape": 0.0,
            "incidence_paths": incidence_paths,
            "model_contract": model_contract,
            "hazard_semantics": hazard_semantics,
        }
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    raw_result = _simulate_sequence(
        dict(dataset.train_state_rows[-1]["state_values"]),
        holdout_rows,
        paths["holdout_hazard_map"],
        incidence_inflow_map=dict((incidence_paths or {}).get("holdout_incidence_inflow_map") or {}),
        incidence_hazard_map=dict((incidence_paths or {}).get("holdout_incidence_hazard_map") or {}),
        incidence_cap_map=dict((incidence_paths or {}).get("holdout_incidence_cap_map") or {}),
        population_denominator_map=dict((incidence_paths or {}).get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=dict((incidence_paths or {}).get("holdout_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict((incidence_paths or {}).get("holdout_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict((incidence_paths or {}).get("holdout_exit_channel_state_outflow_map") or {}),
    )
    positions = _quarter_positions(dataset)
    if observation_cfg is not None:
        observation_model = fit_observation_model(
            dataset,
            paths["train_hazard_map"],
            observation_cfg,
            incidence_paths=incidence_paths,
            damping_cfg=damping_cfg,
        )
        prediction_rows = _apply_observation_model(raw_result["prediction_rows"], observation_model, positions=positions, damping_cfg=damping_cfg)
    else:
        observation_model = None
        prediction_rows = raw_result["prediction_rows"]
    return {
        "prediction_rows": prediction_rows,
        "raw_prediction_rows": raw_result["prediction_rows"],
        "trajectory_rows": raw_result["trajectory_rows"],
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
        "smape": float(smape(prediction_rows, holdout_rows, eps=dataset.eps)),
        "hazard_paths": paths,
        "incidence_paths": incidence_paths,
        "observation_model": observation_model,
        "prior_diagnostics": prior_diagnostics,
        "hidden_diagnostics": hidden_diagnostics,
        "decomposition_diagnostics": decomposition_diagnostics,
        "model_contract": model_contract,
        "hazard_semantics": hazard_semantics,
    }


def simulate_holdout(
    dataset: BlockedTimeDataset,
    hazard_map: dict[str, dict[str, float]],
    *,
    incidence_inflow_map: dict[str, float] | None = None,
    incidence_hazard_map: dict[str, float] | None = None,
    incidence_cap_map: dict[str, float] | None = None,
    population_denominator_map: dict[str, float] | None = None,
    attrition_outflow_map: dict[str, float] | None = None,
    state_attrition_outflow_map: dict[str, dict[str, float]] | None = None,
    exit_channel_state_outflow_map: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not holdout_rows or not dataset.train_state_rows:
        return {"prediction_rows": [], "mae": float("inf"), "smape": 0.0}
    raw_result = _simulate_sequence(
        dict(dataset.train_state_rows[-1]["state_values"]),
        holdout_rows,
        hazard_map,
        incidence_inflow_map=incidence_inflow_map,
        incidence_hazard_map=incidence_hazard_map,
        incidence_cap_map=incidence_cap_map,
        population_denominator_map=population_denominator_map,
        attrition_outflow_map=attrition_outflow_map,
        state_attrition_outflow_map=state_attrition_outflow_map,
        exit_channel_state_outflow_map=exit_channel_state_outflow_map,
    )
    return {"prediction_rows": raw_result["prediction_rows"], "trajectory_rows": raw_result["trajectory_rows"], "mae": float(normalized_mae(raw_result["prediction_rows"], holdout_rows, dataset.metric_scales, eps=dataset.eps)), "smape": float(smape(raw_result["prediction_rows"], holdout_rows, eps=dataset.eps))}

