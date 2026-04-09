from __future__ import annotations

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3._lineage.national_reset_core import PRIMARY_METRICS, quarter_sort_key
from epigraph_ph.phase3.shared.temporal_scaffold import build_shock_regime_basis, build_temporal_basis
from epigraph_ph.runtime import ROOT_DIR, load_tensor_artifact, read_json, save_tensor_artifact, write_json

from .analytics import _factor_static_metrics
from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .analytics import _factor_transition_rows
from .numeric_policy import numerical_guard_entry
from .registry import TRANSITION_NAMES
from .sources import load_transition_research_inputs
from .transition_engine import (
    DOWNSTREAM_TRANSITIONS,
    MECH01E_EXPERIMENT_ID,
    STATE_NAMES,
    _build_observation_payload,
    _diagnosis_flow_mae,
    _holdout_predictions_from_reference,
    _holdout_state_rows_from_reference,
    _load_mech_01e_reference,
    _load_transition_experiment_reference,
    _metric_scales,
    _normalized_mae,
    _pearson_corr,
    _quarter_factor_surface,
    _reference_holdout_hazards,
    _smape,
    _state_row,
    _state_tensor,
    _state_trajectory_rows,
    _train_rows,
    _holdout_rows,
    _train_transition_rows,
)

CHANNEL_NAMES: tuple[str, ...] = ("trend", "shock", "residual")
MONTHS_PER_QUARTER = 3
DECOMP01A_EXPERIMENT_ID = "DECOMP-01A-transition-channel-decomposition"
DECOMP01B_EXPERIMENT_ID = "DECOMP-01B-channel-driver-coupling"
DECOMP01C_EXPERIMENT_ID = "DECOMP-01C-fused-mechanistic-forecast"
DECOMP01D_EXPERIMENT_ID = "DECOMP-01D-skill-gated-fused-forecast"
DECOMP01E_EXPERIMENT_ID = "DECOMP-01E-loo-gated-fused-forecast"
DECOMP01F_EXPERIMENT_ID = "DECOMP-01F-reverse-grasp-peak-clusters"
MECH01B_EXPERIMENT_ID = "MECH-01B-mesoscopic-transition-helpers"
PEAK_CLUSTER_TRANSITIONS: tuple[str, ...] = ("D_to_A", "A_to_L", "L_to_A")
PEAK_CLUSTER_CHANNELS: tuple[str, ...] = ("shock", "residual")
QUARTER_END_MONTHS: dict[str, str] = {
    "Q1": "03",
    "Q2": "06",
    "Q3": "09",
    "Q4": "12",
}


def _quarter_end_month_label(quarter: str) -> str:
    year_text, quarter_text = str(quarter).split("-", maxsplit=1)
    if quarter_text not in QUARTER_END_MONTHS:
        raise ValueError(f"Unsupported quarter label: {quarter}")
    return f"{int(year_text):04d}-{QUARTER_END_MONTHS[quarter_text]}"


def _least_squares_projection(series: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis.size == 0 or series.size == 0:
        return np.zeros((basis.shape[-1],), dtype=np.float32), np.zeros_like(series, dtype=np.float32)
    coefficients, _, _, _ = np.linalg.lstsq(basis.astype(np.float32), series.astype(np.float32), rcond=None)
    coefficients = np.asarray(coefficients, dtype=np.float32)
    reconstruction = (basis.astype(np.float32) @ coefficients).astype(np.float32)
    return coefficients.astype(np.float32), reconstruction.astype(np.float32)


def _mech_01e_hazard_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    summary = dict(reference.get("transition_hazard_summary") or {})
    train_rows = [dict(row) for row in list(summary.get("train_rows") or [])]
    holdout_quarters = [str(value) for value in list((reference.get("evaluation") or {}).get("holdout_quarters") or []) if str(value)]
    holdout_lookup = {
        str(row.get("quarter") or ""): {
            transition: float((row.get("hazards") or {}).get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
        for row in list(summary.get("holdout_rows") or [])
    }
    baseline_holdout_hazards = _reference_holdout_hazards(reference)
    rows: list[dict[str, Any]] = []
    seen_quarters: set[str] = set()
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        if not quarter or quarter in seen_quarters:
            continue
        seen_quarters.add(quarter)
        rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: float((row.get("hazards") or {}).get(transition) or 0.0) for transition in TRANSITION_NAMES},
                "source": "train_transition_reconstruction",
            }
        )
    for quarter in holdout_quarters:
        if quarter in seen_quarters:
            continue
        seen_quarters.add(quarter)
        hazard_map = holdout_lookup.get(quarter) or baseline_holdout_hazards.get(quarter)
        if not hazard_map:
            raise ValueError(f"MECH-01E decomposition requires holdout hazards for {quarter}")
        rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: float(hazard_map.get(transition) or 0.0) for transition in TRANSITION_NAMES},
                "source": "mech_01e_holdout_anchor" if quarter not in holdout_lookup else "mech_01e_holdout_forecast",
            }
        )
    return rows


def _discover_latest_decomp_experiment(experiment_id: str, required_path: str) -> tuple[str, Any]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Any]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / experiment_id
        required_artifact = experiment_dir / required_path
        if required_artifact.exists():
            candidates.append((required_artifact.stat().st_mtime, run_dir.name, experiment_dir))
    if not candidates:
        raise FileNotFoundError(f"No transition research experiment found for {experiment_id}")
    preferred = [row for row in candidates if "pytest" not in str(row[1]).lower()]
    pool = preferred or candidates
    _, run_id, experiment_dir = max(pool, key=lambda row: row[0])
    return run_id, experiment_dir


def _load_decomp_01a_reference() -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_decomp_experiment(
        DECOMP01A_EXPERIMENT_ID,
        "channel_summary.json",
    )
    channel_summary = read_json(experiment_dir / "channel_summary.json", default={})
    tensor = np.asarray(load_tensor_artifact(experiment_dir / "channel_decomposition.npz"), dtype=np.float32)
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "channel_summary": dict(channel_summary or {}),
        "channel_tensor": tensor,
    }


def _load_decomp_01b_reference() -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_decomp_experiment(
        DECOMP01B_EXPERIMENT_ID,
        "channel_driver_weights.json",
    )
    payload = read_json(experiment_dir / "channel_driver_weights.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "payload": dict(payload or {}),
    }


def _aligned_factor_channel_components(
    *,
    factor_series: np.ndarray,
    slow_basis: np.ndarray,
    active_shock_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    trend_coefficients, trend_component = _least_squares_projection(factor_series, slow_basis)
    shock_coefficients, shock_component = _least_squares_projection(
        factor_series - trend_component,
        active_shock_basis,
    )
    residual_component = (factor_series - trend_component - shock_component).astype(np.float32)
    return {
        "trend": trend_component.astype(np.float32),
        "shock": shock_component.astype(np.float32),
        "residual": residual_component.astype(np.float32),
        "trend_coefficients": trend_coefficients.astype(np.float32),
        "shock_coefficients": shock_coefficients.astype(np.float32),
    }


def _quarter_index_lookup(quarter_axis: list[str]) -> dict[str, int]:
    return {str(quarter): index for index, quarter in enumerate(quarter_axis)}


def _driver_component_lookup(
    *,
    ctx: TransitionResearchContext,
    quarter_axis: list[str],
    month_axis: list[str],
    slow_knot_quarters: int,
    medium_block_quarters: int,
    regime_rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    factor_surface = _quarter_factor_surface(ctx)
    temporal_basis = build_temporal_basis(
        month_axis,
        slow_knot_months=slow_knot_quarters,
        medium_block_months=medium_block_quarters,
    )
    slow_basis = np.asarray(temporal_basis["slow_basis"], dtype=np.float32)
    shock_basis = build_shock_regime_basis(month_axis, regime_rows)
    shock_basis_matrix = np.asarray(shock_basis["basis"], dtype=np.float32)
    factor_components: dict[str, dict[str, np.ndarray]] = {}
    for factor_id in sorted(factor_surface):
        factor_series = np.asarray(
            [float((factor_surface.get(factor_id) or {}).get(quarter) or 0.0) for quarter in quarter_axis],
            dtype=np.float32,
        )
        factor_components[factor_id] = _aligned_factor_channel_components(
            factor_series=factor_series,
            slow_basis=slow_basis,
            active_shock_basis=shock_basis_matrix,
        )
    return factor_components


def _transition_channel_lookup(
    channel_tensor: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        transition: {
            channel_name: np.asarray(channel_tensor[transition_idx, :, channel_idx], dtype=np.float32)
            for channel_idx, channel_name in enumerate(CHANNEL_NAMES)
        }
        for transition_idx, transition in enumerate(TRANSITION_NAMES)
    }


def _supported_driver_rows_by_pair(driver_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in driver_rows:
        if not bool(row.get("retained")):
            continue
        key = (str(row.get("transition") or ""), str(row.get("channel") or ""))
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def _empirical_transition_upper_bounds(
    *,
    train_transition_rows: list[dict[str, Any]],
    baseline_holdout_hazards: dict[str, dict[str, float]],
) -> dict[str, float]:
    bounds: dict[str, float] = {}
    for transition in TRANSITION_NAMES:
        observed = [
            float((row.get("hazards") or {}).get(transition) or 0.0)
            for row in train_transition_rows
        ]
        observed.extend(
            float((hazard_map or {}).get(transition) or 0.0)
            for hazard_map in baseline_holdout_hazards.values()
        )
        bounds[transition] = max(observed) if observed else 0.0
    return bounds


def _train_channel_model(
    *,
    train_driver_signal: np.ndarray,
    train_channel_values: np.ndarray,
    eps: float,
) -> dict[str, float]:
    if train_driver_signal.size == 0 or train_channel_values.size == 0:
        return {
            "intercept": 0.0,
            "driver_scale": 0.0,
            "alignment_correlation": 0.0,
            "blend_weight": 0.0,
            "train_driver_mean": 0.0,
            "train_driver_std": 0.0,
        }
    design = np.asarray(
        [[1.0, float(value)] for value in train_driver_signal.tolist()],
        dtype=np.float64,
    )
    target = np.asarray(train_channel_values.tolist(), dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    intercept = float(coefficients[0]) if coefficients.size else 0.0
    driver_scale = float(coefficients[1]) if coefficients.size > 1 else 0.0
    fitted = (design @ coefficients).astype(np.float64)
    residuals = target - fitted
    target_variance = float(np.var(target)) if target.size else 0.0
    residual_mse = float(np.mean(np.square(residuals))) if residuals.size else 0.0
    train_r2 = max(0.0, 1.0 - (residual_mse / max(target_variance, eps))) if target_variance > eps else 0.0
    alignment_correlation = float(
        _pearson_corr(
            x_values=[float(value) for value in train_driver_signal.tolist()],
            y_values=[float(value) for value in train_channel_values.tolist()],
            eps=eps,
        )
    )
    blend_weight = float(abs(alignment_correlation) * abs(alignment_correlation))
    return {
        "intercept": intercept,
        "driver_scale": driver_scale,
        "alignment_correlation": alignment_correlation,
        "blend_weight": blend_weight,
        "train_r2": float(train_r2),
        "train_residual_mse": float(residual_mse),
        "train_driver_mean": float(np.mean(train_driver_signal)),
        "train_driver_std": float(np.std(train_driver_signal)),
    }


def _leave_one_out_channel_r2(
    *,
    train_driver_signal: np.ndarray,
    train_channel_values: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    sample_count = int(train_driver_signal.size)
    if sample_count < 3 or train_channel_values.size != train_driver_signal.size:
        return 0.0, 0.0
    predictions: list[float] = []
    targets: list[float] = []
    for holdout_index in range(sample_count):
        train_mask = np.ones((sample_count,), dtype=bool)
        train_mask[holdout_index] = False
        design = np.asarray(
            [[1.0, float(value)] for value in train_driver_signal[train_mask].tolist()],
            dtype=np.float64,
        )
        target = np.asarray(train_channel_values[train_mask].tolist(), dtype=np.float64)
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        holdout_prediction = float(coefficients[0]) + (float(coefficients[1]) * float(train_driver_signal[holdout_index]))
        predictions.append(holdout_prediction)
        targets.append(float(train_channel_values[holdout_index]))
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    residual_mse = float(np.mean(np.square(target_array - prediction_array))) if target_array.size else 0.0
    target_variance = float(np.var(target_array)) if target_array.size else 0.0
    loo_r2 = max(0.0, 1.0 - (residual_mse / max(target_variance, eps))) if target_variance > eps else 0.0
    return float(loo_r2), float(residual_mse)


def _lag1_persistence(values: np.ndarray, eps: float) -> float:
    if values.size < 3:
        return 0.0
    x_values = np.asarray(values[:-1], dtype=np.float64)
    y_values = np.asarray(values[1:], dtype=np.float64)
    persistence = _pearson_corr(
        x_values=[float(value) for value in x_values.tolist()],
        y_values=[float(value) for value in y_values.tolist()],
        eps=eps,
    )
    return float(np.clip(persistence, 0.0, 1.0))


def _reverse_cluster_pressure(cluster_input: np.ndarray, persistence: float) -> np.ndarray:
    pressure = np.zeros_like(cluster_input, dtype=np.float32)
    running = 0.0
    for index in range(int(cluster_input.shape[0]) - 1, -1, -1):
        running = float(cluster_input[index]) + (float(persistence) * running)
        pressure[index] = float(running)
    return pressure.astype(np.float32)


def _slope_only_fit(x_values: np.ndarray, y_values: np.ndarray, eps: float) -> tuple[float, float]:
    if x_values.size == 0 or y_values.size == 0 or x_values.size != y_values.size:
        return 0.0, 0.0
    numerator = float(np.dot(x_values.astype(np.float64), y_values.astype(np.float64)))
    denominator = float(np.dot(x_values.astype(np.float64), x_values.astype(np.float64)))
    if denominator <= eps:
        return 0.0, 0.0
    slope = numerator / denominator
    fitted = slope * x_values.astype(np.float64)
    residual_mse = float(np.mean(np.square(y_values.astype(np.float64) - fitted))) if y_values.size else 0.0
    return float(slope), float(residual_mse)


def _leave_one_out_slope_only_r2(x_values: np.ndarray, y_values: np.ndarray, eps: float) -> tuple[float, float]:
    sample_count = int(x_values.size)
    if sample_count < 3 or y_values.size != x_values.size:
        return 0.0, 0.0
    predictions: list[float] = []
    targets: list[float] = []
    for holdout_index in range(sample_count):
        train_mask = np.ones((sample_count,), dtype=bool)
        train_mask[holdout_index] = False
        slope, _ = _slope_only_fit(x_values[train_mask], y_values[train_mask], eps)
        predictions.append(float(slope * float(x_values[holdout_index])))
        targets.append(float(y_values[holdout_index]))
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    residual_mse = float(np.mean(np.square(target_array - prediction_array))) if target_array.size else 0.0
    target_variance = float(np.var(target_array)) if target_array.size else 0.0
    loo_r2 = max(0.0, 1.0 - (residual_mse / max(target_variance, eps))) if target_variance > eps else 0.0
    return float(loo_r2), float(residual_mse)


def _simulate_holdout_from_hazard_map(
    *,
    holdout_rows: list[dict[str, Any]],
    hazard_map: dict[str, dict[str, float]],
    testing_share_mean: float,
    current_state_override: dict[str, float],
    probability_lower_bound: float,
    probability_upper_bound: float,
    eps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_state = {state_name: float(current_state_override[state_name]) for state_name in STATE_NAMES}
    forecast_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for target_row in holdout_rows:
        quarter = str(target_row["quarter"])
        predicted_hazards = {
            transition: float(
                np.clip(
                    float((hazard_map.get(quarter) or {}).get(transition) or 0.0),
                    probability_lower_bound,
                    probability_upper_bound,
                )
            )
            for transition in TRANSITION_NAMES
        }
        u_to_d = min(float(current_state["U"]), predicted_hazards["U_to_D"] * float(current_state["U"]))
        d_to_a = min(float(current_state["D"]), predicted_hazards["D_to_A"] * float(current_state["D"]))
        l_to_a = min(float(current_state["L"]), predicted_hazards["L_to_A"] * float(current_state["L"]))
        raw_a_to_v = predicted_hazards["A_to_V"] * float(current_state["A"])
        raw_a_to_l = predicted_hazards["A_to_L"] * float(current_state["A"])
        raw_outflow = raw_a_to_v + raw_a_to_l
        a_scale = min(probability_upper_bound, float(current_state["A"]) / max(raw_outflow, eps)) if raw_outflow > 0.0 else probability_upper_bound
        a_to_v = raw_a_to_v * a_scale
        a_to_l = raw_a_to_l * a_scale
        next_state = {
            "U": max(float(current_state["U"]) - u_to_d, probability_lower_bound),
            "D": max(float(current_state["D"]) + u_to_d - d_to_a, probability_lower_bound),
            "A": max(float(current_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l, probability_lower_bound),
            "V": max(float(current_state["V"]) + a_to_v, probability_lower_bound),
            "L": max(float(current_state["L"]) + a_to_l - l_to_a, probability_lower_bound),
        }
        forecast_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(next_state["D"] + next_state["A"] + next_state["V"] + next_state["L"]),
                "alive_on_art": float(next_state["A"] + next_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": float(testing_share_mean * (next_state["A"] + next_state["V"])),
                "virally_suppressed": float(next_state["V"]),
                "state_values": {state_name: float(next_state[state_name]) for state_name in STATE_NAMES},
            }
        )
        hazard_rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: round(float(predicted_hazards[transition]), 6) for transition in TRANSITION_NAMES},
                "flows": {
                    "U_to_D": round(float(u_to_d), 6),
                    "D_to_A": round(float(d_to_a), 6),
                    "A_to_V": round(float(a_to_v), 6),
                    "A_to_L": round(float(a_to_l), 6),
                    "L_to_A": round(float(l_to_a), 6),
                },
            }
        )
        current_state = next_state
    return forecast_rows, hazard_rows


def run_decomp_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    mech_01e_reference = _load_mech_01e_reference()
    plugin = get_disease_plugin(ctx.run_context.plugin_id)
    phase3_priors = dict(plugin.prior_hyperparameters.get("phase3") or {})
    temporal_cfg = dict(phase3_priors.get("temporal_decomposition") or {})
    shock_cfg = dict(phase3_priors.get("shock_regimes") or {})
    regime_rows = [dict(row) for row in list(shock_cfg.get("regimes") or [])]

    hazard_rows = _mech_01e_hazard_rows(mech_01e_reference)
    quarter_axis = [str(row["quarter"]) for row in hazard_rows]
    month_axis = [_quarter_end_month_label(quarter) for quarter in quarter_axis]
    slow_knot_months = int(temporal_cfg.get("slow_knot_months") or 0)
    medium_block_months = int(temporal_cfg.get("medium_block_months") or 0)
    slow_knot_quarters = max(1, int(ceil(float(slow_knot_months) / float(MONTHS_PER_QUARTER))))
    medium_block_quarters = max(1, int(ceil(float(medium_block_months) / float(MONTHS_PER_QUARTER))))

    temporal_basis = build_temporal_basis(
        month_axis,
        slow_knot_months=slow_knot_quarters,
        medium_block_months=medium_block_quarters,
    )
    shock_basis = build_shock_regime_basis(month_axis, regime_rows)
    slow_basis = np.asarray(temporal_basis["slow_basis"], dtype=np.float32)
    shock_basis_matrix = np.asarray(shock_basis["basis"], dtype=np.float32)

    channel_tensor = np.zeros((len(TRANSITION_NAMES), len(quarter_axis), len(CHANNEL_NAMES)), dtype=np.float32)
    channel_summary_rows: list[dict[str, Any]] = []
    for transition_idx, transition in enumerate(TRANSITION_NAMES):
        series = np.asarray([float(row["hazards"][transition]) for row in hazard_rows], dtype=np.float32)
        trend_coefficients, trend_component = _least_squares_projection(series, slow_basis)
        active_regime_indices = [
            regime_idx
            for regime_idx, regime_row in enumerate(regime_rows)
            if transition in list(regime_row.get("transition_names") or [])
        ]
        active_shock_basis = shock_basis_matrix[:, active_regime_indices] if active_regime_indices else np.zeros((len(quarter_axis), 0), dtype=np.float32)
        shock_coefficients, shock_component = _least_squares_projection(series - trend_component, active_shock_basis)
        residual_component = (series - trend_component - shock_component).astype(np.float32)
        reconstruction = trend_component + shock_component + residual_component
        reconstruction_error = np.abs(series - reconstruction).astype(np.float32)

        channel_tensor[transition_idx, :, 0] = trend_component
        channel_tensor[transition_idx, :, 1] = shock_component
        channel_tensor[transition_idx, :, 2] = residual_component
        channel_summary_rows.append(
            {
                "transition": transition,
                "active_regime_names": [str(regime_rows[index].get("name") or f"regime_{index}") for index in active_regime_indices],
                "trend_coefficient_count": int(trend_coefficients.shape[0]),
                "shock_coefficient_count": int(shock_coefficients.shape[0]),
                "mean_absolute_trend_component": round(float(np.mean(np.abs(trend_component))) if trend_component.size else 0.0, 6),
                "mean_absolute_shock_component": round(float(np.mean(np.abs(shock_component))) if shock_component.size else 0.0, 6),
                "mean_absolute_residual_component": round(float(np.mean(np.abs(residual_component))) if residual_component.size else 0.0, 6),
                "mean_absolute_reconstruction_error": round(float(np.mean(reconstruction_error)) if reconstruction_error.size else 0.0, 10),
                "max_absolute_reconstruction_error": round(float(np.max(reconstruction_error)) if reconstruction_error.size else 0.0, 10),
            }
        )

    tensor_artifact = save_tensor_artifact(
        array=channel_tensor,
        axis_names=["transition", "quarter", "channel"],
        artifact_dir=ctx.experiment_dir,
        stem="channel_decomposition",
        backend="numpy",
        device="cpu",
        notes=["DECOMP-01A trend/shock/residual channelization of the kept MECH-01E transition hazard path"],
        save_pt=False,
    )
    reconstruction_errors = [
        float(row["max_absolute_reconstruction_error"])
        for row in channel_summary_rows
    ]
    float32_eps = float(np.finfo(np.float32).eps)
    channel_summary = {
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "mech_01e_reference_experiment_id": MECH01E_EXPERIMENT_ID,
        "quarter_axis": quarter_axis,
        "month_axis": month_axis,
        "channel_names": list(CHANNEL_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "temporal_basis_summary": {
            "slow_knot_months": slow_knot_months,
            "medium_block_months": medium_block_months,
            "slow_knot_quarters": slow_knot_quarters,
            "medium_block_quarters": medium_block_quarters,
            "slow_basis_shape": list(slow_basis.shape),
            "shock_basis_shape": list(shock_basis_matrix.shape),
            "shock_regime_names": list(shock_basis.get("names") or []),
        },
        "rows": channel_summary_rows,
        "tensor_artifact": tensor_artifact,
    }
    decision_checks = [
        {
            "name": "all_channel_values_finite",
            "passed": bool(np.isfinite(channel_tensor).all()),
            "actual": bool(np.isfinite(channel_tensor).all()),
            "target": True,
        },
        {
            "name": "reconstruction_error_within_float32_guard",
            "passed": bool(all(error <= float32_eps for error in reconstruction_errors)),
            "actual": round(float(max(reconstruction_errors) if reconstruction_errors else 0.0), 10),
            "target": round(float32_eps, 10),
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "DECOMP-01A decomposes the kept MECH-01E hazard path into trend, shock, and residual channels using plugin-backed temporal priors.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "channel_summary.json", channel_summary)

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        },
        coverage_summary={
            "quarter_count": len(quarter_axis),
            "transition_count": len(TRANSITION_NAMES),
            "channel_count": len(CHANNEL_NAMES),
            "shock_regime_count": len(regime_rows),
            "active_shock_transition_count": sum(1 for row in channel_summary_rows if len(list(row["active_regime_names"])) > 0),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "months_per_quarter",
                "value": MONTHS_PER_QUARTER,
                "role": "calendar_resolution_mapping",
                "source_type": "physical_constraint",
                "estimation_data": "calendar convention",
                "estimation_method": "fixed conversion from quarter cadence to month cadence for plugin-backed temporal priors",
                "uncertainty": "none",
                "why_needed": "Converts the plugin's month-scale temporal priors into quarter-scale basis sizes without inventing new decomposition horizons.",
            },
            {
                "name": "slow_knot_months",
                "value": slow_knot_months,
                "role": "trend_basis_prior_horizon",
                "source_type": "bayesian_prior",
                "estimation_data": "HIV plugin phase3 temporal_decomposition prior",
                "estimation_method": "reuse plugin slow_knot_months directly from the disease plugin",
                "uncertainty": "inherits plugin prior uncertainty",
                "why_needed": "Anchors the slow trend channel to the existing plugin temporal prior rather than a manual knot spacing.",
            },
            {
                "name": "medium_block_months",
                "value": medium_block_months,
                "role": "medium_basis_prior_horizon",
                "source_type": "bayesian_prior",
                "estimation_data": "HIV plugin phase3 temporal_decomposition prior",
                "estimation_method": "reuse plugin medium_block_months directly from the disease plugin",
                "uncertainty": "inherits plugin prior uncertainty",
                "why_needed": "Keeps the quarterized temporal basis commensurate with the existing HIV temporal prior.",
            },
            {
                "name": "slow_knot_quarters",
                "value": slow_knot_quarters,
                "role": "quarterized_trend_basis_horizon",
                "source_type": "physical_constraint",
                "estimation_data": "slow_knot_months and months_per_quarter",
                "estimation_method": "ceiling division of plugin slow_knot_months by the fixed months-per-quarter calendar mapping",
                "uncertainty": "none",
                "why_needed": "Transforms the plugin month prior into an exact quarter-grid trend basis for the national hazard series.",
            },
            {
                "name": "medium_block_quarters",
                "value": medium_block_quarters,
                "role": "quarterized_medium_basis_horizon",
                "source_type": "physical_constraint",
                "estimation_data": "medium_block_months and months_per_quarter",
                "estimation_method": "ceiling division of plugin medium_block_months by the fixed months-per-quarter calendar mapping",
                "uncertainty": "none",
                "why_needed": "Keeps the quarter-grid basis aligned with the existing plugin temporal prior scale.",
            },
            {
                "name": "shock_regime_count",
                "value": len(regime_rows),
                "role": "shock_channel_basis_cardinality",
                "source_type": "bayesian_prior",
                "estimation_data": "HIV plugin phase3 shock_regimes prior",
                "estimation_method": "count plugin-defined shock regimes active in the phase3 temporal configuration",
                "uncertainty": "inherits plugin prior uncertainty",
                "why_needed": "Defines the shock channel basis directly from the plugin regime catalog rather than an arbitrary event count.",
            },
            {
                "name": "channel_count",
                "value": len(CHANNEL_NAMES),
                "role": "decomposition_channel_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research plan",
                "estimation_method": "count of the requested HIV hazard channels: trend, shock, residual",
                "uncertainty": "none",
                "why_needed": "Fixes the decomposition layout to the exact plan-approved channels.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="reconstruction_guard",
                why_needed="Bounds acceptable decomposition reconstruction error at floating-point precision because the residual channel is defined as the exact leftover.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "channel_summary": channel_summary,
        "decision": decision,
    }


def run_decomp_01b(ctx: TransitionResearchContext) -> dict[str, Any]:
    decomp_01a_reference = _load_decomp_01a_reference()
    channel_summary = dict(decomp_01a_reference.get("channel_summary") or {})
    channel_tensor = np.asarray(decomp_01a_reference.get("channel_tensor"), dtype=np.float32)
    quarter_axis = [str(value) for value in list(channel_summary.get("quarter_axis") or [])]
    month_axis = [str(value) for value in list(channel_summary.get("month_axis") or [])]
    temporal_basis_summary = dict(channel_summary.get("temporal_basis_summary") or {})
    slow_knot_quarters = int(temporal_basis_summary.get("slow_knot_quarters") or 0)
    medium_block_quarters = int(temporal_basis_summary.get("medium_block_quarters") or 0)

    plugin = get_disease_plugin(ctx.run_context.plugin_id)
    phase3_priors = dict(plugin.prior_hyperparameters.get("phase3") or {})
    shock_cfg = dict(phase3_priors.get("shock_regimes") or {})
    regime_rows = [dict(row) for row in list(shock_cfg.get("regimes") or [])]
    temporal_basis = build_temporal_basis(
        month_axis,
        slow_knot_months=slow_knot_quarters,
        medium_block_months=medium_block_quarters,
    )
    slow_basis = np.asarray(temporal_basis["slow_basis"], dtype=np.float32)
    shock_basis = build_shock_regime_basis(month_axis, regime_rows)
    shock_basis_matrix = np.asarray(shock_basis["basis"], dtype=np.float32)

    inputs = load_transition_research_inputs(ctx)
    factor_surface = _quarter_factor_surface(ctx)
    factor_metrics = _factor_static_metrics(inputs)
    transition_relevance_rows = _factor_transition_rows(inputs)
    factor_ids = sorted(inputs.retained_factor_lookup)
    transition_relevance_lookup = {
        (str(row["factor_id"]), str(row["transition"])): float(row["relevance"])
        for row in transition_relevance_rows
    }
    float32_eps = float(np.finfo(np.float32).eps)
    driver_rows: list[dict[str, Any]] = []
    pair_summary_rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(TRANSITION_NAMES) * len(CHANNEL_NAMES), len(factor_ids)), dtype=np.float32)
    row_labels: list[str] = []

    for transition_idx, transition in enumerate(TRANSITION_NAMES):
        active_regime_indices = [
            regime_idx
            for regime_idx, regime_row in enumerate(regime_rows)
            if transition in list(regime_row.get("transition_names") or [])
        ]
        active_shock_basis = shock_basis_matrix[:, active_regime_indices] if active_regime_indices else np.zeros((len(quarter_axis), 0), dtype=np.float32)
        hazard_channel_lookup = {
            channel_name: np.asarray(channel_tensor[transition_idx, :, channel_idx], dtype=np.float32)
            for channel_idx, channel_name in enumerate(CHANNEL_NAMES)
        }
        factor_component_lookup: dict[str, dict[str, np.ndarray]] = {}
        for factor_id in factor_ids:
            factor_series = np.asarray([float((factor_surface.get(factor_id) or {}).get(quarter) or 0.0) for quarter in quarter_axis], dtype=np.float32)
            factor_component_lookup[factor_id] = _aligned_factor_channel_components(
                factor_series=factor_series,
                slow_basis=slow_basis,
                active_shock_basis=active_shock_basis,
            )
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            label = f"{transition}|{channel_name}"
            row_labels.append(label)
            pair_candidates: list[dict[str, Any]] = []
            for factor_id in factor_ids:
                transition_relevance = float(transition_relevance_lookup.get((factor_id, transition), 0.0))
                if transition_relevance <= 0.0:
                    continue
                factor_components = factor_component_lookup[factor_id]
                factor_channel = np.asarray(factor_components[channel_name], dtype=np.float32)
                hazard_channel = np.asarray(hazard_channel_lookup[channel_name], dtype=np.float32)
                channel_correlation = _pearson_corr(
                    x_values=[float(value) for value in factor_channel.tolist()],
                    y_values=[float(value) for value in hazard_channel.tolist()],
                    eps=float32_eps,
                )
                channel_energy = float(np.mean(np.abs(factor_channel))) if factor_channel.size else 0.0
                structural_strength = float((factor_metrics.get(factor_id) or {}).get("structural_strength") or 0.0)
                pre_weight = transition_relevance * structural_strength * abs(channel_correlation) * channel_energy
                pair_candidates.append(
                    {
                        "factor_id": factor_id,
                        "factor_name": str((inputs.retained_factor_lookup.get(factor_id) or {}).get("factor_name") or factor_id),
                        "transition_relevance": transition_relevance,
                        "structural_strength": structural_strength,
                        "channel_correlation": float(channel_correlation),
                        "channel_energy": float(channel_energy),
                        "pre_weight": float(pre_weight),
                        "trend_coefficient_count": int(factor_components["trend_coefficients"].shape[0]),
                        "shock_coefficient_count": int(factor_components["shock_coefficients"].shape[0]),
                    }
                )

            positive_pre_weights = [float(row["pre_weight"]) for row in pair_candidates if float(row["pre_weight"]) > 0.0]
            pair_threshold = float(np.mean(positive_pre_weights)) if positive_pre_weights else 0.0
            retained_candidates = [
                row for row in pair_candidates
                if float(row["pre_weight"]) > 0.0 and float(row["pre_weight"]) >= pair_threshold
            ]
            retained_weight_total = float(sum(float(row["pre_weight"]) for row in retained_candidates))
            supported = bool(retained_candidates)
            unsupported_reason = ""
            if not pair_candidates:
                unsupported_reason = "no_transition_aligned_factor_candidates"
            elif not positive_pre_weights:
                unsupported_reason = "no_positive_channel_alignment_signal"
            elif not retained_candidates:
                unsupported_reason = "no_driver_above_empirical_pair_mean"

            pair_summary_rows.append(
                {
                    "transition": transition,
                    "channel": channel_name,
                    "candidate_count": len(pair_candidates),
                    "positive_pre_weight_count": len(positive_pre_weights),
                    "retained_driver_count": len(retained_candidates),
                    "pair_threshold": round(float(pair_threshold), 10),
                    "supported": supported,
                    "unsupported_reason": unsupported_reason,
                    "active_regime_names": [str(regime_rows[index].get("name") or f"regime_{index}") for index in active_regime_indices] if channel_name == "shock" else [],
                }
            )
            retained_lookup = {str(row["factor_id"]): dict(row) for row in retained_candidates}
            for factor_idx, factor_id in enumerate(factor_ids):
                retained_row = retained_lookup.get(factor_id)
                normalized_weight = (
                    float(retained_row["pre_weight"]) / retained_weight_total
                    if retained_row is not None and retained_weight_total > float32_eps
                    else 0.0
                )
                matrix[(transition_idx * len(CHANNEL_NAMES)) + channel_idx, factor_idx] = float(normalized_weight)
                source_row = next((row for row in pair_candidates if str(row["factor_id"]) == factor_id), None)
                if source_row is None:
                    continue
                driver_rows.append(
                    {
                        "transition": transition,
                        "channel": channel_name,
                        "factor_id": factor_id,
                        "factor_name": str(source_row["factor_name"]),
                        "transition_relevance": round(float(source_row["transition_relevance"]), 6),
                        "structural_strength": round(float(source_row["structural_strength"]), 6),
                        "channel_correlation": round(float(source_row["channel_correlation"]), 6),
                        "channel_energy": round(float(source_row["channel_energy"]), 6),
                        "pre_weight": round(float(source_row["pre_weight"]), 10),
                        "pair_threshold": round(float(pair_threshold), 10),
                        "retained": bool(retained_row is not None),
                        "normalized_weight": round(float(normalized_weight), 6),
                        "trend_coefficient_count": int(source_row["trend_coefficient_count"]),
                        "shock_coefficient_count": int(source_row["shock_coefficient_count"]),
                    }
                )

    payload = {
        "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
        "decomp_01a_reference_experiment_id": DECOMP01A_EXPERIMENT_ID,
        "quarter_axis": quarter_axis,
        "transition_names": list(TRANSITION_NAMES),
        "channel_names": list(CHANNEL_NAMES),
        "factor_ids": factor_ids,
        "pair_summary_rows": pair_summary_rows,
        "driver_rows": driver_rows,
    }
    write_json(ctx.experiment_dir / "channel_driver_weights.json", payload)

    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title("DECOMP-01B Channel Driver Weights")
    ax.set_xticks(np.arange(len(factor_ids)), labels=factor_ids)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "channel_driver_heatmap.png")
    plt.close(fig)

    supported_pair_count = sum(1 for row in pair_summary_rows if bool(row["supported"]))
    unsupported_pair_count = len(pair_summary_rows) - supported_pair_count
    decision_checks = [
        {
            "name": "all_weight_values_finite",
            "passed": bool(np.isfinite(matrix).all()),
            "actual": bool(np.isfinite(matrix).all()),
            "target": True,
        },
        {
            "name": "all_pairs_explicitly_classified",
            "passed": supported_pair_count + unsupported_pair_count == len(pair_summary_rows),
            "actual": supported_pair_count + unsupported_pair_count,
            "target": len(pair_summary_rows),
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "DECOMP-01B couples aligned mesoscopic trend, shock, and residual factor components to the kept transition-channel decomposition.",
        "checks": decision_checks,
    }

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
        },
        coverage_summary={
            "transition_count": len(TRANSITION_NAMES),
            "channel_count": len(CHANNEL_NAMES),
            "factor_count": len(factor_ids),
            "supported_pair_count": supported_pair_count,
            "unsupported_pair_count": unsupported_pair_count,
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "factor_count",
                "value": len(factor_ids),
                "role": "driver_candidate_grid_cardinality",
                "source_type": "estimated",
                "estimation_data": "retained transition research factor set",
                "estimation_method": "count of retained Phase 1.5/2 factors available for channel-driver coupling",
                "uncertainty": "depends on retained factor registry",
                "why_needed": "Defines the exact candidate driver grid used to couple mesoscopic factors to decomposed transition channels.",
            },
            {
                "name": "transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "transition_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of exact HIV transition channels in the plan",
                "uncertainty": "none",
                "why_needed": "Defines the transition axis for the channel-driver matrix.",
            },
            {
                "name": "channel_count",
                "value": len(CHANNEL_NAMES),
                "role": "channel_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "DECOMP-01A decomposition contract",
                "estimation_method": "count of the aligned trend, shock, and residual channels",
                "uncertainty": "none",
                "why_needed": "Defines the channel axis for the driver matrix with no hidden decomposition family.",
            },
            {
                "name": "slow_knot_quarters",
                "value": slow_knot_quarters,
                "role": "aligned_factor_trend_basis_horizon",
                "source_type": "bayesian_prior",
                "estimation_data": "DECOMP-01A temporal basis summary inherited from the HIV plugin prior",
                "estimation_method": "reuse the quarterized slow basis horizon from the kept decomposition reference",
                "uncertainty": "inherits DECOMP-01A temporal prior uncertainty",
                "why_needed": "Ensures mesoscopic factors are decomposed in the same temporal basis as the kept transition hazards.",
            },
            {
                "name": "medium_block_quarters",
                "value": medium_block_quarters,
                "role": "aligned_factor_medium_basis_horizon",
                "source_type": "bayesian_prior",
                "estimation_data": "DECOMP-01A temporal basis summary inherited from the HIV plugin prior",
                "estimation_method": "reuse the quarterized medium basis horizon from the kept decomposition reference",
                "uncertainty": "inherits DECOMP-01A temporal prior uncertainty",
                "why_needed": "Keeps factor decomposition commensurate with the existing channel decomposition basis.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="correlation_and_normalization_guard",
                why_needed="Prevents undefined correlations and zero-denominator weight normalization for flat factor or channel series.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "payload": payload,
        "decision": decision,
    }


def run_decomp_01c(ctx: TransitionResearchContext) -> dict[str, Any]:
    decomp_01a_reference = _load_decomp_01a_reference()
    decomp_01b_reference = _load_decomp_01b_reference()
    mech_01e_reference = _load_mech_01e_reference()
    mech_01b_reference = _load_transition_experiment_reference(MECH01B_EXPERIMENT_ID)
    skill_gated = ctx.experiment.experiment_id == DECOMP01D_EXPERIMENT_ID
    loo_skill_gated = ctx.experiment.experiment_id == DECOMP01E_EXPERIMENT_ID
    reverse_grasp_peak = ctx.experiment.experiment_id == DECOMP01F_EXPERIMENT_ID
    branch_reference = (
        _load_transition_experiment_reference(DECOMP01E_EXPERIMENT_ID if reverse_grasp_peak else DECOMP01C_EXPERIMENT_ID)
        if skill_gated or loo_skill_gated or reverse_grasp_peak
        else None
    )
    experiment_label = "DECOMP-01C"
    if skill_gated:
        experiment_label = "DECOMP-01D"
    if loo_skill_gated:
        experiment_label = "DECOMP-01E"
    if reverse_grasp_peak:
        experiment_label = "DECOMP-01F"

    payload = _build_observation_payload(ctx)
    holdout_years = sorted(
        {
            int(str(quarter).split("-", maxsplit=1)[0])
            for quarter in list((mech_01e_reference.get("evaluation") or {}).get("holdout_quarters") or [])
            if str(quarter)
        }
    )
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or len(holdout_rows) < 2:
        raise ValueError("DECOMP-01C requires non-empty train rows and at least two holdout rows")

    fit_reference = dict(mech_01e_reference.get("fit_artifact") or {})
    coverage_mean = float(fit_reference.get("coverage_mean") or 0.0)
    suppression_share_mean = float(fit_reference.get("suppression_share_train_mean") or 0.0)
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    lost_gap_share = float(fit_reference.get("lost_gap_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)
    probability_lower_bound = 0.0
    probability_upper_bound = 1.0
    locked_transitions = {"U_to_D"}

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    baseline_holdout_hazards = _reference_holdout_hazards(mech_01e_reference)
    empirical_upper_bounds = _empirical_transition_upper_bounds(
        train_transition_rows=train_transition_rows,
        baseline_holdout_hazards=baseline_holdout_hazards,
    )

    channel_summary = dict(decomp_01a_reference.get("channel_summary") or {})
    channel_tensor = np.asarray(decomp_01a_reference.get("channel_tensor"), dtype=np.float32)
    quarter_axis = [str(value) for value in list(channel_summary.get("quarter_axis") or [])]
    month_axis = [str(value) for value in list(channel_summary.get("month_axis") or [])]
    temporal_basis_summary = dict(channel_summary.get("temporal_basis_summary") or {})
    slow_knot_quarters = int(temporal_basis_summary.get("slow_knot_quarters") or 0)
    medium_block_quarters = int(temporal_basis_summary.get("medium_block_quarters") or 0)
    plugin = get_disease_plugin(ctx.run_context.plugin_id)
    phase3_priors = dict(plugin.prior_hyperparameters.get("phase3") or {})
    regime_rows = [dict(row) for row in list((phase3_priors.get("shock_regimes") or {}).get("regimes") or [])]
    transition_channels = _transition_channel_lookup(channel_tensor)
    factor_components = _driver_component_lookup(
        ctx=ctx,
        quarter_axis=quarter_axis,
        month_axis=month_axis,
        slow_knot_quarters=slow_knot_quarters,
        medium_block_quarters=medium_block_quarters,
        regime_rows=regime_rows,
    )
    quarter_index = _quarter_index_lookup(quarter_axis)
    driver_payload = dict(decomp_01b_reference.get("payload") or {})
    driver_rows = [dict(row) for row in list(driver_payload.get("driver_rows") or [])]
    pair_summary_rows = [dict(row) for row in list(driver_payload.get("pair_summary_rows") or [])]
    retained_driver_lookup = _supported_driver_rows_by_pair(driver_rows)
    pair_support_lookup = {
        (str(row.get("transition") or ""), str(row.get("channel") or "")): bool(row.get("supported"))
        for row in pair_summary_rows
    }

    train_quarters = [str(row["quarter"]) for row in train_rows]
    holdout_quarters = [str(row["quarter"]) for row in holdout_rows]
    holdout_simulation_quarters = holdout_quarters[1:]
    mech_01e_holdout_states = _holdout_state_rows_from_reference(mech_01e_reference, reference_label="MECH-01E reference")
    mech_01e_holdout_predictions = _holdout_predictions_from_reference(mech_01e_reference)
    anchored_state_row = dict(mech_01e_holdout_states[0])
    anchored_prediction_row = dict(mech_01e_holdout_predictions[0])
    anchored_forecast_row = {
        "quarter": str(anchored_state_row["quarter"]),
        "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
        "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
        "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
        "tested_for_viral_load": float(testing_share_mean * (float(anchored_state_row["state_values"]["A"]) + float(anchored_state_row["state_values"]["V"]))),
        "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
        "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    }

    hazard_reconstruction_rows: list[dict[str, Any]] = []
    channel_model_rows: list[dict[str, Any]] = []
    fused_holdout_hazard_map: dict[str, dict[str, float]] = {}
    for transition in TRANSITION_NAMES:
        for channel_name in CHANNEL_NAMES:
            supported = bool(pair_support_lookup.get((transition, channel_name), False))
            retained_rows = list(retained_driver_lookup.get((transition, channel_name), []))
            baseline_series = np.asarray(transition_channels[transition][channel_name], dtype=np.float32)
            driver_signal = np.zeros((len(quarter_axis),), dtype=np.float32)
            for row in retained_rows:
                factor_id = str(row.get("factor_id") or "")
                if not factor_id:
                    continue
                factor_component_lookup = dict(factor_components.get(factor_id) or {})
                factor_component = factor_component_lookup.get(channel_name)
                if factor_component is None:
                    factor_component = np.zeros((len(quarter_axis),), dtype=np.float32)
                factor_component = np.asarray(factor_component, dtype=np.float32)
                driver_signal += float(row.get("normalized_weight") or 0.0) * factor_component

            train_indices = [quarter_index[quarter] for quarter in train_quarters if quarter in quarter_index]
            train_driver_signal = np.asarray([float(driver_signal[index]) for index in train_indices], dtype=np.float32)
            train_channel_values = np.asarray([float(baseline_series[index]) for index in train_indices], dtype=np.float32)
            channel_model = _train_channel_model(
                train_driver_signal=train_driver_signal,
                train_channel_values=train_channel_values,
                eps=eps,
            )
            loo_r2, loo_residual_mse = _leave_one_out_channel_r2(
                train_driver_signal=train_driver_signal,
                train_channel_values=train_channel_values,
                eps=eps,
            )
            peak_persistence = 0.0
            peak_beta = 0.0
            peak_loo_r2 = 0.0
            peak_residual_mse = 0.0
            peak_adjustment_sign = 0.0
            if reverse_grasp_peak and transition in PEAK_CLUSTER_TRANSITIONS and channel_name in PEAK_CLUSTER_CHANNELS:
                signed_effect_series = np.asarray(float(channel_model["driver_scale"]) * driver_signal, dtype=np.float32)
                cluster_input = np.maximum(signed_effect_series, 0.0).astype(np.float32)
                peak_persistence = _lag1_persistence(train_channel_values, eps)
                reverse_pressure = _reverse_cluster_pressure(cluster_input, peak_persistence)
                train_reverse_pressure = np.asarray([float(reverse_pressure[index]) for index in train_indices], dtype=np.float32)
                mapped_train = np.asarray(
                    [
                        float(channel_model["intercept"]) + (float(channel_model["driver_scale"]) * float(train_driver_signal[idx]))
                        for idx in range(train_driver_signal.shape[0])
                    ],
                    dtype=np.float32,
                )
                peak_target = np.maximum(train_channel_values - mapped_train, 0.0).astype(np.float32)
                peak_beta, peak_residual_mse = _slope_only_fit(train_reverse_pressure, peak_target, eps)
                peak_loo_r2, _ = _leave_one_out_slope_only_r2(train_reverse_pressure, peak_target, eps)
            else:
                reverse_pressure = np.zeros((len(quarter_axis),), dtype=np.float32)
            channel_model_rows.append(
                {
                    "transition": transition,
                    "channel": channel_name,
                    "supported": supported,
                    "locked": transition in locked_transitions,
                    "retained_driver_count": len(retained_rows),
                    "intercept": round(float(channel_model["intercept"]), 6),
                    "driver_scale": round(float(channel_model["driver_scale"]), 6),
                    "alignment_correlation": round(float(channel_model["alignment_correlation"]), 6),
                    "blend_weight": round(float(channel_model["blend_weight"]), 6),
                    "train_r2": round(float(channel_model["train_r2"]), 6),
                    "train_residual_mse": round(float(channel_model["train_residual_mse"]), 10),
                    "loo_r2": round(float(loo_r2), 6),
                    "loo_residual_mse": round(float(loo_residual_mse), 10),
                    "peak_persistence": round(float(peak_persistence), 6),
                    "peak_beta": round(float(peak_beta), 8),
                    "peak_loo_r2": round(float(peak_loo_r2), 6),
                    "peak_residual_mse": round(float(peak_residual_mse), 10),
                    "train_driver_mean": round(float(channel_model["train_driver_mean"]), 6),
                    "train_driver_std": round(float(channel_model["train_driver_std"]), 6),
                }
            )

            for quarter in holdout_simulation_quarters:
                if quarter not in quarter_index:
                    continue
                baseline_channel = float(baseline_series[quarter_index[quarter]])
                driver_value = float(driver_signal[quarter_index[quarter]])
                mapped_channel = float(channel_model["intercept"]) + (float(channel_model["driver_scale"]) * driver_value)
                effective_blend_weight = float(channel_model["blend_weight"])
                if skill_gated:
                    effective_blend_weight = min(effective_blend_weight, max(0.0, float(channel_model["train_r2"])))
                if loo_skill_gated or reverse_grasp_peak:
                    effective_blend_weight = min(effective_blend_weight, max(0.0, float(loo_r2)))
                peak_adjustment = 0.0
                if reverse_grasp_peak and transition in PEAK_CLUSTER_TRANSITIONS and channel_name in PEAK_CLUSTER_CHANNELS:
                    peak_adjustment = float(max(0.0, peak_beta) * max(0.0, peak_loo_r2) * float(reverse_pressure[quarter_index[quarter]]))
                if transition in locked_transitions or not supported or not retained_rows:
                    fused_channel = baseline_channel
                else:
                    fused_channel = baseline_channel + (effective_blend_weight * (mapped_channel - baseline_channel)) + peak_adjustment
                fused_holdout_hazard_map.setdefault(quarter, {})
                fused_holdout_hazard_map[quarter].setdefault(transition, 0.0)
                fused_holdout_hazard_map[quarter][transition] += fused_channel
                hazard_reconstruction_rows.append(
                    {
                        "quarter": quarter,
                        "transition": transition,
                        "channel": channel_name,
                        "supported": supported,
                        "locked": transition in locked_transitions,
                        "baseline_channel": round(float(baseline_channel), 8),
                        "driver_signal": round(float(driver_value), 8),
                        "mapped_channel": round(float(mapped_channel), 8),
                        "fused_channel": round(float(fused_channel), 8),
                        "alignment_correlation": round(float(channel_model["alignment_correlation"]), 6),
                        "blend_weight": round(float(channel_model["blend_weight"]), 6),
                        "effective_blend_weight": round(float(effective_blend_weight), 6),
                        "train_r2": round(float(channel_model["train_r2"]), 6),
                        "loo_r2": round(float(loo_r2), 6),
                        "peak_persistence": round(float(peak_persistence), 6),
                        "peak_beta": round(float(peak_beta), 8),
                        "peak_loo_r2": round(float(peak_loo_r2), 6),
                        "peak_adjustment": round(float(peak_adjustment), 8),
                        "retained_driver_count": len(retained_rows),
                    }
                )

    fused_holdout_hazard_rows: list[dict[str, Any]] = []
    for quarter in holdout_simulation_quarters:
        fused_holdout_hazard_map.setdefault(quarter, {})
        fused_row = {"quarter": quarter, "hazards": {}, "base_hazards": {}, "upper_bounds": {}}
        for transition in TRANSITION_NAMES:
            base_hazard = float((baseline_holdout_hazards.get(quarter) or {}).get(transition) or 0.0)
            upper_bound = max(float(empirical_upper_bounds.get(transition) or 0.0), base_hazard)
            fused_hazard = float(fused_holdout_hazard_map.get(quarter, {}).get(transition, base_hazard))
            clipped_hazard = float(np.clip(fused_hazard, probability_lower_bound, upper_bound))
            fused_holdout_hazard_map[quarter][transition] = clipped_hazard
            fused_row["hazards"][transition] = round(float(clipped_hazard), 6)
            fused_row["base_hazards"][transition] = round(float(base_hazard), 6)
            fused_row["upper_bounds"][transition] = round(float(upper_bound), 6)
        fused_holdout_hazard_rows.append(fused_row)

    simulated_forecast_rows, simulated_holdout_transition_rows = _simulate_holdout_from_hazard_map(
        holdout_rows=holdout_rows[1:],
        hazard_map=fused_holdout_hazard_map,
        testing_share_mean=testing_share_mean,
        current_state_override={state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        probability_lower_bound=probability_lower_bound,
        probability_upper_bound=probability_upper_bound,
        eps=eps,
    )
    forecast_rows = [anchored_forecast_row] + simulated_forecast_rows
    holdout_state_rows = [
        {
            "quarter": str(anchored_state_row["quarter"]),
            "diagnosed_plhiv": float(anchored_forecast_row["diagnosed_plhiv"]),
            "alive_on_art": float(anchored_forecast_row["alive_on_art"]),
            "new_diagnosed_cases_period": float(anchored_forecast_row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(anchored_forecast_row["tested_for_viral_load"]),
            "virally_suppressed": float(anchored_forecast_row["virally_suppressed"]),
            "estimated_plhiv": float(sum(float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
    ] + [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(sum(float(row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in simulated_forecast_rows
    ]

    mech_01e_baseline = dict(mech_01e_reference.get("baseline_comparison") or {})
    mech_01b_baseline = dict(mech_01b_reference.get("baseline_comparison") or {})
    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01e_diagnosis_flow_mae, _ = _diagnosis_flow_mae(
        forecast_rows=mech_01e_holdout_predictions,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01b_diagnosis_flow_mae = float(mech_01b_baseline.get("diagnosis_flow_mean_absolute_error") or float("inf"))
    branch_reference_baseline = dict((branch_reference or {}).get("baseline_comparison") or {})
    branch_reference_diagnosis_flow_mae = float(
        branch_reference_baseline.get("diagnosis_flow_mean_absolute_error") or float("inf")
    )
    target_peak_quarter = max(
        [str(row["quarter"]) for row in holdout_rows],
        key=lambda quarter: float(next(row for row in holdout_rows if str(row["quarter"]) == quarter)["alive_on_art"]),
    )
    model_peak_alive_on_art_error = abs(
        float(next(row for row in forecast_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
        - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
    )
    branch_reference_peak_alive_on_art_error = float("inf")
    if branch_reference is not None:
        branch_reference_holdout_predictions = _holdout_predictions_from_reference(branch_reference)
        branch_reference_peak_alive_on_art_error = abs(
            float(next(row for row in branch_reference_holdout_predictions if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
            - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
        )
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01e_reference_mean_absolute_error": float(mech_01e_baseline.get("model_mean_absolute_error") or 0.0),
        "mech_01b_reference_mean_absolute_error": float(mech_01b_baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(mech_01e_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(mech_01e_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01e_reference": model_mae < float(mech_01e_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_mech_01b_reference": model_mae < float(mech_01b_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(mech_01e_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(mech_01e_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01e_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01e_diagnosis_flow_mae), 6),
        "mech_01b_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01b_diagnosis_flow_mae), 6),
        "model_beats_mech_01b_on_mae_or_diagnosis_flow": (
            model_mae < float(mech_01b_baseline.get("model_mean_absolute_error") or float("inf"))
            or float(diagnosis_flow_mae) < float(mech_01b_diagnosis_flow_mae)
        ),
    }
    if skill_gated:
        baseline_comparison["decomp_01c_reference_mean_absolute_error"] = float(
            branch_reference_baseline.get("model_mean_absolute_error") or 0.0
        )
        baseline_comparison["decomp_01c_reference_diagnosis_flow_mean_absolute_error"] = round(
            float(branch_reference_diagnosis_flow_mae), 6
        )
        baseline_comparison["model_beats_decomp_01c_on_mae_or_diagnosis_flow"] = (
            model_mae < float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf"))
            or float(diagnosis_flow_mae) < float(branch_reference_diagnosis_flow_mae)
        )
    if loo_skill_gated:
        baseline_comparison["decomp_01c_reference_mean_absolute_error"] = float(
            branch_reference_baseline.get("model_mean_absolute_error") or 0.0
        )
        baseline_comparison["decomp_01c_reference_diagnosis_flow_mean_absolute_error"] = round(
            float(branch_reference_diagnosis_flow_mae), 6
        )
        baseline_comparison["model_beats_decomp_01c_on_mae_or_diagnosis_flow"] = (
            model_mae < float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf"))
            or float(diagnosis_flow_mae) < float(branch_reference_diagnosis_flow_mae)
        )
    if reverse_grasp_peak:
        baseline_comparison["decomp_01e_reference_mean_absolute_error"] = float(
            branch_reference_baseline.get("model_mean_absolute_error") or 0.0
        )
        baseline_comparison["decomp_01e_reference_peak_alive_on_art_absolute_error"] = round(
            float(branch_reference_peak_alive_on_art_error), 6
        )
        baseline_comparison["model_peak_alive_on_art_absolute_error"] = round(float(model_peak_alive_on_art_error), 6)
        baseline_comparison["peak_target_quarter"] = target_peak_quarter
        baseline_comparison["model_beats_decomp_01e_on_mae_or_peak_error"] = (
            model_mae < float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf"))
            or float(model_peak_alive_on_art_error) < float(branch_reference_peak_alive_on_art_error)
        )
    evaluation = {
        "mode": "transition_research_decomp_fused_mechanistic_forecast",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "channel_names": list(CHANNEL_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
        "decomp_01b_reference_run_id": str(decomp_01b_reference["reference_run_id"]),
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "mech_01b_reference_run_id": str(mech_01b_reference["reference_run_id"]),
        "train_quarters": train_quarters,
        "holdout_quarters": holdout_quarters,
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    if reverse_grasp_peak:
        evaluation["peak_target_quarter"] = target_peak_quarter
        evaluation["peak_alive_on_art_absolute_error"] = round(float(model_peak_alive_on_art_error), 6)
    fit_artifact = {
        "model_family": "transition_research_decomp_fused_mechanistic_forecast",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "channel_names": list(CHANNEL_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
        "decomp_01b_reference_run_id": str(decomp_01b_reference["reference_run_id"]),
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "uses_mech_01e_transition_baseline": True,
        "uses_channel_driver_coupling": True,
        "uses_decomposition_fusion": True,
        "uses_reverse_grasp_peak_clusters": bool(reverse_grasp_peak),
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=[f"{experiment_label} train-state reconstruction inherited from the kept MECH-01E baseline"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=[f"{experiment_label} holdout-state forecast with fused supported channel drivers and locked diagnosis"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    hazard_reconstruction = {
        "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
        "decomp_01b_reference_run_id": str(decomp_01b_reference["reference_run_id"]),
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "skill_gated": bool(skill_gated),
        "loo_skill_gated": bool(loo_skill_gated or reverse_grasp_peak),
        "reverse_grasp_peak": bool(reverse_grasp_peak),
        "locked_transitions": sorted(locked_transitions),
        "downstream_fused_transitions": list(DOWNSTREAM_TRANSITIONS),
        "train_quarters": train_quarters,
        "holdout_quarters": holdout_quarters,
        "rows": hazard_reconstruction_rows,
        "channel_model_rows": channel_model_rows,
        "fused_holdout_hazard_rows": fused_holdout_hazard_rows,
    }
    mechanistic_forecast = {
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "transition_hazards": fused_holdout_hazard_rows,
        "forecast_rows": [
            {
                "quarter": str(row["quarter"]),
                "prediction": {metric_name: round(float(row[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "state_values": {state_name: round(float(row["state_values"][state_name]), 6) for state_name in STATE_NAMES},
            }
            for row in forecast_rows
        ],
    }
    transition_hazard_summary = {
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "train_quarters": train_quarters,
        "holdout_quarters": holdout_quarters,
        "train_rows": train_transition_rows,
        "baseline_holdout_hazards": baseline_holdout_hazards,
        "holdout_rows": [
            {
                "quarter": str(row["quarter"]),
                "hazards": dict(row["hazards"]),
                "base_hazards": {
                    transition: round(float((baseline_holdout_hazards.get(str(row["quarter"])) or {}).get(transition) or 0.0), 6)
                    for transition in TRANSITION_NAMES
                },
                "flows": dict(row["flows"]),
            }
            for row in simulated_holdout_transition_rows
        ],
    }

    max_fused_hazard = (
        max(
            float((row.get("hazards") or {}).get(transition) or 0.0)
            for row in simulated_holdout_transition_rows
            for transition in TRANSITION_NAMES
        )
        if simulated_holdout_transition_rows
        else 0.0
    )
    no_empirical_spike = all(
        float((row.get("hazards") or {}).get(transition) or 0.0) <= float(empirical_upper_bounds.get(transition) or 0.0) + eps
        for row in simulated_holdout_transition_rows
        for transition in TRANSITION_NAMES
    )
    diagnosis_locked_match = all(
        abs(
            float((baseline_holdout_hazards.get(str(row.get("quarter"))) or {}).get("U_to_D") or 0.0)
            - float((row.get("hazards") or {}).get("U_to_D") or 0.0)
        ) <= eps
        for row in simulated_holdout_transition_rows
    )
    decision_checks = [
        {
            "name": (
                "beats_decomp_01e_on_mae_or_peak_error"
                if reverse_grasp_peak
                else "beats_reference_branch_on_mae_or_diagnosis_flow"
            ),
            "passed": bool(
                baseline_comparison["model_beats_decomp_01e_on_mae_or_peak_error"]
                if reverse_grasp_peak
                else (
                    baseline_comparison["model_beats_decomp_01c_on_mae_or_diagnosis_flow"]
                    if skill_gated or loo_skill_gated
                    else baseline_comparison["model_beats_mech_01b_on_mae_or_diagnosis_flow"]
                )
            ),
            "actual": (
                {
                    "model_mean_absolute_error": model_mae,
                    "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
                    "peak_target_quarter": target_peak_quarter,
                }
                if reverse_grasp_peak
                else {
                    "model_mean_absolute_error": model_mae,
                    "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
                }
            ),
            "target": (
                {
                    "decomp_01e_reference_mean_absolute_error": float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf")),
                    "decomp_01e_reference_peak_alive_on_art_absolute_error": round(float(branch_reference_peak_alive_on_art_error), 6),
                    "peak_target_quarter": target_peak_quarter,
                }
                if reverse_grasp_peak
                else (
                    {
                        "decomp_01c_reference_mean_absolute_error": float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf")),
                        "decomp_01c_reference_diagnosis_flow_mean_absolute_error": round(float(branch_reference_diagnosis_flow_mae), 6),
                    }
                    if skill_gated or loo_skill_gated
                    else {
                        "mech_01b_reference_mean_absolute_error": float(mech_01b_baseline.get("model_mean_absolute_error") or float("inf")),
                        "mech_01b_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01b_diagnosis_flow_mae), 6),
                    }
                )
            ),
        },
        {
            "name": "no_empirical_hazard_spikes",
            "passed": bool(no_empirical_spike),
            "actual": round(float(max_fused_hazard), 6),
            "target": {transition: round(float(value), 6) for transition, value in empirical_upper_bounds.items()},
        },
        {
            "name": "diagnosis_hazard_remains_locked_to_mech_01e",
            "passed": bool(diagnosis_locked_match),
            "actual": bool(diagnosis_locked_match),
            "target": True,
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float((row.get("hazards") or {}).get(transition) or 0.0))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float((row.get("hazards") or {}).get(transition) or 0.0))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": (
            "DECOMP-01F applies a reverse-GRASP-style peak and cluster refinement on downstream shock and residual channels after the kept DECOMP-01E fusion, while leaving diagnosis locked on the baseline."
            if reverse_grasp_peak
            else (
                "DECOMP-01D applies train-skill gating to supported channel-driver structure before fusing it back into the anchored MECH-01E mechanistic simulator."
                if skill_gated
                else (
                    "DECOMP-01E applies leave-one-quarter-out channel skill gating to supported channel-driver structure before fusing it back into the anchored MECH-01E mechanistic simulator."
                    if loo_skill_gated
                    else "DECOMP-01C fuses supported channel-driver structure back into the anchored MECH-01E mechanistic simulator while preserving unsupported pairs and diagnosis on the kept baseline."
                )
            )
        ),
        "checks": decision_checks,
    }

    comparison_labels = [experiment_label, "MECH-01E", "MECH-01B", "Carry-forward", "Simple compartmental"]
    comparison_values = [
        model_mae,
        float(mech_01e_baseline.get("model_mean_absolute_error") or 0.0),
        float(mech_01b_baseline.get("model_mean_absolute_error") or 0.0),
        float(mech_01e_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        float(mech_01e_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
    ]
    if skill_gated or loo_skill_gated:
        comparison_labels.insert(1, "DECOMP-01C")
        comparison_values.insert(1, float(branch_reference_baseline.get("model_mean_absolute_error") or 0.0))
    if reverse_grasp_peak:
        comparison_labels.insert(1, "DECOMP-01E")
        comparison_values.insert(1, float(branch_reference_baseline.get("model_mean_absolute_error") or 0.0))
    fig, ax = plt.subplots()
    ax.bar(comparison_labels, comparison_values)
    ax.set_ylabel("Normalized MAE")
    ax.set_title(f"{experiment_label} Forecast vs Baselines")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "forecast_vs_baselines.png")
    plt.close(fig)

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "hazard_reconstruction.json", hazard_reconstruction)
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", mechanistic_forecast)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
            "decomp_01b_reference_run_id": str(decomp_01b_reference["reference_run_id"]),
            "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
            "mech_01b_reference_run_id": str(mech_01b_reference["reference_run_id"]),
            "decomp_01c_reference_run_id": (
                str((branch_reference or {}).get("reference_run_id") or "")
                if (skill_gated or loo_skill_gated)
                else None
            ),
            "decomp_01e_reference_run_id": (
                str((branch_reference or {}).get("reference_run_id") or "")
                if reverse_grasp_peak
                else None
            ),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "holdout_simulation_row_count": len(holdout_rows[1:]),
            "transition_count": len(TRANSITION_NAMES),
            "channel_count": len(CHANNEL_NAMES),
            "locked_transition_count": len(locked_transitions),
            "supported_pair_count": sum(1 for row in pair_summary_rows if bool(row.get("supported"))),
            "unsupported_pair_count": sum(1 for row in pair_summary_rows if not bool(row.get("supported"))),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E data-estimated diagnosed coverage mean so the decomposition fusion only changes supported downstream hazards",
                "uncertainty": "inherits MECH-01E coverage uncertainty",
                "why_needed": "Keeps the latent state reconstruction fixed while evaluating fused channel-driver effects.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E documented suppression share among ART",
                "uncertainty": "inherits MECH-01E service ascertainment uncertainty",
                "why_needed": "Prevents the fused forecast from confounding channel-driver effects with a different A/V split.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E mean viral-load-testing share among ART",
                "uncertainty": "inherits MECH-01E service documentation uncertainty",
                "why_needed": "Keeps the tested-for-VL auxiliary process aligned with the branch baseline while evaluating channel fusion.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "MECH-01E fit artifact inherited from consecutive national train rows",
                "estimation_method": "reuse the kept MECH-01E semi-Markov lost-gap share estimate",
                "uncertainty": "inherits sparse direct L-observation uncertainty",
                "why_needed": "Keeps the D/L split fixed so fused channel drivers are not confounded by a changing latent loss definition.",
            },
            {
                "name": "supported_pair_count",
                "value": sum(1 for row in pair_summary_rows if bool(row.get("supported"))),
                "role": "supported_transition_channel_pair_count",
                "source_type": "estimated",
                "estimation_data": "DECOMP-01B pair summary",
                "estimation_method": "count supported transition-channel pairs retained by the channel-driver coupling step",
                "uncertainty": "depends on DECOMP-01B support classification",
                "why_needed": f"Defines how many transition-channel pairs are eligible for fused driver correction in {experiment_label}.",
            },
            {
                "name": "skill_gate_active",
                "value": 1 if (skill_gated or loo_skill_gated) else 0,
                "role": "skill_gated_fusion_design_flag",
                "source_type": "physical_constraint",
                "estimation_data": "experiment design",
                "estimation_method": "flag indicating whether train-time channel skill gates the effective blend weight",
                "uncertainty": "none",
                "why_needed": f"Records whether {experiment_label} uses raw supported-pair blending or a stricter skill-gated variant.",
            },
            {
                "name": "loo_skill_gate_active",
                "value": 1 if (loo_skill_gated or reverse_grasp_peak) else 0,
                "role": "leave_one_quarter_out_skill_gate_design_flag",
                "source_type": "physical_constraint",
                "estimation_data": "experiment design",
                "estimation_method": "flag indicating whether leave-one-quarter-out channel skill gates the effective blend weight",
                "uncertainty": "none",
                "why_needed": f"Records whether {experiment_label} uses out-of-sample quarter-wise channel skill instead of only in-sample channel fit.",
            },
            {
                "name": "reverse_grasp_peak_active",
                "value": 1 if reverse_grasp_peak else 0,
                "role": "reverse_peak_cluster_refinement_design_flag",
                "source_type": "physical_constraint",
                "estimation_data": "experiment design",
                "estimation_method": "flag indicating whether reverse-GRASP-style peak and cluster refinement is active on supported downstream shock and residual channels",
                "uncertainty": "none",
                "why_needed": "Records whether downstream peak and cluster refinement is layered on top of the kept DECOMP-01E fusion.",
            },
            {
                "name": "locked_transition_count",
                "value": len(locked_transitions),
                "role": "diagnosis_lock_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "branch design inherited from MECH-01E",
                "estimation_method": "count transitions preserved exactly on the kept baseline during fused channel simulation",
                "uncertainty": "none",
                "why_needed": "Records that diagnosis remains locked while channel-driver fusion is tested downstream.",
            },
            {
                "name": "anchored_holdout_quarter_count",
                "value": 1,
                "role": "holdout_anchor_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "branch design inherited from MECH-01E",
                "estimation_method": "count of holdout quarters copied exactly from the kept baseline before fused simulation begins",
                "uncertainty": "none",
                "why_needed": "Keeps the fused forecast commensurate with the kept baseline by fixing the initial holdout state.",
            },
            *(
                [
                    {
                        "name": "decomp_01c_reference_mean_absolute_error",
                        "value": round(float(branch_reference_baseline.get("model_mean_absolute_error") or 0.0), 6),
                        "role": "branch_reference_forecast_gate",
                        "source_type": "estimated",
                        "estimation_data": "kept DECOMP-01C baseline_comparison artifact",
                        "estimation_method": "reuse the latest kept DECOMP-01C normalized holdout MAE as the reference gate for DECOMP-01D",
                        "uncertainty": "inherits DECOMP-01C holdout uncertainty",
                        "why_needed": "Ensures the skill-gated fusion branch is judged against the current kept decomposition baseline rather than an older weaker branch.",
                    }
                ]
                if skill_gated or loo_skill_gated
                else []
            ),
            *(
                [
                    {
                        "name": "decomp_01e_reference_mean_absolute_error",
                        "value": round(float(branch_reference_baseline.get("model_mean_absolute_error") or 0.0), 6),
                        "role": "branch_reference_forecast_gate",
                        "source_type": "estimated",
                        "estimation_data": "kept DECOMP-01E baseline_comparison artifact",
                        "estimation_method": "reuse the latest kept DECOMP-01E normalized holdout MAE as the reference gate for DECOMP-01F",
                        "uncertainty": "inherits DECOMP-01E holdout uncertainty",
                        "why_needed": "Ensures the reverse-GRASP peak branch is judged against the current kept decomposition baseline rather than an older branch.",
                    },
                    {
                        "name": "decomp_01e_reference_peak_alive_on_art_absolute_error",
                        "value": round(float(branch_reference_peak_alive_on_art_error), 6),
                        "role": "peak_quarter_reference_gate",
                        "source_type": "estimated",
                        "estimation_data": f"kept DECOMP-01E prediction and holdout alive_on_art target at {target_peak_quarter}",
                        "estimation_method": "absolute error at the holdout quarter with the highest alive_on_art target",
                        "uncertainty": "depends on holdout alive_on_art observation uncertainty",
                        "why_needed": "Provides the explicit peak-error comparator for DECOMP-01F.",
                    },
                ]
                if reverse_grasp_peak
                else []
            ),
            *(
                [
                    {
                        "name": "peak_cluster_transition_count",
                        "value": len(PEAK_CLUSTER_TRANSITIONS),
                        "role": "reverse_peak_cluster_scope_cardinality",
                        "source_type": "physical_constraint",
                        "estimation_data": "experiment design",
                        "estimation_method": "count downstream transitions eligible for reverse-GRASP-style peak refinement",
                        "uncertainty": "none",
                        "why_needed": "Records the transition scope of the peak-cluster refinement branch.",
                    },
                    {
                        "name": "peak_cluster_channel_count",
                        "value": len(PEAK_CLUSTER_CHANNELS),
                        "role": "reverse_peak_cluster_channel_scope_cardinality",
                        "source_type": "physical_constraint",
                        "estimation_data": "experiment design",
                        "estimation_method": "count channel types eligible for reverse-GRASP-style peak refinement",
                        "uncertainty": "none",
                        "why_needed": "Records the channel scope of the peak-cluster refinement branch.",
                    },
                ]
                if reverse_grasp_peak
                else []
            ),
            {
                "name": "probability_lower_bound",
                "value": probability_lower_bound,
                "role": "hazard_probability_floor",
                "source_type": "physical_constraint",
                "estimation_data": "probability simplex",
                "estimation_method": "lower probability bound for transition hazards",
                "uncertainty": "none",
                "why_needed": "Prevents negative transition hazards during mechanistic simulation.",
            },
            {
                "name": "probability_upper_bound",
                "value": probability_upper_bound,
                "role": "hazard_probability_ceiling",
                "source_type": "physical_constraint",
                "estimation_data": "probability simplex",
                "estimation_method": "upper probability bound for transition hazards",
                "uncertainty": "none",
                "why_needed": "Prevents transition hazards from exceeding probability mass during mechanistic simulation.",
            },
            *[
                {
                    "name": f"empirical_hazard_upper_bound_{transition}",
                    "value": round(float(empirical_upper_bounds[transition]), 6),
                    "role": "transition_hazard_spike_guard",
                    "source_type": "estimated",
                    "estimation_data": "train transition hazards and MECH-01E baseline holdout hazards",
                    "estimation_method": f"maximum observed {transition} hazard across train rows and kept baseline holdout hazards",
                    "uncertainty": "depends on observed hazard support in the retained branch",
                    "why_needed": f"Defines the empirical spike guard used to reject implausible fused {transition} hazards.",
                }
                for transition in TRANSITION_NAMES
            ],
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_reconstruction_and_simulation_guard",
                why_needed="Prevents undefined correlations, zero-denominator state outflow scaling, and exact-equality failures when validating locked hazards.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "hazard_reconstruction": hazard_reconstruction,
        "mechanistic_forecast": mechanistic_forecast,
        "decision": decision,
    }


def run_decomp_01d(ctx: TransitionResearchContext) -> dict[str, Any]:
    return run_decomp_01c(ctx)


def run_decomp_01e(ctx: TransitionResearchContext) -> dict[str, Any]:
    return run_decomp_01c(ctx)


def run_decomp_01f(ctx: TransitionResearchContext) -> dict[str, Any]:
    return run_decomp_01c(ctx)
