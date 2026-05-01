from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import EXIT_CHANNEL_NAMES, EXTERNAL_EXIT_CHANNEL_NAMES, STATE_NAMES, BlockedTimeDataset
from .metrics import quarter_sort_key


@dataclass(slots=True)
class IncidenceFlowConfig:
    ridge_penalty: float = 0.1
    rho_clip: float = 0.9
    trend_scale: float = 1.0


def _quarter_positions(dataset: BlockedTimeDataset) -> dict[str, int]:
    quarters = sorted({str(row.get("quarter") or "") for row in dataset.train_transition_rows + dataset.holdout_rows}, key=quarter_sort_key)
    return {quarter: idx for idx, quarter in enumerate(quarters)}


def _fit_log_positive_ar_trend(
    train_quarters: list[str],
    train_values: list[float],
    forecast_quarters: list[str],
    *,
    ridge_penalty: float,
    rho_clip: float,
    trend_scale: float,
    positions: dict[str, int],
) -> dict[str, Any]:
    def positive_from_eta(value: float) -> float:
        return max(float(np.expm1(float(value))), 0.0)

    transformed = [float(np.log1p(max(float(value), 0.0))) for value in train_values]
    if len(transformed) < 2:
        constant_eta = float(transformed[-1]) if transformed else 0.0
        train_predictions = {quarter: positive_from_eta(constant_eta) for quarter in train_quarters}
        forecast_predictions = {quarter: positive_from_eta(constant_eta) for quarter in forecast_quarters}
        return {
            "train_predictions": train_predictions,
            "forecast_predictions": forecast_predictions,
            "parameters": {"alpha": constant_eta, "slope": 0.0, "rho": 0.0},
        }
    y = np.asarray(transformed[1:], dtype=np.float64)
    x_rows = []
    for idx in range(1, len(transformed)):
        quarter = train_quarters[idx]
        x_rows.append([1.0, float(positions[quarter]) * float(trend_scale), float(transformed[idx - 1])])
    x = np.asarray(x_rows, dtype=np.float64)
    ridge = np.diag([0.0, float(ridge_penalty), float(ridge_penalty)])
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    alpha = float(beta[0])
    slope = float(beta[1])
    rho = float(np.clip(beta[2], -float(rho_clip), float(rho_clip)))
    train_eta_predictions: dict[str, float] = {train_quarters[0]: float(transformed[0])}
    for idx in range(1, len(transformed)):
        quarter = train_quarters[idx]
        predicted = alpha + slope * float(positions[quarter]) * float(trend_scale) + rho * float(transformed[idx - 1])
        train_eta_predictions[quarter] = float(predicted)
    forecast_eta_predictions: dict[str, float] = {}
    last_eta = float(transformed[-1])
    for quarter in forecast_quarters:
        eta = alpha + slope * float(positions[quarter]) * float(trend_scale) + rho * last_eta
        forecast_eta_predictions[quarter] = float(eta)
        last_eta = float(eta)
    return {
        "train_predictions": {quarter: positive_from_eta(eta) for quarter, eta in train_eta_predictions.items()},
        "forecast_predictions": {quarter: positive_from_eta(eta) for quarter, eta in forecast_eta_predictions.items()},
        "parameters": {"alpha": alpha, "slope": slope, "rho": rho},
    }


def fit_incidence_flow_paths(dataset: BlockedTimeDataset, cfg: IncidenceFlowConfig) -> dict[str, Any]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_transition_rows or not holdout_rows:
        return {
            "train_incidence_inflow_map": {},
            "holdout_incidence_inflow_map": {},
            "train_incidence_hazard_map": {},
            "holdout_incidence_hazard_map": {},
            "train_population_denominator_map": {},
            "holdout_population_denominator_map": {},
            "train_attrition_outflow_map": {},
            "holdout_attrition_outflow_map": {},
            "train_state_attrition_outflow_map": {},
            "holdout_state_attrition_outflow_map": {},
            "train_exit_channel_outflow_map": {},
            "holdout_exit_channel_outflow_map": {},
            "train_exit_channel_state_outflow_map": {},
            "holdout_exit_channel_state_outflow_map": {},
            "diagnostics": {},
        }
    positions = _quarter_positions(dataset)
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    incidence_values = [float(((row.get("stock_balance") or {}).get("incidence_inflow")) or 0.0) for row in train_transition_rows]
    incidence_hazard_values = [
        float(((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff")) or 0.0)
        for row in train_transition_rows
    ]
    attrition_values = [float(((row.get("stock_balance") or {}).get("attrition_outflow")) or 0.0) for row in train_transition_rows]
    incidence_fit = _fit_log_positive_ar_trend(
        train_quarters,
        incidence_values,
        holdout_quarters,
        ridge_penalty=cfg.ridge_penalty,
        rho_clip=cfg.rho_clip,
        trend_scale=cfg.trend_scale,
        positions=positions,
    )
    incidence_hazard_fit = _fit_log_positive_ar_trend(
        train_quarters,
        incidence_hazard_values,
        holdout_quarters,
        ridge_penalty=cfg.ridge_penalty,
        rho_clip=cfg.rho_clip,
        trend_scale=cfg.trend_scale,
        positions=positions,
    )
    attrition_fit = _fit_log_positive_ar_trend(
        train_quarters,
        attrition_values,
        holdout_quarters,
        ridge_penalty=cfg.ridge_penalty,
        rho_clip=cfg.rho_clip,
        trend_scale=cfg.trend_scale,
        positions=positions,
    )
    state_attrition_fits: dict[str, dict[str, Any]] = {}
    for state_name in STATE_NAMES:
        values = [
            float((((row.get("stock_balance") or {}).get("state_attrition_outflows") or {}).get(state_name)) or 0.0)
            for row in train_transition_rows
        ]
        state_attrition_fits[state_name] = _fit_log_positive_ar_trend(
            train_quarters,
            values,
            holdout_quarters,
            ridge_penalty=cfg.ridge_penalty,
            rho_clip=cfg.rho_clip,
            trend_scale=cfg.trend_scale,
            positions=positions,
        )
    exit_channel_fits: dict[str, dict[str, Any]] = {}
    for channel_name in EXIT_CHANNEL_NAMES:
        values = [
            float((((row.get("stock_balance") or {}).get("exit_channel_outflows") or {}).get(channel_name)) or 0.0)
            for row in train_transition_rows
        ]
        exit_channel_fits[channel_name] = _fit_log_positive_ar_trend(
            train_quarters,
            values,
            holdout_quarters,
            ridge_penalty=cfg.ridge_penalty,
            rho_clip=cfg.rho_clip,
            trend_scale=cfg.trend_scale,
            positions=positions,
        )
    exit_channel_state_fits: dict[str, dict[str, dict[str, Any]]] = {}
    for channel_name in EXTERNAL_EXIT_CHANNEL_NAMES:
        exit_channel_state_fits[channel_name] = {}
        for state_name in STATE_NAMES:
            values = [
                float(((((row.get("stock_balance") or {}).get("exit_channel_state_outflows") or {}).get(channel_name) or {}).get(state_name)) or 0.0)
                for row in train_transition_rows
            ]
            exit_channel_state_fits[channel_name][state_name] = _fit_log_positive_ar_trend(
                train_quarters,
                values,
                holdout_quarters,
                ridge_penalty=cfg.ridge_penalty,
                rho_clip=cfg.rho_clip,
                trend_scale=cfg.trend_scale,
                positions=positions,
            )
    train_exit_channel_outflow_map = _channel_prediction_map(
        train_quarters,
        {channel: dict(fit["train_predictions"]) for channel, fit in exit_channel_fits.items()},
    )
    holdout_exit_channel_outflow_map = _channel_prediction_map(
        holdout_quarters,
        {channel: dict(fit["forecast_predictions"]) for channel, fit in exit_channel_fits.items()},
    )
    train_exit_channel_state_outflow_map = _exit_channel_state_prediction_map(
        train_quarters,
        {
            channel: {state: dict(fit["train_predictions"]) for state, fit in state_fits.items()}
            for channel, state_fits in exit_channel_state_fits.items()
        },
    )
    holdout_exit_channel_state_outflow_map = _exit_channel_state_prediction_map(
        holdout_quarters,
        {
            channel: {state: dict(fit["forecast_predictions"]) for state, fit in state_fits.items()}
            for channel, state_fits in exit_channel_state_fits.items()
        },
    )
    train_state_attrition_map = _aggregate_exit_channel_state_map(train_exit_channel_state_outflow_map)
    holdout_state_attrition_map = _aggregate_exit_channel_state_map(holdout_exit_channel_state_outflow_map)
    train_population_denominator_map = {
        str(row.get("quarter") or ""): float(((row.get("stock_balance") or {}).get("population_denominator_current")) or 0.0)
        for row in train_transition_rows
        if ((row.get("stock_balance") or {}).get("population_denominator_current")) is not None
    }
    holdout_population_denominator_map = {
        str(row.get("quarter") or ""): float(row.get("population_total") or 0.0)
        for row in holdout_rows
        if row.get("population_total") is not None and float(row.get("population_total") or 0.0) > 0.0
    }
    return {
        "train_incidence_inflow_map": {quarter: float(value) for quarter, value in incidence_fit["train_predictions"].items()},
        "holdout_incidence_inflow_map": {quarter: float(value) for quarter, value in incidence_fit["forecast_predictions"].items()},
        "train_incidence_hazard_map": {quarter: float(value) for quarter, value in incidence_hazard_fit["train_predictions"].items()},
        "holdout_incidence_hazard_map": {quarter: float(value) for quarter, value in incidence_hazard_fit["forecast_predictions"].items()},
        "train_population_denominator_map": train_population_denominator_map,
        "holdout_population_denominator_map": holdout_population_denominator_map,
        "train_attrition_outflow_map": _aggregate_state_map(train_state_attrition_map),
        "holdout_attrition_outflow_map": _aggregate_state_map(holdout_state_attrition_map),
        "train_state_attrition_outflow_map": train_state_attrition_map,
        "holdout_state_attrition_outflow_map": holdout_state_attrition_map,
        "train_exit_channel_outflow_map": train_exit_channel_outflow_map,
        "holdout_exit_channel_outflow_map": holdout_exit_channel_outflow_map,
        "train_exit_channel_state_outflow_map": train_exit_channel_state_outflow_map,
        "holdout_exit_channel_state_outflow_map": holdout_exit_channel_state_outflow_map,
        "diagnostics": {
            "incidence_inflow": {
                "train_mean": float(np.mean(np.asarray(incidence_values, dtype=np.float64))) if incidence_values else 0.0,
                "forecast_mean": float(np.mean(np.asarray(list(incidence_fit["forecast_predictions"].values()), dtype=np.float64))) if incidence_fit["forecast_predictions"] else 0.0,
                "parameters": dict(incidence_fit["parameters"]),
            },
            "incidence_hazard_per_s_eff": {
                "train_mean": float(np.mean(np.asarray(incidence_hazard_values, dtype=np.float64))) if incidence_hazard_values else 0.0,
                "forecast_mean": float(np.mean(np.asarray(list(incidence_hazard_fit["forecast_predictions"].values()), dtype=np.float64))) if incidence_hazard_fit["forecast_predictions"] else 0.0,
                "parameters": dict(incidence_hazard_fit["parameters"]),
                "denominator_contract": "S_eff=max(population_total-current_PLHIV_state_total,0)",
            },
            "attrition_outflow": {
                "train_mean": float(np.mean(np.asarray(attrition_values, dtype=np.float64))) if attrition_values else 0.0,
                "forecast_mean": float(np.mean(np.asarray(list(_aggregate_state_map(holdout_state_attrition_map).values()), dtype=np.float64))) if holdout_state_attrition_map else 0.0,
                "parameters": dict(attrition_fit["parameters"]),
                "state_specific": {
                    state: {
                        "parameters": dict(fit["parameters"]),
                        "train_mean": float(np.mean(np.asarray([
                            float((((row.get("stock_balance") or {}).get("state_attrition_outflows") or {}).get(state)) or 0.0)
                            for row in train_transition_rows
                        ], dtype=np.float64))),
                    }
                    for state, fit in state_attrition_fits.items()
                },
                "exit_channels": {
                    channel: {
                        "parameters": dict(fit["parameters"]),
                        "train_mean": float(np.mean(np.asarray([
                            float((((row.get("stock_balance") or {}).get("exit_channel_outflows") or {}).get(channel)) or 0.0)
                            for row in train_transition_rows
                        ], dtype=np.float64))),
                        "forecast_mean": float(np.mean(np.asarray([
                            float((values or {}).get(channel) or 0.0)
                            for values in holdout_exit_channel_outflow_map.values()
                        ], dtype=np.float64))) if holdout_exit_channel_outflow_map else 0.0,
                    }
                    for channel, fit in exit_channel_fits.items()
                },
            },
        },
    }


def _state_prediction_map(quarters: list[str], predictions_by_state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        quarter: {
            state_name: float((predictions_by_state.get(state_name) or {}).get(quarter) or 0.0)
            for state_name in STATE_NAMES
        }
        for quarter in quarters
    }


def _aggregate_state_map(state_map: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        quarter: float(sum(float((values or {}).get(state_name) or 0.0) for state_name in STATE_NAMES))
        for quarter, values in state_map.items()
    }


def _channel_prediction_map(quarters: list[str], predictions_by_channel: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        quarter: {
            channel_name: float((predictions_by_channel.get(channel_name) or {}).get(quarter) or 0.0)
            for channel_name in EXIT_CHANNEL_NAMES
        }
        for quarter in quarters
    }


def _exit_channel_state_prediction_map(
    quarters: list[str],
    predictions_by_channel_state: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        quarter: {
            channel_name: {
                state_name: float(((predictions_by_channel_state.get(channel_name) or {}).get(state_name) or {}).get(quarter) or 0.0)
                for state_name in STATE_NAMES
            }
            for channel_name in EXTERNAL_EXIT_CHANNEL_NAMES
        }
        for quarter in quarters
    }


def _aggregate_exit_channel_state_map(channel_state_map: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        quarter: {
            state_name: float(
                sum(float(((channels.get(channel_name) or {}).get(state_name)) or 0.0) for channel_name in EXTERNAL_EXIT_CHANNEL_NAMES)
            )
            for state_name in STATE_NAMES
        }
        for quarter, channels in channel_state_map.items()
    }


def carry_forward_incidence_flow_paths(dataset: BlockedTimeDataset, *, mode: str = "last_train") -> dict[str, dict[str, float]]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_transition_rows or not holdout_rows:
        return {
            "incidence_inflow_map": {},
            "incidence_hazard_map": {},
            "population_denominator_map": {},
            "attrition_outflow_map": {},
            "state_attrition_outflow_map": {},
            "exit_channel_outflow_map": {},
            "exit_channel_state_outflow_map": {},
        }
    if str(mode) == "mean_train":
        incidence_value = float(np.mean([float(((row.get("stock_balance") or {}).get("incidence_inflow")) or 0.0) for row in train_transition_rows]))
        incidence_hazard_value = float(np.mean([float(((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff")) or 0.0) for row in train_transition_rows]))
        state_attrition_value = {
            state: float(np.mean([
                float((((row.get("stock_balance") or {}).get("state_attrition_outflows") or {}).get(state)) or 0.0)
                for row in train_transition_rows
            ]))
            for state in STATE_NAMES
        }
        exit_channel_value = {
            channel: float(np.mean([
                float((((row.get("stock_balance") or {}).get("exit_channel_outflows") or {}).get(channel)) or 0.0)
                for row in train_transition_rows
            ]))
            for channel in EXIT_CHANNEL_NAMES
        }
        exit_channel_state_value = {
            channel: {
                state: float(np.mean([
                    float(((((row.get("stock_balance") or {}).get("exit_channel_state_outflows") or {}).get(channel) or {}).get(state)) or 0.0)
                    for row in train_transition_rows
                ]))
                for state in STATE_NAMES
            }
            for channel in EXTERNAL_EXIT_CHANNEL_NAMES
        }
    else:
        last_balance = dict(train_transition_rows[-1].get("stock_balance") or {})
        incidence_value = float(last_balance.get("incidence_inflow") or 0.0)
        incidence_hazard_value = float(last_balance.get("incidence_hazard_per_s_eff") or 0.0)
        state_attrition_value = {
            state: float((last_balance.get("state_attrition_outflows") or {}).get(state) or 0.0)
            for state in STATE_NAMES
        }
        exit_channel_value = {
            channel: float((last_balance.get("exit_channel_outflows") or {}).get(channel) or 0.0)
            for channel in EXIT_CHANNEL_NAMES
        }
        exit_channel_state_value = {
            channel: {
                state: float(((last_balance.get("exit_channel_state_outflows") or {}).get(channel) or {}).get(state) or 0.0)
                for state in STATE_NAMES
            }
            for channel in EXTERNAL_EXIT_CHANNEL_NAMES
        }
    state_attrition_map = {
        str(row.get("quarter") or ""): {state: float(value) for state, value in state_attrition_value.items()}
        for row in holdout_rows
    }
    exit_channel_state_map = {
        str(row.get("quarter") or ""): {
            channel: {state: float(value) for state, value in values.items()}
            for channel, values in exit_channel_state_value.items()
        }
        for row in holdout_rows
    }
    return {
        "incidence_inflow_map": {str(row.get("quarter") or ""): float(incidence_value) for row in holdout_rows},
        "incidence_hazard_map": {str(row.get("quarter") or ""): float(incidence_hazard_value) for row in holdout_rows},
        "population_denominator_map": {
            str(row.get("quarter") or ""): float(row.get("population_total") or 0.0)
            for row in holdout_rows
            if row.get("population_total") is not None and float(row.get("population_total") or 0.0) > 0.0
        },
        "attrition_outflow_map": _aggregate_state_map(state_attrition_map),
        "state_attrition_outflow_map": state_attrition_map,
        "exit_channel_outflow_map": {
            str(row.get("quarter") or ""): {channel: float(value) for channel, value in exit_channel_value.items()}
            for row in holdout_rows
        },
        "exit_channel_state_outflow_map": exit_channel_state_map,
    }


def apply_stock_balance(
    current_state: dict[str, float],
    *,
    incidence_inflow: float | None = None,
    attrition_outflow: float | None = None,
    incidence_hazard: float | None = None,
    incidence_cap: float | None = None,
    population_denominator: float | None = None,
    state_attrition_outflow: dict[str, float] | None = None,
    exit_channel_state_outflows: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    updated_state = {name: max(float(current_state[name]), 0.0) for name in current_state}
    total_before = float(sum(updated_state.values()))
    susceptible_effective = None
    computed_inflow = max(float(incidence_inflow or 0.0), 0.0)
    if population_denominator is not None and float(population_denominator) > 0.0:
        susceptible_effective = max(float(population_denominator) - total_before, 0.0)
        if incidence_hazard is not None:
            computed_inflow = max(float(incidence_hazard), 0.0) * float(susceptible_effective)
    if incidence_cap is not None:
        computed_inflow = min(computed_inflow, max(float(incidence_cap), 0.0))
    updated_state["U"] = float(updated_state["U"] + computed_inflow)
    total_after_inflow = float(sum(updated_state.values()))
    requested_channel_state = {
        channel_name: {
            name: max(float(((exit_channel_state_outflows or {}).get(channel_name) or {}).get(name) or 0.0), 0.0)
            for name in updated_state
        }
        for channel_name in EXTERNAL_EXIT_CHANNEL_NAMES
    }
    has_channel_state = any(
        value > 0.0
        for values in requested_channel_state.values()
        for value in values.values()
    )
    if has_channel_state:
        requested_state_attrition = {
            name: float(sum(float(values.get(name) or 0.0) for values in requested_channel_state.values()))
            for name in updated_state
        }
    else:
        requested_state_attrition = {
            name: max(float((state_attrition_outflow or {}).get(name) or 0.0), 0.0)
            for name in updated_state
        }
    actual_state_attrition: dict[str, float] = {name: 0.0 for name in updated_state}
    actual_channel_state = {
        channel_name: {name: 0.0 for name in updated_state}
        for channel_name in EXTERNAL_EXIT_CHANNEL_NAMES
    }
    if any(value > 0.0 for value in requested_state_attrition.values()):
        for name, requested in requested_state_attrition.items():
            actual = min(float(requested), float(updated_state[name]))
            updated_state[name] = float(updated_state[name] - actual)
            actual_state_attrition[name] = float(actual)
            if has_channel_state and requested > 0.0:
                for channel_name, values in requested_channel_state.items():
                    actual_channel_state[channel_name][name] = float(actual) * float(values.get(name) or 0.0) / float(requested)
    attrition = max(float(attrition_outflow or 0.0), 0.0)
    if not any(value > 0.0 for value in requested_state_attrition.values()) and total_after_inflow > 0.0 and attrition > 0.0:
        scale = max(total_after_inflow - attrition, 0.0) / total_after_inflow
        before_scaled = dict(updated_state)
        updated_state = {name: float(value) * float(scale) for name, value in updated_state.items()}
        actual_state_attrition = {
            name: float(before_scaled[name] - updated_state[name])
            for name in updated_state
        }
    total_after = float(sum(updated_state.values()))
    actual_attrition = float(sum(actual_state_attrition.values()))
    return {
        "state_values": updated_state,
        "stock_balance": {
            "total_before_balance": float(total_before),
            "total_after_inflow": float(total_after_inflow),
            "total_after_balance": float(total_after),
            "population_denominator": None if population_denominator is None else float(population_denominator),
            "susceptible_effective": susceptible_effective,
            "incidence_hazard_per_s_eff": None if incidence_hazard is None else max(float(incidence_hazard), 0.0),
            "incidence_inflow": computed_inflow,
            "attrition_outflow": actual_attrition if any(value > 0.0 for value in requested_state_attrition.values()) else attrition,
            "state_attrition_outflows": actual_state_attrition,
            "state_attrition_requested": requested_state_attrition,
            "exit_channel_state_outflows": actual_channel_state,
            "exit_channel_outflows": {
                channel_name: float(sum(float(values.get(name) or 0.0) for name in updated_state))
                for channel_name, values in actual_channel_state.items()
            },
        },
    }
