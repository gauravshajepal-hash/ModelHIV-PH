from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import MISSING_DATA_LADDER, TRANSITION_NAMES, BlockedTimeDataset
from .metrics import inv_logit, logit, quarter_sort_key


DECOMPOSITION_SCHEMA_VERSION = "phase3_dynamic_hiv_decomposition_controls.v1"
DECOMPOSITION_COMPONENTS: tuple[str, ...] = (
    "trend",
    "reporting_support_shift",
    "residual_shock",
)
STREAM_TO_SUPPORT_METRIC: dict[str, str] = {
    "incidence": "estimated_plhiv",
    "diagnosis": "new_diagnosed_cases_period",
    "art": "alive_on_art",
    "vl": "tested_for_viral_load",
    "suppression": "virally_suppressed",
}
TRANSITION_STREAMS: dict[str, tuple[str, ...]] = {
    "U_to_D": ("diagnosis",),
    "D_to_A": ("art",),
    "A_to_T": ("vl",),
    "T_to_V": ("suppression",),
    "A_to_L": ("art",),
    "T_to_L": ("art", "vl"),
    "V_to_L": ("art", "suppression"),
    "L_to_R": ("art",),
    "R_to_A": ("art",),
}


@dataclass(slots=True)
class DecompositionControlConfig:
    effect_shrinkage: float = 1.0
    use_trend: bool = True
    use_reporting_support_shift: bool = True
    use_residual_shock: bool = True
    enabled_streams: tuple[str, ...] | None = None
    disabled_streams: tuple[str, ...] = ()


def _active_components(cfg: DecompositionControlConfig) -> tuple[str, ...]:
    components: list[str] = []
    if bool(cfg.use_trend):
        components.append("trend")
    if bool(cfg.use_reporting_support_shift):
        components.append("reporting_support_shift")
    if bool(cfg.use_residual_shock):
        components.append("residual_shock")
    return tuple(components)


def _active_streams(cfg: DecompositionControlConfig) -> tuple[str, ...]:
    allowed = set(STREAM_TO_SUPPORT_METRIC) if cfg.enabled_streams is None else {str(value) for value in cfg.enabled_streams}
    disabled = {str(value) for value in cfg.disabled_streams}
    return tuple(stream for stream in STREAM_TO_SUPPORT_METRIC if stream in allowed and stream not in disabled)


def _ridge_solve(x: np.ndarray, y: np.ndarray, *, eps: float) -> np.ndarray:
    if x.size == 0 or y.size == 0:
        return np.zeros(x.shape[1] if x.ndim == 2 else 0, dtype=np.float64)
    variance = float(np.var(y)) if y.size else 0.0
    ridge_scale = max(variance, float(eps)) * max(float(x.shape[1]), 1.0) / max(float(x.shape[0]), 1.0)
    return np.linalg.solve(x.T @ x + ridge_scale * np.eye(x.shape[1], dtype=np.float64), x.T @ y)


def _bounded_effects(
    *,
    x_train: np.ndarray,
    y: np.ndarray,
    x_holdout: np.ndarray,
    eps: float,
    effect_shrinkage: float,
) -> dict[str, Any]:
    if x_train.size == 0 or x_train.shape[1] == 0 or y.size == 0:
        return {
            "train_effect": np.zeros(x_train.shape[0], dtype=np.float64),
            "holdout_effect": np.zeros(x_holdout.shape[0], dtype=np.float64),
            "coefficients": np.zeros(x_train.shape[1] if x_train.ndim == 2 else 0, dtype=np.float64),
            "effect_bound": 0.0,
        }
    beta = _ridge_solve(x_train, y, eps=eps)
    train_effect = np.asarray(x_train @ beta, dtype=np.float64)
    holdout_effect = np.asarray(x_holdout @ beta, dtype=np.float64)
    effect_bound = float(np.max(np.abs(y))) if y.size else 0.0
    raw_bound = float(max(np.max(np.abs(train_effect)) if train_effect.size else 0.0, np.max(np.abs(holdout_effect)) if holdout_effect.size else 0.0))
    if raw_bound > max(effect_bound, eps):
        scale = effect_bound / raw_bound
        train_effect = train_effect * scale
        holdout_effect = holdout_effect * scale
        beta = beta * scale
    shrinkage = max(float(effect_shrinkage), 0.0)
    return {
        "train_effect": np.asarray(train_effect * shrinkage, dtype=np.float64),
        "holdout_effect": np.asarray(holdout_effect * shrinkage, dtype=np.float64),
        "coefficients": np.asarray(beta * shrinkage, dtype=np.float64),
        "effect_bound": effect_bound,
    }


def _positions(train_quarters: list[str], holdout_quarters: list[str]) -> dict[str, int]:
    quarters = sorted(set(train_quarters + holdout_quarters), key=quarter_sort_key)
    return {quarter: idx for idx, quarter in enumerate(quarters)}


def _standardize(train_values: np.ndarray, holdout_values: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if train_values.size == 0:
        return train_values, holdout_values
    center = float(np.mean(train_values))
    scale = float(np.std(train_values))
    if scale <= eps:
        return np.zeros_like(train_values, dtype=np.float64), np.zeros_like(holdout_values, dtype=np.float64)
    return (train_values - center) / scale, (holdout_values - center) / scale


def _fill_train_values(values: list[float | None], *, fallback: float = 0.0) -> list[float]:
    if not values:
        return []
    filled: list[float | None] = list(values)
    last_seen: float | None = None
    for index, value in enumerate(filled):
        if value is None:
            filled[index] = last_seen
        else:
            last_seen = float(value)
    next_seen: float | None = None
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] is None:
            filled[index] = next_seen
        else:
            next_seen = float(filled[index])
    return [float(value if value is not None else fallback) for value in filled]


def _support_rank(row: dict[str, Any] | None, metric_name: str) -> float:
    if row is None or row.get(metric_name) is None:
        return 1.0
    metric_provenance = dict(row.get("metric_provenance") or {})
    tier = str((metric_provenance.get(metric_name) or {}).get("tier") or "rejected_or_quarantined")
    try:
        rank = MISSING_DATA_LADDER.index(tier)
    except ValueError:
        rank = len(MISSING_DATA_LADDER) - 1
    return float(rank) / max(float(len(MISSING_DATA_LADDER) - 1), 1.0)


def _stream_value(stream: str, transition_row: dict[str, Any], row_by_quarter: dict[str, dict[str, Any]]) -> float | None:
    quarter = str(transition_row.get("quarter") or "")
    stock_balance = dict(transition_row.get("stock_balance") or {})
    hazards = dict(transition_row.get("hazards") or {})
    if stream == "incidence":
        value = stock_balance.get("incidence_hazard_per_s_eff")
        return None if value is None else max(float(value), 0.0)
    if stream == "diagnosis":
        return max(float(hazards.get("U_to_D") or 0.0), 0.0)
    if stream == "art":
        return max(float(hazards.get("D_to_A") or 0.0), 0.0)
    source_row = row_by_quarter.get(quarter)
    if source_row is None:
        return None
    alive_on_art = source_row.get("alive_on_art")
    if alive_on_art is None or float(alive_on_art) <= 0.0:
        return None
    if stream == "vl":
        tested = source_row.get("tested_for_viral_load")
        return None if tested is None else float(np.clip(float(tested) / max(float(alive_on_art), 1e-12), 0.0, 1.0))
    if stream == "suppression":
        suppressed = source_row.get("virally_suppressed")
        return None if suppressed is None else float(np.clip(float(suppressed) / max(float(alive_on_art), 1e-12), 0.0, 1.0))
    return None


def _transform_stream(stream: str, value: float, *, eps: float) -> float:
    if stream == "incidence":
        return float(np.log1p(max(float(value), 0.0)))
    return logit(float(value), eps=eps)


def _fit_stream_components(
    *,
    stream: str,
    train_quarters: list[str],
    train_values: list[float],
    support_values: list[float],
    holdout_quarters: list[str],
    positions: dict[str, int],
    eps: float,
) -> dict[str, Any]:
    y = np.asarray([_transform_stream(stream, value, eps=eps) for value in train_values], dtype=np.float64)
    if y.size == 0:
        return {
            "train_components": {quarter: {component: 0.0 for component in DECOMPOSITION_COMPONENTS} for quarter in train_quarters},
            "holdout_components": {quarter: {component: 0.0 for component in DECOMPOSITION_COMPONENTS} for quarter in holdout_quarters},
            "diagnostics": {"observed_count": 0},
        }
    t_train = np.asarray([float(positions[quarter]) for quarter in train_quarters], dtype=np.float64)
    t_holdout = np.asarray([float(positions[quarter]) for quarter in holdout_quarters], dtype=np.float64)
    x_trend = np.column_stack([np.ones_like(t_train), t_train])
    trend_beta = np.linalg.lstsq(x_trend, y, rcond=None)[0]
    trend_train_raw = np.asarray(x_trend @ trend_beta, dtype=np.float64)
    trend_holdout_raw = np.asarray(np.column_stack([np.ones_like(t_holdout), t_holdout]) @ trend_beta, dtype=np.float64) if t_holdout.size else np.zeros(0, dtype=np.float64)
    residual_train_raw = np.asarray(y - trend_train_raw, dtype=np.float64)
    if residual_train_raw.size >= 2:
        prev = residual_train_raw[:-1]
        nxt = residual_train_raw[1:]
        denom = float(prev @ prev)
        rho = 0.0 if denom <= eps else float(np.clip(float(prev @ nxt) / denom, -1.0, 1.0))
    else:
        rho = 0.0
    residual_holdout_values: list[float] = []
    last_residual = float(residual_train_raw[-1]) if residual_train_raw.size else 0.0
    for _quarter in holdout_quarters:
        last_residual = float(rho) * last_residual
        residual_holdout_values.append(last_residual)
    residual_holdout_raw = np.asarray(residual_holdout_values, dtype=np.float64)
    support_array = np.asarray(support_values, dtype=np.float64)
    support_shift_train_raw = np.zeros_like(support_array, dtype=np.float64)
    if support_array.size >= 2:
        support_shift_train_raw[1:] = support_array[1:] - support_array[:-1]
    support_shift_holdout_raw = np.zeros(len(holdout_quarters), dtype=np.float64)
    trend_train, trend_holdout = _standardize(trend_train_raw, trend_holdout_raw, eps=eps)
    support_train, support_holdout = _standardize(support_shift_train_raw, support_shift_holdout_raw, eps=eps)
    residual_train, residual_holdout = _standardize(residual_train_raw, residual_holdout_raw, eps=eps)
    train_components = {
        quarter: {
            "trend": float(trend_train[index]),
            "reporting_support_shift": float(support_train[index]),
            "residual_shock": float(residual_train[index]),
        }
        for index, quarter in enumerate(train_quarters)
    }
    holdout_components = {
        quarter: {
            "trend": float(trend_holdout[index]),
            "reporting_support_shift": float(support_holdout[index]),
            "residual_shock": float(residual_holdout[index]),
        }
        for index, quarter in enumerate(holdout_quarters)
    }
    return {
        "train_components": train_components,
        "holdout_components": holdout_components,
        "diagnostics": {
            "observed_count": int(len([value for value in train_values if np.isfinite(float(value))])),
            "residual_rho": float(rho),
            "trend_slope": float(trend_beta[1]) if len(trend_beta) > 1 else 0.0,
            "support_shift_abs_mean": float(np.mean(np.abs(support_shift_train_raw))) if support_shift_train_raw.size else 0.0,
        },
    }


def build_hiv_decomposition_controls(dataset: BlockedTimeDataset, cfg: DecompositionControlConfig) -> dict[str, Any]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    row_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in list(dataset.train_rows) + list(dataset.holdout_rows)
    }
    positions = _positions(train_quarters, holdout_quarters)
    streams: dict[str, Any] = {}
    active_streams = _active_streams(cfg)
    for stream_name in active_streams:
        support_metric = STREAM_TO_SUPPORT_METRIC[stream_name]
        raw_values = [_stream_value(stream_name, row, row_by_quarter) for row in train_transition_rows]
        filled_values = _fill_train_values(raw_values)
        support_values = [
            _support_rank(row_by_quarter.get(str(row.get("quarter") or "")), support_metric)
            for row in train_transition_rows
        ]
        streams[stream_name] = _fit_stream_components(
            stream=stream_name,
            train_quarters=train_quarters,
            train_values=filled_values,
            support_values=support_values,
            holdout_quarters=holdout_quarters,
            positions=positions,
            eps=dataset.eps,
        )
        streams[stream_name]["diagnostics"]["raw_observed_count"] = int(sum(value is not None for value in raw_values))
        streams[stream_name]["diagnostics"]["support_metric"] = support_metric
    return {
        "schema_version": DECOMPOSITION_SCHEMA_VERSION,
        "active_components": _active_components(cfg),
        "stream_names": active_streams,
        "streams": streams,
        "contract": {
            "train_origin_safe": True,
            "holdout_values": "trend extrapolation plus train residual AR continuation plus carried-forward support status",
            "components": DECOMPOSITION_COMPONENTS,
        },
    }


def _feature_matrix(
    controls: dict[str, Any],
    quarters: list[str],
    streams: tuple[str, ...],
    components: tuple[str, ...],
    *,
    split_name: str,
) -> tuple[np.ndarray, list[str]]:
    columns: list[list[float]] = []
    names: list[str] = []
    key = "train_components" if split_name == "train" else "holdout_components"
    for stream_name in streams:
        stream = dict((controls.get("streams") or {}).get(stream_name) or {})
        component_map = dict(stream.get(key) or {})
        for component in components:
            columns.append([float((component_map.get(quarter) or {}).get(component) or 0.0) for quarter in quarters])
            names.append(f"{stream_name}:{component}")
    if not columns:
        return np.zeros((len(quarters), 0), dtype=np.float64), []
    return np.asarray(np.column_stack(columns), dtype=np.float64), names


def _apply_transition_decomposition(
    *,
    dataset: BlockedTimeDataset,
    paths: dict[str, Any],
    controls: dict[str, Any],
    cfg: DecompositionControlConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    train_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("train_hazard_map") or {}).items()}
    holdout_hazard_map = {quarter: dict(values) for quarter, values in dict(paths.get("holdout_hazard_map") or {}).items()}
    components = tuple(controls.get("active_components") or _active_components(cfg))
    diagnostics: dict[str, Any] = {}
    for transition in TRANSITION_NAMES:
        active_stream_set = set(controls.get("stream_names") or ())
        streams = tuple(stream for stream in TRANSITION_STREAMS.get(transition, ()) if stream in active_stream_set)
        x_train, feature_names = _feature_matrix(controls, train_quarters, streams, components, split_name="train")
        x_holdout, _ = _feature_matrix(controls, holdout_quarters, streams, components, split_name="holdout")
        y = np.asarray([
            logit(float((row.get("hazards") or {}).get(transition) or 0.0), eps=dataset.eps)
            - logit(float((train_hazard_map.get(str(row.get("quarter") or ""), {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            for row in train_transition_rows
        ], dtype=np.float64)
        effect = _bounded_effects(
            x_train=x_train,
            y=y,
            x_holdout=x_holdout,
            eps=dataset.eps,
            effect_shrinkage=cfg.effect_shrinkage,
        )
        train_effect = np.asarray(effect["train_effect"], dtype=np.float64)
        holdout_effect = np.asarray(effect["holdout_effect"], dtype=np.float64)
        for quarter, eta_shift in zip(train_quarters, train_effect):
            base_eta = logit(float((train_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            train_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + float(eta_shift)))
        for quarter, eta_shift in zip(holdout_quarters, holdout_effect):
            base_eta = logit(float((holdout_hazard_map.get(quarter, {}) or {}).get(transition) or 0.0), eps=dataset.eps)
            holdout_hazard_map.setdefault(quarter, {})[transition] = float(inv_logit(base_eta + float(eta_shift)))
        diagnostics[transition] = {
            "streams": list(streams),
            "feature_names": feature_names,
            "coefficients": {name: float(value) for name, value in zip(feature_names, np.asarray(effect["coefficients"], dtype=np.float64))},
            "effect_bound": float(effect["effect_bound"]),
            "mean_abs_train_effect": float(np.mean(np.abs(train_effect))) if train_effect.size else 0.0,
            "mean_abs_holdout_effect": float(np.mean(np.abs(holdout_effect))) if holdout_effect.size else 0.0,
        }
    return {"train_hazard_map": train_hazard_map, "holdout_hazard_map": holdout_hazard_map, "diagnostics": dict(paths.get("diagnostics") or {})}, diagnostics


def _apply_incidence_decomposition(
    *,
    dataset: BlockedTimeDataset,
    incidence_paths: dict[str, Any],
    controls: dict[str, Any],
    cfg: DecompositionControlConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_transition_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    components = tuple(controls.get("active_components") or _active_components(cfg))
    if "incidence" not in set(controls.get("stream_names") or ()):
        diagnostics = {
            "feature_names": [],
            "coefficients": {},
            "effect_bound": 0.0,
            "mean_abs_holdout_effect": 0.0,
            "status": "incidence_stream_disabled",
        }
        return dict(incidence_paths), diagnostics
    x_train, feature_names = _feature_matrix(controls, train_quarters, ("incidence",), components, split_name="train")
    x_holdout, _ = _feature_matrix(controls, holdout_quarters, ("incidence",), components, split_name="holdout")
    y = np.asarray([
        np.log1p(max(float((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff") or 0.0), 0.0))
        - np.log1p(max(float((incidence_paths.get("train_incidence_hazard_map") or {}).get(str(row.get("quarter") or "")) or 0.0), 0.0))
        for row in train_transition_rows
    ], dtype=np.float64)
    effect = _bounded_effects(
        x_train=x_train,
        y=y,
        x_holdout=x_holdout,
        eps=dataset.eps,
        effect_shrinkage=cfg.effect_shrinkage,
    )
    adjusted = dict(incidence_paths)
    train_map = dict(adjusted.get("train_incidence_hazard_map") or {})
    holdout_map = dict(adjusted.get("holdout_incidence_hazard_map") or {})
    for quarter, eta_shift in zip(train_quarters, np.asarray(effect["train_effect"], dtype=np.float64)):
        base = np.log1p(max(float(train_map.get(quarter) or 0.0), 0.0))
        train_map[quarter] = max(float(np.expm1(base + float(eta_shift))), 0.0)
    for quarter, eta_shift in zip(holdout_quarters, np.asarray(effect["holdout_effect"], dtype=np.float64)):
        base = np.log1p(max(float(holdout_map.get(quarter) or 0.0), 0.0))
        holdout_map[quarter] = max(float(np.expm1(base + float(eta_shift))), 0.0)
    adjusted["train_incidence_hazard_map"] = train_map
    adjusted["holdout_incidence_hazard_map"] = holdout_map
    diagnostics = {
        "feature_names": feature_names,
        "coefficients": {name: float(value) for name, value in zip(feature_names, np.asarray(effect["coefficients"], dtype=np.float64))},
        "effect_bound": float(effect["effect_bound"]),
        "mean_abs_holdout_effect": float(np.mean(np.abs(np.asarray(effect["holdout_effect"], dtype=np.float64)))) if len(effect["holdout_effect"]) else 0.0,
    }
    return adjusted, diagnostics


def apply_decomposition_controls(
    *,
    dataset: BlockedTimeDataset,
    paths: dict[str, Any],
    incidence_paths: dict[str, Any] | None,
    cfg: DecompositionControlConfig,
) -> dict[str, Any]:
    controls = build_hiv_decomposition_controls(dataset, cfg)
    adjusted_paths, transition_diagnostics = _apply_transition_decomposition(
        dataset=dataset,
        paths=paths,
        controls=controls,
        cfg=cfg,
    )
    adjusted_incidence = None
    incidence_diagnostics = None
    if incidence_paths is not None:
        adjusted_incidence, incidence_diagnostics = _apply_incidence_decomposition(
            dataset=dataset,
            incidence_paths=incidence_paths,
            controls=controls,
            cfg=cfg,
        )
    return {
        "paths": adjusted_paths,
        "incidence_paths": adjusted_incidence if adjusted_incidence is not None else incidence_paths,
        "diagnostics": {
            "schema_version": DECOMPOSITION_SCHEMA_VERSION,
            "controls": controls,
            "transition_effects": transition_diagnostics,
            "incidence_effects": incidence_diagnostics,
            "bounded_effect_contract": "linear control effects are bounded by the maximum train residual scale for each target",
        },
    }
