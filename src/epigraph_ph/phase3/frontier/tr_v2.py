from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3._lineage.national_reset_core import quarter_sort_key
from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, read_json, utc_now_iso, write_json

from .artifacts import TransitionResearchContext, build_transition_research_context, write_experiment_artifacts
from .phase2_structural_inputs import load_phase2_structural_inputs
from .registry import TRANSITION_NAMES
from .transition_engine import (
    STATE_NAMES,
    _build_observation_payload,
    _coverage_mean,
    _holdout_rows,
    _holdout_years_from_reference,
    _quarter_year,
    _load_front_half_reference,
    _lost_gap_share,
    _metric_scales,
    _normalized_mae,
    _reference_forecast_rows,
    _share_mean,
    _smape,
    _state_row,
    _train_rows,
    _train_transition_rows,
    _with_holdout_forecasts,
)


_HIV_PLUGIN = get_disease_plugin("hiv")
AGE01B_EXPERIMENT_ID = "AGE-01B-youth-diagnosis-modifier"
PEAK01F_EXPERIMENT_ID = "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast"


def _discover_latest_transition_experiment(experiment_id: str) -> tuple[str, Path]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / experiment_id
        evaluation_path = experiment_dir / "evaluation.json"
        if evaluation_path.exists():
            candidates.append((evaluation_path.stat().st_mtime, run_dir.name, experiment_dir))
    if not candidates:
        raise FileNotFoundError(f"No transition research experiment found for {experiment_id}")
    preferred = [row for row in candidates if "pytest" not in str(row[1]).lower()]
    pool = preferred or candidates
    _, run_id, experiment_dir = max(pool, key=lambda row: row[0])
    return run_id, experiment_dir


def _logit(value: float, *, eps: float = 1e-5) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / max(1.0 - clipped, eps)))


def _inv_logit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _load_age01b_lock() -> dict[str, Any]:
    run_id, experiment_dir = _discover_latest_transition_experiment(AGE01B_EXPERIMENT_ID)
    return {
        "reference_run_id": run_id,
        "experiment_id": AGE01B_EXPERIMENT_ID,
        "evaluation": read_json(experiment_dir / "evaluation.json", default={}),
        "baseline_comparison": read_json(experiment_dir / "baseline_comparison.json", default={}),
        "mechanistic_forecast": read_json(experiment_dir / "mechanistic_forecast.json", default={}),
    }


def _load_peak_gates() -> dict[tuple[str, str], float]:
    try:
        _run_id, experiment_dir = _discover_latest_transition_experiment(PEAK01F_EXPERIMENT_ID)
    except FileNotFoundError:
        return {}
    summary = read_json(experiment_dir / "peak_window_gate_summary.json", default={})
    rows = list(summary.get("rows") or []) if isinstance(summary, dict) else []
    gate_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("transition") or ""), str(row.get("quarter") or ""))
        if key[0] and key[1]:
            gate_map[key].append(float(row.get("detector_gate") or 0.0))
    return {key: float(np.mean(values)) for key, values in gate_map.items()}


def _baseline_hazard_map(age_lock: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = list((age_lock.get("mechanistic_forecast") or {}).get("transition_hazards") or [])
    return {
        str(row.get("quarter") or ""): {
            transition: float((row.get("hazards") or {}).get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
        for row in rows
        if str(row.get("quarter") or "")
    }


def _empirical_transition_dataset_for_holdout_years(ctx: TransitionResearchContext, holdout_years: list[int]) -> dict[str, Any]:
    reference = _load_front_half_reference()
    payload = _build_observation_payload(ctx)
    observation_rows = _with_holdout_forecasts(list(payload["rows"]), _reference_forecast_rows(reference))
    train_rows = _train_rows(observation_rows, holdout_years)
    holdout_rows = _holdout_rows(observation_rows, holdout_years)
    coverage_mean = _coverage_mean(reference, dict(payload["estimated_plhiv_by_quarter"]), train_rows)
    suppression_share_mean = _share_mean(train_rows, "virally_suppressed", "alive_on_art")
    testing_share_mean = _share_mean(train_rows, "tested_for_viral_load", "alive_on_art")
    lost_gap_share = _lost_gap_share(train_rows)
    eps = float(np.finfo(np.float32).eps)
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
    holdout_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in holdout_rows
    ]
    return {
        "holdout_years": holdout_years,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "train_state_rows": train_state_rows,
        "holdout_state_rows": holdout_state_rows,
        "train_transition_rows": _train_transition_rows(train_state_rows, holdout_years, eps),
        "testing_share_mean": testing_share_mean,
        "metric_scales": _metric_scales(train_rows + holdout_rows, eps),
        "eps": eps,
    }


def _empirical_transition_dataset(ctx: TransitionResearchContext) -> dict[str, Any]:
    reference = _load_front_half_reference()
    holdout_years = _holdout_years_from_reference(reference)
    return _empirical_transition_dataset_for_holdout_years(ctx, holdout_years)


def _rolling_origin_holdout_splits(
    ctx: TransitionResearchContext,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> list[dict[str, Any]]:
    payload = _build_observation_payload(ctx)
    observation_rows = list(payload.get("rows") or [])
    available_years = sorted(
        {
            _quarter_year(str(row.get("quarter") or ""))
            for row in observation_rows
            if row.get("diagnosed_plhiv") is not None and row.get("alive_on_art") is not None
        }
    )
    bounded_years = [year for year in available_years if int(start_year) <= int(year) <= int(end_year)]
    splits: list[dict[str, Any]] = []
    for train_end_year in bounded_years:
        train_years = [year for year in bounded_years if year <= train_end_year]
        holdout_years = [year for year in bounded_years if train_end_year < year <= train_end_year + int(horizon_years)]
        if len(train_years) < int(min_train_years) or not holdout_years:
            continue
        splits.append(
            {
                "train_end_year": int(train_end_year),
                "train_years": list(train_years),
                "holdout_years": list(holdout_years),
            }
        )
    return splits


def _train_based_baseline_hazard_map(dataset: dict[str, Any], *, mode: str = "last_train") -> dict[str, dict[str, float]]:
    train_transition_rows = list(dataset.get("train_transition_rows") or [])
    holdout_rows = list(dataset.get("holdout_rows") or [])
    if not train_transition_rows or not holdout_rows:
        return {}
    if str(mode) == "mean_train":
        baseline_hazards = {
            transition: float(
                np.mean([float((row.get("hazards") or {}).get(transition) or 0.0) for row in train_transition_rows])
            )
            for transition in TRANSITION_NAMES
        }
    else:
        baseline_hazards = {
            transition: float((train_transition_rows[-1].get("hazards") or {}).get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
    return {
        str(row.get("quarter") or ""): {
            transition: float(baseline_hazards.get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
        for row in holdout_rows
    }


def _quarter_feature_map(structural_inputs: Any) -> dict[str, dict[str, float]]:
    block_index = {block_id: idx for idx, block_id in enumerate(structural_inputs.block_axis)}
    feature_map: dict[str, dict[str, float]] = {}
    for quarter_idx, quarter in enumerate(structural_inputs.quarter_axis):
        feature_map[quarter] = {
            block_id: float(structural_inputs.national_quarter_tensor[0, quarter_idx, idx])
            for block_id, idx in block_index.items()
        }
    return feature_map


def _quarter_hidden_map(structural_inputs: Any) -> dict[str, np.ndarray]:
    hidden_map: dict[str, np.ndarray] = {}
    for quarter_idx, quarter in enumerate(structural_inputs.quarter_axis):
        hidden_map[quarter] = np.asarray(structural_inputs.hidden_mode_quarter_tensor[0, quarter_idx, :], dtype=np.float32)
    return hidden_map


def _direct_support_lookup(structural_inputs: Any) -> dict[tuple[str, str, int], dict[str, float]]:
    frontier_cfg = dict((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("frontier", {}) or {}))
    scale_cfg = dict(frontier_cfg.get("phase2_scale_evidence") or {})
    disagreement_penalty = max(float(scale_cfg.get("dispersion_penalty") or 0.5), 0.0)
    agreement_floor = float(np.clip(float(scale_cfg.get("agreement_floor") or 0.25), 0.05, 1.0))
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    summary_rows = list(getattr(structural_inputs, "direct_edge_summary_rows", []) or [])
    scale_rows = list(getattr(structural_inputs, "direct_edge_scale_rows", []) or [])
    if not summary_rows:
        summary_rows = list(getattr(structural_inputs, "direct_edge_rows", []) or [])
    for row in summary_rows:
        key = (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0))
        item = by_key.setdefault(
            key,
            {
                "weights": [],
                "stabilities": [],
                "scale_weights": {},
                "summary_weight": None,
                "summary_stability": None,
                "support_count": 0,
            },
        )
        item["summary_weight"] = float(row.get("weight") or 0.0)
        item["summary_stability"] = float(row.get("stability") or 0.0)
        item["support_count"] = max(int(item["support_count"]), int(row.get("support_count") or 0))
        for scale_row in list(row.get("scale_rows") or []):
            scale_name = str(scale_row.get("scale") or "")
            if not scale_name:
                continue
            item["scale_weights"][scale_name] = float(scale_row.get("weight") or 0.0)
            item["weights"].append(float(scale_row.get("weight") or 0.0))
            if scale_row.get("stability") is not None:
                item["stabilities"].append(float(scale_row.get("stability") or 0.0))
    for row in scale_rows:
        key = (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0))
        item = by_key.setdefault(
            key,
            {
                "weights": [],
                "stabilities": [],
                "scale_weights": {},
                "summary_weight": None,
                "summary_stability": None,
                "support_count": 0,
            },
        )
        scale_name = str(row.get("scale") or "")
        if scale_name:
            item["scale_weights"][scale_name] = float(row.get("weight") or 0.0)
        item["weights"].append(float(row.get("weight") or 0.0))
        if row.get("stability") is not None:
            item["stabilities"].append(float(row.get("stability") or 0.0))
        item["support_count"] = max(int(item["support_count"]), len(item["scale_weights"]))
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, item in by_key.items():
        weights = np.asarray(item["weights"], dtype=np.float32)
        stabilities = np.asarray(item["stabilities"], dtype=np.float32)
        mean_weight = float(np.mean(weights)) if weights.size else float(item["summary_weight"] or 0.0)
        mean_stability = float(np.mean(stabilities)) if stabilities.size else float(item["summary_stability"] or 0.0)
        scale_count = max(int(item["support_count"]), len(item["scale_weights"]))
        if weights.size <= 1:
            relative_dispersion = 0.0
            sign_consistency = 1.0 if weights.size else 0.0
        else:
            centered_scale = max(abs(float(np.mean(weights))), 1e-6)
            relative_dispersion = float(np.std(weights) / centered_scale)
            signs = np.sign(weights[np.abs(weights) > 1e-6])
            sign_consistency = float(abs(np.sum(signs)) / max(signs.size, 1)) if signs.size else 1.0
        scale_agreement = float(np.clip(sign_consistency / (1.0 + disagreement_penalty * relative_dispersion), agreement_floor, 1.0))
        canonical_weight = float(item["scale_weights"].get("national", item["summary_weight"] if item["summary_weight"] is not None else mean_weight))
        canonical_stability = float(
            item["summary_stability"]
            if item["summary_stability"] is not None
            else mean_stability
        )
        lookup[key] = {
            "weight": canonical_weight,
            "stability": canonical_stability,
            "support_count": int(scale_count),
            "scale_count": int(scale_count),
            "mean_weight": round(mean_weight, 6),
            "mean_stability": round(mean_stability, 6),
            "relative_dispersion": round(relative_dispersion, 6),
            "sign_consistency": round(sign_consistency, 6),
            "scale_agreement": round(scale_agreement, 6),
        }
    return lookup


def _transition_multiscale_support(structural_inputs: Any, transition: str) -> dict[str, Any]:
    frontier_cfg = dict((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("frontier", {}) or {}))
    support_cfg = dict(frontier_cfg.get("phase2_multiscale_support") or {})
    hook_map = dict(support_cfg.get("transition_hook_map") or {})
    relevant_hooks = {str(value) for value in list(hook_map.get(str(transition)) or []) if str(value)}
    if not relevant_hooks:
        return {"multiplier": 1.0, "matching_factor_count": 0, "mean_support_count": 0.0, "factor_ids": []}
    matching_rows = [
        dict(row)
        for row in list(getattr(structural_inputs, "multiscale_support_rows", []) or [])
        if relevant_hooks.intersection({str(value) for value in list(row.get("transition_hooks") or []) if str(value)})
    ]
    if not matching_rows:
        return {"multiplier": 1.0, "matching_factor_count": 0, "mean_support_count": 0.0, "factor_ids": []}
    mean_support = float(np.mean([float(row.get("support_count") or 0.0) for row in matching_rows]))
    support_scale = float(support_cfg.get("support_scale") or 0.08)
    max_multiplier = float(support_cfg.get("max_multiplier") or 1.35)
    multiplier = min(1.0 + support_scale * float(np.log1p(max(mean_support, 0.0))), max_multiplier)
    return {
        "multiplier": float(multiplier),
        "matching_factor_count": int(len(matching_rows)),
        "mean_support_count": round(mean_support, 6),
        "factor_ids": [str(row.get("factor_id") or "") for row in matching_rows],
    }


def _target_prior_map(transition_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    target_cfgs: dict[str, dict[str, Any]] = {}
    explicit = dict(transition_cfg.get("target_blocks") or {})
    for target_block, payload in explicit.items():
        target_cfgs[str(target_block)] = dict(payload or {})
    for target_block, payload in transition_cfg.items():
        if str(target_block) == "target_blocks":
            continue
        if isinstance(payload, dict) and ("lags" in payload or "prior_scale" in payload):
            target_cfgs.setdefault(str(target_block), dict(payload))
    return target_cfgs


def _resolve_target_source_mapping(target_mapping: dict[str, Any], source_block: str) -> dict[str, Any] | None:
    source_cfgs = dict(target_mapping.get("source_blocks") or {})
    if source_cfgs:
        source_mapping = source_cfgs.get(str(source_block))
        if source_mapping is None:
            return None
        merged = dict(target_mapping)
        merged.pop("source_blocks", None)
        merged.update(dict(source_mapping or {}))
        return merged
    return dict(target_mapping)


def _fit_prior_regression(
    *,
    target_values: np.ndarray,
    design_matrix: np.ndarray,
    prior_precisions: np.ndarray,
    prior_means: np.ndarray | None = None,
) -> np.ndarray:
    if design_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    precision_vector = np.asarray(prior_precisions, dtype=np.float32)
    mean_vector = np.zeros_like(precision_vector, dtype=np.float32) if prior_means is None else np.asarray(prior_means, dtype=np.float32)
    gram = design_matrix.T @ design_matrix
    gram += np.diag(precision_vector)
    rhs = design_matrix.T @ target_values
    rhs += precision_vector * mean_vector
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
    return np.asarray(beta, dtype=np.float32)


def _fit_direct_transition_effects(
    *,
    structural_inputs: Any,
    dataset: dict[str, Any],
    include_multiscale: bool = True,
) -> dict[str, Any]:
    frontier_cfg = dict((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("frontier", {}) or {}))
    transition_prior_map = dict(frontier_cfg.get("phase2_transition_prior_map") or {})
    quarter_features = _quarter_feature_map(structural_inputs)
    support_lookup = _direct_support_lookup(structural_inputs)
    summaries: dict[str, Any] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    for transition in TRANSITION_NAMES:
        transition_cfg = dict(transition_prior_map.get(transition) or {})
        target_cfgs = _target_prior_map(transition_cfg)
        train_rows = [row for row in list(dataset["train_transition_rows"]) if str(row.get("quarter") or "") in quarter_features]
        if not train_rows:
            summaries[transition] = {"feature_count": 0, "coefficients": []}
            continue
        multiscale_support = _transition_multiscale_support(structural_inputs, transition) if include_multiscale else {
            "multiplier": 1.0,
            "matching_factor_count": 0,
            "mean_support_count": 0.0,
            "factor_ids": [],
        }
        feature_specs: list[tuple[str, str, int, float, float, dict[str, float]]] = []
        for row in structural_inputs.direct_edge_rows:
            source_block = str(row.get("source") or "")
            target_block = str(row.get("target") or "")
            lag = int(row.get("lag") or 0)
            if target_block not in target_cfgs:
                continue
            mapping = _resolve_target_source_mapping(dict(target_cfgs.get(target_block) or {}), source_block)
            if mapping is None:
                continue
            allowed_lags = [int(value) for value in list(mapping.get("lags") or [])]
            if allowed_lags and lag not in allowed_lags:
                continue
            support = support_lookup.get((source_block, target_block, lag))
            if support is None:
                continue
            prior_scale = float(mapping.get("prior_scale") or 0.25)
            prior_mean_scale = float(mapping.get("prior_mean_scale") or prior_scale)
            feature_specs.append((source_block, target_block, lag, prior_scale, prior_mean_scale, support))
        if not feature_specs:
            summaries[transition] = {"feature_count": 0, "coefficients": []}
            continue
        y = np.asarray([_logit(float((row.get("hazards") or {}).get(transition) or 1e-5)) for row in train_rows], dtype=np.float32)
        intercept = float(np.mean(y))
        X = []
        prior_precisions = []
        for quarter_row in train_rows:
            quarter = str(quarter_row.get("quarter") or "")
            quarter_position = structural_inputs.quarter_axis.index(quarter) if quarter in structural_inputs.quarter_axis else -1
            if quarter_position < 0:
                continue
            row_values = []
            for block_id, _target_block, lag, _prior_scale, _prior_mean_scale, _support in feature_specs:
                source_idx = quarter_position - lag
                if source_idx < 0:
                    row_values.append(0.0)
                    continue
                source_quarter = structural_inputs.quarter_axis[source_idx]
                row_values.append(float(quarter_features.get(source_quarter, {}).get(block_id) or 0.0))
            X.append(row_values)
        X_matrix = np.asarray(X, dtype=np.float32)
        prior_precisions = np.asarray(
            [
                max(
                    float(multiscale_support["multiplier"])
                    * (1.0 + float(support["support_count"]))
                    * max(float(support["stability"]), 0.05)
                    * max(float(support.get("scale_agreement", 1.0)), 0.1),
                    1e-3,
                )
                / max(float(prior_scale) ** 2, 1e-6)
                for _block_id, _target_block, _lag, prior_scale, _prior_mean_scale, support in feature_specs
            ],
            dtype=np.float32,
        )
        prior_means = np.asarray(
            [
                float(prior_mean_scale)
                * float(support["weight"])
                * max(float(support.get("scale_agreement", 1.0)), 0.1)
                for _block_id, _target_block, _lag, _prior_scale, prior_mean_scale, support in feature_specs
            ],
            dtype=np.float32,
        )
        beta = _fit_prior_regression(
            target_values=y - intercept,
            design_matrix=X_matrix,
            prior_precisions=prior_precisions,
            prior_means=prior_means,
        )
        summaries[transition] = {
            "feature_count": int(len(feature_specs)),
            "intercept_logit": round(intercept, 6),
            "multiscale_support": dict(multiscale_support),
            "coefficients": [
                {
                    "block_id": block_id,
                    "target_block_id": target_block,
                    "lag": lag,
                    "coefficient": round(float(beta[idx]), 6),
                    "prior_scale": round(float(prior_scale), 6),
                    "prior_mean": round(float(prior_means[idx]), 6),
                    "prior_precision": round(float(prior_precisions[idx]), 6),
                    "phase2_weight": round(float(support["weight"]), 6),
                    "support_count": int(support["support_count"]),
                    "scale_count": int(support.get("scale_count", support["support_count"])),
                    "stability": round(float(support["stability"]), 6),
                    "scale_agreement": round(float(support.get("scale_agreement", 1.0)), 6),
                    "relative_dispersion": round(float(support.get("relative_dispersion", 0.0)), 6),
                    "multiscale_support_multiplier": round(float(multiscale_support["multiplier"]), 6),
                }
                for idx, (block_id, target_block, lag, prior_scale, _prior_mean_scale, support) in enumerate(feature_specs)
            ],
        }
        for quarter in structural_inputs.quarter_axis:
            quarter_position = structural_inputs.quarter_axis.index(quarter)
            delta = 0.0
            for idx, (block_id, _target_block, lag, _prior_scale, _prior_mean_scale, _support) in enumerate(feature_specs):
                source_idx = quarter_position - lag
                if source_idx < 0:
                    continue
                source_quarter = structural_inputs.quarter_axis[source_idx]
                delta += float(beta[idx]) * float(quarter_features.get(source_quarter, {}).get(block_id) or 0.0)
            predictions[transition][quarter] = float(delta)
    return {"summary": summaries, "quarter_adjustments": {key: dict(value) for key, value in predictions.items()}}


def _hidden_prior_parameters(structural_inputs: Any, hidden_dim: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    quarter_tensor = np.asarray(getattr(structural_inputs, "hidden_mode_quarter_tensor", np.zeros((1, 0, 0), dtype=np.float32)), dtype=np.float32)
    if quarter_tensor.ndim != 3 or quarter_tensor.shape[-1] <= 0:
        return np.ones((hidden_dim,), dtype=np.float32), np.zeros((hidden_dim,), dtype=np.float32), []
    ar1_lookup = {
        int(row.get("hidden_mode") or 0): float(row.get("phi") or 0.0)
        for row in list((getattr(structural_inputs, "hidden_mode_summary", {}) or {}).get("ar1_rows") or [])
    }
    stds = np.nanstd(np.asarray(quarter_tensor[0, :, :hidden_dim], dtype=np.float32), axis=0)
    prior_precisions: list[float] = []
    prior_means: list[float] = []
    diagnostics: list[dict[str, float]] = []
    for mode_idx in range(hidden_dim):
        phi = float(ar1_lookup.get(mode_idx, 0.0))
        scale = max(float(stds[mode_idx]) if mode_idx < stds.shape[0] else 0.0, 1e-3)
        prior_scale = max(scale * (0.50 + 0.50 * abs(phi)), 0.05)
        prior_precisions.append(float(1.0 / max(prior_scale**2, 1e-6)))
        prior_means.append(0.0)
        diagnostics.append(
            {
                "hidden_mode": int(mode_idx),
                "prior_scale": round(float(prior_scale), 6),
                "prior_precision": round(float(prior_precisions[-1]), 6),
                "phi": round(float(phi), 6),
            }
        )
    return np.asarray(prior_precisions, dtype=np.float32), np.asarray(prior_means, dtype=np.float32), diagnostics


def _fit_hidden_transition_effects(
    *,
    structural_inputs: Any,
    dataset: dict[str, Any],
    direct_adjustments: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rank_cap = int((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("frontier", {}) or {}).get("phase2_hidden_shock", {}).get("rank_cap", 3))
    quarter_hidden = _quarter_hidden_map(structural_inputs)
    hidden_dim = min(int(structural_inputs.hidden_mode_quarter_tensor.shape[-1]), max(rank_cap, 0))
    summaries: dict[str, Any] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    if hidden_dim <= 0:
        return {"summary": {transition: {"rank_used": 0, "coefficients": []} for transition in TRANSITION_NAMES}, "quarter_adjustments": {}}
    prior_precisions, prior_means, prior_diagnostics = _hidden_prior_parameters(structural_inputs, hidden_dim)
    for transition in TRANSITION_NAMES:
        train_rows = [row for row in list(dataset["train_transition_rows"]) if str(row.get("quarter") or "") in quarter_hidden]
        if not train_rows:
            summaries[transition] = {"rank_used": 0, "coefficients": []}
            continue
        y = []
        X = []
        for row in train_rows:
            quarter = str(row.get("quarter") or "")
            base_value = _logit(float((row.get("hazards") or {}).get(transition) or 1e-5))
            direct_delta = float((direct_adjustments.get(transition) or {}).get(quarter) or 0.0)
            y.append(base_value - direct_delta)
            X.append(np.asarray(quarter_hidden[quarter][:hidden_dim], dtype=np.float32))
        X_matrix = np.asarray(X, dtype=np.float32)
        y_vector = np.asarray(y, dtype=np.float32)
        intercept = float(np.mean(y_vector))
        beta = _fit_prior_regression(
            target_values=y_vector - intercept,
            design_matrix=X_matrix,
            prior_precisions=prior_precisions,
            prior_means=prior_means,
        )
        summaries[transition] = {
            "rank_used": int(hidden_dim),
            "intercept_logit": round(intercept, 6),
            "basis_kind": str((getattr(structural_inputs, "hidden_mode_summary", {}) or {}).get("basis_kind") or "unknown"),
            "identified_hidden_shock_process": bool((getattr(structural_inputs, "hidden_mode_summary", {}) or {}).get("identified_hidden_shock_process", False)),
            "prior_diagnostics": list(prior_diagnostics),
            "coefficients": [
                {
                    "hidden_mode": int(idx),
                    "coefficient": round(float(beta[idx]), 6),
                    "prior_precision": round(float(prior_precisions[idx]), 6),
                }
                for idx in range(hidden_dim)
            ],
        }
        for quarter, values in quarter_hidden.items():
            predictions[transition][quarter] = float(np.dot(np.asarray(values[:hidden_dim], dtype=np.float32), beta))
    return {"summary": summaries, "quarter_adjustments": {key: dict(value) for key, value in predictions.items()}}


def _simulate_holdout(
    *,
    dataset: dict[str, Any],
    baseline_hazards: dict[str, dict[str, float]],
    direct_adjustments: dict[str, dict[str, float]] | None = None,
    hidden_adjustments: dict[str, dict[str, float]] | None = None,
    peak_gates: dict[tuple[str, str], float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    direct_adjustments = direct_adjustments or {}
    hidden_adjustments = hidden_adjustments or {}
    peak_gates = peak_gates or {}
    current_state = {state_name: float(dataset["train_state_rows"][-1]["state_values"][state_name]) for state_name in STATE_NAMES}
    forecast_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for target_row in sorted(list(dataset["holdout_rows"]), key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))):
        quarter = str(target_row.get("quarter") or "")
        hazards: dict[str, float] = {}
        for transition in TRANSITION_NAMES:
            base_hazard = float((baseline_hazards.get(quarter) or {}).get(transition) or 0.0)
            base_logit = _logit(max(base_hazard, 1e-5))
            direct_delta = float((direct_adjustments.get(transition) or {}).get(quarter) or 0.0)
            hidden_delta = float((hidden_adjustments.get(transition) or {}).get(quarter) or 0.0)
            gate = float(peak_gates.get((transition, quarter), 1.0))
            hazards[transition] = _inv_logit(base_logit + gate * (direct_delta + hidden_delta))
        u_to_d = min(current_state["U"], hazards["U_to_D"] * current_state["U"])
        d_to_a = min(current_state["D"] + u_to_d, hazards["D_to_A"] * current_state["D"])
        a_to_v = min(current_state["A"] + d_to_a, hazards["A_to_V"] * current_state["A"])
        a_to_l = min(current_state["A"] + d_to_a - a_to_v, hazards["A_to_L"] * current_state["A"])
        l_to_a = min(current_state["L"] + a_to_l, hazards["L_to_A"] * current_state["L"])
        current_state = {
            "U": max(current_state["U"] - u_to_d, 0.0),
            "D": max(current_state["D"] + u_to_d - d_to_a, 0.0),
            "A": max(current_state["A"] + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(current_state["V"] + a_to_v, 0.0),
            "L": max(current_state["L"] + a_to_l - l_to_a, 0.0),
        }
        alive_on_art = current_state["A"] + current_state["V"]
        forecast_rows.append(
            {
                "quarter": quarter,
                "state_values": {state_name: round(float(current_state[state_name]), 6) for state_name in STATE_NAMES},
                "diagnosed_plhiv": round(float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]), 6),
                "alive_on_art": round(float(alive_on_art), 6),
                "virally_suppressed": round(float(current_state["V"]), 6),
                "tested_for_viral_load": round(float(dataset["testing_share_mean"] * alive_on_art), 6),
                "new_diagnosed_cases_period": round(float(u_to_d), 6),
            }
        )
        hazard_rows.append({"quarter": quarter, "hazards": {name: round(float(value), 6) for name, value in hazards.items()}})
    prediction_rows = [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
        }
        for row in forecast_rows
    ]
    mae = _normalized_mae(prediction_rows, dataset["holdout_rows"], dataset["metric_scales"], dataset["eps"])
    smape = _smape(prediction_rows, dataset["holdout_rows"], dataset["eps"])
    evaluation = {
        "holdout_quarters": [str(row.get("quarter") or "") for row in list(dataset["holdout_rows"])],
        "holdout_rows": [
            {"quarter": str(target.get("quarter") or ""), "prediction": prediction, "target": dict(target)}
            for prediction, target in zip(prediction_rows, dataset["holdout_rows"])
        ],
        "metric_scales": dataset["metric_scales"],
        "model_mean_absolute_error": round(float(mae), 6),
        "model_smape": round(float(smape), 6),
    }
    return forecast_rows, hazard_rows, evaluation


def _baseline_comparison(age_lock: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(age_lock.get("baseline_comparison") or {})
    return {
        "branch_reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "branch_reference_run_id": str(age_lock.get("reference_run_id") or ""),
        "branch_reference_mean_absolute_error": float(baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_mean_absolute_error": float(evaluation.get("model_mean_absolute_error") or 0.0),
        "model_smape": float(evaluation.get("model_smape") or 0.0),
    }


def _write_standard_experiment_metadata(ctx: TransitionResearchContext, *, description: str) -> None:
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": description,
            "source_run_id": ctx.source_run_id,
        },
        coverage_summary={"source_run_id": ctx.source_run_id, "phase2_dir": str(ctx.phase2_dir), "phase15_dir": str(ctx.phase15_dir)},
        decision={"implemented": True, "scientific_role": "phase2_structural_frontier"},
        numeric_justification=[{"id": "structural_frontier", "justification": "Consume frozen Phase 2 structural payload directly in TR-V2."}],
    )


def run_tr_v2_00(ctx: TransitionResearchContext) -> dict[str, Any]:
    age_lock = _load_age01b_lock()
    _write_standard_experiment_metadata(ctx, description="Exact locked reproduction of the frozen AGE-01B transition frontier baseline.")
    baseline_lock = {
        "experiment_id": AGE01B_EXPERIMENT_ID,
        "reference_run_id": age_lock["reference_run_id"],
        "evaluation_path": str(_discover_latest_transition_experiment(AGE01B_EXPERIMENT_ID)[1] / "evaluation.json"),
        "mechanistic_forecast_path": str(_discover_latest_transition_experiment(AGE01B_EXPERIMENT_ID)[1] / "mechanistic_forecast.json"),
    }
    write_json(ctx.experiment_dir / "baseline_lock.json", baseline_lock)
    write_json(ctx.experiment_dir / "baseline_comparison.json", age_lock["baseline_comparison"])
    write_json(ctx.experiment_dir / "evaluation.json", age_lock["evaluation"])
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", age_lock["mechanistic_forecast"])
    return {
        "baseline_lock": str(ctx.experiment_dir / "baseline_lock.json"),
        "baseline_comparison": str(ctx.experiment_dir / "baseline_comparison.json"),
        "evaluation": str(ctx.experiment_dir / "evaluation.json"),
        "mechanistic_forecast": str(ctx.experiment_dir / "mechanistic_forecast.json"),
    }


def _run_structural_variant(
    *,
    ctx: TransitionResearchContext,
    include_direct: bool,
    include_hidden: bool,
    include_multiscale: bool,
    include_peak_gating: bool,
) -> dict[str, Any]:
    age_lock = _load_age01b_lock()
    structural_inputs = load_phase2_structural_inputs(ctx)
    dataset = _empirical_transition_dataset(ctx)
    peak_gates = _load_peak_gates() if include_peak_gating else {}
    baseline_hazards = _baseline_hazard_map(age_lock)
    direct_fit = _fit_direct_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        include_multiscale=include_multiscale,
    ) if include_direct else {"summary": {}, "quarter_adjustments": {}}
    hidden_fit = _fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments=dict(direct_fit.get("quarter_adjustments") or {}),
    ) if include_hidden else {"summary": {}, "quarter_adjustments": {}}
    forecast_rows, hazard_rows, evaluation = _simulate_holdout(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_adjustments=dict(direct_fit.get("quarter_adjustments") or {}),
        hidden_adjustments=dict(hidden_fit.get("quarter_adjustments") or {}),
        peak_gates=peak_gates,
    )
    return {
        "age_lock": age_lock,
        "structural_inputs": structural_inputs,
        "direct_fit": direct_fit,
        "hidden_fit": hidden_fit,
        "forecast_rows": forecast_rows,
        "hazard_rows": hazard_rows,
        "evaluation": evaluation,
        "baseline_comparison": _baseline_comparison(age_lock, evaluation),
        "peak_gating": {
            "enabled": include_peak_gating,
            "gate_count": len(peak_gates),
        },
        "multiscale_support": {
            "enabled": include_multiscale,
        },
    }


def run_tr_v2_01(ctx: TransitionResearchContext) -> dict[str, Any]:
    _write_standard_experiment_metadata(ctx, description="Direct Phase 2 temporal edges become structured hazard priors on the locked AGE-01B baseline.")
    result = _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=False, include_multiscale=False, include_peak_gating=False)
    write_json(ctx.experiment_dir / "baseline_lock.json", {"reference_experiment_id": AGE01B_EXPERIMENT_ID, "reference_run_id": result["age_lock"]["reference_run_id"]})
    write_json(ctx.experiment_dir / "phase2_direct_hazard_prior_summary.json", result["direct_fit"]["summary"])
    write_json(ctx.experiment_dir / "baseline_comparison.json", result["baseline_comparison"])
    write_json(ctx.experiment_dir / "evaluation.json", result["evaluation"])
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", {"forecast_rows": result["forecast_rows"], "transition_hazards": result["hazard_rows"]})
    return {"experiment_dir": str(ctx.experiment_dir)}


def run_tr_v2_02(ctx: TransitionResearchContext) -> dict[str, Any]:
    _write_standard_experiment_metadata(ctx, description="Direct Phase 2 temporal edges plus hidden low-rank mode shocks modulate the locked AGE-01B hazards.")
    result = _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_multiscale=False, include_peak_gating=False)
    write_json(ctx.experiment_dir / "baseline_lock.json", {"reference_experiment_id": AGE01B_EXPERIMENT_ID, "reference_run_id": result["age_lock"]["reference_run_id"]})
    write_json(ctx.experiment_dir / "phase2_direct_hazard_prior_summary.json", result["direct_fit"]["summary"])
    write_json(ctx.experiment_dir / "phase2_hidden_shock_summary.json", result["hidden_fit"]["summary"])
    write_json(ctx.experiment_dir / "baseline_comparison.json", result["baseline_comparison"])
    write_json(ctx.experiment_dir / "evaluation.json", result["evaluation"])
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", {"forecast_rows": result["forecast_rows"], "transition_hazards": result["hazard_rows"]})
    return {"experiment_dir": str(ctx.experiment_dir)}


def run_tr_v2_03(ctx: TransitionResearchContext) -> dict[str, Any]:
    _write_standard_experiment_metadata(ctx, description="Ablate direct priors, hidden shocks, and peak gating on top of the locked AGE-01B baseline.")
    variants = {
        "direct_only": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=False, include_multiscale=False, include_peak_gating=False),
        "hidden_only": _run_structural_variant(ctx=ctx, include_direct=False, include_hidden=True, include_multiscale=False, include_peak_gating=False),
        "direct_plus_hidden": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_multiscale=False, include_peak_gating=False),
        "direct_plus_multiscale": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=False, include_multiscale=True, include_peak_gating=False),
        "direct_plus_hidden_plus_multiscale": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_multiscale=True, include_peak_gating=False),
        "direct_plus_hidden_plus_multiscale_plus_peak_gating": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_multiscale=True, include_peak_gating=True),
    }
    summary = {
        "rows": [
            {
                "variant_id": variant_id,
                "model_mean_absolute_error": float(result["evaluation"]["model_mean_absolute_error"]),
                "model_smape": float(result["evaluation"]["model_smape"]),
                "direct_feature_count": int(sum(int((result["direct_fit"]["summary"].get(name) or {}).get("feature_count") or 0) for name in TRANSITION_NAMES)),
                "hidden_rank_used": int(sum(int((result["hidden_fit"]["summary"].get(name) or {}).get("rank_used") or 0) for name in TRANSITION_NAMES)),
                "multiscale_enabled": bool(result["multiscale_support"]["enabled"]),
                "peak_gating_enabled": bool(result["peak_gating"]["enabled"]),
            }
            for variant_id, result in variants.items()
        ]
    }
    best_variant = min(summary["rows"], key=lambda row: float(row["model_mean_absolute_error"]))
    selected = variants[str(best_variant["variant_id"])]
    write_json(ctx.experiment_dir / "baseline_lock.json", {"reference_experiment_id": AGE01B_EXPERIMENT_ID, "reference_run_id": selected["age_lock"]["reference_run_id"]})
    write_json(ctx.experiment_dir / "phase2_direct_hazard_prior_summary.json", selected["direct_fit"]["summary"])
    write_json(ctx.experiment_dir / "phase2_hidden_shock_summary.json", selected["hidden_fit"]["summary"])
    write_json(ctx.experiment_dir / "transition_frontier_ablation_summary.json", summary)
    write_json(ctx.experiment_dir / "baseline_comparison.json", selected["baseline_comparison"])
    write_json(ctx.experiment_dir / "evaluation.json", selected["evaluation"])
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", {"forecast_rows": selected["forecast_rows"], "transition_hazards": selected["hazard_rows"]})
    return {"experiment_dir": str(ctx.experiment_dir), "selected_variant": str(best_variant["variant_id"])}


def _rolling_origin_variant_metrics(
    *,
    structural_inputs: Any,
    dataset: dict[str, Any],
    baseline_hazards: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    def _evaluate(
        *,
        direct_summary: dict[str, Any],
        direct_adjustments: dict[str, dict[str, float]],
        hidden_summary: dict[str, Any] | None = None,
        hidden_adjustments: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        _forecast_rows, _hazard_rows, evaluation = _simulate_holdout(
            dataset=dataset,
            baseline_hazards=baseline_hazards,
            direct_adjustments=direct_adjustments,
            hidden_adjustments=hidden_adjustments or {},
            peak_gates={},
        )
        return {
            "mae": float(evaluation["model_mean_absolute_error"]),
            "smape": float(evaluation["model_smape"]),
            "direct_feature_count": float(
                sum(int((direct_summary.get(name) or {}).get("feature_count") or 0) for name in TRANSITION_NAMES)
            ),
            "hidden_rank_used": float(
                sum(int((hidden_summary or {}).get(name, {}).get("rank_used") or 0) for name in TRANSITION_NAMES)
            ),
        }

    no_prior = _evaluate(
        direct_summary={transition: {"feature_count": 0, "coefficients": []} for transition in TRANSITION_NAMES},
        direct_adjustments={},
    )
    direct_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=False)
    direct_only = _evaluate(
        direct_summary=direct_fit["summary"],
        direct_adjustments=direct_fit["quarter_adjustments"],
    )
    direct_multiscale_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=True)
    direct_plus_multiscale = _evaluate(
        direct_summary=direct_multiscale_fit["summary"],
        direct_adjustments=direct_multiscale_fit["quarter_adjustments"],
    )
    hidden_fit = _fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments={},
    )
    hidden_only = _evaluate(
        direct_summary={transition: {"feature_count": 0, "coefficients": []} for transition in TRANSITION_NAMES},
        direct_adjustments={},
        hidden_summary=hidden_fit["summary"],
        hidden_adjustments=hidden_fit["quarter_adjustments"],
    )
    direct_plus_hidden = _evaluate(
        direct_summary=direct_fit["summary"],
        direct_adjustments=direct_fit["quarter_adjustments"],
        hidden_summary=_fit_hidden_transition_effects(
            structural_inputs=structural_inputs,
            dataset=dataset,
            direct_adjustments=direct_fit["quarter_adjustments"],
        )["summary"],
        hidden_adjustments=_fit_hidden_transition_effects(
            structural_inputs=structural_inputs,
            dataset=dataset,
            direct_adjustments=direct_fit["quarter_adjustments"],
        )["quarter_adjustments"],
    )
    hidden_with_direct = _fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments=direct_multiscale_fit["quarter_adjustments"],
    )
    direct_plus_hidden_plus_multiscale = _evaluate(
        direct_summary=direct_multiscale_fit["summary"],
        direct_adjustments=direct_multiscale_fit["quarter_adjustments"],
        hidden_summary=hidden_with_direct["summary"],
        hidden_adjustments=hidden_with_direct["quarter_adjustments"],
    )
    return {
        "no_priors": no_prior,
        "direct_only": direct_only,
        "direct_plus_multiscale": direct_plus_multiscale,
        "hidden_only": hidden_only,
        "direct_plus_hidden": direct_plus_hidden,
        "direct_plus_hidden_plus_multiscale": direct_plus_hidden_plus_multiscale,
    }


def _plot_rolling_origin_lines(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    variants: list[tuple[str, str]],
    title: str,
    ylabel: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_end_years = [int(row["train_end_year"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id, label in variants:
        ax.plot(
            train_end_years,
            [float((row.get("variants") or {}).get(variant_id, {}).get(metric_key) or np.nan) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Train End Year")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_rolling_origin_deltas(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    baseline_variant: str,
    compare_variants: list[tuple[str, str]],
    title: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_end_years = [int(row["train_end_year"]) for row in rows]
    x = np.arange(len(train_end_years), dtype=np.float32)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (variant_id, label) in enumerate(compare_variants):
        delta_values = [
            float((row.get("variants") or {}).get(variant_id, {}).get("mae") or np.nan)
            - float((row.get("variants") or {}).get(baseline_variant, {}).get("mae") or np.nan)
            for row in rows
        ]
        ax.bar(x + (idx - (len(compare_variants) - 1) / 2.0) * width, delta_values, width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in train_end_years])
    ax.set_xlabel("Train End Year")
    ax.set_ylabel("MAE Delta vs No Priors")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _load_early_partial_target_rows(
    ctx: TransitionResearchContext,
    *,
    start_year: int = 2010,
    end_year: int = 2016,
) -> list[dict[str, Any]]:
    archive_rows = list(read_json(ctx.source_run_dir / "harp_archive" / "historical_metric_rows.json", default=[]))
    supported_metrics = {
        "annual_aids_deaths": "mean",
        "annual_new_infections": "mean",
        "art_median_age": "mean",
        "estimated_plhiv": "mean",
        "median_cd4_at_enrollment": "mean",
        "new_diagnosed_cases_monthly": "mean",
        "youth_cases_15_24_period": "mean",
    }
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in archive_rows:
        region = str(row.get("region") or "").lower()
        metric_name = str(row.get("metric_name") or "")
        period = str(row.get("period_end") or row.get("time") or "")
        if region != "national" or metric_name not in supported_metrics:
            continue
        if len(period) < 4 or not period[:4].isdigit():
            continue
        year = int(period[:4])
        if year < int(start_year) or year > int(end_year):
            continue
        value = row.get("value")
        if value is None:
            continue
        grouped[(metric_name, year)].append(float(value))
    output_rows: list[dict[str, Any]] = []
    for (metric_name, year), values in sorted(grouped.items()):
        if not values:
            continue
        output_rows.append(
            {
                "metric_name": metric_name,
                "year": int(year),
                "value": round(float(np.mean(values)), 6),
                "aggregation": supported_metrics[metric_name],
                "observation_count": int(len(values)),
            }
        )
    return output_rows


def _aggregate_quarter_adjustments_to_year_features(quarter_adjustments: dict[str, dict[str, float]]) -> dict[int, dict[str, float]]:
    annual_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for transition, quarter_map in dict(quarter_adjustments or {}).items():
        for quarter, value in dict(quarter_map or {}).items():
            annual_values[int(_quarter_year(str(quarter)))][str(transition)].append(float(value))
    features_by_year: dict[int, dict[str, float]] = {}
    for year, transition_values in annual_values.items():
        feature_row: dict[str, float] = {}
        for transition in TRANSITION_NAMES:
            values = transition_values.get(str(transition), [])
            feature_row[f"{transition}_mean"] = float(np.mean(values)) if values else 0.0
            feature_row[f"{transition}_abs_mean"] = float(np.mean(np.abs(values))) if values else 0.0
        features_by_year[int(year)] = feature_row
    return features_by_year


def _fit_partial_readout(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    ridge_penalty: float = 1.0,
) -> np.ndarray:
    design = np.asarray(train_x, dtype=np.float32)
    target = np.asarray(train_y, dtype=np.float32)
    intercept = np.ones((design.shape[0], 1), dtype=np.float32)
    augmented = np.hstack([intercept, design])
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0], dtype=np.float32) * float(ridge_penalty)
    penalty[0, 0] = 0.0
    rhs = augmented.T @ target
    try:
        beta = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + penalty) @ rhs
    return np.asarray(beta, dtype=np.float32)


def _partial_readout_predict(beta: np.ndarray, feature_row: dict[str, float], feature_names: list[str]) -> float:
    vector = np.asarray([1.0] + [float(feature_row.get(name) or 0.0) for name in feature_names], dtype=np.float32)
    return float(np.dot(vector, np.asarray(beta, dtype=np.float32)))


def _build_partial_observation_features(ctx: TransitionResearchContext) -> dict[str, dict[int, dict[str, float]]]:
    structural_inputs = load_phase2_structural_inputs(ctx)
    dataset = _empirical_transition_dataset(ctx)
    direct_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=False)
    direct_multiscale_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=True)
    hidden_fit = _fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments={},
    )
    hidden_after_direct_multiscale = _fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments=direct_multiscale_fit["quarter_adjustments"],
    )
    direct_features = _aggregate_quarter_adjustments_to_year_features(dict(direct_fit.get("quarter_adjustments") or {}))
    direct_multiscale_features = _aggregate_quarter_adjustments_to_year_features(dict(direct_multiscale_fit.get("quarter_adjustments") or {}))
    hidden_features = _aggregate_quarter_adjustments_to_year_features(dict(hidden_fit.get("quarter_adjustments") or {}))
    combined_features = _aggregate_quarter_adjustments_to_year_features(
        {
            transition: {
                quarter: float((direct_multiscale_fit.get("quarter_adjustments") or {}).get(transition, {}).get(quarter) or 0.0)
                + float((hidden_after_direct_multiscale.get("quarter_adjustments") or {}).get(transition, {}).get(quarter) or 0.0)
                for quarter in structural_inputs.quarter_axis
            }
            for transition in TRANSITION_NAMES
        }
    )
    return {
        "direct_signal": direct_features,
        "direct_plus_multiscale_signal": direct_multiscale_features,
        "hidden_signal": hidden_features,
        "direct_plus_hidden_plus_multiscale_signal": combined_features,
    }


def _partial_observation_splits(
    target_rows: list[dict[str, Any]],
    *,
    start_year: int = 2010,
    end_year: int = 2016,
    min_train_years: int = 3,
    horizon_years: int = 1,
) -> list[dict[str, int]]:
    years = sorted({int(row["year"]) for row in target_rows if int(start_year) <= int(row["year"]) <= int(end_year)})
    splits: list[dict[str, int]] = []
    for train_end_year in years:
        train_years = [year for year in years if year <= train_end_year]
        holdout_years = [year for year in years if train_end_year < year <= train_end_year + int(horizon_years)]
        if len(train_years) < int(min_train_years) or not holdout_years:
            continue
        splits.append({"train_end_year": int(train_end_year), "holdout_year": int(holdout_years[0])})
    return splits


def _plot_partial_observation_metric_bars(
    *,
    output_path: Path,
    summary_rows: list[dict[str, Any]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = sorted({str(row["metric_name"]) for row in summary_rows})
    variant_ids = ["no_signal", "direct_signal", "direct_plus_multiscale_signal", "hidden_signal"]
    x = np.arange(len(metric_names), dtype=np.float32)
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, variant_id in enumerate(variant_ids):
        values = []
        for metric_name in metric_names:
            match = next(
                (row for row in summary_rows if str(row["metric_name"]) == metric_name and str(row["variant_id"]) == variant_id),
                None,
            )
            values.append(float(match["mean_normalized_error"]) if match is not None else np.nan)
        ax.bar(x + (idx - (len(variant_ids) - 1) / 2.0) * width, values, width=width, label=variant_id)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel("Mean Normalized Error")
    ax.set_title("Early-History Partial Observation Benchmark by Metric")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_partial_observation_split_lines(
    *,
    output_path: Path,
    split_rows: list[dict[str, Any]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = sorted({int(row["train_end_year"]) for row in split_rows})
    variant_ids = ["no_signal", "direct_signal", "direct_plus_multiscale_signal", "hidden_signal"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        values = []
        for year in years:
            year_rows = [row for row in split_rows if int(row["train_end_year"]) == year and str(row["variant_id"]) == variant_id]
            values.append(float(np.mean([float(row["normalized_error"]) for row in year_rows])) if year_rows else np.nan)
        ax.plot(years, values, marker="o", linewidth=2, label=variant_id)
    ax.set_xlabel("Train End Year")
    ax.set_ylabel("Mean Normalized Error")
    ax.set_title("Early-History Partial Observation Benchmark by Split")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_tr_v2_rolling_origin_report(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, str]:
    run_context = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    analysis_dir = ensure_dir(run_context.run_dir / "analysis")
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id=plugin_id,
        experiment_id="TR-V2-03-phase2-ablation-suite",
        source_run_id=source_run_id,
    )
    structural_inputs = load_phase2_structural_inputs(ctx)
    splits = _rolling_origin_holdout_splits(
        ctx,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        dataset = _empirical_transition_dataset_for_holdout_years(ctx, list(split["holdout_years"]))
        if not list(dataset.get("train_transition_rows") or []) or not list(dataset.get("holdout_rows") or []):
            continue
        baseline_hazards = _train_based_baseline_hazard_map(dataset, mode="last_train")
        variants = _rolling_origin_variant_metrics(
            structural_inputs=structural_inputs,
            dataset=dataset,
            baseline_hazards=baseline_hazards,
        )
        rows.append(
            {
                "train_end_year": int(split["train_end_year"]),
                "train_years": list(split["train_years"]),
                "holdout_years": list(split["holdout_years"]),
                "variants": variants,
            }
        )
    if not rows:
        raise ValueError("No rolling-origin rows were produced for the requested range.")
    variant_ids = list(rows[0]["variants"].keys())
    summary_rows = []
    for variant_id in variant_ids:
        maes = [float(row["variants"][variant_id]["mae"]) for row in rows]
        smapes = [float(row["variants"][variant_id]["smape"]) for row in rows]
        summary_rows.append(
            {
                "variant_id": variant_id,
                "mean_mae": round(float(np.mean(maes)), 6),
                "median_mae": round(float(np.median(maes)), 6),
                "mean_smape": round(float(np.mean(smapes)), 6),
                "win_count_vs_no_priors": int(
                    sum(
                        float(row["variants"][variant_id]["mae"]) < float(row["variants"]["no_priors"]["mae"])
                        for row in rows
                    )
                )
                if variant_id != "no_priors"
                else 0,
            }
        )
    latest_row = max(rows, key=lambda row: int(row["train_end_year"]))
    payload = {
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "split_config": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": int(horizon_years),
            "baseline_mode": "last_train",
        },
        "rows": rows,
        "summary_rows": summary_rows,
        "latest_split": latest_row,
    }
    json_path = analysis_dir / "tr_v2_rolling_origin_report.json"
    write_json(json_path, payload)
    mae_plot = _plot_rolling_origin_lines(
        output_path=analysis_dir / "tr_v2_rolling_origin_mae.png",
        rows=rows,
        metric_key="mae",
        variants=[
            ("no_priors", "No Priors"),
            ("direct_only", "Direct"),
            ("direct_plus_multiscale", "Direct + Multiscale"),
            ("direct_plus_hidden", "Direct + Hidden"),
        ],
        title="TR-V2 Rolling-Origin MAE",
        ylabel="Normalized MAE",
    )
    smape_plot = _plot_rolling_origin_lines(
        output_path=analysis_dir / "tr_v2_rolling_origin_smape.png",
        rows=rows,
        metric_key="smape",
        variants=[
            ("no_priors", "No Priors"),
            ("direct_only", "Direct"),
            ("direct_plus_multiscale", "Direct + Multiscale"),
            ("direct_plus_hidden", "Direct + Hidden"),
        ],
        title="TR-V2 Rolling-Origin SMAPE",
        ylabel="SMAPE",
    )
    delta_plot = _plot_rolling_origin_deltas(
        output_path=analysis_dir / "tr_v2_rolling_origin_mae_delta.png",
        rows=rows,
        baseline_variant="no_priors",
        compare_variants=[
            ("direct_only", "Direct"),
            ("direct_plus_multiscale", "Direct + Multiscale"),
            ("hidden_only", "Hidden Only"),
        ],
        title="TR-V2 Rolling-Origin MAE Delta vs No Priors",
    )
    markdown_lines = [
        "# TR-V2 Rolling-Origin Report",
        "",
        f"Source run: `{source_run_id}`",
        "",
        "## Summary",
        "",
        "| Variant | Mean MAE | Median MAE | Mean SMAPE | Wins vs No Priors |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: float(item["mean_mae"])):
        markdown_lines.append(
            f"| `{row['variant_id']}` | {float(row['mean_mae']):.6f} | {float(row['median_mae']):.6f} | {float(row['mean_smape']):.6f} | {int(row['win_count_vs_no_priors'])} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Split Table",
            "",
            "| Train End | Holdout | No Priors MAE | Direct MAE | Direct+Multiscale MAE | Hidden Only MAE | Direct+Hidden MAE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        markdown_lines.append(
            f"| {int(row['train_end_year'])} | {', '.join(str(value) for value in row['holdout_years'])} | "
            f"{float(row['variants']['no_priors']['mae']):.6f} | "
            f"{float(row['variants']['direct_only']['mae']):.6f} | "
            f"{float(row['variants']['direct_plus_multiscale']['mae']):.6f} | "
            f"{float(row['variants']['hidden_only']['mae']):.6f} | "
            f"{float(row['variants']['direct_plus_hidden']['mae']):.6f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![Rolling-Origin MAE]({str(Path(mae_plot))})",
            "",
            f"![Rolling-Origin SMAPE]({str(Path(smape_plot))})",
            "",
            f"![Rolling-Origin Delta MAE]({str(Path(delta_plot))})",
            "",
        ]
    )
    md_path = analysis_dir / "tr_v2_rolling_origin_report.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    run_context.update_manifest(
        analysis={
            "tr_v2_rolling_origin_report": {
                "source_run_id": source_run_id,
                "json": str(json_path),
                "markdown": str(md_path),
                "plots": [str(mae_plot), str(smape_plot), str(delta_plot)],
            }
        }
    )
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "mae_plot": str(mae_plot),
        "smape_plot": str(smape_plot),
        "delta_plot": str(delta_plot),
    }


def write_tr_v2_early_history_partial_report(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    start_year: int = 2010,
    end_year: int = 2016,
    min_train_years: int = 3,
    horizon_years: int = 1,
) -> dict[str, str]:
    run_context = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    analysis_dir = ensure_dir(run_context.run_dir / "analysis")
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id=plugin_id,
        experiment_id="TR-V2-03-phase2-ablation-suite",
        source_run_id=source_run_id,
    )
    target_rows = _load_early_partial_target_rows(ctx, start_year=start_year, end_year=end_year)
    feature_surfaces = _build_partial_observation_features(ctx)
    feature_names = [f"{transition}_mean" for transition in TRANSITION_NAMES] + [f"{transition}_abs_mean" for transition in TRANSITION_NAMES]
    target_scales = {
        str(metric_name): max(
            [abs(float(row["value"])) for row in target_rows if str(row["metric_name"]) == str(metric_name)],
            default=1.0,
        )
        for metric_name in sorted({str(row["metric_name"]) for row in target_rows})
    }
    split_defs = _partial_observation_splits(
        target_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    split_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(row["metric_name"]) for row in target_rows}):
        metric_target_rows = [dict(row) for row in target_rows if str(row["metric_name"]) == metric_name]
        for split in split_defs:
            train_targets = [row for row in metric_target_rows if int(row["year"]) <= int(split["train_end_year"])]
            holdout_target = next((row for row in metric_target_rows if int(row["year"]) == int(split["holdout_year"])), None)
            if holdout_target is None or len(train_targets) < int(min_train_years):
                continue
            normalized_scale = max(float(target_scales.get(metric_name) or 1.0), 1e-6)
            no_signal_prediction = float(train_targets[-1]["value"])
            split_rows.append(
                {
                    "metric_name": metric_name,
                    "train_end_year": int(split["train_end_year"]),
                    "holdout_year": int(split["holdout_year"]),
                    "variant_id": "no_signal",
                    "prediction": round(no_signal_prediction, 6),
                    "target": round(float(holdout_target["value"]), 6),
                    "normalized_error": round(abs(no_signal_prediction - float(holdout_target["value"])) / normalized_scale, 6),
                }
            )
            for variant_id, features_by_year in feature_surfaces.items():
                train_years = [int(row["year"]) for row in train_targets if int(row["year"]) in features_by_year]
                if len(train_years) < int(min_train_years) or int(holdout_target["year"]) not in features_by_year:
                    continue
                train_x = np.asarray([[float(features_by_year[year].get(name) or 0.0) for name in feature_names] for year in train_years], dtype=np.float32)
                train_y = np.asarray([float(next(row["value"] for row in train_targets if int(row["year"]) == year)) for year in train_years], dtype=np.float32)
                beta = _fit_partial_readout(train_x=train_x, train_y=train_y, ridge_penalty=1.0)
                prediction = _partial_readout_predict(beta, features_by_year[int(holdout_target["year"])], feature_names)
                split_rows.append(
                    {
                        "metric_name": metric_name,
                        "train_end_year": int(split["train_end_year"]),
                        "holdout_year": int(split["holdout_year"]),
                        "variant_id": variant_id,
                        "prediction": round(float(prediction), 6),
                        "target": round(float(holdout_target["value"]), 6),
                        "normalized_error": round(abs(float(prediction) - float(holdout_target["value"])) / normalized_scale, 6),
                    }
                )
    if not split_rows:
        raise ValueError("No early-history partial-observation benchmark rows were produced.")
    summary_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(row["metric_name"]) for row in split_rows}):
        metric_rows = [row for row in split_rows if str(row["metric_name"]) == metric_name]
        baseline_rows = [row for row in metric_rows if str(row["variant_id"]) == "no_signal"]
        for variant_id in sorted({str(row["variant_id"]) for row in metric_rows}):
            variant_rows = [row for row in metric_rows if str(row["variant_id"]) == variant_id]
            summary_rows.append(
                {
                    "metric_name": metric_name,
                    "variant_id": variant_id,
                    "mean_normalized_error": round(float(np.mean([float(row["normalized_error"]) for row in variant_rows])), 6),
                    "median_normalized_error": round(float(np.median([float(row["normalized_error"]) for row in variant_rows])), 6),
                    "win_count_vs_no_signal": int(
                        sum(
                            float(candidate["normalized_error"]) < float(baseline["normalized_error"])
                            for candidate, baseline in zip(
                                sorted(variant_rows, key=lambda row: (int(row["train_end_year"]), int(row["holdout_year"]))),
                                sorted(baseline_rows, key=lambda row: (int(row["train_end_year"]), int(row["holdout_year"]))),
                            )
                        )
                    )
                    if variant_id != "no_signal"
                    else 0,
                }
            )
    payload = {
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "benchmark_kind": "early_history_partial_observation",
        "quality_regime": "partial_observation_not_comparable_to_full_transition_benchmark",
        "split_config": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": int(horizon_years),
        },
        "target_rows": target_rows,
        "split_rows": split_rows,
        "summary_rows": summary_rows,
    }
    json_path = analysis_dir / "tr_v2_early_history_partial_report.json"
    write_json(json_path, payload)
    metric_plot = _plot_partial_observation_metric_bars(
        output_path=analysis_dir / "tr_v2_early_history_partial_metric_errors.png",
        summary_rows=summary_rows,
    )
    split_plot = _plot_partial_observation_split_lines(
        output_path=analysis_dir / "tr_v2_early_history_partial_split_errors.png",
        split_rows=split_rows,
    )
    markdown_lines = [
        "# TR-V2 Early-History Partial Observation Report",
        "",
        f"Source run: `{source_run_id}`",
        "",
        "This report is a partial-observation benchmark for 2010-2016. It is not directly comparable to the full TR-V2 rolling-origin benchmark because the early years do not have the same national transition targets.",
        "",
        "## Summary",
        "",
        "| Metric | Variant | Mean Normalized Error | Median Normalized Error | Wins vs No Signal |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: (str(item["metric_name"]), float(item["mean_normalized_error"]))):
        markdown_lines.append(
            f"| `{row['metric_name']}` | `{row['variant_id']}` | {float(row['mean_normalized_error']):.6f} | {float(row['median_normalized_error']):.6f} | {int(row['win_count_vs_no_signal'])} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Split Table",
            "",
            "| Metric | Train End | Holdout | Variant | Prediction | Target | Normalized Error |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(split_rows, key=lambda item: (str(item["metric_name"]), int(item["train_end_year"]), str(item["variant_id"]))):
        markdown_lines.append(
            f"| `{row['metric_name']}` | {int(row['train_end_year'])} | {int(row['holdout_year'])} | `{row['variant_id']}` | "
            f"{float(row['prediction']):.6f} | {float(row['target']):.6f} | {float(row['normalized_error']):.6f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![Partial Observation Metric Errors]({str(Path(metric_plot))})",
            "",
            f"![Partial Observation Split Errors]({str(Path(split_plot))})",
            "",
        ]
    )
    md_path = analysis_dir / "tr_v2_early_history_partial_report.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    run_context.update_manifest(
        analysis={
            "tr_v2_early_history_partial_report": {
                "source_run_id": source_run_id,
                "json": str(json_path),
                "markdown": str(md_path),
                "plots": [str(metric_plot), str(split_plot)],
            }
        }
    )
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "metric_plot": str(metric_plot),
        "split_plot": str(split_plot),
    }


def _discover_latest_analysis_report_path(report_filename: str, *, source_run_id: str | None = None) -> Path:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        candidate = run_dir / "analysis" / report_filename
        if not candidate.exists():
            continue
        if source_run_id:
            payload = read_json(candidate, default={})
            if str((payload or {}).get("source_run_id") or "") != str(source_run_id):
                continue
        candidates.append((candidate.stat().st_mtime, candidate))
    if not candidates:
        raise FileNotFoundError(f"No analysis report found for {report_filename} (source_run_id={source_run_id or 'any'})")
    return max(candidates, key=lambda row: row[0])[1]


def _plot_tr_v2_benchmark_dashboard(
    *,
    output_path: Path,
    rolling_payload: dict[str, Any],
    early_payload: dict[str, Any],
    interpretation_lines: list[str],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    rolling_summary = list(rolling_payload.get("summary_rows") or [])
    rolling_rows = list(rolling_payload.get("rows") or [])
    variants = [row for row in rolling_summary if str(row.get("variant_id") or "") in {"no_priors", "direct_only", "direct_plus_multiscale", "direct_plus_hidden"}]
    axes[0, 0].bar(
        [str(row["variant_id"]) for row in variants],
        [float(row["mean_mae"]) for row in variants],
        color=["#5B6C8D", "#2E8B57", "#5AAE61", "#B8860B"],
    )
    axes[0, 0].set_title("2017-2025 Rolling-Origin Mean MAE")
    axes[0, 0].set_ylabel("Mean MAE")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", alpha=0.3)

    train_end_years = [int(row["train_end_year"]) for row in rolling_rows]
    for variant_id, label in [("no_priors", "No Priors"), ("direct_only", "Direct"), ("direct_plus_multiscale", "Direct + Multiscale")]:
        axes[0, 1].plot(
            train_end_years,
            [float((row.get("variants") or {}).get(variant_id, {}).get("mae") or np.nan) for row in rolling_rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[0, 1].set_title("2017-2025 Split-by-Split MAE")
    axes[0, 1].set_xlabel("Train End Year")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    early_summary = list(early_payload.get("summary_rows") or [])
    metric_names = sorted({str(row["metric_name"]) for row in early_summary})
    best_deltas = []
    for metric_name in metric_names:
        metric_rows = [row for row in early_summary if str(row["metric_name"]) == metric_name]
        no_signal = next((row for row in metric_rows if str(row["variant_id"]) == "no_signal"), None)
        structural = [row for row in metric_rows if str(row["variant_id"]) != "no_signal"]
        best_structural = min(structural, key=lambda row: float(row["mean_normalized_error"])) if structural else None
        if no_signal is None or best_structural is None:
            best_deltas.append(np.nan)
            continue
        best_deltas.append(float(best_structural["mean_normalized_error"]) - float(no_signal["mean_normalized_error"]))
    axes[1, 0].bar(metric_names, best_deltas, color=["#B22222" if value > 0 else "#2E8B57" for value in best_deltas])
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].set_title("2010-2016 Best Structural Delta vs No Signal")
    axes[1, 0].set_ylabel("Delta Normalized Error")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].set_title("Interpretation", loc="left", pad=10)
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(f"- {line}" for line in interpretation_lines),
        va="top",
        ha="left",
        fontsize=10,
        wrap=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_tr_v2_benchmark_dashboard_report(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    rolling_origin_run_id: str | None = None,
    early_history_run_id: str | None = None,
) -> dict[str, str]:
    run_context = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    analysis_dir = ensure_dir(run_context.run_dir / "analysis")
    rolling_json_path = (
        ROOT_DIR / "artifacts" / "runs" / rolling_origin_run_id / "analysis" / "tr_v2_rolling_origin_report.json"
        if rolling_origin_run_id
        else _discover_latest_analysis_report_path("tr_v2_rolling_origin_report.json", source_run_id=source_run_id)
    )
    early_json_path = (
        ROOT_DIR / "artifacts" / "runs" / early_history_run_id / "analysis" / "tr_v2_early_history_partial_report.json"
        if early_history_run_id
        else _discover_latest_analysis_report_path("tr_v2_early_history_partial_report.json", source_run_id=source_run_id)
    )
    rolling_payload = dict(read_json(rolling_json_path, default={}) or {})
    early_payload = dict(read_json(early_json_path, default={}) or {})
    if not rolling_payload or not early_payload:
        raise FileNotFoundError("Combined benchmark dashboard requires both rolling-origin and early-history report payloads.")
    rolling_summary = list(rolling_payload.get("summary_rows") or [])
    latest_split = dict(rolling_payload.get("latest_split") or {})
    latest_variants = dict(latest_split.get("variants") or {})
    latest_best = min(latest_variants.items(), key=lambda item: float((item[1] or {}).get("mae") or np.inf)) if latest_variants else ("none", {})
    rolling_winners = [
        str(row["variant_id"])
        for row in rolling_summary
        if str(row.get("variant_id") or "") != "no_priors" and int(row.get("win_count_vs_no_priors") or 0) > 0
    ]
    early_summary = list(early_payload.get("summary_rows") or [])
    metrics = sorted({str(row["metric_name"]) for row in early_summary})
    metrics_where_signal_helps: list[str] = []
    metrics_where_no_signal_wins: list[str] = []
    for metric_name in metrics:
        metric_rows = [row for row in early_summary if str(row["metric_name"]) == metric_name]
        best_row = min(metric_rows, key=lambda row: float(row.get("mean_normalized_error") or np.inf))
        if str(best_row.get("variant_id") or "") == "no_signal":
            metrics_where_no_signal_wins.append(metric_name)
        else:
            metrics_where_signal_helps.append(metric_name)
    interpretation_lines = [
        f"In the fuller 2017-2025 rolling-origin benchmark, structural Phase 2 priors help on later splits but not uniformly; winners vs no-prior include {', '.join(rolling_winners) if rolling_winners else 'none'}.",
        f"On the latest split ending {int(latest_split.get('train_end_year') or 0)} with holdout {', '.join(str(v) for v in latest_split.get('holdout_years') or [])}, the best variant is {latest_best[0]} with MAE {float((latest_best[1] or {}).get('mae') or 0.0):.6f}.",
        f"In the 2010-2016 partial-observation regime, Phase 2-derived signals help most on {', '.join(metrics_where_signal_helps) if metrics_where_signal_helps else 'no early partial metrics'}, while no-signal carry-forward remains strongest on {', '.join(metrics_where_no_signal_wins) if metrics_where_no_signal_wins else 'none'}.",
        "The regimes differ because 2017-2025 has transition-adjacent quarterly national targets, while 2010-2016 only has partial annual national signals, so the mapping from quarterly hazard structure to observed targets is much weaker and more indirect.",
    ]
    dashboard_plot = _plot_tr_v2_benchmark_dashboard(
        output_path=analysis_dir / "tr_v2_benchmark_dashboard.png",
        rolling_payload=rolling_payload,
        early_payload=early_payload,
        interpretation_lines=interpretation_lines,
    )
    payload = {
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "linked_reports": {
            "rolling_origin_json": str(rolling_json_path),
            "rolling_origin_markdown": str(rolling_json_path.with_suffix(".md")),
            "early_history_json": str(early_json_path),
            "early_history_markdown": str(early_json_path.with_suffix(".md")),
        },
        "interpretation_lines": interpretation_lines,
        "rolling_summary_rows": rolling_summary,
        "latest_split": latest_split,
        "early_summary_rows": early_summary,
    }
    json_path = analysis_dir / "tr_v2_benchmark_dashboard_report.json"
    write_json(json_path, payload)
    markdown_lines = [
        "# TR-V2 Benchmark Dashboard",
        "",
        f"Source run: `{source_run_id}`",
        "",
        "## Linked Reports",
        "",
        f"- [2017-2025 Rolling-Origin Report]({str(rolling_json_path.with_suffix('.md'))})",
        f"- [2010-2016 Early-History Partial Report]({str(early_json_path.with_suffix('.md'))})",
        "",
        "## Interpretation",
        "",
    ]
    markdown_lines.extend([f"- {line}" for line in interpretation_lines])
    markdown_lines.extend(
        [
            "",
            "## Dashboard",
            "",
            f"![TR-V2 Benchmark Dashboard]({str(Path(dashboard_plot))})",
            "",
            "## Embedded Benchmark Plots",
            "",
            f"![Rolling-Origin MAE]({str(Path(rolling_json_path.parent / 'tr_v2_rolling_origin_mae.png'))})",
            "",
            f"![Rolling-Origin Delta MAE]({str(Path(rolling_json_path.parent / 'tr_v2_rolling_origin_mae_delta.png'))})",
            "",
            f"![Early Partial Metric Errors]({str(Path(early_json_path.parent / 'tr_v2_early_history_partial_metric_errors.png'))})",
            "",
            f"![Early Partial Split Errors]({str(Path(early_json_path.parent / 'tr_v2_early_history_partial_split_errors.png'))})",
            "",
        ]
    )
    md_path = analysis_dir / "tr_v2_benchmark_dashboard_report.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    run_context.update_manifest(
        analysis={
            "tr_v2_benchmark_dashboard_report": {
                "source_run_id": source_run_id,
                "json": str(json_path),
                "markdown": str(md_path),
                "dashboard_plot": str(dashboard_plot),
            }
        }
    )
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "dashboard_plot": str(dashboard_plot),
    }


__all__ = [
    "run_tr_v2_00",
    "run_tr_v2_01",
    "run_tr_v2_02",
    "run_tr_v2_03",
    "write_tr_v2_benchmark_dashboard_report",
    "write_tr_v2_early_history_partial_report",
    "write_tr_v2_rolling_origin_report",
]
