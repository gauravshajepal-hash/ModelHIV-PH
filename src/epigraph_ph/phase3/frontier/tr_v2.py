from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3._lineage.national_reset_core import quarter_sort_key
from epigraph_ph.runtime import ROOT_DIR, read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .phase2_structural_inputs import load_phase2_structural_inputs
from .registry import TRANSITION_NAMES
from .transition_engine import (
    STATE_NAMES,
    _build_observation_payload,
    _coverage_mean,
    _holdout_rows,
    _holdout_years_from_reference,
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


def _empirical_transition_dataset(ctx: TransitionResearchContext) -> dict[str, Any]:
    reference = _load_front_half_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(reference)
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
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in structural_inputs.direct_edge_rows:
        key = (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0))
        candidate = {
            "weight": float(row.get("weight") or 0.0),
            "stability": float(row.get("stability") or 0.0),
            "support_count": int(row.get("support_count") or 0),
        }
        existing = lookup.get(key)
        if existing is None or (
            candidate["support_count"],
            abs(candidate["weight"]),
            candidate["stability"],
        ) > (
            existing["support_count"],
            abs(existing["weight"]),
            existing["stability"],
        ):
            lookup[key] = candidate
    return lookup


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
) -> np.ndarray:
    if design_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    gram = design_matrix.T @ design_matrix
    gram += np.diag(np.asarray(prior_precisions, dtype=np.float32))
    rhs = design_matrix.T @ target_values
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
    return np.asarray(beta, dtype=np.float32)


def _fit_direct_transition_effects(
    *,
    structural_inputs: Any,
    dataset: dict[str, Any],
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
        feature_specs: list[tuple[str, str, int, float, dict[str, float]]] = []
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
            feature_specs.append((source_block, target_block, lag, prior_scale, support))
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
            for block_id, _target_block, lag, _prior_scale, _support in feature_specs:
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
                1.0
                / max(
                    (float(prior_scale) * (1.0 + float(support["support_count"])) * max(float(support["stability"]), 0.05))
                    ** 2,
                    1e-6,
                )
                for _block_id, _target_block, _lag, prior_scale, support in feature_specs
            ],
            dtype=np.float32,
        )
        beta = _fit_prior_regression(target_values=y - intercept, design_matrix=X_matrix, prior_precisions=prior_precisions)
        summaries[transition] = {
            "feature_count": int(len(feature_specs)),
            "intercept_logit": round(intercept, 6),
            "coefficients": [
                {
                    "block_id": block_id,
                    "target_block_id": target_block,
                    "lag": lag,
                    "coefficient": round(float(beta[idx]), 6),
                    "prior_scale": round(float(prior_scale), 6),
                    "phase2_weight": round(float(support["weight"]), 6),
                    "support_count": int(support["support_count"]),
                    "stability": round(float(support["stability"]), 6),
                }
                for idx, (block_id, target_block, lag, prior_scale, support) in enumerate(feature_specs)
            ],
        }
        for quarter in structural_inputs.quarter_axis:
            quarter_position = structural_inputs.quarter_axis.index(quarter)
            delta = 0.0
            for idx, (block_id, _target_block, lag, _prior_scale, _support) in enumerate(feature_specs):
                source_idx = quarter_position - lag
                if source_idx < 0:
                    continue
                source_quarter = structural_inputs.quarter_axis[source_idx]
                delta += float(beta[idx]) * float(quarter_features.get(source_quarter, {}).get(block_id) or 0.0)
            predictions[transition][quarter] = float(delta)
    return {"summary": summaries, "quarter_adjustments": {key: dict(value) for key, value in predictions.items()}}


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
        prior_precisions = np.ones((hidden_dim,), dtype=np.float32)
        beta = _fit_prior_regression(target_values=y_vector - intercept, design_matrix=X_matrix, prior_precisions=prior_precisions)
        summaries[transition] = {
            "rank_used": int(hidden_dim),
            "intercept_logit": round(intercept, 6),
            "coefficients": [{"hidden_mode": int(idx), "coefficient": round(float(beta[idx]), 6)} for idx in range(hidden_dim)],
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
    include_peak_gating: bool,
) -> dict[str, Any]:
    age_lock = _load_age01b_lock()
    structural_inputs = load_phase2_structural_inputs(ctx)
    dataset = _empirical_transition_dataset(ctx)
    peak_gates = _load_peak_gates() if include_peak_gating else {}
    baseline_hazards = _baseline_hazard_map(age_lock)
    direct_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset) if include_direct else {"summary": {}, "quarter_adjustments": {}}
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
    }


def run_tr_v2_01(ctx: TransitionResearchContext) -> dict[str, Any]:
    _write_standard_experiment_metadata(ctx, description="Direct Phase 2 temporal edges become structured hazard priors on the locked AGE-01B baseline.")
    result = _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=False, include_peak_gating=False)
    write_json(ctx.experiment_dir / "baseline_lock.json", {"reference_experiment_id": AGE01B_EXPERIMENT_ID, "reference_run_id": result["age_lock"]["reference_run_id"]})
    write_json(ctx.experiment_dir / "phase2_direct_hazard_prior_summary.json", result["direct_fit"]["summary"])
    write_json(ctx.experiment_dir / "baseline_comparison.json", result["baseline_comparison"])
    write_json(ctx.experiment_dir / "evaluation.json", result["evaluation"])
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", {"forecast_rows": result["forecast_rows"], "transition_hazards": result["hazard_rows"]})
    return {"experiment_dir": str(ctx.experiment_dir)}


def run_tr_v2_02(ctx: TransitionResearchContext) -> dict[str, Any]:
    _write_standard_experiment_metadata(ctx, description="Direct Phase 2 temporal edges plus hidden low-rank mode shocks modulate the locked AGE-01B hazards.")
    result = _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_peak_gating=False)
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
        "direct_only": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=False, include_peak_gating=False),
        "hidden_only": _run_structural_variant(ctx=ctx, include_direct=False, include_hidden=True, include_peak_gating=False),
        "direct_plus_hidden": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_peak_gating=False),
        "direct_plus_hidden_plus_peak_gating": _run_structural_variant(ctx=ctx, include_direct=True, include_hidden=True, include_peak_gating=True),
    }
    summary = {
        "rows": [
            {
                "variant_id": variant_id,
                "model_mean_absolute_error": float(result["evaluation"]["model_mean_absolute_error"]),
                "model_smape": float(result["evaluation"]["model_smape"]),
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


__all__ = ["run_tr_v2_00", "run_tr_v2_01", "run_tr_v2_02", "run_tr_v2_03"]
