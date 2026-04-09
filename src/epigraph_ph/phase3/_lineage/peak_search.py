from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase3.shared.broad_backtest_support import rolling_origin_splits, run_broad_backtest_trial
from epigraph_ph.phase3.shared.numerics import safe_floor
from epigraph_ph.phase3._lineage.pipeline import RESCUE_INFERENCE_FAMILY, RESCUE_V2_PROFILE_ID, _prepare_frozen_backtest_inputs
from epigraph_ph.phase3._lineage.rescue_core import (
    STATE_NAMES,
    TRANSITION_NAMES,
    _phase3_death_ceiling,
    _phase3_death_scale,
    _phase3_prior,
    _forecast_month_axis,
    _national_reference_mask,
    run_phase3_rescue_core,
)
from epigraph_ph.phase3.shared.temporal_scaffold import build_temporal_basis, logit_clip, sigmoid
from epigraph_ph.runtime import RunContext, ensure_dir, load_tensor_artifact, read_json, write_json


PEAK_TARGETS = {
    "U",
    "D",
    "A",
    "L",
    "diagnosed_stock",
    "art_stock",
    "documented_suppression_gap",
    "deaths",
}


def _default_representation(profile: str) -> str:
    return "clumped" if profile == RESCUE_V2_PROFILE_ID else "unclumped"


def _late_cascade_error(row: dict[str, Any]) -> float:
    bias = dict(row.get("bias_metrics") or {})
    return float(bias.get("documented_suppression_abs_error", 0.0)) + float(bias.get("viral_load_tested_among_art_abs_error", 0.0))


def _promotion_gate(
    *,
    run_id: str,
    plugin_id: str,
    profile: str,
    inference_family: str,
    representation: str,
    tolerance: float,
) -> dict[str, Any]:
    try:
        prepared = _prepare_frozen_backtest_inputs(run_id=run_id, plugin_id=plugin_id, train_years=None, holdout_years=None)
        all_years = sorted(
            {
                *[int(year) for year in list(prepared["backtest_config"].get("train_years") or [])],
                *[int(year) for year in list(prepared["backtest_config"].get("holdout_years") or [])],
            }
        )
        splits = [dict(split) for split in rolling_origin_splits(all_years, min_train_years=3) if int(split["holdout_year"]) >= 2020]
        baseline_representation = _default_representation(profile)
        candidate_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []
        for split in splits:
            prepared_split = _prepare_frozen_backtest_inputs(
                run_id=run_id,
                plugin_id=plugin_id,
                train_years=split["train_years"],
                holdout_years=split["holdout_years"],
            )
            candidate_rows.append(
                run_broad_backtest_trial(
                    prepared=prepared_split,
                    run_id=run_id,
                    plugin_id=plugin_id,
                    profile=profile,
                    inference_family=inference_family,
                    representation=representation,
                    phase_dir_name=f"phase3_peak_search_gate/holdout_{split['holdout_year']}/{representation}",
                )
            )
            baseline_rows.append(
                run_broad_backtest_trial(
                    prepared=prepared_split,
                    run_id=run_id,
                    plugin_id=plugin_id,
                    profile=profile,
                    inference_family=inference_family,
                    representation=baseline_representation,
                    phase_dir_name=f"phase3_peak_search_gate/holdout_{split['holdout_year']}/{baseline_representation}",
                )
            )
    except RuntimeError as exc:
        return {
            "passed": False,
            "representation": representation,
            "baseline_representation": _default_representation(profile),
            "rolling_origin_split_count": 0,
            "mean_model_mae": None,
            "mean_simple_compartmental_mae": None,
            "mean_late_cascade_error": None,
            "baseline_late_cascade_error": None,
            "tolerance": round(float(tolerance), 6),
            "reason": "backend_unavailable_for_requested_inference_family",
            "details": str(exc),
        }
    mean_model_mae = float(np.mean([float(row.get("model_mean_absolute_error", 0.0)) for row in candidate_rows])) if candidate_rows else float("inf")
    mean_simple_mae = float(np.mean([float(row.get("simple_compartmental_mean_absolute_error", 0.0)) for row in candidate_rows])) if candidate_rows else float("inf")
    mean_late_error = float(np.mean([_late_cascade_error(row) for row in candidate_rows])) if candidate_rows else float("inf")
    baseline_late_error = float(np.mean([_late_cascade_error(row) for row in baseline_rows])) if baseline_rows else float("inf")
    passed = bool(candidate_rows) and mean_model_mae <= (mean_simple_mae + tolerance) and mean_late_error <= baseline_late_error
    return {
        "passed": passed,
        "representation": representation,
        "baseline_representation": baseline_representation,
        "rolling_origin_split_count": len(candidate_rows),
        "mean_model_mae": round(mean_model_mae, 6) if np.isfinite(mean_model_mae) else None,
        "mean_simple_compartmental_mae": round(mean_simple_mae, 6) if np.isfinite(mean_simple_mae) else None,
        "mean_late_cascade_error": round(mean_late_error, 6) if np.isfinite(mean_late_error) else None,
        "baseline_late_cascade_error": round(baseline_late_error, 6) if np.isfinite(baseline_late_error) else None,
        "tolerance": round(float(tolerance), 6),
    }


def _national_series(states: np.ndarray, province_axis: list[str]) -> np.ndarray:
    weights = _national_reference_mask(province_axis).astype(np.float32)
    if not np.any(weights > 0):
        weights = np.ones((len(province_axis),), dtype=np.float32) / float(max(len(province_axis), 1))
    return np.sum(states * weights.reshape(len(province_axis), 1, 1), axis=0).astype(np.float32)


def _target_series(states: np.ndarray, province_axis: list[str], target: str) -> np.ndarray:
    national = _national_series(states, province_axis)
    if target == "U":
        return national[:, 0]
    if target == "D":
        return national[:, 1]
    if target == "A":
        return national[:, 2]
    if target == "L":
        return national[:, 4]
    if target == "diagnosed_stock":
        return np.clip(1.0 - national[:, 0], 0.0, 1.0)
    if target == "art_stock":
        return np.clip(national[:, 2] + national[:, 3], 0.0, 1.0)
    if target == "documented_suppression_gap":
        art = np.clip(national[:, 2] + national[:, 3], 0.0, 1.0)
        suppressed = np.clip(national[:, 3], 0.0, 1.0)
        return np.clip(art - suppressed, 0.0, 1.0)
    if target == "deaths":
        return np.clip(national[:, 4] * _phase3_death_scale(), 0.0, _phase3_death_ceiling())
    raise ValueError(f"unsupported peak-search target: {target}")


def _simulate_future_states(initial_state: np.ndarray, transition_probs: np.ndarray) -> tuple[np.ndarray, float]:
    state = np.asarray(initial_state, dtype=np.float32)
    forecasts: list[np.ndarray] = []
    mass_violation_rows: list[float] = []
    state_mass_eps = safe_floor(float(dict(dict(_phase3_prior("frozen_backtest") or {}).get("evolution", {}) or {})["state_mass_eps"]))
    for probs in np.asarray(transition_probs, dtype=np.float32).transpose(1, 0, 2):
        U = state[:, 0]
        D = state[:, 1]
        A = state[:, 2]
        V = state[:, 3]
        L = state[:, 4]
        p_ud = probs[:, 0]
        p_da = probs[:, 1]
        p_av = probs[:, 2]
        p_al = probs[:, 3]
        p_la = probs[:, 4]
        flow_ud = U * p_ud
        flow_da = D * p_da
        flow_av = A * p_av
        flow_al = A * p_al
        flow_la = L * p_la
        state = np.stack(
            [
                U - flow_ud,
                D + flow_ud - flow_da,
                A + flow_da + flow_la - flow_av - flow_al,
                V + flow_av,
                L + flow_al - flow_la,
            ],
            axis=-1,
        ).astype(np.float32)
        mass_violation_rows.append(float(np.mean(np.square(np.sum(state, axis=-1) - 1.0))))
        state = np.clip(state, 0.0, None)
        state = state / np.clip(state.sum(axis=-1, keepdims=True), state_mass_eps, None)
        forecasts.append(state.astype(np.float32))
    return np.asarray(forecasts, dtype=np.float32).transpose(1, 0, 2), float(np.mean(mass_violation_rows) if mass_violation_rows else 0.0)


def _group_transition_templates(group_rows: list[dict[str, Any]], peak_cfg: dict[str, Any]) -> np.ndarray:
    templates = np.zeros((len(group_rows), len(TRANSITION_NAMES)), dtype=np.float32)
    template_cfg = dict(peak_cfg["transition_group_templates"] or {})
    for idx, row in enumerate(group_rows):
        name = str(row.get("group_name") or "").lower()
        matched = None
        for group_name, payload in template_cfg.items():
            keywords = [str(token).lower() for token in list(dict(payload).get("keywords", []) or [])]
            if keywords and any(token in name for token in keywords):
                matched = dict(payload)
                break
            if not keywords and matched is None:
                matched = dict(payload)
        if matched is None:
            continue
        weights = dict(matched.get("weights", {}) or {})
        for transition_idx, transition_name in enumerate(TRANSITION_NAMES):
            templates[idx, transition_idx] = float(weights.get(transition_name, 0.0) or 0.0)
    return templates.astype(np.float32)


def _search_score(
    *,
    series: np.ndarray,
    base_series: np.ndarray,
    peak_index: int,
    offset_params: np.ndarray,
    slow_params: np.ndarray,
    medium_params: np.ndarray,
    group_params: np.ndarray,
    future_transition_probs: np.ndarray,
    state_mass_violation: float,
    weight_cfg: dict[str, Any],
    evolution_cfg: dict[str, Any],
    candidate_center: float,
) -> tuple[float, dict[str, float]]:
    peak_value = float(series[peak_index])
    baseline_value = float(base_series[0]) if base_series.size else safe_floor(None)
    normalized_peak_height = peak_value / max(abs(baseline_value), safe_floor(None))
    prominence = peak_value - float(np.min(series))
    distance_to_center = abs(float(peak_index) - candidate_center) / max(len(series), 1)
    roughness = float(np.mean(np.diff(series) ** 2)) if len(series) > 1 else 0.0
    coefficient_roughness = float(np.mean(np.diff(slow_params, axis=0) ** 2)) if slow_params.shape[0] > 1 else 0.0
    coefficient_roughness += float(np.mean(np.diff(medium_params, axis=0) ** 2)) if medium_params.shape[0] > 1 else 0.0
    latent_code_norm = float(np.mean(offset_params**2) + np.mean(slow_params**2) + np.mean(medium_params**2) + (np.mean(group_params**2) if group_params.size else 0.0))
    transition_floor = float(evolution_cfg["transition_floor"])
    transition_ceiling = float(evolution_cfg["transition_ceiling"])
    bound_violation = float(
        np.mean(np.square(np.clip(future_transition_probs - transition_ceiling, 0.0, None)))
        + np.mean(np.square(np.clip(transition_floor - future_transition_probs, 0.0, None)))
    )
    anchor_mismatch = float(abs(series[0] - base_series[0])) if base_series.size else 0.0
    score = (
        float(weight_cfg["peak_height_weight"]) * normalized_peak_height
        + float(weight_cfg["peak_prominence_weight"]) * prominence
        - float(weight_cfg["peak_time_distance_weight"]) * distance_to_center
        - float(weight_cfg["anchor_mismatch_weight"]) * anchor_mismatch
        - float(weight_cfg["smoothness_weight"]) * (roughness + coefficient_roughness)
        - float(weight_cfg["prior_penalty_weight"]) * latent_code_norm
        - float(weight_cfg["feasibility_weight"]) * bound_violation
        - float(weight_cfg["mass_penalty_weight"]) * float(state_mass_violation)
    )
    return score, {
        "normalized_peak_height": round(normalized_peak_height, 6),
        "peak_prominence": round(prominence, 6),
        "distance_to_candidate_tau": round(distance_to_center, 6),
        "annual_anchor_error": round(anchor_mismatch, 6),
        "coefficient_roughness": round(coefficient_roughness, 6),
        "latent_code_norm": round(latent_code_norm, 6),
        "transition_bound_violation": round(bound_violation, 6),
        "state_mass_violation": round(float(state_mass_violation), 6),
    }


def run_phase3_peak_search(
    *,
    run_id: str,
    plugin_id: str,
    profile: str = RESCUE_V2_PROFILE_ID,
    inference_family: str = RESCUE_INFERENCE_FAMILY,
    target: str,
    representation: str = "hybrid_temporal_multiscale",
    horizon_months: int = 60,
) -> dict[str, Any]:
    if target not in PEAK_TARGETS:
        raise ValueError(f"unsupported phase3 peak-search target: {target}")
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    peak_cfg = dict(_phase3_prior("peak_search") or {})
    tolerance = float(peak_cfg["promotion_tolerance_mae"])
    gate = _promotion_gate(
        run_id=run_id,
        plugin_id=plugin_id,
        profile=profile,
        inference_family=inference_family,
        representation=representation,
        tolerance=tolerance,
    )
    phase_dir = ensure_dir(ctx.run_dir / "phase3_peak_search")
    if not gate["passed"]:
        summary = {
            "available": False,
            "reason": "forecast_gate_not_passed",
            "target_name": target,
            "representation": representation,
            "horizon_months": int(horizon_months),
            "promotion_gate": gate,
        }
        write_json(phase_dir / "peak_search_summary.json", summary)
        write_json(phase_dir / "peak_envelope.json", {"available": False, "promotion_gate": gate})
        write_json(phase_dir / "peak_search_elite_paths.json", {"elite_paths": [], "promotion_gate": gate})
        return summary

    manifest = run_phase3_rescue_core(
        run_id=run_id,
        plugin_id=plugin_id,
        profile_id=profile,
        requested_inference_family=inference_family,
        phase_dir_name="phase3_peak_search/model_fit",
        modifier_representation=representation,
    )
    fit_artifact = read_json(Path(manifest["artifact_paths"]["fit_artifact"]), default={})
    model_artifact = read_json(Path(manifest["artifact_paths"]["model_artifact"]), default={})
    phase3_dir = Path(manifest["artifact_paths"]["fit_artifact"]).parent
    state_estimates = load_tensor_artifact(Path(manifest["artifact_paths"]["state_estimates"])).astype(np.float32)
    transition_probs = load_tensor_artifact(Path(manifest["artifact_paths"]["transition_probabilities"])).astype(np.float32)
    province_axis = list(model_artifact.get("province_axis") or [])
    month_axis = list(model_artifact.get("month_axis") or [])
    future_month_axis = _forecast_month_axis(month_axis[-1] if month_axis else "2025-12", horizon_months)
    temporal_cfg = dict(_phase3_prior("temporal_decomposition") or {})
    frozen_cfg = dict(_phase3_prior("frozen_backtest") or {})
    evolution_cfg = dict(frozen_cfg["evolution"] or {})
    basis = build_temporal_basis(
        future_month_axis,
        slow_knot_months=int(temporal_cfg["slow_knot_months"]),
        medium_block_months=int(temporal_cfg["medium_block_months"]),
    )
    slow_basis = np.asarray(basis["slow_basis"], dtype=np.float32)
    medium_basis = np.asarray(basis["medium_basis"], dtype=np.float32)
    group_rows = list(dict(fit_artifact.get("determinant_modifier_summary") or {}).get("determinant_group_catalog", {}).get("rows", []) or [])
    group_templates = _group_transition_templates(group_rows, peak_cfg)
    base_state = np.asarray(state_estimates[:, -1, :], dtype=np.float32)
    base_transition = np.asarray(transition_probs[:, -1, :], dtype=np.float32)
    base_logit = logit_clip(base_transition, float(temporal_cfg["probability_eps"]))
    base_future, _ = _simulate_future_states(
        base_state,
        np.broadcast_to(base_transition[:, None, :], (base_transition.shape[0], horizon_months, base_transition.shape[-1])),
    )
    base_series = _target_series(base_future, province_axis, target)
    candidate_window_start = int(peak_cfg["candidate_peak_window_start"])
    candidate_window_end = min(int(peak_cfg["candidate_peak_window_end"]), horizon_months)
    candidate_center = float(candidate_window_start + candidate_window_end - 2) / 2.0
    iterations = int(peak_cfg["iterations"])
    samples_per_iteration = int(peak_cfg["samples_per_iteration"])
    elite_count = int(peak_cfg["elite_count"])
    random_seeds = int(peak_cfg["random_seeds_per_run"])
    initial_std_cfg = dict(peak_cfg["initial_search_std"] or {})
    adaptive_std_cfg = dict(peak_cfg["adaptive_search_std"] or {})
    peak_window_q = dict(peak_cfg["peak_window_quantiles"] or {})
    peak_height_q = dict(peak_cfg["peak_height_quantiles"] or {})
    slow_mean = np.zeros((slow_basis.shape[-1], len(TRANSITION_NAMES)), dtype=np.float32)
    medium_mean = np.zeros((medium_basis.shape[-1], len(TRANSITION_NAMES)), dtype=np.float32)
    offset_mean = np.zeros((len(TRANSITION_NAMES),), dtype=np.float32)
    group_mean = np.zeros((len(group_rows),), dtype=np.float32)
    slow_std = np.ones_like(slow_mean, dtype=np.float32) * float(initial_std_cfg["slow"])
    medium_std = np.ones_like(medium_mean, dtype=np.float32) * float(initial_std_cfg["medium"])
    offset_std = np.ones_like(offset_mean, dtype=np.float32) * float(initial_std_cfg["offset"])
    group_std = np.ones_like(group_mean, dtype=np.float32) * float(initial_std_cfg["group"])
    elite_rows: list[dict[str, Any]] = []

    for seed in range(random_seeds):
        rng = np.random.default_rng(17 + seed)
        for _iteration in range(iterations):
            candidate_rows: list[dict[str, Any]] = []
            for _sample in range(samples_per_iteration):
                offset_params = rng.normal(offset_mean, offset_std).astype(np.float32)
                slow_params = rng.normal(slow_mean, slow_std).astype(np.float32) if slow_mean.size else slow_mean
                medium_params = rng.normal(medium_mean, medium_std).astype(np.float32) if medium_mean.size else medium_mean
                group_params = rng.normal(group_mean, group_std).astype(np.float32) if group_mean.size else group_mean
                temporal_effect = (
                    (slow_basis @ slow_params if slow_params.size else 0.0)
                    + (medium_basis @ medium_params if medium_params.size else 0.0)
                )
                group_effect = (group_params.reshape(-1, 1) * group_templates).sum(axis=0) if group_params.size else np.zeros((len(TRANSITION_NAMES),), dtype=np.float32)
                future_logit = base_logit[:, None, :] + offset_params.reshape(1, 1, len(TRANSITION_NAMES)) + temporal_effect.reshape(1, horizon_months, len(TRANSITION_NAMES)) + group_effect.reshape(1, 1, len(TRANSITION_NAMES))
                future_probs = np.clip(sigmoid(future_logit), float(evolution_cfg["transition_floor"]), float(evolution_cfg["transition_ceiling"])).astype(np.float32)
                future_states, state_mass_violation = _simulate_future_states(base_state, future_probs)
                series = _target_series(future_states, province_axis, target)
                search_slice = series[candidate_window_start - 1 : candidate_window_end]
                relative_peak = int(np.argmax(search_slice)) if search_slice.size else 0
                peak_index = max(candidate_window_start - 1, min(candidate_window_end - 1, candidate_window_start - 1 + relative_peak))
                score, diagnostics = _search_score(
                    series=series,
                    base_series=base_series,
                    peak_index=peak_index,
                    offset_params=offset_params,
                    slow_params=np.asarray(slow_params, dtype=np.float32),
                    medium_params=np.asarray(medium_params, dtype=np.float32),
                    group_params=np.asarray(group_params, dtype=np.float32),
                    future_transition_probs=future_probs,
                    state_mass_violation=state_mass_violation,
                    weight_cfg=peak_cfg,
                    evolution_cfg=evolution_cfg,
                    candidate_center=candidate_center,
                )
                candidate_rows.append(
                    {
                        "score": float(score),
                        "peak_index": int(peak_index),
                        "peak_month": future_month_axis[int(peak_index)],
                        "peak_height": round(float(series[int(peak_index)]), 6),
                        "diagnostics": diagnostics,
                        "offset_params": offset_params.round(6).tolist(),
                        "slow_params": np.asarray(slow_params, dtype=np.float32).round(6).tolist(),
                        "medium_params": np.asarray(medium_params, dtype=np.float32).round(6).tolist(),
                        "group_params": np.asarray(group_params, dtype=np.float32).round(6).tolist(),
                    }
                )
            candidate_rows.sort(key=lambda row: float(row["score"]), reverse=True)
            elites = candidate_rows[: max(1, min(elite_count, len(candidate_rows)))]
            elite_rows.extend(elites)
            offset_stack = np.asarray([row["offset_params"] for row in elites], dtype=np.float32)
            offset_mean = offset_stack.mean(axis=0)
            offset_std = np.clip(
                offset_stack.std(axis=0) + float(adaptive_std_cfg["jitter"]),
                float(adaptive_std_cfg["floor"]),
                float(adaptive_std_cfg["ceiling"]),
            )
            if slow_mean.size:
                slow_stack = np.asarray([row["slow_params"] for row in elites], dtype=np.float32)
                slow_mean = slow_stack.mean(axis=0)
                slow_std = np.clip(
                    slow_stack.std(axis=0) + float(adaptive_std_cfg["jitter"]),
                    float(adaptive_std_cfg["floor"]),
                    float(adaptive_std_cfg["ceiling"]),
                )
            if medium_mean.size:
                medium_stack = np.asarray([row["medium_params"] for row in elites], dtype=np.float32)
                medium_mean = medium_stack.mean(axis=0)
                medium_std = np.clip(
                    medium_stack.std(axis=0) + float(adaptive_std_cfg["jitter"]),
                    float(adaptive_std_cfg["floor"]),
                    float(adaptive_std_cfg["ceiling"]),
                )
            if group_mean.size:
                group_stack = np.asarray([row["group_params"] for row in elites], dtype=np.float32)
                group_mean = group_stack.mean(axis=0)
                group_std = np.clip(
                    group_stack.std(axis=0) + float(adaptive_std_cfg["jitter"]),
                    float(adaptive_std_cfg["floor"]),
                    float(adaptive_std_cfg["ceiling"]),
                )

    elite_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    elites = elite_rows[: max(1, elite_count)]
    peak_indices = np.asarray([int(row["peak_index"]) for row in elites], dtype=np.int32)
    peak_heights = np.asarray([float(row["peak_height"]) for row in elites], dtype=np.float32)
    group_stack = np.asarray([row["group_params"] for row in elites], dtype=np.float32) if group_rows and elites else np.zeros((0, len(group_rows)), dtype=np.float32)
    top_precursors = []
    if group_stack.size:
        mean_importance = np.mean(np.abs(group_stack), axis=0)
        ranking = np.argsort(-mean_importance)
        for idx in ranking[: min(5, len(group_rows))]:
            top_precursors.append(
                {
                    "group_name": str(group_rows[idx].get("group_name") or f"group_{idx}"),
                    "mean_absolute_weight": round(float(mean_importance[idx]), 6),
                }
            )
    summary = {
        "available": True,
        "target_name": target,
        "representation": representation,
        "horizon_months": int(horizon_months),
        "mode_peak_month": future_month_axis[int(np.bincount(peak_indices).argmax())] if peak_indices.size else None,
        "peak_window_start": future_month_axis[int(np.quantile(peak_indices, float(peak_window_q["lower"])))] if peak_indices.size else None,
        "peak_window_end": future_month_axis[int(np.quantile(peak_indices, float(peak_window_q["upper"])))] if peak_indices.size else None,
        "earliest_plausible_peak_month": future_month_axis[int(np.min(peak_indices))] if peak_indices.size else None,
        "latest_plausible_peak_month": future_month_axis[int(np.max(peak_indices))] if peak_indices.size else None,
        "mode_peak_height": round(float(np.quantile(peak_heights, float(peak_height_q["median"]))) if peak_heights.size else 0.0, 6),
        "peak_height_p10": round(float(np.quantile(peak_heights, float(peak_height_q["lower"]))) if peak_heights.size else 0.0, 6),
        "peak_height_p50": round(float(np.quantile(peak_heights, float(peak_height_q["median"]))) if peak_heights.size else 0.0, 6),
        "peak_height_p90": round(float(np.quantile(peak_heights, float(peak_height_q["upper"]))) if peak_heights.size else 0.0, 6),
        "shared_precursor_drivers": top_precursors,
        "elite_path_count": len(elites),
        "annual_anchor_fit_summary": {
            "continuity_error": round(float(np.mean([float(row["diagnostics"]["annual_anchor_error"]) for row in elites])) if elites else 0.0, 6),
            "mean_state_mass_violation": round(float(np.mean([float(row["diagnostics"]["state_mass_violation"]) for row in elites])) if elites else 0.0, 6),
            "promotion_gate": gate,
        },
        "promotion_gate": gate,
    }
    envelope = {
        "target_name": target,
        "horizon_months": int(horizon_months),
        "mode_peak_month": summary["mode_peak_month"],
        "peak_window_start": summary["peak_window_start"],
        "peak_window_end": summary["peak_window_end"],
        "earliest_plausible_peak_month": summary["earliest_plausible_peak_month"],
        "latest_plausible_peak_month": summary["latest_plausible_peak_month"],
        "mode_peak_height": summary["mode_peak_height"],
        "peak_height_p10": summary["peak_height_p10"],
        "peak_height_p50": summary["peak_height_p50"],
        "peak_height_p90": summary["peak_height_p90"],
        "shared_precursor_drivers": top_precursors,
        "elite_path_count": len(elites),
        "annual_anchor_fit_summary": summary["annual_anchor_fit_summary"],
    }
    write_json(phase_dir / "peak_search_summary.json", summary)
    write_json(phase_dir / "peak_envelope.json", envelope)
    write_json(phase_dir / "peak_search_elite_paths.json", {"elite_paths": elites, "promotion_gate": gate})
    return summary
