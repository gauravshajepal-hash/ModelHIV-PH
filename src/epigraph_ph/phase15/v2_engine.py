from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase15.latent_measurements import (
    build_sparse_indicator_cube,
    month_slots_for_row,
    normalize_region_label,
    province_region_codes,
    province_region_members,
    row_weight,
)
from epigraph_ph.phase15.missing_information_numerics import (
    apply_missing_information_precision_operator_cpu as _mi_apply_precision_operator_cpu,
    missing_information_backend_runtime as _mi_backend_runtime,
    solve_missing_information_linear_system as _mi_solve_linear_system,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean_value = _weighted_mean(values, weights)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(np.var(values))
    centered = values - mean_value
    return float(np.sum(weights * np.square(centered)) / total)


def _weighted_corr(values: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or targets.size == 0 or values.size != targets.size:
        return 0.0
    vx = _weighted_var(values, weights)
    vy = _weighted_var(targets, weights)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    mx = _weighted_mean(values, weights)
    my = _weighted_mean(targets, weights)
    cov = _weighted_mean((values - mx) * (targets - my), weights)
    corr = cov / max(float(np.sqrt(vx * vy)), 1e-12)
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _inverse_softplus(value: float) -> float:
    magnitude = max(abs(float(value)), 1e-6)
    return float(np.log(np.expm1(magnitude)))


def _softplus(value: float) -> float:
    numeric = float(value)
    if numeric > 20.0:
        return numeric
    return float(np.log1p(np.exp(numeric)))


def _sigmoid(value: float) -> float:
    numeric = float(value)
    if numeric >= 0.0:
        exp_neg = float(np.exp(-numeric))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = float(np.exp(numeric))
    return exp_pos / (1.0 + exp_pos)


def _softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.astype(np.float64)
    shifted = array - float(np.max(array))
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(np.sum(exp_scores), 1e-8, None)


def _posterior_mean_precision(
    residual: np.ndarray,
    *,
    prior_dof: float,
    prior_variance: float,
    floor: float,
    ceiling: float,
) -> float:
    residual_arr = np.asarray(residual, dtype=np.float64).reshape(-1)
    if residual_arr.size == 0:
        return float(floor)
    ssr = float(np.sum(np.square(residual_arr)))
    post_shape = 0.5 * (float(prior_dof) + float(residual_arr.size))
    post_scale = 0.5 * (float(prior_dof) * max(float(prior_variance), 1e-6) + ssr)
    precision = post_shape / max(post_scale, 1e-8)
    return float(np.clip(precision, float(floor), float(ceiling)))


def _phase15_v2_engine_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase15_cfg = dict((plugin.constraint_settings or {}).get("phase15", {}) or {})
    v2_cfg = dict(phase15_cfg.get("latent_blocks_v2", {}) or {})
    engine_cfg = dict(v2_cfg.get("engine", {}) or {})
    return {
        "enabled": bool(v2_cfg.get("enabled", True)),
        "outer_iterations": int(engine_cfg.get("outer_iterations") or 6),
        "inner_gradient_steps": int(engine_cfg.get("inner_gradient_steps") or 20),
        "learning_rate": float(engine_cfg.get("learning_rate") or 0.35),
        "loading_theta_steps": int(engine_cfg.get("loading_theta_steps") or 32),
        "loading_theta_learning_rate": float(engine_cfg.get("loading_theta_learning_rate") or 0.20),
        "loading_theta_ridge": float(engine_cfg.get("loading_theta_ridge") or 0.05),
        "loading_block_prior_precision": float(engine_cfg.get("loading_block_prior_precision") or 0.35),
        "loading_row_scale_precision": float(engine_cfg.get("loading_row_scale_precision") or 6.0),
        "loading_weight_scale_precision": float(engine_cfg.get("loading_weight_scale_precision") or 3.0),
        "loading_sign_conflict_precision": float(engine_cfg.get("loading_sign_conflict_precision") or 4.0),
        "temporal_smoothing_blend": float(engine_cfg.get("temporal_smoothing_blend") or 0.35),
        "aggregation_smoothing_passes": int(engine_cfg.get("aggregation_smoothing_passes") or 2),
        "loading_ridge": float(engine_cfg.get("loading_ridge") or 0.5),
        "minimum_loading_magnitude": float(engine_cfg.get("minimum_loading_magnitude") or 0.05),
        "loading_magnitude_ceiling": float(engine_cfg.get("loading_magnitude_ceiling") or 4.0),
        "aggregation_learning_steps": int(engine_cfg.get("aggregation_learning_steps") or 120),
        "aggregation_learning_rate": float(engine_cfg.get("aggregation_learning_rate") or 0.25),
        "aggregation_gradient_clip": float(engine_cfg.get("aggregation_gradient_clip") or 5.0),
        "aggregation_backtracking_steps": int(engine_cfg.get("aggregation_backtracking_steps") or 8),
        "aggregation_ridge": float(engine_cfg.get("aggregation_ridge") or 0.05),
        "aggregation_feature_floor": float(engine_cfg.get("aggregation_feature_floor") or 1e-4),
        "aggregation_equal_blend": float(engine_cfg.get("aggregation_equal_blend") or 0.05),
        "aggregation_identification_gain_floor": float(engine_cfg.get("aggregation_identification_gain_floor") or 1e-3),
        "aggregation_identification_norm_floor": float(engine_cfg.get("aggregation_identification_norm_floor") or 1e-2),
        "aggregation_identification_dispersion_floor": float(engine_cfg.get("aggregation_identification_dispersion_floor") or 1e-3),
        "aggregation_auto_support_enabled": bool(engine_cfg.get("aggregation_auto_support_enabled", True)),
        "aggregation_auto_feature_min_subnational": int(engine_cfg.get("aggregation_auto_feature_min_subnational") or 3),
        "aggregation_auto_feature_min_aggregate": int(engine_cfg.get("aggregation_auto_feature_min_aggregate") or 1),
        "aggregation_auto_feature_max_names": int(engine_cfg.get("aggregation_auto_feature_max_names") or 16),
        "variance_eps": float(engine_cfg.get("variance_eps") or 1e-4),
        "standardization_eps": float(engine_cfg.get("standardization_eps") or 1e-6),
        "precision_feature_min_rows": int(engine_cfg.get("precision_feature_min_rows") or 4),
        "observation_precision_floor": float(engine_cfg.get("observation_precision_floor") or 0.05),
        "observation_precision_ceiling": float(engine_cfg.get("observation_precision_ceiling") or 50.0),
        "temporal_precision_floor": float(engine_cfg.get("temporal_precision_floor") or 0.1),
        "temporal_precision_ceiling": float(engine_cfg.get("temporal_precision_ceiling") or 25.0),
        "temporal_precision_prior_dof": float(engine_cfg.get("temporal_precision_prior_dof") or 120.0),
        "temporal_precision_prior_variance": float(engine_cfg.get("temporal_precision_prior_variance") or 4.0),
        "regional_precision_floor": float(engine_cfg.get("regional_precision_floor") or 0.05),
        "regional_precision_ceiling": float(engine_cfg.get("regional_precision_ceiling") or 25.0),
        "regional_precision_prior_dof": float(engine_cfg.get("regional_precision_prior_dof") or 120.0),
        "regional_precision_prior_variance": float(engine_cfg.get("regional_precision_prior_variance") or 1.0),
        "state_value_clip": float(engine_cfg.get("state_value_clip") or 6.0),
        "persistence_abs_ceiling": float(dict(v2_cfg.get("state_dynamics") or {}).get("persistence_abs_ceiling") or 0.995),
        "persistence_prior_mean": float(dict(v2_cfg.get("state_dynamics") or {}).get("persistence_prior_mean") or 0.80),
        "persistence_prior_strength": float(dict(v2_cfg.get("state_dynamics") or {}).get("persistence_prior_strength") or 12.0),
        "proxy_weight_names": [
            str(item)
            for item in list(engine_cfg.get("weight_proxy_canonical_names") or ["population_total", "estimated_plhiv", "case_count"])
        ],
        "observation_weight_floor": float(engine_cfg.get("observation_weight_floor") or 0.25),
        "missing_information_enabled": bool(engine_cfg.get("missing_information_enabled", True)),
        "missing_information_correction_prior_dof": float(
            engine_cfg.get("missing_information_correction_prior_dof") or 24.0
        ),
        "missing_information_correction_prior_variance": float(
            engine_cfg.get("missing_information_correction_prior_variance") or 1.0
        ),
        "missing_information_correction_precision_floor": float(
            engine_cfg.get("missing_information_correction_precision_floor") or 0.01
        ),
        "missing_information_correction_precision_ceiling": float(
            engine_cfg.get("missing_information_correction_precision_ceiling") or 25.0
        ),
        "missing_information_donor_precision_prior_dof": float(
            engine_cfg.get("missing_information_donor_precision_prior_dof") or 60.0
        ),
        "missing_information_donor_precision_prior_variance": float(
            engine_cfg.get("missing_information_donor_precision_prior_variance") or 1.0
        ),
        "missing_information_donor_precision_floor": float(
            engine_cfg.get("missing_information_donor_precision_floor") or 0.0
        ),
        "missing_information_donor_precision_ceiling": float(
            engine_cfg.get("missing_information_donor_precision_ceiling") or 25.0
        ),
        "missing_information_backend": str(engine_cfg.get("missing_information_backend") or "auto"),
        "missing_information_preconditioner": str(engine_cfg.get("missing_information_preconditioner") or "temporal_block"),
        "missing_information_preconditioner_cholesky_jitter": float(
            engine_cfg.get("missing_information_preconditioner_cholesky_jitter") or 1e-6
        ),
        "missing_information_preconditioner_cholesky_max_attempts": int(
            engine_cfg.get("missing_information_preconditioner_cholesky_max_attempts") or 4
        ),
        "missing_information_torch_dtype": str(engine_cfg.get("missing_information_torch_dtype") or "float32"),
        "missing_information_torch_compile": bool(engine_cfg.get("missing_information_torch_compile", True)),
        "missing_information_torch_cg_max_iter": int(engine_cfg.get("missing_information_torch_cg_max_iter") or 256),
        "missing_information_torch_cg_retry_max_iter": int(engine_cfg.get("missing_information_torch_cg_retry_max_iter") or 1024),
        "missing_information_torch_cg_rtol": float(engine_cfg.get("missing_information_torch_cg_rtol") or 1e-5),
        "missing_information_torch_cg_atol": float(engine_cfg.get("missing_information_torch_cg_atol") or 1e-7),
        "missing_information_cpu_cg_max_iter": int(engine_cfg.get("missing_information_cpu_cg_max_iter") or 2048),
        "missing_information_fallback_to_cpu": bool(engine_cfg.get("missing_information_fallback_to_cpu", True)),
        "missing_information_retry_on_nonconvergence": bool(
            engine_cfg.get("missing_information_retry_on_nonconvergence", True)
        ),
        "numerical_adequacy_enabled": bool(engine_cfg.get("numerical_adequacy_enabled", True)),
        "numerical_adequacy_tolerance_rtol_values": [
            float(value) for value in list(engine_cfg.get("numerical_adequacy_tolerance_rtol_values") or [1e-4, 1e-5, 1e-6])
        ],
        "numerical_adequacy_tolerance_atol": float(engine_cfg.get("numerical_adequacy_tolerance_atol") or 1e-7),
        "numerical_adequacy_reference_max_provinces": int(
            engine_cfg.get("numerical_adequacy_reference_max_provinces") or 8
        ),
        "numerical_adequacy_reference_max_cells": int(
            engine_cfg.get("numerical_adequacy_reference_max_cells") or 256
        ),
        "pooling_sensitivity_enabled": bool(engine_cfg.get("pooling_sensitivity_enabled", True)),
        "pooling_sensitivity_scales": [float(value) for value in list(engine_cfg.get("pooling_sensitivity_scales") or [0.5, 0.25])],
        "pooling_sensitivity_outer_iterations": int(engine_cfg.get("pooling_sensitivity_outer_iterations") or 3),
        "pooling_sensitivity_inner_gradient_steps": int(engine_cfg.get("pooling_sensitivity_inner_gradient_steps") or 10),
        "calibration_intervals": [float(value) for value in list(engine_cfg.get("calibration_intervals") or [0.5, 0.8, 0.95])],
    }


def _infer_weight_feature_and_supervision_names(
    *,
    normalized_rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    base_names = [str(name) for name in list(cfg.get("proxy_weight_names") or []) if str(name).strip()]
    summary: dict[str, dict[str, Any]] = {}
    for row in normalized_rows:
        if str(row.get("measurement_role") or "") == "context_only":
            continue
        canonical_name = str(row.get("canonical_name") or "")
        if not canonical_name:
            continue
        numeric = _numeric_value(row)
        if numeric is None:
            continue
        item = summary.setdefault(
            canonical_name,
            {
                "province_support": 0,
                "aggregate_support": 0,
                "monthly_support": 0,
                "annual_support": 0,
                "measurement_roles": set(),
                "candidate_blocks": set(),
            },
        )
        geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
        geo_token = str(row.get("geo") or "").strip().lower()
        region_token = normalize_region_label(str(row.get("region") or ""))
        if geo_resolution in {"province", "city"}:
            item["province_support"] += 1
        if geo_resolution == "region" or geo_resolution == "national" or geo_token == "philippines" or region_token == "national":
            item["aggregate_support"] += 1
        time_resolution = str(row.get("time_resolution") or "").strip().lower()
        if time_resolution == "monthly":
            item["monthly_support"] += 1
        if time_resolution == "annual":
            item["annual_support"] += 1
        item["measurement_roles"].add(str(row.get("measurement_role") or ""))
        item["candidate_blocks"].add(str(row.get("candidate_block") or ""))

    auto_candidates: list[tuple[str, int, int]] = []
    base_support_candidates: list[tuple[str, int, int]] = []
    if bool(cfg.get("aggregation_auto_support_enabled", True)):
        for canonical_name, item in summary.items():
            province_support = int(item["province_support"])
            aggregate_support = int(item["aggregate_support"])
            candidate_blocks = {str(value) for value in set(item["candidate_blocks"])}
            if canonical_name in base_names and province_support > 0 and aggregate_support > 0:
                base_support_candidates.append((canonical_name, province_support, aggregate_support))
            if canonical_name in base_names:
                continue
            if province_support < int(cfg["aggregation_auto_feature_min_subnational"]):
                continue
            if aggregate_support < int(cfg["aggregation_auto_feature_min_aggregate"]):
                continue
            if candidate_blocks == {"unassigned"}:
                continue
            auto_candidates.append((canonical_name, province_support, aggregate_support))
        auto_candidates.sort(key=lambda item: (-(item[1] + item[2]), item[0]))
        auto_candidates = auto_candidates[: int(cfg["aggregation_auto_feature_max_names"])]
        base_support_candidates.sort(key=lambda item: (-(item[1] + item[2]), item[0]))

    feature_names: list[str] = []
    seen_names: set[str] = set()
    for canonical_name in list(base_names) + [name for name, _, _ in base_support_candidates] + [name for name, _, _ in auto_candidates]:
        if canonical_name in seen_names:
            continue
        seen_names.add(canonical_name)
        feature_names.append(canonical_name)

    supervision_names = [
        name
        for name, _, _ in list(base_support_candidates) + list(auto_candidates)
        if name in seen_names
    ]
    for canonical_name in feature_names:
        summary.setdefault(
            canonical_name,
            {
                "province_support": 0,
                "aggregate_support": 0,
                "monthly_support": 0,
                "annual_support": 0,
                "measurement_roles": set(),
                "candidate_blocks": set(),
            },
        )

    serializable_summary: dict[str, dict[str, Any]] = {}
    for canonical_name in feature_names:
        item = summary.get(canonical_name) or {}
        serializable_summary[canonical_name] = {
            "province_support": int(item.get("province_support") or 0),
            "aggregate_support": int(item.get("aggregate_support") or 0),
            "monthly_support": int(item.get("monthly_support") or 0),
            "annual_support": int(item.get("annual_support") or 0),
            "measurement_roles": sorted(str(value) for value in list(item.get("measurement_roles") or [])),
            "candidate_blocks": sorted(str(value) for value in list(item.get("candidate_blocks") or [])),
        }
    return feature_names, supervision_names, serializable_summary


def _numeric_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("model_numeric_value")
    if value in (None, ""):
        value = row.get("value")
    numeric = _safe_float(value, default=float("nan"))
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _indicator_transform_kind(canonical_name: str, values: np.ndarray) -> str:
    token = str(canonical_name or "").lower()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "identity"
    ratio_like = any(
        marker in token
        for marker in (
            "rate",
            "share",
            "coverage",
            "suppression",
            "uptake",
            "adherence",
            "retention",
            "prevalence",
            "knowledge",
        )
    )
    if np.all((finite >= 0.0) & (finite <= 1.0)):
        return "logit_01"
    if ratio_like and np.all((finite >= 0.0) & (finite <= 100.0)):
        return "logit_percent"
    positive = finite[finite > 0.0]
    if positive.size >= 4:
        p10 = float(np.percentile(positive, 10))
        p90 = float(np.percentile(positive, 90))
        if p10 > 0.0 and p90 / p10 >= 10.0:
            return "log1p"
    return "identity"


def _apply_indicator_transform(value: float, transform_kind: str, clip_eps: float) -> float:
    numeric = float(value)
    if transform_kind == "logit_01":
        clipped = float(np.clip(numeric, clip_eps, 1.0 - clip_eps))
        return float(np.log(clipped / max(1.0 - clipped, clip_eps)))
    if transform_kind == "logit_percent":
        clipped = float(np.clip(numeric / 100.0, clip_eps, 1.0 - clip_eps))
        return float(np.log(clipped / max(1.0 - clipped, clip_eps)))
    if transform_kind == "log1p":
        return float(np.log1p(max(numeric, 0.0)))
    return numeric


def _build_indicator_transform_stats(
    normalized_rows: list[dict[str, Any]],
    retained_canonical_names: set[str],
    *,
    variance_eps: float,
) -> dict[str, dict[str, float | str]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in normalized_rows:
        if str(row.get("measurement_role") or "") == "context_only":
            continue
        canonical_name = str(row.get("canonical_name") or "")
        if canonical_name not in retained_canonical_names:
            continue
        numeric = _numeric_value(row)
        if numeric is None:
            continue
        grouped[canonical_name].append(float(numeric))

    stats: dict[str, dict[str, float | str]] = {}
    for canonical_name in sorted(grouped):
        values = np.asarray(grouped[canonical_name], dtype=np.float64)
        transform_kind = _indicator_transform_kind(canonical_name, values)
        transformed = np.asarray(
            [_apply_indicator_transform(float(value), transform_kind, clip_eps=1e-3) for value in values],
            dtype=np.float64,
        )
        center = float(np.median(transformed))
        mad = float(np.median(np.abs(transformed - center)))
        scale = max(mad * 1.4826, float(np.std(transformed)), variance_eps)
        stats[canonical_name] = {
            "transform_kind": transform_kind,
            "center": center,
            "scale": scale,
        }
    return stats


def _support_cells(support_row: Mapping[str, Any], month_count: int) -> np.ndarray:
    province_indices = [int(idx) for idx in list(support_row.get("province_indices") or [])]
    month_indices = [int(idx) for idx in list(support_row.get("month_indices") or [])]
    cells = [
        int(province_idx) * int(month_count) + int(month_idx)
        for province_idx in province_indices
        for month_idx in month_indices
    ]
    return np.asarray(cells, dtype=np.int32)


def _retained_indicator_lookup(measurement_spec: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    indicator_to_block: dict[str, str] = {}
    indicator_to_sign: dict[str, str] = {}
    block_display: dict[str, str] = {}
    for block in list(dict(measurement_spec).get("retained_blocks") or []):
        block_id = str(block.get("block_id") or "")
        display_name = str(block.get("display_name") or block_id.replace("_", " ").title())
        block_display[block_id] = display_name
        for indicator in list(block.get("indicator_rows") or []):
            canonical_name = str(indicator.get("canonical_name") or "")
            if not canonical_name:
                continue
            indicator_to_block[canonical_name] = block_id
            indicator_to_sign[canonical_name] = str(indicator.get("expected_sign") or "neutral")
    return indicator_to_block, indicator_to_sign, block_display


def _aggregate_measurement_rows(
    *,
    normalized_rows: list[dict[str, Any]],
    observation_support: Mapping[str, Any],
    measurement_spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    indicator_to_block, indicator_to_sign, block_display = _retained_indicator_lookup(measurement_spec)
    retained_canonical_names = set(indicator_to_block)
    transform_stats = _build_indicator_transform_stats(
        normalized_rows=normalized_rows,
        retained_canonical_names=retained_canonical_names,
        variance_eps=float(cfg["variance_eps"]),
    )
    month_count = len(list(observation_support.get("month_axis") or []))
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for support_row in list(dict(observation_support).get("rows") or []):
        source_row_index_value = support_row.get("source_row_index")
        source_row_index = -1 if source_row_index_value is None else int(source_row_index_value)
        if source_row_index < 0 or source_row_index >= len(normalized_rows):
            continue
        source_row = normalized_rows[source_row_index]
        canonical_name = str(source_row.get("canonical_name") or "")
        if canonical_name not in indicator_to_block:
            continue
        numeric = _numeric_value(source_row)
        if numeric is None:
            continue
        transform_row = dict(transform_stats.get(canonical_name) or {})
        if not transform_row:
            continue
        transformed = _apply_indicator_transform(float(numeric), str(transform_row["transform_kind"]), clip_eps=1e-3)
        standardized_value = (float(transformed) - float(transform_row["center"])) / max(
            float(transform_row["scale"]),
            float(cfg["variance_eps"]),
        )
        base_weight = row_weight(source_row, floor=float(cfg["observation_weight_floor"]))
        reliability = max(0.25, 1.0 - max(0.0, _safe_float(source_row.get("bias_penalty"), default=0.0)))
        support_cells = _support_cells(support_row, month_count)
        if support_cells.size == 0:
            continue
        anchor_status = "anchor" if bool(source_row.get("is_anchor_eligible")) else "non_anchor"
        key = (
            indicator_to_block[canonical_name],
            canonical_name,
            str(indicator_to_sign.get(canonical_name) or "neutral"),
            tuple(int(cell) for cell in support_cells.tolist()),
            str(support_row.get("operator_kind") or ""),
            str(source_row.get("measurement_role") or ""),
            str(source_row.get("source_bank") or ""),
            str(source_row.get("geo_resolution") or ""),
            str(source_row.get("time_resolution") or ""),
            anchor_status,
        )
        item = grouped.setdefault(
            key,
            {
                "block_id": indicator_to_block[canonical_name],
                "display_name": block_display.get(
                    indicator_to_block[canonical_name],
                    indicator_to_block[canonical_name].replace("_", " ").title(),
                ),
                "canonical_name": canonical_name,
                "expected_sign": str(indicator_to_sign.get(canonical_name) or "neutral"),
                "operator_kind": str(support_row.get("operator_kind") or ""),
                "measurement_role": str(source_row.get("measurement_role") or ""),
                "source_bank": str(source_row.get("source_bank") or ""),
                "geo_resolution": str(source_row.get("geo_resolution") or ""),
                "time_resolution": str(source_row.get("time_resolution") or ""),
                "anchor_status": anchor_status,
                "support_cells": np.asarray(support_cells, dtype=np.int32),
                "cell_weight": float(
                    support_row.get("normalized_cell_weight") or (1.0 / max(int(support_row.get("cell_count") or 1), 1))
                ),
                "support_scope": str(support_row.get("support_scope") or ""),
                "time_support_mode": str(support_row.get("time_support_mode") or ""),
                "source_row_indices": [],
                "weight_sum": 0.0,
                "value_sum": 0.0,
                "quality_signal_sum": 0.0,
                "raw_numeric_value_sum": 0.0,
            },
        )
        item["source_row_indices"].append(int(source_row_index))
        item["weight_sum"] += float(base_weight)
        item["value_sum"] += float(base_weight) * float(standardized_value)
        item["quality_signal_sum"] += float(base_weight) * float(reliability)
        item["raw_numeric_value_sum"] += float(base_weight) * float(numeric)

    measurement_rows: list[dict[str, Any]] = []
    for item in grouped.values():
        total_weight = max(float(item["weight_sum"]), 1e-8)
        measurement_rows.append(
            {
                "block_id": str(item["block_id"]),
                "display_name": str(item["display_name"]),
                "canonical_name": str(item["canonical_name"]),
                "expected_sign": str(item["expected_sign"]),
                "operator_kind": str(item["operator_kind"]),
                "measurement_role": str(item["measurement_role"]),
                "source_bank": str(item["source_bank"]),
                "geo_resolution": str(item["geo_resolution"]),
                "time_resolution": str(item["time_resolution"]),
                "anchor_status": str(item["anchor_status"]),
                "support_scope": str(item["support_scope"]),
                "time_support_mode": str(item["time_support_mode"]),
                "support_cells": np.asarray(item["support_cells"], dtype=np.int32),
                "cell_weight": float(item["cell_weight"]),
                "measurement_value": float(item["value_sum"]) / total_weight,
                "base_weight": total_weight,
                "quality_signal": float(item["quality_signal_sum"]) / total_weight,
                "raw_numeric_value": float(item["raw_numeric_value_sum"]) / total_weight,
                "source_row_count": len(list(item["source_row_indices"])),
                "source_row_indices": [int(idx) for idx in list(item["source_row_indices"])],
                "transform_kind": str(transform_stats[str(item["canonical_name"])]["transform_kind"]),
                "transform_center": float(transform_stats[str(item["canonical_name"])]["center"]),
                "transform_scale": float(transform_stats[str(item["canonical_name"])]["scale"]),
            }
        )

    measurement_rows.sort(
        key=lambda row: (
            str(row.get("block_id") or ""),
            str(row.get("canonical_name") or ""),
            str(row.get("measurement_role") or ""),
            str(row.get("source_bank") or ""),
            tuple(
                int(cell)
                for cell in list(
                    np.asarray(row.get("support_cells") if row.get("support_cells") is not None else [], dtype=np.int32).tolist()
                )
            ),
        )
    )
    return {
        "measurement_rows": measurement_rows,
        "transform_stats": transform_stats,
        "block_display": block_display,
    }


def _estimate_aggregation_weights(
    *,
    normalized_rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    region_labels: list[str],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    feature_names, supervision_names, support_summary = _infer_weight_feature_and_supervision_names(
        normalized_rows=normalized_rows,
        cfg=cfg,
    )
    region_codes = province_region_codes(province_axis, region_labels)
    region_members = province_region_members(region_codes)
    province_count = len(province_axis)
    weights = np.ones((province_count,), dtype=np.float64)
    weight_source = "equal_weight_fallback"
    coefficient_values: list[float] = []
    diagnostics: dict[str, float | int | str] = {
        "sample_count": 0,
        "national_sample_count": 0,
        "regional_sample_count": 0,
        "baseline_objective": 0.0,
        "learned_objective": 0.0,
        "objective_gain": 0.0,
        "coefficient_norm": 0.0,
        "weight_dispersion": 0.0,
    }
    if feature_names:
        cube_payload = build_sparse_indicator_cube(
            normalized_rows=normalized_rows,
            province_axis=province_axis,
            month_axis=month_axis,
            canonical_names=feature_names,
            region_labels=region_labels,
            include_national_rows=False,
            observation_weight_floor=float(cfg["observation_weight_floor"]),
        )
        raw_cube = np.asarray(cube_payload["raw_cube"], dtype=np.float64)
        weight_cube = np.asarray(cube_payload["weight_cube"], dtype=np.float64)
        canonical_axis = [str(name) for name in list(cube_payload.get("canonical_axis") or [])]
        transform_stats = _build_indicator_transform_stats(
            normalized_rows=normalized_rows,
            retained_canonical_names=set(canonical_axis),
            variance_eps=float(cfg["variance_eps"]),
        )
        transformed_cube = np.zeros_like(raw_cube, dtype=np.float64)
        for canonical_idx, canonical_name in enumerate(canonical_axis):
            transform_row = dict(transform_stats.get(canonical_name) or {})
            if not transform_row:
                transformed_cube[:, :, canonical_idx] = raw_cube[:, :, canonical_idx]
                continue
            transform_kind = str(transform_row["transform_kind"])
            center = float(transform_row["center"])
            scale = max(float(transform_row["scale"]), float(cfg["variance_eps"]))
            for province_idx in range(raw_cube.shape[0]):
                for month_idx in range(raw_cube.shape[1]):
                    if float(weight_cube[province_idx, month_idx, canonical_idx]) <= 0.0:
                        continue
                    transformed_value = _apply_indicator_transform(
                        float(raw_cube[province_idx, month_idx, canonical_idx]),
                        transform_kind,
                        clip_eps=1e-3,
                    )
                    transformed_cube[province_idx, month_idx, canonical_idx] = (float(transformed_value) - center) / scale
        feature_matrix = np.zeros((province_count, len(canonical_axis)), dtype=np.float64)
        for canonical_idx, canonical_name in enumerate(canonical_axis):
            values = transformed_cube[:, :, canonical_idx]
            support = weight_cube[:, :, canonical_idx]
            support_sum = np.sum(support, axis=1, dtype=np.float64)
            province_feature = np.zeros((province_count,), dtype=np.float64)
            valid = support_sum > 0.0
            province_feature[valid] = np.sum(values * support, axis=1, dtype=np.float64)[valid] / np.clip(support_sum[valid], 1e-8, None)
            if np.any(valid):
                valid_values = province_feature[valid]
                mean_value = float(np.mean(valid_values))
                std_value = float(np.std(valid_values))
                if std_value > float(cfg["aggregation_feature_floor"]):
                    province_feature[valid] = (valid_values - mean_value) / std_value
                else:
                    province_feature[valid] = valid_values - mean_value
            feature_matrix[:, canonical_idx] = province_feature

        canonical_index = {name: idx for idx, name in enumerate(canonical_axis)}
        supervision_name_set = {name for name in supervision_names if name in canonical_index}
        national_samples: list[tuple[int, int, float, float]] = []
        regional_samples: list[tuple[str, int, int, float, float]] = []
        for row in normalized_rows:
            if str(row.get("measurement_role") or "") == "context_only":
                continue
            canonical_name = str(row.get("canonical_name") or "")
            if canonical_name not in supervision_name_set:
                continue
            numeric = _numeric_value(row)
            if numeric is None:
                continue
            transform_row = dict(transform_stats.get(canonical_name) or {})
            if transform_row:
                transformed_target = (
                    _apply_indicator_transform(float(numeric), str(transform_row["transform_kind"]), clip_eps=1e-3)
                    - float(transform_row["center"])
                ) / max(float(transform_row["scale"]), float(cfg["variance_eps"]))
            else:
                transformed_target = float(numeric)
            geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
            geo_token = str(row.get("geo") or "").strip().lower()
            region_token = normalize_region_label(str(row.get("region") or ""))
            month_slots = month_slots_for_row(row, month_axis)
            if not month_slots:
                continue
            sample_weight = row_weight(row, floor=float(cfg["observation_weight_floor"])) / float(max(len(month_slots), 1))
            if geo_resolution == "national" or geo_token == "philippines" or region_token == "national":
                for month_idx in month_slots:
                    national_samples.append((canonical_index[canonical_name], int(month_idx), float(transformed_target), float(sample_weight)))
                continue
            if geo_resolution == "region" and region_token in region_members:
                for month_idx in month_slots:
                    regional_samples.append(
                        (str(region_token), canonical_index[canonical_name], int(month_idx), float(transformed_target), float(sample_weight))
                    )

        proxy_scores = np.zeros((province_count,), dtype=np.float64)
        proxy_hits = np.zeros((province_count,), dtype=np.float64)
        for canonical_idx in range(raw_cube.shape[-1]):
            values = raw_cube[:, :, canonical_idx]
            support = weight_cube[:, :, canonical_idx]
            weighted_sum = np.sum(np.clip(values, 0.0, None) * support, axis=1, dtype=np.float64)
            support_sum = np.sum(support, axis=1, dtype=np.float64)
            valid = support_sum > 0.0
            proxy_scores[valid] += weighted_sum[valid] / np.clip(support_sum[valid], 1e-8, None)
            proxy_hits[valid] += 1.0
        baseline_weights = np.ones((province_count,), dtype=np.float64)
        valid_scores = proxy_hits > 0.0
        if np.any(valid_scores):
            baseline_weights[valid_scores] = proxy_scores[valid_scores] / np.clip(proxy_hits[valid_scores], 1e-8, None)
            weight_source = "proxy_weight_mean"
        baseline_weights = np.clip(baseline_weights, 1e-8, None)
        baseline_weights = baseline_weights / np.clip(np.sum(baseline_weights), 1e-8, None)

        def _objective_and_gradient(beta_value: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            scores = feature_matrix @ beta_value
            national_weights_local = _softmax(scores)
            gradient = np.zeros_like(beta_value)
            objective = 0.0
            total_sample_weight = 0.0
            national_feature_mean = national_weights_local @ feature_matrix if feature_matrix.size else np.zeros((0,), dtype=np.float64)
            for canonical_idx, month_idx, target_value, sample_weight in national_samples:
                province_values = raw_cube[:, month_idx, canonical_idx]
                prediction = float(np.sum(national_weights_local * province_values))
                error = prediction - float(target_value)
                objective += float(sample_weight) * float(error**2)
                total_sample_weight += float(sample_weight)
                weighted_value_features = national_weights_local @ (province_values[:, None] * feature_matrix)
                d_prediction = weighted_value_features - prediction * national_feature_mean
                gradient += 2.0 * float(sample_weight) * error * d_prediction
            for region_label, canonical_idx, month_idx, target_value, sample_weight in regional_samples:
                member_array = np.asarray(region_members[str(region_label)], dtype=np.int32)
                local_scores = scores[member_array]
                local_weights = _softmax(local_scores)
                local_features = feature_matrix[member_array, :]
                province_values = raw_cube[member_array, month_idx, canonical_idx]
                prediction = float(np.sum(local_weights * province_values))
                error = prediction - float(target_value)
                objective += float(sample_weight) * float(error**2)
                total_sample_weight += float(sample_weight)
                local_feature_mean = local_weights @ local_features if local_features.size else np.zeros((0,), dtype=np.float64)
                weighted_value_features = local_weights @ (province_values[:, None] * local_features)
                d_prediction = weighted_value_features - prediction * local_feature_mean
                gradient += 2.0 * float(sample_weight) * error * d_prediction
            normalizer = max(total_sample_weight, 1.0)
            objective = float(objective / normalizer)
            gradient = gradient / normalizer
            objective += float(cfg["aggregation_ridge"]) * float(np.dot(beta_value, beta_value))
            gradient += 2.0 * float(cfg["aggregation_ridge"]) * beta_value
            return float(objective), gradient, national_weights_local

        if national_samples or regional_samples:
            beta = np.zeros((feature_matrix.shape[1],), dtype=np.float64)
            current_objective, _, _ = _objective_and_gradient(beta)
            for _ in range(int(cfg["aggregation_learning_steps"])):
                _, gradient, _ = _objective_and_gradient(beta)
                gradient_norm = float(np.linalg.norm(gradient))
                if gradient_norm > float(cfg["aggregation_gradient_clip"]) > 0.0:
                    gradient = gradient * (float(cfg["aggregation_gradient_clip"]) / max(gradient_norm, 1e-8))
                accepted = False
                step_size = float(cfg["aggregation_learning_rate"])
                for _ in range(int(cfg["aggregation_backtracking_steps"])):
                    candidate_beta = beta - step_size * gradient
                    candidate_objective, _, _ = _objective_and_gradient(candidate_beta)
                    if candidate_objective <= current_objective:
                        beta = candidate_beta
                        current_objective = candidate_objective
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted:
                    break
            learned_objective, _, learned_weights = _objective_and_gradient(beta)
            learned_weights = (
                (1.0 - float(cfg["aggregation_equal_blend"])) * learned_weights
                + float(cfg["aggregation_equal_blend"]) / float(province_count)
            )
            learned_weights = learned_weights / np.clip(np.sum(learned_weights), 1e-8, None)
            baseline_scores = np.clip(baseline_weights, 1e-8, None)

            def _evaluate_weights(weight_vector: np.ndarray) -> float:
                total = 0.0
                total_weight = 0.0
                for canonical_idx, month_idx, target_value, sample_weight in national_samples:
                    province_values = raw_cube[:, month_idx, canonical_idx]
                    prediction = float(np.sum(weight_vector * province_values))
                    total += float(sample_weight) * float((prediction - float(target_value)) ** 2)
                    total_weight += float(sample_weight)
                for region_label, canonical_idx, month_idx, target_value, sample_weight in regional_samples:
                    member_array = np.asarray(region_members[str(region_label)], dtype=np.int32)
                    local_weights = np.clip(weight_vector[member_array], 1e-8, None)
                    local_weights = local_weights / np.clip(np.sum(local_weights), 1e-8, None)
                    province_values = raw_cube[member_array, month_idx, canonical_idx]
                    prediction = float(np.sum(local_weights * province_values))
                    total += float(sample_weight) * float((prediction - float(target_value)) ** 2)
                    total_weight += float(sample_weight)
                return float(total / max(total_weight, 1.0))

            baseline_objective = _evaluate_weights(baseline_scores)
            objective_gain = float(baseline_objective - learned_objective)
            coefficient_norm = float(np.linalg.norm(beta))
            weight_dispersion = float(np.std(learned_weights))
            diagnostics.update(
                {
                    "sample_count": int(len(national_samples) + len(regional_samples)),
                    "national_sample_count": int(len(national_samples)),
                    "regional_sample_count": int(len(regional_samples)),
                    "baseline_objective": round(float(baseline_objective), 6),
                    "learned_objective": round(float(learned_objective), 6),
                    "objective_gain": round(float(objective_gain), 6),
                    "coefficient_norm": round(float(coefficient_norm), 6),
                    "weight_dispersion": round(float(weight_dispersion), 6),
                }
            )
            if (
                objective_gain >= float(cfg["aggregation_identification_gain_floor"])
                and coefficient_norm >= float(cfg["aggregation_identification_norm_floor"])
                and weight_dispersion >= float(cfg["aggregation_identification_dispersion_floor"])
            ):
                weights = learned_weights
                weight_source = "learned_burden_softmax"
                feature_names = list(canonical_axis)
                coefficient_values = [float(value) for value in beta.tolist()]
            else:
                weights = baseline_weights
                weight_source = "unidentified_burden_prior_fallback"
                feature_names = list(canonical_axis)
                coefficient_values = [float(value) for value in beta.tolist()]
        else:
            weights = baseline_weights
            diagnostics.update(
                {
                    "sample_count": 0,
                    "national_sample_count": 0,
                    "regional_sample_count": 0,
                }
            )
    weights = np.clip(weights, 1e-8, None)
    national_weights = weights / np.clip(np.sum(weights), 1e-8, None)
    region_weight_map: dict[str, np.ndarray] = {}
    for region_label, members in region_members.items():
        region_vector = np.zeros((province_count,), dtype=np.float64)
        member_array = np.asarray([int(idx) for idx in members], dtype=np.int32)
        member_weights = np.clip(national_weights[member_array], 1e-8, None)
        member_weights = member_weights / np.clip(np.sum(member_weights), 1e-8, None)
        region_vector[member_array] = member_weights
        region_weight_map[str(region_label)] = region_vector
    return {
        "weight_source": weight_source,
        "national_weights": national_weights,
        "region_weights": region_weight_map,
        "region_codes": region_codes,
        "region_members": region_members,
        "feature_matrix": np.asarray(feature_matrix, dtype=np.float64),
        "feature_axis": list(canonical_axis),
        "feature_names": feature_names,
        "supervision_names": supervision_names,
        "support_summary": support_summary,
        "coefficient_values": coefficient_values,
        "diagnostics": diagnostics,
    }


def _standardize_state(state: np.ndarray, eps: float) -> np.ndarray:
    centered = np.asarray(state, dtype=np.float64) - float(np.mean(state))
    scale = float(np.std(centered))
    if scale <= eps:
        return centered.astype(np.float64)
    return (centered / scale).astype(np.float64)


def _initial_state_from_measurements(
    *,
    block_rows: list[dict[str, Any]],
    province_count: int,
    month_count: int,
    region_members: Mapping[str, list[int]],
    cfg: Mapping[str, Any],
) -> np.ndarray:
    cell_count = province_count * month_count
    numerator = np.zeros((cell_count,), dtype=np.float64)
    denominator = np.zeros((cell_count,), dtype=np.float64)
    for row in block_rows:
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        if cells.size == 0:
            continue
        sign = str(row.get("expected_sign") or "neutral")
        sign_multiplier = -1.0 if sign == "negative" else 1.0
        weight = float(row.get("base_weight") or 1.0) * max(0.25, float(row.get("quality_signal") or 1.0))
        cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
        contribution = sign_multiplier * float(row.get("measurement_value") or 0.0)
        numerator[cells] += weight * cell_weight * contribution
        denominator[cells] += weight * cell_weight

    state = np.zeros((province_count, month_count), dtype=np.float64)
    observed_mask = denominator.reshape(province_count, month_count) > 0.0
    state[observed_mask] = (
        numerator.reshape(province_count, month_count)[observed_mask]
        / np.clip(denominator.reshape(province_count, month_count)[observed_mask], 1e-8, None)
    )

    national_mean = float(np.mean(state[observed_mask])) if np.any(observed_mask) else 0.0
    for members in region_members.values():
        member_array = np.asarray([int(idx) for idx in members], dtype=np.int32)
        region_mask = observed_mask[member_array, :]
        region_mean = float(np.mean(state[member_array, :][region_mask])) if np.any(region_mask) else national_mean
        missing = ~region_mask
        if np.any(missing):
            region_slice = np.asarray(state[member_array, :], dtype=np.float64)
            region_slice[missing] = region_mean
            state[member_array, :] = region_slice

    for _ in range(int(cfg["aggregation_smoothing_passes"])):
        smoothed = state.copy()
        if month_count > 1:
            smoothed[:, 1:-1] = (
                (1.0 - float(cfg["temporal_smoothing_blend"])) * smoothed[:, 1:-1]
                + float(cfg["temporal_smoothing_blend"]) * (state[:, :-2] + state[:, 1:-1] + state[:, 2:]) / 3.0
            )
            smoothed[:, 0] = (state[:, 0] + state[:, min(1, month_count - 1)]) / 2.0
            smoothed[:, -1] = (state[:, -1] + state[:, max(month_count - 2, 0)]) / 2.0
        state = smoothed

    return _standardize_state(state, float(cfg["standardization_eps"]))


def _aggregate_states(
    state: np.ndarray,
    *,
    region_weights: Mapping[str, np.ndarray],
    national_weights: np.ndarray,
    region_axis: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    region_tensor = np.zeros((len(region_axis), state.shape[1]), dtype=np.float64)
    for region_idx, region_label in enumerate(region_axis):
        weights = np.asarray(region_weights[region_label], dtype=np.float64)
        region_tensor[region_idx, :] = np.sum(weights.reshape(-1, 1) * state, axis=0, dtype=np.float64)
    national_state = np.sum(np.asarray(national_weights, dtype=np.float64).reshape(-1, 1) * state, axis=0, dtype=np.float64)
    return region_tensor, national_state


def _row_is_local_support(row: Mapping[str, Any]) -> bool:
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    return geo_resolution in {"province", "city", "municipality"}


def _row_is_aggregate_support(row: Mapping[str, Any]) -> bool:
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    return geo_resolution in {"region", "national"}


def _build_local_support_precision_grid(
    *,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    row_precisions: np.ndarray,
    province_count: int,
    month_count: int,
) -> np.ndarray:
    support_precision = np.zeros((province_count, month_count), dtype=np.float64)
    flat_precision = support_precision.reshape(-1)
    for row_idx, row in enumerate(block_rows):
        if not _row_is_local_support(row):
            continue
        params = indicator_params.get(str(row.get("canonical_name") or ""))
        if not params:
            continue
        cells = np.asarray(row.get("support_cells"), dtype=np.int32)
        if cells.size == 0:
            continue
        cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
        observation_weight = float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx])
        local_precision = observation_weight * float(params["lambda"] ** 2) * float(cell_weight**2)
        flat_precision[cells] += local_precision
    return support_precision


def _build_donor_kernel(
    *,
    feature_matrix: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return {
            "available": False,
            "reason": "missing_feature_matrix",
            "laplacian": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "transition": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "entropy": np.zeros((matrix.shape[0],), dtype=np.float64),
            "effective_donor_count": np.zeros((matrix.shape[0],), dtype=np.float64),
        }
    col_var = np.var(matrix, axis=0, dtype=np.float64)
    valid_cols = col_var > float(cfg["variance_eps"])
    if not np.any(valid_cols):
        return {
            "available": False,
            "reason": "degenerate_feature_matrix",
            "laplacian": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "transition": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "entropy": np.zeros((matrix.shape[0],), dtype=np.float64),
            "effective_donor_count": np.zeros((matrix.shape[0],), dtype=np.float64),
        }
    features = np.asarray(matrix[:, valid_cols], dtype=np.float64)
    covariance = np.cov(features, rowvar=False, dtype=np.float64)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64) + float(cfg["variance_eps"]) * np.eye(features.shape[1], dtype=np.float64)
    inverse_covariance = np.linalg.pinv(covariance)
    delta = features[:, None, :] - features[None, :, :]
    distance = np.einsum("...i,ij,...j->...", delta, inverse_covariance, delta, optimize=True)
    affinity = np.exp(-0.5 * np.clip(distance, 0.0, None))
    np.fill_diagonal(affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)
    degree = np.sum(affinity, axis=1, dtype=np.float64)
    if not np.any(degree > float(cfg["variance_eps"])):
        return {
            "available": False,
            "reason": "zero_affinity",
            "laplacian": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "transition": np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64),
            "entropy": np.zeros((matrix.shape[0],), dtype=np.float64),
            "effective_donor_count": np.zeros((matrix.shape[0],), dtype=np.float64),
        }
    transition = np.zeros_like(affinity, dtype=np.float64)
    valid_rows = degree > float(cfg["variance_eps"])
    transition[valid_rows, :] = affinity[valid_rows, :] / degree[valid_rows, None]
    safe_transition = np.clip(transition, float(cfg["variance_eps"]), None)
    safe_transition[~valid_rows, :] = 1.0
    entropy = -np.sum(transition * np.log(safe_transition), axis=1, dtype=np.float64)
    effective_donor_count = np.exp(entropy)
    laplacian = np.diag(degree) - affinity
    return {
        "available": True,
        "reason": "empirical_mahalanobis_kernel",
        "laplacian": laplacian.astype(np.float64),
        "transition": transition.astype(np.float64),
        "entropy": entropy.astype(np.float64),
        "effective_donor_count": effective_donor_count.astype(np.float64),
    }


def _build_measurement_fit_rows(
    *,
    block_id: str,
    block_display_name: str,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    row_precisions: np.ndarray,
    state: np.ndarray,
    posterior_std: np.ndarray,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    measurement_fit_rows: list[dict[str, Any]] = []
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    posterior_std_vector = np.asarray(posterior_std, dtype=np.float64).reshape(-1)
    for row_idx, row in enumerate(block_rows):
        params = indicator_params[str(row.get("canonical_name") or "")]
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
        predictor = cell_weight * float(np.sum(state_vector[cells]))
        fitted = float(params["alpha"]) + float(params["lambda"]) * predictor
        residual = float(row.get("measurement_value") or 0.0) - fitted
        latent_variance = float(
            np.sum(np.square(np.asarray(posterior_std_vector[cells], dtype=np.float64)) * float(cell_weight**2))
        )
        observation_variance = 1.0 / max(float(row_precisions[row_idx]), float(cfg["variance_eps"]))
        predictive_std = float(
            np.sqrt(max(observation_variance + float(params["lambda"] ** 2) * latent_variance, float(cfg["variance_eps"])))
        )
        standardized_residual = float(residual / max(predictive_std, float(cfg["variance_eps"])))
        measurement_fit_rows.append(
            {
                "block_id": block_id,
                "display_name": block_display_name,
                "canonical_name": str(row.get("canonical_name") or ""),
                "measurement_role": str(row.get("measurement_role") or ""),
                "source_bank": str(row.get("source_bank") or ""),
                "geo_resolution": str(row.get("geo_resolution") or ""),
                "time_resolution": str(row.get("time_resolution") or ""),
                "anchor_status": str(row.get("anchor_status") or ""),
                "support_cell_count": int(np.asarray(row.get("support_cells"), dtype=np.int32).size),
                "observed_value": round(float(row.get("measurement_value") or 0.0), 6),
                "fitted_value": round(float(fitted), 6),
                "residual": round(float(residual), 6),
                "precision": round(float(row_precisions[row_idx]), 6),
                "latent_support_std": round(float(np.sqrt(max(latent_variance, 0.0))), 6),
                "predictive_std": round(float(predictive_std), 6),
                "standardized_residual": round(float(standardized_residual), 6),
            }
        )
    return measurement_fit_rows


def _missing_information_backend_runtime(cfg: Mapping[str, Any]) -> dict[str, Any]:
    requested = str(cfg.get("missing_information_backend") or "auto").strip().lower()
    if requested not in {"auto", "cpu", "torch_cuda"}:
        requested = "auto"
    torch_cuda_available = bool(torch is not None and torch.cuda.is_available())
    if requested == "torch_cuda":
        return {
            "backend": "torch_cuda" if torch_cuda_available else "cpu_sparse",
            "torch_cuda_available": torch_cuda_available,
            "requested_backend": "torch_cuda",
        }
    if requested == "cpu":
        return {
            "backend": "cpu_sparse",
            "torch_cuda_available": torch_cuda_available,
            "requested_backend": "cpu",
        }
    return {
        "backend": "torch_cuda" if torch_cuda_available else "cpu_sparse",
        "torch_cuda_available": torch_cuda_available,
        "requested_backend": "auto",
    }


def _torch_missing_information_dtype(cfg: Mapping[str, Any]) -> Any:
    if torch is None:
        return None
    dtype_name = str(cfg.get("missing_information_torch_dtype") or "float32").strip().lower()
    if dtype_name == "float64":
        return torch.float64
    return torch.float32


def _run_torch_missing_information_cg(
    *,
    province_count: int,
    month_count: int,
    base_diagonal: np.ndarray,
    forcing: np.ndarray,
    temporal_precision: float,
    phi: float,
    donor_laplacian: np.ndarray,
    donor_precision: float,
    constraint_indices: list[np.ndarray],
    constraint_coefficients: list[float],
    constraint_precisions: list[float],
    cfg: Mapping[str, Any],
    rtol_override: float | None = None,
    atol_override: float | None = None,
) -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("torch CUDA backend requested but unavailable")
    device = torch.device("cuda")
    dtype = _torch_missing_information_dtype(cfg)
    cell_count = int(province_count * month_count)
    base_diag_t = torch.as_tensor(np.asarray(base_diagonal, dtype=np.float64), dtype=dtype, device=device)
    forcing_t = torch.as_tensor(np.asarray(forcing, dtype=np.float64), dtype=dtype, device=device)
    temporal_precision_t = torch.as_tensor(float(temporal_precision), dtype=dtype, device=device)
    phi_t = torch.as_tensor(float(phi), dtype=dtype, device=device)
    donor_precision_t = torch.as_tensor(float(donor_precision), dtype=dtype, device=device)
    donor_laplacian_t = (
        torch.as_tensor(np.asarray(donor_laplacian, dtype=np.float64), dtype=dtype, device=device)
        if donor_laplacian.size
        else torch.zeros((0, 0), dtype=dtype, device=device)
    )
    row_index_parts: list[np.ndarray] = []
    col_index_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for row_idx, cells in enumerate(constraint_indices):
        if int(cells.size) == 0:
            continue
        row_index_parts.append(np.full((int(cells.size),), int(row_idx), dtype=np.int64))
        col_index_parts.append(np.asarray(cells, dtype=np.int64))
        value_parts.append(
            np.full((int(cells.size),), float(constraint_coefficients[row_idx]), dtype=np.float64)
        )
    if row_index_parts:
        row_indices = np.concatenate(row_index_parts)
        col_indices = np.concatenate(col_index_parts)
        values = np.concatenate(value_parts)
        constraint_indices_t = torch.as_tensor(
            np.vstack([row_indices, col_indices]),
            dtype=torch.int64,
            device=device,
        )
        constraint_values_t = torch.as_tensor(values, dtype=dtype, device=device)
        constraint_matrix_t = torch.sparse_coo_tensor(
            constraint_indices_t,
            constraint_values_t,
            (len(constraint_indices), cell_count),
            dtype=dtype,
            device=device,
        ).coalesce()
        constraint_precision_t = torch.as_tensor(
            np.asarray(constraint_precisions, dtype=np.float64),
            dtype=dtype,
            device=device,
        )
    else:
        constraint_matrix_t = None
        constraint_precision_t = torch.zeros((0,), dtype=dtype, device=device)

    def _matvec_impl(vector: Any) -> Any:
        x = vector.reshape(province_count, month_count)
        result = base_diag_t.reshape(province_count, month_count) * x
        if month_count > 1 and float(temporal_precision) > 0.0:
            temporal_residual = x[:, 1:] - phi_t * x[:, :-1]
            result = result.clone()
            result[:, 1:] = result[:, 1:] + temporal_precision_t * temporal_residual
            result[:, :-1] = result[:, :-1] - temporal_precision_t * phi_t * temporal_residual
        if donor_laplacian_t.numel() and float(donor_precision) > 0.0:
            result = result + donor_precision_t * (donor_laplacian_t @ x)
        flat_result = result.reshape(-1)
        if constraint_matrix_t is not None:
            projection = torch.sparse.mm(constraint_matrix_t, vector.reshape(-1, 1))
            weighted_projection = constraint_precision_t.reshape(-1, 1) * projection
            flat_result = flat_result + torch.sparse.mm(constraint_matrix_t.transpose(0, 1), weighted_projection).reshape(-1)
        return flat_result

    compiled_matvec = _matvec_impl
    compile_used = False
    compile_enabled = bool(cfg.get("missing_information_torch_compile", True)) and hasattr(torch, "compile")
    if compile_enabled:
        try:
            compiled_matvec = torch.compile(_matvec_impl, mode="reduce-overhead", fullgraph=False)
        except Exception:
            compiled_matvec = _matvec_impl
    if compiled_matvec is not _matvec_impl:
        try:
            _ = compiled_matvec(torch.zeros_like(forcing_t))
            compile_used = True
        except Exception:
            compiled_matvec = _matvec_impl
            compile_used = False

    max_iter = max(8, int(cfg.get("missing_information_torch_cg_max_iter") or 256))
    rtol = float(rtol_override) if rtol_override is not None else float(cfg.get("missing_information_torch_cg_rtol") or 1e-5)
    atol = float(atol_override) if atol_override is not None else float(cfg.get("missing_information_torch_cg_atol") or 1e-7)
    x = torch.zeros_like(forcing_t)
    r = forcing_t - compiled_matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    b_norm = torch.linalg.norm(forcing_t)
    tolerance = torch.maximum(
        torch.as_tensor(atol, dtype=dtype, device=device),
        torch.as_tensor(rtol, dtype=dtype, device=device) * torch.maximum(b_norm, torch.as_tensor(1.0, dtype=dtype, device=device)),
    )
    iteration_count = 0
    residual_norm = torch.sqrt(torch.clamp(rs_old, min=0.0))
    for iteration_idx in range(max_iter):
        if bool((residual_norm <= tolerance).item()):
            break
        ap = compiled_matvec(p)
        denom = torch.dot(p, ap)
        alpha = rs_old / torch.clamp(denom, min=torch.as_tensor(1e-12, dtype=dtype, device=device))
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        residual_norm = torch.sqrt(torch.clamp(rs_new, min=0.0))
        iteration_count = iteration_idx + 1
        if bool((residual_norm <= tolerance).item()):
            rs_old = rs_new
            break
        beta = rs_new / torch.clamp(rs_old, min=torch.as_tensor(1e-12, dtype=dtype, device=device))
        p = r + beta * p
        rs_old = rs_new
    converged = bool((residual_norm <= tolerance).item())
    return {
        "correction_vector": x.detach().cpu().numpy().astype(np.float64),
        "iteration_count": int(iteration_count),
        "residual_norm": float(residual_norm.detach().cpu().item()),
        "tolerance": float(tolerance.detach().cpu().item()),
        "converged": bool(converged),
        "max_iter_reached": bool(iteration_count >= max_iter and not converged),
        "max_iter": int(max_iter),
        "device": str(device),
        "compiled": bool(compile_used),
    }


def _build_missing_information_linear_system(
    *,
    block_id: str,
    block_display_name: str,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    row_precisions: np.ndarray,
    state: np.ndarray,
    posterior_std: np.ndarray,
    province_axis: list[str],
    donor_kernel: Mapping[str, Any],
    temporal_precision: float,
    phi: float,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    province_count, month_count = state.shape
    cell_count = province_count * month_count
    local_support_precision = _build_local_support_precision_grid(
        block_rows=block_rows,
        indicator_params=indicator_params,
        row_precisions=row_precisions,
        province_count=province_count,
        month_count=month_count,
    )
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    reconciliation_rows: list[dict[str, Any]] = []
    latent_discrepancies: list[float] = []
    for row_idx, row in enumerate(block_rows):
        if not _row_is_aggregate_support(row):
            continue
        params = indicator_params.get(str(row.get("canonical_name") or ""))
        if not params:
            continue
        lambda_value = float(params["lambda"])
        if abs(lambda_value) <= float(cfg["variance_eps"]):
            continue
        cells = np.asarray(row.get("support_cells"), dtype=np.int32)
        if cells.size == 0:
            continue
        cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
        predictor = cell_weight * float(np.sum(state_vector[cells]))
        fitted = float(params["alpha"]) + lambda_value * predictor
        residual = float(row.get("measurement_value") or 0.0) - fitted
        latent_discrepancy = float(residual / lambda_value)
        constraint_precision = float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx]) * float(lambda_value**2)
        reconciliation_rows.append(
            {
                "canonical_name": str(row.get("canonical_name") or ""),
                "geo_resolution": str(row.get("geo_resolution") or ""),
                "time_resolution": str(row.get("time_resolution") or ""),
                "measurement_role": str(row.get("measurement_role") or ""),
                "support_cell_count": int(cells.size),
                "support_cells": np.asarray(cells, dtype=np.int32),
                "cell_weight": float(cell_weight),
                "constraint_precision": float(constraint_precision),
                "residual_before": float(residual),
                "latent_discrepancy": float(latent_discrepancy),
            }
        )
        latent_discrepancies.append(float(latent_discrepancy))
    if not reconciliation_rows:
        return {
            "available": False,
            "reason": "missing_aggregate_constraints",
            "block_id": block_id,
            "block_display_name": block_display_name,
            "state": np.asarray(state, dtype=np.float64),
            "posterior_std": np.asarray(posterior_std, dtype=np.float64),
            "province_axis": list(province_axis),
            "summary": {"aggregate_constraint_count": 0},
        }

    correction_prior_precision = _posterior_mean_precision(
        np.asarray(latent_discrepancies, dtype=np.float64),
        prior_dof=float(cfg["missing_information_correction_prior_dof"]),
        prior_variance=float(cfg["missing_information_correction_prior_variance"]),
        floor=float(cfg["missing_information_correction_precision_floor"]),
        ceiling=float(cfg["missing_information_correction_precision_ceiling"]),
    )
    donor_precision = 0.0
    if bool(donor_kernel.get("available")):
        transition = np.asarray(donor_kernel.get("transition"), dtype=np.float64)
        donor_residual = np.asarray(state, dtype=np.float64) - transition @ np.asarray(state, dtype=np.float64)
        donor_precision = _posterior_mean_precision(
            donor_residual.reshape(-1),
            prior_dof=float(cfg["missing_information_donor_precision_prior_dof"]),
            prior_variance=float(cfg["missing_information_donor_precision_prior_variance"]),
            floor=float(cfg["missing_information_donor_precision_floor"]),
            ceiling=float(cfg["missing_information_donor_precision_ceiling"]),
        )

    base_diagonal = np.full((cell_count,), float(correction_prior_precision), dtype=np.float64)
    base_diagonal += np.asarray(local_support_precision, dtype=np.float64).reshape(-1)
    forcing = np.zeros((cell_count,), dtype=np.float64)
    temporal_diag = np.zeros((cell_count,), dtype=np.float64)
    if month_count > 1:
        for province_idx in range(province_count):
            for month_idx in range(1, month_count):
                prev_idx = province_idx * month_count + (month_idx - 1)
                curr_idx = province_idx * month_count + month_idx
                temporal_diag[curr_idx] += float(temporal_precision)
                temporal_diag[prev_idx] += float(temporal_precision) * float(phi**2)
    donor_laplacian = (
        np.asarray(donor_kernel.get("laplacian"), dtype=np.float64)
        if bool(donor_kernel.get("available"))
        else np.zeros((0, 0), dtype=np.float64)
    )
    donor_diag = np.zeros((cell_count,), dtype=np.float64)
    if bool(donor_kernel.get("available")) and donor_precision > 0.0:
        donor_diagonal = np.diag(donor_laplacian) if donor_laplacian.size else np.zeros((0,), dtype=np.float64)
        for month_idx in range(month_count):
            cell_indices = np.asarray(
                [province_idx * month_count + month_idx for province_idx in range(province_count)],
                dtype=np.int32,
            )
            donor_diag[cell_indices] += float(donor_precision) * donor_diagonal
    month_cell_indices = [
        np.asarray([province_idx * month_count + month_idx for province_idx in range(province_count)], dtype=np.int32)
        for month_idx in range(month_count)
    ]
    constraint_terms = [
        (
            np.asarray(row["support_cells"], dtype=np.int32),
            float(row["cell_weight"]),
            float(row["constraint_precision"]),
            float(row["latent_discrepancy"]),
        )
        for row in reconciliation_rows
    ]
    aggregate_diag = np.zeros((cell_count,), dtype=np.float64)
    for cells, coefficient, constraint_precision, latent_discrepancy in constraint_terms:
        aggregate_diag[cells] += constraint_precision * float(coefficient**2)
        forcing[cells] += constraint_precision * latent_discrepancy * coefficient
    diagonal_precision = base_diagonal + temporal_diag + donor_diag + aggregate_diag
    return {
        "available": True,
        "reason": "gaussian_reconciliation_posterior",
        "block_id": block_id,
        "block_display_name": block_display_name,
        "state": np.asarray(state, dtype=np.float64),
        "posterior_std": np.asarray(posterior_std, dtype=np.float64),
        "province_axis": list(province_axis),
        "province_count": int(province_count),
        "month_count": int(month_count),
        "cell_count": int(cell_count),
        "backend_runtime": _mi_backend_runtime(cfg),
        "indicator_params": dict(indicator_params),
        "local_support_precision": np.asarray(local_support_precision, dtype=np.float64),
        "reconciliation_rows": reconciliation_rows,
        "correction_prior_precision": float(correction_prior_precision),
        "donor_precision": float(donor_precision),
        "base_diagonal": np.asarray(base_diagonal, dtype=np.float64),
        "forcing": np.asarray(forcing, dtype=np.float64),
        "temporal_precision": float(temporal_precision),
        "phi": float(phi),
        "donor_laplacian": np.asarray(donor_laplacian, dtype=np.float64),
        "donor_kernel": donor_kernel,
        "constraint_terms": constraint_terms,
        "month_cell_indices": month_cell_indices,
        "diagonal_precision": np.asarray(diagonal_precision, dtype=np.float64),
    }


def _apply_missing_information_precision_operator_cpu(
    system: Mapping[str, Any],
    vector: np.ndarray,
) -> np.ndarray:
    cell_count = int(system["cell_count"])
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    temporal_precision = float(system["temporal_precision"])
    phi = float(system["phi"])
    donor_precision = float(system["donor_precision"])
    donor_laplacian = np.asarray(system["donor_laplacian"], dtype=np.float64)
    x = np.asarray(vector, dtype=np.float64).reshape(cell_count)
    result = np.asarray(system["base_diagonal"], dtype=np.float64) * x
    if month_count > 1 and abs(temporal_precision) > 0.0:
        for province_idx in range(province_count):
            start = province_idx * month_count
            stop = start + month_count
            segment = x[start:stop]
            temporal_residual = segment[1:] - phi * segment[:-1]
            result[start + 1 : stop] += temporal_precision * temporal_residual
            result[start : stop - 1] += -temporal_precision * phi * temporal_residual
    if donor_laplacian.size and donor_precision > 0.0:
        for cell_indices in list(system["month_cell_indices"]):
            result[cell_indices] += donor_precision * (donor_laplacian @ x[cell_indices])
    for cells, coefficient, constraint_precision, _latent_discrepancy in list(system["constraint_terms"]):
        projection = coefficient * float(np.sum(x[cells]))
        result[cells] += constraint_precision * coefficient * projection
    return result


def _solve_missing_information_linear_system(
    *,
    system: Mapping[str, Any],
    cfg: Mapping[str, Any],
    backend: str | None = None,
    rtol_override: float | None = None,
    atol_override: float | None = None,
) -> dict[str, Any]:
    selected_backend = str(backend or dict(system.get("backend_runtime") or {}).get("backend") or "cpu_sparse")
    if selected_backend == "torch_cuda":
        torch_solution = _run_torch_missing_information_cg(
            province_count=int(system["province_count"]),
            month_count=int(system["month_count"]),
            base_diagonal=np.asarray(system["base_diagonal"], dtype=np.float64),
            forcing=np.asarray(system["forcing"], dtype=np.float64),
            temporal_precision=float(system["temporal_precision"]),
            phi=float(system["phi"]),
            donor_laplacian=np.asarray(system["donor_laplacian"], dtype=np.float64),
            donor_precision=float(system["donor_precision"]),
            constraint_indices=[cells for cells, _coefficient, _constraint_precision, _latent_discrepancy in list(system["constraint_terms"])],
            constraint_coefficients=[coefficient for _cells, coefficient, _constraint_precision, _latent_discrepancy in list(system["constraint_terms"])],
            constraint_precisions=[constraint_precision for _cells, _coefficient, constraint_precision, _latent_discrepancy in list(system["constraint_terms"])],
            cfg=cfg,
            rtol_override=rtol_override,
            atol_override=atol_override,
        )
        return {
            "correction_vector": np.asarray(torch_solution["correction_vector"], dtype=np.float64),
            "solver_diagnostics": {
                "backend": "torch_cuda",
                "device": str(torch_solution["device"]),
                "compiled": bool(torch_solution["compiled"]),
                "cg_iteration_count": int(torch_solution["iteration_count"]),
                "cg_residual_norm": round(float(torch_solution["residual_norm"]), 6),
                "cg_tolerance": round(float(torch_solution["tolerance"]), 12),
                "cg_converged": bool(torch_solution["converged"]),
                "cg_max_iter_reached": bool(torch_solution["max_iter_reached"]),
                "cg_max_iter": int(torch_solution["max_iter"]),
                "rtol": float(rtol_override) if rtol_override is not None else float(cfg.get("missing_information_torch_cg_rtol") or 1e-5),
                "atol": float(atol_override) if atol_override is not None else float(cfg.get("missing_information_torch_cg_atol") or 1e-7),
            },
        }

    operator = LinearOperator(
        shape=(int(system["cell_count"]), int(system["cell_count"])),
        matvec=lambda vec: _apply_missing_information_precision_operator_cpu(system, vec),
        dtype=np.float64,
    )
    iteration_counter = {"count": 0}

    def _callback(_xk: np.ndarray) -> None:
        iteration_counter["count"] += 1

    rtol = float(rtol_override) if rtol_override is not None else 1e-6
    atol = float(atol_override) if atol_override is not None else 0.0
    correction_vector, info = cg(
        operator,
        np.asarray(system["forcing"], dtype=np.float64),
        x0=np.zeros((int(system["cell_count"]),), dtype=np.float64),
        rtol=rtol,
        atol=atol,
        callback=_callback,
    )
    if info != 0:
        raise RuntimeError(f"Phase15 missing-information linear solve failed for {system['block_id']} with info={info}")
    residual = np.asarray(system["forcing"], dtype=np.float64) - _apply_missing_information_precision_operator_cpu(
        system,
        np.asarray(correction_vector, dtype=np.float64),
    )
    return {
        "correction_vector": np.asarray(correction_vector, dtype=np.float64),
        "solver_diagnostics": {
            "backend": "cpu_sparse",
            "device": "cpu",
            "compiled": False,
            "cg_iteration_count": int(iteration_counter["count"]),
            "cg_residual_norm": round(float(np.linalg.norm(residual)), 6),
            "cg_tolerance": round(float(max(atol, rtol * max(float(np.linalg.norm(np.asarray(system["forcing"], dtype=np.float64))), 1.0))), 12),
            "cg_converged": True,
            "cg_max_iter_reached": False,
            "cg_max_iter": 0,
            "rtol": float(rtol),
            "atol": float(atol),
        },
    }


def _materialize_missing_information_solution(
    *,
    system: Mapping[str, Any],
    correction_vector: np.ndarray,
    solver_diagnostics: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    block_id = str(system["block_id"])
    block_display_name = str(system["block_display_name"])
    state = np.asarray(system["state"], dtype=np.float64)
    posterior_std = np.asarray(system["posterior_std"], dtype=np.float64)
    diagonal_precision = np.asarray(system["diagonal_precision"], dtype=np.float64)
    local_support_precision = np.asarray(system["local_support_precision"], dtype=np.float64)
    donor_kernel = dict(system.get("donor_kernel") or {})
    indicator_params = dict(system.get("indicator_params") or {})
    reconciliation_rows = list(system.get("reconciliation_rows") or [])
    correction = np.asarray(correction_vector, dtype=np.float64).reshape(province_count, month_count)
    correction_std = np.sqrt(1.0 / np.clip(diagonal_precision, float(cfg["variance_eps"]), None)).reshape(province_count, month_count)
    corrected_state = state + correction
    corrected_posterior_std = np.sqrt(np.square(posterior_std) + np.square(correction_std))
    corrected_state_vector = corrected_state.reshape(-1)
    aggregate_rows: list[dict[str, Any]] = []
    residual_before_values: list[float] = []
    residual_after_values: list[float] = []
    residual_after_vector: list[float] = []
    for row in reconciliation_rows:
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        params = indicator_params[str(row["canonical_name"])]
        predictor_after = float(row["cell_weight"]) * float(np.sum(corrected_state_vector[cells]))
        fitted_after = float(params["alpha"]) + float(params["lambda"]) * predictor_after
        residual_after = float(row["residual_before"]) - float(params["lambda"]) * float(row["cell_weight"]) * float(np.sum(correction.reshape(-1)[cells]))
        aggregate_rows.append(
            {
                "block_id": block_id,
                "display_name": block_display_name,
                "canonical_name": str(row["canonical_name"]),
                "geo_resolution": str(row["geo_resolution"]),
                "time_resolution": str(row["time_resolution"]),
                "measurement_role": str(row["measurement_role"]),
                "support_cell_count": int(row["support_cell_count"]),
                "constraint_precision": round(float(row["constraint_precision"]), 6),
                "latent_discrepancy": round(float(row["latent_discrepancy"]), 6),
                "residual_before": round(float(row["residual_before"]), 6),
                "residual_after": round(float(residual_after), 6),
                "fitted_after": round(float(fitted_after), 6),
            }
        )
        residual_before_values.append(abs(float(row["residual_before"])))
        residual_after_values.append(abs(float(residual_after)))
        residual_after_vector.append(float(residual_after))
    imputation_share = np.abs(correction) / np.maximum(np.abs(corrected_state), float(cfg["variance_eps"]))
    donor_entropy = np.asarray(donor_kernel.get("entropy"), dtype=np.float64)
    donor_effective_count = np.asarray(donor_kernel.get("effective_donor_count"), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for province_idx in range(province_count):
        rows.append(
            {
                "block_id": block_id,
                "display_name": block_display_name,
                "province_index": int(province_idx),
                "correction_values": [round(float(value), 6) for value in correction[province_idx, :].tolist()],
                "correction_std_values": [round(float(value), 6) for value in correction_std[province_idx, :].tolist()],
                "imputation_share_values": [round(float(value), 6) for value in imputation_share[province_idx, :].tolist()],
                "local_support_precision_values": [
                    round(float(value), 6) for value in local_support_precision[province_idx, :].tolist()
                ],
                "donor_entropy": round(float(donor_entropy[province_idx]) if province_idx < donor_entropy.size else 0.0, 6),
                "effective_donor_count": round(
                    float(donor_effective_count[province_idx]) if province_idx < donor_effective_count.size else 0.0,
                    6,
                ),
            }
        )
    return {
        "available": True,
        "reason": str(system.get("reason") or "gaussian_reconciliation_posterior"),
        "state": corrected_state.astype(np.float64),
        "posterior_std": corrected_posterior_std.astype(np.float64),
        "correction": correction.astype(np.float64),
        "correction_std": correction_std.astype(np.float64),
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "aggregate_residual_after_values": np.asarray(residual_after_vector, dtype=np.float64),
        "summary": {
            "aggregate_constraint_count": len(aggregate_rows),
            "correction_prior_precision": round(float(system["correction_prior_precision"]), 6),
            "donor_precision": round(float(system["donor_precision"]), 6),
            "mean_abs_residual_before": round(float(np.mean(residual_before_values)) if residual_before_values else 0.0, 6),
            "mean_abs_residual_after": round(float(np.mean(residual_after_values)) if residual_after_values else 0.0, 6),
            "correction_l2": round(float(np.linalg.norm(correction)), 6),
            "donor_available": bool(donor_kernel.get("available")),
            "solver": dict(solver_diagnostics),
        },
    }


def _missing_information_solution_drift(
    anchor_solution: Mapping[str, Any],
    candidate_solution: Mapping[str, Any],
) -> dict[str, float]:
    anchor_state = np.asarray(anchor_solution.get("state"), dtype=np.float64)
    candidate_state = np.asarray(candidate_solution.get("state"), dtype=np.float64)
    anchor_std = np.asarray(anchor_solution.get("posterior_std"), dtype=np.float64)
    candidate_std = np.asarray(candidate_solution.get("posterior_std"), dtype=np.float64)
    anchor_residuals = np.asarray(anchor_solution.get("aggregate_residual_after_values"), dtype=np.float64)
    candidate_residuals = np.asarray(candidate_solution.get("aggregate_residual_after_values"), dtype=np.float64)
    state_delta = candidate_state - anchor_state
    std_delta = candidate_std - anchor_std
    residual_delta = (
        candidate_residuals - anchor_residuals
        if anchor_residuals.size == candidate_residuals.size
        else np.zeros((0,), dtype=np.float64)
    )
    return {
        "state_mae": round(float(np.mean(np.abs(state_delta))) if state_delta.size else 0.0, 8),
        "state_max_abs": round(float(np.max(np.abs(state_delta))) if state_delta.size else 0.0, 8),
        "posterior_std_mae": round(float(np.mean(np.abs(std_delta))) if std_delta.size else 0.0, 8),
        "posterior_std_max_abs": round(float(np.max(np.abs(std_delta))) if std_delta.size else 0.0, 8),
        "aggregate_residual_mae": round(float(np.mean(np.abs(residual_delta))) if residual_delta.size else 0.0, 8),
        "aggregate_residual_max_abs": round(float(np.max(np.abs(residual_delta))) if residual_delta.size else 0.0, 8),
    }


def _select_missing_information_reference_subset_provinces(
    *,
    system: Mapping[str, Any],
    max_provinces: int,
) -> np.ndarray:
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    if province_count <= 0:
        return np.zeros((0,), dtype=np.int32)
    support_score = np.sum(np.asarray(system["local_support_precision"], dtype=np.float64), axis=1)
    for row in list(system.get("reconciliation_rows") or []):
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        if cells.size == 0:
            continue
        province_indices = np.unique(cells // month_count)
        support_score[province_indices] += float(row["constraint_precision"])
    ordering = np.lexsort((np.arange(province_count, dtype=np.int32), -support_score))
    subset_count = int(max(1, min(int(max_provinces), province_count)))
    return np.asarray(np.sort(ordering[:subset_count]), dtype=np.int32)


def _restrict_missing_information_system_to_provinces(
    *,
    system: Mapping[str, Any],
    province_indices: np.ndarray,
) -> dict[str, Any]:
    province_subset = np.asarray(sorted({int(idx) for idx in np.asarray(province_indices, dtype=np.int32).tolist()}), dtype=np.int32)
    month_count = int(system["month_count"])
    selected_cells = np.asarray(
        [int(province_idx) * month_count + month_idx for province_idx in province_subset for month_idx in range(month_count)],
        dtype=np.int32,
    )
    cell_map = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(selected_cells.tolist())}
    restricted_rows: list[dict[str, Any]] = []
    restricted_constraint_terms: list[tuple[np.ndarray, float, float, float]] = []
    for row in list(system.get("reconciliation_rows") or []):
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        restricted_cells = np.asarray([cell_map[int(cell)] for cell in cells.tolist() if int(cell) in cell_map], dtype=np.int32)
        if restricted_cells.size == 0:
            continue
        restricted_row = dict(row)
        restricted_row["support_cells"] = restricted_cells
        restricted_row["support_cell_count"] = int(restricted_cells.size)
        restricted_rows.append(restricted_row)
        restricted_constraint_terms.append(
            (
                restricted_cells,
                float(row["cell_weight"]),
                float(row["constraint_precision"]),
                float(row["latent_discrepancy"]),
            )
        )
    return {
        **dict(system),
        "province_axis": [str(system["province_axis"][int(idx)]) for idx in province_subset.tolist()],
        "province_count": int(province_subset.size),
        "cell_count": int(selected_cells.size),
        "state": np.asarray(system["state"], dtype=np.float64)[province_subset, :],
        "posterior_std": np.asarray(system["posterior_std"], dtype=np.float64)[province_subset, :],
        "local_support_precision": np.asarray(system["local_support_precision"], dtype=np.float64)[province_subset, :],
        "base_diagonal": np.asarray(system["base_diagonal"], dtype=np.float64)[selected_cells],
        "forcing": np.asarray(system["forcing"], dtype=np.float64)[selected_cells],
        "donor_laplacian": (
            np.asarray(system["donor_laplacian"], dtype=np.float64)[np.ix_(province_subset, province_subset)]
            if np.asarray(system["donor_laplacian"], dtype=np.float64).size
            else np.zeros((0, 0), dtype=np.float64)
        ),
        "reconciliation_rows": restricted_rows,
        "constraint_terms": restricted_constraint_terms,
        "month_cell_indices": [
            np.asarray([local_idx * month_count + month_idx for local_idx in range(int(province_subset.size))], dtype=np.int32)
            for month_idx in range(month_count)
        ],
        "diagonal_precision": np.asarray(system["diagonal_precision"], dtype=np.float64)[selected_cells],
        "constraint_matrix_cpu": None,
        "constraint_precision_vector": None,
    }


def _dense_solve_missing_information_reference(system: Mapping[str, Any]) -> dict[str, Any]:
    cell_count = int(system["cell_count"])
    basis = np.eye(cell_count, dtype=np.float64)
    dense_matrix = np.column_stack(
        [
            _mi_apply_precision_operator_cpu(system, basis[:, column_idx])
            for column_idx in range(cell_count)
        ]
    )
    forcing = np.asarray(system["forcing"], dtype=np.float64)
    correction_vector = np.linalg.solve(dense_matrix, forcing)
    residual = forcing - dense_matrix @ correction_vector
    return {
        "correction_vector": np.asarray(correction_vector, dtype=np.float64),
        "condition_number": float(np.linalg.cond(dense_matrix)),
        "residual_norm": float(np.linalg.norm(residual)),
    }


def _build_missing_information_numerical_adequacy_report(
    *,
    system: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(cfg.get("numerical_adequacy_enabled", True)):
        return {"available": False, "reason": "disabled", "tolerance_rows": [], "direct_reference_rows": [], "summary": {}}
    if not bool(system.get("available")):
        return {
            "available": False,
            "reason": str(system.get("reason") or "unavailable"),
            "tolerance_rows": [],
            "direct_reference_rows": [],
            "summary": {},
        }

    tolerance_values = sorted(
        {
            max(float(value), 1e-12)
            for value in list(cfg.get("numerical_adequacy_tolerance_rtol_values") or [1e-4, 1e-5, 1e-6])
        }
    )
    default_rtol = float(cfg.get("missing_information_torch_cg_rtol") or 1e-5)
    tolerance_values = sorted({*tolerance_values, default_rtol})
    anchor_tolerance = float(min(tolerance_values))
    tolerance_solutions: dict[float, dict[str, Any]] = {}
    tolerance_rows: list[dict[str, Any]] = []
    for tolerance in tolerance_values:
        solve_payload = _mi_solve_linear_system(
            system=system,
            cfg=cfg,
            backend=str(dict(system.get("backend_runtime") or {}).get("backend") or "cpu_sparse"),
            rtol_override=float(tolerance),
            atol_override=float(cfg.get("numerical_adequacy_tolerance_atol") or 1e-7),
        )
        tolerance_solutions[float(tolerance)] = _materialize_missing_information_solution(
            system=system,
            correction_vector=np.asarray(solve_payload["correction_vector"], dtype=np.float64),
            solver_diagnostics=dict(solve_payload["solver_diagnostics"]),
            cfg=cfg,
        )
    anchor_solution = dict(tolerance_solutions[anchor_tolerance])
    for tolerance in tolerance_values:
        solve_solution = dict(tolerance_solutions[float(tolerance)])
        drift = _missing_information_solution_drift(anchor_solution, solve_solution)
        solver = dict(solve_solution.get("summary", {}).get("solver") or {})
        tolerance_rows.append(
            {
                "block_id": str(system["block_id"]),
                "display_name": str(system["block_display_name"]),
                "rtol": float(tolerance),
                "atol": float(cfg.get("numerical_adequacy_tolerance_atol") or 1e-7),
                "solver_backend": str(solver.get("backend") or ""),
                "solver_device": str(solver.get("device") or ""),
                "solver_compiled": bool(solver.get("compiled", False)),
                "cg_iteration_count": int(solver.get("cg_iteration_count") or 0),
                "cg_residual_norm": float(solver.get("cg_residual_norm") or 0.0),
                "cg_tolerance": float(solver.get("cg_tolerance") or 0.0),
                "cg_converged": bool(solver.get("cg_converged", False)),
                "cg_max_iter_reached": bool(solver.get("cg_max_iter_reached", False)),
                "cg_max_iter": int(solver.get("cg_max_iter") or 0),
                "preconditioner": str(solver.get("preconditioner") or ""),
                "preconditioner_jitter": float(solver.get("preconditioner_jitter") or 0.0),
                **drift,
            }
        )

    direct_reference_rows: list[dict[str, Any]] = []
    max_cells = max(1, int(cfg.get("numerical_adequacy_reference_max_cells") or 256))
    max_provinces = max(1, int(cfg.get("numerical_adequacy_reference_max_provinces") or 8))
    province_cap = max(1, min(max_provinces, max_cells // max(1, int(system["month_count"]))))
    province_subset = _select_missing_information_reference_subset_provinces(system=system, max_provinces=province_cap)
    restricted_system = _restrict_missing_information_system_to_provinces(system=system, province_indices=province_subset)
    if int(restricted_system["cell_count"]) <= max_cells and list(restricted_system.get("reconciliation_rows") or []):
        dense_reference = _dense_solve_missing_information_reference(restricted_system)
        iterative_reference = _mi_solve_linear_system(
            system=restricted_system,
            cfg=cfg,
            backend=str(dict(restricted_system.get("backend_runtime") or {}).get("backend") or "cpu_sparse"),
            rtol_override=float(default_rtol),
            atol_override=float(cfg.get("numerical_adequacy_tolerance_atol") or 1e-7),
        )
        dense_solution = _materialize_missing_information_solution(
            system=restricted_system,
            correction_vector=np.asarray(dense_reference["correction_vector"], dtype=np.float64),
            solver_diagnostics={
                "backend": "dense_reference",
                "device": "cpu",
                "compiled": False,
                "cg_iteration_count": 0,
                "cg_residual_norm": round(float(dense_reference["residual_norm"]), 6),
            },
            cfg=cfg,
        )
        iterative_solution = _materialize_missing_information_solution(
            system=restricted_system,
            correction_vector=np.asarray(iterative_reference["correction_vector"], dtype=np.float64),
            solver_diagnostics=dict(iterative_reference["solver_diagnostics"]),
            cfg=cfg,
        )
        drift = _missing_information_solution_drift(dense_solution, iterative_solution)
        iterative_solver = dict(iterative_solution.get("summary", {}).get("solver") or {})
        direct_reference_rows.append(
            {
                "block_id": str(system["block_id"]),
                "display_name": str(system["block_display_name"]),
                "subset_cell_count": int(restricted_system["cell_count"]),
                "subset_province_count": int(restricted_system["province_count"]),
                "subset_provinces": [str(value) for value in list(restricted_system["province_axis"])],
                "condition_number": round(float(dense_reference["condition_number"]), 6),
                "reference_residual_norm": round(float(dense_reference["residual_norm"]), 6),
                "solver_backend": str(iterative_solver.get("backend") or ""),
                "solver_device": str(iterative_solver.get("device") or ""),
                "cg_iteration_count": int(iterative_solver.get("cg_iteration_count") or 0),
                "cg_residual_norm": float(iterative_solver.get("cg_residual_norm") or 0.0),
                "cg_tolerance": float(iterative_solver.get("cg_tolerance") or 0.0),
                "cg_converged": bool(iterative_solver.get("cg_converged", False)),
                "cg_max_iter_reached": bool(iterative_solver.get("cg_max_iter_reached", False)),
                "cg_max_iter": int(iterative_solver.get("cg_max_iter") or 0),
                "preconditioner": str(iterative_solver.get("preconditioner") or ""),
                "preconditioner_jitter": float(iterative_solver.get("preconditioner_jitter") or 0.0),
                **drift,
            }
        )

    tolerance_state_drift = [float(row["state_mae"]) for row in tolerance_rows]
    tolerance_residual_drift = [float(row["aggregate_residual_mae"]) for row in tolerance_rows]
    direct_state_drift = [float(row["state_mae"]) for row in direct_reference_rows]
    direct_residual_drift = [float(row["aggregate_residual_mae"]) for row in direct_reference_rows]
    posterior_std_drift = [float(row["posterior_std_mae"]) for row in tolerance_rows + direct_reference_rows]
    return {
        "available": True,
        "reason": "tolerance_sweep_and_direct_reference",
        "tolerance_rows": tolerance_rows,
        "direct_reference_rows": direct_reference_rows,
        "summary": {
            "anchor_rtol": round(float(anchor_tolerance), 10),
            "tolerance_scenario_count": int(len(tolerance_rows)),
            "direct_reference_case_count": int(len(direct_reference_rows)),
            "tolerance_nonconverged_count": int(sum(1 for row in tolerance_rows if not bool(row.get("cg_converged", False)))),
            "tolerance_iteration_cap_count": int(sum(1 for row in tolerance_rows if bool(row.get("cg_max_iter_reached", False)))),
            "direct_reference_nonconverged_count": int(sum(1 for row in direct_reference_rows if not bool(row.get("cg_converged", False)))),
            "max_tolerance_state_mae": round(max(tolerance_state_drift) if tolerance_state_drift else 0.0, 8),
            "max_tolerance_aggregate_residual_mae": round(max(tolerance_residual_drift) if tolerance_residual_drift else 0.0, 8),
            "max_direct_reference_state_mae": round(max(direct_state_drift) if direct_state_drift else 0.0, 8),
            "max_direct_reference_aggregate_residual_mae": round(max(direct_residual_drift) if direct_residual_drift else 0.0, 8),
            "max_posterior_std_mae": round(max(posterior_std_drift) if posterior_std_drift else 0.0, 8),
        },
    }
def _fit_missing_information_layer(
    *,
    block_id: str,
    block_display_name: str,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    row_precisions: np.ndarray,
    state: np.ndarray,
    posterior_std: np.ndarray,
    province_axis: list[str],
    national_weights: np.ndarray,
    donor_kernel: Mapping[str, Any],
    temporal_precision: float,
    phi: float,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    _ = national_weights
    if not bool(cfg.get("missing_information_enabled", True)):
        return {
            "available": False,
            "reason": "disabled",
            "state": np.asarray(state, dtype=np.float64),
            "posterior_std": np.asarray(posterior_std, dtype=np.float64),
            "summary": {"aggregate_constraint_count": 0},
            "rows": [],
            "aggregate_rows": [],
            "numerical_adequacy": {
                "available": False,
                "reason": "missing_information_disabled",
                "tolerance_rows": [],
                "direct_reference_rows": [],
                "summary": {},
            },
        }
    system = _build_missing_information_linear_system(
        block_id=block_id,
        block_display_name=block_display_name,
        block_rows=block_rows,
        indicator_params=indicator_params,
        row_precisions=row_precisions,
        state=state,
        posterior_std=posterior_std,
        province_axis=province_axis,
        donor_kernel=donor_kernel,
        temporal_precision=temporal_precision,
        phi=phi,
        cfg=cfg,
    )
    if not bool(system.get("available")):
        return {
            "available": False,
            "reason": str(system.get("reason") or "missing_aggregate_constraints"),
            "state": np.asarray(state, dtype=np.float64),
            "posterior_std": np.asarray(posterior_std, dtype=np.float64),
            "summary": {"aggregate_constraint_count": 0},
            "rows": [],
            "aggregate_rows": [],
            "numerical_adequacy": {
                "available": False,
                "reason": str(system.get("reason") or "missing_aggregate_constraints"),
                "tolerance_rows": [],
                "direct_reference_rows": [],
                "summary": {},
            },
        }
    solve_payload = _mi_solve_linear_system(system=system, cfg=cfg)
    solution = _materialize_missing_information_solution(
        system=system,
        correction_vector=np.asarray(solve_payload["correction_vector"], dtype=np.float64),
        solver_diagnostics=dict(solve_payload["solver_diagnostics"]),
        cfg=cfg,
    )
    solution["numerical_adequacy"] = _build_missing_information_numerical_adequacy_report(system=system, cfg=cfg)
    return solution


def _estimate_persistence(
    state: np.ndarray,
    *,
    national_weights: np.ndarray,
    cfg: Mapping[str, Any],
) -> float:
    if state.shape[1] <= 1 or np.asarray(national_weights, dtype=np.float64).size == 0:
        return 0.0
    national_series = np.sum(np.asarray(national_weights, dtype=np.float64).reshape(-1, 1) * state, axis=0, dtype=np.float64)
    prev = np.asarray(national_series[:-1], dtype=np.float64)
    curr = np.asarray(national_series[1:], dtype=np.float64)
    prior_strength = float(cfg["persistence_prior_strength"])
    prior_mean = float(cfg["persistence_prior_mean"])
    denominator = float(np.dot(prev, prev)) + prior_strength
    if denominator <= 1e-8:
        return 0.0
    phi = float((np.dot(prev, curr) + prior_strength * prior_mean) / denominator)
    ceiling = float(cfg["persistence_abs_ceiling"])
    return float(np.clip(phi, -ceiling, ceiling))


def _fit_indicator_parameters(
    *,
    block_rows: list[dict[str, Any]],
    state: np.ndarray,
    row_precisions: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, float | str]]:
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    rows_by_indicator: dict[str, list[int]] = defaultdict(list)
    for row_idx, row in enumerate(block_rows):
        rows_by_indicator[str(row.get("canonical_name") or "")].append(int(row_idx))

    indicator_payloads: dict[str, dict[str, Any]] = {}
    theta_init_values: list[float] = []
    theta_init_weights: list[float] = []
    minimum_loading = float(cfg["minimum_loading_magnitude"])
    loading_ceiling = float(cfg["loading_magnitude_ceiling"])
    for canonical_name, indices in rows_by_indicator.items():
        predictors: list[float] = []
        responses: list[float] = []
        weights: list[float] = []
        signs: list[str] = []
        for row_idx in indices:
            row = block_rows[row_idx]
            cells = np.asarray(row["support_cells"], dtype=np.int32)
            predictor = float(row.get("cell_weight") or (1.0 / max(cells.size, 1))) * float(np.sum(state_vector[cells]))
            response = float(row.get("measurement_value") or 0.0)
            weight = float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx])
            predictors.append(predictor)
            responses.append(response)
            weights.append(weight)
            signs.append(str(row.get("expected_sign") or "neutral"))
        x = np.asarray(predictors, dtype=np.float64)
        y = np.asarray(responses, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        x_mean = _weighted_mean(x, w)
        y_mean = _weighted_mean(y, w)
        centered_x = x - x_mean
        centered_y = y - y_mean
        sign = signs[0] if signs else "neutral"
        empirical_corr = float(_weighted_corr(x, y, w))
        if sign == "neutral":
            sign = "positive" if empirical_corr >= 0.0 else "negative"
        sign_multiplier = -1.0 if sign == "negative" else 1.0
        base_denominator = float(np.sum(w * np.square(centered_x))) + float(cfg["loading_ridge"])
        base_slope = float(np.sum(w * centered_x * centered_y) / max(base_denominator, 1e-8))
        base_magnitude = min(
            max(abs(base_slope), minimum_loading),
            loading_ceiling,
        )
        theta_init = float(_inverse_softplus(max(base_magnitude - minimum_loading, 1e-6)))
        effective_weight = float(np.sum(w))
        sign_conflict = max(0.0, -sign_multiplier * empirical_corr)
        shrinkage_precision = (
            float(cfg["loading_ridge"])
            + float(cfg["loading_row_scale_precision"]) / max(float(len(indices)), 1.0)
            + float(cfg["loading_weight_scale_precision"]) / max(effective_weight, 1.0)
            + float(cfg["loading_sign_conflict_precision"]) * sign_conflict
        )
        indicator_payloads[canonical_name] = {
            "indices": [int(idx) for idx in indices],
            "x": x,
            "y": y,
            "w": w,
            "x_mean": float(x_mean),
            "y_mean": float(y_mean),
            "centered_x": centered_x,
            "centered_y": centered_y,
            "sign": sign,
            "sign_multiplier": sign_multiplier,
            "empirical_corr": empirical_corr,
            "theta_init": theta_init,
            "effective_weight": effective_weight,
            "shrinkage_precision": shrinkage_precision,
            "row_count": len(indices),
        }
        theta_init_values.append(theta_init)
        theta_init_weights.append(max(effective_weight, 1.0))

    if not indicator_payloads:
        return {}

    theta_map = {canonical_name: float(payload["theta_init"]) for canonical_name, payload in indicator_payloads.items()}
    theta_init_array = np.asarray(theta_init_values, dtype=np.float64)
    theta_init_weight_array = np.asarray(theta_init_weights, dtype=np.float64)
    block_mu = _weighted_mean(theta_init_array, theta_init_weight_array)

    for _ in range(int(cfg["loading_theta_steps"])):
        theta_array = np.asarray([float(theta_map[name]) for name in indicator_payloads], dtype=np.float64)
        theta_weight_array = np.asarray([max(float(indicator_payloads[name]["effective_weight"]), 1.0) for name in indicator_payloads], dtype=np.float64)
        block_mu = _weighted_mean(theta_array, theta_weight_array)
        for canonical_name, payload in indicator_payloads.items():
            theta_value = float(theta_map[canonical_name])
            magnitude = minimum_loading + _softplus(theta_value)
            if magnitude > loading_ceiling:
                magnitude = loading_ceiling
            lambda_value = float(payload["sign_multiplier"]) * magnitude
            residual = np.asarray(payload["centered_y"], dtype=np.float64) - lambda_value * np.asarray(payload["centered_x"], dtype=np.float64)
            d_magnitude = _sigmoid(theta_value) if magnitude < loading_ceiling - 1e-8 else 0.0
            data_gradient = -float(np.sum(np.asarray(payload["w"], dtype=np.float64) * residual * np.asarray(payload["centered_x"], dtype=np.float64)))
            theta_gradient = data_gradient * float(payload["sign_multiplier"]) * d_magnitude
            theta_gradient += float(cfg["loading_theta_ridge"]) * theta_value
            theta_gradient += float(payload["shrinkage_precision"]) * (theta_value - block_mu)
            theta_map[canonical_name] = theta_value - float(cfg["loading_theta_learning_rate"]) * theta_gradient

    theta_array = np.asarray([float(theta_map[name]) for name in indicator_payloads], dtype=np.float64)
    theta_weight_array = np.asarray([max(float(indicator_payloads[name]["effective_weight"]), 1.0) for name in indicator_payloads], dtype=np.float64)
    block_mu = _weighted_mean(theta_array, theta_weight_array)

    parameter_map: dict[str, dict[str, float | str]] = {}
    for canonical_name, payload in indicator_payloads.items():
        theta_value = float(theta_map[canonical_name])
        magnitude = minimum_loading + _softplus(theta_value)
        magnitude = min(magnitude, loading_ceiling)
        slope = float(payload["sign_multiplier"]) * magnitude
        intercept = float(payload["y_mean"]) - slope * float(payload["x_mean"])
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        predictions = intercept + slope * x
        residual_mae = float(np.mean(np.abs(y - predictions))) if y.size else 0.0
        parameter_map[canonical_name] = {
            "canonical_name": canonical_name,
            "expected_sign": str(payload["sign"]),
            "alpha": float(intercept),
            "lambda": float(slope),
            "eta": float(theta_value),
            "block_mu": float(block_mu),
            "delta": float(theta_value - block_mu),
            "weighted_corr": float(payload["empirical_corr"]),
            "weighted_mae": residual_mae,
            "row_count": int(payload["row_count"]),
            "effective_weight": float(payload["effective_weight"]),
            "shrinkage_precision": float(payload["shrinkage_precision"]),
            "sign_conflict": float(max(0.0, -float(payload["sign_multiplier"]) * float(payload["empirical_corr"]))),
        }
    return parameter_map


def _build_precision_design(
    block_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str], dict[str, list[str]], np.ndarray]:
    role_values = sorted({str(row.get("measurement_role") or "unknown") for row in block_rows})
    source_values = sorted({str(row.get("source_bank") or "unknown") for row in block_rows})
    geo_values = sorted({str(row.get("geo_resolution") or "unknown") for row in block_rows})
    time_values = sorted({str(row.get("time_resolution") or "unknown") for row in block_rows})
    anchor_values = sorted({str(row.get("anchor_status") or "unknown") for row in block_rows})
    feature_domains = {
        "measurement_role": role_values,
        "source_bank": source_values,
        "geo_resolution": geo_values,
        "time_resolution": time_values,
        "anchor_status": anchor_values,
    }
    column_names = ["intercept", "quality_signal"]
    design_columns: list[np.ndarray] = []
    quality = np.asarray([float(row.get("quality_signal") or 0.0) for row in block_rows], dtype=np.float64)
    quality_centered = quality - float(np.mean(quality)) if quality.size else quality
    design_columns.append(np.ones((len(block_rows),), dtype=np.float64))
    design_columns.append(quality_centered.astype(np.float64))
    for feature_name, values in feature_domains.items():
        baseline = values[0] if values else ""
        for value in values[1:]:
            column_names.append(f"{feature_name}={value}")
            design_columns.append(
                np.asarray(
                    [1.0 if str(row.get(feature_name) or "unknown") == value else 0.0 for row in block_rows],
                    dtype=np.float64,
                )
            )
        if baseline:
            column_names.append(f"{feature_name}={baseline}:baseline")
    design_matrix = np.column_stack(design_columns) if design_columns else np.zeros((len(block_rows), 0), dtype=np.float64)
    return design_matrix, column_names, feature_domains, quality_centered


def _fit_precision_model(
    *,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    state: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if not block_rows:
        return {
            "rows": [],
            "precision_values": np.zeros((0,), dtype=np.float64),
            "feature_columns": [],
            "feature_domains": {},
        }
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    design_matrix, column_names, feature_domains, quality_centered = _build_precision_design(block_rows)
    residuals: list[float] = []
    weights: list[float] = []
    for row in block_rows:
        params = indicator_params[str(row.get("canonical_name") or "")]
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        predictor = float(row.get("cell_weight") or (1.0 / max(cells.size, 1))) * float(np.sum(state_vector[cells]))
        fitted = float(params["alpha"]) + float(params["lambda"]) * predictor
        residual = float(row.get("measurement_value") or 0.0) - fitted
        residuals.append(residual)
        weights.append(float(row.get("base_weight") or 1.0))
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    response = np.log(np.square(residuals_arr) + float(cfg["variance_eps"]))
    ridge = float(cfg["loading_ridge"])
    xtwx = design_matrix.T @ (weights_arr[:, None] * design_matrix)
    xtwy = design_matrix.T @ (weights_arr * response)
    beta = np.linalg.solve(xtwx + ridge * np.eye(xtwx.shape[0], dtype=np.float64), xtwy)
    log_var = design_matrix @ beta
    precision_values = np.clip(np.exp(-log_var), float(cfg["observation_precision_floor"]), float(cfg["observation_precision_ceiling"]))
    rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(block_rows):
        rows.append(
            {
                "canonical_name": str(row.get("canonical_name") or ""),
                "measurement_role": str(row.get("measurement_role") or ""),
                "source_bank": str(row.get("source_bank") or ""),
                "geo_resolution": str(row.get("geo_resolution") or ""),
                "time_resolution": str(row.get("time_resolution") or ""),
                "anchor_status": str(row.get("anchor_status") or ""),
                "quality_signal": round(float(row.get("quality_signal") or 0.0), 6),
                "quality_signal_centered": round(float(quality_centered[row_idx]), 6),
                "precision": round(float(precision_values[row_idx]), 6),
                "log_variance": round(float(log_var[row_idx]), 6),
            }
        )
    coefficient_rows: list[dict[str, Any]] = []
    for column_idx, column_name in enumerate(column_names[: len(beta)]):
        coefficient_rows.append({"feature": column_name, "coefficient": round(float(beta[column_idx]), 6)})
    return {
        "rows": rows,
        "precision_values": precision_values.astype(np.float64),
        "coefficient_rows": coefficient_rows,
        "feature_columns": column_names[: len(beta)],
        "feature_domains": feature_domains,
    }


def _region_mean_grid(
    state: np.ndarray,
    *,
    region_weights: Mapping[str, np.ndarray],
    region_axis: list[str],
    region_codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    region_tensor = np.zeros((len(region_axis), state.shape[1]), dtype=np.float64)
    region_index = {label: idx for idx, label in enumerate(region_axis)}
    province_region_mean = np.zeros_like(state, dtype=np.float64)
    for region_label in region_axis:
        weights = np.asarray(region_weights[region_label], dtype=np.float64)
        region_state = np.sum(weights.reshape(-1, 1) * state, axis=0, dtype=np.float64)
        region_tensor[int(region_index[region_label]), :] = region_state
    for province_idx, region_label in enumerate(region_codes):
        province_region_mean[province_idx, :] = region_tensor[int(region_index[region_label]), :]
    return province_region_mean, region_tensor


def _support_orientation_score(
    *,
    state: np.ndarray,
    block_rows: list[dict[str, Any]],
) -> float:
    if not block_rows:
        return 0.0
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    score = 0.0
    for row in block_rows:
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        predictor = float(row.get("cell_weight") or (1.0 / max(cells.size, 1))) * float(np.sum(state_vector[cells]))
        sign = str(row.get("expected_sign") or "neutral")
        sign_multiplier = -1.0 if sign == "negative" else 1.0
        score += (
            float(row.get("base_weight") or 1.0)
            * max(0.25, float(row.get("quality_signal") or 1.0))
            * sign_multiplier
            * predictor
            * float(row.get("measurement_value") or 0.0)
        )
    return float(score)


def _state_objective(
    *,
    state: np.ndarray,
    block_rows: list[dict[str, Any]],
    indicator_params: Mapping[str, Mapping[str, Any]],
    row_precisions: np.ndarray,
    phi: float,
    temporal_precision: float,
    regional_precision: float,
    region_mean: np.ndarray,
    regional_precision_scale: float = 1.0,
) -> dict[str, float]:
    state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
    obs_loss = 0.0
    for row_idx, row in enumerate(block_rows):
        params = indicator_params[str(row.get("canonical_name") or "")]
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        predictor = float(row.get("cell_weight") or (1.0 / max(cells.size, 1))) * float(np.sum(state_vector[cells]))
        fitted = float(params["alpha"]) + float(params["lambda"]) * predictor
        residual = float(row.get("measurement_value") or 0.0) - fitted
        obs_loss += 0.5 * float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx]) * float(residual**2)
    if state.shape[1] > 1:
        temporal_residual = state[:, 1:] - float(phi) * state[:, :-1]
        temporal_loss = 0.5 * float(temporal_precision) * float(np.sum(np.square(temporal_residual)))
    else:
        temporal_loss = 0.0
    regional_residual = state - region_mean
    effective_regional_precision = float(regional_precision) * float(regional_precision_scale)
    regional_loss = 0.5 * effective_regional_precision * float(np.sum(np.square(regional_residual)))
    total = obs_loss + temporal_loss + regional_loss
    return {
        "observation_loss": float(obs_loss),
        "temporal_loss": float(temporal_loss),
        "regional_loss": float(regional_loss),
        "total_loss": float(total),
    }


def _fit_block_state(
    *,
    block_id: str,
    block_display_name: str,
    block_rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    region_axis: list[str],
    region_codes: list[str],
    region_weights: Mapping[str, np.ndarray],
    national_weights: np.ndarray,
    donor_kernel: Mapping[str, Any],
    cfg: Mapping[str, Any],
    regional_precision_scale: float = 1.0,
) -> dict[str, Any]:
    province_count = len(province_axis)
    month_count = len(month_axis)
    state = _initial_state_from_measurements(
        block_rows=block_rows,
        province_count=province_count,
        month_count=month_count,
        region_members=province_region_members(region_codes),
        cfg=cfg,
    )
    row_precisions = np.ones((len(block_rows),), dtype=np.float64)
    indicator_params: dict[str, dict[str, float | str]] = {}
    precision_model: dict[str, Any] = {"rows": [], "precision_values": row_precisions.copy(), "coefficient_rows": []}
    phi = 0.0
    temporal_precision = float(cfg["temporal_precision_floor"])
    regional_precision = float(cfg["regional_precision_floor"])
    loss_path: list[dict[str, float]] = []

    for _outer in range(int(cfg["outer_iterations"])):
        if _support_orientation_score(state=state, block_rows=block_rows) < 0.0:
            state = -state
        state = _standardize_state(state, float(cfg["standardization_eps"])).reshape(province_count, month_count)
        phi = _estimate_persistence(
            state,
            national_weights=np.asarray(national_weights, dtype=np.float64),
            cfg=cfg,
        )
        indicator_params = _fit_indicator_parameters(
            block_rows=block_rows,
            state=state,
            row_precisions=row_precisions,
            cfg=cfg,
        )
        precision_model = _fit_precision_model(
            block_rows=block_rows,
            indicator_params=indicator_params,
            state=state,
            cfg=cfg,
        )
        row_precisions = np.asarray(precision_model["precision_values"], dtype=np.float64)
        if month_count > 1:
            national_series_for_precision = np.sum(
                np.asarray(national_weights, dtype=np.float64).reshape(-1, 1) * state,
                axis=0,
                dtype=np.float64,
            )
            temporal_residual = national_series_for_precision[1:] - float(phi) * national_series_for_precision[:-1]
            temporal_precision = _posterior_mean_precision(
                temporal_residual,
                prior_dof=float(cfg["temporal_precision_prior_dof"]),
                prior_variance=float(cfg["temporal_precision_prior_variance"]),
                floor=float(cfg["temporal_precision_floor"]),
                ceiling=float(cfg["temporal_precision_ceiling"]),
            )
        else:
            temporal_precision = float(cfg["temporal_precision_floor"])
        region_mean, _region_tensor = _region_mean_grid(
            state=state,
            region_weights=region_weights,
            region_axis=region_axis,
            region_codes=region_codes,
        )
        regional_residual = state - region_mean
        regional_precision = _posterior_mean_precision(
            regional_residual,
            prior_dof=float(cfg["regional_precision_prior_dof"]),
            prior_variance=float(cfg["regional_precision_prior_variance"]),
            floor=float(cfg["regional_precision_floor"]),
            ceiling=float(cfg["regional_precision_ceiling"]),
        )

        for _inner in range(int(cfg["inner_gradient_steps"])):
            gradient = np.zeros_like(state, dtype=np.float64)
            diagonal_precision = np.zeros_like(state, dtype=np.float64)
            state_vector = state.reshape(-1)
            for row_idx, row in enumerate(block_rows):
                params = indicator_params[str(row.get("canonical_name") or "")]
                cells = np.asarray(row["support_cells"], dtype=np.int32)
                cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
                predictor = cell_weight * float(np.sum(state_vector[cells]))
                fitted = float(params["alpha"]) + float(params["lambda"]) * predictor
                residual = float(row.get("measurement_value") or 0.0) - fitted
                observation_weight = float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx])
                coeff = observation_weight * float(params["lambda"]) * cell_weight
                if abs(coeff) <= 1e-12:
                    continue
                flat_grad = gradient.reshape(-1)
                flat_diag = diagonal_precision.reshape(-1)
                flat_grad[cells] -= coeff * residual
                flat_diag[cells] += observation_weight * float(params["lambda"] ** 2) * float(cell_weight**2)
            if month_count > 1:
                temporal_residual = state[:, 1:] - float(phi) * state[:, :-1]
                gradient[:, 1:] += float(temporal_precision) * temporal_residual
                gradient[:, :-1] += -float(temporal_precision) * float(phi) * temporal_residual
                diagonal_precision[:, 1:] += float(temporal_precision)
                diagonal_precision[:, :-1] += float(temporal_precision) * float(phi**2)
            region_mean, _region_tensor = _region_mean_grid(
                state=state,
                region_weights=region_weights,
                region_axis=region_axis,
                region_codes=region_codes,
            )
            regional_residual = state - region_mean
            effective_regional_precision = float(regional_precision) * float(regional_precision_scale)
            gradient += effective_regional_precision * regional_residual
            diagonal_precision += effective_regional_precision
            step = float(cfg["learning_rate"]) * gradient / np.clip(
                diagonal_precision + float(cfg["variance_eps"]),
                float(cfg["variance_eps"]),
                None,
            )
            state = np.clip(state - step, -float(cfg["state_value_clip"]), float(cfg["state_value_clip"]))

        loss_path.append(
            _state_objective(
                state=state,
                block_rows=block_rows,
                indicator_params=indicator_params,
                row_precisions=row_precisions,
                phi=phi,
                temporal_precision=temporal_precision,
                regional_precision=regional_precision,
                region_mean=region_mean,
                regional_precision_scale=regional_precision_scale,
            )
        )

    if _support_orientation_score(state=state, block_rows=block_rows) < 0.0:
        state = -state
    state = _standardize_state(state, float(cfg["standardization_eps"])).reshape(province_count, month_count)
    region_tensor, national_state = _aggregate_states(
        state=state,
        region_weights=region_weights,
        national_weights=national_weights,
        region_axis=region_axis,
    )
    region_mean, region_tensor = _region_mean_grid(
        state=state,
        region_weights=region_weights,
        region_axis=region_axis,
        region_codes=region_codes,
    )
    obs_diagonal = np.zeros_like(state, dtype=np.float64)
    state_vector = state.reshape(-1)
    for row_idx, row in enumerate(block_rows):
        params = indicator_params[str(row.get("canonical_name") or "")]
        cells = np.asarray(row["support_cells"], dtype=np.int32)
        cell_weight = float(row.get("cell_weight") or (1.0 / max(cells.size, 1)))
        observation_weight = float(row.get("base_weight") or 1.0) * float(row_precisions[row_idx])
        obs_diagonal.reshape(-1)[cells] += observation_weight * float(params["lambda"] ** 2) * float(cell_weight**2)

    effective_regional_precision = float(regional_precision) * float(regional_precision_scale)
    posterior_precision = obs_diagonal + effective_regional_precision + float(temporal_precision)
    posterior_std = np.sqrt(1.0 / np.clip(posterior_precision, float(cfg["variance_eps"]), None))
    missing_information = _fit_missing_information_layer(
        block_id=block_id,
        block_display_name=block_display_name,
        block_rows=block_rows,
        indicator_params=indicator_params,
        row_precisions=row_precisions,
        state=state,
        posterior_std=posterior_std,
        province_axis=province_axis,
        national_weights=np.asarray(national_weights, dtype=np.float64),
        donor_kernel=donor_kernel,
        temporal_precision=float(temporal_precision),
        phi=float(phi),
        cfg=cfg,
    )
    state = np.asarray(missing_information["state"], dtype=np.float64)
    posterior_std = np.asarray(missing_information["posterior_std"], dtype=np.float64)
    region_tensor, national_state = _aggregate_states(
        state=state,
        region_weights=region_weights,
        national_weights=national_weights,
        region_axis=region_axis,
    )
    region_mean, region_tensor = _region_mean_grid(
        state=state,
        region_weights=region_weights,
        region_axis=region_axis,
        region_codes=region_codes,
    )
    corrected_loss = _state_objective(
        state=state,
        block_rows=block_rows,
        indicator_params=indicator_params,
        row_precisions=row_precisions,
        phi=phi,
        temporal_precision=temporal_precision,
        regional_precision=regional_precision,
        region_mean=region_mean,
        regional_precision_scale=regional_precision_scale,
    )
    measurement_fit_rows = _build_measurement_fit_rows(
        block_id=block_id,
        block_display_name=block_display_name,
        block_rows=block_rows,
        indicator_params=indicator_params,
        row_precisions=row_precisions,
        state=state,
        posterior_std=posterior_std,
        cfg=cfg,
    )

    indicator_rows: list[dict[str, Any]] = []
    for canonical_name in sorted(indicator_params):
        params = indicator_params[canonical_name]
        indicator_rows.append(
            {
                "block_id": block_id,
                "display_name": block_display_name,
                "canonical_name": canonical_name,
                "expected_sign": str(params["expected_sign"]),
                "alpha": round(float(params["alpha"]), 6),
                "lambda": round(float(params["lambda"]), 6),
                "eta": round(float(params["eta"]), 6),
                "block_mu": round(float(params["block_mu"]), 6),
                "delta": round(float(params["delta"]), 6),
                "weighted_corr": round(float(params["weighted_corr"]), 6),
                "weighted_mae": round(float(params["weighted_mae"]), 6),
                "row_count": int(params["row_count"]),
                "effective_weight": round(float(params["effective_weight"]), 6),
                "shrinkage_precision": round(float(params["shrinkage_precision"]), 6),
                "sign_conflict": round(float(params["sign_conflict"]), 6),
            }
        )

    fit_summary = {
        "block_id": block_id,
        "display_name": block_display_name,
        "row_count": len(block_rows),
        "indicator_count": len(indicator_rows),
        "phi": round(float(phi), 6),
        "temporal_precision": round(float(temporal_precision), 6),
        "regional_precision": round(float(regional_precision), 6),
        "regional_precision_scale": round(float(regional_precision_scale), 6),
        "applied_regional_precision": round(float(effective_regional_precision), 6),
        "final_total_loss": round(float(loss_path[-1]["total_loss"]) if loss_path else 0.0, 6),
        "final_observation_loss": round(float(loss_path[-1]["observation_loss"]) if loss_path else 0.0, 6),
        "final_temporal_loss": round(float(loss_path[-1]["temporal_loss"]) if loss_path else 0.0, 6),
        "final_regional_loss": round(float(loss_path[-1]["regional_loss"]) if loss_path else 0.0, 6),
        "corrected_total_loss": round(float(corrected_loss["total_loss"]), 6),
        "corrected_observation_loss": round(float(corrected_loss["observation_loss"]), 6),
        "corrected_temporal_loss": round(float(corrected_loss["temporal_loss"]), 6),
        "corrected_regional_loss": round(float(corrected_loss["regional_loss"]), 6),
        "missing_information_available": bool(missing_information.get("available")),
        "missing_information_summary": dict(missing_information.get("summary") or {}),
        "numerical_adequacy_available": bool(dict(missing_information.get("numerical_adequacy") or {}).get("available")),
        "numerical_adequacy_summary": dict(dict(missing_information.get("numerical_adequacy") or {}).get("summary") or {}),
    }

    return {
        "block_id": block_id,
        "display_name": block_display_name,
        "state": state.astype(np.float32),
        "region_state": region_tensor.astype(np.float32),
        "national_state": national_state.astype(np.float32),
        "posterior_std": posterior_std.astype(np.float32),
        "indicator_rows": indicator_rows,
        "precision_rows": [{"block_id": block_id, "display_name": block_display_name, **row} for row in list(precision_model.get("rows") or [])],
        "precision_coefficients": [
            {"block_id": block_id, "display_name": block_display_name, **row}
            for row in list(precision_model.get("coefficient_rows") or [])
        ],
        "measurement_fit_rows": measurement_fit_rows,
        "missing_information": missing_information,
        "numerical_adequacy": dict(missing_information.get("numerical_adequacy") or {}),
        "fit_summary": fit_summary,
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _build_calibration_report(
    *,
    measurement_fit_rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    intervals = sorted({float(value) for value in list(cfg.get("calibration_intervals") or [0.5, 0.8, 0.95]) if 0.0 < float(value) < 1.0})
    valid_rows = [
        dict(row)
        for row in measurement_fit_rows
        if np.isfinite(float(row.get("standardized_residual") or 0.0))
        and np.isfinite(float(row.get("predictive_std") or 0.0))
        and float(row.get("predictive_std") or 0.0) > 0.0
    ]
    if not valid_rows:
        return {"method": "phase15_v2_calibration_v1", "available": False, "rows": [], "split_rows": [], "summary": {"row_count": 0}}

    normal = NormalDist()
    z_values = np.asarray([float(row.get("standardized_residual") or 0.0) for row in valid_rows], dtype=np.float64)

    def _rows_for_subset(label: str, subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        subset_z = np.asarray([float(row.get("standardized_residual") or 0.0) for row in subset], dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for interval in intervals:
            z_cut = float(normal.inv_cdf((1.0 + float(interval)) / 2.0))
            observed = float(np.mean(np.abs(subset_z) <= z_cut)) if subset_z.size else 0.0
            rows.append(
                {
                    "segment": label,
                    "interval": round(float(interval), 4),
                    "nominal_coverage": round(float(interval), 6),
                    "observed_coverage": round(float(observed), 6),
                    "coverage_gap": round(float(observed - interval), 6),
                    "row_count": int(subset_z.size),
                }
            )
        return rows

    rows = _rows_for_subset("global", valid_rows)
    split_rows: list[dict[str, Any]] = []
    for field in ("measurement_role", "geo_resolution", "time_resolution"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            grouped[str(row.get(field) or "unknown")].append(row)
        for token, subset in sorted(grouped.items()):
            if len(subset) < 4:
                continue
            for record in _rows_for_subset(f"{field}:{token}", subset):
                split_rows.append(record)

    abs_gap = [abs(float(row["coverage_gap"])) for row in rows]
    return {
        "method": "phase15_v2_calibration_v1",
        "available": True,
        "rows": rows,
        "split_rows": split_rows,
        "summary": {
            "row_count": len(valid_rows),
            "mean_abs_coverage_gap": round(float(np.mean(abs_gap)) if abs_gap else 0.0, 6),
            "max_abs_coverage_gap": round(float(np.max(abs_gap)) if abs_gap else 0.0, 6),
        },
    }


def _pooling_sensitivity_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    adjusted = dict(cfg)
    adjusted["outer_iterations"] = min(int(cfg["outer_iterations"]), int(cfg.get("pooling_sensitivity_outer_iterations", 3)))
    adjusted["inner_gradient_steps"] = min(int(cfg["inner_gradient_steps"]), int(cfg.get("pooling_sensitivity_inner_gradient_steps", 10)))
    return adjusted


def _build_pooling_sensitivity_report(
    *,
    block_axis: list[str],
    block_display: Mapping[str, str],
    rows_by_block: Mapping[str, list[dict[str, Any]]],
    baseline_block_fits: Mapping[str, Mapping[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    region_axis: list[str],
    region_codes: list[str],
    region_weights: Mapping[str, np.ndarray],
    national_weights: np.ndarray,
    donor_kernel: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(cfg.get("pooling_sensitivity_enabled", True)):
        return {"method": "phase15_v2_pooling_sensitivity_v1", "available": False, "rows": [], "summary": {"scenario_count": 0}}
    sensitivity_cfg = _pooling_sensitivity_cfg(cfg)
    scales = [float(value) for value in list(cfg.get("pooling_sensitivity_scales") or []) if float(value) > 0.0]
    rows: list[dict[str, Any]] = []
    for scale in scales:
        for block_id in block_axis:
            baseline = dict(baseline_block_fits.get(block_id) or {})
            if not baseline:
                continue
            scenario_fit = _fit_block_state(
                block_id=block_id,
                block_display_name=str(block_display.get(block_id) or block_id.replace("_", " ").title()),
                block_rows=list(rows_by_block.get(block_id) or []),
                province_axis=province_axis,
                month_axis=month_axis,
                region_axis=region_axis,
                    region_codes=region_codes,
                    region_weights=region_weights,
                    national_weights=np.asarray(national_weights, dtype=np.float64),
                    donor_kernel=donor_kernel,
                    cfg=sensitivity_cfg,
                    regional_precision_scale=float(scale),
                )
            baseline_state = np.asarray(baseline["state"], dtype=np.float64)
            scenario_state = np.asarray(scenario_fit["state"], dtype=np.float64)
            baseline_national = np.asarray(baseline["national_state"], dtype=np.float64)
            scenario_national = np.asarray(scenario_fit["national_state"], dtype=np.float64)
            baseline_std = np.asarray(baseline["posterior_std"], dtype=np.float64)
            scenario_std = np.asarray(scenario_fit["posterior_std"], dtype=np.float64)
            rows.append(
                {
                    "scenario": f"regional_pooling_scale_{scale:.2f}",
                    "block_id": block_id,
                    "display_name": str(block_display.get(block_id) or block_id.replace("_", " ").title()),
                    "regional_precision_scale": round(float(scale), 6),
                    "state_corr": round(float(_safe_corr(baseline_state, scenario_state)), 6),
                    "state_mae": round(float(np.mean(np.abs(scenario_state - baseline_state))), 6),
                    "national_corr": round(float(_safe_corr(baseline_national, scenario_national)), 6),
                    "national_mae": round(float(np.mean(np.abs(scenario_national - baseline_national))), 6),
                    "province_dispersion_ratio": round(
                        float(np.std(scenario_state) / max(np.std(baseline_state), float(cfg["standardization_eps"]))),
                        6,
                    ),
                    "uncertainty_ratio": round(
                        float(np.mean(scenario_std) / max(np.mean(baseline_std), float(cfg["standardization_eps"]))),
                        6,
                    ),
                }
            )
    return {
        "method": "phase15_v2_pooling_sensitivity_v1",
        "available": bool(rows),
        "rows": rows,
        "summary": {
            "scenario_count": len(rows),
            "max_state_mae": round(float(max((float(row["state_mae"]) for row in rows), default=0.0)), 6),
            "min_state_corr": round(float(min((float(row["state_corr"]) for row in rows), default=1.0)), 6),
            "max_uncertainty_ratio": round(float(max((float(row["uncertainty_ratio"]) for row in rows), default=1.0)), 6),
        },
    }


def fit_phase15_v2_map_engine(
    *,
    normalized_rows: list[dict[str, Any]],
    observation_support: Mapping[str, Any],
    measurement_spec: Mapping[str, Any],
    province_axis: list[str],
    month_axis: list[str],
    region_labels: list[str],
    plugin_id: str,
) -> dict[str, Any]:
    cfg = _phase15_v2_engine_cfg(plugin_id)
    aggregation = _estimate_aggregation_weights(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        cfg=cfg,
    )
    region_codes = [str(code) for code in list(aggregation["region_codes"])]
    region_members = {str(key): [int(idx) for idx in list(value)] for key, value in dict(aggregation["region_members"]).items()}
    region_axis = list(region_members.keys())
    measurement_payload = _aggregate_measurement_rows(
        normalized_rows=normalized_rows,
        observation_support=observation_support,
        measurement_spec=measurement_spec,
        cfg=cfg,
    )
    measurement_rows = list(measurement_payload["measurement_rows"])
    rows_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_display = dict(measurement_payload["block_display"])
    for row in measurement_rows:
        rows_by_block[str(row.get("block_id") or "")].append(row)
    block_axis = [
        str(block.get("block_id") or "")
        for block in list(dict(measurement_spec).get("retained_blocks") or [])
        if str(block.get("block_id") or "") in rows_by_block
    ]

    province_count = len(province_axis)
    month_count = len(month_axis)
    block_count = len(block_axis)
    province_state_tensor = np.zeros((province_count, month_count, block_count), dtype=np.float32)
    region_state_tensor = np.zeros((len(region_axis), month_count, block_count), dtype=np.float32)
    national_state_tensor = np.zeros((1, month_count, block_count), dtype=np.float32)
    donor_kernel = _build_donor_kernel(
        feature_matrix=np.asarray(aggregation.get("feature_matrix"), dtype=np.float64),
        cfg=cfg,
    )
    province_state_rows: list[dict[str, Any]] = []
    region_state_rows: list[dict[str, Any]] = []
    national_state_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    indicator_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    precision_coefficient_rows: list[dict[str, Any]] = []
    measurement_fit_rows: list[dict[str, Any]] = []
    missing_information_rows: list[dict[str, Any]] = []
    missing_information_aggregate_rows: list[dict[str, Any]] = []
    numerical_adequacy_tolerance_rows: list[dict[str, Any]] = []
    numerical_adequacy_direct_reference_rows: list[dict[str, Any]] = []
    fit_summary_rows: list[dict[str, Any]] = []
    baseline_block_fits: dict[str, dict[str, Any]] = {}

    for block_idx, block_id in enumerate(block_axis):
        block_rows = list(rows_by_block.get(block_id) or [])
        block_display_name = str(block_display.get(block_id) or block_id.replace("_", " ").title())
        block_fit = _fit_block_state(
            block_id=block_id,
            block_display_name=block_display_name,
            block_rows=block_rows,
            province_axis=province_axis,
            month_axis=month_axis,
            region_axis=region_axis,
            region_codes=region_codes,
            region_weights=aggregation["region_weights"],
            national_weights=np.asarray(aggregation["national_weights"], dtype=np.float64),
            donor_kernel=donor_kernel,
            cfg=cfg,
        )
        baseline_block_fits[block_id] = dict(block_fit)
        province_state_tensor[:, :, block_idx] = np.asarray(block_fit["state"], dtype=np.float32)
        region_state_tensor[:, :, block_idx] = np.asarray(block_fit["region_state"], dtype=np.float32)
        national_state_tensor[0, :, block_idx] = np.asarray(block_fit["national_state"], dtype=np.float32)
        fit_summary_rows.append(dict(block_fit["fit_summary"]))
        indicator_rows.extend(list(block_fit["indicator_rows"]))
        precision_rows.extend(list(block_fit["precision_rows"]))
        precision_coefficient_rows.extend(list(block_fit["precision_coefficients"]))
        measurement_fit_rows.extend(list(block_fit["measurement_fit_rows"]))
        missing_summary = dict(block_fit["missing_information"].get("summary") or {})
        missing_solver = dict(missing_summary.get("solver") or {})
        missing_information_rows.extend(
            [
                {
                    **row,
                    "province": str(province_axis[int(row.get("province_index") or 0)]),
                    "solver_backend": str(missing_solver.get("backend") or ""),
                    "solver_device": str(missing_solver.get("device") or ""),
                    "solver_compiled": bool(missing_solver.get("compiled", False)),
                }
                for row in list(block_fit["missing_information"].get("rows") or [])
            ]
        )
        missing_information_aggregate_rows.extend(list(block_fit["missing_information"].get("aggregate_rows") or []))
        numerical_adequacy_report = dict(block_fit.get("numerical_adequacy") or {})
        numerical_adequacy_tolerance_rows.extend(list(numerical_adequacy_report.get("tolerance_rows") or []))
        numerical_adequacy_direct_reference_rows.extend(list(numerical_adequacy_report.get("direct_reference_rows") or []))
        for province_idx, province in enumerate(province_axis):
            province_state_rows.append(
                {
                    "block_id": block_id,
                    "display_name": block_display_name,
                    "province": str(province),
                    "region": str(region_codes[province_idx]),
                    "state_values": [round(float(value), 6) for value in province_state_tensor[province_idx, :, block_idx].tolist()],
                }
            )
            uncertainty_rows.append(
                {
                    "block_id": block_id,
                    "display_name": block_display_name,
                    "province": str(province),
                    "region": str(region_codes[province_idx]),
                    "posterior_std_values": [
                        round(float(value), 6)
                        for value in np.asarray(block_fit["posterior_std"], dtype=np.float32)[province_idx, :].tolist()
                    ],
                }
            )
        for region_idx, region_label in enumerate(region_axis):
            region_state_rows.append(
                {
                    "block_id": block_id,
                    "display_name": block_display_name,
                    "region": str(region_label),
                    "state_values": [round(float(value), 6) for value in region_state_tensor[region_idx, :, block_idx].tolist()],
                }
            )
        national_state_rows.append(
            {
                "block_id": block_id,
                "display_name": block_display_name,
                "geo": "Philippines",
                "state_values": [round(float(value), 6) for value in national_state_tensor[0, :, block_idx].tolist()],
            }
        )

    serializable_measurement_rows: list[dict[str, Any]] = []
    for row in measurement_rows:
        serializable_row = dict(row)
        serializable_row["support_cells"] = [int(cell) for cell in list(np.asarray(row["support_cells"], dtype=np.int32).tolist())]
        serializable_measurement_rows.append(serializable_row)

    aggregation_rows = {
        "method": "phase15_v2_bottom_up_weights_v2",
        "weight_source": str(aggregation["weight_source"]),
        "province_axis": list(province_axis),
        "region_axis": list(region_axis),
        "feature_names": [str(name) for name in list(aggregation.get("feature_names") or [])],
        "supervision_names": [str(name) for name in list(aggregation.get("supervision_names") or [])],
        "support_summary": dict(aggregation.get("support_summary") or {}),
        "coefficient_values": [round(float(value), 6) for value in list(aggregation.get("coefficient_values") or [])],
        "diagnostics": dict(aggregation.get("diagnostics") or {}),
        "national_weights": [round(float(value), 8) for value in np.asarray(aggregation["national_weights"], dtype=np.float64).tolist()],
        "region_weights": {
            str(region_label): [round(float(value), 8) for value in np.asarray(weights, dtype=np.float64).tolist()]
            for region_label, weights in dict(aggregation["region_weights"]).items()
        },
        "region_codes": [str(value) for value in region_codes],
    }
    calibration = _build_calibration_report(
        measurement_fit_rows=measurement_fit_rows,
        cfg=cfg,
    )
    pooling_sensitivity = _build_pooling_sensitivity_report(
        block_axis=block_axis,
        block_display=block_display,
        rows_by_block=rows_by_block,
        baseline_block_fits=baseline_block_fits,
        province_axis=province_axis,
        month_axis=month_axis,
        region_axis=region_axis,
        region_codes=region_codes,
        region_weights=aggregation["region_weights"],
        national_weights=np.asarray(aggregation["national_weights"], dtype=np.float64),
        donor_kernel=donor_kernel,
        cfg=cfg,
    )

    return {
        "method": "phase15_v2_map_smoother_v1",
        "plugin_id": plugin_id,
        "province_axis": list(province_axis),
        "region_axis": list(region_axis),
        "month_axis": list(month_axis),
        "block_axis": list(block_axis),
        "aggregation": aggregation_rows,
        "measurement_rows": {
            "method": "phase15_v2_measurement_rows_v1",
            "row_count": len(serializable_measurement_rows),
            "rows": serializable_measurement_rows,
        },
        "indicator_parameters": {
            "method": "phase15_v2_hierarchical_signed_loading_v2",
            "rows": indicator_rows,
        },
        "precision_models": {
            "method": "phase15_v2_log_linear_precision_v1",
            "rows": precision_rows,
            "coefficient_rows": precision_coefficient_rows,
        },
        "measurement_fit": {
            "method": "phase15_v2_measurement_fit_v1",
            "rows": measurement_fit_rows,
        },
        "missing_information": {
            "method": "phase15_v2_missing_information_layer_v1",
            "donor_kernel": {
                "available": bool(donor_kernel.get("available")),
                "reason": str(donor_kernel.get("reason") or ""),
                "entropy_values": [round(float(value), 6) for value in np.asarray(donor_kernel.get("entropy"), dtype=np.float64).tolist()],
                "effective_donor_count_values": [
                    round(float(value), 6) for value in np.asarray(donor_kernel.get("effective_donor_count"), dtype=np.float64).tolist()
                ],
            },
            "rows": missing_information_rows,
            "aggregate_rows": missing_information_aggregate_rows,
            "summary": {
                "available_block_count": int(
                    sum(1 for row in fit_summary_rows if bool(row.get("missing_information_available")))
                ),
                "aggregate_constraint_count": int(len(missing_information_aggregate_rows)),
                "solver_backends": sorted(
                    {
                        str(row.get("solver_backend") or "")
                        for row in missing_information_rows
                        if str(row.get("solver_backend") or "")
                    }
                ),
            },
        },
        "calibration": calibration,
        "pooling_sensitivity": pooling_sensitivity,
        "numerical_adequacy": {
            "method": "phase15_v2_numerical_adequacy_v1",
            "available": bool(
                any(bool(row.get("numerical_adequacy_available")) for row in fit_summary_rows)
            ),
            "tolerance_rows": numerical_adequacy_tolerance_rows,
            "direct_reference_rows": numerical_adequacy_direct_reference_rows,
            "summary": {
                "available_block_count": int(
                    sum(1 for row in fit_summary_rows if bool(row.get("numerical_adequacy_available")))
                ),
                "anchor_rtol": round(
                    min(
                        (
                            float(row.get("rtol") or 0.0)
                            for row in numerical_adequacy_tolerance_rows
                            if float(row.get("rtol") or 0.0) > 0.0
                        ),
                        default=0.0,
                    ),
                    10,
                ),
                "tolerance_scenario_count": int(len(numerical_adequacy_tolerance_rows)),
                "direct_reference_case_count": int(len(numerical_adequacy_direct_reference_rows)),
                "tolerance_nonconverged_count": int(
                    sum(1 for row in numerical_adequacy_tolerance_rows if not bool(row.get("cg_converged", False)))
                ),
                "tolerance_iteration_cap_count": int(
                    sum(1 for row in numerical_adequacy_tolerance_rows if bool(row.get("cg_max_iter_reached", False)))
                ),
                "direct_reference_nonconverged_count": int(
                    sum(1 for row in numerical_adequacy_direct_reference_rows if not bool(row.get("cg_converged", False)))
                ),
                "max_tolerance_state_mae": round(
                    max((float(row.get("state_mae") or 0.0) for row in numerical_adequacy_tolerance_rows), default=0.0),
                    8,
                ),
                "max_tolerance_aggregate_residual_mae": round(
                    max((float(row.get("aggregate_residual_mae") or 0.0) for row in numerical_adequacy_tolerance_rows), default=0.0),
                    8,
                ),
                "max_direct_reference_state_mae": round(
                    max((float(row.get("state_mae") or 0.0) for row in numerical_adequacy_direct_reference_rows), default=0.0),
                    8,
                ),
                "max_direct_reference_aggregate_residual_mae": round(
                    max((float(row.get("aggregate_residual_mae") or 0.0) for row in numerical_adequacy_direct_reference_rows), default=0.0),
                    8,
                ),
                "max_posterior_std_mae": round(
                    max(
                        (
                            float(row.get("posterior_std_mae") or 0.0)
                            for row in numerical_adequacy_tolerance_rows + numerical_adequacy_direct_reference_rows
                        ),
                        default=0.0,
                    ),
                    8,
                ),
            },
        },
        "fit_summary": {
            "method": "phase15_v2_map_smoother_v1",
            "rows": fit_summary_rows,
            "global": {
                "block_count": len(block_axis),
                "measurement_row_count": len(serializable_measurement_rows),
                "weight_source": str(aggregation["weight_source"]),
                "aggregation_diagnostics": dict(aggregation.get("diagnostics") or {}),
                "missing_information_summary": {
                    "available_block_count": int(
                        sum(1 for row in fit_summary_rows if bool(row.get("missing_information_available")))
                    ),
                    "aggregate_constraint_count": int(len(missing_information_aggregate_rows)),
                },
                "calibration_summary": dict(calibration.get("summary") or {}),
                "pooling_sensitivity_summary": dict(pooling_sensitivity.get("summary") or {}),
                "numerical_adequacy_summary": {
                    "available_block_count": int(
                        sum(1 for row in fit_summary_rows if bool(row.get("numerical_adequacy_available")))
                    ),
                    "anchor_rtol": round(
                        min(
                            (
                                float(row.get("rtol") or 0.0)
                                for row in numerical_adequacy_tolerance_rows
                                if float(row.get("rtol") or 0.0) > 0.0
                            ),
                            default=0.0,
                    ),
                    10,
                    ),
                    "tolerance_scenario_count": int(len(numerical_adequacy_tolerance_rows)),
                    "direct_reference_case_count": int(len(numerical_adequacy_direct_reference_rows)),
                    "tolerance_nonconverged_count": int(
                        sum(1 for row in numerical_adequacy_tolerance_rows if not bool(row.get("cg_converged", False)))
                    ),
                    "tolerance_iteration_cap_count": int(
                        sum(1 for row in numerical_adequacy_tolerance_rows if bool(row.get("cg_max_iter_reached", False)))
                    ),
                    "direct_reference_nonconverged_count": int(
                        sum(1 for row in numerical_adequacy_direct_reference_rows if not bool(row.get("cg_converged", False)))
                    ),
                    "max_tolerance_state_mae": round(
                        max((float(row.get("state_mae") or 0.0) for row in numerical_adequacy_tolerance_rows), default=0.0),
                        8,
                    ),
                    "max_tolerance_aggregate_residual_mae": round(
                        max((float(row.get("aggregate_residual_mae") or 0.0) for row in numerical_adequacy_tolerance_rows), default=0.0),
                        8,
                    ),
                    "max_direct_reference_state_mae": round(
                        max((float(row.get("state_mae") or 0.0) for row in numerical_adequacy_direct_reference_rows), default=0.0),
                        8,
                    ),
                    "max_direct_reference_aggregate_residual_mae": round(
                        max((float(row.get("aggregate_residual_mae") or 0.0) for row in numerical_adequacy_direct_reference_rows), default=0.0),
                        8,
                    ),
                    "max_posterior_std_mae": round(
                        max(
                            (
                                float(row.get("posterior_std_mae") or 0.0)
                                for row in numerical_adequacy_tolerance_rows + numerical_adequacy_direct_reference_rows
                            ),
                            default=0.0,
                        ),
                        8,
                    ),
                },
            },
        },
        "province_states": {
            "method": "phase15_v2_map_smoother_v1",
            "province_axis": list(province_axis),
            "month_axis": list(month_axis),
            "rows": province_state_rows,
        },
        "region_states": {
            "method": "phase15_v2_map_smoother_v1",
            "region_axis": list(region_axis),
            "month_axis": list(month_axis),
            "rows": region_state_rows,
        },
        "national_states": {
            "method": "phase15_v2_map_smoother_v1",
            "month_axis": list(month_axis),
            "rows": national_state_rows,
        },
        "uncertainty": {
            "method": "phase15_v2_map_smoother_v1",
            "province_axis": list(province_axis),
            "month_axis": list(month_axis),
            "rows": uncertainty_rows,
        },
        "province_state_tensor": province_state_tensor,
        "region_state_tensor": region_state_tensor,
        "national_state_tensor": national_state_tensor,
    }
