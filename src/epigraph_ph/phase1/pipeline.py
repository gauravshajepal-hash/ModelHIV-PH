from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.geography import geo_resolution_label, infer_philippines_geo, infer_region_code, is_national_geo, normalize_geo_label
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.phase1.normalization_helpers import (
    AGE_TOKENS,
    KP_TOKENS,
    SEX_TOKENS,
    bias_fields as _phase1_normalization_bias_fields,
    evidence_class as _evidence_class,
    evidence_weight as _phase1_normalization_evidence_weight,
    geo_resolution as _geo_resolution,
    infer_domain_family as _infer_domain_family,
    infer_from_tokens as _infer_from_tokens,
    infer_region as _infer_region,
    infer_pathway_family as _infer_pathway_family,
    is_valid_observation_year as _is_valid_observation_year,
    normalize_numeric_value as _normalize_numeric_value,
    normalize_unit as _normalize_unit,
    safe_float as _safe_float,
    source_reliability_class as _source_reliability_class,
    text_blob as _text_blob,
    time_components as _time_components,
)
from epigraph_ph.runtime import (
    RunContext,
    choose_jax_device,
    choose_torch_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    read_json,
    save_tensor_artifact,
    to_numpy,
    torch_to_jax_handoff,
    to_torch_tensor,
    utc_now_iso,
    write_boundary_shape_package,
    write_gold_standard_package,
    write_ground_truth_package,
    write_json,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase1_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase1", {}) or {})


def _phase1_required(key: str) -> Any:
    cfg = _phase1_cfg()
    if key not in cfg:
        raise KeyError(f"Missing HIV phase1 constraint setting: {key}")
    return cfg[key]


def _phase1_required_section(key: str) -> dict[str, Any]:
    value = _phase1_required(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase1 constraint setting '{key}' must be a mapping")
    return dict(value)


def _phase1_stabilizer(key: str) -> float:
    stabilizers = dict((_HIV_PLUGIN.numerical_stabilizers or {}).get("phase1", {}) or {})
    if key not in stabilizers:
        raise KeyError(f"Missing HIV phase1 numerical stabilizer: {key}")
    return float(stabilizers[key])


def _phase1_preprocessing_cfg() -> dict[str, Any]:
    return _phase1_required_section("preprocessing")


def _phase1_compute_backend() -> str:
    cfg = _phase1_preprocessing_cfg()
    preferred = str(cfg.get("compute_backend") or "auto").lower()
    prefer_gpu = bool(cfg.get("prefer_gpu", True))
    if preferred == "torch" and torch is not None:
        return "torch"
    if preferred == "numpy":
        return "numpy"
    if torch is not None and choose_torch_device(prefer_gpu=prefer_gpu) == "cuda":
        return "torch"
    return "numpy"


def _evidence_weight(row: dict[str, Any], evidence_class: str) -> float:
    return _phase1_normalization_evidence_weight(row, evidence_class, _phase1_required_section("evidence_class_weights"))


def _bias_fields(row: dict[str, Any], reliability_class: str, time_resolution: str, geo_resolution: str) -> dict[str, Any]:
    return _phase1_normalization_bias_fields(
        row,
        reliability_class,
        time_resolution,
        geo_resolution,
        reliability_class_rules=_phase1_required_section("reliability_class_rules"),
        bias_class_thresholds=_phase1_required_section("bias_class_thresholds"),
        spatial_relevance_by_geo=_phase1_required_section("spatial_relevance_by_geo"),
        bias_penalty_mix=_phase1_required_section("bias_penalty_mix"),
    )


def _canonical_unit_index(normalized_rows: list[dict[str, Any]]) -> dict[str, str]:
    unit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        normalized_unit = str(row.get("normalized_unit") or "")
        if canonical_name and normalized_unit:
            unit_counts[canonical_name][normalized_unit] += 1
    return {
        canonical_name: counts.most_common(1)[0][0]
        for canonical_name, counts in unit_counts.items()
        if counts
    }


def _phase1_tensor_time_key(value: str) -> str:
    resolution, normalized, year, _month = _time_components(value)
    if normalized and normalized != "unknown":
        return normalized
    if year is not None:
        return f"{year:04d}-01"
    return ""


def _build_missing_mask(
    *,
    aligned_tensor: np.ndarray,
    normalized_rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    canonical_axis: list[str],
) -> np.ndarray:
    mask = np.zeros_like(aligned_tensor, dtype=np.float32)
    province_index = {str(name): idx for idx, name in enumerate(province_axis)}
    month_index = {str(name): idx for idx, name in enumerate(month_axis)}
    canonical_index = {str(name): idx for idx, name in enumerate(canonical_axis)}
    for row in normalized_rows:
        if row.get("model_numeric_value") is None:
            continue
        canonical_name = str(row.get("canonical_name") or "")
        if canonical_name not in canonical_index:
            continue
        geo_candidates = [
            normalize_geo_label(str(row.get("geo") or ""), default_country_focus=is_national_geo(str(row.get("geo") or ""))),
            normalize_geo_label(str(row.get("province") or ""), default_country_focus=False),
            normalize_geo_label(str(row.get("region") or ""), default_country_focus=False),
        ]
        province_name = next((candidate for candidate in geo_candidates if candidate and candidate in province_index), "")
        if not province_name and is_national_geo(str(row.get("geo") or "")) and "Philippines" in province_index:
            province_name = "Philippines"
        time_key = _phase1_tensor_time_key(str(row.get("time") or ""))
        if province_name and time_key in month_index:
            mask[province_index[province_name], month_index[time_key], canonical_index[canonical_name]] = 1.0
    return mask.astype(np.float32)


def _build_denominator_tensor(
    *,
    aligned_tensor: np.ndarray,
    normalized_rows: list[dict[str, Any]],
    canonical_axis: list[str],
) -> tuple[np.ndarray, dict[str, str]]:
    rules = _phase1_required_section("denominator_rules")
    min_denominator = _phase1_stabilizer("min_denominator")
    identity_features = {str(name) for name in rules.get("identity_features", [])}
    canonical_denominators = {str(key): str(value) for key, value in dict(rules.get("canonical_denominators", {})).items()}
    default_count_denominator = str(rules.get("default_count_denominator") or "")
    canonical_index = {str(name): idx for idx, name in enumerate(canonical_axis)}
    unit_index = _canonical_unit_index(normalized_rows)
    denominator_tensor = np.ones_like(aligned_tensor, dtype=np.float32)
    denominator_map: dict[str, str] = {}

    for canonical_name in canonical_axis:
        unit = unit_index.get(canonical_name, "")
        denominator_name = ""
        if canonical_name in identity_features:
            denominator_name = "identity"
        elif canonical_name in canonical_denominators:
            denominator_name = canonical_denominators[canonical_name]
        elif unit.startswith("count_") and default_count_denominator and canonical_name != default_count_denominator:
            denominator_name = default_count_denominator
        elif any(token in canonical_name for token in ("_rate", "_share", "_coverage", "_prevalence", "_index")):
            denominator_name = "identity"
        else:
            denominator_name = "identity"

        denominator_map[canonical_name] = denominator_name
        feature_idx = canonical_index[canonical_name]
        if denominator_name == "identity":
            denominator_tensor[:, :, feature_idx] = 1.0
            continue
        if denominator_name not in canonical_index:
            denominator_tensor[:, :, feature_idx] = 1.0
            denominator_map[canonical_name] = "identity_missing"
            continue
        source_idx = canonical_index[denominator_name]
        denominator_tensor[:, :, feature_idx] = np.clip(aligned_tensor[:, :, source_idx], a_min=min_denominator, a_max=None)
    return denominator_tensor.astype(np.float32), denominator_map


def _phase1_quality_weights(
    *,
    density_tensor: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = _phase1_required_section("missing_data")
    beta = float(cfg["quality_beta"])
    gamma = float(cfg["quality_gamma"])
    rho = float(cfg["temporal_decay_rho"])
    minimum_weight = float(cfg["minimum_weight"])
    observed = missing_mask > 0.5
    province_missing_rate = 1.0 - observed.mean(axis=(1, 2))
    density_nan = np.where(observed, density_tensor, np.nan)
    if density_tensor.shape[1] > 1:
        temporal_delta = np.diff(density_nan, axis=1)
        abs_delta = np.abs(temporal_delta)
        valid_delta = np.isfinite(abs_delta)
        delta_sum = np.where(valid_delta, abs_delta, 0.0).sum(axis=(1, 2))
        delta_count = valid_delta.sum(axis=(1, 2))
        province_instability = np.divide(
            delta_sum,
            np.clip(delta_count, a_min=1, a_max=None),
            dtype=np.float32,
        )
    else:
        abs_density = np.abs(density_nan)
        valid_density = np.isfinite(abs_density)
        density_sum = np.where(valid_density, abs_density, 0.0).sum(axis=(1, 2))
        density_count = valid_density.sum(axis=(1, 2))
        province_instability = np.divide(
            density_sum,
            np.clip(density_count, a_min=1, a_max=None),
            dtype=np.float32,
        )
    province_instability = np.nan_to_num(province_instability, nan=0.0, posinf=0.0, neginf=0.0)
    province_weight = 1.0 / (1.0 + beta * province_missing_rate + gamma * province_instability)
    province_weight = np.clip(province_weight, a_min=minimum_weight, a_max=1.0)
    month_distance = np.arange(density_tensor.shape[1] - 1, -1, -1, dtype=np.float32)
    temporal_decay = np.exp(-rho * month_distance)
    quality_weight_tensor = province_weight[:, None, None] * temporal_decay[None, :, None]
    quality_weight_tensor = np.repeat(quality_weight_tensor, density_tensor.shape[2], axis=2)
    meta = {
        "province_missing_rate": province_missing_rate.astype(np.float32).round(6).tolist(),
        "province_instability": province_instability.astype(np.float32).round(6).tolist(),
        "province_weight": province_weight.astype(np.float32).round(6).tolist(),
        "temporal_decay": temporal_decay.astype(np.float32).round(6).tolist(),
    }
    return quality_weight_tensor.astype(np.float32), meta


def _huber_scale_numpy(imputed: np.ndarray, observed: np.ndarray) -> np.ndarray:
    cfg = _phase1_preprocessing_cfg()
    eps = _phase1_stabilizer("huber_eps")
    delta = float(cfg["huber_delta"])
    steps = int(cfg["huber_steps"])
    center = np.nanmedian(np.where(observed, imputed, np.nan), axis=(0, 1), keepdims=True)
    center = np.nan_to_num(center, nan=0.0)
    scale = np.nanmedian(np.abs(np.where(observed, imputed, np.nan) - center), axis=(0, 1), keepdims=True)
    scale = np.nan_to_num(scale, nan=0.0)
    scale = np.clip(scale * 1.4826, a_min=eps, a_max=None)
    for _ in range(max(1, steps)):
        residual = imputed - center
        scaled = residual / scale
        weights = np.where(np.abs(scaled) <= delta, 1.0, delta / np.clip(np.abs(scaled), a_min=eps, a_max=None))
        weights = weights * observed.astype(np.float32)
        weight_sum = np.clip(weights.sum(axis=(0, 1), keepdims=True), a_min=eps, a_max=None)
        center = (weights * imputed).sum(axis=(0, 1), keepdims=True) / weight_sum
        abs_residual = np.abs(imputed - center)
        scale = (weights * abs_residual).sum(axis=(0, 1), keepdims=True) / weight_sum
        scale = np.clip(scale * 1.4826, a_min=eps, a_max=None)
    return np.clip((imputed - center) / scale, a_min=-6.0, a_max=6.0).astype(np.float32)


def _power_transform_numpy(density_nan: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = _phase1_preprocessing_cfg()
    transform_mode = str(cfg.get("transform_mode") or "log1p").lower()
    if transform_mode != "boxcox":
        return np.log1p(np.clip(density_nan, a_min=0.0, a_max=None)), {"transform_mode": "log1p"}
    lam = float(cfg["boxcox_lambda"])
    shift = float(cfg["boxcox_shift"])
    eps = _phase1_stabilizer("boxcox_eps")
    shifted = np.clip(density_nan + shift, a_min=eps, a_max=None)
    if abs(lam) <= 1e-8:
        transformed = np.log(shifted)
    else:
        transformed = (np.power(shifted, lam) - 1.0) / lam
    transformed = np.where(observed, transformed, np.nan)
    return transformed.astype(np.float32), {"transform_mode": "boxcox", "boxcox_lambda": lam, "boxcox_shift": shift}


def _power_transform_torch(density: "torch.Tensor", observed: "torch.Tensor") -> tuple["torch.Tensor", dict[str, Any]]:
    cfg = _phase1_preprocessing_cfg()
    transform_mode = str(cfg.get("transform_mode") or "log1p").lower()
    if transform_mode != "boxcox":
        return torch.log1p(torch.clamp(density, min=0.0)), {"transform_mode": "log1p"}
    lam = float(cfg["boxcox_lambda"])
    shift = float(cfg["boxcox_shift"])
    eps = _phase1_stabilizer("boxcox_eps")
    shifted = torch.clamp(density + shift, min=eps)
    if abs(lam) <= 1e-8:
        transformed = torch.log(shifted)
    else:
        transformed = (torch.pow(shifted, lam) - 1.0) / lam
    nan_fill = torch.full_like(transformed, float("nan"))
    transformed = torch.where(observed, transformed, nan_fill)
    return transformed, {"transform_mode": "boxcox", "boxcox_lambda": lam, "boxcox_shift": shift}


def _huber_scale_torch(imputed: "torch.Tensor", observed: "torch.Tensor") -> "torch.Tensor":
    cfg = _phase1_preprocessing_cfg()
    eps = _phase1_stabilizer("huber_eps")
    delta = float(cfg["huber_delta"])
    steps = int(cfg["huber_steps"])
    nan_fill = torch.full_like(imputed, float("nan"))
    observed_values = torch.where(observed, imputed, nan_fill)
    center = torch.nanmedian(observed_values.reshape(-1, observed_values.shape[-1]), dim=0).values.view(1, 1, -1)
    center = torch.nan_to_num(center, nan=0.0)
    scale = torch.nanmedian(torch.abs(observed_values - center).reshape(-1, observed_values.shape[-1]), dim=0).values.view(1, 1, -1)
    scale = torch.nan_to_num(scale, nan=0.0)
    scale = torch.clamp(scale * 1.4826, min=eps)
    observed_float = observed.to(imputed.dtype)
    for _ in range(max(1, steps)):
        residual = imputed - center
        scaled = residual / scale
        weights = torch.where(
            torch.abs(scaled) <= delta,
            torch.ones_like(scaled),
            delta / torch.clamp(torch.abs(scaled), min=eps),
        )
        weights = weights * observed_float
        weight_sum = torch.clamp(weights.sum(dim=(0, 1), keepdim=True), min=eps)
        center = (weights * imputed).sum(dim=(0, 1), keepdim=True) / weight_sum
        abs_residual = torch.abs(imputed - center)
        scale = (weights * abs_residual).sum(dim=(0, 1), keepdim=True) / weight_sum
        scale = torch.clamp(scale * 1.4826, min=eps)
    return torch.clamp((imputed - center) / scale, min=-6.0, max=6.0)


def _tensor_preprocess(
    aligned: np.ndarray,
    *,
    denominator_tensor: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if aligned.size == 0:
        empty = np.zeros_like(aligned, dtype=np.float32)
        return empty, empty, empty, empty, {"density_scale_default": True}

    backend = _phase1_compute_backend()
    winsor_cfg = _phase1_required_section("winsorization")
    min_denominator = _phase1_stabilizer("min_denominator")
    iqr_eps = _phase1_stabilizer("iqr_eps")
    preprocess_cfg = _phase1_preprocessing_cfg()

    if backend == "torch" and torch is not None:
        device = choose_torch_device(prefer_gpu=bool(preprocess_cfg.get("prefer_gpu", True)))
        aligned_t = to_torch_tensor(aligned, device=device, dtype=torch.float32)
        denominator_t = to_torch_tensor(denominator_tensor, device=device, dtype=torch.float32)
        missing_t = to_torch_tensor(missing_mask, device=device, dtype=torch.float32)
        observed_t = missing_t > 0.5
        density_t = aligned_t / torch.clamp(denominator_t, min=min_denominator)
        nan_fill = torch.full_like(density_t, float("nan"))
        density_nan_t = torch.where(observed_t, density_t, nan_fill)
        feature_has_observation_t = observed_t.any(dim=(0, 1), keepdim=True)
        density_nan_t = torch.where(feature_has_observation_t, density_nan_t, torch.zeros_like(density_nan_t))
        transformed_t, transform_meta = _power_transform_torch(density_nan_t, observed_t)
        flat_transformed_t = transformed_t.reshape(-1, transformed_t.shape[-1])
        q_low_t = torch.nanquantile(flat_transformed_t, float(winsor_cfg["lower_quantile"]), dim=0).view(1, 1, -1)
        q_high_t = torch.nanquantile(flat_transformed_t, float(winsor_cfg["upper_quantile"]), dim=0).view(1, 1, -1)
        q_low_t = torch.nan_to_num(q_low_t, nan=0.0)
        q_high_t = torch.nan_to_num(q_high_t, nan=0.0)
        winsorized_t = torch.minimum(torch.maximum(transformed_t, q_low_t), q_high_t)
        median_t = torch.nanmedian(winsorized_t.reshape(-1, winsorized_t.shape[-1]), dim=0).values.view(1, 1, -1)
        median_t = torch.nan_to_num(median_t, nan=0.0)
        imputed_t = torch.where(torch.isnan(winsorized_t), median_t, winsorized_t)
        scaling_mode = str(preprocess_cfg.get("scaling_mode") or "iqr").lower()
        if scaling_mode == "huber":
            standardized_t = _huber_scale_torch(imputed_t, observed_t)
            scale_meta = {"scaling_mode": "huber"}
        else:
            flat_imputed_t = imputed_t.reshape(-1, imputed_t.shape[-1])
            q75_t = torch.quantile(flat_imputed_t, 0.75, dim=0).view(1, 1, -1)
            q25_t = torch.quantile(flat_imputed_t, 0.25, dim=0).view(1, 1, -1)
            iqr_t = torch.clamp(q75_t - q25_t, min=iqr_eps)
            standardized_t = torch.clamp((imputed_t - median_t) / iqr_t, min=-6.0, max=6.0)
            scale_meta = {"scaling_mode": "iqr"}
        quality_weight_tensor, quality_meta = _phase1_quality_weights(density_tensor=to_numpy(density_t), missing_mask=to_numpy(missing_t))
        quality_t = to_torch_tensor(quality_weight_tensor, device=device, dtype=torch.float32)
        standardized_t = standardized_t * quality_t
        standardized_t = torch.where(observed_t, standardized_t, torch.zeros_like(standardized_t))
        density = density_t
        denominator_tensor = denominator_t
        standardized = standardized_t
        quality_weight_output = quality_t
        dlpack_report: dict[str, Any] = {"source": "torch", "target": "jax", "used_dlpack": False, "reason": "not_requested"}
        if str(preprocess_cfg.get("interop_mode") or "").lower() == "dlpack":
            try:
                _jax_view, dlpack_report = torch_to_jax_handoff(standardized_t, prefer_dlpack=True)
            except Exception as exc:  # pragma: no cover
                dlpack_report = {"source": "torch", "target": "jax", "used_dlpack": False, "reason": f"error:{type(exc).__name__}"}
    else:
        density = np.asarray(aligned, dtype=np.float32) / np.clip(np.asarray(denominator_tensor, dtype=np.float32), a_min=min_denominator, a_max=None)
        observed = np.asarray(missing_mask, dtype=np.float32) > 0.5
        density_nan = np.where(observed, density, np.nan)
        feature_has_observation = np.any(observed, axis=(0, 1), keepdims=True)
        density_nan = np.where(feature_has_observation, density_nan, 0.0)
        transformed, transform_meta = _power_transform_numpy(density_nan, observed)
        q_low = np.nanquantile(transformed, float(winsor_cfg["lower_quantile"]), axis=(0, 1), keepdims=True)
        q_high = np.nanquantile(transformed, float(winsor_cfg["upper_quantile"]), axis=(0, 1), keepdims=True)
        q_low = np.nan_to_num(q_low, nan=0.0)
        q_high = np.nan_to_num(q_high, nan=0.0)
        winsorized = np.clip(transformed, q_low, q_high)
        median = np.nanmedian(winsorized, axis=(0, 1), keepdims=True)
        median = np.nan_to_num(median, nan=0.0)
        imputed = np.where(np.isnan(winsorized), median, winsorized)
        scaling_mode = str(preprocess_cfg.get("scaling_mode") or "iqr").lower()
        if scaling_mode == "huber":
            standardized = _huber_scale_numpy(imputed, observed)
            scale_meta = {"scaling_mode": "huber"}
        else:
            q75 = np.nanquantile(imputed, 0.75, axis=(0, 1), keepdims=True)
            q25 = np.nanquantile(imputed, 0.25, axis=(0, 1), keepdims=True)
            q75 = np.nan_to_num(q75, nan=0.0)
            q25 = np.nan_to_num(q25, nan=0.0)
            iqr = np.clip(q75 - q25, a_min=iqr_eps, a_max=None)
            standardized = np.clip((imputed - median) / iqr, a_min=-6.0, a_max=6.0)
            scale_meta = {"scaling_mode": "iqr"}
        quality_weight_tensor, quality_meta = _phase1_quality_weights(density_tensor=density, missing_mask=missing_mask)
        standardized = standardized * quality_weight_tensor
        standardized = np.where(observed, standardized, 0.0)
        quality_weight_output = quality_weight_tensor.astype(np.float32)
        dlpack_report = {"source": "numpy", "target": "jax", "used_dlpack": False, "reason": "numpy_backend"}
    preprocess_meta = {
        "density_scale_default": False,
        "compute_backend": backend,
        "device": choose_torch_device(prefer_gpu=bool(preprocess_cfg.get("prefer_gpu", True))) if backend == "torch" and torch is not None else "cpu",
        "winsorization": {
            "lower_quantile": float(winsor_cfg["lower_quantile"]),
            "upper_quantile": float(winsor_cfg["upper_quantile"]),
        },
        "missing_imputation": "feature_median",
        "transform": transform_meta,
        "scaling": scale_meta,
        "interop": dlpack_report,
        "quality_weighting": quality_meta,
    }
    return (
        density,
        denominator_tensor,
        standardized,
        quality_weight_output,
        preprocess_meta,
    )


def run_phase1_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    plugin = get_disease_plugin(plugin_id)
    phase1_dir = ensure_dir(ctx.run_dir / "phase1")
    registry_path = ctx.run_dir / "registry" / "subparameter_registry.json"
    registry = read_json(registry_path, default={})
    subparameters = registry.get("subparameters", [])
    alignment_summary = read_json(ctx.run_dir / "phase0" / "extracted" / "alignment_summary.json", default={})
    aligned_tensor = load_tensor_artifact(ctx.run_dir / "phase0" / "extracted" / "aligned_tensor.npz")

    axis_catalogs = {
        "province": alignment_summary.get("province_axis", []),
        "month": alignment_summary.get("month_axis", []),
        "canonical_name": alignment_summary.get("canonical_name_axis", []),
    }

    normalized_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    extra_axis_values: dict[str, set[str]] = defaultdict(set)
    canonical_source_counts: Counter[tuple[str, str]] = Counter()
    catalog_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "canonical_name": "",
            "row_count": 0,
            "numeric_row_count": 0,
            "model_numeric_row_count": 0,
            "source_banks": Counter(),
            "units": Counter(),
            "geo_resolutions": Counter(),
            "regions": Counter(),
            "time_resolutions": Counter(),
            "domain_families": Counter(),
            "pathway_families": Counter(),
            "soft_ontology_tags": Counter(),
            "linkage_targets": Counter(),
            "evidence_classes": Counter(),
        }
    )

    for row in subparameters:
        raw_numeric_value = _safe_float(row.get("value"))
        normalized_unit = _normalize_unit(str(row.get("unit") or ""))
        model_numeric_value = _normalize_numeric_value(raw_numeric_value, normalized_unit)
        blob = _text_blob(row)
        time_resolution, normalized_time, year, month = _time_components(str(row.get("time") or ""))
        sex = row.get("sex") or _infer_from_tokens(blob, SEX_TOKENS, default="")
        age_band = row.get("age_band") or _infer_from_tokens(blob, AGE_TOKENS, default="")
        kp_group = row.get("kp_group") or _infer_from_tokens(blob, KP_TOKENS, default="remaining_population")
        geo = normalize_geo_label(str(row.get("geo") or ""), default_country_focus="philippines" in blob) or ("Philippines" if "philippines" in blob else "")
        geo_match = infer_philippines_geo(f"{geo} {blob}", default_country_focus="philippines" in blob)
        geo_resolution = _geo_resolution(str(geo))
        region = str(row.get("region") or geo_match.region or _infer_region(str(geo), blob))
        province = str(row.get("province") or geo_match.province or (geo if geo_resolution in {"province", "city"} else ""))
        domain_family = _infer_domain_family(row, blob)
        pathway_family = _infer_pathway_family(row, blob)
        evidence_class = _evidence_class(row)
        evidence_weight = _evidence_weight(row, evidence_class)
        reliability_class = _source_reliability_class(row, evidence_class)
        bias_fields = _bias_fields(row, reliability_class, time_resolution, geo_resolution)
        normalized = {
            "normalized_id": row.get("subparameter_id") or row.get("candidate_id"),
            "canonical_name": row.get("canonical_name") or "unknown",
            "candidate_text": row.get("candidate_text") or row.get("parameter_text") or "",
            "raw_numeric_value": raw_numeric_value,
            "model_numeric_value": model_numeric_value,
            "raw_value": row.get("value"),
            "original_unit": row.get("unit") or "",
            "normalized_unit": normalized_unit,
            "geo": geo,
            "geo_resolution": geo_resolution,
            "country": "Philippines" if geo_resolution in {"national", "region", "province", "city"} else ("global" if geo_resolution == "global" else ""),
            "region": region,
            "province": province,
            "time": normalized_time,
            "time_resolution": time_resolution,
            "year": year,
            "month": month,
            "sex": sex,
            "age_band": age_band,
            "kp_group": kp_group,
            "domain_family": domain_family,
            "pathway_family": pathway_family,
            "source_bank": row.get("source_bank") or "",
            "source_id": row.get("source_id") or "",
            "evidence_class": evidence_class,
            "evidence_weight": evidence_weight,
            "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
            "is_direct_measurement": bool(row.get("is_direct_measurement")),
            "is_prior_only": bool(row.get("is_prior_only")),
            "soft_ontology_tags": list(row.get("soft_ontology_tags") or []),
            "soft_subparameter_hints": list(row.get("soft_subparameter_hints") or []),
            "linkage_targets": list(row.get("linkage_targets") or []),
            "literature_ref_details": list(row.get("literature_ref_details") or []),
            **bias_fields,
        }
        normalized_rows.append(normalized)
        canonical_source_counts[(normalized["canonical_name"], str(normalized["source_id"]))] += 1
        roll = catalog_rollup[normalized["canonical_name"]]
        roll["canonical_name"] = normalized["canonical_name"]
        roll["row_count"] += 1
        roll["numeric_row_count"] += 1 if raw_numeric_value is not None else 0
        roll["model_numeric_row_count"] += 1 if model_numeric_value is not None else 0
        roll["source_banks"][normalized["source_bank"]] += 1
        roll["units"][normalized["normalized_unit"]] += 1
        roll["geo_resolutions"][normalized["geo_resolution"]] += 1
        if normalized["region"]:
            roll["regions"][normalized["region"]] += 1
        roll["time_resolutions"][normalized["time_resolution"]] += 1
        roll["domain_families"][normalized["domain_family"]] += 1
        roll["pathway_families"][normalized["pathway_family"]] += 1
        roll["evidence_classes"][normalized["evidence_class"]] += 1
        for tag in normalized["soft_ontology_tags"]:
            roll["soft_ontology_tags"][tag] += 1
        for target in normalized["linkage_targets"]:
            roll["linkage_targets"][target] += 1
        for axis_name in ("geo_resolution", "geo", "country", "region", "time_resolution", "sex", "age_band", "kp_group", "domain_family", "pathway_family", "evidence_class", "source_bank", "normalized_unit"):
            value = normalized.get(axis_name)
            if value not in {"", None}:
                extra_axis_values[axis_name].add(str(value))

    for row in normalized_rows:
        key = (row["canonical_name"], str(row.get("source_id") or ""))
        replication_cfg = _phase1_required_section("replication_weight")
        replication_weight = min(
            float(replication_cfg["ceiling"]),
            float(replication_cfg["base"]) + float(replication_cfg["step"]) * canonical_source_counts[key],
        )
        row["replication_weight"] = round(float(replication_weight), 4)
        row["bias_penalty"] = round(
            float(
                max(
                    0.0,
                    row["bias_penalty"]
                    - float(replication_cfg["bias_penalty_discount"])
                    * min(
                        float(replication_cfg["ceiling"]),
                        max(0.0, replication_weight - float(replication_cfg["bias_penalty_reference"])),
                    ),
                )
            ),
            4,
        )

    province_axis = [normalize_geo_label(str(item), default_country_focus=is_national_geo(str(item))) or str(item) for item in (axis_catalogs["province"] or ["Philippines"])]
    province_axis = list(dict.fromkeys(province_axis))
    axis_catalogs["province"] = province_axis
    month_axis = axis_catalogs["month"] or ["unknown"]
    canonical_axis = axis_catalogs["canonical_name"] or ["numeric_observation"]
    missing_mask = _build_missing_mask(
        aligned_tensor=aligned_tensor,
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        canonical_axis=canonical_axis,
    )
    denominator_tensor, denominator_map = _build_denominator_tensor(
        aligned_tensor=aligned_tensor,
        normalized_rows=normalized_rows,
        canonical_axis=canonical_axis,
    )
    density_tensor, denominator_tensor, standardized_tensor, quality_weight_tensor, preprocess_meta = _tensor_preprocess(
        aligned_tensor,
        denominator_tensor=denominator_tensor,
        missing_mask=missing_mask,
    )
    denominator_tensor_np = to_numpy(denominator_tensor)
    standardized_tensor_np = to_numpy(standardized_tensor)
    quality_weight_tensor_np = to_numpy(quality_weight_tensor)
    for pi, province in enumerate(province_axis):
        for ti, month in enumerate(month_axis):
            for fi, canonical_name in enumerate(canonical_axis):
                tensor_rows.append(
                    {
                        "tensor_row_id": f"{province}:{month}:{canonical_name}",
                        "canonical_name": canonical_name,
                        "model_numeric_value": float(standardized_tensor_np[pi, ti, fi]),
                        "raw_numeric_value": float(aligned_tensor[pi, ti, fi]),
                        "normalized_unit": "standard_score",
                        "geo_resolution": geo_resolution_label(province),
                        "geo": province,
                        "country": "Philippines" if province != "global" else "global",
                        "region": infer_region_code(province),
                        "province": province,
                        "time_resolution": "monthly",
                        "time": month,
                        "year": int(month.split("-")[0]) if re.fullmatch(r"\d{4}-\d{2}", month) else None,
                        "month": int(month.split("-")[1]) if re.fullmatch(r"\d{4}-\d{2}", month) else None,
                        "sex": "",
                        "age_band": "",
                        "kp_group": "remaining_population",
                        "domain_family": next(iter(catalog_rollup.get(canonical_name, {}).get("domain_families", {"mixed": 1})), "mixed"),
                        "pathway_family": next(iter(catalog_rollup.get(canonical_name, {}).get("pathway_families", {"mixed": 1})), "mixed"),
                        "evidence_class": "phase1_standardized_tensor",
                        "evidence_weight": 1.0,
                        "source_bank": "phase1_standardized_tensor",
                        "missing_mask": float(missing_mask[pi, ti, fi]),
                        "quality_weight": float(quality_weight_tensor_np[pi, ti, fi]),
                    }
                )

    parameter_catalog = []
    for canonical_name, roll in sorted(catalog_rollup.items()):
        parameter_catalog.append(
            {
                "canonical_name": canonical_name,
                "row_count": roll["row_count"],
                "numeric_row_count": roll["numeric_row_count"],
                "model_numeric_row_count": roll["model_numeric_row_count"],
                "source_banks": dict(roll["source_banks"]),
                "units": dict(roll["units"]),
                "geo_resolutions": dict(roll["geo_resolutions"]),
                "regions": dict(roll["regions"]),
                "time_resolutions": dict(roll["time_resolutions"]),
                "domain_families": dict(roll["domain_families"]),
                "pathway_families": dict(roll["pathway_families"]),
                "soft_ontology_tags": dict(roll["soft_ontology_tags"]),
                "linkage_targets": dict(roll["linkage_targets"]),
                "evidence_classes": dict(roll["evidence_classes"]),
            }
        )

    axis_catalogs.update({axis: sorted(values) for axis, values in sorted(extra_axis_values.items())})
    normalization_report = {
        "normalized_row_count": len(normalized_rows),
        "tensor_row_count": len(tensor_rows),
        "model_numeric_coverage": round(sum(1 for row in normalized_rows if row["model_numeric_value"] is not None) / len(normalized_rows), 4) if normalized_rows else 0.0,
        "direct_measurement_count": sum(1 for row in normalized_rows if row["is_direct_measurement"]),
        "anchor_eligible_count": sum(1 for row in normalized_rows if row["is_anchor_eligible"]),
        "source_reliability_counts": dict(Counter(row["source_reliability_class"] for row in normalized_rows)),
        "promotion_hint_counts": dict(Counter(row["promotion_eligibility_hint"] for row in normalized_rows)),
        "domain_family_counts": dict(Counter(row["domain_family"] for row in normalized_rows)),
        "pathway_family_counts": dict(Counter(row["pathway_family"] for row in normalized_rows)),
        "kp_group_counts": dict(Counter(row["kp_group"] for row in normalized_rows if row["kp_group"])),
        "sex_counts": dict(Counter(row["sex"] for row in normalized_rows if row["sex"])),
        "age_band_counts": dict(Counter(row["age_band"] for row in normalized_rows if row["age_band"])),
        "missing_mask_fraction": round(float(1.0 - missing_mask.mean()), 6) if missing_mask.size else 0.0,
        "denominator_map": denominator_map,
        "preprocess_meta": preprocess_meta,
    }

    backend_map = detect_backends()
    preprocess_backend = str(preprocess_meta.get("compute_backend") or "numpy")
    preprocess_device = str(preprocess_meta.get("device") or "cpu")
    standardized_artifact = save_tensor_artifact(
        array=standardized_tensor,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="standardized_tensor",
        backend=preprocess_backend,
        device=preprocess_device,
        notes=["phase1_standardized_tensor"],
    )
    denominator_artifact = save_tensor_artifact(
        array=denominator_tensor,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="denominator_tensor",
        backend=preprocess_backend,
        device=preprocess_device,
        notes=["phase1_denominator_tensor"],
    )
    missing_mask_artifact = save_tensor_artifact(
        array=missing_mask,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="missing_mask",
        backend=preprocess_backend,
        device=preprocess_device,
        notes=["phase1_missing_mask"],
    )
    quality_weight_artifact = save_tensor_artifact(
        array=quality_weight_tensor,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="quality_weight_tensor",
        backend=preprocess_backend,
        device=preprocess_device,
        notes=["phase1_quality_weight_tensor"],
    )
    write_json(phase1_dir / "interop_report.json", dict(preprocess_meta.get("interop", {})))

    write_json(phase1_dir / "normalized_subparameters.json", normalized_rows)
    write_json(phase1_dir / "parameter_catalog.json", parameter_catalog)
    write_json(phase1_dir / "axis_catalogs.json", axis_catalogs)
    write_json(phase1_dir / "tensor_rows.json", tensor_rows)
    tensor_schema = {
        "axes": axis_catalogs,
        "value_fields": {
            "aligned_tensor": "phase0/aligned_tensor",
            "standardized_tensor": "phase1/standardized_tensor",
            "denominator_tensor": "phase1/denominator_tensor",
            "missing_mask": "phase1/missing_mask",
            "quality_weight_tensor": "phase1/quality_weight_tensor",
            "raw_numeric_value": "raw_numeric_value",
            "model_numeric_value": "model_numeric_value",
        },
        "default_value_field": "standardized_tensor",
    }
    write_json(phase1_dir / "tensor_schema.json", tensor_schema)
    write_json(phase1_dir / "normalization_report.json", normalization_report)

    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase0" / "parsed"),
        extracted_dir=str(phase1_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase1": "completed"},
        artifact_paths={
            "standardized_tensor": standardized_artifact["value_path"],
            "denominator_tensor": denominator_artifact["value_path"],
            "missing_mask": missing_mask_artifact["value_path"],
            "quality_weight_tensor": quality_weight_artifact["value_path"],
            "normalized_subparameters": str(phase1_dir / "normalized_subparameters.json"),
            "parameter_catalog": str(phase1_dir / "parameter_catalog.json"),
            "axis_catalogs": str(phase1_dir / "axis_catalogs.json"),
            "tensor_rows": str(phase1_dir / "tensor_rows.json"),
            "tensor_schema": str(phase1_dir / "tensor_schema.json"),
            "normalization_report": str(phase1_dir / "normalization_report.json"),
            "interop_report": str(phase1_dir / "interop_report.json"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, backend_map["torch"].selected, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        source_count=registry.get("subparameter_count", 0),
        canonical_candidate_count=len(normalized_rows),
        numeric_observation_count=len(tensor_rows),
        notes=["phase1_preprocessing:robust_tensor_scaling"],
    ).to_dict()
    manifest["profile_id"] = profile
    truth_paths = write_ground_truth_package(
        phase_dir=phase1_dir,
        phase_name="phase1",
        profile_id=profile,
        checks=[
            {"name": "normalized_rows_present", "passed": bool(normalized_rows)},
            {"name": "standardized_tensor_finite", "passed": bool(np.isfinite(standardized_tensor_np).all())},
            {"name": "denominator_tensor_finite", "passed": bool(np.isfinite(denominator_tensor_np).all())},
            {"name": "missing_mask_present", "passed": bool(np.isfinite(missing_mask).all())},
            {
                "name": "no_duplicate_national_labels",
                "passed": not (("Philippines" in province_axis) and any(is_national_geo(name) for name in province_axis if name != "Philippines")),
            },
            {
                "name": "truth_fields_present",
                "passed": all("source_reliability_class" in row and "bias_penalty" in row for row in normalized_rows),
            },
        ],
        truth_sources=sorted(set(str(row.get("source_reliability_class") or "") for row in normalized_rows if row.get("source_reliability_class"))),
        stage_manifest_path=str(phase1_dir / "phase1_manifest.json"),
        summary={
            "normalized_row_count": len(normalized_rows),
            "tensor_row_count": len(tensor_rows),
            "province_count": len(province_axis),
            "month_count": len(month_axis),
            "canonical_name_count": len(canonical_axis),
            "anchor_eligible_count": int(sum(1 for row in normalized_rows if row.get("is_anchor_eligible"))),
        },
    )
    gold_profile = dict((plugin.gold_standard_profiles or {}).get("phase1", {}) or {})
    gold_paths = write_gold_standard_package(
        phase_dir=phase1_dir,
        phase_name="phase1",
        profile_id=profile,
        gold_profile=gold_profile,
        checks=[
            {"name": "gold_standard_profile_declared", "passed": bool(gold_profile)},
            {"name": "measurement_uncertainty_fields_present", "passed": all("source_reliability_class" in row and "bias_penalty" in row for row in normalized_rows)},
            {"name": "denominator_alignment_present", "passed": bool(np.isfinite(denominator_tensor_np).all()) and denominator_tensor_np.shape == standardized_tensor_np.shape and bool(np.any(denominator_tensor_np != 1.0))},
            {"name": "missing_mask_emitted", "passed": missing_mask.shape == standardized_tensor_np.shape},
            {"name": "robust_scaling_documented", "passed": "phase1_preprocessing:robust_tensor_scaling" in manifest.get("notes", []) and bool(normalization_report)},
            {
                "name": "national_label_ambiguity_absent",
                "passed": not (("Philippines" in province_axis) and any(is_national_geo(name) for name in province_axis if name != "Philippines")),
            },
        ],
        stage_manifest_path=str(phase1_dir / "phase1_manifest.json"),
        summary={
            "normalized_row_count": len(normalized_rows),
            "tensor_row_count": len(tensor_rows),
            "anchor_eligible_count": int(sum(1 for row in normalized_rows if row.get("is_anchor_eligible"))),
        },
    )
    boundary_paths = write_boundary_shape_package(
        phase_dir=phase1_dir,
        phase_name="phase1",
        profile_id=profile,
        boundaries=[
            {
                "name": "standardized_tensor",
                "kind": "tensor",
                "path": str(phase1_dir / "standardized_tensor.npz"),
                "expected_shape": list(standardized_tensor_np.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "denominator_tensor",
                "kind": "tensor",
                "path": str(phase1_dir / "denominator_tensor.npz"),
                "expected_shape": list(denominator_tensor_np.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "missing_mask",
                "kind": "tensor",
                "path": str(phase1_dir / "missing_mask.npz"),
                "expected_shape": list(missing_mask.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "quality_weight_tensor",
                "kind": "tensor",
                "path": str(phase1_dir / "quality_weight_tensor.npz"),
                "expected_shape": list(quality_weight_tensor_np.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "normalized_subparameters",
                "kind": "json_rows",
                "path": str(phase1_dir / "normalized_subparameters.json"),
                "min_rows": 1,
                "expected_row_count": len(normalized_rows),
            },
            {
                "name": "tensor_rows",
                "kind": "json_rows",
                "path": str(phase1_dir / "tensor_rows.json"),
                "min_rows": 1,
                "expected_row_count": len(tensor_rows),
            },
            {
                "name": "tensor_schema",
                "kind": "json_dict",
                "path": str(phase1_dir / "tensor_schema.json"),
                "expected_keys": ["axes", "value_fields", "default_value_field"],
            },
        ],
        summary={
            "normalized_row_count": len(normalized_rows),
            "tensor_row_count": len(tensor_rows),
            "province_count": len(province_axis),
            "month_count": len(month_axis),
        },
    )
    manifest["artifact_paths"].update(gold_paths)
    manifest["artifact_paths"].update(truth_paths)
    manifest["artifact_paths"].update(boundary_paths)
    write_json(phase1_dir / "phase1_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase1_build",
        [
            phase1_dir / "standardized_tensor.npz",
            phase1_dir / "denominator_tensor.npz",
            phase1_dir / "missing_mask.npz",
            phase1_dir / "quality_weight_tensor.npz",
            phase1_dir / "normalized_subparameters.json",
            phase1_dir / "parameter_catalog.json",
            phase1_dir / "axis_catalogs.json",
            phase1_dir / "tensor_rows.json",
            phase1_dir / "tensor_schema.json",
            phase1_dir / "normalization_report.json",
            phase1_dir / "interop_report.json",
            phase1_dir / "gold_standard_manifest.json",
            phase1_dir / "gold_standard_checks.json",
            phase1_dir / "gold_standard_summary.json",
            phase1_dir / "boundary_shape_manifest.json",
            phase1_dir / "boundary_shape_checks.json",
            phase1_dir / "boundary_shape_summary.json",
            phase1_dir / "phase1_manifest.json",
        ],
    )
    return manifest
