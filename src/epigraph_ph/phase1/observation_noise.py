from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _observation_noise_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase1_cfg = dict((plugin.constraint_settings or {}).get("phase1", {}) or {})
    cfg = dict(phase1_cfg.get("observation_noise", {}) or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "residual_eps": float(cfg.get("residual_eps") or 1e-4),
        "ridge_penalty": float(cfg.get("ridge_penalty") or 0.15),
        "precision_floor": float(cfg.get("precision_floor") or 0.05),
        "precision_ceiling": float(cfg.get("precision_ceiling") or 25.0),
        "weight_floor": float(cfg.get("weight_floor") or 0.1),
        "weight_ceiling": float(cfg.get("weight_ceiling") or 4.0),
        "minimum_exact_group_count": int(cfg.get("minimum_exact_group_count") or 2),
        "minimum_family_group_count": int(cfg.get("minimum_family_group_count") or 3),
        "minimum_canonical_group_count": int(cfg.get("minimum_canonical_group_count") or 5),
    }


def _transform_kind(canonical_name: str, values: np.ndarray) -> str:
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
            "proportion",
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
        if p10 > 0.0 and p90 / p10 >= 8.0:
            return "log1p"
    return "identity"


def _apply_transform(value: float, transform_kind: str, clip_eps: float) -> float:
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


def _row_geo_key(row: Mapping[str, Any]) -> str:
    for field in ("province", "geo", "region"):
        token = str(row.get(field) or "").strip()
        if token:
            return token.lower()
    return "unknown"


def _row_time_key(row: Mapping[str, Any]) -> str:
    token = str(row.get("time") or "").strip()
    if token:
        return token
    year = row.get("year")
    if isinstance(year, int):
        return f"{year:04d}"
    if isinstance(year, str) and year.isdigit():
        return year
    return "unknown"


def _row_feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "measurement_role": str(row.get("measurement_role") or "unknown"),
        "source_reliability_class": str(row.get("source_reliability_class") or "unknown"),
        "geo_resolution": str(row.get("geo_resolution") or "unknown"),
        "time_resolution": str(row.get("time_resolution") or "unknown"),
        "anchor_status": "anchor" if bool(row.get("is_anchor_eligible")) else "non_anchor",
        "direct_status": "direct" if bool(row.get("is_direct_measurement")) else "indirect",
        "measurement_quality_weight": _safe_float(row.get("measurement_quality_weight"), default=0.0),
        "temporal_freshness_weight": _safe_float(row.get("temporal_freshness_weight"), default=0.0),
        "spatial_relevance_weight": _safe_float(row.get("spatial_relevance_weight"), default=0.0),
        "replication_weight": _safe_float(row.get("replication_weight"), default=0.0),
        "evidence_weight": _safe_float(row.get("evidence_weight"), default=0.0),
    }


def _fit_ridge(design: np.ndarray, target: np.ndarray, ridge_penalty: float) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else 0,), dtype=np.float64)
    gram = x.T @ x + float(ridge_penalty) * np.eye(x.shape[1], dtype=np.float64)
    rhs = x.T @ y
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
    return np.asarray(beta, dtype=np.float64)


def build_observation_noise_model(
    *,
    normalized_rows: list[dict[str, Any]],
    plugin_id: str,
) -> dict[str, Any]:
    cfg = _observation_noise_cfg(plugin_id)
    if not cfg["enabled"]:
        return {"enabled": False, "rows": [], "coefficient_rows": [], "summary": {"row_count": 0}}

    numeric_rows: list[dict[str, Any]] = []
    values_by_canonical: dict[str, list[float]] = defaultdict(list)
    for row in normalized_rows:
        numeric = row.get("model_numeric_value")
        if numeric in (None, ""):
            continue
        numeric_float = _safe_float(numeric, default=float("nan"))
        if not np.isfinite(numeric_float):
            continue
        row_copy = dict(row)
        row_copy["_numeric_value"] = float(numeric_float)
        numeric_rows.append(row_copy)
        values_by_canonical[str(row_copy.get("canonical_name") or "unknown")].append(float(numeric_float))

    if not numeric_rows:
        return {"enabled": True, "rows": [], "coefficient_rows": [], "summary": {"row_count": 0}}

    transform_stats: dict[str, dict[str, float | str]] = {}
    for canonical_name, values in values_by_canonical.items():
        array = np.asarray(values, dtype=np.float64)
        kind = _transform_kind(canonical_name, array)
        transformed = np.asarray(
            [_apply_transform(float(value), kind, clip_eps=1e-3) for value in array],
            dtype=np.float64,
        )
        center = float(np.median(transformed))
        mad = float(np.median(np.abs(transformed - center)))
        scale = max(mad * 1.4826, float(np.std(transformed)), float(cfg["residual_eps"]))
        transform_stats[canonical_name] = {
            "transform_kind": kind,
            "center": center,
            "scale": scale,
        }

    exact_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    family_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    canonical_groups: dict[str, list[float]] = defaultdict(list)
    for row in numeric_rows:
        canonical_name = str(row.get("canonical_name") or "unknown")
        transform_row = transform_stats[canonical_name]
        transformed = _apply_transform(float(row["_numeric_value"]), str(transform_row["transform_kind"]), clip_eps=1e-3)
        row["_transformed_value"] = transformed
        geo_key = _row_geo_key(row)
        time_key = _row_time_key(row)
        exact_groups[(canonical_name, geo_key, time_key)].append(transformed)
        family_groups[(canonical_name, str(row.get("geo_resolution") or "unknown"), str(row.get("time_resolution") or "unknown"))].append(transformed)
        canonical_groups[canonical_name].append(transformed)

    categorical_vocab: dict[str, list[str]] = defaultdict(list)
    row_records: list[dict[str, Any]] = []
    residual_targets: list[float] = []
    for row in numeric_rows:
        canonical_name = str(row.get("canonical_name") or "unknown")
        transformed = float(row["_transformed_value"])
        exact_values = exact_groups[(canonical_name, _row_geo_key(row), _row_time_key(row))]
        family_values = family_groups[
            (
                canonical_name,
                str(row.get("geo_resolution") or "unknown"),
                str(row.get("time_resolution") or "unknown"),
            )
        ]
        canonical_values = canonical_groups[canonical_name]
        support_level = "low_support_prior"
        if len(exact_values) >= int(cfg["minimum_exact_group_count"]):
            baseline = float(np.mean([value for value in exact_values if value != transformed] or exact_values))
            support_level = "exact_cell"
        elif len(family_values) >= int(cfg["minimum_family_group_count"]):
            baseline = float(np.mean([value for value in family_values if value != transformed] or family_values))
            support_level = "family_bucket"
        elif len(canonical_values) >= int(cfg["minimum_canonical_group_count"]):
            baseline = float(np.mean([value for value in canonical_values if value != transformed] or canonical_values))
            support_level = "canonical_global"
        else:
            baseline = float(transform_stats[canonical_name]["center"])
        residual = transformed - baseline
        log_variance_target = float(np.log(float(residual**2) + float(cfg["residual_eps"])))
        features = _row_feature_payload(row)
        features["support_level"] = support_level
        for field in (
            "measurement_role",
            "source_reliability_class",
            "geo_resolution",
            "time_resolution",
            "anchor_status",
            "direct_status",
            "support_level",
        ):
            token = str(features[field])
            if token not in categorical_vocab[field]:
                categorical_vocab[field].append(token)
        row_records.append(
            {
                "row": row,
                "canonical_name": canonical_name,
                "features": features,
                "baseline_value": baseline,
                "transformed_value": transformed,
                "residual": residual,
                "support_level": support_level,
            }
        )
        residual_targets.append(log_variance_target)

    continuous_fields = [
        "measurement_quality_weight",
        "temporal_freshness_weight",
        "spatial_relevance_weight",
        "replication_weight",
        "evidence_weight",
    ]
    categorical_fields = [
        "measurement_role",
        "source_reliability_class",
        "geo_resolution",
        "time_resolution",
        "anchor_status",
        "direct_status",
        "support_level",
    ]
    column_names = ["intercept"]
    for field in continuous_fields:
        column_names.append(field)
    for field in categorical_fields:
        for token in categorical_vocab[field]:
            column_names.append(f"{field}:{token}")

    design = np.zeros((len(row_records), len(column_names)), dtype=np.float64)
    design[:, 0] = 1.0
    continuous_stats: dict[str, tuple[float, float]] = {}
    for field_idx, field in enumerate(continuous_fields, start=1):
        values = np.asarray([float(record["features"][field]) for record in row_records], dtype=np.float64)
        center = float(np.mean(values))
        scale = max(float(np.std(values)), float(cfg["residual_eps"]))
        continuous_stats[field] = (center, scale)
        design[:, field_idx] = (values - center) / scale
    offset = 1 + len(continuous_fields)
    for field in categorical_fields:
        for token in categorical_vocab[field]:
            design[:, offset] = np.asarray([1.0 if str(record["features"][field]) == token else 0.0 for record in row_records], dtype=np.float64)
            offset += 1

    beta = _fit_ridge(design, np.asarray(residual_targets, dtype=np.float64), float(cfg["ridge_penalty"]))
    predicted_log_variance = design @ beta
    precision_values = np.clip(
        np.exp(-predicted_log_variance),
        float(cfg["precision_floor"]),
        float(cfg["precision_ceiling"]),
    )
    median_precision = float(np.median(precision_values[np.isfinite(precision_values)])) if np.any(np.isfinite(precision_values)) else 1.0
    median_precision = max(median_precision, float(cfg["precision_floor"]))
    observation_weights = np.clip(
        precision_values / median_precision,
        float(cfg["weight_floor"]),
        float(cfg["weight_ceiling"]),
    )

    rows: list[dict[str, Any]] = []
    support_counter = Counter()
    for row_idx, record in enumerate(row_records):
        row = record["row"]
        row_id = str(row.get("normalized_id") or row.get("candidate_id") or row_idx)
        support_counter[str(record["support_level"])] += 1
        rows.append(
            {
                "normalized_id": row_id,
                "canonical_name": str(record["canonical_name"]),
                "support_level": str(record["support_level"]),
                "baseline_value": round(float(record["baseline_value"]), 6),
                "transformed_value": round(float(record["transformed_value"]), 6),
                "residual": round(float(record["residual"]), 6),
                "observation_noise_log_variance": round(float(predicted_log_variance[row_idx]), 6),
                "observation_noise_variance": round(float(np.exp(predicted_log_variance[row_idx])), 6),
                "observation_precision": round(float(precision_values[row_idx]), 6),
                "observation_weight": round(float(observation_weights[row_idx]), 6),
                "measurement_role": str(record["features"]["measurement_role"]),
                "source_reliability_class": str(record["features"]["source_reliability_class"]),
                "geo_resolution": str(record["features"]["geo_resolution"]),
                "time_resolution": str(record["features"]["time_resolution"]),
            }
        )

    coefficient_rows = [
        {
            "feature": str(name),
            "coefficient": round(float(value), 6),
        }
        for name, value in zip(column_names, beta.tolist())
    ]
    summary = {
        "row_count": len(rows),
        "median_precision": round(float(median_precision), 6),
        "support_level_counts": dict(sorted(support_counter.items())),
        "precision_min": round(float(np.min(precision_values)), 6),
        "precision_max": round(float(np.max(precision_values)), 6),
        "weight_min": round(float(np.min(observation_weights)), 6),
        "weight_max": round(float(np.max(observation_weights)), 6),
        "continuous_feature_stats": {
            field: {"center": round(float(center), 6), "scale": round(float(scale), 6)}
            for field, (center, scale) in continuous_stats.items()
        },
    }
    return {
        "enabled": True,
        "rows": rows,
        "coefficient_rows": coefficient_rows,
        "summary": summary,
    }


__all__ = ["build_observation_noise_model"]
