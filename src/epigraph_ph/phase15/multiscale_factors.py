from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from epigraph_ph.geography import infer_philippines_geo, is_national_geo


def _zscore_surface(surface: np.ndarray, *, eps: float) -> np.ndarray:
    array = np.asarray(surface, dtype=np.float32)
    if array.size == 0:
        return array
    mean = float(array.mean())
    std = float(array.std())
    centered = array - mean
    if std <= eps:
        return centered.astype(np.float32)
    return (centered / std).astype(np.float32)


def _region_groups(province_axis: list[str]) -> tuple[list[str], list[list[int]], list[int]]:
    region_to_indices: dict[str, list[int]] = defaultdict(list)
    non_national_indices: list[int] = []
    for province_idx, province_name in enumerate(province_axis):
        if is_national_geo(province_name):
            continue
        non_national_indices.append(province_idx)
        match = infer_philippines_geo(province_name, default_country_focus=False)
        region_label = match.region_display or match.region or "Philippines"
        region_to_indices[region_label].append(province_idx)
    if not region_to_indices:
        fallback = non_national_indices or list(range(len(province_axis)))
        region_to_indices["Philippines"] = fallback
    region_axis = sorted(region_to_indices.keys())
    return region_axis, [region_to_indices[name] for name in region_axis], non_national_indices


def _aggregate_surface(surface: np.ndarray, groups: list[list[int]], *, eps: float) -> np.ndarray:
    if surface.size == 0:
        return np.zeros((len(groups), 0), dtype=np.float32)
    rows = []
    for indices in groups:
        if not indices:
            rows.append(np.zeros((surface.shape[1],), dtype=np.float32))
            continue
        rows.append(np.asarray(surface[indices, :], dtype=np.float32).mean(axis=0))
    return _zscore_surface(np.stack(rows, axis=0).astype(np.float32), eps=eps)


def _national_surface(surface: np.ndarray, non_national_indices: list[int], *, eps: float) -> np.ndarray:
    if surface.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    indices = non_national_indices or list(range(surface.shape[0]))
    aggregated = np.asarray(surface[indices, :], dtype=np.float32).mean(axis=0, keepdims=True)
    return _zscore_surface(aggregated, eps=eps)


def _aggregate_uncertainty_surface(surface: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    if surface.size == 0:
        return np.zeros((len(groups), 0), dtype=np.float32)
    rows = []
    for indices in groups:
        if not indices:
            rows.append(np.zeros((surface.shape[1],), dtype=np.float32))
            continue
        rows.append(np.asarray(surface[indices, :], dtype=np.float32).mean(axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def _national_uncertainty_surface(surface: np.ndarray, non_national_indices: list[int]) -> np.ndarray:
    if surface.size == 0:
        return np.zeros((1, 0), dtype=np.float32)
    indices = non_national_indices or list(range(surface.shape[0]))
    return np.asarray(surface[indices, :], dtype=np.float32).mean(axis=0, keepdims=True).astype(np.float32)


def _relationship_adjacency(relationship_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relationship_rows:
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if not left or not right:
            continue
        entry = {
            "left": left,
            "right": right,
            "block_name": str(row.get("block_name") or ""),
            "semantic_index": float(row.get("semantic_index") or 0.0),
            "similarity_score": float(row.get("similarity_score") or 0.0),
            "corr_sim": float(row.get("corr_sim") or 0.0),
            "semantic_sim": float(row.get("semantic_sim") or 0.0),
            "target_sim": float(row.get("target_sim") or 0.0),
            "geo_sim": float(row.get("geo_sim") or 0.0),
        }
        adjacency[left].append(entry)
        adjacency[right].append(entry)
    return adjacency


def _member_weights_for_factor(
    *,
    factor_idx: int,
    member_names: list[str],
    loading_matrix: np.ndarray,
    canonical_index: dict[str, int],
    weight_floor: float,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    if loading_matrix.ndim != 2 or factor_idx >= loading_matrix.shape[0]:
        return {name: 1.0 for name in member_names}
    for member_name in member_names:
        canonical_idx = canonical_index.get(member_name)
        if canonical_idx is None or canonical_idx >= loading_matrix.shape[1]:
            weights[member_name] = 1.0
            continue
        raw = abs(float(loading_matrix[factor_idx, canonical_idx]))
        weights[member_name] = max(weight_floor, raw)
    return weights


def _select_related_members(
    *,
    base_members: list[str],
    factor_block: str,
    scale_name: str,
    relationship_adjacency: dict[str, list[dict[str, Any]]],
    canonical_index: dict[str, int],
    cfg: dict[str, Any],
) -> tuple[list[str], dict[str, float], list[dict[str, Any]]]:
    scale_thresholds = dict(cfg.get("scale_thresholds", {}))
    max_related_members = dict(cfg.get("max_related_members", {}))
    same_block_only = bool(cfg.get("same_block_only", True))
    relation_score_weights = dict(cfg.get("relation_score_weights", {}))
    semantic_weight = float(relation_score_weights.get("semantic_index", 0.6))
    similarity_weight = float(relation_score_weights.get("similarity_score", 0.4))
    threshold = float(scale_thresholds.get(scale_name, 0.55))
    limit = int(max_related_members.get(scale_name, 0))
    candidate_scores: dict[str, float] = {}
    candidate_rows: dict[str, dict[str, Any]] = {}
    base_member_set = set(base_members)
    for member_name in base_members:
        for row in relationship_adjacency.get(member_name, []):
            if same_block_only and str(row.get("block_name") or "") != factor_block:
                continue
            neighbor = str(row["right"] if row["left"] == member_name else row["left"])
            if neighbor in base_member_set or neighbor not in canonical_index:
                continue
            relation_score = semantic_weight * float(row.get("semantic_index") or 0.0) + similarity_weight * float(row.get("similarity_score") or 0.0)
            if relation_score < threshold:
                continue
            if relation_score > candidate_scores.get(neighbor, 0.0):
                candidate_scores[neighbor] = relation_score
                candidate_rows[neighbor] = {
                    "canonical_name": neighbor,
                    "relation_score": round(float(relation_score), 6),
                    "semantic_index": round(float(row.get("semantic_index") or 0.0), 6),
                    "similarity_score": round(float(row.get("similarity_score") or 0.0), 6),
                    "corr_sim": round(float(row.get("corr_sim") or 0.0), 6),
                    "semantic_sim": round(float(row.get("semantic_sim") or 0.0), 6),
                    "target_sim": round(float(row.get("target_sim") or 0.0), 6),
                    "geo_sim": round(float(row.get("geo_sim") or 0.0), 6),
                }
    ranked = sorted(candidate_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:limit]
    return [name for name, _score in ranked], {name: float(score) for name, score in ranked}, [candidate_rows[name] for name, _score in ranked]


def _intra_factor_relationship_boosts(
    *,
    base_members: list[str],
    factor_block: str,
    relationship_adjacency: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    scale_name: str,
) -> dict[str, float]:
    scale_bonus = dict(cfg.get("internal_bonus_scale_by_scale", {}))
    bonus_scale = float(scale_bonus.get(scale_name, 0.0))
    if bonus_scale <= 0.0 or len(base_members) <= 1:
        return {name: 1.0 for name in base_members}
    relation_score_weights = dict(cfg.get("relation_score_weights", {}))
    semantic_weight = float(relation_score_weights.get("semantic_index", 0.6))
    similarity_weight = float(relation_score_weights.get("similarity_score", 0.4))
    relation_floor = float(cfg.get("internal_relation_floor", 0.25))
    same_block_only = bool(cfg.get("same_block_only", True))
    base_member_set = set(base_members)
    boosts: dict[str, float] = {}
    for member_name in base_members:
        scores: list[float] = []
        for row in relationship_adjacency.get(member_name, []):
            if same_block_only and str(row.get("block_name") or "") != factor_block:
                continue
            neighbor = str(row["right"] if row["left"] == member_name else row["left"])
            if neighbor not in base_member_set:
                continue
            relation_score = semantic_weight * float(row.get("semantic_index") or 0.0) + similarity_weight * float(row.get("similarity_score") or 0.0)
            if relation_score >= relation_floor:
                scores.append(relation_score)
        avg_score = float(np.mean(scores)) if scores else 0.0
        boosts[member_name] = 1.0 + bonus_scale * avg_score
    return boosts


def _weighted_surface(
    *,
    standardized_tensor: np.ndarray,
    members: list[str],
    weights: dict[str, float],
    canonical_index: dict[str, int],
    eps: float,
) -> np.ndarray:
    member_indices = [canonical_index[name] for name in members if name in canonical_index]
    if not member_indices:
        return np.zeros(standardized_tensor.shape[:2], dtype=np.float32)
    matrix = np.asarray(standardized_tensor[:, :, member_indices], dtype=np.float32)
    weight_vector = np.asarray([max(eps, float(weights.get(name, 1.0))) for name in members if name in canonical_index], dtype=np.float32)
    weight_vector = weight_vector / np.clip(float(weight_vector.sum()), eps, None)
    combined = np.tensordot(matrix, weight_vector, axes=([2], [0])).astype(np.float32)
    return _zscore_surface(combined, eps=eps)


def _rows_to_block_uncertainty_tensor(
    *,
    rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    block_axis: list[str],
) -> np.ndarray:
    province_index = {name: idx for idx, name in enumerate(province_axis)}
    block_index = {name: idx for idx, name in enumerate(block_axis)}
    tensor = np.zeros((len(province_axis), len(month_axis), len(block_axis)), dtype=np.float32)
    for row in rows:
        province_name = str(row.get("province") or "")
        block_id = str(row.get("block_id") or "")
        values = [float(value) for value in list(row.get("posterior_std_values") or [])]
        if province_name not in province_index or block_id not in block_index or not values:
            continue
        province_idx = province_index[province_name]
        block_idx = block_index[block_id]
        clipped = np.asarray(values[: len(month_axis)], dtype=np.float32)
        tensor[province_idx, : clipped.shape[0], block_idx] = clipped
    return tensor.astype(np.float32)


def _factor_parent_block(
    *,
    base_members: list[str],
    indicator_names_by_block: dict[str, list[str]],
) -> str:
    if not base_members or not indicator_names_by_block:
        return ""
    member_set = {str(name) for name in base_members if str(name)}
    scores: list[tuple[int, str]] = []
    for block_id, indicator_names in indicator_names_by_block.items():
        overlap = len(member_set & {str(name) for name in indicator_names if str(name)})
        if overlap > 0:
            scores.append((overlap, str(block_id)))
    if not scores:
        return ""
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def _concentration_multiplier(weights: dict[str, float], *, members: list[str], eps: float) -> float:
    member_weights = np.asarray([max(eps, float(weights.get(member_name, eps))) for member_name in members], dtype=np.float32)
    total = float(member_weights.sum())
    if total <= eps:
        return 1.0
    normalized = member_weights / total
    concentration = float(np.sum(normalized * normalized))
    return float(np.sqrt(1.0 / max(concentration, eps)))


def build_multiscale_factor_artifacts(
    *,
    standardized_tensor: np.ndarray,
    base_factor_tensor: np.ndarray,
    factor_catalog: list[dict[str, Any]],
    loading_matrix: np.ndarray,
    canonical_axis: list[str],
    province_axis: list[str],
    month_axis: list[str],
    relationship_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    block_uncertainty_rows: list[dict[str, Any]] | None = None,
    indicator_names_by_block: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    enabled = bool(cfg.get("enabled", True))
    if not enabled or not factor_catalog:
        return {
            "catalog_rows": [],
            "axes": {
                "province": list(province_axis),
                "region": [],
                "national": ["Philippines"],
                "factor": [],
                "month": list(month_axis),
            },
            "province_tensor": np.zeros((len(province_axis), len(month_axis), 0), dtype=np.float32),
            "region_tensor": np.zeros((0, len(month_axis), 0), dtype=np.float32),
            "national_tensor": np.zeros((1, len(month_axis), 0), dtype=np.float32),
            "province_uncertainty_tensor": np.zeros((len(province_axis), len(month_axis), 0), dtype=np.float32),
            "region_uncertainty_tensor": np.zeros((0, len(month_axis), 0), dtype=np.float32),
            "national_uncertainty_tensor": np.zeros((1, len(month_axis), 0), dtype=np.float32),
            "summary": {"enabled": False, "factor_count": 0},
        }

    canonical_index = {name: idx for idx, name in enumerate(canonical_axis)}
    relationship_adjacency = _relationship_adjacency(relationship_rows)
    region_axis, region_groups, non_national_indices = _region_groups(province_axis)
    factor_ids = [str(row.get("factor_id") or "") for row in factor_catalog]
    factor_count = len(factor_catalog)
    province_tensor = np.zeros((len(province_axis), len(month_axis), factor_count), dtype=np.float32)
    region_tensor = np.zeros((len(region_axis), len(month_axis), factor_count), dtype=np.float32)
    national_tensor = np.zeros((1, len(month_axis), factor_count), dtype=np.float32)
    province_uncertainty_tensor = np.zeros((len(province_axis), len(month_axis), factor_count), dtype=np.float32)
    region_uncertainty_tensor = np.zeros((len(region_axis), len(month_axis), factor_count), dtype=np.float32)
    national_uncertainty_tensor = np.zeros((1, len(month_axis), factor_count), dtype=np.float32)
    eps = float(cfg.get("svd_eps", 1e-6))
    loading_floor = float(cfg.get("base_loading_floor", 0.1))
    related_floor = float(cfg.get("related_member_weight_floor", 0.08))
    related_scale = float(cfg.get("related_member_weight_scale", 0.75))

    catalog_rows: list[dict[str, Any]] = []
    scale_names = ["province", "region", "national"]
    total_added_by_scale = Counter({scale: 0 for scale in scale_names})
    total_member_by_scale = Counter({scale: 0 for scale in scale_names})
    factors_with_related = Counter({scale: 0 for scale in scale_names})
    factors_with_relationship_weighting = Counter({scale: 0 for scale in scale_names})
    block_uncertainty_rows = list(block_uncertainty_rows or [])
    indicator_names_by_block = {
        str(key): [str(value) for value in list(values or [])]
        for key, values in dict(indicator_names_by_block or {}).items()
    }
    parent_block_axis = sorted(indicator_names_by_block)
    parent_block_uncertainty = _rows_to_block_uncertainty_tensor(
        rows=block_uncertainty_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        block_axis=parent_block_axis,
    )
    parent_block_index = {name: idx for idx, name in enumerate(parent_block_axis)}
    uncertainty_mode = "parent_block_propagation" if parent_block_axis else "unavailable"

    for factor_idx, factor in enumerate(factor_catalog):
        factor_id = str(factor.get("factor_id") or "")
        base_members = [str(name) for name in list(factor.get("member_canonical_names") or []) if str(name) in canonical_index]
        parent_block_id = _factor_parent_block(base_members=base_members, indicator_names_by_block=indicator_names_by_block)
        member_weights = _member_weights_for_factor(
            factor_idx=factor_idx,
            member_names=base_members,
            loading_matrix=loading_matrix,
            canonical_index=canonical_index,
            weight_floor=loading_floor,
        )
        scale_members: dict[str, list[str]] = {}
        scale_added_members: dict[str, list[str]] = {}
        scale_weights: dict[str, dict[str, float]] = {}
        scale_relationships: dict[str, list[dict[str, Any]]] = {}
        scale_uncertainty_multipliers: dict[str, float] = {}

        if not base_members:
            province_surface = np.asarray(base_factor_tensor[:, :, factor_idx], dtype=np.float32)
            province_surface = _zscore_surface(province_surface, eps=eps)
            province_tensor[:, :, factor_idx] = province_surface
            region_tensor[:, :, factor_idx] = _aggregate_surface(province_surface, region_groups, eps=eps)
            national_tensor[:, :, factor_idx] = _national_surface(province_surface, non_national_indices, eps=eps)
            for scale_name in scale_names:
                scale_members[scale_name] = []
                scale_added_members[scale_name] = []
                scale_weights[scale_name] = {}
                scale_relationships[scale_name] = []
                scale_uncertainty_multipliers[scale_name] = 1.0
            catalog_rows.append(
                dict(factor)
                | {
                    "multiscale_members": scale_members,
                    "multiscale_added_members": scale_added_members,
                    "multiscale_member_weights": scale_weights,
                    "multiscale_top_relationships": scale_relationships,
                    "uncertainty_parent_block_id": parent_block_id,
                    "uncertainty_multipliers": scale_uncertainty_multipliers,
                    "multiscale_construction_mode": "carry_forward_base_surface",
                }
            )
            continue

        for scale_name in scale_names:
            related_members, related_scores, related_rows = _select_related_members(
                base_members=base_members,
                factor_block=str(factor.get("block_name") or ""),
                scale_name=scale_name,
                relationship_adjacency=relationship_adjacency,
                canonical_index=canonical_index,
                cfg=cfg,
            )
            all_members = list(base_members) + [name for name in related_members if name not in base_members]
            weights = dict(member_weights)
            internal_boosts = _intra_factor_relationship_boosts(
                base_members=base_members,
                factor_block=str(factor.get("block_name") or ""),
                relationship_adjacency=relationship_adjacency,
                cfg=cfg,
                scale_name=scale_name,
            )
            for member_name, boost in internal_boosts.items():
                weights[member_name] = max(related_floor, float(weights.get(member_name, loading_floor)) * float(boost))
            for member_name, score in related_scores.items():
                weights[member_name] = max(related_floor, float(score) * related_scale)
            surface = _weighted_surface(
                standardized_tensor=standardized_tensor,
                members=all_members,
                weights=weights,
                canonical_index=canonical_index,
                eps=eps,
            )
            if scale_name == "province":
                province_tensor[:, :, factor_idx] = surface
            elif scale_name == "region":
                region_tensor[:, :, factor_idx] = _aggregate_surface(surface, region_groups, eps=eps)
            else:
                national_tensor[:, :, factor_idx] = _national_surface(surface, non_national_indices, eps=eps)
            scale_members[scale_name] = all_members
            scale_added_members[scale_name] = related_members
            scale_weights[scale_name] = {key: round(float(value), 6) for key, value in sorted(weights.items())}
            scale_relationships[scale_name] = related_rows
            scale_uncertainty_multipliers[scale_name] = round(_concentration_multiplier(weights, members=all_members, eps=eps), 6)
            total_added_by_scale[scale_name] += len(related_members)
            total_member_by_scale[scale_name] += len(all_members)
            if related_members:
                factors_with_related[scale_name] += 1
            if any(abs(float(internal_boosts.get(member_name, 1.0)) - 1.0) > 1e-6 for member_name in base_members):
                factors_with_relationship_weighting[scale_name] += 1

        if parent_block_id and parent_block_id in parent_block_index:
            parent_surface = np.asarray(parent_block_uncertainty[:, :, parent_block_index[parent_block_id]], dtype=np.float32)
        elif parent_block_uncertainty.size:
            parent_surface = np.asarray(parent_block_uncertainty.mean(axis=2), dtype=np.float32)
        else:
            parent_surface = np.ones((len(province_axis), len(month_axis)), dtype=np.float32)
        province_uncertainty_tensor[:, :, factor_idx] = parent_surface * float(scale_uncertainty_multipliers.get("province", 1.0))
        region_uncertainty_tensor[:, :, factor_idx] = _aggregate_uncertainty_surface(
            parent_surface * float(scale_uncertainty_multipliers.get("region", 1.0)),
            region_groups,
        )
        national_uncertainty_tensor[:, :, factor_idx] = _national_uncertainty_surface(
            parent_surface * float(scale_uncertainty_multipliers.get("national", 1.0)),
            non_national_indices,
        )

        catalog_rows.append(
            dict(factor)
            | {
                "multiscale_members": scale_members,
                "multiscale_added_members": scale_added_members,
                "multiscale_member_weights": scale_weights,
                "multiscale_top_relationships": scale_relationships,
                "uncertainty_parent_block_id": parent_block_id,
                "uncertainty_multipliers": scale_uncertainty_multipliers,
                "multiscale_construction_mode": "relationship_index_weighted_membership",
            }
        )

    summary = {
        "enabled": True,
        "factor_count": factor_count,
        "relationship_row_count": len(relationship_rows),
        "region_count": len(region_axis),
        "uncertainty_mode": uncertainty_mode,
        "scale_summary": {
            scale_name: {
                "mean_member_count": round(float(total_member_by_scale[scale_name] / max(1, factor_count)), 6),
                "mean_added_member_count": round(float(total_added_by_scale[scale_name] / max(1, factor_count)), 6),
                "factors_with_related_members": int(factors_with_related[scale_name]),
                "factors_with_relationship_weighting": int(factors_with_relationship_weighting[scale_name]),
            }
            for scale_name in scale_names
        },
        "top_region_axis": region_axis[:12],
    }
    return {
        "catalog_rows": catalog_rows,
        "axes": {
            "province": list(province_axis),
            "region": region_axis,
            "national": ["Philippines"],
            "factor": factor_ids,
            "month": list(month_axis),
        },
        "province_tensor": province_tensor.astype(np.float32),
        "region_tensor": region_tensor.astype(np.float32),
        "national_tensor": national_tensor.astype(np.float32),
        "province_uncertainty_tensor": province_uncertainty_tensor.astype(np.float32),
        "region_uncertainty_tensor": region_uncertainty_tensor.astype(np.float32),
        "national_uncertainty_tensor": national_uncertainty_tensor.astype(np.float32),
        "summary": summary,
    }
