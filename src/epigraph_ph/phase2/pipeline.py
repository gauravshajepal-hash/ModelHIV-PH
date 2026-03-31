from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.phase2.block_graph_builder import build_sharded_block_graph_outputs
from epigraph_ph.phase2.rescue_profile import augment_phase2_for_rescue_v2
from epigraph_ph.phase15 import PHASE15_PROFILE_ID
from epigraph_ph.runtime import (
    RunContext,
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

BLOCK_ORDER = ["economics", "logistics", "behavior", "population", "biology", "policy", "cascade", "mixed"]
_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase2_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase2", {}) or {})


def _phase2_required(key: str) -> Any:
    cfg = _phase2_cfg()
    if key not in cfg:
        raise KeyError(f"Missing HIV phase2 constraint setting: {key}")
    return cfg[key]


def _phase2_required_section(key: str) -> dict[str, Any]:
    value = _phase2_required(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase2 constraint setting '{key}' must be a mapping")
    return dict(value)


def _phase2_stabilizer(key: str) -> float:
    stabilizers = dict((_HIV_PLUGIN.numerical_stabilizers or {}).get("phase2", {}) or {})
    if key not in stabilizers:
        raise KeyError(f"Missing HIV phase2 numerical stabilizer: {key}")
    return float(stabilizers[key])


def _phase2_execution_cfg() -> dict[str, Any]:
    return _phase2_required_section("execution")


def _has_directed_cycle(adjacency: np.ndarray, *, threshold: float = 1e-6) -> bool:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return True
    node_count = int(adjacency.shape[0])
    graph = [[idx for idx in range(node_count) if idx != src and abs(float(adjacency[src, idx])) > threshold] for src in range(node_count)]
    visited = [0] * node_count

    def _dfs(node_idx: int) -> bool:
        if visited[node_idx] == 1:
            return True
        if visited[node_idx] == 2:
            return False
        visited[node_idx] = 1
        for next_idx in graph[node_idx]:
            if _dfs(next_idx):
                return True
        visited[node_idx] = 2
        return False

    return any(_dfs(node_idx) for node_idx in range(node_count) if visited[node_idx] == 0)


def _node_tiers(parameter_catalog: list[dict[str, Any]], canonical_axis: list[str]) -> np.ndarray:
    index = {row["canonical_name"]: row for row in parameter_catalog}
    tiers = []
    for name in canonical_axis:
        row = index.get(name, {})
        domain_family = _dominant_key(row.get("domain_families"), "mixed")
        pathway_family = _dominant_key(row.get("pathway_families"), "mixed")
        tiers.append(_tier_for_node(name, domain_family, pathway_family))
    return np.asarray(tiers, dtype=int)


def _dominant_key(counter_map: dict[str, int] | None, default: str = "mixed") -> str:
    if not counter_map:
        return default
    return sorted(counter_map.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def _tier_for_node(canonical_name: str, domain_family: str, pathway_family: str) -> int:
    name = canonical_name.lower()
    if any(token in name for token in ("testing", "diagnos", "viral", "suppression")):
        return 4
    if domain_family in {"biology"} or pathway_family in {"biological_progression"}:
        return 3
    if domain_family in {"behavior"} or pathway_family in {"prevention_access", "testing_uptake", "linkage_to_care", "retention_adherence", "suppression_outcomes"}:
        return 2
    return 1


def _block_from_signals(domain_family: str, pathway_family: str, tags: list[str], targets: list[str]) -> tuple[str, list[str]]:
    candidates = [domain_family, *tags]
    if any(target in {"prevention_access", "testing_uptake", "linkage_to_care", "retention_adherence", "suppression_outcomes"} for target in targets):
        candidates.append("behavior")
    if any(target in {"mobility_network_mixing", "health_system_reach"} for target in targets):
        candidates.append("logistics")
    if any(target in {"biological_progression"} for target in targets):
        candidates.append("biology")
    primary = "mixed"
    for item in BLOCK_ORDER:
        if item in candidates:
            primary = item
            break
    secondary = [item for item in BLOCK_ORDER if item in candidates and item != primary]
    return primary, secondary[:3]


def _flatten_feature_matrix(standardized_tensor: np.ndarray) -> np.ndarray:
    if standardized_tensor.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feature_count = int(standardized_tensor.shape[-1]) if standardized_tensor.ndim == 3 else 0
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.asarray(standardized_tensor, dtype=np.float32).reshape(-1, feature_count)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _phase2_row_month_key(row: dict[str, Any], month_index: dict[str, int]) -> str:
    time_value = str(row.get("time") or "").strip()
    if time_value in month_index:
        return time_value
    year = row.get("year")
    if isinstance(year, int):
        candidate = f"{year:04d}-01"
        if candidate in month_index:
            return candidate
    if isinstance(year, str) and year.isdigit():
        candidate = f"{int(year):04d}-01"
        if candidate in month_index:
            return candidate
    return ""


def _zscore_support_matrix(matrix_tf: np.ndarray) -> np.ndarray:
    if matrix_tf.ndim != 2 or matrix_tf.size == 0:
        return np.zeros_like(matrix_tf, dtype=np.float32)
    output = np.zeros_like(matrix_tf, dtype=np.float32)
    for feature_idx in range(matrix_tf.shape[1]):
        column = np.asarray(matrix_tf[:, feature_idx], dtype=np.float32)
        if not np.any(column):
            continue
        mean = float(column.mean())
        std = float(column.std())
        if std <= 1e-6:
            output[:, feature_idx] = column - mean
        else:
            output[:, feature_idx] = (column - mean) / std
    return output.astype(np.float32)


def _phase2_feature_matrix(
    *,
    standardized_tensor: np.ndarray,
    normalized_rows: list[dict[str, Any]],
    canonical_axis: list[str],
    month_axis: list[str],
    province_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric_matrix = _flatten_feature_matrix(standardized_tensor)
    if not canonical_axis:
        return np.zeros((0, 0), dtype=np.float32), {"used_soft_candidates": False, "soft_feature_count": 0}
    row_count = max(1, len(province_axis) * len(month_axis))
    if numeric_matrix.ndim != 2 or numeric_matrix.shape[1] != len(canonical_axis):
        numeric_matrix = np.zeros((row_count, len(canonical_axis)), dtype=np.float32)
    month_index = {month: idx for idx, month in enumerate(month_axis)}
    canonical_index = {name: idx for idx, name in enumerate(canonical_axis)}
    soft_support = np.zeros((len(month_axis), len(canonical_axis)), dtype=np.float32)
    soft_counts = np.zeros((len(canonical_axis),), dtype=np.float32)
    numeric_counts = np.zeros((len(canonical_axis),), dtype=np.float32)
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        if canonical_name not in canonical_index:
            continue
        month_key = _phase2_row_month_key(row, month_index)
        if not month_key:
            continue
        feature_idx = canonical_index[canonical_name]
        if row.get("model_numeric_value") is not None:
            numeric_counts[feature_idx] += 1.0
            continue
        support = float(row.get("evidence_weight") or 0.0)
        support *= max(0.0, 1.0 - float(row.get("bias_penalty") or 0.0))
        support *= max(0.0, float(row.get("replication_weight") or 0.0))
        if support <= 0.0:
            continue
        soft_support[month_index[month_key], feature_idx] += support
        soft_counts[feature_idx] += 1.0
    soft_matrix_monthly = _zscore_support_matrix(soft_support)
    if len(province_axis) > 0:
        soft_matrix = np.repeat(soft_matrix_monthly[None, :, :], len(province_axis), axis=0).reshape(-1, len(canonical_axis))
    else:
        soft_matrix = soft_matrix_monthly
    blended = np.asarray(numeric_matrix, dtype=np.float32).copy()
    for feature_idx in range(len(canonical_axis)):
        soft_count = float(soft_counts[feature_idx])
        numeric_count = float(numeric_counts[feature_idx])
        if soft_count <= 0.0:
            continue
        if numeric_count <= 0.0:
            blended[:, feature_idx] = soft_matrix[:, feature_idx]
            continue
        blended[:, feature_idx] = (
            numeric_count * blended[:, feature_idx] + soft_count * soft_matrix[:, feature_idx]
        ) / max(numeric_count + soft_count, 1.0)
    report = {
        "used_soft_candidates": bool(np.any(soft_counts > 0)),
        "soft_feature_count": int(np.sum(soft_counts > 0)),
        "numeric_feature_count": int(np.sum(numeric_counts > 0)),
        "matrix_rows": int(blended.shape[0]),
        "matrix_columns": int(blended.shape[1]),
        "top_soft_features": [
            {
                "canonical_name": canonical_axis[idx],
                "soft_count": round(float(soft_counts[idx]), 4),
                "numeric_count": round(float(numeric_counts[idx]), 4),
            }
            for idx in np.argsort(-soft_counts)[:15]
            if soft_counts[idx] > 0
        ],
    }
    return blended.astype(np.float32), report


def _safe_mutual_information(x: np.ndarray, y: np.ndarray, *, bins: int) -> float:
    eps = _phase2_stabilizer("mi_eps")
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    joint, _, _ = np.histogram2d(x, y, bins=max(4, int(bins)))
    if not np.any(joint):
        return 0.0
    joint_prob = joint / max(float(joint.sum()), eps)
    px = joint_prob.sum(axis=1, keepdims=True)
    py = joint_prob.sum(axis=0, keepdims=True)
    valid = joint_prob > 0
    mi = float(np.sum(joint_prob[valid] * np.log((joint_prob[valid] + eps) / ((px @ py)[valid] + eps))))
    return max(0.0, mi)


def _phase2_target_feature_indices(canonical_axis: list[str]) -> list[int]:
    cfg = _phase2_required_section("mi_prefilter")
    target_tokens = [str(token).lower() for token in cfg.get("target_tokens", [])]
    target_indices = [idx for idx, name in enumerate(canonical_axis) if any(token in str(name).lower() for token in target_tokens)]
    return target_indices or list(range(min(3, len(canonical_axis))))


def _mi_prefilter(
    matrix_tf: np.ndarray,
    canonical_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = _phase2_required_section("mi_prefilter")
    feature_count = matrix_tf.shape[1] if matrix_tf.ndim == 2 else 0
    if not cfg.get("enabled", False) or feature_count == 0:
        return np.ones((feature_count,), dtype=np.float32), {"enabled": False, "kept_feature_count": feature_count, "dropped_feature_count": 0, "rows": []}
    bins = int(cfg["hist_bins"])
    threshold = float(cfg["mi_threshold"])
    target_indices = _phase2_target_feature_indices(canonical_axis)
    rows = []
    keep_mask = np.zeros((feature_count,), dtype=np.float32)
    for feature_idx, canonical_name in enumerate(canonical_axis):
        if feature_idx in target_indices:
            keep_mask[feature_idx] = 1.0
            rows.append({"canonical_name": canonical_name, "max_mi_to_target": 1.0, "kept": True, "is_target": True})
            continue
        max_mi = 0.0
        for target_idx in target_indices:
            if target_idx == feature_idx:
                continue
            max_mi = max(max_mi, _safe_mutual_information(matrix_tf[:, feature_idx], matrix_tf[:, target_idx], bins=bins))
        kept = max_mi >= threshold
        keep_mask[feature_idx] = 1.0 if kept else 0.0
        rows.append({"canonical_name": canonical_name, "max_mi_to_target": round(max_mi, 6), "kept": kept, "is_target": False})
    min_keep = min(int(cfg["min_features_to_keep"]), feature_count)
    if int(keep_mask.sum()) < min_keep:
        ranked = sorted(range(feature_count), key=lambda idx: float(rows[idx]["max_mi_to_target"]), reverse=True)
        for feature_idx in ranked[:min_keep]:
            keep_mask[feature_idx] = 1.0
            rows[feature_idx]["kept"] = True
    kept_feature_count = int(keep_mask.sum())
    report = {
        "enabled": True,
        "mi_threshold": threshold,
        "target_indices": target_indices,
        "kept_feature_count": kept_feature_count,
        "dropped_feature_count": int(feature_count - kept_feature_count),
        "rows": rows,
    }
    return keep_mask.astype(np.float32), report


def _normal_survival(z_value: float) -> float:
    return 0.5 * math.erfc(float(z_value) / math.sqrt(2.0))


def _fisher_z_pvalue(corr_value: float, sample_size: int, conditioner_count: int = 0) -> float:
    clip = _phase2_stabilizer("fisher_clip")
    corr_value = float(np.clip(corr_value, -clip, clip))
    dof = max(int(sample_size - conditioner_count - 3), 1)
    fisher = 0.5 * math.log((1.0 + corr_value) / max(1.0 - corr_value, 1e-12))
    z_score = abs(fisher) * math.sqrt(float(dof))
    return min(1.0, max(0.0, 2.0 * _normal_survival(z_score)))


def _partial_correlation(corr: np.ndarray, left_idx: int, right_idx: int, conditioners: tuple[int, ...]) -> float:
    if not conditioners:
        return float(corr[left_idx, right_idx])
    indices = [left_idx, right_idx, *conditioners]
    sub_corr = np.asarray(corr[np.ix_(indices, indices)], dtype=np.float64)
    try:
        precision = np.linalg.pinv(sub_corr)
    except Exception:
        return float(corr[left_idx, right_idx])
    denom = max(float(np.sqrt(abs(precision[0, 0] * precision[1, 1]))), 1e-12)
    value = -float(precision[0, 1]) / denom
    return float(np.clip(value, -_phase2_stabilizer("fisher_clip"), _phase2_stabilizer("fisher_clip")))


def _pc_skeleton_from_fisher_z(matrix_tf: np.ndarray, canonical_axis: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = _phase2_required_section("pc_skeleton")
    feature_count = matrix_tf.shape[1] if matrix_tf.ndim == 2 else 0
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float32), {"available": False, "edge_count": 0, "dropped_edges": []}
    corr = _safe_feature_corr(matrix_tf)
    alpha = float(cfg["alpha"])
    corr_floor = float(cfg["corr_floor"])
    max_conditioners = int(cfg["max_conditioners"])
    candidate_conditioners = int(cfg["candidate_conditioners"])
    skeleton = np.zeros((feature_count, feature_count), dtype=np.float32)
    dropped_edges = []
    sample_size = int(matrix_tf.shape[0])

    for left_idx in range(feature_count):
        for right_idx in range(left_idx + 1, feature_count):
            base_corr = float(corr[left_idx, right_idx])
            if abs(base_corr) < corr_floor:
                dropped_edges.append({"source": canonical_axis[left_idx], "target": canonical_axis[right_idx], "reason": "corr_floor", "p_value": 1.0})
                continue
            p_value = _fisher_z_pvalue(base_corr, sample_size, conditioner_count=0)
            if p_value > alpha:
                dropped_edges.append({"source": canonical_axis[left_idx], "target": canonical_axis[right_idx], "reason": "unconditional_independence", "p_value": round(p_value, 6)})
                continue
            kept = True
            if max_conditioners > 0:
                candidate_order = np.argsort(-(np.abs(corr[left_idx]) + np.abs(corr[right_idx])))
                candidate_order = [idx for idx in candidate_order if idx not in {left_idx, right_idx}][:candidate_conditioners]
                for conditioner_idx in candidate_order:
                    partial_corr = _partial_correlation(corr, left_idx, right_idx, (int(conditioner_idx),))
                    partial_p = _fisher_z_pvalue(partial_corr, sample_size, conditioner_count=1)
                    if partial_p > alpha:
                        kept = False
                        dropped_edges.append(
                            {
                                "source": canonical_axis[left_idx],
                                "target": canonical_axis[right_idx],
                                "reason": f"conditional_independence:{canonical_axis[conditioner_idx]}",
                                "p_value": round(partial_p, 6),
                            }
                        )
                        break
            if kept:
                skeleton[left_idx, right_idx] = 1.0
                skeleton[right_idx, left_idx] = 1.0
    np.fill_diagonal(skeleton, 0.0)
    return skeleton.astype(np.float32), {
        "available": True,
        "alpha": alpha,
        "corr_floor": corr_floor,
        "max_conditioners": max_conditioners,
        "candidate_conditioners": candidate_conditioners,
        "edge_count": int(np.sum(skeleton) / 2),
        "dropped_edges": dropped_edges[:50],
    }


def _safe_feature_corr(matrix_tf: np.ndarray) -> np.ndarray:
    if matrix_tf.ndim != 2 or matrix_tf.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feature_count = int(matrix_tf.shape[1])
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if feature_count == 1:
        return np.zeros((1, 1), dtype=np.float32)
    matrix = np.asarray(matrix_tf, dtype=np.float32)
    if matrix.shape[0] < 2:
        return np.zeros((feature_count, feature_count), dtype=np.float32)
    mean = np.mean(matrix, axis=0, keepdims=True)
    centered = matrix - mean
    var = np.mean(centered * centered, axis=0)
    std = np.sqrt(np.clip(var, a_min=_phase2_stabilizer("corr_eps"), a_max=None))
    normalized = centered / std[None, :]
    corr = (normalized.T @ normalized) / max(matrix.shape[0] - 1, 1)
    corr = np.asarray(corr, dtype=np.float32)
    if corr.shape != (feature_count, feature_count):
        corr = np.reshape(corr, (feature_count, feature_count)).astype(np.float32)
    corr = np.clip(corr, -1.0, 1.0)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, np.where(var > _phase2_stabilizer("corr_eps"), 1.0, 0.0))
    return corr.astype(np.float32)


def _skeleton_from_corr(matrix_tf: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    threshold = float(_phase2_required("skeleton_threshold")) if threshold == 0.05 else float(threshold)
    if matrix_tf.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feature_count = matrix_tf.shape[1]
    if matrix_tf.shape[0] < 2:
        mask = np.ones((feature_count, feature_count), dtype=np.float32)
        np.fill_diagonal(mask, 0.0)
        return mask
    corr = _safe_feature_corr(matrix_tf)
    mask = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def _tier_mask(parameter_catalog: list[dict[str, Any]], canonical_axis: list[str]) -> np.ndarray:
    tiers = _node_tiers(parameter_catalog, canonical_axis)
    mask = (tiers[:, None] <= tiers[None, :]).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def _lag_mask(parameter_catalog: list[dict[str, Any]], canonical_axis: list[str]) -> np.ndarray:
    cfg = _phase2_required_section("lag_rules")
    pathway_by_name = {
        str(row.get("canonical_name") or ""): _dominant_key(row.get("pathway_families"), "mixed")
        for row in parameter_catalog
    }
    blocked_pathway_pairs = {
        (str(left), str(right))
        for left, right in cfg.get("blocked_pathway_pairs", [])
        if left and right
    }
    blocked_canonical_pairs = {
        (str(left), str(right))
        for left, right in cfg.get("blocked_canonical_pairs", [])
        if left and right
    }
    feature_count = len(canonical_axis)
    mask = np.ones((feature_count, feature_count), dtype=np.float32)
    for left_idx, left_name in enumerate(canonical_axis):
        for right_idx, right_name in enumerate(canonical_axis):
            if left_idx == right_idx:
                mask[left_idx, right_idx] = 0.0
                continue
            left_pathway = pathway_by_name.get(left_name, "mixed")
            right_pathway = pathway_by_name.get(right_name, "mixed")
            if (left_pathway, right_pathway) in blocked_pathway_pairs or (left_name, right_name) in blocked_canonical_pairs:
                mask[left_idx, right_idx] = 0.0
    np.fill_diagonal(mask, 0.0)
    return mask


def _notears_optimize(matrix_tf: np.ndarray, admissibility_mask: np.ndarray, *, steps: int = 200) -> np.ndarray:
    notears_cfg = _phase2_required_section("notears")
    feature_count = matrix_tf.shape[1] if matrix_tf.ndim == 2 else 0
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if feature_count == 1:
        return np.zeros((1, 1), dtype=np.float32)
    if torch is None:
        return admissibility_mask.astype(np.float32) * _safe_feature_corr(matrix_tf)
    exec_cfg = _phase2_execution_cfg()
    device = choose_torch_device(prefer_gpu=bool(exec_cfg.get("prefer_gpu", True)))
    X = to_torch_tensor(matrix_tf, device=device, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    mask = to_torch_tensor(admissibility_mask, device=device, dtype=torch.float32)
    W = torch.nn.Parameter(torch.zeros((feature_count, feature_count), dtype=torch.float32, device=device))
    steps = int(notears_cfg["steps"]) if steps == 200 else int(steps)
    optimizer_kind = str(notears_cfg.get("optimizer_kind") or "adam").lower()
    rho = float(notears_cfg.get("rho_init", 1.0))
    rho_multiplier = float(notears_cfg.get("rho_multiplier", 10.0))
    h_tol = float(notears_cfg.get("h_tol", 1e-6))
    weight_threshold = float(notears_cfg.get("weight_threshold", 0.0))
    outer_steps = max(1, int(notears_cfg.get("outer_steps", 1)))
    alpha = torch.tensor(0.0, device=device, dtype=torch.float32)

    def _objective_terms() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        W_masked = W * mask
        W_masked = W_masked - torch.diag(torch.diag(W_masked))
        recon = X @ W_masked
        loss = torch.mean((X - recon) ** 2)
        h = torch.trace(torch.matrix_exp(W_masked * W_masked)) - feature_count
        reg = (
            float(notears_cfg["lasso_penalty"]) * torch.sum(torch.abs(W_masked))
            + float(notears_cfg["ridge_penalty"]) * torch.sum(W_masked ** 2)
            + float(notears_cfg["acyclicity_penalty"]) * (h ** 2)
        )
        return W_masked, loss, h, reg

    if optimizer_kind == "lbfgs":
        for _ in range(outer_steps):
            optimizer = torch.optim.LBFGS(
                [W],
                lr=float(notears_cfg["optimizer_lr"]),
                max_iter=int(notears_cfg.get("lbfgs_max_iter", 20)),
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                _W_masked, loss, h, reg = _objective_terms()
                total = loss + reg + alpha * h + 0.5 * rho * (h ** 2)
                total.backward()
                return total

            optimizer.step(closure)
            with torch.no_grad():
                _W_masked, _loss, h, _reg = _objective_terms()
                h_value = float(torch.abs(h).item())
                if h_value <= h_tol:
                    break
                alpha = alpha + rho * h.detach()
                rho *= rho_multiplier
    else:
        inner_steps = max(1, steps // outer_steps)
        optimizer = torch.optim.Adam([W], lr=float(notears_cfg["optimizer_lr"]))
        for _ in range(outer_steps):
            for _ in range(inner_steps):
                optimizer.zero_grad(set_to_none=True)
                _W_masked, loss, h, reg = _objective_terms()
                total = loss + reg + alpha * h + 0.5 * rho * (h ** 2)
                total.backward()
                optimizer.step()
            with torch.no_grad():
                _W_masked, _loss, h, _reg = _objective_terms()
                h_value = float(torch.abs(h).item())
                if h_value <= h_tol:
                    break
                alpha = alpha + rho * h.detach()
                rho *= rho_multiplier
    result = (W.detach() * mask).cpu().numpy().astype(np.float32)
    if weight_threshold > 0.0:
        result[np.abs(result) < weight_threshold] = 0.0
    np.fill_diagonal(result, 0.0)
    return result


def _project_to_exact_dag(adjacency: np.ndarray, tiers: np.ndarray, canonical_axis: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return np.zeros((0, 0), dtype=np.float32), {"projected": False, "reason": "invalid_shape", "node_order": [], "dropped_edge_count": 0}
    node_count = int(adjacency.shape[0])
    if node_count == 0:
        return adjacency.astype(np.float32), {"projected": True, "reason": "empty", "node_order": [], "dropped_edge_count": 0}
    out_strength = np.abs(adjacency).sum(axis=1)
    in_strength = np.abs(adjacency).sum(axis=0)
    ordering = sorted(
        range(node_count),
        key=lambda idx: (
            int(tiers[idx]) if idx < len(tiers) else 999,
            -(float(out_strength[idx]) - float(in_strength[idx])),
            canonical_axis[idx] if idx < len(canonical_axis) else f"node_{idx}",
        ),
    )
    rank = {node_idx: position for position, node_idx in enumerate(ordering)}
    projected = np.zeros_like(adjacency, dtype=np.float32)
    dropped_edges = []
    for src in range(node_count):
        for dst in range(node_count):
            if src == dst:
                continue
            weight = float(adjacency[src, dst])
            if abs(weight) <= 1e-9:
                continue
            if rank[src] < rank[dst]:
                projected[src, dst] = weight
            else:
                dropped_edges.append(
                    {
                        "source": canonical_axis[src] if src < len(canonical_axis) else f"node_{src}",
                        "target": canonical_axis[dst] if dst < len(canonical_axis) else f"node_{dst}",
                        "weight": round(weight, 6),
                        "abs_weight": round(abs(weight), 6),
                    }
                )
    np.fill_diagonal(projected, 0.0)
    projection_report = {
        "projected": True,
        "reason": "topological_projection",
        "node_order": [
            {
                "rank": int(position),
                "node": canonical_axis[node_idx] if node_idx < len(canonical_axis) else f"node_{node_idx}",
                "tier": int(tiers[node_idx]) if node_idx < len(tiers) else None,
                "net_flow": round(float(out_strength[node_idx] - in_strength[node_idx]), 6),
            }
            for position, node_idx in enumerate(ordering)
        ],
        "dropped_edge_count": len(dropped_edges),
        "dropped_edges": sorted(dropped_edges, key=lambda row: row["abs_weight"], reverse=True)[:25],
        "cycle_free_after_projection": not _has_directed_cycle(projected),
    }
    return projected.astype(np.float32), projection_report


def _adjacency_to_edge_rows(adjacency: np.ndarray, canonical_axis: list[str], *, threshold: float) -> list[dict[str, Any]]:
    rows = []
    if adjacency.ndim != 2:
        return rows
    for i, source_name in enumerate(canonical_axis):
        for j, target_name in enumerate(canonical_axis):
            weight = float(adjacency[i, j]) if adjacency.size else 0.0
            if abs(weight) >= threshold:
                rows.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "weight": round(weight, 6),
                        "abs_weight": round(abs(weight), 6),
                    }
                )
    rows.sort(key=lambda item: item["abs_weight"], reverse=True)
    return rows


def _benchmark_edge_key(source_name: str, target_name: str) -> str:
    return f"{source_name} -> {target_name}"


def _bootstrap_edge_stability(
    matrix_tf: np.ndarray,
    admissibility_mask: np.ndarray,
    tiers: np.ndarray,
    canonical_axis: list[str],
) -> dict[str, Any]:
    cfg = _phase2_required_section("benchmark_artifacts")
    if matrix_tf.ndim != 2 or matrix_tf.shape[0] < 2 or matrix_tf.shape[1] == 0:
        return {"available": False, "reason": "insufficient_matrix", "rows": [], "draw_count": 0}
    draw_count = int(cfg["bootstrap_draws"])
    step_count = int(cfg["bootstrap_steps"])
    edge_threshold = float(cfg["edge_threshold"])
    top_k = int(cfg["top_edges"])
    rng = np.random.default_rng(int(cfg["rng_seed"]))
    edge_counts: Counter[str] = Counter()
    edge_weight_sum: defaultdict[str, float] = defaultdict(float)
    for _ in range(draw_count):
        row_idx = rng.integers(0, matrix_tf.shape[0], size=matrix_tf.shape[0])
        sampled = matrix_tf[row_idx]
        projected, _ = _project_to_exact_dag(_notears_optimize(sampled, admissibility_mask, steps=step_count), tiers, canonical_axis)
        for row in _adjacency_to_edge_rows(projected, canonical_axis, threshold=edge_threshold):
            key = _benchmark_edge_key(row["source"], row["target"])
            edge_counts[key] += 1
            edge_weight_sum[key] += float(row["abs_weight"])
    rows = []
    for key, count in edge_counts.items():
        source_name, target_name = key.split(" -> ", 1)
        rows.append(
            {
                "edge": key,
                "source": source_name,
                "target": target_name,
                "selection_frequency": round(float(count) / max(draw_count, 1), 6),
                "mean_abs_weight": round(float(edge_weight_sum[key]) / max(count, 1), 6),
            }
        )
    rows.sort(key=lambda item: (item["selection_frequency"], item["mean_abs_weight"]), reverse=True)
    return {
        "available": True,
        "draw_count": draw_count,
        "optimizer_steps": step_count,
        "edge_threshold": edge_threshold,
        "rows": rows[:top_k],
    }


def _permutation_null_report(
    matrix_tf: np.ndarray,
    observed_adjacency: np.ndarray,
    admissibility_mask: np.ndarray,
    tiers: np.ndarray,
    canonical_axis: list[str],
) -> dict[str, Any]:
    cfg = _phase2_required_section("benchmark_artifacts")
    if matrix_tf.ndim != 2 or matrix_tf.shape[0] < 2 or matrix_tf.shape[1] == 0:
        return {"available": False, "reason": "insufficient_matrix", "draw_count": 0}
    draw_count = int(cfg["permutation_draws"])
    step_count = int(cfg["permutation_steps"])
    edge_threshold = float(cfg["edge_threshold"])
    rng = np.random.default_rng(int(cfg["rng_seed"]) + 1)
    observed_rows = _adjacency_to_edge_rows(observed_adjacency, canonical_axis, threshold=edge_threshold)
    observed_mean_abs_weight = float(np.mean([row["abs_weight"] for row in observed_rows])) if observed_rows else 0.0
    observed_top_edge = float(observed_rows[0]["abs_weight"]) if observed_rows else 0.0
    perm_edge_counts = []
    perm_mean_abs = []
    perm_top_abs = []
    for _ in range(draw_count):
        permuted = np.asarray(matrix_tf, dtype=np.float32).copy()
        for feature_idx in range(permuted.shape[1]):
            permuted[:, feature_idx] = permuted[rng.permutation(permuted.shape[0]), feature_idx]
        projected, _ = _project_to_exact_dag(_notears_optimize(permuted, admissibility_mask, steps=step_count), tiers, canonical_axis)
        perm_rows = _adjacency_to_edge_rows(projected, canonical_axis, threshold=edge_threshold)
        perm_edge_counts.append(len(perm_rows))
        perm_mean_abs.append(float(np.mean([row["abs_weight"] for row in perm_rows])) if perm_rows else 0.0)
        perm_top_abs.append(float(perm_rows[0]["abs_weight"]) if perm_rows else 0.0)
    return {
        "available": True,
        "draw_count": draw_count,
        "optimizer_steps": step_count,
        "edge_threshold": edge_threshold,
        "observed": {
            "edge_count": len(observed_rows),
            "mean_abs_weight": round(observed_mean_abs_weight, 6),
            "top_edge_abs_weight": round(observed_top_edge, 6),
        },
        "permutation_null": {
            "edge_count_mean": round(float(np.mean(perm_edge_counts)) if perm_edge_counts else 0.0, 6),
            "mean_abs_weight_mean": round(float(np.mean(perm_mean_abs)) if perm_mean_abs else 0.0, 6),
            "top_edge_abs_weight_mean": round(float(np.mean(perm_top_abs)) if perm_top_abs else 0.0, 6),
        },
        "null_gap": {
            "edge_count_gap": round(float(len(observed_rows) - (np.mean(perm_edge_counts) if perm_edge_counts else 0.0)), 6),
            "mean_abs_weight_gap": round(float(observed_mean_abs_weight - (np.mean(perm_mean_abs) if perm_mean_abs else 0.0)), 6),
            "top_edge_abs_weight_gap": round(float(observed_top_edge - (np.mean(perm_top_abs) if perm_top_abs else 0.0)), 6),
        },
    }


def _time_stratified_cv_report(
    matrix_tf: np.ndarray,
    admissibility_mask: np.ndarray,
    tiers: np.ndarray,
    canonical_axis: list[str],
    month_axis: list[str],
    province_axis: list[str],
) -> dict[str, Any]:
    cfg = _phase2_required_section("time_stratified_cv")
    if not cfg.get("enabled", False):
        return {"available": False, "reason": "disabled", "rows": []}
    if matrix_tf.ndim != 2 or matrix_tf.shape[0] < 4 or not month_axis or not province_axis:
        return {"available": False, "reason": "insufficient_matrix", "rows": []}
    month_count = len(month_axis)
    province_count = len(province_axis)
    if matrix_tf.shape[0] != month_count * province_count:
        return {"available": False, "reason": "row_month_mismatch", "rows": []}
    holdout_months = max(1, int(cfg["holdout_months"]))
    min_train_months = max(1, int(cfg["min_train_months"]))
    possible_starts = list(range(min_train_months, month_count, holdout_months))
    if not possible_starts:
        return {"available": False, "reason": "insufficient_months", "rows": []}
    starts = possible_starts[-int(cfg["folds"]) :]
    month_index_per_row = np.tile(np.arange(month_count, dtype=int), province_count)
    optimizer_steps = max(8, int(float(_phase2_required_section("notears")["steps"]) * float(cfg["optimizer_step_fraction"])))
    rows = []
    for fold_id, start in enumerate(starts, start=1):
        stop = min(month_count, start + holdout_months)
        train_mask = month_index_per_row < start
        test_mask = (month_index_per_row >= start) & (month_index_per_row < stop)
        if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 1:
            continue
        train_matrix = matrix_tf[train_mask]
        test_matrix = matrix_tf[test_mask]
        adjacency = _notears_optimize(train_matrix, admissibility_mask, steps=optimizer_steps)
        projected, _ = _project_to_exact_dag(adjacency, tiers, canonical_axis)
        prediction = test_matrix @ projected
        mse = float(np.mean((test_matrix - prediction) ** 2)) if test_matrix.size else 0.0
        rows.append(
            {
                "fold_id": fold_id,
                "train_months": month_axis[:start],
                "test_months": month_axis[start:stop],
                "train_row_count": int(train_mask.sum()),
                "test_row_count": int(test_mask.sum()),
                "test_reconstruction_mse": round(mse, 6),
                "edge_count": int(len(_adjacency_to_edge_rows(projected, canonical_axis, threshold=0.03))),
                "cycle_free": not _has_directed_cycle(projected),
            }
        )
    if not rows:
        return {"available": False, "reason": "no_valid_folds", "rows": []}
    return {
        "available": True,
        "fold_count": len(rows),
        "rows": rows,
        "mean_test_reconstruction_mse": round(float(np.mean([row["test_reconstruction_mse"] for row in rows])), 6),
    }


def _collinearity_report(matrix_tf: np.ndarray, canonical_axis: list[str]) -> dict[str, Any]:
    cfg = _phase2_required_section("collinearity")
    if not cfg.get("enabled", False):
        return {"available": False, "reason": "disabled"}
    if matrix_tf.ndim != 2 or matrix_tf.shape[0] < 2 or matrix_tf.shape[1] == 0:
        return {"available": False, "reason": "insufficient_matrix"}
    centered = np.asarray(matrix_tf, dtype=np.float32) - np.asarray(matrix_tf, dtype=np.float32).mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size == 0:
        condition_number = 0.0
        explained = []
    else:
        condition_number = float(singular_values[0] / max(float(singular_values[-1]), _phase2_stabilizer("condition_eps")))
        total_power = float(np.sum(singular_values ** 2))
        explained = (
            ((singular_values ** 2) / max(total_power, _phase2_stabilizer("condition_eps")))[: int(cfg["max_components"])]
            .astype(np.float32)
            .round(6)
            .tolist()
        )
    corr = _safe_feature_corr(matrix_tf)
    high_pairs = []
    threshold = float(cfg["high_corr_threshold"])
    for left_idx in range(len(canonical_axis)):
        for right_idx in range(left_idx + 1, len(canonical_axis)):
            corr_value = float(corr[left_idx, right_idx])
            if abs(corr_value) >= threshold:
                high_pairs.append(
                    {
                        "left": canonical_axis[left_idx],
                        "right": canonical_axis[right_idx],
                        "corr": round(corr_value, 6),
                    }
                )
    high_pairs.sort(key=lambda row: abs(float(row["corr"])), reverse=True)
    return {
        "available": True,
        "condition_number": round(condition_number, 6),
        "condition_warning": bool(condition_number >= float(cfg["condition_number_warning"])),
        "explained_variance_ratio": explained,
        "high_correlation_pair_count": len(high_pairs),
        "high_correlation_pairs": high_pairs[:50],
    }


def _markov_blanket(adjacency: np.ndarray, canonical_axis: list[str]) -> dict[str, Any]:
    edge_threshold = float(_phase2_required("blanket_edge_threshold"))
    target_tokens = ("testing", "diagnos", "viral", "suppression", "art", "cd4", "case_count")
    target_indices = [idx for idx, name in enumerate(canonical_axis) if any(token in name.lower() for token in target_tokens)]
    if not target_indices and canonical_axis:
        degree = np.abs(adjacency).sum(axis=0) + np.abs(adjacency).sum(axis=1)
        target_indices = [int(np.argmax(degree))]
    blanket_indices: set[int] = set(target_indices)
    for idx in target_indices:
        parents = np.where(np.abs(adjacency[:, idx]) > edge_threshold)[0]
        children = np.where(np.abs(adjacency[idx, :]) > edge_threshold)[0]
        blanket_indices.update(parents.tolist())
        blanket_indices.update(children.tolist())
        for child in children:
            spouses = np.where(np.abs(adjacency[:, child]) > edge_threshold)[0]
            blanket_indices.update(spouses.tolist())
    names = [canonical_axis[idx] for idx in sorted(blanket_indices)]
    targets = [canonical_axis[idx] for idx in target_indices]
    return {
        "target_nodes": targets,
        "blanket_nodes": names,
        "blanket_indices": sorted(blanket_indices),
    }


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else round(num / den, 4)


def _curation_status(profile: dict[str, Any]) -> str:
    thresholds = _phase2_required_section("curation_status_thresholds")
    if profile["curation_score"] >= float(thresholds["promoted_candidate"]) and (profile["numeric_support"] > 0 or profile["anchor_support"] > 0 or profile["blanket_member"]):
        return "promoted_candidate"
    if profile["curation_score"] >= float(thresholds["research_candidate"]):
        return "research_candidate"
    if profile["curation_score"] >= float(thresholds["review"]):
        return "review"
    return "rejected"


def _run_phase2_build_base(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase2_dir = ensure_dir(ctx.run_dir / "phase2")
    normalized = read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    parameter_catalog = read_json(ctx.run_dir / "phase1" / "parameter_catalog.json", default=[])
    axis_catalogs = read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    standardized_tensor = load_tensor_artifact(ctx.run_dir / "phase1" / "standardized_tensor.npz")
    canonical_axis = list(axis_catalogs.get("canonical_name", [])) or [row["canonical_name"] for row in parameter_catalog]
    month_axis = list(axis_catalogs.get("month", []))
    province_axis = list(axis_catalogs.get("province", []))

    time_feature_matrix, feature_matrix_mix_report = _phase2_feature_matrix(
        standardized_tensor=standardized_tensor,
        normalized_rows=normalized,
        canonical_axis=canonical_axis,
        month_axis=month_axis,
        province_axis=province_axis,
    )
    exec_cfg = _phase2_execution_cfg()
    feature_matrix_mix_report["execution_backend"] = "torch" if (torch is not None and choose_torch_device(prefer_gpu=bool(exec_cfg.get("prefer_gpu", True))) == "cuda") else "numpy"
    mi_keep_mask, mi_prefilter_report = _mi_prefilter(time_feature_matrix, canonical_axis)
    feature_matrix_mix_report["mi_kept_feature_count"] = int(mi_prefilter_report.get("kept_feature_count", 0))
    feature_matrix_mix_report["mi_dropped_feature_count"] = int(mi_prefilter_report.get("dropped_feature_count", 0))
    node_tiers = _node_tiers(parameter_catalog, canonical_axis)
    tier_mask = _tier_mask(parameter_catalog, canonical_axis)
    lag_mask = _lag_mask(parameter_catalog, canonical_axis)
    skeleton_threshold = float(_phase2_required("skeleton_threshold"))
    notears_cfg = _phase2_required_section("notears")
    pc_skeleton_mask, pc_skeleton_report = _pc_skeleton_from_fisher_z(time_feature_matrix, canonical_axis)
    corr_skeleton_mask = _skeleton_from_corr(time_feature_matrix, threshold=skeleton_threshold)
    skeleton_mask = corr_skeleton_mask * pc_skeleton_mask
    mi_matrix_mask = np.outer(mi_keep_mask, mi_keep_mask).astype(np.float32)
    final_mask = tier_mask * lag_mask * skeleton_mask * mi_matrix_mask
    raw_adjacency = _notears_optimize(time_feature_matrix, final_mask, steps=int(notears_cfg["steps"]))
    adjacency, dag_projection_report = _project_to_exact_dag(raw_adjacency, node_tiers, canonical_axis)
    edge_scores = _adjacency_to_edge_rows(adjacency, canonical_axis, threshold=0.03)
    bootstrap_edge_stability = _bootstrap_edge_stability(time_feature_matrix, final_mask, node_tiers, canonical_axis)
    permutation_null_report = _permutation_null_report(time_feature_matrix, adjacency, final_mask, node_tiers, canonical_axis)
    time_stratified_cv = _time_stratified_cv_report(
        time_feature_matrix,
        final_mask,
        node_tiers,
        canonical_axis,
        month_axis,
        province_axis,
    )
    collinearity_report = _collinearity_report(time_feature_matrix, canonical_axis)
    interop_report = {"source": "numpy", "target": "jax", "used_dlpack": False, "reason": "torch_unavailable"}
    if torch is not None and str(exec_cfg.get("interop_mode") or "").lower() == "dlpack":
        matrix_tensor = to_torch_tensor(time_feature_matrix, device=choose_torch_device(prefer_gpu=bool(exec_cfg.get("prefer_gpu", True))), dtype=torch.float32)
        _jax_view, interop_report = torch_to_jax_handoff(matrix_tensor, prefer_dlpack=True)

    blanket = _markov_blanket(adjacency, canonical_axis)
    blanket_indices = blanket["blanket_indices"]
    core_tensor = standardized_tensor[:, :, blanket_indices] if blanket_indices else standardized_tensor[:, :, : min(1, standardized_tensor.shape[-1])]

    dag_metrics = {
        canonical_axis[idx]: {
            "in_strength": float(np.abs(adjacency[:, idx]).sum()) if adjacency.size else 0.0,
            "out_strength": float(np.abs(adjacency[idx, :]).sum()) if adjacency.size else 0.0,
            "blanket_member": canonical_axis[idx] in blanket["blanket_nodes"],
        }
        for idx in range(len(canonical_axis))
    }

    candidate_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "support_count": 0,
            "numeric_support": 0,
            "direct_support": 0,
            "anchor_support": 0,
            "evidence_weight_sum": 0.0,
            "source_banks": Counter(),
            "geo_resolutions": Counter(),
            "regions": Counter(),
            "time_resolutions": Counter(),
            "domain_families": Counter(),
            "pathway_families": Counter(),
            "tags": Counter(),
            "targets": Counter(),
            "source_ids": set(),
        }
    )
    cooccurrence: Counter[tuple[str, str]] = Counter()
    source_to_canonicals: dict[str, set[str]] = defaultdict(set)
    linkage_counts: Counter[tuple[str, str]] = Counter()

    for row in normalized:
        canonical_name = str(row.get("canonical_name") or "unknown")
        stats = candidate_stats[canonical_name]
        stats["support_count"] += 1
        stats["numeric_support"] += 1 if row.get("model_numeric_value") is not None else 0
        stats["direct_support"] += 1 if row.get("is_direct_measurement") else 0
        stats["anchor_support"] += 1 if row.get("is_anchor_eligible") else 0
        stats["evidence_weight_sum"] += float(row.get("evidence_weight") or 0.0)
        stats["source_banks"][row.get("source_bank") or ""] += 1
        stats["geo_resolutions"][row.get("geo_resolution") or ""] += 1
        if row.get("region"):
            stats["regions"][row["region"]] += 1
        stats["time_resolutions"][row.get("time_resolution") or ""] += 1
        stats["domain_families"][row.get("domain_family") or "mixed"] += 1
        stats["pathway_families"][row.get("pathway_family") or "mixed"] += 1
        for tag in row.get("soft_ontology_tags") or []:
            stats["tags"][tag] += 1
        for target in row.get("linkage_targets") or []:
            stats["targets"][target] += 1
            linkage_counts[(canonical_name, target)] += 1
        if row.get("source_id"):
            stats["source_ids"].add(str(row["source_id"]))
            source_to_canonicals[str(row["source_id"])].add(canonical_name)

    for canonical_names in source_to_canonicals.values():
        names = sorted(canonical_names)
        for idx, left in enumerate(names):
            for right in names[idx + 1 :]:
                cooccurrence[(left, right)] += 1

    candidate_profiles = []
    for canonical_name, stats in sorted(candidate_stats.items()):
        support_count = stats["support_count"]
        numeric_support = stats["numeric_support"]
        direct_support = stats["direct_support"]
        anchor_support = stats["anchor_support"]
        source_diversity = len(stats["source_banks"])
        geo_diversity = len(stats["geo_resolutions"])
        region_diversity = len(stats["regions"])
        time_diversity = len(stats["time_resolutions"])
        domain_diversity = len(stats["domain_families"])
        pathway_diversity = len(stats["pathway_families"])
        tag_diversity = len(stats["tags"])
        target_diversity = len(stats["targets"])
        evidence_score = min(1.0, (stats["evidence_weight_sum"] / max(1, support_count)))
        score_cfg = _phase2_required_section("stability_score_weights")
        denom_cfg = _phase2_required_section("stability_denominators")
        stability_score = min(
            1.0,
            float(score_cfg["source_diversity"]) * min(1.0, source_diversity / float(denom_cfg["source_diversity"]))
            + float(score_cfg["geo_diversity"]) * min(1.0, geo_diversity / float(denom_cfg["geo_diversity"]))
            + float(score_cfg["region_diversity"]) * min(1.0, region_diversity / float(denom_cfg["region_diversity"]))
            + float(score_cfg["time_diversity"]) * min(1.0, time_diversity / float(denom_cfg["time_diversity"]))
            + float(score_cfg["domain_diversity"]) * min(1.0, domain_diversity / float(denom_cfg["domain_diversity"]))
            + float(score_cfg["pathway_diversity"]) * min(1.0, pathway_diversity / float(denom_cfg["pathway_diversity"]))
            + float(score_cfg["tag_diversity"]) * min(1.0, tag_diversity / float(denom_cfg["tag_diversity"]))
            + float(score_cfg["target_diversity"]) * min(1.0, target_diversity / float(denom_cfg["target_diversity"])),
        )
        dag_metric = dag_metrics.get(canonical_name, {"in_strength": 0.0, "out_strength": 0.0, "blanket_member": False})
        dag_cfg = _phase2_required_section("dag_score_weights")
        dag_score = min(
            1.0,
            float(dag_cfg["in_strength"]) * min(1.0, dag_metric["in_strength"])
            + float(dag_cfg["out_strength"]) * min(1.0, dag_metric["out_strength"])
            + float(dag_cfg["blanket_member"]) * (1.0 if dag_metric["blanket_member"] else 0.0),
        )
        relevance_cfg = _phase2_required_section("relevance_score_weights")
        relevance_denoms = _phase2_required_section("relevance_denominators")
        relevance_score = min(
            1.0,
            float(relevance_cfg["support_count"]) * min(1.0, support_count / float(relevance_denoms["support_count"]))
            + float(relevance_cfg["numeric_support_ratio"]) * _safe_ratio(numeric_support, support_count)
            + float(relevance_cfg["direct_support_ratio"]) * _safe_ratio(direct_support, support_count)
            + float(relevance_cfg["anchor_support_ratio"]) * _safe_ratio(anchor_support, support_count)
            + float(relevance_cfg["target_diversity"]) * min(1.0, target_diversity / float(relevance_denoms["target_diversity"]))
            + float(relevance_cfg["evidence_score"]) * evidence_score
            + float(relevance_cfg["dag_score"]) * dag_score,
        )
        mix_cfg = _phase2_required_section("curation_mix")
        curation_score = round(
            min(
                1.0,
                float(mix_cfg["relevance_score"]) * relevance_score
                + float(mix_cfg["stability_score"]) * stability_score
                + float(mix_cfg["dag_score"]) * dag_score,
            ),
            4,
        )
        dominant_tags = [name for name, _ in stats["tags"].most_common(5)]
        dominant_targets = [name for name, _ in stats["targets"].most_common(5)]
        dominant_domain = _dominant_key(dict(stats["domain_families"]), "mixed")
        dominant_pathway = _dominant_key(dict(stats["pathway_families"]), "mixed")
        primary_block, secondary_blocks = _block_from_signals(dominant_domain, dominant_pathway, dominant_tags, dominant_targets)
        candidate_profile = {
            "canonical_name": canonical_name,
            "support_count": support_count,
            "numeric_support": numeric_support,
            "direct_support": direct_support,
            "anchor_support": anchor_support,
            "numeric_support_ratio": _safe_ratio(numeric_support, support_count),
            "direct_support_ratio": _safe_ratio(direct_support, support_count),
            "anchor_support_ratio": _safe_ratio(anchor_support, support_count),
            "source_diversity": source_diversity,
            "geo_diversity": geo_diversity,
            "region_diversity": region_diversity,
            "time_diversity": time_diversity,
            "domain_diversity": domain_diversity,
            "pathway_diversity": pathway_diversity,
            "tag_diversity": tag_diversity,
            "target_diversity": target_diversity,
            "evidence_score": round(evidence_score, 4),
            "stability_score": round(stability_score, 4),
            "relevance_score": round(relevance_score, 4),
            "dag_score": round(dag_score, 4),
            "in_strength": round(dag_metric["in_strength"], 6),
            "out_strength": round(dag_metric["out_strength"], 6),
            "blanket_member": dag_metric["blanket_member"],
            "curation_score": curation_score,
            "primary_block": primary_block,
            "secondary_blocks": secondary_blocks,
            "dominant_domain_family": dominant_domain,
            "dominant_pathway_family": dominant_pathway,
            "soft_ontology_tags": dominant_tags,
            "linkage_targets": dominant_targets,
            "source_banks": dict(stats["source_banks"]),
            "source_id_count": len(stats["source_ids"]),
            "curation_status": "pending",
        }
        candidate_profile["curation_status"] = _curation_status(candidate_profile)
        candidate_profiles.append(candidate_profile)

    ranked_linkages = []
    for (canonical_name, target), support in linkage_counts.most_common():
        candidate_profile = next(item for item in candidate_profiles if item["canonical_name"] == canonical_name)
        linkage_cfg = _phase2_required_section("linkage_score_weights")
        support_denominator = float(_phase2_required("linkage_support_denominator"))
        score = min(
            1.0,
            float(linkage_cfg["support_count"]) * min(1.0, support / support_denominator)
            + float(linkage_cfg["relevance_score"]) * candidate_profile["relevance_score"]
            + float(linkage_cfg["stability_score"]) * candidate_profile["stability_score"]
            + float(linkage_cfg["dag_score"]) * candidate_profile["dag_score"]
            + float(linkage_cfg["numeric_support_ratio"]) * candidate_profile["numeric_support_ratio"],
        )
        ranked_linkages.append(
            {
                "canonical_name": canonical_name,
                "linkage_target": target,
                "support_count": support,
                "numeric_support": candidate_profile["numeric_support"],
                "source_bank_count": candidate_profile["source_diversity"],
                "linkage_score": round(score, 4),
                "primary_block": candidate_profile["primary_block"],
            }
        )

    param_edges = []
    for (left, right), support in cooccurrence.most_common():
        left_profile = next(item for item in candidate_profiles if item["canonical_name"] == left)
        right_profile = next(item for item in candidate_profiles if item["canonical_name"] == right)
        edge_cfg = _phase2_required_section("parameter_edge_weights")
        score = min(
            1.0,
            float(edge_cfg["support_count"]) * support
            + float(edge_cfg["dag_score"]) * min(left_profile["dag_score"], right_profile["dag_score"])
            + float(edge_cfg["stability_score"]) * min(left_profile["stability_score"], right_profile["stability_score"])
            + float(edge_cfg["relevance_score"]) * min(left_profile["relevance_score"], right_profile["relevance_score"])
            + float(edge_cfg["same_block_bonus"]) * (1.0 if left_profile["primary_block"] == right_profile["primary_block"] else 0.0),
        )
        param_edges.append({"left": left, "right": right, "support_count": support, "edge_score": round(score, 4)})

    candidate_profiles.sort(key=lambda item: (item["curation_score"], item["dag_score"], item["support_count"], item["numeric_support"]), reverse=True)
    curated_candidate_blocks = [item for item in candidate_profiles if item["curation_status"] in {"promoted_candidate", "research_candidate", "review"}]
    block_registry = []
    grouped_blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_profile in candidate_profiles:
        grouped_blocks[candidate_profile["primary_block"]].append(candidate_profile)
    for block_name in BLOCK_ORDER:
        members = grouped_blocks.get(block_name, [])
        if not members:
            continue
        block_registry.append(
            {
                "block_name": block_name,
                "member_count": len(members),
                "promoted_candidate_count": sum(1 for item in members if item["curation_status"] == "promoted_candidate"),
                "research_candidate_count": sum(1 for item in members if item["curation_status"] == "research_candidate"),
                "review_count": sum(1 for item in members if item["curation_status"] == "review"),
                "top_canonical_names": [item["canonical_name"] for item in members[:25]],
            }
        )

    linkage_graph = {
        "nodes": sorted(
            {
                *(f"param:{name}" for name in canonical_axis),
                *(f"target:{item['linkage_target']}" for item in ranked_linkages),
                *(f"block:{item['block_name']}" for item in block_registry),
            }
        ),
        "edges": [
            {
                "source": f"param:{item['source']}",
                "target": f"param:{item['target']}",
                "weight": item["weight"],
            }
            for item in edge_scores
        ]
        + [
            {
                "source": f"param:{item['canonical_name']}",
                "target": f"target:{item['linkage_target']}",
                "weight": item["linkage_score"],
                "support_count": item["support_count"],
            }
            for item in ranked_linkages
        ]
        + [
            {
                "source": f"param:{item['left']}",
                "target": f"param:{item['right']}",
                "weight": item["edge_score"],
                "support_count": item["support_count"],
            }
            for item in param_edges
        ],
        "axis_cardinality": {key: len(value) for key, value in axis_catalogs.items()},
    }

    curation_summary = {
        "candidate_profile_count": len(candidate_profiles),
        "promoted_candidate_count": sum(1 for item in candidate_profiles if item["curation_status"] == "promoted_candidate"),
        "research_candidate_count": sum(1 for item in candidate_profiles if item["curation_status"] == "research_candidate"),
        "review_count": sum(1 for item in candidate_profiles if item["curation_status"] == "review"),
        "rejected_count": sum(1 for item in candidate_profiles if item["curation_status"] == "rejected"),
        "block_counts": {block["block_name"]: block["member_count"] for block in block_registry},
    }
    phase15_catalog_path = ctx.run_dir / "phase15" / "mesoscopic_factor_catalog.json"
    phase15_pool_path = ctx.run_dir / "phase15" / "factor_promotion_pool.json"
    phase15_tensor_path = ctx.run_dir / "phase15" / "mesoscopic_factor_tensor.npz"
    phase15_factor_catalog = read_json(phase15_catalog_path, default=[]) if phase15_catalog_path.exists() else []
    phase15_factor_pool = read_json(phase15_pool_path, default=[]) if phase15_pool_path.exists() else []
    phase15_factor_tensor = load_tensor_artifact(phase15_tensor_path) if phase15_tensor_path.exists() else np.zeros((0, 0, 0), dtype=np.float32)
    block_graph_outputs = build_sharded_block_graph_outputs(
        candidate_profiles=candidate_profiles,
        canonical_axis=canonical_axis,
        parameter_catalog=parameter_catalog,
        time_feature_matrix=time_feature_matrix,
        phase15_factor_catalog=phase15_factor_catalog,
        phase15_factor_pool=phase15_factor_pool,
        phase15_factor_tensor=phase15_factor_tensor,
        skeleton_threshold=skeleton_threshold,
        edge_threshold=0.03,
        notears_steps=int(notears_cfg["steps"]),
        block_cfg=_phase2_required_section("block_graph"),
        node_tiers_fn=_node_tiers,
        tier_mask_fn=_tier_mask,
        lag_mask_fn=_lag_mask,
        mi_prefilter_fn=_mi_prefilter,
        pc_skeleton_fn=_pc_skeleton_from_fisher_z,
        corr_skeleton_fn=_skeleton_from_corr,
        notears_fn=_notears_optimize,
        project_dag_fn=_project_to_exact_dag,
        blanket_fn=_markov_blanket,
    )
    block_candidate_banks = block_graph_outputs["block_candidate_banks"]
    block_graph_bundle = block_graph_outputs["block_graph_bundle"]
    bridge_graph_bundle = block_graph_outputs["bridge_graph_bundle"]
    phase3_target_blankets = block_graph_outputs["phase3_target_blankets"]
    pruning_report = block_graph_outputs["pruning_report"]

    backend_map = detect_backends()
    tier_artifact = save_tensor_artifact(
        array=to_torch_tensor(tier_mask, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else tier_mask,
        axis_names=["source_node", "target_node"],
        artifact_dir=phase2_dir,
        stem="tier_mask",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase2_tier_mask"],
    )
    lag_artifact = save_tensor_artifact(
        array=to_torch_tensor(lag_mask, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else lag_mask,
        axis_names=["source_node", "target_node"],
        artifact_dir=phase2_dir,
        stem="lag_mask",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase2_lag_mask"],
    )
    skeleton_artifact = save_tensor_artifact(
        array=to_torch_tensor(skeleton_mask, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else skeleton_mask,
        axis_names=["source_node", "target_node"],
        artifact_dir=phase2_dir,
        stem="skeleton_mask",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase2_skeleton_mask"],
    )
    dag_artifact = save_tensor_artifact(
        array=to_torch_tensor(adjacency, device=choose_torch_device(prefer_gpu=True), dtype=torch.float32) if torch is not None else adjacency,
        axis_names=["source_node", "target_node"],
        artifact_dir=phase2_dir,
        stem="dag_adjacency",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=True) if torch is not None else "cpu",
        notes=["phase2_notears_adjacency"],
    )
    core_artifact = save_tensor_artifact(
        array=to_torch_tensor(core_tensor, device=choose_torch_device(prefer_gpu=True), dtype=torch.float32) if torch is not None else core_tensor,
        axis_names=["province", "month", "core_feature"],
        artifact_dir=phase2_dir,
        stem="core_feature_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=True) if torch is not None else "cpu",
        notes=["phase2_markov_blanket_features"],
    )
    bridge_adjacency_value = bridge_graph_bundle.get("pruned_adjacency")
    bridge_tensor_value = bridge_graph_bundle.get("retained_factor_tensor")
    bridge_adjacency = np.asarray(
        bridge_adjacency_value if isinstance(bridge_adjacency_value, np.ndarray) else np.zeros((0, 0), dtype=np.float32),
        dtype=np.float32,
    )
    bridge_tensor = np.asarray(
        bridge_tensor_value if isinstance(bridge_tensor_value, np.ndarray) else np.zeros((0, 0, 0), dtype=np.float32),
        dtype=np.float32,
    )
    bridge_artifact = None
    bridge_core_artifact = None
    if bridge_adjacency.size:
        bridge_artifact = save_tensor_artifact(
            array=to_torch_tensor(bridge_adjacency, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else bridge_adjacency,
            axis_names=["source_factor", "target_factor"],
            artifact_dir=phase2_dir,
            stem="bridge_dag_adjacency",
            backend="torch" if torch is not None else "numpy",
            device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
            notes=["phase2_block_graph_bridge_dag"],
        )
    if bridge_tensor.size:
        bridge_core_indices = list(phase3_target_blankets.get("blanket_indices", []))
        bridge_core_tensor = bridge_tensor[:, :, bridge_core_indices] if bridge_core_indices else bridge_tensor[:, :, : min(1, bridge_tensor.shape[-1])]
        bridge_core_artifact = save_tensor_artifact(
            array=to_torch_tensor(bridge_core_tensor, device=choose_torch_device(prefer_gpu=True), dtype=torch.float32) if torch is not None else bridge_core_tensor,
            axis_names=["province", "month", "bridge_factor"],
            artifact_dir=phase2_dir,
            stem="phase3_bridge_core_tensor",
            backend="torch" if torch is not None else "numpy",
            device=choose_torch_device(prefer_gpu=True) if torch is not None else "cpu",
            notes=["phase2_phase3_target_blanket_bridge_factors"],
        )

    write_json(phase2_dir / "edge_scores.json", edge_scores)
    write_json(phase2_dir / "markov_blanket.json", blanket)
    write_json(phase2_dir / "candidate_profiles.json", candidate_profiles)
    write_json(phase2_dir / "curated_candidate_blocks.json", curated_candidate_blocks)
    write_json(phase2_dir / "block_registry.json", block_registry)
    write_json(phase2_dir / "ranked_linkages.json", ranked_linkages)
    write_json(phase2_dir / "parameter_edges.json", param_edges)
    write_json(phase2_dir / "linkage_graph.json", linkage_graph)
    write_json(phase2_dir / "curation_summary.json", curation_summary)
    write_json(phase2_dir / "dag_projection_report.json", dag_projection_report)
    write_json(phase2_dir / "mi_prefilter_report.json", mi_prefilter_report)
    write_json(phase2_dir / "pc_skeleton_report.json", pc_skeleton_report)
    write_json(phase2_dir / "bootstrap_edge_stability.json", bootstrap_edge_stability)
    write_json(phase2_dir / "permutation_null_report.json", permutation_null_report)
    write_json(phase2_dir / "time_stratified_cv_report.json", time_stratified_cv)
    write_json(phase2_dir / "collinearity_report.json", collinearity_report)
    write_json(phase2_dir / "interop_report.json", interop_report)
    write_json(phase2_dir / "feature_matrix_mix_report.json", feature_matrix_mix_report)
    write_json(phase2_dir / "block_candidate_banks.json", {"rows": block_candidate_banks})
    write_json(phase2_dir / "block_graph_bundle.json", block_graph_bundle)
    write_json(phase2_dir / "retained_mesoscopic_factor_catalog.json", {"rows": bridge_graph_bundle.get("retained_factor_rows", [])})
    write_json(phase2_dir / "bridge_dag_report.json", {key: value for key, value in bridge_graph_bundle.items() if key not in {"pruned_adjacency", "retained_factor_tensor", "retained_factor_rows"}})
    write_json(phase2_dir / "phase3_target_blankets.json", phase3_target_blankets)
    write_json(phase2_dir / "pruning_report.json", pruning_report)

    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase1"),
        extracted_dir=str(phase2_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase2": "completed"},
        artifact_paths={
            "tier_mask": tier_artifact["value_path"],
            "lag_mask": lag_artifact["value_path"],
            "skeleton_mask": skeleton_artifact["value_path"],
            "dag_adjacency": dag_artifact["value_path"],
            "edge_scores": str(phase2_dir / "edge_scores.json"),
            "markov_blanket": str(phase2_dir / "markov_blanket.json"),
            "core_feature_tensor": core_artifact["value_path"],
            "candidate_profiles": str(phase2_dir / "candidate_profiles.json"),
            "curated_candidate_blocks": str(phase2_dir / "curated_candidate_blocks.json"),
            "block_registry": str(phase2_dir / "block_registry.json"),
            "ranked_linkages": str(phase2_dir / "ranked_linkages.json"),
            "parameter_edges": str(phase2_dir / "parameter_edges.json"),
            "linkage_graph": str(phase2_dir / "linkage_graph.json"),
            "curation_summary": str(phase2_dir / "curation_summary.json"),
            "dag_projection_report": str(phase2_dir / "dag_projection_report.json"),
            "mi_prefilter_report": str(phase2_dir / "mi_prefilter_report.json"),
            "pc_skeleton_report": str(phase2_dir / "pc_skeleton_report.json"),
            "bootstrap_edge_stability": str(phase2_dir / "bootstrap_edge_stability.json"),
            "permutation_null_report": str(phase2_dir / "permutation_null_report.json"),
            "time_stratified_cv_report": str(phase2_dir / "time_stratified_cv_report.json"),
            "collinearity_report": str(phase2_dir / "collinearity_report.json"),
            "interop_report": str(phase2_dir / "interop_report.json"),
            "feature_matrix_mix_report": str(phase2_dir / "feature_matrix_mix_report.json"),
            "block_candidate_banks": str(phase2_dir / "block_candidate_banks.json"),
            "block_graph_bundle": str(phase2_dir / "block_graph_bundle.json"),
            "retained_mesoscopic_factor_catalog": str(phase2_dir / "retained_mesoscopic_factor_catalog.json"),
            "bridge_dag_report": str(phase2_dir / "bridge_dag_report.json"),
            "phase3_target_blankets": str(phase2_dir / "phase3_target_blankets.json"),
            "pruning_report": str(phase2_dir / "pruning_report.json"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, backend_map["torch"].selected, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        source_count=len(candidate_profiles),
        canonical_candidate_count=len(curated_candidate_blocks),
        notes=["phase2_causal_discovery:torch_notears"],
    ).to_dict()
    manifest["profile_id"] = profile
    truth_paths = write_ground_truth_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        profile_id=profile,
        checks=[
            {"name": "dag_finite", "passed": bool(np.isfinite(adjacency).all())},
            {"name": "blanket_nonempty", "passed": bool(blanket.get("blanket_nodes", []))},
            {"name": "candidate_profiles_present", "passed": bool(candidate_profiles)},
            {"name": "curation_sorted", "passed": [row["curation_score"] for row in candidate_profiles] == sorted((row["curation_score"] for row in candidate_profiles), reverse=True)},
            {"name": "dag_projection_report_present", "passed": dag_projection_report.get("projected") is True},
            {"name": "pc_skeleton_report_present", "passed": pc_skeleton_report.get("available") is True},
            {"name": "time_stratified_cv_present", "passed": bool(time_stratified_cv.get("available"))},
            {"name": "collinearity_report_present", "passed": bool(collinearity_report.get("available"))},
            {"name": "block_candidate_banks_present", "passed": bool(block_candidate_banks)},
            {"name": "block_graph_bundle_present", "passed": bool(block_graph_bundle.get("blocks"))},
        ],
        truth_sources=["prior_truth", "synthetic_truth", "proxy_truth"],
        stage_manifest_path=str(phase2_dir / "phase2_manifest.json"),
        summary={
            "candidate_profile_count": len(candidate_profiles),
            "curated_candidate_count": len(curated_candidate_blocks),
            "blanket_count": len(blanket.get("blanket_nodes", [])),
            "edge_count": len(edge_scores),
            "soft_feature_count": int(feature_matrix_mix_report.get("soft_feature_count", 0)),
            "block_graph_count": len(block_graph_bundle.get("blocks", [])),
        },
    )
    gold_profile = dict((_HIV_PLUGIN.gold_standard_profiles or {}).get("phase2", {}) or {})
    gold_paths = write_gold_standard_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        profile_id=profile,
        gold_profile=gold_profile,
        checks=[
            {"name": "gold_standard_profile_declared", "passed": bool(gold_profile)},
            {"name": "benchmark_family_declared", "passed": str(gold_profile.get("mode") or "") == "benchmark_family"},
            {"name": "acyclicity_exact", "passed": not _has_directed_cycle(adjacency)},
            {"name": "tier_mask_respected", "passed": bool(np.all(np.abs(adjacency[tier_mask == 0]) <= 1e-6))},
            {"name": "lag_mask_respected", "passed": bool(np.all(np.abs(adjacency[lag_mask == 0]) <= 1e-6))},
            {"name": "markov_blanket_present", "passed": bool(blanket.get("blanket_nodes", []))},
            {"name": "permutation_or_bootstrap_benchmark_present", "passed": bootstrap_edge_stability.get("available") or permutation_null_report.get("available")},
            {"name": "time_stratified_cv_present", "passed": bool(time_stratified_cv.get("available"))},
            {"name": "collinearity_audited", "passed": bool(collinearity_report.get("available"))},
            {"name": "block_graph_bundle_present", "passed": bool(block_graph_bundle.get("blocks"))},
        ],
        stage_manifest_path=str(phase2_dir / "phase2_manifest.json"),
        summary={
            "edge_count": len(edge_scores),
            "blanket_count": len(blanket.get("blanket_nodes", [])),
            "cycle_detected": _has_directed_cycle(adjacency),
            "bootstrap_draw_count": int(bootstrap_edge_stability.get("draw_count", 0)),
            "permutation_draw_count": int(permutation_null_report.get("draw_count", 0)),
            "time_cv_fold_count": int(time_stratified_cv.get("fold_count", 0)),
            "soft_feature_count": int(feature_matrix_mix_report.get("soft_feature_count", 0)),
            "bridge_factor_count": int(len(bridge_graph_bundle.get("retained_factor_rows", []))),
        },
    )
    manifest["artifact_paths"].update(gold_paths)
    manifest["artifact_paths"].update(truth_paths)
    boundary_paths = write_boundary_shape_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        profile_id=profile,
        boundaries=[
            {
                "name": "tier_mask",
                "kind": "tensor",
                "path": str(phase2_dir / "tier_mask.npz"),
                "expected_shape": list(tier_mask.shape),
                "expected_axis_names": ["source_node", "target_node"],
                "min_rank": 2,
                "finite_required": True,
            },
            {
                "name": "lag_mask",
                "kind": "tensor",
                "path": str(phase2_dir / "lag_mask.npz"),
                "expected_shape": list(lag_mask.shape),
                "expected_axis_names": ["source_node", "target_node"],
                "min_rank": 2,
                "finite_required": True,
            },
            {
                "name": "skeleton_mask",
                "kind": "tensor",
                "path": str(phase2_dir / "skeleton_mask.npz"),
                "expected_shape": list(skeleton_mask.shape),
                "expected_axis_names": ["source_node", "target_node"],
                "min_rank": 2,
                "finite_required": True,
            },
            {
                "name": "dag_adjacency",
                "kind": "tensor",
                "path": str(phase2_dir / "dag_adjacency.npz"),
                "expected_shape": list(adjacency.shape),
                "expected_axis_names": ["source_node", "target_node"],
                "min_rank": 2,
                "finite_required": True,
            },
            {
                "name": "core_feature_tensor",
                "kind": "tensor",
                "path": str(phase2_dir / "core_feature_tensor.npz"),
                "expected_shape": list(core_tensor.shape),
                "expected_axis_names": ["province", "month", "core_feature"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "candidate_profiles",
                "kind": "json_rows",
                "path": str(phase2_dir / "candidate_profiles.json"),
                "min_rows": 1,
                "expected_row_count": len(candidate_profiles),
            },
            {
                "name": "curation_summary",
                "kind": "json_dict",
                "path": str(phase2_dir / "curation_summary.json"),
                "expected_keys": ["candidate_profile_count", "promoted_candidate_count", "research_candidate_count", "review_count", "rejected_count", "block_counts"],
            },
            {
                "name": "block_candidate_banks",
                "kind": "json_dict",
                "path": str(phase2_dir / "block_candidate_banks.json"),
                "expected_keys": ["rows"],
            },
            {
                "name": "block_graph_bundle",
                "kind": "json_dict",
                "path": str(phase2_dir / "block_graph_bundle.json"),
                "expected_keys": ["blocks"],
            },
            {
                "name": "phase3_target_blankets",
                "kind": "json_dict",
                "path": str(phase2_dir / "phase3_target_blankets.json"),
                "expected_keys": ["target_factor_ids", "blanket_factor_ids", "blanket_indices", "phase3_member_canonical_names"],
            },
        ],
        summary={
            "candidate_profile_count": len(candidate_profiles),
            "edge_count": len(edge_scores),
            "soft_feature_count": int(feature_matrix_mix_report.get("soft_feature_count", 0)),
            "block_graph_count": len(block_graph_bundle.get("blocks", [])),
        },
    )
    manifest["artifact_paths"].update(boundary_paths)
    write_json(phase2_dir / "phase2_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase2_build",
        [
            phase2_dir / "tier_mask.npz",
            phase2_dir / "lag_mask.npz",
            phase2_dir / "skeleton_mask.npz",
            phase2_dir / "dag_adjacency.npz",
            phase2_dir / "edge_scores.json",
            phase2_dir / "markov_blanket.json",
            phase2_dir / "core_feature_tensor.npz",
            phase2_dir / "candidate_profiles.json",
            phase2_dir / "curated_candidate_blocks.json",
            phase2_dir / "block_registry.json",
            phase2_dir / "ranked_linkages.json",
            phase2_dir / "parameter_edges.json",
            phase2_dir / "linkage_graph.json",
            phase2_dir / "curation_summary.json",
            phase2_dir / "dag_projection_report.json",
            phase2_dir / "mi_prefilter_report.json",
            phase2_dir / "pc_skeleton_report.json",
            phase2_dir / "bootstrap_edge_stability.json",
            phase2_dir / "permutation_null_report.json",
            phase2_dir / "time_stratified_cv_report.json",
            phase2_dir / "collinearity_report.json",
            phase2_dir / "interop_report.json",
            phase2_dir / "feature_matrix_mix_report.json",
            phase2_dir / "block_candidate_banks.json",
            phase2_dir / "block_graph_bundle.json",
            phase2_dir / "retained_mesoscopic_factor_catalog.json",
            phase2_dir / "bridge_dag_report.json",
            phase2_dir / "phase3_target_blankets.json",
            phase2_dir / "pruning_report.json",
            phase2_dir / "gold_standard_manifest.json",
            phase2_dir / "gold_standard_checks.json",
            phase2_dir / "gold_standard_summary.json",
            phase2_dir / "boundary_shape_manifest.json",
            phase2_dir / "boundary_shape_checks.json",
            phase2_dir / "boundary_shape_summary.json",
            phase2_dir / "phase2_manifest.json",
            *( [Path(bridge_artifact["value_path"])] if bridge_artifact is not None else [] ),
            *( [Path(bridge_core_artifact["value_path"])] if bridge_core_artifact is not None else [] ),
        ],
    )
    return manifest


def run_phase2_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    manifest = _run_phase2_build_base(run_id=run_id, plugin_id=plugin_id, profile=profile)
    if profile == PHASE15_PROFILE_ID:
        return augment_phase2_for_rescue_v2(run_id=run_id, plugin_id=plugin_id, manifest=manifest)
    return manifest
