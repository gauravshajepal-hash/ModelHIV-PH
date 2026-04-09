from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import load_tensor_artifact, read_json


_HIV_PLUGIN = get_disease_plugin("hiv")


def _safe_zscore_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    centered = array - array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.nan_to_num(centered / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _weighted_standardize(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(matrix, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    if array.ndim != 2 or array.size == 0:
        empty = np.zeros_like(array, dtype=np.float32)
        return empty, np.zeros((array.shape[1],), dtype=np.float32), np.ones((array.shape[1],), dtype=np.float32)
    total = float(np.sum(sample_weights))
    if total <= 0.0:
        mean = array.mean(axis=0)
        var = array.var(axis=0)
    else:
        mean = (sample_weights * array).sum(axis=0) / total
        centered = array - mean
        var = (sample_weights * centered * centered).sum(axis=0) / total
    scale = np.sqrt(np.maximum(var, 1e-6))
    standardized = (array - mean) / scale
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    return standardized.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _soft_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)


def _svd_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return values
    u, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    shrunk = np.maximum(singular_values - float(threshold), 0.0)
    if not np.any(shrunk > 0.0):
        return np.zeros_like(values, dtype=np.float32)
    return (u * shrunk) @ vt


def _estimate_step_size(design_matrix: np.ndarray, ridge_penalty: float) -> float:
    if design_matrix.size == 0:
        return 1.0
    try:
        top_singular = float(np.linalg.svd(design_matrix, compute_uv=False, full_matrices=False)[0])
    except Exception:
        top_singular = float(np.linalg.norm(design_matrix, ord=2))
    lipschitz = max((top_singular * top_singular) / max(1.0, float(design_matrix.shape[0])) + float(ridge_penalty), 1e-6)
    return 1.0 / lipschitz


def _sparse_plus_low_rank_objective(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sparse_matrix: np.ndarray,
    low_rank_matrix: np.ndarray,
    sample_weights: np.ndarray,
    sparse_penalty: float,
    low_rank_penalty: float,
) -> dict[str, float]:
    weighted_design = design_matrix * np.sqrt(sample_weights)[:, None]
    weighted_response = response_matrix * np.sqrt(sample_weights)[:, None]
    residual = weighted_response - weighted_design @ (sparse_matrix + low_rank_matrix)
    mse = 0.5 * float(np.mean(residual * residual)) if residual.size else 0.0
    sparse_norm = float(np.sum(np.abs(sparse_matrix)))
    nuclear_norm = float(np.sum(np.linalg.svd(low_rank_matrix, compute_uv=False, full_matrices=False))) if low_rank_matrix.size else 0.0
    total = mse + float(sparse_penalty) * sparse_norm + float(low_rank_penalty) * nuclear_norm
    return {
        "mse": round(mse, 6),
        "sparse_norm": round(sparse_norm, 6),
        "nuclear_norm": round(nuclear_norm, 6),
        "total": round(total, 6),
    }


def _enforce_no_self_edges(matrix: np.ndarray, block_count: int, max_lag: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    for lag_idx in range(max_lag):
        for block_idx in range(block_count):
            values[lag_idx * block_count + block_idx, block_idx] = 0.0
    return values


def _fit_sparse_plus_low_rank_matrix(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sample_weights: np.ndarray,
    ridge_penalty: float,
    sparse_penalty: float,
    low_rank_penalty: float,
    decomposition_steps: int,
    convergence_tol: float,
    block_count: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    feature_count = int(design_matrix.shape[1]) if design_matrix.ndim == 2 else 0
    target_count = int(response_matrix.shape[1]) if response_matrix.ndim == 2 else 0
    if feature_count == 0 or target_count == 0:
        zeros = np.zeros((feature_count, target_count), dtype=np.float32)
        return zeros, zeros, {"mse": 0.0, "sparse_norm": 0.0, "nuclear_norm": 0.0, "total": 0.0}
    sqrt_weights = np.sqrt(np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1))
    weighted_design = np.asarray(design_matrix, dtype=np.float32) * sqrt_weights
    weighted_response = np.asarray(response_matrix, dtype=np.float32) * sqrt_weights
    gram = weighted_design.T @ weighted_design
    gram += float(ridge_penalty) * np.eye(feature_count, dtype=np.float32)
    rhs = weighted_design.T @ weighted_response
    try:
        coefficient_matrix = np.linalg.solve(gram, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        coefficient_matrix = (np.linalg.pinv(gram) @ rhs).astype(np.float32)
    coefficient_matrix = _enforce_no_self_edges(coefficient_matrix, block_count, max_lag)
    sparse_matrix = _soft_threshold(
        coefficient_matrix,
        float(sparse_penalty) + float(low_rank_penalty),
    )
    sparse_matrix = _enforce_no_self_edges(sparse_matrix, block_count, max_lag)
    low_rank_matrix = _svd_threshold(coefficient_matrix - sparse_matrix, float(low_rank_penalty))
    low_rank_matrix = _enforce_no_self_edges(low_rank_matrix, block_count, max_lag)
    objective = _sparse_plus_low_rank_objective(
        response_matrix=response_matrix,
        design_matrix=design_matrix,
        sparse_matrix=sparse_matrix,
        low_rank_matrix=low_rank_matrix,
        sample_weights=sample_weights,
        sparse_penalty=sparse_penalty,
        low_rank_penalty=low_rank_penalty,
    )
    return sparse_matrix.astype(np.float32), low_rank_matrix.astype(np.float32), objective


def _temporal_block_permutation_indices(
    sample_pairs: list[tuple[int, int]],
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not sample_pairs:
        return np.zeros((0,), dtype=np.int64)
    by_unit: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_idx, (unit_idx, month_idx) in enumerate(sample_pairs):
        by_unit[int(unit_idx)].append((int(month_idx), int(row_idx)))
    mapping = np.arange(len(sample_pairs), dtype=np.int64)
    local_block_length = max(1, int(block_length))
    for rows in by_unit.values():
        if len(rows) <= 1:
            continue
        rows.sort(key=lambda item: (item[0], item[1]))
        ordered_indices = [row_idx for _, row_idx in rows]
        blocks = [
            ordered_indices[offset : offset + local_block_length]
            for offset in range(0, len(ordered_indices), local_block_length)
        ]
        if len(blocks) <= 1:
            shift = int(rng.integers(1, len(ordered_indices)))
            permuted = np.roll(np.asarray(ordered_indices, dtype=np.int64), shift).tolist()
        else:
            order = rng.permutation(len(blocks))
            permuted = [row_idx for block_idx in order.tolist() for row_idx in blocks[int(block_idx)]]
        mapping[np.asarray(ordered_indices, dtype=np.int64)] = np.asarray(permuted, dtype=np.int64)
    return mapping.astype(np.int64)


def _sample_weight_vector(
    uncertainty_tensor: np.ndarray | None,
    sample_pairs: list[tuple[int, int]],
    *,
    max_lag: int,
) -> np.ndarray:
    if uncertainty_tensor is None or not sample_pairs:
        return np.ones((len(sample_pairs),), dtype=np.float32)
    uncertainty = np.asarray(uncertainty_tensor, dtype=np.float32)
    weights = np.ones((len(sample_pairs),), dtype=np.float32)
    for row_idx, (unit_idx, month_idx) in enumerate(sample_pairs):
        if unit_idx >= uncertainty.shape[0] or month_idx >= uncertainty.shape[1]:
            continue
        values = [uncertainty[unit_idx, month_idx, :]]
        for lag in range(1, max_lag + 1):
            values.append(uncertainty[unit_idx, month_idx - lag, :])
        pooled = np.concatenate(values, axis=0)
        mean_var = float(np.mean(np.square(pooled)))
        weights[row_idx] = 1.0 / max(mean_var, 1e-4)
    median = float(np.median(weights)) if weights.size else 1.0
    if median > 0.0:
        weights /= median
    return np.clip(weights, 0.1, 10.0).astype(np.float32)


def _build_sample_arrays(
    *,
    state_tensor: np.ndarray,
    block_axis: list[str],
    phi_by_block: dict[str, float],
    max_lag: int,
    uncertainty_tensor: np.ndarray | None = None,
) -> dict[str, Any]:
    tensor = np.asarray(state_tensor, dtype=np.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    unit_count, month_count, block_count = tensor.shape
    sample_pairs: list[tuple[int, int]] = []
    design_rows: list[np.ndarray] = []
    response_rows: list[np.ndarray] = []
    phi = np.asarray([float(phi_by_block.get(block_id, 0.0)) for block_id in block_axis], dtype=np.float32)
    for unit_idx in range(unit_count):
        for month_idx in range(max_lag, month_count):
            current = tensor[unit_idx, month_idx, :]
            prev = tensor[unit_idx, month_idx - 1, :]
            innovation = current - phi * prev
            lag_parts = [tensor[unit_idx, month_idx - lag, :] for lag in range(1, max_lag + 1)]
            sample_pairs.append((unit_idx, month_idx))
            design_rows.append(np.concatenate(lag_parts, axis=0).astype(np.float32))
            response_rows.append(innovation.astype(np.float32))
    design_matrix = np.vstack(design_rows).astype(np.float32) if design_rows else np.zeros((0, block_count * max_lag), dtype=np.float32)
    response_matrix = np.vstack(response_rows).astype(np.float32) if response_rows else np.zeros((0, block_count), dtype=np.float32)
    sample_weights = _sample_weight_vector(uncertainty_tensor, sample_pairs, max_lag=max_lag)
    return {
        "design_matrix": design_matrix,
        "response_matrix": response_matrix,
        "sample_pairs": sample_pairs,
        "sample_weights": sample_weights,
        "effective_sample_count": len(sample_pairs),
    }


def _feature_rows(block_axis: list[str], max_lag: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_idx = 0
    for lag in range(1, max_lag + 1):
        for source in block_axis:
            rows.append({"feature_index": feature_idx, "source": source, "lag": lag})
            feature_idx += 1
    return rows


def _matrix_rows(
    matrix: np.ndarray,
    *,
    block_axis: list[str],
    max_lag: int,
    threshold: float,
    stability_matrix: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_count = len(block_axis)
    values = np.asarray(matrix, dtype=np.float32)
    for feature_idx in range(values.shape[0]):
        lag = (feature_idx // block_count) + 1
        source = block_axis[feature_idx % block_count]
        for target_idx, target in enumerate(block_axis):
            weight = float(values[feature_idx, target_idx])
            if abs(weight) < float(threshold):
                continue
            stability = float(stability_matrix[feature_idx, target_idx]) if stability_matrix is not None else 1.0
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "lag": lag,
                    "weight": round(weight, 6),
                    "score": round(abs(weight) * stability, 6),
                    "stability": round(stability, 6),
                }
            )
    rows.sort(key=lambda row: (float(row["score"]), float(abs(row["weight"])), str(row["source"]), str(row["target"])), reverse=True)
    return rows


def _innovation_variance_rows(state_tensor: np.ndarray, block_axis: list[str], phi_by_block: dict[str, float]) -> list[dict[str, Any]]:
    tensor = np.asarray(state_tensor, dtype=np.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    rows: list[dict[str, Any]] = []
    for block_idx, block_id in enumerate(block_axis):
        phi = float(phi_by_block.get(block_id, 0.0))
        current = tensor[:, 1:, block_idx]
        prev = tensor[:, :-1, block_idx]
        innovation = current - phi * prev
        variance = float(np.var(innovation)) if innovation.size else 0.0
        rows.append({"block_id": block_id, "phi": round(phi, 6), "innovation_variance": round(variance, 6)})
    return rows


def _estimate_phi_by_series_tensor(state_tensor: np.ndarray, axis_ids: list[str]) -> dict[str, float]:
    tensor = np.asarray(state_tensor, dtype=np.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    phi_by_axis: dict[str, float] = {}
    for axis_idx, axis_id in enumerate(axis_ids):
        if tensor.shape[1] <= 1:
            phi_by_axis[str(axis_id)] = 0.0
            continue
        previous = tensor[:, :-1, axis_idx].reshape(-1).astype(np.float64)
        current = tensor[:, 1:, axis_idx].reshape(-1).astype(np.float64)
        denom = float(np.dot(previous, previous))
        phi = float(np.dot(previous, current) / denom) if denom > 1e-8 else 0.0
        phi_by_axis[str(axis_id)] = float(np.clip(phi, -0.995, 0.995))
    return phi_by_axis


def _forecast_validation_score(
    *,
    state_tensor: np.ndarray,
    sample_weights: np.ndarray,
    phi_by_block: dict[str, float],
    block_axis: list[str],
    max_lag: int,
    ridge_penalty: float,
    sparse_penalty: float,
    low_rank_penalty: float,
) -> tuple[float, int]:
    sample_arrays = _build_sample_arrays(
        state_tensor=state_tensor,
        block_axis=block_axis,
        phi_by_block=phi_by_block,
        max_lag=max_lag,
        uncertainty_tensor=None,
    )
    design_matrix = np.asarray(sample_arrays["design_matrix"], dtype=np.float32)
    response_matrix = np.asarray(sample_arrays["response_matrix"], dtype=np.float32)
    weights = np.asarray(sample_arrays["sample_weights"], dtype=np.float32)
    sample_count = int(design_matrix.shape[0])
    if sample_count <= max(12, 3 * design_matrix.shape[1]):
        return float("inf"), sample_count
    split = max(max_lag + 8, int(np.floor(sample_count * 0.75)))
    split = min(split, sample_count - max(8, design_matrix.shape[1]))
    if split <= max_lag or split >= sample_count:
        return float("inf"), sample_count
    train_design = design_matrix[:split]
    train_response = response_matrix[:split]
    train_weights = weights[:split]
    valid_design = design_matrix[split:]
    valid_response = response_matrix[split:]
    valid_weights = weights[split:]
    if train_design.size == 0 or valid_design.size == 0:
        return float("inf"), sample_count
    standardized_train_design, design_mean, design_scale = _weighted_standardize(train_design, train_weights)
    standardized_train_response, response_mean, response_scale = _weighted_standardize(train_response, train_weights)
    sparse_matrix, low_rank_matrix, _ = _fit_sparse_plus_low_rank_matrix(
        response_matrix=standardized_train_response,
        design_matrix=standardized_train_design,
        sample_weights=train_weights,
        ridge_penalty=ridge_penalty,
        sparse_penalty=sparse_penalty,
        low_rank_penalty=low_rank_penalty,
        decomposition_steps=96,
        convergence_tol=1e-5,
        block_count=len(block_axis),
        max_lag=max_lag,
    )
    valid_standardized_design = (valid_design.astype(np.float32) - design_mean) / np.where(design_scale > 1e-6, design_scale, 1.0)
    valid_standardized_response = (valid_response.astype(np.float32) - response_mean) / np.where(response_scale > 1e-6, response_scale, 1.0)
    valid_standardized_design = np.nan_to_num(valid_standardized_design, nan=0.0, posinf=0.0, neginf=0.0)
    valid_standardized_response = np.nan_to_num(valid_standardized_response, nan=0.0, posinf=0.0, neginf=0.0)
    residual = valid_standardized_response - valid_standardized_design @ (sparse_matrix + low_rank_matrix)
    weighted_residual = residual * np.sqrt(valid_weights)[:, None]
    mse = float(np.mean(weighted_residual * weighted_residual)) if weighted_residual.size else float("inf")
    singular_values = np.linalg.svd(low_rank_matrix, compute_uv=False, full_matrices=False) if low_rank_matrix.size else np.zeros((0,), dtype=np.float32)
    hidden_rank = int(np.sum(singular_values > 1e-6))
    complexity = (
        0.01 * float(np.mean(np.abs(sparse_matrix) > 1e-6))
        + 0.01 * float(hidden_rank)
        + 0.005 * float(max_lag)
    )
    return mse + complexity, sample_count


def _candidate_list(cfg: dict[str, Any], key: str, default: list[float]) -> list[float]:
    value = cfg.get(key)
    if isinstance(value, list) and value:
        return [float(item) for item in value]
    return list(default)


def _select_temporal_hyperparameters(
    *,
    state_tensor: np.ndarray,
    block_axis: list[str],
    phi_by_block: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    lag_candidates = [int(value) for value in cfg.get("candidate_max_lags", [cfg.get("max_lag", 1)])]
    ridge_candidates = _candidate_list(cfg, "candidate_ridge_penalties", [float(cfg.get("ridge_penalty", 0.1))])
    sparse_candidates = _candidate_list(cfg, "candidate_sparse_penalties", [float(cfg.get("sparse_penalty", 0.05))])
    low_rank_candidates = _candidate_list(cfg, "candidate_low_rank_penalties", [float(cfg.get("low_rank_penalty", 0.05))])
    best = {
        "score": float("inf"),
        "max_lag": int(cfg.get("max_lag", 1)),
        "ridge_penalty": float(cfg.get("ridge_penalty", 0.1)),
        "sparse_penalty": float(cfg.get("sparse_penalty", 0.05)),
        "low_rank_penalty": float(cfg.get("low_rank_penalty", 0.05)),
        "effective_sample_count": 0,
    }
    for lag in lag_candidates:
        for ridge_penalty in ridge_candidates:
            for sparse_penalty in sparse_candidates:
                for low_rank_penalty in low_rank_candidates:
                    score, sample_count = _forecast_validation_score(
                        state_tensor=state_tensor,
                        sample_weights=np.ones((1,), dtype=np.float32),
                        phi_by_block=phi_by_block,
                        block_axis=block_axis,
                        max_lag=int(lag),
                        ridge_penalty=float(ridge_penalty),
                        sparse_penalty=float(sparse_penalty),
                        low_rank_penalty=float(low_rank_penalty),
                    )
                    candidate = {
                        "score": score,
                        "max_lag": int(lag),
                        "ridge_penalty": float(ridge_penalty),
                        "sparse_penalty": float(sparse_penalty),
                        "low_rank_penalty": float(low_rank_penalty),
                        "effective_sample_count": int(sample_count),
                    }
                    if (
                        candidate["score"] < best["score"]
                        or (
                            candidate["score"] == best["score"]
                            and (
                                candidate["max_lag"],
                                candidate["ridge_penalty"],
                                candidate["sparse_penalty"],
                                candidate["low_rank_penalty"],
                            )
                            < (
                                best["max_lag"],
                                best["ridge_penalty"],
                                best["sparse_penalty"],
                                best["low_rank_penalty"],
                            )
                        )
                    ):
                        best = candidate
    return best


def _permutation_null_thresholds(
    *,
    design_matrix: np.ndarray,
    response_matrix: np.ndarray,
    sample_weights: np.ndarray,
    sample_pairs: list[tuple[int, int]],
    block_count: int,
    max_lag: int,
    ridge_penalty: float,
    sparse_penalty: float,
    low_rank_penalty: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    permutation_count = max(1, int(cfg.get("null_permutations", 8)))
    quantile = float(cfg.get("null_quantile", 0.95))
    rng = np.random.default_rng(int(cfg.get("rng_seed", 0)) + 911)
    sparse_abs: list[float] = []
    low_rank_abs: list[float] = []
    singular_values: list[float] = []
    null_block_length = max(1, int(cfg.get("null_block_length", cfg.get("bootstrap_block_length", 6))))
    for _ in range(permutation_count):
        permuted_indices = _temporal_block_permutation_indices(
            sample_pairs,
            block_length=null_block_length,
            rng=rng,
        )
        permuted_response = response_matrix[permuted_indices]
        null_sparse, null_low_rank, _ = _fit_sparse_plus_low_rank_matrix(
            response_matrix=permuted_response,
            design_matrix=design_matrix,
            sample_weights=sample_weights,
            ridge_penalty=ridge_penalty,
            sparse_penalty=sparse_penalty,
            low_rank_penalty=low_rank_penalty,
            decomposition_steps=max(32, int(cfg.get("decomposition_steps", 120)) // 2),
            convergence_tol=float(cfg.get("convergence_tol", 1e-5)),
            block_count=block_count,
            max_lag=max_lag,
        )
        sparse_abs.extend(np.abs(null_sparse).reshape(-1).tolist())
        low_rank_abs.extend(np.abs(null_low_rank).reshape(-1).tolist())
        if null_low_rank.size:
            singular_values.extend(np.linalg.svd(null_low_rank, compute_uv=False, full_matrices=False).tolist())
    sparse_threshold = float(np.quantile(np.asarray(sparse_abs or [0.0], dtype=np.float32), quantile))
    low_rank_threshold = float(np.quantile(np.asarray(low_rank_abs or [0.0], dtype=np.float32), quantile))
    rank_threshold = float(np.quantile(np.asarray(singular_values or [0.0], dtype=np.float32), quantile))
    return {
        "direct_edge_threshold": max(sparse_threshold, 1e-6),
        "hidden_edge_threshold": max(low_rank_threshold, 1e-6),
        "rank_threshold": max(rank_threshold, 1e-6),
        "permutation_count": permutation_count,
        "null_quantile": quantile,
        "null_block_length": null_block_length,
        "null_mode": "unit_respecting_temporal_block_permutation",
    }


def _bootstrap_indices(
    *,
    sample_pairs: list[tuple[int, int]],
    unit_count: int,
    month_count: int,
    max_lag: int,
    unit_fraction: float,
    time_fraction: float,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not sample_pairs:
        return np.zeros((0,), dtype=np.int64)
    selected_units = set(
        int(value)
        for value in rng.choice(
            np.arange(unit_count, dtype=np.int64),
            size=max(1, int(np.ceil(unit_count * max(min(unit_fraction, 1.0), 0.05)))),
            replace=True,
        ).tolist()
    )
    available_months = max(1, month_count - max_lag)
    target_months = max(1, int(np.ceil(available_months * max(min(time_fraction, 1.0), 0.10))))
    selected_months: set[int] = set()
    while len(selected_months) < target_months:
        start = int(rng.integers(max_lag, max(month_count, max_lag + 1)))
        for month_idx in range(start, min(month_count, start + max(1, int(block_length)))):
            if month_idx >= max_lag:
                selected_months.add(month_idx)
                if len(selected_months) >= target_months:
                    break
    indices = [
        idx
        for idx, (unit_idx, month_idx) in enumerate(sample_pairs)
        if unit_idx in selected_units and month_idx in selected_months
    ]
    if not indices:
        return np.arange(len(sample_pairs), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _stability_from_bootstrap(
    *,
    design_matrix: np.ndarray,
    response_matrix: np.ndarray,
    sample_weights: np.ndarray,
    sample_pairs: list[tuple[int, int]],
    sparse_matrix: np.ndarray,
    cfg: dict[str, Any],
    unit_count: int,
    month_count: int,
    block_count: int,
    max_lag: int,
) -> tuple[np.ndarray, float]:
    draws = max(0, int(cfg.get("bootstrap_draws", 0)))
    if draws <= 0 or design_matrix.size == 0:
        return np.ones_like(sparse_matrix, dtype=np.float32), 0.0
    edge_threshold = float(cfg.get("edge_threshold", 0.05))
    stability = np.zeros_like(sparse_matrix, dtype=np.float32)
    hidden_ranks: list[float] = []
    rng = np.random.default_rng(int(cfg.get("rng_seed", 0)))
    accepted_draws = 0
    for _ in range(draws):
        indices = _bootstrap_indices(
            sample_pairs=sample_pairs,
            unit_count=unit_count,
            month_count=month_count,
            max_lag=max_lag,
            unit_fraction=float(cfg.get("bootstrap_unit_fraction", 0.35)),
            time_fraction=float(cfg.get("bootstrap_time_fraction", 0.50)),
            block_length=int(cfg.get("bootstrap_block_length", 6)),
            rng=rng,
        )
        subset_design = design_matrix[indices]
        subset_response = response_matrix[indices]
        subset_weights = sample_weights[indices]
        if subset_design.shape[0] < max(8, block_count * max_lag):
            continue
        accepted_draws += 1
        draw_sparse, draw_low_rank, _ = _fit_sparse_plus_low_rank_matrix(
            response_matrix=subset_response,
            design_matrix=subset_design,
            sample_weights=subset_weights,
            ridge_penalty=float(cfg.get("ridge_penalty", 0.1)),
            sparse_penalty=float(cfg.get("sparse_penalty", 0.05)),
            low_rank_penalty=float(cfg.get("low_rank_penalty", 0.05)),
            decomposition_steps=max(24, int(cfg.get("decomposition_steps", 80)) // 2),
            convergence_tol=float(cfg.get("convergence_tol", 1e-5)),
            block_count=block_count,
            max_lag=max_lag,
        )
        same_sign = np.sign(draw_sparse) == np.sign(sparse_matrix)
        active = np.abs(draw_sparse) >= edge_threshold
        stability += (same_sign & active).astype(np.float32)
        if draw_low_rank.size:
            singular_values = np.linalg.svd(draw_low_rank, compute_uv=False, full_matrices=False)
            hidden_ranks.append(float(np.sum(singular_values >= float(cfg.get("rank_threshold", 0.04)))))
    if accepted_draws > 0:
        stability /= float(accepted_draws)
    return stability.astype(np.float32), float(np.mean(hidden_ranks)) if hidden_ranks else 0.0


def _blanket_summary(
    *,
    target_block_ids: list[str],
    edges: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    block_axis: list[str],
    indicator_names_by_block: dict[str, list[str]],
) -> dict[str, Any]:
    blanket: set[str] = set(target_block_ids)
    for row in edges + hidden_rows:
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if source in blanket or target in blanket:
            blanket.add(source)
            blanket.add(target)
    index = {block_id: idx for idx, block_id in enumerate(block_axis)}
    canonical_names: set[str] = set()
    for block_id in blanket:
        for canonical_name in indicator_names_by_block.get(block_id, []):
            canonical_names.add(str(canonical_name))
    blanket_ids = [block_id for block_id in block_axis if block_id in blanket]
    return {
        "target_block_ids": [block_id for block_id in block_axis if block_id in set(target_block_ids)],
        "blanket_block_ids": blanket_ids,
        "blanket_indices": [int(index[block_id]) for block_id in blanket_ids if block_id in index],
        "phase3_member_canonical_names": sorted(canonical_names),
    }


def estimate_latent_temporal_scale_graph(
    *,
    scale_name: str,
    state_tensor: np.ndarray,
    block_axis: list[str],
    target_block_ids: list[str],
    indicator_names_by_block: dict[str, list[str]],
    cfg: dict[str, Any],
    phi_by_block: dict[str, float],
    uncertainty_tensor: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor = np.asarray(state_tensor, dtype=np.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    if tensor.ndim != 3:
        bundle = {
            "scale_name": scale_name,
            "status": "unavailable",
            "reason": "invalid_state_tensor",
            "block_count": len(block_axis),
            "max_lag": int(cfg.get("max_lag", 1)),
            "edge_count": 0,
            "hidden_driver_count": 0,
            "edges": [],
            "hidden_driver_rows": [],
            "innovation_variance_rows": [],
            "feature_rows": _feature_rows(block_axis, int(cfg.get("max_lag", 1))),
            "effective_sample_count": 0,
            "selected_hyperparameters": {},
        }
        return bundle, {"target_block_ids": [], "blanket_block_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
    selected_cfg = _select_temporal_hyperparameters(
        state_tensor=tensor,
        block_axis=block_axis,
        phi_by_block=phi_by_block,
        cfg=cfg,
    )

    max_lag = int(selected_cfg.get("max_lag", cfg.get("max_lag", 1)))
    sample_arrays = _build_sample_arrays(
        state_tensor=tensor,
        block_axis=block_axis,
        phi_by_block=phi_by_block,
        max_lag=max_lag,
        uncertainty_tensor=uncertainty_tensor,
    )
    effective_samples = int(sample_arrays["effective_sample_count"])
    if effective_samples < int(cfg.get("min_effective_samples", 16)):
        bundle = {
            "scale_name": scale_name,
            "status": "unavailable",
            "reason": "insufficient_samples",
            "block_count": len(block_axis),
            "max_lag": max_lag,
            "edge_count": 0,
            "hidden_driver_count": 0,
            "edges": [],
            "hidden_driver_rows": [],
            "innovation_variance_rows": _innovation_variance_rows(tensor, block_axis, phi_by_block),
            "feature_rows": _feature_rows(block_axis, max_lag),
            "effective_sample_count": effective_samples,
            "uncertainty_available": uncertainty_tensor is not None,
            "sample_weight_summary": {"min": 1.0, "median": 1.0, "max": 1.0},
        }
        return bundle, {"target_block_ids": [], "blanket_block_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}

    standardized_design, _, _ = _weighted_standardize(sample_arrays["design_matrix"], sample_arrays["sample_weights"])
    standardized_response, _, _ = _weighted_standardize(sample_arrays["response_matrix"], sample_arrays["sample_weights"])
    sparse_matrix, low_rank_matrix, objective = _fit_sparse_plus_low_rank_matrix(
        response_matrix=standardized_response,
        design_matrix=standardized_design,
        sample_weights=sample_arrays["sample_weights"],
        ridge_penalty=float(selected_cfg.get("ridge_penalty", cfg.get("ridge_penalty", 0.1))),
        sparse_penalty=float(selected_cfg.get("sparse_penalty", cfg.get("sparse_penalty", 0.05))),
        low_rank_penalty=float(selected_cfg.get("low_rank_penalty", cfg.get("low_rank_penalty", 0.05))),
        decomposition_steps=int(cfg.get("decomposition_steps", 120)),
        convergence_tol=float(cfg.get("convergence_tol", 1e-5)),
        block_count=len(block_axis),
        max_lag=max_lag,
    )
    null_thresholds = _permutation_null_thresholds(
        design_matrix=standardized_design,
        response_matrix=standardized_response,
        sample_weights=sample_arrays["sample_weights"],
        sample_pairs=list(sample_arrays["sample_pairs"]),
        block_count=len(block_axis),
        max_lag=max_lag,
        ridge_penalty=float(selected_cfg.get("ridge_penalty", cfg.get("ridge_penalty", 0.1))),
        sparse_penalty=float(selected_cfg.get("sparse_penalty", cfg.get("sparse_penalty", 0.05))),
        low_rank_penalty=float(selected_cfg.get("low_rank_penalty", cfg.get("low_rank_penalty", 0.05))),
        cfg=cfg,
    )
    stability_matrix, hidden_rank_mean = _stability_from_bootstrap(
        design_matrix=standardized_design,
        response_matrix=standardized_response,
        sample_weights=sample_arrays["sample_weights"],
        sample_pairs=sample_arrays["sample_pairs"],
        sparse_matrix=sparse_matrix,
        cfg={**cfg, **selected_cfg, "edge_threshold": float(null_thresholds["direct_edge_threshold"])},
        unit_count=int(tensor.shape[0]),
        month_count=int(tensor.shape[1]),
        block_count=len(block_axis),
        max_lag=max_lag,
    )
    sparse_rows = _matrix_rows(
        sparse_matrix,
        block_axis=block_axis,
        max_lag=max_lag,
        threshold=float(null_thresholds["direct_edge_threshold"]),
        stability_matrix=stability_matrix,
    )
    direct_edges = list(sparse_rows)
    hidden_rows = _matrix_rows(
        low_rank_matrix,
        block_axis=block_axis,
        max_lag=max_lag,
        threshold=float(null_thresholds["hidden_edge_threshold"]),
        stability_matrix=None,
    )
    singular_values = np.linalg.svd(low_rank_matrix, compute_uv=False, full_matrices=False) if low_rank_matrix.size else np.zeros((0,), dtype=np.float32)
    estimated_hidden_rank = int(np.sum(singular_values >= float(null_thresholds["rank_threshold"])))
    if estimated_hidden_rank == 0 and (hidden_rows or float(np.max(np.abs(low_rank_matrix))) > 1e-6):
        estimated_hidden_rank = 1
    hidden_driver_fallback_rows: list[dict[str, Any]] = []
    if not hidden_rows and estimated_hidden_rank > 0:
        hidden_driver_fallback_rows = _matrix_rows(
            low_rank_matrix,
            block_axis=block_axis,
            max_lag=max_lag,
            threshold=0.0,
            stability_matrix=None,
        )
        hidden_driver_fallback_rows = hidden_driver_fallback_rows[:1]
    direct_blanket = _blanket_summary(
        target_block_ids=target_block_ids,
        edges=direct_edges,
        hidden_rows=[],
        block_axis=block_axis,
        indicator_names_by_block=indicator_names_by_block,
    )
    hidden_blanket = _blanket_summary(
        target_block_ids=target_block_ids,
        edges=[],
        hidden_rows=hidden_rows,
        block_axis=block_axis,
        indicator_names_by_block=indicator_names_by_block,
    )
    blanket = {
        **direct_blanket,
        "direct_target_block_ids": list(direct_blanket["target_block_ids"]),
        "direct_blanket_block_ids": list(direct_blanket["blanket_block_ids"]),
        "direct_blanket_indices": list(direct_blanket["blanket_indices"]),
        "direct_phase3_member_canonical_names": list(direct_blanket["phase3_member_canonical_names"]),
        "hidden_blanket_block_ids": list(hidden_blanket["blanket_block_ids"]),
        "hidden_blanket_indices": list(hidden_blanket["blanket_indices"]),
        "hidden_phase3_member_canonical_names": list(hidden_blanket["phase3_member_canonical_names"]),
        "hidden_blanket_source": "threshold_surviving_hidden_driver_rows_only",
    }
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=np.float32)
    bundle = {
        "scale_name": scale_name,
        "status": "completed",
        "block_count": len(block_axis),
        "max_lag": max_lag,
        "edge_count": len(direct_edges),
        "hidden_driver_count": len(hidden_rows),
        "edges": direct_edges,
        "hidden_driver_rows": hidden_rows,
        "hidden_driver_fallback_rows": hidden_driver_fallback_rows,
        "hidden_driver_fallback_used": bool(hidden_driver_fallback_rows and not hidden_rows),
        "innovation_variance_rows": _innovation_variance_rows(tensor, block_axis, phi_by_block),
        "feature_rows": _feature_rows(block_axis, max_lag),
        "effective_sample_count": effective_samples,
        "decomposition_objective": objective,
        "combined_adjacency": np.round((sparse_matrix + low_rank_matrix).astype(np.float32), 6).tolist(),
        "low_rank_adjacency": np.round(low_rank_matrix.astype(np.float32), 6).tolist(),
        "estimated_hidden_rank": estimated_hidden_rank,
        "bootstrap_draws": int(cfg.get("bootstrap_draws", 0)),
        "bootstrap_hidden_rank_mean": round(hidden_rank_mean, 6),
        "edge_threshold": float(null_thresholds["direct_edge_threshold"]),
        "hidden_edge_threshold": float(null_thresholds["hidden_edge_threshold"]),
        "rank_threshold": float(null_thresholds["rank_threshold"]),
        "selected_hyperparameters": {
            "max_lag": int(selected_cfg["max_lag"]),
            "ridge_penalty": round(float(selected_cfg["ridge_penalty"]), 6),
            "sparse_penalty": round(float(selected_cfg["sparse_penalty"]), 6),
            "low_rank_penalty": round(float(selected_cfg["low_rank_penalty"]), 6),
            "selection_score": round(float(selected_cfg["score"]), 6),
        },
        "selection_method": "blocked_forecast_fit_plus_unit_respecting_temporal_block_null",
        "null_thresholds": {
            "direct_edge_threshold": round(float(null_thresholds["direct_edge_threshold"]), 6),
            "hidden_edge_threshold": round(float(null_thresholds["hidden_edge_threshold"]), 6),
            "rank_threshold": round(float(null_thresholds["rank_threshold"]), 6),
            "null_quantile": round(float(null_thresholds["null_quantile"]), 6),
            "permutation_count": int(null_thresholds["permutation_count"]),
            "null_block_length": int(null_thresholds["null_block_length"]),
            "null_mode": str(null_thresholds["null_mode"]),
        },
        "uncertainty_available": uncertainty_tensor is not None,
        "sample_weight_summary": {
            "min": round(float(sample_weights.min()) if sample_weights.size else 1.0, 6),
            "median": round(float(np.median(sample_weights)) if sample_weights.size else 1.0, 6),
            "max": round(float(sample_weights.max()) if sample_weights.size else 1.0, 6),
        },
    }
    return bundle, blanket


def _rows_to_tensor(
    *,
    rows: list[dict[str, Any]],
    unit_axis: list[str],
    month_axis: list[str],
    block_axis: list[str],
    unit_key: str,
    value_key: str,
) -> np.ndarray:
    unit_index = {name: idx for idx, name in enumerate(unit_axis)}
    block_index = {name: idx for idx, name in enumerate(block_axis)}
    tensor = np.full((len(unit_axis), len(month_axis), len(block_axis)), np.nan, dtype=np.float32)
    for row in rows:
        unit_name = str(row.get(unit_key) or "")
        block_id = str(row.get("block_id") or "")
        if unit_name not in unit_index or block_id not in block_index:
            continue
        values = np.asarray(row.get(value_key) or [], dtype=np.float32)
        if values.size != len(month_axis):
            continue
        tensor[unit_index[unit_name], :, block_index[block_id]] = values
    observed = tensor[np.isfinite(tensor)]
    fill_value = float(np.median(observed)) if observed.size else 0.25
    return np.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)


def _indicator_names_by_block(phase15_dir: Path, block_axis: list[str]) -> dict[str, list[str]]:
    parameter_payload = read_json(phase15_dir / "phase15_v2_indicator_parameters.json", default={})
    rows = list(parameter_payload.get("rows") or []) if isinstance(parameter_payload, dict) else []
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        block_id = str(row.get("block_id") or "")
        canonical_name = str(row.get("canonical_name") or "")
        if block_id and canonical_name and canonical_name not in mapping[block_id]:
            mapping[block_id].append(canonical_name)
    if all(mapping.get(block_id) for block_id in block_axis):
        return {block_id: mapping.get(block_id, []) for block_id in block_axis}
    latent_blocks = list((((_HIV_PLUGIN.constraint_settings or {}).get("phase15", {}) or {}).get("latent_blocks", {}) or {}).get("blocks", []))
    fallback: dict[str, list[str]] = {}
    for row in latent_blocks:
        block_id = str(row.get("block_id") or "")
        indicators = dict(row.get("indicators") or {})
        fallback[block_id] = sorted(str(name) for name in indicators)
    return {block_id: mapping.get(block_id) or fallback.get(block_id, []) for block_id in block_axis}


def _phase15_phi_by_block(phase15_dir: Path) -> tuple[list[str], dict[str, float]]:
    fit_summary = read_json(phase15_dir / "phase15_v2_fit_summary.json", default={})
    rows = list(fit_summary.get("rows") or []) if isinstance(fit_summary, dict) else []
    block_axis: list[str] = []
    phi_by_block: dict[str, float] = {}
    for row in rows:
        block_id = str(row.get("block_id") or "")
        if not block_id or block_id in phi_by_block:
            continue
        block_axis.append(block_id)
        phi_by_block[block_id] = float(row.get("phi") or 0.0)
    return block_axis, phi_by_block


def _load_uncertainty_tensors(phase15_dir: Path, block_axis: list[str]) -> dict[str, np.ndarray | None]:
    payload = read_json(phase15_dir / "phase15_v2_uncertainty.json", default={})
    if not payload:
        return {"province": None, "region": None, "national": None}
    province_axis = [str(value) for value in list(payload.get("province_axis") or [])]
    month_axis = [str(value) for value in list(payload.get("month_axis") or [])]
    rows = list(payload.get("rows") or [])
    province_tensor = _rows_to_tensor(
        rows=rows,
        unit_axis=province_axis,
        month_axis=month_axis,
        block_axis=block_axis,
        unit_key="province",
        value_key="posterior_std_values",
    )
    region_map: dict[str, str] = {}
    for row in rows:
        province = str(row.get("province") or "")
        region = str(row.get("region") or "")
        if province and region and province not in region_map:
            region_map[province] = region
    region_axis = sorted({region for region in region_map.values() if region})
    if region_axis:
        region_values = np.zeros((len(region_axis), len(month_axis), len(block_axis)), dtype=np.float32)
        region_counts = np.zeros((len(region_axis), 1, 1), dtype=np.float32)
        region_index = {name: idx for idx, name in enumerate(region_axis)}
        for province_idx, province_name in enumerate(province_axis):
            region_name = region_map.get(province_name)
            if not region_name or region_name not in region_index:
                continue
            idx = region_index[region_name]
            region_values[idx] += province_tensor[province_idx]
            region_counts[idx] += 1.0
        region_tensor = region_values / np.clip(region_counts, 1.0, None)
    else:
        region_tensor = None
    national_tensor = province_tensor.mean(axis=0, keepdims=True) if province_tensor.size else None
    return {"province": province_tensor, "region": region_tensor, "national": national_tensor}


def _validation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    validation = dict(cfg)
    validation["bootstrap_draws"] = min(12, int(cfg.get("bootstrap_draws", 12)))
    validation["decomposition_steps"] = min(120, int(cfg.get("decomposition_steps", 120)))
    validation["min_effective_samples"] = min(24, int(cfg.get("min_effective_samples", 24)))
    return validation


def _build_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = _validation_cfg(cfg)
    block_axis = ["testing_engagement", "care_access_continuity", "suppression_capacity"]
    indicator_names = {
        "testing_engagement": ["testing_rate"],
        "care_access_continuity": ["linkage_to_care"],
        "suppression_capacity": ["viral_suppression_rate"],
    }
    rng = np.random.default_rng(int(cfg.get("rng_seed", 31)))

    recovery_tensor = np.zeros((6, 48, len(block_axis)), dtype=np.float32)
    for unit_idx in range(recovery_tensor.shape[0]):
        hidden = 0.0
        for month_idx in range(2, recovery_tensor.shape[1]):
            prev = recovery_tensor[unit_idx, month_idx - 1]
            prev2 = recovery_tensor[unit_idx, month_idx - 2]
            noise = rng.normal(0.0, 0.07, size=len(block_axis)).astype(np.float32)
            hidden = 0.55 * hidden + float(rng.normal(0.0, 0.08))
            recovery_tensor[unit_idx, month_idx, 0] = 0.45 * prev[0] + 0.25 * hidden + noise[0]
            recovery_tensor[unit_idx, month_idx, 1] = 0.35 * prev[1] + 0.80 * prev2[0] + 0.25 * hidden + noise[1]
            recovery_tensor[unit_idx, month_idx, 2] = 0.30 * prev[2] + 0.75 * prev[1] + 0.25 * hidden + noise[2]
    recovery_bundle, _ = estimate_latent_temporal_scale_graph(
        scale_name="validation_recovery",
        state_tensor=recovery_tensor,
        block_axis=block_axis,
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block=indicator_names,
        cfg=validation_cfg,
        phi_by_block={"testing_engagement": 0.45, "care_access_continuity": 0.35, "suppression_capacity": 0.30},
    )
    recovery_edges = {(row["source"], row["target"], int(row["lag"])) for row in list(recovery_bundle.get("edges") or [])}

    falsification_tensor = np.zeros((6, 48, len(block_axis)), dtype=np.float32)
    for unit_idx in range(falsification_tensor.shape[0]):
        hidden = 0.0
        for month_idx in range(1, falsification_tensor.shape[1]):
            prev = falsification_tensor[unit_idx, month_idx - 1]
            noise = rng.normal(0.0, 0.07, size=len(block_axis)).astype(np.float32)
            hidden = 0.60 * hidden + float(rng.normal(0.0, 0.09))
            falsification_tensor[unit_idx, month_idx, 0] = 0.45 * prev[0] + 0.35 * hidden + noise[0]
            falsification_tensor[unit_idx, month_idx, 1] = 0.40 * prev[1] + 0.30 * hidden + noise[1]
            falsification_tensor[unit_idx, month_idx, 2] = 0.35 * prev[2] + 0.25 * hidden + noise[2]
    falsification_bundle, _ = estimate_latent_temporal_scale_graph(
        scale_name="validation_falsification",
        state_tensor=falsification_tensor,
        block_axis=block_axis,
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block=indicator_names,
        cfg=validation_cfg,
        phi_by_block={"testing_engagement": 0.45, "care_access_continuity": 0.40, "suppression_capacity": 0.35},
    )
    cases = [
        {
            "case_id": "synthetic_multi_lag_recovery",
            "passed": ("testing_engagement", "care_access_continuity", 2) in recovery_edges
            and ("care_access_continuity", "suppression_capacity", 1) in recovery_edges,
            "edge_count": int(recovery_bundle.get("edge_count", 0)),
        },
        {
            "case_id": "synthetic_null_falsification",
            "passed": int(falsification_bundle.get("edge_count", 0)) <= 1,
            "edge_count": int(falsification_bundle.get("edge_count", 0)),
        },
    ]
    return {
        "available": True,
        "method": "latent_temporal_graph_synthetic_validation",
        "cases": cases,
        "summary": {"case_count": len(cases), "passed_case_count": sum(1 for row in cases if row["passed"])},
    }


def build_latent_temporal_graph_outputs(*, phase15_dir: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    enabled = bool(cfg.get("enabled", True))
    bundle = {
        "enabled": enabled,
        "state_source": "phase15_v2_latent_states",
        "block_axis": [],
        "max_lag": int(cfg.get("max_lag", 1)),
        "scales": {},
    }
    blanket_bundle = {
        "enabled": enabled,
        "merged_target_block_ids": [],
        "merged_blanket_block_ids": [],
        "phase3_member_canonical_names": [],
        "direct_target_block_ids": [],
        "direct_blanket_block_ids": [],
        "direct_phase3_member_canonical_names": [],
        "hidden_blanket_block_ids": [],
        "hidden_phase3_member_canonical_names": [],
        "block_support_rows": [],
        "edge_support_rows": [],
        "hidden_driver_support_rows": [],
        "scales": {},
    }
    validation = {"available": False, "reason": "disabled"}
    if not enabled:
        return bundle, blanket_bundle, validation

    block_axis, phi_by_block = _phase15_phi_by_block(phase15_dir)
    if not block_axis:
        for scale_name in ("province", "region", "national"):
            bundle["scales"][scale_name] = {"scale_name": scale_name, "status": "unavailable", "reason": "missing_phase15_v2_fit_summary"}
            blanket_bundle["scales"][scale_name] = {"target_block_ids": [], "blanket_block_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
        return bundle, blanket_bundle, {"available": False, "reason": "missing_phase15_v2_fit_summary"}

    bundle["block_axis"] = list(block_axis)
    indicator_names_by_block = _indicator_names_by_block(phase15_dir, block_axis)
    uncertainty_tensors = _load_uncertainty_tensors(phase15_dir, block_axis)
    target_block_ids = [block_id for block_id in list(cfg.get("phase3_target_block_ids") or []) if block_id in set(block_axis)] or list(block_axis)
    state_paths = {
        "province": phase15_dir / "phase15_v2_province_state_tensor.npz",
        "region": phase15_dir / "phase15_v2_region_state_tensor.npz",
        "national": phase15_dir / "phase15_v2_national_state_tensor.npz",
    }
    block_support = Counter()
    edge_support = Counter()
    hidden_support = Counter()
    canonical_names: set[str] = set()
    hidden_canonical_names: set[str] = set()
    merged_blanket_blocks: set[str] = set()
    hidden_blanket_blocks: set[str] = set()
    merged_target_blocks: set[str] = set()

    for scale_name, path in state_paths.items():
        if not path.exists():
            scale_bundle = {"scale_name": scale_name, "status": "unavailable", "reason": "missing_phase15_v2_state_tensor"}
            scale_blanket = {"target_block_ids": [], "blanket_block_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
        else:
            scale_bundle, scale_blanket = estimate_latent_temporal_scale_graph(
                scale_name=scale_name,
                state_tensor=load_tensor_artifact(path),
                uncertainty_tensor=uncertainty_tensors.get(scale_name),
                block_axis=block_axis,
                target_block_ids=target_block_ids,
                indicator_names_by_block=indicator_names_by_block,
                cfg=cfg,
                phi_by_block=phi_by_block,
            )
            if scale_bundle.get("status") == "completed":
                merged_target_blocks.update(scale_blanket.get("direct_target_block_ids") or scale_blanket["target_block_ids"])
                merged_blanket_blocks.update(scale_blanket.get("direct_blanket_block_ids") or scale_blanket["blanket_block_ids"])
                hidden_blanket_blocks.update(scale_blanket.get("hidden_blanket_block_ids") or [])
                canonical_names.update(scale_blanket.get("direct_phase3_member_canonical_names") or scale_blanket["phase3_member_canonical_names"])
                hidden_canonical_names.update(scale_blanket.get("hidden_phase3_member_canonical_names") or [])
                for block_id in scale_blanket.get("direct_blanket_block_ids") or scale_blanket["blanket_block_ids"]:
                    block_support[block_id] += 1
                for row in list(scale_bundle.get("edges") or []):
                    edge_support[(str(row["source"]), str(row["target"]), int(row["lag"]))] += 1
                for row in list(scale_bundle.get("hidden_driver_rows") or []):
                    hidden_support[(str(row["source"]), str(row["target"]), int(row["lag"]))] += 1
        bundle["scales"][scale_name] = scale_bundle
        blanket_bundle["scales"][scale_name] = scale_blanket

    blanket_bundle["merged_target_block_ids"] = [block_id for block_id in block_axis if block_id in merged_target_blocks]
    blanket_bundle["merged_blanket_block_ids"] = [block_id for block_id in block_axis if block_id in merged_blanket_blocks]
    blanket_bundle["phase3_member_canonical_names"] = sorted(canonical_names)
    blanket_bundle["direct_target_block_ids"] = list(blanket_bundle["merged_target_block_ids"])
    blanket_bundle["direct_blanket_block_ids"] = list(blanket_bundle["merged_blanket_block_ids"])
    blanket_bundle["direct_phase3_member_canonical_names"] = list(blanket_bundle["phase3_member_canonical_names"])
    blanket_bundle["hidden_blanket_block_ids"] = [block_id for block_id in block_axis if block_id in hidden_blanket_blocks]
    blanket_bundle["hidden_phase3_member_canonical_names"] = sorted(hidden_canonical_names)
    blanket_bundle["block_support_rows"] = [{"block_id": block_id, "support_count": int(block_support[block_id])} for block_id in sorted(block_support)]
    blanket_bundle["edge_support_rows"] = [{"source": source, "target": target, "lag": lag, "support_count": int(count)} for (source, target, lag), count in sorted(edge_support.items())]
    blanket_bundle["hidden_driver_support_rows"] = [{"source": source, "target": target, "lag": lag, "support_count": int(count)} for (source, target, lag), count in sorted(hidden_support.items())]
    validation = _build_validation(cfg)
    return bundle, blanket_bundle, validation


__all__ = ["estimate_latent_temporal_scale_graph", "build_latent_temporal_graph_outputs"]
