from __future__ import annotations

from typing import Any

import numpy as np


def soft_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return (np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)).astype(np.float32)


def svd_threshold(array: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return values
    u, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    shrunk = np.maximum(singular_values - float(threshold), 0.0)
    if not np.any(shrunk > 0.0):
        return np.zeros_like(values, dtype=np.float32)
    return ((u * shrunk) @ vt).astype(np.float32)


def enforce_no_self_edges(matrix: np.ndarray, *, block_count: int, max_lag: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    for lag_idx in range(max_lag):
        for block_idx in range(block_count):
            values[lag_idx * block_count + block_idx, block_idx] = 0.0
    return values


def sparse_low_rank_complexity(sparse_matrix: np.ndarray, low_rank_matrix: np.ndarray, *, threshold: float = 1e-6) -> dict[str, int]:
    sparse_values = np.asarray(sparse_matrix, dtype=np.float32)
    low_rank_values = np.asarray(low_rank_matrix, dtype=np.float32)
    nnz_sparse = int(np.sum(np.abs(sparse_values) > float(threshold)))
    if low_rank_values.size == 0:
        hidden_rank = 0
    else:
        singular_values = np.linalg.svd(low_rank_values, compute_uv=False, full_matrices=False)
        hidden_rank = int(np.sum(singular_values > float(threshold)))
    p = int(low_rank_values.shape[0]) if low_rank_values.ndim == 2 else 0
    q = int(low_rank_values.shape[1]) if low_rank_values.ndim == 2 else 0
    low_rank_dof = int(hidden_rank * max(p + q - hidden_rank, 0))
    return {
        "nnz_sparse": nnz_sparse,
        "hidden_rank": hidden_rank,
        "low_rank_degrees_of_freedom": low_rank_dof,
        "total_degrees_of_freedom": nnz_sparse + low_rank_dof,
    }


def weighted_loss_and_gradient(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sample_weights: np.ndarray,
    sparse_matrix: np.ndarray,
    low_rank_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    design = np.asarray(design_matrix, dtype=np.float32)
    response = np.asarray(response_matrix, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
    sample_count = max(1, int(design.shape[0]))
    coefficient = np.asarray(sparse_matrix, dtype=np.float32) + np.asarray(low_rank_matrix, dtype=np.float32)
    residual = design @ coefficient - response
    weighted_residual = residual * weights
    loss = 0.5 * float(np.sum(residual * weighted_residual)) / float(sample_count)
    gradient = (design.T @ weighted_residual) / float(sample_count)
    return loss, gradient.astype(np.float32)


def sparse_plus_low_rank_objective(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sample_weights: np.ndarray,
    sparse_matrix: np.ndarray,
    low_rank_matrix: np.ndarray,
    lambda_sparse: float,
    lambda_low_rank: float,
) -> dict[str, float]:
    loss, _ = weighted_loss_and_gradient(
        response_matrix=response_matrix,
        design_matrix=design_matrix,
        sample_weights=sample_weights,
        sparse_matrix=sparse_matrix,
        low_rank_matrix=low_rank_matrix,
    )
    sparse_norm = float(np.sum(np.abs(np.asarray(sparse_matrix, dtype=np.float32))))
    if np.asarray(low_rank_matrix).size:
        nuclear_norm = float(
            np.sum(np.linalg.svd(np.asarray(low_rank_matrix, dtype=np.float32), compute_uv=False, full_matrices=False))
        )
    else:
        nuclear_norm = 0.0
    total = loss + float(lambda_sparse) * sparse_norm + float(lambda_low_rank) * nuclear_norm
    return {
        "weighted_loss": round(loss, 6),
        "sparse_norm": round(sparse_norm, 6),
        "nuclear_norm": round(nuclear_norm, 6),
        "total": round(total, 6),
    }


def estimate_initial_step_size(design_matrix: np.ndarray, sample_weights: np.ndarray) -> float:
    design = np.asarray(design_matrix, dtype=np.float32)
    if design.size == 0:
        return 1.0
    sqrt_weights = np.sqrt(np.asarray(sample_weights, dtype=np.float32)).reshape(-1, 1)
    weighted_design = design * sqrt_weights
    try:
        top_singular = float(np.linalg.svd(weighted_design, compute_uv=False, full_matrices=False)[0])
    except Exception:
        top_singular = float(np.linalg.norm(weighted_design, ord=2))
    sample_count = max(1.0, float(design.shape[0]))
    lipschitz = max((top_singular * top_singular) / sample_count, 1e-6)
    return 1.0 / lipschitz


def regularization_maxima(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sample_weights: np.ndarray,
    block_count: int,
    max_lag: int,
) -> dict[str, float]:
    zero = np.zeros((design_matrix.shape[1], response_matrix.shape[1]), dtype=np.float32)
    _, gradient = weighted_loss_and_gradient(
        response_matrix=response_matrix,
        design_matrix=design_matrix,
        sample_weights=sample_weights,
        sparse_matrix=zero,
        low_rank_matrix=zero,
    )
    gradient = enforce_no_self_edges(gradient, block_count=block_count, max_lag=max_lag)
    lambda_sparse_max = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    if gradient.size:
        lambda_low_rank_max = float(np.linalg.svd(gradient, compute_uv=False, full_matrices=False)[0])
    else:
        lambda_low_rank_max = 0.0
    return {
        "lambda_sparse_max": max(lambda_sparse_max, 1e-6),
        "lambda_low_rank_max": max(lambda_low_rank_max, 1e-6),
    }


def fit_weighted_sparse_plus_low_rank(
    *,
    response_matrix: np.ndarray,
    design_matrix: np.ndarray,
    sample_weights: np.ndarray,
    lambda_sparse: float,
    lambda_low_rank: float,
    max_iterations: int,
    convergence_tol: float,
    block_count: int,
    max_lag: int,
    initial_step_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    feature_count = int(design_matrix.shape[1]) if np.asarray(design_matrix).ndim == 2 else 0
    target_count = int(response_matrix.shape[1]) if np.asarray(response_matrix).ndim == 2 else 0
    if feature_count == 0 or target_count == 0:
        zeros = np.zeros((feature_count, target_count), dtype=np.float32)
        objective = sparse_plus_low_rank_objective(
            response_matrix=response_matrix,
            design_matrix=design_matrix,
            sample_weights=sample_weights,
            sparse_matrix=zeros,
            low_rank_matrix=zeros,
            lambda_sparse=lambda_sparse,
            lambda_low_rank=lambda_low_rank,
        )
        return zeros, zeros, {
            "converged": True,
            "iterations": 0,
            "step_size": 1.0,
            "objective_trace_summary": {"initial": objective["total"], "best": objective["total"], "final": objective["total"], "count": 1},
            "parameter_delta": 0.0,
            "objective_improvement": 0.0,
            "monotone_backtracking": True,
            "complexity": sparse_low_rank_complexity(zeros, zeros),
            "objective": objective,
        }

    design = np.asarray(design_matrix, dtype=np.float32)
    response = np.asarray(response_matrix, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    base_step_size = float(initial_step_size or estimate_initial_step_size(design, weights))
    step_size = float(base_step_size)
    sparse_prev = np.zeros((feature_count, target_count), dtype=np.float32)
    low_rank_prev = np.zeros((feature_count, target_count), dtype=np.float32)
    sparse_current = np.zeros((feature_count, target_count), dtype=np.float32)
    low_rank_current = np.zeros((feature_count, target_count), dtype=np.float32)
    momentum_prev = 1.0
    current_objective = sparse_plus_low_rank_objective(
        response_matrix=response,
        design_matrix=design,
        sample_weights=weights,
        sparse_matrix=sparse_current,
        low_rank_matrix=low_rank_current,
        lambda_sparse=lambda_sparse,
        lambda_low_rank=lambda_low_rank,
    )
    objective_trace = [float(current_objective["total"])]
    converged = False
    monotone_backtracking = True
    restart_count = 0
    parameter_delta = float("inf")
    objective_improvement = float("inf")
    iterations_run = 0

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_prev * momentum_prev))
        extrapolation = (momentum_prev - 1.0) / max(momentum, 1e-6)
        sparse_extrapolated = sparse_current + extrapolation * (sparse_current - sparse_prev)
        low_rank_extrapolated = low_rank_current + extrapolation * (low_rank_current - low_rank_prev)
        accepted = False
        trial_step = float(step_size)
        base_sparse = sparse_extrapolated
        base_low_rank = low_rank_extrapolated

        for attempt in range(2):
            for _ in range(24):
                _, gradient = weighted_loss_and_gradient(
                    response_matrix=response,
                    design_matrix=design,
                    sample_weights=weights,
                    sparse_matrix=base_sparse,
                    low_rank_matrix=base_low_rank,
                )
                sparse_candidate = enforce_no_self_edges(
                    soft_threshold(base_sparse - trial_step * gradient, trial_step * float(lambda_sparse)),
                    block_count=block_count,
                    max_lag=max_lag,
                )
                low_rank_candidate = enforce_no_self_edges(
                    svd_threshold(base_low_rank - trial_step * gradient, trial_step * float(lambda_low_rank)),
                    block_count=block_count,
                    max_lag=max_lag,
                )
                candidate_objective = sparse_plus_low_rank_objective(
                    response_matrix=response,
                    design_matrix=design,
                    sample_weights=weights,
                    sparse_matrix=sparse_candidate,
                    low_rank_matrix=low_rank_candidate,
                    lambda_sparse=lambda_sparse,
                    lambda_low_rank=lambda_low_rank,
                )
                if float(candidate_objective["total"]) <= float(current_objective["total"]) + 1e-10:
                    accepted = True
                    break
                trial_step *= 0.5
            if accepted:
                break
            if attempt == 0:
                restart_count += 1
                monotone_backtracking = False
                base_sparse = sparse_current
                base_low_rank = low_rank_current
                trial_step = max(float(step_size), float(base_step_size) * 0.25)
                momentum = 1.0
            else:
                break

        if not accepted:
            _, gradient = weighted_loss_and_gradient(
                response_matrix=response,
                design_matrix=design,
                sample_weights=weights,
                sparse_matrix=sparse_current,
                low_rank_matrix=low_rank_current,
            )
            safe_step = max(float(base_step_size) * 1e-3, 1e-6)
            sparse_candidate = enforce_no_self_edges(
                soft_threshold(sparse_current - safe_step * gradient, safe_step * float(lambda_sparse)),
                block_count=block_count,
                max_lag=max_lag,
            )
            low_rank_candidate = enforce_no_self_edges(
                svd_threshold(low_rank_current - safe_step * gradient, safe_step * float(lambda_low_rank)),
                block_count=block_count,
                max_lag=max_lag,
            )
            candidate_objective = sparse_plus_low_rank_objective(
                response_matrix=response,
                design_matrix=design,
                sample_weights=weights,
                sparse_matrix=sparse_candidate,
                low_rank_matrix=low_rank_candidate,
                lambda_sparse=lambda_sparse,
                lambda_low_rank=lambda_low_rank,
            )
            trial_step = safe_step

        sparse_prev = sparse_current
        low_rank_prev = low_rank_current
        sparse_current = np.asarray(sparse_candidate, dtype=np.float32)
        low_rank_current = np.asarray(low_rank_candidate, dtype=np.float32)
        iterations_run = iteration
        step_size = max(min(float(trial_step) * 1.05, float(base_step_size)), float(base_step_size) * 1e-3)
        objective_trace.append(float(candidate_objective["total"]))

        delta_sparse = float(np.linalg.norm(sparse_current - sparse_prev))
        delta_low_rank = float(np.linalg.norm(low_rank_current - low_rank_prev))
        denom = 1.0 + float(np.linalg.norm(sparse_prev)) + float(np.linalg.norm(low_rank_prev))
        parameter_delta = (delta_sparse + delta_low_rank) / denom
        objective_improvement = max(float(current_objective["total"]) - float(candidate_objective["total"]), 0.0) / (
            1.0 + abs(float(current_objective["total"]))
        )
        current_objective = candidate_objective
        momentum_prev = momentum

        if parameter_delta <= float(convergence_tol) and objective_improvement <= float(convergence_tol):
            converged = True
            break

    complexity = sparse_low_rank_complexity(sparse_current, low_rank_current)
    return sparse_current, low_rank_current, {
        "converged": converged,
        "iterations": iterations_run,
        "step_size": round(float(step_size), 8),
        "objective_trace_summary": {
            "initial": round(float(objective_trace[0]), 6),
            "best": round(float(min(objective_trace)), 6),
            "final": round(float(objective_trace[-1]), 6),
            "count": len(objective_trace),
        },
        "parameter_delta": round(float(parameter_delta if np.isfinite(parameter_delta) else 0.0), 8),
        "objective_improvement": round(float(objective_improvement if np.isfinite(objective_improvement) else 0.0), 8),
        "monotone_backtracking": monotone_backtracking,
        "restart_count": int(restart_count),
        "complexity": complexity,
        "objective": current_objective,
    }
