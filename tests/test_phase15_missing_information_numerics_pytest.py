from __future__ import annotations

import numpy as np

from epigraph_ph.phase15.missing_information_numerics import (
    apply_missing_information_precision_operator_cpu,
    solve_missing_information_linear_system,
)


def _build_temporal_only_system() -> dict[str, object]:
    province_count = 6
    month_count = 20
    cell_count = province_count * month_count
    diagonal_precision = np.full((cell_count,), 12.0, dtype=np.float64)
    forcing = np.linspace(0.1, 1.0, cell_count, dtype=np.float64)
    return {
        "province_count": province_count,
        "month_count": month_count,
        "cell_count": cell_count,
        "temporal_precision": 4.0,
        "phi": 0.92,
        "donor_precision": 0.0,
        "donor_laplacian": np.zeros((province_count, province_count), dtype=np.float64),
        "base_diagonal": diagonal_precision.copy(),
        "diagonal_precision": diagonal_precision.copy(),
        "forcing": forcing,
        "constraint_terms": [],
        "backend_runtime": {"backend": "cpu_sparse"},
    }


def _build_cfg(preconditioner: str) -> dict[str, object]:
    return {
        "missing_information_preconditioner": preconditioner,
        "missing_information_preconditioner_cholesky_jitter": 1e-8,
        "missing_information_preconditioner_cholesky_max_attempts": 3,
        "missing_information_cpu_cg_max_iter": 12,
        "missing_information_retry_on_nonconvergence": False,
        "missing_information_torch_cg_rtol": 1e-10,
        "missing_information_torch_cg_atol": 1e-12,
        "variance_eps": 1e-10,
    }


def test_temporal_block_preconditioner_solves_cpu_system_and_reports_backend() -> None:
    system = _build_temporal_only_system()
    preconditioned_result = solve_missing_information_linear_system(
        system=system,
        cfg=_build_cfg("temporal_block"),
        backend="cpu_sparse",
    )

    preconditioned_solver = dict(preconditioned_result["solver_diagnostics"])

    assert preconditioned_solver["preconditioner"] == "temporal_block_cholesky"
    assert preconditioned_solver["backend"] == "cpu_sparse"
    assert preconditioned_solver["stage_count"] == 1
    assert bool(preconditioned_solver["cg_converged"]) is True

    residual = np.asarray(system["forcing"], dtype=np.float64) - apply_missing_information_precision_operator_cpu(
        system,
        np.asarray(preconditioned_result["correction_vector"], dtype=np.float64),
    )
    assert float(np.linalg.norm(residual)) <= float(preconditioned_solver["cg_tolerance"]) * 1.05


def test_cpu_precision_operator_rebuilds_stale_cached_constraint_matrix() -> None:
    full_system = {
        "province_count": 3,
        "month_count": 2,
        "cell_count": 6,
        "temporal_precision": 0.0,
        "phi": 0.0,
        "donor_precision": 0.0,
        "donor_laplacian": np.zeros((3, 3), dtype=np.float64),
        "base_diagonal": np.ones((6,), dtype=np.float64),
        "diagonal_precision": np.ones((6,), dtype=np.float64),
        "forcing": np.ones((6,), dtype=np.float64),
        "constraint_terms": [(np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32), 1.0, 2.0, 0.0)],
    }
    _ = apply_missing_information_precision_operator_cpu(full_system, np.ones((6,), dtype=np.float64))
    assert full_system["constraint_matrix_cpu"].shape == (1, 6)

    restricted_system = {
        **full_system,
        "province_count": 2,
        "cell_count": 4,
        "donor_laplacian": np.zeros((2, 2), dtype=np.float64),
        "base_diagonal": np.ones((4,), dtype=np.float64),
        "diagonal_precision": np.ones((4,), dtype=np.float64),
        "forcing": np.ones((4,), dtype=np.float64),
        "constraint_terms": [(np.asarray([0, 1, 2, 3], dtype=np.int32), 1.0, 2.0, 0.0)],
        "constraint_matrix_cpu": full_system["constraint_matrix_cpu"],
        "constraint_precision_vector": full_system["constraint_precision_vector"],
    }

    result = apply_missing_information_precision_operator_cpu(restricted_system, np.ones((4,), dtype=np.float64))
    assert result.shape == (4,)
    assert restricted_system["constraint_matrix_cpu"].shape == (1, 4)
    assert np.asarray(restricted_system["constraint_precision_vector"], dtype=np.float64).shape == (1,)
