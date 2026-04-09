from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg

try:
    import torch
except Exception:  # pragma: no cover - optional runtime acceleration
    torch = None


def missing_information_backend_runtime(cfg: Mapping[str, Any]) -> dict[str, Any]:
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


def apply_missing_information_precision_operator_cpu(
    system: Mapping[str, Any],
    vector: np.ndarray,
) -> np.ndarray:
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    temporal_precision = float(system["temporal_precision"])
    phi = float(system["phi"])
    donor_precision = float(system["donor_precision"])
    donor_laplacian = np.asarray(system["donor_laplacian"], dtype=np.float64)
    x = np.asarray(vector, dtype=np.float64).reshape(province_count, month_count)
    result = np.asarray(system["base_diagonal"], dtype=np.float64).reshape(province_count, month_count) * x
    if month_count > 1 and abs(temporal_precision) > 0.0:
        temporal_residual = x[:, 1:] - phi * x[:, :-1]
        result[:, 1:] += temporal_precision * temporal_residual
        result[:, :-1] -= temporal_precision * phi * temporal_residual
    if donor_laplacian.size and donor_precision > 0.0:
        result += donor_precision * (donor_laplacian @ x)
    constraint_matrix = system.get("constraint_matrix_cpu")
    constraint_precision_vector = system.get("constraint_precision_vector")
    expected_rows = len(list(system.get("constraint_terms") or []))
    expected_cols = int(system["cell_count"])
    needs_rebuild = (
        constraint_matrix is None
        or getattr(constraint_matrix, "shape", (0, 0))[0] != expected_rows
        or getattr(constraint_matrix, "shape", (0, 0))[1] != expected_cols
        or constraint_precision_vector is None
        or int(np.asarray(constraint_precision_vector, dtype=np.float64).size) != expected_rows
    )
    if needs_rebuild and list(system.get("constraint_terms") or []):
        built_matrix, built_precision = _build_constraint_sparse_matrix(
            constraint_terms=list(system.get("constraint_terms") or []),
            cell_count=int(system["cell_count"]),
        )
        if isinstance(system, dict):
            system["constraint_matrix_cpu"] = built_matrix
            system["constraint_precision_vector"] = np.asarray(built_precision, dtype=np.float64)
        constraint_matrix = built_matrix
        constraint_precision_vector = np.asarray(built_precision, dtype=np.float64)
    elif constraint_precision_vector is not None:
        constraint_precision_vector = np.asarray(constraint_precision_vector, dtype=np.float64)
    flat = x.reshape(-1)
    if getattr(constraint_matrix, "shape", (0, 0))[0] > 0:
        projection = constraint_matrix @ flat
        weighted_projection = np.asarray(constraint_precision_vector, dtype=np.float64) * np.asarray(projection, dtype=np.float64)
        return np.asarray(result.reshape(-1) + constraint_matrix.transpose() @ weighted_projection, dtype=np.float64)
    if list(system.get("constraint_terms") or []):
        flat_result = result.reshape(-1)
        for cells, coefficient, constraint_precision, _latent_discrepancy in list(system["constraint_terms"]):
            projection = coefficient * float(np.sum(flat[cells]))
            flat_result[cells] += constraint_precision * coefficient * projection
        return np.asarray(flat_result, dtype=np.float64)
    return np.asarray(result, dtype=np.float64).reshape(-1)


def _torch_missing_information_dtype(cfg: Mapping[str, Any]) -> Any:
    if torch is None:
        return None
    dtype_name = str(cfg.get("missing_information_torch_dtype") or "float32").strip().lower()
    if dtype_name == "float64":
        return torch.float64
    return torch.float32


def _build_constraint_sparse_matrix(
    *,
    constraint_terms: list[tuple[np.ndarray, float, float, float]],
    cell_count: int,
) -> tuple[Any, np.ndarray]:
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    precision_values = np.zeros((len(constraint_terms),), dtype=np.float64)
    for row_idx, (cells, coefficient, constraint_precision, _latent_discrepancy) in enumerate(constraint_terms):
        cells_arr = np.asarray(cells, dtype=np.int64)
        if cells_arr.size == 0:
            continue
        row_parts.append(np.full((int(cells_arr.size),), int(row_idx), dtype=np.int64))
        col_parts.append(cells_arr)
        value_parts.append(np.full((int(cells_arr.size),), float(coefficient), dtype=np.float64))
        precision_values[row_idx] = float(constraint_precision)
    if not row_parts:
        return None, precision_values
    return (
        csr_matrix(
            (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(len(constraint_terms), cell_count),
            dtype=np.float64,
        ),
        precision_values,
    )


def _factorize_temporal_block_preconditioner_cpu(system: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    diagonal = np.asarray(system["diagonal_precision"], dtype=np.float64).reshape(province_count, month_count)
    blocks = np.zeros((province_count, month_count, month_count), dtype=np.float64)
    diag_idx = np.arange(month_count, dtype=np.int64)
    blocks[:, diag_idx, diag_idx] = diagonal
    off_value = -float(system["temporal_precision"]) * float(system["phi"])
    if month_count > 1 and abs(off_value) > 0.0:
        off_idx = np.arange(month_count - 1, dtype=np.int64)
        blocks[:, off_idx, off_idx + 1] = off_value
        blocks[:, off_idx + 1, off_idx] = off_value
    jitter_base = max(float(cfg.get("missing_information_preconditioner_cholesky_jitter") or 1e-6), float(cfg.get("variance_eps") or 1e-8))
    max_attempts = max(1, int(cfg.get("missing_information_preconditioner_cholesky_max_attempts") or 4))
    identity = np.eye(month_count, dtype=np.float64)[None, :, :]
    jitter = 0.0
    for attempt in range(max_attempts):
        try:
            return {
                "factor": np.linalg.cholesky(blocks + jitter * identity),
                "label": "temporal_block_cholesky",
                "jitter": float(jitter),
            }
        except np.linalg.LinAlgError:
            jitter = jitter_base if attempt == 0 else jitter * 10.0
    raise np.linalg.LinAlgError("Failed to factorize CPU temporal-block preconditioner")


def _apply_temporal_block_preconditioner_cpu(preconditioner: Mapping[str, Any], vector: np.ndarray) -> np.ndarray:
    factor = np.asarray(preconditioner["factor"], dtype=np.float64)
    province_count, month_count, _ = factor.shape
    rhs = np.asarray(vector, dtype=np.float64).reshape(province_count, month_count, 1)
    forward = np.linalg.solve(factor, rhs)
    solution = np.linalg.solve(np.swapaxes(factor, 1, 2), forward)
    return np.asarray(solution, dtype=np.float64).reshape(-1)


def _factorize_temporal_block_preconditioner_torch(system: Mapping[str, Any], cfg: Mapping[str, Any], *, dtype: Any, device: Any) -> dict[str, Any]:
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    diagonal = np.asarray(system["diagonal_precision"], dtype=np.float64).reshape(province_count, month_count)
    blocks = torch.zeros((province_count, month_count, month_count), dtype=dtype, device=device)
    diag_idx = torch.arange(month_count, dtype=torch.int64, device=device)
    blocks[:, diag_idx, diag_idx] = torch.as_tensor(diagonal, dtype=dtype, device=device)
    off_value = -float(system["temporal_precision"]) * float(system["phi"])
    if month_count > 1 and abs(off_value) > 0.0:
        off_idx = torch.arange(month_count - 1, dtype=torch.int64, device=device)
        blocks[:, off_idx, off_idx + 1] = float(off_value)
        blocks[:, off_idx + 1, off_idx] = float(off_value)
    jitter_base = max(float(cfg.get("missing_information_preconditioner_cholesky_jitter") or 1e-6), float(cfg.get("variance_eps") or 1e-8))
    max_attempts = max(1, int(cfg.get("missing_information_preconditioner_cholesky_max_attempts") or 4))
    identity = torch.eye(month_count, dtype=dtype, device=device).unsqueeze(0)
    jitter = 0.0
    for attempt in range(max_attempts):
        factor, info = torch.linalg.cholesky_ex(blocks + jitter * identity)
        if int(torch.max(info).item()) == 0:
            return {"factor": factor, "label": "temporal_block_cholesky", "jitter": float(jitter)}
        jitter = jitter_base if attempt == 0 else jitter * 10.0
    raise RuntimeError("Failed to factorize torch temporal-block preconditioner")


def _apply_temporal_block_preconditioner_torch(preconditioner: Mapping[str, Any], vector: Any) -> Any:
    factor = preconditioner["factor"]
    province_count, month_count, _ = factor.shape
    rhs = vector.reshape(province_count, month_count, 1)
    forward = torch.linalg.solve_triangular(factor, rhs, upper=False)
    solution = torch.linalg.solve_triangular(factor.transpose(-1, -2), forward, upper=True)
    return solution.reshape(-1)


def _run_cpu_pcg(system: Mapping[str, Any], cfg: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    rtol = float(stage["rtol"])
    atol = float(stage["atol"])
    max_iter = int(stage["max_iter"])
    operator = LinearOperator(
        shape=(int(system["cell_count"]), int(system["cell_count"])),
        matvec=lambda vec: apply_missing_information_precision_operator_cpu(system, vec),
        dtype=np.float64,
    )
    preconditioner = None
    preconditioner_label = "identity"
    preconditioner_jitter = 0.0
    if str(cfg.get("missing_information_preconditioner") or "temporal_block").strip().lower() != "identity":
        factorized = _factorize_temporal_block_preconditioner_cpu(system, cfg)
        preconditioner = LinearOperator(
            shape=(int(system["cell_count"]), int(system["cell_count"])),
            matvec=lambda vec: _apply_temporal_block_preconditioner_cpu(factorized, vec),
            dtype=np.float64,
        )
        preconditioner_label = str(factorized["label"])
        preconditioner_jitter = float(factorized["jitter"])
    iteration_counter = {"count": 0}

    def _callback(_xk: np.ndarray) -> None:
        iteration_counter["count"] += 1

    correction_vector, info = cg(
        operator,
        np.asarray(system["forcing"], dtype=np.float64),
        x0=np.zeros((int(system["cell_count"]),), dtype=np.float64),
        rtol=rtol,
        atol=atol,
        maxiter=max_iter,
        M=preconditioner,
        callback=_callback,
    )
    residual = np.asarray(system["forcing"], dtype=np.float64) - apply_missing_information_precision_operator_cpu(system, np.asarray(correction_vector, dtype=np.float64))
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(np.asarray(system["forcing"], dtype=np.float64)))
    tolerance = float(max(atol, rtol * max(rhs_norm, 1.0)))
    if info < 0:
        raise RuntimeError(f"CPU PCG failed for {system['block_id']} with info={info}")
    return {
        "correction_vector": np.asarray(correction_vector, dtype=np.float64),
        "solver_diagnostics": {
            "backend": "cpu_sparse",
            "device": "cpu",
            "compiled": False,
            "preconditioner": preconditioner_label,
            "preconditioner_jitter": round(float(preconditioner_jitter), 12),
            "cg_iteration_count": int(iteration_counter["count"]),
            "cg_residual_norm": round(float(residual_norm), 6),
            "cg_tolerance": round(float(tolerance), 12),
            "cg_converged": bool(residual_norm <= tolerance),
            "cg_max_iter_reached": bool(info > 0),
            "cg_max_iter": int(max_iter),
            "rtol": float(rtol),
            "atol": float(atol),
            "stage_name": str(stage["name"]),
            "dtype": "float64",
        },
    }


def _run_torch_pcg(system: Mapping[str, Any], cfg: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("torch CUDA backend requested but unavailable")
    device = torch.device("cuda")
    dtype_name = str(stage.get("dtype") or cfg.get("missing_information_torch_dtype") or "float32").strip().lower()
    dtype = torch.float64 if dtype_name == "float64" else _torch_missing_information_dtype(cfg)
    province_count = int(system["province_count"])
    month_count = int(system["month_count"])
    cell_count = int(system["cell_count"])
    base_diag_t = torch.as_tensor(np.asarray(system["base_diagonal"], dtype=np.float64), dtype=dtype, device=device)
    forcing_t = torch.as_tensor(np.asarray(system["forcing"], dtype=np.float64), dtype=dtype, device=device)
    donor_laplacian = np.asarray(system["donor_laplacian"], dtype=np.float64)
    donor_laplacian_t = torch.as_tensor(donor_laplacian, dtype=dtype, device=device) if donor_laplacian.size else torch.zeros((0, 0), dtype=dtype, device=device)
    constraint_terms = list(system["constraint_terms"])
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    precision_values = np.zeros((len(constraint_terms),), dtype=np.float64)
    for row_idx, (cells, coefficient, constraint_precision, _latent_discrepancy) in enumerate(constraint_terms):
        cells_arr = np.asarray(cells, dtype=np.int64)
        if cells_arr.size == 0:
            continue
        row_parts.append(np.full((int(cells_arr.size),), int(row_idx), dtype=np.int64))
        col_parts.append(cells_arr)
        value_parts.append(np.full((int(cells_arr.size),), float(coefficient), dtype=np.float64))
        precision_values[row_idx] = float(constraint_precision)
    if row_parts:
        constraint_indices_t = torch.as_tensor(np.vstack([np.concatenate(row_parts), np.concatenate(col_parts)]), dtype=torch.int64, device=device)
        constraint_values_t = torch.as_tensor(np.concatenate(value_parts), dtype=dtype, device=device)
        constraint_matrix_t = torch.sparse_coo_tensor(
            constraint_indices_t,
            constraint_values_t,
            (len(constraint_terms), cell_count),
            dtype=dtype,
            device=device,
        ).coalesce()
        constraint_precision_t = torch.as_tensor(precision_values, dtype=dtype, device=device)
    else:
        constraint_matrix_t = None
        constraint_precision_t = torch.zeros((0,), dtype=dtype, device=device)
    preconditioner = None
    preconditioner_label = "identity"
    preconditioner_jitter = 0.0
    if str(cfg.get("missing_information_preconditioner") or "temporal_block").strip().lower() != "identity":
        preconditioner = _factorize_temporal_block_preconditioner_torch(system, cfg, dtype=dtype, device=device)
        preconditioner_label = str(preconditioner["label"])
        preconditioner_jitter = float(preconditioner["jitter"])

    def _matvec_impl(vector: Any) -> Any:
        x = vector.reshape(province_count, month_count)
        result = base_diag_t.reshape(province_count, month_count) * x
        if month_count > 1 and abs(float(system["temporal_precision"])) > 0.0:
            temporal_residual = x[:, 1:] - float(system["phi"]) * x[:, :-1]
            result = result.clone()
            result[:, 1:] = result[:, 1:] + float(system["temporal_precision"]) * temporal_residual
            result[:, :-1] = result[:, :-1] - float(system["temporal_precision"]) * float(system["phi"]) * temporal_residual
        if donor_laplacian_t.numel() and float(system["donor_precision"]) > 0.0:
            result = result + float(system["donor_precision"]) * (donor_laplacian_t @ x)
        flat_result = result.reshape(-1)
        if constraint_matrix_t is not None:
            projection = torch.sparse.mm(constraint_matrix_t, vector.reshape(-1, 1))
            weighted_projection = constraint_precision_t.reshape(-1, 1) * projection
            flat_result = flat_result + torch.sparse.mm(constraint_matrix_t.transpose(0, 1), weighted_projection).reshape(-1)
        return flat_result

    compiled_matvec = _matvec_impl
    compile_used = False
    if bool(cfg.get("missing_information_torch_compile", True)) and hasattr(torch, "compile"):
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

    max_iter = int(stage["max_iter"])
    rtol = float(stage["rtol"])
    atol = float(stage["atol"])
    x = torch.zeros_like(forcing_t)
    r = forcing_t - compiled_matvec(x)
    z = _apply_temporal_block_preconditioner_torch(preconditioner, r) if preconditioner is not None else r.clone()
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(forcing_t)
    tolerance = torch.maximum(
        torch.as_tensor(atol, dtype=dtype, device=device),
        torch.as_tensor(rtol, dtype=dtype, device=device) * torch.maximum(b_norm, torch.as_tensor(1.0, dtype=dtype, device=device)),
    )
    iteration_count = 0
    residual_norm = torch.linalg.norm(r)
    for iteration_idx in range(max_iter):
        if bool((residual_norm <= tolerance).item()):
            break
        ap = compiled_matvec(p)
        denom = torch.dot(p, ap)
        alpha = rz_old / torch.clamp(denom, min=torch.as_tensor(1e-12, dtype=dtype, device=device))
        x = x + alpha * p
        r = r - alpha * ap
        residual_norm = torch.linalg.norm(r)
        iteration_count = iteration_idx + 1
        if bool((residual_norm <= tolerance).item()):
            break
        z = _apply_temporal_block_preconditioner_torch(preconditioner, r) if preconditioner is not None else r.clone()
        rz_new = torch.dot(r, z)
        beta = rz_new / torch.clamp(rz_old, min=torch.as_tensor(1e-12, dtype=dtype, device=device))
        p = z + beta * p
        rz_old = rz_new
    converged = bool((residual_norm <= tolerance).item())
    return {
        "correction_vector": x.detach().cpu().numpy().astype(np.float64),
        "solver_diagnostics": {
            "backend": "torch_cuda",
            "device": str(device),
            "compiled": bool(compile_used),
            "preconditioner": preconditioner_label,
            "preconditioner_jitter": round(float(preconditioner_jitter), 12),
            "cg_iteration_count": int(iteration_count),
            "cg_residual_norm": round(float(residual_norm.detach().cpu().item()), 6),
            "cg_tolerance": round(float(tolerance.detach().cpu().item()), 12),
            "cg_converged": bool(converged),
            "cg_max_iter_reached": bool(iteration_count >= max_iter and not converged),
            "cg_max_iter": int(max_iter),
            "rtol": float(rtol),
            "atol": float(atol),
            "stage_name": str(stage["name"]),
            "dtype": dtype_name,
        },
    }


def solve_missing_information_linear_system(
    *,
    system: Mapping[str, Any],
    cfg: Mapping[str, Any],
    backend: str | None = None,
    rtol_override: float | None = None,
    atol_override: float | None = None,
) -> dict[str, Any]:
    selected_backend = str(backend or dict(system.get("backend_runtime") or {}).get("backend") or "cpu_sparse")
    rtol = float(rtol_override) if rtol_override is not None else float(cfg.get("missing_information_torch_cg_rtol") or 1e-5)
    atol = float(atol_override) if atol_override is not None else float(cfg.get("missing_information_torch_cg_atol") or 1e-7)
    retry_enabled = bool(cfg.get("missing_information_retry_on_nonconvergence", True))
    stage_rows: list[dict[str, Any]] = []
    if selected_backend == "torch_cuda":
        stage_rows.append({"name": "torch_primary", "backend": "torch_cuda", "max_iter": int(cfg.get("missing_information_torch_cg_max_iter") or 256), "rtol": rtol, "atol": atol, "dtype": str(cfg.get("missing_information_torch_dtype") or "float32")})
        if retry_enabled:
            retry_iter = max(int(cfg.get("missing_information_torch_cg_retry_max_iter") or 1024), stage_rows[0]["max_iter"])
            if retry_iter > stage_rows[0]["max_iter"]:
                stage_rows.append({"name": "torch_retry", "backend": "torch_cuda", "max_iter": retry_iter, "rtol": rtol, "atol": atol, "dtype": str(cfg.get("missing_information_torch_dtype") or "float32")})
            if bool(cfg.get("missing_information_fallback_to_cpu", True)):
                stage_rows.append({"name": "cpu_fallback", "backend": "cpu_sparse", "max_iter": int(cfg.get("missing_information_cpu_cg_max_iter") or 2048), "rtol": rtol, "atol": atol, "dtype": "float64"})
    else:
        stage_rows.append({"name": "cpu_primary", "backend": "cpu_sparse", "max_iter": int(cfg.get("missing_information_cpu_cg_max_iter") or 1024), "rtol": rtol, "atol": atol, "dtype": "float64"})
    stage_results: list[dict[str, Any]] = []
    for stage in stage_rows:
        stage_result = _run_torch_pcg(system, cfg, stage) if stage["backend"] == "torch_cuda" else _run_cpu_pcg(system, cfg, stage)
        stage_results.append(stage_result)
        if bool(stage_result["solver_diagnostics"].get("cg_converged", False)):
            break
    best = min(
        stage_results,
        key=lambda item: float(item["solver_diagnostics"].get("cg_residual_norm") or np.inf) / max(float(item["solver_diagnostics"].get("cg_tolerance") or 1.0), 1e-12),
    )
    diagnostics = dict(best["solver_diagnostics"])
    diagnostics["stage_count"] = len(stage_results)
    diagnostics["stage_rows"] = [dict(result["solver_diagnostics"]) for result in stage_results]
    return {
        "correction_vector": np.asarray(best["correction_vector"], dtype=np.float64),
        "solver_diagnostics": diagnostics,
    }
