from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase2.latent_temporal_graph import (
    _build_sample_arrays,
    _load_uncertainty_tensors,
    _phase15_phi_by_block,
    _weighted_standardize,
)
from epigraph_ph.runtime import load_tensor_artifact, read_json, save_tensor_artifact, write_json


def _edge_summary_rows(scale_bundles: dict[str, Any], support_rows: list[dict[str, Any]], row_key: str) -> list[dict[str, Any]]:
    support_lookup = {
        (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)): int(row.get("support_count") or 0)
        for row in support_rows
    }
    aggregated: dict[tuple[str, str, int], dict[str, Any]] = {}
    for scale_name, scale_bundle in scale_bundles.items():
        if str(scale_bundle.get("status") or "") != "completed":
            continue
        for row in list(scale_bundle.get(row_key) or []):
            key = (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0))
            item = aggregated.setdefault(
                key,
                {
                    "source": key[0],
                    "target": key[1],
                    "lag": key[2],
                    "weights": [],
                    "stabilities": [],
                    "scales": [],
                    "scale_rows": [],
                },
            )
            item["weights"].append(float(row.get("weight") or 0.0))
            if row.get("stability") is not None:
                item["stabilities"].append(float(row.get("stability") or 0.0))
            item["scales"].append(str(scale_name))
            item["scale_rows"].append(
                {
                    "scale": str(scale_name),
                    "weight": round(float(row.get("weight") or 0.0), 6),
                    "stability": round(float(row.get("stability") or 0.0), 6) if row.get("stability") is not None else None,
                }
            )
    rows: list[dict[str, Any]] = []
    for key, payload in sorted(aggregated.items()):
        rows.append(
            {
                "source": key[0],
                "target": key[1],
                "lag": key[2],
                "weight": round(float(np.mean(payload["weights"])) if payload["weights"] else 0.0, 6),
                "stability": round(float(np.mean(payload["stabilities"])) if payload["stabilities"] else 0.0, 6),
                "support_count": int(support_lookup.get(key, 0)),
                "scales": list(payload["scales"]),
                "scale_rows": sorted(list(payload["scale_rows"]), key=lambda row: str(row.get("scale") or "")),
            }
        )
    return rows


def _edge_scale_rows(scale_bundles: dict[str, Any], row_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale_name, scale_bundle in sorted(scale_bundles.items()):
        if str(scale_bundle.get("status") or "") != "completed":
            continue
        for row in list(scale_bundle.get(row_key) or []):
            rows.append(
                {
                    "scale": str(scale_name),
                    "source": str(row.get("source") or ""),
                    "target": str(row.get("target") or ""),
                    "lag": int(row.get("lag") or 0),
                    "weight": round(float(row.get("weight") or 0.0), 6),
                    "stability": round(float(row.get("stability") or 0.0), 6) if row.get("stability") is not None else None,
                }
            )
    return rows


def _optimizer_diagnostics(scale_bundle: dict[str, Any]) -> dict[str, Any]:
    selected = dict(scale_bundle.get("selected_hyperparameters") or {})
    decomposition = dict(scale_bundle.get("decomposition_objective") or {})
    return {
        "status": str(scale_bundle.get("status") or "unknown"),
        "selected_hyperparameters": selected,
        "decomposition_objective": decomposition.get("objective", decomposition),
        "optimizer_diagnostics": {
            "converged": bool(decomposition.get("converged", False)),
            "iterations": int(decomposition.get("iterations", 0)),
            "objective_trace_summary": dict(decomposition.get("objective_trace_summary") or {}),
            "parameter_delta": float(decomposition.get("parameter_delta") or 0.0),
            "objective_improvement": float(decomposition.get("objective_improvement") or 0.0),
            "complexity": dict(decomposition.get("complexity") or {}),
        },
        "uncertainty_available": bool(scale_bundle.get("uncertainty_available", False)),
    }


def _hidden_mode_artifact(
    *,
    phase2_dir: Path,
    phase15_dir: Path,
    latent_bundle: dict[str, Any],
    month_axis: list[str],
) -> tuple[dict[str, Any], str | None]:
    national_bundle = dict((latent_bundle.get("scales") or {}).get("national") or {})
    if str(national_bundle.get("status") or "") != "completed":
        return {"available": False, "reason": "national_latent_scale_unavailable"}, None
    if bool(national_bundle.get("hidden_driver_fallback_used")) or not list(national_bundle.get("hidden_driver_rows") or []):
        return {"available": False, "reason": "no_threshold_surviving_hidden_rows"}, None
    low_rank_adjacency = np.asarray(national_bundle.get("low_rank_adjacency") or [], dtype=np.float32)
    estimated_rank = int(national_bundle.get("estimated_hidden_rank") or 0)
    if low_rank_adjacency.ndim != 2 or low_rank_adjacency.size == 0 or estimated_rank <= 0:
        return {"available": False, "reason": "missing_low_rank_structure"}, None
    block_axis = [str(value) for value in list(latent_bundle.get("block_axis") or [])]
    loaded_block_axis, loaded_phi_by_block = _phase15_phi_by_block(phase15_dir)
    if not block_axis:
        block_axis = list(loaded_block_axis)
    phi_by_block = {block_id: float(loaded_phi_by_block.get(block_id) or 0.0) for block_id in block_axis}
    national_state_tensor = np.asarray(load_tensor_artifact(phase15_dir / "phase15_v2_national_state_tensor.npz"), dtype=np.float32)
    resolved_month_axis = list(month_axis or latent_bundle.get("month_axis") or [])
    if not resolved_month_axis:
        uncertainty_payload = {
            "month_axis": [],
            "available": False,
            "reason": "missing_month_axis",
        }
        return uncertainty_payload, None
    uncertainty_tensors = _load_uncertainty_tensors(phase15_dir, block_axis)
    national_uncertainty = uncertainty_tensors.get("national")
    max_lag = int(national_bundle.get("max_lag") or 1)
    sample_arrays = _build_sample_arrays(
        state_tensor=national_state_tensor,
        block_axis=block_axis,
        phi_by_block=phi_by_block,
        max_lag=max_lag,
        uncertainty_tensor=national_uncertainty,
        month_axis=resolved_month_axis,
    )
    if int(sample_arrays.get("effective_sample_count") or 0) <= 0:
        return {"available": False, "reason": "empty_hidden_sample_arrays"}, None
    standardized_design, _, _ = _weighted_standardize(
        np.asarray(sample_arrays["design_matrix"], dtype=np.float32),
        np.asarray(sample_arrays["sample_weights"], dtype=np.float32),
    )
    if standardized_design.size == 0 or standardized_design.ndim != 2:
        return {"available": False, "reason": "empty_design_matrix"}, None
    low_rank_contribution = standardized_design @ low_rank_adjacency
    if low_rank_contribution.ndim != 2 or low_rank_contribution.size == 0:
        return {"available": False, "reason": "empty_low_rank_contribution"}, None
    sample_u, singular_values, _ = np.linalg.svd(low_rank_contribution, full_matrices=False)
    rank = min(estimated_rank, int(np.sum(singular_values > 1e-6)), int(sample_u.shape[1]))
    if rank <= 0:
        return {"available": False, "reason": "zero_hidden_rank"}, None
    hidden_scores = sample_u[:, :rank] * singular_values[:rank]
    score_tensor = np.zeros((1, len(resolved_month_axis), rank), dtype=np.float32)
    month_counts = np.zeros((len(resolved_month_axis),), dtype=np.float32)
    for row_idx, (_unit_idx, month_idx) in enumerate(list(sample_arrays["sample_pairs"])):
        if month_idx < score_tensor.shape[1]:
            score_tensor[0, month_idx, :] += hidden_scores[row_idx, :]
            month_counts[month_idx] += 1.0
    nonzero = month_counts > 0
    if np.any(nonzero):
        score_tensor[0, nonzero, :] = score_tensor[0, nonzero, :] / month_counts[nonzero, None]
    artifact = save_tensor_artifact(
        array=score_tensor,
        axis_names=["national", "month", "hidden_mode"],
        artifact_dir=phase2_dir,
        stem="phase2_national_hidden_mode_scores",
        backend="numpy",
        device="cpu",
        notes=["phase2_structural_hidden_mode_scores"],
        save_pt=False,
    )
    ar1_rows = []
    for mode_idx in range(rank):
        series = score_tensor[0, :, mode_idx]
        if series.shape[0] <= 1 or float(np.dot(series[:-1], series[:-1])) <= 1e-6:
            phi = 0.0
        else:
            phi = float(np.dot(series[:-1], series[1:]) / max(float(np.dot(series[:-1], series[:-1])), 1e-6))
        ar1_rows.append({"hidden_mode": int(mode_idx), "phi": round(phi, 6)})
    sample_weights = np.asarray(sample_arrays["sample_weights"], dtype=np.float32)
    return {
        "available": True,
        "rank": rank,
        "month_axis": resolved_month_axis,
        "ar1_rows": ar1_rows,
        "basis_kind": "weighted_low_rank_contribution_basis",
        "uncertainty_aware": national_uncertainty is not None,
        "identified_hidden_shock_process": False,
        "sample_weight_summary": {
            "min": round(float(sample_weights.min()) if sample_weights.size else 1.0, 6),
            "median": round(float(np.median(sample_weights)) if sample_weights.size else 1.0, 6),
            "max": round(float(sample_weights.max()) if sample_weights.size else 1.0, 6),
        },
        "tensor_path": artifact["value_path"],
    }, artifact["value_path"]


def _multiscale_support_rows(*, phase15_dir: Path, multiscale_blankets: dict[str, Any]) -> list[dict[str, Any]]:
    factor_catalog = list(read_json(phase15_dir / "multiscale_factor_catalog.json", default=[]))
    factor_lookup = {
        str(row.get("factor_id") or ""): dict(row)
        for row in factor_catalog
        if str(row.get("factor_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for support_row in list(multiscale_blankets.get("factor_support_rows") or []):
        factor_id = str(support_row.get("factor_id") or "")
        catalog_row = dict(factor_lookup.get(factor_id) or {})
        rows.append(
            {
                "factor_id": factor_id,
                "support_count": int(support_row.get("support_count") or 0),
                "block_name": str(catalog_row.get("block_name") or ""),
                "factor_name": str(catalog_row.get("factor_name") or ""),
                "best_target": str(catalog_row.get("best_target") or ""),
                "transition_hooks": [str(value) for value in list(catalog_row.get("transition_hooks") or [])],
                "member_canonical_names": [str(value) for value in list(catalog_row.get("member_canonical_names") or [])],
            }
        )
    return rows


def build_phase2_structural_artifacts(
    *,
    phase2_dir: Path,
    phase15_dir: Path,
    latent_bundle: dict[str, Any],
    latent_blankets: dict[str, Any],
    multiscale_bundle: dict[str, Any],
    multiscale_blankets: dict[str, Any],
) -> dict[str, str]:
    month_axis = list(read_json(phase15_dir / "phase15_v2_uncertainty.json", default={}).get("month_axis") or [])
    latent_scales = dict(latent_bundle.get("scales") or {})
    direct_rows = _edge_summary_rows(
        latent_scales,
        list(latent_blankets.get("edge_support_rows") or []),
        "edges",
    )
    hidden_rows = _edge_summary_rows(
        latent_scales,
        list(latent_blankets.get("hidden_driver_support_rows") or []),
        "hidden_driver_rows",
    )
    direct_scale_rows = _edge_scale_rows(latent_scales, "edges")
    hidden_scale_rows = _edge_scale_rows(latent_scales, "hidden_driver_rows")
    direct_summary_keys = {(str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) for row in direct_rows}
    direct_scale_keys = {(str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) for row in direct_scale_rows}
    hidden_rows = [
        row
        for row in hidden_rows
        if (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) not in direct_summary_keys
    ]
    hidden_scale_rows = [
        row
        for row in hidden_scale_rows
        if (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) not in direct_scale_keys
    ]
    multiscale_rows = _multiscale_support_rows(phase15_dir=phase15_dir, multiscale_blankets=multiscale_blankets)
    hidden_mode_summary, hidden_mode_tensor_path = _hidden_mode_artifact(
        phase2_dir=phase2_dir,
        phase15_dir=phase15_dir,
        latent_bundle=latent_bundle,
        month_axis=month_axis,
    )
    payload = {
        "schema_version": "phase2_structural_payload_v1",
        "scientific_role": "structural_frontier_payload",
        "state_source": str(latent_bundle.get("state_source") or "phase15_v2_latent_states"),
        "block_axis": list(latent_bundle.get("block_axis") or []),
        "month_axis": month_axis,
        "direct_temporal_surface": {
            "target_block_ids": list(latent_blankets.get("direct_target_block_ids") or []),
            "blanket_block_ids": list(latent_blankets.get("direct_blanket_block_ids") or []),
            "phase3_member_canonical_names": list(latent_blankets.get("direct_phase3_member_canonical_names") or []),
        },
        "hidden_driver_surface": {
            "blanket_block_ids": list(latent_blankets.get("hidden_blanket_block_ids") or []),
            "phase3_member_canonical_names": list(latent_blankets.get("hidden_phase3_member_canonical_names") or []),
        },
        "multiscale_support_surface": {
            "blanket_factor_ids": list(multiscale_blankets.get("merged_blanket_factor_ids") or []),
            "target_factor_ids": list(multiscale_blankets.get("merged_target_factor_ids") or []),
            "phase3_member_canonical_names": list(multiscale_blankets.get("phase3_member_canonical_names") or []),
        },
        "direct_temporal_edge_rows": direct_rows,
        "direct_temporal_edge_scale_rows": direct_scale_rows,
        "hidden_driver_rows": hidden_rows,
        "hidden_driver_scale_rows": hidden_scale_rows,
        "multiscale_support_rows": multiscale_rows,
        "direct_support_rows": list(latent_blankets.get("edge_support_rows") or []),
        "hidden_support_rows": list(latent_blankets.get("hidden_driver_support_rows") or []),
        "multiscale_factor_support_rows": list(multiscale_blankets.get("factor_support_rows") or []),
        "optimizer_diagnostics": {
            "latent": {
                scale_name: _optimizer_diagnostics(dict(scale_bundle))
                for scale_name, scale_bundle in dict(latent_bundle.get("scales") or {}).items()
            },
            "multiscale": {
                scale_name: _optimizer_diagnostics(dict(scale_bundle))
                for scale_name, scale_bundle in dict(multiscale_bundle.get("scales") or {}).items()
            },
        },
        "hidden_mode_summary": hidden_mode_summary,
        "artifact_paths": {
            "phase15_national_state_tensor": str(phase15_dir / "phase15_v2_national_state_tensor.npz"),
            "phase15_uncertainty": str(phase15_dir / "phase15_v2_uncertainty.json"),
            "hidden_mode_score_tensor": hidden_mode_tensor_path,
        },
    }
    payload_path = phase2_dir / "phase2_structural_payload.json"
    write_json(payload_path, payload)
    return {"phase2_structural_payload": str(payload_path), "phase2_hidden_mode_scores": str(hidden_mode_tensor_path or "")}
