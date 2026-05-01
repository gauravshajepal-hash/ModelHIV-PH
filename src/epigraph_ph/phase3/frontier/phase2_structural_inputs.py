from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import load_tensor_artifact, read_json

from .artifacts import TransitionResearchContext


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    return f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}"


def _quarterize_tensor(tensor: np.ndarray, month_axis: list[str]) -> tuple[list[str], np.ndarray]:
    values = np.asarray(tensor, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
    grouped: dict[str, list[int]] = {}
    for month_idx, month_label in enumerate(month_axis):
        grouped.setdefault(_quarter_from_month(str(month_label)), []).append(month_idx)
    quarter_axis = sorted(grouped)
    quarter_values = np.zeros((values.shape[0], len(quarter_axis), values.shape[2]), dtype=np.float32)
    for quarter_idx, quarter in enumerate(quarter_axis):
        month_indices = grouped[quarter]
        quarter_values[:, quarter_idx, :] = np.asarray(values[:, month_indices, :], dtype=np.float32).mean(axis=1)
    return quarter_axis, quarter_values


@dataclass(slots=True)
class Phase2StructuralInputs:
    payload: dict[str, Any]
    month_axis: list[str]
    quarter_axis: list[str]
    block_axis: list[str]
    national_state_tensor: np.ndarray
    national_quarter_tensor: np.ndarray
    hidden_mode_tensor: np.ndarray
    hidden_mode_quarter_tensor: np.ndarray
    direct_edge_rows: list[dict[str, Any]]
    direct_edge_scale_rows: list[dict[str, Any]]
    direct_edge_summary_rows: list[dict[str, Any]]
    hidden_driver_rows: list[dict[str, Any]]
    hidden_driver_scale_rows: list[dict[str, Any]]
    hidden_driver_summary_rows: list[dict[str, Any]]
    multiscale_support_rows: list[dict[str, Any]]
    hidden_mode_summary: dict[str, Any]
    optimizer_diagnostics: dict[str, Any]


def load_phase2_structural_inputs(ctx: TransitionResearchContext) -> Phase2StructuralInputs:
    payload_path = ctx.phase2_dir / "phase2_structural_payload.json"
    payload = read_json(payload_path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Frozen Phase 2 structural payload is required but missing: {payload_path}")
    artifact_paths = dict(payload.get("artifact_paths") or {})
    national_state_path = Path(str(artifact_paths.get("phase15_national_state_tensor") or ctx.phase15_dir / "phase15_v2_national_state_tensor.npz"))
    if not national_state_path.exists():
        raise FileNotFoundError(f"Phase 15 national state tensor required by structural frontier is missing: {national_state_path}")
    national_state_tensor = np.asarray(load_tensor_artifact(national_state_path), dtype=np.float32)
    month_axis = [str(value) for value in list(payload.get("month_axis") or [])]
    quarter_axis, national_quarter_tensor = _quarterize_tensor(national_state_tensor, month_axis)
    hidden_mode_path_text = str(artifact_paths.get("hidden_mode_score_tensor") or "").strip()
    hidden_mode_path = Path(hidden_mode_path_text) if hidden_mode_path_text else None
    if hidden_mode_path is not None and hidden_mode_path.exists():
        hidden_mode_tensor = np.asarray(load_tensor_artifact(hidden_mode_path), dtype=np.float32)
        _hidden_quarter_axis, hidden_mode_quarter_tensor = _quarterize_tensor(hidden_mode_tensor, month_axis)
    else:
        hidden_mode_tensor = np.zeros((1, len(month_axis), 0), dtype=np.float32)
        hidden_mode_quarter_tensor = np.zeros((1, len(quarter_axis), 0), dtype=np.float32)
    direct_edge_scale_rows = [dict(row) for row in list(payload.get("direct_temporal_edge_scale_rows") or [])]
    direct_edge_summary_rows = [dict(row) for row in list(payload.get("direct_temporal_edge_rows") or [])]
    hidden_driver_scale_rows = [dict(row) for row in list(payload.get("hidden_driver_scale_rows") or [])]
    hidden_driver_summary_rows = [dict(row) for row in list(payload.get("hidden_driver_rows") or [])]
    national_direct_rows = [dict(row) for row in direct_edge_scale_rows if str(row.get("scale") or "") == "national"]
    national_hidden_rows = [dict(row) for row in hidden_driver_scale_rows if str(row.get("scale") or "") == "national"]
    return Phase2StructuralInputs(
        payload=payload,
        month_axis=month_axis,
        quarter_axis=quarter_axis,
        block_axis=[str(value) for value in list(payload.get("block_axis") or [])],
        national_state_tensor=national_state_tensor,
        national_quarter_tensor=national_quarter_tensor,
        hidden_mode_tensor=hidden_mode_tensor,
        hidden_mode_quarter_tensor=hidden_mode_quarter_tensor,
        direct_edge_rows=national_direct_rows or direct_edge_summary_rows,
        direct_edge_scale_rows=direct_edge_scale_rows,
        direct_edge_summary_rows=direct_edge_summary_rows,
        hidden_driver_rows=national_hidden_rows or hidden_driver_summary_rows,
        hidden_driver_scale_rows=hidden_driver_scale_rows,
        hidden_driver_summary_rows=hidden_driver_summary_rows,
        multiscale_support_rows=[dict(row) for row in list(payload.get("multiscale_support_rows") or [])],
        hidden_mode_summary=dict(payload.get("hidden_mode_summary") or {}),
        optimizer_diagnostics=dict(payload.get("optimizer_diagnostics") or {}),
    )


__all__ = ["Phase2StructuralInputs", "load_phase2_structural_inputs"]
