from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import quarter_sort_key
from .runtime import load_tensor_artifact, read_json


@dataclass(slots=True)
class Phase2StructuralInputs:
    source_run_id: str
    quarter_axis: list[str]
    block_axis: list[str]
    national_quarter_tensor: np.ndarray
    direct_edge_rows: list[dict[str, Any]]
    hidden_driver_rows: list[dict[str, Any]]
    multiscale_support_rows: list[dict[str, Any]]
    hidden_mode_quarter_tensor: np.ndarray
    payload: dict[str, Any]


@dataclass(slots=True)
class DirectPriorFeature:
    transition: str
    source: str
    target: str
    lag: int
    prior_scale: float
    phase2_weight: float
    stability: float
    support_count: int
    tensor_index: int
    edge_key: str = ""
    feature_role: str = "phase2_direct_prior"


@dataclass(slots=True)
class HiddenDriverFeature:
    transition: str
    mode_index: int
    support_scale: float
    stability: float
    support_count: int
    target_blocks: tuple[str, ...]


_PREFERRED_PHASE2_SOURCE_PREFIXES: tuple[str, ...] = (
    "phase0-2-incidence-official-augmented-",
    "tr-v3-phase2-testing-prevention-rebuild-",
    "tr-v3-monthly-phase2-lane-",
    "phase2-replay-source-",
)


def phase2_structural_payload_path(epigraph_root: Path, source_run_id: str) -> Path:
    return (
        Path(epigraph_root)
        / "artifacts"
        / "runs"
        / str(source_run_id)
        / "phase2"
        / "phase2_structural_payload.json"
    )


def resolve_phase2_structural_source_run_id(
    epigraph_root: Path,
    *,
    observation_source_run_id: str,
    preferred: str | None = None,
) -> tuple[str, dict[str, Any]]:
    epigraph_root = Path(epigraph_root)
    if preferred:
        preferred_path = phase2_structural_payload_path(epigraph_root, preferred)
        if preferred_path.exists():
            return str(preferred), {
                "resolution": "explicit_preferred",
                "phase2_payload_path": preferred_path.as_posix(),
            }
        raise FileNotFoundError(f"Preferred Phase 2 source run has no structural payload: {preferred}")

    observation_path = phase2_structural_payload_path(epigraph_root, observation_source_run_id)
    if observation_path.exists():
        return str(observation_source_run_id), {
            "resolution": "observation_source_contains_phase2_payload",
            "phase2_payload_path": observation_path.as_posix(),
        }

    runs_dir = epigraph_root / "artifacts" / "runs"
    candidates = sorted(
        [
            path.parent.parent.name
            for path in runs_dir.glob("*/phase2/phase2_structural_payload.json")
        ]
    )
    for prefix in _PREFERRED_PHASE2_SOURCE_PREFIXES:
        preferred_candidates = [name for name in candidates if str(name).startswith(prefix)]
        if preferred_candidates:
            selected = str(preferred_candidates[-1])
            selected_path = phase2_structural_payload_path(epigraph_root, selected)
            return selected, {
                "resolution": f"latest_matching_prefix:{prefix}",
                "phase2_payload_path": selected_path.as_posix(),
                "candidate_count": len(candidates),
            }
    if not candidates:
        raise FileNotFoundError("No phase2_structural_payload.json artifact was found under artifacts/runs.")
    selected = str(candidates[-1])
    selected_path = phase2_structural_payload_path(epigraph_root, selected)
    return selected, {
        "resolution": "latest_available_phase2_payload",
        "phase2_payload_path": selected_path.as_posix(),
        "candidate_count": len(candidates),
    }


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    return f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}"


def _quarterize_month_tensor(tensor: np.ndarray, month_axis: list[str]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for month_idx, month_label in enumerate(month_axis):
        grouped[_quarter_from_month(str(month_label))].append(month_idx)
    quarter_axis = sorted(grouped, key=quarter_sort_key)
    quarter_tensor = np.asarray(
        [np.mean(np.asarray(tensor[:, grouped[quarter], :], dtype=np.float64), axis=1) for quarter in quarter_axis],
        dtype=np.float32,
    )
    return quarter_axis, np.swapaxes(quarter_tensor, 0, 1)


def _resolve_run_artifact_path(
    *,
    epigraph_root: Path,
    source_run_id: str,
    raw_path: str | None,
    fallback: Path | None = None,
) -> Path | None:
    if raw_path:
        raw_text = str(raw_path).strip()
        if raw_text:
            direct_path = Path(raw_text)
            if direct_path.exists():
                return direct_path
            normalized = raw_text.replace("\\", "/")
            for marker in ("/artifacts/runs/", "artifacts/runs/"):
                marker_index = normalized.lower().find(marker.lower())
                if marker_index >= 0:
                    relative_suffix = normalized[marker_index:].lstrip("/")
                    candidate = epigraph_root / relative_suffix
                    if candidate.exists():
                        return candidate
                    break
            artifact_name = direct_path.name
            if artifact_name:
                run_root = epigraph_root / "artifacts" / "runs" / source_run_id
                for subdir in ("phase2", "phase15"):
                    candidate = run_root / subdir / artifact_name
                    if candidate.exists():
                        return candidate
    if fallback is not None and fallback.exists():
        return fallback
    return fallback


def load_phase2_structural_inputs(epigraph_root: Path, source_run_id: str = "smoke-latent-blocks") -> Phase2StructuralInputs:
    phase2_dir = epigraph_root / "artifacts" / "runs" / source_run_id / "phase2"
    phase15_dir = epigraph_root / "artifacts" / "runs" / source_run_id / "phase15"
    payload = dict(read_json(phase2_dir / "phase2_structural_payload.json", default={}) or {})
    if not payload:
        raise FileNotFoundError(f"Missing structural payload for run {source_run_id}")
    month_axis = [str(value) for value in list(payload.get("month_axis") or [])]
    block_axis = [str(value) for value in list(payload.get("block_axis") or [])]
    artifact_paths = dict(payload.get("artifact_paths") or {})
    national_state_path = _resolve_run_artifact_path(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        raw_path=str(artifact_paths.get("phase15_national_state_tensor") or ""),
        fallback=phase15_dir / "phase15_v2_national_state_tensor.npz",
    )
    if national_state_path is None:
        raise FileNotFoundError(f"Missing national state tensor for run {source_run_id}")
    national_month_tensor = np.asarray(load_tensor_artifact(national_state_path), dtype=np.float32)
    quarter_axis, national_quarter_tensor = _quarterize_month_tensor(national_month_tensor, month_axis)
    hidden_path = str(
        ((payload.get("hidden_mode_summary") or {}).get("tensor_path"))
        or artifact_paths.get("hidden_mode_score_tensor")
        or ""
    )
    resolved_hidden_path = _resolve_run_artifact_path(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        raw_path=hidden_path,
        fallback=phase2_dir / "phase2_national_hidden_mode_scores.npz",
    )
    if resolved_hidden_path is not None and resolved_hidden_path.exists():
        hidden_tensor = np.asarray(load_tensor_artifact(resolved_hidden_path), dtype=np.float32)
        _, hidden_quarter_tensor = _quarterize_month_tensor(hidden_tensor, month_axis)
    else:
        hidden_quarter_tensor = np.zeros((1, len(quarter_axis), 0), dtype=np.float32)
    return Phase2StructuralInputs(
        source_run_id=source_run_id,
        quarter_axis=quarter_axis,
        block_axis=block_axis,
        national_quarter_tensor=national_quarter_tensor,
        direct_edge_rows=list(payload.get("direct_temporal_edge_rows") or []),
        hidden_driver_rows=list(payload.get("hidden_driver_rows") or []),
        multiscale_support_rows=list(payload.get("multiscale_support_rows") or []),
        hidden_mode_quarter_tensor=hidden_quarter_tensor,
        payload=payload,
    )


def build_direct_prior_features(
    structural_inputs: Phase2StructuralInputs,
    transition_prior_map: dict[str, Any],
) -> dict[str, list[DirectPriorFeature]]:
    block_index = {block_id: idx for idx, block_id in enumerate(structural_inputs.block_axis)}
    by_key = {
        (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)): dict(row)
        for row in list(structural_inputs.direct_edge_rows or [])
    }
    features_by_transition: dict[str, list[DirectPriorFeature]] = {}
    for transition, transition_cfg in dict(transition_prior_map or {}).items():
        rows: list[DirectPriorFeature] = []
        target_blocks = dict((transition_cfg or {}).get("target_blocks") or {})
        for target_block, target_cfg in target_blocks.items():
            source_blocks = dict((target_cfg or {}).get("source_blocks") or {})
            for source_block, source_cfg in source_blocks.items():
                for lag in list((source_cfg or {}).get("lags") or []):
                    key = (str(source_block), str(target_block), int(lag))
                    matched = by_key.get(key)
                    if matched is None:
                        continue
                    if source_block not in block_index:
                        continue
                    edge_key = f"direct:{source_block}->{target_block}:lag{int(lag)}"
                    rows.append(
                        DirectPriorFeature(
                            transition=str(transition),
                            source=str(source_block),
                            target=str(target_block),
                            lag=int(lag),
                            prior_scale=float((source_cfg or {}).get("prior_scale") or 0.0),
                            phase2_weight=float(matched.get("weight") or 0.0),
                            stability=float(matched.get("stability") or 0.0),
                            support_count=int(matched.get("support_count") or 0),
                            tensor_index=int(block_index[source_block]),
                            edge_key=edge_key,
                        )
                    )
        features_by_transition[str(transition)] = rows
    return features_by_transition


def filter_direct_prior_features_by_edge_keys(
    features_by_transition: dict[str, list[DirectPriorFeature]],
    allowed_edge_keys: set[str],
    *,
    feature_role: str = "source_stable_determinant_covariate",
) -> dict[str, list[DirectPriorFeature]]:
    allowed = {str(value) for value in allowed_edge_keys}
    filtered: dict[str, list[DirectPriorFeature]] = {}
    for transition, rows in dict(features_by_transition or {}).items():
        filtered[str(transition)] = [
            DirectPriorFeature(
                transition=row.transition,
                source=row.source,
                target=row.target,
                lag=row.lag,
                prior_scale=row.prior_scale,
                phase2_weight=row.phase2_weight,
                stability=row.stability,
                support_count=row.support_count,
                tensor_index=row.tensor_index,
                edge_key=row.edge_key,
                feature_role=feature_role,
            )
            for row in list(rows or [])
            if str(row.edge_key or f"direct:{row.source}->{row.target}:lag{row.lag}") in allowed
        ]
    return filtered


def direct_prior_feature_count(features_by_transition: dict[str, list[DirectPriorFeature]] | None) -> int:
    return int(sum(len(rows) for rows in dict(features_by_transition or {}).values()))


def resolve_phase2_determinant_robustness_report(
    epigraph_root: Path,
    source_run_id: str,
    *,
    preferred: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if preferred:
        path = Path(preferred)
        if path.exists():
            return dict(read_json(path, default={}) or {}), {
                "resolution": "explicit_preferred",
                "path": path.as_posix(),
            }
        return None, {
            "resolution": "explicit_preferred_missing",
            "path": path.as_posix(),
        }
    candidates = [
        Path(epigraph_root) / "artifacts" / "runs" / str(source_run_id) / "phase2" / "determinant_robustness_broad" / "phase2_determinant_robustness_report.json",
        Path(epigraph_root) / "artifacts" / "runs" / str(source_run_id) / "phase2" / "determinant_robustness" / "phase2_determinant_robustness_report.json",
    ]
    for path in candidates:
        if path.exists():
            return dict(read_json(path, default={}) or {}), {
                "resolution": "source_run_default",
                "path": path.as_posix(),
            }
    return None, {
        "resolution": "missing_fail_closed",
        "checked_paths": [path.as_posix() for path in candidates],
    }


def allowed_direct_edge_keys_from_robustness(robustness_report: dict[str, Any] | None) -> set[str]:
    if not robustness_report:
        return set()
    explicit = {
        str(value)
        for value in list(robustness_report.get("phase3_default_allowed_edge_keys") or [])
        if str(value)
    }
    if explicit:
        return explicit
    return {
        str(row.get("edge_key") or "")
        for row in list(robustness_report.get("edge_rows") or [])
        if bool(row.get("phase3_default_allowed")) and str(row.get("edge_key") or "")
    }


def build_hidden_driver_features(
    structural_inputs: Phase2StructuralInputs,
    transition_prior_map: dict[str, Any],
    *,
    rank_cap: int | None = None,
) -> dict[str, list[HiddenDriverFeature]]:
    hidden_tensor = np.asarray(structural_inputs.hidden_mode_quarter_tensor, dtype=np.float32)
    if hidden_tensor.ndim != 3 or hidden_tensor.shape[2] == 0:
        return {str(transition): [] for transition in dict(transition_prior_map or {})}
    available_rank = int(hidden_tensor.shape[2])
    max_rank = available_rank if rank_cap is None else max(0, min(int(rank_cap), available_rank))
    if max_rank == 0:
        return {str(transition): [] for transition in dict(transition_prior_map or {})}
    features_by_transition: dict[str, list[HiddenDriverFeature]] = {}
    hidden_rows = list(structural_inputs.hidden_driver_rows or [])
    for transition, transition_cfg in dict(transition_prior_map or {}).items():
        target_blocks = tuple(str(value) for value in dict((transition_cfg or {}).get("target_blocks") or {}).keys())
        target_block_set = set(target_blocks)
        relevant_rows = [row for row in hidden_rows if str(row.get("target") or "") in target_block_set]
        if not relevant_rows:
            features_by_transition[str(transition)] = []
            continue
        total_support = sum(
            max(float(row.get("stability") or 0.0), 0.05)
            * max(int(row.get("support_count") or 0), 1)
            * abs(float(row.get("weight") or 0.0))
            for row in relevant_rows
        )
        mean_stability = float(np.mean([max(float(row.get("stability") or 0.0), 0.05) for row in relevant_rows]))
        support_count = int(sum(max(int(row.get("support_count") or 0), 1) for row in relevant_rows))
        features_by_transition[str(transition)] = [
            HiddenDriverFeature(
                transition=str(transition),
                mode_index=mode_index,
                support_scale=float(total_support),
                stability=mean_stability,
                support_count=support_count,
                target_blocks=target_blocks,
            )
            for mode_index in range(max_rank)
        ]
    return features_by_transition
