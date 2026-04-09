from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase2.latent_temporal_graph import estimate_latent_temporal_scale_graph, _estimate_phi_by_series_tensor
from epigraph_ph.runtime import load_tensor_artifact, read_json


def _factor_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("factor_id") or ""): dict(row) for row in rows if str(row.get("factor_id") or "")}


def _factor_alias_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    member_names = tuple(sorted(str(name) for name in list(row.get("member_canonical_names") or []) if str(name)))
    block_name = str(row.get("block_name") or "")
    factor_name = str(row.get("factor_name") or "")
    factor_class = str(row.get("factor_class") or "")
    interpretability = str(row.get("interpretability_label") or "")
    network_family = str(row.get("network_feature_family") or "")
    keys: list[tuple[str, str]] = []
    factor_id = str(row.get("factor_id") or "")
    if factor_id:
        keys.append(("factor_id", factor_id))
    if factor_name:
        keys.append(("factor_name", factor_name))
    if factor_name and factor_class:
        keys.append(("factor_class+factor_name", f"{factor_class}|{factor_name}"))
    if factor_name and block_name:
        keys.append(("block_name+factor_name", f"{block_name}|{factor_name}"))
    if factor_name and factor_class and block_name:
        keys.append(("factor_class+block_name+factor_name", f"{factor_class}|{block_name}|{factor_name}"))
    if factor_name and network_family:
        keys.append(("network_feature_family+factor_name", f"{network_family}|{factor_name}"))
    if interpretability and block_name:
        keys.append(("block_name+interpretability", f"{block_name}|{interpretability}"))
    if member_names:
        keys.append(("member_names", "||".join(member_names)))
        if block_name:
            keys.append(("block_name+member_names", f"{block_name}|{'||'.join(member_names)}"))
    return keys


def _catalog_alias_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    unique_keys: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for row in rows:
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        for key in _factor_alias_keys(row):
            if key in ambiguous:
                continue
            existing = unique_keys.get(key)
            if existing is None:
                unique_keys[key] = factor_id
            elif existing != factor_id:
                unique_keys.pop(key, None)
                ambiguous.add(key)
    return unique_keys


def _normalize_retained_factor_rows(
    *,
    retained_factor_rows: list[dict[str, Any]],
    factor_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    alias_lookup = _catalog_alias_lookup(list(factor_lookup.values()))
    normalized_rows: list[dict[str, Any]] = []
    direct_match_count = 0
    alias_match_count = 0
    unmatched_count = 0
    for row in retained_factor_rows:
        raw = dict(row)
        factor_id = str(raw.get("factor_id") or "")
        if factor_id in factor_lookup:
            merged = dict(factor_lookup[factor_id])
            merged.update(raw)
            normalized_rows.append(merged)
            direct_match_count += 1
            continue
        matched_id = None
        for key in _factor_alias_keys(raw):
            matched_id = alias_lookup.get(key)
            if matched_id:
                break
        if matched_id and matched_id in factor_lookup:
            merged = dict(factor_lookup[matched_id])
            raw["input_factor_id"] = factor_id
            raw["factor_id"] = matched_id
            raw["factor_id_normalized"] = True
            merged.update(raw)
            normalized_rows.append(merged)
            alias_match_count += 1
        else:
            unmatched_count += 1
    return normalized_rows, {
        "input_count": len(retained_factor_rows),
        "direct_match_count": direct_match_count,
        "alias_match_count": alias_match_count,
        "unmatched_count": unmatched_count,
        "normalized_count": len(normalized_rows),
    }


def _indicator_names_by_factor(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for row in rows:
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        mapping[factor_id] = sorted({str(name) for name in list(row.get("member_canonical_names") or []) if str(name)})
    return mapping


def _factor_rank(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row.get("predictive_gain") or 0.0),
        float(row.get("survival_score") or 0.0),
        float(row.get("stability_score") or 0.0),
        str(row.get("factor_id") or ""),
    )


def _retained_factor_ids(
    *,
    retained_factor_rows: list[dict[str, Any]],
    factor_index: dict[str, int],
    factor_lookup: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[str]:
    normalized_rows, _ = _normalize_retained_factor_rows(retained_factor_rows=retained_factor_rows, factor_lookup=factor_lookup)
    return [str(row.get("factor_id") or "") for row in normalized_rows if str(row.get("factor_id") or "") in factor_index]


def _target_factor_ids(
    factor_ids: list[str],
    factor_lookup: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[str]:
    del cfg
    return [
        factor_id
        for factor_id in factor_ids
        if list(factor_lookup.get(factor_id, {}).get("transition_hooks") or [])
        or str(factor_lookup.get(factor_id, {}).get("best_target") or "")
    ]


def _scale_tensor(path: Path, factor_ids: list[str], factor_index: dict[str, int]) -> np.ndarray:
    values = np.asarray(load_tensor_artifact(path), dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
    indices = [int(factor_index[factor_id]) for factor_id in factor_ids if factor_id in factor_index]
    if not indices:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return values[:, :, indices].astype(np.float32)


def build_multiscale_dag_outputs(
    *,
    phase15_dir: Path,
    retained_factor_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    enabled = bool(cfg.get("enabled", True))
    bundle = {
        "enabled": enabled,
        "model_family": "multiscale_temporal_factor_graph",
        "factor_count": len(retained_factor_rows),
        "scales": {},
    }
    blankets = {
        "enabled": enabled,
        "model_family": "multiscale_temporal_factor_graph",
        "scientific_role": "support_surface_only",
        "merged_blanket_factor_ids": [],
        "merged_target_factor_ids": [],
        "factor_support_rows": [],
        "edge_support_rows": [],
        "hidden_driver_support_rows": [],
        "phase3_member_canonical_names": [],
        "scales": {},
    }
    if not enabled:
        return bundle, blankets

    axes = read_json(phase15_dir / "multiscale_factor_axes.json", default={})
    factor_axis = [str(value) for value in list(axes.get("factor") or [])]
    factor_index = {factor_id: idx for idx, factor_id in enumerate(factor_axis)}
    factor_catalog = list(read_json(phase15_dir / "multiscale_factor_catalog.json", default=[]))
    factor_lookup = _factor_lookup(factor_catalog)
    normalized_retained_rows, retained_match_summary = _normalize_retained_factor_rows(
        retained_factor_rows=retained_factor_rows,
        factor_lookup=factor_lookup,
    )
    indicator_names = _indicator_names_by_factor(factor_catalog)
    factor_ids = _retained_factor_ids(
        retained_factor_rows=normalized_retained_rows,
        factor_index=factor_index,
        factor_lookup=factor_lookup,
        cfg=cfg,
    )
    target_factor_ids = _target_factor_ids(factor_ids, factor_lookup, cfg)
    bundle["retained_factor_match_summary"] = retained_match_summary
    blankets["retained_factor_match_summary"] = retained_match_summary

    scale_paths = {
        "province": phase15_dir / "multiscale_province_factor_tensor.npz",
        "region": phase15_dir / "multiscale_region_factor_tensor.npz",
        "national": phase15_dir / "multiscale_national_factor_tensor.npz",
    }
    factor_support = Counter()
    edge_support = Counter()
    hidden_support = Counter()
    canonical_names: set[str] = set()
    merged_blanket: set[str] = set()
    merged_target: set[str] = set()

    temporal_cfg = dict(cfg.get("temporal_graph", {}) or {})
    if not temporal_cfg:
        temporal_cfg = {
            "candidate_max_lags": [1, 2],
            "candidate_ridge_penalties": [0.05, 0.1, 0.2],
            "candidate_sparse_penalties": [0.01, 0.02, 0.04],
            "candidate_low_rank_penalties": [0.05, 0.1, 0.2],
            "decomposition_steps": 120,
            "convergence_tol": 1e-5,
            "bootstrap_draws": 12,
            "bootstrap_block_length": 6,
            "bootstrap_unit_fraction": 0.35,
            "bootstrap_time_fraction": 0.50,
            "min_effective_samples": 16,
            "null_permutations": 8,
            "null_quantile": 0.95,
            "rng_seed": 37,
        }

    for scale_name, path in scale_paths.items():
        if not path.exists():
            scale_bundle = {"scale_name": scale_name, "status": "unavailable", "reason": "missing_multiscale_factor_tensor"}
            scale_blanket = {"target_factor_ids": [], "blanket_factor_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
        else:
            scale_tensor = _scale_tensor(path, factor_ids, factor_index)
            if scale_tensor.size == 0 or scale_tensor.shape[-1] < int(cfg.get("min_factor_count", 2)):
                scale_bundle = {
                    "scale_name": scale_name,
                    "status": "unavailable",
                    "reason": "insufficient_retained_factors",
                    "factor_count": int(scale_tensor.shape[-1]) if scale_tensor.ndim == 3 else 0,
                }
                scale_blanket = {"target_factor_ids": [], "blanket_factor_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
            elif not target_factor_ids:
                scale_bundle = {
                    "scale_name": scale_name,
                    "status": "unavailable",
                    "reason": "missing_target_factors",
                    "factor_count": int(scale_tensor.shape[-1]) if scale_tensor.ndim == 3 else 0,
                    "uncertainty_available": False,
                }
                scale_blanket = {"target_factor_ids": [], "blanket_factor_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []}
            else:
                local_factor_ids = [factor_id for factor_id in factor_ids if factor_id in factor_index]
                phi_by_factor = _estimate_phi_by_series_tensor(scale_tensor, local_factor_ids)
                scale_bundle, scale_blanket = estimate_latent_temporal_scale_graph(
                    scale_name=scale_name,
                    state_tensor=scale_tensor,
                    block_axis=local_factor_ids,
                    target_block_ids=target_factor_ids,
                    indicator_names_by_block=indicator_names,
                    cfg=temporal_cfg,
                    phi_by_block=phi_by_factor,
                    uncertainty_tensor=None,
                )
                scale_bundle["graph_family"] = "temporal_factor_graph"
                scale_bundle["factor_count"] = len(local_factor_ids)
                scale_bundle["uncertainty_available"] = False
                scale_bundle["support_surface_kind"] = "multiscale_factor_support"
        bundle["scales"][scale_name] = scale_bundle
        blankets["scales"][scale_name] = scale_blanket
        if scale_bundle.get("status") != "completed":
            continue
        merged_blanket.update(scale_blanket["blanket_block_ids"])
        merged_target.update(scale_blanket["target_block_ids"])
        canonical_names.update(scale_blanket["phase3_member_canonical_names"])
        for factor_id in scale_blanket["blanket_block_ids"]:
            factor_support[factor_id] += 1
        for row in list(scale_bundle.get("edges") or []):
            edge_support[(str(row["source"]), str(row["target"]), int(row["lag"]))] += 1
        for row in list(scale_bundle.get("hidden_driver_rows") or []):
            hidden_support[(str(row["source"]), str(row["target"]), int(row["lag"]))] += 1

    blankets["merged_blanket_factor_ids"] = [factor_id for factor_id in factor_ids if factor_id in merged_blanket]
    blankets["merged_target_factor_ids"] = [factor_id for factor_id in factor_ids if factor_id in merged_target]
    blankets["phase3_member_canonical_names"] = sorted(canonical_names)
    blankets["factor_support_rows"] = [{"factor_id": factor_id, "support_count": int(factor_support[factor_id])} for factor_id in sorted(factor_support)]
    blankets["edge_support_rows"] = [
        {"source": source, "target": target, "lag": lag, "support_count": int(count)}
        for (source, target, lag), count in sorted(edge_support.items())
    ]
    blankets["hidden_driver_support_rows"] = [
        {"source": source, "target": target, "lag": lag, "support_count": int(count)}
        for (source, target, lag), count in sorted(hidden_support.items())
    ]
    return bundle, blankets


__all__ = ["build_multiscale_dag_outputs"]
