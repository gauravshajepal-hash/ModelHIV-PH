from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


def _edge_rows_from_adjacency(adjacency: np.ndarray, axis: list[str], *, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_idx, source in enumerate(axis):
        for right_idx, target in enumerate(axis):
            if left_idx == right_idx:
                continue
            weight = float(adjacency[left_idx, right_idx])
            if abs(weight) <= threshold:
                continue
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "weight": round(weight, 6),
                }
            )
    rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    return rows


def _synthetic_factor_catalog_row(factor: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    block_name = str(factor.get("block_name") or "mixed")
    domain = "mixed"
    if "biology" in block_name:
        domain = "biology"
    elif block_name in {"stigma_behavior_information", "mobility_network_mixing"}:
        domain = "behavior"
    pathway = str(pool.get("best_target") or "")
    if not pathway:
        hooks = list(factor.get("transition_hooks") or [])
        if "suppression_transitions" in hooks:
            pathway = "suppression_outcomes"
        elif "retention_attrition_transitions" in hooks:
            pathway = "retention_adherence"
        elif "linkage_transitions" in hooks:
            pathway = "linkage_to_care"
        elif "diagnosis_transitions" in hooks:
            pathway = "testing_uptake"
        elif "subgroup_allocation_priors" in hooks:
            pathway = "prevention_access"
    return {
        "canonical_name": factor["factor_id"],
        "domain_families": {domain: 1},
        "pathway_families": {pathway or "mixed": 1},
    }


def _target_relevance_score(pool: dict[str, Any], factor: dict[str, Any], target_tokens: list[str]) -> int:
    best_target = str(pool.get("best_target") or factor.get("best_target") or "").lower()
    members = " ".join(str(item) for item in (factor.get("member_canonical_names") or [])).lower()
    score = 0
    if any(token in best_target for token in target_tokens):
        score += 3
    member_hits = sum(1 for token in target_tokens if token in members)
    score += min(member_hits, 3)
    promotion_class = str(pool.get("promotion_class") or factor.get("promotion_class") or "")
    if promotion_class in {"supporting_context", "main_predictive"}:
        score += 1
    return score


def _phase3_target_blanket_from_factors(
    adjacency: np.ndarray,
    factor_rows: list[dict[str, Any]],
    *,
    edge_threshold: float,
) -> dict[str, Any]:
    if adjacency.size == 0 or not factor_rows:
        return {
            "target_factor_ids": [],
            "blanket_factor_ids": [],
            "blanket_indices": [],
            "phase3_member_canonical_names": [],
        }
    target_indices = [idx for idx, row in enumerate(factor_rows) if bool(row.get("phase3_target_relevant"))]
    if not target_indices:
        degree = np.abs(adjacency).sum(axis=0) + np.abs(adjacency).sum(axis=1)
        target_indices = [int(np.argmax(degree))]
    blanket_indices: set[int] = set(target_indices)
    for idx in target_indices:
        parents = np.where(np.abs(adjacency[:, idx]) > edge_threshold)[0]
        children = np.where(np.abs(adjacency[idx, :]) > edge_threshold)[0]
        blanket_indices.update(int(item) for item in parents.tolist())
        blanket_indices.update(int(item) for item in children.tolist())
        for child in children:
            spouses = np.where(np.abs(adjacency[:, child]) > edge_threshold)[0]
            blanket_indices.update(int(item) for item in spouses.tolist())
    blanket_factor_ids = [factor_rows[idx]["factor_id"] for idx in sorted(blanket_indices)]
    phase3_member_canonical_names = sorted(
        {
            member
            for idx in sorted(blanket_indices)
            for member in list(factor_rows[idx].get("member_canonical_names") or [])
        }
    )
    return {
        "target_factor_ids": [factor_rows[idx]["factor_id"] for idx in target_indices],
        "blanket_factor_ids": blanket_factor_ids,
        "blanket_indices": sorted(blanket_indices),
        "phase3_member_canonical_names": phase3_member_canonical_names,
    }


def build_sharded_block_graph_outputs(
    *,
    candidate_profiles: list[dict[str, Any]],
    canonical_axis: list[str],
    parameter_catalog: list[dict[str, Any]],
    time_feature_matrix: np.ndarray,
    phase15_factor_catalog: list[dict[str, Any]],
    phase15_factor_pool: list[dict[str, Any]],
    phase15_factor_tensor: np.ndarray,
    skeleton_threshold: float,
    edge_threshold: float,
    notears_steps: int,
    block_cfg: dict[str, Any],
    node_tiers_fn: Callable[[list[dict[str, Any]], list[str]], np.ndarray],
    tier_mask_fn: Callable[[list[dict[str, Any]], list[str]], np.ndarray],
    lag_mask_fn: Callable[[list[dict[str, Any]], list[str]], np.ndarray],
    mi_prefilter_fn: Callable[[np.ndarray, list[str]], tuple[np.ndarray, dict[str, Any]]],
    pc_skeleton_fn: Callable[[np.ndarray, list[str]], tuple[np.ndarray, dict[str, Any]]],
    corr_skeleton_fn: Callable[[np.ndarray, float], np.ndarray],
    notears_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    project_dag_fn: Callable[[np.ndarray, np.ndarray, list[str]], tuple[np.ndarray, dict[str, Any]]],
    blanket_fn: Callable[[np.ndarray, list[str]], dict[str, Any]],
) -> dict[str, Any]:
    active_statuses = {str(item) for item in block_cfg.get("active_statuses", [])}
    max_candidates_per_block = int(block_cfg["max_candidates_per_block"])
    min_candidates_per_block = int(block_cfg["min_candidates_per_block"])
    retained_factor_classes = {str(item) for item in block_cfg.get("retained_factor_classes", [])}
    bridge_edge_budget = int(block_cfg["bridge_edge_budget_per_block_pair"])
    target_tokens = [str(item).lower() for item in block_cfg.get("phase3_target_tokens", [])]
    max_target_factors = int(block_cfg.get("max_target_factors", 6))

    catalog_by_name = {str(row.get("canonical_name") or ""): row for row in parameter_catalog}
    canonical_index = {name: idx for idx, name in enumerate(canonical_axis)}

    grouped_profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_profiles:
        block_name = str(row.get("primary_block") or "mixed")
        grouped_profiles[block_name].append(row)
    for rows in grouped_profiles.values():
        rows.sort(
            key=lambda item: (
                float(item.get("curation_score") or 0.0),
                float(item.get("dag_score") or 0.0),
                int(item.get("support_count") or 0),
                int(item.get("numeric_support") or 0),
            ),
            reverse=True,
        )

    banks = []
    block_graphs = []
    for block_name, rows in sorted(grouped_profiles.items()):
        retained_rows = [
            row for row in rows
            if str(row.get("curation_status") or "") in active_statuses and str(row.get("canonical_name") or "") in canonical_index
        ][:max_candidates_per_block]
        pruned_count = max(0, len(rows) - len(retained_rows))
        banks.append(
            {
                "block_name": block_name,
                "candidate_count": len(rows),
                "retained_candidate_count": len(retained_rows),
                "pruned_candidate_count": pruned_count,
                "retained_canonical_names": [str(row.get("canonical_name") or "") for row in retained_rows],
                "retained_statuses": [str(row.get("curation_status") or "") for row in retained_rows],
            }
        )
        if len(retained_rows) < min_candidates_per_block:
            block_graphs.append(
                {
                    "block_name": block_name,
                    "status": "skipped_too_small",
                    "retained_candidate_count": len(retained_rows),
                    "edge_count": 0,
                    "blanket_nodes": [],
                }
            )
            continue
        block_axis = [str(row["canonical_name"]) for row in retained_rows]
        block_indices = [canonical_index[name] for name in block_axis]
        submatrix = np.asarray(time_feature_matrix[:, block_indices], dtype=np.float32)
        subcatalog = [dict(catalog_by_name.get(name, {"canonical_name": name})) for name in block_axis]
        mi_keep, mi_report = mi_prefilter_fn(submatrix, block_axis)
        pc_mask, pc_report = pc_skeleton_fn(submatrix, block_axis)
        corr_mask = corr_skeleton_fn(submatrix, threshold=skeleton_threshold)
        tier_mask = tier_mask_fn(subcatalog, block_axis)
        lag_mask = lag_mask_fn(subcatalog, block_axis)
        final_mask = np.outer(mi_keep, mi_keep).astype(np.float32) * pc_mask * corr_mask * tier_mask * lag_mask
        raw_adjacency = notears_fn(submatrix, final_mask, steps=notears_steps)
        node_tiers = node_tiers_fn(subcatalog, block_axis)
        adjacency, dag_report = project_dag_fn(raw_adjacency, node_tiers, block_axis)
        blanket = blanket_fn(adjacency, block_axis)
        block_graphs.append(
            {
                "block_name": block_name,
                "status": "completed",
                "retained_candidate_count": len(retained_rows),
                "mi_kept_feature_count": int(mi_report.get("kept_feature_count", 0)),
                "pc_edge_count": int(pc_report.get("edge_count", 0)),
                "edge_count": len(_edge_rows_from_adjacency(adjacency, block_axis, threshold=edge_threshold)),
                "dag_projection_report": dag_report,
                "markov_blanket": blanket,
                "edges": _edge_rows_from_adjacency(adjacency, block_axis, threshold=edge_threshold)[:100],
            }
        )

    pool_by_id = {str(row.get("factor_id") or ""): row for row in phase15_factor_pool}
    retained_factor_rows = []
    selected_factor_indices = []
    for idx, factor in enumerate(phase15_factor_catalog):
        factor_id = str(factor.get("factor_id") or "")
        pool = dict(pool_by_id.get(factor_id, {}))
        promotion_class = str(pool.get("promotion_class") or factor.get("promotion_class") or "")
        if retained_factor_classes and promotion_class not in retained_factor_classes:
            continue
        factor_row = dict(factor)
        factor_row["promotion_class"] = promotion_class
        factor_row["best_target"] = pool.get("best_target", "")
        factor_row["phase3_target_score"] = _target_relevance_score(pool, factor_row, target_tokens)
        factor_row["phase3_target_relevant"] = False
        retained_factor_rows.append(factor_row)
        selected_factor_indices.append(idx)
    if retained_factor_rows:
        ranked_targets = sorted(
            retained_factor_rows,
            key=lambda row: (
                int(row.get("phase3_target_score") or 0),
                float(row.get("predictive_gain") or 0.0),
                float(row.get("stability_score") or 0.0),
                str(row.get("factor_id") or ""),
            ),
            reverse=True,
        )
        selected_target_ids = {
            str(row.get("factor_id") or "")
            for row in ranked_targets[:max_target_factors]
            if int(row.get("phase3_target_score") or 0) > 0
        }
        for factor_row in retained_factor_rows:
            factor_row["phase3_target_relevant"] = str(factor_row.get("factor_id") or "") in selected_target_ids

    bridge_bundle: dict[str, Any]
    phase3_target_blankets: dict[str, Any]
    if retained_factor_rows and phase15_factor_tensor.ndim == 3 and phase15_factor_tensor.shape[-1] >= len(selected_factor_indices):
        selected_tensor = np.asarray(phase15_factor_tensor[:, :, selected_factor_indices], dtype=np.float32)
        bridge_axis = [str(row["factor_id"]) for row in retained_factor_rows]
        bridge_matrix = selected_tensor.reshape(-1, len(bridge_axis))
        synthetic_catalog = [_synthetic_factor_catalog_row(row, pool_by_id.get(row["factor_id"], {})) for row in retained_factor_rows]
        mi_keep, mi_report = mi_prefilter_fn(bridge_matrix, bridge_axis)
        pc_mask, pc_report = pc_skeleton_fn(bridge_matrix, bridge_axis)
        corr_mask = corr_skeleton_fn(bridge_matrix, threshold=skeleton_threshold)
        tier_mask = tier_mask_fn(synthetic_catalog, bridge_axis)
        lag_mask = lag_mask_fn(synthetic_catalog, bridge_axis)
        final_mask = np.outer(mi_keep, mi_keep).astype(np.float32) * pc_mask * corr_mask * tier_mask * lag_mask
        raw_adjacency = notears_fn(bridge_matrix, final_mask, steps=notears_steps)
        node_tiers = node_tiers_fn(synthetic_catalog, bridge_axis)
        adjacency, dag_report = project_dag_fn(raw_adjacency, node_tiers, bridge_axis)
        edge_rows = _edge_rows_from_adjacency(adjacency, bridge_axis, threshold=edge_threshold)
        kept_edges = []
        budget_counter: dict[tuple[str, str], int] = defaultdict(int)
        pruned_adjacency = np.zeros_like(adjacency, dtype=np.float32)
        factor_by_id = {row["factor_id"]: row for row in retained_factor_rows}
        bridge_index = {factor_id: idx for idx, factor_id in enumerate(bridge_axis)}
        for row in edge_rows:
            source_block = str(factor_by_id[row["source"]].get("block_name") or "mixed")
            target_block = str(factor_by_id[row["target"]].get("block_name") or "mixed")
            pair = (source_block, target_block)
            if budget_counter[pair] >= bridge_edge_budget:
                continue
            budget_counter[pair] += 1
            kept_edges.append(row | {"source_block": source_block, "target_block": target_block})
            left_idx = bridge_index[row["source"]]
            right_idx = bridge_index[row["target"]]
            pruned_adjacency[left_idx, right_idx] = float(row["weight"])
        phase3_target_blankets = _phase3_target_blanket_from_factors(
            pruned_adjacency,
            retained_factor_rows,
            edge_threshold=edge_threshold,
        )
        bridge_bundle = {
            "status": "completed",
            "factor_count": len(retained_factor_rows),
            "selected_factor_ids": bridge_axis,
            "mi_prefilter_report": mi_report,
            "pc_skeleton_report": pc_report,
            "dag_projection_report": dag_report,
            "bridge_edge_budget_per_block_pair": bridge_edge_budget,
            "bridge_edges": kept_edges,
            "blanket_factor_ids": list(phase3_target_blankets.get("blanket_factor_ids", [])),
            "pruned_adjacency": pruned_adjacency,
            "retained_factor_rows": retained_factor_rows,
            "retained_factor_tensor": selected_tensor,
        }
    else:
        bridge_bundle = {
            "status": "unavailable",
            "factor_count": 0,
            "selected_factor_ids": [],
            "bridge_edges": [],
            "blanket_factor_ids": [],
            "pruned_adjacency": np.zeros((0, 0), dtype=np.float32),
            "retained_factor_rows": [],
            "retained_factor_tensor": np.zeros((0, 0, 0), dtype=np.float32),
        }
        phase3_target_blankets = {
            "target_factor_ids": [],
            "blanket_factor_ids": [],
            "blanket_indices": [],
            "phase3_member_canonical_names": [],
        }

    pruning_report = {
        "document_level_pruning": {"delegated_to_phase0": True},
        "candidate_level_pruning": {
            "active_statuses": sorted(active_statuses),
            "retained_candidate_count": int(sum(bank["retained_candidate_count"] for bank in banks)),
            "pruned_candidate_count": int(sum(bank["pruned_candidate_count"] for bank in banks)),
        },
        "within_block_edge_pruning": {
            "block_count": len(block_graphs),
            "completed_block_count": sum(1 for row in block_graphs if row.get("status") == "completed"),
        },
        "cluster_to_factor_collapse": {
            "retained_factor_count": int(len(bridge_bundle["retained_factor_rows"])),
        },
        "cross_block_edge_budget": {
            "bridge_edge_budget_per_block_pair": bridge_edge_budget,
            "retained_bridge_edge_count": int(len(bridge_bundle["bridge_edges"])),
        },
        "target_blanket_pruning": {
            "target_factor_count": int(len(phase3_target_blankets.get("target_factor_ids", []))),
            "blanket_factor_count": int(len(phase3_target_blankets.get("blanket_factor_ids", []))),
            "phase3_member_canonical_count": int(len(phase3_target_blankets.get("phase3_member_canonical_names", []))),
        },
    }

    return {
        "block_candidate_banks": banks,
        "block_graph_bundle": {
            "blocks": block_graphs,
        },
        "bridge_graph_bundle": bridge_bundle,
        "phase3_target_blankets": phase3_target_blankets,
        "pruning_report": pruning_report,
    }
