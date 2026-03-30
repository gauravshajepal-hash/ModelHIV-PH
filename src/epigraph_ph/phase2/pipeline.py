from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
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
    to_torch_tensor,
    utc_now_iso,
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


def _lagged_feature_matrix(standardized_tensor: np.ndarray) -> np.ndarray:
    if standardized_tensor.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # aggregate over provinces -> time x feature
    return standardized_tensor.mean(axis=0).astype(np.float32)


def _safe_feature_corr(matrix_tf: np.ndarray) -> np.ndarray:
    if matrix_tf.ndim != 2 or matrix_tf.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feature_count = int(matrix_tf.shape[1])
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if feature_count == 1:
        return np.zeros((1, 1), dtype=np.float32)
    corr = np.corrcoef(matrix_tf, rowvar=False)
    corr = np.asarray(corr, dtype=np.float32)
    if corr.ndim == 0:
        corr = np.zeros((feature_count, feature_count), dtype=np.float32)
    elif corr.shape != (feature_count, feature_count):
        corr = np.reshape(corr, (feature_count, feature_count)).astype(np.float32)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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
    index = {row["canonical_name"]: row for row in parameter_catalog}
    tiers = []
    for name in canonical_axis:
        row = index.get(name, {})
        domain_family = _dominant_key(row.get("domain_families"), "mixed")
        pathway_family = _dominant_key(row.get("pathway_families"), "mixed")
        tiers.append(_tier_for_node(name, domain_family, pathway_family))
    tiers = np.asarray(tiers, dtype=int)
    mask = (tiers[:, None] <= tiers[None, :]).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def _lag_mask(feature_count: int) -> np.ndarray:
    mask = np.ones((feature_count, feature_count), dtype=np.float32)
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
    device = choose_torch_device(prefer_gpu=True)
    X = to_torch_tensor(matrix_tf, device=device, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    mask = to_torch_tensor(admissibility_mask, device=device, dtype=torch.float32)
    W = torch.nn.Parameter(torch.zeros((feature_count, feature_count), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([W], lr=float(notears_cfg["optimizer_lr"]))
    steps = int(notears_cfg["steps"]) if steps == 200 else int(steps)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        W_masked = W * mask
        W_masked = W_masked - torch.diag(torch.diag(W_masked))
        recon = X @ W_masked
        loss = torch.mean((X - recon) ** 2)
        h = torch.trace(torch.matrix_exp(W_masked * W_masked)) - feature_count
        penalty = (
            float(notears_cfg["lasso_penalty"]) * torch.sum(torch.abs(W_masked))
            + float(notears_cfg["ridge_penalty"]) * torch.sum(W_masked ** 2)
            + float(notears_cfg["acyclicity_penalty"]) * (h ** 2)
        )
        total = loss + penalty
        total.backward()
        optimizer.step()
    result = (W.detach() * mask).cpu().numpy().astype(np.float32)
    np.fill_diagonal(result, 0.0)
    return result


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

    time_feature_matrix = _lagged_feature_matrix(standardized_tensor)
    tier_mask = _tier_mask(parameter_catalog, canonical_axis)
    lag_mask = _lag_mask(len(canonical_axis))
    skeleton_threshold = float(_phase2_required("skeleton_threshold"))
    notears_cfg = _phase2_required_section("notears")
    skeleton_mask = _skeleton_from_corr(time_feature_matrix, threshold=skeleton_threshold)
    final_mask = tier_mask * lag_mask * skeleton_mask
    adjacency = _notears_optimize(time_feature_matrix, final_mask, steps=int(notears_cfg["steps"]))
    edge_scores = []
    for i, source_name in enumerate(canonical_axis):
        for j, target_name in enumerate(canonical_axis):
            weight = float(adjacency[i, j]) if adjacency.size else 0.0
            if abs(weight) >= 0.03:
                edge_scores.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "weight": round(weight, 6),
                        "abs_weight": round(abs(weight), 6),
                    }
                )
    edge_scores.sort(key=lambda item: item["abs_weight"], reverse=True)

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
        profile = {
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
        profile["curation_status"] = _curation_status(profile)
        candidate_profiles.append(profile)

    ranked_linkages = []
    for (canonical_name, target), support in linkage_counts.most_common():
        profile = next(item for item in candidate_profiles if item["canonical_name"] == canonical_name)
        linkage_cfg = _phase2_required_section("linkage_score_weights")
        support_denominator = float(_phase2_required("linkage_support_denominator"))
        score = min(
            1.0,
            float(linkage_cfg["support_count"]) * min(1.0, support / support_denominator)
            + float(linkage_cfg["relevance_score"]) * profile["relevance_score"]
            + float(linkage_cfg["stability_score"]) * profile["stability_score"]
            + float(linkage_cfg["dag_score"]) * profile["dag_score"]
            + float(linkage_cfg["numeric_support_ratio"]) * profile["numeric_support_ratio"],
        )
        ranked_linkages.append(
            {
                "canonical_name": canonical_name,
                "linkage_target": target,
                "support_count": support,
                "numeric_support": profile["numeric_support"],
                "source_bank_count": profile["source_diversity"],
                "linkage_score": round(score, 4),
                "primary_block": profile["primary_block"],
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
    for profile in candidate_profiles:
        grouped_blocks[profile["primary_block"]].append(profile)
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

    write_json(phase2_dir / "edge_scores.json", edge_scores)
    write_json(phase2_dir / "markov_blanket.json", blanket)
    write_json(phase2_dir / "candidate_profiles.json", candidate_profiles)
    write_json(phase2_dir / "curated_candidate_blocks.json", curated_candidate_blocks)
    write_json(phase2_dir / "block_registry.json", block_registry)
    write_json(phase2_dir / "ranked_linkages.json", ranked_linkages)
    write_json(phase2_dir / "parameter_edges.json", param_edges)
    write_json(phase2_dir / "linkage_graph.json", linkage_graph)
    write_json(phase2_dir / "curation_summary.json", curation_summary)

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
        ],
        truth_sources=["prior_truth", "synthetic_truth", "proxy_truth"],
        stage_manifest_path=str(phase2_dir / "phase2_manifest.json"),
        summary={
            "candidate_profile_count": len(candidate_profiles),
            "curated_candidate_count": len(curated_candidate_blocks),
            "blanket_count": len(blanket.get("blanket_nodes", [])),
            "edge_count": len(edge_scores),
        },
    )
    manifest["artifact_paths"].update(truth_paths)
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
            phase2_dir / "phase2_manifest.json",
        ],
    )
    return manifest


def _transition_hook_rank(hooks: list[str]) -> list[str]:
    order = [
        "diagnosis_transitions",
        "linkage_transitions",
        "retention_attrition_transitions",
        "suppression_transitions",
        "subgroup_allocation_priors",
    ]
    return [hook for hook in order if hook in hooks]


def _augment_phase2_for_rescue_v2(*, run_id: str, plugin_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase2_dir = ensure_dir(ctx.run_dir / "phase2")
    factor_catalog = read_json(ctx.run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    factor_pool = read_json(ctx.run_dir / "phase15" / "factor_promotion_pool.json", default=[])
    factor_tensor = load_tensor_artifact(ctx.run_dir / "phase15" / "mesoscopic_factor_tensor.npz")
    stability_report = read_json(ctx.run_dir / "phase15" / "factor_stability_report.json", default=[])
    if not factor_catalog or factor_tensor.size == 0:
        empty = {"rows": [], "summary": {"factor_count": 0}}
        write_json(phase2_dir / "factor_tournament_plan.json", empty)
        write_json(phase2_dir / "factor_tournament_results.json", empty)
        write_json(phase2_dir / "promoted_factor_set.json", [])
        write_json(phase2_dir / "supporting_factor_set.json", [])
        write_json(
            phase2_dir / "promotion_admission.json",
            {
                "status": "none_admitted",
                "reason": "no_factor_catalog_or_tensor",
                "top_near_misses": [],
            },
        )
        write_json(phase2_dir / "promotion_budget_report.json", {"main_predictive_count": 0, "supporting_count": 0})
        write_json(phase2_dir / "stability_gate_report.json", {"rows": stability_report})
        write_json(phase2_dir / "factor_diagnostics.json", {"rows": []})
        manifest["profile_id"] = PHASE15_PROFILE_ID
        manifest.setdefault("artifact_paths", {})
        manifest["artifact_paths"].update(
            {
                "factor_tournament_plan": str(phase2_dir / "factor_tournament_plan.json"),
                "factor_tournament_results": str(phase2_dir / "factor_tournament_results.json"),
                "promoted_factor_set": str(phase2_dir / "promoted_factor_set.json"),
                "supporting_factor_set": str(phase2_dir / "supporting_factor_set.json"),
                "promotion_admission": str(phase2_dir / "promotion_admission.json"),
                "promotion_budget_report": str(phase2_dir / "promotion_budget_report.json"),
                "stability_gate_report": str(phase2_dir / "stability_gate_report.json"),
                "factor_diagnostics": str(phase2_dir / "factor_diagnostics.json"),
            }
        )
        write_json(phase2_dir / "phase2_manifest.json", manifest)
        return manifest

    pool_by_id = {row["factor_id"]: row for row in factor_pool}
    catalog_by_id = {row["factor_id"]: row for row in factor_catalog}
    shortlist_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics = []
    factor_diag_cfg = _phase2_required_section("factor_diagnostic_weights")
    for factor in factor_catalog:
        pool = pool_by_id.get(factor["factor_id"], {})
        score = (
            float(factor_diag_cfg["stability_score"]) * float(pool.get("stability_score", 0.0))
            + float(factor_diag_cfg["predictive_gain"]) * float(pool.get("predictive_gain", 0.0))
            + float(factor_diag_cfg["subnational_anomaly_gain"]) * float(pool.get("subnational_anomaly_gain", 0.0))
            + float(factor_diag_cfg["region_contrast_score"]) * float(pool.get("region_contrast_score", 0.0))
            + float(factor_diag_cfg["source_dropout_robustness"]) * float(pool.get("source_dropout_robustness", 0.0))
            + float(factor_diag_cfg["missing_data_robustness"]) * float(pool.get("missing_data_robustness", 0.0))
            + float(factor_diag_cfg["permutation_null_gap"]) * float(pool.get("permutation_null_gap", 0.0))
        )
        diag_row = {
            "factor_id": factor["factor_id"],
            "factor_name": factor["factor_name"],
            "block_name": factor["block_name"],
            "diagnostic_score": round(float(score), 6),
            "promotion_hint": pool.get("promotion_hint", factor.get("promotion_hint", "candidate_only")),
            "eligible_main_predictive": bool(pool.get("eligible_main_predictive")),
            "eligible_supporting_context": bool(pool.get("eligible_supporting_context")),
            "network_feature_family": pool.get("network_feature_family", factor.get("network_feature_family", "")),
            "best_target": pool.get("best_target", ""),
            "best_subnational_target": pool.get("best_subnational_target", ""),
            "subnational_anomaly_gain": round(float(pool.get("subnational_anomaly_gain", 0.0)), 6),
            "region_contrast_score": round(float(pool.get("region_contrast_score", 0.0)), 6),
            "transition_hooks": _transition_hook_rank(list(factor.get("transition_hooks", []))),
        }
        diagnostics.append(diag_row)
        shortlist_by_block[factor["block_name"]].append(diag_row)

    for block_name, rows in shortlist_by_block.items():
        rows.sort(
            key=lambda item: (
                bool(item["eligible_main_predictive"]),
                bool(item["eligible_supporting_context"]),
                float(item["diagnostic_score"]),
            ),
            reverse=True,
        )
        shortlist_by_block[block_name] = rows[: int(_phase2_required("shortlist_per_block"))]

    tournament_plan = []
    tournament_results = []
    budget_cfg = _phase2_required_section("budgets")
    main_budget = int(budget_cfg["main"])
    support_budget = int(budget_cfg["support"])
    network_main_budget = int(budget_cfg["network_main"])
    promoted_main = []
    promoted_support = []
    network_main_count = 0
    for block_name, rows in shortlist_by_block.items():
        tournament_plan.append({"block_name": block_name, "shortlist_factor_ids": [row["factor_id"] for row in rows], "shortlist_count": len(rows)})
        for rank, row in enumerate(rows):
            pool = pool_by_id.get(row["factor_id"], {})
            factor = catalog_by_id[row["factor_id"]]
            class_decision = "exploratory"
            if (
                row["eligible_main_predictive"]
                and main_budget > 0
                and (network_main_count < network_main_budget or not row["network_feature_family"])
            ):
                class_decision = "main_predictive"
                main_budget -= 1
                if row["network_feature_family"]:
                    network_main_count += 1
            elif row["eligible_supporting_context"] and support_budget > 0:
                class_decision = "supporting_context"
                support_budget -= 1
            factor_row = {
                "factor_id": row["factor_id"],
                "factor_name": row["factor_name"],
                "block_name": block_name,
                "promotion_class": class_decision,
                "factor_class": factor.get("factor_class", "mesoscopic_factor"),
                "transition_hooks": row["transition_hooks"],
                "best_target": row["best_target"],
                "diagnostic_score": row["diagnostic_score"],
                "stability_score": pool.get("stability_score", 0.0),
                "predictive_gain": pool.get("predictive_gain", 0.0),
                "network_feature_family": row["network_feature_family"],
                "tournament_rank": rank + 1,
            }
            tournament_results.append(factor_row)
            if class_decision == "main_predictive":
                promoted_main.append(factor_row)
            elif class_decision == "supporting_context":
                promoted_support.append(factor_row)

    budget_report = {
        "main_predictive_count": len(promoted_main),
        "supporting_count": len(promoted_support),
        "remaining_main_budget": main_budget,
        "remaining_support_budget": support_budget,
        "network_main_predictive_count": network_main_count,
        "max_main_predictive": int(budget_cfg["main"]),
        "max_supporting": int(budget_cfg["support"]),
        "max_network_main_predictive": int(budget_cfg["network_main"]),
    }
    stability_gate_report = {"rows": stability_report}
    top_near_misses = [
        {
            "factor_id": row["factor_id"],
            "factor_name": row["factor_name"],
            "block_name": row["block_name"],
            "diagnostic_score": row["diagnostic_score"],
            "subnational_anomaly_gain": row.get("subnational_anomaly_gain", 0.0),
            "region_contrast_score": row.get("region_contrast_score", 0.0),
            "reason": "failed_main_predictive_gate",
        }
        for row in sorted(diagnostics, key=lambda item: float(item["diagnostic_score"]), reverse=True)[:5]
        if row["factor_id"] not in {item["factor_id"] for item in promoted_main}
    ]
    promotion_admission = (
        {
            "status": "admitted_main_predictive",
            "main_predictive_factor_ids": [row["factor_id"] for row in promoted_main],
            "supporting_factor_ids": [row["factor_id"] for row in promoted_support],
            "top_near_misses": top_near_misses,
        }
        if promoted_main
        else {
            "status": "none_admitted",
            "reason": "no_factor_cleared_subnational_main_predictive_gates",
            "supporting_factor_ids": [row["factor_id"] for row in promoted_support],
            "top_near_misses": top_near_misses,
        }
    )

    write_json(phase2_dir / "factor_tournament_plan.json", tournament_plan)
    write_json(phase2_dir / "factor_tournament_results.json", tournament_results)
    write_json(phase2_dir / "promoted_factor_set.json", promoted_main)
    write_json(phase2_dir / "supporting_factor_set.json", promoted_support)
    write_json(phase2_dir / "promotion_admission.json", promotion_admission)
    write_json(phase2_dir / "promotion_budget_report.json", budget_report)
    write_json(phase2_dir / "stability_gate_report.json", stability_gate_report)
    write_json(phase2_dir / "factor_diagnostics.json", diagnostics)

    manifest = dict(manifest)
    manifest["profile_id"] = PHASE15_PROFILE_ID
    manifest.setdefault("artifact_paths", {})
    manifest["artifact_paths"].update(
        {
            "factor_tournament_plan": str(phase2_dir / "factor_tournament_plan.json"),
            "factor_tournament_results": str(phase2_dir / "factor_tournament_results.json"),
            "promoted_factor_set": str(phase2_dir / "promoted_factor_set.json"),
            "supporting_factor_set": str(phase2_dir / "supporting_factor_set.json"),
            "promotion_admission": str(phase2_dir / "promotion_admission.json"),
            "promotion_budget_report": str(phase2_dir / "promotion_budget_report.json"),
            "stability_gate_report": str(phase2_dir / "stability_gate_report.json"),
            "factor_diagnostics": str(phase2_dir / "factor_diagnostics.json"),
        }
    )
    manifest.setdefault("notes", [])
    manifest["notes"] = list(manifest["notes"]) + ["phase2_rescue_v2:mesoscopic_factor_tournaments"]
    truth_paths = write_ground_truth_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        profile_id=PHASE15_PROFILE_ID,
        checks=[
            {"name": "tournament_results_present", "passed": bool(tournament_results)},
            {"name": "main_budget_respected", "passed": len(promoted_main) <= 8},
            {"name": "support_budget_respected", "passed": len(promoted_support) <= 12},
            {"name": "promotion_admission_present", "passed": True},
            {
                "name": "no_exploratory_promotions",
                "passed": all(pool_by_id.get(row["factor_id"], {}).get("promotion_class") != "exploratory" for row in promoted_main),
            },
            {"name": "network_main_budget_respected", "passed": network_main_count <= 3},
        ],
        truth_sources=["benchmark_truth", "null_test", "synthetic_truth"],
        stage_manifest_path=str(phase2_dir / "phase2_manifest.json"),
        summary={
            "main_predictive_count": len(promoted_main),
            "supporting_context_count": len(promoted_support),
            "diagnostic_factor_count": len(diagnostics),
            "promotion_admission_status": promotion_admission["status"],
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase2_dir / "phase2_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase2_build",
        [
            phase2_dir / "factor_tournament_plan.json",
            phase2_dir / "factor_tournament_results.json",
            phase2_dir / "promoted_factor_set.json",
            phase2_dir / "supporting_factor_set.json",
            phase2_dir / "promotion_admission.json",
            phase2_dir / "promotion_budget_report.json",
            phase2_dir / "stability_gate_report.json",
            phase2_dir / "factor_diagnostics.json",
            phase2_dir / "phase2_manifest.json",
        ],
    )
    return manifest


def run_phase2_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    manifest = _run_phase2_build_base(run_id=run_id, plugin_id=plugin_id, profile=profile)
    if profile == PHASE15_PROFILE_ID:
        return _augment_phase2_for_rescue_v2(run_id=run_id, plugin_id=plugin_id, manifest=manifest)
    return manifest
