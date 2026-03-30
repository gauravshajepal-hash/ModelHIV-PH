from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.geography import macro_region_label
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.phase3.rescue_core import build_observation_ladder
from epigraph_ph.runtime import (
    RunContext,
    choose_torch_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    read_json,
    save_tensor_artifact,
    to_torch_tensor,
    utc_now_iso,
    write_ground_truth_package,
    write_json,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PHASE15_PROFILE_ID = "hiv_rescue_v2"
BLOCK_ORDER = [
    "epidemiology_cascade",
    "biology_severity",
    "economics_affordability",
    "logistics_access",
    "service_delivery_infrastructure",
    "mobility_network_mixing",
    "stigma_behavior_information",
    "population_structure_demography",
    "policy_implementation",
    "environment_disruption",
]
NETWORK_FEATURE_FAMILIES = ["reaction_diffusion", "percolation_fragility", "information_propagation"]
_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase15_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase15", {}) or {})


def _phase15_required(key: str) -> Any:
    cfg = _phase15_cfg()
    if key not in cfg:
        raise KeyError(f"Missing HIV phase15 constraint setting: {key}")
    return cfg[key]


def _phase15_required_section(key: str) -> dict[str, Any]:
    value = _phase15_required(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase15 constraint setting '{key}' must be a mapping")
    return dict(value)


@dataclass(slots=True)
class CandidateProfile:
    canonical_name: str
    canonical_index: int
    block_name: str
    source_mix: dict[str, int]
    lane_mix: dict[str, int]
    source_reliability_mix: dict[str, int]
    soft_tags: list[str]
    linkage_targets: list[str]
    geo_resolutions: dict[str, int]
    mean_surface: float
    std_surface: float
    evidence_weight: float
    bias_penalty: float
    numeric_support: int
    anchor_support: int
    promotion_hint: str
    network_hint: str


def _dominant_key(counter_map: dict[str, int] | Counter[str], default: str = "mixed") -> str:
    if not counter_map:
        return default
    items = sorted(counter_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return items[0][0]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    svd_eps = float(_phase15_required_section("factor_extraction")["svd_eps"])
    if x.size == 0 or y.size == 0:
        return 0.0
    if float(np.std(x)) <= svd_eps or float(np.std(y)) <= svd_eps:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _jaccard(left: list[str], right: list[str]) -> float:
    lset = {item for item in left if item}
    rset = {item for item in right if item}
    if not lset and not rset:
        return 0.0
    return float(len(lset & rset) / max(1, len(lset | rset)))


def _counter_overlap(left: dict[str, int], right: dict[str, int]) -> float:
    lset = {key for key, value in left.items() if value > 0 and key}
    rset = {key for key, value in right.items() if value > 0 and key}
    if not lset and not rset:
        return 0.0
    return float(len(lset & rset) / max(1, len(lset | rset)))


def _block_for_row(row: dict[str, Any]) -> tuple[str, str]:
    domain = str(row.get("domain_family") or "").lower()
    pathway = str(row.get("pathway_family") or "").lower()
    tags = {str(item).lower() for item in row.get("soft_ontology_tags") or []}
    source_bank = str(row.get("source_bank") or "").lower()
    if "biology" in domain or pathway == "biological_progression":
        return "biology_severity", "biology"
    if domain in {"economics"} or "economic" in tags or "afford" in pathway:
        return "economics_affordability", "economics"
    if domain in {"logistics"} or pathway in {"health_system_reach"}:
        return "logistics_access", "logistics"
    if domain in {"population"} or "demograph" in " ".join(tags):
        return "population_structure_demography", "population"
    if domain in {"behavior"} or pathway in {"testing_uptake", "prevention_access", "linkage_to_care"}:
        return "stigma_behavior_information", "behavior"
    if domain in {"policy"}:
        return "policy_implementation", "policy"
    if pathway in {"mobility_network_mixing"} or "mobility" in " ".join(tags) or "migration" in " ".join(tags):
        return "mobility_network_mixing", "mobility"
    if "environment" in tags or "climate" in tags or "typhoon" in tags:
        return "environment_disruption", "environment"
    if any(token in source_bank for token in ("harp", "registry", "phase0_extracted")):
        return "epidemiology_cascade", "cascade"
    return "service_delivery_infrastructure", "service"


def _network_hint(block_name: str) -> str:
    if block_name in {"mobility_network_mixing", "logistics_access"}:
        return "reaction_diffusion"
    if block_name in {"service_delivery_infrastructure", "policy_implementation"}:
        return "percolation_fragility"
    if block_name in {"stigma_behavior_information", "population_structure_demography"}:
        return "information_propagation"
    return ""


def _source_summary(normalized_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"row_count": 0, "bias_penalty_mean": 0.0, "source_banks": Counter()})
    for row in normalized_rows:
        reliability = str(row.get("source_reliability_class") or "unknown")
        item = by_class[reliability]
        item["row_count"] += 1
        item["bias_penalty_mean"] += float(row.get("bias_penalty") or 0.0)
        item["source_banks"][str(row.get("source_bank") or "")] += 1
    rows = []
    for reliability, payload in sorted(by_class.items()):
        count = payload["row_count"]
        rows.append(
            {
                "source_reliability_class": reliability,
                "row_count": count,
                "mean_bias_penalty": round(float(payload["bias_penalty_mean"] / max(1, count)), 6),
                "source_banks": dict(payload["source_banks"]),
            }
        )
    return {"rows": rows}


def _region_label(name: str) -> str:
    return macro_region_label(name)


def _profile_rows(
    normalized_rows: list[dict[str, Any]],
    standardized_tensor: np.ndarray,
    canonical_axis: list[str],
) -> tuple[list[CandidateProfile], dict[str, Any]]:
    axis_index = {name: idx for idx, name in enumerate(canonical_axis)}
    rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_mix": Counter(),
            "lane_mix": Counter(),
            "source_reliability_mix": Counter(),
            "soft_tags": Counter(),
            "linkage_targets": Counter(),
            "geo_resolutions": Counter(),
            "evidence_weight": 0.0,
            "bias_penalty": 0.0,
            "numeric_support": 0,
            "anchor_support": 0,
            "row_count": 0,
            "promotion_hints": Counter(),
            "block_names": Counter(),
        }
    )
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        if canonical_name not in axis_index:
            continue
        block_name, lane_label = _block_for_row(row)
        item = rollup[canonical_name]
        item["source_mix"][str(row.get("source_bank") or "")] += 1
        item["lane_mix"][lane_label] += 1
        item["source_reliability_mix"][str(row.get("source_reliability_class") or "unknown")] += 1
        item["geo_resolutions"][str(row.get("geo_resolution") or "unknown")] += 1
        item["evidence_weight"] += float(row.get("evidence_weight") or 0.0)
        item["bias_penalty"] += float(row.get("bias_penalty") or 0.0)
        item["numeric_support"] += 1 if row.get("model_numeric_value") is not None else 0
        item["anchor_support"] += 1 if row.get("is_anchor_eligible") else 0
        item["row_count"] += 1
        item["promotion_hints"][str(row.get("promotion_eligibility_hint") or "candidate_only")] += 1
        item["block_names"][block_name] += 1
        for tag in row.get("soft_ontology_tags") or []:
            item["soft_tags"][str(tag)] += 1
        for target in row.get("linkage_targets") or []:
            item["linkage_targets"][str(target)] += 1

    profiles: list[CandidateProfile] = []
    block_catalog: dict[str, list[str]] = defaultdict(list)
    for canonical_name in canonical_axis:
        if canonical_name not in rollup:
            continue
        item = rollup[canonical_name]
        idx = axis_index[canonical_name]
        surface = standardized_tensor[:, :, idx]
        block_name = _dominant_key(item["block_names"], "service_delivery_infrastructure")
        profile = CandidateProfile(
            canonical_name=canonical_name,
            canonical_index=idx,
            block_name=block_name,
            source_mix=dict(item["source_mix"]),
            lane_mix=dict(item["lane_mix"]),
            source_reliability_mix=dict(item["source_reliability_mix"]),
            soft_tags=[key for key, _value in item["soft_tags"].most_common(8)],
            linkage_targets=[key for key, _value in item["linkage_targets"].most_common(8)],
            geo_resolutions=dict(item["geo_resolutions"]),
            mean_surface=round(float(surface.mean()), 6),
            std_surface=round(float(surface.std()), 6),
            evidence_weight=round(float(item["evidence_weight"] / max(1, item["row_count"])), 6),
            bias_penalty=round(float(item["bias_penalty"] / max(1, item["row_count"])), 6),
            numeric_support=int(item["numeric_support"]),
            anchor_support=int(item["anchor_support"]),
            promotion_hint=_dominant_key(item["promotion_hints"], "candidate_only"),
            network_hint=_network_hint(block_name),
        )
        profiles.append(profile)
        block_catalog[block_name].append(canonical_name)
    block_payload = {
        "rows": [
            {"block_name": block_name, "member_count": len(names), "members": sorted(names)}
            for block_name, names in sorted(block_catalog.items())
        ]
    }
    return profiles, block_payload


def _signature_matrix(profiles: list[CandidateProfile]) -> np.ndarray:
    block_index = {name: idx for idx, name in enumerate(BLOCK_ORDER)}
    lane_vocab = ["cascade", "biology", "economics", "logistics", "behavior", "population", "policy", "mobility", "environment", "service"]
    matrix = np.zeros((len(profiles), len(BLOCK_ORDER) + len(lane_vocab) + 6), dtype=np.float32)
    for idx, profile in enumerate(profiles):
        matrix[idx, block_index.get(profile.block_name, 0)] = 1.0
        offset = len(BLOCK_ORDER)
        for lane_name, lane_count in profile.lane_mix.items():
            if lane_name in lane_vocab:
                matrix[idx, offset + lane_vocab.index(lane_name)] = float(lane_count)
        offset += len(lane_vocab)
        matrix[idx, offset + 0] = float(profile.evidence_weight)
        matrix[idx, offset + 1] = float(1.0 - profile.bias_penalty)
        matrix[idx, offset + 2] = float(profile.numeric_support > 0)
        matrix[idx, offset + 3] = float(profile.anchor_support > 0)
        matrix[idx, offset + 4] = float(profile.mean_surface)
        matrix[idx, offset + 5] = float(profile.std_surface)
    row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norm[row_norm == 0.0] = 1.0
    return matrix / row_norm


def _similarity_score(
    left: CandidateProfile,
    right: CandidateProfile,
    standardized_tensor: np.ndarray,
    signature_cosine: float,
) -> float:
    weight_cfg = _phase15_required_section("similarity_weights")
    quality_cfg = _phase15_required_section("similarity_quality")
    left_surface = standardized_tensor[:, :, left.canonical_index].reshape(-1)
    right_surface = standardized_tensor[:, :, right.canonical_index].reshape(-1)
    corr_sim = max(0.0, _safe_corr(left_surface, right_surface))
    semantic_sim = _jaccard(left.soft_tags, right.soft_tags)
    target_sim = _jaccard(left.linkage_targets, right.linkage_targets)
    source_sim = _counter_overlap(left.source_mix, right.source_mix)
    geo_sim = _counter_overlap(left.geo_resolutions, right.geo_resolutions)
    lane_sim = _counter_overlap(left.lane_mix, right.lane_mix)
    base = (
        float(weight_cfg["signature_cosine"]) * signature_cosine
        + float(weight_cfg["corr_sim"]) * corr_sim
        + float(weight_cfg["semantic_sim"]) * semantic_sim
        + float(weight_cfg["target_sim"]) * target_sim
        + float(weight_cfg["source_sim"]) * source_sim
        + float(weight_cfg["geo_sim"]) * geo_sim
        + float(weight_cfg["lane_sim"]) * lane_sim
    )
    quality = float(quality_cfg["evidence_weight_mix"]) * (left.evidence_weight + right.evidence_weight) + float(quality_cfg["bias_penalty_mix"]) * ((1.0 - left.bias_penalty) + (1.0 - right.bias_penalty)) / 2.0
    return float(np.clip(base * np.clip(quality, float(quality_cfg["quality_floor"]), 1.0), 0.0, 1.0))


def _build_similarity_graph(profiles: list[CandidateProfile], standardized_tensor: np.ndarray) -> tuple[list[dict[str, Any]], list[list[int]]]:
    if not profiles:
        return [], []
    signatures = _signature_matrix(profiles)
    cosine = np.clip(signatures @ signatures.T, 0.0, 1.0)
    adjacency: dict[int, set[int]] = defaultdict(set)
    edges: list[dict[str, Any]] = []
    block_to_indices: dict[str, list[int]] = defaultdict(list)
    similarity_cfg = _phase15_required_section("similarity_quality")
    for idx, profile in enumerate(profiles):
        block_to_indices[profile.block_name].append(idx)
    for indices in block_to_indices.values():
        for local_idx, idx in enumerate(indices):
            scored: list[tuple[int, float]] = []
            for jdx in indices[local_idx + 1 :]:
                score = _similarity_score(profiles[idx], profiles[jdx], standardized_tensor, float(cosine[idx, jdx]))
                if score >= float(similarity_cfg["within_block_threshold"]):
                    scored.append((jdx, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            for jdx, score in scored[: int(similarity_cfg["within_block_top_k"])]:
                adjacency[idx].add(jdx)
                adjacency[jdx].add(idx)
                edges.append(
                    {
                        "left": profiles[idx].canonical_name,
                        "right": profiles[jdx].canonical_name,
                        "block_name": profiles[idx].block_name,
                        "similarity_score": round(float(score), 6),
                    }
                )
    visited: set[int] = set()
    components: list[list[int]] = []
    for idx in range(len(profiles)):
        if idx in visited:
            continue
        stack = [idx]
        component: list[int] = []
        visited.add(idx)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return edges, components


def _cluster_payload(
    profiles: list[CandidateProfile],
    components: list[list[int]],
    standardized_tensor: np.ndarray,
) -> list[dict[str, Any]]:
    payload = []
    cluster_id = 0
    for component in components:
        component_profiles = [profiles[idx] for idx in component]
        block_name = _dominant_key(Counter(profile.block_name for profile in component_profiles), "service_delivery_infrastructure")
        centroid = np.mean([standardized_tensor[:, :, profile.canonical_index] for profile in component_profiles], axis=0)
        payload.append(
            {
                "cluster_id": f"cluster_{cluster_id:04d}",
                "block_name": block_name,
                "member_canonical_names": [profile.canonical_name for profile in component_profiles],
                "member_count": len(component_profiles),
                "cross_block_member_count": len({profile.block_name for profile in component_profiles}),
                "mean_signal_strength": round(float(np.mean(np.abs(centroid))), 6),
            }
        )
        cluster_id += 1
    return payload


def _factor_surface_from_cluster(
    cluster_profiles: list[CandidateProfile],
    standardized_tensor: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    factor_cfg = _phase15_required_section("factor_extraction")
    indices = [profile.canonical_index for profile in cluster_profiles]
    weights = np.asarray([max(float(factor_cfg["composite_weight_floor"]), profile.evidence_weight * (1.0 - 0.5 * profile.bias_penalty)) for profile in cluster_profiles], dtype=np.float32)
    weights = weights / np.clip(weights.sum(), float(factor_cfg["svd_eps"]), None)
    surfaces = standardized_tensor[:, :, indices].astype(np.float32)
    province_count, month_count, feature_count = surfaces.shape
    if feature_count == 1:
        return surfaces[:, :, 0], {cluster_profiles[0].canonical_name: 1.0}
    matrix = surfaces.reshape(province_count * month_count, feature_count)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    weighted = matrix * np.sqrt(weights.reshape(1, -1))
    try:
        _u, _s, vt = np.linalg.svd(weighted, full_matrices=False)
        loading = vt[0]
    except np.linalg.LinAlgError:
        loading = weights
    combined = matrix @ loading
    if _safe_corr(combined, matrix @ weights) < 0.0:
        loading = -loading
        combined = -combined
    combined = combined.reshape(province_count, month_count)
    combined = combined - combined.mean()
    scale = float(combined.std())
    if scale > float(factor_cfg["svd_eps"]):
        combined = combined / scale
    loading = loading / np.clip(np.linalg.norm(loading), float(factor_cfg["svd_eps"]), None)
    return combined.astype(np.float32), {
        cluster_profiles[idx].canonical_name: round(float(loading[idx]), 6) for idx in range(len(cluster_profiles))
    }


def _adjacency_from_profiles(features: np.ndarray, region_labels: list[str], top_k: int = 3) -> np.ndarray:
    graph_cfg = _phase15_required_section("network_graph")
    top_k = int(graph_cfg["adjacency_top_k"]) if top_k == 3 else int(top_k)
    province_count = features.shape[0]
    if province_count <= 1:
        return np.eye(province_count, dtype=np.float32)
    row_norm = np.linalg.norm(features, axis=1, keepdims=True)
    row_norm[row_norm == 0.0] = 1.0
    normed = features / row_norm
    cosine = np.clip(normed @ normed.T, 0.0, 1.0)
    adjacency = np.zeros((province_count, province_count), dtype=np.float32)
    for idx in range(province_count):
        scores = []
        for jdx in range(province_count):
            if idx == jdx:
                continue
            score = float(cosine[idx, jdx])
            if region_labels[idx] == region_labels[jdx] and region_labels[idx] != "mixed":
                score += float(graph_cfg["same_region_bonus"])
            scores.append((jdx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        for jdx, score in scores[:top_k]:
            adjacency[idx, jdx] = max(0.0, min(1.0, score))
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _normalized_graph_weights(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1, keepdims=True)
    degree[degree == 0.0] = 1.0
    return adjacency / degree


def _largest_component_ratio(adjacency: np.ndarray) -> float:
    edge_threshold = float(_phase15_required_section("network_graph")["positive_edge_threshold"])
    if adjacency.shape[0] <= 1:
        return 1.0
    visited: set[int] = set()
    best = 0
    for idx in range(adjacency.shape[0]):
        if idx in visited:
            continue
        stack = [idx]
        visited.add(idx)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            neighbors = np.where(adjacency[current] > edge_threshold)[0]
            for neighbor in neighbors:
                if int(neighbor) not in visited:
                    visited.add(int(neighbor))
                    stack.append(int(neighbor))
        best = max(best, size)
    return float(best / max(1, adjacency.shape[0]))


def _per_node_fragility(adjacency: np.ndarray) -> np.ndarray:
    province_count = adjacency.shape[0]
    scores = np.zeros((province_count,), dtype=np.float32)
    base_ratio = _largest_component_ratio(adjacency)
    for idx in range(province_count):
        mask = np.ones((province_count,), dtype=bool)
        mask[idx] = False
        reduced = adjacency[mask][:, mask]
        reduced_ratio = _largest_component_ratio(reduced) if reduced.size else 0.0
        scores[idx] = float(max(0.0, base_ratio - reduced_ratio))
    return scores


def _shortest_path_matrix(adjacency: np.ndarray) -> np.ndarray:
    graph_cfg = _phase15_required_section("network_graph")
    edge_threshold = float(graph_cfg["positive_edge_threshold"])
    shortest_path_floor = float(graph_cfg["shortest_path_floor"])
    province_count = adjacency.shape[0]
    dist = np.full((province_count, province_count), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    positive = adjacency > edge_threshold
    dist[positive] = 1.0 / np.clip(adjacency[positive], shortest_path_floor, None)
    for k in range(province_count):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def _network_feature_bundle(
    standardized_tensor: np.ndarray,
    profiles: list[CandidateProfile],
    province_axis: list[str],
    month_axis: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    if standardized_tensor.size == 0 or not profiles:
        return (
            np.zeros((len(province_axis), len(month_axis), 0), dtype=np.float32),
            [],
            {"graphs": []},
            np.zeros((0, len(province_axis), len(province_axis)), dtype=np.float32),
            [],
        )
    by_block: dict[str, list[int]] = defaultdict(list)
    for profile in profiles:
        by_block[profile.block_name].append(profile.canonical_index)

    def _block_surface(*names: str) -> np.ndarray:
        indices = [canonical for name in names for canonical in by_block.get(name, [])]
        if not indices:
            return np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
        return standardized_tensor[:, :, indices].mean(axis=-1).astype(np.float32)

    region_labels = [_region_label(name) for name in province_axis]
    mobility_base = _block_surface("mobility_network_mixing", "logistics_access")
    service_base = _block_surface("service_delivery_infrastructure", "policy_implementation", "epidemiology_cascade")
    information_base = _block_surface("stigma_behavior_information", "population_structure_demography")

    mobility_graph = _adjacency_from_profiles(np.stack([mobility_base.mean(axis=1), service_base.mean(axis=1)], axis=-1), region_labels)
    service_graph = _adjacency_from_profiles(np.stack([service_base.mean(axis=1), mobility_base.mean(axis=1)], axis=-1), region_labels)
    info_graph = _adjacency_from_profiles(np.stack([information_base.mean(axis=1), mobility_base.mean(axis=1)], axis=-1), region_labels)

    mobility_w = _normalized_graph_weights(mobility_graph)
    service_w = _normalized_graph_weights(service_graph)
    info_w = _normalized_graph_weights(info_graph)

    degree_mob = mobility_graph.sum(axis=1).astype(np.float32)
    graph_cfg = _phase15_required_section("network_graph")
    accessibility = 1.0 / np.clip(_shortest_path_matrix(service_graph).mean(axis=1), float(graph_cfg["accessibility_floor"]), None)
    fragility = _per_node_fragility(service_graph)
    redundancy = 1.0 - fragility
    single_point = np.maximum(0.0, fragility - fragility.mean())
    isolation = 1.0 - np.clip(info_graph.sum(axis=1) / np.clip(info_graph.sum(axis=1).max(), float(graph_cfg["isolation_eps"]), None), 0.0, 1.0)

    features = []
    for month_idx, _month in enumerate(month_axis):
        diffusion = mobility_w @ mobility_base[:, month_idx] - mobility_base[:, month_idx]
        inbound = mobility_w @ np.maximum(mobility_base[:, month_idx], 0.0)
        outbound = degree_mob * np.maximum(mobility_base[:, month_idx], 0.0)
        care_gradient = service_w @ service_base[:, month_idx] - service_base[:, month_idx]
        continuity_stress = np.abs(diffusion) + np.maximum(0.0, -care_gradient)
        awareness = info_w @ information_base[:, month_idx]
        stigma = info_w @ np.maximum(-information_base[:, month_idx], 0.0)
        testing_intent = info_w @ np.maximum(mobility_base[:, month_idx], 0.0)
        rumor = np.abs(stigma - awareness)
        month_features = np.stack(
            [
                diffusion,
                inbound,
                outbound,
                care_gradient,
                accessibility,
                continuity_stress,
                fragility,
                redundancy,
                single_point,
                awareness,
                stigma,
                testing_intent,
                rumor,
                isolation,
            ],
            axis=-1,
        ).astype(np.float32)
        features.append(month_features)
    tensor = np.stack(features, axis=1).astype(np.float32)

    feature_specs = [
        ("mobility_diffusion_pressure", "reaction_diffusion", "mobility_network_mixing"),
        ("mobility_inbound_exposure_flux", "reaction_diffusion", "mobility_network_mixing"),
        ("mobility_outbound_leakage_flux", "reaction_diffusion", "mobility_network_mixing"),
        ("care_access_gradient", "reaction_diffusion", "logistics_access"),
        ("treatment_hub_accessibility", "reaction_diffusion", "service_delivery_infrastructure"),
        ("continuity_of_care_stress", "reaction_diffusion", "service_delivery_infrastructure"),
        ("service_giant_component_loss", "percolation_fragility", "service_delivery_infrastructure"),
        ("service_redundancy_score", "percolation_fragility", "service_delivery_infrastructure"),
        ("service_single_point_failure_score", "percolation_fragility", "policy_implementation"),
        ("awareness_propagation_score", "information_propagation", "stigma_behavior_information"),
        ("stigma_diffusion_score", "information_propagation", "stigma_behavior_information"),
        ("testing_intent_spread_score", "information_propagation", "stigma_behavior_information"),
        ("rumor_vulnerability_proxy", "information_propagation", "stigma_behavior_information"),
        ("community_information_isolation_score", "information_propagation", "population_structure_demography"),
    ]
    feature_catalog = []
    for idx, (name, family, block_name) in enumerate(feature_specs):
        feature_catalog.append(
            {
                "factor_id": f"network_factor_{idx:03d}",
                "factor_name": name,
                "factor_class": "network_feature",
                "network_feature_family": family,
                "block_name": block_name,
                "member_canonical_names": [],
                "promotion_class": "scientific_retained",
                "transition_hooks": [],
            }
        )
    graph_bundle = {
        "graphs": [
            {"graph_name": "mobility_graph", "edge_count": int(np.sum(mobility_graph > 0.05) // 2), "largest_component_ratio": round(_largest_component_ratio(mobility_graph), 6)},
            {"graph_name": "service_graph", "edge_count": int(np.sum(service_graph > 0.05) // 2), "largest_component_ratio": round(_largest_component_ratio(service_graph), 6)},
            {"graph_name": "information_graph", "edge_count": int(np.sum(info_graph > 0.05) // 2), "largest_component_ratio": round(_largest_component_ratio(info_graph), 6)},
        ],
        "province_axis": province_axis,
    }
    operator_tensor = np.stack([mobility_w, service_w, info_w], axis=0).astype(np.float32)
    operator_catalog = [
        {"operator_id": "mobility_operator", "operator_name": "mobility_operator", "network_feature_family": "reaction_diffusion"},
        {"operator_id": "service_operator", "operator_name": "service_operator", "network_feature_family": "percolation_fragility"},
        {"operator_id": "information_operator", "operator_name": "information_operator", "network_feature_family": "information_propagation"},
    ]
    return tensor, feature_catalog, graph_bundle, operator_tensor, operator_catalog


def _environment_masks(
    province_axis: list[str],
    month_axis: list[str],
    tensor_rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    province_regions = np.asarray([_region_label(name) for name in province_axis])
    region_masks = {
        "luzon": province_regions == "luzon",
        "visayas": province_regions == "visayas",
        "mindanao": province_regions == "mindanao",
    }
    year_values = []
    for month in month_axis:
        if isinstance(month, str) and len(month) >= 4 and month[:4].isdigit():
            year_values.append(int(month[:4]))
        else:
            year_values.append(0)
    years = np.asarray(year_values, dtype=np.int32)
    if np.any(years > 0):
        time_masks = {
            "pre_pandemic": years < 2020,
            "pandemic": (years >= 2020) & (years <= 2022),
            "post_pandemic": years >= 2023,
        }
    else:
        midpoint = max(1, len(month_axis) // 2)
        time_masks = {"pre_pandemic": np.arange(len(month_axis)) < midpoint, "pandemic": np.arange(len(month_axis)) >= midpoint, "post_pandemic": np.zeros((len(month_axis),), dtype=bool)}
    province_counts: Counter[str] = Counter()
    for row in tensor_rows:
        province_counts[str(row.get("province") or row.get("geo") or "")] += 1
    counts = np.asarray([province_counts.get(name, 0) for name in province_axis], dtype=np.float32)
    median = float(np.median(counts)) if counts.size else 0.0
    data_quality_masks = {"high_data": counts >= median, "sparse_data": counts < median}
    return {"region_family": region_masks, "time_family": time_masks, "data_quality_family": data_quality_masks}


def _subnational_anomaly_score(surface: np.ndarray, target_surface: np.ndarray) -> float:
    if surface.size == 0 or target_surface.size == 0:
        return 0.0
    surface_centered = surface - surface.mean(axis=0, keepdims=True)
    target_centered = target_surface - target_surface.mean(axis=0, keepdims=True)
    return abs(_safe_corr(surface_centered.reshape(-1), target_centered.reshape(-1)))


def _region_contrast_score(surface: np.ndarray, region_masks: dict[str, np.ndarray]) -> float:
    if surface.size == 0:
        return 0.0
    regional_profiles = []
    for mask in region_masks.values():
        if not np.any(mask):
            continue
        regional_profiles.append(surface[mask].mean(axis=0))
    if len(regional_profiles) < 2:
        return 0.0
    stacked = np.stack(regional_profiles, axis=0)
    between = float(np.mean(np.std(stacked, axis=0)))
    within = float(np.mean(np.std(surface, axis=0)))
    if within <= 1e-6:
        return 0.0
    return float(np.clip(between / within, 0.0, 1.0))


def _factor_stability(
    factor_tensor: np.ndarray,
    factor_catalog: list[dict[str, Any]],
    observation_targets: dict[str, np.ndarray],
    environment_masks: dict[str, dict[str, np.ndarray]],
    mesoscopic_factor_members: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_candidates = {
        "diagnosed_stock": observation_targets["diagnosed_stock"],
        "art_stock": observation_targets["art_stock"],
        "documented_suppression": observation_targets["documented_suppression"],
        "testing_coverage": observation_targets["testing_coverage"],
    }
    stability_rows = []
    promotion_pool = []
    rng = np.random.default_rng(17)
    stability_cfg = _phase15_required_section("stability")
    score_weights = dict(stability_cfg["score_weights"])
    main_thresholds = dict(stability_cfg["main_thresholds"])
    support_thresholds = dict(stability_cfg["support_thresholds"])
    for factor_idx, factor in enumerate(factor_catalog):
        surface = factor_tensor[:, :, factor_idx]
        flat = surface.reshape(-1)
        target_scores = {target_name: _safe_corr(flat, target_surface.reshape(-1)) for target_name, target_surface in target_candidates.items()}
        subnational_scores = {
            target_name: _subnational_anomaly_score(surface, target_surface)
            for target_name, target_surface in target_candidates.items()
        }
        best_target = max(target_scores.items(), key=lambda item: abs(item[1]))
        best_subnational = max(subnational_scores.items(), key=lambda item: abs(item[1]))
        env_signs = []
        env_scores = []
        for family, masks in environment_masks.items():
            for _env_name, mask in masks.items():
                if family == "time_family":
                    env_surface = surface[:, mask]
                    env_target = target_candidates[best_target[0]][:, mask]
                else:
                    env_surface = surface[mask]
                    env_target = target_candidates[best_target[0]][mask]
                if env_surface.size < 4 or env_target.size < 4:
                    continue
                corr = _safe_corr(env_surface.reshape(-1), env_target.reshape(-1))
                env_signs.append(np.sign(corr))
                env_scores.append(abs(corr))
        sign_stability = float(np.mean(np.asarray(env_signs) == np.sign(best_target[1]))) if env_signs else 0.0
        stability = float(np.mean(env_scores)) if env_scores else 0.0
        region_contrast = _region_contrast_score(surface, environment_masks.get("region_family", {}))
        permuted = []
        target_flat = target_candidates[best_target[0]].reshape(-1).copy()
        for _ in range(int(stability_cfg["permutation_draws"])):
            rng.shuffle(target_flat)
            permuted.append(abs(_safe_corr(flat, target_flat)))
        null_gap = float(abs(best_target[1]) - np.mean(permuted))
        missing_mask = rng.uniform(size=surface.shape) > float(stability_cfg["missing_dropout_rate"])
        missing_surface = surface * missing_mask
        missing_robustness = float(abs(_safe_corr(surface.reshape(-1), missing_surface.reshape(-1))))
        has_non_seed_member = any("seed" not in member for member in mesoscopic_factor_members.get(factor["factor_id"], []))
        source_dropout_robustness = (
            float(stability_cfg["source_dropout_network"])
            if factor.get("factor_class") == "network_feature"
            else (float(stability_cfg["source_dropout_non_seed"]) if has_non_seed_member else float(stability_cfg["source_dropout_seed_only"]))
        )
        predictive_gain = float(abs(best_target[1]))
        subnational_gain = float(abs(best_subnational[1]))
        stability_score = float(
            np.clip(
                float(score_weights["predictive_gain"]) * predictive_gain
                + float(score_weights["subnational_gain"]) * subnational_gain
                + float(score_weights["stability"]) * stability
                + float(score_weights["sign_stability"]) * sign_stability
                + float(score_weights["missing_robustness"]) * missing_robustness
                + float(score_weights["source_dropout_robustness"]) * source_dropout_robustness
                + float(score_weights["region_contrast"]) * region_contrast,
                0.0,
                1.0,
            )
        )
        eligible_main = bool(
            (predictive_gain >= float(main_thresholds["predictive_gain"]) or subnational_gain >= float(main_thresholds["subnational_gain"]))
            and stability_score >= float(main_thresholds["stability_score"])
            and sign_stability >= float(main_thresholds["sign_stability"])
            and null_gap >= float(main_thresholds["null_gap"])
            and region_contrast >= float(main_thresholds["region_contrast"])
            and factor.get("promotion_hint", "candidate_only") != "candidate_only"
        )
        eligible_support = bool(
            predictive_gain >= float(support_thresholds["predictive_gain"])
            or subnational_gain >= float(support_thresholds["subnational_gain"])
            or factor.get("factor_class") == "network_feature"
        )
        row = {
            "factor_id": factor["factor_id"],
            "factor_name": factor["factor_name"],
            "block_name": factor["block_name"],
            "best_target": best_target[0],
            "best_target_correlation": round(float(best_target[1]), 6),
            "best_subnational_target": best_subnational[0],
            "best_subnational_correlation": round(float(best_subnational[1]), 6),
            "predictive_gain": round(predictive_gain, 6),
            "subnational_anomaly_gain": round(subnational_gain, 6),
            "region_contrast_score": round(region_contrast, 6),
            "sign_stability": round(sign_stability, 6),
            "predictive_gain_stability": round(stability, 6),
            "missing_data_robustness": round(missing_robustness, 6),
            "source_dropout_robustness": round(source_dropout_robustness, 6),
            "permutation_null_gap": round(null_gap, 6),
            "stability_score": round(stability_score, 6),
            "eligible_main_predictive": eligible_main,
            "eligible_supporting_context": eligible_support,
            "network_feature_family": factor.get("network_feature_family", ""),
        }
        stability_rows.append(row)
        promotion_pool.append(
            row
            | {
                "promotion_class": "scientific_retained" if eligible_main else ("supporting_context" if eligible_support else "exploratory"),
                "factor_class": factor["factor_class"],
                "transition_hooks": factor.get("transition_hooks", []),
                "promotion_hint": factor.get("promotion_hint", "candidate_only"),
            }
        )
    return stability_rows, promotion_pool


def run_phase15_build(*, run_id: str, plugin_id: str, profile: str = PHASE15_PROFILE_ID) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase15_dir = ensure_dir(ctx.run_dir / "phase15")
    normalized_rows = read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    parameter_catalog = read_json(ctx.run_dir / "phase1" / "parameter_catalog.json", default=[])
    axis_catalogs = read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    tensor_rows = read_json(ctx.run_dir / "phase1" / "tensor_rows.json", default=[])
    standardized_tensor = load_tensor_artifact(ctx.run_dir / "phase1" / "standardized_tensor.npz")

    province_axis = list(axis_catalogs.get("province", [])) or ["national"]
    month_axis = list(axis_catalogs.get("month", [])) or ["unknown"]
    canonical_axis = list(axis_catalogs.get("canonical_name", []))

    source_reliability = _source_summary(normalized_rows)
    profiles, block_catalog = _profile_rows(normalized_rows, standardized_tensor, canonical_axis)
    similarity_edges, components = _build_similarity_graph(profiles, standardized_tensor)
    clusters = _cluster_payload(profiles, components, standardized_tensor)

    factor_catalog = []
    factor_surfaces = []
    loading_matrix = np.zeros((len(components), len(canonical_axis)), dtype=np.float32)
    mesoscopic_factor_members: dict[str, list[str]] = {}
    for factor_idx, component in enumerate(components):
        cluster_profiles = [profiles[idx] for idx in component]
        factor_surface, loadings = _factor_surface_from_cluster(cluster_profiles, standardized_tensor)
        factor_id = f"factor_{factor_idx:04d}"
        factor_name = f"{_dominant_key(Counter(profile.block_name for profile in cluster_profiles), 'service')}_factor_{factor_idx:04d}"
        best_block = _dominant_key(Counter(profile.block_name for profile in cluster_profiles), "service_delivery_infrastructure")
        promotion_hint = "promotion_eligible" if any(profile.promotion_hint == "promotion_eligible" for profile in cluster_profiles) else "candidate_only"
        network_hint = _dominant_key(Counter(profile.network_hint for profile in cluster_profiles if profile.network_hint), "")
        factor_catalog.append(
            {
                "factor_id": factor_id,
                "factor_name": factor_name,
                "factor_class": "mesoscopic_factor",
                "block_name": best_block,
                "member_canonical_names": [profile.canonical_name for profile in cluster_profiles],
                "source_mix": dict(sum((Counter(profile.source_mix) for profile in cluster_profiles), Counter())),
                "source_reliability_mix": dict(sum((Counter(profile.source_reliability_mix) for profile in cluster_profiles), Counter())),
                "lane_mix": dict(sum((Counter(profile.lane_mix) for profile in cluster_profiles), Counter())),
                "missingness_rate": round(float(np.mean(np.abs(factor_surface) <= 1e-8)), 6),
                "interpretability_label": best_block,
                "promotion_hint": promotion_hint,
                "network_feature_family": network_hint,
                "transition_hooks": [],
                "promotion_class": "scientific_retained",
            }
        )
        factor_surfaces.append(factor_surface)
        mesoscopic_factor_members[factor_id] = [profile.canonical_name for profile in cluster_profiles]
        for canonical_name, loading in loadings.items():
            if canonical_name in canonical_axis:
                loading_matrix[factor_idx, canonical_axis.index(canonical_name)] = float(loading)

    network_tensor, network_factors, graph_bundle, operator_tensor, operator_catalog = _network_feature_bundle(standardized_tensor, profiles, province_axis, month_axis)
    for network_factor in network_factors:
        factor_catalog.append(
            network_factor
            | {
                "source_mix": {},
                "source_reliability_mix": {},
                "lane_mix": {},
                "missingness_rate": 0.0,
                "interpretability_label": network_factor["network_feature_family"],
                "promotion_hint": "promotion_eligible",
            }
        )
        mesoscopic_factor_members[network_factor["factor_id"]] = []
    factor_tensor = np.stack(factor_surfaces, axis=-1).astype(np.float32) if factor_surfaces else np.zeros((len(province_axis), len(month_axis), 0), dtype=np.float32)
    if network_tensor.size:
        factor_tensor = np.concatenate([factor_tensor, network_tensor], axis=-1) if factor_tensor.size else network_tensor
        loading_matrix = np.pad(loading_matrix, ((0, network_tensor.shape[-1]), (0, 0)), mode="constant")

    _observation_ladder, observation_targets, _target_rows = build_observation_ladder(
        standardized_tensor=standardized_tensor,
        normalized_rows=normalized_rows,
        parameter_catalog=parameter_catalog,
        canonical_axis=canonical_axis,
        province_axis=province_axis,
        month_axis=month_axis,
    )
    environments = _environment_masks(province_axis, month_axis, tensor_rows)
    stability_rows, promotion_pool = _factor_stability(factor_tensor, factor_catalog, observation_targets, environments, mesoscopic_factor_members)
    stability_by_id = {row["factor_id"]: row for row in stability_rows}
    for factor in factor_catalog:
        factor.update(stability_by_id.get(factor["factor_id"], {}))
        if factor.get("best_target") == "diagnosed_stock":
            factor["transition_hooks"] = ["diagnosis_transitions"]
        elif factor.get("best_target") == "art_stock":
            factor["transition_hooks"] = ["linkage_transitions", "retention_attrition_transitions"]
        elif factor.get("best_target") == "documented_suppression":
            factor["transition_hooks"] = ["suppression_transitions"]
        elif factor.get("best_target") == "testing_coverage":
            factor["transition_hooks"] = ["diagnosis_transitions", "subgroup_allocation_priors"]
        if factor.get("network_feature_family") == "reaction_diffusion":
            factor["transition_hooks"] = sorted(set(factor["transition_hooks"]) | {"diagnosis_transitions", "retention_attrition_transitions"})
        if factor.get("network_feature_family") == "percolation_fragility":
            factor["transition_hooks"] = sorted(set(factor["transition_hooks"]) | {"retention_attrition_transitions", "suppression_transitions"})
        if factor.get("network_feature_family") == "information_propagation":
            factor["transition_hooks"] = sorted(set(factor["transition_hooks"]) | {"diagnosis_transitions", "linkage_transitions", "subgroup_allocation_priors"})

    factor_rows = []
    for factor_idx, factor in enumerate(factor_catalog):
        surface = factor_tensor[:, :, factor_idx]
        for province_idx, province in enumerate(province_axis):
            for month_idx, month in enumerate(month_axis):
                factor_rows.append(
                    {
                        "factor_id": factor["factor_id"],
                        "factor_name": factor["factor_name"],
                        "province": province,
                        "time": month,
                        "value": round(float(surface[province_idx, month_idx]), 6),
                    }
                )

    factor_artifact = save_tensor_artifact(
        array=to_torch_tensor(factor_tensor, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else factor_tensor,
        axis_names=["province", "month", "factor"],
        artifact_dir=phase15_dir,
        stem="mesoscopic_factor_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase15_mesoscopic_factor_tensor"],
        save_pt=False,
    )
    loadings_artifact = save_tensor_artifact(
        array=loading_matrix,
        axis_names=["factor", "canonical_name"],
        artifact_dir=phase15_dir,
        stem="factor_loading_matrix",
        backend="numpy",
        device="cpu",
        notes=["phase15_factor_loading_matrix"],
        save_pt=False,
    )
    network_artifact = save_tensor_artifact(
        array=network_tensor,
        axis_names=["province", "month", "network_feature"],
        artifact_dir=phase15_dir,
        stem="network_feature_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase15_network_feature_tensor"],
        save_pt=False,
    )
    operator_artifact = save_tensor_artifact(
        array=operator_tensor,
        axis_names=["network_operator", "province", "province_neighbor"],
        artifact_dir=phase15_dir,
        stem="network_operator_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase15_network_operator_tensor"],
        save_pt=False,
    )

    write_json(phase15_dir / "source_reliability.json", source_reliability)
    write_json(phase15_dir / "candidate_similarity_graph.json", {"edge_count": len(similarity_edges), "edges": similarity_edges})
    write_json(phase15_dir / "domain_block_catalog.json", block_catalog)
    write_json(phase15_dir / "cluster_catalog.json", clusters)
    write_json(phase15_dir / "mesoscopic_factor_catalog.json", factor_catalog)
    write_json(phase15_dir / "factor_rows.json", factor_rows)
    write_json(phase15_dir / "network_graph_bundle.json", graph_bundle)
    write_json(phase15_dir / "network_feature_catalog.json", network_factors)
    write_json(phase15_dir / "network_operator_catalog.json", operator_catalog)
    write_json(phase15_dir / "stability_environments.json", {family: list(masks.keys()) for family, masks in environments.items()})
    write_json(phase15_dir / "factor_stability_report.json", stability_rows)
    write_json(phase15_dir / "factor_promotion_pool.json", promotion_pool)

    backend_map = detect_backends()
    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase1"),
        extracted_dir=str(phase15_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase15": "completed"},
        artifact_paths={
            "source_reliability": str(phase15_dir / "source_reliability.json"),
            "candidate_similarity_graph": str(phase15_dir / "candidate_similarity_graph.json"),
            "domain_block_catalog": str(phase15_dir / "domain_block_catalog.json"),
            "cluster_catalog": str(phase15_dir / "cluster_catalog.json"),
            "mesoscopic_factor_catalog": str(phase15_dir / "mesoscopic_factor_catalog.json"),
            "mesoscopic_factor_tensor": factor_artifact["value_path"],
            "factor_loading_matrix": loadings_artifact["value_path"],
            "factor_rows": str(phase15_dir / "factor_rows.json"),
            "network_graph_bundle": str(phase15_dir / "network_graph_bundle.json"),
            "network_feature_catalog": str(phase15_dir / "network_feature_catalog.json"),
            "network_feature_tensor": network_artifact["value_path"],
            "network_operator_catalog": str(phase15_dir / "network_operator_catalog.json"),
            "network_operator_tensor": operator_artifact["value_path"],
            "factor_stability_report": str(phase15_dir / "factor_stability_report.json"),
            "factor_promotion_pool": str(phase15_dir / "factor_promotion_pool.json"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, False, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        source_count=len(normalized_rows),
        canonical_candidate_count=len(profiles),
        numeric_observation_count=int(factor_tensor.shape[-1]),
        notes=["phase15_mesoscopic_factor_engine", "phase15_network_features:reaction_diffusion_percolation_information_propagation"],
    ).to_dict()
    manifest["profile_id"] = profile
    truth_paths = write_ground_truth_package(
        phase_dir=phase15_dir,
        phase_name="phase15",
        profile_id=profile,
        checks=[
            {"name": "factor_catalog_present", "passed": bool(factor_catalog)},
            {"name": "factor_tensor_finite", "passed": bool(np.isfinite(factor_tensor).all())},
            {"name": "network_tensor_finite", "passed": bool(np.isfinite(network_tensor).all())},
            {"name": "operator_tensor_finite", "passed": bool(np.isfinite(operator_tensor).all())},
            {"name": "factor_count_matches_tensor", "passed": int(factor_tensor.shape[-1]) == len(factor_catalog)},
            {"name": "stability_report_matches_catalog", "passed": len(stability_rows) == len(factor_catalog)},
        ],
        truth_sources=NETWORK_FEATURE_FAMILIES[:],
        stage_manifest_path=str(phase15_dir / "phase15_manifest.json"),
        summary={
            "factor_count": len(factor_catalog),
            "network_feature_count": int(network_tensor.shape[-1]) if network_tensor.ndim == 3 else 0,
            "cluster_count": len(clusters),
            "similarity_edge_count": len(similarity_edges),
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase15_dir / "phase15_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase15_build",
        [
            phase15_dir / "source_reliability.json",
            phase15_dir / "candidate_similarity_graph.json",
            phase15_dir / "domain_block_catalog.json",
            phase15_dir / "cluster_catalog.json",
            phase15_dir / "mesoscopic_factor_catalog.json",
            phase15_dir / "mesoscopic_factor_tensor.npz",
            phase15_dir / "factor_loading_matrix.npz",
            phase15_dir / "factor_rows.json",
            phase15_dir / "network_graph_bundle.json",
            phase15_dir / "network_feature_catalog.json",
            phase15_dir / "network_feature_tensor.npz",
            phase15_dir / "network_operator_catalog.json",
            phase15_dir / "network_operator_tensor.npz",
            phase15_dir / "factor_stability_report.json",
            phase15_dir / "factor_promotion_pool.json",
            phase15_dir / "phase15_manifest.json",
        ],
    )
    return manifest
