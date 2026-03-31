from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np


def _adjacency_from_profiles(features: np.ndarray, region_labels: list[str], *, graph_cfg: dict[str, Any]) -> np.ndarray:
    top_k = int(graph_cfg["adjacency_top_k"])
    province_count = features.shape[0]
    if province_count <= 1:
        return np.eye(province_count, dtype=np.float32)
    row_norm = np.linalg.norm(features, axis=1, keepdims=True)
    row_norm[row_norm == 0.0] = 1.0
    normed = features / row_norm
    cosine = np.clip(normed @ normed.T, 0.0, 1.0)
    adjacency = np.zeros((province_count, province_count), dtype=np.float32)
    same_region_bonus = float(graph_cfg["same_region_bonus"])
    for idx in range(province_count):
        scores = []
        for jdx in range(province_count):
            if idx == jdx:
                continue
            score = float(cosine[idx, jdx])
            if region_labels[idx] == region_labels[jdx] and region_labels[idx] != "mixed":
                score += same_region_bonus
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


def _largest_component_ratio(adjacency: np.ndarray, *, edge_threshold: float) -> float:
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
                neighbor_idx = int(neighbor)
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    stack.append(neighbor_idx)
        best = max(best, size)
    return float(best / max(1, adjacency.shape[0]))


def _per_node_fragility(adjacency: np.ndarray, *, edge_threshold: float) -> np.ndarray:
    province_count = adjacency.shape[0]
    scores = np.zeros((province_count,), dtype=np.float32)
    base_ratio = _largest_component_ratio(adjacency, edge_threshold=edge_threshold)
    for idx in range(province_count):
        mask = np.ones((province_count,), dtype=bool)
        mask[idx] = False
        reduced = adjacency[mask][:, mask]
        reduced_ratio = _largest_component_ratio(reduced, edge_threshold=edge_threshold) if reduced.size else 0.0
        scores[idx] = float(max(0.0, base_ratio - reduced_ratio))
    return scores


def _shortest_path_matrix(adjacency: np.ndarray, *, graph_cfg: dict[str, Any]) -> np.ndarray:
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


def build_network_feature_bundle(
    standardized_tensor: np.ndarray,
    profiles: list[Any],
    province_axis: list[str],
    month_axis: list[str],
    *,
    graph_cfg: dict[str, Any],
    region_labels: list[str],
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
        by_block[str(profile.block_name)].append(int(profile.canonical_index))

    def _block_surface(*names: str) -> np.ndarray:
        indices = [canonical for name in names for canonical in by_block.get(name, [])]
        if not indices:
            return np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
        return standardized_tensor[:, :, indices].mean(axis=-1).astype(np.float32)

    mobility_base = _block_surface("mobility_network_mixing", "logistics_access")
    service_base = _block_surface("service_delivery_infrastructure", "policy_implementation", "epidemiology_cascade")
    information_base = _block_surface("stigma_behavior_information", "population_structure_demography")

    mobility_graph = _adjacency_from_profiles(
        np.stack([mobility_base.mean(axis=1), service_base.mean(axis=1)], axis=-1),
        region_labels,
        graph_cfg=graph_cfg,
    )
    service_graph = _adjacency_from_profiles(
        np.stack([service_base.mean(axis=1), mobility_base.mean(axis=1)], axis=-1),
        region_labels,
        graph_cfg=graph_cfg,
    )
    info_graph = _adjacency_from_profiles(
        np.stack([information_base.mean(axis=1), mobility_base.mean(axis=1)], axis=-1),
        region_labels,
        graph_cfg=graph_cfg,
    )

    mobility_w = _normalized_graph_weights(mobility_graph)
    service_w = _normalized_graph_weights(service_graph)
    info_w = _normalized_graph_weights(info_graph)

    degree_mob = mobility_graph.sum(axis=1).astype(np.float32)
    accessibility = 1.0 / np.clip(_shortest_path_matrix(service_graph, graph_cfg=graph_cfg).mean(axis=1), float(graph_cfg["accessibility_floor"]), None)
    fragility = _per_node_fragility(service_graph, edge_threshold=float(graph_cfg["positive_edge_threshold"]))
    redundancy = 1.0 - fragility
    single_point = np.maximum(0.0, fragility - fragility.mean())
    isolation = 1.0 - np.clip(
        info_graph.sum(axis=1) / np.clip(info_graph.sum(axis=1).max(), float(graph_cfg["isolation_eps"]), None),
        0.0,
        1.0,
    )

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
                "promotion_class": "unranked",
                "transition_hooks": [],
            }
        )
    graph_bundle = {
        "graphs": [
            {
                "graph_name": "mobility_graph",
                "edge_count": int(np.sum(mobility_graph > 0.05) // 2),
                "largest_component_ratio": round(
                    _largest_component_ratio(mobility_graph, edge_threshold=float(graph_cfg["positive_edge_threshold"])),
                    6,
                ),
            },
            {
                "graph_name": "service_graph",
                "edge_count": int(np.sum(service_graph > 0.05) // 2),
                "largest_component_ratio": round(
                    _largest_component_ratio(service_graph, edge_threshold=float(graph_cfg["positive_edge_threshold"])),
                    6,
                ),
            },
            {
                "graph_name": "information_graph",
                "edge_count": int(np.sum(info_graph > 0.05) // 2),
                "largest_component_ratio": round(
                    _largest_component_ratio(info_graph, edge_threshold=float(graph_cfg["positive_edge_threshold"])),
                    6,
                ),
            },
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


def build_environment_masks(
    province_axis: list[str],
    month_axis: list[str],
    tensor_rows: list[dict[str, Any]],
    *,
    region_labels: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    province_regions = np.asarray(region_labels)
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
        time_masks = {
            "pre_pandemic": np.arange(len(month_axis)) < midpoint,
            "pandemic": np.arange(len(month_axis)) >= midpoint,
            "post_pandemic": np.zeros((len(month_axis),), dtype=bool),
        }
    province_counts: Counter[str] = Counter()
    for row in tensor_rows:
        province_counts[str(row.get("province") or row.get("geo") or "")] += 1
    counts = np.asarray([province_counts.get(name, 0) for name in province_axis], dtype=np.float32)
    median = float(np.median(counts)) if counts.size else 0.0
    data_quality_masks = {"high_data": counts >= median, "sparse_data": counts < median}
    return {"region_family": region_masks, "time_family": time_masks, "data_quality_family": data_quality_masks}


def _subnational_anomaly_score(surface: np.ndarray, target_surface: np.ndarray, *, corr_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    if surface.size == 0 or target_surface.size == 0:
        return 0.0
    surface_centered = surface - surface.mean(axis=0, keepdims=True)
    target_centered = target_surface - target_surface.mean(axis=0, keepdims=True)
    return abs(corr_fn(surface_centered.reshape(-1), target_centered.reshape(-1)))


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
    if within <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.clip(between / within, 0.0, 1.0))


def compute_factor_stability(
    factor_tensor: np.ndarray,
    factor_catalog: list[dict[str, Any]],
    observation_targets: dict[str, np.ndarray],
    environment_masks: dict[str, dict[str, np.ndarray]],
    mesoscopic_factor_members: dict[str, list[str]],
    *,
    stability_cfg: dict[str, Any],
    corr_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    factor_catalog_by_id = {str(row.get("factor_id") or ""): row for row in factor_catalog}
    def _mae(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return 1.0
        return float(np.mean(np.abs(left - right)))

    def _smape(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return 1.0
        denom = np.abs(left) + np.abs(right) + 1e-6
        return float(np.mean(2.0 * np.abs(left - right) / denom))

    def _time_holdout_columns(month_count: int, *, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if month_count <= 1:
            return np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32)
        holdout_fraction = float(cfg["holdout_fraction"])
        min_holdout = max(1, int(cfg["holdout_min_months"]))
        holdout_months = max(min_holdout, int(np.ceil(month_count * holdout_fraction)))
        holdout_months = min(max(1, month_count - 1), holdout_months)
        split = month_count - holdout_months
        train_idx = np.arange(0, split, dtype=np.int32)
        holdout_idx = np.arange(split, month_count, dtype=np.int32)
        return train_idx, holdout_idx

    def _fit_linear_predictor(train_x: np.ndarray, train_y: np.ndarray, pred_x: np.ndarray) -> np.ndarray:
        x = train_x.reshape(-1).astype(np.float64)
        y = train_y.reshape(-1).astype(np.float64)
        pred = pred_x.reshape(-1).astype(np.float64)
        if x.size == 0 or y.size == 0 or pred.size == 0:
            return np.zeros_like(pred, dtype=np.float64)
        if float(np.std(x)) <= 1e-8:
            return np.full_like(pred, float(np.mean(y)), dtype=np.float64)
        design = np.stack([np.ones_like(x), x], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred_design = np.stack([np.ones_like(pred), pred], axis=1)
        return pred_design @ coeffs

    def _holdout_metrics(surface: np.ndarray, target_surface: np.ndarray, *, cfg: dict[str, Any]) -> dict[str, float]:
        month_count = int(surface.shape[1]) if surface.ndim == 2 else 0
        if month_count <= 1:
            return {
                "holdout_month_count": 0,
                "holdout_mae": 1.0,
                "holdout_smape": 1.0,
                "baseline_holdout_mae": 1.0,
                "baseline_holdout_smape": 1.0,
                "holdout_mae_improvement": 0.0,
                "holdout_smape_improvement": 0.0,
                "calibration_score": 0.0,
                "leakage_check_passed": False,
            }
        train_idx, holdout_idx = _time_holdout_columns(month_count, cfg=cfg)
        leakage_check_passed = bool(len(train_idx) > 0 and len(holdout_idx) > 0 and int(train_idx.max()) < int(holdout_idx.min()))
        train_x = surface[:, train_idx]
        holdout_x = surface[:, holdout_idx]
        train_y = target_surface[:, train_idx]
        holdout_y = target_surface[:, holdout_idx]
        pred_holdout = _fit_linear_predictor(train_x, train_y, holdout_x).reshape(holdout_y.shape)
        baseline = np.repeat(train_y[:, [-1]], len(holdout_idx), axis=1)
        train_pred = _fit_linear_predictor(train_x, train_y, train_x).reshape(train_y.shape)
        residual_scale = float(np.std(train_y - train_pred))
        residual_scale = max(residual_scale, 1e-6)
        coverage = float(np.mean(np.abs(holdout_y - pred_holdout) <= residual_scale))
        target_coverage = float(cfg["target_coverage"])
        calibration_score = float(np.clip(1.0 - abs(coverage - target_coverage) / max(target_coverage, 1e-6), 0.0, 1.0))
        model_mae = _mae(holdout_y, pred_holdout)
        baseline_mae = _mae(holdout_y, baseline)
        model_smape = _smape(holdout_y, pred_holdout)
        baseline_smape = _smape(holdout_y, baseline)
        mae_improvement = float(np.clip((baseline_mae - model_mae) / max(baseline_mae, 1e-6), -1.0, 1.0))
        smape_improvement = float(np.clip((baseline_smape - model_smape) / max(baseline_smape, 1e-6), -1.0, 1.0))
        return {
            "holdout_month_count": int(len(holdout_idx)),
            "holdout_mae": round(model_mae, 6),
            "holdout_smape": round(model_smape, 6),
            "baseline_holdout_mae": round(baseline_mae, 6),
            "baseline_holdout_smape": round(baseline_smape, 6),
            "holdout_mae_improvement": round(mae_improvement, 6),
            "holdout_smape_improvement": round(smape_improvement, 6),
            "calibration_score": round(calibration_score, 6),
            "leakage_check_passed": leakage_check_passed,
        }

    def _resampling_instability_penalty(surface: np.ndarray, target_surface: np.ndarray, *, cfg: dict[str, Any]) -> float:
        province_count = int(surface.shape[0]) if surface.ndim == 2 else 0
        if province_count <= 1:
            return 0.0
        draw_count = max(1, int(cfg["resample_draws"]))
        sample_fraction = float(cfg["resample_fraction"])
        draws: list[float] = []
        for _ in range(draw_count):
            sample_size = max(1, int(np.ceil(province_count * sample_fraction)))
            sample_idx = rng.choice(province_count, size=sample_size, replace=sample_size > province_count)
            sample_metrics = _holdout_metrics(surface[sample_idx], target_surface[sample_idx], cfg=cfg)
            draws.append(float(sample_metrics["holdout_mae_improvement"]))
        if not draws:
            return 1.0
        penalty = float(np.clip(np.std(np.asarray(draws, dtype=np.float32)) / max(float(cfg["resample_penalty_scale"]), 1e-6), 0.0, 1.0))
        return penalty

    target_candidates = {
        "diagnosed_stock": observation_targets["diagnosed_stock"],
        "art_stock": observation_targets["art_stock"],
        "documented_suppression": observation_targets["documented_suppression"],
        "testing_coverage": observation_targets["testing_coverage"],
    }
    stability_rows = []
    promotion_pool = []
    rng = np.random.default_rng(17)
    score_weights = dict(stability_cfg["score_weights"])
    tournament_weights = dict(stability_cfg["tournament_weights"])
    penalty_weights = dict(stability_cfg["penalty_weights"])
    for factor_idx, factor in enumerate(factor_catalog):
        surface = factor_tensor[:, :, factor_idx]
        flat = surface.reshape(-1)
        target_scores = {target_name: corr_fn(flat, target_surface.reshape(-1)) for target_name, target_surface in target_candidates.items()}
        subnational_scores = {
            target_name: _subnational_anomaly_score(surface, target_surface, corr_fn=corr_fn)
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
                corr = corr_fn(env_surface.reshape(-1), env_target.reshape(-1))
                env_signs.append(np.sign(corr))
                env_scores.append(abs(corr))
        sign_stability = float(np.mean(np.asarray(env_signs) == np.sign(best_target[1]))) if env_signs else 0.0
        stability = float(np.mean(env_scores)) if env_scores else 0.0
        region_contrast = _region_contrast_score(surface, environment_masks.get("region_family", {}))
        permuted = []
        target_flat = target_candidates[best_target[0]].reshape(-1).copy()
        for _ in range(int(stability_cfg["permutation_draws"])):
            rng.shuffle(target_flat)
            permuted.append(abs(corr_fn(flat, target_flat)))
        null_gap = float(abs(best_target[1]) - np.mean(permuted))
        missing_mask = rng.uniform(size=surface.shape) > float(stability_cfg["missing_dropout_rate"])
        missing_surface = surface * missing_mask
        missing_robustness = float(abs(corr_fn(surface.reshape(-1), missing_surface.reshape(-1))))
        has_non_seed_member = any("seed" not in member for member in mesoscopic_factor_members.get(factor["factor_id"], []))
        source_dropout_robustness = (
            float(stability_cfg["source_dropout_network"])
            if factor.get("factor_class") == "network_feature"
            else (
                float(stability_cfg["source_dropout_non_seed"])
                if has_non_seed_member
                else float(stability_cfg["source_dropout_seed_only"])
            )
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
        member_count = len(mesoscopic_factor_members.get(factor["factor_id"], []))
        finite_check_passed = bool(np.isfinite(surface).all())
        shape_check_passed = bool(surface.ndim == 2 and surface.shape[0] > 0 and surface.shape[1] > 0)
        support_check_passed = bool(member_count >= int(stability_cfg["min_member_count"]) or factor.get("factor_class") == "network_feature")
        holdout_metrics = _holdout_metrics(surface, target_candidates[best_target[0]], cfg=stability_cfg)
        sparsity_penalty = float(
            np.clip(
                max(0, member_count - 1) / max(float(stability_cfg["sparsity_member_denominator"]), 1e-6),
                0.0,
                1.0,
            )
        )
        resampling_penalty = _resampling_instability_penalty(surface, target_candidates[best_target[0]], cfg=stability_cfg)
        survives_holdout = bool(
            float(holdout_metrics["holdout_mae_improvement"]) > 0.0
            or float(holdout_metrics["holdout_smape_improvement"]) > 0.0
        )
        hard_checks_passed = bool(
            finite_check_passed
            and shape_check_passed
            and support_check_passed
            and bool(holdout_metrics["leakage_check_passed"])
        )
        survival_score = float(
            np.clip(
                float(tournament_weights["mae_improvement"]) * max(0.0, float(holdout_metrics["holdout_mae_improvement"]))
                + float(tournament_weights["smape_improvement"]) * max(0.0, float(holdout_metrics["holdout_smape_improvement"]))
                + float(tournament_weights["calibration"]) * float(holdout_metrics["calibration_score"])
                + float(tournament_weights["predictive_gain"]) * predictive_gain
                + float(tournament_weights["subnational_gain"]) * subnational_gain
                + float(tournament_weights["stability"]) * stability
                + float(tournament_weights["sign_stability"]) * sign_stability
                + float(tournament_weights["missing_robustness"]) * missing_robustness
                + float(tournament_weights["source_dropout_robustness"]) * source_dropout_robustness
                + float(tournament_weights["region_contrast"]) * region_contrast
                - float(penalty_weights["sparsity"]) * sparsity_penalty
                - float(penalty_weights["resampling_instability"]) * resampling_penalty,
                0.0,
                1.0,
            )
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
            "finite_check_passed": finite_check_passed,
            "shape_check_passed": shape_check_passed,
            "support_check_passed": support_check_passed,
            "leakage_check_passed": bool(holdout_metrics["leakage_check_passed"]),
            "hard_checks_passed": hard_checks_passed,
            "member_count": member_count,
            "holdout_month_count": int(holdout_metrics["holdout_month_count"]),
            "holdout_mae": float(holdout_metrics["holdout_mae"]),
            "holdout_smape": float(holdout_metrics["holdout_smape"]),
            "baseline_holdout_mae": float(holdout_metrics["baseline_holdout_mae"]),
            "baseline_holdout_smape": float(holdout_metrics["baseline_holdout_smape"]),
            "holdout_mae_improvement": float(holdout_metrics["holdout_mae_improvement"]),
            "holdout_smape_improvement": float(holdout_metrics["holdout_smape_improvement"]),
            "calibration_score": float(holdout_metrics["calibration_score"]),
            "sparsity_penalty": round(sparsity_penalty, 6),
            "resampling_stability_penalty": round(resampling_penalty, 6),
            "survives_holdout": survives_holdout,
            "survival_score": round(survival_score, 6),
            "network_feature_family": factor.get("network_feature_family", ""),
        }
        stability_rows.append(row)
    primary_per_block = max(1, int(stability_cfg["primary_survivors_per_block"]))
    secondary_per_block = max(0, int(stability_cfg["secondary_survivors_per_block"]))
    require_holdout_survival = bool(stability_cfg.get("require_holdout_survival_for_promotion", True))
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        by_block[str(row["block_name"])].append(row)
    row_by_id = {str(row["factor_id"]): row for row in stability_rows}
    for block_rows in by_block.values():
        block_rows.sort(
            key=lambda item: (
                bool(item["hard_checks_passed"]),
                bool(item["survives_holdout"]),
                float(item["survival_score"]),
                float(item["holdout_mae_improvement"]),
                float(item["holdout_smape_improvement"]),
                -float(item["sparsity_penalty"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(block_rows):
            if not bool(row["hard_checks_passed"]):
                survival_class = "discarded"
            elif require_holdout_survival and not bool(row["survives_holdout"]):
                survival_class = "discarded"
            elif rank < primary_per_block:
                survival_class = "survivor_primary"
            elif rank < primary_per_block + secondary_per_block:
                survival_class = "survivor_secondary"
            else:
                survival_class = "reserve"
            row["survival_rank_in_block"] = int(rank + 1)
            row["survival_class"] = survival_class
            promotion_pool.append(
                row
                | {
                    "promotion_class": survival_class,
                    "factor_class": factor_catalog_by_id.get(str(row["factor_id"]), {}).get("factor_class", "mesoscopic_factor"),
                    "transition_hooks": factor_catalog_by_id.get(str(row["factor_id"]), {}).get("transition_hooks", []),
                }
            )
    return stability_rows, promotion_pool
