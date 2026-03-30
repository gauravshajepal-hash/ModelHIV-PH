from __future__ import annotations

import os
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.core.node_graph import build_node_graph_bundle
from epigraph_ph.geography import infer_region_code
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.runtime import (
    RunContext,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    save_tensor_artifact,
    set_global_seed,
    utc_now_iso,
    write_ground_truth_package,
    write_json,
    read_json,
)

try:
    import ray
except Exception:  # pragma: no cover
    ray = None


_HIV_PLUGIN = get_disease_plugin("hiv")
_PHASE4_SETTINGS = dict(_HIV_PLUGIN.policy_settings.get("phase4", {}) or {})
INTERVENTION_CHANNELS = list(_PHASE4_SETTINGS.get("intervention_channels", []))
OBJECTIVE_NAMES = list(_PHASE4_SETTINGS.get("objectives", []))
CHANNEL_EFFECTS = np.asarray(_PHASE4_SETTINGS.get("channel_effects", []), dtype=np.float32)
DEFAULT_THRESHOLDS = np.asarray(_PHASE4_SETTINGS.get("activation_thresholds", []), dtype=np.float32)
DEFAULT_SATURATION = np.asarray(_PHASE4_SETTINGS.get("saturation", []), dtype=np.float32)


def _phase4_setting(key: str, default: Any) -> Any:
    return _PHASE4_SETTINGS.get(key, default)


def _load_phase4_inputs(run_dir) -> dict[str, Any]:
    phase2_dir = run_dir / "phase2"
    phase3_dir = run_dir / "phase3"
    return {
        "candidate_profiles": read_json(phase2_dir / "candidate_profiles.json", default=[]),
        "edge_scores": read_json(phase2_dir / "edge_scores.json", default=[]),
        "ranked_linkages": read_json(phase2_dir / "ranked_linkages.json", default=[]),
        "state_estimates": load_tensor_artifact(phase3_dir / "state_estimates.npz"),
        "forecast_states": load_tensor_artifact(phase3_dir / "forecast_states.npz"),
        "transition_parameters": read_json(phase3_dir / "transition_parameters.json", default={}),
        "validation_artifact": read_json(phase3_dir / "validation_artifact.json", default={}),
        "fit_artifact": read_json(phase3_dir / "fit_artifact.json", default={}),
        "subgroup_weight_summary": read_json(phase3_dir / "subgroup_weight_summary.json", default={}),
        "province_archetypes": read_json(phase3_dir / "province_archetype_mixture.json", default={}),
        "axis_catalogs": read_json(run_dir / "phase1" / "axis_catalogs.json", default={}),
    }


def _channel_alpha(counterfactual_rankings: list[dict[str, Any]]) -> np.ndarray:
    counter_cfg = dict(_phase4_setting("counterfactual", {}) or {})
    alpha = np.ones((len(INTERVENTION_CHANNELS),), dtype=np.float32) * float(counter_cfg.get("dirichlet_base_alpha", 1.0))
    for row in counterfactual_rankings:
        channel = row["channel"]
        idx = INTERVENTION_CHANNELS.index(channel)
        alpha[idx] += float(row["counterfactual_score"]) * float(counter_cfg.get("dirichlet_score_scale", 0.0))
    return np.clip(alpha, float(counter_cfg.get("alpha_floor", 0.0)), None)


def _counterfactual_rankings(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    counter_cfg = dict(_phase4_setting("counterfactual", {}) or {})
    latest_states = np.asarray(inputs["state_estimates"])[:, -1, :]
    mean_state = latest_states.mean(axis=0) if latest_states.size else np.zeros((5,), dtype=np.float32)
    V_gap = max(0.0, float(counter_cfg.get("suppression_gap_target", 0.0)) - float(mean_state[3]))
    U_burden = float(mean_state[0])
    D_burden = float(mean_state[1])
    A_burden = float(mean_state[2])
    L_burden = float(mean_state[4])
    blanket_bonus = {
        row.get("canonical_name", ""): float(counter_cfg.get("blanket_bonus", 0.0)) if row.get("blanket_member") else 0.0
        for row in inputs["candidate_profiles"]
    }
    edge_support = {}
    for row in inputs["ranked_linkages"]:
        edge_support.setdefault(row["linkage_target"], 0.0)
        edge_support[row["linkage_target"]] += float(row.get("linkage_score", 0.0))
    channel_targets = dict(counter_cfg.get("channel_targets", {}) or {})
    burden_context = {"U": U_burden, "D": D_burden, "A": A_burden, "L": L_burden, "V_gap": V_gap}
    base_scores = {
        channel: sum(float(weight) * float(burden_context.get(name, 0.0)) for name, weight in dict(weights).items())
        for channel, weights in dict(counter_cfg.get("base_scores", {}) or {}).items()
    }
    rankings = []
    for channel in INTERVENTION_CHANNELS:
        target_bonus = sum(edge_support.get(target, 0.0) for target in list(channel_targets.get(channel, []) or []))
        domain_bonus = 0.0
        for profile in inputs["candidate_profiles"]:
            if channel.startswith("testing") and "testing" in profile.get("canonical_name", "").lower():
                domain_bonus += float(counter_cfg.get("testing_domain_bonus", 0.0))
            if channel == "art_retention" and profile.get("primary_block") == "biology":
                domain_bonus += float(counter_cfg.get("art_biology_bonus", 0.0))
            domain_bonus += blanket_bonus.get(profile.get("canonical_name", ""), 0.0) * float(counter_cfg.get("blanket_bonus_scale", 0.0))
        rankings.append(
            {
                "channel": channel,
                "counterfactual_score": round(
                    min(
                        float(_phase4_setting("counterfactual_score_ceiling", 1.0)),
                        float(base_scores.get(channel, 0.0))
                        + float(counter_cfg.get("target_bonus_scale", 0.0)) * target_bonus
                        + domain_bonus,
                    ),
                    6,
                ),
                "causal_targets": list(channel_targets.get(channel, []) or []),
                "expected_direction": "improve_cascade",
            }
        )
    rankings.sort(key=lambda item: item["counterfactual_score"], reverse=True)
    return rankings


def _realized_intensity(action: np.ndarray) -> np.ndarray:
    activated = (action >= DEFAULT_THRESHOLDS).astype(np.float32)
    return activated * (1.0 - np.exp(-DEFAULT_SATURATION * action))


def _sample_transition_probabilities(mean_transition: np.ndarray, *, concentration: float, rng: np.random.Generator) -> np.ndarray:
    stochastic_cfg = dict(_phase4_setting("stochastic_control", {}) or {})
    clip_floor = float(stochastic_cfg.get("transition_clip_floor", stochastic_cfg.get("transition_floor", 0.0)))
    clip_ceiling = float(stochastic_cfg.get("transition_clip_ceiling", stochastic_cfg.get("transition_ceiling", 1.0)))
    beta_floor = float(stochastic_cfg.get("beta_floor", 0.0))
    clipped = np.clip(np.asarray(mean_transition, dtype=np.float32), clip_floor, clip_ceiling)
    effective_concentration = max(float(concentration), 1.0)
    alpha = np.clip(clipped * effective_concentration, beta_floor, None)
    beta = np.clip((1.0 - clipped) * effective_concentration, beta_floor, None)
    return rng.beta(alpha, beta).astype(np.float32)


def _simulate_policy(
    latest_states: np.ndarray,
    base_transition: np.ndarray,
    action: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stochastic_cfg = dict(_phase4_setting("stochastic_control", {}) or {})
    trajectory_count = max(1, int(stochastic_cfg.get("trajectory_count", 1)))
    transition_concentration = float(stochastic_cfg.get("transition_concentration", 1.0))
    rng = np.random.default_rng(int(stochastic_cfg.get("rng_seed", 0)))
    intensity = _realized_intensity(action)
    delta = intensity @ CHANNEL_EFFECTS
    rollout_decay = float(_phase4_setting("rollout_decay", 0.0))
    feasibility_variance_weight = float(_phase4_setting("feasibility_variance_weight", 0.0))
    transition_floor = float(stochastic_cfg.get("transition_floor", 0.0))
    transition_ceiling = float(stochastic_cfg.get("transition_ceiling", 1.0))
    trajectory_states = []
    trajectory_objectives = []
    for _ in range(trajectory_count):
        state = latest_states.astype(np.float32).copy()
        rollout_states = []
        rollout_objectives = []
        for step in range(horizon):
            decay = np.exp(-rollout_decay * step)
            adjusted_mean = np.clip(base_transition + delta * decay, transition_floor, transition_ceiling)
            adjusted = _sample_transition_probabilities(adjusted_mean, concentration=transition_concentration, rng=rng)
            p_ud, p_da, p_av, p_al, p_la = [adjusted[:, idx] for idx in range(adjusted.shape[-1])]
            U = state[:, 0]
            D = state[:, 1]
            A = state[:, 2]
            V = state[:, 3]
            L = state[:, 4]
            flow_ud = U * p_ud
            flow_da = D * p_da
            flow_av = A * p_av
            flow_al = A * p_al
            flow_la = L * p_la
            state = np.stack(
                [
                    U - flow_ud,
                    D + flow_ud - flow_da,
                    A + flow_da + flow_la - flow_av - flow_al,
                    V + flow_av,
                    L + flow_al - flow_la,
                ],
                axis=-1,
            )
            state = np.clip(state, 0.0, None)
            state = state / np.clip(state.sum(axis=-1, keepdims=True), 1e-6, None)
            suppression_reward = float(state[:, 3].mean())
            inequality_reward = float(-np.var(state[:, 3]))
            incidence_reward = float(-state[:, 0].mean())
            feasibility_reward = float(-np.mean(np.maximum(0.0, DEFAULT_THRESHOLDS - action)) - feasibility_variance_weight * np.var(action))
            rollout_states.append(state.copy())
            rollout_objectives.append([suppression_reward, inequality_reward, incidence_reward, feasibility_reward])
        trajectory_states.append(np.asarray(rollout_states, dtype=np.float32))
        trajectory_objectives.append(np.asarray(rollout_objectives, dtype=np.float32))
    state_tensor = np.asarray(trajectory_states, dtype=np.float32)
    objective_tensor = np.asarray(trajectory_objectives, dtype=np.float32)
    return (
        state_tensor.mean(axis=0),
        objective_tensor.mean(axis=0),
        {
            "trajectory_count": trajectory_count,
            "transition_concentration": round(float(transition_concentration), 6),
            "final_objective_std": np.round(objective_tensor[:, -1, :].std(axis=0), 6).tolist() if objective_tensor.size else [0.0 for _ in OBJECTIVE_NAMES],
        },
    )


def _non_dominated(frontier_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for idx, row in enumerate(frontier_rows):
        dominated = False
        for jdx, other in enumerate(frontier_rows):
            if idx == jdx:
                continue
            better_or_equal = all(other["objectives"][name] >= row["objectives"][name] for name in OBJECTIVE_NAMES)
            strictly_better = any(other["objectives"][name] > row["objectives"][name] for name in OBJECTIVE_NAMES)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(row)
    kept.sort(key=lambda item: item["scalar_score"], reverse=True)
    return kept


def _node_graph_enabled() -> bool:
    return str(os.getenv("EPIGRAPH_ENABLE_NODE_GRAPH", "")).strip().lower() in {"1", "true", "yes", "on"}


def _write_node_graph_report(phase4_dir, bundle: dict[str, Any], *, enabled: bool) -> None:
    lines = [
        "# Phase 4 Node Graph Report",
        "",
        f"- Node graph enabled: `{str(enabled).lower()}`",
        f"- Region count: `{bundle.get('debug_summary', {}).get('region_count', 0)}`",
        f"- Node count: `{bundle.get('debug_summary', {}).get('node_count', 0)}`",
        "",
        "## Block Evidence",
        "",
    ]
    block_evidence = bundle.get("block_evidence", {})
    for block_id, row in sorted(block_evidence.items()):
        lines.append(
            f"- `{block_id}` prior={row.get('prior_reliability')} posterior={row.get('posterior_reliability')} truth_gain={row.get('truth_gain')} stability={row.get('stability')}"
        )
    lines.extend(["", "## Region Signals", ""])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in bundle.get("region_node_states", []):
        grouped.setdefault(str(row.get("region_id") or ""), []).append(row)
    for region_id in sorted(grouped):
        penalties = bundle.get("risk_penalty_by_region", {}).get(region_id, 0.0)
        bonuses = bundle.get("risk_bonus_by_region", {}).get(region_id, 0.0)
        veto = bundle.get("risk_veto_flag_by_region", {}).get(region_id, False)
        lines.append(f"### {region_id}")
        lines.append(f"- bonus={bonuses} penalty={penalties} veto={str(veto).lower()}")
        for row in sorted(grouped[region_id], key=lambda item: abs(float(item.get("raw_signal") or 0.0)), reverse=True)[:4]:
            lines.append(
                f"- `{row.get('block_id')}` signal={row.get('raw_signal')} weight={row.get('final_node_weight')} direction={row.get('signal_direction')}"
            )
        lines.append("")
    (phase4_dir / "node_graph_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _apply_node_graph_adjustment(policy_frontier: list[dict[str, Any]], bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not policy_frontier:
        return policy_frontier, {"enabled": False, "penalty_mean": 0.0, "support_mean": 0.0}
    node_cfg = dict(_phase4_setting("node_graph", {}) or {})
    penalty_mean = float(np.mean(list((bundle.get("decision_penalty_by_region") or {}).values()) or [0.0]))
    support_mean = float(np.mean(list((bundle.get("decision_support_by_region") or {}).values()) or [0.0]))
    veto_count = int(sum(1 for flag in (bundle.get("risk_veto_flag_by_region") or {}).values() if flag))
    multiplier = max(
        float(node_cfg.get("frontier_multiplier_floor", 0.0)),
        1.0
        + float(node_cfg.get("frontier_support_multiplier", 0.0)) * support_mean
        - float(node_cfg.get("frontier_penalty_multiplier", 0.0)) * penalty_mean
        - float(node_cfg.get("frontier_veto_penalty", 0.0)) * veto_count,
    )
    adjusted = []
    for row in policy_frontier:
        patched = dict(row)
        patched["node_graph_scalar_multiplier"] = round(multiplier, 6)
        patched["scalar_score"] = round(float(row.get("scalar_score") or 0.0) * multiplier, 6)
        adjusted.append(patched)
    adjusted.sort(key=lambda item: item["scalar_score"], reverse=True)
    return adjusted, {
        "enabled": True,
        "penalty_mean": round(penalty_mean, 6),
        "support_mean": round(support_mean, 6),
        "veto_count": veto_count,
        "scalar_multiplier": round(multiplier, 6),
    }


def _province_region_map(inputs: dict[str, Any]) -> dict[str, str]:
    subgroup_rows = inputs.get("subgroup_weight_summary", {}).get("rows", [])
    mapping = {
        str(row.get("province") or ""): str(row.get("region") or "")
        for row in subgroup_rows
        if row.get("province") and row.get("region")
    }
    province_axis = list(inputs.get("fit_artifact", {}).get("axis_catalogs", {}).get("province", []))
    if not province_axis:
        province_axis = list(inputs.get("axis_catalogs", {}).get("province", []))
    for province in province_axis:
        mapping.setdefault(str(province), infer_region_code(str(province)) or "region_unknown")
    return mapping


def _node_channel_surface(bundle: dict[str, Any]) -> dict[str, dict[str, float]]:
    node_cfg = dict(_phase4_setting("node_graph", {}) or {})
    channel_map = dict(node_cfg.get("channel_map", {}) or {})
    out: dict[str, dict[str, float]] = {}
    for row in bundle.get("region_node_states", []):
        region = str(row.get("region_id") or "")
        block_id = str(row.get("block_id") or "")
        if not region or block_id not in channel_map:
            continue
        signal = float(row.get("raw_signal") or 0.0)
        weight = float(row.get("final_node_weight") or 0.0)
        region_store = out.setdefault(region, {channel: 0.0 for channel in INTERVENTION_CHANNELS})
        for channel, factor in channel_map[block_id].items():
            region_store[channel] += signal * weight * factor
    return {region: {channel: round(float(value), 6) for channel, value in values.items()} for region, values in out.items()}


def _regional_priority_surface(inputs: dict[str, Any], bundle: dict[str, Any], *, node_graph_active: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_cfg = dict(_phase4_setting("regional_priority_weights", {}) or {})
    node_cfg = dict(_phase4_setting("node_graph", {}) or {})
    province_axis = list(inputs.get("fit_artifact", {}).get("axis_catalogs", {}).get("province", []))
    if not province_axis:
        province_axis = list(inputs.get("axis_catalogs", {}).get("province", []))
    if not province_axis and np.asarray(inputs.get("state_estimates")).ndim >= 1:
        province_axis = [f"province_{idx:03d}" for idx in range(int(np.asarray(inputs["state_estimates"]).shape[0]))]
    latest_states = np.asarray(inputs["state_estimates"])[:, -1, :] if inputs["state_estimates"].size else np.zeros((len(province_axis), 5), dtype=np.float32)
    province_to_region = _province_region_map(inputs)
    by_region: dict[str, dict[str, Any]] = {}
    for idx, province in enumerate(province_axis):
        region = province_to_region.get(str(province), "region_unknown")
        store = by_region.setdefault(region, {"states": [], "provinces": []})
        store["states"].append(latest_states[idx])
        store["provinces"].append(str(province))
    validation_rows = {
        str(row.get("region") or ""): row
        for row in inputs.get("validation_artifact", {}).get("regional_metrics", [])
        if row.get("region")
    }
    archetype_region_mix = dict(inputs.get("province_archetypes", {}).get("region_archetype_mixture", {}) or {})
    node_channel_surface = _node_channel_surface(bundle)
    region_rows = []
    for region, payload in sorted(by_region.items()):
        if region == "national":
            continue
        state_matrix = np.asarray(payload["states"], dtype=np.float32)
        state_mean = state_matrix.mean(axis=0) if state_matrix.size else np.zeros((5,), dtype=np.float32)
        U, D, A, V, L = [float(state_mean[idx]) for idx in range(5)]
        validation = validation_rows.get(region, {})
        diagnosed_mae = float(validation.get("diagnosed_mae") or 0.0)
        art_mae = float(validation.get("art_mae") or 0.0)
        suppression_mae = float(validation.get("suppression_mae") or 0.0)
        archetype_mix = archetype_region_mix.get(region, {}) or {}
        urban_mix = float(archetype_mix.get("urban_high_throughput", 0.0))
        migrant_mix = float(archetype_mix.get("migrant_corridor", 0.0))
        remote_mix = float(archetype_mix.get("remote_island", 0.0))
        fragile_mix = float(archetype_mix.get("fragile_service_network", 0.0))
        under_reporting_mix = float(archetype_mix.get("under_reporting_province", 0.0))
        base_scores = {}
        for channel in INTERVENTION_CHANNELS:
            weights = dict(policy_cfg.get(channel, {}) or {})
            score = 0.0
            score += float(weights.get("U", 0.0)) * U
            score += float(weights.get("D", 0.0)) * D
            score += float(weights.get("A", 0.0)) * A
            score += float(weights.get("L", 0.0)) * L
            score += float(weights.get("A_plus_L", 0.0)) * (A + L)
            score += float(weights.get("diagnosed_mae", 0.0)) * diagnosed_mae
            score += float(weights.get("art_mae", 0.0)) * art_mae
            score += float(weights.get("suppression_mae", 0.0)) * suppression_mae
            score += float(weights.get("under_reporting", 0.0)) * under_reporting_mix
            score += float(weights.get("urban", 0.0)) * urban_mix
            score += float(weights.get("migrant", 0.0)) * migrant_mix
            score += float(weights.get("remote", 0.0)) * remote_mix
            score += float(weights.get("fragile", 0.0)) * fragile_mix
            base_scores[channel] = max(0.0, score)
        node_bonus = float(bundle.get("risk_bonus_by_region", {}).get(region, 0.0))
        node_penalty = float(bundle.get("risk_penalty_by_region", {}).get(region, 0.0))
        node_veto = bool(bundle.get("risk_veto_flag_by_region", {}).get(region, False)) if node_graph_active else False
        decision_penalty = float(bundle.get("decision_penalty_by_region", {}).get(region, 0.0))
        decision_support = float(bundle.get("decision_support_by_region", {}).get(region, 0.0))
        node_multiplier = (
            max(
                float(node_cfg.get("decision_multiplier_floor", 0.0)),
                1.0
                + float(node_cfg.get("decision_support_multiplier", 0.0)) * decision_support
                - float(node_cfg.get("decision_penalty_multiplier", 0.0)) * decision_penalty,
            )
            if node_graph_active
            else 1.0
        )
        channel_adjustments = node_channel_surface.get(region, {})
        adjusted_scores = {}
        for channel in INTERVENTION_CHANNELS:
            score = base_scores[channel]
            if node_graph_active:
                score = max(
                    0.0,
                    score
                    + float(node_cfg.get("policy_score_bonus_scale", 0.0)) * node_bonus
                    - float(node_cfg.get("policy_score_penalty_scale", 0.0)) * node_penalty
                    + float(channel_adjustments.get(channel, 0.0)),
                )
                score = 0.0 if node_veto else score * node_multiplier
            adjusted_scores[channel] = round(float(score), 6)
        region_rows.append(
            {
                "region": region,
                "province_count": len(payload["provinces"]),
                "latest_state": {state: round(float(state_mean[idx]), 6) for idx, state in enumerate(["U", "D", "A", "V", "L"])},
                "validation": {
                    "diagnosed_mae": round(diagnosed_mae, 6),
                    "art_mae": round(art_mae, 6),
                    "suppression_mae": round(suppression_mae, 6),
                },
                "archetype_mixture": {key: round(float(value), 6) for key, value in archetype_mix.items()},
                "base_channel_scores": {channel: round(float(score), 6) for channel, score in base_scores.items()},
                "adjusted_channel_scores": adjusted_scores,
                "node_graph": {
                    "active": node_graph_active,
                    "bonus": round(node_bonus, 6),
                    "penalty": round(node_penalty, 6),
                    "decision_penalty": round(decision_penalty, 6),
                    "decision_support": round(decision_support, 6),
                    "multiplier": round(node_multiplier, 6),
                    "vetoed": node_veto,
                    "channel_adjustments": {channel: round(float(channel_adjustments.get(channel, 0.0)), 6) for channel in INTERVENTION_CHANNELS},
                },
            }
        )
    summary = {
        "region_count": len(region_rows),
        "eligible_region_count": sum(1 for row in region_rows if not row["node_graph"]["vetoed"]),
        "vetoed_region_count": sum(1 for row in region_rows if row["node_graph"]["vetoed"]),
        "node_graph_active": node_graph_active,
    }
    return region_rows, summary


def _regional_allocation_surface(
    *,
    selected_policy: dict[str, Any],
    region_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    channel_totals = {channel: float(selected_policy.get("action", {}).get(channel, 0.0)) for channel in INTERVENTION_CHANNELS}
    allocation_cfg = dict(_phase4_setting("regional_allocation", {}) or {})
    allocation_rows = []
    for channel in INTERVENTION_CHANNELS:
        allocatable_rows = [row for row in region_rows if str(row.get("region") or "") != "national"]
        scores = np.asarray([float(row["adjusted_channel_scores"].get(channel, 0.0)) for row in allocatable_rows], dtype=np.float32)
        eligible = np.asarray([0.0 if bool(row["node_graph"]["vetoed"]) else 1.0 for row in allocatable_rows], dtype=np.float32)
        scores = scores * eligible
        if scores.size == 0:
            continue
        total = float(scores.sum())
        if total <= float(allocation_cfg.get("zero_budget_eps", 0.0)):
            eligible_total = float(eligible.sum())
            weights = (eligible / eligible_total) if eligible_total > float(allocation_cfg.get("zero_budget_eps", 0.0)) else np.zeros_like(scores)
        else:
            weights = scores / total
        for row, weight in zip(allocatable_rows, weights, strict=True):
            allocation_rows.append(
                {
                    "region": row["region"],
                    "channel": channel,
                    "channel_budget": round(channel_totals[channel], 6),
                    "regional_weight": round(float(weight), 6),
                    "regional_allocation": round(float(channel_totals[channel] * weight), 6),
                    "eligible": not bool(row["node_graph"]["vetoed"]),
                }
            )
    by_region: dict[str, dict[str, float]] = {}
    for row in allocation_rows:
        region_store = by_region.setdefault(row["region"], {})
        region_store[row["channel"]] = row["regional_allocation"]
    return {
        "rows": allocation_rows,
        "region_allocations": {
            region: {channel: round(float(values.get(channel, 0.0)), 6) for channel in INTERVENTION_CHANNELS}
            for region, values in sorted(by_region.items())
        },
        "notes": [
            "Global channel simplex is selected first; regional allocation is applied second.",
            "Node graph influences allocation eligibility and channel-specific regional weights only.",
            "National pseudo-regions are diagnostic only and are excluded from allocatable regional budgets.",
        ],
    }


def _policy_frontier(inputs: dict[str, Any], counterfactual_rankings: list[dict[str, Any]], *, rollouts: int | None = None, horizon: int | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    search_cfg = dict(_phase4_setting("policy_search", {}) or {})
    rollouts = int(rollouts if rollouts is not None else search_cfg.get("rollouts", 1))
    horizon = int(horizon if horizon is not None else search_cfg.get("horizon", 1))
    rng = np.random.default_rng(int(search_cfg.get("rng_seed", 0)))
    latest_states = np.asarray(inputs["state_estimates"])[:, -1, :]
    base_transition = np.asarray(inputs["forecast_states"])[:, 0, :]
    map_cfg = dict(_phase4_setting("base_transition_map", {}) or {})
    mapped = []
    for transition_name in ["U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A"]:
        row_cfg = dict(map_cfg.get(transition_name, {}) or {})
        intercept = float(row_cfg.get("intercept", 0.0))
        state_index = int(row_cfg.get("state_index", 0))
        slope = float(row_cfg.get("slope", 0.0))
        floor = float(row_cfg.get("floor", 0.0))
        ceiling = float(row_cfg.get("ceiling", 1.0))
        mapped.append(np.clip(intercept + slope * base_transition[:, state_index], floor, ceiling))
    base_transition = np.stack(mapped, axis=-1).astype(np.float32)
    alpha = _channel_alpha(counterfactual_rankings)
    policies = rng.dirichlet(alpha, size=rollouts).astype(np.float32)
    scalar_weights = np.asarray(_phase4_setting("scalar_objective_weights", []), dtype=np.float32)
    frontier_rows = []
    objective_tensor = []
    for policy_id, action in enumerate(policies):
        rollout_states, rollout_objectives, stochastic_summary = _simulate_policy(latest_states, base_transition, action, horizon=horizon)
        final_objective = rollout_objectives[-1]
        scalar_score = float(np.dot(final_objective, scalar_weights))
        frontier_rows.append(
            {
                "policy_id": f"policy_{policy_id:04d}",
                "action": {INTERVENTION_CHANNELS[idx]: round(float(action[idx]), 6) for idx in range(len(INTERVENTION_CHANNELS))},
                "intensity": {INTERVENTION_CHANNELS[idx]: round(float(_realized_intensity(action)[idx]), 6) for idx in range(len(INTERVENTION_CHANNELS))},
                "objectives": {OBJECTIVE_NAMES[idx]: round(float(final_objective[idx]), 6) for idx in range(len(OBJECTIVE_NAMES))},
                "stochastic_summary": stochastic_summary,
                "scalar_score": round(scalar_score, 6),
                "horizon": horizon,
            }
        )
        objective_tensor.append(rollout_objectives)
    return _non_dominated(frontier_rows), np.asarray(objective_tensor, dtype=np.float32)


def _run_phase4(run_id: str, plugin_id: str, *, mode: str, profile: str = "legacy") -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase4_dir = ensure_dir(ctx.run_dir / "phase4")
    set_global_seed(int(dict(_phase4_setting("policy_search", {}) or {}).get("rng_seed", 23)))
    fit_artifact = read_json(ctx.run_dir / "phase3" / "fit_artifact.json", default={})
    if profile in {"hiv_rescue_v1", "hiv_rescue_v2"} and not bool(fit_artifact.get("phase4_ready")):
        blocked = {
            "profile_id": profile,
            "mode": mode,
            "phase4_ready": False,
            "reason": "phase3 rescue core has not cleared phase4 readiness gates",
            "blocking_gates": read_json(ctx.run_dir / "phase3" / "benchmark_gate_report.json", default={}).get("primary_gates", []),
        }
        write_json(phase4_dir / "phase4_blocked.json", blocked)
        node_bundle = build_node_graph_bundle(run_dir=ctx.run_dir, plugin_id=plugin_id).to_dict()
        write_json(phase4_dir / "node_graph_bundle.json", node_bundle)
        write_json(phase4_dir / "node_graph_adjustment.json", {"enabled": False, "reason": "phase4_blocked"})
        _write_node_graph_report(phase4_dir, node_bundle, enabled=False)
        backend_map = detect_backends()
        manifest = Phase0ManifestArtifact(
            plugin_id=plugin_id,
            run_id=run_id,
            generated_at=utc_now_iso(),
            raw_dir=str(ctx.run_dir / "phase0" / "raw"),
            parsed_dir=str(ctx.run_dir / "phase3"),
            extracted_dir=str(phase4_dir),
            index_dir=str(ctx.run_dir / "phase0" / "index"),
            stage_status={"phase4": "blocked"},
            artifact_paths={
                "phase4_blocked": str(phase4_dir / "phase4_blocked.json"),
                "node_graph_bundle": str(phase4_dir / "node_graph_bundle.json"),
                "node_graph_adjustment": str(phase4_dir / "node_graph_adjustment.json"),
                "node_graph_report": str(phase4_dir / "node_graph_report.md"),
            },
            backend_status={
                "torch": Phase0BackendStatus("torch", backend_map["torch"].available, False, notes=backend_map["torch"].device),
                "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
                "ray": Phase0BackendStatus("ray", ray is not None, False, notes="blocked_in_rescue_profile"),
            },
            notes=["phase4_blocked:rescue_profile_not_ready"],
        ).to_dict()
        truth_paths = write_ground_truth_package(
            phase_dir=phase4_dir,
            phase_name="phase4",
            profile_id=profile,
            checks=[
                {"name": "phase4_blocked_visible", "passed": True},
                {"name": "phase4_ready_false", "passed": blocked.get("phase4_ready") is False},
                {"name": "blocking_gates_present", "passed": bool(blocked.get("blocking_gates", []))},
                {"name": "node_graph_bundle_present", "passed": bool(node_bundle.get("region_node_states", []))},
            ],
            truth_sources=["benchmark_truth", "prior_truth"],
            stage_manifest_path=str(phase4_dir / "phase4_manifest.json"),
            summary={"mode": mode, "blocked_gate_count": len(blocked.get("blocking_gates", []))},
        )
        manifest["artifact_paths"].update(truth_paths)
        write_json(phase4_dir / "phase4_manifest.json", manifest)
        ctx.record_stage_outputs(
            f"phase4_{mode}",
            [
                phase4_dir / "phase4_blocked.json",
                phase4_dir / "node_graph_bundle.json",
                phase4_dir / "node_graph_adjustment.json",
                phase4_dir / "node_graph_report.md",
                phase4_dir / "phase4_manifest.json",
            ],
        )
        return manifest
    inputs = _load_phase4_inputs(ctx.run_dir)
    counterfactual_rankings = _counterfactual_rankings(inputs)
    policy_frontier, rollout_tensor = _policy_frontier(inputs, counterfactual_rankings)
    node_bundle = build_node_graph_bundle(run_dir=ctx.run_dir, plugin_id=plugin_id).to_dict()
    node_graph_active = _node_graph_enabled()
    selected_policy = policy_frontier[0] if policy_frontier else {
        "policy_id": "policy_empty",
        "action": {channel: round(1.0 / len(INTERVENTION_CHANNELS), 6) for channel in INTERVENTION_CHANNELS},
        "intensity": {channel: round(float(_realized_intensity(np.ones((len(INTERVENTION_CHANNELS),), dtype=np.float32) / len(INTERVENTION_CHANNELS))[idx]), 6) for idx, channel in enumerate(INTERVENTION_CHANNELS)},
        "objectives": {name: 0.0 for name in OBJECTIVE_NAMES},
        "scalar_score": 0.0,
        "horizon": 0,
    }
    regional_risk_scores, regional_risk_summary = _regional_priority_surface(inputs, node_bundle, node_graph_active=node_graph_active)
    regional_allocation_surface = _regional_allocation_surface(selected_policy=selected_policy, region_rows=regional_risk_scores)
    node_adjustment = {
        "enabled": node_graph_active,
        "reason": "regional_allocation_seam" if node_graph_active else "disabled",
        **regional_risk_summary,
    }
    selected_policy = dict(selected_policy)
    selected_policy["regional_allocations"] = regional_allocation_surface["region_allocations"]
    mpc_plan = {
        "control_horizon": int(dict(_phase4_setting("mpc", {}) or {}).get("control_horizon", 2)),
        "execution_steps": [
            {
                "step": 1,
                "policy_id": selected_policy["policy_id"],
                "action": selected_policy["action"],
                "regional_allocations": selected_policy["regional_allocations"],
            },
            {
                "step": 2,
                "policy_id": selected_policy["policy_id"],
                "action": selected_policy["action"],
                "regional_allocations": selected_policy["regional_allocations"],
            },
        ],
        "replan_step": int(dict(_phase4_setting("mpc", {}) or {}).get("replan_step", 3)),
        "notes": ["phase4_mpc:commit_first_two_steps_only"],
    }
    rollout_diagnostics = {
        "mode": mode,
        "policy_count": len(policy_frontier),
        "raw_rollout_count": int(rollout_tensor.shape[0]) if rollout_tensor.size else 0,
        "trajectory_count": int((_phase4_setting("stochastic_control", {}) or {}).get("trajectory_count", 0)),
        "transition_concentration": float((_phase4_setting("stochastic_control", {}) or {}).get("transition_concentration", 0.0)),
        "objective_names": OBJECTIVE_NAMES,
        "intervention_channels": INTERVENTION_CHANNELS,
        "phase3_claim_eligible": bool(inputs["validation_artifact"].get("claim_eligible")),
        "ray_available": ray is not None,
        "execution_backend": "ray" if ray is not None else "local_numpy",
    }
    rollout_artifact = save_tensor_artifact(
        array=rollout_tensor,
        axis_names=["policy", "horizon_step", "objective"],
        artifact_dir=phase4_dir,
        stem="rollout_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase4_rollout_objectives"],
        save_pt=False,
    )
    write_json(phase4_dir / "counterfactual_rankings.json", counterfactual_rankings)
    write_json(phase4_dir / "policy_frontier.json", policy_frontier)
    write_json(phase4_dir / "selected_policy.json", selected_policy)
    write_json(phase4_dir / "mpc_plan.json", mpc_plan)
    write_json(phase4_dir / "rollout_diagnostics.json", rollout_diagnostics)
    write_json(phase4_dir / "node_graph_bundle.json", node_bundle)
    write_json(phase4_dir / "node_graph_adjustment.json", node_adjustment)
    write_json(phase4_dir / "regional_risk_scores.json", {"rows": regional_risk_scores, "summary": regional_risk_summary})
    write_json(phase4_dir / "regional_allocation_surface.json", regional_allocation_surface)
    write_json(
        phase4_dir / "regional_selected_policy.json",
        {
            "policy_id": selected_policy["policy_id"],
            "action": selected_policy["action"],
            "regional_allocations": selected_policy["regional_allocations"],
            "node_graph_active": node_graph_active,
        },
    )
    _write_node_graph_report(phase4_dir, node_bundle, enabled=bool(node_adjustment.get("enabled")))

    backend_map = detect_backends()
    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase3"),
        extracted_dir=str(phase4_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase4": "completed"},
        artifact_paths={
            "counterfactual_rankings": str(phase4_dir / "counterfactual_rankings.json"),
            "policy_frontier": str(phase4_dir / "policy_frontier.json"),
            "selected_policy": str(phase4_dir / "selected_policy.json"),
            "mpc_plan": str(phase4_dir / "mpc_plan.json"),
            "rollout_diagnostics": str(phase4_dir / "rollout_diagnostics.json"),
            "rollout_tensor": rollout_artifact["value_path"],
            "node_graph_bundle": str(phase4_dir / "node_graph_bundle.json"),
            "node_graph_adjustment": str(phase4_dir / "node_graph_adjustment.json"),
            "regional_risk_scores": str(phase4_dir / "regional_risk_scores.json"),
            "regional_allocation_surface": str(phase4_dir / "regional_allocation_surface.json"),
            "regional_selected_policy": str(phase4_dir / "regional_selected_policy.json"),
            "node_graph_report": str(phase4_dir / "node_graph_report.md"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, False, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
            "ray": Phase0BackendStatus("ray", ray is not None, ray is not None, notes="local_rollout_orchestrator" if ray is not None else "fallback_local_numpy"),
        },
        source_count=len(counterfactual_rankings),
        canonical_candidate_count=len(policy_frontier),
        notes=["phase4_policy_engine:dirichlet_counterfactual_mpc"],
    ).to_dict()
    truth_paths = write_ground_truth_package(
        phase_dir=phase4_dir,
        phase_name="phase4",
        profile_id=profile,
        checks=[
            {"name": "counterfactual_rankings_present", "passed": bool(counterfactual_rankings)},
            {"name": "policy_frontier_present", "passed": bool(policy_frontier)},
            {"name": "rollout_tensor_finite", "passed": bool(np.isfinite(rollout_tensor).all())},
            {
                "name": "selected_policy_is_simplex",
                "passed": abs(sum(float(value) for value in selected_policy.get("action", {}).values()) - 1.0) <= 1e-4,
            },
            {"name": "node_graph_bundle_present", "passed": bool(node_bundle.get("region_node_states", []))},
            {"name": "regional_risk_scores_present", "passed": bool(regional_risk_scores)},
            {"name": "regional_allocations_present", "passed": bool(regional_allocation_surface.get("rows", []))},
        ],
        truth_sources=["benchmark_truth", "prior_truth", "synthetic_truth"],
        stage_manifest_path=str(phase4_dir / "phase4_manifest.json"),
        summary={
            "mode": mode,
            "counterfactual_count": len(counterfactual_rankings),
            "frontier_count": len(policy_frontier),
            "region_count": regional_risk_summary["region_count"],
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase4_dir / "phase4_manifest.json", manifest)
    ctx.record_stage_outputs(
        f"phase4_{mode}",
        [
            phase4_dir / "counterfactual_rankings.json",
            phase4_dir / "policy_frontier.json",
            phase4_dir / "selected_policy.json",
            phase4_dir / "mpc_plan.json",
            phase4_dir / "rollout_diagnostics.json",
            phase4_dir / "rollout_tensor.npz",
            phase4_dir / "node_graph_bundle.json",
            phase4_dir / "node_graph_adjustment.json",
            phase4_dir / "regional_risk_scores.json",
            phase4_dir / "regional_allocation_surface.json",
            phase4_dir / "regional_selected_policy.json",
            phase4_dir / "node_graph_report.md",
            phase4_dir / "phase4_manifest.json",
        ],
    )
    return manifest


def run_phase4_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    return _run_phase4(run_id, plugin_id, mode="build", profile=profile)


def run_phase4_simulate(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    return _run_phase4(run_id, plugin_id, mode="simulate", profile=profile)


def run_phase4_optimize(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    return _run_phase4(run_id, plugin_id, mode="optimize", profile=profile)
