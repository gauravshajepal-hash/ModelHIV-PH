from __future__ import annotations

from typing import Any, Callable

import numpy as np


def channel_alpha(
    counterfactual_rankings: list[dict[str, Any]],
    *,
    intervention_channels: list[str],
    phase4_setting: Callable[[str, Any], Any],
) -> np.ndarray:
    counter_cfg = dict(phase4_setting("counterfactual", {}) or {})
    alpha = np.ones((len(intervention_channels),), dtype=np.float32) * float(counter_cfg.get("dirichlet_base_alpha", 1.0))
    for row in counterfactual_rankings:
        channel = row["channel"]
        idx = intervention_channels.index(channel)
        alpha[idx] += float(row["counterfactual_score"]) * float(counter_cfg.get("dirichlet_score_scale", 0.0))
    return np.clip(alpha, float(counter_cfg.get("alpha_floor", 0.0)), None)


def counterfactual_rankings(
    inputs: dict[str, Any],
    *,
    intervention_channels: list[str],
    phase4_setting: Callable[[str, Any], Any],
) -> list[dict[str, Any]]:
    counter_cfg = dict(phase4_setting("counterfactual", {}) or {})
    latest_states = np.asarray(inputs["state_estimates"])[:, -1, :]
    mean_state = latest_states.mean(axis=0) if latest_states.size else np.zeros((5,), dtype=np.float32)
    v_gap = max(0.0, float(counter_cfg.get("suppression_gap_target", 0.0)) - float(mean_state[3]))
    burden_context = {
        "U": float(mean_state[0]),
        "D": float(mean_state[1]),
        "A": float(mean_state[2]),
        "L": float(mean_state[4]),
        "V_gap": v_gap,
    }
    blanket_bonus = {
        row.get("canonical_name", ""): float(counter_cfg.get("blanket_bonus", 0.0)) if row.get("blanket_member") else 0.0
        for row in inputs["candidate_profiles"]
    }
    edge_support: dict[str, float] = {}
    for row in inputs["ranked_linkages"]:
        edge_support.setdefault(row["linkage_target"], 0.0)
        edge_support[row["linkage_target"]] += float(row.get("linkage_score", 0.0))
    channel_targets = dict(counter_cfg.get("channel_targets", {}) or {})
    base_scores = {
        channel: sum(float(weight) * float(burden_context.get(name, 0.0)) for name, weight in dict(weights).items())
        for channel, weights in dict(counter_cfg.get("base_scores", {}) or {}).items()
    }
    rankings = []
    for channel in intervention_channels:
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
                        float(phase4_setting("counterfactual_score_ceiling", 1.0)),
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


def uniform_action(intervention_channels: list[str]) -> np.ndarray:
    channel_count = max(len(intervention_channels), 1)
    return (np.ones((channel_count,), dtype=np.float32) / float(channel_count)).astype(np.float32)


def channel_one_hot(intervention_channels: list[str], channel_name: str) -> np.ndarray:
    action = np.zeros((len(intervention_channels),), dtype=np.float32)
    if channel_name in intervention_channels:
        action[intervention_channels.index(channel_name)] = 1.0
    return action


def top_counterfactual_mix(counterfactual_rankings_rows: list[dict[str, Any]], *, intervention_channels: list[str], top_k: int) -> np.ndarray:
    top_rows = list(counterfactual_rankings_rows[:top_k])
    if not top_rows:
        return uniform_action(intervention_channels)
    scores = np.asarray([max(float(row.get("counterfactual_score") or 0.0), 1e-6) for row in top_rows], dtype=np.float32)
    weights = scores / np.clip(scores.sum(), 1e-6, None)
    action = np.zeros((len(intervention_channels),), dtype=np.float32)
    for weight, row in zip(weights, top_rows):
        channel_name = str(row.get("channel") or "")
        if channel_name in intervention_channels:
            action[intervention_channels.index(channel_name)] += float(weight)
    return action.astype(np.float32)


def build_policy_comparator_report(
    *,
    selected_policy: dict[str, Any],
    rollout_context: dict[str, Any],
    counterfactual_rankings_rows: list[dict[str, Any]],
    horizon: int,
    intervention_channels: list[str],
    phase4_setting: Callable[[str, Any], Any],
    evaluate_policy_action: Callable[..., tuple[dict[str, Any], np.ndarray]],
) -> dict[str, Any]:
    analysis_cfg = dict(phase4_setting("analysis", {}) or {})
    selected_action = np.asarray([float(selected_policy.get("action", {}).get(channel, 0.0)) for channel in intervention_channels], dtype=np.float32)
    candidate_actions = [
        ("selected_policy", selected_action),
        ("uniform_policy", uniform_action(intervention_channels)),
    ]
    if counterfactual_rankings_rows:
        best_channel = str(counterfactual_rankings_rows[0].get("channel") or "")
        candidate_actions.append((f"single_channel_{best_channel}", channel_one_hot(intervention_channels, best_channel)))
        candidate_actions.append(
            (
                f"top_{int(analysis_cfg.get('top_counterfactual_channels', 2))}_counterfactual_mix",
                top_counterfactual_mix(
                    counterfactual_rankings_rows,
                    intervention_channels=intervention_channels,
                    top_k=int(analysis_cfg.get("top_counterfactual_channels", 2)),
                ),
            )
        )
    rows = []
    for label, action in candidate_actions:
        row, _ = evaluate_policy_action(policy_id=label, action=action, rollout_context=rollout_context, horizon=horizon, policy_label=label)
        row["comparator"] = label
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("scalar_score") or 0.0), reverse=True)
    selected_row = next((row for row in rows if row["comparator"] == "selected_policy"), rows[0] if rows else {})
    baseline_row = next((row for row in rows if row["comparator"] == "uniform_policy"), {})
    return {
        "available": bool(rows),
        "comparators": rows,
        "selected_policy_scalar_score": float(selected_row.get("scalar_score") or 0.0),
        "uniform_policy_scalar_score": float(baseline_row.get("scalar_score") or 0.0),
        "selected_minus_uniform": round(
            float(selected_row.get("scalar_score") or 0.0) - float(baseline_row.get("scalar_score") or 0.0),
            6,
        ),
        "best_comparator": rows[0]["comparator"] if rows else None,
    }


def shift_action_mass(action: np.ndarray, *, from_index: int, to_index: int, shift: float) -> np.ndarray:
    shifted = np.asarray(action, dtype=np.float32).copy()
    actual_shift = min(float(shift), float(shifted[from_index]))
    shifted[from_index] -= actual_shift
    shifted[to_index] += actual_shift
    shifted = np.clip(shifted, 0.0, None)
    shifted = shifted / np.clip(shifted.sum(), 1e-8, None)
    return shifted.astype(np.float32)


def build_sensitivity_analysis(
    *,
    selected_policy: dict[str, Any],
    rollout_context: dict[str, Any],
    horizon: int,
    intervention_channels: list[str],
    objective_names: list[str],
    phase4_setting: Callable[[str, Any], Any],
    evaluate_policy_action: Callable[..., tuple[dict[str, Any], np.ndarray]],
) -> dict[str, Any]:
    analysis_cfg = dict(phase4_setting("analysis", {}) or {})
    base_action = np.asarray([float(selected_policy.get("action", {}).get(channel, 0.0)) for channel in intervention_channels], dtype=np.float32)
    if base_action.size == 0:
        return {"available": False, "reason": "no_action"}
    base_row, _ = evaluate_policy_action(policy_id="selected_policy", action=base_action, rollout_context=rollout_context, horizon=horizon, policy_label="selected_policy")
    donor_index = int(np.argmax(base_action))
    shift = float(analysis_cfg.get("sensitivity_shift", 0.0))
    perturbations = []
    for target_index, channel_name in enumerate(intervention_channels):
        if target_index == donor_index:
            continue
        perturbed_action = shift_action_mass(base_action, from_index=donor_index, to_index=target_index, shift=shift)
        row, _ = evaluate_policy_action(
            policy_id=f"sensitivity_{intervention_channels[donor_index]}_to_{channel_name}",
            action=perturbed_action,
            rollout_context=rollout_context,
            horizon=horizon,
            policy_label=f"{intervention_channels[donor_index]}->{channel_name}",
        )
        perturbations.append(
            {
                "from_channel": intervention_channels[donor_index],
                "to_channel": channel_name,
                "shift": round(float(np.abs(perturbed_action[target_index] - base_action[target_index])), 6),
                "scalar_score_delta": round(float(row["scalar_score"]) - float(base_row["scalar_score"]), 6),
                "objective_delta": {
                    name: round(float(row["objectives"][name]) - float(base_row["objectives"][name]), 6)
                    for name in objective_names
                },
            }
        )
    scalar_deltas = [abs(float(row["scalar_score_delta"])) for row in perturbations]
    return {
        "available": bool(perturbations),
        "base_policy": {
            "policy_id": str(selected_policy.get("policy_id") or "selected_policy"),
            "donor_channel": intervention_channels[donor_index],
            "scalar_score": float(base_row["scalar_score"]),
        },
        "shift": round(shift, 6),
        "perturbations": perturbations,
        "summary": {
            "perturbation_count": len(perturbations),
            "max_abs_scalar_score_delta": round(max(scalar_deltas) if scalar_deltas else 0.0, 6),
            "mean_abs_scalar_score_delta": round(float(np.mean(scalar_deltas)) if scalar_deltas else 0.0, 6),
        },
    }


def non_dominated(frontier_rows: list[dict[str, Any]], *, objective_names: list[str]) -> list[dict[str, Any]]:
    kept = []
    for idx, row in enumerate(frontier_rows):
        dominated = False
        for jdx, other in enumerate(frontier_rows):
            if idx == jdx:
                continue
            better_or_equal = all(other["objectives"][name] >= row["objectives"][name] for name in objective_names)
            strictly_better = any(other["objectives"][name] > row["objectives"][name] for name in objective_names)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(row)
    kept.sort(key=lambda item: item["scalar_score"], reverse=True)
    return kept

