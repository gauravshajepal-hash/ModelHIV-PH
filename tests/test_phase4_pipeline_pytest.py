from __future__ import annotations

import math

import numpy as np

from epigraph_ph.core.node_graph import build_node_graph_bundle
from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase4.pipeline import _regional_allocation_surface, _simulate_policy
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase4_legacy_frontier_contract(legacy_full_run_dir) -> None:
    manifest = read_json(legacy_full_run_dir / "phase4" / "phase4_manifest.json", default={})
    truth_summary = read_json(legacy_full_run_dir / "phase4" / "ground_truth_summary.json", default={})
    rankings = read_json(legacy_full_run_dir / "phase4" / "counterfactual_rankings.json", default=[])
    frontier = read_json(legacy_full_run_dir / "phase4" / "policy_frontier.json", default=[])
    selected = read_json(legacy_full_run_dir / "phase4" / "selected_policy.json", default={})
    regional_risk = read_json(legacy_full_run_dir / "phase4" / "regional_risk_scores.json", default={})
    regional_surface = read_json(legacy_full_run_dir / "phase4" / "regional_allocation_surface.json", default={})
    regional_selected = read_json(legacy_full_run_dir / "phase4" / "regional_selected_policy.json", default={})
    mpc_plan = read_json(legacy_full_run_dir / "phase4" / "mpc_plan.json", default={})
    node_bundle = read_json(legacy_full_run_dir / "phase4" / "node_graph_bundle.json", default={})
    node_adjustment = read_json(legacy_full_run_dir / "phase4" / "node_graph_adjustment.json", default={})
    rollout_diagnostics = read_json(legacy_full_run_dir / "phase4" / "rollout_diagnostics.json", default={})
    rollout = load_tensor_artifact(legacy_full_run_dir / "phase4" / "rollout_tensor.npz")

    assert manifest.get("stage_status", {}).get("phase4") == "completed"
    assert rankings
    assert frontier
    assert selected.get("action", {})
    assert mpc_plan.get("execution_steps", [])
    assert np.isfinite(rollout).all()
    action = selected["action"]
    assert math.isclose(sum(float(value) for value in action.values()), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    ranking_scores = [row["counterfactual_score"] for row in rankings]
    assert ranking_scores == sorted(ranking_scores, reverse=True)
    assert truth_summary.get("phase_name") == "phase4"
    assert node_bundle.get("region_node_states", [])
    assert regional_risk.get("rows", [])
    assert regional_surface.get("rows", [])
    assert regional_selected.get("regional_allocations", {})
    assert node_adjustment.get("enabled") is False
    assert int(rollout_diagnostics.get("trajectory_count", 0)) >= 1


def test_phase4_policy_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase4_settings = (plugin.policy_settings or {}).get("phase4", {})
    node_graph_settings = (plugin.constraint_settings or {}).get("node_graph", {})
    archetype_settings = (plugin.prior_hyperparameters or {}).get("province_archetypes", {})

    assert phase4_settings.get("channel_effects")
    assert len(phase4_settings.get("activation_thresholds", [])) == len(phase4_settings.get("intervention_channels", []))
    assert len(phase4_settings.get("saturation", [])) == len(phase4_settings.get("intervention_channels", []))
    assert phase4_settings.get("node_graph", {}).get("decision_multiplier_floor") is not None
    assert phase4_settings.get("regional_allocation", {}).get("zero_budget_eps") is not None
    assert node_graph_settings.get("confidence_mix", {}).get("posterior_weight") is not None
    assert archetype_settings.get("default_feature_value") is not None


def test_node_graph_bundle_has_one_state_per_region_per_enabled_node(legacy_full_run_dir) -> None:
    plugin = get_disease_plugin("hiv")
    enabled_nodes = [row["block_id"] for row in plugin.node_graph_defaults if row.get("enabled", True)]
    bundle = build_node_graph_bundle(run_dir=legacy_full_run_dir, plugin_id="hiv")

    states = bundle.region_node_states
    assert states

    regions = sorted({row.region_id for row in states})
    observed_pairs = {(row.region_id, row.block_id) for row in states}
    expected_pairs = {(region, block_id) for region in regions for block_id in enabled_nodes}

    assert observed_pairs == expected_pairs
    assert bundle.debug_summary.get("region_count") == len(regions)
    assert bundle.debug_summary.get("enabled_node_count") == len(enabled_nodes)


def test_phase4_rescue_profiles_are_blocked(rescue_v1_run_dir, rescue_v2_run_dir) -> None:
    for run_dir in (rescue_v1_run_dir, rescue_v2_run_dir):
        manifest = read_json(run_dir / "phase4" / "phase4_manifest.json", default={})
        blocked = read_json(run_dir / "phase4" / "phase4_blocked.json", default={})
        node_bundle = read_json(run_dir / "phase4" / "node_graph_bundle.json", default={})
        truth_summary = read_json(run_dir / "phase4" / "ground_truth_summary.json", default={})
        assert manifest.get("stage_status", {}).get("phase4") == "blocked"
        assert blocked.get("phase4_ready") is False
        assert blocked.get("profile_id") in {"hiv_rescue_v1", "hiv_rescue_v2"}
        assert blocked.get("blocking_gates", [])
        assert truth_summary.get("phase_name") == "phase4"
        assert node_bundle.get("region_node_states", [])


def test_regional_allocation_surface_respects_veto_zero_weight() -> None:
    selected_policy = {
        "policy_id": "policy_test",
        "action": {"testing_expansion": 0.6, "linkage_support": 0.4, "art_retention": 0.0, "viral_monitoring": 0.0, "transport_support": 0.0, "stigma_reduction": 0.0, "workforce_deployment": 0.0},
    }
    region_rows = [
        {
            "region": "ncr",
            "adjusted_channel_scores": {"testing_expansion": 0.9, "linkage_support": 0.3, "art_retention": 0.0, "viral_monitoring": 0.0, "transport_support": 0.0, "stigma_reduction": 0.0, "workforce_deployment": 0.0},
            "node_graph": {"vetoed": False},
        },
        {
            "region": "region_xi",
            "adjusted_channel_scores": {"testing_expansion": 0.0, "linkage_support": 0.0, "art_retention": 0.0, "viral_monitoring": 0.0, "transport_support": 0.0, "stigma_reduction": 0.0, "workforce_deployment": 0.0},
            "node_graph": {"vetoed": True},
        },
    ]

    surface = _regional_allocation_surface(selected_policy=selected_policy, region_rows=region_rows)

    by_region = surface["region_allocations"]
    assert by_region["region_xi"]["testing_expansion"] == 0.0
    assert by_region["region_xi"]["linkage_support"] == 0.0
    assert math.isclose(by_region["ncr"]["testing_expansion"], 0.6, abs_tol=1e-6)


def test_regional_allocation_surface_excludes_national_pseudo_region() -> None:
    selected_policy = {
        "policy_id": "policy_test",
        "action": {
            "testing_expansion": 0.5,
            "linkage_support": 0.5,
            "art_retention": 0.0,
            "viral_monitoring": 0.0,
            "transport_support": 0.0,
            "stigma_reduction": 0.0,
            "workforce_deployment": 0.0,
        },
    }
    region_rows = [
        {
            "region": "national",
            "adjusted_channel_scores": {channel: 1.0 for channel in selected_policy["action"]},
            "node_graph": {"vetoed": False},
        },
        {
            "region": "ncr",
            "adjusted_channel_scores": {
                "testing_expansion": 0.8,
                "linkage_support": 0.4,
                "art_retention": 0.0,
                "viral_monitoring": 0.0,
                "transport_support": 0.0,
                "stigma_reduction": 0.0,
                "workforce_deployment": 0.0,
            },
            "node_graph": {"vetoed": False},
        },
    ]

    surface = _regional_allocation_surface(selected_policy=selected_policy, region_rows=region_rows)

    assert "national" not in surface["region_allocations"]
    assert math.isclose(surface["region_allocations"]["ncr"]["testing_expansion"], 0.5, abs_tol=1e-6)


def test_regional_allocation_surface_preserves_channel_budget() -> None:
    selected_policy = {
        "policy_id": "policy_budget",
        "action": {
            "testing_expansion": 0.35,
            "linkage_support": 0.25,
            "art_retention": 0.15,
            "viral_monitoring": 0.10,
            "transport_support": 0.05,
            "stigma_reduction": 0.05,
            "workforce_deployment": 0.05,
        },
    }
    region_rows = [
        {
            "region": "ncr",
            "adjusted_channel_scores": {
                "testing_expansion": 0.9,
                "linkage_support": 0.4,
                "art_retention": 0.3,
                "viral_monitoring": 0.2,
                "transport_support": 0.1,
                "stigma_reduction": 0.2,
                "workforce_deployment": 0.3,
            },
            "node_graph": {"vetoed": False},
        },
        {
            "region": "region_vii",
            "adjusted_channel_scores": {
                "testing_expansion": 0.3,
                "linkage_support": 0.6,
                "art_retention": 0.5,
                "viral_monitoring": 0.4,
                "transport_support": 0.6,
                "stigma_reduction": 0.3,
                "workforce_deployment": 0.2,
            },
            "node_graph": {"vetoed": False},
        },
        {
            "region": "region_xi",
            "adjusted_channel_scores": {
                "testing_expansion": 0.2,
                "linkage_support": 0.2,
                "art_retention": 0.2,
                "viral_monitoring": 0.2,
                "transport_support": 0.2,
                "stigma_reduction": 0.2,
                "workforce_deployment": 0.2,
            },
            "node_graph": {"vetoed": True},
        },
    ]

    surface = _regional_allocation_surface(selected_policy=selected_policy, region_rows=region_rows)

    for channel, budget in selected_policy["action"].items():
        channel_sum = sum(float(row["regional_allocation"]) for row in surface["rows"] if row["channel"] == channel)
        assert math.isclose(channel_sum, float(budget), abs_tol=2e-6)
    assert surface["region_allocations"]["region_xi"]["testing_expansion"] == 0.0


def test_phase4_stochastic_policy_rollout_returns_expected_trajectories() -> None:
    latest_states = np.asarray(
        [
            [0.52, 0.18, 0.17, 0.08, 0.05],
            [0.44, 0.20, 0.20, 0.10, 0.06],
        ],
        dtype=np.float32,
    )
    base_transition = np.asarray(
        [
            [0.10, 0.12, 0.11, 0.04, 0.07],
            [0.09, 0.11, 0.10, 0.05, 0.06],
        ],
        dtype=np.float32,
    )
    action = np.asarray([0.22, 0.18, 0.16, 0.12, 0.10, 0.12, 0.10], dtype=np.float32)

    rollout_states, rollout_objectives, stochastic_summary = _simulate_policy(
        latest_states,
        base_transition,
        action,
        horizon=4,
    )

    assert rollout_states.shape == (4, 2, 5)
    assert rollout_objectives.shape == (4, 4)
    assert np.isfinite(rollout_states).all()
    assert np.isfinite(rollout_objectives).all()
    assert int(stochastic_summary.get("trajectory_count", 0)) >= 1
    assert len(stochastic_summary.get("final_objective_std", [])) == 4
