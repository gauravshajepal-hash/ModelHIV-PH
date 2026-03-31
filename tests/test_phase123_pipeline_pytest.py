from __future__ import annotations

import numpy as np

from epigraph_ph.phase3.rescue_core import _build_cd4_overlay, build_observation_ladder, simulate_transition_step_numpy
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase1234_build_smoke(legacy_full_run_dir) -> None:
    run_dir = legacy_full_run_dir
    phase1 = read_json(run_dir / "phase1" / "phase1_manifest.json", default={})
    phase2 = read_json(run_dir / "phase2" / "phase2_manifest.json", default={})
    phase3 = read_json(run_dir / "phase3" / "phase3_manifest.json", default={})
    phase4 = read_json(run_dir / "phase4" / "phase4_manifest.json", default={})
    assert phase1["stage_status"]["phase1"] == "completed"
    assert phase2["stage_status"]["phase2"] == "completed"
    assert phase3["stage_status"]["phase3"] == "completed"
    assert phase4["stage_status"]["phase4"] == "completed"
    tensor_rows = read_json(run_dir / "phase1" / "tensor_rows.json", default=[])
    axis_catalogs = read_json(run_dir / "phase1" / "axis_catalogs.json", default={})
    normalization_report = read_json(run_dir / "phase1" / "normalization_report.json", default={})
    candidate_profiles = read_json(run_dir / "phase2" / "candidate_profiles.json", default=[])
    curated_blocks = read_json(run_dir / "phase2" / "curated_candidate_blocks.json", default=[])
    markov_blanket = read_json(run_dir / "phase2" / "markov_blanket.json", default={})
    inference_ready = read_json(run_dir / "phase3" / "inference_ready_candidates.json", default=[])
    fit_artifact = read_json(run_dir / "phase3" / "fit_artifact.json", default={})
    validation_artifact = read_json(run_dir / "phase3" / "validation_artifact.json", default={})
    transition_parameters = read_json(run_dir / "phase3" / "transition_parameters.json", default={})
    counterfactual_rankings = read_json(run_dir / "phase4" / "counterfactual_rankings.json", default=[])
    selected_policy = read_json(run_dir / "phase4" / "selected_policy.json", default={})
    mpc_plan = read_json(run_dir / "phase4" / "mpc_plan.json", default={})
    state_estimates = load_tensor_artifact(run_dir / "phase3" / "state_estimates.npz")
    rollout_tensor = load_tensor_artifact(run_dir / "phase4" / "rollout_tensor.npz")
    assert tensor_rows != []
    assert axis_catalogs != {}
    assert normalization_report.get("tensor_row_count", 0) > 0
    assert candidate_profiles != []
    assert curated_blocks != []
    assert markov_blanket.get("blanket_nodes", []) != []
    assert inference_ready != []
    assert fit_artifact.get("fit_rows", []) != []
    assert validation_artifact.get("validation_gates", []) != []
    assert transition_parameters.get("transition_names", []) != []
    assert counterfactual_rankings != []
    assert selected_policy.get("action", {}) != {}
    assert mpc_plan.get("execution_steps", []) != []
    assert state_estimates.shape[-1] == 5
    assert rollout_tensor.shape[-1] == 4


def test_rescue_transition_step_conserves_mass() -> None:
    current = np.zeros((2, 3, 2, 2, 5, 4), dtype=np.float32)
    current[..., 0, 0] = 1.0
    transition_probs = np.zeros((2, 3, 2, 2, 4, 5), dtype=np.float32)
    transition_probs[..., 0] = 0.12
    transition_probs[..., 1] = 0.10
    transition_probs[..., 2] = 0.09
    transition_probs[..., 3] = 0.04
    transition_probs[..., 4] = 0.06
    next_state = simulate_transition_step_numpy(current, transition_probs)
    assert np.all(next_state >= 0.0)
    assert np.allclose(next_state.sum(axis=(-2, -1)), 1.0, atol=1e-5)


def test_rescue_observation_ladder_and_cd4_overlay() -> None:
    standardized_tensor = np.asarray(
        [
            [[2.0, 1.5, 1.0, 0.8], [1.8, 1.4, 1.1, 0.9]],
            [[1.2, 1.0, 0.7, 0.6], [1.1, 0.9, 0.6, 0.5]],
        ],
        dtype=np.float32,
    )
    parameter_catalog = [
        {"canonical_name": "diagnosed_cases", "domain_families": {"cascade": 3}, "pathway_families": {"testing_uptake": 2}, "evidence_classes": {"observed_numeric": 2}, "numeric_row_count": 2, "source_banks": {"phase0_extracted": 2}},
        {"canonical_name": "art_retention", "domain_families": {"policy": 1}, "pathway_families": {"retention_adherence": 2}, "evidence_classes": {"observed_numeric": 1}, "numeric_row_count": 1, "source_banks": {"phase0_extracted": 1}},
        {"canonical_name": "viral_suppression", "domain_families": {"biology": 2}, "pathway_families": {"suppression_outcomes": 3}, "evidence_classes": {"numeric_prior": 2}, "numeric_row_count": 2, "source_banks": {"phase0_extracted": 2}},
        {"canonical_name": "testing_coverage", "domain_families": {"behavior": 1}, "pathway_families": {"testing_uptake": 3}, "evidence_classes": {"observed_numeric": 1}, "numeric_row_count": 1, "source_banks": {"phase0_extracted": 1}},
    ]
    ladder, targets, target_rows = build_observation_ladder(
        standardized_tensor=standardized_tensor,
        normalized_rows=[],
        parameter_catalog=parameter_catalog,
        canonical_axis=[row["canonical_name"] for row in parameter_catalog],
        province_axis=["a", "b"],
        month_axis=["2025-01", "2025-02"],
    )
    assert len(ladder) == 5
    assert len(target_rows) == 2 * 2 * 5
    assert targets["diagnosed_stock"].shape == (2, 2)
    assert np.all(targets["diagnosed_stock"] >= targets["art_stock"])
    assert np.all(targets["art_stock"] >= targets["documented_suppression"])
    overlay, summary = _build_cd4_overlay(targets, ["a", "b"], ["remaining_population", "msm"], ["15_24", "25_34"], ["male", "female"])
    assert overlay.shape == (2, 2, 2, 2, 4, 2)
    assert np.allclose(overlay.sum(axis=4), 1.0, atol=1e-5)
    assert summary["simplex_max_error"] <= 1e-5


def test_phase3_rescue_build_smoke_and_phase4_block(rescue_v1_run_dir) -> None:
    run_dir = rescue_v1_run_dir
    phase3 = read_json(run_dir / "phase3" / "phase3_manifest.json", default={})
    phase4 = read_json(run_dir / "phase4" / "phase4_manifest.json", default={})

    assert phase3["stage_status"]["phase3"] == "completed"
    assert phase4["stage_status"]["phase4"] == "blocked"

    fit_artifact = read_json(run_dir / "phase3" / "fit_artifact.json", default={})
    validation_artifact = read_json(run_dir / "phase3" / "validation_artifact.json", default={})
    rescue_core_spec = read_json(run_dir / "phase3" / "rescue_core_spec.json", default={})
    guardrails = read_json(run_dir / "phase3" / "mechanistic_guardrails.json", default={})
    benchmark_report = read_json(run_dir / "phase3" / "benchmark_gate_report.json", default={})
    reference_check = read_json(run_dir / "phase3" / "reference_check_official.json", default={})
    blocked = read_json(run_dir / "phase4" / "phase4_blocked.json", default={})
    determinant_modifiers = read_json(run_dir / "phase3" / "determinant_modifiers.json", default={})
    subgroup_coupling = read_json(run_dir / "phase3" / "subgroup_coupling.json", default={})
    state_estimates = load_tensor_artifact(run_dir / "phase3" / "state_estimates.npz")
    forecast_states = load_tensor_artifact(run_dir / "phase3" / "forecast_states.npz")
    cd4_overlay = load_tensor_artifact(run_dir / "phase3" / "cd4_overlay_tensor.npz")

    assert fit_artifact.get("profile_id") == "hiv_rescue_v1"
    assert fit_artifact.get("phase4_ready") is False
    assert rescue_core_spec.get("state_catalog") == ["U", "D", "A", "V", "L"]
    assert rescue_core_spec.get("duration_catalog") == ["0_2", "3_5", "6_11", "12_plus"]
    assert validation_artifact.get("phase4_ready") is False
    assert benchmark_report.get("primary_gates", []) != []
    assert "comparisons" in reference_check
    assert blocked.get("phase4_ready") is False
    assert guardrails.get("phase4_ready") is False
    assert determinant_modifiers.get("covariate_names", []) != []
    assert subgroup_coupling.get("kp_coupling_matrix", []) != []
    assert state_estimates.shape[-1] == 5
    assert forecast_states.shape[-1] == 5
    assert cd4_overlay.shape[4] == 4


def test_phase3_rescue_jax_svi_smoke(rescue_v1_jax_run_dir) -> None:
    run_dir = rescue_v1_jax_run_dir
    fit_artifact = read_json(run_dir / "phase3" / "fit_artifact.json", default={})
    manifest = read_json(run_dir / "phase3" / "phase3_manifest.json", default={})
    assert fit_artifact.get("inference_family") == "jax_svi"
    assert manifest.get("backend_status", {}).get("jax", {}).get("available") is True


def test_phase15_build_and_rescue_v2_smoke(rescue_v2_run_dir) -> None:
    run_dir = rescue_v2_run_dir
    phase15 = read_json(run_dir / "phase15" / "phase15_manifest.json", default={})
    phase2 = read_json(run_dir / "phase2" / "phase2_manifest.json", default={})
    phase3 = read_json(run_dir / "phase3" / "phase3_manifest.json", default={})
    phase4 = read_json(run_dir / "phase4" / "phase4_manifest.json", default={})

    assert phase15["stage_status"]["phase15"] == "completed"
    assert phase2["profile_id"] == "hiv_rescue_v2"
    assert phase3["stage_status"]["phase3"] == "completed"
    assert phase4["stage_status"]["phase4"] == "blocked"

    phase15_manifest = read_json(run_dir / "phase15" / "phase15_manifest.json", default={})
    factor_catalog = read_json(run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    factor_pool = read_json(run_dir / "phase15" / "factor_promotion_pool.json", default=[])
    promoted = read_json(run_dir / "phase2" / "promoted_factor_set.json", default=[])
    supporting = read_json(run_dir / "phase2" / "supporting_factor_set.json", default=[])
    fit_artifact = read_json(run_dir / "phase3" / "fit_artifact.json", default={})
    determinant_modifiers = read_json(run_dir / "phase3" / "determinant_modifiers.json", default={})
    reference_check = read_json(run_dir / "phase3" / "reference_check_official.json", default={})
    alignment_summary = read_json(run_dir / "phase0" / "extracted" / "alignment_summary.json", default={})
    blocked = read_json(run_dir / "phase4" / "phase4_blocked.json", default={})
    factor_tensor = load_tensor_artifact(run_dir / "phase15" / "mesoscopic_factor_tensor.npz")
    network_tensor = load_tensor_artifact(run_dir / "phase15" / "network_feature_tensor.npz")
    network_operator_tensor = load_tensor_artifact(run_dir / "phase15" / "network_operator_tensor.npz")

    assert phase15_manifest.get("profile_id") == "hiv_rescue_v2"
    assert factor_catalog != []
    assert factor_pool != []
    assert fit_artifact.get("profile_id") == "hiv_rescue_v2"
    if promoted or supporting:
        assert determinant_modifiers.get("selected_determinant_modifiers", []) != []
    else:
        assert determinant_modifiers.get("selected_determinant_modifiers", []) == []
    assert "comparisons" in reference_check
    assert blocked.get("phase4_ready") is False
    assert len(alignment_summary.get("province_axis", [])) > 1
    assert factor_tensor.shape[-1] == len(factor_catalog)
    assert network_tensor.shape[-1] >= 3
    assert network_operator_tensor.shape[0] == 3
    assert fit_artifact.get("subgroup_coupling_summary", {}).get("metapopulation_engine", {}).get("enabled") is True
    assert len(promoted) <= 8
    assert len(supporting) <= 12
