from __future__ import annotations

import math

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3.pipeline import _adaptive_frozen_tuning_candidates, _trial_calibration_overrides
from epigraph_ph.phase3.rescue_core import (
    _enforce_cascade_ordering,
    _fallback_observation_targets,
    _inject_harp_program_targets,
    _province_operator_jax,
    _state_initialization_prior,
    _surface_from_sparse_support,
)
from epigraph_ph.runtime import load_tensor_artifact, read_json

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None


ALLOWED_OBSERVATION_CLASSES = {"direct_observed", "bounded_observed", "proxy_observed", "prior_only"}


def test_phase3_prior_hyperparameters_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase3_priors = (plugin.prior_hyperparameters or {}).get("phase3", {})

    assert phase3_priors.get("transition_prior")
    assert phase3_priors.get("kp_prior")
    assert phase3_priors.get("observation_fallback", {}).get("second95_prior") is not None
    assert phase3_priors.get("state_initialization", {}).get("loss_share_lower") is not None
    assert phase3_priors.get("cd4_overlay", {}).get("high_suppression_weight") is not None
    assert phase3_priors.get("subgroup_hyperpriors", {}).get("feature_coef_sigma") is not None
    assert phase3_priors.get("subgroup_hyperpriors", {}).get("state_effect_sigma") is not None
    assert phase3_priors.get("subgroup_hyperpriors", {}).get("transition_effect_sigma") is not None
    assert phase3_priors.get("cd4_hyperpriors", {}).get("age_low_sigma") is not None
    assert phase3_priors.get("cd4_hyperpriors", {}).get("kp_sex_high_sigma") is not None
    assert phase3_priors.get("cd4_hyperpriors", {}).get("transition_coef_sigma") is not None
    frozen_cfg = phase3_priors.get("frozen_backtest", {})
    assert frozen_cfg.get("default_initial_state") is not None
    assert frozen_cfg.get("empirical_transition_targets", {}).get("positive_mass_weight") is not None
    assert frozen_cfg.get("semi_markov_hyperpriors", {}).get("duration_prior_mean") is not None
    assert frozen_cfg.get("posterior_inference", {}).get("optimizer_lr") is not None
    assert frozen_cfg.get("fallback_posterior", {}).get("duration_mean") is not None
    assert frozen_cfg.get("evolution", {}).get("transition_floor") is not None
    assert frozen_cfg.get("legacy_selection", {}).get("mass_error_tolerance") is not None
    assert frozen_cfg.get("tuning", {}).get("adaptive_rules", {}).get("final_fit_steps") is not None


def test_phase3_legacy_output_contract(legacy_full_run_dir) -> None:
    fit_artifact = read_json(legacy_full_run_dir / "phase3" / "fit_artifact.json", default={})
    truth_summary = read_json(legacy_full_run_dir / "phase3" / "ground_truth_summary.json", default={})
    validation = read_json(legacy_full_run_dir / "phase3" / "validation_artifact.json", default={})
    transitions = read_json(legacy_full_run_dir / "phase3" / "transition_parameters.json", default={})
    state_estimates = load_tensor_artifact(legacy_full_run_dir / "phase3" / "state_estimates.npz")

    assert fit_artifact.get("fit_rows", [])
    assert validation.get("validation_gates", [])
    assert transitions.get("transition_names", [])
    assert state_estimates.shape[-1] == 5
    assert np.isfinite(state_estimates).all()
    assert truth_summary.get("phase_name") == "phase3"


def test_phase3_rescue_v1_truth_artifacts(rescue_v1_run_dir) -> None:
    spec = read_json(rescue_v1_run_dir / "phase3" / "rescue_core_spec.json", default={})
    fit_artifact = read_json(rescue_v1_run_dir / "phase3" / "fit_artifact.json", default={})
    validation = read_json(rescue_v1_run_dir / "phase3" / "validation_artifact.json", default={})
    ladder = read_json(rescue_v1_run_dir / "phase3" / "observation_ladder.json", default=[])
    gate_report = read_json(rescue_v1_run_dir / "phase3" / "benchmark_gate_report.json", default={})
    guardrails = read_json(rescue_v1_run_dir / "phase3" / "mechanistic_guardrails.json", default={})
    reference_check = read_json(rescue_v1_run_dir / "phase3" / "reference_check_official.json", default={})
    harp_check = read_json(rescue_v1_run_dir / "phase3" / "reference_check_harp.json", default={})
    national_anchor = read_json(rescue_v1_run_dir / "phase3" / "national_anchor_surfaces.json", default={})
    harp_surfaces = read_json(rescue_v1_run_dir / "phase3" / "harp_program_surfaces.json", default={})
    linkage_surfaces = read_json(rescue_v1_run_dir / "phase3" / "linkage_anchor_surfaces.json", default={})
    suppression_surfaces = read_json(rescue_v1_run_dir / "phase3" / "suppression_anchor_surfaces.json", default={})
    observation_support = read_json(rescue_v1_run_dir / "phase3" / "observation_support_summary.json", default={})
    truth_summary = read_json(rescue_v1_run_dir / "phase3" / "ground_truth_summary.json", default={})
    state_estimates = load_tensor_artifact(rescue_v1_run_dir / "phase3" / "state_estimates.npz")
    forecast_states = load_tensor_artifact(rescue_v1_run_dir / "phase3" / "forecast_states.npz")
    cd4_overlay = load_tensor_artifact(rescue_v1_run_dir / "phase3" / "cd4_overlay_tensor.npz")

    assert spec.get("state_catalog") == ["U", "D", "A", "V", "L"]
    assert fit_artifact.get("phase4_ready") is False
    assert "harp_program_penalty" in fit_artifact.get("loss_breakdown", {})
    assert "linkage_penalty" in fit_artifact.get("loss_breakdown", {})
    assert "suppression_penalty" in fit_artifact.get("loss_breakdown", {})
    assert validation.get("phase4_ready") is False
    assert gate_report.get("primary_gates", [])
    assert guardrails.get("phase4_ready") is False
    assert reference_check.get("comparisons", []) != []
    assert harp_check.get("comparisons", []) != []
    assert ladder
    assert all(row["observation_class"] in ALLOWED_OBSERVATION_CLASSES for row in ladder)
    assert np.isfinite(state_estimates).all()
    assert np.isfinite(forecast_states).all()
    assert np.isfinite(cd4_overlay).all()
    assert np.all(state_estimates >= 0.0)
    state_mass = state_estimates.sum(axis=-1)
    assert np.all(state_mass <= 1.0001)
    assert float(state_mass.mean()) >= 0.95
    assert np.allclose(cd4_overlay.sum(axis=4), 1.0, atol=1e-4)
    assert national_anchor.get("month_axis", []) != []
    assert harp_surfaces.get("month_axis", []) != []
    assert linkage_surfaces.get("month_axis", []) != []
    assert suppression_surfaces.get("month_axis", []) != []
    assert observation_support.get("rows", []) != []
    assert fit_artifact.get("observation_support_summary", {}).get("rows", []) != []
    assert truth_summary.get("phase_name") == "phase3"


def test_phase3_rescue_v2_factor_and_metapop_contract(rescue_v2_run_dir) -> None:
    fit_artifact = read_json(rescue_v2_run_dir / "phase3" / "fit_artifact.json", default={})
    determinant_modifiers = read_json(rescue_v2_run_dir / "phase3" / "determinant_modifiers.json", default={})
    subgroup_coupling = read_json(rescue_v2_run_dir / "phase3" / "subgroup_coupling.json", default={})
    province_archetypes = read_json(rescue_v2_run_dir / "phase3" / "province_archetype_mixture.json", default={})
    synthetic_pretraining = read_json(rescue_v2_run_dir / "phase3" / "synthetic_pretraining_summary.json", default={})
    subgroup_prior_learning = read_json(rescue_v2_run_dir / "phase3" / "subgroup_prior_learning_summary.json", default={})
    cd4_prior_learning = read_json(rescue_v2_run_dir / "phase3" / "cd4_prior_learning_summary.json", default={})
    reference_check = read_json(rescue_v2_run_dir / "phase3" / "reference_check_official.json", default={})
    harp_check = read_json(rescue_v2_run_dir / "phase3" / "reference_check_harp.json", default={})
    linkage_surfaces = read_json(rescue_v2_run_dir / "phase3" / "linkage_anchor_surfaces.json", default={})
    suppression_surfaces = read_json(rescue_v2_run_dir / "phase3" / "suppression_anchor_surfaces.json", default={})
    subgroup_summary = read_json(rescue_v2_run_dir / "phase3" / "subgroup_weight_summary.json", default={})

    assert fit_artifact.get("profile_id") == "hiv_rescue_v2"
    assert "harp_program_penalty" in fit_artifact.get("loss_breakdown", {})
    assert "linkage_penalty" in fit_artifact.get("loss_breakdown", {})
    assert "suppression_penalty" in fit_artifact.get("loss_breakdown", {})
    assert determinant_modifiers.get("selected_determinant_modifiers", []) != []
    assert subgroup_coupling.get("metapopulation_engine", {}).get("enabled") is True
    assert set(subgroup_coupling.get("metapopulation_engine", {}).get("operator_names", [])) == {
        "mobility_operator",
        "service_operator",
        "information_operator",
    }
    assert reference_check.get("comparisons", []) != []
    assert harp_check.get("comparisons", []) != []
    assert linkage_surfaces.get("month_axis", []) != []
    assert suppression_surfaces.get("month_axis", []) != []
    assert province_archetypes.get("rows", []) != []
    assert synthetic_pretraining.get("library")
    assert fit_artifact.get("subgroup_prior_learning_summary", {}).get("feature_names", []) != []
    assert fit_artifact.get("cd4_prior_learning_summary", {}).get("age_low_offset", []) != []
    assert subgroup_prior_learning.get("feature_names", []) != []
    assert subgroup_prior_learning.get("kp_feature_coefficients", []) != []
    assert subgroup_prior_learning.get("mean_kp_distribution", []) != []
    assert cd4_prior_learning.get("age_low_offset", []) != []
    assert cd4_prior_learning.get("kp_sex_high_offset", []) != []
    rows = subgroup_summary.get("rows", [])
    assert rows
    kp_vectors = {tuple(sorted(row.get("kp_distribution", {}).items())) for row in rows}
    assert len(kp_vectors) > 1


def test_phase3_frozen_history_backtest_contract(rescue_v2_backtest_run_dir) -> None:
    backtest_dir = rescue_v2_backtest_run_dir / "phase3_frozen_backtest"
    manifest = read_json(backtest_dir / "phase3_manifest.json", default={})
    fit_artifact = read_json(backtest_dir / "fit_artifact.json", default={})
    validation = read_json(backtest_dir / "validation_artifact.json", default={})
    spec = read_json(backtest_dir / "frozen_history_backtest_spec.json", default={})
    evaluation = read_json(backtest_dir / "frozen_history_backtest_evaluation.json", default={})
    state_estimates = load_tensor_artifact(backtest_dir / "state_estimates.npz")
    forecast_states = load_tensor_artifact(backtest_dir / "forecast_states.npz")

    assert manifest.get("artifact_paths", {}).get("frozen_history_backtest_spec")
    assert manifest.get("artifact_paths", {}).get("frozen_history_backtest_evaluation")
    assert spec.get("train_years") == [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    assert spec.get("holdout_years") == [2024]
    assert spec.get("forecast_horizon", 0) >= 1
    assert evaluation.get("summary", {}).get("comparison_count", 0) >= 1
    assert evaluation.get("holdout_reference_check", {}).get("comparisons", []) != []
    assert evaluation.get("carry_forward_baseline", {}).get("available") is True
    assert fit_artifact.get("frozen_history_backtest", {}).get("comparison_count", 0) >= 1
    assert validation.get("frozen_history_backtest", {}).get("summary", {}).get("comparison_count", 0) >= 1
    assert np.isfinite(state_estimates).all()
    assert np.isfinite(forecast_states).all()
    assert state_estimates.shape[-1] == 5
    assert forecast_states.shape[-1] == 5


def test_phase3_jax_svi_contract(rescue_v1_jax_run_dir) -> None:
    fit_artifact = read_json(rescue_v1_jax_run_dir / "phase3" / "fit_artifact.json", default={})
    spec = read_json(rescue_v1_jax_run_dir / "phase3" / "rescue_core_spec.json", default={})
    manifest = read_json(rescue_v1_jax_run_dir / "phase3" / "phase3_manifest.json", default={})
    subgroup_prior_learning = read_json(rescue_v1_jax_run_dir / "phase3" / "subgroup_prior_learning_summary.json", default={})
    cd4_prior_learning = read_json(rescue_v1_jax_run_dir / "phase3" / "cd4_prior_learning_summary.json", default={})

    assert fit_artifact.get("inference_family") == "jax_svi"
    assert spec.get("requested_inference_family") == "jax_svi"
    assert spec.get("resolved_inference_family") == "jax_svi"
    assert manifest.get("backend_status", {}).get("jax", {}).get("available") is True
    assert fit_artifact.get("subgroup_prior_learning_summary", {}).get("feature_names", []) != []
    assert fit_artifact.get("cd4_prior_learning_summary", {}).get("age_low_offset", []) != []
    assert subgroup_prior_learning.get("mean_kp_distribution", []) != []
    assert cd4_prior_learning.get("kp_sex_high_offset", []) != []


def test_phase3_no_silent_fallback_and_truth_surfaces(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase3" / "phase3_manifest.json", default={})
    reference_check = read_json(rescue_v2_run_dir / "phase3" / "reference_check_official.json", default={})
    harp_check = read_json(rescue_v2_run_dir / "phase3" / "reference_check_harp.json", default={})
    observation_targets = read_json(rescue_v2_run_dir / "phase3" / "observation_targets.json", default=[])
    residuals = read_json(rescue_v2_run_dir / "phase3" / "observation_residuals.json", default=[])

    assert manifest.get("artifact_paths", {}).get("reference_check_official")
    assert manifest.get("artifact_paths", {}).get("reference_check_harp")
    assert manifest.get("artifact_paths", {}).get("linkage_anchor_surfaces")
    assert manifest.get("artifact_paths", {}).get("suppression_anchor_surfaces")
    assert manifest.get("artifact_paths", {}).get("observation_support_summary")
    assert manifest.get("artifact_paths", {}).get("province_archetype_mixture")
    assert manifest.get("artifact_paths", {}).get("synthetic_pretraining_summary")
    assert manifest.get("artifact_paths", {}).get("subgroup_prior_learning_summary")
    assert manifest.get("artifact_paths", {}).get("cd4_prior_learning_summary")
    assert observation_targets
    assert residuals
    assert reference_check.get("verdict")
    assert reference_check.get("comparisons", []) != []
    assert harp_check.get("verdict")
    assert harp_check.get("comparisons", []) != []


def test_surface_from_sparse_support_ignores_missing_cells() -> None:
    standardized = np.zeros((3, 2, 2), dtype=np.float32)
    support = np.zeros_like(standardized)
    standardized[0, 0, 0] = -2.0
    standardized[1, 0, 0] = 2.0
    support[0, 0, 0] = 1.0
    support[1, 0, 0] = 1.0
    matches = [(0, 1.0, {"canonical_name": "case_count", "evidence_classes": {"observed_numeric": 1}})]
    surface, meta = _surface_from_sparse_support(standardized, support, matches, ["A", "B", "C"])
    assert meta["observed_cell_fraction"] > 0.0
    assert surface.shape == (3, 2)
    assert float(surface[0, 0]) != float(surface[1, 0])


def test_phase3_cascade_ordering_treats_testing_as_vl_tested_stock() -> None:
    targets = {
        "diagnosed_stock": np.asarray([[0.60]], dtype=np.float32),
        "art_stock": np.asarray([[0.50]], dtype=np.float32),
        "documented_suppression": np.asarray([[0.45]], dtype=np.float32),
        "testing_coverage": np.asarray([[0.70]], dtype=np.float32),
        "deaths": np.asarray([[0.02]], dtype=np.float32),
    }

    ordered = _enforce_cascade_ordering(targets)

    assert float(ordered["documented_suppression"][0, 0]) <= float(ordered["testing_coverage"][0, 0])
    assert float(ordered["testing_coverage"][0, 0]) <= float(ordered["art_stock"][0, 0])
    assert float(ordered["art_stock"][0, 0]) <= float(ordered["diagnosed_stock"][0, 0])
    assert math.isclose(float(ordered["testing_coverage"][0, 0]), 0.5, rel_tol=1e-5, abs_tol=1e-5)


def test_phase3_fallback_targets_use_anchor_curves_when_surfaces_are_empty() -> None:
    month_axis = ["2022-01", "2025-01"]
    empty = np.zeros((1, len(month_axis)), dtype=np.float32)
    repaired = _fallback_observation_targets(
        {
            "diagnosed_stock": empty.copy(),
            "art_stock": empty.copy(),
            "documented_suppression": empty.copy(),
            "testing_coverage": empty.copy(),
            "deaths": empty.copy(),
        },
        month_axis=month_axis,
    )

    assert float(repaired["diagnosed_stock"].sum()) > 0.0
    assert float(repaired["art_stock"].sum()) > 0.0
    assert float(repaired["documented_suppression"].sum()) > 0.0
    assert float(repaired["testing_coverage"].sum()) > 0.0
    assert np.all(repaired["testing_coverage"] <= repaired["art_stock"] + 1e-6)
    assert np.all(repaired["art_stock"] <= repaired["diagnosed_stock"] + 1e-6)


def test_phase3_harp_injection_can_pull_national_target_down() -> None:
    month_axis = ["2025-01"]
    province_axis = ["Philippines"]
    observation_ladder = [
        {"target_name": "diagnosed_stock", "observation_class": "proxy_observed", "weight": 0.35},
        {"target_name": "art_stock", "observation_class": "proxy_observed", "weight": 0.35},
        {"target_name": "documented_suppression", "observation_class": "proxy_observed", "weight": 0.35},
        {"target_name": "testing_coverage", "observation_class": "proxy_observed", "weight": 0.35},
        {"target_name": "deaths", "observation_class": "proxy_observed", "weight": 0.35},
    ]
    observation_targets = {
        "diagnosed_stock": np.asarray([[0.90]], dtype=np.float32),
        "art_stock": np.asarray([[0.80]], dtype=np.float32),
        "documented_suppression": np.asarray([[0.70]], dtype=np.float32),
        "testing_coverage": np.asarray([[0.78]], dtype=np.float32),
        "deaths": np.asarray([[0.02]], dtype=np.float32),
    }

    _, updated_targets, _, harp_summary = _inject_harp_program_targets(
        observation_ladder=observation_ladder,
        observation_targets=observation_targets,
        province_axis=province_axis,
        month_axis=month_axis,
    )

    assert harp_summary.get("applied") is True
    assert float(updated_targets["diagnosed_stock"][0, 0]) < 0.90
    assert float(updated_targets["diagnosed_stock"][0, 0]) < 0.70
    assert float(updated_targets["testing_coverage"][0, 0]) <= float(updated_targets["art_stock"][0, 0]) + 1e-6


def test_phase3_jax_operator_preserves_row_count_when_available() -> None:
    if jnp is None:
        return
    state = jnp.asarray(
        [
            [0.60, 0.20, 0.10, 0.05, 0.05],
            [0.35, 0.25, 0.20, 0.10, 0.10],
            [0.20, 0.30, 0.25, 0.15, 0.10],
        ],
        dtype=jnp.float32,
    )
    operator = jnp.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.70, 0.10],
            [0.10, 0.25, 0.65],
        ],
        dtype=jnp.float32,
    )

    mixed = np.asarray(_province_operator_jax(state, operator))

    assert mixed.shape == (3, 5)
    assert np.isfinite(mixed).all()
    assert np.allclose(mixed.sum(axis=1), 1.0, atol=1e-5)


def test_phase3_state_initialization_prior_uses_support_weighting() -> None:
    observation_targets = {
        "diagnosed_stock": np.asarray([[0.70], [0.70]], dtype=np.float32),
        "art_stock": np.asarray([[0.52], [0.52]], dtype=np.float32),
        "documented_suppression": np.asarray([[0.31], [0.31]], dtype=np.float32),
        "testing_coverage": np.asarray([[0.40], [0.40]], dtype=np.float32),
        "deaths": np.asarray([[0.02], [0.02]], dtype=np.float32),
    }
    support_bundle = {
        "targets": {
            "diagnosed_stock": {"support_strength": np.asarray([[0.95], [0.10]], dtype=np.float32)},
            "art_stock": {"support_strength": np.asarray([[0.95], [0.10]], dtype=np.float32)},
            "documented_suppression": {"support_strength": np.asarray([[0.95], [0.10]], dtype=np.float32)},
            "testing_coverage": {"support_strength": np.asarray([[0.95], [0.10]], dtype=np.float32)},
        }
    }

    prior = _state_initialization_prior(observation_targets, support_bundle=support_bundle)

    assert prior["mean"].shape == (2, 1, 5)
    assert np.allclose(prior["mean"].sum(axis=-1), 1.0, atol=1e-5)
    assert float(prior["concentration"][0, 0]) > float(prior["concentration"][1, 0])


def test_phase3_tuning_overrides_replace_diagnosis_and_linkage_priors() -> None:
    overrides = _trial_calibration_overrides(
        {
            "u_to_d_prior": 0.02,
            "d_to_a_prior": 0.18,
            "diagnosed_penalty_scale": 22.0,
            "linkage_penalty_scale": 30.0,
        }
    )
    assert round(float(overrides["transition_prior_override"][0]), 3) == 0.02
    assert round(float(overrides["transition_prior_override"][1]), 3) == 0.18
    assert float(overrides["diagnosed_penalty_scale"]) == 22.0
    assert float(overrides["linkage_penalty_scale"]) == 30.0


def test_phase3_adaptive_tuning_candidates_push_diagnosis_lower_and_linkage_higher() -> None:
    candidates = _adaptive_frozen_tuning_candidates(
        {
            "calibration_overrides": {
                "transition_prior_override": [0.02, 0.14, 0.11, 0.05, 0.08],
                "diagnosed_penalty_scale": 20.0,
                "official_reference_penalty_scale": 18.0,
                "national_anchor_penalty_scale": 18.0,
                "harp_program_penalty_scale": 42.0,
                "linkage_penalty_scale": 28.0,
            }
        }
    )
    assert candidates
    assert min(float(row["u_to_d_prior"]) for row in candidates) < 0.02
    assert max(float(row["d_to_a_prior"]) for row in candidates) > 0.14
