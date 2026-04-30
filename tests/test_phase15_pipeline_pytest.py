from __future__ import annotations

import json

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3.rescue_core import build_archive_aware_observation_targets
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase15_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase15_cfg = (plugin.constraint_settings or {}).get("phase15", {})

    assert phase15_cfg.get("similarity_weights", {}).get("signature_cosine") is not None
    assert phase15_cfg.get("similarity_quality", {}).get("within_block_threshold") is not None
    assert phase15_cfg.get("factor_extraction", {}).get("svd_eps") is not None
    assert phase15_cfg.get("multiscale", {}).get("enabled") is True
    assert phase15_cfg.get("multiscale", {}).get("scale_thresholds", {}).get("province") is not None
    assert phase15_cfg.get("multiscale", {}).get("max_related_members", {}).get("national") is not None
    assert phase15_cfg.get("multiscale", {}).get("internal_bonus_scale_by_scale", {}).get("national") is not None
    assert phase15_cfg.get("network_graph", {}).get("adjacency_top_k") is not None
    assert phase15_cfg.get("network_graph", {}).get("positive_edge_threshold") is not None
    assert phase15_cfg.get("latent_blocks", {}).get("enabled") is True
    assert phase15_cfg.get("latent_blocks", {}).get("province_factor_graph", {}).get("enabled") is True
    assert phase15_cfg.get("latent_blocks", {}).get("blocks")
    assert phase15_cfg.get("latent_blocks_v2", {}).get("enabled") is True
    assert phase15_cfg.get("latent_blocks_v2", {}).get("aggregation_weights", {}).get("province_to_national") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("observation_model", {}).get("annual_aggregation") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("state_dynamics", {}).get("persistence_abs_ceiling") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("variance_model", {}).get("innovation_scale_prior") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("outer_iterations") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("weight_proxy_canonical_names")
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("numerical_adequacy_enabled") is True
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("missing_information_preconditioner") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("missing_information_torch_cg_retry_max_iter") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("missing_information_cpu_cg_max_iter") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("numerical_adequacy_tolerance_rtol_values")
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("numerical_adequacy_reference_max_cells") is not None
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("pooling_sensitivity_enabled") is True
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("pooling_sensitivity_scales")
    assert phase15_cfg.get("latent_blocks_v2", {}).get("engine", {}).get("calibration_intervals")
    testing_block = next(row for row in phase15_cfg.get("latent_blocks", {}).get("blocks", []) if row.get("block_id") == "testing_prevention_reach")
    suppression_block = next(row for row in phase15_cfg.get("latent_blocks", {}).get("blocks", []) if row.get("block_id") == "suppression_capacity")
    assert "annual_hiv_tests_volume_per_100k" in testing_block.get("indicators", {})
    assert "viral_load" in suppression_block.get("indicators", {})
    assert phase15_cfg.get("stability", {}).get("permutation_draws") is not None
    assert phase15_cfg.get("stability", {}).get("score_weights", {}).get("predictive_gain") is not None
    assert phase15_cfg.get("stability", {}).get("tournament_weights", {}).get("mae_improvement") is not None
    assert phase15_cfg.get("stability", {}).get("penalty_weights", {}).get("sparsity") is not None
    assert phase15_cfg.get("stability", {}).get("require_holdout_survival_for_promotion") is True
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("trials") is not None
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("primary_survivors_range") is not None
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("representation_mix_bonus_scale") is not None


def test_phase15_archive_aware_targets_use_harp_archive(tmp_path) -> None:
    archive_dir = tmp_path / "harp_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "harp_program_points.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "year": 2024,
                        "month": "2024-12",
                        "effective_month": "2024-12",
                        "source_month": "2024-12",
                        "estimated_plhiv": 100000.0,
                        "diagnosed": 60000.0,
                        "on_art": 42000.0,
                        "viral_load_tested": 30000.0,
                        "suppressed": 24000.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (archive_dir / "diagnosis_flow_points.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "period_start": "2024-10",
                        "period_end": "2024-12",
                        "diagnosed_share": 0.012,
                        "weight": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ladder, targets, _rows, summary = build_archive_aware_observation_targets(
        standardized_tensor=np.zeros((1, 4, 0), dtype=np.float32),
        normalized_rows=[],
        parameter_catalog=[],
        canonical_axis=[],
        province_axis=["Philippines"],
        month_axis=["2024-09", "2024-10", "2024-11", "2024-12"],
        run_dir=tmp_path,
    )

    ladder_map = {row["target_name"]: row for row in ladder}
    assert summary["harp_program_support"]["applied"] is True
    assert summary["diagnosis_flow_summary"]["point_count"] == 1
    assert ladder_map["diagnosed_stock"]["harp_program_support"] is True
    assert float(targets["diagnosed_stock"][0, -1]) > 0.0


def test_phase15_manifest_and_factor_contract(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})
    truth_summary = read_json(rescue_v2_run_dir / "phase15" / "ground_truth_summary.json", default={})
    factor_catalog = read_json(rescue_v2_run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    factor_rows = read_json(rescue_v2_run_dir / "phase15" / "factor_rows.json", default=[])
    factor_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "mesoscopic_factor_tensor.npz")
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})

    assert manifest.get("profile_id") == "hiv_rescue_v2"
    assert manifest.get("stage_status", {}).get("phase15") == "completed"
    assert factor_catalog
    assert factor_tensor.shape[-1] == len(factor_catalog)
    assert factor_tensor.shape[:2] == (len(axis_catalogs.get("province", [])), len(axis_catalogs.get("month", [])))
    assert len(factor_rows) == factor_tensor.shape[0] * factor_tensor.shape[1] * factor_tensor.shape[2]
    assert np.isfinite(factor_tensor).all()
    assert truth_summary.get("phase_name") == "phase15"


def test_phase15_emits_national_latent_scaffold_artifacts(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})
    derived_rows = read_json(rescue_v2_run_dir / "phase15" / "archive_derived_indicator_rows.json", default=[])
    augmented_audit = read_json(rescue_v2_run_dir / "phase15" / "augmented_latent_observability_audit.json", default={})
    phase15_v2_spec = read_json(rescue_v2_run_dir / "phase15" / "phase15_v2_model_spec.json", default={})
    measurement_spec = read_json(rescue_v2_run_dir / "phase15" / "national_block_measurement_spec.json", default={})
    states = read_json(rescue_v2_run_dir / "phase15" / "national_block_states.json", default={})
    identification = read_json(rescue_v2_run_dir / "phase15" / "national_block_identification_report.json", default={})
    numerical_adequacy = read_json(rescue_v2_run_dir / "phase15" / "phase15_v2_numerical_adequacy.json", default={})

    assert manifest.get("artifact_paths", {}).get("archive_derived_indicator_rows")
    assert manifest.get("artifact_paths", {}).get("augmented_latent_observability_audit")
    assert manifest.get("artifact_paths", {}).get("phase15_v2_model_spec")
    assert manifest.get("artifact_paths", {}).get("phase15_v2_numerical_adequacy")
    assert manifest.get("artifact_paths", {}).get("national_block_measurement_spec")
    assert manifest.get("artifact_paths", {}).get("national_block_states")
    assert derived_rows
    assert augmented_audit.get("rows")
    assert phase15_v2_spec.get("model_id") == "phase15_v2_mixed_frequency_hierarchical_dynamic_factor_model"
    assert phase15_v2_spec.get("equations")
    assert numerical_adequacy.get("method") == "phase15_v2_numerical_adequacy_v1"
    assert "summary" in numerical_adequacy
    assert measurement_spec.get("method") == "signed_weighted_national_scaffold_v1"
    assert "retained_blocks" in measurement_spec
    assert int(measurement_spec.get("retained_block_count") or 0) >= 2
    assert states.get("method") == "signed_weighted_national_scaffold_v1"
    month_axis = states.get("month_axis", [])
    for row in states.get("rows", []):
        assert len(row.get("state_values", [])) == len(month_axis)
    assert identification.get("is_scaffold") is True


def test_phase15_emits_province_factor_graph_artifacts(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})
    states = read_json(rescue_v2_run_dir / "phase15" / "province_block_states.json", default={})
    uncertainty = read_json(rescue_v2_run_dir / "phase15" / "province_block_uncertainty.json", default={})
    deviations = read_json(rescue_v2_run_dir / "phase15" / "province_loading_deviations.json", default={})
    identification = read_json(rescue_v2_run_dir / "phase15" / "province_block_identification_report.json", default={})
    tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "province_block_state_tensor.npz")
    region_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "region_block_state_tensor.npz")
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})

    assert manifest.get("artifact_paths", {}).get("province_block_state_tensor")
    assert manifest.get("artifact_paths", {}).get("province_block_states")
    assert manifest.get("artifact_paths", {}).get("province_block_uncertainty")
    assert manifest.get("artifact_paths", {}).get("province_loading_deviations")
    assert states.get("method") == "province_factor_graph_scaffold_v1"
    assert uncertainty.get("method") == "province_factor_graph_scaffold_v1"
    assert deviations.get("method") == "province_factor_graph_scaffold_v1"
    assert identification.get("is_scaffold") is True
    assert tensor.shape[:2] == (len(axis_catalogs.get("province", [])), len(axis_catalogs.get("month", [])))
    assert np.isfinite(tensor).all()
    assert np.isfinite(region_tensor).all()
    for row in states.get("rows", []):
        assert len(row.get("state_values", [])) == len(states.get("month_axis", []))
    for row in uncertainty.get("rows", []):
        assert len(row.get("posterior_std_values", [])) == len(uncertainty.get("month_axis", []))


def test_phase15_source_reliability_and_stability_bounds(rescue_v2_run_dir) -> None:
    source_reliability = read_json(rescue_v2_run_dir / "phase15" / "source_reliability.json", default={})
    normalized_rows = read_json(rescue_v2_run_dir / "phase1" / "normalized_subparameters.json", default=[])
    stability_rows = read_json(rescue_v2_run_dir / "phase15" / "factor_stability_report.json", default=[])
    promotion_pool = read_json(rescue_v2_run_dir / "phase15" / "factor_promotion_pool.json", default=[])

    assert sum(row["row_count"] for row in source_reliability.get("rows", [])) == len(normalized_rows)
    assert stability_rows
    assert promotion_pool
    allowed_classes = {"discarded", "reserve", "survivor_primary", "survivor_secondary"}
    for row in stability_rows:
        for key in (
            "predictive_gain",
            "subnational_anomaly_gain",
            "region_contrast_score",
            "sign_stability",
            "predictive_gain_stability",
            "missing_data_robustness",
            "source_dropout_robustness",
            "stability_score",
            "calibration_score",
            "sparsity_penalty",
            "resampling_stability_penalty",
            "survival_score",
        ):
            assert 0.0 <= float(row[key]) <= 1.0
        assert isinstance(bool(row["hard_checks_passed"]), bool)
        assert isinstance(bool(row["survives_holdout"]), bool)
    for row in promotion_pool:
        assert row["promotion_class"] in allowed_classes


def test_phase15_bayesian_survival_artifacts(rescue_v2_run_dir) -> None:
    baseline_rows = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_tournament_baseline.json", default=[])
    baseline_pool = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_pool_baseline.json", default=[])
    optimized_rows = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_tournament_optimized.json", default=[])
    optimized_pool = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_pool_optimized.json", default=[])
    active_rows = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_tournament.json", default=[])
    active_pool = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_pool.json", default=[])
    bayes_report = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_bayesian_optimization.json", default={})

    assert baseline_rows
    assert baseline_pool
    assert optimized_rows
    assert optimized_pool
    assert active_rows
    assert active_pool
    assert "enabled" in bayes_report
    assert "active_variant" in bayes_report
    if bayes_report.get("enabled"):
        assert bayes_report.get("trial_count", 0) >= 1
        assert "best_objective" in bayes_report
        assert "default_objective" in bayes_report
        assert "best_score_params" in bayes_report
        assert "best_primary_per_block" in bayes_report
        assert "best_secondary_per_block" in bayes_report
        assert "best_representation_mix" in bayes_report
        assert bayes_report["active_variant"] in {"baseline", "optimized"}


def test_phase15_relationship_explorer_artifacts(rescue_v2_run_dir) -> None:
    relationship_index = read_json(rescue_v2_run_dir / "phase15" / "semantic_relationship_index.json", default={})
    manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})

    assert relationship_index.get("relationship_count", 0) >= 1
    assert relationship_index.get("rows")
    assert manifest.get("artifact_paths", {}).get("semantic_relationship_index")
    assert "semantic_relationship_bubble_chart" in manifest.get("artifact_paths", {})


def test_phase15_multiscale_factor_artifacts(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})
    factor_catalog = read_json(rescue_v2_run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    multiscale_catalog = read_json(rescue_v2_run_dir / "phase15" / "multiscale_factor_catalog.json", default=[])
    multiscale_axes = read_json(rescue_v2_run_dir / "phase15" / "multiscale_factor_axes.json", default={})
    multiscale_summary = read_json(rescue_v2_run_dir / "phase15" / "multiscale_factor_summary.json", default={})
    province_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "multiscale_province_factor_tensor.npz")
    region_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "multiscale_region_factor_tensor.npz")
    national_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "multiscale_national_factor_tensor.npz")

    assert len(multiscale_catalog) == len(factor_catalog)
    assert province_tensor.shape[-1] == len(factor_catalog)
    assert region_tensor.shape[-1] == len(factor_catalog)
    assert national_tensor.shape[-1] == len(factor_catalog)
    assert np.isfinite(province_tensor).all()
    assert np.isfinite(region_tensor).all()
    assert np.isfinite(national_tensor).all()
    assert multiscale_axes.get("factor") == [row.get("factor_id") for row in factor_catalog]
    assert len(multiscale_axes.get("national", [])) == 1
    assert multiscale_summary.get("enabled") is True
    assert "scale_summary" in multiscale_summary
    assert manifest.get("artifact_paths", {}).get("multiscale_factor_catalog")
    assert manifest.get("artifact_paths", {}).get("multiscale_province_factor_tensor")


def test_phase15_survivors_must_pass_holdout(rescue_v2_run_dir) -> None:
    promotion_pool = read_json(rescue_v2_run_dir / "phase15" / "factor_survival_pool.json", default=[])

    promoted = [
        row for row in promotion_pool if row.get("promotion_class") in {"survivor_primary", "survivor_secondary"}
    ]
    if promoted:
        assert all(bool(row.get("survives_holdout")) for row in promoted)
    else:
        assert all(
            (not bool(row.get("hard_checks_passed"))) or (not bool(row.get("survives_holdout")))
            for row in promotion_pool
        )


def test_phase15_network_features_and_operator_contract(rescue_v2_run_dir) -> None:
    factor_catalog = read_json(rescue_v2_run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    network_bundle = read_json(rescue_v2_run_dir / "phase15" / "network_graph_bundle.json", default={})
    operator_catalog = read_json(rescue_v2_run_dir / "phase15" / "network_operator_catalog.json", default=[])
    network_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "network_feature_tensor.npz")
    operator_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase15" / "network_operator_tensor.npz")

    families = {row.get("network_feature_family") for row in factor_catalog if row.get("factor_class") == "network_feature"}
    assert {"reaction_diffusion", "percolation_fragility", "information_propagation"}.issubset(families)
    assert network_tensor.shape[-1] >= 3
    assert operator_tensor.shape[0] == 3
    assert {row["operator_name"] for row in operator_catalog} == {"mobility_operator", "service_operator", "information_operator"}
    assert len(network_bundle.get("graphs", [])) == 3
    assert np.isfinite(network_tensor).all()
    assert np.isfinite(operator_tensor).all()


def test_phase15_overload_is_bounded_by_factorization(rescue_v2_run_dir) -> None:
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})
    factor_catalog = read_json(rescue_v2_run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    graph = read_json(rescue_v2_run_dir / "phase15" / "candidate_similarity_graph.json", default={})
    canonical_count = len(axis_catalogs.get("canonical_name", []))
    assert len(factor_catalog) <= canonical_count + 20
    assert graph.get("edge_count", 0) <= max(1, canonical_count * 4)
