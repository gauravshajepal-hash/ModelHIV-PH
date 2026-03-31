from __future__ import annotations

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase15_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase15_cfg = (plugin.constraint_settings or {}).get("phase15", {})

    assert phase15_cfg.get("similarity_weights", {}).get("signature_cosine") is not None
    assert phase15_cfg.get("similarity_quality", {}).get("within_block_threshold") is not None
    assert phase15_cfg.get("factor_extraction", {}).get("svd_eps") is not None
    assert phase15_cfg.get("network_graph", {}).get("adjacency_top_k") is not None
    assert phase15_cfg.get("network_graph", {}).get("positive_edge_threshold") is not None
    assert phase15_cfg.get("stability", {}).get("permutation_draws") is not None
    assert phase15_cfg.get("stability", {}).get("score_weights", {}).get("predictive_gain") is not None
    assert phase15_cfg.get("stability", {}).get("tournament_weights", {}).get("mae_improvement") is not None
    assert phase15_cfg.get("stability", {}).get("penalty_weights", {}).get("sparsity") is not None
    assert phase15_cfg.get("stability", {}).get("require_holdout_survival_for_promotion") is True
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("trials") is not None
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("primary_survivors_range") is not None
    assert phase15_cfg.get("stability", {}).get("bayesian_optimization", {}).get("representation_mix_bonus_scale") is not None


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
