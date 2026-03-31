from __future__ import annotations

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase2 import run_phase2_merge_shard_summaries
from epigraph_ph.phase2.pipeline import (
    _has_directed_cycle,
    _notears_optimize,
    _phase2_feature_matrix,
    _safe_feature_corr,
    _skeleton_from_corr,
)
from epigraph_ph.runtime import ROOT_DIR, load_tensor_artifact, read_json, write_json


ALLOWED_TRANSITION_HOOKS = {
    "diagnosis_transitions",
    "linkage_transitions",
    "retention_attrition_transitions",
    "suppression_transitions",
    "subgroup_allocation_priors",
}


def test_phase2_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase2_cfg = (plugin.constraint_settings or {}).get("phase2", {})

    assert phase2_cfg.get("skeleton_threshold") is not None
    assert phase2_cfg.get("mi_prefilter", {}).get("mi_threshold") is not None
    assert phase2_cfg.get("pc_skeleton", {}).get("alpha") is not None
    assert phase2_cfg.get("lag_rules", {}).get("blocked_pathway_pairs") is not None
    assert phase2_cfg.get("execution", {}).get("interop_mode") is not None
    assert phase2_cfg.get("blanket_edge_threshold") is not None
    assert phase2_cfg.get("linkage_support_denominator") is not None
    assert phase2_cfg.get("shortlist_per_block") is not None
    assert phase2_cfg.get("notears", {}).get("optimizer_lr") is not None
    assert phase2_cfg.get("notears", {}).get("optimizer_kind") is not None
    assert phase2_cfg.get("notears", {}).get("steps") is not None
    assert phase2_cfg.get("time_stratified_cv", {}).get("folds") is not None
    assert phase2_cfg.get("collinearity", {}).get("high_corr_threshold") is not None
    assert phase2_cfg.get("stability_score_weights", {}).get("source_diversity") is not None
    assert phase2_cfg.get("stability_denominators", {}).get("source_diversity") is not None
    assert phase2_cfg.get("curation_status_thresholds", {}).get("promoted_candidate") is not None
    assert phase2_cfg.get("factor_diagnostic_weights", {}).get("stability_score") is not None
    assert phase2_cfg.get("benchmark_artifacts", {}).get("bootstrap_draws") is not None
    assert phase2_cfg.get("budgets", {}).get("main") is not None
    assert phase2_cfg.get("block_graph", {}).get("max_candidates_per_block") is not None
    assert phase2_cfg.get("block_graph", {}).get("bridge_edge_budget_per_block_pair") is not None
    assert phase2_cfg.get("block_graph", {}).get("max_target_factors") is not None


def test_phase2_legacy_masks_and_dag_contract(legacy_full_run_dir) -> None:
    manifest = read_json(legacy_full_run_dir / "phase2" / "phase2_manifest.json", default={})
    truth_summary = read_json(legacy_full_run_dir / "phase2" / "ground_truth_summary.json", default={})
    candidate_profiles = read_json(legacy_full_run_dir / "phase2" / "candidate_profiles.json", default=[])
    blanket = read_json(legacy_full_run_dir / "phase2" / "markov_blanket.json", default={})
    bootstrap = read_json(legacy_full_run_dir / "phase2" / "bootstrap_edge_stability.json", default={})
    permutation = read_json(legacy_full_run_dir / "phase2" / "permutation_null_report.json", default={})
    time_cv = read_json(legacy_full_run_dir / "phase2" / "time_stratified_cv_report.json", default={})
    collinearity = read_json(legacy_full_run_dir / "phase2" / "collinearity_report.json", default={})
    interop = read_json(legacy_full_run_dir / "phase2" / "interop_report.json", default={})
    mi_prefilter = read_json(legacy_full_run_dir / "phase2" / "mi_prefilter_report.json", default={})
    pc_skeleton = read_json(legacy_full_run_dir / "phase2" / "pc_skeleton_report.json", default={})
    block_banks = read_json(legacy_full_run_dir / "phase2" / "block_candidate_banks.json", default={})
    block_bundle = read_json(legacy_full_run_dir / "phase2" / "block_graph_bundle.json", default={})
    pruning = read_json(legacy_full_run_dir / "phase2" / "pruning_report.json", default={})
    tier_mask = load_tensor_artifact(legacy_full_run_dir / "phase2" / "tier_mask.npz")
    lag_mask = load_tensor_artifact(legacy_full_run_dir / "phase2" / "lag_mask.npz")
    skeleton_mask = load_tensor_artifact(legacy_full_run_dir / "phase2" / "skeleton_mask.npz")
    dag = load_tensor_artifact(legacy_full_run_dir / "phase2" / "dag_adjacency.npz")

    assert manifest.get("stage_status", {}).get("phase2") == "completed"
    assert candidate_profiles
    assert blanket.get("blanket_nodes", [])
    assert tier_mask.shape == lag_mask.shape == skeleton_mask.shape == dag.shape
    assert np.isfinite(dag).all()
    assert np.allclose(np.diag(dag), 0.0, atol=1e-6)
    assert _has_directed_cycle(dag) is False
    scores = [row["curation_score"] for row in candidate_profiles]
    assert scores == sorted(scores, reverse=True)
    assert bootstrap.get("available") is True
    assert permutation.get("available") is True
    assert mi_prefilter.get("enabled") is True
    assert mi_prefilter.get("kept_feature_count", 0) > 0
    assert pc_skeleton.get("available") is True
    assert "edge_count" in pc_skeleton
    assert time_cv.get("available") is True
    assert time_cv.get("fold_count", 0) >= 1
    assert collinearity.get("available") is True
    assert "condition_number" in collinearity
    assert "used_dlpack" in interop
    assert block_banks.get("rows")
    assert block_bundle.get("blocks")
    assert pruning.get("candidate_level_pruning", {}).get("retained_candidate_count", 0) >= 0
    assert np.sum(lag_mask == 0.0) > dag.shape[0]
    assert truth_summary.get("phase_name") == "phase2"


def test_phase2_rescue_v2_tournament_budgets(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase2" / "phase2_manifest.json", default={})
    truth_summary = read_json(rescue_v2_run_dir / "phase2" / "ground_truth_summary.json", default={})
    tournament_plan = read_json(rescue_v2_run_dir / "phase2" / "factor_tournament_plan.json", default={})
    tournament_results = read_json(rescue_v2_run_dir / "phase2" / "factor_tournament_results.json", default={})
    promoted = read_json(rescue_v2_run_dir / "phase2" / "promoted_factor_set.json", default=[])
    supporting = read_json(rescue_v2_run_dir / "phase2" / "supporting_factor_set.json", default=[])
    admission = read_json(rescue_v2_run_dir / "phase2" / "promotion_admission.json", default={})
    budget_report = read_json(rescue_v2_run_dir / "phase2" / "promotion_budget_report.json", default={})

    assert manifest.get("profile_id") == "hiv_rescue_v2"
    assert tournament_plan
    assert tournament_results
    assert len(promoted) <= 8
    assert len(supporting) <= 12
    assert budget_report.get("main_predictive_count", len(promoted)) <= 8
    assert budget_report.get("supporting_count", len(supporting)) <= 12
    assert all(plan["shortlist_count"] <= 6 for plan in tournament_plan)
    assert admission.get("status") in {"admitted_main_predictive", "none_admitted"}
    assert truth_summary.get("phase_name") == "phase2"


def test_phase2_transition_hook_whitelist_and_network_budget(rescue_v2_run_dir) -> None:
    promoted = read_json(rescue_v2_run_dir / "phase2" / "promoted_factor_set.json", default=[])
    supporting = read_json(rescue_v2_run_dir / "phase2" / "supporting_factor_set.json", default=[])
    diagnostics = read_json(rescue_v2_run_dir / "phase2" / "factor_diagnostics.json", default=[])
    bridge_report = read_json(rescue_v2_run_dir / "phase2" / "bridge_dag_report.json", default={})
    target_blankets = read_json(rescue_v2_run_dir / "phase2" / "phase3_target_blankets.json", default={})
    promoted_network = 0
    for row in promoted + supporting:
        assert set(row.get("transition_hooks", [])).issubset(ALLOWED_TRANSITION_HOOKS)
        if row.get("network_feature_family"):
            promoted_network += 1 if row in promoted else 0
        assert row.get("promotion_class") in {"main_predictive", "supporting_context"}
    assert promoted_network <= 3
    assert len(diagnostics) >= len(promoted)
    assert bridge_report.get("status") in {"completed", "unavailable"}
    assert "blanket_factor_ids" in target_blankets


def test_phase2_no_exploratory_factor_is_promoted(rescue_v2_run_dir) -> None:
    pool = read_json(rescue_v2_run_dir / "phase15" / "factor_promotion_pool.json", default=[])
    promoted = read_json(rescue_v2_run_dir / "phase2" / "promoted_factor_set.json", default=[])
    pool_by_id = {row["factor_id"]: row for row in pool}
    for row in promoted:
        assert pool_by_id[row["factor_id"]]["eligible_main_predictive"] is True
        assert pool_by_id[row["factor_id"]]["promotion_class"] != "exploratory"


def test_phase2_explicitly_admits_when_no_main_predictive_exists(rescue_v2_run_dir) -> None:
    promoted = read_json(rescue_v2_run_dir / "phase2" / "promoted_factor_set.json", default=[])
    admission = read_json(rescue_v2_run_dir / "phase2" / "promotion_admission.json", default={})

    if promoted:
        assert admission.get("status") == "admitted_main_predictive"
        assert admission.get("main_predictive_factor_ids", []) != []
    else:
        assert admission.get("status") == "none_admitted"
        assert admission.get("reason")


def test_phase2_safe_feature_corr_preserves_square_shape_for_single_feature() -> None:
    matrix = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    corr = _safe_feature_corr(matrix)
    skeleton = _skeleton_from_corr(matrix)

    assert corr.shape == (1, 1)
    assert skeleton.shape == (1, 1)
    assert np.isfinite(corr).all()
    assert np.allclose(corr, 0.0, atol=1e-6)
    assert np.allclose(skeleton, 0.0, atol=1e-6)


def test_phase2_notears_single_feature_never_drops_matrix_shape() -> None:
    matrix = np.asarray([[0.2], [0.4], [0.6], [0.8]], dtype=np.float32)
    adjacency = _notears_optimize(matrix, np.ones((1, 1), dtype=np.float32), steps=8)

    assert adjacency.shape == (1, 1)
    assert np.isfinite(adjacency).all()
    assert float(adjacency[0, 0]) == 0.0


def test_phase2_feature_matrix_blends_soft_candidates_without_dropping_shape() -> None:
    standardized = np.zeros((2, 3, 2), dtype=np.float32)
    standardized[:, :, 0] = np.asarray([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=np.float32)
    normalized_rows = [
        {
            "canonical_name": "testing_uptake",
            "time": "2021-01",
            "year": 2021,
            "model_numeric_value": None,
            "evidence_weight": 0.8,
            "bias_penalty": 0.1,
            "replication_weight": 0.6,
        },
        {
            "canonical_name": "testing_uptake",
            "time": "2022-01",
            "year": 2022,
            "model_numeric_value": None,
            "evidence_weight": 0.9,
            "bias_penalty": 0.1,
            "replication_weight": 0.7,
        },
        {
            "canonical_name": "case_count",
            "time": "2021-01",
            "year": 2021,
            "model_numeric_value": 0.4,
            "evidence_weight": 1.0,
            "bias_penalty": 0.0,
            "replication_weight": 1.0,
        },
    ]
    matrix, report = _phase2_feature_matrix(
        standardized_tensor=standardized,
        normalized_rows=normalized_rows,
        canonical_axis=["case_count", "testing_uptake"],
        month_axis=["2021-01", "2022-01", "2023-01"],
        province_axis=["Region I", "Region II"],
    )

    assert matrix.shape == (6, 2)
    assert np.isfinite(matrix).all()
    assert report["used_soft_candidates"] is True
    assert report["soft_feature_count"] >= 1
    assert report["matrix_rows"] == 6
    assert report["matrix_columns"] == 2


def test_phase2_merge_shard_summaries_aggregates_retained_factors() -> None:
    run_ids = [
        "pytest-phase2-merge-shard-a",
        "pytest-phase2-merge-shard-b",
        "pytest-phase2-merge-merged",
    ]
    for run_id in run_ids:
        run_dir = ROOT_DIR / "artifacts" / "runs" / run_id
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)

    shard_a_dir = ROOT_DIR / "artifacts" / "runs" / "pytest-phase2-merge-shard-a" / "phase2"
    shard_b_dir = ROOT_DIR / "artifacts" / "runs" / "pytest-phase2-merge-shard-b" / "phase2"
    shard_a_dir.mkdir(parents=True, exist_ok=True)
    shard_b_dir.mkdir(parents=True, exist_ok=True)

    shared_row = {
        "factor_id": "f_shared",
        "factor_class": "network",
        "block_name": "testing",
        "factor_name": "testing_uptake",
        "interpretability_label": "Testing uptake",
        "best_target": "diagnosed_share",
        "network_feature_family": "testing",
        "transition_hooks": ["diagnosis_transitions"],
        "member_canonical_names": ["testing_uptake", "facility_access"],
        "promotion_class": "main_predictive",
        "phase3_target_relevant": True,
        "predictive_gain": 0.7,
        "stability_score": 0.8,
        "region_contrast_score": 0.4,
        "subnational_anomaly_gain": 0.3,
        "phase3_target_score": 0.75,
    }
    bridge_row_a = {
        "factor_id": "f_a",
        "factor_class": "network",
        "block_name": "mobility",
        "factor_name": "mobility_network_mixing",
        "interpretability_label": "Mobility mixing",
        "best_target": "diagnosed_share",
        "network_feature_family": "mobility",
        "transition_hooks": ["diagnosis_transitions"],
        "member_canonical_names": ["mobility_network_mixing"],
        "promotion_class": "supporting_context",
        "phase3_target_relevant": True,
        "predictive_gain": 0.5,
        "stability_score": 0.6,
        "region_contrast_score": 0.4,
        "subnational_anomaly_gain": 0.2,
        "phase3_target_score": 0.6,
    }
    bridge_row_b = {
        "factor_id": "f_b",
        "factor_class": "network",
        "block_name": "policy",
        "factor_name": "policy_implementation_weakness",
        "interpretability_label": "Policy weakness",
        "best_target": "diagnosed_share",
        "network_feature_family": "policy",
        "transition_hooks": ["diagnosis_transitions"],
        "member_canonical_names": ["policy_implementation_weakness"],
        "promotion_class": "supporting_context",
        "phase3_target_relevant": True,
        "predictive_gain": 0.55,
        "stability_score": 0.65,
        "region_contrast_score": 0.35,
        "subnational_anomaly_gain": 0.25,
        "phase3_target_score": 0.61,
    }

    write_json(shard_a_dir / "retained_mesoscopic_factor_catalog.json", {"rows": [shared_row, bridge_row_a, bridge_row_b]})
    write_json(
        shard_a_dir / "phase3_target_blankets.json",
        {
            "target_factor_ids": ["f_shared"],
            "blanket_factor_ids": ["f_shared", "f_a", "f_b"],
            "blanket_indices": [0, 1, 2],
            "phase3_member_canonical_names": ["testing_uptake", "mobility_network_mixing", "policy_implementation_weakness"],
        },
    )
    write_json(shard_b_dir / "retained_mesoscopic_factor_catalog.json", {"rows": [dict(shared_row, factor_id="f_shared_2"), dict(bridge_row_a, factor_id="f_a_2")]})
    write_json(
        shard_b_dir / "phase3_target_blankets.json",
        {
            "target_factor_ids": ["f_shared_2"],
            "blanket_factor_ids": ["f_shared_2", "f_a_2"],
            "blanket_indices": [0, 1],
            "phase3_member_canonical_names": ["testing_uptake", "mobility_network_mixing"],
        },
    )

    result = run_phase2_merge_shard_summaries(
        run_id="pytest-phase2-merge-merged",
        plugin_id="hiv",
        source_run_ids=["pytest-phase2-merge-shard-a", "pytest-phase2-merge-shard-b"],
        bridge_edge_budget_per_block_pair=1,
    )

    merged = read_json(ROOT_DIR / "artifacts" / "runs" / "pytest-phase2-merge-merged" / "phase2" / "merged_retained_factor_summary.json", default={})
    bridge = read_json(ROOT_DIR / "artifacts" / "runs" / "pytest-phase2-merge-merged" / "phase2" / "merged_bridge_summary.json", default={})
    blanket = read_json(ROOT_DIR / "artifacts" / "runs" / "pytest-phase2-merge-merged" / "phase2" / "merged_phase3_target_blanket_summary.json", default={})

    assert result["summary"]["merged_factor_count"] == 3
    shared = next(row for row in merged["rows"] if row["factor_name"] == "testing_uptake")
    assert shared["shard_count"] == 2
    assert shared["phase3_blanket_shard_count"] == 2
    assert bridge["retained_edge_count"] >= 1
    assert "testing_uptake" in blanket["member_canonical_names"]
