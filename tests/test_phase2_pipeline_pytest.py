from __future__ import annotations

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase2.pipeline import _notears_optimize, _safe_feature_corr, _skeleton_from_corr
from epigraph_ph.runtime import load_tensor_artifact, read_json


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
    assert phase2_cfg.get("blanket_edge_threshold") is not None
    assert phase2_cfg.get("linkage_support_denominator") is not None
    assert phase2_cfg.get("shortlist_per_block") is not None
    assert phase2_cfg.get("notears", {}).get("optimizer_lr") is not None
    assert phase2_cfg.get("notears", {}).get("steps") is not None
    assert phase2_cfg.get("stability_score_weights", {}).get("source_diversity") is not None
    assert phase2_cfg.get("stability_denominators", {}).get("source_diversity") is not None
    assert phase2_cfg.get("curation_status_thresholds", {}).get("promoted_candidate") is not None
    assert phase2_cfg.get("factor_diagnostic_weights", {}).get("stability_score") is not None
    assert phase2_cfg.get("budgets", {}).get("main") is not None


def test_phase2_legacy_masks_and_dag_contract(legacy_full_run_dir) -> None:
    manifest = read_json(legacy_full_run_dir / "phase2" / "phase2_manifest.json", default={})
    truth_summary = read_json(legacy_full_run_dir / "phase2" / "ground_truth_summary.json", default={})
    candidate_profiles = read_json(legacy_full_run_dir / "phase2" / "candidate_profiles.json", default=[])
    blanket = read_json(legacy_full_run_dir / "phase2" / "markov_blanket.json", default={})
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
    scores = [row["curation_score"] for row in candidate_profiles]
    assert scores == sorted(scores, reverse=True)
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
    promoted_network = 0
    for row in promoted + supporting:
        assert set(row.get("transition_hooks", [])).issubset(ALLOWED_TRANSITION_HOOKS)
        if row.get("network_feature_family"):
            promoted_network += 1 if row in promoted else 0
        assert row.get("promotion_class") in {"main_predictive", "supporting_context"}
    assert promoted_network <= 3
    assert len(diagnostics) >= len(promoted)


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
