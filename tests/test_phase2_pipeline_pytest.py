from __future__ import annotations

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3.shared.phase2_inputs import load_phase2_compatibility_payload
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase2_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase2_cfg = (plugin.constraint_settings or {}).get("phase2", {})

    assert phase2_cfg.get("multiscale_dag", {}).get("enabled") is True
    assert phase2_cfg.get("multiscale_dag", {}).get("temporal_graph", {}).get("candidate_max_lags") is not None
    assert phase2_cfg.get("multiscale_dag", {}).get("temporal_graph", {}).get("lambda_s_fractions") is not None
    assert phase2_cfg.get("multiscale_dag", {}).get("temporal_graph", {}).get("lambda_l_fractions") is not None
    assert phase2_cfg.get("multiscale_dag", {}).get("temporal_graph", {}).get("selection_metric") == "weighted_bic"
    assert phase2_cfg.get("multiscale_dag", {}).get("uncertainty_mode") == "parent_block_propagation"
    assert phase2_cfg.get("latent_temporal_graph", {}).get("enabled") is True
    assert max(phase2_cfg.get("latent_temporal_graph", {}).get("candidate_max_lags", [0])) >= 2
    assert phase2_cfg.get("latent_temporal_graph", {}).get("lambda_s_fractions") is not None
    assert phase2_cfg.get("latent_temporal_graph", {}).get("lambda_l_fractions") is not None
    assert phase2_cfg.get("latent_temporal_graph", {}).get("selection_metric") == "weighted_bic"
    assert phase2_cfg.get("latent_temporal_graph", {}).get("bootstrap_draws") is not None
    assert phase2_cfg.get("latent_temporal_graph", {}).get("null_permutations") is not None
    assert phase2_cfg.get("phase3_compatibility", {}).get("predictive_budget") is not None
    assert phase2_cfg.get("phase3_compatibility", {}).get("context_budget") is not None


def test_phase2_legacy_build_keeps_phase3_compatibility_surface(legacy_full_run_dir) -> None:
    manifest = read_json(legacy_full_run_dir / "phase2" / "phase2_manifest.json", default={})
    candidate_profiles = read_json(legacy_full_run_dir / "phase2" / "candidate_profiles.json", default=[])
    curated_blocks = read_json(legacy_full_run_dir / "phase2" / "curated_candidate_blocks.json", default=[])
    markov_blanket = read_json(legacy_full_run_dir / "phase2" / "markov_blanket.json", default={})
    multiscale_bundle = read_json(legacy_full_run_dir / "phase2" / "multiscale_dag_bundle.json", default={})
    latent_bundle = read_json(legacy_full_run_dir / "phase2" / "latent_temporal_graph_bundle.json", default={})
    phase3_target_blankets = read_json(legacy_full_run_dir / "phase2" / "phase3_target_blankets.json", default={})
    core_tensor = load_tensor_artifact(legacy_full_run_dir / "phase2" / "core_feature_tensor.npz")

    assert manifest.get("stage_status", {}).get("phase2") == "completed"
    assert candidate_profiles
    assert curated_blocks
    assert markov_blanket.get("blanket_nodes", [])
    assert phase3_target_blankets.get("phase3_member_canonical_names", []) != []
    assert core_tensor.ndim == 3
    assert np.isfinite(core_tensor).all()
    assert multiscale_bundle.get("enabled") is True
    assert latent_bundle.get("enabled") is True
    assert set(multiscale_bundle.get("scales", {}).keys()) == {"province", "region", "national"}
    assert set(latent_bundle.get("scales", {}).keys()) == {"province", "region", "national"}


def test_phase2_rescue_v2_emits_restored_multiscale_and_latent_outputs(rescue_v2_run_dir) -> None:
    manifest = read_json(rescue_v2_run_dir / "phase2" / "phase2_manifest.json", default={})
    multiscale_bundle = read_json(rescue_v2_run_dir / "phase2" / "multiscale_dag_bundle.json", default={})
    multiscale_blankets = read_json(rescue_v2_run_dir / "phase2" / "multiscale_phase3_target_blankets.json", default={})
    latent_bundle = read_json(rescue_v2_run_dir / "phase2" / "latent_temporal_graph_bundle.json", default={})
    latent_blankets = read_json(rescue_v2_run_dir / "phase2" / "latent_temporal_phase3_target_blankets.json", default={})
    validation = read_json(rescue_v2_run_dir / "phase2" / "latent_temporal_graph_validation.json", default={})
    retained_predictive = read_json(rescue_v2_run_dir / "phase2" / "retained_predictive_factor_set.json", default=[])
    retained_context = read_json(rescue_v2_run_dir / "phase2" / "retained_context_factor_set.json", default=[])
    retained_catalog = read_json(rescue_v2_run_dir / "phase2" / "retained_mesoscopic_factor_catalog.json", default={})

    assert manifest.get("profile_id") == "hiv_rescue_v2"
    assert multiscale_bundle.get("enabled") is True
    assert set(multiscale_bundle.get("scales", {}).keys()) == {"province", "region", "national"}
    assert multiscale_blankets.get("enabled") is True
    assert set(multiscale_blankets.get("scales", {}).keys()) == {"province", "region", "national"}
    assert latent_bundle.get("enabled") is True
    assert latent_bundle.get("state_source") == "phase15_v2_latent_states"
    assert set(latent_bundle.get("scales", {}).keys()) == {"province", "region", "national"}
    assert latent_blankets.get("enabled") is True
    assert set(latent_blankets.get("scales", {}).keys()) == {"province", "region", "national"}
    assert validation.get("available") is True
    assert validation.get("summary", {}).get("case_count") == 2
    assert len(retained_predictive) <= 8
    assert len(retained_context) <= 12
    assert isinstance(retained_catalog.get("rows", []), list)


def test_phase2_restored_outputs_feed_phase3_compatibility_aliases(rescue_v2_run_dir) -> None:
    candidate_profiles = read_json(rescue_v2_run_dir / "phase2" / "candidate_profiles.json", default=[])
    markov_blanket = read_json(rescue_v2_run_dir / "phase2" / "markov_blanket.json", default={})
    ranked_linkages = read_json(rescue_v2_run_dir / "phase2" / "ranked_linkages.json", default=[])
    edge_scores = read_json(rescue_v2_run_dir / "phase2" / "edge_scores.json", default=[])
    hidden_edge_scores = read_json(rescue_v2_run_dir / "phase2" / "hidden_driver_edge_scores.json", default=[])
    multiscale_edge_scores = read_json(rescue_v2_run_dir / "phase2" / "multiscale_edge_scores.json", default=[])
    eligibility_surfaces = read_json(rescue_v2_run_dir / "phase2" / "phase3_eligibility_surfaces.json", default={})
    frozen_payload = read_json(rescue_v2_run_dir / "phase2" / "phase3_compatibility_payload.json", default={})
    promoted = read_json(rescue_v2_run_dir / "phase2" / "promoted_factor_set.json", default=[])
    supporting = read_json(rescue_v2_run_dir / "phase2" / "supporting_factor_set.json", default=[])
    retained_predictive = read_json(rescue_v2_run_dir / "phase2" / "retained_predictive_factor_set.json", default=[])
    retained_context = read_json(rescue_v2_run_dir / "phase2" / "retained_context_factor_set.json", default=[])
    core_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase2" / "core_feature_tensor.npz")

    assert candidate_profiles
    assert markov_blanket.get("blanket_nodes", []) != []
    assert ranked_linkages != []
    assert edge_scores != []
    assert isinstance(hidden_edge_scores, list)
    assert isinstance(multiscale_edge_scores, list)
    assert set(eligibility_surfaces.keys()) == {"direct_temporal", "hidden_driver", "multiscale_support"}
    assert frozen_payload.get("core_feature_tensor_path")
    assert promoted == retained_predictive
    assert supporting == retained_context
    assert core_tensor.ndim == 3
    assert np.isfinite(core_tensor).all()


def test_phase2_frozen_compatibility_payload_preserves_separated_surfaces(rescue_v2_run_dir) -> None:
    payload = load_phase2_compatibility_payload(rescue_v2_run_dir)

    assert payload.get("compatibility_payload_source") == "frozen_artifact"
    assert payload.get("eligibility_surfaces", {}).get("direct_temporal", {}).get("blanket_nodes", []) == payload.get("markov_blanket", {}).get("blanket_nodes", [])
    assert payload.get("core_feature_tensor_array") is not None
    assert payload.get("direct_feature_tensor_array").shape == payload.get("core_feature_tensor_array").shape
    assert payload.get("hidden_driver_feature_tensor_array").ndim == 3
    assert payload.get("multiscale_support_feature_tensor_array").ndim == 3
    for row in list(payload.get("candidate_profiles") or []):
        assert bool(row.get("blanket_member")) == bool(row.get("direct_temporal_member"))


def test_phase2_emits_cross_phase_trust_audit_and_updates_stage_manifests(rescue_v2_run_dir) -> None:
    trust_audit = read_json(rescue_v2_run_dir / "analysis" / "phase_trust_audit.json", default={})
    phase2_manifest = read_json(rescue_v2_run_dir / "phase2" / "phase2_manifest.json", default={})
    phase0_manifest = read_json(rescue_v2_run_dir / "phase0" / "phase0_manifest.json", default={})
    phase1_manifest = read_json(rescue_v2_run_dir / "phase1" / "phase1_manifest.json", default={})
    phase15_manifest = read_json(rescue_v2_run_dir / "phase15" / "phase15_manifest.json", default={})

    assert trust_audit.get("phase_rows")
    phase2_summary = trust_audit.get("phase_rows", {}).get("phase2", {}).get("summary", {})
    assert phase2_summary.get("latent_completed_scale_count") is not None
    assert phase2_summary.get("multiscale_completed_scale_count") is not None
    assert phase2_manifest.get("artifact_paths", {}).get("phase_trust_audit")
    assert phase2_manifest.get("artifact_paths", {}).get("phase_trust_audit_md")
    for manifest in (phase0_manifest, phase1_manifest, phase15_manifest, phase2_manifest):
        assert manifest.get("artifact_status")
        assert manifest.get("trust_status")
        assert isinstance(manifest.get("trust_summary"), dict)
