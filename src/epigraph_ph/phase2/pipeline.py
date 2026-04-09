from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.models import Phase0BackendStatus
from epigraph_ph.phase15 import PHASE15_PROFILE_ID
from epigraph_ph.phase2.latent_temporal_graph import build_latent_temporal_graph_outputs
from epigraph_ph.phase2.multiscale_dag import build_multiscale_dag_outputs
from epigraph_ph.phase2.phase3_compat import build_phase3_compatibility_artifacts, select_retained_factor_rows
from epigraph_ph.phase2.structural_payload import build_phase2_structural_artifacts
from epigraph_ph.runtime import RunContext, detect_backends, ensure_dir, read_json, utc_now_iso, write_boundary_shape_package, write_gold_standard_package, write_ground_truth_package, write_json
from epigraph_ph.validate.phase_trust_audit import build_phase_trust_audit


_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase2_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase2", {}) or {})


def _phase2_required_section(key: str) -> dict[str, Any]:
    value = _phase2_cfg().get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing HIV phase2 constraint setting: {key}")
    return dict(value)


def _empty_multiscale_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = {
        "enabled": True,
        "factor_count": 0,
        "scales": {scale: {"scale_name": scale, "status": "unavailable", "factor_count": 0, "edge_count": 0, "edges": []} for scale in ("province", "region", "national")},
    }
    blankets = {
        "enabled": True,
        "merged_blanket_factor_ids": [],
        "merged_target_factor_ids": [],
        "factor_support_rows": [],
        "edge_support_rows": [],
        "phase3_member_canonical_names": [],
        "scales": {scale: {"target_factor_ids": [], "blanket_factor_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []} for scale in ("province", "region", "national")},
    }
    return bundle, blankets


def _empty_latent_bundle(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = {
        "enabled": True,
        "state_source": "phase15_v2_latent_states",
        "block_axis": [],
        "max_lag": int(cfg.get("max_lag", 1)),
        "scales": {scale: {"scale_name": scale, "status": "unavailable", "reason": "phase15_missing"} for scale in ("province", "region", "national")},
    }
    blankets = {
        "enabled": True,
        "merged_target_block_ids": [],
        "merged_blanket_block_ids": [],
        "phase3_member_canonical_names": [],
        "block_support_rows": [],
        "edge_support_rows": [],
        "hidden_driver_support_rows": [],
        "scales": {scale: {"target_block_ids": [], "blanket_block_ids": [], "blanket_indices": [], "phase3_member_canonical_names": []} for scale in ("province", "region", "national")},
    }
    validation = {"available": False, "reason": "phase15_missing"}
    return bundle, blankets, validation


def _phase2_backend_status() -> dict[str, dict[str, Any]]:
    backends = detect_backends()
    return {
        name: Phase0BackendStatus(
            name=name,
            available=bool(payload.available),
            selected=bool(payload.available),
            notes=str(payload.notes or ""),
        ).to_dict()
        for name, payload in backends.items()
    }


def _gold_profile() -> dict[str, Any]:
    return {
        "mode": "restored_contract_validation",
        "truth_claim_level": "artifact_and_interface",
        "description": "Phase 2 restores the latent-temporal graph and multiscale factor contract and emits the compatibility artifacts Phase 3 still consumes.",
        "benchmark_policy": "contract_first",
        "standards": [
            {"name": "latent_temporal_graph_bundle", "expectation": "present"},
            {"name": "multiscale_dag_bundle", "expectation": "present"},
            {"name": "phase3_compatibility_artifacts", "expectation": "present"},
        ],
        "judging_principles": ["preserve downstream contracts", "prefer latent-state structure over NOTEARS feature graphs"],
        "required_claims": ["phase2_restored", "phase3_legacy_compatible"],
        "related_layers": ["phase1", "phase15", "phase3"],
    }


def run_phase2_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase2_dir_path = ctx.run_dir / "phase2"
    if phase2_dir_path.exists():
        shutil.rmtree(phase2_dir_path, ignore_errors=True)
    phase2_dir = ensure_dir(phase2_dir_path)
    phase1_dir = ctx.run_dir / "phase1"
    phase15_dir = ctx.run_dir / "phase15"
    multiscale_cfg = _phase2_required_section("multiscale_dag")
    latent_cfg = _phase2_required_section("latent_temporal_graph")
    compat_cfg = _phase2_required_section("phase3_compatibility")

    predictive_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    retained_catalog = {
        "rows": [],
        "predictive_count": 0,
        "context_count": 0,
        "selection_method": "phase15_survival_classes_intersect_multiscale_catalog",
    }
    if phase15_dir.exists() and (phase15_dir / "factor_promotion_pool.json").exists():
        predictive_rows, context_rows, retained_catalog = select_retained_factor_rows(
            phase15_dir=phase15_dir,
            predictive_budget=int(compat_cfg.get("predictive_budget", 8)),
            context_budget=int(compat_cfg.get("context_budget", 12)),
        )

    if phase15_dir.exists() and (phase15_dir / "multiscale_factor_axes.json").exists():
        multiscale_bundle, multiscale_blankets = build_multiscale_dag_outputs(
            phase15_dir=phase15_dir,
            retained_factor_rows=[*predictive_rows, *context_rows],
            cfg=multiscale_cfg,
        )
    else:
        multiscale_bundle, multiscale_blankets = _empty_multiscale_bundle()

    if phase15_dir.exists() and (phase15_dir / "phase15_v2_fit_summary.json").exists():
        latent_bundle, latent_blankets, latent_validation = build_latent_temporal_graph_outputs(
            phase15_dir=phase15_dir,
            cfg=latent_cfg,
        )
    else:
        latent_bundle, latent_blankets, latent_validation = _empty_latent_bundle(latent_cfg)

    write_json(phase2_dir / "multiscale_dag_bundle.json", multiscale_bundle)
    write_json(phase2_dir / "multiscale_phase3_target_blankets.json", multiscale_blankets)
    write_json(phase2_dir / "latent_temporal_graph_bundle.json", latent_bundle)
    write_json(phase2_dir / "latent_temporal_phase3_target_blankets.json", latent_blankets)
    write_json(phase2_dir / "latent_temporal_graph_validation.json", latent_validation)

    compat_paths = build_phase3_compatibility_artifacts(
        phase2_dir=phase2_dir,
        phase1_dir=phase1_dir,
        predictive_rows=predictive_rows,
        context_rows=context_rows,
        retained_catalog=retained_catalog,
        multiscale_blankets=multiscale_blankets,
        latent_blankets=latent_blankets,
        multiscale_bundle=multiscale_bundle,
        latent_bundle=latent_bundle,
    )
    structural_paths = build_phase2_structural_artifacts(
        phase2_dir=phase2_dir,
        phase15_dir=phase15_dir,
        latent_bundle=latent_bundle,
        latent_blankets=latent_blankets,
        multiscale_bundle=multiscale_bundle,
        multiscale_blankets=multiscale_blankets,
    )

    artifact_paths = {
        "multiscale_dag_bundle": str(phase2_dir / "multiscale_dag_bundle.json"),
        "multiscale_phase3_target_blankets": str(phase2_dir / "multiscale_phase3_target_blankets.json"),
        "latent_temporal_graph_bundle": str(phase2_dir / "latent_temporal_graph_bundle.json"),
        "latent_temporal_phase3_target_blankets": str(phase2_dir / "latent_temporal_phase3_target_blankets.json"),
        "latent_temporal_graph_validation": str(phase2_dir / "latent_temporal_graph_validation.json"),
        "phase_trust_audit": str(ctx.run_dir / "analysis" / "phase_trust_audit.json"),
        "phase_trust_audit_md": str(ctx.run_dir / "analysis" / "phase_trust_audit.md"),
    }
    artifact_paths.update(compat_paths)
    artifact_paths.update(structural_paths)
    compatibility_manifest = {
        "family": "legacy_phase3_phase4_compatibility_bridge",
        "scientific_role": "non_core_compatibility_only",
        "derivation": "downstream_aliases_from_restored_phase2_outputs",
        "artifacts": compat_paths,
    }
    write_json(phase2_dir / "phase2_compatibility_manifest.json", compatibility_manifest)
    artifact_paths["phase2_compatibility_manifest"] = str(phase2_dir / "phase2_compatibility_manifest.json")
    structural_manifest = {
        "family": "phase2_structural_frontier_payload",
        "scientific_role": "core_structural_frontier_input",
        "derivation": "direct_temporal_hidden_driver_multiscale_support_split",
        "artifacts": structural_paths,
    }
    write_json(phase2_dir / "phase2_structural_manifest.json", structural_manifest)
    artifact_paths["phase2_structural_manifest"] = str(phase2_dir / "phase2_structural_manifest.json")
    manifest = {
        "phase_name": "phase2",
        "profile_id": profile,
        "generated_at": utc_now_iso(),
        "stage_status": {"phase2": "completed"},
        "backend_status": _phase2_backend_status(),
        "artifact_paths": artifact_paths,
        "latent_temporal_completed_scales": sum(1 for row in dict(latent_bundle.get("scales") or {}).values() if row.get("status") == "completed"),
        "multiscale_completed_scales": sum(1 for row in dict(multiscale_bundle.get("scales") or {}).values() if row.get("status") == "completed"),
        "retained_predictive_count": len(predictive_rows),
        "retained_context_count": len(context_rows),
        "restored_contract": "latent_temporal_multiscale_v1",
        "scientific_core_family": "latent_temporal_multiscale",
        "compatibility_family": "legacy_phase3_phase4_compatibility_bridge",
        "phase15_available": phase15_dir.exists(),
    }
    manifest_path = phase2_dir / "phase2_manifest.json"
    write_json(manifest_path, manifest)

    truth_checks = [
        {"name": "candidate_profiles_present", "passed": bool(Path(compat_paths["candidate_profiles"]).exists())},
        {"name": "core_feature_tensor_present", "passed": bool(Path(compat_paths["core_feature_tensor"]).exists())},
        {"name": "latent_temporal_bundle_present", "passed": bool(Path(artifact_paths["latent_temporal_graph_bundle"]).exists())},
        {"name": "multiscale_bundle_present", "passed": bool(Path(artifact_paths["multiscale_dag_bundle"]).exists())},
        {"name": "structural_payload_present", "passed": bool(Path(artifact_paths["phase2_structural_payload"]).exists())},
    ]
    boundary_paths = [{"name": name, "kind": "json_or_tensor", "path": path} for name, path in artifact_paths.items()]
    ground_truth_paths = write_ground_truth_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        checks=truth_checks,
        summary={"phase_name": "phase2", "profile_id": profile},
        profile_id=profile,
        truth_sources=["phase1", "phase15", "phase3_compatibility", "phase2_structural_payload"],
        stage_manifest_path=str(manifest_path),
    )
    gold_paths = write_gold_standard_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        checks=truth_checks,
        gold_profile=_gold_profile(),
        summary={"phase_name": "phase2", "profile_id": profile},
        profile_id=profile,
        stage_manifest_path=str(manifest_path),
    )
    boundary_paths_written = write_boundary_shape_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        boundaries=boundary_paths,
        profile_id=profile,
        summary={"phase_name": "phase2", "profile_id": profile},
    )
    manifest["artifact_paths"].update(ground_truth_paths)
    manifest["artifact_paths"].update(gold_paths)
    manifest["artifact_paths"].update(boundary_paths_written)
    write_json(manifest_path, manifest)

    trust_report = build_phase_trust_audit(run_dir=ctx.run_dir, plugin_id=plugin_id)
    manifest = read_json(manifest_path, default={})
    manifest["artifact_paths"]["phase_trust_audit"] = str(trust_report["artifacts"]["json"])
    manifest["artifact_paths"]["phase_trust_audit_md"] = str(trust_report["artifacts"]["md"])
    write_json(manifest_path, manifest)
    return manifest


__all__ = ["run_phase2_build"]
