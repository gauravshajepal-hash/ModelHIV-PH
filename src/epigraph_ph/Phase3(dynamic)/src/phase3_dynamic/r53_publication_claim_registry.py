from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .runtime import ensure_dir, read_json, write_json


R53_SCHEMA_VERSION = "phase3_dynamic.r53_publication_claim_registry.v1"
R53_RUN_ID = "p3d-r53-publication-claim-registry-20260503-s00"
R42_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r42-r41-champion-hardening-20260502-s00"
    / "analysis"
    / "r42_r41_champion_hardening_report.json"
)
R46_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r47-phase2-current-lineage-driver-gate-20260503-s01"
    / "analysis"
    / "r46_phase2_lineage_driver_gate_report.json"
)
R52_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r52-subnational-coherence-gate-20260503-s00"
    / "analysis"
    / "r52_subnational_coherence_gate_report.json"
)
R54_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r54-national-total-regional-adapter-20260503-s00"
    / "analysis"
    / "r54_national_total_regional_adapter_report.json"
)
R55_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r55-adapter-split-stability-gate-20260503-s00"
    / "analysis"
    / "r55_adapter_split_stability_gate_report.json"
)
R56_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r56-split-guarded-regional-selector-20260503-s00"
    / "analysis"
    / "r56_split_guarded_regional_selector_report.json"
)
R57_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r57-regional-candidate-ceiling-diagnostic-20260503-s00"
    / "analysis"
    / "r57_regional_candidate_ceiling_diagnostic_report.json"
)
R58_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r58-pareto-simplex-regional-ensemble-20260503-s00"
    / "analysis"
    / "r58_pareto_simplex_regional_ensemble_report.json"
)
R59_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r59-anchor-stable-pareto-ensemble-20260503-s00"
    / "analysis"
    / "r59_anchor_stable_pareto_ensemble_report.json"
)
R60_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r60-regional-experiment-queue-20260503-s00"
    / "analysis"
    / "r60_regional_experiment_queue_report.json"
)
R61_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r61-split-risk-regional-router-20260503-s00"
    / "analysis"
    / "r61_split_risk_regional_router_report.json"
)
R62_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r62-leakage-expert-student-gate-20260503-s00"
    / "analysis"
    / "r62_leakage_expert_student_gate_report.json"
)
R63_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r63-region-metric-online-expert-gate-20260503-s00"
    / "analysis"
    / "r63_region_metric_online_expert_gate_report.json"
)
R64_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r64-leakage-support-gap-prioritizer-20260503-s00"
    / "analysis"
    / "r64_leakage_support_gap_prioritizer_report.json"
)
R65_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r65-transmission-model-readiness-gate-20260503-s00"
    / "analysis"
    / "r65_transmission_model_readiness_gate_report.json"
)
R66_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r66-scientific-source-base-20260506-s00"
    / "analysis"
    / "r66_scientific_source_base_report.json"
)
R67_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r67-transmission-model-family-queue-20260506-s00"
    / "analysis"
    / "r67_transmission_model_family_queue_report.json"
)
R68_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r68-bulk-external-source-ingest-20260506-s00"
    / "analysis"
    / "r68_bulk_external_source_ingest_report.json"
)
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r69-bulk-signal-feature-compiler-20260506-s00"
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
R70_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r70-scientific-model-build-queue-20260506-s00"
    / "analysis"
    / "r70_scientific_model_build_queue_report.json"
)
R71_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r71-service-intensity-capacity-branch-20260506-s00"
    / "analysis"
    / "r71_service_intensity_capacity_branch_report.json"
)
R72_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r72-service-feature-selector-gate-20260506-s00"
    / "analysis"
    / "r72_service_feature_selector_gate_report.json"
)
R73_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r73-external-signal-lag-falsification-20260506-s00"
    / "analysis"
    / "r73_external_signal_lag_falsification_report.json"
)
R74_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r74-external-tail-risk-selector-20260506-s00"
    / "analysis"
    / "r74_external_tail_risk_selector_report.json"
)
R75_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r75-bulk-unaids-annual-challenge-20260507-s00"
    / "analysis"
    / "r75_bulk_unaids_annual_challenge_report.json"
)
R76_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r76-public-domain-annual-comparator-20260507-s00"
    / "analysis"
    / "r76_public_domain_annual_comparator_report.json"
)
R77_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r77-public-proxy-annual-gate-20260507-s00"
    / "analysis"
    / "r77_public_proxy_annual_gate_report.json"
)
R78_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r78-public-annual-family-expansion-20260507-s00"
    / "analysis"
    / "r78_public_annual_family_expansion_report.json"
)
R79_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r79-expanded-public-proxy-annual-gate-20260507-s00"
    / "analysis"
    / "r79_expanded_public_proxy_annual_gate_report.json"
)
R80_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r80-public-annual-projection-head-20260507-s00"
    / "analysis"
    / "r80_public_annual_projection_head_report.json"
)
R81_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r81-phase2-knob-admissibility-gate-20260507-s00"
    / "analysis"
    / "r81_phase2_knob_admissibility_gate_report.json"
)
R82_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r82-quarterly-annual-bridge-gate-20260507-s00"
    / "analysis"
    / "r82_quarterly_annual_bridge_gate_report.json"
)
R83_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r83-quarterly-emission-bridge-audit-20260507-s00"
    / "analysis"
    / "r83_quarterly_emission_bridge_audit_report.json"
)
R84_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r84-conserved-quarterly-annual-ledger-20260507-s00"
    / "analysis"
    / "r84_conserved_quarterly_annual_ledger_report.json"
)
R85_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r85-annual-ledger-forecast-grid-20260507-s00"
    / "analysis"
    / "r85_annual_ledger_forecast_grid_report.json"
)
R86_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r86-annual-calibrated-forecast-grid-ledger-20260507-s00"
    / "analysis"
    / "r86_annual_calibrated_forecast_grid_ledger_report.json"
)
R87_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r87-train-backtested-emission-process-calibration-20260507-s00"
    / "analysis"
    / "r87_train_backtested_emission_process_calibration_report.json"
)
R88_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r88-guarded-annual-ledger-selector-20260507-s00"
    / "analysis"
    / "r88_guarded_annual_ledger_selector_report.json"
)
R89_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r89-incidence-mortality-mechanism-support-gate-20260507-s00"
    / "analysis"
    / "r89_incidence_mortality_mechanism_support_gate_report.json"
)
R90_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r90-claim-grade-gate-20260509-s00"
    / "analysis"
    / "r90_claim_grade_gate_report.json"
)
R91_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r91-mechanism-support-expansion-gate-20260509-s00"
    / "analysis"
    / "r91_mechanism_support_expansion_gate_report.json"
)
R92_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r92-process-repair-experiment-queue-20260509-s00"
    / "analysis"
    / "r92_process_repair_experiment_queue_report.json"
)
R93_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r93-open-public-incumbent-comparator-20260510-s00"
    / "analysis"
    / "r93_open_public_incumbent_comparator_report.json"
)


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _national_claim(r42: dict[str, Any], path: Path) -> dict[str, Any]:
    blockers = list(r42.get("blockers") or [])
    strict_status = str(r42.get("strict_gate_status") or "")
    promotion_claim = str(r42.get("promotion_claim") or "")
    promoted = not blockers and strict_status == "pass" and promotion_claim == "freeze_as_national_research_champion"
    return {
        "claim_id": "national_r41_research_champion",
        "claim_scope": "national",
        "claim_status": "promoted" if promoted else "blocked",
        "model_family": r42.get("candidate_family") or "r41_monotone_growth_component_process",
        "primary_gate": strict_status,
        "blockers": blockers,
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R41 is the current national research champion under the strict mapped carry-forward/R10 and annual challenge gates."
            if promoted
            else "National champion claim is blocked until R42 strict gate passes."
        ),
        "claim_limit": "National forecast/readout champion; not a subnational validation claim and not a causal determinant claim.",
    }


def _subnational_claim(r52: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r52.get("coherence_gate") or {})
    blockers = list(r52.get("blockers") or gate.get("blockers") or [])
    promoted = str(r52.get("status") or "") == "coherent_subnational_champion_promoted" and not blockers
    return {
        "claim_id": "regional_hasp_cascade_readout_champion",
        "claim_scope": "regional",
        "claim_status": "promoted" if promoted else "blocked",
        "model_family": gate.get("promoted_candidate_family"),
        "primary_gate": gate.get("status"),
        "blockers": blockers,
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "A regional HASP cascade readout/proxy champion improves regional errors without worsening aggregate mass or regional allocation share versus carry-forward."
            if promoted
            else "Regional claim remains blocked until coherence gate passes."
        ),
        "claim_limit": "Regional cascade readout/proxy claim only; not province-level validation, not a mechanistic regional transition claim, and not determinant-driven scenario validation.",
        "key_metrics": {
            "promoted_mean_regional_normalized_absolute_error": gate.get("promoted_mean_regional_normalized_absolute_error"),
            "promoted_mean_aggregate_mass_normalized_absolute_error": gate.get("promoted_mean_aggregate_mass_normalized_absolute_error"),
            "promoted_mean_regional_share_half_l1_error": gate.get("promoted_mean_regional_share_half_l1_error"),
            "carry_forward_mean_regional_normalized_absolute_error": gate.get("carry_forward_mean_regional_normalized_absolute_error"),
            "carry_forward_mean_aggregate_mass_normalized_absolute_error": gate.get("carry_forward_mean_aggregate_mass_normalized_absolute_error"),
            "carry_forward_mean_regional_share_half_l1_error": gate.get("carry_forward_mean_regional_share_half_l1_error"),
        },
    }


def _regional_adapter_claim(r54: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r54.get("adapter_gate") or {})
    blockers = list(r54.get("blockers") or gate.get("blockers") or [])
    promoted = str(r54.get("status") or "") == "national_total_regional_adapter_promoted" and not blockers
    return {
        "claim_id": "regional_hasp_national_total_adapter_champion",
        "claim_scope": "regional",
        "claim_status": "promoted" if promoted else "blocked",
        "model_family": gate.get("promoted_candidate_family"),
        "primary_gate": gate.get("status"),
        "blockers": blockers,
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "A national-total-constrained regional HASP adapter improves regional error, aggregate mass error, and regional share allocation versus both carry-forward and the R52 regional reference."
            if promoted
            else "Regional national-total adapter claim remains blocked until R54 passes."
        ),
        "claim_limit": "Regional adapter/readout claim only; national totals and regional shares are borrowed from existing forecast candidates, not fitted determinant effects or province-level mechanisms.",
        "key_metrics": {
            "promoted_mean_regional_normalized_absolute_error": gate.get("promoted_mean_regional_normalized_absolute_error"),
            "promoted_mean_aggregate_mass_normalized_absolute_error": gate.get("promoted_mean_aggregate_mass_normalized_absolute_error"),
            "promoted_mean_regional_share_half_l1_error": gate.get("promoted_mean_regional_share_half_l1_error"),
            "r52_reference_mean_regional_normalized_absolute_error": gate.get("r52_reference_mean_regional_normalized_absolute_error"),
            "r52_reference_mean_aggregate_mass_normalized_absolute_error": gate.get("r52_reference_mean_aggregate_mass_normalized_absolute_error"),
            "r52_reference_mean_regional_share_half_l1_error": gate.get("r52_reference_mean_regional_share_half_l1_error"),
        },
    }


def _determinant_claim(r46: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r46.get("lineage_gate") or {})
    status = str(r46.get("status") or gate.get("status") or "")
    strict_count = int(gate.get("strict_phase3_prior_count") or 0)
    sensitivity_count = int(gate.get("sensitivity_only_driver_count") or 0)
    if status == "strict_determinant_priors_ready" and strict_count > 0:
        claim_status = "strict_priors_ready"
        allowed = "Phase 2 determinant edges may enter Phase 3 as strict fitted priors."
    elif status == "sensitivity_only_determinant_scenarios" and sensitivity_count > 0:
        claim_status = "sensitivity_only"
        allowed = "Phase 2 determinant edges may be used only as labeled sensitivity/scenario sliders."
    else:
        claim_status = "blocked"
        allowed = "Phase 2 determinants are blocked from Phase 3 claims."
    return {
        "claim_id": "phase2_determinant_driver_claim",
        "claim_scope": "national_and_regional_scenarios",
        "claim_status": claim_status,
        "model_family": "phase2_direct_and_hidden_structural_payload",
        "primary_gate": status,
        "blockers": list(r46.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "No fitted determinant prior or intervention-effect claim until source-family re-estimation and blocked-time recovery both pass.",
        "key_metrics": {
            "strict_phase3_prior_count": strict_count,
            "sensitivity_only_driver_count": sensitivity_count,
            "completed_source_family_count": gate.get("completed_source_family_count"),
            "completed_source_families": gate.get("completed_source_families"),
        },
    }


def _adapter_stability_claim(r55: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r55.get("split_stability_gate") or {})
    status = str(r55.get("status") or gate.get("status") or "")
    strict = status == "strict_split_stable_adapter_promoted"
    limited = status == "adapter_mean_promoted_but_split_stability_limited"
    if strict:
        claim_status = "split_stable"
        allowed = "The R54 regional adapter is split-stable across all scored period-metric checks."
    elif limited:
        claim_status = "mean_promoted_split_limited"
        allowed = "The R54 regional adapter is mean-promoted, but some period-metric splits regress versus carry-forward or the R52 reference."
    else:
        claim_status = "blocked"
        allowed = "The R54 regional adapter has no split-stability support."
    return {
        "claim_id": "regional_adapter_split_stability_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": r55.get("promoted_candidate_family"),
        "primary_gate": status,
        "blockers": list(r55.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "If split-stability is limited, report R54 as a mean-level adapter improvement with localized regressions, not as uniformly better across all periods and cascade streams.",
        "key_metrics": {
            "split_metric_count": gate.get("split_metric_count"),
            "failure_counts": gate.get("failure_counts"),
            "period_failure_counts": gate.get("period_failure_counts"),
            "metric_failure_counts": gate.get("metric_failure_counts"),
        },
    }


def _split_guarded_selector_claim(r56: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r56.get("selector_gate") or {})
    status = str(r56.get("status") or gate.get("status") or "")
    promoted = status == "split_guarded_regional_selector_promoted"
    diagnostic = status == "split_guarded_regional_selector_diagnostic_only"
    if promoted:
        claim_status = "promoted"
        allowed = "The R56 split-guarded regional selector is promoted as a stricter subnational readout champion."
    elif diagnostic:
        claim_status = "diagnostic_only"
        allowed = "R56 is a negative/diagnostic result: prior split guarding did not beat R52/R54 under the strict contract."
    else:
        claim_status = "blocked"
        allowed = "R56 has no supported subnational selector claim."
    return {
        "claim_id": "regional_split_guarded_selector_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("selector_family") or "split_guarded_regional_selector",
        "primary_gate": status,
        "blockers": list(r56.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Selector/readout stability claim only; it uses prior split diagnostics, not determinant effects or province-level mechanisms.",
        "key_metrics": {
            "selector_mean_regional_normalized_absolute_error": gate.get("selector_mean_regional_normalized_absolute_error"),
            "selector_mean_aggregate_mass_normalized_absolute_error": gate.get("selector_mean_aggregate_mass_normalized_absolute_error"),
            "selector_mean_regional_share_half_l1_error": gate.get("selector_mean_regional_share_half_l1_error"),
            "r54_reference_mean_regional_normalized_absolute_error": gate.get("r54_reference_mean_regional_normalized_absolute_error"),
            "r54_reference_mean_aggregate_mass_normalized_absolute_error": gate.get("r54_reference_mean_aggregate_mass_normalized_absolute_error"),
            "r54_reference_mean_regional_share_half_l1_error": gate.get("r54_reference_mean_regional_share_half_l1_error"),
            "split_stability_status": gate.get("split_stability_status"),
        },
    }


def _regional_candidate_ceiling_claim(r57: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r57.get("ceiling_gate") or {})
    status = str(r57.get("status") or gate.get("status") or "")
    diagnostic = status == "candidate_space_has_unrealized_signal_diagnostic"
    return {
        "claim_id": "regional_candidate_space_ceiling_diagnostic",
        "claim_scope": "regional",
        "claim_status": "diagnostic_only" if diagnostic else "blocked",
        "model_family": gate.get("ceiling_family") or "leakage_labeled_candidate_ceiling",
        "primary_gate": status,
        "blockers": list(r57.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "A leakage-labeled ceiling diagnostic shows that existing regional candidate families contain unrealized signal; this is not a promotable model."
            if diagnostic
            else "No regional candidate-space ceiling claim is supported."
        ),
        "claim_limit": "Diagnostic ceiling only; it uses same-split holdout target errors and must never be reported as train-origin forecast performance.",
        "key_metrics": {
            "ceiling_mean_regional_normalized_absolute_error": gate.get("ceiling_mean_regional_normalized_absolute_error"),
            "ceiling_mean_aggregate_mass_normalized_absolute_error": gate.get("ceiling_mean_aggregate_mass_normalized_absolute_error"),
            "ceiling_mean_regional_share_half_l1_error": gate.get("ceiling_mean_regional_share_half_l1_error"),
            "r54_reference_mean_regional_normalized_absolute_error": gate.get("r54_reference_mean_regional_normalized_absolute_error"),
            "r54_reference_mean_aggregate_mass_normalized_absolute_error": gate.get("r54_reference_mean_aggregate_mass_normalized_absolute_error"),
            "r54_reference_mean_regional_share_half_l1_error": gate.get("r54_reference_mean_regional_share_half_l1_error"),
            "improvement": gate.get("improvement"),
        },
    }


def _regional_pareto_ensemble_claim(r58: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r58.get("ensemble_gate") or {})
    status = str(r58.get("status") or gate.get("status") or "")
    if status == "pareto_simplex_regional_ensemble_promoted":
        claim_status = "promoted"
        allowed = "The R58 Pareto-simplex regional ensemble is promoted as a split-stable regional readout champion."
    elif status == "pareto_simplex_regional_ensemble_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "The R58 Pareto-simplex regional ensemble improves mean regional/mass/share errors versus R54, but split-stability remains limited."
    elif status == "pareto_simplex_regional_ensemble_diagnostic_only":
        claim_status = "diagnostic_only"
        allowed = "R58 is diagnostic-only and does not improve the current regional references."
    else:
        claim_status = "blocked"
        allowed = "R58 has no supported regional ensemble claim."
    return {
        "claim_id": "regional_pareto_simplex_ensemble_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("ensemble_family") or "pareto_simplex_regional_ensemble",
        "primary_gate": status,
        "blockers": list(r58.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Regional readout ensemble only; no determinant, province-level, or causal transition claim. If split-limited, report as mean-level improvement only.",
        "key_metrics": {
            "ensemble_mean_regional_normalized_absolute_error": gate.get("ensemble_mean_regional_normalized_absolute_error"),
            "ensemble_mean_aggregate_mass_normalized_absolute_error": gate.get("ensemble_mean_aggregate_mass_normalized_absolute_error"),
            "ensemble_mean_regional_share_half_l1_error": gate.get("ensemble_mean_regional_share_half_l1_error"),
            "r54_reference_mean_regional_normalized_absolute_error": gate.get("r54_reference_mean_regional_normalized_absolute_error"),
            "r54_reference_mean_aggregate_mass_normalized_absolute_error": gate.get("r54_reference_mean_aggregate_mass_normalized_absolute_error"),
            "r54_reference_mean_regional_share_half_l1_error": gate.get("r54_reference_mean_regional_share_half_l1_error"),
            "split_stability_status": gate.get("split_stability_status"),
        },
    }


def _regional_anchor_stable_ensemble_claim(r59: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r59.get("ensemble_gate") or {})
    status = str(r59.get("status") or gate.get("status") or "")
    if status == "anchor_stable_pareto_ensemble_promoted":
        claim_status = "promoted"
        allowed = "The R59 anchor-stable Pareto ensemble is promoted as a split-stable regional readout champion."
    elif status == "anchor_stable_pareto_ensemble_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "The R59 anchor-stable Pareto ensemble improves mean regional/mass/share errors versus R54, but split-stability remains limited."
    elif status == "anchor_stable_pareto_ensemble_diagnostic_only":
        claim_status = "diagnostic_only"
        allowed = "R59 is diagnostic-only and does not improve the current regional references."
    else:
        claim_status = "blocked"
        allowed = "R59 has no supported regional ensemble claim."
    return {
        "claim_id": "regional_anchor_stable_pareto_ensemble_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("ensemble_family") or "anchor_stable_pareto_ensemble",
        "primary_gate": status,
        "blockers": list(r59.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Regional readout ensemble only; estimated PLHIV is treated as a stock anchor, not a causal regional process. If split-limited, report as mean-level improvement only.",
        "key_metrics": {
            "ensemble_mean_regional_normalized_absolute_error": gate.get("ensemble_mean_regional_normalized_absolute_error"),
            "ensemble_mean_aggregate_mass_normalized_absolute_error": gate.get("ensemble_mean_aggregate_mass_normalized_absolute_error"),
            "ensemble_mean_regional_share_half_l1_error": gate.get("ensemble_mean_regional_share_half_l1_error"),
            "r54_reference_mean_regional_normalized_absolute_error": gate.get("r54_reference_mean_regional_normalized_absolute_error"),
            "r54_reference_mean_aggregate_mass_normalized_absolute_error": gate.get("r54_reference_mean_aggregate_mass_normalized_absolute_error"),
            "r54_reference_mean_regional_share_half_l1_error": gate.get("r54_reference_mean_regional_share_half_l1_error"),
            "split_stability_status": gate.get("split_stability_status"),
        },
    }


def _regional_experiment_queue_claim(r60: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r60.get("queue_gate") or {})
    status = str(r60.get("status") or gate.get("status") or "")
    if status == "regional_experiment_queue_strict_champion_promoted":
        claim_status = "promoted"
        allowed = "The R60 broad regional experiment queue found a strict split-stable regional readout champion."
    elif status == "regional_experiment_queue_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "The R60 broad regional experiment queue improves mean regional/mass/share errors over R59, but no queued candidate is split-stable."
    elif status == "regional_experiment_queue_diagnostic_only":
        claim_status = "diagnostic_only"
        allowed = "R60 is diagnostic-only and did not improve the current regional references."
    else:
        claim_status = "blocked"
        allowed = "R60 has no supported regional experiment-queue claim."
    return {
        "claim_id": "regional_r60_experiment_queue_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("best_candidate_family") or "regional_experiment_queue",
        "primary_gate": status,
        "blockers": list(r60.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Regional readout search claim only; if split-limited, do not claim uniform improvement across all period-metric splits.",
        "key_metrics": {
            "experiment_count": gate.get("experiment_count"),
            "mean_pass_count": gate.get("mean_pass_count"),
            "strict_pass_count": gate.get("strict_pass_count"),
            "best_mean_regional_normalized_absolute_error": gate.get("best_mean_regional_normalized_absolute_error"),
            "best_mean_aggregate_mass_normalized_absolute_error": gate.get("best_mean_aggregate_mass_normalized_absolute_error"),
            "best_mean_regional_share_half_l1_error": gate.get("best_mean_regional_share_half_l1_error"),
            "best_split_gate_status": gate.get("best_split_gate_status"),
        },
    }


def _regional_split_risk_router_claim(r61: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r61.get("router_gate") or {})
    status = str(r61.get("status") or gate.get("status") or "")
    if status == "split_risk_regional_router_strict_champion_promoted":
        claim_status = "promoted"
        allowed = "The R61 split-risk router found a strict split-stable regional readout champion without same-holdout residual routing."
    elif status == "split_risk_regional_router_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "The R61 split-risk router preserves the R60 mean contract, but split stability remains limited."
    elif status in {"split_risk_regional_router_mean_vs_r59_only", "split_risk_regional_router_diagnostic_only"}:
        claim_status = "diagnostic_only"
        allowed = "R61 is a negative/diagnostic result: conservative stream routing did not produce a stricter regional champion."
    else:
        claim_status = "blocked"
        allowed = "R61 has no supported split-risk router claim."
    return {
        "claim_id": "regional_r61_split_risk_router_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("best_candidate_family") or "split_risk_regional_router",
        "primary_gate": status,
        "blockers": list(r61.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Router/readout claim only. A negative R61 result supports the conclusion that the remaining regional failures are process/support-lineage problems, not safely fixable by stream routing.",
        "key_metrics": {
            "experiment_count": gate.get("experiment_count"),
            "mean_pass_count": gate.get("mean_pass_count"),
            "mean_and_r60_nonregression_count": gate.get("mean_and_r60_nonregression_count"),
            "strict_pass_count": gate.get("strict_pass_count"),
            "best_mean_regional_normalized_absolute_error": gate.get("best_mean_regional_normalized_absolute_error"),
            "best_mean_aggregate_mass_normalized_absolute_error": gate.get("best_mean_aggregate_mass_normalized_absolute_error"),
            "best_mean_regional_share_half_l1_error": gate.get("best_mean_regional_share_half_l1_error"),
            "best_split_gate_status": gate.get("best_split_gate_status"),
        },
    }


def _regional_leakage_expert_student_claim(r62: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r62.get("leakage_student_gate") or {})
    status = str(r62.get("status") or gate.get("status") or "")
    if status == "leakage_student_strict_champion_promoted":
        claim_status = "promoted"
        allowed = "A blocked-time leakage student is promoted; the same-split leakage expert remains diagnostic-only."
    elif status == "leakage_student_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "A blocked-time leakage student improves the mean contract but remains split-limited; the same-split expert is not promotable."
    elif status == "leakage_expert_only_diagnostic":
        claim_status = "diagnostic_only"
        allowed = "The leakage expert is useful only as an oracle/teacher diagnostic; blocked-time students do not preserve the current R60 contract."
    else:
        claim_status = "blocked"
        allowed = "No leakage-expert/student claim is supported."
    return {
        "claim_id": "regional_r62_leakage_expert_student_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": gate.get("best_student_family") or gate.get("expert_family") or "leakage_expert_student_gate",
        "primary_gate": status,
        "blockers": list(r62.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Same-split leakage expert predictions must never be used as promoted forecasts. Only blocked-time students may promote, and only if they pass the existing R60/R55 gates.",
        "key_metrics": {
            "expert_mean_regional_normalized_absolute_error": gate.get("expert_mean_regional_normalized_absolute_error"),
            "expert_mean_aggregate_mass_normalized_absolute_error": gate.get("expert_mean_aggregate_mass_normalized_absolute_error"),
            "expert_mean_regional_share_half_l1_error": gate.get("expert_mean_regional_share_half_l1_error"),
            "best_student_mean_regional_normalized_absolute_error": gate.get("best_student_mean_regional_normalized_absolute_error"),
            "best_student_mean_aggregate_mass_normalized_absolute_error": gate.get("best_student_mean_aggregate_mass_normalized_absolute_error"),
            "best_student_mean_regional_share_half_l1_error": gate.get("best_student_mean_regional_share_half_l1_error"),
            "student_mean_pass_count": gate.get("student_mean_pass_count"),
            "student_r60_safe_count": gate.get("student_r60_safe_count"),
            "student_strict_pass_count": gate.get("student_strict_pass_count"),
        },
    }


def _regional_online_expert_claim(r63: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r63.get("online_expert_gate") or {})
    status = str(r63.get("status") or gate.get("status") or "")
    if status == "region_metric_online_expert_student_promoted":
        claim_status = "promoted"
        allowed = "The region-metric online expert student is promoted; the same-holdout region-metric oracle remains diagnostic-only."
    elif status == "region_metric_online_expert_student_mean_promoted_split_limited":
        claim_status = "mean_promoted_split_limited"
        allowed = "The region-metric online expert student improves the mean contract but remains split-limited."
    elif status == "region_metric_online_expert_diagnostic_only":
        claim_status = "diagnostic_only"
        allowed = "The region-metric oracle exposes a large ceiling, but the blocked-time online student does not preserve the R60 contract."
    else:
        claim_status = "blocked"
        allowed = "No region-metric online expert claim is supported."
    return {
        "claim_id": "regional_r63_region_metric_online_expert_claim",
        "claim_scope": "regional",
        "claim_status": claim_status,
        "model_family": "region_metric_online_expert_student",
        "primary_gate": status,
        "blockers": list(r63.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Same-holdout region-metric oracle predictions are diagnostic only. The failed student means current regional history is too short/nonstationary for an online expert overlay claim.",
        "key_metrics": {
            "oracle_mean_regional_normalized_absolute_error": gate.get("oracle_mean_regional_normalized_absolute_error"),
            "oracle_mean_aggregate_mass_normalized_absolute_error": gate.get("oracle_mean_aggregate_mass_normalized_absolute_error"),
            "oracle_mean_regional_share_half_l1_error": gate.get("oracle_mean_regional_share_half_l1_error"),
            "student_mean_regional_normalized_absolute_error": gate.get("student_mean_regional_normalized_absolute_error"),
            "student_mean_aggregate_mass_normalized_absolute_error": gate.get("student_mean_aggregate_mass_normalized_absolute_error"),
            "student_mean_regional_share_half_l1_error": gate.get("student_mean_regional_share_half_l1_error"),
            "student_split_gate_status": gate.get("student_split_gate_status"),
        },
    }


def _regional_leakage_support_gap_claim(r64: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r64.get("support_gap_gate") or {})
    status = str(r64.get("status") or gate.get("status") or "")
    ready = status == "leakage_support_gap_priorities_ready"
    return {
        "claim_id": "regional_r64_leakage_support_gap_priorities",
        "claim_scope": "regional_evidence_acquisition",
        "claim_status": "diagnostic_ready" if ready else "blocked",
        "model_family": "leakage_support_gap_prioritizer",
        "primary_gate": status,
        "blockers": list(r64.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R64 identifies leakage-derived evidence acquisition priorities; it does not produce forecasts."
            if ready
            else "R64 did not produce evidence acquisition priorities."
        ),
        "claim_limit": "Diagnostic/evidence-prioritization only. R64 rows cannot be used as fitted corrections or target-derived model inputs.",
        "key_metrics": {
            "gap_row_count": gate.get("gap_row_count"),
            "top_metric": gate.get("top_metric"),
            "top_metric_stream": gate.get("top_metric_stream"),
            "top_region": gate.get("top_region"),
        },
    }


def _transmission_readiness_claim(r65: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r65.get("readiness_gate") or {})
    status = str(r65.get("status") or gate.get("status") or "")
    ready = status == "full_transmission_model_ready"
    return {
        "claim_id": "phase3_full_transmission_model_readiness",
        "claim_scope": "national_and_regional_transmission",
        "claim_status": "ready" if ready else "not_yet_ready",
        "model_family": "full_transmission_readiness_gate",
        "primary_gate": status,
        "blockers": list(r65.get("blockers") or gate.get("blockers") or []),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "Full national/regional transmission model readiness gate passed."
            if ready
            else "Full transmission claims are blocked; national readout is ready, but regional full-transmission modeling remains unsupported."
        ),
        "claim_limit": "Readiness-gate claim only. Does not create a new forecast or imply regional transmission mechanisms are validated.",
        "key_metrics": {
            "national_forecast_readout_ready": gate.get("national_forecast_readout_ready"),
            "regional_full_transmission_ready": gate.get("regional_full_transmission_ready"),
            "blockers": gate.get("blockers"),
        },
    }


def _scientific_source_base_claim(r66: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r66.get("source_base_gate") or {})
    status = str(r66.get("status") or gate.get("status") or "")
    ready = status == "source_base_ready_for_model_family_queue"
    return {
        "claim_id": "phase3_scientific_source_base",
        "claim_scope": "source_lineage_and_model_inputs",
        "claim_status": "diagnostic_ready" if ready else "blocked",
        "model_family": "scientific_source_base",
        "primary_gate": status,
        "blockers": [] if ready else ["source_base_not_ready"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R66 provides a tracked source base for transmission-model ideation and bounded experiments."
            if ready
            else "R66 source base is not ready."
        ),
        "claim_limit": "Source/readiness claim only. It does not validate any new model family or determinant effect.",
        "key_metrics": {
            "local_source_count": gate.get("local_source_count"),
            "downloaded_source_count": gate.get("downloaded_source_count"),
            "quarantined_source_count": gate.get("quarantined_source_count"),
            "determinant_module_count": gate.get("determinant_module_count"),
        },
    }


def _model_family_queue_claim(r67: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r67.get("model_family_gate") or {})
    status = str(r67.get("status") or gate.get("status") or "")
    ready = status == "model_family_queue_ready"
    return {
        "claim_id": "phase3_transmission_model_family_queue",
        "claim_scope": "model_family_hypotheses",
        "claim_status": "experiment_queue_ready" if ready else "blocked",
        "model_family": "ranked_transmission_model_family_queue",
        "primary_gate": status,
        "blockers": [] if ready else ["model_family_queue_not_ready"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R67 defines source-backed transmission-model families with equations, overfit guards, and promotion gates."
            if ready
            else "R67 model-family queue is not ready."
        ),
        "claim_limit": "Hypothesis/experiment-queue claim only. No family is promoted until it beats locked blocked-time gates.",
        "key_metrics": {
            "ready_bounded_family_count": gate.get("ready_bounded_family_count"),
            "determinant_locked_family_count": gate.get("determinant_locked_family_count"),
            "top_ready_families": gate.get("top_ready_families"),
        },
    }


def _bulk_external_ingest_claim(r68: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r68.get("bulk_ingest_gate") or {})
    status = str(r68.get("status") or gate.get("status") or "")
    ready = status == "bulk_external_sources_ingested"
    return {
        "claim_id": "phase3_bulk_external_source_ingest",
        "claim_scope": "external_source_lineage",
        "claim_status": "diagnostic_ready" if ready else "blocked",
        "model_family": "bulk_external_source_ingest",
        "primary_gate": status,
        "blockers": [] if ready else ["bulk_external_sources_not_ingested"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R68 safely downloaded bulk UNAIDS/Google sources and extracted Philippines rows with allowed-use roles."
            if ready
            else "R68 bulk external source ingest is not ready."
        ),
        "claim_limit": "Source-ingest claim only. Bulk rows cannot become training targets unless their allowed-use role permits it.",
        "key_metrics": {
            "downloaded_source_count": gate.get("downloaded_source_count"),
            "failed_source_count": gate.get("failed_source_count"),
            "extracted_philippines_row_count": gate.get("extracted_philippines_row_count"),
        },
    }


def _bulk_signal_feature_claim(r69: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r69.get("signal_feature_gate") or {})
    status = str(r69.get("status") or gate.get("status") or "")
    ready = status == "bulk_signal_features_ready"
    return {
        "claim_id": "phase3_bulk_signal_feature_compiler",
        "claim_scope": "model_feature_inputs",
        "claim_status": "feature_tables_ready" if ready else "blocked",
        "model_family": "bulk_signal_feature_compiler",
        "primary_gate": status,
        "blockers": [] if ready else ["bulk_signal_features_not_ready"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R69 compiled extracted bulk sources into model-ready external challenge, program support, KP, policy, and mobility feature tables."
            if ready
            else "R69 bulk signal feature tables are not ready."
        ),
        "claim_limit": "Feature-readiness claim only. Determinants remain sensitivity-only unless R46 source-stability passes.",
        "key_metrics": {
            "compiled_signal_row_count": gate.get("compiled_signal_row_count"),
            "feature_table_ready_family_count": gate.get("feature_table_ready_family_count"),
            "row_counts": r69.get("row_counts"),
        },
    }


def _scientific_model_build_queue_claim(r70: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r70.get("build_queue_gate") or {})
    status = str(r70.get("status") or gate.get("status") or "")
    ready = status == "scientific_model_build_queue_ready"
    return {
        "claim_id": "phase3_scientific_model_build_queue",
        "claim_scope": "bounded_experiment_queue",
        "claim_status": "experiment_queue_ready" if ready else "blocked",
        "model_family": "scientific_model_build_queue",
        "primary_gate": status,
        "blockers": [] if ready else ["scientific_model_build_queue_not_ready"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "R70 defines the next staged multi-hundred-step build queue with explicit promotion gates."
            if ready
            else "R70 scientific model build queue is not ready."
        ),
        "claim_limit": "Experiment-queue claim only. No R70 step is a promoted model until separately fitted, gated, and registered.",
        "key_metrics": {
            "queued_step_count": gate.get("queued_step_count"),
            "bounded_candidate_step_count": gate.get("bounded_candidate_step_count"),
            "sensitivity_or_falsification_step_count": gate.get("sensitivity_or_falsification_step_count"),
            "auxiliary_diagnostic_step_count": gate.get("auxiliary_diagnostic_step_count"),
        },
    }


def _external_signal_branch_claim(
    report: dict[str, Any],
    path: Path,
    *,
    claim_id: str,
    gate_key: str,
    model_family: str,
    promoted_status: str,
) -> dict[str, Any]:
    gate = dict(report.get(gate_key) or {})
    status = str(report.get("status") or gate.get("status") or "")
    if status == promoted_status:
        claim_status = "promoted"
        allowed = f"{model_family} is promoted under its locked blocked-time gate."
    elif status:
        claim_status = "diagnostic_only"
        allowed = f"{model_family} is a registered negative/diagnostic result and must not replace R41."
    else:
        claim_status = "blocked"
        allowed = f"{model_family} has no usable gate artifact."
    return {
        "claim_id": claim_id,
        "claim_scope": "national_external_signal_modeling",
        "claim_status": claim_status,
        "model_family": model_family,
        "primary_gate": status,
        "blockers": [] if claim_status != "blocked" else ["external_signal_gate_missing"],
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "External support/mobility/program signals are not determinant-effect claims. Diagnostic-only branches may inform failure anatomy but cannot become paper champion results.",
        "key_metrics": gate,
    }


def _bulk_unaids_annual_challenge_claim(r75: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r75.get("bulk_unaids_annual_gate") or {})
    status = str(r75.get("status") or gate.get("status") or "")
    passed = status == "bulk_unaids_annual_challenge_pass"
    if passed:
        claim_status = "external_validation_passed"
        allowed = "The annual weak-measurement/mass-balance head beats annual carry-forward against bulk UNAIDS all-ages incidence, AIDS deaths, and PLHIV validation targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "Bulk UNAIDS annual challenge is registered but did not pass the external validation gate."
    else:
        claim_status = "blocked"
        allowed = "Bulk UNAIDS annual challenge artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r75_bulk_unaids_annual_challenge",
        "claim_scope": "national_external_annual_validation",
        "claim_status": claim_status,
        "model_family": gate.get("best_candidate_family") or "bulk_unaids_annual_challenge",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if passed else ["bulk_unaids_annual_challenge_not_passed"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "External annual validation only. This does not prove superiority over AEM/Spectrum projections unless direct AEM/Spectrum output comparisons are later added.",
        "key_metrics": gate,
    }


def _public_domain_annual_comparator_claim(r76: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r76.get("public_domain_annual_gate") or {})
    status = str(r76.get("status") or gate.get("status") or "")
    ready = status == "public_domain_annual_comparator_ready"
    if ready:
        claim_status = "public_comparator_ready"
        allowed = "A transparent public-domain annual comparator is ready as an AEM/Spectrum-style proxy benchmark built only from public UNAIDS annual targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "Public-domain annual comparator is registered but did not beat public carry-forward under the gate."
    else:
        claim_status = "blocked"
        allowed = "Public-domain annual comparator artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r76_public_domain_annual_comparator",
        "claim_scope": "national_public_domain_annual_benchmark",
        "claim_status": claim_status,
        "model_family": "public_train_selected_annual_proxy",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if ready else ["public_domain_annual_comparator_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "This is not official AEM/Spectrum output. It is a reproducible public-data incumbent proxy for annual incidence, AIDS deaths, and PLHIV only.",
        "key_metrics": gate,
    }


def _public_proxy_annual_gate_claim(r77: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r77.get("public_proxy_annual_gate") or {})
    status = str(r77.get("status") or gate.get("status") or "")
    passed = status == "annual_model_beats_public_proxy"
    if passed:
        claim_status = "annual_public_proxy_beaten"
        allowed = "The Phase 3 annual head beats the transparent public-domain annual proxy on matched UNAIDS-style annual targets."
    elif status:
        claim_status = "annual_superiority_blocked"
        allowed = "The public-domain annual proxy remains the incumbent annual benchmark; Phase 3 annual superiority claims are blocked."
    else:
        claim_status = "blocked"
        allowed = "Public-proxy annual gate artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r77_public_proxy_annual_gate",
        "claim_scope": "national_public_domain_annual_superiority_gate",
        "claim_status": claim_status,
        "model_family": gate.get("model_family") or "phase3_annual_head",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if passed else ["public_proxy_annual_gate_not_passed"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "This gate compares against a reproducible public-data proxy, not official AEM/Spectrum outputs. If blocked, annual validation claims must be framed as carry-forward-beating only.",
        "key_metrics": gate,
    }


def _expanded_public_annual_comparator_claim(r78: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r78.get("expanded_public_annual_gate") or {})
    status = str(r78.get("status") or gate.get("status") or "")
    promoted = status == "expanded_public_annual_comparator_promoted"
    if promoted:
        claim_status = "expanded_public_comparator_promoted"
        allowed = "The expanded public-domain annual comparator supersedes R76 on held-out public UNAIDS-style annual targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "Expanded public annual comparator is registered but did not beat the R76 public proxy."
    else:
        claim_status = "blocked"
        allowed = "Expanded public annual comparator artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r78_expanded_public_annual_comparator",
        "claim_scope": "national_public_domain_annual_benchmark",
        "claim_status": claim_status,
        "model_family": "public_train_selected_annual_proxy_v2",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if promoted else ["expanded_public_annual_comparator_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "This remains a public-data annual proxy benchmark, not official AEM/Spectrum output and not a full quarterly cascade model.",
        "key_metrics": gate,
    }


def _expanded_public_proxy_annual_gate_claim(r79: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r79.get("expanded_public_proxy_annual_gate") or {})
    status = str(r79.get("status") or gate.get("status") or "")
    passed = status == "annual_model_beats_expanded_public_proxy"
    if passed:
        claim_status = "annual_expanded_public_proxy_beaten"
        allowed = "The Phase 3 annual head beats the expanded public-domain annual proxy on matched public annual targets."
    elif status:
        claim_status = "annual_superiority_blocked"
        allowed = "The expanded public-domain annual proxy remains the incumbent annual benchmark; annual superiority claims are blocked."
    else:
        claim_status = "blocked"
        allowed = "Expanded public-proxy annual gate artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r79_expanded_public_proxy_annual_gate",
        "claim_scope": "national_public_domain_annual_superiority_gate",
        "claim_status": claim_status,
        "model_family": gate.get("model_family") or "phase3_annual_head",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if passed else ["expanded_public_proxy_annual_gate_not_passed"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "This is a strict benchmark gate against the expanded public proxy, not a direct official AEM/Spectrum comparison.",
        "key_metrics": gate,
    }


def _public_annual_projection_head_claim(r80: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r80.get("public_annual_projection_gate") or {})
    status = str(r80.get("status") or gate.get("status") or "")
    ready = status == "public_annual_projection_head_ready"
    if ready:
        claim_status = "public_projection_ready"
        allowed = "A public-domain AEM/Spectrum-style annual projection head is available for incidence, AIDS deaths, and PLHIV sensitivity comparisons."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "Public annual projection head is registered but did not pass its readiness gate."
    else:
        claim_status = "blocked"
        allowed = "Public annual projection head artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r80_public_annual_projection_head",
        "claim_scope": "national_public_domain_annual_projection",
        "claim_status": claim_status,
        "model_family": "public_train_selected_annual_proxy_v2_projection",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if ready else ["public_annual_projection_head_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Projection head is public-data proxy output only. It is not official AEM/Spectrum and not evidence that quarterly cascade dynamics beat official models.",
        "key_metrics": gate,
    }


def _phase2_knob_admissibility_claim(r81: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r81.get("phase2_knob_gate") or {})
    status = str(r81.get("status") or gate.get("status") or "")
    if status == "phase2_quantitative_knobs_ready":
        claim_status = "quantitative_knobs_ready"
        allowed = "Phase 2 outputs include strict quantitative priors that may modulate model components under the recorded gate."
    elif status == "phase2_directional_sensitivity_knobs_only":
        claim_status = "directional_sensitivity_only"
        allowed = "Phase 2 outputs may label directional sensitivity scenarios, but cannot supply numeric intervention effect sizes."
    elif status:
        claim_status = "blocked"
        allowed = "Phase 2 knob gate is registered but does not allow model-output modulation."
    else:
        claim_status = "blocked"
        allowed = "Phase 2 knob admissibility artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r81_phase2_knob_admissibility",
        "claim_scope": "phase2_to_phase3_scenario_knobs",
        "claim_status": claim_status,
        "model_family": "phase2_structural_knob_gate",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if status == "phase2_quantitative_knobs_ready" else ["phase2_quantitative_knobs_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "No causal or numeric policy effect claim is allowed unless strict quantitative Phase 2 priors survive blocked-time falsification.",
        "key_metrics": gate,
    }


def _quarterly_annual_bridge_claim(r82: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r82.get("quarterly_annual_bridge_gate") or {})
    status = str(r82.get("status") or gate.get("status") or "")
    ready = status == "quarterly_annual_bridge_ready"
    if ready:
        claim_status = "bridge_ready"
        allowed = "Quarterly Phase 3 outputs can be annualized to the public annual proxy targets without annual-target leakage."
    elif status:
        claim_status = "bridge_blocked"
        allowed = "Quarterly Phase 3 outputs cannot yet be annualized into incidence, AIDS deaths, and total PLHIV public annual targets."
    else:
        claim_status = "blocked"
        allowed = "Quarterly-annual bridge artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r82_quarterly_annual_bridge",
        "claim_scope": "quarterly_mechanistic_to_public_annual_bridge",
        "claim_status": claim_status,
        "model_family": "quarterly_annual_bridge_gate",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if ready else ["quarterly_annual_bridge_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "Annual weak-measurement heads are not accepted as quarterly mechanistic bridges.",
        "key_metrics": gate,
    }


def _quarterly_emission_bridge_claim(r83: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r83.get("quarterly_emission_bridge_gate") or {})
    status = str(r83.get("status") or gate.get("status") or "")
    if status == "quarterly_emission_bridge_ready":
        claim_status = "bridge_ready"
        allowed = "Current quarterly Phase 3 candidates emit incidence, AIDS deaths, and total PLHIV so public annual targets can be scored without annual-head leakage."
    elif status == "quarterly_emission_bridge_partial_diagnostic":
        claim_status = "partial_diagnostic"
        allowed = "Some annual quantities can be reconstructed from quarterly emissions, but full incidence/deaths/PLHIV annual-mechanistic claims remain blocked."
    elif status:
        claim_status = "bridge_blocked"
        allowed = "Current quarterly Phase 3 candidates do not emit the required public annual challenge quantities directly."
    else:
        claim_status = "blocked"
        allowed = "Quarterly emission bridge audit artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r83_quarterly_emission_bridge_audit",
        "claim_scope": "actual_quarterly_candidate_emissions_to_public_annual_targets",
        "claim_status": claim_status,
        "model_family": "quarterly_emission_bridge_audit",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "bridge_ready" else ["quarterly_emission_bridge_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "R83 inspects actual prediction-row emissions; annual weak-measurement heads, diagnosed-stock PLHIV proxies, and aggregate attrition death proxies are not accepted.",
        "key_metrics": gate,
    }


def _conserved_quarterly_annual_ledger_claim(r84: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r84.get("conserved_quarterly_annual_ledger_gate") or {})
    status = str(r84.get("status") or gate.get("status") or "")
    if status == "conserved_quarterly_annual_ledger_pass":
        claim_status = "ledger_pass"
        allowed = "The conserved dynamic simulator emits annual incidence, AIDS deaths, and total PLHIV and beats carry-forward under validation-only annual targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "The conserved dynamic simulator emits annual ledger quantities, but the annual public-target comparison is not promotion-grade."
    else:
        claim_status = "blocked"
        allowed = "Conserved quarterly annual ledger artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r84_conserved_quarterly_annual_ledger",
        "claim_scope": "mechanistic_quarterly_state_ledger_to_public_annual_targets",
        "claim_status": claim_status,
        "model_family": "conserved_dynamic_quarterly_ledger",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "ledger_pass" else ["conserved_quarterly_annual_ledger_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "If diagnostic-only, R84 may be cited as a mechanistic bridge audit, not as evidence that the quarterly model beats official annual comparators.",
        "key_metrics": gate,
    }


def _annual_ledger_forecast_grid_claim(r85: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r85.get("annual_ledger_forecast_grid_gate") or {})
    status = str(r85.get("status") or gate.get("status") or "")
    if status == "annual_ledger_forecast_grid_pass":
        claim_status = "forecast_grid_pass"
        allowed = "The quarterly simulator emits complete annual ledger quantities on an unscored forecast grid and beats carry-forward under validation-only annual targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "The forecast-grid annual ledger is evaluable, but it is not promotion-grade against carry-forward and interval coverage gates."
    else:
        claim_status = "blocked"
        allowed = "Annual ledger forecast-grid artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r85_annual_ledger_forecast_grid",
        "claim_scope": "complete_quarterly_emission_grid_to_public_annual_targets",
        "claim_status": claim_status,
        "model_family": "conserved_dynamic_forecast_grid_ledger",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "forecast_grid_pass" else ["annual_ledger_forecast_grid_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "R85 repairs quarterly emission coverage only; if diagnostic-only, it must not be reported as an official annual-model win.",
        "key_metrics": gate,
    }


def _annual_calibrated_forecast_grid_ledger_claim(r86: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r86.get("annual_calibrated_forecast_grid_ledger_gate") or {})
    status = str(r86.get("status") or gate.get("status") or "")
    if status == "annual_calibrated_forecast_grid_ledger_pass":
        claim_status = "scoped_annual_model_win"
        allowed = (
            "The complete quarterly ledger with train-origin annual weak-measurement calibration beats carry-forward "
            "on held-out annual incidence, AIDS deaths, and PLHIV targets without using holdout annual targets as training truth."
        )
    elif status:
        claim_status = "diagnostic_only"
        allowed = "The annual-calibrated forecast-grid ledger is evaluable, but it is not promotion-grade against carry-forward and interval gates."
    else:
        claim_status = "blocked"
        allowed = "Annual-calibrated forecast-grid ledger artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r86_annual_calibrated_forecast_grid_ledger",
        "claim_scope": "train_origin_annual_calibration_on_complete_quarterly_ledger",
        "claim_status": claim_status,
        "model_family": "annual_calibrated_conserved_forecast_grid_ledger",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "scoped_annual_model_win" else ["annual_calibrated_forecast_grid_ledger_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "This is a scoped annual-ledger win, not a proof that raw quarterly mechanistic incidence/death emissions "
            "or broad official-model replacement claims are solved."
        ),
        "key_metrics": gate,
    }


def _train_backtested_emission_process_calibration_claim(r87: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r87.get("train_backtested_emission_process_calibration_gate") or {})
    status = str(r87.get("status") or gate.get("status") or "")
    if status == "train_backtested_emission_process_calibration_pass":
        claim_status = "raw_process_calibration_win"
        allowed = "Train-backtested raw quarterly emission calibration beats carry-forward on validation-only annual targets."
    elif status:
        claim_status = "diagnostic_only"
        allowed = "Train-backtested raw quarterly emission calibration is evaluable but does not beat carry-forward."
    else:
        claim_status = "blocked"
        allowed = "Train-backtested raw emission calibration artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r87_train_backtested_emission_process_calibration",
        "claim_scope": "raw_quarterly_emission_process_calibration_to_annual_targets",
        "claim_status": claim_status,
        "model_family": "train_backtested_raw_emission_process_calibration",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "raw_process_calibration_win" else ["train_backtested_emission_process_calibration_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": "R87 currently falsifies free ratio/trend rescaling when diagnostic-only; do not cite it as an annual model win.",
        "key_metrics": gate,
    }


def _guarded_annual_ledger_selector_claim(r88: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r88.get("guarded_annual_ledger_selector_gate") or {})
    status = str(r88.get("status") or gate.get("status") or "")
    if status == "guarded_annual_ledger_selector_pass":
        claim_status = "guarded_annual_model_win"
        allowed = (
            "The guarded annual ledger beats carry-forward by retaining the raw quarterly process only where train-window "
            "rolling evidence beats carry-forward and falling back to a conservative carry-forward prior otherwise."
        )
    elif status:
        claim_status = "diagnostic_only"
        allowed = "The guarded annual ledger selector is evaluable but not promotion-grade."
    else:
        claim_status = "blocked"
        allowed = "Guarded annual ledger selector artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r88_guarded_annual_ledger_selector",
        "claim_scope": "guarded_raw_process_or_carry_forward_annual_ledger",
        "claim_status": claim_status,
        "model_family": "guarded_raw_process_or_carry_forward_annual_ledger",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "guarded_annual_model_win" else ["guarded_annual_ledger_selector_not_promoted"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R88 is a conservative guarded model, not a proof that incidence or AIDS-death raw processes are identified; "
            "its win comes from rejecting weak raw channels and preserving the stable PLHIV stock process."
        ),
        "key_metrics": gate,
    }


def _incidence_mortality_mechanism_support_claim(r89: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r89.get("incidence_mortality_mechanism_support_gate") or {})
    status = str(r89.get("status") or gate.get("status") or "")
    if status == "incidence_mortality_mechanism_support_ready":
        claim_status = "mechanism_support_ready"
        allowed = "Direct incidence/mortality process support is sufficient to attempt a raw incidence/death mechanism claim."
    elif status:
        claim_status = "diagnostic_only"
        allowed = (
            "Raw incidence/death process claims are blocked: direct incidence support is absent or the direct reported-death "
            "bridge does not beat carry-forward."
        )
    else:
        claim_status = "blocked"
        allowed = "Incidence/mortality mechanism-support artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r89_incidence_mortality_mechanism_support_gate",
        "claim_scope": "raw_incidence_and_mortality_process_admissibility",
        "claim_status": claim_status,
        "model_family": "reported_death_bridge_and_incidence_support_gate",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status == "mechanism_support_ready" else ["incidence_mortality_mechanism_support_not_ready"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R89 is an admissibility gate, not a forecast champion. When diagnostic-only, it explicitly preserves R86/R88 "
            "as scoped annual/readout wins and blocks claims that incidence or AIDS-death raw mechanisms are identified."
        ),
        "key_metrics": gate,
    }


def _claim_grade_gate_claim(r90: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r90.get("claim_grade_gate") or {})
    status = str(r90.get("status") or gate.get("status") or "")
    if status == "claim_grade_annual_readout_and_mechanisms_ready":
        claim_status = "annual_readout_and_mechanisms_claim_grade_ready"
        allowed = "R90 clears annual/readout and raw incidence/death mechanism-support claims against the locked artifacts."
    elif status == "claim_grade_annual_readout_ready_mechanisms_blocked":
        claim_status = "annual_readout_claim_grade_ready_mechanisms_blocked"
        allowed = (
            "R90 clears R86/R88 as scoped annual/readout claims while explicitly blocking raw incidence/death mechanism claims."
        )
    elif status:
        claim_status = "diagnostic_only"
        allowed = "R90 is evaluable but one or more claim-grade annual/readout requirements failed."
    else:
        claim_status = "blocked"
        allowed = "R90 claim-grade gate artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r90_claim_grade_gate",
        "claim_scope": "publication_claim_grade_adjudication_for_r86_r88_r89",
        "claim_status": claim_status,
        "model_family": "claim_grade_gate_over_locked_annual_and_mechanism_artifacts",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status != "blocked" else ["r90_claim_grade_gate_missing"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R90 is an adjudication gate only. It does not create a new forecast model, refit parameters, or convert "
            "annual/readout wins into identified incidence, mortality, determinant, or official-model-replacement claims."
        ),
        "key_metrics": gate,
    }


def _mechanism_support_expansion_claim(r91: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r91.get("mechanism_support_expansion_gate") or {})
    status = str(r91.get("status") or gate.get("status") or "")
    if status == "mechanism_support_expansion_ready":
        claim_status = "mechanism_support_ready"
        allowed = "R91 found direct/process incidence support and source-stable mortality bridge support sufficient for a mechanism branch."
    elif status == "proxy_bridge_signal_detected_but_mechanism_claim_blocked":
        claim_status = "proxy_signal_diagnostic_only"
        allowed = (
            "R91 detected at least one proxy bridge signal, but mechanism claims remain blocked by missing direct support "
            "or source-family instability."
        )
    elif status:
        claim_status = "diagnostic_only"
        allowed = "R91 is a negative mechanism-support diagnostic: proxy bridges do not clear the carry-forward/source-stability gate."
    else:
        claim_status = "blocked"
        allowed = "R91 mechanism-support expansion artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r91_mechanism_support_expansion_gate",
        "claim_scope": "incidence_and_mortality_mechanism_support_expansion",
        "claim_status": claim_status,
        "model_family": "train_origin_proxy_bridge_and_source_family_ablation_gate",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status != "blocked" else ["r91_mechanism_support_expansion_missing"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R91 can only promote mechanism-support admissibility. Diagnosis-flow bridges are proxy evidence, not direct "
            "incidence truth, and source-unstable mortality bridges cannot support process claims."
        ),
        "key_metrics": gate,
    }


def _process_repair_experiment_queue_claim(r92: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r92.get("process_repair_gate") or {})
    status = str(r92.get("status") or gate.get("status") or "")
    if status == "process_repair_mechanism_support_ready":
        claim_status = "mechanism_support_ready"
        allowed = "R92 found a train-origin process-repair branch that clears direct support, carry-forward, coverage, and source-family gates."
    elif status == "mortality_process_signal_detected_mechanism_claim_blocked":
        claim_status = "mortality_signal_diagnostic_only"
        allowed = (
            "R92 detected a mortality/death-ascertainment process signal, but mechanism claims remain blocked by "
            "missing direct incidence support or source-family instability."
        )
    elif status == "readout_process_signal_detected_mechanism_claim_blocked":
        claim_status = "readout_signal_diagnostic_only"
        allowed = "R92 detected a readout/proxy signal, but it is not admissible as a raw incidence or mortality mechanism claim."
    elif status == "process_repair_diagnostic_only":
        claim_status = "diagnostic_only"
        allowed = "R92 did not find a process-repair branch strong enough for mechanism-support promotion."
    else:
        claim_status = "blocked"
        allowed = "R92 process-repair artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r92_process_repair_experiment_queue",
        "claim_scope": "incidence_and_mortality_process_repair_queue",
        "claim_status": claim_status,
        "model_family": r92.get("candidate_family") or "process_repair_train_selected_observation_bridge",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status != "blocked" else ["r92_process_repair_missing"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R92 is an experiment-queue and support-repair diagnostic unless it reaches mechanism_support_ready. "
            "Proxy diagnosis-flow and reported-death bridges are not direct incidence/death mechanisms."
        ),
        "key_metrics": {
            "direct_incidence_process_support_count": gate.get("direct_incidence_process_support_count"),
            "diagnosis_flow_proxy_support_count": gate.get("diagnosis_flow_proxy_support_count"),
            "direct_reported_death_support_count": gate.get("direct_reported_death_support_count"),
            "incidence_mechanism_bridge": gate.get("incidence_mechanism_bridge"),
            "mortality_mechanism_bridge": gate.get("mortality_mechanism_bridge"),
            "mortality_source_family_stable": gate.get("mortality_source_family_stable"),
        },
    }


def _open_public_incumbent_comparator_claim(r93: dict[str, Any], path: Path) -> dict[str, Any]:
    gate = dict(r93.get("open_public_incumbent_comparator_gate") or {})
    status = str(r93.get("status") or gate.get("status") or "")
    if status == "public_incumbent_comparator_ready_model_beats_incumbent":
        claim_status = "annual_superiority_ready_against_open_public_incumbent"
        allowed = "R93 clears a broad annual superiority claim against the open public AEM/Spectrum-style incumbent comparator."
    elif status == "public_incumbent_comparator_ready_model_blocked":
        claim_status = "annual_superiority_blocked_by_open_public_incumbent"
        allowed = (
            "R93 freezes the open public annual incumbent comparator and blocks broad annual superiority claims "
            "until the Phase 3 annual head beats it on matched blocked-time scores."
        )
    elif status == "public_incumbent_comparator_blocked":
        claim_status = "comparator_blocked"
        allowed = "R93 could not establish an evaluable public incumbent comparator."
    else:
        claim_status = "blocked"
        allowed = "R93 open public incumbent comparator artifact is missing or not evaluable."
    return {
        "claim_id": "phase3_r93_open_public_incumbent_comparator",
        "claim_scope": "public_annual_incumbent_comparison",
        "claim_status": claim_status,
        "model_family": r93.get("candidate_family") or "open_public_aem_spectrum_style_annual_incumbent_comparator",
        "primary_gate": status,
        "blockers": list(gate.get("blockers") or ([] if claim_status != "blocked" else ["r93_open_public_incumbent_missing"])),
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed,
        "claim_limit": (
            "R93 compares against an open public incumbent proxy, not official Philippines AEM/Spectrum files. "
            "It governs annual incidence/deaths/PLHIV superiority claims only, not quarterly cascade, determinant, or subnational claims."
        ),
        "key_metrics": {
            "annual_superiority_status": gate.get("annual_superiority_status"),
            "incumbent_family": gate.get("incumbent_family"),
            "incumbent_mean_norm_error": gate.get("incumbent_mean_norm_error"),
            "matched_model_mean_norm_error": gate.get("matched_model_mean_norm_error"),
            "matched_incumbent_mean_norm_error": gate.get("matched_incumbent_mean_norm_error"),
            "matched_model_minus_incumbent_mean_norm_error": gate.get("matched_model_minus_incumbent_mean_norm_error"),
            "matched_model_interval_coverage": gate.get("matched_model_interval_coverage"),
            "matched_incumbent_interval_coverage": gate.get("matched_incumbent_interval_coverage"),
        },
    }


def _registry_gate(claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("claim_id") or ""): dict(row) for row in claim_rows}
    national_ok = str((by_id.get("national_r41_research_champion") or {}).get("claim_status")) == "promoted"
    regional_readout_ok = str((by_id.get("regional_hasp_cascade_readout_champion") or {}).get("claim_status")) == "promoted"
    regional_adapter_ok = str((by_id.get("regional_hasp_national_total_adapter_champion") or {}).get("claim_status")) == "promoted"
    regional_split_guarded_ok = str((by_id.get("regional_split_guarded_selector_claim") or {}).get("claim_status")) == "promoted"
    regional_pareto_ensemble_status = str((by_id.get("regional_pareto_simplex_ensemble_claim") or {}).get("claim_status"))
    regional_pareto_ensemble_ok = regional_pareto_ensemble_status == "promoted"
    regional_anchor_ensemble_status = str((by_id.get("regional_anchor_stable_pareto_ensemble_claim") or {}).get("claim_status"))
    regional_anchor_ensemble_ok = regional_anchor_ensemble_status == "promoted"
    regional_r60_status = str((by_id.get("regional_r60_experiment_queue_claim") or {}).get("claim_status"))
    regional_r60_ok = regional_r60_status == "promoted"
    regional_r61_status = str((by_id.get("regional_r61_split_risk_router_claim") or {}).get("claim_status"))
    regional_r61_ok = regional_r61_status == "promoted"
    regional_r62_status = str((by_id.get("regional_r62_leakage_expert_student_claim") or {}).get("claim_status"))
    regional_r62_ok = regional_r62_status == "promoted"
    regional_r63_status = str((by_id.get("regional_r63_region_metric_online_expert_claim") or {}).get("claim_status"))
    regional_r63_ok = regional_r63_status == "promoted"
    regional_r64_status = str((by_id.get("regional_r64_leakage_support_gap_priorities") or {}).get("claim_status"))
    transmission_readiness_status = str((by_id.get("phase3_full_transmission_model_readiness") or {}).get("claim_status"))
    source_base_status = str((by_id.get("phase3_scientific_source_base") or {}).get("claim_status"))
    model_family_queue_status = str((by_id.get("phase3_transmission_model_family_queue") or {}).get("claim_status"))
    bulk_ingest_status = str((by_id.get("phase3_bulk_external_source_ingest") or {}).get("claim_status"))
    bulk_signal_status = str((by_id.get("phase3_bulk_signal_feature_compiler") or {}).get("claim_status"))
    model_build_queue_status = str((by_id.get("phase3_scientific_model_build_queue") or {}).get("claim_status"))
    r71_external_status = str((by_id.get("phase3_r71_service_intensity_capacity_branch") or {}).get("claim_status"))
    r72_external_status = str((by_id.get("phase3_r72_service_feature_selector_gate") or {}).get("claim_status"))
    r73_external_status = str((by_id.get("phase3_r73_external_signal_lag_falsification") or {}).get("claim_status"))
    r74_external_status = str((by_id.get("phase3_r74_external_tail_risk_selector") or {}).get("claim_status"))
    r75_bulk_annual_status = str((by_id.get("phase3_r75_bulk_unaids_annual_challenge") or {}).get("claim_status"))
    r76_public_annual_status = str((by_id.get("phase3_r76_public_domain_annual_comparator") or {}).get("claim_status"))
    r77_public_proxy_gate_status = str((by_id.get("phase3_r77_public_proxy_annual_gate") or {}).get("claim_status"))
    r78_expanded_public_annual_status = str((by_id.get("phase3_r78_expanded_public_annual_comparator") or {}).get("claim_status"))
    r79_expanded_public_proxy_gate_status = str((by_id.get("phase3_r79_expanded_public_proxy_annual_gate") or {}).get("claim_status"))
    r80_public_projection_status = str((by_id.get("phase3_r80_public_annual_projection_head") or {}).get("claim_status"))
    r81_phase2_knob_status = str((by_id.get("phase3_r81_phase2_knob_admissibility") or {}).get("claim_status"))
    r82_quarterly_annual_bridge_status = str((by_id.get("phase3_r82_quarterly_annual_bridge") or {}).get("claim_status"))
    r83_quarterly_emission_bridge_status = str((by_id.get("phase3_r83_quarterly_emission_bridge_audit") or {}).get("claim_status"))
    r84_conserved_ledger_status = str((by_id.get("phase3_r84_conserved_quarterly_annual_ledger") or {}).get("claim_status"))
    r85_forecast_grid_status = str((by_id.get("phase3_r85_annual_ledger_forecast_grid") or {}).get("claim_status"))
    r86_annual_calibrated_ledger_status = str((by_id.get("phase3_r86_annual_calibrated_forecast_grid_ledger") or {}).get("claim_status"))
    r87_process_calibration_status = str((by_id.get("phase3_r87_train_backtested_emission_process_calibration") or {}).get("claim_status"))
    r88_guarded_annual_status = str((by_id.get("phase3_r88_guarded_annual_ledger_selector") or {}).get("claim_status"))
    r89_mechanism_support_status = str((by_id.get("phase3_r89_incidence_mortality_mechanism_support_gate") or {}).get("claim_status"))
    r90_claim_grade_status = str((by_id.get("phase3_r90_claim_grade_gate") or {}).get("claim_status"))
    r91_mechanism_expansion_status = str((by_id.get("phase3_r91_mechanism_support_expansion_gate") or {}).get("claim_status"))
    r92_process_repair_status = str((by_id.get("phase3_r92_process_repair_experiment_queue") or {}).get("claim_status"))
    r93_public_incumbent_status = str((by_id.get("phase3_r93_open_public_incumbent_comparator") or {}).get("claim_status"))
    subnational_ok = regional_r63_ok or regional_r62_ok or regional_r61_ok or regional_r60_ok or regional_anchor_ensemble_ok or regional_pareto_ensemble_ok or regional_split_guarded_ok or regional_adapter_ok or regional_readout_ok
    adapter_stability_status = str((by_id.get("regional_adapter_split_stability_claim") or {}).get("claim_status"))
    split_guarded_selector_status = str((by_id.get("regional_split_guarded_selector_claim") or {}).get("claim_status"))
    candidate_ceiling_status = str((by_id.get("regional_candidate_space_ceiling_diagnostic") or {}).get("claim_status"))
    determinant_status = str((by_id.get("phase2_determinant_driver_claim") or {}).get("claim_status"))
    blockers: list[str] = []
    if not national_ok:
        blockers.append("national_champion_not_promoted")
    if not subnational_ok:
        blockers.append("subnational_champion_not_promoted")
    if determinant_status == "strict_priors_ready":
        status = "national_subnational_and_strict_determinant_claims_ready" if not blockers else "claim_registry_blocked"
    elif determinant_status == "sensitivity_only":
        status = "national_and_subnational_readout_champions_ready_determinants_sensitivity_only" if not blockers else "claim_registry_blocked"
    else:
        blockers.append("determinant_scenarios_blocked")
        status = "claim_registry_blocked"
    return {
        "status": status,
        "blockers": blockers,
        "national_champion_promoted": national_ok,
        "subnational_champion_promoted": subnational_ok,
        "regional_readout_promoted": regional_readout_ok,
        "regional_adapter_promoted": regional_adapter_ok,
        "regional_split_guarded_selector_promoted": regional_split_guarded_ok,
        "regional_pareto_ensemble_promoted": regional_pareto_ensemble_ok,
        "regional_pareto_ensemble_status": regional_pareto_ensemble_status,
        "regional_anchor_stable_ensemble_promoted": regional_anchor_ensemble_ok,
        "regional_anchor_stable_ensemble_status": regional_anchor_ensemble_status,
        "regional_r60_experiment_queue_promoted": regional_r60_ok,
        "regional_r60_experiment_queue_status": regional_r60_status,
        "regional_r61_split_risk_router_promoted": regional_r61_ok,
        "regional_r61_split_risk_router_status": regional_r61_status,
        "regional_r62_leakage_student_promoted": regional_r62_ok,
        "regional_r62_leakage_student_status": regional_r62_status,
        "regional_r63_online_expert_promoted": regional_r63_ok,
        "regional_r63_online_expert_status": regional_r63_status,
        "regional_r64_support_gap_status": regional_r64_status,
        "phase3_full_transmission_readiness_status": transmission_readiness_status,
        "phase3_scientific_source_base_status": source_base_status,
        "phase3_model_family_queue_status": model_family_queue_status,
        "phase3_bulk_external_ingest_status": bulk_ingest_status,
        "phase3_bulk_signal_feature_status": bulk_signal_status,
        "phase3_scientific_model_build_queue_status": model_build_queue_status,
        "phase3_r71_external_signal_status": r71_external_status,
        "phase3_r72_external_signal_status": r72_external_status,
        "phase3_r73_external_signal_status": r73_external_status,
        "phase3_r74_external_signal_status": r74_external_status,
        "phase3_r75_bulk_unaids_annual_status": r75_bulk_annual_status,
        "phase3_r76_public_domain_annual_status": r76_public_annual_status,
        "phase3_r77_public_proxy_annual_gate_status": r77_public_proxy_gate_status,
        "phase3_r78_expanded_public_annual_status": r78_expanded_public_annual_status,
        "phase3_r79_expanded_public_proxy_annual_gate_status": r79_expanded_public_proxy_gate_status,
        "phase3_r80_public_annual_projection_status": r80_public_projection_status,
        "phase3_r81_phase2_knob_status": r81_phase2_knob_status,
        "phase3_r82_quarterly_annual_bridge_status": r82_quarterly_annual_bridge_status,
        "phase3_r83_quarterly_emission_bridge_status": r83_quarterly_emission_bridge_status,
        "phase3_r84_conserved_ledger_status": r84_conserved_ledger_status,
        "phase3_r85_forecast_grid_status": r85_forecast_grid_status,
        "phase3_r86_annual_calibrated_ledger_status": r86_annual_calibrated_ledger_status,
        "phase3_r87_process_calibration_status": r87_process_calibration_status,
        "phase3_r88_guarded_annual_status": r88_guarded_annual_status,
        "phase3_r89_mechanism_support_status": r89_mechanism_support_status,
        "phase3_r90_claim_grade_status": r90_claim_grade_status,
        "phase3_r91_mechanism_expansion_status": r91_mechanism_expansion_status,
        "phase3_r92_process_repair_status": r92_process_repair_status,
        "phase3_r93_public_incumbent_status": r93_public_incumbent_status,
        "regional_adapter_stability_status": adapter_stability_status,
        "regional_split_guarded_selector_status": split_guarded_selector_status,
        "regional_candidate_ceiling_status": candidate_ceiling_status,
        "determinant_claim_status": determinant_status,
        "contract": (
            "The publication registry separates promoted forecast/readout claims from sensitivity-only determinant claims. "
            "No claim may be made in a paper unless it appears here with its evidence artifact and allowed-use limit."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("registry_gate") or {})
    lines = [
        "# Phase 3 R53 Publication Claim Registry",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- National champion promoted: `{gate.get('national_champion_promoted')}`",
        f"- Subnational champion promoted: `{gate.get('subnational_champion_promoted')}`",
        f"- Regional readout promoted: `{gate.get('regional_readout_promoted')}`",
        f"- Regional national-total adapter promoted: `{gate.get('regional_adapter_promoted')}`",
        f"- Regional split-guarded selector promoted: `{gate.get('regional_split_guarded_selector_promoted')}`",
        f"- Regional Pareto ensemble promoted: `{gate.get('regional_pareto_ensemble_promoted')}`",
        f"- Regional Pareto ensemble status: `{gate.get('regional_pareto_ensemble_status')}`",
        f"- Regional anchor-stable ensemble promoted: `{gate.get('regional_anchor_stable_ensemble_promoted')}`",
        f"- Regional anchor-stable ensemble status: `{gate.get('regional_anchor_stable_ensemble_status')}`",
        f"- Regional R60 experiment queue promoted: `{gate.get('regional_r60_experiment_queue_promoted')}`",
        f"- Regional R60 experiment queue status: `{gate.get('regional_r60_experiment_queue_status')}`",
        f"- Regional R61 split-risk router promoted: `{gate.get('regional_r61_split_risk_router_promoted')}`",
        f"- Regional R61 split-risk router status: `{gate.get('regional_r61_split_risk_router_status')}`",
        f"- Regional R62 leakage student promoted: `{gate.get('regional_r62_leakage_student_promoted')}`",
        f"- Regional R62 leakage student status: `{gate.get('regional_r62_leakage_student_status')}`",
        f"- Regional R63 online expert promoted: `{gate.get('regional_r63_online_expert_promoted')}`",
        f"- Regional R63 online expert status: `{gate.get('regional_r63_online_expert_status')}`",
        f"- Regional R64 support-gap status: `{gate.get('regional_r64_support_gap_status')}`",
        f"- Phase 3 full-transmission readiness status: `{gate.get('phase3_full_transmission_readiness_status')}`",
        f"- Phase 3 scientific source-base status: `{gate.get('phase3_scientific_source_base_status')}`",
        f"- Phase 3 model-family queue status: `{gate.get('phase3_model_family_queue_status')}`",
        f"- Phase 3 bulk external ingest status: `{gate.get('phase3_bulk_external_ingest_status')}`",
        f"- Phase 3 bulk signal feature status: `{gate.get('phase3_bulk_signal_feature_status')}`",
        f"- Phase 3 scientific model build queue status: `{gate.get('phase3_scientific_model_build_queue_status')}`",
        f"- Phase 3 R71 external signal status: `{gate.get('phase3_r71_external_signal_status')}`",
        f"- Phase 3 R72 external signal status: `{gate.get('phase3_r72_external_signal_status')}`",
        f"- Phase 3 R73 external signal status: `{gate.get('phase3_r73_external_signal_status')}`",
        f"- Phase 3 R74 external signal status: `{gate.get('phase3_r74_external_signal_status')}`",
        f"- Phase 3 R75 bulk UNAIDS annual status: `{gate.get('phase3_r75_bulk_unaids_annual_status')}`",
        f"- Phase 3 R76 public-domain annual status: `{gate.get('phase3_r76_public_domain_annual_status')}`",
        f"- Phase 3 R77 public-proxy annual gate status: `{gate.get('phase3_r77_public_proxy_annual_gate_status')}`",
        f"- Phase 3 R78 expanded public annual status: `{gate.get('phase3_r78_expanded_public_annual_status')}`",
        f"- Phase 3 R79 expanded public-proxy annual gate status: `{gate.get('phase3_r79_expanded_public_proxy_annual_gate_status')}`",
        f"- Phase 3 R80 public annual projection status: `{gate.get('phase3_r80_public_annual_projection_status')}`",
        f"- Phase 3 R81 Phase 2 knob status: `{gate.get('phase3_r81_phase2_knob_status')}`",
        f"- Phase 3 R82 quarterly-annual bridge status: `{gate.get('phase3_r82_quarterly_annual_bridge_status')}`",
        f"- Phase 3 R83 quarterly emission bridge status: `{gate.get('phase3_r83_quarterly_emission_bridge_status')}`",
        f"- Phase 3 R84 conserved ledger status: `{gate.get('phase3_r84_conserved_ledger_status')}`",
        f"- Phase 3 R85 forecast-grid status: `{gate.get('phase3_r85_forecast_grid_status')}`",
        f"- Phase 3 R86 annual-calibrated ledger status: `{gate.get('phase3_r86_annual_calibrated_ledger_status')}`",
        f"- Phase 3 R87 process calibration status: `{gate.get('phase3_r87_process_calibration_status')}`",
        f"- Phase 3 R88 guarded annual status: `{gate.get('phase3_r88_guarded_annual_status')}`",
        f"- Phase 3 R89 mechanism support status: `{gate.get('phase3_r89_mechanism_support_status')}`",
        f"- Phase 3 R90 claim-grade status: `{gate.get('phase3_r90_claim_grade_status')}`",
        f"- Phase 3 R91 mechanism expansion status: `{gate.get('phase3_r91_mechanism_expansion_status')}`",
        f"- Phase 3 R92 process-repair status: `{gate.get('phase3_r92_process_repair_status')}`",
        f"- Phase 3 R93 public-incumbent status: `{gate.get('phase3_r93_public_incumbent_status')}`",
        f"- Regional adapter stability status: `{gate.get('regional_adapter_stability_status')}`",
        f"- Regional split-guarded selector status: `{gate.get('regional_split_guarded_selector_status')}`",
        f"- Regional candidate ceiling status: `{gate.get('regional_candidate_ceiling_status')}`",
        f"- Determinant claim status: `{gate.get('determinant_claim_status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Claims",
        "",
        "| Claim | Scope | Status | Family | Allowed Claim | Limit |",
        "|---|---|---|---|---|---|",
    ]
    for row in list(report.get("claim_rows") or []):
        lines.append(
            f"| `{row.get('claim_id')}` | `{row.get('claim_scope')}` | `{row.get('claim_status')}` | "
            f"`{row.get('model_family')}` | {row.get('allowed_claim')} | {row.get('claim_limit')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r53_publication_claim_registry(
    *,
    run_id: str = R53_RUN_ID,
    r42_report_path: Path | None = None,
    r46_report_path: Path | None = None,
    r52_report_path: Path | None = None,
    r54_report_path: Path | None = None,
    r55_report_path: Path | None = None,
    r56_report_path: Path | None = None,
    r57_report_path: Path | None = None,
    r58_report_path: Path | None = None,
    r59_report_path: Path | None = None,
    r60_report_path: Path | None = None,
    r61_report_path: Path | None = None,
    r62_report_path: Path | None = None,
    r63_report_path: Path | None = None,
    r64_report_path: Path | None = None,
    r65_report_path: Path | None = None,
    r66_report_path: Path | None = None,
    r67_report_path: Path | None = None,
    r68_report_path: Path | None = None,
    r69_report_path: Path | None = None,
    r70_report_path: Path | None = None,
    r71_report_path: Path | None = None,
    r72_report_path: Path | None = None,
    r73_report_path: Path | None = None,
    r74_report_path: Path | None = None,
    r75_report_path: Path | None = None,
    r76_report_path: Path | None = None,
    r77_report_path: Path | None = None,
    r78_report_path: Path | None = None,
    r79_report_path: Path | None = None,
    r80_report_path: Path | None = None,
    r81_report_path: Path | None = None,
    r82_report_path: Path | None = None,
    r83_report_path: Path | None = None,
    r84_report_path: Path | None = None,
    r85_report_path: Path | None = None,
    r86_report_path: Path | None = None,
    r87_report_path: Path | None = None,
    r88_report_path: Path | None = None,
    r89_report_path: Path | None = None,
    r90_report_path: Path | None = None,
    r91_report_path: Path | None = None,
    r92_report_path: Path | None = None,
    r93_report_path: Path | None = None,
) -> dict[str, Any]:
    r42_path = Path(r42_report_path) if r42_report_path is not None else R42_DEFAULT_REPORT
    r46_path = Path(r46_report_path) if r46_report_path is not None else R46_DEFAULT_REPORT
    r52_path = Path(r52_report_path) if r52_report_path is not None else R52_DEFAULT_REPORT
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r55_path = Path(r55_report_path) if r55_report_path is not None else R55_DEFAULT_REPORT
    r56_path = Path(r56_report_path) if r56_report_path is not None else R56_DEFAULT_REPORT
    r57_path = Path(r57_report_path) if r57_report_path is not None else R57_DEFAULT_REPORT
    r58_path = Path(r58_report_path) if r58_report_path is not None else R58_DEFAULT_REPORT
    r59_path = Path(r59_report_path) if r59_report_path is not None else R59_DEFAULT_REPORT
    r60_path = Path(r60_report_path) if r60_report_path is not None else R60_DEFAULT_REPORT
    r61_path = Path(r61_report_path) if r61_report_path is not None else R61_DEFAULT_REPORT
    r62_path = Path(r62_report_path) if r62_report_path is not None else R62_DEFAULT_REPORT
    r63_path = Path(r63_report_path) if r63_report_path is not None else R63_DEFAULT_REPORT
    r64_path = Path(r64_report_path) if r64_report_path is not None else R64_DEFAULT_REPORT
    r65_path = Path(r65_report_path) if r65_report_path is not None else R65_DEFAULT_REPORT
    r66_path = Path(r66_report_path) if r66_report_path is not None else R66_DEFAULT_REPORT
    r67_path = Path(r67_report_path) if r67_report_path is not None else R67_DEFAULT_REPORT
    r68_path = Path(r68_report_path) if r68_report_path is not None else R68_DEFAULT_REPORT
    r69_path = Path(r69_report_path) if r69_report_path is not None else R69_DEFAULT_REPORT
    r70_path = Path(r70_report_path) if r70_report_path is not None else R70_DEFAULT_REPORT
    r71_path = Path(r71_report_path) if r71_report_path is not None else R71_DEFAULT_REPORT
    r72_path = Path(r72_report_path) if r72_report_path is not None else R72_DEFAULT_REPORT
    r73_path = Path(r73_report_path) if r73_report_path is not None else R73_DEFAULT_REPORT
    r74_path = Path(r74_report_path) if r74_report_path is not None else R74_DEFAULT_REPORT
    r75_path = Path(r75_report_path) if r75_report_path is not None else R75_DEFAULT_REPORT
    r76_path = Path(r76_report_path) if r76_report_path is not None else R76_DEFAULT_REPORT
    r77_path = Path(r77_report_path) if r77_report_path is not None else R77_DEFAULT_REPORT
    r78_path = Path(r78_report_path) if r78_report_path is not None else R78_DEFAULT_REPORT
    r79_path = Path(r79_report_path) if r79_report_path is not None else R79_DEFAULT_REPORT
    r80_path = Path(r80_report_path) if r80_report_path is not None else R80_DEFAULT_REPORT
    r81_path = Path(r81_report_path) if r81_report_path is not None else R81_DEFAULT_REPORT
    r82_path = Path(r82_report_path) if r82_report_path is not None else R82_DEFAULT_REPORT
    r83_path = Path(r83_report_path) if r83_report_path is not None else R83_DEFAULT_REPORT
    r84_path = Path(r84_report_path) if r84_report_path is not None else R84_DEFAULT_REPORT
    r85_path = Path(r85_report_path) if r85_report_path is not None else R85_DEFAULT_REPORT
    r86_path = Path(r86_report_path) if r86_report_path is not None else R86_DEFAULT_REPORT
    r87_path = Path(r87_report_path) if r87_report_path is not None else R87_DEFAULT_REPORT
    r88_path = Path(r88_report_path) if r88_report_path is not None else R88_DEFAULT_REPORT
    r89_path = Path(r89_report_path) if r89_report_path is not None else R89_DEFAULT_REPORT
    r90_path = Path(r90_report_path) if r90_report_path is not None else R90_DEFAULT_REPORT
    r91_path = Path(r91_report_path) if r91_report_path is not None else R91_DEFAULT_REPORT
    r92_path = Path(r92_report_path) if r92_report_path is not None else R92_DEFAULT_REPORT
    r93_path = Path(r93_report_path) if r93_report_path is not None else R93_DEFAULT_REPORT
    r42 = _load_report(r42_path)
    r46 = _load_report(r46_path)
    r52 = _load_report(r52_path)
    r54 = _load_report(r54_path)
    r55 = _load_report(r55_path)
    r56 = _load_report(r56_path)
    r57 = _load_report(r57_path)
    r58 = _load_report(r58_path)
    r59 = _load_report(r59_path)
    r60 = _load_report(r60_path)
    r61 = _load_report(r61_path)
    r62 = _load_report(r62_path)
    r63 = _load_report(r63_path)
    r64 = _load_report(r64_path)
    r65 = _load_report(r65_path)
    r66 = _load_report(r66_path)
    r67 = _load_report(r67_path)
    r68 = _load_report(r68_path)
    r69 = _load_report(r69_path)
    r70 = _load_report(r70_path)
    r71 = _load_report(r71_path)
    r72 = _load_report(r72_path)
    r73 = _load_report(r73_path)
    r74 = _load_report(r74_path)
    r75 = _load_report(r75_path)
    r76 = _load_report(r76_path)
    r77 = _load_report(r77_path)
    r78 = _load_report(r78_path)
    r79 = _load_report(r79_path)
    r80 = _load_report(r80_path)
    r81 = _load_report(r81_path)
    r82 = _load_report(r82_path)
    r83 = _load_report(r83_path)
    r84 = _load_report(r84_path)
    r85 = _load_report(r85_path)
    r86 = _load_report(r86_path)
    r87 = _load_report(r87_path)
    r88 = _load_report(r88_path)
    r89 = _load_report(r89_path)
    r90 = _load_report(r90_path)
    r91 = _load_report(r91_path)
    r92 = _load_report(r92_path)
    r93 = _load_report(r93_path)
    claim_rows = [
        _national_claim(r42, r42_path),
        _subnational_claim(r52, r52_path),
        _regional_adapter_claim(r54, r54_path),
        _adapter_stability_claim(r55, r55_path),
        _split_guarded_selector_claim(r56, r56_path),
        _regional_candidate_ceiling_claim(r57, r57_path),
        _regional_pareto_ensemble_claim(r58, r58_path),
        _regional_anchor_stable_ensemble_claim(r59, r59_path),
        _regional_experiment_queue_claim(r60, r60_path),
        _regional_split_risk_router_claim(r61, r61_path),
        _regional_leakage_expert_student_claim(r62, r62_path),
        _regional_online_expert_claim(r63, r63_path),
        _regional_leakage_support_gap_claim(r64, r64_path),
        _transmission_readiness_claim(r65, r65_path),
        _scientific_source_base_claim(r66, r66_path),
        _model_family_queue_claim(r67, r67_path),
        _bulk_external_ingest_claim(r68, r68_path),
        _bulk_signal_feature_claim(r69, r69_path),
        _scientific_model_build_queue_claim(r70, r70_path),
        _external_signal_branch_claim(
            r71,
            r71_path,
            claim_id="phase3_r71_service_intensity_capacity_branch",
            gate_key="service_intensity_capacity_gate",
            model_family="r71_service_intensity_capacity_branch",
            promoted_status="r71_forecast_safe_promoted",
        ),
        _external_signal_branch_claim(
            r72,
            r72_path,
            claim_id="phase3_r72_service_feature_selector_gate",
            gate_key="selector_gate",
            model_family="r72_train_backtested_service_feature_selector",
            promoted_status="r72_selector_promoted",
        ),
        _external_signal_branch_claim(
            r73,
            r73_path,
            claim_id="phase3_r73_external_signal_lag_falsification",
            gate_key="external_signal_gate",
            model_family="r73_external_signal_lag_falsification",
            promoted_status="r73_forecast_external_signal_promoted",
        ),
        _external_signal_branch_claim(
            r74,
            r74_path,
            claim_id="phase3_r74_external_tail_risk_selector",
            gate_key="tail_risk_gate",
            model_family="r74_external_tail_risk_selector",
            promoted_status="r74_tail_risk_selector_promoted",
        ),
        _bulk_unaids_annual_challenge_claim(r75, r75_path),
        _public_domain_annual_comparator_claim(r76, r76_path),
        _public_proxy_annual_gate_claim(r77, r77_path),
        _expanded_public_annual_comparator_claim(r78, r78_path),
        _expanded_public_proxy_annual_gate_claim(r79, r79_path),
        _public_annual_projection_head_claim(r80, r80_path),
        _phase2_knob_admissibility_claim(r81, r81_path),
        _quarterly_annual_bridge_claim(r82, r82_path),
        _quarterly_emission_bridge_claim(r83, r83_path),
        _conserved_quarterly_annual_ledger_claim(r84, r84_path),
        _annual_ledger_forecast_grid_claim(r85, r85_path),
        _annual_calibrated_forecast_grid_ledger_claim(r86, r86_path),
        _train_backtested_emission_process_calibration_claim(r87, r87_path),
        _guarded_annual_ledger_selector_claim(r88, r88_path),
        _incidence_mortality_mechanism_support_claim(r89, r89_path),
        _claim_grade_gate_claim(r90, r90_path),
        _mechanism_support_expansion_claim(r91, r91_path),
        _process_repair_experiment_queue_claim(r92, r92_path),
        _open_public_incumbent_comparator_claim(r93, r93_path),
        _determinant_claim(r46, r46_path),
    ]
    gate = _registry_gate(claim_rows)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r53_publication_claim_registry_report.json"
    md_path = analysis_dir / "r53_publication_claim_registry_report.md"
    csv_path = analysis_dir / "r53_claim_rows.csv"
    report = {
        "schema_version": R53_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "registry_gate": gate,
        "claim_rows": claim_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "claim_rows_csv": csv_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(csv_path, claim_rows)
    _write_markdown(md_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R53 publication claim registry.")
    parser.add_argument("--run-id", default=R53_RUN_ID)
    parser.add_argument("--r42-report-path", default=None)
    parser.add_argument("--r46-report-path", default=None)
    parser.add_argument("--r52-report-path", default=None)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r55-report-path", default=None)
    parser.add_argument("--r56-report-path", default=None)
    parser.add_argument("--r57-report-path", default=None)
    parser.add_argument("--r58-report-path", default=None)
    parser.add_argument("--r59-report-path", default=None)
    parser.add_argument("--r60-report-path", default=None)
    parser.add_argument("--r61-report-path", default=None)
    parser.add_argument("--r62-report-path", default=None)
    parser.add_argument("--r63-report-path", default=None)
    parser.add_argument("--r64-report-path", default=None)
    parser.add_argument("--r65-report-path", default=None)
    parser.add_argument("--r66-report-path", default=None)
    parser.add_argument("--r67-report-path", default=None)
    parser.add_argument("--r68-report-path", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--r70-report-path", default=None)
    parser.add_argument("--r71-report-path", default=None)
    parser.add_argument("--r72-report-path", default=None)
    parser.add_argument("--r73-report-path", default=None)
    parser.add_argument("--r74-report-path", default=None)
    parser.add_argument("--r75-report-path", default=None)
    parser.add_argument("--r76-report-path", default=None)
    parser.add_argument("--r77-report-path", default=None)
    parser.add_argument("--r78-report-path", default=None)
    parser.add_argument("--r79-report-path", default=None)
    parser.add_argument("--r80-report-path", default=None)
    parser.add_argument("--r81-report-path", default=None)
    parser.add_argument("--r82-report-path", default=None)
    parser.add_argument("--r83-report-path", default=None)
    parser.add_argument("--r84-report-path", default=None)
    parser.add_argument("--r85-report-path", default=None)
    parser.add_argument("--r86-report-path", default=None)
    parser.add_argument("--r87-report-path", default=None)
    parser.add_argument("--r88-report-path", default=None)
    parser.add_argument("--r89-report-path", default=None)
    parser.add_argument("--r90-report-path", default=None)
    parser.add_argument("--r91-report-path", default=None)
    parser.add_argument("--r92-report-path", default=None)
    parser.add_argument("--r93-report-path", default=None)
    args = parser.parse_args()
    run_r53_publication_claim_registry(
        run_id=str(args.run_id),
        r42_report_path=None if args.r42_report_path is None else Path(args.r42_report_path),
        r46_report_path=None if args.r46_report_path is None else Path(args.r46_report_path),
        r52_report_path=None if args.r52_report_path is None else Path(args.r52_report_path),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r55_report_path=None if args.r55_report_path is None else Path(args.r55_report_path),
        r56_report_path=None if args.r56_report_path is None else Path(args.r56_report_path),
        r57_report_path=None if args.r57_report_path is None else Path(args.r57_report_path),
        r58_report_path=None if args.r58_report_path is None else Path(args.r58_report_path),
        r59_report_path=None if args.r59_report_path is None else Path(args.r59_report_path),
        r60_report_path=None if args.r60_report_path is None else Path(args.r60_report_path),
        r61_report_path=None if args.r61_report_path is None else Path(args.r61_report_path),
        r62_report_path=None if args.r62_report_path is None else Path(args.r62_report_path),
        r63_report_path=None if args.r63_report_path is None else Path(args.r63_report_path),
        r64_report_path=None if args.r64_report_path is None else Path(args.r64_report_path),
        r65_report_path=None if args.r65_report_path is None else Path(args.r65_report_path),
        r66_report_path=None if args.r66_report_path is None else Path(args.r66_report_path),
        r67_report_path=None if args.r67_report_path is None else Path(args.r67_report_path),
        r68_report_path=None if args.r68_report_path is None else Path(args.r68_report_path),
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        r70_report_path=None if args.r70_report_path is None else Path(args.r70_report_path),
        r71_report_path=None if args.r71_report_path is None else Path(args.r71_report_path),
        r72_report_path=None if args.r72_report_path is None else Path(args.r72_report_path),
        r73_report_path=None if args.r73_report_path is None else Path(args.r73_report_path),
        r74_report_path=None if args.r74_report_path is None else Path(args.r74_report_path),
        r75_report_path=None if args.r75_report_path is None else Path(args.r75_report_path),
        r76_report_path=None if args.r76_report_path is None else Path(args.r76_report_path),
        r77_report_path=None if args.r77_report_path is None else Path(args.r77_report_path),
        r78_report_path=None if args.r78_report_path is None else Path(args.r78_report_path),
        r79_report_path=None if args.r79_report_path is None else Path(args.r79_report_path),
        r80_report_path=None if args.r80_report_path is None else Path(args.r80_report_path),
        r81_report_path=None if args.r81_report_path is None else Path(args.r81_report_path),
        r82_report_path=None if args.r82_report_path is None else Path(args.r82_report_path),
        r83_report_path=None if args.r83_report_path is None else Path(args.r83_report_path),
        r84_report_path=None if args.r84_report_path is None else Path(args.r84_report_path),
        r85_report_path=None if args.r85_report_path is None else Path(args.r85_report_path),
        r86_report_path=None if args.r86_report_path is None else Path(args.r86_report_path),
        r87_report_path=None if args.r87_report_path is None else Path(args.r87_report_path),
        r88_report_path=None if args.r88_report_path is None else Path(args.r88_report_path),
        r89_report_path=None if args.r89_report_path is None else Path(args.r89_report_path),
        r90_report_path=None if args.r90_report_path is None else Path(args.r90_report_path),
        r91_report_path=None if args.r91_report_path is None else Path(args.r91_report_path),
        r92_report_path=None if args.r92_report_path is None else Path(args.r92_report_path),
        r93_report_path=None if args.r93_report_path is None else Path(args.r93_report_path),
    )


if __name__ == "__main__":
    _main()
