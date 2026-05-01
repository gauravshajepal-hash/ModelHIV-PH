from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import write_json


MODEL_CONTRACT_SCHEMA_VERSION = "phase3_model_contract.v1"
HAZARD_SEMANTICS_SCHEMA_VERSION = "phase3_hazard_semantics.v1"
OBSERVATION_CONTRACT_SCHEMA_VERSION = "epigraph_observation_allowed_use.v1"
OBSERVATION_ROLE_LEDGER_SCHEMA_VERSION = "phase3_observation_role_ledger_schema.v1"
LINEAGE_LOCK_SCHEMA_VERSION = "epigraph_project_lineage_lock.v1"


def build_model_contract(
    *,
    incidence_enabled: bool,
    observation_calibration_enabled: bool,
    phase2_direct_enabled: bool,
    phase2_hidden_enabled: bool,
    decomposition_enabled: bool = False,
) -> dict[str, Any]:
    if incidence_enabled and observation_calibration_enabled:
        model_kind = "mechanistic_transition_with_latent_inflow_and_observation_calibration"
    elif incidence_enabled:
        model_kind = "mechanistic_transition_with_latent_inflow_forecast"
    elif observation_calibration_enabled:
        model_kind = "mechanistic_transition_with_observation_calibration"
    else:
        model_kind = "mechanistic_transition_forecast"
    return {
        "schema_version": MODEL_CONTRACT_SCHEMA_VERSION,
        "model_kind": model_kind,
        "primary_output_contract": "blocked_time_cascade_forecast",
        "transition_contract": "hazards_fit_from_train_only_reconstructed_state_transitions",
        "incidence_contract": (
            "train_only_effective_population_incidence_to_U_with_evidence_typed_exit_channels"
            if incidence_enabled
            else "not_used"
        ),
        "observation_contract": (
            "train_only_observation_calibration"
            if observation_calibration_enabled
            else "raw_state_simulation_outputs"
        ),
        "phase2_contract": {
            "direct_terms": (
                "forecast_origin_safe_structured_priors"
                if phase2_direct_enabled
                else "not_used"
            ),
            "hidden_terms": (
                "forecast_origin_safe_shared_latent_shocks"
                if phase2_hidden_enabled
                else "not_used"
            ),
        },
        "decomposition_contract": (
            "train_origin_hiv_stream_decomposition_controls"
            if decomposition_enabled
            else "not_used"
        ),
        "allowed_claims": [
            "development_blocked_time_forecast",
            "forecast_origin_hazard_state_research",
            "structured_prior_ablation_when_phase2_terms_enabled",
        ],
        "disallowed_claims": [
            "validated_incidence_truth",
            "validated_incidence_mechanism",
            "validated_causal_transition_effect",
            "validated_provincial_truth",
            "policy_effect_estimate",
        ],
    }


def build_hazard_semantics(
    *,
    incidence_enabled: bool,
    observation_calibration_enabled: bool,
    phase2_direct_enabled: bool,
    phase2_hidden_enabled: bool,
    decomposition_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": HAZARD_SEMANTICS_SCHEMA_VERSION,
        "fitted_transition_hazard": {
            "status": "emitted",
            "meaning": "hazard parameters fitted on train-only reconstructed state transitions",
            "allowed_use": [
                "mechanistic_research_sidecar",
                "state_simulation_inside_blocked_time_forecast",
            ],
            "not_allowed_use": [
                "directly_observed_hazard_truth",
                "causal_transition_effect",
            ],
        },
        "forecast_transition_hazard": {
            "status": "emitted",
            "meaning": "future hazard path forecast from train-only fitted hazard dynamics",
            "allowed_use": [
                "holdout_state_simulation",
                "blocked_time_forecast_comparison",
            ],
            "not_allowed_use": [
                "post_hoc_hazard_validation_without_transition_targets",
            ],
        },
        "latent_incidence_inflow": {
            "status": "emitted" if incidence_enabled else "not_emitted",
            "meaning": "train-only inferred S_eff -> incidence -> U inflow with state-specific exit sidecars from reconstructed stock balance",
            "allowed_use": [
                "state_replenishment_inside_blocked_time_forecast",
                "diagnostic_mass_balance_analysis",
            ],
            "not_allowed_use": [
                "validated_incidence_truth",
                "validated_population_denominator_hazard_claim",
                "causal_incidence_effect",
            ],
        },
        "state_specific_exit_flows": {
            "status": "emitted" if incidence_enabled else "not_emitted",
            "meaning": "state-indexed external exits/removals split into evidence-typed mortality/removal, treatment non-initiation, and unresolved-removal channels",
            "allowed_use": [
                "conserved_state_simulation",
                "loss_to_follow_up_or_mortality_hypothesis_generation",
            ],
            "not_allowed_use": [
                "observed_state_specific_mortality_truth",
                "validated_treatment_attrition_mechanism",
            ],
        },
        "care_leakage_channels": {
            "status": "emitted" if incidence_enabled else "not_emitted",
            "meaning": "ART interruption, re-engagement, VL testing, VL-tested unsuppressed, and suppression are represented as explicit back-half states/transitions rather than endpoint corrections",
            "allowed_use": [
                "channel_ablation_diagnostics",
                "support_aware_failure_anatomy",
                "third_95_process_hypothesis_generation_when_support_gates_pass",
            ],
            "not_allowed_use": [
                "validated_policy_effect_estimate",
                "identified_individual_care_history",
            ],
        },
        "predictive_observation_head": {
            "status": "emitted" if observation_calibration_enabled else "not_emitted",
            "meaning": "train-only calibration from simulated state outputs to observed cascade metrics",
            "allowed_use": ["observed_endpoint_forecast"],
            "not_allowed_use": ["mechanistic_transition_identification"],
        },
        "diagnostic_derived_hazard": {
            "status": "not_emitted_by_phase3_dynamic_hazard_model",
            "meaning": "reserved for observation-first models such as R10 when hazards are derived after endpoint prediction",
            "allowed_use": ["diagnostic_visualization", "model_interpretation_with_warning"],
            "not_allowed_use": ["fitted_mechanistic_hazard_claim"],
        },
        "phase2_effect_semantics": {
            "direct_terms": (
                "structured_prior_on_hazard_residuals"
                if phase2_direct_enabled
                else "not_used"
            ),
            "hidden_terms": (
                "shared_latent_shock_sidecar_on_hazard_residuals"
                if phase2_hidden_enabled
                else "not_used"
            ),
        },
        "hiv_decomposition_controls": {
            "status": "emitted" if decomposition_enabled else "not_emitted",
            "meaning": "train-origin decomposition of HIV streams into trend, reporting/support shift, and residual-shock controls used as bounded hazard drivers",
            "allowed_use": [
                "blocked_time_hazard_driver_ablation",
                "bounded_scenario_diagnostic_if_gates_pass",
            ],
            "not_allowed_use": [
                "causal_reporting_shift_claim",
                "policy_effect_estimate",
                "future_observation_leakage",
            ],
        },
    }


def build_observation_allowed_use_schema() -> dict[str, Any]:
    return {
        "schema_version": OBSERVATION_CONTRACT_SCHEMA_VERSION,
        "required_fields": [
            "phase",
            "artifact_path",
            "metric_name",
            "geography",
            "time_label",
            "time_support",
            "value_role",
            "measurement_role",
            "source_tier",
            "provenance_tier",
            "allowed_downstream_roles",
            "disallowed_claims",
        ],
        "time_support_values": [
            "instant_or_snapshot",
            "monthly_period",
            "quarterly_period",
            "annual_period",
            "multi_period_bridge",
            "latent_time_support",
        ],
        "value_roles": [
            "direct_measurement",
            "anchor_truth",
            "bridge_observed",
            "rule_based_extrapolated",
            "latent_imputed",
            "proxy_prior",
            "context_only",
            "model_output",
        ],
        "allowed_downstream_roles": [
            "validation_truth",
            "training_likelihood",
            "train_only_imputation",
            "structured_prior",
            "auxiliary_regularization",
            "contextual_interpretation",
            "diagnostic_visualization",
        ],
        "claim_levels": {
            "validation_truth": "requires direct or explicitly accepted anchor evidence for the target geography/time support",
            "training_likelihood": "requires non-future observed or accepted bridge evidence inside the training split",
            "structured_prior": "allowed for Phase 2 direct/hidden structures after forecast-origin safety checks",
            "auxiliary_regularization": "allowed for subnational proxy evidence that is not validated provincial truth",
            "contextual_interpretation": "allowed for literature, qualitative, or weak proxy evidence",
        },
        "always_disallowed_without_extra_validation": [
            "causal_policy_effect",
            "validated_mechanistic_transition",
            "validated_provincial_truth",
            "third_95_strong_mechanistic_claim",
        ],
    }


def build_observation_role_ledger_schema() -> dict[str, Any]:
    return {
        "schema_version": OBSERVATION_ROLE_LEDGER_SCHEMA_VERSION,
        "required_fields": [
            "source_id",
            "source_path",
            "metric_id",
            "time_start",
            "time_end",
            "time_granularity",
            "geography",
            "population",
            "value",
            "unit",
            "extraction_method",
            "source_tier",
            "observation_role",
            "allowed_use",
            "support_partition",
            "leakage_status",
            "measurement_semantics",
            "row_hash",
        ],
        "observation_roles": [
            "direct_target",
            "auxiliary_likelihood",
            "validation_only",
            "prior_context",
            "quarantined",
        ],
        "measurement_semantics": [
            "stock_anchor",
            "flow_count",
            "modeled_estimate",
            "proportion",
            "denominator",
            "reporting_process_covariate",
            "determinant_covariate",
        ],
        "support_partitions": [
            "common_support",
            "expanded_support",
        ],
        "leakage_status_values": [
            "split_unassigned",
            "train_only",
            "holdout_observed",
            "quarantined_future",
        ],
        "contract_notes": [
            "quarantined rows are not scorable unless an explicit diagnostic override is present",
            "validation_only rows may not be promoted to direct training truth by downstream experiments",
            "support_partition is defined relative to the selected baseline archive run",
        ],
    }


def build_project_lineage_lock(project_root: Path, phase3_dynamic_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    phase3_dynamic_root = phase3_dynamic_root.resolve()
    root_phase3 = project_root / "src" / "epigraph_ph" / "phase3"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": LINEAGE_LOCK_SCHEMA_VERSION,
        "project_root": project_root.as_posix(),
        "canonical_phase3_dynamics_path": phase3_dynamic_root.as_posix(),
        "canonical_phase3_status": "active_for_new_phase3_dynamics_claims",
        "fixed_contracts": [
            "GAP-P3D-001 train-only state parameters and metric scales",
            "GAP-P3D-002 forecast-origin-safe Phase 2 direct and hidden priors",
            "GAP-P3D-003 model_contract and hazard_semantics emitted for Phase3(dynamic)",
        ],
        "quarantined_paths": [
            {
                "path": root_phase3.as_posix(),
                "status": "quarantined_for_new_scientific_claims",
                "allowed_use": "historical_reference_only_until_patched_and_replayed_under_current_contracts",
                "reason": "root Phase 3 contains stale pre-contract experiments and may not share Phase3(dynamic) leakage fixes",
            },
            {
                "path": (project_root / "artifacts" / "runs").as_posix(),
                "status": "mixed_lineage_archive",
                "allowed_use": "evidence_source_with_per_run_contract_check",
                "reason": "root artifact runs include multiple historical contracts and cannot be treated as one homogeneous validation source",
            },
        ],
        "promotion_rule": "only artifacts with current model_contract, hazard_semantics, benchmark_contract, support-tier reporting, and train-only proof can support new champion claims",
    }


def write_project_contract_artifacts(*, project_root: Path, phase3_dynamic_root: Path) -> dict[str, Path]:
    audit_dir = project_root / "artifacts" / "scientific_audits"
    lineage_path = audit_dir / "project_lineage_lock_20260424.json"
    observation_path = audit_dir / "observation_allowed_use_schema_20260424.json"
    ledger_schema_path = audit_dir / "observation_role_ledger_schema_20260425.json"
    write_json(lineage_path, build_project_lineage_lock(project_root, phase3_dynamic_root))
    write_json(observation_path, build_observation_allowed_use_schema())
    write_json(ledger_schema_path, build_observation_role_ledger_schema())
    return {
        "lineage_lock": lineage_path,
        "observation_allowed_use_schema": observation_path,
        "observation_role_ledger_schema": ledger_schema_path,
    }
