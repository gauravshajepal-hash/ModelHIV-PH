from __future__ import annotations

from dataclasses import dataclass
from datetime import date

TRANSITION_NAMES: tuple[str, ...] = (
    "U_to_D",
    "D_to_A",
    "A_to_V",
    "A_to_L",
    "L_to_A",
)

KP_COLLAPSED_NAMES: tuple[str, ...] = ("msm", "tgw", "other")

ANALYTICS_EXPERIMENT_IDS: tuple[str, ...] = (
    "AN-01A-national-yearly-factor-evolution",
    "AN-01B-regional-yearly-factor-evolution",
    "AN-01C-factor-importance-drift-report",
    "AN-02A-factor-to-transition-map",
    "AN-02B-factor-to-kp-map",
    "AN-02C-kp-transition-relevance-drift",
    "AN-03A-phase2-aggregation-loss-diagnostic",
    "AN-03B-phase2-partial-denoising-diagnostic",
    "AN-03C-phase3-strict-spec-gap-audit",
    "DIAG-01A-strict-diagnosis-kernel-blocked-time",
    "DIAG-01B-diagnosis-budget-sweep",
    "DIAG-02A-integrated-champion-strict-promotion",
    "DIAG-02B-blocked-time-incidence-ud-bundle-search",
    "DIAG-02C-blocked-time-incidence-family-search",
    "AGE-01A-age-evidence-audit",
    "HMBA-00-hierarchical-contract-freeze",
    "HMBA-01-module-local-direct-bundle-promotion",
    "HMBA-02-hierarchical-geography-layer",
    "HMBA-03-joint-hierarchical-autoresearch-loop",
    "HMBA-03A-interpretation-dashboard",
)

MECHANISTIC_EXPERIMENT_IDS: tuple[str, ...] = (
    "AGE-01B-youth-diagnosis-modifier",
    "AGE-01C-youth-downstream-modifier",
    "MECH-01A-national-udavl-baseline",
    "MECH-01B-mesoscopic-transition-helpers",
    "MECH-01C-residual-transition-helpers",
    "MECH-01D-diagnosis-locked-residual-helpers",
    "MECH-01E-anchored-downstream-residual-helpers",
    "KP-01A-national-kp-lite-overlay",
    "DECOMP-01A-transition-channel-decomposition",
    "DECOMP-01B-channel-driver-coupling",
    "DECOMP-01C-fused-mechanistic-forecast",
    "DECOMP-01D-skill-gated-fused-forecast",
    "DECOMP-01E-loo-gated-fused-forecast",
    "DECOMP-01F-reverse-grasp-peak-clusters",
    "PEAK-01A-regional-kp-window-detector",
    "PEAK-01B-detector-gated-fused-forecast",
    "PEAK-01C-region-only-window-detector",
    "PEAK-01D-region-only-gated-fused-forecast",
    "PEAK-01E-region-plus-kp-modifier-detector",
    "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast",
)

TR_V2_EXPERIMENT_IDS: tuple[str, ...] = (
    "TR-V2-00-age01b-baseline-lock",
    "TR-V2-01-phase2-direct-hazard-priors",
    "TR-V2-02-phase2-hidden-shock-hazards",
    "TR-V2-03-phase2-ablation-suite",
    "PHASE3-V2-INT-explicit-incidence-autoresearch",
)


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    experiment_id: str
    cli_name: str
    description: str
    expected_outputs: tuple[str, ...]


EXPERIMENT_REGISTRY: dict[str, ExperimentDefinition] = {
    "AN-01A-national-yearly-factor-evolution": ExperimentDefinition(
        experiment_id="AN-01A-national-yearly-factor-evolution",
        cli_name="an-01a",
        description="Yearly national mesoscopic factor evolution.",
        expected_outputs=(
            "yearly_factor_scores_national.json",
            "national_factor_heatmap.png",
            "national_factor_trajectories.png",
            "ranked_yearly_tables.json",
        ),
    ),
    "AN-01B-regional-yearly-factor-evolution": ExperimentDefinition(
        experiment_id="AN-01B-regional-yearly-factor-evolution",
        cli_name="an-01b",
        description="Yearly regional mesoscopic factor evolution.",
        expected_outputs=(
            "yearly_factor_scores_region.json",
            "regional_factor_heatmap.png",
            "regional_factor_trajectories.png",
        ),
    ),
    "AN-01C-factor-importance-drift-report": ExperimentDefinition(
        experiment_id="AN-01C-factor-importance-drift-report",
        cli_name="an-01c",
        description="Drift report for national and regional factor scores.",
        expected_outputs=(
            "factor_drift_report.json",
            "factor_drift_summary.png",
        ),
    ),
    "AN-02A-factor-to-transition-map": ExperimentDefinition(
        experiment_id="AN-02A-factor-to-transition-map",
        cli_name="an-02a",
        description="Map retained factors onto exact transition channels.",
        expected_outputs=(
            "factor_transition_relevance.json",
            "factor_transition_heatmap.png",
        ),
    ),
    "AN-02B-factor-to-kp-map": ExperimentDefinition(
        experiment_id="AN-02B-factor-to-kp-map",
        cli_name="an-02b",
        description="Map retained factors onto collapsed KP relevance groups.",
        expected_outputs=(
            "factor_kp_relevance.json",
            "factor_kp_heatmap.png",
        ),
    ),
    "AN-02C-kp-transition-relevance-drift": ExperimentDefinition(
        experiment_id="AN-02C-kp-transition-relevance-drift",
        cli_name="an-02c",
        description="Combine yearly factor evolution with KP and transition relevance.",
        expected_outputs=(
            "kp_transition_relevance.json",
            "kp_transition_drift.png",
        ),
    ),
    "AN-03A-phase2-aggregation-loss-diagnostic": ExperimentDefinition(
        experiment_id="AN-03A-phase2-aggregation-loss-diagnostic",
        cli_name="an-03a",
        description="Quantify how much provincially promoted Phase 2 signal is lost when collapsing province-level factor fields into the national mechanistic frontier.",
        expected_outputs=(
            "aggregation_loss_diagnostic.json",
            "provincial_evidence_summary.json",
            "module_bundle_seed_report.json",
            "aggregation_loss_top_factors.png",
            "aggregation_loss_module_heatmap.png",
        ),
    ),
    "AN-03B-phase2-partial-denoising-diagnostic": ExperimentDefinition(
        experiment_id="AN-03B-phase2-partial-denoising-diagnostic",
        cli_name="an-03b",
        description="Test data-driven partial denoising on retained province-level Phase 2 factor fields before wiring any denoising into HMBA optimization.",
        expected_outputs=(
            "partial_denoising_factor_delta.json",
            "partial_denoising_module_delta.json",
            "partial_denoising_summary.json",
            "partial_denoising_module_delta.png",
            "partial_denoising_rank_compression.png",
            "partial_denoising_priority_comparison.png",
        ),
    ),
    "AN-03C-phase3-strict-spec-gap-audit": ExperimentDefinition(
        experiment_id="AN-03C-phase3-strict-spec-gap-audit",
        cli_name="an-03c",
        description="Audit the strict Phase 3 mathematical spec against the implemented code and quantify the conditional gap between the current age-in-state branches and a stricter spec surrogate.",
        expected_outputs=(
            "spec_vs_code_table.json",
            "spec_vs_code_table.md",
            "strict_spec_divergence_summary.json",
            "diagnosis_divergence.json",
            "care_divergence.json",
            "combined_divergence.json",
            "strict_spec_gap_dashboard.md",
            "strict_spec_gap_loss_delta.png",
            "strict_spec_prediction_divergence.png",
            "strict_spec_bic_trace.png",
        ),
    ),
    "DIAG-01A-strict-diagnosis-kernel-blocked-time": ExperimentDefinition(
        experiment_id="DIAG-01A-strict-diagnosis-kernel-blocked-time",
        cli_name="diag-01a",
        description="Implement a stricter multi-parameter diagnosis-delay kernel on top of the current diagnosis branch and evaluate it with a frozen blocked-time gate before touching the combined model.",
        expected_outputs=(
            "blocked_time_contract.json",
            "candidate_frontier.json",
            "blocked_time_evaluation.json",
            "blocked_time_model_comparison.json",
            "blocked_time_prediction_rows.json",
            "strict_diagnosis_dashboard.md",
            "blocked_time_candidate_frontier.png",
            "blocked_time_model_comparison.png",
            "blocked_time_predictions.png",
            "blocked_time_kernel_profile.png",
        ),
    ),
    "DIAG-01B-diagnosis-budget-sweep": ExperimentDefinition(
        experiment_id="DIAG-01B-diagnosis-budget-sweep",
        cli_name="diag-01b",
        description="Audit diagnosis-kernel ranking stability under a predeclared compute-budget ladder before increasing optimizer effort in publication-facing blocked-time evaluation.",
        expected_outputs=(
            "budget_sweep_contract.json",
            "budget_sweep_rows.json",
            "budget_sweep_champions.json",
            "budget_sweep_summary.json",
            "budget_sweep_dashboard.md",
            "budget_sweep_loss_curves.png",
            "budget_sweep_champion_trace.png",
            "budget_sweep_compute_scatter.png",
        ),
    ),
    "DIAG-02A-integrated-champion-strict-promotion": ExperimentDefinition(
        experiment_id="DIAG-02A-integrated-champion-strict-promotion",
        cli_name="diag-02a",
        description="Promote the strict diagnosis kernel into the current integrated PHASE3-V2 champion while keeping downstream care fixed, then compare the promoted branch against the champion under a blocked-time gate.",
        expected_outputs=(
            "promotion_blocked_time_contract.json",
            "promotion_candidate_frontier.json",
            "promotion_evaluation.json",
            "promotion_model_comparison.json",
            "promotion_prediction_rows.json",
            "promotion_dashboard.md",
            "promotion_candidate_frontier.png",
            "promotion_model_comparison.png",
            "promotion_predictions.png",
            "promotion_kernel_profile.png",
        ),
    ),
    "DIAG-02B-blocked-time-incidence-ud-bundle-search": ExperimentDefinition(
        experiment_id="DIAG-02B-blocked-time-incidence-ud-bundle-search",
        cli_name="diag-02b",
        description="Fix the promoted strict diagnosis branch as the reference, keep downstream care fixed, and search only incidence and U_to_D direct Phase 2 bundles under the blocked-time contract.",
        expected_outputs=(
            "bundle_search_contract.json",
            "bundle_search_frontier.json",
            "bundle_search_traces.json",
            "bundle_search_evaluation.json",
            "bundle_search_model_comparison.json",
            "bundle_search_prediction_rows.json",
            "bundle_search_dashboard.md",
            "bundle_search_frontier.png",
            "bundle_search_traces.png",
            "bundle_search_model_comparison.png",
            "bundle_search_predictions.png",
            "bundle_search_selection.png",
        ),
    ),
    "DIAG-02C-blocked-time-incidence-family-search": ExperimentDefinition(
        experiment_id="DIAG-02C-blocked-time-incidence-family-search",
        cli_name="diag-02c",
        description="Freeze the promoted strict diagnosis branch as the reference, keep downstream care fixed, and search stronger incidence-module families under the blocked-time contract.",
        expected_outputs=(
            "incidence_family_contract.json",
            "incidence_family_frontier.json",
            "incidence_family_evaluation.json",
            "incidence_family_model_comparison.json",
            "incidence_family_prediction_rows.json",
            "incidence_family_dashboard.md",
            "incidence_family_frontier.png",
            "incidence_family_model_comparison.png",
            "incidence_family_predictions.png",
            "incidence_family_trajectories.png",
        ),
    ),
    "HMBA-00-hierarchical-contract-freeze": ExperimentDefinition(
        experiment_id="HMBA-00-hierarchical-contract-freeze",
        cli_name="hmba-00",
        description="Freeze the hierarchical module-bundle autoresearch contract by materializing the national observed panel, provincial auxiliary evidence panel, and module seed manifest from the current provincial evidence overlap.",
        expected_outputs=(
            "contract_snapshot.json",
            "module_seed_manifest.json",
            "evidence_panel_summary.json",
            "national_observation_values.npz",
            "national_observation_mask.npz",
            "provincial_auxiliary_state_shares.npz",
            "regional_auxiliary_state_shares.npz",
            "national_auxiliary_state_shares.npz",
            "coverage_frontier.png",
            "module_seed_frontier.png",
        ),
    ),
    "HMBA-01-module-local-direct-bundle-promotion": ExperimentDefinition(
        experiment_id="HMBA-01-module-local-direct-bundle-promotion",
        cli_name="hmba-01",
        description="Run exact module-local direct-bundle promotion on top of the frozen HMBA-00 contract while keeping hidden Phase 2 structure separate.",
        expected_outputs=(
            "module_local_frontier.json",
            "module_promotion_results.json",
            "combined_promoted_model.json",
            "single_bundle_gain_heatmap.png",
            "module_promotion_trace.png",
            "combined_model_comparison.png",
        ),
    ),
    "HMBA-02-hierarchical-geography-layer": ExperimentDefinition(
        experiment_id="HMBA-02-hierarchical-geography-layer",
        cli_name="hmba-02",
        description="Fit the actual HMBA hierarchical national-region-province direct-score layer on top of the frozen HMBA contract while keeping hidden Phase 2 structure separate.",
        expected_outputs=(
            "hierarchical_module_coefficients.json",
            "hierarchical_auxiliary_evaluation.json",
            "hierarchical_combined_model.json",
            "hierarchical_auxiliary_fit_comparison.png",
            "hierarchical_pooling_profile.png",
            "hierarchical_combined_model_comparison.png",
        ),
    ),
    "HMBA-03-joint-hierarchical-autoresearch-loop": ExperimentDefinition(
        experiment_id="HMBA-03-joint-hierarchical-autoresearch-loop",
        cli_name="hmba-03",
        description="Run a joint autoresearch loop that searches module bundles and hierarchy depth together under the HMBA national-plus-auxiliary dual gate.",
        expected_outputs=(
            "joint_search_frontier.json",
            "joint_search_trace.json",
            "joint_hierarchical_model.json",
            "joint_search_frontier.png",
            "joint_depth_assignment.png",
            "joint_model_comparison.png",
        ),
    ),
    "HMBA-03A-interpretation-dashboard": ExperimentDefinition(
        experiment_id="HMBA-03A-interpretation-dashboard",
        cli_name="hmba-03a",
        description="Generate a paper-quality interpretation dashboard from the emitted HMBA-03 search frontier, accepted trace, and spike candidates.",
        expected_outputs=(
            "interpretation_dashboard.json",
            "interpretation_dashboard.md",
            "accepted_step_effect_table.json",
            "module_effect_table.json",
            "spike_candidate_annotations.json",
            "zoomed_loss_deltas.png",
            "accepted_step_deltas.png",
            "module_effect_table.png",
            "spike_candidate_annotations.png",
            "interpretation_dashboard.png",
        ),
    ),
    "AGE-01A-age-evidence-audit": ExperimentDefinition(
        experiment_id="AGE-01A-age-evidence-audit",
        cli_name="age-01a",
        description="Audit direct, auxiliary, and prior-supported age evidence before enabling any youth-conditioned transition branch.",
        expected_outputs=(
            "age_evidence_audit.json",
            "age_signal_coverage_summary.json",
            "age_evidence_heatmap.png",
        ),
    ),
    "AGE-01B-youth-diagnosis-modifier": ExperimentDefinition(
        experiment_id="AGE-01B-youth-diagnosis-modifier",
        cli_name="age-01b",
        description="Apply a youth-share-based diagnosis modifier on top of the kept PEAK-01F branch while leaving downstream hazards unchanged.",
        expected_outputs=(
            "baseline_comparison.json",
            "evaluation.json",
            "mechanistic_forecast.json",
            "age_diagnosis_modifier_summary.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "AGE-01C-youth-downstream-modifier": ExperimentDefinition(
        experiment_id="AGE-01C-youth-downstream-modifier",
        cli_name="age-01c",
        description="Apply youth-share-based downstream age modulation only on D_to_A and A_to_V on top of the kept AGE-01B branch.",
        expected_outputs=(
            "baseline_comparison.json",
            "evaluation.json",
            "mechanistic_forecast.json",
            "age_downstream_modifier_summary.json",
            "age_transition_effect_heatmap.png",
            "forecast_vs_age01b.png",
        ),
    ),
    "MECH-01A-national-udavl-baseline": ExperimentDefinition(
        experiment_id="MECH-01A-national-udavl-baseline",
        cli_name="mech-01a",
        description="National U/D/A/V/L baseline built on the current winning front-half forecast.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "MECH-01B-mesoscopic-transition-helpers": ExperimentDefinition(
        experiment_id="MECH-01B-mesoscopic-transition-helpers",
        cli_name="mech-01b",
        description="National U/D/A/V/L mechanistic baseline with transition-specific mesoscopic helpers.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "mesoscopic_transition_helper_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "MECH-01C-residual-transition-helpers": ExperimentDefinition(
        experiment_id="MECH-01C-residual-transition-helpers",
        cli_name="mech-01c",
        description="Residual mesoscopic corrections layered onto the fixed MECH-01A transition path.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "residual_transition_helper_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "MECH-01D-diagnosis-locked-residual-helpers": ExperimentDefinition(
        experiment_id="MECH-01D-diagnosis-locked-residual-helpers",
        cli_name="mech-01d",
        description="Residual mesoscopic corrections on downstream transitions only, with U_to_D locked to the MECH-01A baseline.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "residual_transition_helper_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "MECH-01E-anchored-downstream-residual-helpers": ExperimentDefinition(
        experiment_id="MECH-01E-anchored-downstream-residual-helpers",
        cli_name="mech-01e",
        description="Downstream residual corrections anchored to the first MECH-01A holdout state, with U_to_D locked to the MECH-01A baseline.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "residual_transition_helper_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "KP-01A-national-kp-lite-overlay": ExperimentDefinition(
        experiment_id="KP-01A-national-kp-lite-overlay",
        cli_name="kp-01a",
        description="Weak national KP overlay on top of the kept MECH-01E branch, applied only to downstream hazards.",
        expected_outputs=(
            "fit_artifact.json",
            "evaluation.json",
            "transition_hazard_summary.json",
            "kp_overlay_summary.json",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "DECOMP-01A-transition-channel-decomposition": ExperimentDefinition(
        experiment_id="DECOMP-01A-transition-channel-decomposition",
        cli_name="decomp-01a",
        description="Decompose the kept MECH-01E national transition hazards into HIV-specific trend, shock, and residual channels.",
        expected_outputs=(
            "channel_decomposition.npz",
            "channel_summary.json",
        ),
    ),
    "DECOMP-01B-channel-driver-coupling": ExperimentDefinition(
        experiment_id="DECOMP-01B-channel-driver-coupling",
        cli_name="decomp-01b",
        description="Couple transition-channel hazards to mesoscopic drivers using aligned factor trend, shock, and residual components.",
        expected_outputs=(
            "channel_driver_weights.json",
            "channel_driver_heatmap.png",
        ),
    ),
    "DECOMP-01C-fused-mechanistic-forecast": ExperimentDefinition(
        experiment_id="DECOMP-01C-fused-mechanistic-forecast",
        cli_name="decomp-01c",
        description="Fuse supported transition-channel drivers back into the anchored MECH-01E mechanistic simulator while preserving unsupported pairs on the kept baseline.",
        expected_outputs=(
            "hazard_reconstruction.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "DECOMP-01D-skill-gated-fused-forecast": ExperimentDefinition(
        experiment_id="DECOMP-01D-skill-gated-fused-forecast",
        cli_name="decomp-01d",
        description="Apply train-skill gating to supported transition-channel drivers before fusing them back into the anchored MECH-01E mechanistic simulator.",
        expected_outputs=(
            "hazard_reconstruction.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "DECOMP-01E-loo-gated-fused-forecast": ExperimentDefinition(
        experiment_id="DECOMP-01E-loo-gated-fused-forecast",
        cli_name="decomp-01e",
        description="Apply leave-one-quarter-out channel skill gating to supported transition-channel drivers before fusing them back into the anchored MECH-01E mechanistic simulator.",
        expected_outputs=(
            "hazard_reconstruction.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "DECOMP-01F-reverse-grasp-peak-clusters": ExperimentDefinition(
        experiment_id="DECOMP-01F-reverse-grasp-peak-clusters",
        cli_name="decomp-01f",
        description="Apply a reverse-GRASP-style peak and cluster refinement on downstream transition channels after the kept DECOMP-01E fusion.",
        expected_outputs=(
            "hazard_reconstruction.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "PEAK-01A-regional-kp-window-detector": ExperimentDefinition(
        experiment_id="PEAK-01A-regional-kp-window-detector",
        cli_name="peak-01a",
        description="Detect regional and KP-conditioned downstream peak windows from quarter-level mesoscopic fields and decompose them into transition-channel gate signals.",
        expected_outputs=(
            "peak_window_detector.json",
            "peak_window_heatmap.png",
        ),
    ),
    "PEAK-01B-detector-gated-fused-forecast": ExperimentDefinition(
        experiment_id="PEAK-01B-detector-gated-fused-forecast",
        cli_name="peak-01b",
        description="Use the regional/KP peak-window detector only to gate downstream shock and residual fusion on top of the kept DECOMP-01E branch.",
        expected_outputs=(
            "peak_window_gate_summary.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "PEAK-01C-region-only-window-detector": ExperimentDefinition(
        experiment_id="PEAK-01C-region-only-window-detector",
        cli_name="peak-01c",
        description="Detect regional downstream peak windows from quarter-level mesoscopic fields without KP conditioning.",
        expected_outputs=(
            "peak_window_detector.json",
            "peak_window_heatmap.png",
        ),
    ),
    "PEAK-01D-region-only-gated-fused-forecast": ExperimentDefinition(
        experiment_id="PEAK-01D-region-only-gated-fused-forecast",
        cli_name="peak-01d",
        description="Use the region-only peak-window detector to gate downstream shock and residual fusion on top of the kept DECOMP-01E branch.",
        expected_outputs=(
            "peak_window_gate_summary.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "PEAK-01E-region-plus-kp-modifier-detector": ExperimentDefinition(
        experiment_id="PEAK-01E-region-plus-kp-modifier-detector",
        cli_name="peak-01e",
        description="Reintroduce KP only as a modifier on top of supported region-only peak-window detector pairs.",
        expected_outputs=(
            "peak_window_detector.json",
            "peak_window_heatmap.png",
        ),
    ),
    "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast": ExperimentDefinition(
        experiment_id="PEAK-01F-region-plus-kp-modifier-gated-fused-forecast",
        cli_name="peak-01f",
        description="Use the region-first plus KP-modifier detector to gate downstream shock and residual fusion and compare it against the kept PEAK-01D branch.",
        expected_outputs=(
            "peak_window_gate_summary.json",
            "mechanistic_forecast.json",
            "forecast_vs_baselines.png",
        ),
    ),
    "TR-V2-00-age01b-baseline-lock": ExperimentDefinition(
        experiment_id="TR-V2-00-age01b-baseline-lock",
        cli_name="tr-v2-00",
        description="Lock the transition frontier to the frozen AGE-01B baseline artifact for exact reproducibility.",
        expected_outputs=("baseline_lock.json", "baseline_comparison.json", "evaluation.json", "mechanistic_forecast.json"),
    ),
    "TR-V2-01-phase2-direct-hazard-priors": ExperimentDefinition(
        experiment_id="TR-V2-01-phase2-direct-hazard-priors",
        cli_name="tr-v2-01",
        description="Use direct Phase 2 temporal edges as structured hazard priors on top of the locked AGE-01B baseline.",
        expected_outputs=(
            "baseline_lock.json",
            "phase2_direct_hazard_prior_summary.json",
            "baseline_comparison.json",
            "evaluation.json",
            "mechanistic_forecast.json",
        ),
    ),
    "TR-V2-02-phase2-hidden-shock-hazards": ExperimentDefinition(
        experiment_id="TR-V2-02-phase2-hidden-shock-hazards",
        cli_name="tr-v2-02",
        description="Add separate hidden Phase 2 shock channels on top of direct hazard priors and the locked AGE-01B baseline.",
        expected_outputs=(
            "baseline_lock.json",
            "phase2_direct_hazard_prior_summary.json",
            "phase2_hidden_shock_summary.json",
            "baseline_comparison.json",
            "evaluation.json",
            "mechanistic_forecast.json",
        ),
    ),
    "TR-V2-03-phase2-ablation-suite": ExperimentDefinition(
        experiment_id="TR-V2-03-phase2-ablation-suite",
        cli_name="tr-v2-03",
        description="Ablate direct priors, hidden shocks, and peak gating on top of the locked AGE-01B baseline.",
        expected_outputs=(
            "baseline_lock.json",
            "phase2_direct_hazard_prior_summary.json",
            "phase2_hidden_shock_summary.json",
            "transition_frontier_ablation_summary.json",
            "baseline_comparison.json",
            "evaluation.json",
            "mechanistic_forecast.json",
        ),
    ),
    "PHASE3-V2-INT-explicit-incidence-autoresearch": ExperimentDefinition(
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        cli_name="phase3-v2-int",
        description="Run the integrated explicit-incidence Phase 3 autoresearch loop: Stage 1 mechanistic calibration followed by Stage 2 bounded target-seeking toward 95-95-95.",
        expected_outputs=(
            "epoch_spec.json",
            "data_snapshot_manifest.json",
            "evaluation_contract.json",
            "baseline_comparison.json",
            "baseline_comparison.png",
            "champion_table.json",
            "challenger_log.json",
            "rejected_mutations.json",
            "stage1_calibration_frontier.json",
            "stage1_calibration_frontier.png",
            "stage1_fit_vs_observed.png",
            "stage2_target_frontier.json",
            "stage2_target_frontier.png",
            "stage2_cascade_projection.png",
            "promotion_audit.json",
            "mechanistic_spec.json",
            "evaluation.json",
            "state_trajectory_rows.json",
            "transition_hazard_summary.json",
            "observation_process_summary.json",
            "cascade_target_projection.json",
            "mechanistic_forecast.json",
        ),
    ),
}

CLI_TO_EXPERIMENT_ID: dict[str, str] = {
    definition.cli_name: definition.experiment_id for definition in EXPERIMENT_REGISTRY.values()
}


def get_experiment_definition(experiment_id: str) -> ExperimentDefinition:
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise KeyError(f"unknown transition research experiment: {experiment_id}")
    return EXPERIMENT_REGISTRY[experiment_id]


def get_experiment_id_for_cli(cli_name: str) -> str:
    if cli_name not in CLI_TO_EXPERIMENT_ID:
        raise KeyError(f"unknown transition research CLI experiment: {cli_name}")
    return CLI_TO_EXPERIMENT_ID[cli_name]


def list_cli_names() -> list[str]:
    return sorted(CLI_TO_EXPERIMENT_ID)


def make_transition_run_id(*, experiment_id: str, seed: int = 0, today: date | None = None) -> str:
    stamp = (today or date.today()).strftime("%Y%m%d")
    return f"tr-{stamp}-s{seed:02d}-{experiment_id}"
