from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from epigraph_ph.cli.main import build_parser
from epigraph_ph.phase3.frontier.aggregation_diagnostic import run_an03a
from epigraph_ph.phase3.frontier.age_research import run_age_01a, run_age_01b, run_age_01c
from epigraph_ph.phase3.frontier.analytics import compute_an01a_outputs
from epigraph_ph.phase3.frontier.cli import run_phase3_transition_report
from epigraph_ph.phase3.frontier.denoising_diagnostic import run_an03b
from epigraph_ph.phase3.frontier.artifacts import build_transition_research_context
from epigraph_ph.phase3.frontier.decomposition import run_decomp_01a, run_decomp_01b, run_decomp_01c, run_decomp_01d, run_decomp_01e, run_decomp_01f
from epigraph_ph.phase3.frontier.hierarchical_autoresearch import _hmba03_auxiliary_gate, run_hmba_00
from epigraph_ph.phase3.frontier.integrated_autoresearch import (
    CandidateConfig,
    _build_blocked_time_contract as _build_integrated_blocked_time_contract,
    _branch_candidate_configs,
    _build_candidate_configs,
    _build_snapshot,
    _serialize_year_metrics,
)
from epigraph_ph.phase3.frontier.strict_diagnosis_kernel_research import _build_blocked_time_contract, _build_candidates, _budget_multipliers
from epigraph_ph.phase3.frontier.analytics import _factor_transition_rows, run_an02b, run_an02c
from epigraph_ph.phase3.frontier.peak_windows import run_peak_01a, run_peak_01b, run_peak_01c, run_peak_01d, run_peak_01e, run_peak_01f
from epigraph_ph.phase3.frontier.sources import load_transition_research_inputs
from epigraph_ph.phase3.frontier.transition_engine import run_kp_01a, run_mech_01a, run_mech_01c, run_mech_01d, run_mech_01e


SOURCE_RUN_ID = "audit-phase0-reuse-s00-20260331"
AGE_SOURCE_RUN_ID = (
    "audit-phase0-reuse-s00-20260402-age-composite"
    if Path("D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260402-age-composite").exists()
    else SOURCE_RUN_ID
)


def test_transition_research_cli_and_phase1_5_alias_parse() -> None:
    parser = build_parser()

    phase15_args = parser.parse_args(["phase1_5", "build", "--run-id", "alias-test"])
    assert phase15_args.command == "phase1_5"
    assert phase15_args.phase1_5_command == "build"

    transition_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "an-01a",
            "--run-id",
            "tr-parse-test",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert transition_args.command == "phase3"
    assert transition_args.phase3_command == "transition-research"
    assert transition_args.phase3_transition_research_command == "an-01a"

    tr_v2_baseline_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "tr-v2-00",
            "--run-id",
            "tr-v2-parse-baseline",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert tr_v2_baseline_args.phase3_transition_research_command == "tr-v2-00"

    tr_v2_direct_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "tr-v2-01",
            "--run-id",
            "tr-v2-parse-direct",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert tr_v2_direct_args.phase3_transition_research_command == "tr-v2-01"

    tr_v2_hidden_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "tr-v2-02",
            "--run-id",
            "tr-v2-parse-hidden",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert tr_v2_hidden_args.phase3_transition_research_command == "tr-v2-02"

    tr_v2_ablation_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "tr-v2-03",
            "--run-id",
            "tr-v2-parse-ablation",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert tr_v2_ablation_args.phase3_transition_research_command == "tr-v2-03"

    rolling_report_args = parser.parse_args(
        [
            "phase3",
            "transition-report",
            "rolling-origin",
            "--run-id",
            "tr-report-rolling",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert rolling_report_args.phase3_transition_report_command == "rolling-origin"

    early_partial_args = parser.parse_args(
        [
            "phase3",
            "transition-report",
            "early-history-partial",
            "--run-id",
            "tr-report-early",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert early_partial_args.phase3_transition_report_command == "early-history-partial"

    dashboard_args = parser.parse_args(
        [
            "phase3",
            "transition-report",
            "benchmark-dashboard",
            "--run-id",
            "tr-report-dashboard",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert dashboard_args.phase3_transition_report_command == "benchmark-dashboard"

    phase3_v2_int_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "phase3-v2-int",
            "--run-id",
            "phase3-v2-int-parse",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert phase3_v2_int_args.phase3_transition_research_command == "phase3-v2-int"

    an03a_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "an-03a",
            "--run-id",
            "tr-parse-an03a",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert an03a_args.phase3_transition_research_command == "an-03a"

    an03b_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "an-03b",
            "--run-id",
            "tr-parse-an03b",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert an03b_args.phase3_transition_research_command == "an-03b"

    an03c_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "an-03c",
            "--run-id",
            "tr-parse-an03c",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert an03c_args.phase3_transition_research_command == "an-03c"

    diag01a_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "diag-01a",
            "--run-id",
            "tr-parse-diag01a",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert diag01a_args.phase3_transition_research_command == "diag-01a"

    diag01b_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "diag-01b",
            "--run-id",
            "tr-parse-diag01b",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert diag01b_args.phase3_transition_research_command == "diag-01b"

    diag02a_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "diag-02a",
            "--run-id",
            "tr-parse-diag02a",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert diag02a_args.phase3_transition_research_command == "diag-02a"

    diag02b_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "diag-02b",
            "--run-id",
            "tr-parse-diag02b",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert diag02b_args.phase3_transition_research_command == "diag-02b"

    diag02c_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "diag-02c",
            "--run-id",
            "tr-parse-diag02c",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert diag02c_args.phase3_transition_research_command == "diag-02c"

    hmba00_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "hmba-00",
            "--run-id",
            "tr-parse-hmba00",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert hmba00_args.phase3_transition_research_command == "hmba-00"

    hmba01_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "hmba-01",
            "--run-id",
            "tr-parse-hmba01",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert hmba01_args.phase3_transition_research_command == "hmba-01"

    hmba02_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "hmba-02",
            "--run-id",
            "tr-parse-hmba02",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert hmba02_args.phase3_transition_research_command == "hmba-02"

    hmba03_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "hmba-03",
            "--run-id",
            "tr-parse-hmba03",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert hmba03_args.phase3_transition_research_command == "hmba-03"

    hmba03a_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "hmba-03a",
            "--run-id",
            "tr-parse-hmba03a",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert hmba03a_args.phase3_transition_research_command == "hmba-03a"

    mech_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "mech-01a",
            "--run-id",
            "tr-parse-mech",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert mech_args.phase3_transition_research_command == "mech-01a"

    mech_helper_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "mech-01b",
            "--run-id",
            "tr-parse-mech-helper",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert mech_helper_args.phase3_transition_research_command == "mech-01b"

    mech_residual_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "mech-01c",
            "--run-id",
            "tr-parse-mech-residual",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert mech_residual_args.phase3_transition_research_command == "mech-01c"

    mech_diagnosis_locked_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "mech-01d",
            "--run-id",
            "tr-parse-mech-diagnosis-locked",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert mech_diagnosis_locked_args.phase3_transition_research_command == "mech-01d"

    mech_anchored_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "mech-01e",
            "--run-id",
            "tr-parse-mech-anchored",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert mech_anchored_args.phase3_transition_research_command == "mech-01e"

    kp_overlay_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "kp-01a",
            "--run-id",
            "tr-parse-kp-overlay",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert kp_overlay_args.phase3_transition_research_command == "kp-01a"

    decomp_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01a",
            "--run-id",
            "tr-parse-decomp",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_args.phase3_transition_research_command == "decomp-01a"

    decomp_driver_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01b",
            "--run-id",
            "tr-parse-decomp-driver",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_driver_args.phase3_transition_research_command == "decomp-01b"

    decomp_fused_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01c",
            "--run-id",
            "tr-parse-decomp-fused",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_fused_args.phase3_transition_research_command == "decomp-01c"

    decomp_skill_gated_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01d",
            "--run-id",
            "tr-parse-decomp-skill-gated",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_skill_gated_args.phase3_transition_research_command == "decomp-01d"

    decomp_loo_gated_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01e",
            "--run-id",
            "tr-parse-decomp-loo-gated",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_loo_gated_args.phase3_transition_research_command == "decomp-01e"

    decomp_reverse_grasp_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "decomp-01f",
            "--run-id",
            "tr-parse-decomp-reverse-grasp",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert decomp_reverse_grasp_args.phase3_transition_research_command == "decomp-01f"

    peak_detector_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01a",
            "--run-id",
            "tr-parse-peak-detector",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_detector_args.phase3_transition_research_command == "peak-01a"

    peak_gated_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01b",
            "--run-id",
            "tr-parse-peak-gated",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_gated_args.phase3_transition_research_command == "peak-01b"

    peak_region_only_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01c",
            "--run-id",
            "tr-parse-peak-region-only",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_region_only_args.phase3_transition_research_command == "peak-01c"

    peak_region_gated_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01d",
            "--run-id",
            "tr-parse-peak-region-gated",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_region_gated_args.phase3_transition_research_command == "peak-01d"

    peak_kp_modifier_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01e",
            "--run-id",
            "tr-parse-peak-kp-modifier",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_kp_modifier_args.phase3_transition_research_command == "peak-01e"


def test_phase3_v2_int_snapshot_and_candidate_space() -> None:
    parser = build_parser()
    ctx = build_transition_research_context(
        run_id="phase3-v2-int-snapshot",
        plugin_id="hiv",
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        source_run_id=SOURCE_RUN_ID,
    )
    snapshot = _build_snapshot(ctx)
    candidate_configs = _build_candidate_configs(snapshot)
    branch_candidate_configs = _branch_candidate_configs(snapshot, candidate_configs[0])

    assert snapshot.historical_quarters
    assert snapshot.historical_quarters[0] == "2010-Q1"
    assert snapshot.historical_end_quarter.startswith("2025-")
    assert snapshot.historical_years[0] == 2010
    assert snapshot.historical_years[-1] == 2025
    assert snapshot.target_horizon.endswith("Q4")
    assert snapshot.population_denominator.shape[0] == len(snapshot.model_quarters)
    assert candidate_configs
    assert {config.diagnosis_family for config in candidate_configs} == {"hazard"}
    assert {config.care_family for config in candidate_configs} == {"markov"}
    assert {config.diagnosis_family for config in branch_candidate_configs} == {"hazard", "delay"}
    assert {config.care_family for config in branch_candidate_configs} == {"markov", "semi_markov"}


def test_phase3_v2_int_blocked_time_contract_and_provenance() -> None:
    ctx = build_transition_research_context(
        run_id="phase3-v2-int-publication-contract",
        plugin_id="hiv",
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        source_run_id=SOURCE_RUN_ID,
    )
    snapshot = _build_snapshot(ctx)
    contract = _build_integrated_blocked_time_contract(snapshot)

    assert contract.train_diagnosis_quarters
    assert contract.validation_quarters
    assert contract.holdout_quarters
    assert contract.train_end_quarter == contract.train_diagnosis_quarters[-1]
    assert contract.validation_start_quarter == contract.validation_quarters[0]
    assert contract.validation_end_quarter == contract.validation_quarters[-1]
    assert contract.holdout_start_quarter == contract.holdout_quarters[0]
    assert contract.holdout_end_quarter == contract.holdout_quarters[-1]
    assert contract.train_end_quarter < contract.validation_start_quarter
    assert contract.validation_end_quarter < contract.holdout_start_quarter
    assert contract.holdout_end_quarter == snapshot.evidence_provenance["metrics"]["new_diagnosed_cases_period"]["last_observed_quarter"]
    assert snapshot.evidence_provenance["metrics"]["estimated_plhiv"]["role"] == "auxiliary_latent_population_size"
    assert snapshot.evidence_provenance["metrics"]["estimated_plhiv"]["evidence_tier_counts"]["model_estimated_total"] > 0
    assert "archive_program_observation_metrics" in snapshot.evidence_provenance["summary"]["direct_vs_contextual_split"]
    assert snapshot.phase2_insertion_contract["structured_prior_status"] is False
    assert snapshot.phase2_insertion_contract["province_resolved_hidden_dynamics_status"] is False


def test_phase3_v2_int_yearly_reporting_marks_unobserved_diagnosis_flow() -> None:
    ctx = build_transition_research_context(
        run_id="phase3-v2-int-yearly-contract",
        plugin_id="hiv",
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        source_run_id=SOURCE_RUN_ID,
    )
    snapshot = _build_snapshot(ctx)
    serialized = _serialize_year_metrics(
        snapshot,
        {
            "2010": {"primary_loss": 0.1, "diag_flow_loss": float("inf")},
            "2025": {"primary_loss": 0.2, "diag_flow_loss": 0.3},
        },
    )

    assert serialized["2010"]["diag_flow_loss"] is None
    assert serialized["2010"]["diagnosis_flow_observed"] is False
    assert serialized["2010"]["diagnosis_flow_status"] == "not observed"
    assert serialized["2025"]["diag_flow_loss"] == 0.3
    assert serialized["2025"]["diagnosis_flow_observed"] is True
    assert serialized["2025"]["diagnosis_flow_status"] == "observed"


def test_hmba03_auxiliary_gate_rejects_absolute_spikes() -> None:
    module_summaries = {
        "U_to_D": {"feature_ids": ["f1"], "depth_metrics": {"national": 4.09, "region": 0.73, "province": 0.47}},
        "D_to_A": {"feature_ids": ["f2"], "depth_metrics": {"national": 0.20, "region": 0.15, "province": 0.10}},
    }
    depth_map = {"U_to_D": "national", "D_to_A": "province"}
    gate = _hmba03_auxiliary_gate(module_summaries, depth_map, absolute_ceiling=0.287076)

    assert gate["passes_auxiliary_gate"] is False
    assert gate["max_selected_module_rmse"] > gate["absolute_auxiliary_ceiling"]
    assert any(str(row["module_name"]) == "U_to_D" for row in gate["spike_modules"])


def test_diag_01a_blocked_time_contract_and_candidate_widths() -> None:
    ctx = build_transition_research_context(
        run_id="diag-01a-contract",
        plugin_id="hiv",
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        source_run_id=SOURCE_RUN_ID,
    )
    snapshot = _build_snapshot(ctx)
    contract = _build_blocked_time_contract(snapshot)
    reference_config = CandidateConfig(
        candidate_id="diag-delay-care-markov-h05-obsoff",
        diagnosis_family="delay",
        care_family="markov",
        hidden_rank=5,
        use_observation_covariates=False,
    )
    candidates = _build_candidates(reference_config, contract)

    assert contract.train_diagnosis_quarters
    assert contract.validation_quarters
    assert contract.holdout_quarters
    assert contract.train_end_quarter < contract.validation_start_quarter
    assert contract.validation_end_quarter < contract.holdout_start_quarter
    assert len(contract.train_diagnosis_quarters) + len(contract.validation_quarters) + len(contract.holdout_quarters) == len(
        contract.diagnosis_flow_observed_quarters
    )
    assert {candidate.diagnosis_kind for candidate in candidates} == {"hazard", "delay", "strict"}
    strict_widths = [candidate.kernel_width for candidate in candidates if candidate.diagnosis_kind == "strict"]
    assert strict_widths[0] == 2
    assert strict_widths[-1] == len(contract.train_diagnosis_quarters)


def test_diag_01b_budget_ladder_is_predeclared_and_monotone() -> None:
    ctx = build_transition_research_context(
        run_id="diag-01b-contract",
        plugin_id="hiv",
        experiment_id="PHASE3-V2-INT-explicit-incidence-autoresearch",
        source_run_id=SOURCE_RUN_ID,
    )
    snapshot = _build_snapshot(ctx)
    contract = _build_blocked_time_contract(snapshot)
    multipliers = _budget_multipliers(contract)

    assert multipliers[0] == 1.0
    assert multipliers == sorted(multipliers)
    assert len(set(multipliers)) == len(multipliers)
    assert multipliers[-1] == float(max(2, len(contract.train_diagnosis_quarters)))
    assert all(next_value <= current_value * 2.0 for current_value, next_value in zip(multipliers, multipliers[1:]))


def test_rebuilt_harp_archive_expands_early_national_support() -> None:
    parser = build_parser()
    metric_rows = json.loads(
        (Path("D:/EpiGraph_PH/artifacts/runs") / SOURCE_RUN_ID / "harp_archive" / "historical_metric_rows.json").read_text(encoding="utf-8")
    )
    national_rows = [dict(row) for row in metric_rows if str(row.get("region") or "").lower() == "national"]

    diagnosed_times = {str(row.get("time")) for row in national_rows if str(row.get("metric_name")) == "diagnosed_plhiv"}
    art_rows = [row for row in national_rows if str(row.get("metric_name")) == "alive_on_art"]
    total_times = {str(row.get("time")) for row in national_rows if str(row.get("metric_name")) == "estimated_plhiv"}

    assert "2010-12" in diagnosed_times
    assert any(str(row.get("time")) == "2012-06" and float(row.get("value") or 0.0) == 2761.0 for row in art_rows)
    assert any(str(row.get("time")) == "2012-09" and float(row.get("value") or 0.0) == 3115.0 for row in art_rows)
    assert "2010-01" in total_times

    peak_kp_modifier_gated_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "peak-01f",
            "--run-id",
            "tr-parse-peak-kp-modifier-gated",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert peak_kp_modifier_gated_args.phase3_transition_research_command == "peak-01f"

    age_audit_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "age-01a",
            "--run-id",
            "tr-parse-age-audit",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert age_audit_args.phase3_transition_research_command == "age-01a"

    age_modifier_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "age-01b",
            "--run-id",
            "tr-parse-age-modifier",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert age_modifier_args.phase3_transition_research_command == "age-01b"

    age_downstream_args = parser.parse_args(
        [
            "phase3",
            "transition-research",
            "age-01c",
            "--run-id",
            "tr-parse-age-downstream",
            "--source-run-id",
            AGE_SOURCE_RUN_ID,
        ]
    )
    assert age_downstream_args.phase3_transition_research_command == "age-01c"


def test_an01a_live_source_covers_expected_years_and_factors() -> None:
    ctx = build_transition_research_context(
        run_id="tr-pytest-an01a",
        plugin_id="hiv",
        experiment_id="AN-01A-national-yearly-factor-evolution",
        source_run_id=SOURCE_RUN_ID,
    )
    inputs = load_transition_research_inputs(ctx)
    payload = compute_an01a_outputs(inputs)

    assert payload["years"][0] == "2010"
    assert payload["years"][-1] == "2025"
    assert len(payload["factor_ids"]) >= 8
    assert payload["score_matrix"].shape == (len(payload["factor_ids"]), len(payload["years"]))
    assert any(entry["name"] == "analysis_year_ceiling" and entry["value"] == 2025 for entry in payload["numeric_justification"])


def test_transition_report_dispatch_passes_expected_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs: object) -> dict[str, str]:
        calls.append(dict(kwargs))
        return {"markdown": "ok"}

    monkeypatch.setattr("epigraph_ph.phase3.frontier.cli.REPORT_DISPATCH", {"rolling-origin": fake_report})

    result = run_phase3_transition_report(
        run_id="report-run",
        plugin_id="hiv",
        source_run_id=SOURCE_RUN_ID,
        cli_report_name="rolling-origin",
        start_year=2012,
        end_year=2024,
        min_train_years=4,
        horizon_years=2,
    )

    assert result == {"markdown": "ok"}
    assert calls == [
        {
            "run_id": "report-run",
            "plugin_id": "hiv",
            "source_run_id": SOURCE_RUN_ID,
            "start_year": 2012,
            "end_year": 2024,
            "min_train_years": 4,
            "horizon_years": 2,
        }
    ]


def test_an02_transition_and_kp_maps_reflect_sparse_kp_support() -> None:
    ctx = build_transition_research_context(
        run_id="tr-pytest-an02",
        plugin_id="hiv",
        experiment_id="AN-02B-factor-to-kp-map",
        source_run_id=SOURCE_RUN_ID,
    )
    inputs = load_transition_research_inputs(ctx)
    transition_rows = _factor_transition_rows(inputs)

    factor_count = len(inputs.retained_factor_lookup)
    assert len(transition_rows) == factor_count * 5
    factor_0003_rows = [row for row in transition_rows if row["factor_id"] == "factor_0003"]
    assert any(row["transition"] == "U_to_D" and float(row["relevance"]) > 0.0 for row in factor_0003_rows)

    result = run_an02b(ctx)
    kp_rows = result["rows"]
    assert len(kp_rows) == factor_count * 3
    assert any("current_phase3_subgroup_summary_has_no_explicit_tgw_mass" in row["low_confidence_reasons"] for row in kp_rows)
    factor_0003_kp_rows = [row for row in kp_rows if row["factor_id"] == "factor_0003"]
    assert all(float(row["relevance"]) == 0.0 for row in factor_0003_kp_rows)


def test_an02c_writes_expected_artifacts() -> None:
    run_id = "tr-pytest-an02c"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AN-02C-kp-transition-relevance-drift",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_an02c(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert result["artifacts"]["decision"].endswith("decision.json")
    assert (experiment_dir / "kp_transition_relevance.json").exists()
    assert (experiment_dir / "kp_transition_drift.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()


def test_an03a_writes_aggregation_loss_artifacts() -> None:
    run_id = "tr-pytest-an03a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AN-03A-phase2-aggregation-loss-diagnostic",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_an03a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert result["artifacts"]["decision"].endswith("decision.json")
    assert result["provincial_evidence_run_id"]
    assert (experiment_dir / "aggregation_loss_diagnostic.json").exists()
    assert (experiment_dir / "provincial_evidence_summary.json").exists()
    assert (experiment_dir / "module_bundle_seed_report.json").exists()
    assert (experiment_dir / "aggregation_loss_top_factors.png").exists()
    assert (experiment_dir / "aggregation_loss_module_heatmap.png").exists()
    top_row = result["rows"][0]
    assert "hierarchical_priority_score" in top_row
    assert "transition_priority" in top_row
    incidence_bundles = result["module_bundle_seed_report"]["module_bundle_rankings"]["incidence"]
    assert incidence_bundles


def test_hmba00_writes_contract_and_panel_artifacts() -> None:
    run_id = "tr-pytest-hmba00"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="HMBA-00-hierarchical-contract-freeze",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_hmba_00(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert result["decision"]["completed"] is True
    assert (experiment_dir / "contract_snapshot.json").exists()
    assert (experiment_dir / "module_seed_manifest.json").exists()
    assert (experiment_dir / "evidence_panel_summary.json").exists()
    assert (experiment_dir / "national_observation_values.npz").exists()
    assert (experiment_dir / "national_observation_mask.npz").exists()
    assert (experiment_dir / "provincial_auxiliary_state_shares.npz").exists()
    assert (experiment_dir / "regional_auxiliary_state_shares.npz").exists()
    assert (experiment_dir / "national_auxiliary_state_shares.npz").exists()
    assert (experiment_dir / "coverage_frontier.png").exists()
    assert (experiment_dir / "module_seed_frontier.png").exists()
    assert result["contract_snapshot"]["contract_terms"]["province_truth_available"] is False
    assert result["module_seed_manifest"]["module_bundle_rankings"]["incidence"]
    assert result["evidence_panel_summary"]["provincial_auxiliary_panel"]["province_count"] > 0


def test_an03b_writes_partial_denoising_artifacts() -> None:
    run_id = "tr-pytest-an03b"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AN-03B-phase2-partial-denoising-diagnostic",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_an03b(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert result["decision"]["completed"] is True
    assert (experiment_dir / "partial_denoising_factor_delta.json").exists()
    assert (experiment_dir / "partial_denoising_module_delta.json").exists()
    assert (experiment_dir / "partial_denoising_summary.json").exists()
    assert (experiment_dir / "partial_denoising_module_delta.png").exists()
    assert (experiment_dir / "partial_denoising_rank_compression.png").exists()
    assert (experiment_dir / "partial_denoising_priority_comparison.png").exists()
    assert (experiment_dir / "raw_module_seed_manifest.json").exists()
    assert (experiment_dir / "denoised_module_seed_manifest.json").exists()
    assert result["paper_archive"]["figure_count"] >= 3


def test_age_01a_writes_age_audit_artifacts() -> None:
    run_id = "tr-pytest-age01a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AGE-01A-age-evidence-audit",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_age_01a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "age_evidence_audit.json").exists()
    assert (experiment_dir / "age_signal_coverage_summary.json").exists()


def test_age_01b_writes_modifier_artifacts() -> None:
    run_id = "tr-pytest-age01b"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AGE-01B-youth-diagnosis-modifier",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_age_01b(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "baseline_comparison.json").exists()
    assert (experiment_dir / "evaluation.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "age_diagnosis_modifier_summary.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert "baseline_comparison" in result
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert result["baseline_comparison"]["branch_reference_experiment_id"] == "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast"
    assert result["evaluation"]["mode"] == "age_youth_diagnosis_modifier"


def test_age_01c_writes_downstream_modifier_artifacts() -> None:
    run_id = "tr-pytest-age01c"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="AGE-01C-youth-downstream-modifier",
        source_run_id=AGE_SOURCE_RUN_ID,
    )
    result = run_age_01c(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "baseline_comparison.json").exists()
    assert (experiment_dir / "evaluation.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "age_downstream_modifier_summary.json").exists()
    assert (experiment_dir / "age_transition_effect_heatmap.png").exists()
    assert (experiment_dir / "forecast_vs_age01b.png").exists()
    assert "baseline_comparison" in result
    assert result["baseline_comparison"]["branch_reference_experiment_id"] == "AGE-01B-youth-diagnosis-modifier"
    assert result["evaluation"]["mode"] == "age_youth_downstream_modifier"


def test_mech_01a_writes_state_and_transition_artifacts() -> None:
    run_id = "tr-pytest-mech01a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="MECH-01A-national-udavl-baseline",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_mech_01a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert result["decision"]["keep"] is True
    assert (experiment_dir / "fit_artifact.json").exists()
    assert (experiment_dir / "evaluation.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()

    state_estimates = np.load(experiment_dir / "state_estimates.npz")["values"]
    forecast_states = np.load(experiment_dir / "forecast_states.npz")["values"]
    assert state_estimates.shape[1] == 5
    assert forecast_states.shape[1] == 5


def test_mech_01b_writes_helper_artifacts() -> None:
    run_id = "tr-pytest-mech01b"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="MECH-01B-mesoscopic-transition-helpers",
        source_run_id=SOURCE_RUN_ID,
    )
    from epigraph_ph.phase3.frontier.transition_engine import run_mech_01b

    result = run_mech_01b(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "mesoscopic_transition_helper_summary.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert "mesoscopic_transition_helper_summary" in result
    helper_summary = result["mesoscopic_transition_helper_summary"]
    assert len(helper_summary["transition_model_rows"]) == 5
    assert any(int(row["eligible_factor_count"]) > 0 for row in helper_summary["transition_model_rows"])


def test_mech_01c_writes_residual_helper_artifacts() -> None:
    run_id = "tr-pytest-mech01c"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="MECH-01C-residual-transition-helpers",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_mech_01c(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "residual_transition_helper_summary.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert "residual_transition_helper_summary" in result
    helper_summary = result["residual_transition_helper_summary"]
    assert len(helper_summary["transition_model_rows"]) == 5
    assert any(int(row["eligible_factor_count"]) > 0 for row in helper_summary["transition_model_rows"])


def test_mech_01d_writes_diagnosis_locked_residual_artifacts() -> None:
    run_id = "tr-pytest-mech01d"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="MECH-01D-diagnosis-locked-residual-helpers",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_mech_01d(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "residual_transition_helper_summary.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert "residual_transition_helper_summary" in result
    helper_summary = result["residual_transition_helper_summary"]
    assert helper_summary["locked_transitions"] == ["U_to_D"]
    assert len(helper_summary["transition_model_rows"]) == 5


def test_mech_01e_writes_anchored_residual_artifacts() -> None:
    run_id = "tr-pytest-mech01e"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="MECH-01E-anchored-downstream-residual-helpers",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_mech_01e(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "residual_transition_helper_summary.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    helper_summary = result["residual_transition_helper_summary"]
    assert helper_summary["locked_transitions"] == ["U_to_D"]
    assert helper_summary["anchored_holdout_quarter"] == "2025-Q1"


def test_kp_01a_writes_overlay_artifacts() -> None:
    run_id = "tr-pytest-kp01a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="KP-01A-national-kp-lite-overlay",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_kp_01a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "kp_overlay_summary.json").exists()
    assert (experiment_dir / "transition_hazard_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    overlay_summary = result["kp_overlay_summary"]
    assert overlay_summary["locked_transitions"] == ["U_to_D"]
    assert overlay_summary["anchored_holdout_quarter"] == "2025-Q1"
    assert overlay_summary["kp_support_strength"] > 0.0


def test_decomp_01a_writes_channel_artifacts() -> None:
    run_id = "tr-pytest-decomp01a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01A-transition-channel-decomposition",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "channel_decomposition.npz").exists()
    assert (experiment_dir / "channel_summary.json").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    decomposition = np.load(experiment_dir / "channel_decomposition.npz")["values"]
    assert decomposition.shape[0] == 5
    assert decomposition.shape[2] == 3


def test_decomp_01b_writes_driver_artifacts() -> None:
    run_id = "tr-pytest-decomp01b"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01B-channel-driver-coupling",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01b(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "channel_driver_weights.json").exists()
    assert (experiment_dir / "channel_driver_heatmap.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    payload = result["payload"]
    assert len(payload["pair_summary_rows"]) == 15
    assert any(bool(row["supported"]) for row in payload["pair_summary_rows"])


def test_decomp_01c_writes_fused_forecast_artifacts() -> None:
    run_id = "tr-pytest-decomp01c"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01C-fused-mechanistic-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01c(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "hazard_reconstruction.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert (experiment_dir / "evaluation.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result


def test_decomp_01d_writes_skill_gated_forecast_artifacts() -> None:
    run_id = "tr-pytest-decomp01d"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01D-skill-gated-fused-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01d(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "hazard_reconstruction.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result


def test_decomp_01e_writes_loo_gated_forecast_artifacts() -> None:
    run_id = "tr-pytest-decomp01e"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01E-loo-gated-fused-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01e(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "hazard_reconstruction.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result


def test_decomp_01f_writes_reverse_grasp_forecast_artifacts() -> None:
    run_id = "tr-pytest-decomp01f"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="DECOMP-01F-reverse-grasp-peak-clusters",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_decomp_01f(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "hazard_reconstruction.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result
    baseline = result["baseline_comparison"]
    assert "decomp_01e_reference_mean_absolute_error" in baseline
    assert "decomp_01e_reference_peak_alive_on_art_absolute_error" in baseline
    assert "peak_target_quarter" in baseline


def test_peak_01a_writes_detector_artifacts() -> None:
    run_id = "tr-pytest-peak01a"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01A-regional-kp-window-detector",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01a(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_detector.json").exists()
    assert (experiment_dir / "peak_window_heatmap.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "pair_rows" in result
    assert len(result["pair_rows"]) > 0


def test_peak_01b_writes_detector_gated_forecast_artifacts() -> None:
    run_id = "tr-pytest-peak01b"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01B-detector-gated-fused-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01b(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_gate_summary.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result


def test_peak_01c_writes_region_only_detector_artifacts() -> None:
    run_id = "tr-pytest-peak01c"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01C-region-only-window-detector",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01c(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_detector.json").exists()
    assert (experiment_dir / "peak_window_heatmap.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert result["coverage_summary"]["detector_mode"] == "region_only"
    assert all(float(row["kp_signal"]) == 0.0 for row in result["quarter_rows"])


def test_peak_01d_writes_region_only_gated_forecast_artifacts() -> None:
    run_id = "tr-pytest-peak01d"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01D-region-only-gated-fused-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01d(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_gate_summary.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result


def test_peak_01e_writes_region_plus_kp_modifier_detector_artifacts() -> None:
    run_id = "tr-pytest-peak01e"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01E-region-plus-kp-modifier-detector",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01e(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_detector.json").exists()
    assert (experiment_dir / "peak_window_heatmap.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert result["coverage_summary"]["detector_mode"] == "region_plus_kp_modifier"
    assert "pair_rows" in result


def test_peak_01f_writes_region_plus_kp_modifier_gated_forecast_artifacts() -> None:
    run_id = "tr-pytest-peak01f"
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id="hiv",
        experiment_id="PEAK-01F-region-plus-kp-modifier-gated-fused-forecast",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_peak_01f(ctx)
    experiment_dir = Path(ctx.experiment_dir)

    assert (experiment_dir / "peak_window_gate_summary.json").exists()
    assert (experiment_dir / "mechanistic_forecast.json").exists()
    assert (experiment_dir / "forecast_vs_baselines.png").exists()
    assert (experiment_dir / "numeric_justification.json").exists()
    assert result["decision"]["completed"] is True
    assert "baseline_comparison" in result
