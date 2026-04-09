from __future__ import annotations

from pathlib import Path

import numpy as np

from epigraph_ph.cli.main import build_parser
from epigraph_ph.phase3.frontier.age_research import run_age_01a, run_age_01b, run_age_01c
from epigraph_ph.phase3.frontier.analytics import compute_an01a_outputs
from epigraph_ph.phase3.frontier.artifacts import build_transition_research_context
from epigraph_ph.phase3.frontier.decomposition import run_decomp_01a, run_decomp_01b, run_decomp_01c, run_decomp_01d, run_decomp_01e, run_decomp_01f
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
