from __future__ import annotations

from pathlib import Path

from epigraph_ph.cli.main import build_parser
from epigraph_ph.phase3.incidence.artifacts import build_incidence_research_context
from epigraph_ph.phase3.incidence.audits import run_inc_00a, run_inc_00b, run_inc_00c
from epigraph_ph.phase3.incidence.modeling import run_inc_01b, run_inc_01d
from epigraph_ph.phase3._lineage.shocks import run_shock_00a


SOURCE_RUN_ID = (
    "audit-phase0-reuse-s00-20260402-age-composite"
    if Path("D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260402-age-composite").exists()
    else "audit-phase0-reuse-s00-20260331"
)


def test_incidence_research_cli_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase3",
            "incidence-research",
            "inc-00a",
            "--run-id",
            "inc-parse-test",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert args.command == "phase3"
    assert args.phase3_command == "incidence-research"
    assert args.phase3_incidence_research_command == "inc-00a"

    args_01d = parser.parse_args(
        [
            "phase3",
            "incidence-research",
            "inc-01d",
            "--run-id",
            "inc-parse-01d",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert args_01d.phase3_incidence_research_command == "inc-01d"

    args_01b = parser.parse_args(
        [
            "phase3",
            "incidence-research",
            "inc-01b",
            "--run-id",
            "inc-parse-01b",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert args_01b.phase3_incidence_research_command == "inc-01b"

    args_shock = parser.parse_args(
        [
            "phase3",
            "incidence-research",
            "shock-00a",
            "--run-id",
            "inc-parse-shock",
            "--source-run-id",
            SOURCE_RUN_ID,
        ]
    )
    assert args_shock.phase3_incidence_research_command == "shock-00a"


def test_inc_00a_evidence_audit_live_run() -> None:
    ctx = build_incidence_research_context(
        run_id="inc-pytest-00a",
        plugin_id="hiv",
        experiment_id="INC-00A-incidence-evidence-audit",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_inc_00a(ctx)
    assert result["decision"]["passed"] is True
    assert "recommended_starting_branch" in result["decision"]
    assert (ctx.experiment_dir / "incidence_evidence_audit.json").exists()
    assert (ctx.experiment_dir / "incidence_support_heatmap.png").exists()


def test_inc_00b_and_inc_00c_live_runs() -> None:
    ctx_b = build_incidence_research_context(
        run_id="inc-pytest-00b",
        plugin_id="hiv",
        experiment_id="INC-00B-population-denominator-audit",
        source_run_id=SOURCE_RUN_ID,
    )
    result_b = run_inc_00b(ctx_b)
    assert result_b["decision"]["passed"] is True
    assert (ctx_b.experiment_dir / "population_denominator_audit.json").exists()
    assert (ctx_b.experiment_dir / "population_alignment_summary.json").exists()

    ctx_c = build_incidence_research_context(
        run_id="inc-pytest-00c",
        plugin_id="hiv",
        experiment_id="INC-00C-incidence-identifiability-audit",
        source_run_id=SOURCE_RUN_ID,
    )
    result_c = run_inc_00c(ctx_c)
    assert result_c["decision"]["passed"] is True
    assert result_c["decision"]["identifiability_class"] in {
        "weakly_identified_requires_locked_diagnosis",
        "direct_inflow_training_support_present",
    }
    assert (ctx_c.experiment_dir / "incidence_identifiability_audit.json").exists()
    assert (ctx_c.experiment_dir / "identifiability_stress_table.json").exists()


def test_inc_01d_and_inc_01b_live_runs() -> None:
    ctx_d = build_incidence_research_context(
        run_id="inc-pytest-01d",
        plugin_id="hiv",
        experiment_id="INC-01D-diagnosis-locked-incidence-branch",
        source_run_id=SOURCE_RUN_ID,
    )
    result_d = run_inc_01d(ctx_d)
    assert result_d["decision"]["completed"] is True
    assert (ctx_d.experiment_dir / "incidence_flow_summary.json").exists()
    assert (ctx_d.experiment_dir / "diagnosis_locked_incidence_summary.json").exists()
    assert (ctx_d.experiment_dir / "state_estimates.npz").exists()
    assert (ctx_d.experiment_dir / "forecast_states.npz").exists()

    ctx_b = build_incidence_research_context(
        run_id="inc-pytest-01b",
        plugin_id="hiv",
        experiment_id="INC-01B-backlog-vs-incidence-swap-stress-test",
        source_run_id=SOURCE_RUN_ID,
    )
    result_b = run_inc_01b(ctx_b)
    assert result_b["decision"]["completed"] is True
    assert (ctx_b.experiment_dir / "swap_stress_summary.json").exists()
    assert (ctx_b.experiment_dir / "identifiability_margin.json").exists()


def test_shock_00a_live_run() -> None:
    ctx = build_incidence_research_context(
        run_id="inc-pytest-shock-00a",
        plugin_id="hiv",
        experiment_id="SHOCK-00A-covid-shock-subparameter-audit",
        source_run_id=SOURCE_RUN_ID,
    )
    result = run_shock_00a(ctx)
    assert result["decision"]["passed"] is True
    assert "recommended_next_branch" in result["decision"]
    assert (ctx.experiment_dir / "shock_subparameter_audit.json").exists()
    assert (ctx.experiment_dir / "shock_signal_coverage_summary.json").exists()
    assert (ctx.experiment_dir / "shock_factor_family_heatmap.png").exists()
