from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_grasp_falsification_batch as batch


def test_plateau_runs_detects_flat_segment() -> None:
    payload = batch._plateau_runs_from_values(
        ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"],
        [100.0, 100.0, 100.0, 120.0],
    )

    assert payload["run_count"] >= 1
    assert payload["max_run_length"] >= 3


def test_shock_gate_stops_when_null_or_calibration_fail() -> None:
    gate = batch._shock_gate_decision(
        check01_exact={"decision": {"shared_burst_quarter_count": 2, "diagnosed_only_share": 0.0}},
        check01_dense={"decision": {"shared_burst_quarter_count": 3, "diagnosed_only_share": 0.0}},
        check02_exact={"decision": {"status": "plateau_signal_nontrivial"}},
        check02_dense={"decision": {"status": "plateau_signal_nontrivial"}},
        null_exact={"quarterly_summary": {"candidate_mean_mae": 0.070}},
        winner_exact={"quarterly_summary": {"candidate_mean_mae": 0.069}},
        null_dense={"quarterly_summary": {"candidate_mean_mae": 0.120}},
        winner_dense={"quarterly_summary": {"candidate_mean_mae": 0.085}},
        cal02={"decision": {"status": "calibration_repair_not_yet_sufficient"}},
    )

    assert gate["status"] == "stop_before_shock_overlay"
    assert gate["null_blocks"] is True
    assert gate["calibration_ok"] is False


def test_markdown_report_mentions_falsification_sections() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "harp-archive-wdi-standard-20260412-s19",
        "variant": "evidence-to-model-loop",
        "recommendation": {
            "exact_candidate_id": "EXP-R10-M1-F1",
            "dense_candidate_id": "EXP-R10-DENSE-M1-H1",
            "grasp_gate_status": "stop_before_shock_overlay",
        },
        "artifacts": {
            "check_01_markdown_exact": "exp_g_check_01_exact.md",
            "check_01_markdown_dense": "exp_g_check_01_dense.md",
            "check_02_markdown": "exp_g_check_02_plateau_census.md",
            "null_markdown": "exp_g_null_01_piecewise_baseline.md",
            "calibration_markdown": "exp_cal_02_dense_diagnosed_recalibration.md",
            "shock_markdown": "exp_r10_g1_01_min.md",
        },
    }

    text = batch._markdown_report(payload)

    assert "TR-V3 GRASP Falsification Batch" in text
    assert "exp_g_check_01_exact.md" in text
    assert "exp_g_check_02_plateau_census.md" in text
    assert "exp_g_null_01_piecewise_baseline.md" in text
    assert "exp_cal_02_dense_diagnosed_recalibration.md" in text
