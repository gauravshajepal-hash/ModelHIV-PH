from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_phase2_testing_indicator_ablation_batch as batch


def test_score_row_extracts_gate_fields() -> None:
    payload = {
        "overall_decision": "revert_to_two_block_kernel",
        "support_gate": {"decision": "keep", "block_retained": True, "indicator_count": 2, "max_indicator_share": 0.6},
        "edge_gate": {"decision": "revert", "edge_row": {"target": "mobility_exposure_pressure", "mean_score": 0.2, "survive_top2_count": 1}},
        "circularity_gate": {"decision": "revert", "sign_agreement_rate": 0.625, "mean_abs_ratio": 1.564},
        "baseline_preservation_gate": {"decision": "revert", "sign_agreement_rate": 0.833, "mean_abs_delta_of_delta": 59.39},
        "testing_scenario_gate": {"decision": "keep", "nonnull_terminal_count": 10, "max_abs_terminal_delta": 999.616},
        "archive_alignment_gate": {"decision": "keep"},
    }

    row = batch._score_row("full_block", payload)

    assert row["variant"] == "full_block"
    assert row["edge_target"] == "mobility_exposure_pressure"
    assert row["testing_max_abs_delta"] == 999.616


def test_markdown_report_lists_variants() -> None:
    payload = {
        "generated_at": "2026-04-19T00:00:00+00:00",
        "baseline_two_block_run_id": "two-block",
        "base_monthly_run_id": "monthly",
        "summary_rows": [
            {
                "variant": "full_block",
                "overall_decision": "revert_to_two_block_kernel",
                "support_decision": "keep",
                "edge_target": "mobility_exposure_pressure",
                "edge_decision": "revert",
                "circularity_decision": "revert",
                "baseline_decision": "revert",
                "testing_decision": "keep",
                "testing_max_abs_delta": 12.5,
            }
        ],
        "artifacts": {
            "gate_matrix": "gate.png",
            "terminal_delta_bars": "bars.png",
        },
    }

    text = batch._markdown_report(payload)

    assert "full_block" in text
    assert "mobility_exposure_pressure" in text
