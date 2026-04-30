from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_phase2_two_block_kernel_batch as batch


def test_active_scenario_summary_marks_nonzero_kernel_as_keep() -> None:
    payload = {
        "terminal_delta_rows": [
            {
                "contract": "exact_only",
                "scenario": "disruption_recovery",
                "diagnosed_plhiv_delta": 12.0,
                "alive_on_art_delta": 0.0,
                "new_diagnosed_cases_period_delta": -4.0,
            },
            {
                "contract": "purged_dense",
                "scenario": "mobility_spike",
                "diagnosed_plhiv_delta": 0.0,
                "alive_on_art_delta": 0.0,
                "new_diagnosed_cases_period_delta": 0.0,
            },
        ]
    }

    summary = batch._active_scenario_summary(payload)

    assert summary["decision"] == "keep"
    assert summary["nonnull_terminal_count"] == 2
    assert summary["max_abs_terminal_delta"] == 12.0
