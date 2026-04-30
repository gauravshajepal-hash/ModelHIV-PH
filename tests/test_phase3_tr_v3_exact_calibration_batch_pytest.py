from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_exact_calibration_batch as batch


def test_exact_batch_markdown_mentions_crossfit_candidate() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "archive-sample",
        "rows": [
            {
                "experiment_id": "EXP-R10-M1-F1-C1",
                "quarterly_mean_mae": 0.068,
                "quarterly_baseline_mae": 0.228,
                "diagnosed_raw_mae": 2100.0,
                "art_raw_mae": 2700.0,
                "flow_raw_mae": 790.0,
                "suppression_honesty_flags": {"scored_direct_support": 5},
            }
        ],
        "lockbox": {"winner": {"experiment_id": "EXP-R10-M1-F1-C1", "quarterly_mean_mae": 0.044}},
        "decision": {"winner_id": "EXP-R10-M1-F1-C1", "status": "promote", "why": "Improved exact MAE."},
    }
    markdown = batch._markdown_report(payload)
    assert "EXP-R10-M1-F1-C1" in markdown
    assert "promote" in markdown
