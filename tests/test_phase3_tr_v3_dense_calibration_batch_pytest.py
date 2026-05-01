from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_dense_calibration_batch as batch


def test_markdown_report_mentions_new_dense_candidate() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "archive-sample",
        "rows": [
            {
                "contract": "purged_dense",
                "experiment_id": "EXP-R10-DENSE-M1-C1-H1",
                "quarterly_mean_mae": 0.0851,
                "quarterly_baseline_mae": 0.1440,
                "diagnosed_raw_mae": 4000.0,
                "art_raw_mae": 1500.0,
                "flow_raw_mae": 590.0,
                "suppression_honesty_flags": {"candidate": "unsupported_or_unclaimed"},
            }
        ],
        "lockbox": {
            "purged_dense": {"winner": {"experiment_id": "EXP-R10-DENSE-M1-C1-H1", "quarterly_mean_mae": 0.0637}}
        },
        "decision": {
            "winner_id": "EXP-R10-DENSE-M1-C1-H1",
            "status": "promote",
            "why": "Crossfit diagnosed calibration improved the primary purged-dense MAE.",
        },
    }
    markdown = batch._markdown_report(payload)
    assert "EXP-R10-DENSE-M1-C1-H1" in markdown
    assert "promote" in markdown


def test_result_row_reads_endpoint_metrics_from_quarterly_summary() -> None:
    row = batch._result_row(
        {
            "experiment_id": "EXP-R10-DENSE-M1-C1-H1",
            "quarterly_summary": {
                "candidate_mean_mae": 0.081,
                "carry_forward_mean_mae": 0.144,
                "endpoint_audit_summary": {
                    "candidate": {
                        "by_metric": {
                            "diagnosed_plhiv": {"raw_mae": 4100.0},
                            "alive_on_art": {"raw_mae": 1500.0},
                            "new_diagnosed_cases_period": {"raw_mae": 580.0},
                        }
                    },
                    "suppression_honesty_flags": {"unsupported_or_unclaimed": 5},
                },
            },
        },
        contract_name="purged_dense",
    )
    assert row["diagnosed_raw_mae"] == 4100.0
    assert row["art_raw_mae"] == 1500.0
    assert row["flow_raw_mae"] == 580.0
