from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as batch


def test_coverage_summary_emits_metric_rows() -> None:
    quarterly_rows = [
        {
            "holdout_target_rows": [
                {
                    "quarter": "2024-Q1",
                    "diagnosed_plhiv": 100.0,
                    "alive_on_art": 60.0,
                    "new_diagnosed_cases_period": 10.0,
                    "metric_tiers": {
                        "diagnosed_plhiv": "exact_observed",
                        "alive_on_art": "exact_observed",
                        "new_diagnosed_cases_period": "exact_observed",
                    },
                }
            ],
            "candidate_prediction_rows": [
                {
                    "quarter": "2024-Q1",
                    "diagnosed_plhiv": 103.0,
                    "alive_on_art": 62.0,
                    "new_diagnosed_cases_period": 9.0,
                }
            ],
        }
        for _ in range(6)
    ]

    payload = batch._coverage_summary(quarterly_rows, allowed_tiers={"exact_observed"})

    assert payload["levels"] == [0.5, 0.8, 0.95]
    assert "diagnosed_plhiv" in payload["by_metric"]
    assert "50" in payload["by_metric"]["diagnosed_plhiv"]["overall"]


def test_markdown_report_mentions_new_eval_sections() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "harp-archive-wdi-standard-20260412-s19",
        "variant": "evidence-to-model-loop",
        "recommendation": {
            "exact_candidate_id": "EXP-R10-M1-F1",
            "dense_candidate_id": "EXP-R10-DENSE-M1-H1",
            "dense_transfer_status": "revert_transfer",
        },
        "artifacts": {
            "eval_01_markdown": "exp_eval_01_frozen_protocol_rebuild.md",
            "eval_02_markdown": "exp_eval_02_endpoint_tier_era_audit.md",
            "eval_03_markdown": "exp_eval_03_calibration_interval_coverage.md",
            "dense_transfer_markdown": "exp_r10_dense_m1_f1_h1.md",
        },
    }

    text = batch._markdown_report(payload)

    assert "TR-V3 Eval Hardening Batch" in text
    assert "exp_eval_01_frozen_protocol_rebuild.md" in text
    assert "exp_eval_02_endpoint_tier_era_audit.md" in text
    assert "exp_eval_03_calibration_interval_coverage.md" in text
    assert "exp_r10_dense_m1_f1_h1.md" in text
