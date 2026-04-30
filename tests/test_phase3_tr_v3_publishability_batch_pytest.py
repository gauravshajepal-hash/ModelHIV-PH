from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_publishability_batch as batch


def test_lockbox_contract_choices_and_scoring_tiers_are_stable() -> None:
    assert batch.LOCKBOX_CONTRACT_CHOICES == ("exact_only", "legacy_dense", "purged_dense")
    assert batch._scoring_tiers("exact_only") == {"exact_observed"}
    assert batch._scoring_tiers("legacy_dense") == {"exact_observed", "bridge_observed"}
    assert batch._scoring_tiers("purged_dense") == {"exact_observed", "bridge_observed"}


def test_markdown_report_renders_core_sections() -> None:
    payload = {
        "generated_at": "2026-04-13T00:00:00+00:00",
        "archive_run_id": "harp-archive-wdi-standard-20260412-s19",
        "variant": "evidence-to-model-loop",
        "lockbox_note": "retroactive",
        "frozen_contract_audit": {
            "rows": [
                {
                    "contract": "exact_only",
                    "experiment_id": "EXP-R10-M1",
                    "quarterly_mean_mae": 0.07,
                    "quarterly_baseline_mae": 0.22,
                    "diagnosed_raw_mae": 1000.0,
                    "art_raw_mae": 900.0,
                    "flow_raw_mae": 300.0,
                    "suppression_honesty_flags": {"scored_direct_support": 5},
                }
            ]
        },
        "lockbox": {
            "holdout_years": [2025],
            "contracts": {
                "exact_only": {
                    "graph_file": "lockbox_exact_only.png",
                    "rows": [
                        {
                            "experiment_id": "EXP-R10-M1",
                            "quarterly_mean_mae": 0.08,
                            "quarterly_baseline_mae": 0.2,
                            "annual_overlay_error": 0.03,
                            "suppression_honesty_flag": "scored_direct_support",
                        }
                    ],
                }
            },
        },
        "calibration": {
            "EXP-R10-M1": {
                "graph_file": "EXP-R10-M1_calibration.png",
                "quarterly_mean_mae": 0.07,
                "quarterly_baseline_mean_mae": 0.22,
                "by_metric": {
                    "diagnosed_plhiv": {
                        "count": 4,
                        "mean_residual": -10.0,
                        "median_residual": -8.0,
                        "std_residual": 5.0,
                        "q10_residual": -15.0,
                        "q90_residual": -2.0,
                    }
                },
            }
        },
        "support_exploration": {
            "susceptible_contract": {
                "decision": {"status": "annual_sidecar_only", "why": "annual only"},
                "annual_years": {"joint_support_years": [2010, 2011]},
            },
            "richer_leakage_contract": {
                "decision": {"status": "defer_richer_leakage", "why": "still weak"},
                "annual_aids_deaths_years": [2010, 2011],
            },
        },
    }

    text = batch._markdown_report(payload)

    assert "TR-V3 Publishability Batch" in text
    assert "Frozen Contract Audit" in text
    assert "Lockbox" in text
    assert "Calibration And Uncertainty" in text
    assert "Support Exploration" in text
    assert "Explicit S(t)" in text
    assert "Richer Leakage" in text
