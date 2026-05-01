from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_probabilistic_batch as batch


def test_interval_decision_keeps_bootstrap_when_it_wins_cleanly() -> None:
    decision = batch._interval_decision(
        {
            "primary_normalized_wis": 0.12,
            "coverage_gap": 0.08,
        },
        {
            "primary_normalized_wis": 0.10,
            "coverage_gap": 0.07,
            "lockbox": {"summary": {"primary_normalized_wis": 0.11}},
            "lockbox_baselines": {"conformal_primary_normalized_wis": 0.13},
        },
    )
    assert decision["winner"] == "EXP-UQ-02"
    assert decision["status"] == "keep_bootstrap"


def test_markdown_report_mentions_uq_and_risk_sections() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "archive-sample",
        "exact": {
            "experiment_id": "EXP-R10-M1-F1-C1",
            "contract_name": "exact_only",
            "conformal": {
                "summary": {"80": {"mean_normalized_wis": 0.1, "coverage": 0.8}, "95": {"mean_normalized_wis": 0.2, "coverage": 0.94}},
                "primary_normalized_wis": 0.15,
                "coverage_gap": 0.01,
                "lockbox": {"summary": {"primary_normalized_wis": 0.14}},
            },
            "bootstrap": {
                "summary": {"80": {"mean_normalized_wis": 0.11, "coverage": 0.79}, "95": {"mean_normalized_wis": 0.21, "coverage": 0.93}},
                "primary_normalized_wis": 0.16,
                "coverage_gap": 0.02,
                "lockbox": {"summary": {"primary_normalized_wis": 0.15}},
            },
            "risk": {
                "contract": "exact_only",
                "by_quantile": {
                    "q80": {
                        "status": "no_incremental_risk_signal",
                        "event_count": 3,
                        "metrics": {
                            "candidate_brier": 0.2,
                            "prevalence_brier": 0.19,
                            "persistence_brier": 0.21,
                            "candidate_recall_at_k": 0.33,
                            "persistence_recall_at_k": 0.0,
                        },
                    }
                },
                "decision": {"status": "revert", "why": "No signal."},
            },
            "decision": {"winner": "EXP-UQ-01", "status": "keep_conformal", "why": "Safer."},
        },
        "dense": {
            "experiment_id": "EXP-R10-DENSE-M1-C1-H1",
            "contract_name": "purged_dense",
            "conformal": {
                "summary": {"80": {"mean_normalized_wis": 0.09, "coverage": 0.81}, "95": {"mean_normalized_wis": 0.18, "coverage": 0.95}},
                "primary_normalized_wis": 0.135,
                "coverage_gap": 0.005,
                "lockbox": {"summary": {"primary_normalized_wis": 0.12}},
            },
            "bootstrap": {
                "summary": {"80": {"mean_normalized_wis": 0.08, "coverage": 0.8}, "95": {"mean_normalized_wis": 0.17, "coverage": 0.94}},
                "primary_normalized_wis": 0.125,
                "coverage_gap": 0.01,
                "lockbox": {"summary": {"primary_normalized_wis": 0.11}},
            },
            "risk": {
                "contract": "purged_dense",
                "by_quantile": {
                    "q80": {
                        "status": "incremental_risk_signal",
                        "event_count": 6,
                        "metrics": {
                            "candidate_brier": 0.14,
                            "prevalence_brier": 0.18,
                            "persistence_brier": 0.17,
                            "candidate_recall_at_k": 0.5,
                            "persistence_recall_at_k": 0.33,
                        },
                    }
                },
                "decision": {"status": "keep", "why": "Signal."},
            },
            "decision": {"winner": "EXP-UQ-02", "status": "keep_bootstrap", "why": "Better WIS."},
        },
    }
    markdown = batch._markdown_report(payload)
    assert "EXP-UQ-01" in markdown
    assert "EXP-UQ-02" in markdown
    assert "High-Error Event Prediction" in markdown
    assert "EXP-R10-M1-F1-C1" in markdown
