from __future__ import annotations

from phase3_dynamic.loop import _build_data_provenance_payload, _markdown_report


def test_markdown_report_renders_missing_data_ladder_section() -> None:
    payload = {
        "family_name": "TR-V3-00",
        "source_run_id": "smoke-latent-blocks",
        "generated_at": "2026-04-12T00:00:00+00:00",
        "decision": "revert",
        "decision_reason": "Benchmark gate failed.",
        "best_candidate": {
            "config": {"dynamic_cfg": {"ridge_penalty": 0.01}},
            "score": {
                "candidate_mean_mae": 0.2,
                "carry_forward_mean_mae": 0.1,
                "candidate_worst_mae": 0.3,
                "carry_forward_worst_mae": 0.2,
            },
            "rows": [
                {
                    "train_end_year": 2021,
                    "holdout_years": [2022],
                    "carry_forward": {"mae": 0.1},
                    "candidate": {"mae": 0.2},
                }
            ],
        },
        "data_provenance": {
            "ladder": [
                "exact_observed",
                "bridge_observed",
                "rule_based_extrapolated",
                "latent_imputed",
                "rejected_or_quarantined",
            ],
            "overall_observation_rows": {
                "row_tier_counts": {
                    "exact_observed": 4,
                    "bridge_observed": 1,
                    "rule_based_extrapolated": 0,
                    "latent_imputed": 0,
                    "rejected_or_quarantined": 0,
                }
            },
            "benchmark_weighted": {
                "holdout_state_rows": {
                    "row_tier_counts": {
                        "exact_observed": 0,
                        "bridge_observed": 0,
                        "rule_based_extrapolated": 1,
                        "latent_imputed": 1,
                        "rejected_or_quarantined": 0,
                    }
                }
            },
        },
    }

    report = _markdown_report(payload)

    assert "## Data Provenance" in report
    assert "Missing-data ladder" in report
    assert "Overall observation rows" in report
    assert "Benchmark-weighted holdout state" in report


def test_build_data_provenance_payload_includes_benchmark_weighted_rows() -> None:
    payload = _build_data_provenance_payload(
        [
            {
                "quarter": "2024-Q1",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 80.0,
                "new_diagnosed_cases_period": 10.0,
                "metric_provenance": {
                    "diagnosed_plhiv": {"tier": "exact_observed"},
                    "alive_on_art": {"tier": "exact_observed"},
                    "new_diagnosed_cases_period": {"tier": "bridge_observed"},
                },
                "row_provenance_tier": "bridge_observed",
            }
        ],
        {
            "rows": [
                {
                    "dataset_provenance": {
                        "observation_rows": {
                            "row_count": 1,
                            "row_tier_counts": {
                                "exact_observed": 1,
                                "bridge_observed": 0,
                                "rule_based_extrapolated": 0,
                                "latent_imputed": 0,
                                "rejected_or_quarantined": 0,
                            },
                        }
                    }
                }
            ]
        },
    )

    assert payload["benchmark_weighted"]["observation_rows"]["row_count"] == 1


def test_markdown_report_renders_observation_score_ledger_section() -> None:
    payload = {
        "family_name": "TR-V3-00",
        "source_run_id": "smoke-latent-blocks",
        "generated_at": "2026-04-12T00:00:00+00:00",
        "decision": "keep",
        "decision_reason": "Benchmark gate passed.",
        "best_candidate": {
            "config": {"dynamic_cfg": {"ridge_penalty": 0.01}},
            "score": {
                "candidate_mean_mae": 0.1,
                "carry_forward_mean_mae": 0.2,
                "candidate_worst_mae": 0.15,
                "carry_forward_worst_mae": 0.25,
            },
            "rows": [
                {
                    "train_end_year": 2021,
                    "holdout_years": [2022],
                    "carry_forward": {"mae": 0.2},
                    "candidate": {"mae": 0.1},
                }
            ],
        },
        "observation_score_ledger_summary": {
            "schema_version": "phase3_dynamic_observation_score_ledger.v1",
            "entry_count": 3,
            "candidate_scored_entry_count": 2,
            "carry_forward_scored_entry_count": 2,
            "metrics": {
                "diagnosed_plhiv": {
                    "entry_count": 1,
                    "candidate_scored_count": 1,
                    "carry_forward_scored_count": 1,
                    "metric_tier_counts": {
                        "exact_observed": 1,
                        "bridge_observed": 0,
                        "rule_based_extrapolated": 0,
                        "latent_imputed": 0,
                        "rejected_or_quarantined": 0,
                    },
                }
            },
        },
    }

    report = _markdown_report(payload)

    assert "## Observation-to-Score Ledger" in report
    assert "Candidate scored entries" in report
    assert "`diagnosed_plhiv`" in report
