from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_current_champion_exact_support_partition_batch as batch


def test_partition_summary_splits_common_and_new() -> None:
    result = {
        "quarterly_rows": [
            {
                "holdout_target_rows": [
                    {
                        "quarter": "2020-Q1",
                        "diagnosed_plhiv": 100.0,
                        "alive_on_art": 80.0,
                        "new_diagnosed_cases_period": 10.0,
                        "diagnosed_plhiv_tier": "exact_observed",
                        "alive_on_art_tier": "exact_observed",
                        "new_diagnosed_cases_period_tier": "exact_observed",
                    },
                    {
                        "quarter": "2021-Q1",
                        "diagnosed_plhiv": 120.0,
                        "alive_on_art": 90.0,
                        "new_diagnosed_cases_period": 12.0,
                        "diagnosed_plhiv_tier": "exact_observed",
                        "alive_on_art_tier": "exact_observed",
                        "new_diagnosed_cases_period_tier": "exact_observed",
                    },
                ],
                "candidate_prediction_rows": [
                    {"quarter": "2020-Q1", "diagnosed_plhiv": 102.0, "alive_on_art": 81.0, "new_diagnosed_cases_period": 11.0},
                    {"quarter": "2021-Q1", "diagnosed_plhiv": 110.0, "alive_on_art": 70.0, "new_diagnosed_cases_period": 10.0},
                ],
            }
        ]
    }
    summary = batch._partition_summary(
        result,
        common_quarters={
            "diagnosed_plhiv": {"2020-Q1"},
            "alive_on_art": {"2020-Q1"},
            "new_diagnosed_cases_period": {"2020-Q1"},
        },
        new_quarters={
            "diagnosed_plhiv": {"2021-Q1"},
            "alive_on_art": {"2021-Q1"},
            "new_diagnosed_cases_period": {"2021-Q1"},
        },
        metric_scales={
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 100.0,
            "new_diagnosed_cases_period": 10.0,
        },
    )
    assert summary["common"]["by_metric"]["diagnosed_plhiv"]["raw_mae"] == 2.0
    assert summary["new"]["by_metric"]["alive_on_art"]["raw_mae"] == 20.0
    assert summary["common"]["quarterly_mean_mae"] is not None
    assert summary["new"]["quarterly_mean_mae"] is not None


def test_run_batch_writes_support_partition_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        batch,
        "_evaluate_spec_list",
        lambda *, archive_run_id, specs: [
            {
                "experiment_id": getattr(specs[0], "experiment_id", "EXP-R10-EXACT-CHAMPION"),
                "quarterly_rows": [
                    {
                        "holdout_target_rows": [
                            {
                                "quarter": "2020-Q1",
                                "diagnosed_plhiv": 100.0,
                                "alive_on_art": 80.0,
                                "new_diagnosed_cases_period": 10.0,
                                "diagnosed_plhiv_tier": "exact_observed",
                                "alive_on_art_tier": "exact_observed",
                                "new_diagnosed_cases_period_tier": "exact_observed",
                            },
                            {
                                "quarter": "2021-Q1",
                                "diagnosed_plhiv": 120.0,
                                "alive_on_art": 90.0,
                                "new_diagnosed_cases_period": 12.0,
                                "diagnosed_plhiv_tier": "exact_observed",
                                "alive_on_art_tier": "exact_observed",
                                "new_diagnosed_cases_period_tier": "exact_observed",
                            },
                        ],
                        "candidate_prediction_rows": [
                            {"quarter": "2020-Q1", "diagnosed_plhiv": 101.0, "alive_on_art": 81.0, "new_diagnosed_cases_period": 10.5},
                            {"quarter": "2021-Q1", "diagnosed_plhiv": 118.0, "alive_on_art": 86.0, "new_diagnosed_cases_period": 11.5},
                        ],
                    }
                ],
                "quarterly_summary": {
                    "candidate_mean_mae": 0.08,
                    "candidate_worst_mae": 0.15,
                    "endpoint_audit_summary": {
                        "holdout_support_counts": {
                            "diagnosed_plhiv": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                            "alive_on_art": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                            "new_diagnosed_cases_period": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                        },
                        "suppression_honesty_flags": {"scored_direct_support": 2},
                    },
                },
                "annual_summary": {"candidate_mean_incidence_error": 0.03},
                "best_candidate": {"transition_model": "direct_observation_repair"},
            }
        ]
        if len(specs) == 1
        else [
            {
                "experiment_id": spec.experiment_id,
                "quarterly_rows": [
                    {
                        "holdout_target_rows": [
                            {
                                "quarter": "2020-Q1",
                                "diagnosed_plhiv": 100.0,
                                "alive_on_art": 80.0,
                                "new_diagnosed_cases_period": 10.0,
                                "diagnosed_plhiv_tier": "exact_observed",
                                "alive_on_art_tier": "exact_observed",
                                "new_diagnosed_cases_period_tier": "exact_observed",
                            },
                            {
                                "quarter": "2021-Q1",
                                "diagnosed_plhiv": 120.0,
                                "alive_on_art": 90.0,
                                "new_diagnosed_cases_period": 12.0,
                                "diagnosed_plhiv_tier": "exact_observed",
                                "alive_on_art_tier": "exact_observed",
                                "new_diagnosed_cases_period_tier": "exact_observed",
                            },
                        ],
                        "candidate_prediction_rows": [
                            {"quarter": "2020-Q1", "diagnosed_plhiv": 100.5, "alive_on_art": 80.5, "new_diagnosed_cases_period": 10.5},
                            {"quarter": "2021-Q1", "diagnosed_plhiv": 100.0, "alive_on_art": 70.0, "new_diagnosed_cases_period": 11.0},
                        ],
                    }
                ],
                "quarterly_summary": {
                    "candidate_mean_mae": 0.07 if "piecewise" in spec.experiment_id else 0.09,
                    "candidate_worst_mae": 0.14,
                    "endpoint_audit_summary": {
                        "holdout_support_counts": {
                            "diagnosed_plhiv": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                            "alive_on_art": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                            "new_diagnosed_cases_period": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                        },
                        "suppression_honesty_flags": {"scored_direct_support": 2},
                    },
                },
                "annual_summary": {"candidate_mean_incidence_error": 0.03},
                "best_candidate": {"transition_model": "direct_observation_repair"},
            }
            for spec in specs
        ],
    )

    payload = batch.run_tr_v3_current_champion_exact_support_partition_batch(
        run_id="support-partition",
        merged_archive_run_id="merged",
        baseline_archive_run_id="baseline",
    )

    assert payload["decision"]["status"] in {
        "keep_local_exact_refresh",
        "reopen_broader_model_family_exploration",
    }
    assert (
        tmp_path
        / "artifacts"
        / "runs"
        / "support-partition"
        / "analysis"
        / "tr_v3_current_champion_exact_support_partition_batch_report.json"
    ).exists()
