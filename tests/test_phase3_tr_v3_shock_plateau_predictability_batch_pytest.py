from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_shock_plateau_predictability_batch as batch


def test_lagged_shared_shock_predictability_reports_insufficient_events() -> None:
    payload = {
        "ordered_quarters": [
            {"quarter": "2020-Q1", "shared_burst": False, "diagnosed_only_burst": False, "mean_abs_z": 0.2, "burst_metrics": [], "metrics": {}},
            {"quarter": "2020-Q2", "shared_burst": False, "diagnosed_only_burst": False, "mean_abs_z": 0.1, "burst_metrics": [], "metrics": {}},
            {"quarter": "2020-Q3", "shared_burst": True, "diagnosed_only_burst": False, "mean_abs_z": 2.0, "burst_metrics": ["diagnosed_plhiv", "alive_on_art"], "metrics": {}},
            {"quarter": "2020-Q4", "shared_burst": False, "diagnosed_only_burst": False, "mean_abs_z": 0.2, "burst_metrics": [], "metrics": {}},
            {"quarter": "2021-Q1", "shared_burst": False, "diagnosed_only_burst": False, "mean_abs_z": 0.2, "burst_metrics": [], "metrics": {}},
            {"quarter": "2021-Q2", "shared_burst": False, "diagnosed_only_burst": False, "mean_abs_z": 0.1, "burst_metrics": [], "metrics": {}},
        ]
    }
    result = batch._lagged_shared_shock_predictability(payload, contract_name="exact_only")
    assert result["status"] == "insufficient_events"


def test_markdown_report_mentions_recommendation() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "archive-sample",
        "exact_experiment_id": "EXP-R10-M1-F1-C1",
        "dense_experiment_id": "EXP-R10-DENSE-M1-C1-H1",
        "shock_predictability": {
            "exact_only": {"status": "insufficient_events", "event_count": 1, "metrics": {}},
            "purged_dense": {"status": "no_incremental_predictive_signal", "event_count": 5, "metrics": {}},
        },
        "plateau_predictability": {
            "exact_only": {"metrics": {}},
            "purged_dense": {"metrics": {}},
        },
        "decision": {
            "shock": "do_not_promote_shock_model",
            "plateau": "pursue_plateau_sidecar",
            "why": "Example decision.",
        },
    }
    markdown = batch._markdown_report(payload)
    assert "do_not_promote_shock_model" in markdown
    assert "pursue_plateau_sidecar" in markdown
