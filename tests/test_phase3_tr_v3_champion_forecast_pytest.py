from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_champion_forecast as champion


def test_future_quarters_roll_over_year_boundary() -> None:
    assert champion._future_quarters("2025-Q4", 4) == [
        "2026-Q1",
        "2026-Q2",
        "2026-Q3",
        "2026-Q4",
    ]


def test_collect_metric_residuals_filters_by_allowed_tiers() -> None:
    quarterly_rows = [
        {
            "holdout_target_rows": [
                {
                    "quarter": "2024-Q1",
                    "diagnosed_plhiv": 100.0,
                    "alive_on_art": 50.0,
                    "new_diagnosed_cases_period": 10.0,
                    "metric_tiers": {
                        "diagnosed_plhiv": "exact_observed",
                        "alive_on_art": "bridge_observed",
                        "new_diagnosed_cases_period": "rule_based_extrapolated",
                    },
                }
            ],
            "candidate_prediction_rows": [
                {
                    "quarter": "2024-Q1",
                    "diagnosed_plhiv": 110.0,
                    "alive_on_art": 48.0,
                    "new_diagnosed_cases_period": 12.0,
                }
            ],
        }
    ]

    residuals = champion._collect_metric_residuals(
        quarterly_rows,
        allowed_tiers={"exact_observed", "bridge_observed"},
    )

    assert residuals["diagnosed_plhiv"] == [10.0]
    assert residuals["alive_on_art"] == [-2.0]
    assert residuals["new_diagnosed_cases_period"] == []


def test_apply_empirical_intervals_adds_metric_bands() -> None:
    forecast_rows = [
        {
            "quarter": "2026-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 70.0,
            "new_diagnosed_cases_period": 8.0,
            "virally_suppressed": 50.0,
        }
    ]
    residual_payload = {
        "metrics": {
            "diagnosed_plhiv": {"q_low": -5.0, "q_high": 7.0},
            "alive_on_art": {"q_low": -3.0, "q_high": 4.0},
            "new_diagnosed_cases_period": {"q_low": -2.0, "q_high": 3.0},
            "virally_suppressed": {"q_low": -4.0, "q_high": 5.0},
        }
    }

    enriched = champion._apply_empirical_intervals(forecast_rows, residual_payload)

    assert enriched[0]["diagnosed_plhiv_lower"] == 115.0
    assert enriched[0]["diagnosed_plhiv_upper"] == 127.0
    assert enriched[0]["alive_on_art_lower"] == 67.0
    assert enriched[0]["alive_on_art_upper"] == 74.0
    assert enriched[0]["new_diagnosed_cases_period_lower"] == 6.0
    assert enriched[0]["new_diagnosed_cases_period_upper"] == 11.0
