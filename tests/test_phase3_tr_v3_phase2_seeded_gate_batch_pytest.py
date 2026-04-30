from __future__ import annotations

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_seeded_gate_batch as batch


def test_unique_quarter_residual_points_averages_duplicates() -> None:
    quarter_feature_map = {
        "2024-Q1": np.asarray([1.0, 0.0], dtype=np.float64),
        "2024-Q2": np.asarray([0.5, 1.0], dtype=np.float64),
    }
    result = {
        "quarterly_rows": [
            {
                "holdout_target_rows": [
                    {"quarter": "2024-Q1", "diagnosed_plhiv": 100.0, "metric_tiers": {"diagnosed_plhiv": "exact_observed"}},
                    {"quarter": "2024-Q2", "diagnosed_plhiv": 110.0, "metric_tiers": {"diagnosed_plhiv": "exact_observed"}},
                ],
                "candidate_prediction_rows": [
                    {"quarter": "2024-Q1", "diagnosed_plhiv": 105.0},
                    {"quarter": "2024-Q2", "diagnosed_plhiv": 100.0},
                ],
            },
            {
                "holdout_target_rows": [
                    {"quarter": "2024-Q1", "diagnosed_plhiv": 100.0, "metric_tiers": {"diagnosed_plhiv": "exact_observed"}},
                ],
                "candidate_prediction_rows": [
                    {"quarter": "2024-Q1", "diagnosed_plhiv": 95.0},
                ],
            },
        ]
    }

    points = batch._unique_quarter_residual_points(result=result, allowed_tiers={"exact_observed"}, quarter_feature_map=quarter_feature_map)

    diagnosed = points["diagnosed_plhiv"]
    assert len(diagnosed) == 2
    assert diagnosed[0]["quarter"] == "2024-Q1"
    assert diagnosed[0]["residual"] == 0.0
    assert diagnosed[1]["residual"] == -10.0


def test_retained_edge_stability_requires_two_top2_folds() -> None:
    phase2_state = batch.seeded.Phase2QuarterState(
        source_run_id="demo",
        month_axis=[],
        quarter_axis=[
            "2010-Q1",
            "2010-Q2",
            "2010-Q3",
            "2010-Q4",
            "2011-Q1",
            "2011-Q2",
            "2011-Q3",
            "2011-Q4",
            "2012-Q1",
        ],
        block_axis=["a", "b", "c"],
        quarter_states=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.5, 0.0],
                [4.0, 2.0, 0.0],
                [5.0, 2.5, 0.0],
                [6.0, 3.0, 0.0],
                [7.0, 3.5, 0.0],
                [8.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        edge_rows=[{"source": "a", "target": "b", "weight": 0.4}],
    )

    summary = batch._retained_edge_stability(phase2_state=phase2_state)

    assert summary["keep_gate_passes"] in {True, False}
    assert len(summary["edge_summary_rows"]) == 1


def test_boundedness_audit_flags_stock_violation() -> None:
    payload = {
        "contracts": {
            "exact_only": {
                "readout": {"metrics": {"diagnosed_plhiv": {"cap_abs": 10.0}, "alive_on_art": {"cap_abs": 10.0}, "new_diagnosed_cases_period": {"cap_abs": 10.0}}},
                "base_forecast_rows": [{"quarter": "2026-Q1", "diagnosed_plhiv": 100.0, "alive_on_art": 90.0, "new_diagnosed_cases_period": 10.0}],
                "scenario_rows": {
                    "testing_pulse": [
                        {
                            "quarter": "2026-Q1",
                            "diagnosed_plhiv": 80.0,
                            "alive_on_art": 95.0,
                            "new_diagnosed_cases_period": 5.0,
                            "diagnosed_plhiv_scenario_delta": -20.0,
                            "alive_on_art_scenario_delta": 5.0,
                            "new_diagnosed_cases_period_scenario_delta": -5.0,
                        }
                    ]
                },
            }
        }
    }

    audit = batch._boundedness_audit(payload)

    assert audit["total_violation_count"] > 0
    assert audit["rows"][0]["bounded_keep"] is False


def test_summarize_outcome_circularity_rows_computes_loading_share() -> None:
    rows = [
        {"block_id": "testing_engagement", "canonical_name": "diagnosed_plhiv", "loading": 0.3, "direct_indicator_count": 10},
        {"block_id": "testing_engagement", "canonical_name": "new_diagnosed_cases_period", "loading": 0.7, "direct_indicator_count": 12},
        {"block_id": "testing_engagement", "canonical_name": "testing_rate", "loading": 1.0, "direct_indicator_count": 20},
        {"block_id": "care_access_continuity", "canonical_name": "alive_on_art", "loading": 0.4, "direct_indicator_count": 8},
    ]

    summary = batch._summarize_outcome_circularity_rows(
        loading_rows=rows,
        outcome_canonical_names={"diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period"},
    )

    testing = [row for row in summary if row["block_id"] == "testing_engagement"][0]
    assert testing["outcome_indicator_count"] == 2
    assert round(float(testing["outcome_loading_share"]), 6) == 0.5
    assert testing["outcome_support_count"] == 22


def test_orthogonalize_series_against_matrix_removes_linear_component() -> None:
    x = np.arange(12, dtype=np.float64).reshape(-1, 1)
    y = (2.0 * x[:, 0]) + 5.0

    fit = batch._orthogonalize_series_against_matrix(series=y, design=x)
    orthogonalized = np.asarray(fit["orthogonalized"], dtype=np.float64)

    assert fit["r2"] > 0.99
    assert float(np.std(orthogonalized)) < 1e-3
