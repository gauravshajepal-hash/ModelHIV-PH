from __future__ import annotations

from pathlib import Path

import pytest

from epigraph_ph.phase3.dense_quarterly_panel import _reconciled_stock_path, build_dense_quarterly_panel_rows
from epigraph_ph.phase3.tr_v3_experiment_suite import _filtered_normalized_mae


def test_dense_quarterly_panel_live_archive_smoke() -> None:
    archive_run_id = "harp-archive-wdi-standard-20260411-s09"
    archive_path = Path("D:/EpiGraph_PH/artifacts/runs") / archive_run_id / "harp_archive" / "historical_metric_rows.json"
    if not archive_path.exists():
        pytest.skip("Live archive is not available.")
    payload = build_dense_quarterly_panel_rows(archive_run_id)
    assert payload["summary"]["quarter_count"] >= 60
    assert payload["summary"]["score_eligible_quarters"] >= 40
    assert 2010 in payload["summary"]["score_eligible_years"]
    assert 2025 in payload["summary"]["score_eligible_years"]


def test_dense_quarterly_panel_max_quarter_filters_future_rows() -> None:
    archive_run_id = "harp-archive-wdi-standard-20260411-s09"
    archive_path = Path("D:/EpiGraph_PH/artifacts/runs") / archive_run_id / "harp_archive" / "historical_metric_rows.json"
    if not archive_path.exists():
        pytest.skip("Live archive is not available.")
    payload = build_dense_quarterly_panel_rows(archive_run_id, max_quarter="2012-Q4")
    assert payload["summary"]["available_through_quarter"] == "2012-Q4"
    assert payload["summary"]["quarter_range"][1] == "2012-Q4"


def test_filtered_normalized_mae_skips_rule_based_targets() -> None:
    prediction_rows = [
        {"quarter": "2024-Q1", "diagnosed_plhiv": 10.0, "alive_on_art": 5.0, "new_diagnosed_cases_period": 2.0},
        {"quarter": "2024-Q2", "diagnosed_plhiv": 12.0, "alive_on_art": 6.0, "new_diagnosed_cases_period": 3.0},
    ]
    target_rows = [
        {
            "quarter": "2024-Q1",
            "diagnosed_plhiv": 9.0,
            "diagnosed_plhiv_tier": "exact_observed",
            "alive_on_art": 5.0,
            "alive_on_art_tier": "rule_based_extrapolated",
            "new_diagnosed_cases_period": 2.0,
            "new_diagnosed_cases_period_tier": "bridge_observed",
        },
        {
            "quarter": "2024-Q2",
            "diagnosed_plhiv": 20.0,
            "diagnosed_plhiv_tier": "rule_based_extrapolated",
            "alive_on_art": 6.0,
            "alive_on_art_tier": "rule_based_extrapolated",
            "new_diagnosed_cases_period": 7.0,
            "new_diagnosed_cases_period_tier": "rule_based_extrapolated",
        },
    ]
    score = _filtered_normalized_mae(
        prediction_rows,
        target_rows,
        {"diagnosed_plhiv": 20.0, "alive_on_art": 10.0, "new_diagnosed_cases_period": 10.0},
        allowed_tiers={"exact_observed", "bridge_observed"},
        eps=1e-6,
    )
    assert score == pytest.approx((abs(10.0 - 9.0) / 20.0 + abs(2.0 - 2.0) / 10.0) / 2.0)


def test_reconciled_stock_path_backcasts_from_anchor_using_flows_and_deaths() -> None:
    values, methods = _reconciled_stock_path(
        ["2010-Q1", "2010-Q2", "2010-Q3", "2010-Q4"],
        anchors={"2010-Q4": 100.0},
        inflows={"2010-Q2": 10.0, "2010-Q3": 20.0, "2010-Q4": 30.0},
        outflows={"2010-Q2": 1.0, "2010-Q3": 2.0, "2010-Q4": 3.0},
        floor_map={},
    )
    assert values["2010-Q4"] == pytest.approx(100.0)
    assert values["2010-Q3"] == pytest.approx(73.0)
    assert values["2010-Q2"] == pytest.approx(55.0)
    assert values["2010-Q1"] == pytest.approx(46.0)
    assert methods["2010-Q3"] == "reverse_flow_death"
