from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_indicator_inventory_batch as batch


def test_cadence_class_detects_biannual_and_singleton() -> None:
    assert batch._cadence_class({"2020", "2022", "2024"}, {"annual"}, 3) == "biannual"
    assert batch._cadence_class({"2024"}, {"annual"}, 1) == "single_reading"


def test_current_status_prefers_active_and_excluded_states() -> None:
    assert (
        batch._current_status(
            indicator_name="care_access_continuity",
            active_structural_names={"care_access_continuity"},
            active_full_eval_names=set(),
            active_excluded_names=set(),
            candidate_structural_names=set(),
            national_numeric_obs=10,
            subnational_numeric_obs=0,
        )
        == "active_structural"
    )
    assert (
        batch._current_status(
            indicator_name="diagnosed_plhiv",
            active_structural_names=set(),
            active_full_eval_names={"diagnosed_plhiv"},
            active_excluded_names={"diagnosed_plhiv"},
            candidate_structural_names=set(),
            national_numeric_obs=87,
            subnational_numeric_obs=0,
        )
        == "evaluation_only"
    )


def test_summarize_historical_panel_rows_extracts_numeric_fields() -> None:
    rows = [
        {
            "time": "2020-12",
            "year": 2020,
            "estimated_plhiv": 12000,
            "diagnosed_plhiv": 8000,
            "alive_on_art": None,
        },
        {
            "time": "2021-12",
            "year": 2021,
            "estimated_plhiv": 13000,
            "diagnosed_plhiv": 9000,
            "alive_on_art": 5000,
        },
    ]

    summary = batch._summarize_historical_panel_rows(rows)

    assert summary["estimated_plhiv"]["numeric_obs_count"] == 2
    assert summary["diagnosed_plhiv"]["cadence_class"] == "annual"
    assert summary["alive_on_art"]["numeric_obs_count"] == 1
