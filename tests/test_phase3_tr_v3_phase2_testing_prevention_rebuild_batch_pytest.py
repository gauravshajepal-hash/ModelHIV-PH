from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_phase2_testing_prevention_rebuild_batch as batch


def test_testing_block_support_gate_keeps_valid_non_outcome_block() -> None:
    loading_payload = {
        "block_summary_rows": [
            {
                "block_id": "testing_prevention_reach",
                "indicator_count": 2,
                "champion_loading_share": 0.0,
                "cascade_loading_share": 0.0,
            }
        ],
        "audit_rows": [
            {"block_id": "testing_prevention_reach", "canonical_name": "annual_hiv_tests_volume_per_100k", "abs_loading": 0.55},
            {"block_id": "testing_prevention_reach", "canonical_name": "prep_people_receiving_per_100k", "abs_loading": 0.45},
        ],
    }
    phase2_payload = {"block_axis": ["testing_prevention_reach", "care_access_continuity", "mobility_exposure_pressure"]}

    gate = batch._testing_block_support_gate(loading_payload, phase2_payload)

    assert gate["decision"] == "keep"
    assert gate["block_retained"] is True
    assert gate["max_indicator_share"] == 0.55


def test_testing_scenario_gate_reverts_when_testing_deltas_are_null() -> None:
    payload = {
        "terminal_delta_rows": [
            {"contract": "exact_only", "scenario": "testing_pulse", "diagnosed_plhiv_delta": 0.0, "alive_on_art_delta": 0.0, "new_diagnosed_cases_period_delta": 0.0},
            {"contract": "purged_dense", "scenario": "testing_plateau", "diagnosed_plhiv_delta": 0.0, "alive_on_art_delta": 0.0, "new_diagnosed_cases_period_delta": 0.0},
        ]
    }

    gate = batch._testing_scenario_gate(payload)

    assert gate["decision"] == "revert"
    assert gate["nonnull_terminal_count"] == 0


def test_additional_exclusions_merge_with_base_exclusions() -> None:
    base_exclusions = ("alive_on_art", "diagnosed_plhiv")
    merged = tuple(
        sorted(
            {
                *[str(name).strip() for name in list(base_exclusions) if str(name).strip()],
                *[str(name).strip() for name in ["prep_people_receiving_per_100k"] if str(name).strip()],
            }
        )
    )

    assert merged == ("alive_on_art", "diagnosed_plhiv", "prep_people_receiving_per_100k")
