from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_phase2_sidecar_ablation_batch as batch


def test_merge_exclusions_preserves_existing_and_adds_new() -> None:
    base_report = {
        "structural_exclusion_summary": {
            "excluded_canonicals": [
                "alive_on_art",
                "diagnosed_plhiv",
                "new_diagnosed_cases_period",
            ]
        }
    }

    merged = batch._merge_exclusions(
        base_report,
        ("tested_for_viral_load", "virally_suppressed", "alive_on_art"),
    )

    assert merged == (
        "alive_on_art",
        "diagnosed_plhiv",
        "new_diagnosed_cases_period",
        "tested_for_viral_load",
        "virally_suppressed",
    )


def test_scenario_filtered_sign_agreement_uses_requested_scenarios_only() -> None:
    rows = [
        {"scenario": "disruption_recovery", "sign_agrees": True},
        {"scenario": "disruption_recovery", "sign_agrees": False},
        {"scenario": "mobility_spike", "sign_agrees": True},
        {"scenario": "testing_pulse", "sign_agrees": False},
    ]

    score = batch._scenario_filtered_sign_agreement(
        rows,
        scenarios={"disruption_recovery", "mobility_spike"},
    )

    assert score == 2.0 / 3.0
