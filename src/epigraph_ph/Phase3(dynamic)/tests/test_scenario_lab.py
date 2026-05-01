from __future__ import annotations

from phase3_dynamic.scenario_lab import (
    _apply_projection_constraints,
    _forecast_admissibility,
    _future_quarters,
    _scenario_specs,
)


def test_future_quarters_emit_complete_horizon() -> None:
    assert _future_quarters(2026, 2027) == [
        "2026-Q1",
        "2026-Q2",
        "2026-Q3",
        "2026-Q4",
        "2027-Q1",
        "2027-Q2",
        "2027-Q3",
        "2027-Q4",
    ]


def test_scenario_specs_keep_strict_status_quo_and_label_exploratory_knobs() -> None:
    specs = _scenario_specs(
        {
            "strict_allowed_direct_edge_count": 0,
            "exploratory_direct_edge_count": 3,
            "touched_blocks": [
                "care_access_continuity",
                "mobility_exposure_pressure",
                "structural_barrier_pressure",
            ],
        }
    )

    assert specs[0]["scenario_id"] == "strict_reference_status_quo"
    assert specs[0]["allowed_use"] == "development_projection_reference"
    assert "source_stable_determinant_bundle" not in {row["scenario_id"] for row in specs}
    exploratory = [row for row in specs if str(row["scenario_id"]).startswith("exploratory_")]
    assert exploratory
    assert all(row["allowed_use"] == "exploratory_sensitivity_not_policy_effect" for row in exploratory)


def test_projection_constraints_cap_incidence_and_removal_without_hand_tuned_values() -> None:
    incidence, attrition, diagnostics = _apply_projection_constraints(
        incidence_map={"2026-Q1": 50.0, "2026-Q2": 200.0},
        attrition_map={"2026-Q1": 3.0, "2026-Q2": 99.0},
        constraints={
            "incidence_cap_per_quarter": 100.0,
            "mortality_removal_cap_per_quarter": 10.0,
        },
    )

    assert incidence == {"2026-Q1": 50.0, "2026-Q2": 100.0}
    assert attrition == {"2026-Q1": 3.0, "2026-Q2": 10.0}
    assert diagnostics["incidence_clipped_quarter_count"] == 1
    assert diagnostics["attrition_clipped_quarter_count"] == 1


def test_forecast_admissibility_blocks_failed_reference_even_if_stable() -> None:
    result = _forecast_admissibility(
        near_horizon_gate={"trust_2035_projection": True},
        long_horizon_stability={"unstable_scenario_count": 0},
        reference_quality={"beats_carry_forward": False},
    )

    assert result["status"] == "diagnostic_only_not_publication_forecast"
    assert "strict_reference_does_not_beat_carry_forward" in result["blockers"]
