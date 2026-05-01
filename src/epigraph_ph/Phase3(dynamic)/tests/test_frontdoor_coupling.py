from __future__ import annotations

import pytest

from phase3_dynamic.frontdoor_coupling import (
    _bounded_flow_hazard,
    _promotion_gate,
    _simulate_sequence_with_frontdoor_targets,
)


def test_bounded_flow_hazard_clips_to_empirical_bounds() -> None:
    result = _bounded_flow_hazard(
        target_flow=50.0,
        denominator=100.0,
        floor=0.05,
        ceiling=0.3,
        eps=1e-9,
    )

    assert result["raw_hazard"] == pytest.approx(0.5)
    assert result["bounded_hazard"] == pytest.approx(0.3)
    assert result["implied_flow"] == pytest.approx(30.0)


def test_frontdoor_coupling_moves_diagnosis_pressure_into_art_linkage() -> None:
    result = _simulate_sequence_with_frontdoor_targets(
        initial_state={"U": 100.0, "D": 10.0, "A": 0.0, "T": 0.0, "V": 0.0, "L": 0.0, "R": 0.0},
        target_rows=[{"quarter": "2025-Q1"}],
        hazard_map={"2025-Q1": {"U_to_D": 0.1, "D_to_A": 0.1}},
        u_to_d_flow_targets={"2025-Q1": 30.0},
        u_to_d_bounds={"floor": 0.0, "ceiling": 0.3},
        d_to_a_available_bounds={"floor": 0.0, "ceiling": 0.5},
        linkage_response={"status": "completed", "linkage_flow_ratio": 0.5},
        eps=1e-9,
        incidence_inflow_map={"2025-Q1": 0.0},
    )

    prediction = result["prediction_rows"][0]
    state = result["trajectory_rows"][0]["state_values"]
    coupling = result["frontdoor_coupling_rows"][0]

    assert prediction["new_diagnosed_cases_period"] == pytest.approx(30.0)
    assert coupling["diagnosis_pressure_delta"] == pytest.approx(20.0)
    assert coupling["target_d_to_a_flow"] == pytest.approx(11.0)
    assert coupling["realized_d_to_a_flow"] == pytest.approx(11.0)
    assert state["D"] == pytest.approx(29.0)
    assert state["A"] == pytest.approx(11.0)


def test_frontdoor_promotion_requires_lifted_r10_gate() -> None:
    gate = _promotion_gate(
        one_year_gate={"status": "pass", "blockers": []},
        shock_gate={"status": "fail", "blockers": ["candidate_lifted_path_mean_not_better_than_r10_reference"]},
    )

    assert gate["status"] == "reject_frontdoor_coupling_claim"
    assert "fails_shock_aware_lifted_trajectory_gate" in gate["blockers"]
