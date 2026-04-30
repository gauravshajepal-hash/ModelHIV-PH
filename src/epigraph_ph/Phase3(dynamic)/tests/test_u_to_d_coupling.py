from __future__ import annotations

import pytest

from phase3_dynamic.u_to_d_coupling import (
    _bounded_u_to_d_hazard,
    _promotion_gate,
    _repair_summary,
    _simulate_sequence_with_u_to_d_targets,
)


def test_bounded_u_to_d_hazard_uses_empirical_bounds() -> None:
    result = _bounded_u_to_d_hazard(
        target_diagnosis_flow=30.0,
        undiagnosed_state=100.0,
        floor=0.01,
        ceiling=0.2,
        eps=1e-9,
    )

    assert result["raw_hazard"] == pytest.approx(0.3)
    assert result["bounded_hazard"] == pytest.approx(0.2)
    assert result["implied_flow"] == pytest.approx(20.0)


def test_u_to_d_coupling_moves_mass_through_state_transition() -> None:
    result = _simulate_sequence_with_u_to_d_targets(
        initial_state={"U": 100.0, "D": 0.0, "A": 0.0, "T": 0.0, "V": 0.0, "L": 0.0, "R": 0.0},
        target_rows=[{"quarter": "2025-Q1"}],
        hazard_map={"2025-Q1": {"U_to_D": 0.0}},
        u_to_d_flow_targets={"2025-Q1": 30.0},
        u_to_d_bounds={"floor": 0.0, "ceiling": 0.2},
        eps=1e-9,
        incidence_inflow_map={"2025-Q1": 0.0},
    )

    prediction = result["prediction_rows"][0]
    state = result["trajectory_rows"][0]["state_values"]
    coupling = result["u_to_d_coupling_rows"][0]

    assert prediction["new_diagnosed_cases_period"] == pytest.approx(20.0)
    assert state["U"] == pytest.approx(80.0)
    assert state["D"] == pytest.approx(20.0)
    assert coupling["target_diagnosis_flow"] == pytest.approx(30.0)
    assert coupling["implied_u_to_d_flow"] == pytest.approx(20.0)


def test_u_to_d_promotion_requires_short_horizon_and_r10_lifted_gate() -> None:
    gate = _promotion_gate(
        one_year_gate={"status": "pass", "blockers": []},
        shock_gate={"status": "fail", "blockers": ["candidate_lifted_path_mean_not_better_than_r10_reference"]},
    )

    assert gate["status"] == "reject_short_horizon_coupling_claim"
    assert "fails_shock_aware_lifted_trajectory_gate" in gate["blockers"]


def test_repair_summary_separates_targets_from_gate_sources() -> None:
    summary = _repair_summary(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "repair_diagnostics_summary": {
                    "status": "completed",
                    "u_to_d_target_count": 1,
                    "u_to_d_target_source_counts": {"monthly_reporting_nowcast": 1},
                    "selective_gate_source_counts": {"monthly_reporting_nowcast": 1, "strict_base": 3},
                },
            },
            {
                "train_end_year": 2022,
                "holdout_years": [2023],
                "repair_diagnostics_summary": {
                    "status": "completed",
                    "u_to_d_target_count": 0,
                    "u_to_d_target_source_counts": {},
                    "selective_gate_source_counts": {"strict_base": 4},
                },
            },
        ]
    )

    assert summary["total_u_to_d_target_count"] == 1
    assert summary["u_to_d_target_source_counts"] == {"monthly_reporting_nowcast": 1}
    assert summary["selective_gate_source_counts"] == {"monthly_reporting_nowcast": 1, "strict_base": 7}
