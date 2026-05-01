from __future__ import annotations

from phase3_dynamic.incidence_readout_full_gate import (
    _annual_incidence_gate_from_rows,
    _full_path_gate,
    _promotion_gate,
    _select_measurement_readout_map,
    _select_state_incidence_map,
)


def test_full_gate_keeps_annual_readout_out_of_state_incidence() -> None:
    result = {
        "incidence_map": {"2025-Q4": 100.0},
        "annual_measurement_readout_incidence_map": {"2025-Q4": 1000.0},
    }

    assert _select_state_incidence_map(result) == {"2025-Q4": 100.0}
    assert _select_measurement_readout_map(result) == {"2025-Q4": 1000.0}


def test_annual_incidence_gate_requires_beating_carry_and_r10() -> None:
    rows = [
        {
            "incidence_annual_score_entries": [
                {
                    "candidate_norm_error": 0.08,
                    "carry_forward_norm_error": 0.4,
                }
            ]
        }
    ]

    passed = _annual_incidence_gate_from_rows(
        rows,
        {"reference_annual_incidence_error": 0.09},
    )
    assert passed["status"] == "pass"

    failed = _annual_incidence_gate_from_rows(
        rows,
        {"reference_annual_incidence_error": 0.07},
    )
    assert failed["status"] == "fail"
    assert "annual_incidence_not_better_than_r10" in failed["blockers"]


def test_full_path_gate_requires_mean_and_worst_superiority() -> None:
    passed = _full_path_gate(
        rows=[
            {"candidate_mae": 0.1, "carry_forward_mae": 0.2},
            {"candidate_mae": 0.2, "carry_forward_mae": 0.3},
        ],
        gate_name="synthetic",
    )
    assert passed["status"] == "pass"

    failed = _full_path_gate(
        rows=[
            {"candidate_mae": 0.1, "carry_forward_mae": 0.2},
            {"candidate_mae": 0.5, "carry_forward_mae": 0.3},
        ],
        gate_name="synthetic",
    )
    assert failed["status"] == "fail"
    assert "candidate_full_path_worst_worse_than_carry_forward" in failed["blockers"]


def test_full_cascade_promotion_rejects_narrow_incidence_only_win() -> None:
    gate = _promotion_gate(
        annual_gate={"status": "pass", "blockers": []},
        one_year_gate={"status": "fail", "blockers": ["candidate_full_path_mean_not_better_than_carry_forward"]},
        five_year_gate={"status": "pass", "blockers": []},
        shock_gate={"status": "pass", "blockers": []},
        long_horizon_status={"status": "pass"},
    )

    assert gate["status"] == "reject_full_cascade_claim"
    assert "fails_one_year_full_path_gate" in gate["blockers"]
    assert "annual incidence measurement-readout" in gate["claim_boundary"]
