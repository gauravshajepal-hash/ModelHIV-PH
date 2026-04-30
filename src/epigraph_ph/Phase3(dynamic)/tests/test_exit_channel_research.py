from __future__ import annotations

from phase3_dynamic.exit_channel_research import _zero_external_channel, _zero_transition


def test_zero_external_channel_preserves_other_channel_state_flows() -> None:
    result = _zero_external_channel(
        {
            "2025-Q1": {
                "mortality_removal": {"U": 1.0, "D": 2.0},
                "treatment_non_initiation": {"D": 3.0},
            }
        },
        "mortality_removal",
    )

    assert result["2025-Q1"]["mortality_removal"] == {"U": 0.0, "D": 0.0}
    assert result["2025-Q1"]["treatment_non_initiation"] == {"D": 3.0}


def test_zero_transition_only_removes_requested_hazard() -> None:
    result = _zero_transition(
        {
            "2025-Q1": {
                "A_to_L": 0.2,
                "L_to_A": 0.1,
            }
        },
        "A_to_L",
    )

    assert result["2025-Q1"]["A_to_L"] == 0.0
    assert result["2025-Q1"]["L_to_A"] == 0.1
