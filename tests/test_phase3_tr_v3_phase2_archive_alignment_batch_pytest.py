from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_phase2_archive_alignment_batch as batch


def test_adequacy_gate_reverts_proxy_only_suppression_block() -> None:
    loading_payload = {
        "audit_rows": [
            {"block_id": "suppression_capacity", "category": "context_or_proxy", "loading": 0.6},
            {"block_id": "suppression_capacity", "category": "context_or_proxy", "loading": 0.4},
        ],
        "block_summary_rows": [
            {
                "block_id": "suppression_capacity",
                "indicator_count": 5,
                "singleton_loading_share": 1.0,
                "mean_ppc_corr": 0.0,
            }
        ],
    }

    result = batch._adequacy_gate(loading_payload)

    assert result["decision"] == "revert"
    assert result["reasons"]["context_or_proxy_share"] == 1.0


def test_scenario_sign_agreement_filters_selected_scenarios() -> None:
    rows = [
        {"scenario": "disruption_recovery", "sign_agrees": True},
        {"scenario": "disruption_recovery", "sign_agrees": False},
        {"scenario": "mobility_spike", "sign_agrees": True},
        {"scenario": "testing_pulse", "sign_agrees": False},
    ]

    assert batch._scenario_sign_agreement(rows, scenarios={"disruption_recovery", "mobility_spike"}) == 2.0 / 3.0


def test_adequacy_gate_is_not_applicable_when_suppression_is_inactive() -> None:
    result = batch._adequacy_gate(
        {"audit_rows": [], "block_summary_rows": []},
        active_block_subset=("care_access_continuity", "mobility_exposure_pressure"),
    )

    assert result["decision"] == "not_applicable"
    assert result["reasons"]["suppression_inactive"] is True
