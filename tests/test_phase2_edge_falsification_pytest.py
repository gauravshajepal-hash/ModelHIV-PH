from __future__ import annotations

from epigraph_ph.phase2.edge_falsification import (
    _ledger_block_support,
    _source_ablation_for_edge,
    _source_reestimated_ablation_for_edge,
    _time_window_for_edge,
)


def test_source_ablation_blocks_edge_when_single_family_carries_target_block() -> None:
    support = _ledger_block_support(
        [
            {
                "candidate_block": "mobility_exposure_pressure",
                "source_family": "openalex",
                "row_count": 4,
                "has_verifiable_locator": True,
                "years": ["2021"],
            },
            {
                "candidate_block": "structural_barrier_pressure",
                "source_family": "world_bank_wdi",
                "row_count": 3,
                "has_verifiable_locator": True,
                "years": ["2021"],
            },
        ]
    )

    result = _source_ablation_for_edge(
        {"source": "mobility_exposure_pressure", "target": "structural_barrier_pressure", "lag": 1},
        support,
    )

    assert result["passed"] is False
    assert any(row["ablated_source_family"] == "world_bank_wdi" and not row["survives"] for row in result["ablation_rows"])


def test_validation_only_rows_do_not_promote_phase2_prior_support() -> None:
    support = _ledger_block_support(
        [
            {
                "candidate_block": "mobility_exposure_pressure",
                "source_family": "unaids",
                "row_count": 4,
                "has_verifiable_locator": True,
                "allowed_use": "validation_or_auxiliary_only",
                "years": ["2024"],
            },
            {
                "candidate_block": "structural_barrier_pressure",
                "source_family": "openalex",
                "row_count": 3,
                "has_verifiable_locator": True,
                "allowed_use": "prior_context_only",
                "years": ["2024"],
            },
        ]
    )

    assert support["block_row_counts_all_verifiable"]["mobility_exposure_pressure"] == 4
    assert "mobility_exposure_pressure" not in support["block_row_counts"]
    assert support["excluded_allowed_use_counts"]["validation_or_auxiliary_only"] == 4
    result = _source_ablation_for_edge(
        {"source": "mobility_exposure_pressure", "target": "structural_barrier_pressure", "lag": 1},
        support,
    )
    assert result["passed"] is False


def test_time_window_edge_requires_same_sign_survival() -> None:
    payload = {
        "window_scale_rows": [
            {
                "window_id": "early",
                "scale": "national",
                "status": "completed",
                "falsification_window": True,
                "effective_sample_count": 20,
            },
            {
                "window_id": "late",
                "scale": "national",
                "status": "completed",
                "falsification_window": True,
                "effective_sample_count": 20,
            },
        ],
        "window_edge_rows": {
            "early": {"national": [{"edge_kind": "direct", "source": "a", "target": "b", "lag": 1, "weight": 0.2}]},
            "late": {"national": [{"edge_kind": "direct", "source": "a", "target": "b", "lag": 1, "weight": -0.2}]},
        },
    }

    result = _time_window_for_edge(
        edge={"source": "a", "target": "b", "lag": 1, "weight": 0.3},
        kind="direct",
        original_scales=["national"],
        time_window_payload=payload,
    )

    assert result["passed"] is False
    assert result["status_counts"]["survived"] == 1
    assert result["status_counts"]["sign_conflict"] == 1


def test_source_reestimated_ablation_requires_completed_family_survival() -> None:
    report = {
        "edge_summary": {
            "direct:a->b:lag1": {
                "evaluated_family_count": 2,
                "survived_family_count": 1,
                "absent_family_count": 1,
                "sign_conflict_family_count": 0,
                "passed": False,
            }
        },
        "edge_family_rows": [
            {
                "edge_key": "direct:a->b:lag1",
                "excluded_source_family": "psa",
                "ablation_status": "completed",
                "reestimated_status": "survived",
            },
            {
                "edge_key": "direct:a->b:lag1",
                "excluded_source_family": "world_bank_wdi",
                "ablation_status": "completed",
                "reestimated_status": "absent",
            },
        ],
    }

    result = _source_reestimated_ablation_for_edge(
        edge_key="direct:a->b:lag1",
        source_reestimated_report=report,
    )

    assert result["available"] is True
    assert result["passed"] is False
    assert result["evaluated_family_count"] == 2
    assert result["absent_family_count"] == 1
