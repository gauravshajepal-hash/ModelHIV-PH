from __future__ import annotations

from epigraph_ph.phase2.determinant_robustness import _binomial_upper_tail, _status_for_edge


def test_determinant_robustness_requires_all_completed_refits_for_default_use() -> None:
    source_report = {
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
                "excluded_source_family": "google_mobility",
                "ablation_status": "completed",
                "reestimated_status": "absent",
            },
        ],
    }
    edge_report = {
        "phase3_prior_eligible": False,
        "source_ablation": {"passed": True},
        "time_window_falsification": {"passed": True},
        "weight": 0.3,
    }

    result = _status_for_edge(
        edge_key="direct:a->b:lag1",
        source_reestimated_report=source_report,
        edge_falsification_row=edge_report,
    )

    assert result["phase3_default_allowed"] is False
    assert result["exploratory_covariate_only"] is True
    assert result["survival_fraction"] == 0.5


def test_exact_binomial_tail_has_no_threshold_side_effect() -> None:
    assert _binomial_upper_tail(2, 2) == 0.25
    assert _binomial_upper_tail(0, 0) is None
