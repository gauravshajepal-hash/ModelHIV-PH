from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_monthly_edge_audit_batch as batch


def test_edge_delta_rows_marks_new_and_retained_edges() -> None:
    baseline = [
        {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.2, "stability": 0.5},
        {"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.1, "stability": 0.5},
    ]
    candidate = [
        {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.3, "stability": 0.7},
        {"source": "testing_engagement", "target": "suppression_capacity", "lag": 1, "weight": 0.4, "stability": 0.8},
    ]

    rows = batch._edge_delta_rows(baseline, candidate)
    lookup = {(row["source"], row["target"], row["lag"]): row for row in rows}

    retained = lookup[("testing_engagement", "care_access_continuity", 1)]
    assert retained["edge_status"] == "retained"
    assert retained["baseline_score"] == 0.1
    assert retained["candidate_score"] == 0.21

    new_edge = lookup[("testing_engagement", "suppression_capacity", 1)]
    assert new_edge["edge_status"] == "new"
    assert new_edge["baseline_score"] == 0.0
    assert new_edge["candidate_score"] == 0.32

    dropped = lookup[("care_access_continuity", "suppression_capacity", 1)]
    assert dropped["edge_status"] == "dropped"
    assert dropped["candidate_score"] == 0.0


def test_confounder_flags_fire_on_support_collapse() -> None:
    flags = batch._confounder_flags(
        {
            "baseline_diagnosis_flow_point_count": 151,
            "candidate_diagnosis_flow_point_count": 0,
            "baseline_harp_program_point_count": 16,
            "candidate_harp_program_point_count": 9,
            "baseline_diagnosis_flow_file_count": 151,
            "candidate_diagnosis_flow_file_count": 0,
        }
    )

    assert len(flags) == 3
    assert "diagnosis-flow support dropped" in flags[0]
