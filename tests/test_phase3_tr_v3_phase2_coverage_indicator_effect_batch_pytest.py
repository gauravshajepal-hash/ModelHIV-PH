from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_phase2_coverage_indicator_effect_batch as batch
from epigraph_ph.runtime import write_json


def test_edge_signature_rows_extracts_sorted_direct_edges() -> None:
    payload = {
        "direct_temporal_edge_rows": [
            {"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.2, "stability": 0.6},
            {"source": "testing_prevention_reach", "target": "care_access_continuity", "lag": 1, "weight": 0.3, "stability": 0.8},
        ]
    }

    rows = batch._edge_signature_rows(payload)

    assert rows == [
        {"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.2, "stability": 0.6},
        {"source": "testing_prevention_reach", "target": "care_access_continuity", "lag": 1, "weight": 0.3, "stability": 0.8},
    ]


def test_score_variant_flags_testing_restore_and_overlap() -> None:
    report = {
        "structural_monthly_row_count": 645,
        "phase1_summary": {"canonical_count": 27},
        "rebuilt_phase2_summary": {"block_count": 4, "direct_temporal_edge_count": 1},
    }
    phase2_payload = {
        "direct_temporal_edge_rows": [
            {"source": "testing_prevention_reach", "target": "care_access_continuity", "lag": 1, "weight": 0.25, "stability": 0.8}
        ]
    }

    row = batch._score_variant(
        label="exclude_x",
        excluded_canonicals=("x",),
        report=report,
        phase2_payload=phase2_payload,
        baseline_edge_set={("testing_engagement", "care_access_continuity", 1)},
        merged_all_edge_set={("care_access_continuity", "suppression_capacity", 1)},
    )

    assert row["testing_edge_present"] is True
    assert row["restores_baseline_testing_edge"] is True
    assert row["baseline_overlap_count"] == 0
    assert row["merged_all_overlap_count"] == 0


def test_run_batch_writes_summary_with_variant_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    baseline_run = tmp_path / "artifacts" / "runs" / "baseline"
    (baseline_run / "analysis").mkdir(parents=True)
    (baseline_run / "phase2").mkdir(parents=True)
    write_json(
        baseline_run / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json",
        {
            "structural_monthly_row_count": 697,
            "phase1_summary": {"canonical_count": 22},
            "rebuilt_phase2_summary": {"block_count": 4, "direct_temporal_edge_count": 2},
        },
    )
    write_json(
        baseline_run / "phase2" / "phase2_structural_payload.json",
        {
            "direct_temporal_edge_rows": [
                {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.2, "stability": 0.9},
                {"source": "testing_engagement", "target": "suppression_capacity", "lag": 1, "weight": 0.3, "stability": 0.7},
            ]
        },
    )

    def fake_run(*, run_id: str, **_: object) -> dict[str, object]:
        run_dir = tmp_path / "artifacts" / "runs" / str(run_id)
        (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2").mkdir(parents=True, exist_ok=True)
        if str(run_id).endswith("merged_all"):
            edge_rows = [{"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.2, "stability": 0.6}]
        else:
            edge_rows = [{"source": "testing_prevention_reach", "target": "care_access_continuity", "lag": 1, "weight": 0.25, "stability": 0.8}]
        write_json(
            run_dir / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json",
            {
                "structural_monthly_row_count": 645,
                "phase1_summary": {"canonical_count": 27},
                "rebuilt_phase2_summary": {"block_count": 4, "direct_temporal_edge_count": len(edge_rows)},
            },
        )
        write_json(run_dir / "phase2" / "phase2_structural_payload.json", {"direct_temporal_edge_rows": edge_rows})
        return {}

    monkeypatch.setattr(batch.monthly_lane, "run_tr_v3_monthly_phase2_lane_batch", fake_run)

    payload = batch.run_tr_v3_phase2_coverage_indicator_effect_batch(
        run_id="coverage-effect",
        baseline_run_id="baseline",
        source_run_id="source",
        coverage_run_id="coverage",
    )

    assert len(payload["summary_rows"]) == 1 + len(batch.ADDED_COVERAGE_CANONICALS)
    assert payload["summary_rows"][0]["variant"] == "merged_all"
    assert (tmp_path / "artifacts" / "runs" / "coverage-effect" / "analysis" / "tr_v3_phase2_coverage_indicator_effect_batch_report.json").exists()
