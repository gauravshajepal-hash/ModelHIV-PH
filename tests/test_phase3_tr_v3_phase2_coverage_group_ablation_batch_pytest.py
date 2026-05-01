from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_phase2_coverage_group_ablation_batch as batch
from epigraph_ph.runtime import write_json


def test_group_constants_cover_expected_added_canonicals() -> None:
    assert set(batch.LOAD_BEARING_TRIO) <= set(batch.ALL_ADDED_COVERAGE_CANONICALS)
    assert len(batch.LOAD_BEARING_TRIO) == 3


def test_run_batch_writes_group_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.effect, "ROOT_DIR", tmp_path)
    baseline_run = tmp_path / "artifacts" / "runs" / "baseline"
    (baseline_run / "phase2").mkdir(parents=True)
    write_json(
        baseline_run / "phase2" / "phase2_structural_payload.json",
        {
            "direct_temporal_edge_rows": [
                {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.2, "stability": 0.9},
                {"source": "testing_engagement", "target": "suppression_capacity", "lag": 1, "weight": 0.3, "stability": 0.7},
            ]
        },
    )

    def fake_run(*, run_id: str, structural_excluded_canonicals: tuple[str, ...], **_: object) -> dict[str, object]:
        run_dir = tmp_path / "artifacts" / "runs" / str(run_id)
        (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (run_dir / "phase2").mkdir(parents=True, exist_ok=True)
        if not structural_excluded_canonicals:
            edge_rows = [{"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.2, "stability": 0.6}]
            row_count = 720
            canonical_count = 27
        elif tuple(structural_excluded_canonicals) == batch.LOAD_BEARING_TRIO:
            edge_rows = []
            row_count = 707
            canonical_count = 24
        else:
            edge_rows = []
            row_count = 697
            canonical_count = 22
        write_json(
            run_dir / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json",
            {
                "structural_monthly_row_count": row_count,
                "phase1_summary": {"canonical_count": canonical_count},
                "rebuilt_phase2_summary": {"block_count": 4, "direct_temporal_edge_count": len(edge_rows)},
            },
        )
        write_json(run_dir / "phase2" / "phase2_structural_payload.json", {"direct_temporal_edge_rows": edge_rows})
        return {}

    monkeypatch.setattr(batch.monthly_lane, "run_tr_v3_monthly_phase2_lane_batch", fake_run)

    payload = batch.run_tr_v3_phase2_coverage_group_ablation_batch(
        run_id="coverage-group",
        baseline_run_id="baseline",
        source_run_id="source",
        coverage_run_id="coverage",
    )

    assert [row["variant"] for row in payload["summary_rows"]] == [
        "merged_all",
        "exclude_load_bearing_trio",
        "exclude_all_added_coverage",
    ]
    assert (tmp_path / "artifacts" / "runs" / "coverage-group" / "analysis" / "tr_v3_phase2_coverage_group_ablation_batch_report.json").exists()
