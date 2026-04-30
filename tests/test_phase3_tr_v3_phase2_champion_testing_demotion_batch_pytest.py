from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_phase2_champion_testing_demotion_batch as batch
from epigraph_ph.runtime import write_json


def test_run_batch_writes_demotion_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.monthly_lane, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.champion_equivalence, "ROOT_DIR", tmp_path)

    candidate_analysis_dir = tmp_path / "artifacts" / "runs" / "candidate-merged" / "analysis"
    candidate_analysis_dir.mkdir(parents=True)
    write_json(
        candidate_analysis_dir / "tr_v3_monthly_phase2_lane_batch_report.json",
        {
            "source_run_id": "source-run",
            "coverage_run_id": "coverage-run",
            "start_month": "2010-01",
        },
    )

    monthly_calls: list[dict[str, object]] = []

    def _fake_monthly(**kwargs):
        monthly_calls.append(dict(kwargs))
        analysis_dir = tmp_path / "artifacts" / "runs" / str(kwargs["run_id"]) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            analysis_dir / "tr_v3_monthly_phase2_lane_batch_report.json",
            {
                "source_run_id": kwargs["source_run_id"],
                "coverage_run_id": kwargs["coverage_run_id"],
                "start_month": kwargs["start_month"],
            },
        )
        return {"run_id": kwargs["run_id"]}

    monkeypatch.setattr(batch.monthly_lane, "run_tr_v3_monthly_phase2_lane_batch", _fake_monthly)
    monkeypatch.setattr(
        batch.champion_equivalence,
        "run_tr_v3_phase2_champion_equivalence_batch",
        lambda **kwargs: {
            "family_overlap_rows": [
                {"block_family": "testing_family", "candidate_indicator_count": 0, "indicator_jaccard": 0.0},
                {"block_family": "care_access_continuity", "candidate_indicator_count": 2, "indicator_jaccard": 1.0},
                {"block_family": "suppression_capacity", "candidate_indicator_count": 1, "indicator_jaccard": 1.0},
                {"block_family": "mobility_exposure_pressure", "candidate_indicator_count": 1, "indicator_jaccard": 1.0},
            ],
            "testing_scenario_rows": [
                {"candidate_abs_delta": 0.0},
                {"candidate_abs_delta": 0.0},
            ],
        },
    )
    monkeypatch.setattr(
        batch.champion_equivalence,
        "_seeded_report",
        lambda run_id: {
            "terminal_delta_rows": [
                {"scenario": "disruption_recovery", "diagnosed_plhiv_delta": 5.0, "alive_on_art_delta": 1.0, "new_diagnosed_cases_period_delta": 0.0},
                {"scenario": "mobility_spike", "diagnosed_plhiv_delta": 2.0, "alive_on_art_delta": 0.0, "new_diagnosed_cases_period_delta": 0.0},
            ]
        },
    )

    payload = batch.run_tr_v3_phase2_champion_testing_demotion_batch(
        run_id="demotion",
        baseline_monthly_run_id="baseline-old",
        candidate_monthly_run_id="candidate-merged",
    )

    assert monthly_calls
    assert tuple(monthly_calls[0]["structural_excluded_canonicals"]) == batch.TESTING_FAMILY_CANONICALS
    assert payload["decision"] == "keep_testing_as_measurement_sidecar"
    assert (tmp_path / "artifacts" / "runs" / "demotion" / "analysis" / "tr_v3_phase2_champion_testing_demotion_batch_report.json").exists()
