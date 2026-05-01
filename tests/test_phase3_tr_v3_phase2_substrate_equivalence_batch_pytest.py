from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_phase2_substrate_equivalence_batch as batch
from epigraph_ph.runtime import write_json


def test_family_name_maps_testing_blocks_together() -> None:
    assert batch._family_name("testing_engagement") == "testing_family"
    assert batch._family_name("testing_prevention_reach") == "testing_family"
    assert batch._family_name("care_access_continuity") == "care_access_continuity"


def test_testing_equivalence_rows_aligns_baseline_and_candidate() -> None:
    baseline_rows = [
        {"block_id": "testing_engagement", "canonical_name": "diagnosed_plhiv", "loading": 0.4, "direct_indicator_count": 3, "measurement_time_mix": "monthly_only"},
        {"block_id": "testing_engagement", "canonical_name": "new_diagnosed_cases_period", "loading": 0.3, "direct_indicator_count": 3, "measurement_time_mix": "monthly_only"},
    ]
    candidate_rows = [
        {"block_id": "testing_prevention_reach", "canonical_name": "hiv_test_positivity_percent", "loading": 0.2, "direct_indicator_count": 1, "measurement_time_mix": "annual_only"},
        {"block_id": "testing_prevention_reach", "canonical_name": "new_diagnosed_cases_period", "loading": 0.1, "direct_indicator_count": 0, "measurement_time_mix": "none"},
    ]

    rows = batch._testing_equivalence_rows(baseline_rows, candidate_rows)

    keyed = {row["canonical_name"]: row for row in rows}
    assert keyed["diagnosed_plhiv"]["baseline_abs_loading"] == 0.4
    assert keyed["diagnosed_plhiv"]["candidate_abs_loading"] == 0.0
    assert keyed["hiv_test_positivity_percent"]["baseline_abs_loading"] == 0.0
    assert keyed["hiv_test_positivity_percent"]["candidate_abs_loading"] == 0.2
    assert keyed["new_diagnosed_cases_period"]["baseline_direct_count"] == 3
    assert keyed["new_diagnosed_cases_period"]["candidate_direct_count"] == 0


def test_run_batch_writes_equivalence_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.loading_sanity, "ROOT_DIR", tmp_path)
    base_loading_run = tmp_path / "artifacts" / "runs" / "substrate-baseline-loading" / "analysis"
    candidate_loading_run = tmp_path / "artifacts" / "runs" / "substrate-candidate-loading" / "analysis"
    base_loading_run.mkdir(parents=True)
    candidate_loading_run.mkdir(parents=True)
    write_json(
        base_loading_run / "tr_v3_monthly_loading_sanity_batch_report.json",
        {
            "audit_rows": [
                {"block_id": "testing_engagement", "canonical_name": "diagnosed_plhiv", "loading": 0.4, "direct_indicator_count": 3, "measurement_time_mix": "monthly_only"},
                {"block_id": "care_access_continuity", "canonical_name": "alive_on_art", "loading": 0.5, "direct_indicator_count": 2, "measurement_time_mix": "monthly_only"},
            ],
            "block_summary_rows": [
                {"block_id": "testing_engagement", "indicator_count": 1, "total_loading_mass": 0.4, "champion_loading_share": 1.0, "cascade_loading_share": 0.0, "singleton_loading_share": 0.0, "annual_only_loading_share": 0.0, "mean_ppc_corr": 0.8, "max_risk_score": 4},
                {"block_id": "care_access_continuity", "indicator_count": 1, "total_loading_mass": 0.5, "champion_loading_share": 1.0, "cascade_loading_share": 0.0, "singleton_loading_share": 0.0, "annual_only_loading_share": 0.0, "mean_ppc_corr": 0.9, "max_risk_score": 3},
            ],
        },
    )
    write_json(
        candidate_loading_run / "tr_v3_monthly_loading_sanity_batch_report.json",
        {
            "audit_rows": [
                {"block_id": "testing_prevention_reach", "canonical_name": "hiv_test_positivity_percent", "loading": 0.2, "direct_indicator_count": 1, "measurement_time_mix": "annual_only"},
                {"block_id": "care_access_continuity", "canonical_name": "alive_on_art", "loading": 0.45, "direct_indicator_count": 2, "measurement_time_mix": "monthly_only"},
            ],
            "block_summary_rows": [
                {"block_id": "testing_prevention_reach", "indicator_count": 1, "total_loading_mass": 0.2, "champion_loading_share": 0.0, "cascade_loading_share": 0.0, "singleton_loading_share": 1.0, "annual_only_loading_share": 1.0, "mean_ppc_corr": 0.6, "max_risk_score": 2},
                {"block_id": "care_access_continuity", "indicator_count": 1, "total_loading_mass": 0.45, "champion_loading_share": 1.0, "cascade_loading_share": 0.0, "singleton_loading_share": 0.0, "annual_only_loading_share": 0.0, "mean_ppc_corr": 0.88, "max_risk_score": 3},
            ],
        },
    )

    payload = batch.run_tr_v3_phase2_substrate_equivalence_batch(
        run_id="substrate",
        baseline_monthly_run_id="baseline",
        candidate_monthly_run_id="candidate",
    )

    families = {row["block_family"] for row in payload["family_overlap_rows"]}
    assert "testing_family" in families
    assert "care_access_continuity" in families
    assert (tmp_path / "artifacts" / "runs" / "substrate" / "analysis" / "tr_v3_phase2_substrate_equivalence_batch_report.json").exists()
