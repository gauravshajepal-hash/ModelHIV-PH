from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_phase2_champion_equivalence_batch as batch
from epigraph_ph.runtime import write_json


def test_current_winner_configs_uses_predictive_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        batch.suite,
        "default_predictive_candidate_id",
        lambda contract_name: "EXACT-ID" if contract_name == "exact_only" else "DENSE-ID",
    )

    configs = batch._current_winner_configs()

    assert configs["exact_only"]["winner_id"] == "EXACT-ID"
    assert configs["purged_dense"]["winner_id"] == "DENSE-ID"
    assert configs["purged_dense"]["suite_contract"] == "purged_dense"


def test_testing_readout_rows_extracts_current_testing_features() -> None:
    baseline_report = {
        "contracts": {
            "exact_only": {
                "readout": {
                    "feature_names": ["testing_engagement", "testing_engagement_delta", "care_access_continuity"],
                    "metrics": {
                        "diagnosed_plhiv": {
                            "beta": [2.0, -1.0, 0.5],
                            "scale": 0.1,
                        }
                    },
                }
            }
        }
    }
    candidate_report = {
        "contracts": {
            "exact_only": {
                "readout": {
                    "feature_names": ["testing_prevention_reach", "testing_prevention_reach_delta", "care_access_continuity"],
                    "metrics": {
                        "diagnosed_plhiv": {
                            "beta": [1.5, 0.25, 0.5],
                            "scale": 0.2,
                        }
                    },
                }
            }
        }
    }

    rows = batch._testing_readout_rows(baseline_report, candidate_report)
    keyed = {(row["metric"], row["feature_kind"]): row for row in rows if row["contract"] == "exact_only"}

    assert keyed[("diagnosed_plhiv", "level")]["baseline_feature_name"] == "testing_engagement"
    assert keyed[("diagnosed_plhiv", "level")]["candidate_feature_name"] == "testing_prevention_reach"
    assert keyed[("diagnosed_plhiv", "level")]["baseline_scaled_coeff"] == 0.2
    assert keyed[("diagnosed_plhiv", "level")]["candidate_scaled_coeff"] == 0.30000000000000004
    assert keyed[("diagnosed_plhiv", "delta")]["baseline_scaled_coeff"] == -0.1
    assert keyed[("diagnosed_plhiv", "delta")]["candidate_scaled_coeff"] == 0.05


def test_run_batch_writes_champion_equivalence_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.substrate, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.seeded, "ROOT_DIR", tmp_path)

    for run_name, payload in {
        "champion-baseline-loading": {
            "audit_rows": [
                {"block_id": "testing_engagement", "canonical_name": "diagnosed_plhiv", "loading": 0.4, "direct_indicator_count": 3, "measurement_time_mix": "monthly_only"},
            ],
            "block_summary_rows": [
                {"block_id": "testing_engagement", "indicator_count": 1, "total_loading_mass": 0.4, "champion_loading_share": 1.0, "cascade_loading_share": 0.0, "singleton_loading_share": 0.0, "annual_only_loading_share": 0.0, "mean_ppc_corr": 0.8, "max_risk_score": 4},
            ],
        },
        "champion-candidate-loading": {
            "audit_rows": [
                {"block_id": "testing_prevention_reach", "canonical_name": "hiv_test_positivity_percent", "loading": 0.25, "direct_indicator_count": 1, "measurement_time_mix": "annual_only"},
            ],
            "block_summary_rows": [
                {"block_id": "testing_prevention_reach", "indicator_count": 1, "total_loading_mass": 0.25, "champion_loading_share": 0.0, "cascade_loading_share": 0.0, "singleton_loading_share": 1.0, "annual_only_loading_share": 1.0, "mean_ppc_corr": 0.6, "max_risk_score": 2},
            ],
        },
    }.items():
        analysis_dir = tmp_path / "artifacts" / "runs" / run_name / "analysis"
        analysis_dir.mkdir(parents=True)
        write_json(analysis_dir / "tr_v3_monthly_loading_sanity_batch_report.json", payload)

    for run_name, feature_name in {
        "champion-baseline-seeded": "testing_engagement",
        "champion-candidate-seeded": "testing_prevention_reach",
    }.items():
        analysis_dir = tmp_path / "artifacts" / "runs" / run_name / "analysis"
        analysis_dir.mkdir(parents=True)
        write_json(
            analysis_dir / "tr_v3_phase2_seeded_champion_batch_report.json",
            {
                "contracts": {
                    "exact_only": {
                        "readout": {
                            "feature_names": [feature_name, f"{feature_name}_delta"],
                            "metrics": {
                                "diagnosed_plhiv": {"beta": [1.0, -1.0], "scale": 0.2},
                                "alive_on_art": {"beta": [0.5, 0.1], "scale": 0.2},
                                "new_diagnosed_cases_period": {"beta": [0.25, 0.2], "scale": 0.2},
                            },
                        }
                    }
                },
                "terminal_delta_rows": [
                    {
                        "contract": "exact_only",
                        "scenario": "testing_pulse",
                        "diagnosed_plhiv_delta": 10.0,
                        "alive_on_art_delta": 5.0,
                        "new_diagnosed_cases_period_delta": 2.0,
                    },
                    {
                        "contract": "exact_only",
                        "scenario": "testing_plateau",
                        "diagnosed_plhiv_delta": 12.0,
                        "alive_on_art_delta": 4.0,
                        "new_diagnosed_cases_period_delta": 1.0,
                    },
                ],
            },
        )

    payload = batch.run_tr_v3_phase2_champion_equivalence_batch(
        run_id="champion",
        baseline_monthly_run_id="baseline",
        candidate_monthly_run_id="candidate",
    )

    assert payload["current_champions"]["exact_only"] == batch.suite.default_predictive_candidate_id("exact_only")
    assert payload["testing_equivalence_rows"]
    assert payload["testing_readout_rows"]
    assert payload["testing_scenario_rows"]
    assert (tmp_path / "artifacts" / "runs" / "champion" / "analysis" / "tr_v3_phase2_champion_equivalence_batch_report.json").exists()
