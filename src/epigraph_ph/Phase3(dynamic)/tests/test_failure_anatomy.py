from __future__ import annotations

import phase3_dynamic.failure_anatomy as failure_anatomy
from phase3_dynamic.failure_anatomy import _config_from_report, _flatten_scored_entries, markdown_failure_anatomy_report, summarize_scored_entries
from phase3_dynamic.model import HiddenDriverConfig, IncidenceFlowConfig


def test_flatten_scored_entries_and_summary_capture_metric_and_year_deltas() -> None:
    rows = [
        {
            "scoring_details": {
                "metric_scales": {
                    "diagnosed_plhiv": 100.0,
                    "alive_on_art": 80.0,
                    "new_diagnosed_cases_period": 20.0,
                },
                "holdout_rows": [
                    {
                        "quarter": "2024-Q1",
                        "diagnosed_plhiv": 100.0,
                        "alive_on_art": 80.0,
                        "new_diagnosed_cases_period": 10.0,
                        "metric_provenance": {
                            "diagnosed_plhiv": {"tier": "exact_observed", "source_id": "diag-src"},
                            "alive_on_art": {"tier": "bridge_observed", "source_id": "art-src"},
                            "new_diagnosed_cases_period": {"tier": "exact_observed", "source_id": "flow-src"},
                        },
                    }
                ],
                "candidate_prediction_rows": [
                    {
                        "quarter": "2024-Q1",
                        "diagnosed_plhiv": 150.0,
                        "alive_on_art": 120.0,
                        "new_diagnosed_cases_period": 30.0,
                    }
                ],
                "carry_forward_prediction_rows": [
                    {
                        "quarter": "2024-Q1",
                        "diagnosed_plhiv": 105.0,
                        "alive_on_art": 82.0,
                        "new_diagnosed_cases_period": 12.0,
                    }
                ],
            }
        }
    ]

    entries = _flatten_scored_entries(rows)
    summary = summarize_scored_entries(entries)

    assert len(entries) == 3
    diagnosed_entry = next(entry for entry in entries if entry["metric_name"] == "diagnosed_plhiv")
    assert diagnosed_entry["candidate_norm_error"] == 0.5
    assert diagnosed_entry["carry_forward_norm_error"] == 0.05
    assert summary["metric_rows"][0]["entry_count"] == 1
    assert summary["year_rows"][0]["year"] == 2024
    assert summary["worst_rows"][0]["metric_name"] == "diagnosed_plhiv"


def test_config_from_report_preserves_hidden_transition_weights() -> None:
    spec = _config_from_report(
        {
            "dynamic_cfg": {"ridge_penalty": 0.01, "rho_clip": 0.8, "trend_scale": 1.0},
            "incidence_cfg": {"ridge_penalty": 0.01, "rho_clip": 0.8, "trend_scale": 1.0},
            "hidden_cfg": {
                "precision_scale": 0.25,
                "residual_ridge": 0.01,
                "max_effect": 0.1,
                "rank_cap": 1,
                "transition_weights": {"U_to_D": 1.0, "A_to_V": 0.0},
            },
        }
    )

    assert isinstance(spec["hidden_cfg"], HiddenDriverConfig)
    assert isinstance(spec["incidence_cfg"], IncidenceFlowConfig)
    assert spec["hidden_cfg"].transition_weights == {"U_to_D": 1.0, "A_to_V": 0.0}


def test_markdown_failure_anatomy_report_renders_problem_year_section() -> None:
    report = markdown_failure_anatomy_report(
        {
            "generated_at": "2026-04-24T00:00:00+00:00",
            "reports": [
                {
                    "family_name": "TR-V3-04d",
                    "run_id": "run-1",
                    "report_path": "/tmp/report.json",
                    "source_run_id": "source-1",
                    "decision": "keep",
                    "decision_reason": "Gate passed.",
                    "score_summary": {"candidate_mean_mae": 1.0, "carry_forward_mean_mae": 0.1, "candidate_worst_mae": 2.0, "carry_forward_worst_mae": 0.2},
                    "metric_error_summary": [],
                    "year_error_summary": [],
                    "worst_quarter_deltas": [],
                    "problem_year_replays": [
                        {
                            "holdout_year": 2022,
                            "raw_vs_calibrated": {
                                "alive_on_art": {
                                    "observed_mean": 100.0,
                                    "raw_mean": 90.0,
                                    "calibrated_mean": 5000.0,
                                    "raw_mae": 10.0,
                                    "calibrated_mae": 4900.0,
                                }
                            },
                            "train_mass_diagnostics": {
                                "first_train_state_total": 19000.0,
                                "last_train_state_total": 142000.0,
                                "raw_train_last_state_total": 19000.0,
                                "train_mass_gap": 123000.0,
                                "total_inferred_inflow": 180000.0,
                                "total_inferred_attrition": 57000.0,
                            },
                            "observation_coefficients": {
                                "alive_on_art": {"raw_scale": 87.5, "time_slope": 998.6}
                            },
                            "incidence_diagnostics": {
                                "incidence_inflow": {"train_mean": 1000.0, "forecast_mean": 1100.0}
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert "Problem-Year Replays" in report
    assert "Holdout 2022" in report
    assert "`alive_on_art.raw_scale`" in report
    assert "Total inferred inflow" in report
    assert "`incidence_inflow`" in report


def test_failure_replay_uses_phase2_source_from_report_context(monkeypatch, tmp_path) -> None:
    observed_phase2_sources: list[str] = []

    class FakeDataset:
        holdout_rows = [{"quarter": "2024-Q1"}]

    def fake_load_phase2(_root, source_run_id):
        observed_phase2_sources.append(str(source_run_id))
        return object()

    monkeypatch.setattr(failure_anatomy, "build_observation_rows", lambda *_args, **_kwargs: [{"quarter": "2023-Q1"}])
    monkeypatch.setattr(failure_anatomy, "build_blocked_time_dataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(failure_anatomy, "load_phase2_structural_inputs", fake_load_phase2)
    monkeypatch.setattr(failure_anatomy, "build_direct_prior_features", lambda *_args, **_kwargs: {"U_to_D": []})
    monkeypatch.setattr(
        failure_anatomy,
        "forecast_dynamic_baseline",
        lambda *_args, **_kwargs: {
            "mae": 0.0,
            "smape": 0.0,
            "raw_prediction_rows": [],
            "prediction_rows": [],
            "observation_model": {},
            "incidence_paths": None,
        },
    )
    monkeypatch.setattr(failure_anatomy, "_train_mass_diagnostics", lambda *_args, **_kwargs: {})

    rows = failure_anatomy._replay_top_problem_years(
        epigraph_root=tmp_path,
        source_run_id="expanded-harp-source",
        phase2_source_run_id="phase2-structure-source",
        report_config={
            "dynamic_cfg": {"ridge_penalty": 0.01, "rho_clip": 0.8, "trend_scale": 1.0},
            "prior_cfg": {"effect_scale": 1.0, "precision_scale": 1.0, "residual_ridge": 0.1, "max_effect": 0.2},
        },
        holdout_years=[2024],
    )

    assert rows[0]["holdout_year"] == 2024
    assert observed_phase2_sources == ["phase2-structure-source"]
