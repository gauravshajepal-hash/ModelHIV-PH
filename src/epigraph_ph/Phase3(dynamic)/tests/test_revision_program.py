from __future__ import annotations

import phase3_dynamic.revision_program as revision_program


def test_build_stage_payload_emits_registry_compatible_contract_fields(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        revision_program,
        "_load_reference_benchmarks",
        lambda epigraph_root, sandbox_root: {
            "frozen_exact_r10": {"candidate_mean_mae": 0.3, "candidate_worst_mae": 0.35},
            "dense_r10_readout": {"candidate_mean_mae": 0.25, "candidate_worst_mae": 0.3},
            "inflow_diagnostic": {"candidate_mean_mae": 0.5, "candidate_worst_mae": 0.6},
        },
    )

    holdout_row = {
        "quarter": "2021-Q1",
        "diagnosed_plhiv": 100.0,
        "alive_on_art": 80.0,
        "new_diagnosed_cases_period": 10.0,
        "metric_provenance": {
            "diagnosed_plhiv": {
                "tier": "exact_observed",
                "aggregation_mode": "quarterly_observed",
                "source_id": "diagnosed",
                "source_quality_tier": "official_local_corpus",
                "measurement_class": "program_observed",
                "series_kind": "quarterly_snapshot",
                "support_partition": "common_support",
                "observation_role": "direct_target",
                "row_hash": "hash-diagnosed",
            },
            "alive_on_art": {
                "tier": "exact_observed",
                "aggregation_mode": "quarterly_observed",
                "source_id": "art",
                "source_quality_tier": "official_local_corpus",
                "measurement_class": "program_observed",
                "series_kind": "quarterly_snapshot",
                "support_partition": "common_support",
                "observation_role": "direct_target",
                "row_hash": "hash-art",
            },
            "new_diagnosed_cases_period": {
                "tier": "bridge_observed",
                "aggregation_mode": "monthly_to_quarter_sum",
                "source_id": "flow",
                "source_quality_tier": "official_local_corpus",
                "measurement_class": "program_observed",
                "series_kind": "monthly_series",
                "support_partition": "expanded_support",
                "observation_role": "direct_target",
                "row_hash": "hash-flow",
            },
        },
        "row_provenance_tier": "bridge_observed",
    }
    split_row = {
        "train_end_year": 2020,
        "holdout_years": [2021],
        "carry_forward": {"mae": 0.4, "smape": 0.4},
        "candidate": {"mae": 0.2, "smape": 0.2},
        "dataset_provenance": {
            "observation_rows": {
                "row_count": 1,
                "row_tier_counts": {
                    "exact_observed": 0,
                    "bridge_observed": 1,
                    "rule_based_extrapolated": 0,
                    "latent_imputed": 0,
                    "rejected_or_quarantined": 0,
                },
            }
        },
        "scoring_details": {
            "metric_scales": {
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 80.0,
                "new_diagnosed_cases_period": 10.0,
            },
            "eps": 1e-6,
            "holdout_rows": [holdout_row],
            "candidate_prediction_rows": [
                {
                    "quarter": "2021-Q1",
                    "diagnosed_plhiv": 98.0,
                    "alive_on_art": 79.0,
                    "new_diagnosed_cases_period": 9.0,
                }
            ],
            "carry_forward_prediction_rows": [
                {
                    "quarter": "2021-Q1",
                    "diagnosed_plhiv": 95.0,
                    "alive_on_art": 75.0,
                    "new_diagnosed_cases_period": 8.0,
                }
            ],
        },
    }
    spec = {
        "dynamic_cfg": revision_program.DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        "observation_cfg": revision_program.ObservationModelConfig(),
    }
    best_candidate = {
        "config": revision_program._spec_payload(spec),
        "model_contract": revision_program._candidate_model_contract(spec),
        "hazard_semantics": revision_program._candidate_hazard_semantics(spec),
        "score": {
            "candidate_mean_mae": 0.2,
            "carry_forward_mean_mae": 0.4,
            "candidate_mean_smape": 0.2,
            "carry_forward_mean_smape": 0.4,
            "candidate_worst_mae": 0.2,
            "carry_forward_worst_mae": 0.4,
        },
        "rows": [split_row],
    }

    payload = revision_program._build_stage_payload(
        revision_stage="REV-01",
        run_id="rev-test",
        source_run_id="source-run",
        baseline_source_run_id="baseline-run",
        start_year=2020,
        end_year=2021,
        min_train_years=1,
        horizon_years=1,
        observation_rows=[holdout_row],
        observation_role_ledger={"summary": {"row_count": 1}, "rows": []},
        best_candidate=best_candidate,
        all_candidates=[best_candidate],
        module_name=None,
        epigraph_root=tmp_path,
        phase2_context=None,
    )

    assert payload["decision"] == "keep"
    assert payload["model_contract"]["schema_version"] == "phase3_model_contract.v1"
    assert payload["hazard_semantics"]["schema_version"] == "phase3_hazard_semantics.v1"
    assert payload["benchmark_contract"]["start_year"] == 2020
    assert payload["observation_score_ledger_summary"]["entry_count"] == 3
    assert payload["data_provenance"]["overall_observation_rows"]["row_count"] == 1
    assert payload["archive_drift_gate"]["status"] == "emitted"
    assert payload["lockbox_contract"]["status"] == "deferred_until_final_publication_lockbox"


def test_phase2_revision_resolver_prefers_admissible_feature_payload(monkeypatch, tmp_path) -> None:
    for run_id in ("phase2-empty", "phase2-rich"):
        phase2_dir = tmp_path / "artifacts" / "runs" / run_id / "phase2"
        phase2_dir.mkdir(parents=True)
        (phase2_dir / "phase2_structural_payload.json").write_text("{}", encoding="utf-8")

    class FakeStructuralInputs:
        def __init__(self, run_id: str) -> None:
            self.source_run_id = run_id
            self.direct_edge_rows = [1] if run_id == "phase2-rich" else []
            self.hidden_driver_rows = [1, 2] if run_id == "phase2-rich" else []

    monkeypatch.setattr(
        revision_program,
        "load_phase2_structural_inputs",
        lambda _root, run_id: FakeStructuralInputs(str(run_id)),
    )
    monkeypatch.setattr(
        revision_program,
        "build_direct_prior_features",
        lambda structural_inputs, _prior_map: {"U_to_D": [object(), object()]}
        if structural_inputs.source_run_id == "phase2-rich"
        else {"U_to_D": []},
    )
    monkeypatch.setattr(
        revision_program,
        "build_hidden_driver_features",
        lambda structural_inputs, _prior_map: {"U_to_D": [object()]}
        if structural_inputs.source_run_id == "phase2-rich"
        else {"U_to_D": []},
    )

    resolved, diagnostics = revision_program._resolve_phase2_source_for_revision(
        tmp_path,
        observation_source_run_id="expanded-observation-run",
        preferred=None,
    )

    assert resolved == "phase2-rich"
    assert diagnostics["resolution"] == "feature_aware_max_admissible_phase3_features"
    assert diagnostics["selected_direct_prior_feature_count"] == 2
    assert diagnostics["selected_hidden_driver_feature_count"] == 1
