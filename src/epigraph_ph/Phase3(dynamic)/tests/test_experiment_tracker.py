from __future__ import annotations

import json

from phase3_dynamic.experiment_tracker import build_experiment_registry, write_experiment_registry


def test_experiment_tracker_extracts_artifact_run_and_declared_experiment(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    analysis = root / "artifacts" / "runs" / "tr-v3-test-sandbox" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "tr_v3_test_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-10T12:00:00+00:00",
                "run_id": "tr-v3-test-sandbox",
                "family_name": "TR-V3-TEST",
                "source_run_id": "source",
                "loop_variant": "evidence-to-model-loop",
                "decision": "keep",
                "decision_reason": "test gate passed",
                "model_contract": {
                    "schema_version": "phase3_model_contract.v1",
                    "model_kind": "mechanistic_transition_forecast",
                    "primary_output_contract": "blocked_time_cascade_forecast",
                },
                "hazard_semantics": {
                    "schema_version": "phase3_hazard_semantics.v1",
                    "diagnostic_derived_hazard": {"status": "not_emitted_by_phase3_dynamic_hazard_model"},
                },
                "benchmark_contract": {"start_year": 2017, "end_year": 2025},
                "data_provenance": {"overall_observation_rows": {"row_count": 1}},
                "observation_score_ledger_summary": {
                    "schema_version": "phase3_dynamic_observation_score_ledger.v1",
                    "entry_count": 3,
                    "candidate_scored_entry_count": 3,
                    "carry_forward_scored_entry_count": 3,
                    "metrics": {},
                },
                "candidate_count": 1,
                "split_count": 2,
                "best_candidate": {
                    "config": {"alpha": 1},
                    "score": {
                        "candidate_mean_mae": 0.1,
                        "candidate_worst_mae": 0.2,
                        "carry_forward_mean_mae": 0.3,
                        "carry_forward_worst_mae": 0.4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (analysis / "tr_v3_test_report.md").write_text("# test\n", encoding="utf-8")
    (root / "AUTORESEARCH.md").write_text(
        "### EXP-X1\n\nObjective:\n- tracked objective\n\nCurrent implementation status:\n- implemented\n",
        encoding="utf-8",
    )

    registry = build_experiment_registry(root)

    assert registry["artifact_runs"][0]["run_id"] == "tr-v3-test-sandbox"
    assert registry["artifact_runs"][0]["score"]["candidate_mean_mae"] == 0.1
    assert registry["declared_experiments"][0]["experiment_id"] == "EXP-X1"
    assert registry["champion_by_kept_mean_mae"]["run_id"] == "tr-v3-test-sandbox"
    assert registry["champion_by_claim_aware_promotion"]["run_id"] == "tr-v3-test-sandbox"
    assert registry["artifact_runs"][0]["claim_promotion_review"]["promotion_eligible"] is True


def test_missing_observation_score_ledger_quarantines_run(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    analysis = root / "artifacts" / "runs" / "tr-v3-no-ledger-sandbox" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "tr_v3_no_ledger_report.json").write_text(
        json.dumps(
            {
                "run_id": "tr-v3-no-ledger-sandbox",
                "decision": "keep",
                "model_contract": {
                    "schema_version": "phase3_model_contract.v1",
                    "model_kind": "mechanistic_transition_forecast",
                },
                "hazard_semantics": {
                    "schema_version": "phase3_hazard_semantics.v1",
                    "diagnostic_derived_hazard": {"status": "not_emitted_by_phase3_dynamic_hazard_model"},
                },
                "benchmark_contract": {"start_year": 2017, "end_year": 2025},
                "data_provenance": {"overall_observation_rows": {"row_count": 1}},
                "best_candidate": {
                    "score": {
                        "candidate_mean_mae": 0.05,
                        "candidate_worst_mae": 0.08,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    registry = build_experiment_registry(root)

    assert registry["champion_by_kept_mean_mae"]["run_id"] == "tr-v3-no-ledger-sandbox"
    assert registry["champion_by_claim_aware_promotion"] is None
    assert "missing_observation_score_ledger_summary" in registry["artifact_runs"][0]["claim_promotion_review"]["blockers"]


def test_carry_forward_failure_quarantines_claim_promotion(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    analysis = root / "artifacts" / "runs" / "tr-v3-loses-to-cf-sandbox" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "tr_v3_loses_to_cf_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-24T19:43:42+00:00",
                "run_id": "tr-v3-loses-to-cf-sandbox",
                "family_name": "TR-V3-TEST",
                "source_run_id": "source",
                "loop_variant": "evidence-to-model-loop",
                "decision": "keep",
                "decision_reason": "family-local gate passed",
                "model_contract": {
                    "schema_version": "phase3_model_contract.v1",
                    "model_kind": "mechanistic_transition_forecast",
                    "primary_output_contract": "blocked_time_cascade_forecast",
                },
                "hazard_semantics": {
                    "schema_version": "phase3_hazard_semantics.v1",
                    "diagnostic_derived_hazard": {"status": "not_emitted_by_phase3_dynamic_hazard_model"},
                },
                "benchmark_contract": {"start_year": 2010, "end_year": 2025},
                "data_provenance": {"overall_observation_rows": {"row_count": 10}},
                "observation_score_ledger_summary": {
                    "schema_version": "phase3_dynamic_observation_score_ledger.v1",
                    "entry_count": 10,
                    "candidate_scored_entry_count": 10,
                    "carry_forward_scored_entry_count": 10,
                    "metrics": {},
                },
                "best_candidate": {
                    "config": {"alpha": 1},
                    "score": {
                        "candidate_mean_mae": 9.0,
                        "candidate_worst_mae": 12.0,
                        "carry_forward_mean_mae": 1.0,
                        "carry_forward_worst_mae": 2.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    registry = build_experiment_registry(root)

    assert registry["champion_by_kept_mean_mae"]["run_id"] == "tr-v3-loses-to-cf-sandbox"
    assert registry["champion_by_claim_aware_promotion"] is None
    blockers = registry["artifact_runs"][0]["claim_promotion_review"]["blockers"]
    assert "fails_carry_forward_mean_gate" in blockers
    assert "fails_carry_forward_worst_gate" in blockers


def test_write_experiment_registry_emits_json_and_markdown(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    (root / "artifacts" / "runs").mkdir(parents=True)

    registry = write_experiment_registry(sandbox_root=root)

    assert registry["schema_version"] == "phase3_dynamic_experiment_registry.v2"
    assert (root / "artifacts" / "experiment_tracking" / "experiment_registry.json").exists()
    assert (root / "artifacts" / "experiment_tracking" / "experiment_registry.md").exists()


def test_missing_model_contract_quarantines_legacy_runs(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    analysis = root / "artifacts" / "runs" / "tr-v3-legacy-sandbox" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "tr_v3_legacy_report.json").write_text(
        json.dumps(
            {
                "run_id": "tr-v3-legacy-sandbox",
                "decision": "keep",
                "best_candidate": {
                    "score": {
                        "candidate_mean_mae": 0.01,
                        "candidate_worst_mae": 0.02,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    registry = build_experiment_registry(root)

    assert registry["champion_by_kept_mean_mae"]["run_id"] == "tr-v3-legacy-sandbox"
    assert registry["champion_by_claim_aware_promotion"] is None
    assert "missing_model_contract" in registry["artifact_runs"][0]["claim_promotion_review"]["blockers"]


def test_contract_only_stage_is_not_legacy_or_claim_champion(tmp_path) -> None:
    root = tmp_path / "Phase3(dynamic)"
    analysis = root / "artifacts" / "runs" / "rev-00-contract" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "rev_00_report.json").write_text(
        json.dumps(
            {
                "run_id": "rev-00-contract",
                "decision": "contract_only",
                "benchmark_contract": {"status": "contract_only_stage"},
                "data_provenance": {"observation_role_ledger_summary": {"row_count": 1}},
                "observation_score_ledger_summary": {
                    "schema_version": "phase3_dynamic_observation_score_ledger.v1",
                    "entry_count": 0,
                    "candidate_scored_entry_count": 0,
                    "carry_forward_scored_entry_count": 0,
                    "metrics": {},
                },
                "claim_card": {
                    "schema_version": "phase3_dynamic_claim_card.v1",
                    "promotion_eligible": False,
                    "blockers": ["non_model_or_diagnostic_stage"],
                },
                "best_candidate": {
                    "score": {
                        "candidate_mean_mae": 0.0,
                        "candidate_worst_mae": 0.0,
                        "carry_forward_mean_mae": 0.0,
                        "carry_forward_worst_mae": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    registry = build_experiment_registry(root)

    assert registry["champion_by_kept_mean_mae"] is None
    assert registry["champion_by_claim_aware_promotion"] is None
    blockers = registry["artifact_runs"][0]["claim_promotion_review"]["blockers"]
    assert blockers == ["non_model_or_diagnostic_stage"]
