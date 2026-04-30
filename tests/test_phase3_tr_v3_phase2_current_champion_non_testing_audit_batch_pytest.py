from __future__ import annotations

from pathlib import Path

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_current_champion_non_testing_audit_batch as batch


def test_placebo_separation_rows_respects_current_winners() -> None:
    rows = batch._placebo_separation_rows(
        [
            {"contract": "exact_only", "scenario": "disruption_recovery", "mean_mae_delta": -1.0, "mean_directional_fit_rate": 0.9},
            {"contract": "exact_only", "scenario": "disruption_recovery_placebo", "mean_mae_delta": 0.2, "mean_directional_fit_rate": 0.5},
            {"contract": "purged_dense", "scenario": "mobility_spike", "mean_mae_delta": -0.5, "mean_directional_fit_rate": 0.8},
            {"contract": "purged_dense", "scenario": "mobility_spike_placebo", "mean_mae_delta": 0.4, "mean_directional_fit_rate": 0.1},
        ],
        winner_configs=batch._current_winner_configs(),
    )

    keyed = {(row["contract"], row["scenario"]): row for row in rows}
    assert keyed[("exact_only", "disruption_recovery")]["separation_keep"] is True
    assert keyed[("purged_dense", "mobility_spike")]["separation_keep"] is True


def test_run_batch_writes_non_testing_audit_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.testing_demotion, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.champion_equivalence, "ROOT_DIR", tmp_path)

    monkeypatch.setattr(
        batch.testing_demotion,
        "run_tr_v3_phase2_champion_testing_demotion_batch",
        lambda **kwargs: {
            "decision": "keep_testing_as_measurement_sidecar",
            "demoted_monthly_run_id": "demoted-monthly",
        },
    )
    monkeypatch.setattr(
        batch.seeded,
        "_load_phase2_quarter_state",
        lambda run_id: batch.seeded.Phase2QuarterState(
            source_run_id=str(run_id),
            month_axis=["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
            quarter_axis=["2025-Q1", "2025-Q2"],
            block_axis=["care_access_continuity", "mobility_exposure_pressure"],
            quarter_states=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            edge_rows=[],
        ),
    )
    monkeypatch.setattr(batch.seeded_gate, "_load_block_loading_rows", lambda run_id: [{"block_id": "care_access_continuity", "canonical_name": "alive_on_art"}])
    monkeypatch.setattr(
        batch.seeded_gate,
        "_summarize_outcome_circularity_rows",
        lambda **kwargs: [{"block_id": "care_access_continuity", "outcome_loading_share": 0.2}],
    )
    monkeypatch.setattr(
        batch.seeded_gate,
        "_orthogonalize_phase2_state_against_outcomes",
        lambda **kwargs: (
            batch.seeded.Phase2QuarterState(
                source_run_id="demoted-monthly",
                month_axis=["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
                quarter_axis=["2025-Q1", "2025-Q2"],
                block_axis=["care_access_continuity", "mobility_exposure_pressure"],
                quarter_states=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
                edge_rows=[],
            ),
            [{"block_id": "care_access_continuity", "r_squared": 0.1}],
        ),
    )
    monkeypatch.setattr(
        batch,
        "_seeded_payload_from_phase2_state",
        lambda **kwargs: {
            "contracts": {
                "exact_only": {
                    "quarterly_mean_mae": 0.07,
                    "readout": {"feature_names": ["care_access_continuity"], "metrics": {}},
                    "base_forecast_rows": [],
                    "scenario_rows": {},
                },
                "purged_dense": {
                    "quarterly_mean_mae": 0.08,
                    "readout": {"feature_names": ["care_access_continuity"], "metrics": {}},
                    "base_forecast_rows": [],
                    "scenario_rows": {},
                },
            },
            "terminal_delta_rows": [],
        },
    )
    monkeypatch.setattr(
        batch,
        "_archive_alignment_rows_filtered",
        lambda *args, **kwargs: [
            {"contract": "exact_only", "metric": "diagnosed_plhiv", "sign_agrees": True, "delta_of_delta": 0.2, "aligned_terminal_delta": 1.0, "baseline_terminal_delta": 1.2, "scenario": "disruption_recovery"},
            {"contract": "purged_dense", "metric": "alive_on_art", "sign_agrees": True, "delta_of_delta": 0.1, "aligned_terminal_delta": 0.8, "baseline_terminal_delta": 1.0, "scenario": "mobility_spike"},
        ],
    )
    monkeypatch.setattr(
        batch.prior_audit,
        "_replay_contract_scenarios",
        lambda **kwargs: [
            {"contract": kwargs["contract_name"], "scenario": "disruption_recovery", "split_idx": 0, "mae_delta": -0.2, "directional_fit_rate": 0.9, "improved_fraction": 0.7, "terminal_max_abs_delta": 10.0},
            {"contract": kwargs["contract_name"], "scenario": "mobility_spike", "split_idx": 0, "mae_delta": -0.1, "directional_fit_rate": 0.8, "improved_fraction": 0.6, "terminal_max_abs_delta": 12.0},
            {"contract": kwargs["contract_name"], "scenario": "disruption_recovery_placebo", "split_idx": 0, "mae_delta": 0.3, "directional_fit_rate": 0.3, "improved_fraction": 0.2, "terminal_max_abs_delta": 4.0},
            {"contract": kwargs["contract_name"], "scenario": "mobility_spike_placebo", "split_idx": 0, "mae_delta": 0.2, "directional_fit_rate": 0.2, "improved_fraction": 0.2, "terminal_max_abs_delta": 3.0},
        ],
    )

    payload = batch.run_tr_v3_phase2_current_champion_non_testing_audit_batch(
        run_id="current-non-testing",
        baseline_monthly_run_id="baseline-old",
        candidate_monthly_run_id="candidate-merged",
        forecast_horizon_quarters=4,
    )

    assert payload["gate_summary"]["overall_decision"] == "keep_non_testing_sidecar"
    assert (tmp_path / "artifacts" / "runs" / "current-non-testing" / "analysis" / "tr_v3_phase2_current_champion_non_testing_audit_batch_report.json").exists()
