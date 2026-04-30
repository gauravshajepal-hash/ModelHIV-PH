from __future__ import annotations

from pathlib import Path

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_admissibility_backtest_batch as batch
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded


def test_placebo_shock_preserves_norm_for_active_two_block_kernel() -> None:
    block_axis = ["care_access_continuity", "mobility_exposure_pressure"]
    scales = np.ones(2, dtype=np.float64)

    real = batch._shock_matrix_for_replay(
        scenario_name="mobility_spike",
        steps=4,
        block_axis=block_axis,
        block_scales=scales,
    )
    placebo = batch._shock_matrix_for_replay(
        scenario_name="mobility_spike_placebo",
        steps=4,
        block_axis=block_axis,
        block_scales=scales,
    )

    assert np.isclose(np.linalg.norm(real), np.linalg.norm(placebo))
    assert np.allclose(placebo[:, 1], 0.0)
    assert np.any(np.abs(placebo[:, 0]) > 0.0)


def test_placebo_separation_marks_real_better_than_placebo() -> None:
    summary_rows = [
        {
            "contract": "exact_only",
            "scenario": "disruption_recovery",
            "mean_mae_delta": -2.0,
            "mean_directional_fit_rate": 0.8,
        },
        {
            "contract": "exact_only",
            "scenario": "disruption_recovery_placebo",
            "mean_mae_delta": -1.0,
            "mean_directional_fit_rate": 0.55,
        },
        {
            "contract": "exact_only",
            "scenario": "mobility_spike",
            "mean_mae_delta": -0.5,
            "mean_directional_fit_rate": 0.76,
        },
        {
            "contract": "exact_only",
            "scenario": "mobility_spike_placebo",
            "mean_mae_delta": 0.2,
            "mean_directional_fit_rate": 0.51,
        },
    ]

    rows = batch._placebo_separation_rows(summary_rows)

    assert len(rows) == 2
    assert all(bool(row["separation_keep"]) for row in rows)
    assert all(float(row["mae_margin_vs_placebo"]) > 0.0 for row in rows)


def test_run_batch_writes_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(batch.two_block, "ROOT_DIR", tmp_path)

    aligned_seeded_run = "admissibility-two-block-aligned-seeded"

    def fake_two_block_run(**_: object) -> dict[str, object]:
        return {
            "monthly_phase2_run_id": "monthly-run",
            "aligned_archive_run_id": "aligned-archive",
            "overall_decision": "keep_minimal_two_block_kernel",
            "archive_alignment_gate": {"active_sign_agreement": 1.0, "active_mean_abs_delta": 0.0},
            "active_scenario_summary": {"max_abs_terminal_delta": 12.0},
        }

    def fake_seeded_report(run_id: str) -> dict[str, object]:
        assert run_id == aligned_seeded_run
        return {
            "terminal_delta_rows": [
                {
                    "contract": "exact_only",
                    "scenario": "disruption_recovery",
                    "diagnosed_plhiv_delta": 5.0,
                    "alive_on_art_delta": -1.0,
                    "new_diagnosed_cases_period_delta": 2.0,
                }
            ]
        }

    phase2_state = seeded.Phase2QuarterState(
        source_run_id="monthly-run",
        month_axis=["2025-01", "2025-02", "2025-03"],
        quarter_axis=["2025-Q1", "2025-Q2", "2025-Q3"],
        block_axis=["care_access_continuity", "mobility_exposure_pressure"],
        quarter_states=np.asarray([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2]], dtype=np.float64),
        edge_rows=[],
    )

    monkeypatch.setattr(batch.two_block, "run_tr_v3_phase2_two_block_kernel_batch", fake_two_block_run)
    monkeypatch.setattr(batch, "_seeded_report", fake_seeded_report)
    monkeypatch.setattr(batch.seeded, "_load_phase2_quarter_state", lambda _: phase2_state)
    monkeypatch.setattr(batch.seeded, "_filter_phase2_quarter_state", lambda phase2_state, **_: phase2_state)
    monkeypatch.setattr(batch.seeded_gate, "_load_block_loading_rows", lambda _: [])
    monkeypatch.setattr(
        batch.seeded_gate,
        "_summarize_outcome_circularity_rows",
        lambda **_: [
            {
                "block_id": "care_access_continuity",
                "outcome_loading_share": 0.0,
            },
            {
                "block_id": "mobility_exposure_pressure",
                "outcome_loading_share": 0.0,
            },
        ],
    )
    monkeypatch.setattr(batch.seeded_gate, "_orthogonalize_phase2_state_against_outcomes", lambda **_: (phase2_state, []))
    monkeypatch.setattr(
        batch.seeded_gate,
        "_seeded_payload_from_phase2_state",
        lambda **_: {
            "terminal_delta_rows": [
                {
                    "contract": "exact_only",
                    "scenario": "disruption_recovery",
                    "diagnosed_plhiv_delta": 4.5,
                    "alive_on_art_delta": -1.0,
                    "new_diagnosed_cases_period_delta": 1.5,
                }
            ]
        },
    )
    monkeypatch.setattr(
        batch.seeded_gate,
        "_archive_alignment_rows",
        lambda *_: [
            {
                "contract": "exact_only",
                "scenario": "disruption_recovery",
                "metric": "diagnosed_plhiv",
                "baseline_terminal_delta": 5.0,
                "aligned_terminal_delta": 4.5,
                "delta_of_delta": -0.5,
                "sign_agrees": True,
            }
        ],
    )
    monkeypatch.setattr(
        batch,
        "_replay_contract_scenarios",
        lambda **kwargs: [
            {
                "contract": str(kwargs["contract_name"]),
                "split_idx": 0,
                "scenario": "disruption_recovery",
                "row_count": 4,
                "base_mae": 10.0,
                "scenario_mae": 8.0,
                "mae_delta": -2.0,
                "improved_fraction": 0.75,
                "directional_fit_rate": 0.8,
                "terminal_max_abs_delta": 12.0,
            },
            {
                "contract": str(kwargs["contract_name"]),
                "split_idx": 0,
                "scenario": "mobility_spike",
                "row_count": 4,
                "base_mae": 10.0,
                "scenario_mae": 9.0,
                "mae_delta": -1.0,
                "improved_fraction": 0.5,
                "directional_fit_rate": 0.75,
                "terminal_max_abs_delta": 6.0,
            },
            {
                "contract": str(kwargs["contract_name"]),
                "split_idx": 0,
                "scenario": "disruption_recovery_placebo",
                "row_count": 4,
                "base_mae": 10.0,
                "scenario_mae": 9.5,
                "mae_delta": -0.5,
                "improved_fraction": 0.5,
                "directional_fit_rate": 0.55,
                "terminal_max_abs_delta": 12.0,
            },
            {
                "contract": str(kwargs["contract_name"]),
                "split_idx": 0,
                "scenario": "mobility_spike_placebo",
                "row_count": 4,
                "base_mae": 10.0,
                "scenario_mae": 10.5,
                "mae_delta": 0.5,
                "improved_fraction": 0.25,
                "directional_fit_rate": 0.4,
                "terminal_max_abs_delta": 6.0,
            },
        ],
    )

    payload = batch.run_tr_v3_phase2_admissibility_backtest_batch(
        run_id="admissibility",
        active_block_subset=("care_access_continuity", "mobility_exposure_pressure"),
    )

    assert payload["gate_summary"]["overall_decision"] == "keep_provisional_two_block_sidecar"
    assert (tmp_path / "artifacts" / "runs" / "admissibility" / "analysis" / "tr_v3_phase2_admissibility_backtest_batch_report.json").exists()
