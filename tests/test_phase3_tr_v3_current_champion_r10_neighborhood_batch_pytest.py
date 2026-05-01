from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_current_champion_r10_neighborhood_batch as batch


def test_overall_decision_thresholds() -> None:
    assert (
        batch._overall_decision(
            [
                {
                    "contract": "exact_only",
                    "current_champion_kept": True,
                    "baseline_comparison": {"decision": "stable"},
                },
                {
                    "contract": "purged_dense",
                    "current_champion_kept": True,
                    "baseline_comparison": {"decision": "stable"},
                },
            ]
        )
        == "keep_current_champions_on_merged_archive"
    )
    assert (
        batch._overall_decision(
            [
                {
                    "contract": "exact_only",
                    "current_champion_kept": False,
                    "baseline_comparison": {"decision": "moderate_drift"},
                },
                {
                    "contract": "purged_dense",
                    "current_champion_kept": True,
                    "baseline_comparison": {"decision": "stable"},
                },
            ]
        )
        == "promote_narrow_r10_refresh"
    )
    assert (
        batch._overall_decision(
            [
                {
                    "contract": "exact_only",
                    "current_champion_kept": False,
                    "baseline_comparison": {"decision": "severe_drift"},
                },
                {
                    "contract": "purged_dense",
                    "current_champion_kept": False,
                    "baseline_comparison": {"decision": "moderate_drift"},
                },
            ]
        )
        == "reopen_broader_model_family_exploration"
    )


def test_result_row_reads_top_level_honesty_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        batch.compatibility,
        "_collect_absolute_residual_rows",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        batch.compatibility,
        "_support_rows",
        lambda *args, **kwargs: [],
    )
    row = batch._result_row(
        {
            "experiment_id": "EXP-R10-EXACT-CHAMPION",
            "quarterly_summary": {
                "candidate_mean_mae": 0.1,
                "carry_forward_mean_mae": 0.2,
                "candidate_worst_mae": 0.3,
                "endpoint_audit_summary": {
                    "candidate": {},
                    "suppression_honesty_flags": {"scored_direct_support": 7},
                },
            },
            "annual_summary": {"candidate_mean_incidence_error": 0.02},
        },
        contract_name="exact_only",
        archive_variant="merged",
        allowed_tiers={"exact_observed"},
    )
    assert row["honesty_flags"] == {"scored_direct_support": 7}


def test_run_batch_writes_neighborhood_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        batch,
        "_run_contract_neighborhood",
        lambda **kwargs: {
            "contract": str(kwargs["contract_name"]),
            "current_champion_id": str(kwargs["current_champion_id"]),
            "merged_rows": [
                {
                    "experiment_id": str(kwargs["current_champion_id"]),
                    "quarterly_mean_mae": 0.08,
                    "quarterly_baseline_mae": 0.14,
                    "quarterly_worst_mae": 0.16,
                    "annual_mean_incidence_error": 0.03,
                    "diagnosed_residual_p90": 100.0,
                    "art_residual_p90": 90.0,
                    "flow_residual_p90": 40.0,
                    "score_tuple": [0.08, 0.16, 0.03],
                },
                {
                    "experiment_id": "ALT",
                    "quarterly_mean_mae": 0.075,
                    "quarterly_baseline_mae": 0.14,
                    "quarterly_worst_mae": 0.15,
                    "annual_mean_incidence_error": 0.031,
                    "diagnosed_residual_p90": 95.0,
                    "art_residual_p90": 85.0,
                    "flow_residual_p90": 38.0,
                    "score_tuple": [0.075, 0.15, 0.031],
                },
            ],
            "merged_winner": {"experiment_id": "ALT"},
            "merged_current_champion": {"experiment_id": str(kwargs["current_champion_id"])},
            "current_champion_rank": 2,
            "current_champion_kept": False,
            "baseline_comparison": {
                "baseline_experiment_id": str(kwargs["current_champion_id"]),
                "merged_experiment_id": "ALT",
                "mean_mae_ratio": 1.10,
                "worst_mae_ratio": 1.08,
                "residual_p90_ratio_mean": 1.12,
                "mean_exact_share_delta": 0.0,
                "worsened_honesty_flags": {},
                "honesty_flag_worsened_count": 0,
                "decision": "stable",
            },
            "winner_improvement_vs_current_champion": {
                "mean_mae_delta": 0.005,
                "worst_mae_delta": 0.01,
                "diagnosed_p90_delta": 5.0,
                "art_p90_delta": 5.0,
                "flow_p90_delta": 2.0,
            },
        },
    )

    payload = batch.run_tr_v3_current_champion_r10_neighborhood_batch(
        run_id="r10-neighborhood",
        merged_archive_run_id="merged-archive",
        baseline_archive_run_id="baseline-archive",
    )

    assert payload["overall_decision"] == "promote_narrow_r10_refresh"
    assert (tmp_path / "artifacts" / "runs" / "r10-neighborhood" / "analysis" / "tr_v3_current_champion_r10_neighborhood_batch_report.json").exists()
