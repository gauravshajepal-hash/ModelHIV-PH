from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_current_champion_expanded_harp_compatibility_batch as batch


def test_contract_decision_thresholds() -> None:
    assert (
        batch._contract_decision(
            mean_mae_ratio=1.05,
            worst_mae_ratio=1.08,
            residual_p90_ratio_mean=1.10,
            honesty_flag_worsened_count=0,
            mean_exact_share_delta=0.01,
        )
        == "stable"
    )
    assert (
        batch._contract_decision(
            mean_mae_ratio=1.20,
            worst_mae_ratio=1.10,
            residual_p90_ratio_mean=1.05,
            honesty_flag_worsened_count=0,
            mean_exact_share_delta=0.01,
        )
        == "moderate_drift"
    )
    assert (
        batch._contract_decision(
            mean_mae_ratio=1.10,
            worst_mae_ratio=1.12,
            residual_p90_ratio_mean=1.10,
            honesty_flag_worsened_count=1,
            mean_exact_share_delta=0.01,
        )
        == "severe_drift"
    )


def test_support_rows_reads_top_level_holdout_support_counts() -> None:
    rows = batch._support_rows(
        {
            "endpoint_audit_summary": {
                "candidate": {},
                "holdout_support_counts": {
                    "diagnosed_plhiv": {"exact_observed": 4, "bridge_observed": 1, "scored": 5},
                    "alive_on_art": {"exact_observed": 3, "bridge_observed": 0, "scored": 3},
                    "new_diagnosed_cases_period": {"exact_observed": 2, "bridge_observed": 0, "scored": 2},
                },
            }
        },
        contract_name="exact_only",
        archive_variant="baseline",
    )
    lookup = {row["metric"]: row for row in rows}
    assert lookup["diagnosed_plhiv"]["exact_count"] == 4
    assert lookup["diagnosed_plhiv"]["scored_count"] == 5


def test_run_batch_writes_compatibility_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(batch, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        batch,
        "_current_winner_configs",
        lambda: {
            "exact_only": {
                "winner_id": "EXP-R10-EXACT-CHAMPION",
                "suite_contract": "exact_only",
                "forecast_contract": "exact_only",
                "allowed_tiers": {"exact_observed"},
            },
            "purged_dense": {
                "winner_id": "EXP-R10-DENSE-CHAMPION",
                "suite_contract": "purged_dense",
                "forecast_contract": "dense_train_observed_score",
                "allowed_tiers": {"exact_observed", "bridge_observed"},
            },
        },
    )
    monkeypatch.setattr(
        batch.monthly_lane,
        "_copy_harp_archive",
        lambda *args, **kwargs: {
            "mode": "baseline_plus_coverage_multinational_merge",
            "coverage_multinational_row_count": 100,
            "preserved_observed_program_panel": True,
            "preserved_diagnosis_flow_points": True,
        },
    )

    def fake_eval(*, archive_run_id: str, contract_name: str, archive_variant: str, winner_config: dict[str, object], forecast_horizon_quarters: int) -> dict[str, object]:
        merged = str(archive_variant) == "merged"
        base_mean = 0.07 if contract_name == "exact_only" else 0.09
        mean_mae = base_mean * (1.0 if not merged else 1.05)
        worst_mae = 0.14 * (1.0 if not merged else 1.03)
        residual_scale = 100.0 * (1.0 if not merged else 1.10)
        exact_share = 1.0 if contract_name == "exact_only" else (0.45 if not merged else 0.47)
        return {
            "archive_run_id": archive_run_id,
            "archive_variant": archive_variant,
            "contract": contract_name,
            "winner_id": str(winner_config["winner_id"]),
            "quarterly_summary": {
                "candidate_mean_mae": mean_mae,
                "candidate_worst_mae": worst_mae,
            },
            "annual_summary": {"candidate_mean_incidence_error": 0.03 if not merged else 0.031},
            "honesty_flags": {"scored_direct_support": 5},
            "residual_rows": [
                {
                    "contract": contract_name,
                    "archive_variant": archive_variant,
                    "metric": metric_name,
                    "tier": "overall",
                    "count": 10,
                    "abs_residual_mean": residual_scale * 0.6,
                    "abs_residual_p90": residual_scale,
                }
                for metric_name in batch.METRIC_ORDER
            ],
            "support_rows": [
                {
                    "contract": contract_name,
                    "archive_variant": archive_variant,
                    "metric": metric_name,
                    "exact_count": 10,
                    "bridge_count": 0 if contract_name == "exact_only" else 12,
                    "scored_count": 10 if contract_name == "exact_only" else 22,
                    "exact_share": exact_share,
                }
                for metric_name in batch.METRIC_ORDER
            ],
            "forecast_rows": [
                {
                    "quarter": "2026-Q1",
                    "diagnosed_plhiv": 1000.0 + (20.0 if merged else 0.0),
                    "alive_on_art": 800.0 + (10.0 if merged else 0.0),
                    "new_diagnosed_cases_period": 100.0 + (5.0 if merged else 0.0),
                },
                {
                    "quarter": "2026-Q4",
                    "diagnosed_plhiv": 1100.0 + (25.0 if merged else 0.0),
                    "alive_on_art": 850.0 + (12.0 if merged else 0.0),
                    "new_diagnosed_cases_period": 95.0 + (3.0 if merged else 0.0),
                },
            ],
        }

    monkeypatch.setattr(batch, "_evaluate_archive_contract", fake_eval)

    payload = batch.run_tr_v3_current_champion_expanded_harp_compatibility_batch(
        run_id="compat",
        baseline_archive_run_id="baseline-archive",
        coverage_archive_run_id="coverage-archive",
        forecast_horizon_quarters=4,
    )

    assert payload["overall_decision"] == "keep_current_champions"
    assert (tmp_path / "artifacts" / "runs" / "compat" / "analysis" / "tr_v3_current_champion_expanded_harp_compatibility_batch_report.json").exists()
