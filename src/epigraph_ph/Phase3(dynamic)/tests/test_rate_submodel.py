from __future__ import annotations

import pytest

from phase3_dynamic.rate_submodel import (
    BASELINE_FAMILY,
    RateTarget,
    _claim_grade_support_gate,
    _score_family_gate,
    forecast_rate_family,
)


def _target(quarter: str, rate_name: str, value: float) -> RateTarget:
    return RateTarget(
        quarter=quarter,
        rate_name=rate_name,
        value=value,
        ordinal=int(quarter[:4]) * 4,
        provenance={"support_partition": "common_support"},
    )


def test_last_observed_rate_carry_forward_uses_latest_supported_train_rate() -> None:
    result = forecast_rate_family(
        [
            _target("2020-Q1", "vl_testing_given_art", 0.4),
            _target("2021-Q1", "vl_testing_given_art", 0.6),
        ],
        ["2022-Q1", "2022-Q2"],
        family=BASELINE_FAMILY,
        eps=1e-9,
    )

    assert result["predictions"][("2022-Q1", "vl_testing_given_art")] == 0.6
    assert result["predictions"][("2022-Q2", "vl_testing_given_art")] == 0.6


def test_logit_linear_trend_forecasts_supported_rates() -> None:
    result = forecast_rate_family(
        [
            _target("2020-Q1", "vl_testing_given_art", 0.2),
            _target("2021-Q1", "vl_testing_given_art", 0.4),
            _target("2022-Q1", "vl_testing_given_art", 0.6),
        ],
        ["2023-Q1"],
        family="logit_linear_trend",
        eps=1e-9,
    )

    assert result["diagnostics"]["vl_testing_given_art"]["status"] == "fit"
    assert 0.0 < result["predictions"][("2023-Q1", "vl_testing_given_art")] < 1.0
    assert result["predictions"][("2023-Q1", "vl_testing_given_art")] > 0.6


def test_rate_family_gate_promotes_only_true_improvement_over_last_observed() -> None:
    passing = _score_family_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "baseline": {"score": {"status": "scored", "mean_mae": 0.2, "worst_mae": 0.3, "target_count": 2}},
                "candidate": {"score": {"status": "scored", "mean_mae": 0.1, "worst_mae": 0.25, "target_count": 2}},
            }
        ],
        family="synthetic_candidate",
        gate_name="synthetic_gate",
    )
    failing = _score_family_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "baseline": {"score": {"status": "scored", "mean_mae": 0.2, "worst_mae": 0.3, "target_count": 2}},
                "candidate": {"score": {"status": "scored", "mean_mae": 0.2, "worst_mae": 0.3, "target_count": 2}},
            }
        ],
        family="synthetic_candidate",
        gate_name="synthetic_gate",
    )

    assert passing["status"] == "pass"
    assert passing["claim_status"] == "supports_conditional_rate_improvement_claim"
    assert failing["status"] == "fail"
    assert "candidate_mean_not_better_than_last_observed_rate_carry_forward" in failing["blockers"]


def test_rate_family_gate_blocks_worst_error_regression() -> None:
    gate = _score_family_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "baseline": {"score": {"status": "scored", "mean_mae": 0.2, "worst_mae": 0.25, "target_count": 2}},
                "candidate": {"score": {"status": "scored", "mean_mae": 0.1, "worst_mae": 0.3, "target_count": 2}},
            }
        ],
        family="synthetic_candidate",
        gate_name="synthetic_gate",
    )

    assert gate["status"] == "fail"
    assert "candidate_worst_regresses_against_last_observed_rate_carry_forward" in gate["blockers"]


def _metric_score(vl_mean: float, vl_worst: float, suppression_mean: float, suppression_worst: float) -> dict[str, object]:
    return {
        "status": "scored",
        "mean_mae": (vl_mean + suppression_mean) / 2.0,
        "worst_mae": max(vl_worst, suppression_worst),
        "metric_rows": [
            {
                "metric_name": "vl_testing_given_art",
                "status": "scored",
                "target_count": 2,
                "mean_mae": vl_mean,
                "worst_mae": vl_worst,
            },
            {
                "metric_name": "suppression_given_vl_tested",
                "status": "scored",
                "target_count": 2,
                "mean_mae": suppression_mean,
                "worst_mae": suppression_worst,
            },
        ],
    }


def _claim_grade_horizon(holdouts: list[list[int]], *, candidate_better: bool = True) -> dict[str, object]:
    rows = []
    split_rows = []
    for holdout_years in holdouts:
        rows.append(
            {
                "status": "comparable",
                "holdout_years": holdout_years,
                "baseline_mean_mae": 0.2,
                "candidate_mean_mae": 0.1 if candidate_better else 0.2,
                "baseline_worst_mae": 0.25,
                "candidate_worst_mae": 0.2 if candidate_better else 0.25,
            }
        )
        split_rows.append(
            {
                "holdout_years": holdout_years,
                "baseline": {"score": _metric_score(0.2, 0.25, 0.2, 0.25)},
                "candidate": {
                    "score": _metric_score(
                        0.1 if candidate_better else 0.2,
                        0.2 if candidate_better else 0.25,
                        0.1 if candidate_better else 0.2,
                        0.2 if candidate_better else 0.25,
                    )
                },
            }
        )
    return {
        "gate": {
            "status": "pass",
            "comparable_split_count": len(holdouts),
            "rows": rows,
        },
        "split_rows": split_rows,
    }


def test_claim_grade_gate_requires_enough_supported_blocked_splits() -> None:
    horizon = _claim_grade_horizon([[2025]])
    gate = _claim_grade_support_gate(horizon, horizon, family="train_mean")

    assert gate["status"] == "fail"
    assert "insufficient_one_year_comparable_splits_for_claim_grade" in gate["blockers"]
    assert "missing_post_covid_rebound_coverage_in_both_blocked_gates" in gate["blockers"]


def test_claim_grade_gate_passes_only_with_era_coverage_and_per_rate_improvement() -> None:
    horizon = _claim_grade_horizon([[2023], [2025]])
    gate = _claim_grade_support_gate(horizon, horizon, family="train_mean")

    assert gate["status"] == "pass"
    assert gate["claim_status"] == "publication_grade_third_95_rate_process"
