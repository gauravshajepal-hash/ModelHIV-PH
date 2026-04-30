from __future__ import annotations

import pytest

from phase3_dynamic.metrics import (
    SUPPORT_AWARE_BACK_HALF_METRICS,
    SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
    support_aware_metric_scales,
    support_aware_normalized_mae,
    support_aware_rate_mae,
)


def _row(
    quarter: str,
    *,
    tested: float | None,
    suppressed: float | None,
    alive: float | None = 200.0,
    tier: str = "exact_observed",
) -> dict[str, object]:
    return {
        "quarter": quarter,
        "alive_on_art": alive,
        "tested_for_viral_load": tested,
        "virally_suppressed": suppressed,
        "metric_provenance": {
            "alive_on_art": {
                "tier": tier,
                "observation_role": "direct_target",
                "support_partition": "common_support",
            },
            "tested_for_viral_load": {
                "tier": tier,
                "observation_role": "direct_target",
                "support_partition": "common_support",
            },
            "virally_suppressed": {
                "tier": tier,
                "observation_role": "direct_target",
                "support_partition": "common_support",
            },
        },
    }


def test_support_aware_metric_scales_use_supported_train_rows_only() -> None:
    train_rows = [
        _row("2020-Q1", tested=100.0, suppressed=80.0),
        _row("2020-Q2", tested=10000.0, suppressed=9000.0, tier="rule_based_extrapolated"),
    ]

    scales = support_aware_metric_scales(train_rows, SUPPORT_AWARE_BACK_HALF_METRICS, eps=1e-9)

    assert scales["tested_for_viral_load"] == 100.0
    assert scales["virally_suppressed"] == 80.0


def test_support_aware_normalized_mae_scores_only_direct_observed_targets() -> None:
    targets = [
        _row("2021-Q1", tested=100.0, suppressed=80.0),
        _row("2021-Q2", tested=200.0, suppressed=160.0, tier="rule_based_extrapolated"),
    ]
    predictions = [
        {"quarter": "2021-Q1", "tested_for_viral_load": 90.0, "virally_suppressed": 72.0},
        {"quarter": "2021-Q2", "tested_for_viral_load": 0.0, "virally_suppressed": 0.0},
    ]
    scales = {"tested_for_viral_load": 100.0, "virally_suppressed": 80.0}

    result = support_aware_normalized_mae(
        predictions,
        targets,
        scales,
        metric_names=SUPPORT_AWARE_BACK_HALF_METRICS,
        eps=1e-9,
    )

    assert result["status"] == "scored"
    assert result["supported_target_count"] == 2
    assert result["scored_entry_count"] == 2
    assert result["mean_normalized_mae"] == pytest.approx(0.1)


def test_support_aware_rate_mae_scores_conditional_rates_without_holdout_scale() -> None:
    train_rows = [_row("2020-Q1", alive=200.0, tested=100.0, suppressed=80.0)]
    targets = [_row("2021-Q1", alive=200.0, tested=100.0, suppressed=80.0)]
    predictions = [{"quarter": "2021-Q1", "alive_on_art": 200.0, "tested_for_viral_load": 80.0, "virally_suppressed": 60.0}]

    result = support_aware_rate_mae(
        predictions,
        targets,
        train_rows,
        rate_names=SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
        eps=1e-9,
    )

    assert result["status"] == "scored"
    assert result["supported_target_count"] == 2
    assert result["scored_entry_count"] == 2
    assert result["mean_normalized_mae"] == pytest.approx(0.075)


def test_support_aware_rate_mae_requires_train_supported_conditional_rate() -> None:
    train_rows = [_row("2020-Q1", alive=200.0, tested=100.0, suppressed=80.0, tier="rule_based_extrapolated")]
    targets = [_row("2021-Q1", alive=200.0, tested=100.0, suppressed=80.0)]
    predictions = [{"quarter": "2021-Q1", "alive_on_art": 200.0, "tested_for_viral_load": 80.0, "virally_suppressed": 60.0}]

    result = support_aware_rate_mae(
        predictions,
        targets,
        train_rows,
        rate_names=SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
        eps=1e-9,
    )

    assert result["status"] == "not_train_supported"
    assert result["supported_target_count"] == 2
    assert result["scored_entry_count"] == 0
