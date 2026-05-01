from __future__ import annotations

from phase3_dynamic.monthly_shock import (
    _feature_for_month,
    _fit_linear_effect,
    _linear_effect,
    _month_index,
    _monthly_feature_table,
)


def _row(month: str, value: float) -> dict[str, object]:
    return {
        "metric_name": "new_diagnosed_cases_period",
        "value": value,
        "series_kind": "monthly_count",
        "source_quality_tier": "official_doh_archive",
        "measurement_class": "flow_count",
        "source_id": "synthetic",
        "evidence_confidence": 1.0,
        "_month_index": _month_index(month),
        "_contract": {
            "observation_role": "direct_target",
            "allowed_use": "train_target",
            "support_partition": "common_support",
        },
    }


def test_monthly_feature_table_is_forecast_origin_safe() -> None:
    train_end = _month_index("2020-02")
    table = _monthly_feature_table(
        [_row("2020-01", 10.0), _row("2020-02", 11.0), _row("2020-03", 1000.0)],
        train_end_month=train_end,
    )

    assert _month_index("2020-03") not in table
    future = _feature_for_month(table, _month_index("2020-03"), train_end_month=train_end, phi=0.0)
    assert future["residual_shock"] == 0.0


def test_monthly_linear_effect_is_bounded_by_train_residual_support() -> None:
    fit = _fit_linear_effect(
        ["2020-Q1", "2020-Q2"],
        {
            "2020-Q1": {"monthly_deviation": 1.0},
            "2020-Q2": {"monthly_deviation": -1.0},
        },
        {"2020-Q1": 0.2, "2020-Q2": -0.2},
        ["monthly_deviation"],
    )

    assert fit["train_row_count"] == 2
    assert _linear_effect({"monthly_deviation": 10.0}, fit, ["monthly_deviation"]) <= 0.2
