from __future__ import annotations

from phase3_dynamic.monthly_latent_state import (
    REPORTING_STREAM_METRICS,
    _build_stream_process,
    _stream_feature_for_month,
)
from phase3_dynamic.monthly_shock import _month_index


def _row(month: str, metric_name: str, value: float) -> dict[str, object]:
    return {
        "metric_name": metric_name,
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


def test_monthly_latent_stream_ignores_future_rows_after_forecast_origin() -> None:
    train_end = _month_index("2020-03")
    process = _build_stream_process(
        [
            _row("2020-01", "new_diagnosed_cases_period", 10.0),
            _row("2020-02", "new_diagnosed_cases_period", 11.0),
            _row("2020-03", "new_diagnosed_cases_period", 12.0),
            _row("2020-04", "new_diagnosed_cases_period", 10000.0),
        ],
        stream_name="reporting",
        metric_names=REPORTING_STREAM_METRICS,
        anchor_metric="new_diagnosed_cases_period",
        train_end_month=train_end,
    )

    assert _month_index("2020-04") not in process["month_table"]
    future = _stream_feature_for_month(process, _month_index("2020-04"))
    assert abs(future["reporting_intensity"]) <= 1.0


def test_monthly_latent_stream_uses_multiple_metric_residual_matrix() -> None:
    process = _build_stream_process(
        [
            _row("2020-01", "new_diagnosed_cases_period", 10.0),
            _row("2020-02", "new_diagnosed_cases_period", 20.0),
            _row("2020-03", "new_diagnosed_cases_period", 40.0),
            _row("2020-01", "diagnosed_plhiv", 100.0),
            _row("2020-02", "diagnosed_plhiv", 130.0),
            _row("2020-03", "diagnosed_plhiv", 170.0),
        ],
        stream_name="reporting",
        metric_names=REPORTING_STREAM_METRICS,
        anchor_metric="new_diagnosed_cases_period",
        train_end_month=_month_index("2020-03"),
    )

    assert process["active_metric_names"] == ["diagnosed_plhiv", "new_diagnosed_cases_period"]
    assert process["month_table"]
