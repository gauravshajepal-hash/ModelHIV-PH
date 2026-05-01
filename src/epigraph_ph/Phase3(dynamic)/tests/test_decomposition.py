from __future__ import annotations

import numpy as np

from phase3_dynamic.decomposition import (
    DecompositionControlConfig,
    _active_streams,
    _bounded_effects,
    _fit_stream_components,
)


def test_stream_decomposition_forecasts_without_future_values() -> None:
    result = _fit_stream_components(
        stream="diagnosis",
        train_quarters=["2020-Q1", "2020-Q2", "2020-Q3"],
        train_values=[0.1, 0.2, 0.3],
        support_values=[0.0, 0.5, 0.5],
        holdout_quarters=["2020-Q4", "2021-Q1"],
        positions={
            "2020-Q1": 0,
            "2020-Q2": 1,
            "2020-Q3": 2,
            "2020-Q4": 3,
            "2021-Q1": 4,
        },
        eps=1e-9,
    )

    assert set(result["train_components"]["2020-Q2"]) == {
        "trend",
        "reporting_support_shift",
        "residual_shock",
    }
    assert set(result["holdout_components"]) == {"2020-Q4", "2021-Q1"}
    assert result["diagnostics"]["residual_rho"] <= 1.0


def test_bounded_effects_cannot_exceed_train_residual_scale() -> None:
    x_train = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    x_holdout = np.asarray([[100.0]], dtype=np.float64)
    y = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)

    result = _bounded_effects(
        x_train=x_train,
        y=y,
        x_holdout=x_holdout,
        eps=1e-9,
        effect_shrinkage=DecompositionControlConfig().effect_shrinkage,
    )

    assert abs(float(result["holdout_effect"][0])) <= max(abs(value) for value in y)


def test_decomposition_config_can_disable_streams_for_ablation() -> None:
    streams = _active_streams(DecompositionControlConfig(disabled_streams=("incidence", "vl")))

    assert "incidence" not in streams
    assert "vl" not in streams
    assert "diagnosis" in streams
