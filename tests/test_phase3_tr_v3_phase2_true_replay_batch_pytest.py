from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_true_replay_batch as batch


def _metric_diag(train_quarters: list[str], fitted_level: float, residuals: list[float]) -> dict[str, object]:
    return {
        "train_quarters": list(train_quarters),
        "train_observed": [float(fitted_level + residual) for residual in residuals],
        "train_fitted": [float(fitted_level) for _ in residuals],
        "train_supported": [True for _ in residuals],
    }


def _synthetic_structural_inputs() -> SimpleNamespace:
    quarter_axis = ["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4", "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4"]
    national_quarter_tensor = np.asarray(
        [
            [
                [0.0, 0.0],
                [4.0, 2.0],
                [2.0, 1.0],
                [1.0, 0.5],
                [0.5, 0.25],
                [0.25, 0.125],
                [0.125, 0.0625],
                [0.0625, 0.03125],
            ]
        ],
        dtype=np.float32,
    )
    hidden_mode_quarter_tensor = np.asarray(
        [
            [
                [0.0],
                [2.0],
                [1.0],
                [0.5],
                [0.25],
                [0.125],
                [0.0625],
                [0.03125],
            ]
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        quarter_axis=quarter_axis,
        national_quarter_tensor=national_quarter_tensor,
        hidden_mode_quarter_tensor=hidden_mode_quarter_tensor,
    )


def test_lagged_feature_map_uses_prior_quarter() -> None:
    cfg = next(candidate for candidate in batch.TRUE_REPLAY_CANDIDATES if candidate.config_id == "P2-TRUE-BOTH-r1-w25")
    feature_map = batch._lagged_feature_map(_synthetic_structural_inputs(), cfg)

    assert "2021-Q1" not in feature_map
    assert np.allclose(feature_map["2021-Q2"], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(feature_map["2021-Q3"], np.asarray([4.0, 2.0, 2.0], dtype=np.float32))


def test_true_replay_model_emits_finite_corrections() -> None:
    quarters = ["2021-Q2", "2021-Q3", "2021-Q4", "2022-Q1", "2022-Q2", "2022-Q3"]
    residuals = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    transition_diagnostics = {
        "D_to_A": {
            "diagnosed_level_diagnostics": _metric_diag(quarters, 100.0, residuals),
            "art_level_diagnostics": _metric_diag(quarters, 80.0, [value * 0.5 for value in residuals]),
        },
        "A_to_V": {
            "flow_level_diagnostics": _metric_diag(quarters, 10.0, [value * 0.25 for value in residuals]),
        },
    }
    cfg = next(candidate for candidate in batch.TRUE_REPLAY_CANDIDATES if candidate.config_id == "P2-TRUE-BLOCK-r1-w50")
    model = batch._fit_true_replay_model(transition_diagnostics, _synthetic_structural_inputs(), cfg)

    assert model is not None
    base_prediction_rows = [
        {"quarter": "2022-Q3", "diagnosed_plhiv": 100.0, "alive_on_art": 80.0, "new_diagnosed_cases_period": 10.0},
        {"quarter": "2022-Q4", "diagnosed_plhiv": 100.0, "alive_on_art": 80.0, "new_diagnosed_cases_period": 10.0},
    ]
    corrected_rows, correction_rows = batch._corrected_prediction_rows(base_prediction_rows, model, cfg)

    assert len(corrected_rows) == 2
    assert len(correction_rows) == 2
    assert np.isfinite(float(corrected_rows[0]["diagnosed_plhiv"]))
    assert np.isfinite(float(corrected_rows[0]["alive_on_art"]))
    assert np.isfinite(float(corrected_rows[0]["new_diagnosed_cases_period"]))
    assert float(corrected_rows[0]["diagnosed_plhiv"]) >= float(base_prediction_rows[0]["diagnosed_plhiv"])
    assert 0.0 <= float(corrected_rows[0]["alive_on_art"]) <= float(corrected_rows[0]["diagnosed_plhiv"])
