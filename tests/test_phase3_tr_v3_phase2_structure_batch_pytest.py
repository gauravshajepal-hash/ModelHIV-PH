from __future__ import annotations

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_structure_batch as batch


def _metric_diag(train_quarters: list[str], fitted_level: float, residuals: list[float]) -> dict[str, object]:
    return {
        "train_quarters": list(train_quarters),
        "train_observed": [float(fitted_level + residual) for residual in residuals],
        "train_fitted": [float(fitted_level) for _ in residuals],
        "train_supported": [True for _ in residuals],
    }


def test_build_train_residual_state_extracts_metric_vectors() -> None:
    quarters = ["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4", "2022-Q1", "2022-Q2"]
    residuals = [6.0, 3.0, 1.5, 0.75, 0.375, 0.1875]
    transition_diagnostics = {
        "D_to_A": {
            "diagnosed_level_diagnostics": _metric_diag(quarters, 100.0, residuals),
            "art_level_diagnostics": _metric_diag(quarters, 80.0, [value * 0.5 for value in residuals]),
        },
        "A_to_V": {
            "flow_level_diagnostics": _metric_diag(quarters, 10.0, [value * 0.25 for value in residuals]),
        },
    }

    payload = batch._build_train_residual_state(transition_diagnostics)

    assert payload["quarters"] == quarters
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    assert matrix.shape == (6, 3)
    assert np.allclose(matrix[0], np.asarray([6.0, 3.0, 1.5], dtype=np.float64))


def test_phase2_structural_model_emits_finite_constraint_preserving_corrections() -> None:
    train_quarters = ["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4", "2022-Q1", "2022-Q2"]
    train_residuals = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    transition_diagnostics = {
        "D_to_A": {
            "diagnosed_level_diagnostics": _metric_diag(train_quarters, 100.0, train_residuals),
            "art_level_diagnostics": _metric_diag(train_quarters, 80.0, [value * 0.5 for value in train_residuals]),
        },
        "A_to_V": {
            "flow_level_diagnostics": _metric_diag(train_quarters, 10.0, [value * 0.25 for value in train_residuals]),
        },
    }
    cfg = next(candidate for candidate in batch.STRUCTURAL_CANDIDATES if candidate.config_id == "P2-FULL-s10-l10-w50")
    model = batch._fit_phase2_structural_model(transition_diagnostics, cfg)

    assert model is not None
    base_prediction_rows = [
        {"quarter": "2022-Q3", "diagnosed_plhiv": 100.0, "alive_on_art": 80.0, "new_diagnosed_cases_period": 10.0},
        {"quarter": "2022-Q4", "diagnosed_plhiv": 100.0, "alive_on_art": 80.0, "new_diagnosed_cases_period": 10.0},
    ]
    corrected_rows, _ = batch._corrected_prediction_rows(base_prediction_rows, model, cfg)
    first = dict(corrected_rows[0])
    second = dict(corrected_rows[1])

    assert np.isfinite(first["diagnosed_plhiv"])
    assert np.isfinite(first["alive_on_art"])
    assert np.isfinite(first["new_diagnosed_cases_period"])
    assert first["diagnosed_plhiv"] > float(base_prediction_rows[0]["diagnosed_plhiv"])
    assert first["alive_on_art"] > float(base_prediction_rows[0]["alive_on_art"])
    assert first["new_diagnosed_cases_period"] > float(base_prediction_rows[0]["new_diagnosed_cases_period"])
    assert 0.0 <= first["alive_on_art"] <= first["diagnosed_plhiv"]
    assert 0.0 <= second["alive_on_art"] <= second["diagnosed_plhiv"]
