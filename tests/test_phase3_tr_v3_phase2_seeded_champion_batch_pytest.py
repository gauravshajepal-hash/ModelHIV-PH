from __future__ import annotations

import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as batch


def test_feature_matrix_stacks_levels_and_deltas() -> None:
    state_rows = np.asarray(
        [
            [1.0, 2.0],
            [2.5, 5.0],
            [4.0, 8.5],
        ],
        dtype=np.float64,
    )

    features = batch._feature_matrix(state_rows)

    assert features.shape == (3, 4)
    assert features[0].tolist() == [1.0, 2.0, 0.0, 0.0]
    assert features[2].tolist() == [4.0, 8.5, 1.5, 3.5]


def test_propagate_structural_states_uses_edge_and_persistence() -> None:
    kernel = {
        "intercept": np.asarray([0.0, 0.1], dtype=np.float64),
        "persistence": np.asarray([0.5, 0.25], dtype=np.float64),
        "edge_matrix": np.asarray([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64),
    }
    last_state = np.asarray([2.0, 1.0], dtype=np.float64)
    shock_matrix = np.zeros((2, 2), dtype=np.float64)

    path = batch._propagate_structural_states(kernel=kernel, last_state=last_state, steps=2, shock_matrix=shock_matrix)

    assert path.shape == (2, 2)
    assert np.allclose(path[0], [1.0, 0.95])
    assert np.allclose(path[1], [0.5, 0.6375])


def test_bounded_metric_correction_clips_large_adjustment() -> None:
    readout_metric = {
        "beta": [10.0, -5.0],
        "scale": 0.4,
        "cap_abs": 3.0,
    }

    correction = batch._bounded_metric_correction(readout_metric, np.asarray([2.0, 1.0], dtype=np.float64))

    assert correction == 3.0


def test_filter_phase2_quarter_state_subsets_blocks_and_edges() -> None:
    state = batch.Phase2QuarterState(
        source_run_id="run-a",
        month_axis=["2025-01", "2025-02", "2025-03"],
        quarter_axis=["2025-Q1"],
        block_axis=["care_access_continuity", "suppression_capacity", "mobility_exposure_pressure"],
        quarter_states=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        edge_rows=[
            {"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1, "weight": 0.3},
            {"source": "suppression_capacity", "target": "mobility_exposure_pressure", "lag": 1, "weight": 0.2},
        ],
    )

    filtered = batch._filter_phase2_quarter_state(
        state,
        active_block_subset=("care_access_continuity", "mobility_exposure_pressure"),
    )

    assert filtered.block_axis == ["care_access_continuity", "mobility_exposure_pressure"]
    assert filtered.quarter_states.tolist() == [[1.0, 3.0]]
    assert filtered.edge_rows == []


def test_scenario_shock_matrix_uses_testing_prevention_reach_when_present() -> None:
    shocks = batch._scenario_shock_matrix(
        "testing_pulse",
        steps=3,
        block_axis=["testing_prevention_reach", "care_access_continuity"],
        block_scales=np.asarray([2.0, 1.0], dtype=np.float64),
    )

    assert shocks[:, 0].tolist() == [2.0, 1.1, 0.4]
    assert shocks[:, 1].tolist() == [0.0, 0.0, 0.0]
