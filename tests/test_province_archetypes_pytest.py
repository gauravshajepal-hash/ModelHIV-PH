from __future__ import annotations

import numpy as np

from epigraph_ph.core.province_archetypes import (
    ARCHETYPE_ORDER,
    build_synthetic_province_library,
    infer_province_archetype_priors,
)


def test_synthetic_province_library_shapes() -> None:
    month_axis = [f"2024-{idx:02d}" for idx in range(1, 7)]
    bundle = build_synthetic_province_library(month_axis=month_axis)

    assert bundle["month_axis"] == month_axis
    assert len(bundle["archetypes"]) == len(ARCHETYPE_ORDER)
    assert len(bundle["synthetic_trajectories"]) == len(ARCHETYPE_ORDER)
    for row in bundle["synthetic_trajectories"]:
        trajectory = np.asarray(row["state_trajectory"], dtype=np.float32)
        assert trajectory.shape == (len(month_axis), 5)
        assert np.allclose(trajectory.sum(axis=-1), 1.0, atol=1e-4)


def test_archetype_priors_are_soft_and_nonconstant() -> None:
    province_axis = ["Metro Manila", "Basilan", "Cebu", "Davao City"]
    month_axis = [f"2024-{idx:02d}" for idx in range(1, 5)]
    observation_targets = {
        "diagnosed_stock": np.asarray(
            [
                [0.78, 0.79, 0.80, 0.81],
                [0.44, 0.45, 0.45, 0.46],
                [0.65, 0.66, 0.67, 0.68],
                [0.61, 0.62, 0.63, 0.64],
            ],
            dtype=np.float32,
        ),
        "art_stock": np.asarray(
            [
                [0.53, 0.54, 0.55, 0.56],
                [0.24, 0.25, 0.25, 0.26],
                [0.39, 0.40, 0.41, 0.42],
                [0.36, 0.37, 0.38, 0.39],
            ],
            dtype=np.float32,
        ),
        "documented_suppression": np.asarray(
            [
                [0.37, 0.38, 0.39, 0.40],
                [0.12, 0.12, 0.13, 0.13],
                [0.24, 0.25, 0.25, 0.26],
                [0.19, 0.20, 0.21, 0.22],
            ],
            dtype=np.float32,
        ),
        "testing_coverage": np.asarray(
            [
                [0.29, 0.30, 0.31, 0.32],
                [0.08, 0.08, 0.09, 0.09],
                [0.18, 0.19, 0.20, 0.21],
                [0.15, 0.16, 0.17, 0.18],
            ],
            dtype=np.float32,
        ),
    }
    subgroup_summary = {
        "rows": [
            {
                "province": "Metro Manila",
                "region": "ncr",
                "evidence_strength": 18.0,
                "network_signal": {"urbanity": 0.95, "accessibility": 0.90, "awareness": 0.78, "stress": 0.35},
                "kp_distribution": {"remaining_population": 0.50, "msm": 0.35, "tgw": 0.05},
            },
            {
                "province": "Basilan",
                "region": "barmm",
                "evidence_strength": 3.0,
                "network_signal": {"urbanity": 0.16, "accessibility": 0.18, "awareness": 0.24, "stress": 0.70},
                "kp_distribution": {"remaining_population": 0.84, "msm": 0.09, "tgw": 0.01},
            },
            {
                "province": "Cebu",
                "region": "region_vii",
                "evidence_strength": 10.0,
                "network_signal": {"urbanity": 0.74, "accessibility": 0.69, "awareness": 0.57, "stress": 0.58},
                "kp_distribution": {"remaining_population": 0.58, "msm": 0.27, "tgw": 0.03},
            },
            {
                "province": "Davao City",
                "region": "region_xi",
                "evidence_strength": 4.0,
                "network_signal": {"urbanity": 0.60, "accessibility": 0.54, "awareness": 0.46, "stress": 0.48},
                "kp_distribution": {"remaining_population": 0.61, "msm": 0.23, "tgw": 0.03},
            },
        ]
    }

    priors = infer_province_archetype_priors(
        province_axis=province_axis,
        month_axis=month_axis,
        subgroup_summary=subgroup_summary,
        observation_targets=observation_targets,
    )

    assert priors["mixture_matrix"].shape == (len(province_axis), len(ARCHETYPE_ORDER))
    assert np.allclose(priors["mixture_matrix"].sum(axis=1), 1.0, atol=1e-5)
    assert priors["transition_prior_shift"].shape == (len(province_axis), 5)
    assert priors["observation_weight"].shape == (len(province_axis),)
    assert len({row["dominant_archetype"] for row in priors["rows"]}) >= 2
    assert float(np.max(priors["synthetic_pretraining_weight"])) > float(np.min(priors["synthetic_pretraining_weight"]))
