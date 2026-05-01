from __future__ import annotations

PHASE2_TRANSITION_PRIOR_MAP: dict[str, dict[str, dict[str, dict[str, dict[str, list[int] | float]]]]] = {
    "U_to_D": {
        "target_blocks": {
            "structural_barrier_pressure": {
                "source_blocks": {
                    "mobility_exposure_pressure": {"lags": [1], "prior_scale": 0.2},
                }
            },
            "mobility_exposure_pressure": {
                "source_blocks": {
                    "structural_barrier_pressure": {"lags": [1, 2], "prior_scale": 0.35},
                }
            },
        }
    },
    "D_to_A": {
        "target_blocks": {
            "mobility_exposure_pressure": {
                "source_blocks": {
                    "care_access_continuity": {"lags": [1], "prior_scale": 0.6},
                    "structural_barrier_pressure": {"lags": [1], "prior_scale": 0.25},
                }
            },
            "care_access_continuity": {
                "source_blocks": {
                    "mobility_exposure_pressure": {"lags": [1], "prior_scale": 0.2},
                }
            },
        }
    },
    "A_to_T": {
        "target_blocks": {
            "suppression_capacity": {
                "source_blocks": {
                    "care_access_continuity": {"lags": [1], "prior_scale": 0.2},
                }
            },
        }
    },
    "T_to_V": {
        "target_blocks": {
            "suppression_capacity": {
                "source_blocks": {
                    "care_access_continuity": {"lags": [1], "prior_scale": 0.25},
                }
            },
        }
    },
    "A_to_L": {
        "target_blocks": {
            "mobility_exposure_pressure": {
                "source_blocks": {
                    "structural_barrier_pressure": {"lags": [1, 2], "prior_scale": 0.55},
                }
            },
            "structural_barrier_pressure": {
                "source_blocks": {
                    "mobility_exposure_pressure": {"lags": [1], "prior_scale": 0.2},
                }
            },
        }
    },
    "T_to_L": {
        "target_blocks": {
            "mobility_exposure_pressure": {
                "source_blocks": {
                    "structural_barrier_pressure": {"lags": [1, 2], "prior_scale": 0.35},
                }
            },
        }
    },
    "V_to_L": {
        "target_blocks": {
            "structural_barrier_pressure": {
                "source_blocks": {
                    "mobility_exposure_pressure": {"lags": [1], "prior_scale": 0.2},
                }
            },
        }
    },
    "L_to_R": {
        "target_blocks": {
            "suppression_capacity": {
                "source_blocks": {
                    "care_access_continuity": {"lags": [1], "prior_scale": 0.55},
                }
            },
        }
    },
    "R_to_A": {
        "target_blocks": {
            "care_access_continuity": {
                "source_blocks": {
                    "suppression_capacity": {"lags": [1], "prior_scale": 0.25},
                }
            },
        }
    },
}
