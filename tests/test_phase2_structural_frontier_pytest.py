from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from epigraph_ph.phase2.phase3_compat import build_phase3_compatibility_artifacts, build_phase3_compatibility_payload
from epigraph_ph.phase3.frontier.phase2_structural_inputs import load_phase2_structural_inputs
from epigraph_ph.phase3.frontier.tr_v2 import _fit_direct_transition_effects
from epigraph_ph.phase3.shared.phase2_inputs import load_phase2_compatibility_payload
from epigraph_ph.runtime import read_json, save_tensor_artifact, write_json


def test_phase2_compatibility_loader_requires_frozen_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase2_dir = run_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Frozen Phase 2 compatibility payload is required"):
        load_phase2_compatibility_payload(run_dir)


def test_phase2_compatibility_loader_uses_frozen_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase2_dir = run_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    direct_artifact = save_tensor_artifact(
        array=np.ones((2, 3, 4), dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="direct_feature_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    hidden_artifact = save_tensor_artifact(
        array=np.zeros((2, 3, 2), dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="hidden_feature_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    multiscale_artifact = save_tensor_artifact(
        array=np.full((2, 3, 1), 2.0, dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="multiscale_feature_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase2_dir / "phase3_compatibility_payload.json",
        {
            "core_feature_tensor_path": direct_artifact["value_path"],
            "direct_feature_tensor_path": direct_artifact["value_path"],
            "hidden_driver_feature_tensor_path": hidden_artifact["value_path"],
            "multiscale_support_feature_tensor_path": multiscale_artifact["value_path"],
            "markov_blanket": {"blanket_nodes": ["testing_engagement"]},
            "eligibility_surfaces": {"direct_temporal": {"blanket_nodes": ["testing_engagement"]}},
        },
    )

    payload = load_phase2_compatibility_payload(run_dir)

    assert payload["compatibility_payload_source"] == "frozen_artifact"
    assert payload["direct_feature_tensor_array"].shape == (2, 3, 4)
    assert payload["hidden_driver_feature_tensor_array"].shape == (2, 3, 2)
    assert payload["multiscale_support_feature_tensor_array"].shape == (2, 3, 1)


def test_phase2_compatibility_payload_counts_direct_support_from_measurement_role(tmp_path: Path) -> None:
    phase1_dir = tmp_path / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        phase1_dir / "axis_catalogs.json",
        {
            "canonical": ["testing_prevention_reach", "care_access_continuity"],
        },
    )
    save_tensor_artifact(
        array=np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase1_dir,
        stem="standardized_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase1_dir / "normalized_subparameters.json",
        [
            {
                "canonical_name": "testing_prevention_reach",
                "measurement_role": "direct_indicator",
                "model_numeric_value": 1.0,
                "is_anchor_eligible": True,
                "evidence_weight": 2.0,
                "observation_weight": 3.0,
                "domain_family": "testing",
                "pathway_family": "testing",
                "source_bank": "harp",
                "geo_resolution": "national",
                "region": "national",
                "time": "2025-Q1",
                "soft_ontology_tags": ["testing"],
                "linkage_targets": ["diagnosed_stock"],
            },
            {
                "canonical_name": "testing_prevention_reach",
                "observation_role": "direct_indicator",
                "model_numeric_value": 0.5,
                "is_anchor_eligible": False,
                "evidence_weight": 1.0,
                "quality_weight": 1.5,
                "domain_family": "testing",
                "pathway_family": "testing",
                "source_bank": "legacy_harp",
                "geo_resolution": "national",
                "region": "national",
                "time": "2025-Q2",
            },
        ],
    )

    payload = build_phase3_compatibility_payload(
        phase1_dir=phase1_dir,
        predictive_rows=[],
        context_rows=[],
        retained_catalog={"rows": []},
        multiscale_blankets={"phase3_member_canonical_names": ["care_access_continuity"]},
        latent_blankets={
            "direct_phase3_member_canonical_names": ["testing_prevention_reach"],
            "hidden_phase3_member_canonical_names": ["care_access_continuity"],
        },
        multiscale_bundle={"scales": {}},
        latent_bundle={"scales": {}},
    )

    testing_row = next(
        row for row in payload["candidate_profiles"] if row["canonical_name"] == "testing_prevention_reach"
    )

    assert testing_row["direct_support"] == 2
    assert testing_row["direct_support_ratio"] == pytest.approx(1.0)
    assert payload["compatibility_contract"]["scientific_status"] == "historical_bridge_only"
    assert payload["compatibility_contract"]["canonical_frontier_input"] == "phase2_structural_payload"


def test_phase2_compatibility_artifacts_freeze_bridge_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase1_dir = run_dir / "phase1"
    phase2_dir = run_dir / "phase2"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        phase1_dir / "axis_catalogs.json",
        {
            "canonical": ["testing_prevention_reach"],
        },
    )
    save_tensor_artifact(
        array=np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase1_dir,
        stem="standardized_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase1_dir / "normalized_subparameters.json",
        [
            {
                "canonical_name": "testing_prevention_reach",
                "measurement_role": "direct_indicator",
                "model_numeric_value": 1.0,
                "evidence_weight": 1.0,
            }
        ],
    )

    build_phase3_compatibility_artifacts(
        phase2_dir=phase2_dir,
        phase1_dir=phase1_dir,
        predictive_rows=[],
        context_rows=[],
        retained_catalog={"rows": []},
        multiscale_blankets={"phase3_member_canonical_names": []},
        latent_blankets={"direct_phase3_member_canonical_names": ["testing_prevention_reach"]},
        multiscale_bundle={"scales": {}},
        latent_bundle={"scales": {}},
    )

    frozen_payload = read_json(phase2_dir / "phase3_compatibility_payload.json", default={})
    loaded_payload = load_phase2_compatibility_payload(run_dir)

    assert frozen_payload.get("compatibility_contract", {}).get("scientific_status") == "historical_bridge_only"
    assert loaded_payload.get("compatibility_contract", {}).get("scientific_status") == "historical_bridge_only"
    assert loaded_payload.get("compatibility_payload_source") == "frozen_artifact"


def test_phase2_structural_loader_requires_frozen_payload(tmp_path: Path) -> None:
    ctx = SimpleNamespace(phase2_dir=tmp_path / "phase2", phase15_dir=tmp_path / "phase15")
    ctx.phase2_dir.mkdir(parents=True, exist_ok=True)
    ctx.phase15_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Frozen Phase 2 structural payload is required"):
        load_phase2_structural_inputs(ctx)


def test_phase2_structural_loader_quarterizes_states_and_hidden_modes(tmp_path: Path) -> None:
    phase2_dir = tmp_path / "phase2"
    phase15_dir = tmp_path / "phase15"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase15_dir.mkdir(parents=True, exist_ok=True)

    national_state_artifact = save_tensor_artifact(
        array=np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2),
        axis_names=["national", "month", "block"],
        artifact_dir=phase15_dir,
        stem="phase15_v2_national_state_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    hidden_mode_artifact = save_tensor_artifact(
        array=np.arange(1 * 6 * 3, dtype=np.float32).reshape(1, 6, 3),
        axis_names=["national", "month", "hidden_mode"],
        artifact_dir=phase2_dir,
        stem="phase2_national_hidden_mode_scores",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase2_dir / "phase2_structural_payload.json",
        {
            "schema_version": "phase2_structural_payload_v1",
            "month_axis": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
            "block_axis": ["testing_engagement", "care_access_continuity"],
            "direct_temporal_edge_rows": [
                {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.5, "stability": 0.8, "support_count": 3}
            ],
            "hidden_driver_rows": [
                {"source": "testing_engagement", "target": "care_access_continuity", "lag": 1, "weight": 0.4, "support_count": 2}
            ],
            "multiscale_support_rows": [{"factor_id": "factor_01", "support_count": 2}],
            "artifact_paths": {
                "phase15_national_state_tensor": national_state_artifact["value_path"],
                "hidden_mode_score_tensor": hidden_mode_artifact["value_path"],
            },
        },
    )

    ctx = SimpleNamespace(phase2_dir=phase2_dir, phase15_dir=phase15_dir)
    loaded = load_phase2_structural_inputs(ctx)

    assert loaded.month_axis == ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    assert loaded.quarter_axis == ["2025-Q1", "2025-Q2"]
    assert loaded.national_quarter_tensor.shape == (1, 2, 2)
    assert loaded.hidden_mode_quarter_tensor.shape == (1, 2, 3)
    assert loaded.direct_edge_rows[0]["source"] == "testing_engagement"
    assert loaded.multiscale_support_rows[0]["factor_id"] == "factor_01"


def test_phase2_structural_loader_prefers_national_scale_rows_over_aggregated_summary(tmp_path: Path) -> None:
    phase2_dir = tmp_path / "phase2"
    phase15_dir = tmp_path / "phase15"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase15_dir.mkdir(parents=True, exist_ok=True)

    national_state_artifact = save_tensor_artifact(
        array=np.arange(1 * 4 * 1, dtype=np.float32).reshape(1, 4, 1),
        axis_names=["national", "month", "block"],
        artifact_dir=phase15_dir,
        stem="phase15_v2_national_state_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase2_dir / "phase2_structural_payload.json",
        {
            "schema_version": "phase2_structural_payload_v1",
            "month_axis": ["2025-01", "2025-02", "2025-03", "2025-04"],
            "block_axis": ["source_a"],
            "direct_temporal_edge_rows": [
                {"source": "summary_source", "target": "summary_target", "lag": 1, "weight": 0.9, "stability": 0.9, "support_count": 3}
            ],
            "direct_temporal_edge_scale_rows": [
                {"scale": "province", "source": "prov_source", "target": "prov_target", "lag": 1, "weight": 0.3, "stability": 0.4},
                {"scale": "national", "source": "nat_source", "target": "nat_target", "lag": 2, "weight": 0.7, "stability": 0.8},
            ],
            "hidden_driver_rows": [],
            "hidden_driver_scale_rows": [],
            "multiscale_support_rows": [],
            "artifact_paths": {
                "phase15_national_state_tensor": national_state_artifact["value_path"],
            },
        },
    )

    ctx = SimpleNamespace(phase2_dir=phase2_dir, phase15_dir=phase15_dir)
    loaded = load_phase2_structural_inputs(ctx)

    assert [(row["source"], row["target"], int(row["lag"])) for row in loaded.direct_edge_rows] == [("nat_source", "nat_target", 2)]
    assert [(row["source"], row["target"], int(row["lag"])) for row in loaded.direct_edge_summary_rows] == [("summary_source", "summary_target", 1)]


def test_tr_v2_direct_priors_preserve_target_block_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(
            constraint_settings={
                "phase3": {
                    "frontier": {
                        "phase2_transition_prior_map": {
                            "U_to_D": {
                                "target_blocks": {
                                    "target_b": {
                                        "source_blocks": {
                                            "source_a": {"lags": [1], "prior_scale": 0.5},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
    )

    structural_inputs = SimpleNamespace(
        direct_edge_rows=[
            {"source": "source_a", "target": "target_b", "lag": 1, "weight": 0.5, "stability": 0.8, "support_count": 3},
            {"source": "source_c", "target": "target_b", "lag": 1, "weight": 0.5, "stability": 0.8, "support_count": 3},
            {"source": "source_a", "target": "target_c", "lag": 1, "weight": 0.5, "stability": 0.8, "support_count": 3},
        ],
        multiscale_support_rows=[],
        quarter_axis=["2025-Q1", "2025-Q2"],
        block_axis=["source_a"],
        national_quarter_tensor=np.asarray([[[1.0], [2.0]]], dtype=np.float32),
    )
    dataset = {
        "train_transition_rows": [
            {"quarter": "2025-Q2", "hazards": {"U_to_D": 0.25}},
        ]
    }

    result = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset)
    summary = dict(result["summary"]["U_to_D"])

    assert summary["feature_count"] == 1
    assert summary["coefficients"][0]["block_id"] == "source_a"
    assert summary["coefficients"][0]["target_block_id"] == "target_b"
    assert int(summary["coefficients"][0]["lag"]) == 1


def test_fit_prior_regression_uses_nonzero_prior_means() -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    beta = tr_v2._fit_prior_regression(
        target_values=np.zeros((4,), dtype=np.float32),
        design_matrix=np.zeros((4, 1), dtype=np.float32),
        prior_precisions=np.asarray([25.0], dtype=np.float32),
        prior_means=np.asarray([0.4], dtype=np.float32),
    )

    assert beta.shape == (1,)
    assert float(beta[0]) == pytest.approx(0.4, abs=1e-6)


def test_tr_v2_explicit_target_block_ablation_beats_legacy_and_no_priors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(
            constraint_settings={
                "phase3": {
                    "frontier": {
                        "phase2_transition_prior_map": {
                            "U_to_D": {
                                "target_blocks": {
                                    "target_good": {
                                        "source_blocks": {
                                            "source_good": {"lags": [1], "prior_scale": 0.6},
                                        }
                                    },
                                    "target_missing": {
                                        "source_blocks": {
                                            "source_bad": {"lags": [1], "prior_scale": 1.2},
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
        ),
    )

    quarter_axis = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2025-Q5", "2025-Q6"]
    source_good = np.asarray([0.0, 0.2, 0.4, 0.7, 1.0, 1.1], dtype=np.float32)
    # Looks correlated in the training window, then spikes on the holdout-driving quarter.
    source_bad = np.asarray([0.0, 0.15, 0.35, 0.65, 3.5, 0.1], dtype=np.float32)
    structural_inputs = SimpleNamespace(
        direct_edge_rows=[
            {"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.35, "stability": 0.8, "support_count": 3},
            {"source": "source_bad", "target": "wrong_target", "lag": 1, "weight": 0.85, "stability": 0.95, "support_count": 4},
        ],
        multiscale_support_rows=[],
        quarter_axis=quarter_axis,
        block_axis=["source_good", "source_bad"],
        national_quarter_tensor=np.asarray([np.stack([source_good, source_bad], axis=-1)], dtype=np.float32),
    )

    def true_hazard(prev_source_good: float) -> float:
        return float(tr_v2._inv_logit(-2.0 + 1.2 * prev_source_good))

    train_transition_rows = []
    for idx, quarter in enumerate(quarter_axis[1:5], start=1):
        train_transition_rows.append(
            {
                "quarter": quarter,
                "hazards": {
                    "U_to_D": true_hazard(float(source_good[idx - 1])),
                    "D_to_A": 1e-5,
                    "A_to_V": 1e-5,
                    "A_to_L": 1e-5,
                    "L_to_A": 1e-5,
                },
            }
        )

    current_state = {"U": 1000.0, "D": 100.0, "A": 50.0, "V": 20.0, "L": 10.0}
    holdout_u_to_d = true_hazard(float(source_good[4])) * current_state["U"]
    dataset = {
        "train_transition_rows": train_transition_rows,
        "train_state_rows": [{"state_values": current_state}],
        "holdout_rows": [
            {
                "quarter": "2025-Q6",
                "diagnosed_plhiv": 180.0 + holdout_u_to_d,
                "alive_on_art": 70.0,
                "new_diagnosed_cases_period": holdout_u_to_d,
            }
        ],
        "metric_scales": {
            "diagnosed_plhiv": 1.0,
            "alive_on_art": 1.0,
            "new_diagnosed_cases_period": 1.0,
        },
        "eps": 1e-6,
        "testing_share_mean": 0.5,
    }
    baseline_hazards = {
        "2025-Q6": {
            "U_to_D": 0.15,
            "D_to_A": 1e-5,
            "A_to_V": 1e-5,
            "A_to_L": 1e-5,
            "L_to_A": 1e-5,
        }
    }

    def evaluate_variant(direct_adjustments: dict[str, dict[str, float]]) -> dict[str, float]:
        _forecast_rows, _hazard_rows, evaluation = tr_v2._simulate_holdout(
            dataset=dataset,
            baseline_hazards=baseline_hazards,
            direct_adjustments=direct_adjustments,
            hidden_adjustments={},
            peak_gates={},
        )
        return {
            "mae": float(evaluation["model_mean_absolute_error"]),
            "smape": float(evaluation["model_smape"]),
        }

    current_fit = _fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset)
    quarter_features = tr_v2._quarter_feature_map(structural_inputs)
    y = np.asarray(
        [
            tr_v2._logit(float((row.get("hazards") or {}).get("U_to_D") or 1e-5))
            for row in train_transition_rows
        ],
        dtype=np.float32,
    )
    intercept = float(np.mean(y))
    X_matrix = np.asarray(
        [
            [
                float(quarter_features.get(quarter_axis[quarter_axis.index(str(row.get("quarter") or "")) - 1], {}).get("source_good") or 0.0),
                float(quarter_features.get(quarter_axis[quarter_axis.index(str(row.get("quarter") or "")) - 1], {}).get("source_bad") or 0.0),
            ]
            for row in train_transition_rows
        ],
        dtype=np.float32,
    )
    legacy_beta = tr_v2._fit_prior_regression(
        target_values=y - intercept,
        design_matrix=X_matrix,
        prior_precisions=np.asarray(
            [
                1.0 / max(((0.6 * (1.0 + 3.0) * 0.8) ** 2), 1e-6),
                1.0 / max(((1.2 * (1.0 + 4.0) * 0.95) ** 2), 1e-6),
            ],
            dtype=np.float32,
        ),
    )
    legacy_adjustments = {"U_to_D": {}}
    for quarter_idx, quarter in enumerate(quarter_axis):
        source_idx = quarter_idx - 1
        delta = 0.0
        if source_idx >= 0:
            source_quarter = quarter_axis[source_idx]
            delta += float(legacy_beta[0]) * float(quarter_features.get(source_quarter, {}).get("source_good") or 0.0)
            delta += float(legacy_beta[1]) * float(quarter_features.get(source_quarter, {}).get("source_bad") or 0.0)
        legacy_adjustments["U_to_D"][quarter] = delta

    no_prior_metrics = evaluate_variant({})
    explicit_metrics = evaluate_variant(current_fit["quarter_adjustments"])
    legacy_metrics = evaluate_variant(legacy_adjustments)

    assert current_fit["summary"]["U_to_D"]["feature_count"] == 1
    assert explicit_metrics["mae"] < no_prior_metrics["mae"]
    assert explicit_metrics["mae"] < legacy_metrics["mae"]
    assert explicit_metrics["smape"] < no_prior_metrics["smape"]


def test_tr_v2_hidden_basis_adjustment_beats_no_hidden_on_synthetic_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(constraint_settings={"phase3": {"frontier": {"phase2_hidden_shock": {"rank_cap": 1}}}}),
    )

    quarter_axis = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2025-Q5", "2025-Q6"]
    hidden_signal = np.asarray([0.0, 0.1, 0.3, 0.5, 0.9, 1.2], dtype=np.float32)
    structural_inputs = SimpleNamespace(
        quarter_axis=quarter_axis,
        hidden_mode_quarter_tensor=np.asarray(hidden_signal.reshape(1, -1, 1), dtype=np.float32),
        hidden_mode_summary={
            "basis_kind": "weighted_low_rank_contribution_basis",
            "identified_hidden_shock_process": False,
            "ar1_rows": [{"hidden_mode": 0, "phi": 0.8}],
        },
    )
    train_rows = []
    for idx, quarter in enumerate(quarter_axis[1:5], start=1):
        train_rows.append(
            {
                "quarter": quarter,
                "hazards": {
                    "U_to_D": float(tr_v2._inv_logit(-2.1 + 0.9 * hidden_signal[idx])),
                    "D_to_A": 1e-5,
                    "A_to_V": 1e-5,
                    "A_to_L": 1e-5,
                    "L_to_A": 1e-5,
                },
            }
        )
    dataset = {
        "train_transition_rows": train_rows,
    }

    hidden_fit = tr_v2._fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments={},
    )

    summary = hidden_fit["summary"]["U_to_D"]
    assert summary["rank_used"] == 1
    assert summary["basis_kind"] == "weighted_low_rank_contribution_basis"
    assert summary["identified_hidden_shock_process"] is False
    assert summary["prior_diagnostics"][0]["prior_precision"] != pytest.approx(1.0)
    assert abs(float(summary["coefficients"][0]["coefficient"])) > 0.0


def test_tr_v2_multiscale_support_modulates_prior_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(
            constraint_settings={
                "phase3": {
                    "frontier": {
                        "phase2_multiscale_support": {
                            "support_scale": 0.12,
                            "max_multiplier": 1.4,
                            "transition_hook_map": {"U_to_D": ["diagnosis_transitions"]},
                        },
                        "phase2_transition_prior_map": {
                            "U_to_D": {
                                "target_blocks": {
                                    "target_good": {
                                        "source_blocks": {
                                            "source_good": {"lags": [1], "prior_scale": 0.4, "prior_mean_scale": 0.5},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        ),
    )

    quarter_axis = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2025-Q5", "2025-Q6"]
    source_good = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0, 1.1], dtype=np.float32)
    structural_inputs = SimpleNamespace(
        direct_edge_rows=[
            {"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.45, "stability": 0.8, "support_count": 3},
        ],
        multiscale_support_rows=[
            {"factor_id": "factor_01", "support_count": 4, "transition_hooks": ["diagnosis_transitions"]},
        ],
        quarter_axis=quarter_axis,
        block_axis=["source_good"],
        national_quarter_tensor=np.asarray([source_good.reshape(-1, 1)], dtype=np.float32),
    )
    train_transition_rows = []
    for idx, quarter in enumerate(quarter_axis[1:5], start=1):
        train_transition_rows.append(
            {
                "quarter": quarter,
                "hazards": {
                    "U_to_D": float(tr_v2._inv_logit(-2.0 + 1.0 * source_good[idx - 1])),
                    "D_to_A": 1e-5,
                    "A_to_V": 1e-5,
                    "A_to_L": 1e-5,
                    "L_to_A": 1e-5,
                },
            }
        )
    dataset = {"train_transition_rows": train_transition_rows}

    without_multiscale = tr_v2._fit_direct_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        include_multiscale=False,
    )
    with_multiscale = tr_v2._fit_direct_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        include_multiscale=True,
    )

    no_multi_summary = without_multiscale["summary"]["U_to_D"]["coefficients"][0]
    with_multi_summary = with_multiscale["summary"]["U_to_D"]["coefficients"][0]

    assert without_multiscale["summary"]["U_to_D"]["multiscale_support"]["multiplier"] == pytest.approx(1.0)
    assert with_multiscale["summary"]["U_to_D"]["multiscale_support"]["multiplier"] > 1.0
    assert with_multi_summary["prior_precision"] > no_multi_summary["prior_precision"]


def test_tr_v2_scale_disagreement_softens_direct_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(
            constraint_settings={
                "phase3": {
                    "frontier": {
                        "phase2_scale_evidence": {"dispersion_penalty": 0.5, "agreement_floor": 0.25},
                        "phase2_transition_prior_map": {
                            "U_to_D": {
                                "target_blocks": {
                                    "target_good": {
                                        "source_blocks": {
                                            "source_good": {"lags": [1], "prior_scale": 0.4, "prior_mean_scale": 0.5},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        ),
    )

    quarter_axis = ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2025-Q5"]
    source_good = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    dataset = {
        "train_transition_rows": [
            {"quarter": "2025-Q2", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * source_good[0]))}},
            {"quarter": "2025-Q3", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * source_good[1]))}},
            {"quarter": "2025-Q4", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * source_good[2]))}},
            {"quarter": "2025-Q5", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * source_good[3]))}},
        ]
    }
    coherent_inputs = SimpleNamespace(
        direct_edge_rows=[{"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8, "support_count": 2}],
        direct_edge_summary_rows=[{"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8, "support_count": 2}],
        direct_edge_scale_rows=[
            {"scale": "national", "source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8},
            {"scale": "province", "source": "source_good", "target": "target_good", "lag": 1, "weight": 0.35, "stability": 0.75},
        ],
        multiscale_support_rows=[],
        quarter_axis=quarter_axis,
        block_axis=["source_good"],
        national_quarter_tensor=np.asarray([source_good.reshape(-1, 1)], dtype=np.float32),
    )
    conflicting_inputs = SimpleNamespace(
        direct_edge_rows=[{"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8, "support_count": 2}],
        direct_edge_summary_rows=[{"source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8, "support_count": 2}],
        direct_edge_scale_rows=[
            {"scale": "national", "source": "source_good", "target": "target_good", "lag": 1, "weight": 0.4, "stability": 0.8},
            {"scale": "province", "source": "source_good", "target": "target_good", "lag": 1, "weight": -0.35, "stability": 0.75},
        ],
        multiscale_support_rows=[],
        quarter_axis=quarter_axis,
        block_axis=["source_good"],
        national_quarter_tensor=np.asarray([source_good.reshape(-1, 1)], dtype=np.float32),
    )

    coherent = tr_v2._fit_direct_transition_effects(structural_inputs=coherent_inputs, dataset=dataset, include_multiscale=False)
    conflicting = tr_v2._fit_direct_transition_effects(structural_inputs=conflicting_inputs, dataset=dataset, include_multiscale=False)

    coherent_row = coherent["summary"]["U_to_D"]["coefficients"][0]
    conflicting_row = conflicting["summary"]["U_to_D"]["coefficients"][0]

    assert coherent_row["scale_agreement"] > conflicting_row["scale_agreement"]
    assert coherent_row["prior_precision"] > conflicting_row["prior_precision"]
    assert abs(float(coherent_row["prior_mean"])) > abs(float(conflicting_row["prior_mean"]))


def test_tr_v2_replay_from_frozen_structural_payload_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import epigraph_ph.phase3.frontier.tr_v2 as tr_v2

    monkeypatch.setattr(
        tr_v2,
        "_HIV_PLUGIN",
        SimpleNamespace(
            constraint_settings={
                "phase3": {
                    "frontier": {
                        "phase2_transition_prior_map": {
                            "U_to_D": {
                                "target_blocks": {
                                    "target_good": {
                                        "source_blocks": {
                                            "source_good": {"lags": [1], "prior_scale": 0.4, "prior_mean_scale": 0.5},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ),
    )

    phase2_dir = tmp_path / "phase2"
    phase15_dir = tmp_path / "phase15"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase15_dir.mkdir(parents=True, exist_ok=True)
    national_state_artifact = save_tensor_artifact(
        array=np.asarray([[[0.0], [0.2], [0.5], [0.8], [1.0], [1.1]]], dtype=np.float32),
        axis_names=["national", "month", "block"],
        artifact_dir=phase15_dir,
        stem="phase15_v2_national_state_tensor",
        backend="numpy",
        device="cpu",
        save_pt=False,
    )
    write_json(
        phase2_dir / "phase2_structural_payload.json",
        {
            "schema_version": "phase2_structural_payload_v1",
            "month_axis": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
            "block_axis": ["source_good"],
            "direct_temporal_edge_rows": [
                {
                    "source": "source_good",
                    "target": "target_good",
                    "lag": 1,
                    "weight": 0.45,
                    "stability": 0.8,
                    "support_count": 2,
                    "scale_rows": [
                        {"scale": "national", "weight": 0.45, "stability": 0.8},
                        {"scale": "province", "weight": 0.40, "stability": 0.75},
                    ],
                }
            ],
            "direct_temporal_edge_scale_rows": [
                {"scale": "national", "source": "source_good", "target": "target_good", "lag": 1, "weight": 0.45, "stability": 0.8},
                {"scale": "province", "source": "source_good", "target": "target_good", "lag": 1, "weight": 0.40, "stability": 0.75},
            ],
            "hidden_driver_rows": [],
            "hidden_driver_scale_rows": [],
            "multiscale_support_rows": [],
            "artifact_paths": {
                "phase15_national_state_tensor": national_state_artifact["value_path"],
            },
        },
    )
    ctx = SimpleNamespace(phase2_dir=phase2_dir, phase15_dir=phase15_dir)
    loaded_a = load_phase2_structural_inputs(ctx)
    loaded_b = load_phase2_structural_inputs(ctx)
    dataset = {
        "train_transition_rows": [
            {"quarter": "2025-Q2", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * 0.0))}},
            {"quarter": "2025-Q3", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * 0.2))}},
            {"quarter": "2025-Q4", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * 0.5))}},
            {"quarter": "2025-Q5", "hazards": {"U_to_D": float(tr_v2._inv_logit(-2.0 + 0.8 * 0.8))}},
        ]
    }

    fit_a = tr_v2._fit_direct_transition_effects(structural_inputs=loaded_a, dataset=dataset, include_multiscale=False)
    fit_b = tr_v2._fit_direct_transition_effects(structural_inputs=loaded_b, dataset=dataset, include_multiscale=False)

    assert fit_a["summary"] == fit_b["summary"]
    assert fit_a["quarter_adjustments"] == fit_b["quarter_adjustments"]
