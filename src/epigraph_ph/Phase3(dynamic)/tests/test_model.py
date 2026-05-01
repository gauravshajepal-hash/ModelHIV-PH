from __future__ import annotations

import numpy as np

from phase3_dynamic.data import build_blocked_time_dataset
from phase3_dynamic.model import (
    DampingConfig,
    DynamicBaselineConfig,
    HiddenDriverConfig,
    IncidenceFlowConfig,
    ObservationModelConfig,
    ShockConfig,
    _direct_prior_design,
    _hidden_driver_design,
    carry_forward_hazards,
    fit_dynamic_hazard_paths,
    forecast_dynamic_baseline,
    simulate_holdout,
)
from phase3_dynamic.incidence import apply_stock_balance
from phase3_dynamic.phase2 import (
    DirectPriorFeature,
    HiddenDriverFeature,
    Phase2StructuralInputs,
    filter_direct_prior_features_by_edge_keys,
)


def _synthetic_rows(*, regime_break: bool = False) -> list[dict[str, float]]:
    quarters = [f"{year}-Q1" for year in range(2017, 2025)]
    state = {"U": 1000.0, "D": 200.0, "A": 150.0, "V": 100.0, "L": 50.0}
    rows = []
    last_u_to_d = None
    for idx, quarter in enumerate(quarters):
        diagnosed = state["D"] + state["A"] + state["V"] + state["L"]
        alive_on_art = state["A"] + state["V"]
        rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": alive_on_art,
                "new_diagnosed_cases_period": last_u_to_d,
                "tested_for_viral_load": 0.55 * alive_on_art,
                "virally_suppressed": 0.62 * alive_on_art,
                "estimated_plhiv": state["U"] + diagnosed,
                "population_total": 1_000_000.0 + 10_000.0 * idx,
            }
        )
        if regime_break:
            u_to_d_h = 0.04 if idx < 5 else 0.14
        else:
            u_to_d_h = 0.05 + 0.01 * idx
        d_to_a_h = 0.45
        a_to_v_h = 0.30
        a_to_l_h = 0.05
        l_to_a_h = 0.10
        last_u_to_d = u_to_d_h * state["U"]
        d_to_a = d_to_a_h * state["D"]
        a_to_v = a_to_v_h * state["A"]
        a_to_l = a_to_l_h * state["A"]
        l_to_a = l_to_a_h * state["L"]
        state = {
            "U": max(state["U"] - last_u_to_d, 0.0),
            "D": max(state["D"] + last_u_to_d - d_to_a, 0.0),
            "A": max(state["A"] + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(state["V"] + a_to_v, 0.0),
            "L": max(state["L"] + a_to_l - l_to_a, 0.0),
        }
    return rows


def _synthetic_hidden_rows() -> tuple[list[dict[str, float]], Phase2StructuralInputs, dict[str, list[HiddenDriverFeature]]]:
    quarters = [f"{year}-Q1" for year in range(2017, 2025)]
    hidden_mode = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    state = {"U": 1000.0, "D": 200.0, "A": 150.0, "V": 100.0, "L": 50.0}
    rows: list[dict[str, float]] = []
    last_u_to_d = None
    for quarter, mode in zip(quarters, hidden_mode):
        diagnosed = state["D"] + state["A"] + state["V"] + state["L"]
        alive_on_art = state["A"] + state["V"]
        rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": alive_on_art,
                "new_diagnosed_cases_period": last_u_to_d,
                "tested_for_viral_load": 0.55 * alive_on_art,
                "virally_suppressed": 0.62 * alive_on_art,
                "estimated_plhiv": state["U"] + diagnosed,
                "population_total": 1_000_000.0,
            }
        )
        u_to_d_h = 0.06 + 0.015 * float(mode)
        d_to_a_h = 0.45
        a_to_v_h = 0.30
        a_to_l_h = 0.05
        l_to_a_h = 0.10
        last_u_to_d = u_to_d_h * state["U"]
        d_to_a = d_to_a_h * state["D"]
        a_to_v = a_to_v_h * state["A"]
        a_to_l = a_to_l_h * state["A"]
        l_to_a = l_to_a_h * state["L"]
        state = {
            "U": max(state["U"] - last_u_to_d, 0.0),
            "D": max(state["D"] + last_u_to_d - d_to_a, 0.0),
            "A": max(state["A"] + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(state["V"] + a_to_v, 0.0),
            "L": max(state["L"] + a_to_l - l_to_a, 0.0),
        }
    structural_inputs = Phase2StructuralInputs(
        source_run_id="synthetic-hidden",
        quarter_axis=quarters,
        block_axis=["dummy_block"],
        national_quarter_tensor=np.zeros((1, len(quarters), 1), dtype=np.float32),
        direct_edge_rows=[],
        hidden_driver_rows=[{"source": "care_access_continuity", "target": "mobility_exposure_pressure", "lag": 1, "weight": 0.5, "stability": 1.0, "support_count": 1}],
        multiscale_support_rows=[],
        hidden_mode_quarter_tensor=np.asarray(hidden_mode, dtype=np.float32).reshape(1, len(quarters), 1),
        payload={},
    )
    hidden_features = {
        "U_to_D": [
            HiddenDriverFeature(
                transition="U_to_D",
                mode_index=0,
                support_scale=1.0,
                stability=1.0,
                support_count=1,
                target_blocks=("mobility_exposure_pressure",),
            )
        ]
    }
    return rows, structural_inputs, hidden_features


def _synthetic_growth_rows() -> list[dict[str, float]]:
    quarters = [f"{year}-Q1" for year in range(2017, 2025)]
    state = {"U": 1000.0, "D": 200.0, "A": 150.0, "V": 100.0, "L": 50.0}
    rows = []
    last_u_to_d = None
    for idx, quarter in enumerate(quarters):
        diagnosed = state["D"] + state["A"] + state["V"] + state["L"]
        alive_on_art = state["A"] + state["V"]
        rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": alive_on_art,
                "new_diagnosed_cases_period": last_u_to_d,
                "tested_for_viral_load": 0.55 * alive_on_art,
                "virally_suppressed": 0.62 * alive_on_art,
                "estimated_plhiv": state["U"] + diagnosed,
                "population_total": 1_000_000.0 + 10_000.0 * idx,
            }
        )
        inflow = 80.0 + 10.0 * float(idx)
        u_to_d_h = 0.06
        d_to_a_h = 0.45
        a_to_v_h = 0.30
        a_to_l_h = 0.05
        l_to_a_h = 0.10
        state["U"] += inflow
        last_u_to_d = u_to_d_h * state["U"]
        d_to_a = d_to_a_h * state["D"]
        a_to_v = a_to_v_h * state["A"]
        a_to_l = a_to_l_h * state["A"]
        l_to_a = l_to_a_h * state["L"]
        state = {
            "U": max(state["U"] - last_u_to_d, 0.0),
            "D": max(state["D"] + last_u_to_d - d_to_a, 0.0),
            "A": max(state["A"] + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(state["V"] + a_to_v, 0.0),
            "L": max(state["L"] + a_to_l - l_to_a, 0.0),
        }
    return rows


def test_dynamic_baseline_beats_carry_forward_on_trending_hazard() -> None:
    dataset = build_blocked_time_dataset(_synthetic_rows(), [2024])
    carry = simulate_holdout(dataset, carry_forward_hazards(dataset, mode="last_train"))
    dynamic = forecast_dynamic_baseline(dataset, DynamicBaselineConfig())
    assert dynamic["mae"] < carry["mae"]
    assert dynamic["model_contract"]["schema_version"] == "phase3_model_contract.v1"
    assert dynamic["hazard_semantics"]["fitted_transition_hazard"]["status"] == "emitted"
    assert dynamic["hazard_semantics"]["diagnostic_derived_hazard"]["status"] == "not_emitted_by_phase3_dynamic_hazard_model"


def test_phase2_direct_gate_fails_closed_when_no_source_stable_edges() -> None:
    rows = {
        "U_to_D": [
            DirectPriorFeature(
                transition="U_to_D",
                source="a",
                target="b",
                lag=1,
                prior_scale=1.0,
                phase2_weight=0.5,
                stability=1.0,
                support_count=2,
                tensor_index=0,
                edge_key="direct:a->b:lag1",
            )
        ]
    }

    filtered = filter_direct_prior_features_by_edge_keys(rows, set())

    assert filtered["U_to_D"] == []


def test_observation_model_emits_auxiliary_predictions() -> None:
    dataset = build_blocked_time_dataset(_synthetic_rows(), [2024])
    result = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(), observation_cfg=ObservationModelConfig())
    assert result["observation_model"] is not None
    assert all(row.get("tested_for_viral_load") is not None for row in result["prediction_rows"])
    assert all(row.get("virally_suppressed") is not None for row in result["prediction_rows"])
    assert result["model_contract"]["model_kind"] == "mechanistic_transition_with_observation_calibration"
    assert result["hazard_semantics"]["predictive_observation_head"]["status"] == "emitted"
    assert result["observation_model"]["primary_coefficients"]["diagnosed_plhiv"]["time_slope"] == 0.0


def test_shock_layer_improves_regime_break_synthetic() -> None:
    dataset = build_blocked_time_dataset(_synthetic_rows(regime_break=True), [2023])
    baseline = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(ridge_penalty=0.1, rho_clip=0.8, trend_scale=1.0))
    shocked = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.1, rho_clip=0.8, trend_scale=1.0),
        shock_cfg=ShockConfig(shock_phi=0.8, shock_scale=1.0, gate_z=0.5),
    )
    assert shocked["mae"] <= baseline["mae"]
    paths = fit_dynamic_hazard_paths(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.1, rho_clip=0.8, trend_scale=1.0),
        shock_cfg=ShockConfig(shock_phi=0.8, shock_scale=1.0, gate_z=0.5),
    )
    assert paths["diagnostics"]["U_to_D"]["last_residual"] != 0.0


def test_damping_shrinks_forecast_toward_last_train_anchor() -> None:
    dataset = build_blocked_time_dataset(_synthetic_rows(), [2024])
    undamped_paths = fit_dynamic_hazard_paths(dataset, DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0))
    damped_paths = fit_dynamic_hazard_paths(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        damping_cfg=DampingConfig(
            min_dynamic_weight=0.10,
            gain_weight=0.0,
            residual_weight=0.0,
            horizon_decay=0.5,
            calibration_floor=0.2,
            calibration_gain_weight=0.5,
        ),
    )
    last_train_hazard = dataset.train_transition_rows[-1]["hazards"]["U_to_D"]
    holdout_quarter = str(dataset.holdout_rows[0]["quarter"])
    undamped = undamped_paths["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    damped = damped_paths["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    assert abs(damped - last_train_hazard) <= abs(undamped - last_train_hazard)


def test_hidden_driver_layer_emits_operational_hidden_adjustment() -> None:
    rows, structural_inputs, hidden_features = _synthetic_hidden_rows()
    dataset = build_blocked_time_dataset(rows, [2024])
    baseline = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0))
    hidden = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        structural_inputs=structural_inputs,
        hidden_driver_features=hidden_features,
        hidden_cfg=HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.4, rank_cap=1),
    )
    holdout_quarter = str(dataset.holdout_rows[0]["quarter"])
    baseline_hazard = baseline["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    hidden_hazard = hidden["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    assert hidden["hidden_diagnostics"] is not None
    assert hidden["hidden_diagnostics"]["U_to_D"]["feature_count"] == 1
    assert abs(hidden_hazard - baseline_hazard) > 1e-9
    assert np.isfinite(hidden["mae"])


def test_phase2_design_matrices_are_forecast_origin_safe() -> None:
    quarter_axis = ["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4", "2022-Q1", "2022-Q2"]
    base_direct_tensor = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0, 1000.0], dtype=np.float32).reshape(1, 6, 1)
    base_hidden_tensor = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0, 1000.0], dtype=np.float32).reshape(1, 6, 1)
    perturbed_direct_tensor = base_direct_tensor.copy()
    perturbed_hidden_tensor = base_hidden_tensor.copy()
    perturbed_direct_tensor[0, 4:, 0] = np.asarray([1_000_000.0, -1_000_000.0], dtype=np.float32)
    perturbed_hidden_tensor[0, 4:, 0] = np.asarray([1_000_000.0, -1_000_000.0], dtype=np.float32)

    def _inputs(direct_tensor: np.ndarray, hidden_tensor: np.ndarray) -> Phase2StructuralInputs:
        return Phase2StructuralInputs(
            source_run_id="synthetic-contract",
            quarter_axis=quarter_axis,
            block_axis=["direct_block"],
            national_quarter_tensor=direct_tensor,
            direct_edge_rows=[],
            hidden_driver_rows=[],
            multiscale_support_rows=[],
            hidden_mode_quarter_tensor=hidden_tensor,
            payload={},
        )

    direct_feature = DirectPriorFeature(
        transition="U_to_D",
        source="direct_block",
        target="target_block",
        lag=1,
        prior_scale=1.0,
        phase2_weight=1.0,
        stability=1.0,
        support_count=1,
        tensor_index=0,
    )
    hidden_feature = HiddenDriverFeature(
        transition="U_to_D",
        mode_index=0,
        support_scale=1.0,
        stability=1.0,
        support_count=1,
        target_blocks=("target_block",),
    )
    train_quarters = ["2021-Q2", "2021-Q3", "2021-Q4"]
    holdout_quarters = ["2022-Q1", "2022-Q2"]

    direct_base = _direct_prior_design(_inputs(base_direct_tensor, base_hidden_tensor), [direct_feature], train_quarters, holdout_quarters, eps=1e-12)
    direct_perturbed = _direct_prior_design(_inputs(perturbed_direct_tensor, base_hidden_tensor), [direct_feature], train_quarters, holdout_quarters, eps=1e-12)
    hidden_base = _hidden_driver_design(_inputs(base_direct_tensor, base_hidden_tensor), [hidden_feature], train_quarters, holdout_quarters, eps=1e-12)
    hidden_perturbed = _hidden_driver_design(_inputs(base_direct_tensor, perturbed_hidden_tensor), [hidden_feature], train_quarters, holdout_quarters, eps=1e-12)

    assert np.array_equal(direct_base[1], direct_perturbed[1])
    assert np.array_equal(hidden_base[1], hidden_perturbed[1])
    assert direct_base[-1]["forecast_origin_quarter"] == "2021-Q4"
    assert hidden_base[-1]["forecast_origin_quarter"] == "2021-Q4"
    assert direct_base[-1]["holdout_future_source_count"] == 1
    assert hidden_base[-1]["holdout_future_source_count"] == 2
    assert direct_base[-1]["forecast_origin_contract"] == "train_history_only_with_carry_forward_for_future_phase2_values"

def test_hidden_driver_damping_reports_reduced_weight() -> None:
    rows, structural_inputs, hidden_features = _synthetic_hidden_rows()
    dataset = build_blocked_time_dataset(rows, [2024])
    hidden = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        structural_inputs=structural_inputs,
        hidden_driver_features=hidden_features,
        hidden_cfg=HiddenDriverConfig(
            precision_scale=0.25,
            residual_ridge=0.01,
            max_effect=0.4,
            rank_cap=1,
            min_effect_weight=0.2,
            gain_weight=0.0,
            shock_penalty_weight=1.0,
            horizon_decay=0.5,
        ),
    )
    assert hidden['hidden_diagnostics'] is not None
    assert hidden['hidden_diagnostics']['U_to_D']['base_weight'] == 0.2
    assert hidden['hidden_diagnostics']['U_to_D']['effective_decay'] <= 0.5


def test_hidden_driver_transition_gate_can_zero_low_gain_transition() -> None:
    rows, structural_inputs, hidden_features = _synthetic_hidden_rows()
    dataset = build_blocked_time_dataset(rows, [2024])
    baseline = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0))
    hidden = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        structural_inputs=structural_inputs,
        hidden_driver_features=hidden_features,
        hidden_cfg=HiddenDriverConfig(
            precision_scale=0.25,
            residual_ridge=0.01,
            max_effect=0.4,
            rank_cap=1,
            gate_gain_weight=1.0,
            gate_recent_penalty=0.0,
            gate_threshold=1.0,
            blocked_weight=0.0,
        ),
    )
    holdout_quarter = str(dataset.holdout_rows[0]["quarter"])
    baseline_hazard = baseline["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    hidden_hazard = hidden["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    assert hidden['hidden_diagnostics'] is not None
    assert hidden['hidden_diagnostics']['U_to_D']['gate_passed'] is False
    assert hidden['hidden_diagnostics']['U_to_D']['transition_weight'] == 0.0
    assert abs(hidden_hazard - baseline_hazard) <= 1e-12


def test_hidden_driver_transition_weights_zero_selected_transition() -> None:
    rows, structural_inputs, hidden_features = _synthetic_hidden_rows()
    dataset = build_blocked_time_dataset(rows, [2024])
    baseline = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0))
    hidden = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        structural_inputs=structural_inputs,
        hidden_driver_features=hidden_features,
        hidden_cfg=HiddenDriverConfig(
            precision_scale=0.25,
            residual_ridge=0.01,
            max_effect=0.4,
            rank_cap=1,
            transition_weights={'U_to_D': 0.0},
        ),
    )
    holdout_quarter = str(dataset.holdout_rows[0]["quarter"])
    baseline_hazard = baseline["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    hidden_hazard = hidden["hazard_paths"]["holdout_hazard_map"][holdout_quarter]["U_to_D"]
    assert hidden['hidden_diagnostics'] is not None
    assert hidden['hidden_diagnostics']['U_to_D']['transition_weight'] == 0.0
    assert abs(hidden_hazard - baseline_hazard) <= 1e-12


def test_incidence_sidecar_allows_total_state_growth() -> None:
    dataset = build_blocked_time_dataset(_synthetic_growth_rows(), [2024])
    closed = forecast_dynamic_baseline(dataset, DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0))
    open_result = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        incidence_cfg=IncidenceFlowConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
    )
    closed_total = float(sum(float(value) for value in dict(closed["trajectory_rows"][-1]["state_values"]).values()))
    open_total = float(sum(float(value) for value in dict(open_result["trajectory_rows"][-1]["state_values"]).values()))
    assert open_result["incidence_paths"] is not None
    assert open_total > closed_total
    assert open_result["model_contract"]["incidence_contract"] == "train_only_effective_population_incidence_to_U_with_evidence_typed_exit_channels"
    assert open_result["hazard_semantics"]["latent_incidence_inflow"]["status"] == "emitted"
    assert open_result["hazard_semantics"]["state_specific_exit_flows"]["status"] == "emitted"
    assert open_result["hazard_semantics"]["care_leakage_channels"]["status"] == "emitted"
    assert open_result["incidence_paths"]["holdout_incidence_hazard_map"]
    assert open_result["incidence_paths"]["holdout_state_attrition_outflow_map"]
    assert open_result["incidence_paths"]["holdout_exit_channel_state_outflow_map"]


def test_stock_balance_uses_s_eff_incidence_and_state_specific_exits() -> None:
    result = apply_stock_balance(
        {"U": 100.0, "D": 20.0, "A": 10.0, "V": 5.0, "L": 5.0},
        incidence_hazard=0.01,
        population_denominator=1_000.0,
        state_attrition_outflow={"U": 1.0, "D": 2.0, "A": 3.0, "V": 4.0, "L": 5.0},
    )

    balance = result["stock_balance"]
    state = result["state_values"]
    assert balance["susceptible_effective"] == 860.0
    assert balance["incidence_inflow"] == 8.6
    assert balance["attrition_outflow"] == 15.0
    assert state["U"] == 107.6
    assert state["D"] == 18.0
    assert state["A"] == 7.0
    assert state["V"] == 1.0
    assert state["L"] == 0.0


def test_stock_balance_preserves_exit_channel_labels_when_capping_by_state() -> None:
    result = apply_stock_balance(
        {"U": 0.0, "D": 5.0, "A": 0.0, "V": 0.0, "L": 0.0},
        exit_channel_state_outflows={
            "mortality_removal": {"D": 2.0},
            "treatment_non_initiation": {"D": 6.0},
        },
    )

    balance = result["stock_balance"]
    assert balance["state_attrition_outflows"]["D"] == 5.0
    assert balance["exit_channel_state_outflows"]["mortality_removal"]["D"] == 1.25
    assert balance["exit_channel_state_outflows"]["treatment_non_initiation"]["D"] == 3.75
    assert balance["exit_channel_outflows"]["mortality_removal"] == 1.25
    assert balance["exit_channel_outflows"]["treatment_non_initiation"] == 3.75
