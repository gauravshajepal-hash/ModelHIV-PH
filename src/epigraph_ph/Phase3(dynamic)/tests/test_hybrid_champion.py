from __future__ import annotations

from phase3_dynamic.hybrid_champion import (
    EndpointHead,
    _apply_endpoint_head,
    _base_head_family,
    _detect_shock_catalog,
    _delegate_family_for_horizon,
    _enforce_endpoint_cone,
    _fit_endpoint_blend_weight,
    _horizon_step_weights,
    _is_conditional_rate_family,
    _is_lifted_residual_family,
    _is_multihorizon_family,
    _metric_feature_row,
    _promotion_gate,
    _regime_summaries,
    _shock_aware_lifted_gate_from_evaluated,
    _annotate_residual_shock_entries,
    _shock_nowcast_feature_row,
    _state_feature_row,
    _uses_monthly_shock_hazard,
)


def test_endpoint_cone_enforces_cascade_order_and_nonnegative_flow() -> None:
    row = _enforce_endpoint_cone(
        {
            "quarter": "2026-Q1",
            "diagnosed_plhiv": 10.0,
            "alive_on_art": 20.0,
            "virally_suppressed": 30.0,
            "new_diagnosed_cases_period": -5.0,
        }
    )

    assert row["diagnosed_plhiv"] == 30.0
    assert row["alive_on_art"] == 20.0
    assert row["virally_suppressed"] == 20.0
    assert row["new_diagnosed_cases_period"] == 0.0


def test_r10_like_endpoint_features_are_lagged_and_train_only_inputs() -> None:
    features = _metric_feature_row(
        family="endpoint_r10_like_delta",
        raw_value=130.0,
        previous_value=100.0,
        previous_previous_value=80.0,
    )

    assert features == [1.0, 100.0, 30.0, 20.0]


def test_endpoint_blend_weight_is_fit_from_train_residuals() -> None:
    weight = _fit_endpoint_blend_weight(
        train_rows=[
            {"quarter": "2020-Q1", "diagnosed_plhiv": 100.0, "alive_on_art": 50.0, "new_diagnosed_cases_period": 10.0},
            {"quarter": "2020-Q2", "diagnosed_plhiv": 120.0, "alive_on_art": 60.0, "new_diagnosed_cases_period": 11.0},
        ],
        backbone_rows=[
            {"quarter": "2020-Q2", "diagnosed_plhiv": 100.0, "alive_on_art": 50.0, "new_diagnosed_cases_period": 10.0},
        ],
        endpoint_rows=[
            {"quarter": "2020-Q2", "diagnosed_plhiv": 120.0, "alive_on_art": 60.0, "new_diagnosed_cases_period": 11.0},
        ],
    )

    assert weight == 1.0


def test_state_multihorizon_feature_uses_latent_state_and_horizon_position() -> None:
    features = _state_feature_row(
        metric_name="diagnosed_plhiv",
        raw_row={
            "diagnosed_plhiv": 120.0,
            "incident_infections_period": 7.0,
            "net_attrition_outflow_period": 3.0,
        },
        trajectory_row={
            "state_values": {"U": 40.0, "D": 10.0, "A": 20.0, "V": 30.0, "L": 5.0},
            "stock_balance": {"incidence_inflow": 8.0, "attrition_outflow": 4.0},
        },
        previous_value=100.0,
        previous_previous_value=90.0,
        step_index=1,
        horizon_quarters=4,
    )

    assert features == [1.0, 120.0, 100.0, 10.0, 65.0, 65.0, 50.0, 30.0, 40.0, 105.0, 8.0, 4.0, 0.5]


def test_state_feature_can_append_bounded_shock_nowcast_drivers() -> None:
    features = _state_feature_row(
        metric_name="diagnosed_plhiv",
        raw_row={"diagnosed_plhiv": 120.0},
        trajectory_row={"state_values": {"U": 0.0, "D": 10.0, "A": 20.0, "V": 30.0, "L": 0.0}},
        previous_value=100.0,
        previous_previous_value=90.0,
        step_index=0,
        horizon_quarters=1,
        shock_features=[2.0, -2.0],
    )

    assert features[-2:] == [1.0, -1.0]


def test_lifted_residual_head_is_state_based_and_bounded() -> None:
    head = EndpointHead(
        family="monthly_joint_lifted_residual_state",
        metric_coefficients={
            "diagnosed_plhiv": [1000.0] + [0.0] * 12,
            "alive_on_art": [1000.0] + [0.0] * 12,
            "new_diagnosed_cases_period": [1000.0] + [0.0] * 12,
            "virally_suppressed": [1000.0] + [0.0] * 12,
        },
        train_row_count={
            "diagnosed_plhiv": 2,
            "alive_on_art": 2,
            "new_diagnosed_cases_period": 2,
            "virally_suppressed": 2,
        },
        feature_centers={
            "diagnosed_plhiv": [0.0] * 13,
            "alive_on_art": [0.0] * 13,
            "new_diagnosed_cases_period": [0.0] * 13,
            "virally_suppressed": [0.0] * 13,
        },
        feature_scales={
            "diagnosed_plhiv": [1.0] * 13,
            "alive_on_art": [1.0] * 13,
            "new_diagnosed_cases_period": [1.0] * 13,
            "virally_suppressed": [1.0] * 13,
        },
        correction_bounds={
            "diagnosed_plhiv": 10.0,
            "alive_on_art": 10.0,
            "new_diagnosed_cases_period": 10.0,
            "virally_suppressed": 10.0,
        },
    )

    rows = _apply_endpoint_head(
        head=head,
        train_rows=[
            {
                "quarter": "2020-Q4",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 80.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 40.0,
            }
        ],
        backbone_rows=[
            {
                "quarter": "2021-Q1",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 80.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 40.0,
            }
        ],
        trajectory_rows=[
            {"quarter": "2021-Q1", "state_values": {"U": 0.0, "D": 20.0, "A": 40.0, "V": 40.0, "L": 0.0}}
        ],
    )

    assert _is_multihorizon_family("monthly_joint_lifted_residual_state")
    assert _is_lifted_residual_family("monthly_joint_lifted_residual_state")
    assert rows[0]["diagnosed_plhiv"] == 110.0
    assert rows[0]["alive_on_art"] == 90.0
    assert rows[0]["new_diagnosed_cases_period"] == 15.0


def test_conditional_rate_head_derives_back_half_from_upstream_rates() -> None:
    head = EndpointHead(
        family="monthly_joint_state_multihorizon_conditional_rates",
        metric_coefficients={
            "diagnosed_plhiv": [120.0] + [0.0] * 12,
            "alive_on_art": [0.0] + [0.0] * 12,
            "new_diagnosed_cases_period": [8.0] + [0.0] * 12,
            "virally_suppressed": [0.0] + [0.0] * 12,
        },
        train_row_count={
            "diagnosed_plhiv": 2,
            "alive_on_art": 2,
            "new_diagnosed_cases_period": 2,
            "virally_suppressed": 2,
        },
        feature_centers={
            "diagnosed_plhiv": [0.0] * 13,
            "alive_on_art": [0.0] * 13,
            "new_diagnosed_cases_period": [0.0] * 13,
            "virally_suppressed": [0.0] * 13,
        },
        feature_scales={
            "diagnosed_plhiv": [1.0] * 13,
            "alive_on_art": [1.0] * 13,
            "new_diagnosed_cases_period": [1.0] * 13,
            "virally_suppressed": [1.0] * 13,
        },
        correction_bounds={
            "diagnosed_plhiv": 0.0,
            "alive_on_art": 0.0,
            "new_diagnosed_cases_period": 0.0,
            "virally_suppressed": 0.0,
        },
    )

    rows = _apply_endpoint_head(
        head=head,
        train_rows=[
            {
                "quarter": "2020-Q4",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 70.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 35.0,
            }
        ],
        backbone_rows=[
            {
                "quarter": "2021-Q1",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 70.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 35.0,
            }
        ],
        trajectory_rows=[
            {"quarter": "2021-Q1", "state_values": {"U": 0.0, "D": 20.0, "A": 40.0, "V": 40.0, "L": 0.0}}
        ],
    )

    assert _is_multihorizon_family("monthly_joint_state_multihorizon_conditional_rates")
    assert _is_conditional_rate_family("monthly_joint_state_multihorizon_conditional_rates")
    assert rows[0]["diagnosed_plhiv"] == 120.0
    assert rows[0]["alive_on_art"] == 60.0
    assert rows[0]["virally_suppressed"] == 30.0
    assert rows[0]["new_diagnosed_cases_period"] == 8.0


def test_shock_nowcast_feature_uses_monthly_deviation_and_support_shift() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "new_diagnosed_cases_period": 100.0,
            "metric_provenance": {
                "new_diagnosed_cases_period": {
                    "aggregation_mode": "monthly_to_quarter_sum",
                    "series_kind": "monthly_count",
                    "tier": "bridge_observed",
                }
            },
        },
        {
            "quarter": "2020-Q2",
            "new_diagnosed_cases_period": 105.0,
            "metric_provenance": {
                "new_diagnosed_cases_period": {
                    "aggregation_mode": "monthly_to_quarter_sum",
                    "series_kind": "monthly_count",
                    "tier": "bridge_observed",
                }
            },
        },
        {
            "quarter": "2020-Q3",
            "new_diagnosed_cases_period": 300.0,
            "metric_provenance": {
                "new_diagnosed_cases_period": {
                    "aggregation_mode": "quarterly_observed",
                    "series_kind": "quarterly_count",
                    "tier": "exact_observed",
                }
            },
        },
    ]

    features = _shock_nowcast_feature_row(rows, metric_name="new_diagnosed_cases_period", step_index=0)

    assert len(features) == 5
    assert features[0] > 0.0
    assert features[2] == 1.0


def test_shock_nowcast_readout_correction_is_data_bounded() -> None:
    head = EndpointHead(
        family="state_multihorizon_shock_nowcast",
        metric_coefficients={
            "diagnosed_plhiv": [1000.0] + [0.0] * 16,
            "alive_on_art": [1000.0] + [0.0] * 16,
            "new_diagnosed_cases_period": [1000.0] + [0.0] * 16,
            "virally_suppressed": [1000.0] + [0.0] * 16,
        },
        train_row_count={
            "diagnosed_plhiv": 2,
            "alive_on_art": 2,
            "new_diagnosed_cases_period": 2,
            "virally_suppressed": 2,
        },
        feature_centers={
            "diagnosed_plhiv": [0.0] * 17,
            "alive_on_art": [0.0] * 17,
            "new_diagnosed_cases_period": [0.0] * 17,
            "virally_suppressed": [0.0] * 17,
        },
        feature_scales={
            "diagnosed_plhiv": [1.0] * 17,
            "alive_on_art": [1.0] * 17,
            "new_diagnosed_cases_period": [1.0] * 17,
            "virally_suppressed": [1.0] * 17,
        },
        correction_bounds={
            "diagnosed_plhiv": 10.0,
            "alive_on_art": 10.0,
            "new_diagnosed_cases_period": 10.0,
            "virally_suppressed": 10.0,
        },
    )

    rows = _apply_endpoint_head(
        head=head,
        train_rows=[
            {
                "quarter": "2020-Q4",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 50.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 30.0,
            }
        ],
        backbone_rows=[
            {
                "quarter": "2021-Q1",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 50.0,
                "new_diagnosed_cases_period": 5.0,
                "virally_suppressed": 30.0,
            }
        ],
        trajectory_rows=[
            {"quarter": "2021-Q1", "state_values": {"U": 0.0, "D": 50.0, "A": 20.0, "V": 30.0, "L": 0.0}}
        ],
    )

    assert rows[0]["diagnosed_plhiv"] <= 110.0
    assert rows[0]["new_diagnosed_cases_period"] <= 15.0


def test_horizon_step_weights_drop_to_zero_when_step_has_no_training_support() -> None:
    weights = _horizon_step_weights(
        [
            {"step_index": 0},
            {"step_index": 0},
            {"step_index": 1},
        ]
    )

    assert weights == [1.0, 0.5]


def test_horizon_gated_family_uses_endpoint_short_and_state_long() -> None:
    assert _delegate_family_for_horizon("horizon_gated_endpoint_state", holdout_year_count=1) == "endpoint_raw_affine"
    assert _delegate_family_for_horizon("horizon_gated_endpoint_state", holdout_year_count=5) == "state_multihorizon_support_decay"
    assert _delegate_family_for_horizon("conditional_horizon_gated_endpoint_state", holdout_year_count=1) == "endpoint_raw_affine"
    assert _delegate_family_for_horizon("conditional_horizon_gated_endpoint_state", holdout_year_count=5) == "state_multihorizon_conditional_rates"


def test_monthly_shock_family_keeps_same_readout_family_but_routes_hazard_layer() -> None:
    assert _base_head_family("monthly_shock_horizon_gated_endpoint_state") == "horizon_gated_endpoint_state"
    assert _base_head_family("monthly_latent_horizon_gated_endpoint_state") == "horizon_gated_endpoint_state"
    assert _base_head_family("monthly_joint_horizon_gated_endpoint_state") == "horizon_gated_endpoint_state"
    assert _base_head_family("monthly_joint_lifted_residual_state") == "lifted_residual_state"
    assert _base_head_family("backhalf_channel_horizon_gated_endpoint_state") == "horizon_gated_endpoint_state"
    assert _uses_monthly_shock_hazard("monthly_shock_state_multihorizon_support_decay")
    assert not _uses_monthly_shock_hazard("state_multihorizon_support_decay")
    assert _delegate_family_for_horizon("monthly_shock_horizon_gated_endpoint_state", holdout_year_count=1) == "monthly_shock_endpoint_raw_affine"
    assert _delegate_family_for_horizon("monthly_shock_horizon_gated_endpoint_state", holdout_year_count=5) == "monthly_shock_state_multihorizon_support_decay"
    assert _delegate_family_for_horizon("monthly_latent_horizon_gated_endpoint_state", holdout_year_count=1) == "monthly_latent_endpoint_raw_affine"
    assert _delegate_family_for_horizon("monthly_latent_horizon_gated_endpoint_state", holdout_year_count=5) == "monthly_latent_state_multihorizon_support_decay"
    assert _delegate_family_for_horizon("monthly_joint_horizon_gated_endpoint_state", holdout_year_count=1) == "monthly_joint_endpoint_raw_affine"
    assert _delegate_family_for_horizon("monthly_joint_horizon_gated_endpoint_state", holdout_year_count=5) == "monthly_joint_state_multihorizon_support_decay"
    assert _delegate_family_for_horizon("monthly_joint_conditional_horizon_gated_endpoint_state", holdout_year_count=1) == "monthly_joint_endpoint_raw_affine"
    assert _delegate_family_for_horizon("monthly_joint_conditional_horizon_gated_endpoint_state", holdout_year_count=5) == "monthly_joint_state_multihorizon_conditional_rates"
    assert _delegate_family_for_horizon("backhalf_channel_horizon_gated_endpoint_state", holdout_year_count=1) == "backhalf_channel_endpoint_raw_affine"
    assert _delegate_family_for_horizon("backhalf_channel_horizon_gated_endpoint_state", holdout_year_count=5) == "backhalf_channel_state_multihorizon_support_decay"


def test_hybrid_promotion_requires_carry_forward_and_bounded_scenario_gates() -> None:
    gate = _promotion_gate(
        score={
            "split_count": 2,
            "candidate_mean_mae": 0.2,
            "carry_forward_mean_mae": 0.3,
            "candidate_worst_mae": 0.4,
            "carry_forward_worst_mae": 0.5,
        },
        near_horizon_gate={"status": "pass"},
        long_horizon_status={"status": "pass"},
    )
    assert gate["status"] == "promote"

    failed = _promotion_gate(
        score={
            "split_count": 2,
            "candidate_mean_mae": 0.2,
            "carry_forward_mean_mae": 0.3,
            "candidate_worst_mae": 0.4,
            "carry_forward_worst_mae": 0.5,
        },
        near_horizon_gate={"status": "fail"},
        long_horizon_status={"status": "pass"},
    )
    assert failed["status"] == "reject"
    assert "fails_bounded_near_horizon_scenario_gate" in failed["blockers"]


def test_shock_catalog_detects_non_covid_signal_jump() -> None:
    rows = [
        {"quarter": "2018-Q4", "diagnosed_plhiv": 100.0},
        {"quarter": "2019-Q4", "diagnosed_plhiv": 110.0},
        {"quarter": "2020-Q4", "diagnosed_plhiv": 120.0},
        {"quarter": "2021-Q4", "diagnosed_plhiv": 130.0},
        {"quarter": "2022-Q4", "diagnosed_plhiv": 600.0},
        {"quarter": "2023-Q4", "diagnosed_plhiv": 620.0},
    ]

    catalog = _detect_shock_catalog(rows)

    assert "2022" in catalog["year_labels"]
    assert "detected_signal_shock:diagnosed_plhiv" in catalog["year_labels"]["2022"]


def test_shock_aware_lifted_gate_requires_beating_carry_and_r10_on_path() -> None:
    evaluated = [
        {
            "family": "synthetic",
            "train_end_year": 2021,
            "holdout_years": [2022],
            "metric_scales": {"diagnosed_plhiv": 100.0},
            "holdout_rows": [
                {"quarter": "2022-Q1", "diagnosed_plhiv": 100.0},
                {"quarter": "2022-Q2", "diagnosed_plhiv": 100.0},
            ],
            "prediction_rows": [
                {"quarter": "2022-Q1", "diagnosed_plhiv": 110.0},
                {"quarter": "2022-Q2", "diagnosed_plhiv": 110.0},
            ],
            "carry_forward_prediction_rows": [
                {"quarter": "2022-Q1", "diagnosed_plhiv": 130.0},
                {"quarter": "2022-Q2", "diagnosed_plhiv": 130.0},
            ],
        }
    ]
    catalog = {"year_labels": {"2022": ["detected_signal_shock:diagnosed_plhiv"]}}
    r10 = {
        "score_rows": [
            {
                "archive_variant": "merged",
                "experiment_id": "EXP-R10",
                "contract": "exact_only",
                "quarterly_mean_mae": 0.2,
            }
        ]
    }

    gate = _shock_aware_lifted_gate_from_evaluated(
        family="synthetic",
        evaluated_rows=evaluated,
        shock_catalog=catalog,
        r10_baseline=r10,
    )

    assert gate["status"] == "pass"
    assert gate["overall"]["candidate_minus_carry_forward_mean_mae"] < 0.0

    strict_r10 = {"score_rows": [{**r10["score_rows"][0], "quarterly_mean_mae": 0.05}]}
    failed = _shock_aware_lifted_gate_from_evaluated(
        family="synthetic",
        evaluated_rows=evaluated,
        shock_catalog=catalog,
        r10_baseline=strict_r10,
    )

    assert failed["status"] == "fail"
    assert "candidate_lifted_path_mean_not_better_than_r10_reference" in failed["blockers"]


def test_residual_shock_labels_are_data_adaptive_not_event_specific() -> None:
    entries = [
        {"regime_labels": ["stable"], "carry_forward_mae": 0.1, "candidate_mae": 0.1},
        {"regime_labels": ["stable"], "carry_forward_mae": 0.1, "candidate_mae": 0.1},
        {"regime_labels": ["stable"], "carry_forward_mae": 0.1, "candidate_mae": 0.1},
        {"regime_labels": ["stable"], "carry_forward_mae": 1.0, "candidate_mae": 0.2},
    ]

    annotated = _annotate_residual_shock_entries(entries)
    summaries = _regime_summaries(annotated)

    assert "residual_shock:carry_forward_path_error_outlier" in annotated[-1]["regime_labels"]
    assert any(row["regime_label"] == "residual_shock:carry_forward_path_error_outlier" for row in summaries)
