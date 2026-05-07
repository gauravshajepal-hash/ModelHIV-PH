from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from phase3_dynamic.r11_sparse_state_space import (
    _apply_horizon_adaptive_constrained_shape_head,
    _candidate_predictions,
    _conditional_rate_gate,
    _fit_datv_transition_process,
    _fit_era_datv_transition_process,
    _fit_back_half_rate_model,
    _fit_horizon_adaptive_shape_selector,
    _fit_linkage_lag_kernel,
    _fit_r19_lineage_rate_state_model,
    _fit_r20_service_capacity_transition,
    _fit_r21_diagnosis_flow_selector,
    _fit_r25_predictive_endpoint_head_from_records,
    _r22_select_program_metric_policy,
    _fit_r34_art_ratio_selector,
    _r34_art_ratio_guarded_predictions,
    _r34_art_ratio_prediction,
    _r35_art_ratio_flow_guarded_predictions,
    _fit_r36_metric_policy_selector,
    _r36_component_policy_cascade_predictions,
    _r39_fixed_policy_predictions,
    _r40_p90_policy_selector_predictions,
    _r41_growth_phase_policy,
    _r41_monotone_growth_component_predictions,
    _fit_trajectory_shape_head,
    _build_r12_04_source_lineage_ablation_report,
    _build_r12_05_lineage_stratified_contract_report,
    _build_r12_06_doh_quarterly_support_adjudication_report,
    _build_r12_07_horizon_specific_evidence_router_report,
    _build_r12_08_route_aware_two_head_candidate_report,
    _build_r12_official_annual_challenge_gate_report,
    _fit_r12_annual_anchor_head_selector,
    _predict_back_half_rate,
    _predict_r19_lineage_rate_state,
    _predict_r20_service_capacity_transition,
    _r11_multi_horizon_report,
    _r10_lifted_gate,
    _select_horizon_matched_r10_reference,
    _fit_support_partition_calibration_model,
    _predict_support_partition_calibration,
    project_cascade_stock_row,
    stock_consistency_gate,
)
from phase3_dynamic.r13_priority_experiments import _r13_priority_experiment_specs
from phase3_dynamic.r26_r10_teacher_fusion import _fit_metric_policy as _fit_r26_metric_policy
from phase3_dynamic.r27_r19_r10_complementarity import (
    _fit_metric_policy as _fit_r27_metric_policy,
    _summary_row as _r27_summary_row,
)
from phase3_dynamic.r28_r10_contract_lineage_audit import (
    _lineage_id as _r28_lineage_id,
    _provenance_coverage_rows as _r28_provenance_coverage_rows,
)
from phase3_dynamic.r29_strict_ledger_matched_r10_gate import (
    _is_program_lineage as _r29_is_program_lineage,
    _score_records as _r29_score_records,
)
from phase3_dynamic.r30_strict_r13_family_scan import (
    _reference_scope_for_row_scope as _r30_reference_scope_for_row_scope,
    _scan_r13_results as _r30_scan_r13_results,
)
from phase3_dynamic.r31_strict_policy_probe import (
    _median_velocity as _r31_median_velocity,
    _positive_velocity as _r31_positive_velocity,
)
from phase3_dynamic.r32_strict_composite_replay import (
    _candidate_gate_rows as _r32_candidate_gate_rows,
    _candidate_value as _r32_candidate_value,
)
from phase3_dynamic.r33_h5_backhalf_process_audit import (
    _art_policy_prediction as _r33_art_policy_prediction,
    _rate_policy_prediction as _r33_rate_policy_prediction,
    _score_art_policy_records as _r33_score_art_policy_records,
    _score_rate_policy_records as _r33_score_rate_policy_records,
)
from phase3_dynamic.r34_art_ratio_aggregate_replay import (
    _aggregate_candidate_rows as _r34_aggregate_candidate_rows,
)
from phase3_dynamic.r37_fixed_policy_strict_scan import (
    _candidate_specs as _r37_candidate_specs,
    _predict_candidate_row as _r37_predict_candidate_row,
)
from phase3_dynamic.r38_train_selected_policy_strict_gate import (
    _select_candidate_from_train as _r38_select_candidate_from_train,
)
from phase3_dynamic.r42_r41_champion_hardening import (
    _ablate_training_rows as _r42_ablate_training_rows,
    _geography_claim_report as _r42_geography_claim_report,
    _source_family_counts as _r42_source_family_counts,
    _source_family_from_provenance as _r42_source_family_from_provenance,
)
from phase3_dynamic.r43_subnational_evidence_intake import (
    _parse_regional_cascade_line as _r43_parse_regional_cascade_line,
)
from phase3_dynamic.r44_subnational_horizon_gate import (
    _period_from_pdf_name as _r44_period_from_pdf_name,
    _subnational_horizon_gate as _r44_subnational_horizon_gate,
)
from phase3_dynamic.r45_phase01215_determinant_readiness_audit import (
    _capability_status as _r45_capability_status,
    _scenario_readiness as _r45_scenario_readiness,
)
from phase3_dynamic.r46_phase2_lineage_driver_gate import (
    _edge_driver_status as _r46_edge_driver_status,
    _lineage_gate as _r46_lineage_gate,
    _module_targets_for_edge as _r46_module_targets_for_edge,
)
from phase3_dynamic.r48_subnational_proxy_champion_contract import (
    _gate as _r48_gate,
    _projection_cascade as _r48_projection_cascade,
    _similarity_kernel as _r48_similarity_kernel,
)
from phase3_dynamic.r49_subnational_module_champion_gate import (
    _gate as _r49_gate,
    _module_local_prediction_rows as _r49_module_local_prediction_rows,
    _module_selector_table as _r49_module_selector_table,
)
from phase3_dynamic.r50_subnational_residual_anatomy import (
    _gate as _r50_gate,
    _metric_best_family_table as _r50_metric_best_family_table,
    _region_best_family_table as _r50_region_best_family_table,
    _region_metric_score_table as _r50_region_metric_score_table,
)
from phase3_dynamic.r51_train_only_regional_selector import (
    _gate as _r51_gate,
    _select_family_from_prior_scores as _r51_select_family_from_prior_scores,
    _train_only_selector_prediction_rows as _r51_train_only_selector_prediction_rows,
)
from phase3_dynamic.r52_subnational_coherence_gate import (
    _coherence_rows as _r52_coherence_rows,
    _gate as _r52_gate,
)
from phase3_dynamic.r53_publication_claim_registry import (
    _adapter_stability_claim as _r53_adapter_stability_claim,
    _determinant_claim as _r53_determinant_claim,
    _regional_adapter_claim as _r53_regional_adapter_claim,
    _registry_gate as _r53_registry_gate,
)
from phase3_dynamic.r54_national_total_regional_adapter import (
    _adapter_prediction_rows as _r54_adapter_prediction_rows,
    _gate as _r54_gate,
)
from phase3_dynamic.r55_adapter_split_stability_gate import (
    _comparison_rows as _r55_comparison_rows,
    _gate as _r55_gate,
)
from phase3_dynamic.r56_split_guarded_regional_selector import (
    _gate as _r56_gate,
    _select_family_from_prior_scores as _r56_select_family_from_prior_scores,
    _split_guarded_prediction_rows as _r56_split_guarded_prediction_rows,
)
from phase3_dynamic.r57_regional_candidate_ceiling_diagnostic import (
    _ceiling_prediction_rows as _r57_ceiling_prediction_rows,
    _ceiling_selection_rows as _r57_ceiling_selection_rows,
    _gate as _r57_gate,
)
from phase3_dynamic.r58_pareto_simplex_regional_ensemble import (
    _fit_simplex_weights as _r58_fit_simplex_weights,
    _gate as _r58_gate,
    _pareto_candidate_families as _r58_pareto_candidate_families,
)
from phase3_dynamic.r59_anchor_stable_pareto_ensemble import (
    _anchor_stable_prediction_rows as _r59_anchor_stable_prediction_rows,
    _gate as _r59_gate,
)
from phase3_dynamic.r60_regional_experiment_queue import (
    CARRY_FORWARD_FAMILY as _r60_carry_forward_family,
    REFERENCE_FAMILY as _r60_reference_family,
    _candidate_mean_gate as _r60_candidate_mean_gate,
    _candidate_set_families as _r60_candidate_set_families,
    _experiment_specs as _r60_experiment_specs,
)
from phase3_dynamic.r61_split_risk_regional_router import (
    _copy_metric_from_family as _r61_copy_metric_from_family,
    _r60_nonregression as _r61_r60_nonregression,
    _r61_experiment_specs as _r61_experiment_specs,
)
from phase3_dynamic.r62_leakage_expert_student_gate import (
    _adapter_components as _r62_adapter_components,
    _component_prior_family as _r62_component_prior_family,
    _frequency_prior_family as _r62_frequency_prior_family,
    _most_recent_prior_family as _r62_most_recent_prior_family,
)
from phase3_dynamic.r63_region_metric_online_expert_gate import (
    _candidate_error_rows as _r63_candidate_error_rows,
    _region_metric_student_prediction_rows as _r63_region_metric_student_prediction_rows,
)
from phase3_dynamic.r64_leakage_support_gap_prioritizer import (
    _aggregate_priority as _r64_aggregate_priority,
    _evidence_recommendation as _r64_evidence_recommendation,
    _gap_rows as _r64_gap_rows,
)
from phase3_dynamic.r65_transmission_model_readiness_gate import (
    _gate as _r65_gate,
    _module_status_rows as _r65_module_status_rows,
    _state_equation_rows as _r65_state_equation_rows,
)
from phase3_dynamic.r66_scientific_source_base import (
    _classify_local_source as _r66_classify_local_source,
    _gate as _r66_gate,
    _module_coverage_rows as _r66_module_coverage_rows,
    _support_gap_rows as _r66_support_gap_rows,
    _wdi_specs as _r66_wdi_specs,
)
from phase3_dynamic.r67_transmission_model_family_queue import (
    _gate as _r67_gate,
    _model_family_rows as _r67_model_family_rows,
    _next_experiment_rows as _r67_next_experiment_rows,
)
from phase3_dynamic.r68_bulk_external_source_ingest import (
    _bulk_source_specs as _r68_bulk_source_specs,
    _extract_philippines_from_csv_stream as _r68_extract_philippines_from_csv_stream,
    _gate as _r68_gate,
)
from phase3_dynamic.r69_bulk_signal_feature_compiler import (
    _compile_google_quarterly_rows as _r69_compile_google_quarterly_rows,
    _compile_unaids_estimates as _r69_compile_unaids_estimates,
    _gate as _r69_gate,
    _readiness_rows as _r69_readiness_rows,
)
from phase3_dynamic.r70_scientific_model_build_queue import (
    _gate as _r70_gate,
    _queue_rows as _r70_queue_rows,
)
from phase3_dynamic.r71_service_intensity_capacity_branch import (
    R71_METRICS as _r71_metrics,
    _compile_feature_rows as _r71_compile_feature_rows,
    _feature_predictions as _r71_feature_predictions,
    _fit_linear_feature_model as _r71_fit_linear_feature_model,
    _gate as _r71_gate,
)
from phase3_dynamic.r72_service_feature_selector_gate import (
    _gate as _r72_gate,
    _select_metric_families as _r72_select_metric_families,
)
from phase3_dynamic.r73_external_signal_lag_falsification import (
    _feature_catalog as _r73_feature_catalog,
    _feature_family as _r73_feature_family,
    _gate as _r73_gate,
)
from phase3_dynamic.r74_external_tail_risk_selector import (
    _gate as _r74_gate,
    _metric_summary as _r74_metric_summary,
)
from phase3_dynamic.r75_bulk_unaids_annual_challenge import (
    _bulk_unaids_target_rows as _r75_bulk_unaids_target_rows,
    _gate as _r75_gate,
    _merge_external_targets_into_observations as _r75_merge_external_targets_into_observations,
    _rolling_annual_splits as _r75_rolling_annual_splits,
)
from phase3_dynamic.r76_public_domain_annual_comparator import (
    PUBLIC_COMPARATOR_FAMILIES as _r76_public_comparator_families,
    _gate as _r76_gate,
    _log_linear_prediction_rows as _r76_log_linear_prediction_rows,
    _select_metric_families as _r76_select_metric_families,
)
from phase3_dynamic.r77_public_proxy_annual_gate import (
    _gate as _r77_gate,
    _matched_proxy_rows as _r77_matched_proxy_rows,
)
from phase3_dynamic.r78_public_annual_family_expansion import (
    _fit_piecewise_log_linear_metric as _r78_fit_piecewise_log_linear_metric,
    _fit_theil_sen_log_metric as _r78_fit_theil_sen_log_metric,
    _gate as _r78_gate,
    _predict_piecewise_log_linear_metric as _r78_predict_piecewise_log_linear_metric,
    _predict_theil_sen_log_metric as _r78_predict_theil_sen_log_metric,
)
from phase3_dynamic.r79_expanded_public_proxy_annual_gate import (
    _expanded_gate as _r79_expanded_gate,
)
from phase3_dynamic.r80_public_annual_projection_head import (
    _gate as _r80_gate,
    _horizon_bucket as _r80_horizon_bucket,
    _residual_quantiles as _r80_residual_quantiles,
)
from phase3_dynamic.r81_phase2_knob_admissibility_gate import (
    _driver_kind as _r81_driver_kind,
    _gate as _r81_gate,
    _knob_rows as _r81_knob_rows,
)
from phase3_dynamic.r82_quarterly_annual_bridge_gate import (
    _bridge_rows as _r82_bridge_rows,
    _gate as _r82_gate,
)
from phase3_dynamic.r83_quarterly_emission_bridge_audit import (
    _annualized_candidate_value as _r83_annualized_candidate_value,
    _gate as _r83_gate,
)
from phase3_dynamic.r84_conserved_quarterly_annual_ledger import _gate as _r84_gate
from phase3_dynamic.r85_annual_ledger_forecast_grid import (
    _forecast_grid_rows as _r85_forecast_grid_rows,
    _gate as _r85_gate,
    _quarter_grid as _r85_quarter_grid,
)
from phase3_dynamic.r86_annual_calibrated_forecast_grid_ledger import (
    _distribute_annual_total_by_quarter_shape as _r86_distribute_annual_total_by_quarter_shape,
    _gate as _r86_gate,
)
from phase3_dynamic.model import _apply_observation_model as _phase3_apply_observation_model
from phase3_dynamic.model import _simulate_sequence as _phase3_simulate_sequence


def test_r13_priority_experiment_specs_are_ordered_and_claim_scoped() -> None:
    specs = _r13_priority_experiment_specs()

    assert len(specs) == 50
    assert [spec["priority"] for spec in specs] == list(range(1, 51))
    assert specs[0]["experiment_id"] == "R13-001"
    assert specs[-1]["experiment_id"] == "R13-050"
    assert all(spec.get("hypothesis") for spec in specs)
    assert all(spec.get("metrics") for spec in specs)
    assert any(spec["family"] == "official_annual_challenge_gate" for spec in specs)
    assert any(spec["family"] == "r14_two_factor_program_process" for spec in specs)
    assert any(spec["family"] == "r19_joint_service_cascade_process" for spec in specs)
    assert any(spec["family"] == "r22_program_metric_coupled_process" for spec in specs)
    assert any(spec["family"] == "r20_service_capacity_process" for spec in specs)
    assert any(spec["family"] == "r21_diagnosis_flow_guarded_process" for spec in specs)
    assert any(spec.get("phase2_status") == "locked_until_source_stable" for spec in specs)


def test_r26_metric_policy_requires_worst_case_nonregression() -> None:
    records = [
        {
            "metric_name": "alive_on_art",
            "target_value": 10.0,
            "r10_value": 9.0,
            "carry_forward_value": 10.0,
            "scale": 1.0,
        },
        {
            "metric_name": "alive_on_art",
            "target_value": 10.0,
            "r10_value": 10.0,
            "carry_forward_value": 12.0,
            "scale": 1.0,
        },
    ]

    policy = _fit_r26_metric_policy(records, "alive_on_art")

    assert policy["kind"] == "identity"


def test_r27_metric_policy_can_select_r19_when_train_safe() -> None:
    records = [
        {
            "metric_name": "alive_on_art",
            "target_value": 20.0,
            "r10_value": 10.0,
            "r19_value": 20.0,
            "carry_forward_value": 12.0,
            "scale": 1.0,
        },
        {
            "metric_name": "alive_on_art",
            "target_value": 30.0,
            "r10_value": 15.0,
            "r19_value": 30.0,
            "carry_forward_value": 18.0,
            "scale": 1.0,
        },
    ]

    policy = _fit_r27_metric_policy(records, "alive_on_art")

    assert policy["kind"] in {"r19", "r10_r19_blend"}


def test_r27_summary_row_rejects_oracle_tie_with_r10() -> None:
    row = _r27_summary_row(
        row_scope="program_best_phase3",
        source_row={
            "horizon_years": 5,
            "family": "r22_program_metric_coupled_process",
            "r10_comparable_candidate_mean_mae": 0.18,
            "r10_horizon_reference_mae": 0.13,
        },
    )

    assert row["fusion_mean_mae"] == 0.13
    assert row["status"] == "fail"
    assert "oracle_not_strictly_better_than_matched_r10" in row["blockers"]


def test_r28_lineage_id_preserves_source_measurement_and_cadence() -> None:
    lineage = _r28_lineage_id(
        {
            "source_quality_tier": "official_doh_archive",
            "measurement_class": "program_observed_harp",
            "series_kind": "quarterly_snapshot",
        }
    )

    assert lineage == "official_doh_archive|program_observed_harp|quarterly_snapshot"


def test_r28_provenance_coverage_flags_unmapped_entries() -> None:
    rows = _r28_provenance_coverage_rows(
        [
            {"horizon_years": 5, "unknown_provenance": False},
            {"horizon_years": 5, "unknown_provenance": True},
            {"horizon_years": 3, "unknown_provenance": True},
        ]
    )

    by_horizon = {row["horizon_years"]: row for row in rows}
    assert by_horizon[5]["unknown_provenance_count"] == 1
    assert by_horizon[5]["unknown_provenance_share"] == 0.5
    assert by_horizon[3]["unknown_provenance_share"] == 1.0


def test_r29_program_lineage_filter_accepts_only_doh_program_rows() -> None:
    assert _r29_is_program_lineage(
        {"source_lineage": "official_doh_archive|program_observed_harp|quarterly_snapshot"}
    )
    assert not _r29_is_program_lineage(
        {"source_lineage": "official_user_provided_slide|program_observed_harp|annual_snapshot"}
    )


def test_r29_score_records_computes_strict_mapped_r10_reference() -> None:
    score = _r29_score_records(
        [
            {"r10_norm_error": 0.1, "carry_forward_norm_error": 0.3},
            {"r10_norm_error": 0.2, "carry_forward_norm_error": 0.1},
        ]
    )

    assert score["entry_count"] == 2
    assert score["matched_r10_mean_mae"] == pytest.approx(0.15)
    assert score["carry_forward_mean_mae"] == pytest.approx(0.2)
    assert score["matched_r10_better_share"] == pytest.approx(0.5)


def test_r30_reference_scope_maps_program_to_program_reference() -> None:
    assert _r30_reference_scope_for_row_scope("program") == "program_mapped"
    assert _r30_reference_scope_for_row_scope("all") == "all_mapped"


def test_r30_scan_requires_all_required_horizons() -> None:
    rows = _r30_scan_r13_results(
        r13_report={
            "results": [
                {
                    "experiment_id": "R13-X",
                    "family": "candidate",
                    "row_scope": "all",
                    "decision": "diagnostic",
                    "horizon_rows": [
                        {"horizon_years": 1, "r10_comparable_candidate_mean_mae": 0.05},
                    ],
                }
            ]
        },
        strict_reference={("all_mapped", 1): 0.1},
    )

    assert rows[0]["status"] == "fail"
    assert "missing_required_horizons_h3_h5" in rows[0]["blockers"]


def test_r31_velocity_helpers_are_train_derived() -> None:
    history = [
        (1, 2020, 10.0),
        (2, 2020, 12.0),
        (3, 2020, 11.0),
        (4, 2020, 15.0),
    ]

    assert _r31_positive_velocity(history) == pytest.approx(3.0)
    assert _r31_median_velocity(history) == pytest.approx(2.0)


def test_r32_positive_flow_composite_uses_train_policy_only_for_flow() -> None:
    history = {
        "new_diagnosed_cases_period": [
            (8080, 2020, 10.0),
            (8081, 2020, 12.0),
        ],
        "diagnosed_plhiv": [],
        "alive_on_art": [],
    }
    record = {
        "metric_name": "new_diagnosed_cases_period",
        "train_end_year": 2020,
        "quarter": "2020-Q3",
    }
    value = _r32_candidate_value(
        record=record,
        candidate_row={"new_diagnosed_cases_period": 1.0},
        candidate={
            "candidate_id": "flow",
            "base_family": "r18_evidence_backed_art_process",
            "flow_policy": "positive_velocity",
        },
        history_by_metric=history,
    )

    assert value == pytest.approx(14.0)


def test_r32_candidate_gate_requires_all_strict_scopes_to_pass() -> None:
    scored_rows = [
        {
            "candidate_id": "candidate",
            "horizon_years": horizon,
            "candidate_norm_error": 0.05,
            "carry_forward_norm_error": 0.20,
            "unknown_provenance": False,
            "source_lineage": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        }
        for horizon in (1, 3, 5)
    ]
    strict_reference_rows = [
        {
            "scope": scope,
            "horizon_years": horizon,
            "matched_r10_mean_mae": 0.10,
        }
        for scope, horizons in {"all_mapped": (1, 3, 5), "program_mapped": (3, 5)}.items()
        for horizon in horizons
    ]

    rows = _r32_candidate_gate_rows(
        scored_rows_by_candidate={"candidate": scored_rows},
        strict_reference_rows=strict_reference_rows,
    )

    assert len(rows) == 5
    assert all(row["status"] == "pass" for row in rows)


def test_r33_ratio_art_policy_is_train_only_and_stock_cone_safe() -> None:
    history = {
        "diagnosed_plhiv": [
            (8080, 2020, 100.0),
            (8081, 2020, 120.0),
        ],
        "alive_on_art": [],
        "new_diagnosed_cases_period": [],
    }
    ratio_history = [
        (8080, 2020, 0.50),
        (8081, 2020, 0.60),
    ]
    record = {
        "train_end_year": 2020,
        "quarter": "2020-Q3",
    }

    value = _r33_art_policy_prediction(
        record=record,
        policy_id="diagnosed_ratio_positive_velocity__rate_carry_forward",
        history_by_metric=history,
        art_diagnosed_ratio_history=ratio_history,
    )

    assert value == pytest.approx(84.0)


def test_r33_rate_policy_uses_only_train_history() -> None:
    history = [
        (8080, 2020, 0.20),
        (8081, 2020, 0.30),
        (8082, 2021, 0.90),
    ]

    value = _r33_rate_policy_prediction(
        history=history,
        train_end_year=2020,
        target_quarter="2020-Q3",
        policy_id="carry_forward",
    )

    assert value == pytest.approx(0.30)


def test_r33_art_score_blocks_strict_r10_regression() -> None:
    records = [
        {
            "train_end_year": 2020,
            "quarter": "2020-Q2",
            "target_value": 100.0,
            "scale": 100.0,
            "r10_norm_error": 0.01,
            "carry_forward_norm_error": 0.30,
            "diagnosed_target_value": 150.0,
        }
    ]
    history = {
        "alive_on_art": [
            (8079, 2020, 50.0),
            (8080, 2020, 60.0),
        ],
        "diagnosed_plhiv": [],
        "new_diagnosed_cases_period": [],
    }

    score = _r33_score_art_policy_records(
        records=records,
        policy_id="direct_carry_forward",
        history_by_metric=history,
        art_diagnosed_ratio_history=[],
    )

    assert score["status"] == "fail"
    assert "policy_not_better_than_strict_r10" in score["blockers"]


def test_r33_rate_carry_forward_is_baseline_not_promotional() -> None:
    records = [
        {
            "train_end_year": 2020,
            "quarter": "2020-Q2",
            "target_rate": 0.5,
        }
    ]
    history = [
        (8079, 2020, 0.4),
        (8080, 2020, 0.5),
    ]

    score = _r33_score_rate_policy_records(
        records=records,
        history=history,
        policy_id="carry_forward",
    )

    assert score["status"] == "fail"
    assert "conditional_carry_forward_is_baseline_not_promotional" in score["blockers"]


def test_r34_art_ratio_prediction_clips_to_output_diagnosed_stock() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
        },
        {
            "quarter": "2020-Q2",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
        },
    ]

    value = _r34_art_ratio_prediction(
        rows,
        "2020-Q3",
        "diagnosed_ratio_positive_velocity__rate_carry_forward",
        output_diagnosed_value=80.0,
    )

    assert value == pytest.approx(80.0)


def test_r34_selector_fails_closed_when_no_policy_beats_carry_forward() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
        },
    ]

    selector = _fit_r34_art_ratio_selector(rows, max_horizon_years=1)

    assert selector["status"] == "failed_closed"
    assert selector["selected_policy_id"] == "none"


def test_r34_guarded_predictions_only_mutate_art_and_back_half_rates() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 12.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 140.0,
            "alive_on_art": 98.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 49.0,
            "virally_suppressed": 40.0,
        },
    ]
    holdout_rows = [
        {
            "quarter": "2023-Q1",
            "diagnosed_plhiv": 160.0,
            "alive_on_art": 128.0,
            "new_diagnosed_cases_period": 16.0,
            "tested_for_viral_load": 64.0,
            "virally_suppressed": 52.0,
        }
    ]

    rows, summary = _r34_art_ratio_guarded_predictions(train_rows, holdout_rows)

    assert rows
    assert rows[0]["diagnosed_plhiv"] >= rows[0]["alive_on_art"]
    assert rows[0]["alive_on_art"] >= rows[0]["tested_for_viral_load"]
    assert rows[0]["tested_for_viral_load"] >= rows[0]["virally_suppressed"]
    assert summary["base_family"] == "r19_joint_service_cascade_process"


def test_r34_aggregate_replay_replaces_only_art_metric() -> None:
    r32_lookup = {
        ("program_mapped", 5, "diagnosed_plhiv"): {
            "candidate_mean_mae": 0.10,
            "entry_count": 10,
        },
        ("program_mapped", 5, "alive_on_art"): {
            "candidate_mean_mae": 0.20,
            "entry_count": 10,
        },
        ("program_mapped", 5, "new_diagnosed_cases_period"): {
            "candidate_mean_mae": 0.10,
            "entry_count": 10,
        },
    }
    r33_lookup = {
        ("program_mapped", 5, "alive_on_art"): {
            "candidate_mean_mae": 0.05,
            "entry_count": 10,
            "source": "r33",
        }
    }
    strict_reference_rows = [
        {
            "scope": "program_mapped",
            "horizon_years": 5,
            "matched_r10_mean_mae": 0.09,
            "carry_forward_mean_mae": 0.30,
        }
    ]

    gate_rows, metric_rows = _r34_aggregate_candidate_rows(
        r32_lookup=r32_lookup,
        r33_art_lookup=r33_lookup,
        strict_reference_rows=strict_reference_rows,
    )

    program_h5 = [
        row
        for row in gate_rows
        if row["scope"] == "program_mapped" and row["horizon_years"] == 5
    ][0]
    art_row = [
        row
        for row in metric_rows
        if row["scope"] == "program_mapped" and row["horizon_years"] == 5 and row["metric_name"] == "alive_on_art"
    ][0]

    assert program_h5["candidate_mean_mae"] == pytest.approx(0.0833333333)
    assert art_row["candidate_mean_mae"] == pytest.approx(0.05)
    assert art_row["source"] == "r33"


def test_r35_flow_repair_is_train_positive_velocity_only() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 140.0,
            "alive_on_art": 98.0,
            "new_diagnosed_cases_period": 18.0,
            "tested_for_viral_load": 49.0,
            "virally_suppressed": 40.0,
        },
    ]
    holdout_rows = [
        {
            "quarter": "2023-Q1",
            "diagnosed_plhiv": 160.0,
            "alive_on_art": 128.0,
            "new_diagnosed_cases_period": 22.0,
            "tested_for_viral_load": 64.0,
            "virally_suppressed": 52.0,
        }
    ]

    rows, summary = _r35_art_ratio_flow_guarded_predictions(train_rows, holdout_rows)

    assert rows[0]["new_diagnosed_cases_period"] == pytest.approx(22.0)
    assert summary["flow_policy_id"] == "positive_velocity"


def test_r36_metric_selector_requires_worst_case_nonregression() -> None:
    rows = [
        {"quarter": "2020-Q1", "diagnosed_plhiv": 100.0},
        {"quarter": "2021-Q1", "diagnosed_plhiv": 100.0},
        {"quarter": "2022-Q1", "diagnosed_plhiv": 100.0},
    ]

    selector = _fit_r36_metric_policy_selector(
        rows,
        metric_name="diagnosed_plhiv",
        max_horizon_years=1,
    )

    assert selector["status"] == "failed_closed"
    assert selector["selected_policy_id"] == "carry_forward"


def test_r36_component_policy_cascade_preserves_stock_cone() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 140.0,
            "alive_on_art": 98.0,
            "new_diagnosed_cases_period": 18.0,
            "tested_for_viral_load": 49.0,
            "virally_suppressed": 40.0,
        },
    ]
    holdout_rows = [{"quarter": "2023-Q1"}]

    rows, summary = _r36_component_policy_cascade_predictions(train_rows, holdout_rows)

    assert rows[0]["diagnosed_plhiv"] >= rows[0]["alive_on_art"]
    assert rows[0]["alive_on_art"] >= rows[0]["tested_for_viral_load"]
    assert rows[0]["tested_for_viral_load"] >= rows[0]["virally_suppressed"]
    assert set(summary["selected_policies"]) == {
        "diagnosed_plhiv",
        "new_diagnosed_cases_period",
        "alive_on_art",
    }


def test_r37_candidate_specs_are_bounded_and_predeclared() -> None:
    specs = _r37_candidate_specs()

    assert len(specs) == 54
    assert all(spec["candidate_id"] for spec in specs)
    assert {
        "candidate_id",
        "diagnosed_policy",
        "flow_policy",
        "art_policy",
    } == set(specs[0])


def test_r37_fixed_policy_prediction_preserves_cascade_cone() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
    ]
    process = {
        "status": "completed",
        "rate_models": {
            "vl_tested_among_art": {
                "status": "completed",
                "selected_variant": "carry_rate",
                "last_rate": 0.5,
            },
            "suppressed_among_vl_tested": {
                "status": "completed",
                "selected_variant": "carry_rate",
                "last_rate": 0.8,
            },
        },
    }

    row = _r37_predict_candidate_row(
        train_rows=train_rows,
        holdout_row={"quarter": "2022-Q1"},
        back_half_process=process,
        candidate={
            "candidate_id": "candidate",
            "diagnosed_policy": "positive_velocity",
            "flow_policy": "positive_velocity",
            "art_policy": "diagnosed_ratio_positive_velocity__rate_logit_velocity",
        },
    )

    assert row["diagnosed_plhiv"] >= row["alive_on_art"]
    assert row["alive_on_art"] >= row["tested_for_viral_load"]
    assert row["tested_for_viral_load"] >= row["virally_suppressed"]


def test_r38_selector_uses_internal_train_records_only() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 140.0,
            "alive_on_art": 98.0,
            "new_diagnosed_cases_period": 18.0,
            "tested_for_viral_load": 49.0,
            "virally_suppressed": 40.0,
        },
    ]

    selector = _r38_select_candidate_from_train(rows, max_horizon_years=1)

    assert selector["status"] in {"completed", "fallback_no_internal_records"}
    assert selector["selected_candidate"]["candidate_id"]


def test_r39_fixed_policy_preserves_stock_cone() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
    ]
    holdout_rows = [{"quarter": "2022-Q1"}]

    rows, summary = _r39_fixed_policy_predictions(train_rows, holdout_rows)

    assert rows[0]["diagnosed_plhiv"] >= rows[0]["alive_on_art"]
    assert rows[0]["alive_on_art"] >= rows[0]["tested_for_viral_load"]
    assert rows[0]["tested_for_viral_load"] >= rows[0]["virally_suppressed"]
    assert summary["fixed_policy"]["flow_policy"] == "positive_velocity"


def test_r40_p90_selector_preserves_stock_cone() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 140.0,
            "alive_on_art": 98.0,
            "new_diagnosed_cases_period": 18.0,
            "tested_for_viral_load": 49.0,
            "virally_suppressed": 40.0,
        },
    ]
    holdout_rows = [{"quarter": "2023-Q1"}]

    rows, summary = _r40_p90_policy_selector_predictions(train_rows, holdout_rows)

    assert rows[0]["diagnosed_plhiv"] >= rows[0]["alive_on_art"]
    assert rows[0]["alive_on_art"] >= rows[0]["tested_for_viral_load"]
    assert rows[0]["tested_for_viral_load"] >= rows[0]["virally_suppressed"]
    assert set(summary["selected_policies"]) == {
        "diagnosed_plhiv",
        "new_diagnosed_cases_period",
        "alive_on_art",
    }


def test_r41_growth_phase_policy_is_train_sign_based() -> None:
    growing = [
        {"quarter": "2020-Q1", "new_diagnosed_cases_period": 10.0},
        {"quarter": "2021-Q1", "new_diagnosed_cases_period": 12.0},
    ]
    shrinking = [
        {"quarter": "2020-Q1", "new_diagnosed_cases_period": 12.0},
        {"quarter": "2021-Q1", "new_diagnosed_cases_period": 10.0},
    ]

    assert _r41_growth_phase_policy(growing, "new_diagnosed_cases_period") == "positive_velocity"
    assert _r41_growth_phase_policy(shrinking, "new_diagnosed_cases_period") == "median_velocity"


def test_r41_monotone_growth_component_preserves_stock_cone() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 25.0,
            "virally_suppressed": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 72.0,
            "new_diagnosed_cases_period": 14.0,
            "tested_for_viral_load": 36.0,
            "virally_suppressed": 30.0,
        },
    ]
    holdout_rows = [{"quarter": "2022-Q1"}]

    rows, summary = _r41_monotone_growth_component_predictions(train_rows, holdout_rows)

    assert rows[0]["diagnosed_plhiv"] >= rows[0]["alive_on_art"]
    assert rows[0]["alive_on_art"] >= rows[0]["tested_for_viral_load"]
    assert rows[0]["tested_for_viral_load"] >= rows[0]["virally_suppressed"]
    assert summary["selected_policy"]["flow_policy"] == "positive_velocity"


def test_r42_source_family_ablation_masks_training_metrics_only() -> None:
    source_family = _r42_source_family_from_provenance(
        {
            "source_tier": "official_doh_archive",
            "measurement_class": "program_observed_harp",
            "series_kind": "monthly_snapshot",
        }
    )
    rows = [
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 80.0,
            "new_diagnosed_cases_period": 10.0,
            "metric_provenance": {
                "diagnosed_plhiv": {
                    "source_tier": "official_doh_archive",
                    "measurement_class": "program_observed_harp",
                    "series_kind": "monthly_snapshot",
                },
                "alive_on_art": {
                    "source_tier": "official_user_provided_slide",
                    "measurement_class": "program_observed_harp",
                    "series_kind": "annual_snapshot",
                },
            },
        }
    ]

    ablated, summary = _r42_ablate_training_rows(rows, source_family)

    assert ablated[0]["diagnosed_plhiv"] is None
    assert ablated[0]["alive_on_art"] == 80.0
    assert summary["removed_metric_counts"] == {"diagnosed_plhiv": 1}
    assert rows[0]["diagnosed_plhiv"] == 100.0


def test_r42_source_family_counts_use_r10_comparable_metric_provenance() -> None:
    rows = [
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 100.0,
            "metric_provenance": {
                "diagnosed_plhiv": {
                    "source_tier": "official_doh_archive",
                    "measurement_class": "program_observed_harp",
                    "series_kind": "quarterly_snapshot",
                },
                "population_total": {
                    "source_tier": "official_wdi_population",
                    "measurement_class": "external_reference_population",
                    "series_kind": "annual_series",
                },
            },
        }
    ]

    counts = _r42_source_family_counts(rows)

    assert counts == [
        {
            "source_family": "official_doh_archive|program_observed_harp|quarterly_snapshot",
            "entry_count": 1,
            "metric_counts": {"diagnosed_plhiv": 1},
        }
    ]


def test_r42_geography_claim_boundary_blocks_subnational_claim_without_direct_support() -> None:
    report = _r42_geography_claim_report(
        {
            "rows": [
                {
                    "geography": "national",
                    "observation_role": "direct_target",
                    "metric_id": "diagnosed_plhiv",
                }
            ]
        }
    )

    assert report["status"] == "national_only_validation"
    assert report["nonnational_geographies"] == []
    assert "Subnational modeling" in report["claim_boundary"]


def test_r43_parse_regional_cascade_line_extracts_hasp_anchor() -> None:
    row = _r43_parse_regional_cascade_line(
        "       NCR          73,700        46,695         63%          29,094        62%         16,468         57%            14,692        89%           50%"
    )

    assert row == {
        "region": "NCR",
        "estimated_plhiv": 73700,
        "diagnosed_plhiv": 46695,
        "first_95": 0.63,
        "alive_on_art": 29094,
        "second_95": 0.62,
        "tested_for_viral_load": 16468,
        "vl_testing_coverage": 0.57,
        "vl_testing_coverage_semantics": "reported_percent",
        "virally_suppressed": 14692,
        "suppression_among_tested": 0.89,
        "third_95": 0.5,
    }


def test_r43_parse_regional_cascade_line_derives_missing_vl_coverage() -> None:
    row = _r43_parse_regional_cascade_line(
        "       NCR          60,800        41,472         68%          26,015        63%         13,096            11,716        89%           45%"
    )

    assert row is not None
    assert row["region"] == "NCR"
    assert row["tested_for_viral_load"] == 13096
    assert row["alive_on_art"] == 26015
    assert row["vl_testing_coverage"] == pytest.approx(13096 / 26015)
    assert row["vl_testing_coverage_semantics"] == "derived_vl_tested_over_art_no_reported_column"
    assert row["third_95"] == 0.45


def test_r44_subnational_horizon_gate_requires_comparable_periods() -> None:
    single_period_rows = [
        {"region": "NCR", "period_id": "2025-Q2", "diagnosed_plhiv": 10.0},
        {"region": "Region 3", "period_id": "2025-Q2", "diagnosed_plhiv": 5.0},
    ]
    single = _r44_subnational_horizon_gate(single_period_rows)

    assert single["status"] == "single_period_auxiliary_only"
    assert "need_at_least_two_regional_cascade_periods_for_blocked_subnational_gate" in single["blockers"]

    multi = _r44_subnational_horizon_gate(
        [
            *single_period_rows,
            {"region": "NCR", "period_id": "2024-Q2", "diagnosed_plhiv": 8.0},
            {"region": "Region 3", "period_id": "2024-Q2", "diagnosed_plhiv": 4.0},
        ]
    )

    assert multi["status"] == "multi_period_subnational_gate_ready"
    assert multi["overlapping_regions"] == ["NCR", "Region 3"]


def test_r44_period_parser_uses_hasp_quarter_name() -> None:
    period = _r44_period_from_pdf_name(Path("2025_Q2-HIV-AIDS-Surveillance-Report-of-the-Philippines.pdf"))

    assert period["period_id"] == "2025-Q2"
    assert period["time_end"] == "2025-06"
    assert period["time_granularity"] == "quarterly_snapshot"

    reversed_period = _r44_period_from_pdf_name(Path("HASP-Q3-2024-1.pdf"))
    assert reversed_period["period_id"] == "2024-Q3"
    assert reversed_period["time_end"] == "2024-09"


def test_r45_capability_and_readiness_contract(tmp_path) -> None:
    source_file = tmp_path / "src" / "epigraph_ph" / "phase2" / "edge_falsification.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("placebo ablation time_window falsification\n", encoding="utf-8")
    capability = {
        "capability_id": "edge_gate",
        "phase": "phase2",
        "relative_path": "src/epigraph_ph/phase2/edge_falsification.py",
        "evidence_tokens": ("placebo", "ablation", "time_window", "falsification"),
        "scientific_role": "test",
    }

    row = _r45_capability_status(tmp_path, capability)
    readiness = _r45_scenario_readiness(
        [row],
        {"structural_payload_count": 1, "falsification_artifact_count": 1},
    )

    assert row["status"] == "present"
    assert row["missing_tokens"] == []
    assert readiness["status"] == "ready_for_phase3_scenario_interface"

    blocked = _r45_scenario_readiness([row], {"structural_payload_count": 0, "falsification_artifact_count": 1})
    assert blocked["status"] == "blocked_pending_phase2_payload_and_falsification"
    assert "missing_phase2_structural_payload_artifact" in blocked["blockers"]


def test_r46_lineage_gate_blocks_source_stable_edges_that_fail_time_windows() -> None:
    edge = {
        "edge_kind": "direct",
        "source": "mobility_exposure_pressure",
        "target": "structural_barrier_pressure",
        "support_ablation_passed": True,
        "source_reestimated_passed": True,
        "blocked_time_passed": False,
        "sign_conflict_family_count": 0,
    }
    row = {"driver_status": _r46_edge_driver_status(edge)}
    gate = _r46_lineage_gate([row], {"completed_source_family_count": 5})

    assert row["driver_status"] == "source_stable_but_time_blocked_sensitivity_only"
    assert gate["status"] == "sensitivity_only_determinant_scenarios"
    assert "no_direct_edge_survives_strict_phase3_prior_gate" in gate["blockers"]
    assert _r46_module_targets_for_edge(edge) == ["ART_retention", "D_to_A", "U_to_D", "incidence"]


def test_r46_lineage_gate_promotes_strict_direct_prior() -> None:
    edge = {
        "edge_kind": "direct",
        "phase3_default_allowed": True,
        "source": "care_access_continuity",
        "target": "suppression_capacity",
    }
    row = {"driver_status": _r46_edge_driver_status(edge)}
    gate = _r46_lineage_gate([row], {"completed_source_family_count": 2})

    assert row["driver_status"] == "strict_phase3_prior"
    assert gate["status"] == "strict_determinant_priors_ready"
    assert gate["blockers"] == []


def test_r48_similarity_kernel_is_leave_self_out_and_normalized() -> None:
    distances, weights, bandwidth = _r48_similarity_kernel(
        feature_matrix=np.asarray([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float64),
        regions=["A", "B", "C"],
    )

    assert bandwidth > 0.0
    assert distances.shape == (3, 3)
    assert weights.shape == (3, 3)
    assert np.allclose(np.diag(weights), 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 1] > weights[0, 2]


def test_r48_projection_cascade_enforces_stock_cone() -> None:
    projected = _r48_projection_cascade(
        {
            "estimated_plhiv": 50.0,
            "diagnosed_plhiv": 80.0,
            "alive_on_art": 90.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 120.0,
        }
    )

    assert projected["estimated_plhiv"] == 80.0
    assert projected["diagnosed_plhiv"] == 80.0
    assert projected["alive_on_art"] == 80.0
    assert projected["tested_for_viral_load"] == 70.0
    assert projected["virally_suppressed"] == 70.0


def test_r48_gate_promotes_only_if_beats_required_baselines() -> None:
    promoted = _r48_gate(
        [
            {
                "candidate_family": "similarity_proxy_log_delta",
                "mean_normalized_absolute_error": 0.1,
            },
            {
                "candidate_family": "aggregate_log_trend",
                "mean_normalized_absolute_error": 0.2,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_normalized_absolute_error": 0.3,
            },
        ]
    )
    blocked = _r48_gate(
        [
            {
                "candidate_family": "regional_carry_forward",
                "mean_normalized_absolute_error": 0.1,
            },
            {
                "candidate_family": "similarity_proxy_log_delta",
                "mean_normalized_absolute_error": 0.2,
            },
        ]
    )

    assert promoted["status"] == "subnational_proxy_champion_promoted"
    assert promoted["blockers"] == []
    assert blocked["status"] == "no_subnational_proxy_champion"
    assert "best_candidate_is_carry_forward" in blocked["blockers"]


def test_r49_module_selector_promotes_only_metric_local_improvements() -> None:
    rows = []
    for metric in [
        "estimated_plhiv",
        "diagnosed_plhiv",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
    ]:
        rows.extend(
            [
                {
                    "candidate_family": "regional_carry_forward",
                    "holdout_period": "2025-Q2",
                    "metric_name": metric,
                    "normalized_absolute_error": 0.20,
                },
                {
                    "candidate_family": "aggregate_log_trend",
                    "holdout_period": "2025-Q2",
                    "metric_name": metric,
                    "normalized_absolute_error": 0.10 if metric == "alive_on_art" else 0.25,
                },
                {
                    "candidate_family": "similarity_proxy_log_delta",
                    "holdout_period": "2025-Q2",
                    "metric_name": metric,
                    "normalized_absolute_error": 0.15 if metric == "diagnosed_plhiv" else 0.30,
                },
            ]
        )

    table = _r49_module_selector_table(rows)
    by_metric = {row["metric_name"]: row for row in table}

    assert by_metric["alive_on_art"]["status"] == "module_candidate_promoted"
    assert by_metric["alive_on_art"]["selected_candidate_family"] == "aggregate_log_trend"
    assert by_metric["diagnosed_plhiv"]["status"] == "module_candidate_promoted"
    assert by_metric["diagnosed_plhiv"]["selected_candidate_family"] == "similarity_proxy_log_delta"
    assert by_metric["estimated_plhiv"]["status"] == "module_locked_to_carry_forward"
    assert by_metric["estimated_plhiv"]["selected_candidate_family"] == "regional_carry_forward"


def test_r49_module_local_predictions_reproject_mixed_metrics_into_stock_cone() -> None:
    prediction_rows = [
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q2",
            "region": "NCR",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 90.0,
            "alive_on_art": 80.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 60.0,
        },
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2025-Q2",
            "region": "NCR",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 90.0,
            "alive_on_art": 120.0,
            "tested_for_viral_load": 110.0,
            "virally_suppressed": 95.0,
        },
    ]
    selector = [
        {"metric_name": "estimated_plhiv", "selected_candidate_family": "regional_carry_forward"},
        {"metric_name": "diagnosed_plhiv", "selected_candidate_family": "regional_carry_forward"},
        {"metric_name": "alive_on_art", "selected_candidate_family": "aggregate_log_trend"},
        {"metric_name": "tested_for_viral_load", "selected_candidate_family": "aggregate_log_trend"},
        {"metric_name": "virally_suppressed", "selected_candidate_family": "aggregate_log_trend"},
    ]

    rows = _r49_module_local_prediction_rows(prediction_rows, selector)

    assert len(rows) == 1
    assert rows[0]["estimated_plhiv"] == 100.0
    assert rows[0]["diagnosed_plhiv"] == 90.0
    assert rows[0]["alive_on_art"] == 90.0
    assert rows[0]["tested_for_viral_load"] == 90.0
    assert rows[0]["virally_suppressed"] == 90.0
    assert rows[0]["projection_adjusted"] is True
    assert rows[0]["alive_on_art_source_candidate_family"] == "aggregate_log_trend"


def test_r49_gate_requires_module_selector_to_beat_r48_best() -> None:
    selector_rows = [
        {
            "metric_name": "alive_on_art",
            "status": "module_candidate_promoted",
        }
    ]
    promoted = _r49_gate(
        module_selector_table=selector_rows,
        combined_candidate_table=[
            {"candidate_family": "module_local_selector", "mean_normalized_absolute_error": 0.08},
            {"candidate_family": "aggregate_log_trend", "mean_normalized_absolute_error": 0.10},
            {"candidate_family": "regional_carry_forward", "mean_normalized_absolute_error": 0.20},
        ],
    )
    blocked = _r49_gate(
        module_selector_table=selector_rows,
        combined_candidate_table=[
            {"candidate_family": "aggregate_log_trend", "mean_normalized_absolute_error": 0.07},
            {"candidate_family": "module_local_selector", "mean_normalized_absolute_error": 0.08},
            {"candidate_family": "regional_carry_forward", "mean_normalized_absolute_error": 0.20},
        ],
    )

    assert promoted["status"] == "module_local_subnational_champion_promoted"
    assert promoted["blockers"] == []
    assert blocked["status"] == "module_local_selector_diagnostic_only"
    assert "module_selector_regresses_vs_best_single_family_r48" in blocked["blockers"]


def test_r50_region_residual_anatomy_detects_region_proxy_signal() -> None:
    score_rows = [
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2025-Q2",
            "region": "A",
            "metric_name": "alive_on_art",
            "absolute_error": 10.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "similarity_proxy_log_delta",
            "holdout_period": "2025-Q2",
            "region": "A",
            "metric_name": "alive_on_art",
            "absolute_error": 5.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q2",
            "region": "A",
            "metric_name": "alive_on_art",
            "absolute_error": 20.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "module_local_selector",
            "holdout_period": "2025-Q2",
            "region": "A",
            "metric_name": "alive_on_art",
            "absolute_error": 8.0,
            "target_abs": 100.0,
        },
    ]

    region_metric = _r50_region_metric_score_table(score_rows)
    metric_best = _r50_metric_best_family_table(region_metric)
    region_best = _r50_region_best_family_table(
        [
            {
                "candidate_family": row["candidate_family"],
                "region": row["region"],
                "mean_metric_normalized_absolute_error": row["normalized_absolute_error"],
            }
            for row in region_metric
        ]
    )
    gate = _r50_gate(
        region_best,
        [
            {
                "metric_name": "alive_on_art",
                "best_candidate_family": "similarity_proxy_log_delta",
            }
        ],
    )

    assert region_best[0]["similarity_beats_aggregate"] is True
    assert metric_best[0]["best_candidate_family"] == "similarity_proxy_log_delta"
    assert gate["status"] == "residual_anatomy_ready_for_region_specific_selector"
    assert gate["regions_where_similarity_proxy_beats_aggregate"] == ["A"]


def test_r51_prior_selector_uses_only_earlier_blocked_scores() -> None:
    score_rows = [
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "absolute_error": 20.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "similarity_proxy_log_delta",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "absolute_error": 10.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q2",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "absolute_error": 0.0,
            "target_abs": 100.0,
        },
    ]

    early_family, early_decision = _r51_select_family_from_prior_scores(
        score_rows,
        region="NCR",
        metric="alive_on_art",
        holdout_period="2024-Q3",
        default_family="aggregate_log_trend",
    )
    late_family, late_decision = _r51_select_family_from_prior_scores(
        score_rows,
        region="NCR",
        metric="alive_on_art",
        holdout_period="2025-Q2",
        default_family="aggregate_log_trend",
    )

    assert early_family == "aggregate_log_trend"
    assert early_decision["selection_status"] == "default_no_prior_region_metric_scores"
    assert late_family == "similarity_proxy_log_delta"
    assert late_decision["selection_status"] == "selected_from_prior_region_metric_scores"


def test_r51_train_only_predictions_record_decisions_and_project_cone() -> None:
    prediction_rows = [
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 90.0,
            "alive_on_art": 80.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 60.0,
        },
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2025-Q2",
            "region": "NCR",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 90.0,
            "alive_on_art": 80.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 60.0,
        },
        {
            "candidate_family": "similarity_proxy_log_delta",
            "holdout_period": "2025-Q2",
            "region": "NCR",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 90.0,
            "alive_on_art": 120.0,
            "tested_for_viral_load": 110.0,
            "virally_suppressed": 95.0,
        },
    ]
    score_rows = [
        {
            "candidate_family": "similarity_proxy_log_delta",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "absolute_error": 1.0,
            "target_abs": 100.0,
        },
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "absolute_error": 10.0,
            "target_abs": 100.0,
        },
    ]

    rows, decisions = _r51_train_only_selector_prediction_rows(
        prediction_rows=prediction_rows,
        score_rows=score_rows,
        default_family="aggregate_log_trend",
    )

    by_period = {row["holdout_period"]: row for row in rows}
    late_art_decision = [
        row
        for row in decisions
        if row["holdout_period"] == "2025-Q2" and row["metric_name"] == "alive_on_art"
    ][0]
    assert by_period["2025-Q2"]["alive_on_art_source_candidate_family"] == "similarity_proxy_log_delta"
    assert by_period["2025-Q2"]["alive_on_art"] == 90.0
    assert by_period["2025-Q2"]["projection_adjusted"] is True
    assert late_art_decision["selection_status"] == "selected_from_prior_region_metric_scores"


def test_r51_gate_blocks_if_selector_regresses_vs_aggregate() -> None:
    blocked = _r51_gate(
        [
            {"candidate_family": "aggregate_log_trend", "mean_normalized_absolute_error": 0.08},
            {"candidate_family": "train_only_region_metric_selector", "mean_normalized_absolute_error": 0.09},
            {"candidate_family": "regional_carry_forward", "mean_normalized_absolute_error": 0.20},
        ],
        [{"selection_status": "selected_from_prior_region_metric_scores"}],
    )

    assert blocked["status"] == "train_only_regional_selector_diagnostic_only"
    assert "selector_regresses_vs_aggregate_trend" in blocked["blockers"]


def test_r52_coherence_rows_separate_mass_and_share_error() -> None:
    rows_by_key = {
        ("2025-Q2", "A"): {"estimated_plhiv": 100.0, "diagnosed_plhiv": 80.0, "alive_on_art": 60.0, "tested_for_viral_load": 40.0, "virally_suppressed": 30.0},
        ("2025-Q2", "B"): {"estimated_plhiv": 100.0, "diagnosed_plhiv": 20.0, "alive_on_art": 10.0, "tested_for_viral_load": 5.0, "virally_suppressed": 4.0},
    }
    prediction_rows = [
        {
            "candidate_family": "x",
            "holdout_period": "2025-Q2",
            "region": "A",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 50.0,
            "alive_on_art": 60.0,
            "tested_for_viral_load": 40.0,
            "virally_suppressed": 30.0,
        },
        {
            "candidate_family": "x",
            "holdout_period": "2025-Q2",
            "region": "B",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 50.0,
            "alive_on_art": 10.0,
            "tested_for_viral_load": 5.0,
            "virally_suppressed": 4.0,
        },
    ]

    rows = _r52_coherence_rows(rows_by_key, prediction_rows)
    diagnosed = [row for row in rows if row["metric_name"] == "diagnosed_plhiv"][0]

    assert diagnosed["aggregate_mass_normalized_absolute_error"] == pytest.approx(0.0)
    assert diagnosed["regional_share_half_l1_error"] == pytest.approx(0.3)


def test_r52_gate_requires_regional_mass_and_share_nonregression() -> None:
    promoted = _r52_gate(
        [
            {
                "candidate_family": "train_only_region_metric_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.03,
                "mean_regional_share_half_l1_error": 0.02,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.06,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.03,
            },
        ]
    )
    blocked = _r52_gate(
        [
            {
                "candidate_family": "train_only_region_metric_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.05,
                "mean_regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.06,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.03,
            },
        ]
    )

    assert promoted["status"] == "coherent_subnational_champion_promoted"
    assert promoted["blockers"] == []
    assert blocked["status"] == "subnational_selector_not_coherent_enough"
    assert "selector_worsens_aggregate_mass_error" in blocked["blockers"]
    assert "selector_worsens_regional_share_error" in blocked["blockers"]


def test_r53_registry_separates_promoted_champions_from_sensitivity_determinants(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    adapter = _r53_regional_adapter_claim(
        {
            "status": "national_total_regional_adapter_promoted",
            "adapter_gate": {
                "status": "national_total_regional_adapter_promoted",
                "promoted_candidate_family": "national_total_adapter__total=x__share=y",
            },
        },
        artifact,
    )
    stability = _r53_adapter_stability_claim(
        {
            "status": "adapter_mean_promoted_but_split_stability_limited",
            "split_stability_gate": {
                "status": "adapter_mean_promoted_but_split_stability_limited",
                "split_metric_count": 30,
            },
        },
        artifact,
    )
    determinant = _r53_determinant_claim(
        {
            "status": "sensitivity_only_determinant_scenarios",
            "lineage_gate": {
                "status": "sensitivity_only_determinant_scenarios",
                "strict_phase3_prior_count": 0,
                "sensitivity_only_driver_count": 3,
                "completed_source_family_count": 5,
            },
        },
        artifact,
    )
    gate = _r53_registry_gate(
        [
            {"claim_id": "national_r41_research_champion", "claim_status": "promoted"},
            {"claim_id": "regional_hasp_cascade_readout_champion", "claim_status": "promoted"},
            adapter,
            stability,
            determinant,
        ]
    )

    assert adapter["claim_status"] == "promoted"
    assert stability["claim_status"] == "mean_promoted_split_limited"
    assert determinant["claim_status"] == "sensitivity_only"
    assert gate["status"] == "national_and_subnational_readout_champions_ready_determinants_sensitivity_only"
    assert gate["regional_adapter_promoted"] is True
    assert gate["blockers"] == []


def test_r54_adapter_uses_forecast_total_and_forecast_shares() -> None:
    prediction_rows = [
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q2",
            "region": "A",
            "estimated_plhiv": 60.0,
            "diagnosed_plhiv": 50.0,
            "alive_on_art": 40.0,
            "tested_for_viral_load": 20.0,
            "virally_suppressed": 10.0,
        },
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q2",
            "region": "B",
            "estimated_plhiv": 40.0,
            "diagnosed_plhiv": 30.0,
            "alive_on_art": 20.0,
            "tested_for_viral_load": 10.0,
            "virally_suppressed": 5.0,
        },
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2025-Q2",
            "region": "A",
            "estimated_plhiv": 90.0,
            "diagnosed_plhiv": 54.0,
            "alive_on_art": 36.0,
            "tested_for_viral_load": 18.0,
            "virally_suppressed": 9.0,
        },
        {
            "candidate_family": "aggregate_log_trend",
            "holdout_period": "2025-Q2",
            "region": "B",
            "estimated_plhiv": 10.0,
            "diagnosed_plhiv": 6.0,
            "alive_on_art": 4.0,
            "tested_for_viral_load": 2.0,
            "virally_suppressed": 1.0,
        },
    ]

    rows, diagnostics = _r54_adapter_prediction_rows(
        prediction_rows,
        total_families=("regional_carry_forward",),
        share_families=("aggregate_log_trend",),
    )

    by_region = {row["region"]: row for row in rows}
    assert diagnostics[0]["status"] == "completed"
    assert by_region["A"]["estimated_plhiv"] == pytest.approx(90.0)
    assert by_region["B"]["estimated_plhiv"] == pytest.approx(10.0)
    assert by_region["A"]["estimated_plhiv_national_total_source_family"] == "regional_carry_forward"
    assert by_region["A"]["estimated_plhiv_regional_share_source_family"] == "aggregate_log_trend"


def test_r54_gate_requires_no_regression_against_r52_reference() -> None:
    promoted = _r54_gate(
        [
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.10,
                "mean_aggregate_mass_normalized_absolute_error": 0.10,
                "mean_regional_share_half_l1_error": 0.10,
            },
            {
                "candidate_family": "national_total_adapter__total=x__share=y",
                "mean_regional_normalized_absolute_error": 0.08,
                "mean_aggregate_mass_normalized_absolute_error": 0.08,
                "mean_regional_share_half_l1_error": 0.08,
            },
        ],
        r52_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.09,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.09,
            "promoted_mean_regional_share_half_l1_error": 0.09,
        },
    )
    blocked = _r54_gate(
        [
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.10,
                "mean_aggregate_mass_normalized_absolute_error": 0.10,
                "mean_regional_share_half_l1_error": 0.10,
            },
            {
                "candidate_family": "national_total_adapter__total=x__share=y",
                "mean_regional_normalized_absolute_error": 0.095,
                "mean_aggregate_mass_normalized_absolute_error": 0.08,
                "mean_regional_share_half_l1_error": 0.08,
            },
        ],
        r52_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.09,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.09,
            "promoted_mean_regional_share_half_l1_error": 0.09,
        },
    )

    assert promoted["status"] == "national_total_regional_adapter_promoted"
    assert promoted["blockers"] == []
    assert blocked["status"] == "national_total_regional_adapter_diagnostic_only"
    assert "no_national_total_adapter_beats_carry_forward_and_r52_reference" in blocked["blockers"]


def test_r55_split_stability_gate_counts_local_regressions() -> None:
    comparison = _r55_comparison_rows(
        regional_score_rows=[
            {
                "candidate_family": "promoted",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "normalized_absolute_error": 0.10,
            },
            {
                "candidate_family": "regional_carry_forward",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "normalized_absolute_error": 0.11,
            },
            {
                "candidate_family": "module_local_selector",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "normalized_absolute_error": 0.09,
            },
        ],
        coherence_rows=[
            {
                "candidate_family": "promoted",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "aggregate_mass_normalized_absolute_error": 0.05,
                "regional_share_half_l1_error": 0.03,
            },
            {
                "candidate_family": "regional_carry_forward",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "aggregate_mass_normalized_absolute_error": 0.06,
                "regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "module_local_selector",
                "holdout_period": "2025-Q2",
                "metric_name": "alive_on_art",
                "aggregate_mass_normalized_absolute_error": 0.04,
                "regional_share_half_l1_error": 0.02,
            },
        ],
        promoted_family="promoted",
    )
    gate = _r55_gate(comparison)

    assert comparison[0]["regional_nonregression_vs_carry_forward"] is True
    assert comparison[0]["regional_nonregression_vs_reference"] is False
    assert gate["status"] == "adapter_mean_promoted_but_split_stability_limited"
    assert gate["failure_counts"]["regional_nonregression_vs_reference"] == 1


def test_r56_split_guarded_selector_uses_only_prior_splits() -> None:
    selected, reason = _r56_select_family_from_prior_scores(
        [
            {
                "candidate_family": "a",
                "prior_scored_period_count": 1,
                "prior_strict_pass_count": 0,
                "prior_failed_check_count": 1,
                "prior_mean_regional_normalized_absolute_error": 0.01,
                "prior_mean_aggregate_mass_normalized_absolute_error": 0.01,
                "prior_mean_regional_share_half_l1_error": 0.01,
            },
            {
                "candidate_family": "b",
                "prior_scored_period_count": 1,
                "prior_strict_pass_count": 1,
                "prior_failed_check_count": 0,
                "prior_mean_regional_normalized_absolute_error": 0.02,
                "prior_mean_aggregate_mass_normalized_absolute_error": 0.02,
                "prior_mean_regional_share_half_l1_error": 0.02,
            },
        ]
    )
    assert selected == "b"
    assert reason == "prior_all_splits_strict_nonregression"

    prediction_rows = []
    for period in ("2025-Q1", "2025-Q2"):
        for family, value in (
            ("module_local_selector", 10.0),
            ("regional_carry_forward", 11.0),
            ("adapter", 8.0 if period == "2025-Q2" else 12.0),
        ):
            prediction_rows.append(
                {
                    "candidate_family": family,
                    "holdout_period": period,
                    "region": "A",
                    "estimated_plhiv": 20.0,
                    "diagnosed_plhiv": value,
                    "alive_on_art": 7.0,
                    "tested_for_viral_load": 6.0,
                    "virally_suppressed": 5.0,
                }
            )
    regional_score_rows = [
        {
            "candidate_family": "module_local_selector",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "normalized_absolute_error": 0.05,
        },
        {
            "candidate_family": "regional_carry_forward",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "normalized_absolute_error": 0.10,
        },
        {
            "candidate_family": "adapter",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "normalized_absolute_error": 0.01,
        },
    ]
    coherence_rows = [
        {
            "candidate_family": family,
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "aggregate_mass_normalized_absolute_error": mass,
            "regional_share_half_l1_error": share,
        }
        for family, mass, share in (
            ("module_local_selector", 0.05, 0.05),
            ("regional_carry_forward", 0.10, 0.10),
            ("adapter", 0.01, 0.01),
        )
    ]

    rows, selection_rows, prior_rows = _r56_split_guarded_prediction_rows(
        prediction_rows,
        regional_score_rows=regional_score_rows,
        coherence_rows=coherence_rows,
        candidate_families=["module_local_selector", "adapter", "regional_carry_forward"],
    )

    q1_diag = next(row for row in selection_rows if row["holdout_period"] == "2025-Q1" and row["metric_name"] == "diagnosed_plhiv")
    q2_diag = next(row for row in selection_rows if row["holdout_period"] == "2025-Q2" and row["metric_name"] == "diagnosed_plhiv")
    q2_prediction = next(row for row in rows if row["holdout_period"] == "2025-Q2" and row["region"] == "A")

    assert q1_diag["selected_candidate_family"] == "module_local_selector"
    assert q1_diag["selection_reason"] == "default_no_prior_split"
    assert q2_diag["selected_candidate_family"] == "adapter"
    assert q2_prediction["diagnosed_plhiv"] == pytest.approx(8.0)
    assert any(row["candidate_family"] == "adapter" for row in prior_rows)


def test_r56_gate_requires_split_stability_and_no_regression_against_references() -> None:
    promoted = _r56_gate(
        [
            {
                "candidate_family": "split_guarded_regional_selector",
                "mean_regional_normalized_absolute_error": 0.04,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.07,
                "mean_aggregate_mass_normalized_absolute_error": 0.07,
                "mean_regional_share_half_l1_error": 0.07,
            },
            {
                "candidate_family": "module_local_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.05,
                "mean_regional_share_half_l1_error": 0.05,
            },
        ],
        split_gate={"status": "strict_split_stable_adapter_promoted"},
        r54_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.045,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.045,
            "promoted_mean_regional_share_half_l1_error": 0.045,
        },
    )
    blocked = _r56_gate(
        [
            {
                "candidate_family": "split_guarded_regional_selector",
                "mean_regional_normalized_absolute_error": 0.046,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.07,
                "mean_aggregate_mass_normalized_absolute_error": 0.07,
                "mean_regional_share_half_l1_error": 0.07,
            },
            {
                "candidate_family": "module_local_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.05,
                "mean_regional_share_half_l1_error": 0.05,
            },
        ],
        split_gate={"status": "strict_split_stable_adapter_promoted"},
        r54_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.045,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.045,
            "promoted_mean_regional_share_half_l1_error": 0.045,
        },
    )

    assert promoted["status"] == "split_guarded_regional_selector_promoted"
    assert blocked["status"] == "split_guarded_regional_selector_diagnostic_only"
    assert "selector_does_not_beat_r54_regional_error" in blocked["blockers"]


def test_r57_ceiling_selection_is_leakage_labeled_and_not_promotable() -> None:
    prediction_rows = [
        {
            "candidate_family": family,
            "holdout_period": "2025-Q1",
            "region": "A",
            "estimated_plhiv": 20.0,
            "diagnosed_plhiv": value,
            "alive_on_art": 7.0,
            "tested_for_viral_load": 6.0,
            "virally_suppressed": 5.0,
        }
        for family, value in (("a", 12.0), ("b", 8.0))
    ]
    regional_score_rows = [
        {
            "candidate_family": "a",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "normalized_absolute_error": 0.20,
        },
        {
            "candidate_family": "b",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "normalized_absolute_error": 0.05,
        },
    ]
    coherence_rows = [
        {
            "candidate_family": "a",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "aggregate_mass_normalized_absolute_error": 0.20,
            "regional_share_half_l1_error": 0.20,
        },
        {
            "candidate_family": "b",
            "holdout_period": "2025-Q1",
            "metric_name": "diagnosed_plhiv",
            "aggregate_mass_normalized_absolute_error": 0.05,
            "regional_share_half_l1_error": 0.05,
        },
    ]

    selection = _r57_ceiling_selection_rows(
        prediction_rows=prediction_rows,
        regional_score_rows=regional_score_rows,
        coherence_rows=coherence_rows,
    )
    ceiling_rows = _r57_ceiling_prediction_rows(prediction_rows, selection)
    gate = _r57_gate(
        [
            {
                "candidate_family": "leakage_labeled_candidate_ceiling",
                "mean_regional_normalized_absolute_error": 0.04,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.04,
            }
        ],
        r54_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.06,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.06,
            "promoted_mean_regional_share_half_l1_error": 0.06,
        },
    )

    diagnosed_selection = next(row for row in selection if row["metric_name"] == "diagnosed_plhiv")
    assert diagnosed_selection["selected_candidate_family"] == "b"
    assert diagnosed_selection["leakage_status"] == "uses_same_split_targets_for_diagnostic_ceiling_only"
    assert ceiling_rows[0]["diagnosed_plhiv"] == pytest.approx(8.0)
    assert gate["status"] == "candidate_space_has_unrealized_signal_diagnostic"
    assert gate["claim_status"] == "diagnostic_only"
    assert "uses_same_split_holdout_targets" in gate["blockers"]


def test_r58_pareto_frontier_removes_dominated_candidates() -> None:
    families = _r58_pareto_candidate_families(
        [
            {
                "candidate_family": "a",
                "mean_regional_normalized_absolute_error": 0.10,
                "mean_aggregate_mass_normalized_absolute_error": 0.10,
                "mean_regional_share_half_l1_error": 0.10,
            },
            {
                "candidate_family": "b",
                "mean_regional_normalized_absolute_error": 0.09,
                "mean_aggregate_mass_normalized_absolute_error": 0.10,
                "mean_regional_share_half_l1_error": 0.10,
            },
            {
                "candidate_family": "c",
                "mean_regional_normalized_absolute_error": 0.11,
                "mean_aggregate_mass_normalized_absolute_error": 0.08,
                "mean_regional_share_half_l1_error": 0.10,
            },
        ]
    )

    assert "a" not in families
    assert families == ["b", "c"]


def test_r58_simplex_weights_are_nonnegative_and_sum_to_one() -> None:
    design = np.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float64)
    target = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    weights = _r58_fit_simplex_weights(design, target)

    assert np.all(weights >= 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    assert weights[0] > weights[1]


def test_r58_gate_separates_mean_promotion_from_split_stability() -> None:
    gate = _r58_gate(
        [
            {
                "candidate_family": "pareto_simplex_regional_ensemble",
                "mean_regional_normalized_absolute_error": 0.04,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.07,
                "mean_aggregate_mass_normalized_absolute_error": 0.07,
                "mean_regional_share_half_l1_error": 0.07,
            },
            {
                "candidate_family": "module_local_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.05,
                "mean_regional_share_half_l1_error": 0.05,
            },
        ],
        split_gate={"status": "adapter_mean_promoted_but_split_stability_limited"},
        r54_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.045,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.045,
            "promoted_mean_regional_share_half_l1_error": 0.045,
        },
    )

    assert gate["status"] == "pareto_simplex_regional_ensemble_mean_promoted_split_limited"
    assert gate["blockers"] == ["ensemble_not_strict_split_stable"]


def test_r59_anchor_stable_predictions_use_reference_for_stock_anchor() -> None:
    prediction_rows = [
        {
            "candidate_family": "module_local_selector",
            "holdout_period": "2025-Q1",
            "region": "A",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 70.0,
            "alive_on_art": 60.0,
            "tested_for_viral_load": 40.0,
            "virally_suppressed": 30.0,
        },
        {
            "candidate_family": "adapter",
            "holdout_period": "2025-Q1",
            "region": "A",
            "estimated_plhiv": 120.0,
            "diagnosed_plhiv": 72.0,
            "alive_on_art": 61.0,
            "tested_for_viral_load": 41.0,
            "virally_suppressed": 31.0,
        },
    ]

    rows, weights = _r59_anchor_stable_prediction_rows(
        prediction_rows,
        rows_by_key={},
        pareto_families=["adapter"],
        default_family="adapter",
    )

    assert rows[0]["estimated_plhiv"] == pytest.approx(100.0)
    assert rows[0]["diagnosed_plhiv"] == pytest.approx(72.0)
    anchor_weight = next(row for row in weights if row["metric_name"] == "estimated_plhiv")
    assert anchor_weight["weight_status"] == "anchor_metric_reference_family"


def test_r59_gate_renames_r58_gate_statuses() -> None:
    gate = _r59_gate(
        [
            {
                "candidate_family": "anchor_stable_pareto_ensemble",
                "mean_regional_normalized_absolute_error": 0.04,
                "mean_aggregate_mass_normalized_absolute_error": 0.04,
                "mean_regional_share_half_l1_error": 0.04,
            },
            {
                "candidate_family": "regional_carry_forward",
                "mean_regional_normalized_absolute_error": 0.07,
                "mean_aggregate_mass_normalized_absolute_error": 0.07,
                "mean_regional_share_half_l1_error": 0.07,
            },
            {
                "candidate_family": "module_local_selector",
                "mean_regional_normalized_absolute_error": 0.05,
                "mean_aggregate_mass_normalized_absolute_error": 0.05,
                "mean_regional_share_half_l1_error": 0.05,
            },
        ],
        split_gate={"status": "adapter_mean_promoted_but_split_stability_limited"},
        r54_gate={
            "promoted_mean_regional_normalized_absolute_error": 0.045,
            "promoted_mean_aggregate_mass_normalized_absolute_error": 0.045,
            "promoted_mean_regional_share_half_l1_error": 0.045,
        },
    )

    assert gate["status"] == "anchor_stable_pareto_ensemble_mean_promoted_split_limited"


def test_project_cascade_stock_row_enforces_nonnegative_cascade_cone() -> None:
    result = project_cascade_stock_row(
        {
            "quarter": "2025-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 120.0,
            "tested_for_viral_load": 80.0,
            "virally_suppressed": -5.0,
        }
    )

    assert result["changed"] is True
    assert result["projected"] == {
        "diagnosed_plhiv": 100.0,
        "alive_on_art": 100.0,
        "tested_for_viral_load": 80.0,
        "virally_suppressed": 0.0,
    }
    assert result["changed_metrics"] == ["alive_on_art", "virally_suppressed"]


def test_conditional_rate_gate_ignores_machine_epsilon_noise() -> None:
    gate = _conditional_rate_gate(
        [
            {
                "rate_id": "suppressed_among_vl_tested",
                "entry_count": 4,
                "candidate_mean_rate_error": 0.1 + 1.0e-17,
                "carry_forward_mean_rate_error": 0.1,
                "candidate_minus_carry_forward_mean_rate_error": 1.0e-17,
                "worst_candidate_minus_carry_forward_rate_error": 0.0,
            }
        ]
    )

    assert gate["status"] == "pass"


def test_stock_consistency_gate_rejects_diagnosed_or_art_worsening() -> None:
    gate = stock_consistency_gate(
        {
            "metric_anatomy": [
                {
                    "metric_name": "diagnosed_plhiv",
                    "candidate_mean_norm_error": 0.2,
                    "carry_forward_mean_norm_error": 0.1,
                    "candidate_minus_carry_forward_mean_norm_error": 0.1,
                    "worst_candidate_minus_carry_forward_norm_error": 0.2,
                },
                {
                    "metric_name": "alive_on_art",
                    "candidate_mean_norm_error": 0.05,
                    "carry_forward_mean_norm_error": 0.1,
                    "candidate_minus_carry_forward_mean_norm_error": -0.05,
                    "worst_candidate_minus_carry_forward_norm_error": -0.01,
                },
            ]
        }
    )

    assert gate["status"] == "fail"
    assert "diagnosed_plhiv_mean_worse_than_carry_forward" in gate["blockers"]


def test_stock_consistency_gate_passes_strict_non_regression() -> None:
    gate = stock_consistency_gate(
        {
            "metric_anatomy": [
                {
                    "metric_name": "diagnosed_plhiv",
                    "candidate_mean_norm_error": 0.1,
                    "carry_forward_mean_norm_error": 0.2,
                    "candidate_minus_carry_forward_mean_norm_error": -0.1,
                    "worst_candidate_minus_carry_forward_norm_error": 0.0,
                },
                {
                    "metric_name": "alive_on_art",
                    "candidate_mean_norm_error": 0.1,
                    "carry_forward_mean_norm_error": 0.1,
                    "candidate_minus_carry_forward_mean_norm_error": 0.0,
                    "worst_candidate_minus_carry_forward_norm_error": 0.0,
                },
            ]
        }
    )

    assert gate["status"] == "pass"
    assert gate["blockers"] == []


def test_linkage_lag_kernel_selects_train_supported_delay() -> None:
    rows = [
        {"quarter": "2020-Q1", "alive_on_art": 100.0, "new_diagnosed_cases_period": 20.0},
        {"quarter": "2020-Q2", "alive_on_art": 110.0, "new_diagnosed_cases_period": 1.0},
        {"quarter": "2020-Q3", "alive_on_art": 110.5, "new_diagnosed_cases_period": 10.0},
        {"quarter": "2020-Q4", "alive_on_art": 115.5, "new_diagnosed_cases_period": 1.0},
    ]

    model = _fit_linkage_lag_kernel(rows)

    assert model["status"] == "completed"
    assert model["selected_lag_quarters"] == 1
    assert model["selected_coefficient"] == 0.5


def test_support_partition_calibration_uses_partition_residual() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "exact_observed"}},
        },
        {
            "quarter": "2020-Q2",
            "diagnosed_plhiv": 110.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        },
        {
            "quarter": "2020-Q3",
            "diagnosed_plhiv": 121.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        },
    ]
    holdout_row = {
        "quarter": "2020-Q4",
        "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
    }

    model = _fit_support_partition_calibration_model(rows, "diagnosed_plhiv")
    prediction = _predict_support_partition_calibration(model, holdout_row, "diagnosed_plhiv")

    assert model["status"] == "completed"
    assert prediction is not None
    assert prediction > 121.0


def test_linkage_lag_plus_support_partition_combines_kept_branches() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "tested_for_viral_load": 30.0,
            "virally_suppressed": 20.0,
            "new_diagnosed_cases_period": 20.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        },
        {
            "quarter": "2020-Q2",
            "diagnosed_plhiv": 110.0,
            "alive_on_art": 60.0,
            "tested_for_viral_load": 32.0,
            "virally_suppressed": 22.0,
            "new_diagnosed_cases_period": 1.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        },
        {
            "quarter": "2020-Q3",
            "diagnosed_plhiv": 121.0,
            "alive_on_art": 60.5,
            "tested_for_viral_load": 34.0,
            "virally_suppressed": 24.0,
            "new_diagnosed_cases_period": 10.0,
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        },
    ]
    holdout = [
        {
            "quarter": "2020-Q4",
            "metric_provenance": {"diagnosed_plhiv": {"support_partition": "bridge_observed"}},
        }
    ]

    predictions, summary = _candidate_predictions(
        rows,
        holdout,
        family="linkage_lag_plus_support_partition",
    )

    assert summary["family"] == "linkage_lag_plus_support_partition"
    assert summary["linkage_lag_kernel"]["status"] == "completed"
    assert predictions[0]["diagnosed_plhiv"] > 121.0
    assert predictions[0]["alive_on_art"] >= 60.5
    assert predictions[0]["diagnosed_plhiv"] >= predictions[0]["alive_on_art"]


def test_r10_style_readout_teacher_is_train_only_selector() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "tested_for_viral_load": 30.0,
            "virally_suppressed": 20.0,
            "new_diagnosed_cases_period": 20.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 115.0,
            "alive_on_art": 60.0,
            "tested_for_viral_load": 32.0,
            "virally_suppressed": 22.0,
            "new_diagnosed_cases_period": 18.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 130.0,
            "alive_on_art": 70.0,
            "tested_for_viral_load": 34.0,
            "virally_suppressed": 24.0,
            "new_diagnosed_cases_period": 16.0,
        },
    ]
    holdout = [{"quarter": "2023-Q1"}]

    predictions, summary = _candidate_predictions(
        rows,
        holdout,
        family="r10_style_readout_teacher",
    )

    assert len(predictions) == 1
    assert summary["family"] == "r10_style_readout_teacher"
    assert "selected_metric_families" in summary
    assert "R10 target leakage" in summary["contract"]


def test_back_half_rate_model_predicts_bounded_conditional_rate() -> None:
    rows = [
        {"quarter": "2020-Q1", "alive_on_art": 100.0, "tested_for_viral_load": 40.0},
        {"quarter": "2020-Q2", "alive_on_art": 100.0, "tested_for_viral_load": 50.0},
        {"quarter": "2020-Q3", "alive_on_art": 100.0, "tested_for_viral_load": 60.0},
    ]

    model = _fit_back_half_rate_model(
        rows,
        rate_id="vl_tested_among_art",
        numerator_metric="tested_for_viral_load",
        denominator_metric="alive_on_art",
    )
    prediction = _predict_back_half_rate(model, {"quarter": "2020-Q4"}, variant="median_delta")

    assert model["status"] == "completed"
    assert prediction is not None
    assert 0.0 <= prediction <= 1.0
    assert prediction > model["last_rate"]


def test_r19_lineage_rate_state_model_predicts_bounded_lineage_rate() -> None:
    provenance = {
        "alive_on_art": {
            "source_tier": "official_doh_archive",
            "measurement_class": "program_observed_harp",
            "series_kind": "quarterly_snapshot",
            "support_partition": "common_support",
            "aggregation_mode": "quarterly",
        },
        "tested_for_viral_load": {
            "source_tier": "official_doh_archive",
            "measurement_class": "program_observed_harp",
            "series_kind": "quarterly_snapshot",
            "support_partition": "common_support",
            "aggregation_mode": "quarterly",
        },
    }
    rows = [
        {
            "quarter": "2020-Q1",
            "alive_on_art": 100.0,
            "tested_for_viral_load": 40.0,
            "metric_provenance": provenance,
        },
        {
            "quarter": "2020-Q2",
            "alive_on_art": 100.0,
            "tested_for_viral_load": 50.0,
            "metric_provenance": provenance,
        },
        {
            "quarter": "2020-Q3",
            "alive_on_art": 100.0,
            "tested_for_viral_load": 60.0,
            "metric_provenance": provenance,
        },
    ]

    model = _fit_r19_lineage_rate_state_model(
        rows,
        rate_id="vl_tested_among_art",
        numerator_metric="tested_for_viral_load",
        denominator_metric="alive_on_art",
    )
    prediction, detail = _predict_r19_lineage_rate_state(
        model,
        {"quarter": "2020-Q4", "metric_provenance": provenance},
    )

    assert model["status"] == "completed"
    assert model["lineage_count"] == 1
    assert prediction is not None
    assert 0.0 <= prediction <= 1.0
    assert detail["lineage_logit_bias"] != 0.0 or model["median_quarterly_logit_slope"] != 0.0


def test_r20_service_capacity_transition_predicts_bounded_stock() -> None:
    rows = [
        {"quarter": "2020-Q1", "alive_on_art": 100.0, "tested_for_viral_load": 40.0},
        {"quarter": "2020-Q2", "alive_on_art": 110.0, "tested_for_viral_load": 55.0},
        {"quarter": "2020-Q3", "alive_on_art": 120.0, "tested_for_viral_load": 70.0},
    ]
    latent_process = {
        "status": "completed",
        "latent_intensity_by_ordinal": {},
    }
    shock_process = {
        "status": "completed",
        "latent_shock_by_ordinal": {},
        "median_program_volume_shock": 0.0,
    }

    model = _fit_r20_service_capacity_transition(
        rows,
        state_metric="tested_for_viral_load",
        capacity_metric="alive_on_art",
        availability_process=latent_process,
        shock_process=shock_process,
    )
    prediction, detail = _predict_r20_service_capacity_transition(
        model,
        previous_state=70.0,
        capacity_value=130.0,
        availability=0.0,
        shock=0.0,
    )

    assert model["status"] == "completed"
    assert prediction is not None
    assert 0.0 <= prediction <= 130.0
    assert detail["service_gap"] == pytest.approx(60.0)


def test_r21_diagnosis_flow_selector_is_lead_aware_and_guarded() -> None:
    provenance = {
        metric: {
            "source_tier": "official_doh_archive",
            "measurement_class": "program_observed_harp",
            "series_kind": "quarterly_snapshot",
            "support_partition": "common_support",
            "aggregation_mode": "quarterly",
        }
        for metric in [
            "diagnosed_plhiv",
            "alive_on_art",
            "tested_for_viral_load",
            "virally_suppressed",
            "new_diagnosed_cases_period",
        ]
    }
    rows = []
    for index, year in enumerate(range(2018, 2024)):
        flow = 10.0 + 2.0 * index
        diagnosed = 100.0 + 12.0 * index
        art = 70.0 + 9.0 * index
        rows.append(
            {
                "quarter": f"{year}-Q4",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": art,
                "tested_for_viral_load": 0.6 * art,
                "virally_suppressed": 0.72 * 0.6 * art,
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    selector = _fit_r21_diagnosis_flow_selector(rows, max_horizon_years=3)

    assert selector["reference_family"] == "r19_joint_service_cascade_process"
    assert selector["record_count"] > 0
    assert "global" in selector["policy_by_lead"]


def test_r22_program_metric_policy_requires_mean_and_worst_improvement() -> None:
    improving_records = [
        {
            "policy_key": "alive_on_art|lead3",
            "metric_global_key": "alive_on_art|global",
            "base_prediction": 10.0,
            "selected_prediction": 12.0,
            "carry_forward_prediction": 10.0,
            "target_value": 12.0,
            "scale": 1.0,
        },
        {
            "policy_key": "alive_on_art|lead3",
            "metric_global_key": "alive_on_art|global",
            "base_prediction": 14.0,
            "selected_prediction": 13.0,
            "carry_forward_prediction": 14.0,
            "target_value": 13.0,
            "scale": 1.0,
        },
    ]
    selected = _r22_select_program_metric_policy(improving_records, policy_key="alive_on_art|lead3")

    assert selected["status"] == "completed"
    assert selected["blend_weight"] > 0.0

    mixed_records = [
        {
            "policy_key": "alive_on_art|lead5",
            "metric_global_key": "alive_on_art|global",
            "base_prediction": 10.0,
            "selected_prediction": 12.0,
            "carry_forward_prediction": 10.0,
            "target_value": 12.0,
            "scale": 1.0,
        },
        {
            "policy_key": "alive_on_art|lead5",
            "metric_global_key": "alive_on_art|global",
            "base_prediction": 10.0,
            "selected_prediction": 20.0,
            "carry_forward_prediction": 10.0,
            "target_value": 10.0,
            "scale": 1.0,
        },
    ]
    failed = _r22_select_program_metric_policy(mixed_records, policy_key="alive_on_art|lead5")

    assert failed["status"] == "failed_closed"
    assert failed["blend_weight"] == 0.0


def test_r25_endpoint_head_selects_only_walk_forward_supported_policy() -> None:
    records = [
        {
            "train_end_year": 2018,
            "quarter": "2019-Q4",
            "metric_name": "alive_on_art",
            "lead_years": 1,
            "target_value": 121.0,
            "base_prediction": 100.0,
            "carry_forward_prediction": 121.0,
            "scale": 100.0,
            "base_norm_error": 0.21,
            "log_residual": 0.19062035960864987,
        },
        {
            "train_end_year": 2019,
            "quarter": "2020-Q4",
            "metric_name": "alive_on_art",
            "lead_years": 1,
            "target_value": 121.0,
            "base_prediction": 100.0,
            "carry_forward_prediction": 121.0,
            "scale": 100.0,
            "base_norm_error": 0.21,
            "log_residual": 0.19062035960864987,
        },
        {
            "train_end_year": 2020,
            "quarter": "2021-Q4",
            "metric_name": "alive_on_art",
            "lead_years": 1,
            "target_value": 121.0,
            "base_prediction": 100.0,
            "carry_forward_prediction": 121.0,
            "scale": 100.0,
            "base_norm_error": 0.21,
            "log_residual": 0.19062035960864987,
        },
    ]

    head = _fit_r25_predictive_endpoint_head_from_records(records, max_horizon_years=1)

    assert head["status"] == "completed"
    assert head["selected_policy_count"] == 1
    policy = head["policy_by_metric_lead"]["alive_on_art|lead1"]
    assert policy["policy"] in {"log_residual", "carry_blend"}
    row = next(item for item in head["policy_rows"] if item["key"] == "alive_on_art|lead1")
    assert row["selected"] is True


def test_back_half_conditional_rates_preserve_front_half_and_cascade() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 80.0,
            "tested_for_viral_load": 40.0,
            "virally_suppressed": 20.0,
            "new_diagnosed_cases_period": 10.0,
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 110.0,
            "alive_on_art": 90.0,
            "tested_for_viral_load": 54.0,
            "virally_suppressed": 32.4,
            "new_diagnosed_cases_period": 12.0,
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 100.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 49.0,
            "new_diagnosed_cases_period": 14.0,
        },
    ]
    holdout = [{"quarter": "2023-Q1"}]

    predictions, summary = _candidate_predictions(
        rows,
        holdout,
        family="back_half_conditional_rates",
    )

    assert summary["family"] == "back_half_conditional_rates"
    assert summary["back_half_rate_process"]["status"] == "completed"
    assert predictions[0]["diagnosed_plhiv"] >= predictions[0]["alive_on_art"]
    assert predictions[0]["alive_on_art"] >= predictions[0]["tested_for_viral_load"]
    assert predictions[0]["tested_for_viral_load"] >= predictions[0]["virally_suppressed"]


def test_r11_multi_horizon_report_materializes_lifted_gate() -> None:
    rows = []
    for index, year in enumerate(range(2018, 2025)):
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": 100.0 + 10.0 * index,
                "alive_on_art": 80.0 + 9.0 * index,
                "tested_for_viral_load": 40.0 + 5.0 * index,
                "virally_suppressed": 30.0 + 4.0 * index,
                "new_diagnosed_cases_period": 10.0 + index,
            }
        )

    report = _r11_multi_horizon_report(
        experiment_id="R11-15-test",
        family="back_half_conditional_rates",
        rows=rows,
        start_year=2018,
        end_year=2024,
        min_train_years=2,
        horizons=(1, 2),
        r10_reference_mae=0.0,
    )

    assert report["schema_version"] == "phase3_dynamic.r11_multi_horizon_lifted_report.v1"
    assert [row["horizon_years"] for row in report["horizon_rows"]] == [1, 2]
    assert report["multi_horizon_gate"]["status"] in {"pass", "fail"}
    assert report["r10_lifted_gate"]["status"] == "fail"


def test_horizon_matched_r10_reference_selects_best_r10_family_row() -> None:
    reference = _select_horizon_matched_r10_reference(
        {
            "results": [
                {
                    "experiment_id": "EXP-R10-DENSE-CHAMPION",
                    "family": "repair",
                    "quarterly_summary": {
                        "candidate_mean_mae": 0.20,
                        "carry_forward_mean_mae": 0.30,
                    },
                },
                {
                    "experiment_id": "EXP-R10-DENSE-M1-H1",
                    "family": "repair",
                    "quarterly_summary": {
                        "candidate_mean_mae": 0.10,
                        "carry_forward_mean_mae": 0.30,
                    },
                },
                {
                    "experiment_id": "EXP-R1",
                    "family": "repair",
                    "quarterly_summary": {
                        "candidate_mean_mae": 0.01,
                        "carry_forward_mean_mae": 0.30,
                    },
                },
            ]
        }
    )

    assert reference["available"] is True
    assert reference["reference_experiment_id"] == "EXP-R10-DENSE-M1-H1"
    assert reference["reference_quarterly_mean_mae"] == 0.10
    assert reference["reference_metric_scope"] == [
        "diagnosed_plhiv",
        "alive_on_art",
        "new_diagnosed_cases_period",
    ]


def test_r10_lifted_gate_uses_horizon_matched_comparable_score() -> None:
    gate = _r10_lifted_gate(
        [
            {
                "horizon_years": 1,
                "candidate_mean_mae": 0.50,
                "r10_comparable_candidate_mean_mae": 0.09,
                "r10_horizon_reference_mae": 0.10,
                "r10_horizon_reference_experiment_id": "EXP-R10-DENSE-M1-H1",
                "r10_horizon_reference_source": "horizon_matched_replay",
            },
            {
                "horizon_years": 3,
                "candidate_mean_mae": 0.50,
                "r10_comparable_candidate_mean_mae": 0.20,
                "r10_horizon_reference_mae": 0.10,
                "r10_horizon_reference_experiment_id": "EXP-R10-DENSE-M1-H1",
                "r10_horizon_reference_source": "horizon_matched_replay",
            },
        ],
        r10_reference_mae=None,
    )

    assert gate["status"] == "fail"
    assert gate["horizon_rows"][0]["status"] == "pass"
    assert gate["horizon_rows"][1]["status"] == "fail"
    assert gate["blockers"] == ["h3_candidate_mean_not_better_than_r10_reference"]


def test_trajectory_shape_head_is_train_origin_model() -> None:
    rows = []
    for index, year in enumerate(range(2018, 2025)):
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": 100.0 + 8.0 * index + float(index * index),
                "alive_on_art": 80.0 + 7.0 * index + float(index * index) * 0.8,
                "tested_for_viral_load": 40.0 + 5.0 * index,
                "virally_suppressed": 30.0 + 4.0 * index,
                "new_diagnosed_cases_period": 10.0 + index,
            }
        )

    model = _fit_trajectory_shape_head(rows, max_horizon_years=3)
    predictions, summary = _candidate_predictions(
        rows[:-1],
        [rows[-1]],
        family="trajectory_shape_head",
    )

    assert model["status"] == "completed"
    assert "correction_rows" in model
    assert summary["family"] == "trajectory_shape_head"
    assert summary["trajectory_shape_head"]["status"] == "completed"
    assert predictions[0]["diagnosed_plhiv"] >= predictions[0]["alive_on_art"]
    assert predictions[0]["alive_on_art"] >= predictions[0]["tested_for_viral_load"]


def test_constrained_shape_head_preserves_back_half_conditional_rates() -> None:
    rows = []
    for index, year in enumerate(range(2018, 2026)):
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": 120.0 + 8.0 * index + float(index * index),
                "alive_on_art": 90.0 + 7.0 * index + 0.7 * float(index * index),
                "tested_for_viral_load": 45.0 + 5.0 * index,
                "virally_suppressed": 30.0 + 4.0 * index,
                "new_diagnosed_cases_period": 14.0 + index,
            }
        )
    train_rows = rows[:-1]
    holdout_rows = [rows[-1]]

    base_predictions, _base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="back_half_conditional_rates",
    )
    constrained_predictions, summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="constrained_trajectory_shape_head",
    )

    def rate(row: dict[str, float], numerator: str, denominator: str) -> float:
        return float(row[numerator]) / float(row[denominator])

    base = base_predictions[0]
    constrained = constrained_predictions[0]
    assert summary["family"] == "constrained_trajectory_shape_head"
    assert summary["constrained_shape_contract"]["directly_corrected_metrics"] == [
        "alive_on_art",
        "new_diagnosed_cases_period",
    ]
    assert constrained["diagnosed_plhiv"] >= constrained["alive_on_art"]
    assert constrained["alive_on_art"] >= constrained["tested_for_viral_load"]
    assert constrained["tested_for_viral_load"] >= constrained["virally_suppressed"]
    assert rate(constrained, "tested_for_viral_load", "alive_on_art") == pytest.approx(
        rate(base, "tested_for_viral_load", "alive_on_art")
    )
    assert rate(constrained, "virally_suppressed", "tested_for_viral_load") == pytest.approx(
        rate(base, "virally_suppressed", "tested_for_viral_load")
    )


def test_horizon_adaptive_shape_selector_uses_predeclared_policy_and_preserves_rates() -> None:
    rows = []
    for index, year in enumerate(range(2018, 2027)):
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": 140.0 + 9.0 * index + 0.6 * float(index * index),
                "alive_on_art": 100.0 + 7.0 * index + 0.5 * float(index * index),
                "tested_for_viral_load": 50.0 + 4.0 * index,
                "virally_suppressed": 35.0 + 3.0 * index,
                "new_diagnosed_cases_period": 15.0 + 1.5 * index,
            }
        )
    train_rows = rows[:-1]
    holdout_rows = [rows[-1]]
    base_predictions, _base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="back_half_conditional_rates",
    )
    adaptive_predictions, summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="horizon_adaptive_constrained_shape_head",
    )
    fitted_selector = _fit_horizon_adaptive_shape_selector(train_rows, max_horizon_years=1)
    selector = summary["horizon_adaptive_shape_selector"]

    assert summary["family"] == "horizon_adaptive_constrained_shape_head"
    assert fitted_selector["status"] == "completed"
    assert selector["baseline_policy_id"] == "art_plus_diagnosis_flow"
    assert selector["selected_policy_id"] in {
        "identity_r11_14",
        "diagnosis_flow_only",
        "art_only",
        "art_plus_diagnosis_flow",
        "diagnosed_stock_only",
        "r10_scope_stocks",
        "r10_scope_all",
    }
    assert any(row["selected"] for row in selector["policy_rows"])
    base = base_predictions[0]
    adaptive = adaptive_predictions[0]
    assert adaptive["diagnosed_plhiv"] >= adaptive["alive_on_art"]
    assert adaptive["alive_on_art"] >= adaptive["tested_for_viral_load"]
    assert adaptive["tested_for_viral_load"] >= adaptive["virally_suppressed"]
    assert adaptive["tested_for_viral_load"] / adaptive["alive_on_art"] == pytest.approx(
        base["tested_for_viral_load"] / base["alive_on_art"]
    )
    assert adaptive["virally_suppressed"] / adaptive["tested_for_viral_load"] == pytest.approx(
        base["virally_suppressed"] / base["tested_for_viral_load"]
    )


def test_horizon_adaptive_apply_can_correct_diagnosed_stock_without_breaking_cone() -> None:
    base_prediction = {
        "quarter": "2025-Q1",
        "diagnosed_plhiv": 100.0,
        "alive_on_art": 80.0,
        "tested_for_viral_load": 40.0,
        "virally_suppressed": 30.0,
        "new_diagnosed_cases_period": 12.0,
    }
    holdout_row = {"quarter": "2025-Q1"}
    shape_head = {
        "correction_by_metric_lead": {
            "diagnosed_plhiv|lead1": -1.0,
            "alive_on_art|lead1": 1.0,
        }
    }
    selector = {"selected_policy_id": "r10_scope_stocks"}

    prediction = _apply_horizon_adaptive_constrained_shape_head(
        base_prediction,
        holdout_row,
        shape_head,
        selector,
        train_end_year=2024,
    )

    assert prediction["diagnosed_plhiv"] >= prediction["alive_on_art"]
    assert prediction["alive_on_art"] >= prediction["tested_for_viral_load"]
    assert prediction["tested_for_viral_load"] >= prediction["virally_suppressed"]
    assert prediction["tested_for_viral_load"] / prediction["alive_on_art"] == pytest.approx(0.5)
    assert prediction["virally_suppressed"] / prediction["tested_for_viral_load"] == pytest.approx(0.75)


def test_datv_transition_process_fits_bounded_stock_flow_coefficients() -> None:
    rows = []
    diagnosed = 100.0
    art = 70.0
    for index, year in enumerate(range(2018, 2026)):
        flow = 12.0 + float(index)
        diagnosed = 0.98 * diagnosed + 0.90 * flow
        art = 0.94 * art + 0.50 * flow + 0.04 * max(diagnosed - art, 0.0)
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": min(art, diagnosed),
                "tested_for_viral_load": min(0.55 * art, art),
                "virally_suppressed": min(0.70 * 0.55 * art, 0.55 * art),
                "new_diagnosed_cases_period": flow,
            }
        )

    process = _fit_datv_transition_process(rows)
    diagnosed_model = process["diagnosed_stock_transition"]
    art_model = process["art_stock_transition"]

    assert process["status"] == "completed"
    assert 0.0 <= diagnosed_model["retention_coefficient"] <= 1.0
    assert 0.0 <= diagnosed_model["diagnosis_flow_coefficient"] <= 1.0
    assert 0.0 <= art_model["art_retention_coefficient"] <= 1.0
    assert 0.0 <= art_model["diagnosis_linkage_coefficient"] <= 1.0
    assert 0.0 <= art_model["diagnosed_gap_linkage_coefficient"] <= 1.0
    assert art_model["selected_lag_quarters"] in {0, 1, 2, 4}


def test_datv_transition_process_candidate_preserves_stock_cone_and_back_half_rates() -> None:
    rows = []
    diagnosed = 120.0
    art = 85.0
    for index, year in enumerate(range(2018, 2027)):
        flow = 14.0 + float(index)
        diagnosed = 0.99 * diagnosed + 0.85 * flow
        art = 0.95 * art + 0.45 * flow + 0.03 * max(diagnosed - art, 0.0)
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": min(art, diagnosed),
                "tested_for_viral_load": min(0.60 * art, art),
                "virally_suppressed": min(0.75 * 0.60 * art, 0.60 * art),
                "new_diagnosed_cases_period": flow,
            }
        )

    train_rows = rows[:-1]
    holdout_rows = [rows[-1]]
    predictions, summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="datv_transition_process",
    )
    prediction = predictions[0]

    assert summary["family"] == "datv_transition_process"
    assert summary["transition_process"]["status"] == "completed"
    assert prediction["diagnosed_plhiv"] >= prediction["alive_on_art"]
    assert prediction["alive_on_art"] >= prediction["tested_for_viral_load"]
    assert prediction["tested_for_viral_load"] >= prediction["virally_suppressed"]
    assert prediction["tested_for_viral_load"] / prediction["alive_on_art"] <= 1.0
    assert prediction["virally_suppressed"] / prediction["tested_for_viral_load"] <= 1.0


def test_era_datv_transition_process_fits_removal_and_reporting_shift_terms() -> None:
    rows = []
    diagnosed = 140.0
    art = 95.0
    for index, year in enumerate(range(2018, 2027)):
        flow = 12.0 + float(index)
        reporting_shift = 0.0 if index < 5 else 8.0
        diagnosed = diagnosed + 0.75 * flow - 0.02 * diagnosed + reporting_shift
        art = art + 0.35 * flow + 0.02 * max(diagnosed - art, 0.0) - 0.03 * art + 0.5 * reporting_shift
        support = "bridge" if index < 5 else "expanded"
        provenance = {
            metric: {
                "support_partition": support,
                "tier": "direct",
                "aggregation_mode": "quarterly",
            }
            for metric in [
                "diagnosed_plhiv",
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ]
        }
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": min(art, diagnosed),
                "tested_for_viral_load": min(0.60 * art, art),
                "virally_suppressed": min(0.75 * 0.60 * art, 0.60 * art),
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    process = _fit_era_datv_transition_process(rows)
    diagnosed_model = process["diagnosed_stock_transition"]
    art_model = process["art_stock_transition"]

    assert process["status"] == "completed"
    assert 0.0 <= diagnosed_model["diagnosed_removal_fraction"] <= 1.0
    assert 0.0 <= art_model["art_removal_fraction"] <= 1.0
    assert diagnosed_model["era_count"] >= 2
    assert art_model["era_count"] >= 2
    assert diagnosed_model["reporting_shift_by_era"]
    assert art_model["reporting_shift_by_era"]


def test_era_datv_transition_process_candidate_preserves_stock_cone() -> None:
    rows = []
    diagnosed = 150.0
    art = 100.0
    for index, year in enumerate(range(2018, 2028)):
        flow = 15.0 + float(index)
        support = "legacy" if index < 6 else "expanded"
        shift = 0.0 if support == "legacy" else 6.0
        diagnosed = diagnosed + 0.70 * flow - 0.01 * diagnosed + shift
        art = art + 0.30 * flow + 0.03 * max(diagnosed - art, 0.0) - 0.02 * art + 0.5 * shift
        provenance = {
            metric: {
                "support_partition": support,
                "tier": "direct",
                "aggregation_mode": "quarterly",
            }
            for metric in [
                "diagnosed_plhiv",
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ]
        }
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": min(art, diagnosed),
                "tested_for_viral_load": min(0.55 * art, art),
                "virally_suppressed": min(0.70 * 0.55 * art, 0.55 * art),
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    predictions, summary = _candidate_predictions(
        rows[:-1],
        [rows[-1]],
        family="era_datv_transition_process",
    )
    prediction = predictions[0]

    assert summary["family"] == "era_datv_transition_process"
    assert summary["transition_process"]["status"] == "completed"
    assert prediction["diagnosed_plhiv"] >= prediction["alive_on_art"]
    assert prediction["alive_on_art"] >= prediction["tested_for_viral_load"]
    assert prediction["tested_for_viral_load"] >= prediction["virally_suppressed"]


@pytest.mark.parametrize(
    "family",
    [
        "horizon_family_selector",
        "diagnosis_flow_input_repair_process",
        "support_era_diagnosis_flow_process",
        "stock_flow_reconciliation_process",
        "diagnosed_reporting_bias_process",
        "art_horizon_selector_process",
        "diagnosis_lag_stock_process",
        "multi_horizon_weighted_process",
        "r10_scope_teacher_stock_process",
        "conditional_rate_horizon_selector",
        "r12_long_horizon_stock_shape_process",
        "r12_da_process_split_transition",
        "r12_da_residual_source_process",
        "r12_route_aware_two_head_process",
        "r12_stock_cone_safe_annual_trajectory_process",
        "r12_program_nowcast_mixed_quarterly_process",
    ],
)
def test_r11_21_to_r12_candidate_families_preserve_stock_cone(family: str) -> None:
    rows = []
    diagnosed = 150.0
    art = 105.0
    for index, year in enumerate(range(2016, 2028)):
        flow = 14.0 + float(index % 4)
        support = "legacy" if index < 6 else "expanded"
        shift = 0.0 if support == "legacy" else 4.0
        diagnosed = diagnosed + 0.65 * flow - 0.01 * diagnosed + shift
        art = art + 0.32 * flow + 0.025 * max(diagnosed - art, 0.0) - 0.018 * art + 0.4 * shift
        provenance = {
            metric: {
                "support_partition": support,
                "tier": "direct",
                "aggregation_mode": "quarterly",
            }
            for metric in [
                "diagnosed_plhiv",
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ]
        }
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": min(art, diagnosed),
                "tested_for_viral_load": min(0.58 * art, art),
                "virally_suppressed": min(0.72 * 0.58 * art, 0.58 * art),
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    predictions, summary = _candidate_predictions(rows[:-1], [rows[-1]], family=family)
    prediction = predictions[0]

    assert summary["family"] == family
    assert prediction["diagnosed_plhiv"] >= prediction["alive_on_art"]
    assert prediction["alive_on_art"] >= prediction["tested_for_viral_load"]
    assert prediction["tested_for_viral_load"] >= prediction["virally_suppressed"]


def test_r12_04_source_lineage_ablation_is_diagnostic_only() -> None:
    rows = []
    diagnosed = 120.0
    art = 85.0
    for index, year in enumerate(range(2012, 2026)):
        flow = 9.0 + float(index % 3)
        support = "bridge_observed" if year < 2019 else "expanded_harp"
        series_kind = "annual_bridge" if year < 2019 else "monthly_harp"
        diagnosed = diagnosed + 0.7 * flow + 1.5
        art = min(diagnosed, art + 0.45 * flow + 1.0)
        provenance = {
            metric: {
                "source_id": f"{series_kind}_{metric}",
                "source_tier": "official",
                "measurement_class": "stock" if metric != "new_diagnosed_cases_period" else "flow",
                "series_kind": series_kind,
                "support_partition": support,
                "aggregation_mode": "quarterly_from_monthly" if year >= 2019 else "annual_to_quarter",
                "observation_role": "direct_target",
                "allowed_use": "training",
            }
            for metric in [
                "diagnosed_plhiv",
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ]
        }
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": art,
                "tested_for_viral_load": 0.6 * art,
                "virally_suppressed": 0.75 * 0.6 * art,
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    report = _build_r12_04_source_lineage_ablation_report(
        rows=rows,
        start_year=2012,
        end_year=2025,
        min_train_years=4,
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 3,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 0.01,
                }
            ]
        },
        horizons=(3,),
    )

    assert report["experiment_id"] == "R12-04"
    assert report["prediction_mutation"] == "forbidden"
    assert report["decision"] == "keep_as_observation_lineage_diagnostic"
    assert report["score_record_count"] > 0
    assert "support_partition" in report["lineage_axes"]
    assert report["axis_row_count"] > 0


def test_r12_05_lineage_stratified_contract_masks_other_lineages() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [120.0, 82.0],
        "doh_monthly": [80.0, 55.0],
        "slide_annual_anchor": [150.0, 102.0],
    }
    years = list(range(2012, 2026))
    for index, year in enumerate(years):
        lineage_id = ("doh_quarterly", "doh_monthly", "slide_annual_anchor")[index % 3]
        diagnosed, art = state_by_lineage[lineage_id]
        flow = 8.0 + float(index % 4)
        diagnosed = diagnosed + 0.8 * flow
        art = min(diagnosed, art + 0.5 * flow)
        state_by_lineage[lineage_id] = [diagnosed, art]
        source_tier, measurement_class, series_kind = source_families[lineage_id].split("|")
        provenance = {
            metric: {
                "source_id": f"{lineage_id}_{metric}",
                "source_tier": source_tier,
                "measurement_class": measurement_class,
                "series_kind": series_kind,
                "support_partition": "common_support",
                "aggregation_mode": "quarterly_observed",
                "observation_role": "direct_target",
                "allowed_use": "training",
            }
            for metric in [
                "diagnosed_plhiv",
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ]
        }
        rows.append(
            {
                "quarter": f"{year}-Q1",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": art,
                "tested_for_viral_load": 0.6 * art,
                "virally_suppressed": 0.7 * 0.6 * art,
                "new_diagnosed_cases_period": flow,
                "metric_provenance": provenance,
            }
        )

    report = _build_r12_05_lineage_stratified_contract_report(
        rows=rows,
        start_year=2012,
        end_year=2025,
        min_train_years=4,
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 3,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                }
            ]
        },
        horizons=(3,),
    )

    assert report["experiment_id"] == "R12-05"
    assert report["prediction_mutation"] == "forbidden"
    assert report["decision"] == "keep_as_lineage_stratified_observation_operator_contract"
    assert report["score_record_count"] > 0
    assert {row["lineage_id"] for row in report["lineage_manifests"]} == set(source_families)
    assert all(set(row["metric_counts"]) for row in report["lineage_manifests"])
    assert report["five_year_dynamics_claim_status"] == "frozen_not_promoted"


def test_r12_06_doh_quarterly_support_adjudication_is_diagnostic_only() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [140.0, 95.0],
        "doh_monthly": [138.0, 94.0],
        "slide_annual_anchor": [142.0, 96.0],
    }
    for year in range(2018, 2026):
        for lineage_id, source_family in source_families.items():
            diagnosed, art = state_by_lineage[lineage_id]
            flow = 10.0 + float((year + len(lineage_id)) % 5)
            diagnosed = diagnosed + 0.75 * flow + (1.0 if lineage_id == "slide_annual_anchor" else 0.0)
            art = min(diagnosed, art + 0.5 * flow)
            state_by_lineage[lineage_id] = [diagnosed, art]
            source_tier, measurement_class, series_kind = source_family.split("|")
            provenance = {
                metric: {
                    "source_id": f"{lineage_id}_{metric}",
                    "source_tier": source_tier,
                    "measurement_class": measurement_class,
                    "series_kind": series_kind,
                    "support_partition": "common_support",
                    "aggregation_mode": "quarterly_observed",
                    "observation_role": "direct_target",
                    "allowed_use": "training",
                }
                for metric in [
                    "diagnosed_plhiv",
                    "alive_on_art",
                    "tested_for_viral_load",
                    "virally_suppressed",
                    "new_diagnosed_cases_period",
                ]
            }
            quarter = "Q4" if lineage_id == "slide_annual_anchor" else ("Q2" if lineage_id == "doh_monthly" else "Q3")
            rows.append(
                {
                    "quarter": f"{year}-{quarter}",
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "tested_for_viral_load": 0.62 * art,
                    "virally_suppressed": 0.74 * 0.62 * art,
                    "new_diagnosed_cases_period": flow,
                    "metric_provenance": provenance,
                }
            )

    report = _build_r12_06_doh_quarterly_support_adjudication_report(
        rows=rows,
        start_year=2018,
        end_year=2025,
        production_min_train_years=5,
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 1,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 2,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
            ]
        },
    )

    assert report["experiment_id"] == "R12-06"
    assert report["prediction_mutation"] == "forbidden"
    assert report["decision"] == "keep_as_doh_quarterly_support_adequacy_adjudication"
    assert report["production_min_train_years"] == 5
    assert report["short_horizon_manifest"]["score_record_count"] > 0
    assert report["bridge_consistency"]["bridge_pair_count"] > 0
    assert report["bridge_consistency"]["contract"].count("same calendar year") == 1


def test_r12_07_horizon_specific_router_separates_nowcast_and_trajectory_evidence() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [125.0, 88.0],
        "doh_monthly": [123.0, 87.0],
        "slide_annual_anchor": [130.0, 91.0],
    }
    for year in range(2013, 2026):
        for lineage_id, source_family in source_families.items():
            diagnosed, art = state_by_lineage[lineage_id]
            flow = 7.0 + float((year + len(lineage_id)) % 6)
            diagnosed = diagnosed + 0.82 * flow + (0.6 if lineage_id == "slide_annual_anchor" else 0.0)
            art = min(diagnosed, art + 0.48 * flow)
            state_by_lineage[lineage_id] = [diagnosed, art]
            source_tier, measurement_class, series_kind = source_family.split("|")
            provenance = {
                metric: {
                    "source_id": f"{lineage_id}_{metric}",
                    "source_tier": source_tier,
                    "measurement_class": measurement_class,
                    "series_kind": series_kind,
                    "support_partition": "common_support",
                    "aggregation_mode": "quarterly_observed",
                    "observation_role": "direct_target",
                    "allowed_use": "training",
                }
                for metric in [
                    "diagnosed_plhiv",
                    "alive_on_art",
                    "tested_for_viral_load",
                    "virally_suppressed",
                    "new_diagnosed_cases_period",
                ]
            }
            quarter = "Q4" if lineage_id == "slide_annual_anchor" else ("Q2" if lineage_id == "doh_monthly" else "Q3")
            rows.append(
                {
                    "quarter": f"{year}-{quarter}",
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "tested_for_viral_load": 0.6 * art,
                    "virally_suppressed": 0.72 * 0.6 * art,
                    "new_diagnosed_cases_period": flow,
                    "metric_provenance": provenance,
                }
            )

    report = _build_r12_07_horizon_specific_evidence_router_report(
        rows=rows,
        start_year=2013,
        end_year=2025,
        production_min_train_years=4,
        diagnostic_min_train_years=1,
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 1,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 3,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 5,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
            ]
        },
    )

    assert report["experiment_id"] == "R12-07"
    assert report["prediction_mutation"] == "forbidden"
    assert report["decision"] == "keep_as_horizon_specific_evidence_router"
    assert report["score_record_count"] > 0
    route_ids = {row["route_id"] for row in report["route_manifests"]}
    assert route_ids == {"program_nowcast", "annual_trajectory_anchor"}
    program_manifest = next(row for row in report["route_manifests"] if row["route_id"] == "program_nowcast")
    trajectory_manifest = next(row for row in report["route_manifests"] if row["route_id"] == "annual_trajectory_anchor")
    assert program_manifest["horizons"] == [1, 2]
    assert trajectory_manifest["horizons"] == [3, 5]
    assert program_manifest["min_train_contract"] == "diagnostic_short_horizon"
    assert trajectory_manifest["min_train_contract"] == "production"
    assert {row["route_id"] for row in report["route_horizon_rows"]} == route_ids


def test_r12_08_route_aware_two_head_candidate_uses_separate_route_gates() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [125.0, 88.0],
        "doh_monthly": [123.0, 87.0],
        "slide_annual_anchor": [130.0, 91.0],
    }
    for year in range(2013, 2026):
        for lineage_id, source_family in source_families.items():
            diagnosed, art = state_by_lineage[lineage_id]
            flow = 7.0 + float((year + len(lineage_id)) % 6)
            diagnosed = diagnosed + 0.82 * flow + (0.6 if lineage_id == "slide_annual_anchor" else 0.0)
            art = min(diagnosed, art + 0.48 * flow)
            state_by_lineage[lineage_id] = [diagnosed, art]
            source_tier, measurement_class, series_kind = source_family.split("|")
            provenance = {
                metric: {
                    "source_id": f"{lineage_id}_{metric}",
                    "source_tier": source_tier,
                    "measurement_class": measurement_class,
                    "series_kind": series_kind,
                    "support_partition": "common_support",
                    "aggregation_mode": "quarterly_observed",
                    "observation_role": "direct_target",
                    "allowed_use": "training",
                }
                for metric in [
                    "diagnosed_plhiv",
                    "alive_on_art",
                    "tested_for_viral_load",
                    "virally_suppressed",
                    "new_diagnosed_cases_period",
                ]
            }
            quarter = "Q4" if lineage_id == "slide_annual_anchor" else ("Q2" if lineage_id == "doh_monthly" else "Q3")
            rows.append(
                {
                    "quarter": f"{year}-{quarter}",
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "tested_for_viral_load": 0.6 * art,
                    "virally_suppressed": 0.72 * 0.6 * art,
                    "new_diagnosed_cases_period": flow,
                    "metric_provenance": provenance,
                }
            )

    report = _build_r12_08_route_aware_two_head_candidate_report(
        rows=rows,
        start_year=2013,
        end_year=2025,
        production_min_train_years=4,
        diagnostic_min_train_years=1,
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 1,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 3,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 5,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
            ]
        },
    )

    assert report["experiment_id"] == "R12-08"
    assert report["prediction_mutation"] == "enabled_route_aware_two_head"
    assert report["candidate_family"] == "r12_route_aware_two_head_process"
    assert report["score_record_count"] > 0
    route_ids = {row["route_id"] for row in report["route_manifests"]}
    assert route_ids == {"program_nowcast", "annual_trajectory_anchor"}
    assert {row["candidate_family"] for row in report["route_horizon_rows"]} == {"r12_route_aware_two_head_process"}
    assert report["decision"] in {
        "keep_as_route_aware_two_head_candidate",
        "keep_as_route_aware_nowcast_candidate",
        "reject_route_aware_candidate",
        "reject_route_aware_candidate_for_support_gap",
    }


def test_r12_09_stock_cone_safe_annual_trajectory_head_is_lineage_scoped() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [125.0, 88.0],
        "doh_monthly": [123.0, 87.0],
        "slide_annual_anchor": [130.0, 91.0],
    }
    for year in range(2013, 2026):
        for lineage_id, source_family in source_families.items():
            diagnosed, art = state_by_lineage[lineage_id]
            flow = 7.0 + float((year + len(lineage_id)) % 6)
            diagnosed = diagnosed + 0.82 * flow + (0.6 if lineage_id == "slide_annual_anchor" else 0.0)
            art = min(diagnosed, art + 0.48 * flow)
            state_by_lineage[lineage_id] = [diagnosed, art]
            source_tier, measurement_class, series_kind = source_family.split("|")
            provenance = {
                metric: {
                    "source_id": f"{lineage_id}_{metric}",
                    "source_tier": source_tier,
                    "measurement_class": measurement_class,
                    "series_kind": series_kind,
                    "support_partition": "common_support",
                    "aggregation_mode": "quarterly_observed",
                    "observation_role": "direct_target",
                    "allowed_use": "training",
                }
                for metric in [
                    "diagnosed_plhiv",
                    "alive_on_art",
                    "tested_for_viral_load",
                    "virally_suppressed",
                    "new_diagnosed_cases_period",
                ]
            }
            quarter = "Q4" if lineage_id == "slide_annual_anchor" else ("Q2" if lineage_id == "doh_monthly" else "Q3")
            rows.append(
                {
                    "quarter": f"{year}-{quarter}",
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "tested_for_viral_load": 0.6 * art,
                    "virally_suppressed": 0.72 * 0.6 * art,
                    "new_diagnosed_cases_period": flow,
                    "metric_provenance": provenance,
                }
            )

    selector = _fit_r12_annual_anchor_head_selector(rows)
    assert selector["status"] == "completed"
    assert selector["lineage_id"] == "slide_annual_anchor"
    assert selector["selected_family"] in {
        "multi_horizon_weighted_process",
        "support_reporting_bias",
        "local_level_filter",
    }

    predictions, summary = _candidate_predictions(
        rows[:24],
        rows[24:],
        family="r12_stock_cone_safe_annual_trajectory_process",
    )
    assert predictions
    assert summary["annual_anchor_selector"]["lineage_id"] == "slide_annual_anchor"
    assert any(row["annual_anchor_stock_row"] for row in summary["mutation_rows"])

    report = _build_r12_08_route_aware_two_head_candidate_report(
        rows=rows,
        start_year=2013,
        end_year=2025,
        production_min_train_years=4,
        diagnostic_min_train_years=1,
        candidate_family="r12_stock_cone_safe_annual_trajectory_process",
        experiment_id="R12-09",
        schema_version="phase3_dynamic.r12_09_stock_cone_safe_annual_trajectory_candidate.v1",
        prediction_mutation="enabled_stock_cone_safe_annual_trajectory_head",
        r10_horizon_replay={
            "horizon_rows": [
                {
                    "horizon_years": 1,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 3,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
                {
                    "horizon_years": 5,
                    "status": "pass",
                    "reference_experiment_id": "EXP-R10-TEST",
                    "reference_quarterly_mean_mae": 10.0,
                },
            ]
        },
    )

    assert report["experiment_id"] == "R12-09"
    assert report["schema_version"] == "phase3_dynamic.r12_09_stock_cone_safe_annual_trajectory_candidate.v1"
    assert report["prediction_mutation"] == "enabled_stock_cone_safe_annual_trajectory_head"
    assert report["candidate_family"] == "r12_stock_cone_safe_annual_trajectory_process"
    assert {row["candidate_family"] for row in report["route_horizon_rows"]} == {
        "r12_stock_cone_safe_annual_trajectory_process"
    }


def test_r12_10_official_annual_challenge_keeps_required_heads_out_of_training() -> None:
    rows = []
    for index, year in enumerate(range(2015, 2026)):
        diagnosed = 120.0 + 8.0 * index
        art = 84.0 + 7.0 * index
        provenance = {
            "annual_new_infections": {
                "source_id": "wdi_sh_hiv_incd_tl",
                "source_tier": "canonical_external_reference_no_local_overlap",
                "support_partition": "common_support",
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
                "measurement_semantics": "modeled_estimate",
            },
            "annual_aids_deaths": {
                "source_id": "official_local_corpus_deaths",
                "source_tier": "official_local_corpus",
                "support_partition": "common_support",
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
                "measurement_semantics": "modeled_estimate",
            },
            "estimated_plhiv": {
                "source_id": "spectrum_like_total",
                "source_tier": "external_multinational_hiv_panel",
                "support_partition": "common_support",
                "observation_role": "auxiliary_likelihood",
                "allowed_use": "auxiliary_likelihood",
                "measurement_semantics": "modeled_estimate",
            },
        }
        for metric in [
            "diagnosed_plhiv",
            "alive_on_art",
            "tested_for_viral_load",
            "virally_suppressed",
        ]:
            provenance[metric] = {
                "source_id": f"annual_slide_{metric}",
                "source_tier": "official_user_provided_slide",
                "support_partition": "common_support",
                "observation_role": "direct_target",
                "allowed_use": "training",
                "measurement_semantics": "stock_anchor",
            }
        rows.append(
            {
                "quarter": f"{year}-Q4",
                "annual_new_infections": 10.0 + index,
                "annual_aids_deaths": 2.0 + 0.2 * index,
                "estimated_plhiv": 150.0 + 9.0 * index,
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": art,
                "tested_for_viral_load": 0.62 * art,
                "virally_suppressed": 0.74 * 0.62 * art,
                "metric_provenance": provenance,
            }
        )

    report = _build_r12_official_annual_challenge_gate_report(
        rows=rows,
        start_year=2015,
        end_year=2025,
        min_train_years=3,
        candidate_families=("multi_horizon_weighted_process",),
        horizons=(1,),
    )

    required_records = [
        row
        for row in report["score_records"]
        if row["metric_name"] in {"annual_new_infections", "annual_aids_deaths", "estimated_plhiv"}
    ]
    assert report["experiment_id"] == "R12-10A"
    assert report["status"] == "pass"
    assert required_records
    assert all(row["training_use"] == "train_origin_weak_measurement_head" for row in required_records)
    assert all(row["candidate_value"] is not None for row in required_records)
    assert report["blockers"] == []
    assert report["scored_required_model_head_count"] > 0
    assert report["annual_measurement_head_rows"]
    plhiv_head_rows = [
        row
        for row in report["annual_measurement_head_rows"]
        if row["metric_name"] == "estimated_plhiv" and row.get("conservation_residual") is not None
    ]
    assert plhiv_head_rows
    assert all(row["joint_conservation_status"] == "completed" for row in plhiv_head_rows)
    assert all(abs(float(row["conservation_residual"])) <= 1e-9 for row in plhiv_head_rows)
    assert all(row.get("mass_balance_plhiv") is not None for row in plhiv_head_rows)
    assert report["scored_cascade_metric_count"] > 0


def test_r12_10_program_nowcast_branch_is_doh_program_scoped() -> None:
    source_families = {
        "doh_quarterly": "official_doh_archive|program_observed_harp|quarterly_snapshot",
        "doh_monthly": "official_doh_archive|program_observed_harp|monthly_snapshot",
        "slide_annual_anchor": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    }
    rows = []
    state_by_lineage = {
        "doh_quarterly": [130.0, 90.0],
        "doh_monthly": [128.0, 89.0],
        "slide_annual_anchor": [135.0, 93.0],
    }
    for year in range(2013, 2026):
        for lineage_id, source_family in source_families.items():
            diagnosed, art = state_by_lineage[lineage_id]
            flow = 8.0 + float((year + len(lineage_id)) % 5)
            diagnosed = diagnosed + 0.8 * flow + (0.4 if lineage_id == "slide_annual_anchor" else 0.0)
            art = min(diagnosed, art + 0.5 * flow)
            state_by_lineage[lineage_id] = [diagnosed, art]
            source_tier, measurement_class, series_kind = source_family.split("|")
            provenance = {
                metric: {
                    "source_id": f"{lineage_id}_{metric}",
                    "source_tier": source_tier,
                    "measurement_class": measurement_class,
                    "series_kind": series_kind,
                    "support_partition": "common_support",
                    "aggregation_mode": "quarterly_observed",
                    "observation_role": "direct_target",
                    "allowed_use": "training",
                }
                for metric in [
                    "diagnosed_plhiv",
                    "alive_on_art",
                    "tested_for_viral_load",
                    "virally_suppressed",
                    "new_diagnosed_cases_period",
                ]
            }
            quarter = "Q4" if lineage_id == "slide_annual_anchor" else ("Q2" if lineage_id == "doh_monthly" else "Q3")
            rows.append(
                {
                    "quarter": f"{year}-{quarter}",
                    "diagnosed_plhiv": diagnosed,
                    "alive_on_art": art,
                    "tested_for_viral_load": 0.60 * art,
                    "virally_suppressed": 0.72 * 0.60 * art,
                    "new_diagnosed_cases_period": flow,
                    "metric_provenance": provenance,
                }
            )

    train_rows = [row for row in rows if int(row["quarter"][:4]) <= 2022]
    holdout_rows = [row for row in rows if int(row["quarter"][:4]) == 2023]
    predictions, summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r12_program_nowcast_mixed_quarterly_process",
    )

    assert predictions
    assert summary["family"] == "r12_program_nowcast_mixed_quarterly_process"
    assert summary["monthly_reporting_state_process"]["program_train_row_count"] > 0
    assert summary["monthly_reporting_state_process"]["status"] == "completed"
    latent_process = summary["monthly_reporting_state_process"]["latent_reporting_intensity_process"]
    assert latent_process["status"] == "completed"
    assert latent_process["latent_intensity_by_ordinal"]
    mutation_by_quarter = {row["quarter"]: row for row in summary["mutation_rows"]}
    assert mutation_by_quarter["2023-Q2"]["program_row"] is True
    assert mutation_by_quarter["2023-Q3"]["program_row"] is True
    assert mutation_by_quarter["2023-Q2"]["latent_reporting_intensity"] is not None
    assert mutation_by_quarter["2023-Q4"]["program_row"] is False
    assert mutation_by_quarter["2023-Q4"]["mutated_metrics"] == []

    r14_predictions, r14_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r14_two_factor_program_process",
    )

    assert r14_predictions
    assert r14_summary["family"] == "r14_two_factor_program_process"
    assert r14_summary["two_factor_monthly_state_process"]["status"] == "completed"
    r14_mutation_by_quarter = {row["quarter"]: row for row in r14_summary["mutation_rows"]}
    assert r14_mutation_by_quarter["2023-Q2"]["latent_support_reporting_availability"] is not None
    assert r14_mutation_by_quarter["2023-Q2"]["latent_program_volume_shock"] is not None
    assert r14_mutation_by_quarter["2023-Q4"]["program_row"] is False

    r14b_predictions, r14b_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r14_program_long_horizon_calibrated_process",
    )

    assert r14b_predictions
    assert r14b_summary["family"] == "r14_program_long_horizon_calibrated_process"
    assert r14b_summary["long_horizon_drift_model"]["reference_family"] == "r14_two_factor_program_process"
    assert set(r14b_summary["long_horizon_drift_model"]["corrected_metrics"]) == {
        "diagnosed_plhiv",
        "alive_on_art",
        "new_diagnosed_cases_period",
    }
    r14b_mutation_by_quarter = {row["quarter"]: row for row in r14b_summary["mutation_rows"]}
    assert r14b_mutation_by_quarter["2023-Q2"]["program_row"] is True
    assert r14b_mutation_by_quarter["2023-Q4"]["program_row"] is False

    r15_predictions, r15_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r15_velocity_envelope_process",
    )

    assert r15_predictions
    assert r15_summary["family"] == "r15_velocity_envelope_process"
    assert r15_summary["velocity_envelope_selector"]["reference_family"] == "r14_program_long_horizon_calibrated_process"
    assert set(r15_summary["selected_policy_by_metric"]) == {
        "diagnosed_plhiv",
        "alive_on_art",
        "new_diagnosed_cases_period",
    }

    r16_predictions, r16_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r16_support_cadence_stock_process",
    )

    assert r16_predictions
    assert r16_summary["family"] == "r16_support_cadence_stock_process"
    assert r16_summary["support_cadence_stock_selector"]["reference_family"] == "r15_velocity_envelope_process"
    assert r16_summary["support_cadence_stock_selector"]["selected_diagnosed_cap_policy"]
    assert r16_summary["support_cadence_stock_selector"]["selected_flow_support_cadence_policy"]

    r17_predictions, r17_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r17_art_flow_teacher_process",
    )

    assert r17_predictions
    assert r17_summary["family"] == "r17_art_flow_teacher_process"
    assert r17_summary["art_flow_teacher_selector"]["reference_family"] == "r16_support_cadence_stock_process"
    assert r17_summary["art_flow_teacher_selector"]["teacher_source"] == "frozen_horizon_matched_EXP_R10_replay"

    r18_predictions, r18_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r18_evidence_backed_art_process",
    )

    assert r18_predictions
    assert r18_summary["family"] == "r18_evidence_backed_art_process"
    assert r18_summary["art_process_selector"]["reference_family"] == "r16_support_cadence_stock_process"
    assert r18_summary["evidence_backed_art_process"]["art_stock_transition"]["status"] == "completed"
    assert r18_summary["evidence_backed_art_process"]["art_retention_fraction"] is not None

    r19_predictions, r19_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r19_joint_service_cascade_process",
    )

    assert r19_predictions
    assert r19_summary["family"] == "r19_joint_service_cascade_process"
    assert r19_summary["joint_service_selector"]["reference_family"] == "r18_evidence_backed_art_process"
    assert r19_summary["joint_service_process"]["vl_testing_channel"]["status"] == "completed"
    assert r19_summary["joint_service_process"]["suppression_channel"]["status"] == "completed"
    assert r19_summary["joint_service_process"]["diagnosis_flow_shape_process"]["status"] == "completed"


def test_r60_experiment_queue_has_broad_mode_coverage() -> None:
    specs = _r60_experiment_specs()

    assert len(specs) == 127
    assert [spec["experiment_id"] for spec in specs[:3]] == ["R60-001", "R60-002", "R60-003"]
    assert specs[-1]["experiment_id"] == "R60-127"
    assert {spec["mode"] for spec in specs} == {
        "anchor_simplex",
        "conditional_backhalf_rate",
        "previous_failure_fallback",
    }
    assert sum(1 for spec in specs if spec["mode"] == "anchor_simplex") == 96
    assert sum(1 for spec in specs if spec["mode"] == "conditional_backhalf_rate") == 27
    assert sum(1 for spec in specs if spec["mode"] == "previous_failure_fallback") == 4
    assert any(
        spec["mode"] == "anchor_simplex"
        and set(spec["anchor_metrics"]) == {
            "estimated_plhiv",
            "diagnosed_plhiv",
            "alive_on_art",
            "tested_for_viral_load",
            "virally_suppressed",
        }
        for spec in specs
    )


def test_r60_candidate_set_filters_keep_only_defensible_families() -> None:
    rows = [
        {
            "candidate_family": _r60_carry_forward_family,
            "mean_regional_normalized_absolute_error": 0.10,
            "mean_aggregate_mass_normalized_absolute_error": 0.10,
            "mean_regional_share_half_l1_error": 0.10,
        },
        {
            "candidate_family": _r60_reference_family,
            "mean_regional_normalized_absolute_error": 0.08,
            "mean_aggregate_mass_normalized_absolute_error": 0.09,
            "mean_regional_share_half_l1_error": 0.09,
        },
        {
            "candidate_family": "dominates_reference",
            "mean_regional_normalized_absolute_error": 0.07,
            "mean_aggregate_mass_normalized_absolute_error": 0.08,
            "mean_regional_share_half_l1_error": 0.08,
        },
        {
            "candidate_family": "regional_only_improvement",
            "mean_regional_normalized_absolute_error": 0.06,
            "mean_aggregate_mass_normalized_absolute_error": 0.12,
            "mean_regional_share_half_l1_error": 0.11,
        },
        {
            "candidate_family": "dominated",
            "mean_regional_normalized_absolute_error": 0.12,
            "mean_aggregate_mass_normalized_absolute_error": 0.12,
            "mean_regional_share_half_l1_error": 0.12,
        },
    ]

    pareto = _r60_candidate_set_families(rows, kind="pareto")
    carry_safe = _r60_candidate_set_families(rows, kind="carry_nonregress_all")
    mixed_safe = _r60_candidate_set_families(rows, kind="r52_nonregress_regional_and_carry_mass_share")

    assert "dominates_reference" in pareto
    assert "regional_only_improvement" in pareto
    assert "dominated" not in pareto
    assert "regional_only_improvement" not in carry_safe
    assert "dominates_reference" in carry_safe
    assert _r60_reference_family in mixed_safe
    assert "regional_only_improvement" not in mixed_safe
    assert "dominated" not in mixed_safe


def test_r60_mean_gate_requires_current_r59_nonregression() -> None:
    carry = {
        "mean_regional_normalized_absolute_error": 0.10,
        "mean_aggregate_mass_normalized_absolute_error": 0.10,
        "mean_regional_share_half_l1_error": 0.10,
    }
    reference = {
        "mean_regional_normalized_absolute_error": 0.09,
        "mean_aggregate_mass_normalized_absolute_error": 0.09,
        "mean_regional_share_half_l1_error": 0.09,
    }
    r54_gate = {
        "promoted_mean_regional_normalized_absolute_error": 0.08,
        "promoted_mean_aggregate_mass_normalized_absolute_error": 0.08,
        "promoted_mean_regional_share_half_l1_error": 0.08,
    }
    r59_gate = {
        "ensemble_mean_regional_normalized_absolute_error": 0.075,
        "ensemble_mean_aggregate_mass_normalized_absolute_error": 0.075,
        "ensemble_mean_regional_share_half_l1_error": 0.075,
    }
    passing = {
        "mean_regional_normalized_absolute_error": 0.074,
        "mean_aggregate_mass_normalized_absolute_error": 0.075,
        "mean_regional_share_half_l1_error": 0.075,
    }
    failing = {
        "mean_regional_normalized_absolute_error": 0.074,
        "mean_aggregate_mass_normalized_absolute_error": 0.076,
        "mean_regional_share_half_l1_error": 0.075,
    }

    pass_status, pass_blockers = _r60_candidate_mean_gate(
        passing,
        carry=carry,
        reference=reference,
        r54_gate=r54_gate,
        r59_gate=r59_gate,
    )
    fail_status, fail_blockers = _r60_candidate_mean_gate(
        failing,
        carry=carry,
        reference=reference,
        r54_gate=r54_gate,
        r59_gate=r59_gate,
    )

    assert pass_status is True
    assert pass_blockers == []
    assert fail_status is False
    assert "r59_mass_regressed" in fail_blockers


def test_r61_specs_include_conservative_and_train_origin_router_branches() -> None:
    specs = _r61_experiment_specs("r60_best", "r54_best")

    assert len(specs) == 8
    assert [spec["experiment_id"] for spec in specs] == [f"R61-{index:03d}" for index in range(1, 9)]
    assert {spec["mode"] for spec in specs} == {
        "best_r60_replay",
        "metric_backoff",
        "train_prior_metric_router",
    }
    assert any(
        spec["mode"] == "metric_backoff"
        and spec["metric_family"].get("tested_for_viral_load") == _r60_carry_forward_family
        and spec["metric_family"].get("virally_suppressed") == _r60_carry_forward_family
        for spec in specs
    )
    assert any(spec["mode"] == "train_prior_metric_router" and spec["default_family"] == _r60_reference_family for spec in specs)


def test_r61_metric_backoff_copies_whole_metric_stream_and_projects_cone() -> None:
    base_rows = [
        {
            "candidate_family": "base",
            "holdout_period": "2025-Q1",
            "region": "A",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 80.0,
            "alive_on_art": 70.0,
            "tested_for_viral_load": 60.0,
            "virally_suppressed": 55.0,
        }
    ]
    source_rows = [
        {
            "candidate_family": _r60_carry_forward_family,
            "holdout_period": "2025-Q1",
            "region": "A",
            "estimated_plhiv": 100.0,
            "diagnosed_plhiv": 80.0,
            "alive_on_art": 70.0,
            "tested_for_viral_load": 65.0,
            "virally_suppressed": 66.0,
        }
    ]

    rows = _r61_copy_metric_from_family(
        base_rows,
        source_rows,
        metric_family={
            "tested_for_viral_load": _r60_carry_forward_family,
            "virally_suppressed": _r60_carry_forward_family,
        },
        candidate_family="r61_test",
    )

    assert len(rows) == 1
    assert rows[0]["candidate_family"] == "r61_test"
    assert rows[0]["tested_for_viral_load"] == 65.0
    assert rows[0]["virally_suppressed"] == 65.0
    assert rows[0]["tested_for_viral_load_source_family"] == _r60_carry_forward_family
    assert rows[0]["virally_suppressed_source_family"] == _r60_carry_forward_family


def test_r61_r60_nonregression_blocks_any_mean_regression() -> None:
    r60_gate = {
        "best_mean_regional_normalized_absolute_error": 0.05,
        "best_mean_aggregate_mass_normalized_absolute_error": 0.04,
        "best_mean_regional_share_half_l1_error": 0.02,
    }

    passing, passing_blockers = _r61_r60_nonregression(
        {
            "mean_regional_normalized_absolute_error": 0.05,
            "mean_aggregate_mass_normalized_absolute_error": 0.04,
            "mean_regional_share_half_l1_error": 0.02,
        },
        r60_gate,
    )
    failing, failing_blockers = _r61_r60_nonregression(
        {
            "mean_regional_normalized_absolute_error": 0.049,
            "mean_aggregate_mass_normalized_absolute_error": 0.041,
            "mean_regional_share_half_l1_error": 0.02,
        },
        r60_gate,
    )

    assert passing is True
    assert passing_blockers == []
    assert failing is False
    assert failing_blockers == ["r60_mass_regressed"]


def test_r62_prior_oracle_family_policies_use_only_previous_periods() -> None:
    selection_rows = [
        {
            "holdout_period": "2024-Q3",
            "metric_name": "alive_on_art",
            "selected_candidate_family": "family_a",
        },
        {
            "holdout_period": "2024-Q4",
            "metric_name": "alive_on_art",
            "selected_candidate_family": "family_b",
        },
        {
            "holdout_period": "2025-Q1",
            "metric_name": "alive_on_art",
            "selected_candidate_family": "future_family",
        },
        {
            "holdout_period": "2024-Q3",
            "metric_name": "diagnosed_plhiv",
            "selected_candidate_family": "wrong_metric",
        },
    ]

    recency_family, recency_reason, recency_count = _r62_most_recent_prior_family(
        selection_rows,
        period="2025-Q1",
        metric="alive_on_art",
        default_family="default",
    )
    frequency_family, frequency_reason, frequency_count = _r62_frequency_prior_family(
        selection_rows,
        period="2025-Q1",
        metric="alive_on_art",
        default_family="default",
    )
    default_family, default_reason, default_count = _r62_most_recent_prior_family(
        selection_rows,
        period="2024-Q3",
        metric="alive_on_art",
        default_family="default",
    )

    assert recency_family == "family_b"
    assert recency_reason == "most_recent_prior_oracle_label"
    assert recency_count == 2
    assert frequency_family in {"family_a", "family_b"}
    assert frequency_reason == "most_frequent_prior_oracle_label"
    assert frequency_count == 2
    assert default_family == "default"
    assert default_reason == "default_no_prior_oracle_label"
    assert default_count == 0


def test_r62_component_policy_splits_total_and_share_families() -> None:
    family = "national_total_adapter__total=aggregate_log_trend__share=similarity_proxy_log_delta"
    assert _r62_adapter_components(family) == ("aggregate_log_trend", "similarity_proxy_log_delta")
    assert _r62_adapter_components("module_local_selector") == ("module_local_selector", "module_local_selector")

    total, share, reason, count = _r62_component_prior_family(
        [
            {
                "holdout_period": "2024-Q3",
                "metric_name": "tested_for_viral_load",
                "selected_candidate_family": family,
            },
            {
                "holdout_period": "2024-Q4",
                "metric_name": "tested_for_viral_load",
                "selected_candidate_family": "national_total_adapter__total=module_local_selector__share=aggregate_log_trend",
            },
        ],
        period="2025-Q1",
        metric="tested_for_viral_load",
        default_family="regional_carry_forward",
        policy="component_recency",
    )

    assert (total, share) == ("module_local_selector", "aggregate_log_trend")
    assert reason == "most_recent_prior_oracle_components"
    assert count == 2


def test_r63_candidate_error_rows_are_region_metric_specific() -> None:
    rows = [
        {
            "candidate_family": "a",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "estimated_plhiv": 9.0,
            "diagnosed_plhiv": 6.0,
            "alive_on_art": 4.0,
            "tested_for_viral_load": 2.0,
            "virally_suppressed": 1.0,
        }
    ]
    targets = {
        ("2024-Q3", "NCR"): {
            "estimated_plhiv": 10.0,
            "diagnosed_plhiv": 5.0,
            "alive_on_art": 4.0,
            "tested_for_viral_load": 1.0,
            "virally_suppressed": 1.0,
        }
    }

    error_rows = _r63_candidate_error_rows(rows, targets)
    by_metric = {row["metric_name"]: row for row in error_rows}

    assert by_metric["estimated_plhiv"]["absolute_error"] == 1.0
    assert by_metric["diagnosed_plhiv"]["normalized_absolute_error"] == 0.2
    assert by_metric["alive_on_art"]["absolute_error"] == 0.0


def test_r63_student_uses_prior_region_metric_errors_only() -> None:
    source_rows = []
    for period, value_a, value_b in [
        ("2024-Q3", 10.0, 20.0),
        ("2024-Q4", 11.0, 21.0),
    ]:
        source_rows.extend(
            [
                {
                    "candidate_family": "a",
                    "holdout_period": period,
                    "region": "NCR",
                    "estimated_plhiv": value_a,
                    "diagnosed_plhiv": value_a,
                    "alive_on_art": value_a,
                    "tested_for_viral_load": value_a,
                    "virally_suppressed": value_a,
                },
                {
                    "candidate_family": "b",
                    "holdout_period": period,
                    "region": "NCR",
                    "estimated_plhiv": value_b,
                    "diagnosed_plhiv": value_b,
                    "alive_on_art": value_b,
                    "tested_for_viral_load": value_b,
                    "virally_suppressed": value_b,
                },
            ]
        )
    error_rows = [
        {
            "candidate_family": "a",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": metric,
            "normalized_absolute_error": 0.5,
        }
        for metric in ["estimated_plhiv", "diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"]
    ] + [
        {
            "candidate_family": "b",
            "holdout_period": "2024-Q3",
            "region": "NCR",
            "metric_name": metric,
            "normalized_absolute_error": 0.1,
        }
        for metric in ["estimated_plhiv", "diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"]
    ]

    rows, selections = _r63_region_metric_student_prediction_rows(
        source_rows,
        error_rows,
        default_family="a",
    )

    selected_q3 = [row for row in selections if row["holdout_period"] == "2024-Q3"]
    selected_q4 = [row for row in selections if row["holdout_period"] == "2024-Q4"]
    q4_prediction = next(row for row in rows if row["holdout_period"] == "2024-Q4")
    assert {row["selected_source_family"] for row in selected_q3} == {"a"}
    assert {row["selected_source_family"] for row in selected_q4} == {"b"}
    assert q4_prediction["diagnosed_plhiv"] == 21.0
    assert all(row["leakage_training_status"] == "uses_prior_region_metric_errors_only" for row in selected_q4)


def test_r64_gap_rows_rank_student_oracle_gap_and_recommend_evidence() -> None:
    oracle_rows = [
        {
            "holdout_period": "2025-Q1",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "selected_source_family": "oracle_family",
            "selected_normalized_absolute_error": 0.1,
        }
    ]
    student_rows = [
        {
            "holdout_period": "2025-Q1",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "selected_source_family": "student_family",
        }
    ]
    error_rows = [
        {
            "candidate_family": "student_family",
            "holdout_period": "2025-Q1",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "normalized_absolute_error": 0.4,
            "target_value": 100.0,
        },
        {
            "candidate_family": "r60",
            "holdout_period": "2025-Q1",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "normalized_absolute_error": 0.3,
            "target_value": 100.0,
        },
        {
            "candidate_family": "oracle_family",
            "holdout_period": "2025-Q1",
            "region": "NCR",
            "metric_name": "alive_on_art",
            "normalized_absolute_error": 0.1,
            "target_value": 100.0,
        },
    ]

    rows = _r64_gap_rows(
        oracle_rows=oracle_rows,
        student_rows=student_rows,
        candidate_error_rows=error_rows,
        r60_family="r60",
    )

    assert len(rows) == 1
    assert rows[0]["student_gap_vs_oracle"] == pytest.approx(0.3)
    assert rows[0]["r60_gap_vs_oracle"] == pytest.approx(0.2)
    assert rows[0]["metric_stream"] == "art_stock_retention"
    assert "retention" in rows[0]["evidence_recommendation"]


def test_r64_aggregate_priority_orders_by_positive_gap() -> None:
    rows = [
        {
            "metric_name": "alive_on_art",
            "oracle_source_family": "a",
            "student_gap_vs_oracle": 0.5,
            "r60_gap_vs_oracle": 0.4,
            "evidence_recommendation": _r64_evidence_recommendation("alive_on_art"),
        },
        {
            "metric_name": "diagnosed_plhiv",
            "oracle_source_family": "b",
            "student_gap_vs_oracle": 0.1,
            "r60_gap_vs_oracle": 0.1,
            "evidence_recommendation": _r64_evidence_recommendation("diagnosed_plhiv"),
        },
    ]

    priorities = _r64_aggregate_priority(rows, ("metric_name",), label="metric")

    assert priorities[0]["metric_name"] == "alive_on_art"
    assert priorities[0]["sum_positive_student_gap_vs_oracle"] == pytest.approx(0.5)
    assert priorities[1]["metric_name"] == "diagnosed_plhiv"


def test_r65_module_status_blocks_regional_transmission_until_service_support() -> None:
    rows = _r65_module_status_rows(
        r42={
            "strict_gate_status": "pass",
            "promotion_claim": "freeze_as_national_research_champion",
        },
        r46={
            "lineage_gate": {
                "strict_phase3_prior_count": 0,
                "sensitivity_only_driver_count": 3,
            }
        },
        r53={
            "registry_gate": {
                "regional_adapter_promoted": True,
                "regional_readout_promoted": True,
                "regional_r60_experiment_queue_status": "mean_promoted_split_limited",
            }
        },
        r60={"queue_gate": {"status": "regional_experiment_queue_mean_promoted_split_limited"}},
        r64={"support_gap_gate": {"top_metric_stream": "vl_suppression_service"}},
    )
    by_module = {row["module_id"]: row for row in rows}
    gate = _r65_gate(rows)

    assert by_module["national_forecast_readout"]["readiness_status"] == "ready"
    assert by_module["national_full_transmission_scenario"]["readiness_status"] == "sensitivity_only"
    assert by_module["regional_full_transmission_model"]["readiness_status"] == "blocked"
    assert "vl_suppression_service" in by_module["regional_full_transmission_model"]["primary_blocker"]
    assert gate["status"] == "full_transmission_model_not_yet_ready"
    assert gate["national_forecast_readout_ready"] is True
    assert gate["regional_full_transmission_ready"] is False


def test_r65_state_equation_rows_include_incidence_and_service_states() -> None:
    rows = _r65_state_equation_rows()
    equations = {row["state_or_transition"]: row for row in rows}

    assert "S_eff -> I" in equations
    assert "A -> VL_tested -> suppressed" in equations
    assert "highest_regional_blocker" == equations["A -> VL_tested -> suppressed"]["current_status"]


def test_r66_classifies_local_sources_by_allowed_use(tmp_path: Path) -> None:
    hasp_pdf = tmp_path / "2025_Q2-HIV-AIDS-Surveillance-Report-of-the-Philippines-1.pdf"
    incidence_csv = tmp_path / "New HIV infections_New HIV infections - All ages_Population_ All ages.csv"
    kp_csv = tmp_path / "Men who have sex with men_HIV prevalence among men who have sex with men_Population_ Total.csv"
    unknown_xlsx = tmp_path / "unknown.xlsx"
    for path in (hasp_pdf, incidence_csv, kp_csv, unknown_xlsx):
        path.write_text("source", encoding="utf-8")

    hasp = _r66_classify_local_source(hasp_pdf)
    incidence = _r66_classify_local_source(incidence_csv)
    kp = _r66_classify_local_source(kp_csv)
    unknown = _r66_classify_local_source(unknown_xlsx)

    assert hasp["observation_role"] == "direct_target"
    assert "vl_suppression_service" in hasp["module_targets"]
    assert incidence["observation_role"] == "validation_only"
    assert "quarterly training truth" in incidence["evidence_note"]
    assert kp["allowed_use"] == "determinant_sensitivity_until_source_stable"
    assert unknown["observation_role"] == "quarantined"


def test_r66_module_coverage_and_gap_actions_are_source_aware(tmp_path: Path) -> None:
    source_rows = []
    for filename in [
        "2025_Q2-HIV-AIDS-Surveillance-Report-of-the-Philippines-1.pdf",
        "Treatment cascade_People living with HIV who have suppressed viral loads (%)_Population_ All ages.csv",
        "Men who have sex with men_Men who have sex with men_ Size estimate.csv",
    ]:
        path = tmp_path / filename
        path.write_text("source", encoding="utf-8")
        source_rows.append(_r66_classify_local_source(path))

    module_rows = _r66_module_coverage_rows(source_rows)
    module_by_id = {row["module_target"]: row for row in module_rows}
    gaps = _r66_support_gap_rows(
        {
            "metric_stream_priority_rows": [
                {
                    "metric_stream": "vl_suppression_service",
                    "entry_count": 10,
                    "sum_positive_student_gap_vs_oracle": 1.5,
                }
            ]
        },
        module_rows,
    )
    gate = _r66_gate(source_rows, module_rows)

    assert module_by_id["vl_suppression_service"]["usable_supported_count"] >= 1
    assert module_by_id["kp_overlay"]["determinant_context_count"] == 1
    assert gaps[0]["evidence_action"].startswith("extract regional VL-tested")
    assert gate["status"] == "source_base_ready_for_model_family_queue"
    assert _r66_wdi_specs()[0]["indicator"] == "SP.POP.TOTL"


def test_r67_model_queue_keeps_determinants_locked_and_leakage_diagnostic() -> None:
    coverage = {
        "diagnosis_reporting": {"usable_supported_count": 2, "determinant_context_count": 0},
        "art_retention": {"usable_supported_count": 2, "determinant_context_count": 0},
        "vl_suppression_service": {"usable_supported_count": 2, "determinant_context_count": 0},
        "mortality_reporting": {"usable_supported_count": 1, "determinant_context_count": 0},
        "incidence_validation": {"usable_supported_count": 1, "determinant_context_count": 0},
        "incidence_pressure": {"usable_supported_count": 0, "determinant_context_count": 3},
        "kp_overlay": {"usable_supported_count": 0, "determinant_context_count": 2},
        "regional_shrinkage": {"usable_supported_count": 0, "determinant_context_count": 2},
        "prep_persistence": {"usable_supported_count": 1, "determinant_context_count": 0},
    }

    rows = _r67_model_family_rows(coverage)
    by_id = {row["family_id"]: row for row in rows}
    experiments = _r67_next_experiment_rows(rows)
    gate = _r67_gate(rows)

    assert len(rows) == 12
    assert by_id["R67-M04_kp_metapopulation_transmission_patch"]["readiness_status"] == "sensitivity_only_until_R46_source_stable"
    assert by_id["R67-M08_online_expert_leakage_teacher_student"]["readiness_status"] == "diagnostic_only_until_student_passes_R60_contract"
    assert by_id["R67-M12_identifiability_first_null_model"]["readiness_status"] == "ready_as_guardrail_for_all_R67_models"
    assert any(row["next_action"] == "run_phase2_source_family_falsification_before_model_fit" for row in experiments)
    assert gate["ready_bounded_family_count"] >= 4


def test_r68_bulk_specs_have_allowed_use_and_google_is_context_only() -> None:
    specs = _r68_bulk_source_specs()
    by_id = {spec["source_id"]: spec for spec in specs}

    assert {"unaids_estimates_2025", "unaids_gam_2025", "unaids_kp_atlas_2025", "unaids_ncpi_2025", "google_global_mobility_report"} <= set(by_id)
    assert by_id["unaids_estimates_2025"]["observation_role"] == "validation_only"
    assert by_id["google_global_mobility_report"]["observation_role"] == "prior_context"
    assert "2020_2022" in by_id["google_global_mobility_report"]["allowed_use"]


def test_r68_extracts_only_philippines_rows_from_csv_stream(tmp_path: Path) -> None:
    source = io.BytesIO(
        b"country_region_code,country_region,date,value\n"
        b"PH,Philippines,2020-01-01,1\n"
        b"US,United States,2020-01-01,2\n"
        b"PHL,Philippines,2020-01-02,3\n"
    )
    output = tmp_path / "ph.csv"

    result = _r68_extract_philippines_from_csv_stream(source, output, source_member="test.csv")
    lines = output.read_text(encoding="utf-8").strip().splitlines()

    assert result["rows_scanned"] == 4
    assert result["philippines_rows_written"] == 2
    assert len(lines) == 3
    assert "United States" not in output.read_text(encoding="utf-8")


def test_r68_gate_requires_downloaded_and_extracted_rows() -> None:
    gate = _r68_gate(
        [
            {"download_status": "downloaded"},
            {"download_status": "download_failed"},
        ],
        [{"philippines_rows_written": 3}],
    )

    assert gate["status"] == "bulk_external_sources_ingested"
    assert gate["downloaded_source_count"] == 1
    assert gate["failed_source_count"] == 1
    assert gate["extracted_philippines_row_count"] == 3


def test_r69_compiles_annual_estimates_into_validation_modules(tmp_path: Path) -> None:
    source = tmp_path / "estimates.csv"
    source.write_text(
        "Indicator,Indicator_GId,Unit,Subgroup,Subgroup_Val_GId,Area,Area ID,Area Level,Time Period,Source,Data value,Formatted,Data_Denominator,Footnote\n"
        "New HIV infections,NEW_INFECTIONS,Number,Total,TOTAL,Philippines,PHL,2,2024,UNAIDS_Estimates_,1200,,,\n"
        "AIDS-related deaths,AIDS_DEATHS,Number,Total,TOTAL,Philippines,PHL,2,2024,UNAIDS_Estimates_,30,,,\n",
        encoding="utf-8",
    )

    rows = _r69_compile_unaids_estimates(source)
    by_module = {row["module_target"]: row for row in rows}

    assert by_module["incidence_validation"]["time_period"] == "2024-Q4"
    assert by_module["incidence_validation"]["allowed_use"] == "validation_only_or_weak_measurement"
    assert by_module["mortality_reporting"]["value"] == pytest.approx(30.0)


def test_r69_compiles_google_daily_rows_to_quarterly_features(tmp_path: Path) -> None:
    source = tmp_path / "mobility.csv"
    source.write_text(
        "country_region_code,country_region,sub_region_1,date,retail_and_recreation_percent_change_from_baseline,workplaces_percent_change_from_baseline\n"
        "PH,Philippines,,2020-02-15,10,5\n"
        "PH,Philippines,,2020-03-15,-10,-5\n"
        "PH,Philippines,NCR,2020-04-15,-20,-10\n",
        encoding="utf-8",
    )

    rows = _r69_compile_google_quarterly_rows(source)
    national_retail = next(row for row in rows if row["geography"] == "national" and row["indicator"] == "retail_and_recreation_percent_change_from_baseline")

    assert national_retail["time_period"] == "2020-Q1"
    assert national_retail["value"] == pytest.approx(0.0)
    assert national_retail["days_observed"] == 2
    assert any(row["geography"] == "NCR" and row["time_period"] == "2020-Q2" for row in rows)


def test_r69_readiness_marks_m01_and_m06_ready_when_signals_exist() -> None:
    gam_rows = [
        {"module_target": "diagnosis_reporting"},
        {"module_target": "art_retention"},
        {"module_target": "vl_suppression_service"},
        {"module_target": "prep_persistence"},
    ]
    mobility_rows = [{"module_target": "reporting_disruption"}]
    annual_rows = [{"module_target": "incidence_validation"}, {"module_target": "mortality_reporting"}]
    kp_rows = [{"module_target": "kp_overlay"}]
    policy_rows = [{"module_target": "structural_policy"}]

    rows = _r69_readiness_rows(
        annual_rows=annual_rows,
        gam_rows=gam_rows,
        kp_rows=kp_rows,
        policy_rows=policy_rows,
        mobility_rows=mobility_rows,
    )
    by_family = {row["family_id"]: row for row in rows}
    gate = _r69_gate(rows, total_rows=9)

    assert by_family["R67-M01_hidden_service_intensity_state_space"]["readiness_status"] == "feature_table_ready"
    assert by_family["R67-M06_service_capacity_queue_control"]["readiness_status"] == "feature_table_ready"
    assert gate["status"] == "bulk_signal_features_ready"


def test_r70_queue_prioritizes_ready_bounded_national_steps() -> None:
    family_rows = [
        {
            "family_id": "R67-M01_hidden_service_intensity_state_space",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service",
            "feature_readiness_status": "feature_table_ready",
            "readiness_status": "ready_for_bounded_branch",
            "overfit_guard": "train-only",
            "leakage_policy": "no leakage",
            "source_domain": "astronomy",
            "equation": "Y=qX",
        },
        {
            "family_id": "R67-M04_kp_metapopulation_transmission_patch",
            "module_targets": "incidence_pressure|kp_overlay",
            "feature_readiness_status": "feature_table_ready",
            "readiness_status": "sensitivity_only_until_R46_source_stable",
            "overfit_guard": "source-stable only",
            "leakage_policy": "no validation training",
            "source_domain": "metapopulation",
            "equation": "I=S lambda",
        },
    ]

    rows = _r70_queue_rows(family_rows, max_steps=120)
    gate = _r70_gate(rows)

    assert rows[0]["family_id"] == "R67-M01_hidden_service_intensity_state_space"
    assert rows[0]["geography_scope"] == "national"
    assert rows[0]["allowed_use"] == "bounded_candidate_experiment"
    assert any(row["allowed_use"] == "sensitivity_or_falsification_until_R46_source_stable" for row in rows)
    assert gate["status"] == "scientific_model_build_queue_ready"
    assert gate["queued_step_count"] == 120


def test_r71_compiles_google_and_gam_features_from_r69_paths(tmp_path: Path) -> None:
    mobility = tmp_path / "mobility.csv"
    gam = tmp_path / "gam.csv"
    mobility.write_text(
        "geography,indicator,time_period,value\n"
        "national,workplaces_percent_change_from_baseline,2020-Q1,-20\n"
        "NCR,workplaces_percent_change_from_baseline,2020-Q1,-30\n",
        encoding="utf-8",
    )
    gam.write_text(
        "module_target,time_period,value\n"
        "diagnosis_reporting,2020-Q4,10\n"
        "diagnosis_reporting,2020-Q4,90\n",
        encoding="utf-8",
    )
    report = {
        "artifact_paths": {
            "google_mobility_quarterly_csv": mobility.as_posix(),
            "gam_program_support_csv": gam.as_posix(),
        }
    }

    rows = _r71_compile_feature_rows(report)

    assert rows["2020-Q1"]["google::workplaces_percent_change_from_baseline"] == pytest.approx(-20.0)
    assert rows["2020-Q4"]["gam::diagnosis_reporting::count"] == pytest.approx(2.0)
    assert rows["2020-Q4"]["gam::diagnosis_reporting::log_sum"] == pytest.approx(np.log1p(100.0))


def test_r71_metrics_are_unique_after_back_half_extension() -> None:
    assert len(_r71_metrics) == len(set(_r71_metrics))
    assert {"tested_for_viral_load", "virally_suppressed"} <= set(_r71_metrics)


def test_r71_feature_model_uses_train_rows_and_freezes_forecast_features() -> None:
    train_rows = [
        {"quarter": f"2020-Q{quarter}", "diagnosed_plhiv": 100.0 + 10.0 * quarter, "alive_on_art": 80.0, "new_diagnosed_cases_period": 10.0}
        for quarter in range(1, 5)
    ] + [
        {"quarter": f"2021-Q{quarter}", "diagnosed_plhiv": 150.0 + 10.0 * quarter, "alive_on_art": 90.0, "new_diagnosed_cases_period": 12.0}
        for quarter in range(1, 5)
    ]
    holdout_rows = [
        {"quarter": "2022-Q1", "diagnosed_plhiv": 210.0, "alive_on_art": 100.0, "new_diagnosed_cases_period": 13.0}
    ]
    feature_rows = {
        "2021-Q4": {"google::workplaces_percent_change_from_baseline": -5.0},
        "2022-Q1": {"google::workplaces_percent_change_from_baseline": -50.0},
    }

    model = _r71_fit_linear_feature_model(train_rows, "diagnosed_plhiv", feature_rows, mode="reporting_state")
    predictions, summary = _r71_feature_predictions(
        train_rows,
        holdout_rows,
        feature_rows,
        policy="forecast_safe",
        family="r71_forecast_safe_service_intensity_capacity",
    )

    assert model["status"] == "completed"
    assert summary["policy"] == "forecast_safe"
    assert predictions[0]["quarter"] == "2022-Q1"
    assert predictions[0]["diagnosed_plhiv"] >= 0.0


def test_r71_gate_promotes_only_forecast_safe_branch() -> None:
    summary_rows = [
        {"family": "carry_forward", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.5},
        {"family": "r41_research_champion_reference", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.4},
        {"family": "r71_forecast_safe_service_intensity_capacity", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.3},
        {"family": "r71_contemporaneous_nowcast_service_intensity_capacity", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.2},
    ]

    gate = _r71_gate(summary_rows)

    assert gate["status"] == "r71_forecast_safe_promoted"
    assert gate["forecast_safe_beats_carry_forward"] is True
    assert gate["nowcast_beats_carry_forward"] is True


def test_r72_selector_falls_back_to_r41_without_internal_history() -> None:
    train_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 80.0,
            "new_diagnosed_cases_period": 10.0,
            "tested_for_viral_load": 40.0,
            "virally_suppressed": 35.0,
        }
    ]

    selected = _r72_select_metric_families(train_rows, {}, min_train_years=5)

    assert set(selected) == set(_r71_metrics)
    assert set(selected.values()) == {"r41_research_champion_reference"}


def test_r72_gate_requires_selector_to_beat_carry_and_r41() -> None:
    summary_rows = [
        {"family": "carry_forward", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.4},
        {"family": "r41_research_champion_reference", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.3},
        {"family": "r72_train_backtested_service_feature_selector", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.2},
    ]
    selector_rows = [
        {
            "selected_metric_families": {
                "diagnosed_plhiv": "r71_forecast_safe_service_intensity_capacity",
                "alive_on_art": "r41_research_champion_reference",
            }
        }
    ]

    gate = _r72_gate(summary_rows, selector_rows)

    assert gate["status"] == "r72_selector_promoted"
    assert gate["selector_beats_carry_forward"] is True
    assert gate["selector_beats_r41"] is True
    assert gate["selected_feature_metric_count"] == 1


def test_r73_feature_catalog_keeps_only_known_external_signal_families() -> None:
    feature_rows = {
        "2020-Q1": {
            "google::workplaces_percent_change_from_baseline": -20.0,
            "gam::diagnosis_reporting::count": 2.0,
            "gam::unknown_module::count": 1.0,
        }
    }

    catalog = _r73_feature_catalog(feature_rows)

    assert _r73_feature_family("google::workplaces_percent_change_from_baseline") == "google_mobility"
    assert {row["feature_family"] for row in catalog} == {"gam_diagnosis_reporting", "google_mobility"}


def test_r73_gate_never_promotes_oracle_only_signal() -> None:
    summary_rows = [
        {"family": "carry_forward", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.4},
        {"family": "r41_research_champion_reference", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.3},
        {"family": "r73_forecast_origin_external_signal_selector", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.35},
        {"family": "r73_oracle_contemporaneous_external_signal_selector", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.2},
    ]
    selection_rows = [
        {"policy": "oracle_contemporaneous", "selection": "feature_model"},
        {"policy": "forecast_origin", "selection": "base_family"},
    ]

    gate = _r73_gate(summary_rows, selection_rows)

    assert gate["status"] == "r73_diagnostic_only"
    assert gate["oracle_beats_r41"] is True
    assert gate["forecast_beats_r41"] is False


def test_r74_metric_summary_reports_mean_and_p90_by_metric() -> None:
    rows = [
        {"metric_name": "alive_on_art", "normalized_absolute_error": 0.1},
        {"metric_name": "alive_on_art", "normalized_absolute_error": 0.3},
        {"metric_name": "diagnosed_plhiv", "normalized_absolute_error": 0.2},
    ]

    summary = _r74_metric_summary(rows)

    assert summary["alive_on_art"]["mean_nae"] == pytest.approx(0.2)
    assert summary["alive_on_art"]["p90_nae"] == pytest.approx(0.28)
    assert summary["diagnosed_plhiv"]["mean_nae"] == pytest.approx(0.2)


def test_r74_gate_requires_mean_nonregression_and_p90_gain() -> None:
    summary_rows = [
        {"family": "carry_forward", "metric_name": "__overall__", "mean_normalized_absolute_error": 0.4},
        {
            "family": "r41_research_champion_reference",
            "metric_name": "__overall__",
            "mean_normalized_absolute_error": 0.3,
            "p90_normalized_absolute_error": 0.6,
        },
        {
            "family": "r74_p90_safe_external_tail_selector",
            "metric_name": "__overall__",
            "mean_normalized_absolute_error": 0.3,
            "p90_normalized_absolute_error": 0.5,
        },
    ]
    policy_rows = [{"use_r73": True}, {"use_r73": False}]

    gate = _r74_gate(summary_rows, policy_rows)

    assert gate["status"] == "r74_tail_risk_selector_promoted"
    assert gate["selector_mean_nonregression_vs_r41"] is True
    assert gate["selector_p90_beats_r41"] is True


def test_r75_extracts_all_ages_unaids_required_targets_with_intervals(tmp_path: Path) -> None:
    source = tmp_path / "annual.csv"
    source.write_text(
        "source_id,module_target,time_period,indicator,indicator_gid,subgroup,unit,value\n"
        "unaids_estimates_2025,incidence_validation,2024-Q4,New HIV Infections,NEW_INFECTIONS,All ages estimate,Number,100\n"
        "unaids_estimates_2025,incidence_validation,2024-Q4,New HIV Infections,NEW_INFECTIONS,All ages lower estimate,Number,80\n"
        "unaids_estimates_2025,incidence_validation,2024-Q4,New HIV Infections,NEW_INFECTIONS,All ages upper estimate,Number,120\n"
        "unaids_estimates_2025,incidence_validation,2024-Q4,New HIV Infections,NEW_INFECTIONS,Adults (15+) estimate,Number,90\n"
        "unaids_estimates_2025,mortality_reporting,2024-Q4,AIDS-related deaths,AIDS_DEATHS,All ages estimate,Number,10\n"
        "unaids_estimates_2025,plhiv_stock_validation,2024-Q4,People living with HIV,PLWH,All ages estimate,Number,1000\n",
        encoding="utf-8",
    )

    rows = _r75_bulk_unaids_target_rows(source, external_start_year=2010)
    by_metric = {row["metric_name"]: row for row in rows}

    assert by_metric["annual_new_infections"]["target_value"] == pytest.approx(100.0)
    assert by_metric["annual_new_infections"]["target_lower"] == pytest.approx(80.0)
    assert by_metric["annual_new_infections"]["target_upper"] == pytest.approx(120.0)
    assert by_metric["annual_new_infections"]["metric_provenance"]["observation_role"] == "validation_only"
    assert by_metric["annual_aids_deaths"]["target_value"] == pytest.approx(10.0)
    assert by_metric["estimated_plhiv"]["target_value"] == pytest.approx(1000.0)


def test_r75_merge_targets_marks_validation_only_metric_provenance() -> None:
    observation_rows = [{"quarter": "2024-Q4", "diagnosed_plhiv": 100.0}]
    target_rows = [
        {
            "quarter": "2024-Q4",
            "metric_name": "annual_new_infections",
            "target_value": 50.0,
            "target_lower": 40.0,
            "target_upper": 60.0,
            "target_interval_available": True,
            "metric_provenance": {"observation_role": "validation_only", "allowed_use": "validation_only"},
        }
    ]

    rows = _r75_merge_external_targets_into_observations(observation_rows, target_rows)

    assert rows[0]["diagnosed_plhiv"] == pytest.approx(100.0)
    assert rows[0]["annual_new_infections"] == pytest.approx(50.0)
    assert rows[0]["annual_new_infections_target_lower"] == pytest.approx(40.0)
    assert rows[0]["metric_provenance"]["annual_new_infections"]["allowed_use"] == "validation_only"


def test_r75_gate_promotes_only_when_best_beats_carry_and_interval_coverage_nonregresses() -> None:
    family_rows = [
        {
            "candidate_family": "candidate",
            "candidate_mean_norm_error": 0.2,
            "carry_forward_mean_norm_error": 0.3,
            "candidate_interval_coverage": 0.8,
            "carry_forward_interval_coverage": 0.7,
        }
    ]
    score_rows = [
        {"metric_name": "annual_new_infections", "candidate_norm_error": 0.1, "observation_role": "validation_only", "allowed_use": "validation_only"},
        {"metric_name": "annual_aids_deaths", "candidate_norm_error": 0.2, "observation_role": "validation_only", "allowed_use": "validation_only"},
        {"metric_name": "estimated_plhiv", "candidate_norm_error": 0.3, "observation_role": "validation_only", "allowed_use": "validation_only"},
    ]

    gate = _r75_gate(family_rows, score_rows, target_rows=[{"metric_name": "annual_new_infections"}])

    assert gate["status"] == "bulk_unaids_annual_challenge_pass"
    assert gate["best_candidate_family"] == "candidate"


def test_r75_annual_splits_never_extend_beyond_external_end_year() -> None:
    rows = [{"quarter": f"{year}-Q4"} for year in range(2010, 2026)]

    splits = _r75_rolling_annual_splits(rows, start_year=2019, end_year=2024, min_train_years=5, horizons=(5,))

    assert splits
    assert all(max(split["holdout_years"]) <= 2024 for split in splits)


def test_r76_log_linear_public_trend_forecasts_required_metrics() -> None:
    train_rows = [
        {
            "quarter": f"{year}-Q4",
            "annual_new_infections": float(100 + 10 * (year - 2020)),
            "annual_aids_deaths": float(10 + year - 2020),
            "estimated_plhiv": float(1000 + 100 * (year - 2020)),
        }
        for year in range(2020, 2023)
    ]
    holdout_rows = [{"quarter": "2023-Q4"}]

    rows = _r76_log_linear_prediction_rows(train_rows, holdout_rows)

    assert rows[0]["annual_new_infections"] > train_rows[-1]["annual_new_infections"]
    assert rows[0]["annual_aids_deaths"] > train_rows[-1]["annual_aids_deaths"]
    assert rows[0]["estimated_plhiv"] > train_rows[-1]["estimated_plhiv"]


def test_r76_selector_uses_only_declared_public_comparator_families() -> None:
    rows = [
        {
            "quarter": f"{year}-Q4",
            "annual_new_infections": float(100 + 5 * (year - 2010)),
            "annual_aids_deaths": float(10 + year - 2010),
            "estimated_plhiv": float(1000 + 50 * (year - 2010)),
        }
        for year in range(2010, 2020)
    ]

    selected = _r76_select_metric_families(rows, min_train_years=5, horizons=(1,))

    assert set(selected) == {"annual_new_infections", "annual_aids_deaths", "estimated_plhiv"}
    assert set(selected.values()) <= set(_r76_public_comparator_families)


def test_r76_gate_requires_selected_proxy_to_beat_public_carry_forward() -> None:
    family_rows = [
        {
            "candidate_family": "public_train_selected_annual_proxy",
            "candidate_mean_norm_error": 0.2,
            "candidate_interval_coverage": 0.8,
        },
        {
            "candidate_family": "public_carry_forward",
            "candidate_mean_norm_error": 0.3,
            "candidate_interval_coverage": 0.7,
        },
    ]
    score_rows = [
        {"observation_role": "validation_only", "allowed_use": "validation_only", "candidate_norm_error": 0.2}
    ]

    gate = _r76_gate(family_rows, score_rows, target_rows=[{"metric_name": "annual_new_infections"}])

    assert gate["status"] == "public_domain_annual_comparator_ready"
    assert gate["selected_minus_public_carry_forward_mean_norm_error"] == pytest.approx(-0.1)


def test_r77_matches_phase3_annual_head_to_public_proxy_by_holdout_key() -> None:
    r75 = {
        "bulk_unaids_annual_gate": {"best_candidate_family": "model"},
        "score_rows": [
            {
                "candidate_family": "model",
                "horizon_years": 1,
                "train_end_year": 2022,
                "holdout_years": [2023],
                "quarter": "2023-Q4",
                "year": 2023,
                "metric_name": "annual_new_infections",
                "candidate_norm_error": 0.2,
                "candidate_value": 120.0,
                "target_value": 100.0,
                "candidate_interval_covered": True,
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
            }
        ],
    }
    r76 = {
        "score_rows": [
            {
                "candidate_family": "public_train_selected_annual_proxy",
                "horizon_years": 1,
                "train_end_year": 2022,
                "quarter": "2023-Q4",
                "metric_name": "annual_new_infections",
                "candidate_norm_error": 0.1,
                "candidate_value": 110.0,
                "candidate_interval_covered": True,
            }
        ]
    }

    rows = _r77_matched_proxy_rows(r75, r76)

    assert len(rows) == 1
    assert rows[0]["model_minus_public_proxy_norm_error"] == pytest.approx(0.1)


def test_r77_gate_blocks_annual_superiority_when_public_proxy_is_better() -> None:
    family_rows = [
        {
            "model_family": "model",
            "public_proxy_family": "public_train_selected_annual_proxy",
            "model_mean_norm_error": 0.24,
            "public_proxy_mean_norm_error": 0.20,
            "model_interval_coverage": 0.75,
            "public_proxy_interval_coverage": 0.88,
        }
    ]
    matched_rows = [
        {"observation_role": "validation_only", "allowed_use": "validation_only"},
    ]

    gate = _r77_gate(matched_rows, family_rows)

    assert gate["status"] == "annual_model_blocked_by_public_proxy"
    assert "model_worse_than_public_annual_proxy" in gate["blockers"]
    assert "model_interval_coverage_worse_than_public_annual_proxy" in gate["blockers"]


def test_r78_theil_sen_log_trend_is_train_fitted_and_forecasts_positive() -> None:
    train_rows = [
        {"quarter": f"{year}-Q4", "annual_new_infections": float(100 + 8 * (year - 2020))}
        for year in range(2020, 2024)
    ]

    model = _r78_fit_theil_sen_log_metric(train_rows, "annual_new_infections")
    prediction = _r78_predict_theil_sen_log_metric(model, "2024-Q4")

    assert model["status"] == "completed"
    assert prediction is not None
    assert prediction > train_rows[-1]["annual_new_infections"]


def test_r78_piecewise_log_trend_learns_breakpoint_from_train_years() -> None:
    train_rows = [
        {"quarter": f"{year}-Q4", "annual_aids_deaths": float(10 + 2 * (year - 2018) + max(0, year - 2020) * 5)}
        for year in range(2018, 2024)
    ]

    model = _r78_fit_piecewise_log_linear_metric(train_rows, "annual_aids_deaths")
    prediction = _r78_predict_piecewise_log_linear_metric(model, "2024-Q4")

    assert model["status"] == "completed"
    assert 2018 < model["breakpoint_year"] < 2023
    assert prediction is not None
    assert prediction > 0


def test_r78_gate_requires_expanded_selector_to_beat_r76_reference() -> None:
    family_rows = [
        {
            "candidate_family": "public_train_selected_annual_proxy_v2",
            "candidate_mean_norm_error": 0.18,
            "candidate_interval_coverage": 0.9,
        }
    ]
    r76_reference = {"candidate_mean_norm_error": 0.20, "candidate_interval_coverage": 0.8}
    score_rows = [
        {"observation_role": "validation_only", "allowed_use": "validation_only", "candidate_norm_error": 0.18}
    ]

    gate = _r78_gate(family_rows, score_rows, target_rows=[{"metric_name": "annual_new_infections"}], r76_reference=r76_reference)

    assert gate["status"] == "expanded_public_annual_comparator_promoted"


def test_r79_expanded_public_gate_renames_public_proxy_status() -> None:
    family_rows = [
        {
            "model_family": "model",
            "public_proxy_family": "public_train_selected_annual_proxy_v2",
            "model_mean_norm_error": 0.3,
            "public_proxy_mean_norm_error": 0.2,
            "model_interval_coverage": 0.7,
            "public_proxy_interval_coverage": 0.9,
        }
    ]
    matched_rows = [{"observation_role": "validation_only", "allowed_use": "validation_only"}]

    gate = _r79_expanded_gate(matched_rows, family_rows)

    assert gate["status"] == "annual_model_blocked_by_expanded_public_proxy"


def test_r80_horizon_bucket_uses_declared_backtest_horizons() -> None:
    assert _r80_horizon_bucket(1) == 1
    assert _r80_horizon_bucket(2) == 3
    assert _r80_horizon_bucket(6) == 5


def test_r80_residual_quantiles_use_selected_proxy_metric_errors() -> None:
    rows = [
        {
            "candidate_family": "public_train_selected_annual_proxy_v2",
            "metric_name": "annual_new_infections",
            "horizon_years": 1,
            "candidate_norm_error": 0.1,
        },
        {
            "candidate_family": "public_train_selected_annual_proxy_v2",
            "metric_name": "annual_new_infections",
            "horizon_years": 1,
            "candidate_norm_error": 0.3,
        },
    ]

    quantiles = _r80_residual_quantiles(rows, metric_name="annual_new_infections", horizon_bucket=1)

    assert quantiles["q50_norm_abs_error"] == pytest.approx(0.2)
    assert quantiles["q95_norm_abs_error"] > quantiles["q50_norm_abs_error"]


def test_r80_gate_requires_promoted_r78_before_projection_claim() -> None:
    projection_rows = [{"metric_name": "annual_new_infections"}]
    r78 = {"expanded_public_annual_gate": {"status": "expanded_public_annual_comparator_promoted"}}
    selected = {
        "annual_new_infections": "public_piecewise_log_linear_trend",
        "annual_aids_deaths": "public_log_linear_trend",
        "estimated_plhiv": "public_joint_local_level_mass_balance",
    }

    gate = _r80_gate(projection_rows, r78, selected)

    assert gate["status"] == "public_annual_projection_head_ready"


def test_r81_driver_kind_keeps_time_blocked_edges_directional_only() -> None:
    row = {
        "edge_kind": "direct",
        "driver_status": "source_stable_but_time_blocked_sensitivity_only",
    }

    assert _r81_driver_kind(row) == "directional_sensitivity_knob"


def test_r81_knob_rows_do_not_allow_quantitative_modulation_for_time_blocked_edges() -> None:
    r46 = {
        "driver_rows": [
            {
                "edge_key": "direct:mobility_exposure_pressure->structural_barrier_pressure:lag1",
                "edge_kind": "direct",
                "source": "mobility_exposure_pressure",
                "target": "structural_barrier_pressure",
                "lag": 1,
                "baseline_weight": 0.1,
                "driver_status": "source_stable_but_time_blocked_sensitivity_only",
                "phase3_modules": ["incidence"],
                "source_reestimated_passed": True,
                "support_ablation_passed": True,
                "blocked_time_passed": False,
            }
        ]
    }

    rows = _r81_knob_rows(r46)

    assert rows[0]["directional_scenario_allowed"] is True
    assert rows[0]["quantitative_modulation_allowed"] is False
    assert rows[0]["intervention_claim_allowed"] is False
    assert rows[0]["primary_outputs"] == ["annual_new_infections", "incident_infections_period"]


def test_r81_gate_reports_directional_sensitivity_when_no_strict_prior_exists() -> None:
    knob_rows = [
        {
            "knob_kind": "directional_sensitivity_knob",
            "quantitative_modulation_allowed": False,
        }
    ]
    r46 = {"lineage_gate": {"status": "sensitivity_only_determinant_scenarios"}}
    r80 = {"public_annual_projection_gate": {"status": "public_annual_projection_head_ready"}}

    gate = _r81_gate(knob_rows, r46, r80)

    assert gate["status"] == "phase2_directional_sensitivity_knobs_only"
    assert "no_strict_phase2_quantitative_priors" in gate["blockers"]


def test_r82_bridge_rows_require_explicit_incidence_mortality_and_plhiv_outputs() -> None:
    rows = _r82_bridge_rows(
        quarterly_outputs=("incident_infections_period", "estimated_plhiv"),
        annual_metrics=("annual_new_infections", "annual_aids_deaths", "estimated_plhiv"),
    )
    by_metric = {row["annual_metric"]: row for row in rows}

    assert by_metric["annual_new_infections"]["bridge_status"] == "bridge_available"
    assert by_metric["annual_aids_deaths"]["bridge_status"] == "blocked_missing_quarterly_outputs"
    assert by_metric["estimated_plhiv"]["bridge_status"] == "bridge_available"


def test_r82_gate_blocks_when_any_required_annual_metric_lacks_quarterly_bridge() -> None:
    bridge_rows = [
        {"annual_metric": "annual_new_infections", "bridge_status": "bridge_available"},
        {"annual_metric": "annual_aids_deaths", "bridge_status": "blocked_missing_quarterly_outputs"},
    ]
    r80 = {"public_annual_projection_gate": {"status": "public_annual_projection_head_ready"}}

    gate = _r82_gate(bridge_rows, r80)

    assert gate["status"] == "quarterly_annual_bridge_blocked"
    assert "annual_aids_deaths_blocked_missing_quarterly_outputs" in gate["blockers"]


def test_r83_annualizes_only_complete_four_quarter_incidence_emissions() -> None:
    rows = [
        {"quarter": "2024-Q1", "incident_infections_period": 10.0},
        {"quarter": "2024-Q2", "incident_infections_period": 20.0},
        {"quarter": "2024-Q3", "incident_infections_period": 30.0},
        {"quarter": "2024-Q4", "incident_infections_period": 40.0},
    ]

    annualized = _r83_annualized_candidate_value(rows, year=2024, annual_metric="annual_new_infections")

    assert annualized["coverage_status"] == "annualized_from_quarterly_emission"
    assert annualized["candidate_value"] == 100.0


def test_r83_blocks_incomplete_or_proxy_annual_emissions() -> None:
    incomplete_incidence = _r83_annualized_candidate_value(
        [{"quarter": "2024-Q1", "incident_infections_period": 10.0}],
        year=2024,
        annual_metric="annual_new_infections",
    )
    attrition_is_not_deaths = _r83_annualized_candidate_value(
        [
            {"quarter": "2024-Q1", "net_attrition_outflow_period": 1.0},
            {"quarter": "2024-Q2", "net_attrition_outflow_period": 1.0},
            {"quarter": "2024-Q3", "net_attrition_outflow_period": 1.0},
            {"quarter": "2024-Q4", "net_attrition_outflow_period": 1.0},
        ],
        year=2024,
        annual_metric="annual_aids_deaths",
    )

    assert incomplete_incidence["coverage_status"] == "blocked_incomplete_four_quarter_emission"
    assert attrition_is_not_deaths["coverage_status"] == "blocked_missing_quarterly_outputs"
    assert "aids_deaths_period" in attrition_is_not_deaths["missing_required_outputs"]


def test_r83_gate_keeps_partial_bridge_diagnostic_not_full_claim() -> None:
    score_rows = [
        {
            "metric_name": "annual_new_infections",
            "candidate_norm_error": 0.2,
            "carry_forward_norm_error": 0.5,
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        },
        {
            "metric_name": "annual_aids_deaths",
            "candidate_norm_error": None,
            "carry_forward_norm_error": 0.5,
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        },
    ]
    family_rows = [
        {
            "candidate_family": "demo",
            "candidate_mean_norm_error": 0.2,
            "carry_forward_mean_norm_error": 0.5,
        }
    ]

    gate = _r83_gate(score_rows=score_rows, coverage_rows=[], target_rows=[{"year": 2024}], family_rows=family_rows)

    assert gate["status"] == "quarterly_emission_bridge_partial_diagnostic"
    assert "annual_aids_deaths_not_emitted_by_quarterly_candidate" in gate["blockers"]
    assert "estimated_plhiv_not_emitted_by_quarterly_candidate" in gate["blockers"]


def test_r84_dynamic_simulator_emits_conserved_annual_ledger_metrics() -> None:
    initial = {"U": 5.0, "D": 0.0, "A": 0.0, "T": 0.0, "V": 0.0, "L": 0.0, "R": 0.0}

    result = _phase3_simulate_sequence(
        initial,
        [{"quarter": "2024-Q1"}],
        {"2024-Q1": {}},
        incidence_inflow_map={"2024-Q1": 10.0},
        exit_channel_state_outflow_map={"2024-Q1": {"mortality_removal": {"U": 2.0}}},
    )

    prediction = result["prediction_rows"][0]
    assert prediction["incident_infections_period"] == 10.0
    assert prediction["aids_deaths_period"] == 2.0
    assert prediction["estimated_plhiv"] == 13.0


def test_r84_observation_model_preserves_conserved_ledger_metrics() -> None:
    rows = [
        {
            "quarter": "2024-Q1",
            "diagnosed_plhiv": 10.0,
            "alive_on_art": 9.0,
            "new_diagnosed_cases_period": 1.0,
            "tested_for_viral_load": 8.0,
            "virally_suppressed": 7.0,
            "estimated_plhiv": 20.0,
            "incident_infections_period": 3.0,
            "aids_deaths_period": 2.0,
            "net_attrition_outflow_period": 4.0,
        }
    ]

    calibrated = _phase3_apply_observation_model(rows, {"primary_coefficients": {}, "share_forecasts": {}}, positions={"2024-Q1": 1})

    assert calibrated[0]["estimated_plhiv"] == 20.0
    assert calibrated[0]["incident_infections_period"] == 3.0
    assert calibrated[0]["aids_deaths_period"] == 2.0
    assert calibrated[0]["net_attrition_outflow_period"] == 4.0


def test_r84_gate_blocks_incomplete_target_coverage() -> None:
    score_rows = [
        {
            "metric_name": "annual_new_infections",
            "candidate_norm_error": 0.1,
            "carry_forward_norm_error": 0.2,
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        },
        {
            "metric_name": "annual_new_infections",
            "candidate_norm_error": None,
            "carry_forward_norm_error": 0.2,
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        },
    ]
    family_rows = [
        {
            "candidate_mean_norm_error": 0.1,
            "carry_forward_mean_norm_error": 0.2,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 1.0,
        }
    ]

    gate = _r84_gate(score_rows=score_rows, family_rows=family_rows, target_rows=[{"year": 2024}])

    assert gate["status"] == "conserved_quarterly_annual_ledger_diagnostic_only"
    assert "annual_new_infections_incomplete_target_coverage" in gate["blockers"]


def test_r85_quarter_grid_expands_each_holdout_year() -> None:
    assert _r85_quarter_grid([2024, 2025]) == [
        "2024-Q1",
        "2024-Q2",
        "2024-Q3",
        "2024-Q4",
        "2025-Q1",
        "2025-Q2",
        "2025-Q3",
        "2025-Q4",
    ]


def test_r85_forecast_grid_strips_targets_and_uses_train_population_context() -> None:
    rows = [
        {
            "quarter": "2023-Q4",
            "population_total": 1000.0,
            "metric_provenance": {"population_total": {"source_id": "train_population"}},
        },
        {
            "quarter": "2024-Q4",
            "diagnosed_plhiv": 10.0,
            "alive_on_art": 9.0,
            "annual_new_infections": 11.0,
            "annual_aids_deaths": 1.0,
            "estimated_plhiv": 20.0,
            "population_total": 1200.0,
            "metric_provenance": {
                "annual_new_infections": {"allowed_use": "validation_only"},
                "population_total": {"source_id": "future_population"},
            },
        },
    ]

    grid = _r85_forecast_grid_rows(rows, holdout_years=[2024], train_end_year=2023)
    q4 = [row for row in grid if row["quarter"] == "2024-Q4"][0]

    assert len(grid) == 4
    assert "annual_new_infections" not in q4
    assert "diagnosed_plhiv" not in q4
    assert q4["population_total"] == 1000.0
    assert q4["metric_provenance"]["population_total"]["source_anchor_quarter"] == "2023-Q4"


def test_r85_gate_promotes_complete_better_forecast_grid() -> None:
    score_rows = []
    for metric_name in ("annual_new_infections", "annual_aids_deaths", "estimated_plhiv"):
        score_rows.append(
            {
                "metric_name": metric_name,
                "candidate_norm_error": 0.1,
                "carry_forward_norm_error": 0.2,
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
            }
        )
    family_rows = [
        {
            "candidate_mean_norm_error": 0.1,
            "carry_forward_mean_norm_error": 0.2,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 1.0,
        }
    ]

    gate = _r85_gate(score_rows=score_rows, family_rows=family_rows, target_rows=[{"year": 2024}])

    assert gate["status"] == "annual_ledger_forecast_grid_pass"
    assert gate["blockers"] == []


def test_r86_distributes_annual_total_by_dynamic_quarter_shape() -> None:
    rows = [
        {"quarter": "2024-Q1", "incident_infections_period": 1.0},
        {"quarter": "2024-Q2", "incident_infections_period": 1.0},
        {"quarter": "2024-Q3", "incident_infections_period": 2.0},
        {"quarter": "2024-Q4", "incident_infections_period": 6.0},
    ]

    distribution = _r86_distribute_annual_total_by_quarter_shape(
        rows,
        annual_total=100.0,
        quarterly_metric="incident_infections_period",
    )

    assert distribution["2024-Q1"] == 10.0
    assert distribution["2024-Q2"] == 10.0
    assert distribution["2024-Q3"] == 20.0
    assert distribution["2024-Q4"] == 60.0
    assert sum(value for key, value in distribution.items() if key.startswith("2024-")) == 100.0
    assert distribution["_shape_status"] == "base_dynamic_quarter_shape"


def test_r86_distributes_annual_total_uniformly_when_shape_is_zero() -> None:
    rows = [
        {"quarter": "2024-Q1", "aids_deaths_period": 0.0},
        {"quarter": "2024-Q2", "aids_deaths_period": 0.0},
        {"quarter": "2024-Q3", "aids_deaths_period": 0.0},
        {"quarter": "2024-Q4", "aids_deaths_period": 0.0},
    ]

    distribution = _r86_distribute_annual_total_by_quarter_shape(
        rows,
        annual_total=8.0,
        quarterly_metric="aids_deaths_period",
    )

    assert [distribution[f"2024-Q{quarter}"] for quarter in range(1, 5)] == [2.0, 2.0, 2.0, 2.0]
    assert distribution["_shape_status"] == "uniform_no_positive_quarter_shape"


def test_r86_gate_promotes_complete_better_annual_calibrated_ledger() -> None:
    score_rows = []
    for metric_name in ("annual_new_infections", "annual_aids_deaths", "estimated_plhiv"):
        score_rows.append(
            {
                "metric_name": metric_name,
                "candidate_norm_error": 0.1,
                "carry_forward_norm_error": 0.2,
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
            }
        )
    family_rows = [
        {
            "candidate_mean_norm_error": 0.1,
            "carry_forward_mean_norm_error": 0.2,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 1.0,
        }
    ]

    gate = _r86_gate(score_rows=score_rows, family_rows=family_rows, target_rows=[{"year": 2024}])

    assert gate["status"] == "annual_calibrated_forecast_grid_ledger_pass"
    assert gate["blockers"] == []
