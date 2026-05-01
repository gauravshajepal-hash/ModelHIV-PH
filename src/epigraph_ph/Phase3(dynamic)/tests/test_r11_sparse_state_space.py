from __future__ import annotations

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
    _fit_trajectory_shape_head,
    _build_r12_04_source_lineage_ablation_report,
    _build_r12_05_lineage_stratified_contract_report,
    _build_r12_06_doh_quarterly_support_adjudication_report,
    _build_r12_07_horizon_specific_evidence_router_report,
    _build_r12_08_route_aware_two_head_candidate_report,
    _build_r12_official_annual_challenge_gate_report,
    _fit_r12_annual_anchor_head_selector,
    _predict_back_half_rate,
    _r11_multi_horizon_report,
    _r10_lifted_gate,
    _select_horizon_matched_r10_reference,
    _fit_support_partition_calibration_model,
    _predict_support_partition_calibration,
    project_cascade_stock_row,
    stock_consistency_gate,
)
from phase3_dynamic.r13_priority_experiments import _r13_priority_experiment_specs


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
    assert any(spec["family"] == "r16_support_cadence_stock_process" for spec in specs)
    assert any(spec.get("phase2_status") == "locked_until_source_stable" for spec in specs)


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
