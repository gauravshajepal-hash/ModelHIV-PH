from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from phase3_dynamic.diagnosis_incidence_repair import (
    DiagnosisFlowRepairHead,
    IncidenceReadoutRepairHead,
    MonthlyReportingNowcastHead,
    _annual_incidence_entries,
    _apply_monthly_reporting_nowcast_head,
    _convolve_monthly_diagnoses,
    _feature_names,
    _fit_backlog_late_emission_model,
    _fit_delay_kernel_from_prior,
    _fit_monthly_reporting_nowcast_head,
    _late_delay_prior_by_lag,
    _month_index,
    _r10_annual_incidence_error,
    _simulate_backlog_late_emission,
    apply_diagnosis_flow_repair_head,
    apply_incidence_readout_repair_head,
    apply_selective_repair_gate,
)
from phase3_dynamic.runtime import write_json


def test_feature_names_rejects_unsupported_repair_family() -> None:
    with pytest.raises(ValueError, match="Unsupported diagnosis-flow repair family"):
        _feature_names("raw_cartesian_factor_sweep")


def test_apply_repair_head_changes_only_diagnosis_flow() -> None:
    head = DiagnosisFlowRepairHead(
        family="diag_flow_incidence_only",
        feature_names=["intercept", "log_incidence_inflow"],
        coefficients=[0.0, 1.0],
        train_row_count=4,
    )
    base_rows = [
        {
            "quarter": "2025-Q1",
            "diagnosed_plhiv": 100.0,
            "new_diagnosed_cases_period": 5.0,
            "alive_on_art": 80.0,
        }
    ]
    trajectory_rows = [
        {
            "quarter": "2025-Q1",
            "stock_balance": {"incidence_inflow": 12.0},
            "state_values": {"U": 50.0},
        }
    ]
    repaired = apply_diagnosis_flow_repair_head(
        head=head,
        base_rows=base_rows,
        backbone_rows=base_rows,
        trajectory_rows=trajectory_rows,
    )
    assert repaired[0]["new_diagnosed_cases_period"] == pytest.approx(12.0)
    assert repaired[0]["diagnosed_plhiv"] == 100.0
    assert repaired[0]["alive_on_art"] == 80.0


def test_apply_incidence_readout_changes_only_incidence_readout() -> None:
    head = IncidenceReadoutRepairHead(
        family="incidence_readout_from_diag_flow",
        feature_names=["intercept", "log_diagnosis_flow", "horizon_fraction"],
        coefficients=[0.0, 1.0, 0.0],
        train_row_count=4,
    )
    base_rows = [
        {
            "quarter": "2025-Q1",
            "diagnosed_plhiv": 100.0,
            "new_diagnosed_cases_period": 12.0,
            "incident_infections_period": 20.0,
        }
    ]
    repaired = apply_incidence_readout_repair_head(head=head, base_rows=base_rows)
    assert repaired[0]["incident_infections_period"] == pytest.approx(12.0)
    assert repaired[0]["new_diagnosed_cases_period"] == 12.0
    assert repaired[0]["diagnosed_plhiv"] == 100.0


def test_delay_kernel_fit_recovers_nonnegative_convolution() -> None:
    incidence = {0: 10.0, 1: 0.0, 2: 20.0, 3: 0.0}
    diagnosis = {
        1: 10.0,
        3: 20.0,
    }
    fit = _fit_delay_kernel_from_prior(
        diagnosis_by_month=diagnosis,
        incidence_prior_by_month=incidence,
    )
    assert fit is not None
    kernel = dict(zip(fit["lags"], fit["kernel"]))
    assert sum(fit["kernel"]) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in fit["kernel"])
    assert kernel[1] > 0.99


def test_late_delay_prior_moves_mass_to_longer_lags_when_late_severity_is_high() -> None:
    prior = _late_delay_prior_by_lag(
        lags=[0, 1, 2],
        diagnosis_by_month={2: 10.0, 3: 10.0},
        incidence_prior_by_month={0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0},
        late_severity_by_month={2: 1.0, 3: 1.0},
    )
    assert prior is not None
    assert prior[2] > prior[1] > prior[0]


def test_late_constrained_delay_kernel_records_independent_constraint() -> None:
    fit = _fit_delay_kernel_from_prior(
        diagnosis_by_month={2: 10.0, 3: 10.0},
        incidence_prior_by_month={0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0},
        late_severity_by_month={2: 1.0, 3: 1.0},
    )
    assert fit is not None
    kernel = dict(zip(fit["lags"], fit["kernel"]))
    assert fit["late_constraint"]["status"] == "applied"
    assert fit["late_constraint"]["late_severity_month_count"] == 2
    assert kernel[2] > kernel[1] > kernel[0]


def test_delay_convolution_generates_monthly_diagnosis_from_infections() -> None:
    diagnosis = _convolve_monthly_diagnoses(
        incidence_by_month={0: 10.0, 1: 20.0},
        lags=[0, 1],
        kernel=[0.25, 0.75],
        ascertainment_scale=2.0,
        months=[0, 1, 2],
    )
    assert diagnosis[0] == pytest.approx(5.0)
    assert diagnosis[1] == pytest.approx(25.0)
    assert diagnosis[2] == pytest.approx(30.0)


def test_backlog_late_emission_simulation_conserves_hidden_population() -> None:
    simulated = _simulate_backlog_late_emission(
        incidence_by_month={0: 10.0},
        months=[0],
        early_to_late_hazard=0.25,
        early_diagnosis_hazard=0.1,
        late_diagnosis_hazard=0.4,
        initial_early_undiagnosed=2.0,
        initial_late_undiagnosed=3.0,
    )
    row = simulated[0]
    start_mass = 2.0 + 3.0 + 10.0
    end_mass = row["early_undiagnosed_end"] + row["late_undiagnosed_end"] + row["new_diagnosed_cases_period"]
    assert end_mass == pytest.approx(start_mass)
    assert row["late_diagnosis_share"] >= 0.0
    assert row["late_diagnosis_share"] <= 1.0


def test_backlog_late_emission_fit_uses_late_presenter_observations() -> None:
    diagnosis_by_month = {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0}
    incidence_prior_by_month = {0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0}
    fit = _fit_backlog_late_emission_model(
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=incidence_prior_by_month,
        emission_observations={
            "late_share_by_month": {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5},
            "advanced_count_by_month": {3: 2.0},
            "metric_counts": {"median_cd4_at_enrollment": 4, "advanced_hiv_cases_period": 1},
            "late_share_month_count": 4,
            "advanced_count_month_count": 1,
        },
    )
    assert fit is not None
    assert fit["late_share_emission_month_count"] == 4
    assert fit["advanced_count_emission_month_count"] == 1
    assert fit["train_loss"] >= 0.0
    assert fit["emission_summary"]["metric_counts"]["median_cd4_at_enrollment"] == 4


def test_annual_incidence_validation_requires_complete_year() -> None:
    validation_targets = {
        2024: {"quarter": "2024-Q4", "annual_new_infections": 100.0},
        2025: {"quarter": "2025-Q4", "annual_new_infections": 100.0},
    }
    complete_rows = [
        {"quarter": "2024-Q1", "incident_infections_period": 20.0},
        {"quarter": "2024-Q2", "incident_infections_period": 20.0},
        {"quarter": "2024-Q3", "incident_infections_period": 30.0},
        {"quarter": "2024-Q4", "incident_infections_period": 40.0},
        {"quarter": "2025-Q1", "incident_infections_period": 10.0},
        {"quarter": "2025-Q2", "incident_infections_period": 10.0},
        {"quarter": "2025-Q4", "incident_infections_period": 10.0},
    ]
    entries = _annual_incidence_entries(
        candidate_rows=complete_rows,
        base_rows=complete_rows,
        carry_rows=complete_rows,
        validation_targets=validation_targets,
        train_end_year=2023,
        horizon_years=1,
        repair_family="baseline_no_repair",
    )
    assert [entry["year"] for entry in entries] == [2024]
    assert entries[0]["candidate_annual_incidence"] == 110.0
    assert entries[0]["candidate_norm_error"] == pytest.approx(0.1)


def test_r10_annual_incidence_error_reads_full_report(tmp_path: Path) -> None:
    source_report = tmp_path / "r10_report.json"
    write_json(
        source_report,
        {
            "contracts": [
                {
                    "contract": "purged_dense",
                    "baseline_current_champion": {
                        "contract": "purged_dense",
                        "archive_variant": "baseline",
                        "experiment_id": "EXP-R10-DENSE-CHAMPION",
                        "annual_mean_incidence_error": 0.03,
                    },
                    "merged_current_champion": {
                        "contract": "purged_dense",
                        "archive_variant": "merged",
                        "experiment_id": "EXP-R10-DENSE-CHAMPION",
                        "annual_mean_incidence_error": 0.08,
                    },
                }
            ]
        },
    )
    assert _r10_annual_incidence_error(
        {"source_report": source_report.as_posix()},
        {
            "reference_contract": "purged_dense",
            "reference_experiment_id": "EXP-R10-DENSE-CHAMPION",
        },
    ) == 0.08


def _monthly_diag_row(month: str, value: float) -> dict[str, object]:
    return {
        "metric_name": "new_diagnosed_cases_period",
        "value": value,
        "series_kind": "monthly_snapshot",
        "_month_index": _month_index(month),
        "_contract": {"observation_role": "direct_target"},
    }


def test_monthly_reporting_nowcast_completes_partial_harp_months(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _monthly_diag_row("2022-01", 10.0),
        _monthly_diag_row("2022-02", 10.0),
        _monthly_diag_row("2022-03", 10.0),
        _monthly_diag_row("2022-04", 10.0),
        _monthly_diag_row("2023-01", 20.0),
    ]
    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair.load_monthly_signal_rows",
        lambda _context: rows,
    )
    dataset = SimpleNamespace(
        train_rows=[
            {"quarter": "2022-Q1", "new_diagnosed_cases_period": 60.0},
            {"quarter": "2022-Q2", "new_diagnosed_cases_period": 40.0},
        ],
        train_transition_rows=[
            {"quarter": "2022-Q1", "stock_balance": {"incidence_inflow": 120.0}},
            {"quarter": "2022-Q2", "stock_balance": {"incidence_inflow": 80.0}},
        ],
    )
    head = _fit_monthly_reporting_nowcast_head(
        family="monthly_reporting_nowcast",
        dataset=dataset,
        monthly_context=object(),
    )

    assert head is not None
    assert head.completion_by_month_mask["123"] == pytest.approx(2.0)
    assert head.completion_by_month_mask["1"] == pytest.approx(4.0)

    repaired, updated_head, diagnostics = _apply_monthly_reporting_nowcast_head(
        head=head,
        monthly_context=object(),
        base_rows=[{"quarter": "2023-Q1", "new_diagnosed_cases_period": 999.0, "incident_infections_period": 999.0}],
    )

    assert updated_head is not None
    assert repaired[0]["new_diagnosed_cases_period"] == pytest.approx(80.0)
    assert repaired[0]["incident_infections_period"] >= 0.0
    assert diagnostics[0]["month_mask"] == "1"
    assert diagnostics[0]["contract"].startswith("uses direct monthly HARP")


def test_monthly_reporting_nowcast_flow_only_preserves_incidence_readout(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _monthly_diag_row("2022-01", 10.0),
        _monthly_diag_row("2022-02", 10.0),
        _monthly_diag_row("2022-03", 10.0),
        _monthly_diag_row("2023-01", 20.0),
    ]
    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair.load_monthly_signal_rows",
        lambda _context: rows,
    )
    dataset = SimpleNamespace(
        train_rows=[{"quarter": "2022-Q1", "new_diagnosed_cases_period": 60.0}],
        train_transition_rows=[{"quarter": "2022-Q1", "stock_balance": {"incidence_inflow": 120.0}}],
    )
    head = _fit_monthly_reporting_nowcast_head(
        family="monthly_reporting_nowcast_flow_only",
        dataset=dataset,
        monthly_context=object(),
    )
    assert head is not None

    repaired, _updated_head, diagnostics = _apply_monthly_reporting_nowcast_head(
        head=head,
        monthly_context=object(),
        base_rows=[{"quarter": "2023-Q1", "new_diagnosed_cases_period": 999.0, "incident_infections_period": 777.0}],
        update_incidence=False,
    )

    assert repaired[0]["new_diagnosed_cases_period"] == pytest.approx(40.0)
    assert repaired[0]["incident_infections_period"] == pytest.approx(777.0)
    assert diagnostics[0]["incidence_updated"] is False


def test_selective_repair_gate_separates_nowcast_backlog_and_base(monkeypatch: pytest.MonkeyPatch) -> None:
    head = MonthlyReportingNowcastHead(
        family="selective_monthly_nowcast_backlog",
        completion_by_month_mask={},
        training_partial_by_month_mask={},
        lead_completion_by_month_mask={},
        training_lead_partial_by_month_mask={},
        fallback_completion_scale=1.0,
        fallback_lead_completion_scale=1.0,
        incidence_feature_names=["intercept"],
        incidence_coefficients=[0.0],
        incidence_prediction_floor=0.0,
        incidence_prediction_ceiling=10.0,
        train_quarter_count=1,
        direct_monthly_row_count=1,
        train_partial_quarter_count=1,
        holdout_nowcast_quarter_count=0,
    )
    base_rows = [
        {"quarter": "2023-Q1", "new_diagnosed_cases_period": 10.0},
        {"quarter": "2023-Q2", "new_diagnosed_cases_period": 20.0},
        {"quarter": "2023-Q3", "new_diagnosed_cases_period": 30.0},
    ]

    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair.apply_backlog_late_emission_branch",
        lambda **_kwargs: (
            [
                {"quarter": "2023-Q1", "new_diagnosed_cases_period": 100.0},
                {"quarter": "2023-Q2", "new_diagnosed_cases_period": 200.0},
                {"quarter": "2023-Q3", "new_diagnosed_cases_period": 300.0},
            ],
            object(),
        ),
    )
    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair._apply_monthly_reporting_nowcast_head",
        lambda **kwargs: (
            [
                {"quarter": "2023-Q1", "new_diagnosed_cases_period": 1000.0},
                {"quarter": "2023-Q2", "new_diagnosed_cases_period": 20.0},
                {"quarter": "2023-Q3", "new_diagnosed_cases_period": 30.0},
            ],
            kwargs["head"],
            [{"quarter": "2023-Q1", "nowcast_source": "same_quarter_partial_months"}],
        ),
    )
    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair._late_evidence_supported_quarters",
        lambda *_args, **_kwargs: {"2023-Q2": {"metric_names": ["median_cd4_at_enrollment"]}},
    )
    monkeypatch.setattr(
        "phase3_dynamic.diagnosis_incidence_repair._late_evidence_agreement_by_quarter",
        lambda **_kwargs: {
            "2023-Q2": {
                "agreement_supported": True,
                "candidate_late_evidence_loss": 0.1,
                "neutral_late_evidence_loss": 0.2,
            }
        },
    )

    selected, backlog_head, updated_head, nowcast_diagnostics, gate_diagnostics = apply_selective_repair_gate(
        repair_family="selective_monthly_nowcast_backlog",
        dataset=SimpleNamespace(),
        monthly_context=object(),
        base_rows=base_rows,
        monthly_nowcast_head=head,
    )

    assert backlog_head is not None
    assert updated_head is head
    assert nowcast_diagnostics[0]["quarter"] == "2023-Q1"
    assert [row["new_diagnosed_cases_period"] for row in selected] == [1000.0, 200.0, 30.0]
    assert [row["selective_repair_source"] for row in selected] == [
        "monthly_reporting_nowcast",
        "backlog_late_emission",
        "strict_base",
    ]
    assert gate_diagnostics[2]["selection_reason"] == "no_monthly_nowcast_or_late_diagnosis_support"
