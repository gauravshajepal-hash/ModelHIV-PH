from __future__ import annotations

from pathlib import Path

from phase3_dynamic.r53_publication_claim_registry import _process_repair_experiment_queue_claim
from phase3_dynamic.r92_process_repair_experiment_queue import (
    _family_predictions,
    _gate,
    _log_ar_forecast,
)


def _row(quarter: str, proxy_metric: str, proxy_value: float, *, target_metric: str, target_value: float | None = None) -> dict:
    row = {
        "quarter": quarter,
        proxy_metric: proxy_value,
        "metric_provenance": {
            proxy_metric: {
                "source_tier": "official_doh_archive",
                "measurement_class": "program_observed_harp",
                "series_kind": "quarterly_snapshot",
                "observation_role": "direct_target",
                "allowed_use": "direct_target",
            }
        },
    }
    if target_value is not None:
        row[target_metric] = target_value
        row["metric_provenance"][target_metric] = {
            "source_tier": "external_official_model_estimate",
            "measurement_class": "model_estimate",
            "series_kind": "annual",
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        }
    return row


def test_r92_log_ar_forecast_is_data_fit_and_nonnegative() -> None:
    forecast = _log_ar_forecast({2020: 10.0, 2021: 20.0, 2022: 30.0}, train_end_year=2022, forecast_years=[2023, 2024])

    assert set(forecast) == {2023, 2024}
    assert all(value >= 0.0 for value in forecast.values())


def test_r92_proxy_family_prediction_uses_proxy_history() -> None:
    rows = [
        _row("2020-Q4", "new_diagnosed_cases_period", 100.0, target_metric="annual_new_infections", target_value=200.0),
        _row("2021-Q4", "new_diagnosed_cases_period", 150.0, target_metric="annual_new_infections", target_value=300.0),
        _row("2022-Q4", "new_diagnosed_cases_period", 225.0, target_metric="annual_new_infections"),
    ]

    predictions = _family_predictions(
        rows,
        target_metric="annual_new_infections",
        proxy_metric="new_diagnosed_cases_period",
        family="proxy_ar_ratio_median",
        train_end_year=2021,
        forecast_years=[2022],
    )

    assert predictions[2022] > 0.0


def test_r92_gate_detects_signal_but_blocks_unsupported_mechanism() -> None:
    support_rows = [
        {"metric_name": "incident_infections_period", "count": 0},
        {"metric_name": "new_diagnosed_cases_period", "count": 12},
        {"metric_name": "deaths_reported_period", "count": 10},
    ]
    mechanism_family_rows = [
        {
            "candidate_family": "incidence_proxy_diagnosis_flow_bridge:r92_process_repair:mechanism",
            "candidate_mean_norm_error": 0.05,
            "carry_forward_mean_norm_error": 0.3,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 0.5,
        },
        {
            "candidate_family": "mortality_reported_death_bridge:r92_process_repair:mechanism",
            "candidate_mean_norm_error": 0.2,
            "carry_forward_mean_norm_error": 0.5,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 0.6,
        },
    ]
    readout_family_rows = [
        {
            "candidate_family": "incidence_proxy_diagnosis_flow_bridge:r92_process_repair:readout",
            "candidate_mean_norm_error": 0.05,
            "carry_forward_mean_norm_error": 0.3,
            "candidate_interval_coverage": 1.0,
            "carry_forward_interval_coverage": 0.5,
        }
    ]
    ablation_rows = [
        {
            "bridge_id": "mortality_reported_death_bridge",
            "ablation_status": "evaluable",
            "ablated_candidate_mean_norm_error": 0.6,
            "ablated_carry_forward_mean_norm_error": 0.5,
        }
    ]

    gate = _gate(
        support_rows=support_rows,
        mechanism_family_rows=mechanism_family_rows,
        readout_family_rows=readout_family_rows,
        ablation_rows=ablation_rows,
    )

    assert gate["status"] == "mortality_process_signal_detected_mechanism_claim_blocked"
    assert "direct_incidence_process_support_absent" in gate["blockers"]
    assert "mortality_process_repair_not_source_family_stable" in gate["blockers"]


def test_r92_registry_claim_keeps_signal_diagnostic(tmp_path: Path) -> None:
    artifact = tmp_path / "r92.json"
    artifact.write_text("{}", encoding="utf-8")
    claim = _process_repair_experiment_queue_claim(
        {
            "status": "mortality_process_signal_detected_mechanism_claim_blocked",
            "candidate_family": "process_repair_train_selected_observation_bridge",
            "process_repair_gate": {
                "status": "mortality_process_signal_detected_mechanism_claim_blocked",
                "blockers": ["direct_incidence_process_support_absent"],
                "direct_incidence_process_support_count": 0,
            },
        },
        artifact,
    )

    assert claim["claim_status"] == "mortality_signal_diagnostic_only"
    assert "direct_incidence_process_support_absent" in claim["blockers"]
