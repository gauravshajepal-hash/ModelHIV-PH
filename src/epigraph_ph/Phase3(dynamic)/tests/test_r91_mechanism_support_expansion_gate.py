from __future__ import annotations

from phase3_dynamic.r91_mechanism_support_expansion_gate import (
    _bridge_pairs,
    _fit_bridge_policy,
    _gate,
    _proxy_annual_series,
)


def _row(quarter: str, metric: str, value: float, *, annual_target: float | None = None) -> dict:
    row = {
        "quarter": quarter,
        metric: value,
        "metric_provenance": {
            metric: {
                "source_id": "unit_test_source",
                "source_tier": "official_doh_archive",
                "measurement_class": "program_observed_harp",
                "series_kind": "quarterly_snapshot",
                "observation_role": "direct_target",
                "allowed_use": "direct_target",
            }
        },
    }
    if annual_target is not None:
        row["annual_aids_deaths"] = annual_target
        row["metric_provenance"]["annual_aids_deaths"] = {
            "source_id": "unit_test_annual",
            "source_tier": "external_official_model_estimate",
            "measurement_class": "model_estimate",
            "series_kind": "annual",
            "observation_role": "validation_only",
            "allowed_use": "validation_only",
        }
    return row


def test_r91_proxy_annual_series_annualizes_partial_year() -> None:
    rows = [
        _row("2020-Q1", "deaths_reported_period", 10.0),
        _row("2020-Q3", "deaths_reported_period", 30.0),
    ]

    series = _proxy_annual_series(rows, "deaths_reported_period")

    assert series[2020]["observed_quarter_count"] == 2
    assert series[2020]["annualized_proxy_value"] == 80.0
    assert series[2020]["coverage_status"] == "partial_year_annualized"


def test_r91_fit_bridge_policy_uses_train_ratio_and_selected_proxy() -> None:
    rows = [
        _row("2020-Q4", "deaths_reported_period", 100.0, annual_target=200.0),
        _row("2021-Q4", "deaths_reported_period", 150.0, annual_target=300.0),
    ]
    proxy_by_year = _proxy_annual_series(rows, "deaths_reported_period")
    pairs = _bridge_pairs(rows, target_metric="annual_aids_deaths", proxy_by_year=proxy_by_year, train_end_year=2021)

    model = _fit_bridge_policy(
        pairs=pairs,
        proxy_by_year=proxy_by_year,
        train_end_year=2021,
        policy="median_ratio_last_any_proxy",
    )

    assert model["status"] == "completed"
    assert model["ratio"] == 0.5
    assert model["candidate_value"] == 300.0


def test_r91_gate_blocks_proxy_only_incidence_and_unstable_death_bridge() -> None:
    support_rows = [
        {"metric_name": "incident_infections_period", "count": 0},
        {"metric_name": "new_diagnosed_cases_period", "count": 10},
        {"metric_name": "deaths_reported_period", "count": 10},
    ]
    family_rows = [
        {
            "candidate_family": "incidence_proxy_diagnosis_flow_bridge:train_selected_proxy_bridge",
            "candidate_mean_norm_error": 0.5,
            "carry_forward_mean_norm_error": 0.4,
            "candidate_interval_coverage": 0.2,
            "carry_forward_interval_coverage": 0.5,
        },
        {
            "candidate_family": "mortality_reported_death_bridge:train_selected_proxy_bridge",
            "candidate_mean_norm_error": 0.4,
            "carry_forward_mean_norm_error": 0.5,
            "candidate_interval_coverage": 0.6,
            "carry_forward_interval_coverage": 0.6,
        },
    ]
    ablation_rows = [
        {
            "bridge_id": "mortality_reported_death_bridge",
            "ablation_status": "evaluable",
            "ablated_candidate_mean_norm_error": 0.6,
            "ablated_carry_forward_mean_norm_error": 0.5,
        }
    ]

    gate = _gate(support_rows=support_rows, family_rows=family_rows, ablation_rows=ablation_rows)

    assert gate["status"] == "proxy_bridge_signal_detected_but_mechanism_claim_blocked"
    assert "direct_incidence_process_support_absent" in gate["blockers"]
    assert "mortality_bridge_not_source_family_stable" in gate["blockers"]
