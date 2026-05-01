from __future__ import annotations

import pytest

from phase3_dynamic.backlog_residual_anatomy import _attach_r10_proxy, _entry_rows, _summarize_group


def test_entry_rows_keeps_annual_incidence_as_validation_only_stream() -> None:
    rows = _entry_rows(
        [
            {
                "repair_family": "monthly_backlog_late_emission",
                "base_family": "base",
                "delegate_family": "delegate",
                "train_end_year": 2023,
                "horizon_years": 1,
                "metric_entries": [],
                "incidence_validation_entries": [
                    {
                        "year": 2024,
                        "target_annual_new_infections": 100.0,
                        "candidate_annual_incidence": 110.0,
                        "base_annual_incidence": 105.0,
                        "carry_forward_annual_incidence": 120.0,
                        "candidate_norm_error": 0.10,
                        "base_norm_error": 0.05,
                        "carry_forward_norm_error": 0.20,
                        "candidate_minus_base_norm_error": 0.05,
                        "candidate_minus_carry_forward_norm_error": -0.10,
                    }
                ],
            }
        ]
    )
    assert rows[0]["metric_name"] == "annual_new_infections"
    assert rows[0]["stream"] == "incidence_validation"
    assert rows[0]["comparison_scope"] == "validation_only_annual_incidence"


def test_attach_r10_proxy_marks_aggregate_metric_and_annual_scopes() -> None:
    entries = [
        {
            "metric_name": "diagnosed_plhiv",
            "observed_value": 200.0,
            "candidate_norm_error": 0.2,
            "base_norm_error": 0.3,
            "carry_forward_norm_error": 0.4,
            "candidate_minus_base_norm_error": -0.1,
            "candidate_minus_carry_forward_norm_error": -0.2,
        },
        {
            "metric_name": "annual_new_infections",
            "observed_value": 100.0,
            "candidate_norm_error": 0.5,
            "base_norm_error": 0.4,
            "carry_forward_norm_error": 0.7,
            "candidate_minus_base_norm_error": 0.1,
            "candidate_minus_carry_forward_norm_error": -0.2,
        },
    ]
    rows = _summarize_group(entries, group_keys=("metric_name",))
    attached = _attach_r10_proxy(
        rows,
        entries=entries,
        r10_metric_reference={"diagnosed_plhiv": {"raw_mean_abs_error": 20.0, "raw_p90_abs_error": 30.0}},
        r10_annual_incidence_error=0.25,
    )
    by_metric = {row["metric_name"]: row for row in attached}
    assert by_metric["diagnosed_plhiv"]["r10_norm_mae_proxy"] == pytest.approx(0.1)
    assert by_metric["diagnosed_plhiv"]["r10_proxy_scope"].startswith("r10_aggregate_metric")
    assert by_metric["annual_new_infections"]["r10_norm_mae_proxy"] == 0.25
    assert by_metric["annual_new_infections"]["r10_proxy_scope"] == "r10_annual_incidence_validation_error"
