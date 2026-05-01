from __future__ import annotations

from phase3_dynamic.lifted_residual_anatomy import (
    _attach_r10_metric_proxy,
    _metric_entry_rows,
    _summarize_group,
)


def test_metric_entry_rows_decomposes_candidate_carry_errors_by_metric() -> None:
    evaluated = [
        {
            "family": "demo",
            "train_end_year": 2024,
            "holdout_years": [2025],
            "gate_horizon_years": 1,
            "metric_scales": {"diagnosed_plhiv": 100.0},
            "holdout_rows": [{"quarter": "2025-Q1", "diagnosed_plhiv": 120.0}],
            "prediction_rows": [{"quarter": "2025-Q1", "diagnosed_plhiv": 110.0}],
            "carry_forward_prediction_rows": [{"quarter": "2025-Q1", "diagnosed_plhiv": 90.0}],
        }
    ]
    entries = _metric_entry_rows(evaluated, {"year_labels": {}})
    assert len(entries) == 1
    assert entries[0]["metric_name"] == "diagnosed_plhiv"
    assert entries[0]["stream"] == "diagnosis_stock"
    assert entries[0]["candidate_norm_error"] == 0.1
    assert entries[0]["carry_forward_norm_error"] == 0.3
    assert entries[0]["candidate_minus_carry_forward_norm_error"] == -0.2


def test_metric_rows_carry_explicit_r10_proxy_limitation() -> None:
    entries = [
        {
            "metric_name": "diagnosed_plhiv",
            "stream": "diagnosis_stock",
            "observed_value": 200.0,
            "candidate_norm_error": 0.20,
            "carry_forward_norm_error": 0.30,
            "candidate_minus_carry_forward_norm_error": -0.10,
            "candidate_abs_error": 40.0,
            "carry_forward_abs_error": 60.0,
        }
    ]
    metric_rows = _summarize_group(entries, group_keys=("metric_name", "stream"))
    attached = _attach_r10_metric_proxy(
        metric_rows,
        entries,
        {"diagnosed_plhiv": {"raw_mean_abs_error": 20.0, "raw_p90_abs_error": 25.0, "count": 5.0}},
    )
    assert attached[0]["r10_norm_mae_proxy"] == 0.1
    assert attached[0]["candidate_minus_r10_norm_mae_proxy"] == 0.1
