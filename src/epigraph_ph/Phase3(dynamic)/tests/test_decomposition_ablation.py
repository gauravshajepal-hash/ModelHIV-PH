from __future__ import annotations

from phase3_dynamic.decomposition_ablation import _attach_full_deltas, _variant_specs
from phase3_dynamic.decomposition_research import _score_supported_back_half_gate, _score_supported_rate_gate


def test_decomposition_ablation_specs_include_component_and_stream_variants() -> None:
    variant_ids = {row["variant_id"] for row in _variant_specs()}

    assert {"trend_only", "support_shift_only", "shock_only"}.issubset(variant_ids)
    assert {"lean_shock_art", "lean_shock_art_back_half"}.issubset(variant_ids)
    assert {"without_incidence", "without_diagnosis", "without_art", "without_vl", "without_suppression"}.issubset(variant_ids)


def test_decomposition_ablation_delta_uses_full_variant_as_reference() -> None:
    rows = _attach_full_deltas(
        [
            {
                "variant_id": "full",
                "one_year_gate": {"candidate_mean_mae": 1.0},
                "five_year_gate": {"candidate_mean_mae": 2.0},
            },
            {
                "variant_id": "without_incidence",
                "variant_class": "stream_removal",
                "one_year_gate": {"candidate_mean_mae": 1.5},
                "five_year_gate": {"candidate_mean_mae": 1.0},
            },
        ]
    )

    ablation = rows[1]["ablation_summary"]
    assert ablation["one_year_delta_vs_full"] == 0.5
    assert ablation["five_year_delta_vs_full"] == -1.0
    assert ablation["interpretation"] == "removed control has horizon-dependent effect"


def test_decomposition_ablation_delta_treats_tiny_deltas_as_indistinguishable() -> None:
    rows = _attach_full_deltas(
        [
            {
                "variant_id": "full",
                "one_year_gate": {"candidate_mean_mae": 1.0},
                "five_year_gate": {"candidate_mean_mae": 2.0},
            },
            {
                "variant_id": "without_vl",
                "variant_class": "stream_removal",
                "one_year_gate": {"candidate_mean_mae": 1.0 + 1e-9},
                "five_year_gate": {"candidate_mean_mae": 2.0 - 1e-9},
            },
        ]
    )

    assert rows[1]["ablation_summary"]["interpretation"] == "removed control is numerically indistinguishable from full"


def test_support_aware_back_half_gate_reports_claim_status() -> None:
    gate = _score_supported_back_half_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "support_aware_back_half": {
                    "candidate": {
                        "status": "scored",
                        "mean_normalized_mae": 0.2,
                        "supported_target_count": 2,
                        "metric_rows": [],
                    },
                    "baseline": {
                        "status": "scored",
                        "mean_normalized_mae": 0.3,
                        "supported_target_count": 2,
                        "metric_rows": [],
                    },
                    "carry_forward": {
                        "status": "not_evaluable",
                        "mean_normalized_mae": float("inf"),
                        "supported_target_count": 2,
                        "metric_rows": [],
                    },
                },
            }
        ],
        gate_name="synthetic_back_half",
    )

    assert gate["status"] == "pass"
    assert gate["claim_status"] == "supports_back_half_improvement_claim"
    assert gate["carry_forward_comparable"] is False


def test_support_aware_rate_gate_reports_claim_status_and_train_support() -> None:
    gate = _score_supported_rate_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "support_aware_back_half_rates": {
                    "candidate": {
                        "status": "scored",
                        "mean_normalized_mae": 0.04,
                        "supported_target_count": 2,
                        "metric_rows": [
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                        ],
                    },
                    "baseline": {
                        "status": "scored",
                        "mean_normalized_mae": 0.08,
                        "supported_target_count": 2,
                        "metric_rows": [
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                        ],
                    },
                    "carry_forward": {
                        "status": "not_evaluable",
                        "mean_normalized_mae": float("inf"),
                        "supported_target_count": 2,
                        "metric_rows": [],
                    },
                },
            }
        ],
        gate_name="synthetic_rate",
    )

    assert gate["status"] == "pass"
    assert gate["claim_status"] == "supports_conditional_back_half_improvement_claim"
    assert gate["carry_forward_comparable"] is False


def test_support_aware_rate_gate_treats_machine_epsilon_as_non_regression() -> None:
    gate = _score_supported_rate_gate(
        [
            {
                "train_end_year": 2021,
                "holdout_years": [2022],
                "support_aware_back_half_rates": {
                    "candidate": {
                        "status": "scored",
                        "mean_normalized_mae": 0.1000000000001,
                        "supported_target_count": 1,
                        "metric_rows": [
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                        ],
                    },
                    "baseline": {
                        "status": "scored",
                        "mean_normalized_mae": 0.1,
                        "supported_target_count": 1,
                        "metric_rows": [
                            {"missing_prediction_count": 0, "train_supported_count": 1, "supported_target_count": 1},
                        ],
                    },
                    "carry_forward": {
                        "status": "not_evaluable",
                        "mean_normalized_mae": float("inf"),
                        "supported_target_count": 1,
                        "metric_rows": [],
                    },
                },
            }
        ],
        gate_name="synthetic_rate_tolerance",
    )

    assert gate["status"] == "pass"
    assert gate["claim_status"] == "conditional_back_half_non_regression_only"
