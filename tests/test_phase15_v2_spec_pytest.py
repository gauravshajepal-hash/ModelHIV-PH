from __future__ import annotations

from epigraph_ph.phase15.v2_spec import build_phase15_v2_model_spec


def test_phase15_v2_model_spec_declares_bottom_up_mixed_frequency_contract() -> None:
    normalized_rows = [
        {
            "canonical_name": "testing_rate",
            "measurement_role": "direct_indicator",
            "observation_operator": "monthly_snapshot",
            "geo_resolution": "province",
            "time_resolution": "monthly",
        },
        {
            "canonical_name": "poverty_rate",
            "measurement_role": "proxy_indicator",
            "observation_operator": "annual_snapshot",
            "geo_resolution": "region",
            "time_resolution": "annual",
        },
    ]
    observability_audit = {
        "rows": [
            {
                "canonical_name": "testing_rate",
                "expected_sign": "positive",
                "direct_indicator_count": 3,
                "proxy_indicator_count": 0,
                "numeric_row_count": 3,
                "national_support_count": 1,
                "regional_support_count": 1,
                "province_support_count": 1,
                "monthly_support_count": 3,
                "annual_support_count": 0,
                "eligible_for_national_likelihood": True,
                "eligible_for_province_graph": True,
            },
            {
                "canonical_name": "prevention_coverage",
                "expected_sign": "positive",
                "direct_indicator_count": 2,
                "proxy_indicator_count": 0,
                "numeric_row_count": 2,
                "national_support_count": 0,
                "regional_support_count": 1,
                "province_support_count": 1,
                "monthly_support_count": 2,
                "annual_support_count": 0,
                "eligible_for_national_likelihood": True,
                "eligible_for_province_graph": True,
            },
            {
                "canonical_name": "poverty_rate",
                "expected_sign": "positive",
                "direct_indicator_count": 4,
                "proxy_indicator_count": 0,
                "numeric_row_count": 4,
                "national_support_count": 1,
                "regional_support_count": 1,
                "province_support_count": 1,
                "monthly_support_count": 0,
                "annual_support_count": 4,
                "eligible_for_national_likelihood": True,
                "eligible_for_province_graph": True,
            },
            {
                "canonical_name": "education",
                "expected_sign": "negative",
                "direct_indicator_count": 2,
                "proxy_indicator_count": 0,
                "numeric_row_count": 2,
                "national_support_count": 1,
                "regional_support_count": 1,
                "province_support_count": 0,
                "monthly_support_count": 0,
                "annual_support_count": 2,
                "eligible_for_national_likelihood": True,
                "eligible_for_province_graph": False,
            },
        ]
    }
    axis_catalogs = {
        "province": ["Cebu", "Bohol"],
        "month": ["2025-01", "2025-02"],
        "canonical_name": ["testing_rate", "prevention_coverage", "poverty_rate", "education"],
    }

    spec = build_phase15_v2_model_spec(
        normalized_rows=normalized_rows,
        observability_audit=observability_audit,
        axis_catalogs=axis_catalogs,
        plugin_id="hiv",
    )

    assert spec["model_id"] == "phase15_v2_mixed_frequency_hierarchical_dynamic_factor_model"
    assert spec["selected_autoresearch_variant"] == "evidence-to-model-loop"
    equations = {row["equation_id"]: row for row in spec["equations"]}
    assert "sum_p omega_{p,b} x_{p,t,b}" in equations["E3"]["math"]
    assert "H_i[p,t]" in equations["E5"]["math"]
    assert "softplus" in equations["E4"]["math"]
    assert "log tau_i" in equations["E7"]["math"]
    assert spec["current_repo_state"]["province_count"] == 2
    assert spec["current_repo_state"]["month_count"] == 2
    sign_prior_names = {row["canonical_name"] for row in spec["sign_priors"]}
    assert "testing_rate" in sign_prior_names
    assert "poverty_rate" in sign_prior_names
    assert "false_wins" in spec["evaluation"]
    assert "required_checks" in spec["evaluation"]
