from __future__ import annotations

import numpy as np

from epigraph_ph.phase15.v2_engine import fit_phase15_v2_map_engine
from epigraph_ph.phase15.v2_spec import build_phase15_v2_observation_support


def _row(
    canonical_name: str,
    province: str,
    month: str,
    value: float,
    *,
    measurement_role: str = "direct_indicator",
) -> dict[str, object]:
    return {
        "canonical_name": canonical_name,
        "candidate_block": "synthetic_barrier_block",
        "measurement_role": measurement_role,
        "expected_sign": "positive" if canonical_name == "cash_instability" else "negative",
        "source_bank": "synthetic_source",
        "geo_resolution": "province",
        "province": province,
        "region": "Region VII",
        "time": month,
        "time_resolution": "monthly",
        "temporal_precision": "monthly_snapshot",
        "model_numeric_value": value,
        "evidence_weight": 1.0,
        "quality_weight": 1.0,
        "bias_penalty": 0.0,
        "is_anchor_eligible": False,
        "is_direct_measurement": measurement_role == "direct_indicator",
    }


def test_phase15_v2_engine_learns_signed_loadings_and_bottom_up_aggregation() -> None:
    province_axis = ["Cebu", "Bohol"]
    month_axis = ["2025-01", "2025-02"]
    region_labels = ["Region VII", "Region VII"]
    normalized_rows = [
        _row("cash_instability", "Cebu", "2025-01", 0.20),
        _row("cash_instability", "Cebu", "2025-02", 0.35),
        _row("cash_instability", "Bohol", "2025-01", 0.85),
        _row("cash_instability", "Bohol", "2025-02", 1.05),
        _row("education", "Cebu", "2025-01", 1.20),
        _row("education", "Cebu", "2025-02", 1.05),
        _row("education", "Bohol", "2025-01", 0.55),
        _row("cash_instability", "Bohol", "2025-02", 99.0, measurement_role="context_only"),
    ]
    measurement_spec = {
        "retained_blocks": [
            {
                "block_id": "synthetic_barrier_block",
                "display_name": "Synthetic Barrier Block",
                "indicator_rows": [
                    {"canonical_name": "cash_instability", "expected_sign": "positive"},
                    {"canonical_name": "education", "expected_sign": "negative"},
                ],
            }
        ]
    }
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )

    result = fit_phase15_v2_map_engine(
        normalized_rows=normalized_rows,
        observation_support=observation_support,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        plugin_id="hiv",
    )

    indicator_rows = {row["canonical_name"]: row for row in result["indicator_parameters"]["rows"]}
    assert indicator_rows["cash_instability"]["lambda"] > 0.0
    assert indicator_rows["education"]["lambda"] < 0.0
    assert abs(float(indicator_rows["cash_instability"]["lambda"])) <= 4.0
    assert abs(float(indicator_rows["education"]["lambda"])) <= 4.0
    assert abs(float(indicator_rows["cash_instability"]["eta"])) < 50.0

    national_weights = np.asarray(result["aggregation"]["national_weights"], dtype=np.float64)
    province_tensor = np.asarray(result["province_state_tensor"], dtype=np.float64)
    national_tensor = np.asarray(result["national_state_tensor"], dtype=np.float64)
    expected_national = np.sum(national_weights.reshape(-1, 1) * province_tensor[:, :, 0], axis=0)
    assert np.allclose(national_tensor[0, :, 0], expected_national, atol=1e-6)

    measurement_rows = list(result["measurement_rows"]["rows"])
    assert measurement_rows
    assert all(str(row["measurement_role"]) != "context_only" for row in measurement_rows)
    assert result["measurement_rows"]["row_count"] == 7
    calibration = dict(result["calibration"])
    assert calibration["available"] is True
    assert calibration["summary"]["row_count"] == len(result["measurement_fit"]["rows"])
    assert calibration["rows"]
    pooling = dict(result["pooling_sensitivity"])
    assert pooling["available"] is True
    assert pooling["summary"]["scenario_count"] >= 1
    assert pooling["rows"]

    uncertainty_rows = {row["province"]: row for row in result["uncertainty"]["rows"]}
    cebu_std = float(np.mean(np.asarray(uncertainty_rows["Cebu"]["posterior_std_values"], dtype=np.float64)))
    bohol_std = float(np.mean(np.asarray(uncertainty_rows["Bohol"]["posterior_std_values"], dtype=np.float64)))
    assert bohol_std > cebu_std


def test_phase15_v2_engine_learns_burden_weighted_aggregation_from_national_proxy_rows() -> None:
    province_axis = ["Cebu", "Bohol"]
    month_axis = ["2025-01", "2025-02"]
    region_labels = ["Region VII", "Region VII"]
    normalized_rows = [
        _row("cash_instability", "Cebu", "2025-01", 0.20),
        _row("cash_instability", "Cebu", "2025-02", 0.30),
        _row("cash_instability", "Bohol", "2025-01", 0.75),
        _row("cash_instability", "Bohol", "2025-02", 0.90),
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.8,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.8,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
    ]
    measurement_spec = {
        "retained_blocks": [
            {
                "block_id": "synthetic_barrier_block",
                "display_name": "Synthetic Barrier Block",
                "indicator_rows": [
                    {"canonical_name": "cash_instability", "expected_sign": "positive"},
                ],
            }
        ]
    }
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )

    result = fit_phase15_v2_map_engine(
        normalized_rows=normalized_rows,
        observation_support=observation_support,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        plugin_id="hiv",
    )

    assert result["aggregation"]["weight_source"] == "learned_burden_softmax"
    assert "population_total" in result["aggregation"]["feature_names"]
    national_weights = np.asarray(result["aggregation"]["national_weights"], dtype=np.float64)
    assert abs(float(national_weights[1] - national_weights[0])) > 0.25
    diagnostics = dict(result["aggregation"]["diagnostics"])
    assert float(diagnostics["objective_gain"]) > 0.0
    assert float(diagnostics["coefficient_norm"]) > 0.0


def test_phase15_v2_engine_marks_conflicting_burden_weight_objective_as_unidentified() -> None:
    province_axis = ["Cebu", "Bohol"]
    month_axis = ["2025-01"]
    region_labels = ["Region VII", "Region VII"]
    normalized_rows = [
        _row("cash_instability", "Cebu", "2025-01", 0.30),
        _row("cash_instability", "Bohol", "2025-01", 0.30),
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.5,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
    ]
    measurement_spec = {
        "retained_blocks": [
            {
                "block_id": "synthetic_barrier_block",
                "display_name": "Synthetic Barrier Block",
                "indicator_rows": [
                    {"canonical_name": "cash_instability", "expected_sign": "positive"},
                ],
            }
        ]
    }
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )

    result = fit_phase15_v2_map_engine(
        normalized_rows=normalized_rows,
        observation_support=observation_support,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        plugin_id="hiv",
    )

    assert result["aggregation"]["weight_source"] == "unidentified_burden_prior_fallback"
    diagnostics = dict(result["aggregation"]["diagnostics"])
    assert int(diagnostics["sample_count"]) == 2
    assert float(diagnostics["objective_gain"]) <= 1e-6


def test_phase15_v2_missing_information_layer_reduces_aggregate_residual_and_inflates_weak_province_uncertainty() -> None:
    province_axis = ["Cebu", "Bohol"]
    month_axis = ["2025-01", "2025-02"]
    region_labels = ["Region VII", "Region VII"]
    normalized_rows = [
        _row("cash_instability", "Cebu", "2025-01", 0.15),
        _row("cash_instability", "Cebu", "2025-02", 0.20),
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Cebu",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "province",
            "province": "Bohol",
            "region": "Region VII",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 3.0,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "population_total",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.8,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "case_count",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "proxy_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 2.8,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": False,
            "is_direct_measurement": False,
        },
        {
            "canonical_name": "cash_instability",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "direct_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-01",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 0.95,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": True,
            "is_direct_measurement": True,
        },
        {
            "canonical_name": "cash_instability",
            "candidate_block": "synthetic_barrier_block",
            "measurement_role": "direct_indicator",
            "expected_sign": "positive",
            "source_bank": "synthetic_source",
            "geo_resolution": "national",
            "geo": "Philippines",
            "region": "national",
            "time": "2025-02",
            "time_resolution": "monthly",
            "temporal_precision": "monthly_snapshot",
            "model_numeric_value": 1.05,
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "bias_penalty": 0.0,
            "is_anchor_eligible": True,
            "is_direct_measurement": True,
        },
    ]
    measurement_spec = {
        "retained_blocks": [
            {
                "block_id": "synthetic_barrier_block",
                "display_name": "Synthetic Barrier Block",
                "indicator_rows": [
                    {"canonical_name": "cash_instability", "expected_sign": "positive"},
                ],
            }
        ]
    }
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )

    result = fit_phase15_v2_map_engine(
        normalized_rows=normalized_rows,
        observation_support=observation_support,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        plugin_id="hiv",
    )

    missing_information = dict(result["missing_information"])
    assert missing_information["summary"]["available_block_count"] == 1
    aggregate_rows = list(missing_information["aggregate_rows"])
    assert aggregate_rows
    before = np.mean([abs(float(row["residual_before"])) for row in aggregate_rows])
    after = np.mean([abs(float(row["residual_after"])) for row in aggregate_rows])
    assert after < before

    province_rows = {str(row["province"]): row for row in missing_information["rows"]}
    cebu_share = float(np.mean(np.asarray(province_rows["Cebu"]["imputation_share_values"], dtype=np.float64)))
    bohol_share = float(np.mean(np.asarray(province_rows["Bohol"]["imputation_share_values"], dtype=np.float64)))
    assert bohol_share > cebu_share

    cebu_correction = float(np.mean(np.abs(np.asarray(province_rows["Cebu"]["correction_values"], dtype=np.float64))))
    bohol_correction = float(np.mean(np.abs(np.asarray(province_rows["Bohol"]["correction_values"], dtype=np.float64))))
    assert bohol_correction > cebu_correction

    uncertainty_rows = {row["province"]: row for row in result["uncertainty"]["rows"]}
    cebu_std = float(np.mean(np.asarray(uncertainty_rows["Cebu"]["posterior_std_values"], dtype=np.float64)))
    bohol_std = float(np.mean(np.asarray(uncertainty_rows["Bohol"]["posterior_std_values"], dtype=np.float64)))
    assert bohol_std > cebu_std
    solver_backends = list(missing_information["summary"].get("solver_backends") or [])
    assert solver_backends
    assert solver_backends[0] in {"torch_cuda", "cpu_sparse"}
    assert province_rows["Cebu"]["solver_backend"] in {"torch_cuda", "cpu_sparse"}
    block_solver = dict(result["fit_summary"]["rows"][0]["missing_information_summary"]["solver"])
    assert str(block_solver["preconditioner"]) in {"temporal_block_cholesky", "identity"}
    numerical_adequacy = dict(result["numerical_adequacy"])
    assert numerical_adequacy["summary"]["available_block_count"] == 1
    tolerance_rows = list(numerical_adequacy.get("tolerance_rows") or [])
    direct_reference_rows = list(numerical_adequacy.get("direct_reference_rows") or [])
    assert tolerance_rows
    assert direct_reference_rows
    assert max(float(row["posterior_std_mae"]) for row in tolerance_rows) == 0.0
    assert max(float(row["posterior_std_mae"]) for row in direct_reference_rows) == 0.0
    assert min(float(row["rtol"]) for row in tolerance_rows) == float(numerical_adequacy["summary"]["anchor_rtol"])
    assert int(numerical_adequacy["summary"]["tolerance_nonconverged_count"]) >= 0
    assert int(numerical_adequacy["summary"]["tolerance_iteration_cap_count"]) >= 0
    assert all("cg_converged" in row for row in tolerance_rows)
    assert all("cg_max_iter_reached" in row for row in tolerance_rows)
    assert all("cg_converged" in row for row in direct_reference_rows)
    assert all("preconditioner" in row for row in tolerance_rows)
    assert max(float(row["state_mae"]) for row in tolerance_rows) < 0.25
    assert max(float(row["state_mae"]) for row in direct_reference_rows) < 0.25
