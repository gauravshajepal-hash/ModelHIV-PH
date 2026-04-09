from __future__ import annotations

import math

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase1.latent_observability import build_latent_observability_audit
from epigraph_ph.phase1.observation_noise import build_observation_noise_model
from epigraph_ph.phase1.pipeline import (
    _bias_fields,
    _normalize_numeric_value,
    _normalize_unit,
    _source_reliability_class,
)
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase1_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase1_cfg = (plugin.constraint_settings or {}).get("phase1", {})
    literature_cfg = (plugin.constraint_settings or {}).get("literature_review", {})

    assert phase1_cfg.get("denominator_rules", {}).get("default_count_denominator") is not None
    assert phase1_cfg.get("winsorization", {}).get("lower_quantile") is not None
    assert phase1_cfg.get("preprocessing", {}).get("transform_mode") is not None
    assert phase1_cfg.get("preprocessing", {}).get("scaling_mode") is not None
    assert phase1_cfg.get("missing_data", {}).get("quality_beta") is not None
    assert phase1_cfg.get("evidence_class_weights", {}).get("observed_numeric") is not None
    assert phase1_cfg.get("reliability_class_rules", {}).get("official_routine_anchor") is not None
    assert phase1_cfg.get("bias_class_thresholds", {}).get("measurement_low") is not None
    assert phase1_cfg.get("spatial_relevance_by_geo", {}).get("province") is not None
    assert phase1_cfg.get("bias_penalty_mix", {}).get("measurement") is not None
    assert phase1_cfg.get("replication_weight", {}).get("base") is not None
    assert phase1_cfg.get("observation_noise", {}).get("enabled") is True
    assert phase1_cfg.get("observability_gates", {}).get("subnational_inference", {}).get("province_support_min") is not None
    assert literature_cfg.get("tier_quality_weight", {}).get("tier1_official_anchor") is not None
    assert literature_cfg.get("quality_score_weights", {}).get("tier") is not None
    assert literature_cfg.get("promotion_thresholds", {}).get("strong_numeric_confidence") is not None


def test_phase1_unit_and_reliability_helpers() -> None:
    assert _normalize_unit("%") == "percent"
    assert _normalize_unit("million") == "count_million"
    assert _normalize_numeric_value(25.0, "percent") == 0.25
    assert _normalize_numeric_value(2.0, "count_million") == 2_000_000.0
    row = {
        "source_bank": "phase0_extracted",
        "is_anchor_eligible": True,
        "is_direct_measurement": True,
        "is_prior_only": False,
    }
    reliability = _source_reliability_class(row, "observed_numeric")
    assert reliability == "official_routine_anchor"
    bias = _bias_fields(row, reliability, "monthly", "province")
    assert 0.0 <= bias["bias_penalty"] <= 1.0
    assert 0.0 <= bias["source_tier_weight"] <= 1.0
    assert bias["promotion_eligibility_hint"] == "promotion_eligible"


def test_phase1_normalized_rows_have_bias_and_truth_fields(rescue_v2_run_dir) -> None:
    rows = read_json(rescue_v2_run_dir / "phase1" / "normalized_subparameters.json", default=[])
    assert rows
    required = {
        "source_reliability_class",
        "measurement_bias_class",
        "sampling_bias_class",
        "reporting_delay_class",
        "promotion_eligibility_hint",
        "source_tier_weight",
        "measurement_quality_weight",
        "temporal_freshness_weight",
        "spatial_relevance_weight",
        "replication_weight",
        "bias_penalty",
        "observation_precision",
        "observation_weight",
        "signal_family",
        "payload_family",
        "geo_binding_class",
        "candidate_block",
        "expected_sign",
        "measurement_role",
        "observation_operator",
    }
    for row in rows:
        assert required.issubset(row.keys())
        for key in (
            "source_tier_weight",
            "measurement_quality_weight",
            "temporal_freshness_weight",
            "spatial_relevance_weight",
            "replication_weight",
            "bias_penalty",
            "observation_precision",
            "observation_weight",
        ):
            value = float(row[key])
            assert math.isfinite(value)
            assert value >= 0.0


def test_phase1_emits_latent_observability_artifacts(rescue_v2_run_dir) -> None:
    audit = read_json(rescue_v2_run_dir / "phase1" / "latent_observability_audit.json", default={})
    split = read_json(rescue_v2_run_dir / "phase1" / "direct_vs_contextual_split.json", default={})

    assert audit.get("rows")
    assert split.get("rows")
    first = audit["rows"][0]
    assert isinstance(first.get("eligible_for_national_likelihood"), bool)
    assert isinstance(first.get("eligible_for_province_graph"), bool)
    assert isinstance(first.get("eligible_for_subnational_inference"), bool)
    assert "subnational_support_score" in first
    assert "aggregate_support_score" in first
    assert "direct_indicator_count" in first
    summary = split.get("summary", {})
    assert "row_counts" in summary
    assert summary["row_counts"].get("context_only", 0) >= 1
    assert int(audit.get("summary", {}).get("eligible_for_subnational_inference_count", 0)) >= 1


def test_phase1_subnational_inference_gate_requires_real_subnational_support() -> None:
    audit = build_latent_observability_audit(
        normalized_rows=[
            {
                "canonical_name": "cash_instability",
                "model_numeric_value": 0.21,
                "measurement_role": "direct_indicator",
                "geo_resolution": "national",
                "time_resolution": "annual",
                "candidate_block": "structural_barrier_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.11,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.12,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.13,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.14,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.15,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
            {
                "canonical_name": "mobility_network_mixing",
                "model_numeric_value": 0.16,
                "measurement_role": "direct_indicator",
                "geo_resolution": "region",
                "time_resolution": "monthly",
                "candidate_block": "mobility_exposure_pressure",
                "expected_sign": "positive",
            },
        ],
        parameter_catalog=[],
        plugin_id="hiv",
    )
    rows = {row["canonical_name"]: row for row in audit["rows"]}
    assert rows["cash_instability"]["eligible_for_province_graph"] is True
    assert rows["cash_instability"]["eligible_for_subnational_inference"] is False
    assert rows["mobility_network_mixing"]["eligible_for_subnational_inference"] is True


def test_phase1_observation_noise_model_learns_higher_precision_for_repeated_anchor_rows() -> None:
    model = build_observation_noise_model(
        normalized_rows=[
            {
                "normalized_id": "anchor_1",
                "canonical_name": "testing_rate",
                "model_numeric_value": 0.52,
                "measurement_role": "direct_indicator",
                "source_reliability_class": "official_routine_anchor",
                "geo_resolution": "province",
                "time_resolution": "monthly",
                "is_anchor_eligible": True,
                "is_direct_measurement": True,
                "measurement_quality_weight": 0.95,
                "temporal_freshness_weight": 0.90,
                "spatial_relevance_weight": 0.95,
                "replication_weight": 0.90,
                "evidence_weight": 1.0,
                "province": "Cebu",
                "time": "2025-01",
            },
            {
                "normalized_id": "anchor_2",
                "canonical_name": "testing_rate",
                "model_numeric_value": 0.51,
                "measurement_role": "direct_indicator",
                "source_reliability_class": "official_routine_anchor",
                "geo_resolution": "province",
                "time_resolution": "monthly",
                "is_anchor_eligible": True,
                "is_direct_measurement": True,
                "measurement_quality_weight": 0.95,
                "temporal_freshness_weight": 0.90,
                "spatial_relevance_weight": 0.95,
                "replication_weight": 0.90,
                "evidence_weight": 1.0,
                "province": "Cebu",
                "time": "2025-01",
            },
            {
                "normalized_id": "proxy_1",
                "canonical_name": "testing_rate",
                "model_numeric_value": 0.80,
                "measurement_role": "proxy_indicator",
                "source_reliability_class": "proxy_inferred",
                "geo_resolution": "national",
                "time_resolution": "annual",
                "is_anchor_eligible": False,
                "is_direct_measurement": False,
                "measurement_quality_weight": 0.35,
                "temporal_freshness_weight": 0.40,
                "spatial_relevance_weight": 0.35,
                "replication_weight": 0.10,
                "evidence_weight": 0.35,
                "geo": "Philippines",
                "time": "2025",
            },
            {
                "normalized_id": "proxy_2",
                "canonical_name": "testing_rate",
                "model_numeric_value": 0.10,
                "measurement_role": "proxy_indicator",
                "source_reliability_class": "proxy_inferred",
                "geo_resolution": "national",
                "time_resolution": "annual",
                "is_anchor_eligible": False,
                "is_direct_measurement": False,
                "measurement_quality_weight": 0.35,
                "temporal_freshness_weight": 0.40,
                "spatial_relevance_weight": 0.35,
                "replication_weight": 0.10,
                "evidence_weight": 0.35,
                "geo": "Philippines",
                "time": "2025",
            },
        ],
        plugin_id="hiv",
    )
    rows = {row["normalized_id"]: row for row in model["rows"]}
    assert rows["anchor_1"]["observation_precision"] > rows["proxy_1"]["observation_precision"]
    assert rows["anchor_2"]["observation_weight"] > rows["proxy_2"]["observation_weight"]


def test_phase1_tensor_contract_and_sanity(rescue_v2_run_dir) -> None:
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})
    truth_summary = read_json(rescue_v2_run_dir / "phase1" / "ground_truth_summary.json", default={})
    tensor_rows = read_json(rescue_v2_run_dir / "phase1" / "tensor_rows.json", default=[])
    report = read_json(rescue_v2_run_dir / "phase1" / "normalization_report.json", default={})
    interop_report = read_json(rescue_v2_run_dir / "phase1" / "interop_report.json", default={})
    tensor_schema = read_json(rescue_v2_run_dir / "phase1" / "tensor_schema.json", default={})
    standardized = load_tensor_artifact(rescue_v2_run_dir / "phase1" / "standardized_tensor.npz")
    denominators = load_tensor_artifact(rescue_v2_run_dir / "phase1" / "denominator_tensor.npz")
    missing_mask = load_tensor_artifact(rescue_v2_run_dir / "phase1" / "missing_mask.npz")
    quality_weight_tensor = load_tensor_artifact(rescue_v2_run_dir / "phase1" / "quality_weight_tensor.npz")

    province_axis = axis_catalogs.get("province", [])
    month_axis = axis_catalogs.get("month", [])
    canonical_axis = axis_catalogs.get("canonical_name", [])
    assert standardized.shape == (len(province_axis), len(month_axis), len(canonical_axis))
    assert denominators.shape == standardized.shape
    assert missing_mask.shape == standardized.shape
    assert quality_weight_tensor.shape == standardized.shape
    assert np.isfinite(standardized).all()
    assert np.isfinite(denominators).all()
    assert np.isfinite(missing_mask).all()
    assert np.isfinite(quality_weight_tensor).all()
    assert np.any(denominators != 1.0)
    assert set(np.unique(missing_mask)).issubset({0.0, 1.0})
    assert len(tensor_rows) == len(province_axis) * len(month_axis) * len(canonical_axis)
    assert len({row["tensor_row_id"] for row in tensor_rows}) == len(tensor_rows)
    assert report.get("normalized_row_count", 0) >= report.get("anchor_eligible_count", 0)
    assert report.get("tensor_row_count", 0) == len(tensor_rows)
    assert len(report.get("source_reliability_counts", {})) > 0
    assert report.get("source_registry_mode") in {"registry", "phase0_candidates_fallback"}
    assert report.get("preprocess_meta", {}).get("transform", {}).get("transform_mode") in {"log1p", "boxcox"}
    assert report.get("preprocess_meta", {}).get("scaling", {}).get("scaling_mode") in {"iqr", "huber"}
    assert "used_dlpack" in interop_report
    assert "missing_mask" in tensor_schema.get("value_fields", {})
    assert "quality_weight_tensor" in tensor_schema.get("value_fields", {})
    assert truth_summary.get("phase_name") == "phase1"


def test_phase1_geography_is_split_and_not_collapsed(rescue_v2_run_dir) -> None:
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})
    province_axis = axis_catalogs.get("province", [])
    assert len(province_axis) > 1
    assert any(name not in {"Philippines", "national"} for name in province_axis)
    assert "Philippines" in province_axis
    assert not any(name in {"national", "National"} for name in province_axis)


def test_phase1_month_axis_uses_valid_observation_years(rescue_v2_run_dir) -> None:
    axis_catalogs = read_json(rescue_v2_run_dir / "phase1" / "axis_catalogs.json", default={})
    month_axis = axis_catalogs.get("month", [])
    assert month_axis
    for value in month_axis:
        year = int(str(value)[:4])
        assert 1980 <= year <= 2026
