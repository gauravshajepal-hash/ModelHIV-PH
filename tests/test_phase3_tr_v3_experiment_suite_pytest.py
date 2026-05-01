from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite


def test_build_experiment_suite_specs_contains_expected_blocks() -> None:
    specs = suite.build_experiment_suite_specs()
    ids = {spec.experiment_id for spec in specs}
    spec_map = {spec.experiment_id: spec for spec in specs}
    assert "EXP-B1" in ids
    assert "EXP-B0" in ids
    assert "EXP-B2" in ids
    assert "EXP-V1" in ids
    assert "EXP-V2" in ids
    assert "EXP-S1-A1" in ids
    assert "EXP-L2-A1" in ids
    assert "EXP-05a-05" in ids
    assert "EXP-N3" in ids
    assert "EXP-R1" in ids
    assert "EXP-R2" in ids
    assert "EXP-R3" in ids
    assert "EXP-R4" in ids
    assert "EXP-R5" in ids
    assert "EXP-R6" in ids
    assert "EXP-R7" in ids
    assert "EXP-R8" in ids
    assert "EXP-R9" in ids
    assert "EXP-R10" in ids
    assert "EXP-R10-M1" in ids
    assert "EXP-R10-M1-B1" in ids
    assert "EXP-R10-M1-F1" in ids
    assert "EXP-R10-M1-F1-C1" in ids
    assert "EXP-R10-M2" in ids
    assert "EXP-R10-CHAMPION" in ids
    assert "EXP-R10-EXACT-CHAMPION" in ids
    assert "EXP-R10-DENSE-CHAMPION" in ids
    assert "EXP-R10-DENSE-H1" in ids
    assert "EXP-R10-DENSE-M1" in ids
    assert "EXP-R10-DENSE-M1-H1" in ids
    assert "EXP-R10-DENSE-M1-C1-H1" in ids
    assert "EXP-R10-DENSE-M1-B1-H1" in ids
    assert "EXP-R10-DENSE-M1-F1-H1" in ids
    assert "EXP-R10-DENSE-M2" in ids
    assert "EXP-R11" in ids
    assert "EXP-L1" in ids
    assert "EXP-05b-full-open-leaky-model" in ids
    assert spec_map["EXP-B0"].diagnostic_kind == "contract_ablation"
    assert spec_map["EXP-B2"].diagnostic_kind == "dense_contract_summary"
    assert spec_map["EXP-V1"].diagnostic_kind == "purged_dense_contract"
    assert spec_map["EXP-V2"].diagnostic_kind == "endpoint_tier_audit"
    assert spec_map["EXP-S1-A1"].diagnostic_kind == "susceptible_sidecar"
    assert spec_map["EXP-L2-A1"].diagnostic_kind == "late_leakage_sensitivity"
    assert spec_map["EXP-L1"].family == "05a"
    assert spec_map["EXP-R1"].family == "repair"
    assert spec_map["EXP-R1"].transition_model == "strict_support_drift"
    assert spec_map["EXP-R1"].emit_hazard_curves is True
    assert spec_map["EXP-R2"].family == "repair"
    assert spec_map["EXP-R2"].transition_model == "care_share_repair"
    assert spec_map["EXP-R3"].family == "repair"
    assert spec_map["EXP-R3"].transition_model == "art_stock_repair"
    assert spec_map["EXP-R4"].family == "repair"
    assert spec_map["EXP-R4"].transition_model == "hybrid_care_repair"
    assert spec_map["EXP-R5"].family == "repair"
    assert spec_map["EXP-R5"].transition_model == "hybrid_d_to_a_only"
    assert spec_map["EXP-R6"].family == "repair"
    assert spec_map["EXP-R6"].transition_model == "d_to_a_only_repair"
    assert spec_map["EXP-R7"].family == "repair"
    assert spec_map["EXP-R7"].transition_model == "regime_aware_d_to_a_repair"
    assert spec_map["EXP-R8"].family == "repair"
    assert spec_map["EXP-R8"].transition_model == "two_regime_d_to_a_repair"
    assert spec_map["EXP-R9"].family == "repair"
    assert spec_map["EXP-R9"].transition_model == "art_delta_repair"
    assert spec_map["EXP-R10"].family == "repair"
    assert spec_map["EXP-R10"].transition_model == "direct_observation_repair"
    assert spec_map["EXP-R10-M1"].transition_model == "direct_observation_joint_consistency"
    assert spec_map["EXP-R10-M1-B1"].transition_model == "direct_observation_joint_consistency_bias_corrected"
    assert spec_map["EXP-R10-M1-F1"].transition_model == "direct_observation_joint_consistency_bias_corrected"
    assert spec_map["EXP-R10-M1-F1"].repair_params["flow_consistency_weight"] == 1.0
    assert spec_map["EXP-R10-M1-F1-C1"].transition_model == "direct_observation_joint_consistency_crossfit_calibrated"
    assert spec_map["EXP-R10-M1-F1-C1"].repair_params["flow_consistency_weight"] == 1.0
    assert spec_map["EXP-R10-M1-F1-C1"].repair_params["use_bias_correction"] is True
    assert spec_map["EXP-R10-M2"].transition_model == "direct_observation_bias_corrected"
    assert spec_map["EXP-R10-CHAMPION"].family == "repair"
    assert spec_map["EXP-R10-CHAMPION"].transition_model == "direct_observation_repair"
    assert spec_map["EXP-R10-CHAMPION"].repair_params["flow_weight"] == 0.5
    assert spec_map["EXP-R10-EXACT-CHAMPION"].benchmark_role == "predictive_exact_candidate"
    assert spec_map["EXP-R10-DENSE-CHAMPION"].benchmark_role == "predictive_dense_candidate"
    assert spec_map["EXP-R10-EXACT-CHAMPION"].repair_params["flow_series_model"] == "delta"
    assert spec_map["EXP-R10-DENSE-CHAMPION"].repair_params["art_series_model"] == "delta"
    assert spec_map["EXP-R10-DENSE-H1"].repair_params["suppression_fallback_mode"] == "unclaimed"
    assert spec_map["EXP-R10-DENSE-M1"].transition_model == "direct_observation_joint_consistency"
    assert spec_map["EXP-R10-DENSE-M1-H1"].repair_params["suppression_fallback_mode"] == "unclaimed"
    assert spec_map["EXP-R10-DENSE-M1-C1-H1"].transition_model == "direct_observation_joint_consistency_crossfit_calibrated"
    assert spec_map["EXP-R10-DENSE-M1-C1-H1"].repair_params["suppression_fallback_mode"] == "unclaimed"
    assert spec_map["EXP-R10-DENSE-M1-B1-H1"].transition_model == "direct_observation_joint_consistency_bias_corrected"
    assert spec_map["EXP-R10-DENSE-M1-B1-H1"].repair_params["suppression_fallback_mode"] == "unclaimed"
    assert spec_map["EXP-R10-DENSE-M1-F1-H1"].transition_model == "direct_observation_joint_consistency"
    assert spec_map["EXP-R10-DENSE-M1-F1-H1"].repair_params["suppression_fallback_mode"] == "unclaimed"
    assert spec_map["EXP-R10-DENSE-M1-F1-H1"].repair_params["flow_consistency_weight"] == 1.0
    assert spec_map["EXP-R10-DENSE-M2"].transition_model == "direct_observation_bias_corrected"
    assert spec_map["EXP-R11"].transition_model == "direct_observation_mechanistic_overlay"
    assert spec_map["EXP-L1"].diagnostic_kind is None
    assert spec_map["EXP-L1"].transition_ridge_multipliers == {"A_to_L": 8.0, "L_to_A": 8.0}
    skipped = {spec.experiment_id for spec in specs if spec.status == "skipped"}
    assert {"EXP-M1", "EXP-05b-full-open-leaky-model"} <= skipped


def test_quarterly_benchmark_candidate_registry_is_stable() -> None:
    assert suite.default_quarterly_benchmark_candidate_id() == "EXP-R10-EXACT-CHAMPION"
    assert suite.default_predictive_candidate_id("exact_only") == "EXP-R10-EXACT-CHAMPION"
    assert suite.default_predictive_candidate_id("dense_train_observed_score") == "EXP-R10-DENSE-CHAMPION"
    assert suite.mechanistic_research_anchor_id() == "EXP-R1"
    assert "EXP-B0" in suite.canonical_design_experiment_ids()
    assert "EXP-05b-full-open-leaky-model" in suite.canonical_design_experiment_ids()


def test_tier_mapping_helpers_are_stable() -> None:
    assert suite._tier_from_snapshot_source_kind("exact_snapshot") == "exact_observed"
    assert suite._tier_from_snapshot_source_kind("monthly_bridge") == "bridge_observed"
    assert suite._tier_from_snapshot_source_kind("monthly_aggregate") == "bridge_observed"
    assert suite._annual_row_tier({"source_quality_tier": "official_wdi_unaids_hiv_series"}) == "exact_observed"
    assert suite._annual_row_tier({"source_quality_tier": "model_estimate"}) == "rule_based_extrapolated"


def test_availability_payload_smoke_on_live_archive() -> None:
    archive_run_id = suite._latest_standard_archive_run()
    archive_path = Path(suite.repo_root()) / "artifacts" / "runs" / archive_run_id / "harp_archive" / "historical_metric_rows.json"
    if not archive_path.exists():
        pytest.skip("Live archive is not available.")
    payload = suite._build_availability_payload(archive_run_id)
    assert payload["quarterly"]["exact_observation_row_count"] >= 1
    assert "diagnosed_plhiv" in payload["quarterly"]["metrics"]
    assert "annual_new_infections" in payload["annual"]["metrics"]
    assert len(payload["quarterly"]["quarters"]) >= payload["quarterly"]["exact_observation_row_count"]


def test_strict_support_transition_model_stays_bounded_on_sparse_zero_rows() -> None:
    train_transition_rows = [
        {
            "quarter": "2012-Q1",
            "hazards": {"U_to_D": 0.0},
            "support_flags": {"U_to_D": False},
            "support_sources": {"U_to_D": "unsupported_diagnosis_flow"},
        },
        {
            "quarter": "2012-Q2",
            "hazards": {"U_to_D": 0.02},
            "support_flags": {"U_to_D": True},
            "support_sources": {"U_to_D": "observed_diagnosis_flow"},
        },
        {
            "quarter": "2012-Q3",
            "hazards": {"U_to_D": 0.035},
            "support_flags": {"U_to_D": True},
            "support_sources": {"U_to_D": "observed_diagnosis_flow"},
        },
        {
            "quarter": "2012-Q4",
            "hazards": {"U_to_D": 0.0},
            "support_flags": {"U_to_D": False},
            "support_sources": {"U_to_D": "unsupported_diagnosis_flow"},
        },
    ]
    controls_train = {"A": [0.0, 0.1, 0.2, 0.1], "C": [0.0, 0.0, 0.0, 0.0], "R": [0.0, 0.0, 0.0, 0.0]}

    result = suite._fit_strict_support_transition_model(
        train_transition_rows,
        ["2013-Q1", "2013-Q2"],
        "U_to_D",
        controls_train,
        ("A",),
        cfg=suite.DynamicControlConfig(ridge_penalty=0.01, rho_clip=0.9, trend_scale=1.0),
    )

    diagnostics = dict(result["diagnostics"])
    assert diagnostics["model_kind"] == "strict_support_drift"
    assert diagnostics["support_count"] == 2
    assert max(result["forecast_map"].values()) <= 0.05
    assert min(result["forecast_map"].values()) >= 0.0


def test_save_hazard_curve_graph_writes_png() -> None:
    result = {
        "experiment_id": "EXP-R1",
        "quarterly_rows": [
            {
                "holdout_years": [2013],
                "transition_diagnostics": {
                    "U_to_D": {
                        "train_quarters": ["2012-Q1", "2012-Q2"],
                        "train_observed": [0.01, 0.02],
                        "train_fitted": [0.01, 0.018],
                        "train_supported": [False, True],
                        "forecast_quarters": ["2013-Q1", "2013-Q2"],
                        "forecast_hazards": [0.018, 0.019],
                        "support_count": 1,
                        "forecast_lower": 0.0,
                        "forecast_upper": 0.03,
                    }
                },
            }
        ],
    }
    tmp_dir = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"hazard_curve_{uuid.uuid4().hex}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "hazard.png"
    try:
        assert suite._save_hazard_curve_graph(result, path) is True
        assert path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_controls_for_transition_rows_aligns_by_quarter_not_position() -> None:
    train_rows = [
        {"quarter": "2018-Q1"},
        {"quarter": "2018-Q2"},
        {"quarter": "2018-Q3"},
        {"quarter": "2018-Q4"},
    ]
    train_transition_rows = [
        {"quarter": "2018-Q2"},
        {"quarter": "2018-Q4"},
    ]
    controls_full = {"A": [1.0, 2.0, 3.0, 4.0], "C": [10.0, 20.0, 30.0, 40.0]}

    aligned = suite._controls_for_transition_rows(train_rows, train_transition_rows, controls_full)

    assert aligned == {"A": [2.0, 4.0], "C": [20.0, 40.0]}


def test_fit_supported_share_series_returns_bounded_forecast() -> None:
    train_rows = [
        {"quarter": "2018-Q4", "diagnosed_plhiv": 100.0, "alive_on_art": 40.0, "diagnosed_plhiv_tier": "exact_observed", "alive_on_art_tier": "exact_observed"},
        {"quarter": "2019-Q4", "diagnosed_plhiv": 120.0, "alive_on_art": 54.0, "diagnosed_plhiv_tier": "exact_observed", "alive_on_art_tier": "exact_observed"},
        {"quarter": "2020-Q4", "diagnosed_plhiv": 150.0, "alive_on_art": 75.0, "diagnosed_plhiv_tier": "exact_observed", "alive_on_art_tier": "exact_observed"},
    ]
    result = suite._fit_supported_share_series(
        train_rows,
        ["2021-Q4", "2022-Q4"],
        numerator_metric="alive_on_art",
        denominator_metric="diagnosed_plhiv",
    )
    forecast = dict(result["forecast_map"])
    assert result["diagnostics"]["support_count"] == 3
    assert 0.0 <= forecast["2021-Q4"] <= 1.0
    assert 0.0 <= forecast["2022-Q4"] <= 1.0


def test_fit_supported_series_delta_mode_returns_nonnegative_levels() -> None:
    train_rows = [
        {"quarter": "2018-Q4", "alive_on_art": 40.0, "alive_on_art_tier": "exact_observed"},
        {"quarter": "2019-Q4", "alive_on_art": 54.0, "alive_on_art_tier": "exact_observed"},
        {"quarter": "2020-Q4", "alive_on_art": 75.0, "alive_on_art_tier": "exact_observed"},
    ]
    result = suite._fit_supported_series(
        train_rows,
        ["2021-Q4", "2022-Q4"],
        metric_name="alive_on_art",
        model_kind="delta",
        recent_blend_weight=0.75,
    )
    forecast = dict(result["forecast_map"])
    assert result["diagnostics"]["model_kind"] == "level_from_delta_bounded_drift"
    assert forecast["2021-Q4"] >= 0.0
    assert forecast["2022-Q4"] >= 0.0


def test_fit_supported_series_piecewise_mode_emits_changepoints() -> None:
    train_rows = [
        {"quarter": "2018-Q1", "diagnosed_plhiv": 100.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2018-Q2", "diagnosed_plhiv": 110.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2018-Q3", "diagnosed_plhiv": 120.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2018-Q4", "diagnosed_plhiv": 130.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2019-Q1", "diagnosed_plhiv": 130.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2019-Q2", "diagnosed_plhiv": 130.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2019-Q3", "diagnosed_plhiv": 130.0, "diagnosed_plhiv_tier": "exact_observed"},
        {"quarter": "2019-Q4", "diagnosed_plhiv": 130.0, "diagnosed_plhiv_tier": "exact_observed"},
    ]

    result = suite._fit_supported_series(
        train_rows,
        ["2020-Q1", "2020-Q2"],
        metric_name="diagnosed_plhiv",
        model_kind="piecewise",
    )

    diagnostics = dict(result["diagnostics"])
    assert diagnostics["model_kind"] == "piecewise_linear_changepoint"
    assert diagnostics["segment_count"] >= 1
    assert result["forecast_map"]["2020-Q1"] >= 0.0


def test_last_available_level_ignores_tier_restrictions() -> None:
    train_rows = [
        {"quarter": "2021-Q4", "virally_suppressed": None},
        {"quarter": "2022-Q4", "virally_suppressed": 1200.0, "virally_suppressed_tier": "rule_based_extrapolated"},
        {"quarter": "2023-Q4", "virally_suppressed": 1400.0, "virally_suppressed_tier": "latent_imputed"},
    ]

    value, count = suite._last_available_level(train_rows, metric_name="virally_suppressed")

    assert value == 1400.0


def test_susceptible_identification_contract_allows_annual_sidecar_only(monkeypatch: pytest.MonkeyPatch) -> None:
    annual_rows = []
    for year in range(2010, 2021):
        annual_rows.extend(
            [
                {"metric_name": "population_total", "year": year, "value": 1000000.0 + year},
                {"metric_name": "estimated_plhiv", "year": year, "value": 10000.0 + year},
                {"metric_name": "annual_new_infections", "year": year, "value": 1200.0 + year},
            ]
        )

    monkeypatch.setattr(suite, "_build_dense_contract_payload", lambda archive_run_id: {"rows": [], "summary": {}})
    monkeypatch.setattr(suite, "build_annual_anchor_rows", lambda archive_run_id: list(annual_rows))

    payload = suite._build_susceptible_identification_contract_payload("archive-run")

    assert payload["decision"]["status"] == "annual_sidecar_only"
    assert len(payload["annual_years"]["joint_support_years"]) == 11
    assert payload["allowed_blocks"]["quarterly_explicit_s"] == "not_identifiable_for_primary_loop"


def test_susceptible_sidecar_payload_reports_pressure_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        suite,
        "_build_susceptible_identification_contract_payload",
        lambda archive_run_id: {
            "archive_run_id": archive_run_id,
            "annual_years": {"joint_support_years": list(range(2016, 2025))},
            "proxy_rows": [
                {
                    "year": 2020,
                    "population_total": 1000000.0,
                    "estimated_plhiv": 10000.0,
                    "annual_new_infections": 1200.0,
                    "susceptible_proxy": 990000.0,
                    "susceptible_fraction": 0.99,
                }
            ],
            "decision": {"status": "annual_sidecar_only"},
        },
    )

    payload = suite._build_susceptible_sidecar_payload("archive-run")

    assert payload["decision"]["status"] == "annual_sidecar_only"
    assert payload["pressure_rows"][0]["annual_incidence_per_100k_susceptible"] > 0.0


def test_richer_leakage_identification_contract_stays_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv_tier": "exact_observed",
            "alive_on_art_tier": "bridge_observed",
            "virally_suppressed_tier": "bridge_observed",
            "new_diagnosed_cases_period_tier": "exact_observed",
            "deaths_reported_period_tier": "bridge_observed",
        },
        {
            "quarter": "2020-Q2",
            "diagnosed_plhiv_tier": "exact_observed",
            "alive_on_art_tier": "bridge_observed",
            "virally_suppressed_tier": "bridge_observed",
            "new_diagnosed_cases_period_tier": "exact_observed",
            "deaths_reported_period_tier": "bridge_observed",
        },
    ]
    annual_rows = [
        {"metric_name": "annual_aids_deaths", "year": 2020, "value": 1000.0},
    ]

    monkeypatch.setattr(suite, "_build_dense_contract_payload", lambda archive_run_id: {"rows": list(dense_rows), "summary": {}})
    monkeypatch.setattr(suite, "build_annual_anchor_rows", lambda archive_run_id: list(annual_rows))

    payload = suite._build_richer_leakage_identification_contract_payload("archive-run")

    assert payload["decision"]["status"] == "defer_richer_leakage"
    assert payload["allowed_blocks"]["annual_mortality_sanity"] == "allowed_as_sidecar_only"
    assert payload["yearly_support"][0]["can_score_art_leakage_net"] is True


def test_late_leakage_sensitivity_payload_stays_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    dense_rows = [
        {
            "quarter": "2023-Q3",
            "alive_on_art": 100.0,
            "alive_on_art_tier": "exact_observed",
            "virally_suppressed": 60.0,
            "virally_suppressed_tier": "exact_observed",
            "deaths_reported_period": 2.0,
            "deaths_reported_period_tier": "bridge_observed",
        },
        {
            "quarter": "2023-Q4",
            "alive_on_art": 110.0,
            "alive_on_art_tier": "exact_observed",
            "virally_suppressed": 62.0,
            "virally_suppressed_tier": "exact_observed",
            "deaths_reported_period": 3.0,
            "deaths_reported_period_tier": "bridge_observed",
        },
    ]
    monkeypatch.setattr(suite, "_build_dense_contract_payload", lambda archive_run_id: {"rows": list(dense_rows)})

    payload = suite._build_late_leakage_sensitivity_payload("archive-run")

    assert payload["decision"]["status"] == "late_window_sensitivity_only"
    assert payload["proxy_rows"][0]["shortfall_vs_share_carry"] >= 0.0


def test_suppression_honesty_flag_recognizes_unclaimed_mode() -> None:
    dataset = type("Dataset", (), {"holdout_rows": [{"quarter": "2025-Q1", "virally_suppressed": None}]})()

    flag = suite._suppression_honesty_flag(
        dataset,
        {"A_to_V": {"support_count": 0, "model_kind": "direct_observation_repair_suppression_unclaimed"}},
        allowed_tiers={"exact_observed"},
    )

    assert flag == "unsupported_or_unclaimed"


def test_supported_stock_flow_residual_bounds_uses_supported_rows_only() -> None:
    rows = [
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 100.0,
            "diagnosed_plhiv_tier": "exact_observed",
            "new_diagnosed_cases_period": 8.0,
            "new_diagnosed_cases_period_tier": "exact_observed",
        },
        {
            "quarter": "2021-Q2",
            "diagnosed_plhiv": 111.0,
            "diagnosed_plhiv_tier": "bridge_observed",
            "new_diagnosed_cases_period": 10.0,
            "new_diagnosed_cases_period_tier": "bridge_observed",
        },
        {
            "quarter": "2021-Q3",
            "diagnosed_plhiv": 130.0,
            "diagnosed_plhiv_tier": "rule_based_extrapolated",
            "new_diagnosed_cases_period": 30.0,
            "new_diagnosed_cases_period_tier": "rule_based_extrapolated",
        },
    ]

    lower, upper, count = suite._supported_stock_flow_residual_bounds(rows)

    assert count == 1
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_raw_endpoint_audit_reports_metric_and_tier_breakdown() -> None:
    predictions = [
        {"quarter": "2024-Q1", "diagnosed_plhiv": 100.0, "alive_on_art": 60.0, "new_diagnosed_cases_period": 10.0},
        {"quarter": "2024-Q2", "diagnosed_plhiv": 110.0, "alive_on_art": 66.0, "new_diagnosed_cases_period": 8.0},
    ]
    targets = [
        {
            "quarter": "2024-Q1",
            "diagnosed_plhiv": 102.0,
            "diagnosed_plhiv_tier": "exact_observed",
            "alive_on_art": 62.0,
            "alive_on_art_tier": "bridge_observed",
            "new_diagnosed_cases_period": 9.0,
            "new_diagnosed_cases_period_tier": "bridge_observed",
        },
        {
            "quarter": "2024-Q2",
            "diagnosed_plhiv": 109.0,
            "diagnosed_plhiv_tier": "bridge_observed",
            "alive_on_art": 65.0,
            "alive_on_art_tier": "bridge_observed",
            "new_diagnosed_cases_period": 7.0,
            "new_diagnosed_cases_period_tier": "exact_observed",
        },
    ]

    audit = suite._raw_endpoint_audit(
        predictions,
        targets,
        {"diagnosed_plhiv": 110.0, "alive_on_art": 66.0, "new_diagnosed_cases_period": 10.0},
        allowed_tiers={"exact_observed", "bridge_observed"},
        eps=1e-6,
    )

    assert audit["by_metric"]["diagnosed_plhiv"]["count"] == 2
    assert audit["by_metric_tier"]["diagnosed_plhiv"]["exact_observed"]["count"] == 1
    assert audit["by_metric_tier"]["diagnosed_plhiv"]["bridge_observed"]["count"] == 1


def test_build_endpoint_tier_audit_payload_tracks_new_r10_variants() -> None:
    payload = suite._build_endpoint_tier_audit_payload(
        [
            {
                "experiment_id": "EXP-R10-EXACT-CHAMPION",
                "status": "executed",
                "family": "repair",
                "decision": "keep",
                "quarterly_summary": {
                    "candidate_mean_mae": 0.1,
                    "carry_forward_mean_mae": 0.2,
                    "endpoint_audit_summary": {"suppression_honesty_flags": {"unsupported_level_carry": 3}},
                },
            },
            {
                "experiment_id": "EXP-R10-DENSE-CHAMPION",
                "status": "executed",
                "family": "repair",
                "decision": "keep",
                "quarterly_summary": {
                    "candidate_mean_mae": 0.09,
                    "carry_forward_mean_mae": 0.14,
                    "endpoint_audit_summary": {"suppression_honesty_flags": {"unsupported_level_carry": 3}},
                },
            },
            {
                "experiment_id": "EXP-R1",
                "status": "executed",
                "family": "repair",
                "decision": "revert",
                "quarterly_summary": {
                    "candidate_mean_mae": 0.3,
                    "carry_forward_mean_mae": 0.2,
                    "endpoint_audit_summary": {"suppression_honesty_flags": {"unsupported_or_unclaimed": 3}},
                },
            },
            {
                "experiment_id": "EXP-R10-M1",
                "status": "executed",
                "family": "repair",
                "decision": "revert",
                "quarterly_summary": {"candidate_mean_mae": 0.11, "carry_forward_mean_mae": 0.2, "endpoint_audit_summary": {}},
            },
            {
                "experiment_id": "EXP-R10-M2",
                "status": "executed",
                "family": "repair",
                "decision": "revert",
                "quarterly_summary": {"candidate_mean_mae": 0.12, "carry_forward_mean_mae": 0.2, "endpoint_audit_summary": {}},
            },
            {
                "experiment_id": "EXP-R10-DENSE-M1-F1-H1",
                "status": "executed",
                "family": "repair",
                "decision": "keep",
                "quarterly_summary": {"candidate_mean_mae": 0.08, "carry_forward_mean_mae": 0.14, "endpoint_audit_summary": {}},
            },
            {
                "experiment_id": "EXP-R11",
                "status": "executed",
                "family": "repair",
                "decision": "revert",
                "quarterly_summary": {"candidate_mean_mae": 0.15, "carry_forward_mean_mae": 0.2, "endpoint_audit_summary": {}},
            },
        ],
        quarterly_contract="exact_only",
    )

    tracked_ids = [row["experiment_id"] for row in payload["tracked_results"]]
    assert "EXP-R10-M1" in tracked_ids
    assert "EXP-R10-M2" in tracked_ids
    assert "EXP-R10-DENSE-M1-F1-H1" in tracked_ids
    assert "EXP-R11" in tracked_ids


def test_simulate_art_stock_repair_hits_art_target_and_uses_carry_share() -> None:
    result = suite._simulate_art_stock_repair(
        {"U": 80.0, "D": 20.0, "A": 30.0, "V": 10.0, "L": 5.0},
        [{"quarter": "2021-Q1"}],
        {"2021-Q1": {"U_to_D": 0.05, "A_to_L": 0.02, "L_to_A": 0.10}},
        {"2021-Q1": 50.0},
        suppression_share_carry=0.25,
        eps=1e-6,
    )

    prediction = result["prediction_rows"][0]
    trajectory = result["trajectory_rows"][0]
    assert prediction["alive_on_art"] == pytest.approx(50.0)
    assert prediction["virally_suppressed"] == pytest.approx(12.5)
    assert 0.0 <= trajectory["hazards"]["D_to_A"] <= 1.0
    assert 0.0 <= trajectory["hazards"]["A_to_V"] <= 1.0


def test_care_support_mode_prefers_strict_only_with_enough_support() -> None:
    assert suite._care_support_mode(support_count=4, total_count=10) == "strict_support"
    assert suite._care_support_mode(support_count=3, total_count=10) == "repair"
    assert suite._care_support_mode(support_count=4, total_count=20) == "repair"
    assert suite._care_support_mode(support_count=6, total_count=10, min_count=6, min_fraction=0.5) == "strict_support"
    assert suite._care_support_mode(support_count=5, total_count=10, min_count=6, min_fraction=0.5) == "repair"


def test_d_to_a_blend_weight_is_zero_without_support_and_one_for_strong_support() -> None:
    assert suite._d_to_a_blend_weight(
        {"support_count": 0, "train_quarters": ["2020-Q1"], "train_supported": [False]},
        total_count=8,
        min_count=4,
        min_fraction=0.35,
    ) == 0.0
    assert suite._d_to_a_blend_weight(
        {"support_count": 5, "train_quarters": ["2020-Q1", "2020-Q2"], "train_supported": [True, True]},
        total_count=8,
        min_count=4,
        min_fraction=0.35,
    ) == 1.0
    moderate = suite._d_to_a_blend_weight(
        {
            "support_count": 2,
            "train_quarters": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"],
            "train_supported": [False, True, False, True],
        },
        total_count=8,
        min_count=4,
        min_fraction=0.35,
    )
    assert 0.0 < moderate < 1.0


def test_recent_supported_block_indices_finds_trailing_block_only() -> None:
    rows = [
        {"quarter": "2019-Q1", "support_flags": {"D_to_A": False}},
        {"quarter": "2019-Q2", "support_flags": {"D_to_A": True}},
        {"quarter": "2019-Q3", "support_flags": {"D_to_A": True}},
        {"quarter": "2019-Q4", "support_flags": {"D_to_A": False}},
        {"quarter": "2020-Q1", "support_flags": {"D_to_A": True}},
        {"quarter": "2020-Q2", "support_flags": {"D_to_A": True}},
        {"quarter": "2020-Q3", "support_flags": {"D_to_A": True}},
    ]
    block = suite._recent_supported_block_indices(rows, "D_to_A", min_supported=3, max_inner_gap=2, max_tail_gap=1)
    assert block == [4, 5, 6]


def test_fit_supported_delta_series_returns_forecast_map() -> None:
    train_rows = [
        {"quarter": "2018-Q1", "alive_on_art": 100.0, "alive_on_art_tier": "exact_observed"},
        {"quarter": "2018-Q2", "alive_on_art": 110.0, "alive_on_art_tier": "exact_observed"},
        {"quarter": "2018-Q4", "alive_on_art": 130.0, "alive_on_art_tier": "bridge_observed"},
        {"quarter": "2019-Q2", "alive_on_art": 150.0, "alive_on_art_tier": "exact_observed"},
    ]
    result = suite._fit_supported_delta_series(
        train_rows,
        ["2019-Q3", "2019-Q4"],
        metric_name="alive_on_art",
        recent_blend_weight=0.75,
    )
    forecast = dict(result["forecast_map"])
    assert result["diagnostics"]["support_count"] == 3
    assert "2019-Q3" in forecast
    assert "2019-Q4" in forecast


def test_crossfit_supported_metric_correction_handles_short_history() -> None:
    spec = next(spec for spec in suite.build_experiment_suite_specs() if spec.experiment_id == "EXP-R10-DENSE-M1-C1-H1")
    correction, diagnostics = suite._crossfit_supported_metric_correction(
        [
            {
                "quarter": "2024-Q4",
                "diagnosed_plhiv": 100.0,
                "diagnosed_plhiv_tier": "exact_observed",
                "alive_on_art": 60.0,
                "alive_on_art_tier": "exact_observed",
                "new_diagnosed_cases_period": 10.0,
                "new_diagnosed_cases_period_tier": "exact_observed",
            }
        ],
        spec,
        metric_name="diagnosed_plhiv",
        diagnosed_weight=1.0,
        art_weight=1.0,
        flow_weight=1.0,
        diagnosed_series_model="level",
        art_series_model="delta",
        flow_series_model="level",
        diagnosed_recent_blend_weight=1.0,
        art_recent_blend_weight=1.0,
        flow_recent_blend_weight=1.0,
        suppression_carry_weight=1.0,
        suppression_fallback_mode="unclaimed",
        use_joint_consistency=True,
        use_bias_correction=False,
        diagnosed_bias_weight=1.0,
        art_bias_weight=1.0,
        flow_bias_weight=1.0,
        flow_consistency_weight=0.0,
        min_train_years=3,
        min_points=5,
        recent_pool=12,
    )
    assert correction == 0.0
    assert diagnostics["status"] == "insufficient_train_history"
