from __future__ import annotations

from pathlib import Path

import pytest

from phase3_dynamic.data import build_blocked_time_dataset
from phase3_dynamic.incidence_module_search import (
    _apply_determinant_bundle_keep_rules,
    _fit_annual_measurement_readout_head,
    _fit_weak_annual_measurement_calibrator,
    _quarter_rate_to_interval_incidence,
    _metric_is_kp_pressure,
    _r10_source_report_annual_incidence_error,
    _train_annual_stock_balance_incidence,
    fit_incidence_module_head,
    s_eff_incidence_to_u,
)


def _synthetic_incidence_rows() -> list[dict[str, float]]:
    state_u = 1000.0
    diagnosed = 300.0
    rows: list[dict[str, float]] = []
    for idx, year in enumerate(range(2017, 2025)):
        quarter = f"{year}-Q1"
        population = 1_000_000.0 + 5000.0 * idx
        estimated = state_u + diagnosed
        rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": 0.7 * diagnosed,
                "tested_for_viral_load": 0.55 * 0.7 * diagnosed,
                "virally_suppressed": 0.45 * 0.7 * diagnosed,
                "estimated_plhiv": estimated,
                "population_total": population,
                "new_diagnosed_cases_period": 0.05 * state_u if idx else None,
            }
        )
        incidence = 0.00002 * max(population - estimated, 0.0)
        diagnosed_flow = 0.05 * state_u
        state_u = max(state_u + incidence - diagnosed_flow, 0.0)
        diagnosed = max(diagnosed + diagnosed_flow, 0.0)
    return rows


def _synthetic_quarterly_incidence_rows() -> list[dict[str, float]]:
    state_u = 1000.0
    diagnosed = 300.0
    rows: list[dict[str, float]] = []
    for idx in range(8 * 4):
        year = 2017 + idx // 4
        quarter = idx % 4 + 1
        population = 1_000_000.0 + 1000.0 * idx
        diagnosis_flow = 8.0 + float(idx % 5)
        incidence = 20.0 + 1.5 * float(idx)
        state_u = max(state_u + incidence - diagnosis_flow, 0.0)
        diagnosed = max(diagnosed + diagnosis_flow, 0.0)
        estimated = state_u + diagnosed
        rows.append(
            {
                "quarter": f"{year}-Q{quarter}",
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": 0.7 * diagnosed,
                "tested_for_viral_load": 0.55 * 0.7 * diagnosed,
                "virally_suppressed": 0.45 * 0.7 * diagnosed,
                "estimated_plhiv": estimated,
                "population_total": population,
                "new_diagnosed_cases_period": diagnosis_flow,
            }
        )
    return rows


def test_s_eff_incidence_to_u_uses_explicit_population_denominator() -> None:
    dataset = build_blocked_time_dataset(_synthetic_incidence_rows(), [2024])
    quarter = str(dataset.holdout_rows[0]["quarter"])
    incidence_map, trajectory_rows = s_eff_incidence_to_u(
        dataset=dataset,
        hazard_map={quarter: 0.00001},
    )

    row = trajectory_rows[0]
    expected = _quarter_rate_to_interval_incidence(
        0.00001,
        float(row["susceptible_effective"]),
        int(row["interval_quarters"]),
    )
    assert incidence_map[quarter] == pytest.approx(expected)
    assert row["state_values_after_incidence"]["U"] >= dataset.train_state_rows[-1]["state_values"]["U"]
    assert row["contract"].startswith("standalone interval-aware incidence module")
    assert row["hazard_semantics"].startswith("continuous per-quarter")


def test_incidence_head_is_not_diagnosis_flow_readout(tmp_path: Path) -> None:
    dataset = build_blocked_time_dataset(_synthetic_incidence_rows(), [2024])
    head = fit_incidence_module_head(
        family="s_eff_hazard_ar",
        dataset=dataset,
        epigraph_root=tmp_path,
        source_run_id="missing",
        baseline_source_run_id="missing",
    )

    assert head is not None
    assert "lag_log_s_eff_hazard" in head.feature_names
    assert all("diagnosis_flow" not in name for name in head.feature_names)
    assert head.training_objective == "train_stock_balance_s_eff_incidence_hazard_with_annual_incidence_validation_only"


def test_r10_mechanistic_head_uses_hazard_delta_features_not_diagnosis_flow(tmp_path: Path) -> None:
    dataset = build_blocked_time_dataset(_synthetic_quarterly_incidence_rows(), [2024])
    head = fit_incidence_module_head(
        family="s_eff_hazard_r10_mechanistic",
        dataset=dataset,
        epigraph_root=tmp_path,
        source_run_id="missing",
        baseline_source_run_id="missing",
    )

    assert head is not None
    assert "lag_log_s_eff_rate_delta" in head.feature_names
    assert "lag_log_s_eff_rate_curvature" in head.feature_names
    assert all("diagnosis_flow" not in name for name in head.feature_names)


def test_determinant_bundle_keep_rule_requires_mechanistic_counterpart_improvement() -> None:
    summaries = _apply_determinant_bundle_keep_rules(
        [
            {
                "family": "s_eff_hazard_r10_mechanistic",
                "candidate_norm_mae": 0.5,
                "blockers": [],
                "promotion_eligible": True,
            },
            {
                "family": "s_eff_hazard_r10_mechanistic_kp_pressure",
                "candidate_norm_mae": 0.6,
                "blockers": [],
                "promotion_eligible": True,
            },
        ]
    )

    bundle = next(row for row in summaries if row["family"] == "s_eff_hazard_r10_mechanistic_kp_pressure")
    assert bundle["determinant_bundle_counterpart"] == "s_eff_hazard_r10_mechanistic"
    assert "determinant_bundle_not_better_than_mechanistic_counterpart" in bundle["blockers"]
    assert not bundle["promotion_eligible"]


def test_weak_annual_measurement_calibrator_uses_train_years_only() -> None:
    dataset = build_blocked_time_dataset(_synthetic_quarterly_incidence_rows(), [2024])
    raw_annual = _train_annual_stock_balance_incidence(dataset)
    targets = {
        year: {
            "quarter": f"{year}-Q4",
            "annual_new_infections": raw_value * 1.5,
            "metric_provenance": {"annual_new_infections": {"allowed_use": "validation_only"}},
        }
        for year, raw_value in raw_annual.items()
    }
    targets[2024] = {
        "quarter": "2024-Q4",
        "annual_new_infections": 1_000_000_000.0,
        "metric_provenance": {"annual_new_infections": {"allowed_use": "validation_only"}},
    }

    calibrator = _fit_weak_annual_measurement_calibrator(
        dataset=dataset,
        targets=targets,
        train_end_year=2023,
    )

    assert calibrator is not None
    assert calibrator.train_annual_count >= 3
    assert max(calibrator.train_years) <= 2023
    assert 2024 not in calibrator.train_years
    assert calibrator.training_objective.startswith("weak_measurement")


def test_annual_measurement_readout_uses_pre_holdout_measurements_only() -> None:
    targets = {
        2020: {"annual_new_infections": 100.0},
        2021: {"annual_new_infections": 120.0},
        2022: {"annual_new_infections": 140.0},
        2023: {"annual_new_infections": 9999.0},
    }

    head = _fit_annual_measurement_readout_head(targets=targets, train_end_year=2022)

    assert head is not None
    assert head.train_years == [2020, 2021, 2022]
    assert head.readout_family == "r10_log_delta_unclipped_annual_measurement_readout"
    assert head.forecast_log_floor <= min(head.train_log_values)
    assert head.forecast_log_ceiling >= max(head.train_log_values)


def test_kp_pressure_filter_excludes_incidence_validation_metrics() -> None:
    assert _metric_is_kp_pressure("men_who_have_sex_with_men_hiv_prevalence_among_men_who_have_sex_with_men_population_total")
    assert not _metric_is_kp_pressure("new_hiv_infections_hiv_incidence_as_new_infections_per_1000_uninfected_population_all_ages_population_all")


def test_r10_reference_prefers_expanded_harp_purged_dense_merged(tmp_path: Path) -> None:
    source_report = tmp_path / "r10_source_report.json"
    source_report.write_text(
        """
{
  "contracts": [
    {
      "contract": "exact_only",
      "baseline_current_champion": {
        "contract": "exact_only",
        "archive_variant": "baseline",
        "experiment_id": "EXP-R10-EXACT-CHAMPION",
        "annual_mean_incidence_error": 0.03
      },
      "merged_current_champion": {
        "contract": "exact_only",
        "archive_variant": "merged",
        "experiment_id": "EXP-R10-EXACT-CHAMPION",
        "annual_mean_incidence_error": 0.06
      }
    },
    {
      "contract": "purged_dense",
      "baseline_current_champion": {
        "contract": "purged_dense",
        "archive_variant": "baseline",
        "experiment_id": "EXP-R10-DENSE-CHAMPION",
        "annual_mean_incidence_error": 0.04
      },
      "merged_current_champion": {
        "contract": "purged_dense",
        "archive_variant": "merged",
        "experiment_id": "EXP-R10-DENSE-CHAMPION",
        "annual_mean_incidence_error": 0.08
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    assert _r10_source_report_annual_incidence_error(source_report) == pytest.approx(0.08)
