from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as batch
from epigraph_ph.runtime import write_json


def test_harp_month_axis_builds_contiguous_range(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    harp_dir = run_dir / "harp_archive"
    harp_dir.mkdir(parents=True)
    write_json(
        harp_dir / "observed_program_panel.json",
        {"rows": [{"time": "2010-01"}, {"time": "2010-03"}]},
    )
    write_json(
        harp_dir / "diagnosis_flow_points.json",
        {"points": [{"effective_month": "2010-06"}]},
    )
    write_json(
        harp_dir / "harp_program_points.json",
        {"points": [{"month": "2010-04"}]},
    )

    axis = batch._harp_month_axis(run_dir, start_month="2010-01")

    assert axis == ["2010-01", "2010-02", "2010-03", "2010-04", "2010-05", "2010-06"]


def test_filtered_monthly_rows_excludes_annual_and_non_national() -> None:
    rows = [
        {"time": "2010", "geo": "Philippines", "region": "national", "geo_resolution": "national"},
        {"time": "2010-01", "geo": "Philippines", "region": "national", "geo_resolution": "national"},
        {"time": "2010-02", "geo": "Cebu", "region": "region_vii", "geo_resolution": "province"},
        {"time": "2010-03", "geo": "Philippines", "region": "national", "geo_resolution": "national"},
    ]

    filtered = batch._filtered_monthly_rows(rows, start_month="2010-01", end_month="2010-03")

    assert [row["time"] for row in filtered] == ["2010-01", "2010-03"]
    assert all(row["province"] == "Philippines" for row in filtered)
    assert all(row["geo_resolution"] == "national" for row in filtered)


def test_axis_gap_summary_flags_irregular_legacy_axis() -> None:
    summary = batch._axis_gap_summary(["2010-01", "2010-02", "2010-06"])

    assert summary["irregular"] is True
    assert summary["gap_count"] == 1
    assert summary["largest_gap_months"] == 4


def test_apply_structural_canonical_exclusions_filters_requested_rows() -> None:
    rows = [
        {"canonical_name": "diagnosed_plhiv", "time": "2010-01"},
        {"canonical_name": "new_diagnosed_cases_period", "time": "2010-01"},
        {"canonical_name": "alive_on_art", "time": "2010-01"},
        {"canonical_name": "tested_for_viral_load", "time": "2010-01"},
    ]

    kept, summary = batch._apply_structural_canonical_exclusions(
        rows,
        excluded_canonicals=("diagnosed_plhiv", "new_diagnosed_cases_period", "alive_on_art"),
    )

    assert [row["canonical_name"] for row in kept] == ["tested_for_viral_load"]
    assert summary["full_row_count"] == 4
    assert summary["structural_row_count"] == 1
    assert summary["excluded_row_count"] == 3
    assert summary["excluded_counts_by_canonical"] == {
        "alive_on_art": 1,
        "diagnosed_plhiv": 1,
        "new_diagnosed_cases_period": 1,
    }


def test_canonical_axis_keeps_new_harp_canonicals() -> None:
    axis = batch._canonical_axis(
        {"canonical_name": ["clinics_per_capita", "mobility_network_mixing"]},
        [
            {"canonical_name": "clinics_per_capita"},
            {"canonical_name": "diagnosed_plhiv"},
            {"canonical_name": "alive_on_art"},
        ],
    )

    assert axis == ["clinics_per_capita", "alive_on_art", "diagnosed_plhiv"]


def test_harp_phase1_rows_include_direct_heads_and_flow(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    harp_dir = run_dir / "harp_archive"
    harp_dir.mkdir(parents=True)
    write_json(
        harp_dir / "observed_program_panel.json",
        {
            "rows": [
                {
                    "metric_name": "diagnosed_plhiv",
                    "time": "2010-01",
                    "value": 4367,
                    "unit": "count_people",
                    "source_id": "obs_1",
                    "source_label": "Obs 1",
                    "source_url": "https://example.com/obs_1",
                    "temporal_precision": "monthly_snapshot",
                    "evidence_confidence": 0.92,
                    "source_quality_tier": "official_doh_archive",
                },
                {
                    "metric_name": "alive_on_art",
                    "time": "2010-01",
                    "value": 2087,
                    "unit": "count_people",
                    "source_id": "obs_2",
                    "source_label": "Obs 2",
                    "source_url": "https://example.com/obs_2",
                    "temporal_precision": "monthly_snapshot",
                    "evidence_confidence": 0.92,
                    "source_quality_tier": "official_doh_archive",
                },
            ]
        },
    )
    write_json(
        harp_dir / "diagnosis_flow_points.json",
        {
            "points": [
                {
                    "diagnosed_count": 143,
                    "effective_month": "2010-01",
                    "source_id": "flow_1",
                    "source_label": "Flow 1",
                    "source_url": "https://example.com/flow_1",
                    "temporal_precision": "monthly_snapshot",
                    "evidence_confidence": 0.92,
                }
            ]
        },
    )
    write_json(
        harp_dir / "harp_program_points.json",
        {
            "points": [
                {
                    "diagnosed": 4367,
                    "on_art": 2087,
                    "viral_load_tested": 1000,
                    "suppressed": 800,
                    "effective_month": "2010-01",
                    "month": "2010-01",
                    "label": "Program 1",
                    "source_url": "https://example.com/program_1",
                    "temporal_precision": "monthly_snapshot",
                }
            ]
        },
    )

    rows = batch._harp_monthly_phase1_rows(
        source_run_dir=run_dir,
        plugin_id="hiv",
        start_month="2010-01",
        end_month="2010-12",
    )

    canonicals = {row["canonical_name"] for row in rows}
    assert "diagnosed_plhiv" in canonicals
    assert "alive_on_art" in canonicals
    assert "new_diagnosed_cases_period" in canonicals
    assert "art_uptake_rate" in canonicals
    direct_row = next(row for row in rows if row["canonical_name"] == "diagnosed_plhiv")
    assert direct_row["source_bank"] == "phase1_harp_observed_program_panel"
    assert direct_row["measurement_role"] == "direct_indicator"
    assert direct_row["time"] == "2010-01"


def test_harp_phase1_rows_include_historical_panel_anchors(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    harp_dir = run_dir / "harp_archive"
    harp_dir.mkdir(parents=True)
    write_json(harp_dir / "observed_program_panel.json", {"rows": []})
    write_json(harp_dir / "diagnosis_flow_points.json", {"points": []})
    write_json(harp_dir / "harp_program_points.json", {"points": []})
    write_json(
        harp_dir / "historical_harp_panel.json",
        {
            "rows": [
                {
                    "time": "2010-01",
                    "year": 2010,
                    "alive_on_art": 1000,
                    "estimated_plhiv": 19000,
                }
            ]
        },
    )

    rows = batch._harp_monthly_phase1_rows(
        source_run_dir=run_dir,
        plugin_id="hiv",
        start_month="2010-01",
        end_month="2010-12",
    )

    keyed = {(row["canonical_name"], row["time"], row["source_bank"]): row for row in rows}
    assert ("alive_on_art", "2010-01", "phase1_harp_historical_panel") in keyed
    assert ("estimated_plhiv", "2010-01", "phase1_harp_historical_panel") in keyed


def test_harp_rows_from_historical_metrics_normalizes_upstream_counts_per_100k(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    harp_dir = run_dir / "harp_archive"
    harp_dir.mkdir(parents=True)
    write_json(
        harp_dir / "historical_metric_rows.json",
        [
            {
                "metric_name": "population_total",
                "time": "2023-12",
                "year": 2023,
                "value": 1000000.0,
                "source_id": "population_total",
            }
        ],
    )
    write_json(
        harp_dir / "multinational_hiv_metric_rows.json",
        [
            {
                "metric_name": "annual_hiv_tests_volume",
                "time": "2023-01",
                "year": 2023,
                "value": 25000.0,
                "source_id": "tests_volume",
                "source_label": "Tests volume",
                "source_quality_tier": "external_multinational_hiv_panel",
            },
            {
                "metric_name": "prep_people_receiving",
                "time": "2023-01",
                "year": 2023,
                "value": 1500.0,
                "source_id": "prep_people",
                "source_label": "PrEP people",
                "source_quality_tier": "external_multinational_hiv_panel",
            },
        ],
    )

    rows = batch._harp_rows_from_historical_metrics(
        source_run_dir=run_dir,
        plugin_id="hiv",
        start_month="2023-01",
        end_month="2023-12",
    )

    keyed = {(row["canonical_name"], row["time"]): row for row in rows}
    assert keyed[("annual_hiv_tests_volume_per_100k", "2023-01")]["raw_numeric_value"] == 2500.0
    assert keyed[("prep_people_receiving_per_100k", "2023-01")]["raw_numeric_value"] == 150.0
    assert keyed[("annual_hiv_tests_volume_per_100k", "2023-01")]["measurement_role"] == "direct_indicator"


def test_harp_rows_from_historical_metrics_passes_through_annual_testing_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "source_run"
    harp_dir = run_dir / "harp_archive"
    harp_dir.mkdir(parents=True)
    write_json(harp_dir / "historical_metric_rows.json", [])
    write_json(
        harp_dir / "multinational_hiv_metric_rows.json",
        [
            {
                "metric_name": "hiv_test_positivity_percent",
                "time": "2024-01",
                "year": 2024,
                "value": 1.7,
                "source_id": "positivity",
                "source_label": "Positivity",
                "unit": "percent",
                "source_quality_tier": "external_multinational_hiv_panel",
            },
            {
                "metric_name": "late_hiv_diagnosis_percent",
                "time": "2020-01",
                "year": 2020,
                "value": 42.0,
                "source_id": "late_dx",
                "source_label": "Late diagnosis",
                "unit": "percent",
                "source_quality_tier": "external_multinational_hiv_panel",
            },
        ],
    )

    rows = batch._harp_rows_from_historical_metrics(
        source_run_dir=run_dir,
        plugin_id="hiv",
        start_month="2020-01",
        end_month="2024-12",
    )

    keyed = {(row["canonical_name"], row["time"]): row for row in rows}
    assert keyed[("hiv_test_positivity_percent", "2024-01")]["raw_numeric_value"] == 1.7
    assert keyed[("late_hiv_diagnosis_percent", "2020-01")]["raw_numeric_value"] == 42.0
    assert keyed[("hiv_test_positivity_percent", "2024-01")]["time_resolution"] == "annual"


def test_copy_harp_archive_with_coverage_merge_preserves_monthly_surfaces(tmp_path: Path) -> None:
    source_run_dir = tmp_path / "source_run"
    source_harp_dir = source_run_dir / "harp_archive"
    source_harp_dir.mkdir(parents=True)
    coverage_run_dir = tmp_path / "coverage_run"
    coverage_harp_dir = coverage_run_dir / "harp_archive"
    coverage_harp_dir.mkdir(parents=True)
    target_run_dir = tmp_path / "target_run"

    write_json(
        source_harp_dir / "observed_program_panel.json",
        {"rows": [{"metric_name": "alive_on_art", "time": "2024-01", "value": 100.0}]},
    )
    write_json(
        source_harp_dir / "diagnosis_flow_points.json",
        {"points": [{"effective_month": "2024-01", "diagnosed_count": 12.0}]},
    )
    write_json(
        source_harp_dir / "historical_metric_rows_harp_only.json",
        [{"metric_name": "estimated_plhiv", "time": "2024-01", "value": 1000.0, "source_id": "base_est"}],
    )
    write_json(
        source_harp_dir / "historical_metric_rows.json",
        [{"metric_name": "estimated_plhiv", "time": "2024-01", "value": 1000.0, "source_id": "base_est"}],
    )
    write_json(source_harp_dir / "multinational_hiv_metric_rows.json", [])

    write_json(
        coverage_harp_dir / "multinational_hiv_metric_rows.json",
        [
            {
                "metric_name": "annual_hiv_tests_volume",
                "time": "2024-01",
                "value": 25000.0,
                "source_id": "coverage_tests",
            }
        ],
    )
    write_json(
        coverage_harp_dir / "multinational_hiv_series_inventory.json",
        [{"source_id": "coverage_tests", "metric_name": "annual_hiv_tests_volume"}],
    )

    summary = batch._copy_harp_archive(
        source_run_dir,
        target_run_dir,
        coverage_run_dir=coverage_run_dir,
    )

    merged_harp_dir = target_run_dir / "harp_archive"
    observed_panel = (merged_harp_dir / "observed_program_panel.json")
    diagnosis_flow = (merged_harp_dir / "diagnosis_flow_points.json")
    historical_harp_only = (merged_harp_dir / "historical_metric_rows_harp_only.json")
    historical_metric_rows = (merged_harp_dir / "historical_metric_rows.json")
    multinational_rows = (merged_harp_dir / "multinational_hiv_metric_rows.json")

    assert summary["mode"] == "baseline_plus_coverage_multinational_merge"
    assert summary["preserved_observed_program_panel"] is True
    assert summary["preserved_diagnosis_flow_points"] is True
    assert batch.read_json(observed_panel, default={})["rows"][0]["metric_name"] == "alive_on_art"
    assert batch.read_json(diagnosis_flow, default={})["points"][0]["diagnosed_count"] == 12.0
    merged_harp_only_rows = batch.read_json(historical_harp_only, default=[])
    merged_historical_rows = batch.read_json(historical_metric_rows, default=[])
    merged_multinational_rows = batch.read_json(multinational_rows, default=[])
    assert {row["metric_name"] for row in merged_harp_only_rows} == {"estimated_plhiv", "annual_hiv_tests_volume"}
    assert {row["metric_name"] for row in merged_historical_rows} == {"estimated_plhiv", "annual_hiv_tests_volume"}
    assert merged_multinational_rows[0]["metric_name"] == "annual_hiv_tests_volume"
