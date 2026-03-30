from __future__ import annotations

from epigraph_ph.harp_archive.pipeline import (
    YEAR_RANGE,
    _backtest_assessment,
    _build_frozen_backtest_artifacts,
    _deduplicate_metric_rows,
    _extract_core_team_cascade,
    _extract_core_team_series,
    _extract_core_team_subnational_kp,
    _extract_generic_harp_snapshot,
    _read_tabular_seed_rows,
    _extract_surveillance_kp_profile,
    run_harp_archive_build,
)
from epigraph_ph.runtime import ROOT_DIR, read_json


def test_extract_core_team_series_from_sample_page() -> None:
    years = " ".join(str(year) for year in YEAR_RANGE)
    infections = " ".join(str(value) for value in range(5000, 5000 + len(YEAR_RANGE)))
    plhiv = " ".join(str(value) for value in range(120000, 120000 + len(YEAR_RANGE)))
    deaths = " ".join(str(value) for value in range(500, 500 + len(YEAR_RANGE)))
    sample = f"{years} {infections} {plhiv} {deaths}"

    rows = _extract_core_team_series(sample, "core_team_2025", "Core team", 12)

    assert rows
    metrics = {row["metric_name"] for row in rows}
    assert {"annual_new_infections", "estimated_plhiv", "annual_aids_deaths"} <= metrics
    assert min(int(row["year"]) for row in rows) == 2010
    assert max(int(row["year"]) for row in rows) == 2025


def test_extract_core_team_cascade_snapshot() -> None:
    sample = (
        "Philippine HIV Care Cascade as of December 2024 "
        "216,900 135,026 90,854 43,534 41,164"
    )
    rows = _extract_core_team_cascade(sample, "core_team_2025", "Core team", 13)

    assert len(rows) == 5
    assert {row["measurement_class"] for row in rows} >= {"program_observed_harp", "model_estimate"}
    metrics = {row["metric_name"]: row["value"] for row in rows}
    assert metrics["diagnosed_plhiv"] == 135026.0
    assert metrics["virally_suppressed"] == 41164.0


def test_extract_surveillance_national_kp_profile() -> None:
    sample = (
        "PLHIV by key population, 2021 (N=215,400) "
        "Males having sex with Males (MSM) 78% "
        "Female- 6% "
        "Person who Inject Drugs (PWID)- 2% "
        "Other males- 13%"
    )
    anchor = _extract_surveillance_kp_profile(sample, "surveillance", "Surveillance", 3)

    assert anchor is not None
    assert anchor["group_kind"] == "kp_distribution"
    assert abs(sum(anchor["mapped_distribution"].values()) - 1.0) < 1e-6
    assert anchor["mapped_distribution"]["msm"] > 0.7


def test_extract_core_team_subnational_kp_anchors() -> None:
    sample = (
        "Subnational Model Prevention Coverage MSM & TGW Estimates (2025) "
        "NCR 27% 324,600 "
        "Cebu City 29% 19,000 "
        "Cebu Province 24% 51,100 "
        "Angeles City 39% 10,000 "
        "National 29% 1,234,500"
    )
    rows = _extract_core_team_subnational_kp(sample, "core_team_2025", "Core team", 18)

    assert len(rows) >= 5
    assert any(row["geo"] == "National Capital Region" or row["region"] == "ncr" for row in rows)
    assert any(row["geo"] == "Philippines" for row in rows)


def test_backtest_assessment_admits_gap_aware_panel() -> None:
    metric_rows = [
        {"metric_name": "estimated_plhiv", "year": year, "measurement_class": "model_estimate"}
        for year in range(2010, 2025)
    ] + [
        {"metric_name": "diagnosed_plhiv", "year": 2024, "measurement_class": "program_observed_harp"},
        {"metric_name": "alive_on_art", "year": 2024, "measurement_class": "program_observed_harp"},
    ]
    assessment = _backtest_assessment(metric_rows)

    assert assessment["backtest_ready"] is False
    assert "historical_harp_program_series_incomplete" in assessment["blocking_reasons"]


def test_extract_generic_harp_snapshot() -> None:
    sample = (
        "HIV Care Cascade and AIDS and ART Registry for 2020 "
        "estimated PLHIV 115,000 diagnosed PLHIV 78,291 alive on ART 47,977 "
        "viral load tested 8,155 virally suppressed 7,666"
    )
    rows = _extract_generic_harp_snapshot(sample, "manual_2020", "Manual 2020", 4)

    assert len(rows) >= 4
    by_metric = {row["metric_name"]: row["value"] for row in rows}
    assert by_metric["diagnosed_plhiv"] == 78291.0
    assert by_metric["alive_on_art"] == 47977.0


def test_read_tabular_seed_rows_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "historical_harp_panel.csv"
    csv_path.write_text(
        "year,metric_name,value,measurement_class,geo\n"
        "2021,diagnosed_plhiv,90000,program_observed_harp,Philippines\n"
        "2021,alive_on_art,56000,program_observed_harp,Philippines\n",
        encoding="utf-8",
    )

    rows = _read_tabular_seed_rows(csv_path, source_id="manual_seed", source_label="Manual Seed")

    assert len(rows) == 2
    assert {row["metric_name"] for row in rows} == {"diagnosed_plhiv", "alive_on_art"}
    assert all(row["measurement_class"] == "program_observed_harp" for row in rows)


def test_frozen_backtest_artifacts_reflect_holdout() -> None:
    metric_rows = []
    for year in range(2020, 2025):
        metric_rows.extend(
            [
                {"metric_name": "diagnosed_plhiv", "year": year, "value": 70000 + 1000 * (year - 2020), "measurement_class": "program_observed_harp"},
                {"metric_name": "alive_on_art", "year": year, "value": 40000 + 800 * (year - 2020), "measurement_class": "program_observed_harp"},
            ]
        )
    spec, summary = _build_frozen_backtest_artifacts(metric_rows)

    assert spec["ready_for_model_backtest"] is True
    assert spec["holdout_years"] == [2024]
    assert summary["comparison_count"] >= 2


def test_packaged_curated_seed_makes_archive_backtest_ready() -> None:
    run_id = "pytest-harp-archive-full"
    run_harp_archive_build(run_id=run_id, plugin_id="hiv")
    archive_dir = ROOT_DIR / "artifacts" / "runs" / run_id / "harp_archive"

    assessment = read_json(archive_dir / "backtest_assessment.json", default={})
    spec = read_json(archive_dir / "frozen_backtest_spec.json", default={})
    summary = read_json(archive_dir / "frozen_backtest_summary.json", default={})
    panel = read_json(archive_dir / "historical_harp_panel.json", default={})

    assert assessment.get("backtest_ready") is True
    assert assessment.get("coverage_summary", {}).get("program_observed_year_count", 0) >= 5
    assert spec.get("ready_for_model_backtest") is True
    assert spec.get("holdout_years") == [2024]
    assert summary.get("comparison_count", 0) >= 4
    assert any(int(row.get("year") or 0) == 2024 for row in panel.get("rows", []))


def test_packaged_seed_survives_manual_seed_dir_override(tmp_path) -> None:
    run_id = "pytest-harp-archive-manual-override"
    run_harp_archive_build(run_id=run_id, plugin_id="hiv", manual_seed_dir=str(tmp_path))
    archive_dir = ROOT_DIR / "artifacts" / "runs" / run_id / "harp_archive"

    manifest = read_json(archive_dir / "archive_source_manifest.json", default={})
    assessment = read_json(archive_dir / "backtest_assessment.json", default={})

    assert any(row.get("source_id") == "curated_historical_harp_2017_2024" for row in manifest.get("sources", []))
    assert assessment.get("backtest_ready") is True


def test_curated_estimated_plhiv_wins_dedup_for_historical_panel() -> None:
    rows = [
        {
            "year": 2023,
            "metric_name": "estimated_plhiv",
            "measurement_class": "model_estimate",
            "geo": "Philippines",
            "value": 45047,
            "source_id": "core_team_2025",
            "evidence_confidence": 0.98,
        },
        {
            "year": 2023,
            "metric_name": "estimated_plhiv",
            "measurement_class": "model_estimate",
            "geo": "Philippines",
            "value": 189000,
            "source_id": "curated_historical_harp_2017_2024",
            "evidence_confidence": 0.90,
        },
    ]

    deduped = _deduplicate_metric_rows(rows)

    assert len(deduped) == 1
    assert deduped[0]["value"] == 189000
