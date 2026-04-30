from __future__ import annotations

import json
import shutil
from pathlib import Path

import epigraph_ph.harp_archive.doh_hiv_sti_archive as archive_module
import epigraph_ph.harp_archive.pipeline as pipeline_module
import requests
from epigraph_ph.harp_archive.pipeline import (
    YEAR_RANGE,
    _backtest_assessment,
    _build_frozen_backtest_artifacts,
    _promote_reused_archive_ocr_manifest_to_shared,
    _deduplicate_metric_rows,
    _archive_source_signature,
    _reuse_cached_archive_build_if_current,
    _extract_core_team_cascade,
    _extract_core_team_series,
    _extract_core_team_subnational_kp,
    _extract_generic_harp_snapshot,
    _resolve_local_seed_path,
    _read_tabular_seed_rows,
    _extract_surveillance_kp_profile,
    _panel_from_metric_rows,
    run_harp_archive_build,
)
from epigraph_ph.harp_archive.doh_hiv_sti_archive import (
    archive_ocr_settings,
    archive_seed_path,
    build_diagnosis_flow_points,
    build_harp_program_points,
    download_archive_pdfs,
    extract_art_summary_rows,
    extract_art_treatment_outcome_rows,
    extract_diagnosis_summary_rows,
    extract_continuum_metric_rows,
    load_archive_seed_rows,
    materialize_archive_ocr_payload,
)
from epigraph_ph.runtime import ROOT_DIR, read_json, write_json


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
    assert {str(row["time"])[-3:] for row in rows} == {"-12"}


def test_extract_core_team_series_reads_explicit_annual_deaths_block() -> None:
    sample = (
        "Annual new HIV infections, 2010-2024 "
        "Estimated PLHIV "
        "2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 "
        "Spectrum 2025 4,600 5,600 6,700 7,900 9,000 10,300 11,800 13,400 14,900 16,400 18,300 20,900 23,900 26,900 29,800 "
        "Spectrum 2025 300 400 500 600 600 600 600 700 800 900 1100 1300 1600 1900 2100 "
        "300 2100 0 500 1000 1500 2000 "
        "Annual AIDS deaths Annual AIDS Deaths, 2010-2024"
    )

    rows = _extract_core_team_series(sample, "core_team_2025", "Core team", 12)
    deaths = {(int(row["year"]), row["metric_name"]): float(row["value"]) for row in rows}

    assert deaths[(2022, "annual_aids_deaths")] == 1600.0
    assert deaths[(2024, "annual_aids_deaths")] == 2100.0


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
    assert {row["time"] for row in rows} == {"2024-12"}


def test_extract_continuum_metric_rows_handles_ohasis_quarterly_layout() -> None:
    sample = (
        "95-95-95 ACCOMPLISHMENT\n"
        "ThelatestPhilippineHIVestimatesshowthatbytheendof2023 there will be189.000estimated People Living with HIV.\n"
        "Of the estimated PLHIV,111,031（59%) cases have been diagnosed or laboratory-confirmed and currently living or not reported to have died, as of June 2023.\n"
        "Further,70,916 PLHIV are currently on life-saving Anti-retroviral Therapy(ART),ofwhich,26,006(37%)PLHIVhave been tested for viral load(VL) in the past12 months.Among those tested forVL,22,690(87%)are virally suppressed.\n"
    )

    rows = extract_continuum_metric_rows(
        {
            "source_id": "probe_2023_q2",
            "label": "2023 April-June",
            "source_year": 2023,
            "source_url": "",
            "effective_month": "2023-06",
            "start_month": "2023-04",
            "end_month": "2023-06",
            "temporal_precision": "quarterly_snapshot",
        },
        [{"page_number": 1, "text": sample}],
    )
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["estimated_plhiv"] == 189000.0
    assert by_metric["diagnosed_plhiv"] == 111031.0
    assert by_metric["alive_on_art"] == 70916.0


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
    assert {str(row["time"])[-3:] for row in rows} == {"-12"}


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


def test_resolve_local_seed_path_recovers_moved_local_pdf(monkeypatch, tmp_path) -> None:
    relocated_root = tmp_path / "Bioarxiv"
    relocated_root.mkdir(parents=True)
    relocated_pdf = relocated_root / "2025 PH HIV Estimates_Core team_for WHO.pdf"
    relocated_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pipeline_module, "_local_seed_search_roots", lambda desktop_seed_dir=None: [relocated_root])
    monkeypatch.setattr(pipeline_module, "local_archive_pdf_corpus_rows", lambda: [])

    resolved = _resolve_local_seed_path(
        {
            "source_id": "core_team_2025",
            "path": Path(r"C:\Users\gaura\OneDrive\Desktop\2025 PH HIV Estimates_Core team_for WHO.pdf"),
            "source_kind": "local_pdf",
        }
    )

    assert resolved == relocated_pdf


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
    merged_rows = read_json(archive_dir / "historical_metric_rows.json", default=[])
    canonical_rows = read_json(archive_dir / "historical_metric_rows_harp_only.json", default=[])
    wdi_rows = read_json(archive_dir / "wdi_hiv_rows.json", default=[])
    wdi_overlap = read_json(archive_dir / "wdi_hiv_overlap_summary.json", default={})

    assert assessment.get("backtest_ready") is True
    assert assessment.get("coverage_summary", {}).get("program_observed_year_count", 0) >= 5
    assert spec.get("ready_for_model_backtest") is True
    assert spec.get("holdout_years") == [2025]
    assert summary.get("comparison_count", 0) >= 4
    assert any(int(row.get("year") or 0) == 2025 for row in panel.get("rows", []))
    assert merged_rows
    assert canonical_rows
    assert wdi_rows
    assert not any(str(row.get("measurement_class")) == "external_reference_wdi" for row in canonical_rows)
    assert any(str(row.get("measurement_class")) == "external_reference_wdi" for row in merged_rows)
    assert any(str(row.get("source_tier")) == "overlap_validated_external_reference" for row in wdi_rows)
    assert wdi_overlap.get("art_coverage_overlap", {}).get("comparison_rows")


def test_packaged_seed_survives_manual_seed_dir_override(tmp_path) -> None:
    run_id = "pytest-harp-archive-manual-override"
    run_harp_archive_build(run_id=run_id, plugin_id="hiv", manual_seed_dir=str(tmp_path))
    archive_dir = ROOT_DIR / "artifacts" / "runs" / run_id / "harp_archive"

    manifest = read_json(archive_dir / "archive_source_manifest.json", default={})
    assessment = read_json(archive_dir / "backtest_assessment.json", default={})

    assert any(row.get("source_id") == "curated_historical_harp_2017_2024" for row in manifest.get("sources", []))
    assert assessment.get("backtest_ready") is True


def test_run_harp_archive_build_reuses_matching_prior_run_without_download(monkeypatch) -> None:
    prior_run_id = "pytest-harp-archive-prior-reuse-source"
    current_run_id = "pytest-harp-archive-prior-reuse-target"
    prior_run_dir = ROOT_DIR / "artifacts" / "runs" / prior_run_id
    current_run_dir = ROOT_DIR / "artifacts" / "runs" / current_run_id
    shutil.rmtree(prior_run_dir, ignore_errors=True)
    shutil.rmtree(current_run_dir, ignore_errors=True)

    fake_archive_seed_rows = [
        {
            "source_id": "doh_hiv_sti_2024_2024_october_december",
            "label": "2024 October - December",
            "source_kind": "doh_hiv_sti_archive_pdf",
            "source_label": "DOH HIV/STI Archive",
            "source_year": 2024,
            "source_url": "https://drive.google.com/file/d/test-file/view",
            "file_id": "test-file",
            "temporal_precision": "quarterly_snapshot",
            "effective_month": "2024-12",
            "start_month": "2024-10",
            "end_month": "2024-12",
        }
    ]
    prior_archive_dir = prior_run_dir / "harp_archive"
    prior_artifact_paths = pipeline_module._archive_required_artifact_paths(prior_archive_dir)
    for path in prior_artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            path.write_text("year,metric_name,value\n", encoding="utf-8")
        else:
            write_json(path, {})

    build_cache_key = pipeline_module._archive_build_cache_key(fake_archive_seed_rows)
    write_json(
        prior_archive_dir / "archive_source_manifest.json",
        {
            "build_version": pipeline_module._HARP_ARCHIVE_BUILD_VERSION,
            "build_cache_key": build_cache_key,
            "source_manifest_rows": pipeline_module._archive_source_manifest_rows(fake_archive_seed_rows),
            "sources": fake_archive_seed_rows,
        },
    )
    write_json(
        prior_archive_dir / "ocr_corpus_manifest.json",
        {
            "cache_settings": pipeline_module._archive_ocr_cache_settings(),
            "documents": [],
        },
    )
    write_json(
        prior_archive_dir / "harp_archive_manifest.json",
        {
            "run_id": prior_run_id,
            "plugin_id": "hiv",
            "build_version": pipeline_module._HARP_ARCHIVE_BUILD_VERSION,
            "build_cache_key": build_cache_key,
            "artifact_paths": {key: str(path) for key, path in prior_artifact_paths.items()},
        },
    )
    write_json(prior_archive_dir / "historical_harp_panel.json", {"rows": [{"year": 2024, "metric_name": "diagnosed_plhiv"}]})

    monkeypatch.setattr(pipeline_module, "_materialize_local_sources", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline_module, "load_archive_seed_rows", lambda *args, **kwargs: list(fake_archive_seed_rows))

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("download_archive_pdfs should not run when a matching prior build exists")

    monkeypatch.setattr(pipeline_module, "download_archive_pdfs", _unexpected_download)

    manifest = run_harp_archive_build(run_id=current_run_id, plugin_id="hiv")
    current_archive_dir = current_run_dir / "harp_archive"
    current_manifest = read_json(current_archive_dir / "harp_archive_manifest.json", default={})

    assert manifest.get("reused_existing_build") is True
    assert current_manifest.get("reused_from_run_id") == prior_run_id
    assert (current_archive_dir / "historical_harp_panel.json").exists()


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


def test_doh_ground_truth_overlay_wins_dedup_for_overlap_years() -> None:
    rows = [
        {
            "year": 2024,
            "metric_name": "alive_on_art",
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "value": 90854,
            "source_id": "core_team_2025",
            "evidence_confidence": 0.99,
        },
        {
            "year": 2024,
            "metric_name": "alive_on_art",
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "value": 90568,
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "evidence_confidence": 1.0,
        },
    ]

    deduped = _deduplicate_metric_rows(rows)

    assert len(deduped) == 1
    assert deduped[0]["value"] == 90568


def test_doh_archive_seed_loads_and_preserves_quarter_ranges() -> None:
    rows = load_archive_seed_rows(archive_seed_path())

    assert len(rows) >= 150
    q4_2024 = next(row for row in rows if row["label"] == "2024 October - December")
    assert q4_2024["temporal_precision"] == "quarterly_snapshot"
    assert q4_2024["effective_month"] == "2024-12"


def test_doh_official_cascade_seed_uses_year_end_months() -> None:
    seed_path = ROOT_DIR / "src" / "epigraph_ph" / "harp_archive" / "seeds" / "doh_official_cascade_ground_truth_2018_2025.csv"

    rows = _read_tabular_seed_rows(seed_path, source_id="doh_official_cascade_ground_truth_2018_2025", source_label="DOH Official Cascade")
    diagnosed_2022 = next(row for row in rows if row["metric_name"] == "diagnosed_plhiv" and int(row["year"]) == 2022)
    art_2024 = next(row for row in rows if row["metric_name"] == "alive_on_art" and int(row["year"]) == 2024)

    assert diagnosed_2022["time"] == "2022-12"
    assert art_2024["time"] == "2024-12"


def test_curated_harp_panel_seed_uses_year_end_months() -> None:
    seed_path = ROOT_DIR / "src" / "epigraph_ph" / "harp_archive" / "seeds" / "historical_harp_panel_curated.csv"

    rows = _read_tabular_seed_rows(seed_path, source_id="curated_historical_harp_2017_2024", source_label="Curated Historical HARP")
    diagnosed_2022 = next(row for row in rows if row["metric_name"] == "diagnosed_plhiv" and int(row["year"]) == 2022)
    art_2024 = next(row for row in rows if row["metric_name"] == "alive_on_art" and int(row["year"]) == 2024)

    assert diagnosed_2022["time"] == "2022-12"
    assert art_2024["time"] == "2024-12"


def test_build_diagnosis_flow_points_dedupes_and_normalizes_against_estimated_plhiv() -> None:
    rows = [
        {
            "metric_name": "estimated_plhiv",
            "time": "2023-01",
            "value": 1000.0,
            "source_id": "estimate_jan",
            "source_label": "Estimate Jan",
        },
        {
            "metric_name": "estimated_plhiv",
            "time": "2023-02",
            "value": 1100.0,
            "source_id": "estimate_feb",
            "source_label": "Estimate Feb",
        },
        {
            "metric_name": "estimated_plhiv",
            "time": "2023-03",
            "value": 1200.0,
            "source_id": "estimate_mar",
            "source_label": "Estimate Mar",
        },
        {
            "metric_name": "new_diagnosed_cases_monthly",
            "time": "2023-01",
            "value": 10.0,
            "source_id": "old_jan",
            "source_label": "Old January report",
            "evidence_confidence": 0.4,
            "temporal_precision": "monthly_snapshot",
        },
        {
            "metric_name": "new_diagnosed_cases_monthly",
            "time": "2023-01",
            "value": 12.0,
            "source_id": "new_jan",
            "source_label": "New January report",
            "evidence_confidence": 0.9,
            "temporal_precision": "monthly_snapshot",
        },
        {
            "metric_name": "new_diagnosed_cases_period",
            "time": "2023-03",
            "period_start": "2023-02",
            "period_end": "2023-03",
            "value": 33.0,
            "source_id": "quarter_q1",
            "source_label": "February to March report",
            "evidence_confidence": 0.8,
            "temporal_precision": "quarterly_snapshot",
            "series_kind": "quarterly_snapshot",
        },
        {
            "metric_name": "advanced_hiv_cases_period",
            "time": "2023-03",
            "period_start": "2023-02",
            "period_end": "2023-03",
            "value": 9.0,
            "source_id": "advanced_q1",
            "source_label": "Advanced cases February to March",
            "evidence_confidence": 0.7,
        },
    ]

    points = build_diagnosis_flow_points(rows)

    assert len(points) == 2
    january = next(point for point in points if point["period_start"] == "2023-01")
    quarter = next(point for point in points if point["period_start"] == "2023-02")
    assert january["source_id"] == "new_jan"
    assert january["diagnosed_count"] == 12.0
    assert january["estimated_plhiv"] == 1000.0
    assert january["diagnosed_share"] == 0.012
    assert quarter["estimated_plhiv"] == 1150.0
    assert round(float(quarter["diagnosed_share"]), 6) == round(33.0 / 1150.0, 6)
    assert quarter["advanced_hiv_cases"] == 9.0
    assert round(float(quarter["advanced_hiv_share"]), 6) == round(9.0 / 33.0, 6)


def test_download_archive_pdfs_tolerates_request_timeouts(tmp_path, monkeypatch) -> None:
    class _TimeoutSession:
        def get(self, *_args, **_kwargs):
            raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(archive_module, "_download_session", lambda: _TimeoutSession())
    rows = [
        {
            "source_id": "timeout_case",
            "label": "2024 January",
            "file_id": "fake-file-id",
        }
    ]

    hydrated = download_archive_pdfs(archive_rows=rows, raw_dir=tmp_path, request_timeout_seconds=0.01, pause_seconds=0.0)

    assert len(hydrated) == 1
    assert hydrated[0]["download_status"] == "failed"
    assert hydrated[0]["local_path"] == ""
    assert "Timeout" in str(hydrated[0]["download_error"])


def test_download_archive_pdfs_uses_powershell_fallback_when_requests_fail(tmp_path, monkeypatch) -> None:
    class _TimeoutSession:
        def get(self, *_args, **_kwargs):
            raise requests.ConnectionError("simulated connection error")

    def _fake_powershell_download(_url: str, destination: Path, *, timeout_seconds: float) -> tuple[bool, str]:
        assert timeout_seconds == 0.01
        destination.write_bytes(b"%PDF-1.3\n%fake\n")
        return True, ""

    monkeypatch.setattr(archive_module, "_download_session", lambda: _TimeoutSession())
    monkeypatch.setattr(archive_module, "_powershell_download_pdf", _fake_powershell_download)
    rows = [
        {
            "source_id": "fallback_case",
            "label": "2024 January",
            "file_id": "fake-file-id",
        }
    ]

    hydrated = download_archive_pdfs(archive_rows=rows, raw_dir=tmp_path, request_timeout_seconds=0.01, pause_seconds=0.0)

    assert len(hydrated) == 1
    assert hydrated[0]["download_status"] == "downloaded"
    assert hydrated[0]["local_path"]
    assert Path(hydrated[0]["local_path"]).exists()


def test_reuse_cached_archive_build_accepts_legacy_ocr_settings_manifest(tmp_path) -> None:
    artifact_path = tmp_path / "historical_metric_rows.json"
    artifact_path.write_text("{}", encoding="utf-8")
    source_rows = [
        {
            "source_id": "archive_2024_q4",
            "source_kind": "doh_hiv_sti_archive_pdf",
            "label": "2024 October - December",
            "local_path": "D:\\archive\\2024_q4.pdf",
            "checksum": "abc123",
            "download_status": "downloaded",
        }
    ]
    write_json(
        tmp_path / "harp_archive_manifest.json",
        {
            "build_version": "harp_archive_cache_v1",
            "artifact_paths": {"historical_metric_rows": str(artifact_path)},
        },
    )
    write_json(
        tmp_path / "archive_source_manifest.json",
        {
            "build_version": "harp_archive_cache_v1",
            "source_signature": _archive_source_signature(source_rows),
        },
    )
    write_json(
        tmp_path / "ocr_corpus_manifest.json",
        {
            "build_version": "harp_archive_cache_v1",
            "ocr_settings": archive_ocr_settings(),
        },
    )

    reused = _reuse_cached_archive_build_if_current(
        archive_dir=tmp_path,
        source_rows=source_rows,
        force_refresh=False,
    )

    assert reused is not None
    assert reused["reused_existing_build"] is True


def test_shared_archive_ocr_cache_reuses_payload_across_run_dirs(tmp_path, monkeypatch) -> None:
    shared_dir = tmp_path / "shared_ocr"
    monkeypatch.setenv("EPIGRAPH_SHARED_OCR_DIR", str(shared_dir))

    source_bytes = b"%PDF-1.4\nshared test payload\n"
    run1_pdf = tmp_path / "run1" / "raw" / "report.pdf"
    run2_pdf = tmp_path / "run2" / "raw" / "report.pdf"
    run1_pdf.parent.mkdir(parents=True, exist_ok=True)
    run2_pdf.parent.mkdir(parents=True, exist_ok=True)
    run1_pdf.write_bytes(source_bytes)
    run2_pdf.write_bytes(source_bytes)

    call_counter = {"count": 0}

    def fake_ocr(local_path: Path, *, force_ocr_every_page=None, max_pages=None):
        call_counter["count"] += 1
        return {
            "local_path": str(local_path),
            "checksum": archive_module.sha256_file(local_path),
            "page_count": 1,
            "pages": [{"page_number": 1, "text": "shared page text"}],
            "runtime": archive_ocr_settings(),
            "force_ocr_every_page": bool(force_ocr_every_page),
            "max_pages": int(max_pages or 0),
            "generated_at": "2026-04-01T00:00:00+00:00",
        }

    monkeypatch.setattr(archive_module, "ocr_archive_pdf_payload", fake_ocr)

    first = materialize_archive_ocr_payload(
        {"source_id": "report_run1", "label": "Run 1", "source_year": 2024, "source_url": "", "local_path": str(run1_pdf)},
        ocr_dir=tmp_path / "run1" / "ocr_corpus",
    )
    second = materialize_archive_ocr_payload(
        {"source_id": "report_run2", "label": "Run 2", "source_year": 2024, "source_url": "", "local_path": str(run2_pdf)},
        ocr_dir=tmp_path / "run2" / "ocr_corpus",
    )

    assert call_counter["count"] == 1
    assert first["from_cache"] is False
    assert first["from_shared_cache"] is False
    assert second["from_cache"] is True
    assert second["from_shared_cache"] is True
    assert first["shared_cache_key"] == second["shared_cache_key"]
    assert first["shared_artifact_path"] == second["shared_artifact_path"]
    assert Path(second["shared_artifact_path"]).exists()
    assert Path(second["run_artifact_path"]).exists()
    assert second["pages"][0]["text"] == "shared page text"


def test_shared_archive_ocr_cache_invalidates_on_cache_settings_change(tmp_path, monkeypatch) -> None:
    shared_dir = tmp_path / "shared_ocr"
    monkeypatch.setenv("EPIGRAPH_SHARED_OCR_DIR", str(shared_dir))

    pdf_path = tmp_path / "raw" / "report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\nsettings invalidation\n")

    call_counter = {"count": 0}

    def fake_ocr(local_path: Path, *, force_ocr_every_page=None, max_pages=None):
        call_counter["count"] += 1
        return {
            "local_path": str(local_path),
            "checksum": archive_module.sha256_file(local_path),
            "page_count": 1,
            "pages": [{"page_number": 1, "text": f"cache-settings-{max_pages}"}],
            "runtime": archive_ocr_settings(),
            "force_ocr_every_page": bool(force_ocr_every_page),
            "max_pages": int(max_pages or 0),
            "generated_at": "2026-04-01T00:00:00+00:00",
        }

    monkeypatch.setattr(archive_module, "ocr_archive_pdf_payload", fake_ocr)

    first = materialize_archive_ocr_payload(
        {"source_id": "report_one", "label": "Report", "source_year": 2024, "source_url": "", "local_path": str(pdf_path)},
        ocr_dir=tmp_path / "run1" / "ocr_corpus",
        max_pages=1,
    )
    second = materialize_archive_ocr_payload(
        {"source_id": "report_two", "label": "Report", "source_year": 2024, "source_url": "", "local_path": str(pdf_path)},
        ocr_dir=tmp_path / "run2" / "ocr_corpus",
        max_pages=2,
    )

    assert call_counter["count"] == 2
    assert first["shared_cache_key"] != second["shared_cache_key"]
    assert first["from_shared_cache"] is False
    assert second["from_shared_cache"] is False


def test_reused_archive_manifest_promotes_legacy_local_ocr_sidecars_to_shared(tmp_path, monkeypatch) -> None:
    shared_dir = tmp_path / "shared_ocr"
    monkeypatch.setenv("EPIGRAPH_SHARED_OCR_DIR", str(shared_dir))

    archive_dir = tmp_path / "harp_archive"
    run_artifact_path = archive_dir / "ocr_corpus" / "legacy_report.json"
    run_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "local_path": str(tmp_path / "raw" / "legacy_report.pdf"),
        "checksum": "legacy-checksum",
        "page_count": 1,
        "pages": [{"page_number": 1, "text": "legacy cached text"}],
        "runtime": archive_ocr_settings(),
        "force_ocr_every_page": True,
        "max_pages": 0,
        "generated_at": "2026-04-01T00:00:00+00:00",
    }
    write_json(run_artifact_path, legacy_payload)
    write_json(
        archive_dir / "ocr_corpus_manifest.json",
        {
            "documents": [
                {
                    "source_id": "legacy_report",
                    "ocr_artifact_path": str(run_artifact_path),
                }
            ]
        },
    )

    _promote_reused_archive_ocr_manifest_to_shared(archive_dir)

    expected_cache_settings = archive_module.archive_ocr_cache_settings(
        force_ocr_every_page=True,
        max_pages=0,
    )
    expected_key = archive_module.shared_archive_ocr_cache_key(
        checksum="legacy-checksum",
        cache_settings=expected_cache_settings,
    )
    expected_shared_path = archive_module.shared_archive_ocr_artifact_path(expected_key)

    assert expected_shared_path.exists()


def test_extract_continuum_metric_rows_from_recent_quarter_text() -> None:
    report = {
        "source_id": "probe_2024_q4",
        "label": "2024 October - December",
        "source_year": 2024,
        "source_url": "",
        "effective_month": "2024-12",
        "start_month": "2024-10",
        "end_month": "2024-12",
        "temporal_precision": "quarterly_snapshot",
    }
    sample = (
        "HIV & AIDS CONTINUUM OF CARE "
        "there will be 215,400 estimated People Living with HIV (PLHIV) in the country. "
        "As of December 2024, of the estimated PLHIV, 135,026 (63%) of the estimated PLHIV have been diagnosed. "
        "Further, 90,854 (67%) PLHIV are currently on life-saving Antiretroviral Therapy (ART), "
        "of which, 41,860 (46%) PLHIV have been tested for viral load (VL) in the past 12 months. "
        "Among those tested for VL, 36,723 (88%) were virally suppressed."
    )

    rows = extract_continuum_metric_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["estimated_plhiv"] == 215400.0
    assert by_metric["diagnosed_plhiv"] == 135026.0
    assert by_metric["alive_on_art"] == 90854.0
    assert by_metric["tested_for_viral_load"] == 41860.0
    assert by_metric["virally_suppressed"] == 36723.0


def test_extract_continuum_metric_rows_accepts_qualified_percent_phrasing() -> None:
    report = {
        "source_id": "probe_2025_q1",
        "label": "2025 January - March",
        "source_year": 2025,
        "source_url": "",
        "effective_month": "2025-03",
        "start_month": "2025-01",
        "end_month": "2025-03",
        "temporal_precision": "quarterly_snapshot",
    }
    sample = (
        "HIV & AIDS CONTINUUM OF CARE "
        "The latest Philippine HIV estimates show that by the end of 2025, there will be 252,800 estimated People Living with HIV (PLHIV) in the country. "
        "As of March 2025, 139,610 (55% of the estimated) PLHIV have been diagnosed or laboratory-confirmed. "
        "Further, 92,712 (66% of the diagnosed) PLHIV are currently on life-saving Antiretroviral Therapy (ART), "
        "of which, 41,786 (45%) PLHIV have been tested for viral load (VL) in the past 12 months. "
        "Among those tested for VL, 36,630 (88%) were virally suppressed."
    )

    rows = extract_continuum_metric_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["estimated_plhiv"] == 252800.0
    assert by_metric["diagnosed_plhiv"] == 139610.0
    assert by_metric["alive_on_art"] == 92712.0
    assert by_metric["tested_for_viral_load"] == 41786.0
    assert by_metric["virally_suppressed"] == 36630.0


def test_extract_diagnosis_summary_rows_captures_youth_quick_facts_table() -> None:
    report = {
        "source_id": "probe_2010_dec",
        "label": "2010 December",
        "source_year": 2010,
        "source_url": "",
        "effective_month": "2010-12",
        "start_month": "2010-12",
        "end_month": "2010-12",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "In December 2010, there were 174 confirmed HIV-positive individuals reported.\n"
        "Table 1. Quick Facts\n"
        "Demographic Data\n"
        "Dec\n2010\n"
        "Total Reported Cases\n174\n"
        "Youth 15-24yo\n59\n1,213\n"
        "Children <15yo\n0\n55\n"
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["youth_cases_15_24_period"] == 59.0


def test_extract_diagnosis_summary_rows_handles_monthly_numeric_counts_and_deaths() -> None:
    report = {
        "source_id": "probe_2022_jan",
        "label": "2022 January",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-01",
        "start_month": "2022-01",
        "end_month": "2022-01",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "In January 2022, there were 875 confirmed HIV-positive individuals reported to the HIV/AIDS & ART Registry of the Philippines. "
        "In January 2022, there were 33 reported deaths due to any cause among people with HIV."
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["new_diagnosed_cases_period"] == 875.0
    assert by_metric["deaths_reported_period"] == 33.0


def test_extract_diagnosis_summary_rows_handles_collapsed_ocr_and_spelled_out_deaths() -> None:
    report = {
        "source_id": "probe_2022_oct",
        "label": "2022 October",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-10",
        "start_month": "2022-10",
        "end_month": "2022-10",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "InOctober2022,therewere1,383confirmedHIV-positiveindividuals reportedtotheHIV/AIDS&ARTRegistry of the Philippines. "
        "Sixty-five deathswere newly reported,bringing the total deaths reported this year to 878e."
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["new_diagnosed_cases_period"] == 1383.0
    assert by_metric["deaths_reported_period"] == 65.0


def test_extract_diagnosis_summary_rows_handles_interleaved_summary_table_ocr() -> None:
    report = {
        "source_id": "probe_2022_march",
        "label": "2022 March",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-03",
        "start_month": "2022-03",
        "end_month": "2022-03",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "NEWLYDIAGNOSEDCASES\n"
        "InMarch2022,therewere1,539confirmedHIV-positive\n"
        "Table1:SummaryofHivdiagnosesanddeaths\n"
        "Mar Jan- Jan2017- Jan1984-\n"
        "individualsreportedtotheHIV/AIDS&ARTRegistryofthe\n"
        "Philippines(HARP).This wasa48%increase comparedto\n"
        "Totalreportedcases 1,539 3,468 58,900 97,792\n"
        "Reporteddeaths9 66 177 3,629 5,548\n"
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["new_diagnosed_cases_period"] == 1539.0
    assert by_metric["deaths_reported_period"] == 66.0


def test_extract_diagnosis_summary_rows_captures_cumulative_cases_and_deaths_from_table() -> None:
    report = {
        "source_id": "probe_2022_june",
        "label": "2022 June",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-06",
        "start_month": "2022-06",
        "end_month": "2022-06",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "NEWLY DIAGNOSED CASES\n"
        "Total reported cases 1,472 7,444 62,876 101,768\n"
        "Reporteddeaths9 36 246 3,698 5,616\n"
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}, {"page_number": 2, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["diagnosed_cases_cumulative"] == 101768.0
    assert by_metric["deaths_reported_cumulative"] == 5616.0


def test_extract_art_summary_rows_captures_monthly_alive_on_art_bridge_stock() -> None:
    report = {
        "source_id": "probe_2022_january",
        "label": "2022 January",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-01",
        "start_month": "2022-01",
        "end_month": "2022-01",
        "temporal_precision": "monthly_snapshot",
    }
    sample = "A total of 56,982 people living with HIV (PLHIV) were presently on ART as of January 2022."

    rows = extract_art_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["alive_on_art"] == 56982.0


def test_extract_art_treatment_outcome_rows_captures_direct_ltfu_transfer_stop_and_death() -> None:
    report = {
        "source_id": "probe_2025_q2",
        "label": "2025 Q2",
        "source_year": 2025,
        "source_url": "",
        "effective_month": "2025-06",
        "start_month": "2025-04",
        "end_month": "2025-06",
        "temporal_precision": "quarterly_snapshot",
    }
    sample = (
        "Table 4. Number of PLHIV ever enrolled to ART by treatment outcome and region as of June 2025\n"
        "Treatment Outcome Alive on ART Lost to Follow-up24 (n= 29,286) Dead (n= 6,255) "
        "Trans out (Overseas)25 (n= 13) Stopped26 (n= 5)\n"
        "Among the 131,381 people living with HIV (PLHIV) who have ever been enrolled on antiretroviral therapy (ART) since 2002, "
        "a total of 95,556 individuals were alive on ART as of June 2025.\n"
        "As of June 2025, 29,304 (22%) individuals who were previously on ART were no longer receiving treatment. "
        "This group includes 29,286 individuals who were lost to follow-up, 5 who refused to continue ART for various reasons, "
        "and 13 who reported migrating overseas."
    )

    rows = extract_art_treatment_outcome_rows(report, [{"page_number": 4, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["art_ever_enrolled_cumulative"] == 131381.0
    assert by_metric["art_ltfu_cumulative"] == 29286.0
    assert by_metric["art_deaths_cumulative"] == 6255.0
    assert by_metric["art_transfer_out_overseas_cumulative"] == 13.0
    assert by_metric["art_stopped_refused_cumulative"] == 5.0
    assert by_metric["art_no_longer_on_treatment_cumulative"] == 29304.0
    assert {row["temporal_precision"] for row in rows} == {"monthly_snapshot"}


def test_extract_quick_facts_stock_rows_ignores_non_quick_facts_summary_pages() -> None:
    report = {
        "source_id": "probe_2023_may",
        "label": "2023 May",
        "source_year": 2023,
        "source_url": "",
        "effective_month": "2023-05",
        "start_month": "2023-05",
        "end_month": "2023-05",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "SUMMARYOFNEWLYDIAGNOSEDCASES\n"
        "InMay2023,therewere1,256confirmedHIV-positiveindividualsreportedtotheHIV/AIDS&ARTRegistryofthePhilippines.\n"
        "Table1:NumberofdiagnosedHiVcasesbymodeoftransmissionandsex,Jan-May2023(N=7,315)\n"
        "Totalreported cases,May2023\n"
        "1,186\n70\n323\n132\n1,256\n48\n"
    )

    rows = archive_module.extract_quick_facts_stock_rows(report, [{"page_number": 1, "text": sample}])

    assert rows == []


def test_extract_quick_facts_stock_rows_rejects_impossible_cumulative_alive_counts() -> None:
    report = {
        "source_id": "probe_2022_oct_quickfacts",
        "label": "2022 October",
        "source_year": 2022,
        "source_url": "",
        "effective_month": "2022-10",
        "start_month": "2022-10",
        "end_month": "2022-10",
        "temporal_precision": "monthly_snapshot",
    }
    sample = (
        "InOctober2022,therewere1,383confirmedHIV-positiveindividuals reportedtotheHIV/AIDS&ARTRegistry of the Philippines.\n"
        "Table 1. Quick Facts\n"
        "Demographic Data\n"
        "Oct\nJan-Oct 2022\nJan1984-\n"
        "Total reported cases\n513\n1,383\n109,282\n"
        "Reported deaths\n202\n878\n6,351\n"
    )

    rows = archive_module.extract_quick_facts_stock_rows(report, [{"page_number": 1, "text": sample}])

    assert rows == []


def test_extract_diagnosis_summary_rows_captures_post_2014_youth_narrative_layout() -> None:
    report = {
        "source_id": "sample_2019_q1",
        "label": "2019 Q1 HARP",
        "source_label": "2019 Q1 HARP",
        "source_year": 2019,
        "year": 2019,
        "time": "2019-03",
        "effective_month": "2019-03",
        "start_month": "2019-01",
        "end_month": "2019-03",
        "period_start": "2019-01",
        "period_end": "2019-03",
        "temporal_precision": "quarter",
        "geo": "Philippines",
        "region": "national",
        "province": "Philippines",
        "evidence_confidence": 0.92,
        "measurement_class": "surveillance_report",
        "series_kind": "period_total",
    }
    sample = (
        "HIV/AIDS & ART Registry of the Philippines\n"
        "From January to March 2019, there were 2,946 newly diagnosed HIV-positive individuals reported.\n"
        "Of the reported cases, 31% (926) were 15-24 years old.\n"
    )

    rows = extract_diagnosis_summary_rows(report, [{"page_number": 1, "text": sample}])
    by_metric = {row["metric_name"]: row["value"] for row in rows}

    assert by_metric["youth_cases_15_24_period"] == 926.0


def test_build_harp_program_points_uses_latest_time_and_estimate_rows() -> None:
    metric_rows = [
        {
            "year": 2024,
            "time": "2024-03",
            "metric_name": "estimated_plhiv",
            "value": 210000.0,
            "measurement_class": "model_estimate",
            "source_id": "seed_a",
            "source_label": "Seed A",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 215400.0,
            "measurement_class": "model_estimate",
            "source_id": "seed_b",
            "source_label": "Seed B",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "diagnosed_plhiv",
            "value": 135026.0,
            "measurement_class": "program_observed_harp",
            "source_id": "seed_b",
            "source_label": "Q4 2024",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "alive_on_art",
            "value": 90854.0,
            "measurement_class": "program_observed_harp",
            "source_id": "seed_b",
            "source_label": "Q4 2024",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "tested_for_viral_load",
            "value": 41860.0,
            "measurement_class": "program_observed_harp",
            "source_id": "seed_b",
            "source_label": "Q4 2024",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "virally_suppressed",
            "value": 36723.0,
            "measurement_class": "program_observed_harp",
            "source_id": "seed_b",
            "source_label": "Q4 2024",
        },
    ]

    points = build_harp_program_points(metric_rows)

    assert len(points) == 1
    assert points[0]["month"] == "2024-12"
    assert points[0]["estimated_plhiv"] == 215400.0


def test_panel_from_metric_rows_prefers_latest_time_within_year() -> None:
    metric_rows = [
        {
            "year": 2024,
            "time": "2024-03",
            "metric_name": "diagnosed_plhiv",
            "value": 120000.0,
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "source_id": "older",
            "evidence_confidence": 0.8,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "diagnosed_plhiv",
            "value": 135026.0,
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "source_id": "newer",
            "evidence_confidence": 0.7,
        },
    ]

    panel = _panel_from_metric_rows(metric_rows)
    row_2024 = next(row for row in panel["rows"] if row["year"] == 2024)

    assert row_2024["diagnosed_plhiv"] == 135026.0
    assert row_2024["time"] == "2024-12"


def test_panel_from_metric_rows_prefers_official_seed_over_later_lower_priority_row() -> None:
    metric_rows = [
        {
            "year": 2024,
            "time": "2024-01",
            "metric_name": "estimated_plhiv",
            "value": 215400.0,
            "measurement_class": "model_estimate",
            "geo": "Philippines",
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "evidence_confidence": 1.0,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 216900.0,
            "measurement_class": "model_estimate",
            "geo": "Philippines",
            "source_id": "core_team_2025",
            "evidence_confidence": 0.99,
        },
    ]

    panel = _panel_from_metric_rows(metric_rows)
    row_2024 = next(row for row in panel["rows"] if row["year"] == 2024)

    assert row_2024["estimated_plhiv"] == 215400.0
    assert row_2024["time"] == "2024-01"


def test_deduplicate_metric_rows_preserves_distinct_months() -> None:
    rows = [
        {
            "year": 2024,
            "time": "2024-03",
            "metric_name": "diagnosed_plhiv",
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "value": 120000.0,
            "source_id": "q1",
            "evidence_confidence": 0.8,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "diagnosed_plhiv",
            "measurement_class": "program_observed_harp",
            "geo": "Philippines",
            "value": 135026.0,
            "source_id": "q4",
            "evidence_confidence": 0.8,
        },
    ]

    deduped = _deduplicate_metric_rows(rows)

    assert len(deduped) == 2


def test_reuse_cached_archive_build_if_current_accepts_matching_manifest(tmp_path) -> None:
    archive_dir = tmp_path / "harp_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifact = archive_dir / "historical_metric_rows.json"
    artifact.write_text("[]", encoding="utf-8")
    source_rows = [
        {
            "source_id": "seed_a",
            "source_kind": "packaged_csv",
            "label": "Seed A",
            "local_path": str(tmp_path / "seed_a.csv"),
            "checksum": "abc123",
            "download_status": "downloaded",
        }
    ]
    source_manifest = {
        "generated_at": "2026-04-01T00:00:00Z",
        "build_version": "harp_archive_cache_v1",
        "source_signature": [
            {
                "source_id": "seed_a",
                "source_kind": "packaged_csv",
                "label": "Seed A",
                "local_path": str(tmp_path / "seed_a.csv"),
                "checksum": "abc123",
                "download_status": "downloaded",
            }
        ],
        "sources": source_rows,
    }
    ocr_manifest = {
        "generated_at": "2026-04-01T00:00:00Z",
        "build_version": "harp_archive_cache_v1",
        "ocr_settings": archive_ocr_settings(),
        "documents": [],
    }
    manifest = {
        "build_version": "harp_archive_cache_v1",
        "artifact_paths": {
            "historical_metric_rows": str(artifact),
            "archive_source_manifest": str(archive_dir / "archive_source_manifest.json"),
            "ocr_corpus_manifest": str(archive_dir / "ocr_corpus_manifest.json"),
        },
    }
    (archive_dir / "archive_source_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    (archive_dir / "ocr_corpus_manifest.json").write_text(json.dumps(ocr_manifest), encoding="utf-8")
    (archive_dir / "harp_archive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reused = _reuse_cached_archive_build_if_current(
        archive_dir=archive_dir,
        source_rows=source_rows,
        force_refresh=False,
    )

    assert reused is not None
    assert reused["reused_existing_build"] is True
