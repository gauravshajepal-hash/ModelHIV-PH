from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import epigraph_ph.aidsdatahub.extractor as extractor


def test_discover_detail_urls_from_listing_html() -> None:
    html = """
    <html>
      <body>
        <a href="/resource/philippines-country-data-2023">Country Data</a>
        <a href="/index.php/resource/hiv-aids-and-art-registry-philippines-january-2021">Registry</a>
        <a href="/sites/default/files/resource/unaids-data-2024-en.pdf">Direct PDF</a>
        <a href="/resource/thailand-country-data-2023">Other Country</a>
      </body>
    </html>
    """

    urls = extractor._discover_detail_urls_from_html(html, base_url="https://www.aidsdatahub.org/country-snapshot/philippines")

    assert "https://www.aidsdatahub.org/resource/philippines-country-data-2023" in urls
    assert "https://www.aidsdatahub.org/index.php/resource/hiv-aids-and-art-registry-philippines-january-2021" in urls
    assert "https://www.aidsdatahub.org/sites/default/files/resource/unaids-data-2024-en.pdf" not in urls
    assert all("thailand" not in url.lower() for url in urls)


def test_parse_temporal_metadata_from_title() -> None:
    monthly = extractor._parse_temporal_metadata_from_title(
        "HIV/AIDS and ART Registry of the Philippines: January 2021",
        2021,
    )
    quarterly = extractor._parse_temporal_metadata_from_title(
        "HIV & AIDS Surveillance of the Philippines OCT-DEC 2023",
        2023,
    )

    assert monthly["temporal_precision"] == "monthly_snapshot"
    assert monthly["effective_month"] == "2021-01"
    assert quarterly["temporal_precision"] == "quarterly_snapshot"
    assert quarterly["start_month"] == "2023-10"
    assert quarterly["end_month"] == "2023-12"


def test_hydrate_resource_record_uses_curated_override() -> None:
    session = SimpleNamespace(get=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be used")))
    record = extractor.hydrate_resource_record(
        "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-january-2021",
        session=session,
    )
    assert record.pdf_url == "https://www.aidsdatahub.org/sites/default/files/resource/eb-harp-january-aidsreg2021.pdf"
    assert record.resource_kind == "harp_registry"


def test_download_pdf_reuses_cached_artifact_on_network_failure(monkeypatch) -> None:
    resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/philippines-country-data-2023",
        title="Philippines Country Data 2023",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/phillippines-data-book-2023.pdf",
        release_year=2023,
        resource_kind="annual_country_summary",
    )
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp/aidsdatahub_download_cache_pytest")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cached_pdf = tmp_path / "cached.pdf"
    cached_pdf.write_bytes(b"%PDF-1.4 cached")

    class _FailingSession:
        def get(self, *args, **kwargs):
            raise extractor.requests.ConnectionError("offline")

    monkeypatch.setattr(extractor, "_find_cached_download", lambda target_name: cached_pdf)
    local_path = extractor._download_pdf(resource, session=_FailingSession(), download_dir=tmp_path / "downloads", refresh=True)
    assert local_path is not None
    assert local_path.read_bytes() == b"%PDF-1.4 cached"


def test_country_anchor_rows_extract_core_series() -> None:
    sample = """
    PHILIPPINES | COUNTRY DATA
    New HIV infections
    New HIV infections (all ages)
    4700
    10 000
    24 000
    AIDS-related deaths
    AIDS-related deaths (all ages)
    <500
    <500
    1500
    People living with HIV
    People living with HIV (all ages)
    17 000
    53 000
    160 000
    """

    rows = extractor._country_anchor_rows(
        sample,
        release_year=2023,
        source_id="aidsdatahub_2023",
        source_label="UNAIDS DATA 2023",
        source_url="https://www.aidsdatahub.org/sites/default/files/resource/phillippines-data-book-2023.pdf",
    )

    assert {(row["metric_name"], row["year"]) for row in rows} >= {
        ("annual_new_infections", 2010),
        ("annual_new_infections", 2015),
        ("annual_new_infections", 2022),
        ("annual_aids_deaths", 2010),
        ("estimated_plhiv", 2022),
    }
    upper_bound_rows = [row for row in rows if row["metric_name"] == "annual_aids_deaths" and row["year"] == 2010]
    assert upper_bound_rows
    assert upper_bound_rows[0]["value"] == 500.0
    assert upper_bound_rows[0]["value_semantics"] == "upper_bound"


def test_country_snapshot_card_rows_extract_core_metrics() -> None:
    sample = """
    PHILIPPINES
    COUNTRY SNAPSHOT, 2018
    People living with HIV (PLHIV)
    68 000
    Low estimate
    61 000
    High estimate 76 000
    New HIV infections
    12 000
    Low estimate
    11 000
    High estimate 13 000
    People on ART (2017)
    24 754
    AIDS-related deaths
    760
    Low estimate
    510
    High estimate 1 000
    TREATMENT CASCADE, 2017
    68,000
    68,000
    48,259
    24,754
    6,230
    5,873
    0
    """

    rows = extractor._country_snapshot_card_rows(
        sample,
        release_year=2019,
        source_id="aidsdatahub_2019_card",
        source_label="Country Snapshot 2019 Philippines",
        source_url="https://www.aidsdatahub.org/resource/philippines-country-snapshot-2019",
    )

    lookup = {(row["metric_name"], row["year"], row["value_semantics"]): row["value"] for row in rows}
    assert lookup[("estimated_plhiv", 2018, "point_estimate")] == 68000.0
    assert lookup[("annual_new_infections", 2018, "point_estimate")] == 12000.0
    assert lookup[("annual_aids_deaths", 2018, "point_estimate")] == 760.0
    assert lookup[("alive_on_art", 2017, "point_estimate")] == 24754.0
    assert lookup[("diagnosed_plhiv", 2017, "point_estimate")] == 48259.0
    assert lookup[("tested_for_viral_load", 2017, "point_estimate")] == 6230.0
    assert lookup[("virally_suppressed", 2017, "point_estimate")] == 5873.0



def test_technical_report_year_series_rows_extracts_table_series() -> None:
    sample = """
    Table 6. Proportion of HRG Per Site Who Knew of Three Correct Ways of Preventing HIV Transmission, BSS, 1997-2003
    Age 1997 1998 1999 2000 2001 2002 2003
    RFSW 56 54 60 59 61 61 55
    FLSW 25 27 27 32 30 30 26
    MSM 1 4 6 1 61 21 61
    Figure 7. Ignore Me
    >50 40-49 30-39 20-29 10-19 <10 No age reported
    """

    rows = extractor._technical_report_year_series_rows(
        sample,
        source_id="aidsdatahub_tech_2003",
        source_label="Philippines HIV/AIDS Surveillance Technical Report 2003",
        source_url="https://www.aidsdatahub.org/sites/default/files/resource/philippines-hiv-aids-surveillance-technical-report-2003.pdf",
    )

    assert rows
    assert {row["subgroup"] for row in rows} >= {"RFSW", "FLSW", "MSM"}
    assert all(row["metric_name"].startswith("technical_report_") for row in rows)
    assert not any(row["subgroup"].startswith(">") for row in rows)


def test_run_aidsdatahub_extract_writes_panel(monkeypatch) -> None:
    annual_resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/philippines-country-data-2023",
        title="Philippines Country Data 2023",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/phillippines-data-book-2023.pdf",
        release_year=2023,
        resource_kind="annual_country_summary",
    )
    harp_resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-january-2021",
        title="HIV/AIDS and ART Registry of the Philippines: January 2021",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/hiv-aids-and-art-registry-philippines-january-2021.pdf",
        release_year=2021,
        resource_kind="harp_registry",
    )
    run_dir = Path("D:/EpiGraph_PH/artifacts/test_tmp/aidsdatahub_extract_pytest")
    shutil.rmtree(run_dir, ignore_errors=True)

    monkeypatch.setattr(
        extractor.RunContext,
        "create",
        classmethod(lambda cls, *, run_id, plugin_id: SimpleNamespace(run_id=run_id, plugin_id=plugin_id, run_dir=run_dir)),
    )
    monkeypatch.setattr(
        extractor,
        "discover_philippines_resources",
        lambda **kwargs: [annual_resource.detail_url, harp_resource.detail_url],
    )
    monkeypatch.setattr(
        extractor,
        "hydrate_resource_record",
        lambda detail_url, **kwargs: annual_resource if detail_url == annual_resource.detail_url else harp_resource,
    )

    def _fake_download(resource, *, session, download_dir, refresh):
        path = Path(download_dir) / f"{extractor._safe_ascii_label(resource.title)}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return path

    monkeypatch.setattr(extractor, "_download_pdf", _fake_download)
    monkeypatch.setattr(
        extractor,
        "_pdf_text",
        lambda *args, **kwargs: """
        New HIV infections (all ages) 4700 10 000 24 000
        AIDS-related deaths (all ages) <500 <500 1500
        People living with HIV (all ages) 17 000 53 000 160 000
        """,
    )
    monkeypatch.setattr(extractor, "extract_archive_pdf_pages", lambda *args, **kwargs: [{"page_number": 1, "text": "synthetic"}])
    monkeypatch.setattr(
        extractor,
        "extract_archive_metric_rows",
        lambda report_row, page_texts: [
            {
                "year": 2021,
                "time": "2021-01",
                "metric_name": "diagnosed_plhiv",
                "value": 88964.0,
                "unit": "count_people",
                "source_id": report_row["source_id"],
                "source_label": report_row["label"],
                "measurement_class": "program_observed_harp",
                "series_kind": "monthly_snapshot",
                "geo": "Philippines",
                "region": "national",
                "province": "Philippines",
            }
        ],
    )

    outputs = extractor.run_aidsdatahub_philippines_extract(run_id="pytest-aidsdatahub", plugin_id="hiv")
    rows = extractor.read_json(Path(outputs["historical_metric_rows_path"]), default=[])
    manifest = extractor.read_json(Path(outputs["manifest_path"]), default={})

    assert len(rows) >= 7
    assert any(row["metric_name"] == "annual_new_infections" for row in rows)
    assert any(row["metric_name"] == "diagnosed_plhiv" for row in rows)
    assert manifest["metric_row_count"] == len(rows)
    assert Path(outputs["historical_metric_rows_csv_path"]).exists()


def test_phase0_ocr_fallback_recovers_unsupported_surveillance_pdf(monkeypatch) -> None:
    resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/philippines-hiv-aids-surveillance-technical-report-2003",
        title="Philippines HIV/AIDS Surveillance Technical Report 2003",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/philippines-hiv-aids-surveillance-technical-report-2003.pdf",
        release_year=2003,
        resource_kind="surveillance_technical_report",
    )
    run_dir = Path("D:/EpiGraph_PH/artifacts/test_tmp/aidsdatahub_extract_phase0_fallback_pytest")
    shutil.rmtree(run_dir, ignore_errors=True)

    monkeypatch.setattr(
        extractor.RunContext,
        "create",
        classmethod(lambda cls, *, run_id, plugin_id: SimpleNamespace(run_id=run_id, plugin_id=plugin_id, run_dir=run_dir)),
    )
    monkeypatch.setattr(extractor, "discover_philippines_resources", lambda **kwargs: [resource.detail_url])
    monkeypatch.setattr(extractor, "hydrate_resource_record", lambda detail_url, **kwargs: resource)

    def _fake_download(current, *, session, download_dir, refresh):
        path = Path(download_dir) / f"{extractor._safe_ascii_label(current.title)}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return path

    monkeypatch.setattr(extractor, "_download_pdf", _fake_download)
    monkeypatch.setattr(extractor, "_phase0_ocr_fallback_blocks", lambda *args, **kwargs: ([{"page_number": 1, "text": "diagnosed cases 1993 42"}], {"status": "parsed", "backend": "auto", "page_count": 1, "parsed_page_count": 1, "failure_notes": []}))
    monkeypatch.setattr(extractor, "extract_archive_metric_rows", lambda report_row, page_texts: [{"year": 2003, "time": "2003-12", "metric_name": "diagnosed_plhiv", "value": 42.0, "unit": "count_people", "source_id": report_row["source_id"], "source_label": report_row["label"], "measurement_class": "program_observed_harp", "series_kind": "annual_snapshot", "geo": "Philippines", "region": "national", "province": "Philippines"}])

    outputs = extractor.run_aidsdatahub_philippines_extract(
        run_id="pytest-aidsdatahub-phase0-fallback",
        plugin_id="hiv",
        use_phase0_ocr_fallback=True,
    )
    rows = extractor.read_json(Path(outputs["historical_metric_rows_path"]), default=[])
    fallback_manifest = extractor.read_json(Path(outputs["phase0_ocr_fallback_manifest_path"]), default=[])

    assert any(row["metric_name"] == "diagnosed_plhiv" for row in rows)
    assert fallback_manifest
    assert fallback_manifest[0]["fallback_status"] == "parsed"
    assert fallback_manifest[0]["fallback_row_count"] == 1


def test_text_rich_technical_report_skips_phase0_ocr_fallback(monkeypatch) -> None:
    resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/philippines-hiv-aids-surveillance-technical-report-2003",
        title="Philippines HIV/AIDS Surveillance Technical Report 2003",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/philippines-hiv-aids-surveillance-technical-report-2003.pdf",
        release_year=2003,
        resource_kind="surveillance_technical_report",
    )
    run_dir = Path("D:/EpiGraph_PH/artifacts/test_tmp/aidsdatahub_extract_text_rich_skip_pytest")
    shutil.rmtree(run_dir, ignore_errors=True)

    monkeypatch.setattr(
        extractor.RunContext,
        "create",
        classmethod(lambda cls, *, run_id, plugin_id: SimpleNamespace(run_id=run_id, plugin_id=plugin_id, run_dir=run_dir)),
    )
    monkeypatch.setattr(extractor, "discover_philippines_resources", lambda **kwargs: [resource.detail_url])
    monkeypatch.setattr(extractor, "hydrate_resource_record", lambda detail_url, **kwargs: resource)

    def _fake_download(current, *, session, download_dir, refresh):
        path = Path(download_dir) / f"{extractor._safe_ascii_label(current.title)}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return path

    monkeypatch.setattr(extractor, "_download_pdf", _fake_download)
    monkeypatch.setattr(extractor, "_pdf_text", lambda *args, **kwargs: "Table 1 " + ("Alpha beta gamma " * 120))
    monkeypatch.setattr(extractor, "_technical_report_year_series_rows", lambda *args, **kwargs: [])

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError("Phase 0 fallback should not run for text-rich technical reports")

    monkeypatch.setattr(extractor, "_phase0_ocr_fallback_blocks", _unexpected_fallback)

    outputs = extractor.run_aidsdatahub_philippines_extract(
        run_id="pytest-aidsdatahub-text-rich-skip",
        plugin_id="hiv",
        use_phase0_ocr_fallback=True,
    )
    fallback_manifest = extractor.read_json(Path(outputs["phase0_ocr_fallback_manifest_path"]), default=[])

    assert fallback_manifest
    assert fallback_manifest[0]["fallback_status"] == "skipped_text_rich_native_first"


def test_annual_country_summary_fallback_uses_snapshot_parser(monkeypatch) -> None:
    resource = extractor.ResourceRecord(
        detail_url="https://www.aidsdatahub.org/resource/philippines-country-snapshot-2019",
        title="Country Snapshot 2019 Philippines",
        pdf_url="https://www.aidsdatahub.org/sites/default/files/resource/philippines-country-snapshot-2019.pdf",
        release_year=2019,
        resource_kind="annual_country_summary",
    )
    run_dir = Path("D:/EpiGraph_PH/artifacts/test_tmp/aidsdatahub_extract_country_card_fallback_pytest")
    shutil.rmtree(run_dir, ignore_errors=True)

    monkeypatch.setattr(
        extractor.RunContext,
        "create",
        classmethod(lambda cls, *, run_id, plugin_id: SimpleNamespace(run_id=run_id, plugin_id=plugin_id, run_dir=run_dir)),
    )
    monkeypatch.setattr(extractor, "discover_philippines_resources", lambda **kwargs: [resource.detail_url])
    monkeypatch.setattr(extractor, "hydrate_resource_record", lambda detail_url, **kwargs: resource)

    def _fake_download(current, *, session, download_dir, refresh):
        path = Path(download_dir) / f"{extractor._safe_ascii_label(current.title)}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return path

    monkeypatch.setattr(extractor, "_download_pdf", _fake_download)
    monkeypatch.setattr(extractor, "_pdf_text", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        extractor,
        "_phase0_ocr_fallback_blocks",
        lambda *args, **kwargs: (
            [
                {
                    "page_number": 1,
                    "text": "COUNTRY SNAPSHOT, 2018 People living with HIV (PLHIV) 68 000 Low estimate 61 000 High estimate 76 000 "
                    "New HIV infections 12 000 Low estimate 11 000 High estimate 13 000 AIDS-related deaths 760 Low estimate 510 High estimate 1 000",
                }
            ],
            {"status": "parsed", "backend": "lighton_local", "page_count": 1, "parsed_page_count": 1, "failure_notes": []},
        ),
    )

    outputs = extractor.run_aidsdatahub_philippines_extract(
        run_id="pytest-aidsdatahub-country-card-fallback",
        plugin_id="hiv",
        use_phase0_ocr_fallback=True,
    )
    rows = extractor.read_json(Path(outputs["historical_metric_rows_path"]), default=[])

    assert any(row["metric_name"] == "estimated_plhiv" and row["year"] == 2018 for row in rows)
