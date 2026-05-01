from __future__ import annotations

import csv
import shutil
from pathlib import Path
from uuid import uuid4

import epigraph_ph.harp_archive.pipeline as pipeline_module
from epigraph_ph.harp_archive.multinational_hiv_import import (
    extract_multinational_hiv_rows_from_sources,
    materialize_multinational_hiv_sources,
)
from epigraph_ph.harp_archive.pipeline import run_harp_archive_build
from epigraph_ph.runtime import ROOT_DIR, read_json


def _workspace_tmp_dir(stem: str) -> Path:
    path = ROOT_DIR / "tmp" / f"{stem}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_simple_panel(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_extract_multinational_hiv_rows_from_sources_derives_counts() -> None:
    tmp_dir = _workspace_tmp_dir("pytest-harp-multinational-import")
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        estimated_path = raw_dir / "estimated.csv"
        _write_simple_panel(
            estimated_path,
            ["Country", "2020", "2021"],
            [["Philippines", "100000", "110000"]],
        )
        art_path = raw_dir / "art.csv"
        _write_simple_panel(
            art_path,
            ["Country", "2020", "2021"],
            [["Philippines", "20000", "25000"]],
        )
        diagnosed_share_path = raw_dir / "diagnosed_share.csv"
        _write_simple_panel(
            diagnosed_share_path,
            ["Country", "2020", "2021"],
            [["Philippines", "50", "60"]],
        )
        suppressed_share_path = raw_dir / "suppressed_share.csv"
        _write_simple_panel(
            suppressed_share_path,
            ["Country", "2020", "2021"],
            [["Philippines", "20", "30"]],
        )

        source_rows = [
            {
                "source_id": "unaids_estimated_plhiv_all_ages",
                "label": "Estimated PLHIV",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(estimated_path),
            },
            {
                "source_id": "unaids_alive_on_art_all_ages",
                "label": "Alive on ART",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(art_path),
            },
            {
                "source_id": "unaids_known_status_share_all_ages",
                "label": "Known status share",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(diagnosed_share_path),
            },
            {
                "source_id": "unaids_suppressed_share_all_ages",
                "label": "Suppressed share",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(suppressed_share_path),
            },
        ]

        metric_rows, inventory_rows = extract_multinational_hiv_rows_from_sources(source_rows)

        assert inventory_rows
        value_lookup = {
            (str(row["metric_name"]), int(row["year"])): float(row["value"])
            for row in metric_rows
        }
        assert value_lookup[("estimated_plhiv", 2020)] == 100000.0
        assert value_lookup[("alive_on_art", 2021)] == 25000.0
        assert value_lookup[("diagnosed_plhiv", 2020)] == 50000.0
        assert value_lookup[("diagnosed_plhiv", 2021)] == 66000.0
        assert value_lookup[("virally_suppressed", 2020)] == 20000.0
        assert value_lookup[("virally_suppressed", 2021)] == 33000.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_harp_archive_build_ingests_multinational_hiv_sources(monkeypatch) -> None:
    tmp_dir = _workspace_tmp_dir("pytest-harp-multinational-build")
    data_dir = tmp_dir / "HIV_Data"
    run_id = "pytest-harp-archive-multinational-import"
    run_dir = ROOT_DIR / "artifacts" / "runs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)

    try:
        _write_simple_panel(
            data_dir / "People living with HIV_People living with HIV - All ages_Population_ All.csv",
            ["Country", "2020", "2021"],
            [["Philippines", "100000", "110000"]],
        )
        _write_simple_panel(
            data_dir / "Treatment cascade_People living with HIV receiving ART (#)_Population_ All ages.csv",
            ["Country", "2020", "2021"],
            [["Philippines", "20000", "25000"]],
        )
        _write_simple_panel(
            data_dir / "Treatment cascade_People living with HIV who know their status (%)_Population_ All ages.csv",
            ["Country", "2020", "2021"],
            [["Philippines", "50", "60"]],
        )
        _write_simple_panel(
            data_dir / "Treatment cascade_People living with HIV who have suppressed viral loads (%)_Population_ All ages.csv",
            ["Country", "2020", "2021"],
            [["Philippines", "20", "30"]],
        )

        monkeypatch.setattr(pipeline_module, "_materialize_local_sources", lambda *args, **kwargs: [])
        monkeypatch.setattr(pipeline_module, "load_archive_seed_rows", lambda *args, **kwargs: [])
        monkeypatch.setattr(pipeline_module, "download_archive_pdfs", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            "epigraph_ph.harp_archive.multinational_hiv_import.default_multinational_hiv_data_dir",
            lambda: data_dir,
        )
        monkeypatch.setattr(
            pipeline_module,
            "_wdi_hiv_workbook_input",
            lambda: {"path": "", "exists": False, "checksum": ""},
        )

        manifest = run_harp_archive_build(run_id=run_id, plugin_id="hiv")

        archive_dir = run_dir / "harp_archive"
        rows = read_json(archive_dir / "historical_metric_rows_harp_only.json", default=[])
        multi_rows = read_json(archive_dir / "multinational_hiv_metric_rows.json", default=[])
        inventory = read_json(archive_dir / "multinational_hiv_series_inventory.json", default=[])

        assert manifest
        assert multi_rows
        assert inventory
        assert any(str(row.get("metric_name")) == "estimated_plhiv" and int(row.get("year") or 0) == 2020 for row in rows)
        assert any(str(row.get("metric_name")) == "diagnosed_plhiv" and int(row.get("year") or 0) == 2021 for row in rows)
        assert any(str(row.get("metric_name")) == "virally_suppressed" and int(row.get("year") or 0) == 2021 for row in rows)
        assert any(str(row.get("source_id")) == "unaids_derived_diagnosed_plhiv" for row in multi_rows)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_materialize_multinational_hiv_sources_discovers_priority_tier_files(monkeypatch) -> None:
    tmp_dir = _workspace_tmp_dir("pytest-harp-multinational-auto-discovery")
    data_dir = tmp_dir / "HIV_Data"
    run_dir = tmp_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        _write_simple_panel(
            data_dir / "Treatment cascade_Testing and treatment cascade - All ages.csv",
            ["", "2015", "2015", "2015"],
            [
                [
                    "Country",
                    "Percent of people living with HIV who know their status",
                    "Coverage of people living with HIV receiving ART",
                    "Percent of people living with HIV who have suppressed viral loads",
                ],
                ["Philippines", "50", "20", "10"],
            ],
        )
        _write_simple_panel(
            data_dir / "Men who have sex with men_HIV testing and status awareness among men who have sex with men_Population_ Total.csv",
            ["Country", "Most recent data as of 2024", "Most recent data as of 2024_Footnote"],
            [["Philippines", "44.2", "Source: Behavioral surveillance survey, 2024"]],
        )

        monkeypatch.setattr(
            "epigraph_ph.harp_archive.multinational_hiv_import.default_multinational_hiv_data_dir",
            lambda: data_dir,
        )

        source_rows = materialize_multinational_hiv_sources(run_dir)

        source_ids = {str(row.get("source_id") or "") for row in source_rows}
        assert any(source_id.startswith("unaids_auto_treatment_cascade_testing_and_treatment_cascade_all_ages") for source_id in source_ids)
        assert any(source_id.startswith("unaids_auto_men_who_have_sex_with_men_hiv_testing_and_status_awareness") for source_id in source_ids)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_extract_multinational_hiv_rows_from_sources_handles_multi_series_and_latest_value() -> None:
    tmp_dir = _workspace_tmp_dir("pytest-harp-multinational-parser-modes")
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        multi_series_path = raw_dir / "cascade.csv"
        _write_simple_panel(
            multi_series_path,
            ["", "2015", "2015", "2016", "2016"],
            [
                [
                    "Country",
                    "Percent of people living with HIV who know their status",
                    "Coverage of people living with HIV receiving ART",
                    "Percent of people living with HIV who know their status",
                    "Coverage of people living with HIV receiving ART",
                ],
                ["Philippines", "50", "20", "55", "22"],
            ],
        )
        latest_path = raw_dir / "kp.csv"
        _write_simple_panel(
            latest_path,
            ["Country", "Most recent data as of 2024", "Most recent data as of 2024_Footnote"],
            [["Philippines", "44.2", "Source: Behavioral surveillance survey, 2024"]],
        )

        source_rows = [
            {
                "source_id": "unaids_auto_treatment_cascade_testing_and_treatment_cascade_all_ages",
                "label": "UNAIDS Treatment cascade_Testing and treatment cascade - All ages",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(multi_series_path),
                "metric_name": "treatment_cascade_testing_and_treatment_cascade_all_ages",
                "unit": "annual_value",
                "measurement_class": "external_reference_unaids_auto",
                "parse_mode": "auto",
            },
            {
                "source_id": "unaids_auto_msm_testing_status_awareness",
                "label": "UNAIDS MSM HIV testing and status awareness",
                "source_kind": "unaids_multinational_csv",
                "local_path": str(latest_path),
                "metric_name": "men_who_have_sex_with_men_hiv_testing_and_status_awareness_among_men_who_have_sex_with_men_population_total",
                "unit": "percent",
                "measurement_class": "external_reference_unaids_auto",
                "parse_mode": "auto",
            },
        ]

        metric_rows, inventory_rows = extract_multinational_hiv_rows_from_sources(source_rows)

        value_lookup = {(str(row["metric_name"]), int(row["year"])): float(row["value"]) for row in metric_rows}
        assert value_lookup[("treatment_cascade_testing_and_treatment_cascade_all_ages__percent_of_people_living_with_hiv_who_know_their_status", 2015)] == 50.0
        assert value_lookup[("treatment_cascade_testing_and_treatment_cascade_all_ages__coverage_of_people_living_with_hiv_receiving_art", 2016)] == 22.0
        assert value_lookup[("men_who_have_sex_with_men_hiv_testing_and_status_awareness_among_men_who_have_sex_with_men_population_total", 2024)] == 44.2
        assert any(str(row.get("metric_name") or "").startswith("treatment_cascade_testing_and_treatment_cascade_all_ages__") for row in inventory_rows)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
