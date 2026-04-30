from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from openpyxl import Workbook

from epigraph_ph.cli.main import build_parser
from epigraph_ph.harp_archive.wdi_hiv_import import extract_wdi_hiv_rows, run_harp_archive_wdi_hiv_extract
from epigraph_ph.runtime import ROOT_DIR, read_json


def _build_sample_wdi_workbook(path: Path) -> Path:
    workbook = Workbook()
    data_sheet = workbook.active
    data_sheet.title = "Data"
    data_sheet.append(
        [
            "Series Name",
            "Series Code",
            "Country Name",
            "Country Code",
            "2021 [YR2021]",
            "2022 [YR2022]",
        ]
    )
    data_sheet.append(
        [
            "Children (ages 0-14) newly infected with HIV",
            "SH.HIV.INCD.14",
            "Philippines",
            "PHL",
            500,
            600,
        ]
    )
    data_sheet.append(
        [
            "Antiretroviral therapy coverage (% of people living with HIV)",
            "SH.HIV.ARTC.ZS",
            "Philippines",
            "PHL",
            35.0,
            38.5,
        ]
    )
    data_sheet.append(
        [
            "GDP (current US$)",
            "NY.GDP.MKTP.CD",
            "Philippines",
            "PHL",
            100.0,
            200.0,
        ]
    )

    metadata_sheet = workbook.create_sheet("Series - Metadata")
    metadata_sheet.append(
        [
            "Code",
            "License Type",
            "Indicator Name",
            "Long definition",
            "Source",
            "Topic",
            "Dataset",
            "Unit of measure",
            "Periodicity",
        ]
    )
    metadata_sheet.append(
        [
            "SH.HIV.INCD.14",
            "CC BY-4.0",
            "Children (ages 0-14) newly infected with HIV",
            "Number of children newly infected with HIV.",
            "UNAIDS estimates",
            "Health: Risk factors",
            "WB_WDI",
            "Number",
            "Annual",
        ]
    )
    metadata_sheet.append(
        [
            "SH.HIV.ARTC.ZS",
            "CC BY-4.0",
            "Antiretroviral therapy coverage (% of people living with HIV)",
            "ART coverage.",
            "UNAIDS estimates",
            "Health: Risk factors",
            "WB_WDI",
            "%",
            "Annual",
        ]
    )
    metadata_sheet.append(
        [
            "NY.GDP.MKTP.CD",
            "CC BY-4.0",
            "GDP (current US$)",
            "GDP",
            "World Bank",
            "Economy",
            "WB_WDI",
            "Number",
            "Annual",
        ]
    )
    workbook.save(path)
    return path


def _workspace_tmp_dir(stem: str) -> Path:
    path = ROOT_DIR / "tmp" / f"{stem}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_extract_wdi_hiv_rows_filters_to_hiv_series() -> None:
    tmp_dir = _workspace_tmp_dir("pytest-wdi-import")
    workbook_path = _build_sample_wdi_workbook(tmp_dir / "sample_wdi.xlsx")

    try:
        metric_rows, series_rows = extract_wdi_hiv_rows(workbook_path)

        assert len(metric_rows) == 4
        assert len(series_rows) == 2
        assert {row["series_code"] for row in series_rows} == {"SH.HIV.INCD.14", "SH.HIV.ARTC.ZS"}
        assert {row["metric_name"] for row in metric_rows} == {"wdi_sh_hiv_incd_14", "wdi_sh_hiv_artc_zs"}
        assert {row["unit"] for row in metric_rows if row["series_code"] == "SH.HIV.INCD.14"} == {"count_people"}
        assert {row["unit"] for row in metric_rows if row["series_code"] == "SH.HIV.ARTC.ZS"} == {"percent"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_harp_archive_wdi_hiv_extract_writes_artifacts() -> None:
    tmp_dir = _workspace_tmp_dir("pytest-wdi-import")
    workbook_path = _build_sample_wdi_workbook(tmp_dir / "sample_wdi.xlsx")
    run_id = "pytest-harp-archive-wdi-import"
    run_dir = ROOT_DIR / "artifacts" / "runs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)

    try:
        manifest = run_harp_archive_wdi_hiv_extract(
            run_id=run_id,
            plugin_id="hiv",
            workbook_path=workbook_path,
        )

        artifact_dir = run_dir / "harp_archive_wdi"
        assert manifest["metric_row_count"] == 4
        assert manifest["series_count"] == 2
        assert (artifact_dir / "wdi_hiv_metric_rows.json").exists()
        assert (artifact_dir / "wdi_hiv_series_inventory.json").exists()
        persisted_manifest = read_json(artifact_dir / "wdi_hiv_manifest.json", default={})
        assert persisted_manifest["metric_row_count"] == 4
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_parser_accepts_harp_archive_wdi_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["harp-archive", "extract-wdi-hiv", "--run-id", "pytest-run"])

    assert args.command == "harp-archive"
    assert args.harp_archive_command == "extract-wdi-hiv"
    assert args.country_code == "PHL"
