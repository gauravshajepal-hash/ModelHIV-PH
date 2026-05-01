from __future__ import annotations

import shutil
from pathlib import Path

from epigraph_ph.cli.main import build_parser
from epigraph_ph.harp_archive.wdi_hiv_merge import run_harp_archive_merge_wdi_hiv
from epigraph_ph.runtime import ROOT_DIR, read_json, write_json


def test_run_harp_archive_merge_wdi_hiv_writes_tiered_rows() -> None:
    archive_run_id = "pytest-harp-wdi-merge-source-archive"
    wdi_run_id = "pytest-harp-wdi-merge-source-wdi"
    target_run_id = "pytest-harp-wdi-merge-target"

    archive_dir = ROOT_DIR / "artifacts" / "runs" / archive_run_id / "harp_archive"
    wdi_dir = ROOT_DIR / "artifacts" / "runs" / wdi_run_id / "harp_archive_wdi"
    target_dir = ROOT_DIR / "artifacts" / "runs" / target_run_id

    shutil.rmtree(archive_dir.parent.parent, ignore_errors=True)
    shutil.rmtree(wdi_dir.parent.parent, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    wdi_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        archive_dir / "historical_metric_rows.json",
        [
            {
                "metric_name": "annual_new_infections",
                "year": 2021,
                "time": "2021-01",
                "value": 20900.0,
                "province": "Philippines",
                "source_id": "core_team_2025",
            }
        ],
    )
    write_json(
        archive_dir / "historical_harp_panel.json",
        {
            "rows": [
                {
                    "year": 2021,
                    "time": "2021-01",
                    "alive_on_art": 51863.0,
                    "estimated_plhiv": 142000.0,
                }
            ]
        },
    )
    write_json(
        wdi_dir / "wdi_hiv_metric_rows.json",
        [
            {
                "metric_name": "wdi_sh_hiv_incd_tl",
                "series_code": "SH.HIV.INCD.TL",
                "series_name": "Adults (ages 15+) and children (ages 0-14) newly infected with HIV",
                "year": 2021,
                "time": "2021-01",
                "value": 21000.0,
                "unit": "count_people",
                "source_id": "wdi_sh_hiv_incd_tl",
                "source_label": "World Development Indicators HIV Series",
            },
            {
                "metric_name": "wdi_sh_hiv_0014",
                "series_code": "SH.HIV.0014",
                "series_name": "Children (0-14) living with HIV",
                "year": 2021,
                "time": "2021-01",
                "value": 1000.0,
                "unit": "count_people",
                "source_id": "wdi_sh_hiv_0014",
                "source_label": "World Development Indicators HIV Series",
            },
        ],
    )

    try:
        manifest = run_harp_archive_merge_wdi_hiv(
            run_id=target_run_id,
            plugin_id="hiv",
            archive_run_id=archive_run_id,
            wdi_run_id=wdi_run_id,
        )

        out_dir = ROOT_DIR / "artifacts" / "runs" / target_run_id / "harp_archive_wdi_merge"
        merged_rows = read_json(out_dir / "wdi_hiv_rows_merged.json", default=[])
        overlap = read_json(out_dir / "wdi_hiv_overlap_summary.json", default={})

        assert manifest["merged_wdi_row_count"] == 2
        overlap_row = next(row for row in merged_rows if row["series_code"] == "SH.HIV.INCD.TL")
        reference_row = next(row for row in merged_rows if row["series_code"] == "SH.HIV.0014")
        assert overlap_row["metric_name"] == "annual_new_infections"
        assert overlap_row["source_tier"] == "overlap_validated_external_reference"
        assert reference_row["source_tier"] == "reference_only_external_series"
        assert overlap["annual_new_infections_overlap"]["comparison_rows"][0]["year"] == 2021
    finally:
        shutil.rmtree(archive_dir.parent.parent, ignore_errors=True)
        shutil.rmtree(wdi_dir.parent.parent, ignore_errors=True)
        shutil.rmtree(target_dir, ignore_errors=True)


def test_cli_parser_accepts_harp_archive_merge_wdi_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "harp-archive",
            "merge-wdi-hiv",
            "--run-id",
            "pytest-run",
            "--archive-run-id",
            "archive-source",
            "--wdi-run-id",
            "wdi-source",
        ]
    )

    assert args.command == "harp-archive"
    assert args.harp_archive_command == "merge-wdi-hiv"
    assert args.archive_run_id == "archive-source"
    assert args.wdi_run_id == "wdi-source"
