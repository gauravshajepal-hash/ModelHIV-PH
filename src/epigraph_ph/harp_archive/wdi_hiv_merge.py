from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, utc_now_iso, write_json

_OVERLAP_VALIDATED_CODE_MAP: dict[str, dict[str, str]] = {
    "SH.HIV.INCD.TL": {
        "metric_name": "annual_new_infections",
        "integration_role": "canonical_metric_external_reference",
    },
    "SH.HIV.ARTC.ZS": {
        "metric_name": "art_coverage_percent",
        "integration_role": "derived_metric_external_reference",
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_harp_rows(archive_run_id: str) -> list[dict[str, Any]]:
    path = ROOT_DIR / "artifacts" / "runs" / archive_run_id / "harp_archive" / "historical_metric_rows.json"
    return list(_read_json(path))


def _load_harp_panel_rows(archive_run_id: str) -> list[dict[str, Any]]:
    path = ROOT_DIR / "artifacts" / "runs" / archive_run_id / "harp_archive" / "historical_harp_panel.json"
    payload = dict(_read_json(path))
    return list(payload.get("rows", []))


def _load_wdi_rows(wdi_run_id: str) -> list[dict[str, Any]]:
    path = ROOT_DIR / "artifacts" / "runs" / wdi_run_id / "harp_archive_wdi" / "wdi_hiv_metric_rows.json"
    return list(_read_json(path))


def _art_coverage_from_harp_panel(panel_rows: list[dict[str, Any]]) -> dict[int, float]:
    coverage: dict[int, float] = {}
    for row in panel_rows:
        year = int(row.get("year") or 0)
        alive = row.get("alive_on_art")
        plhiv = row.get("estimated_plhiv")
        if year and alive not in (None, "") and plhiv not in (None, "", 0):
            coverage[year] = 100.0 * float(alive) / float(plhiv)
    return coverage


def _annual_new_infections_from_harp_rows(harp_rows: list[dict[str, Any]]) -> dict[int, float]:
    by_year: dict[int, set[float]] = {}
    for row in harp_rows:
        if row.get("province") != "Philippines":
            continue
        if row.get("metric_name") != "annual_new_infections":
            continue
        year = int(row.get("year") or 0)
        if not year:
            continue
        by_year.setdefault(year, set()).add(float(row["value"]))
    return {year: next(iter(values)) for year, values in by_year.items() if len(values) == 1}


def _overlap_summary(
    *,
    harp_rows: list[dict[str, Any]],
    harp_panel_rows: list[dict[str, Any]],
    wdi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    wdi_by_code_year: dict[tuple[str, int], float] = {}
    for row in wdi_rows:
        wdi_by_code_year[(str(row["series_code"]), int(row["year"]))] = float(row["value"])

    annual_infections = _annual_new_infections_from_harp_rows(harp_rows)
    art_coverage = _art_coverage_from_harp_panel(harp_panel_rows)

    infections_rows: list[dict[str, Any]] = []
    infection_abs_pct_errors: list[float] = []
    for year in sorted(set(annual_infections) & {year for code, year in wdi_by_code_year if code == "SH.HIV.INCD.TL"}):
        w = wdi_by_code_year[("SH.HIV.INCD.TL", year)]
        h = annual_infections[year]
        delta = w - h
        pct = (delta / h * 100.0) if h else None
        if pct is not None:
            infection_abs_pct_errors.append(abs(pct))
        infections_rows.append({"year": year, "wdi_value": w, "harp_value": h, "delta": delta, "delta_pct": pct})

    art_rows: list[dict[str, Any]] = []
    art_abs_pp_errors: list[float] = []
    for year in sorted(set(art_coverage) & {year for code, year in wdi_by_code_year if code == "SH.HIV.ARTC.ZS"}):
        w = wdi_by_code_year[("SH.HIV.ARTC.ZS", year)]
        h = art_coverage[year]
        delta = w - h
        art_abs_pp_errors.append(abs(delta))
        art_rows.append({"year": year, "wdi_value": w, "harp_derived_value": h, "delta_pp": delta})

    return {
        "annual_new_infections_overlap": {
            "series_code": "SH.HIV.INCD.TL",
            "harp_metric_name": "annual_new_infections",
            "comparison_rows": infections_rows,
            "mean_absolute_percent_error": (sum(infection_abs_pct_errors) / len(infection_abs_pct_errors)) if infection_abs_pct_errors else None,
        },
        "art_coverage_overlap": {
            "series_code": "SH.HIV.ARTC.ZS",
            "harp_derived_metric_name": "alive_on_art_over_estimated_plhiv_percent",
            "comparison_rows": art_rows,
            "mean_absolute_percentage_point_error": (sum(art_abs_pp_errors) / len(art_abs_pp_errors)) if art_abs_pp_errors else None,
        },
    }


def _validated_series_codes(overlap_summary: dict[str, Any]) -> set[str]:
    validated: set[str] = set()
    annual_rows = list(dict(overlap_summary.get("annual_new_infections_overlap") or {}).get("comparison_rows") or [])
    if annual_rows:
        validated.add("SH.HIV.INCD.TL")
    art_rows = list(dict(overlap_summary.get("art_coverage_overlap") or {}).get("comparison_rows") or [])
    if art_rows:
        validated.add("SH.HIV.ARTC.ZS")
    return validated


def _merged_wdi_rows(wdi_rows: list[dict[str, Any]], *, validated_codes: set[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in wdi_rows:
        code = str(row["series_code"])
        mapped = _OVERLAP_VALIDATED_CODE_MAP.get(code)
        payload = dict(row)
        payload["source_family"] = "wdi_unaids_hiv"
        if mapped is not None:
            payload["metric_name"] = mapped["metric_name"]
            if code in validated_codes:
                payload["source_tier"] = "overlap_validated_external_reference"
                payload["integration_role"] = f"{mapped['integration_role']}_overlap_validated"
                payload["overlap_validation_status"] = "validated_against_existing_harp_overlap"
            else:
                payload["source_tier"] = "canonical_external_reference_no_local_overlap"
                payload["integration_role"] = f"{mapped['integration_role']}_no_local_overlap"
                payload["overlap_validation_status"] = "no_overlap_rows_in_target_archive"
        else:
            payload["source_tier"] = "reference_only_external_series"
            payload["integration_role"] = "standalone_reference_series"
            payload["overlap_validation_status"] = "no_direct_harp_counterpart"
        payload["measurement_class"] = "external_reference_wdi"
        payload["source_quality_tier"] = "official_wdi_unaids_hiv_series"
        merged.append(payload)
    merged.sort(key=lambda item: (str(item["metric_name"]), int(item["year"]), str(item["source_id"])))
    return merged


def merge_wdi_hiv_rows_with_harp(
    *,
    harp_rows: list[dict[str, Any]],
    harp_panel_rows: list[dict[str, Any]],
    wdi_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    overlap_summary = _overlap_summary(
        harp_rows=harp_rows,
        harp_panel_rows=harp_panel_rows,
        wdi_rows=wdi_rows,
    )
    validated_codes = _validated_series_codes(overlap_summary)
    merged_wdi_rows = _merged_wdi_rows(wdi_rows, validated_codes=validated_codes)
    return merged_wdi_rows, overlap_summary, validated_codes


def run_harp_archive_merge_wdi_hiv(
    *,
    run_id: str,
    plugin_id: str = "hiv",
    archive_run_id: str,
    wdi_run_id: str,
) -> dict[str, Any]:
    harp_rows = _load_harp_rows(archive_run_id)
    harp_panel_rows = _load_harp_panel_rows(archive_run_id)
    wdi_rows = _load_wdi_rows(wdi_run_id)

    merged_wdi_rows, overlap_summary, validated_codes = merge_wdi_hiv_rows_with_harp(
        harp_rows=harp_rows,
        harp_panel_rows=harp_panel_rows,
        wdi_rows=wdi_rows,
    )
    combined_rows = list(harp_rows) + merged_wdi_rows
    combined_rows.sort(
        key=lambda item: (
            str(item.get("metric_name") or ""),
            str(item.get("time") or ""),
            str(item.get("source_id") or ""),
        )
    )

    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    out_dir = ensure_dir(Path(ctx.run_dir) / "harp_archive_wdi_merge")
    merged_rows_json = out_dir / "historical_metric_rows_merged.json"
    merged_rows_csv = out_dir / "historical_metric_rows_merged.csv"
    merged_wdi_json = out_dir / "wdi_hiv_rows_merged.json"
    overlap_json = out_dir / "wdi_hiv_overlap_summary.json"

    write_json(merged_rows_json, combined_rows)
    _write_csv(merged_rows_csv, combined_rows)
    write_json(merged_wdi_json, merged_wdi_rows)
    write_json(overlap_json, overlap_summary)

    manifest = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "generated_at": utc_now_iso(),
        "archive_run_id": archive_run_id,
        "wdi_run_id": wdi_run_id,
        "harp_row_count": len(harp_rows),
        "wdi_row_count": len(wdi_rows),
        "merged_wdi_row_count": len(merged_wdi_rows),
        "combined_row_count": len(combined_rows),
        "overlap_validated_series_codes": sorted(validated_codes),
        "canonical_reference_without_local_overlap_series_codes": sorted(
            {str(row["series_code"]) for row in merged_wdi_rows if str(row["source_tier"]) == "canonical_external_reference_no_local_overlap"}
        ),
        "reference_only_series_codes": sorted(
            {str(row["series_code"]) for row in merged_wdi_rows if str(row["source_tier"]) == "reference_only_external_series"}
        ),
        "artifact_paths": {
            "historical_metric_rows_merged_json": str(merged_rows_json),
            "historical_metric_rows_merged_csv": str(merged_rows_csv),
            "wdi_hiv_rows_merged_json": str(merged_wdi_json),
            "wdi_hiv_overlap_summary_json": str(overlap_json),
        },
    }
    write_json(out_dir / "wdi_hiv_merge_manifest.json", manifest)
    return manifest
