from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r66_scientific_source_base import R66_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R68_SCHEMA_VERSION = "phase3_dynamic.r68_bulk_external_source_ingest.v1"
R68_RUN_ID = "p3d-r68-bulk-external-source-ingest-20260506-s00"
R66_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R66_RUN_ID
    / "analysis"
    / "r66_scientific_source_base_report.json"
)
DEFAULT_MAX_BYTES = 4 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
PH_TOKENS = {"philippines", "phl"}
csv.field_size_limit(sys.maxsize)


def _bulk_source_specs() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "unaids_estimates_2025",
            "source_family": "unaids_aidsinfo_estimates_bulk_zip",
            "url": "https://aidsinfo.unaids.org/public/documents/Estimates_2025_en.zip",
            "filename": "Estimates_2025_en.zip",
            "allowed_use": "annual_official_hiv_estimates_external_challenge",
            "observation_role": "validation_only",
            "module_targets": "incidence_validation|mortality_reporting|plhiv_stock_validation|annual_challenge",
            "measurement_semantics": "modeled_estimate|stock_anchor|flow_count",
        },
        {
            "source_id": "unaids_gam_2025",
            "source_family": "unaids_gam_bulk_zip",
            "url": "https://aidsinfo.unaids.org/public/documents/GAM_2025_en.zip",
            "filename": "GAM_2025_en.zip",
            "allowed_use": "program_context_auxiliary_or_external_challenge",
            "observation_role": "auxiliary_likelihood",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|prep_persistence|annual_challenge",
            "measurement_semantics": "country_reported|stock_anchor|flow_count|proportion",
        },
        {
            "source_id": "unaids_kp_atlas_2025",
            "source_family": "unaids_key_population_bulk_zip",
            "url": "https://aidsinfo.unaids.org/public/documents/KPAtlasDB_2025_en.zip",
            "filename": "KPAtlasDB_2025_en.zip",
            "allowed_use": "determinant_sensitivity_until_source_stable",
            "observation_role": "prior_context",
            "module_targets": "incidence_pressure|kp_overlay|regional_shrinkage",
            "measurement_semantics": "determinant_covariate|denominator|proportion",
        },
        {
            "source_id": "unaids_ncpi_2025",
            "source_family": "unaids_laws_policies_bulk_zip",
            "url": "https://aidsinfo.unaids.org/public/documents/NCPI_2025_en.zip",
            "filename": "NCPI_2025_en.zip",
            "allowed_use": "determinant_sensitivity_until_source_stable",
            "observation_role": "prior_context",
            "module_targets": "structural_policy|diagnosis_reporting|art_retention|incidence_pressure",
            "measurement_semantics": "determinant_covariate|policy_context",
        },
        {
            "source_id": "google_global_mobility_report",
            "source_family": "google_covid19_mobility_global_csv",
            "url": "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
            "filename": "Global_Mobility_Report.csv",
            "allowed_use": "reporting_disruption_or_mobility_sensitivity_covariate_2020_2022",
            "observation_role": "prior_context",
            "module_targets": "mobility_proxy|reporting_disruption|diagnosis_reporting",
            "measurement_semantics": "reporting_process_covariate|determinant_covariate",
        },
    ]


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


def _download_stream(url: str, path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, timeout_seconds: int = 90) -> dict[str, Any]:
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        return {
            "download_status": "downloaded",
            "content_length": path.stat().st_size,
            "bytes_downloaded": path.stat().st_size,
            "sha256": _sha256(path),
            "error": "reused_existing_download",
        }
    tmp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ModelHIV-PH-R68/1.0"})
    digest = hashlib.sha256()
    bytes_written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, tmp_path.open("wb") as handle:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return {
                    "download_status": "skipped_size_cap",
                    "content_length": int(content_length),
                    "bytes_downloaded": 0,
                    "sha256": None,
                    "error": f"content_length_exceeds_max_bytes:{content_length}>{max_bytes}",
                }
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    return {
                        "download_status": "stopped_size_cap",
                        "content_length": int(content_length) if content_length else None,
                        "bytes_downloaded": bytes_written,
                        "sha256": None,
                        "error": f"bytes_downloaded_exceeds_max_bytes:{bytes_written}>{max_bytes}",
                    }
                digest.update(chunk)
                handle.write(chunk)
        tmp_path.replace(path)
        return {
            "download_status": "downloaded",
            "content_length": path.stat().st_size,
            "bytes_downloaded": bytes_written,
            "sha256": digest.hexdigest(),
            "error": "",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "download_status": "download_failed",
            "content_length": None,
            "bytes_downloaded": bytes_written,
            "sha256": None,
            "error": str(exc),
        }


def _row_is_philippines(row: list[str]) -> bool:
    normalized = {cell.strip().lower() for cell in row if cell is not None}
    return bool(normalized & PH_TOKENS)


def _extract_philippines_from_csv_stream(input_handle: BinaryIO, output_path: Path, *, source_member: str) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    read_rows = 0
    written_rows = 0
    header: list[str] | None = None
    text = io.TextIOWrapper(input_handle, encoding="utf-8-sig", errors="replace", newline="")
    reader = csv.reader(text)
    with output_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        for row in reader:
            read_rows += 1
            if header is None:
                header = row
                writer.writerow(row)
                continue
            if _row_is_philippines(row):
                writer.writerow(row)
                written_rows += 1
    return {
        "source_member": source_member,
        "rows_scanned": read_rows,
        "philippines_rows_written": written_rows,
        "output_path": output_path.as_posix(),
        "output_sha256": _sha256(output_path) if output_path.exists() else None,
        "output_bytes": output_path.stat().st_size if output_path.exists() else None,
    }


def _extract_philippines_from_zip(zip_path: Path, output_dir: Path, *, source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not zip_path.exists() or zip_path.suffix.lower() != ".zip":
        return rows
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                lower = member.filename.lower()
                if member.is_dir() or not lower.endswith(".csv"):
                    rows.append(
                        {
                            "source_member": member.filename,
                            "rows_scanned": 0,
                            "philippines_rows_written": 0,
                            "output_path": "",
                            "output_sha256": None,
                            "output_bytes": None,
                            "skip_reason": "non_csv_member",
                        }
                    )
                    continue
                safe_name = member.filename.replace("/", "__").replace("\\", "__")
                output_path = output_dir / f"{source_id}__{safe_name}"
                with archive.open(member, "r") as handle:
                    extracted = _extract_philippines_from_csv_stream(handle, output_path, source_member=member.filename)
                rows.append({**extracted, "skip_reason": ""})
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError, csv.Error) as exc:
        rows.append(
            {
                "source_member": "",
                "rows_scanned": 0,
                "philippines_rows_written": 0,
                "output_path": "",
                "output_sha256": None,
                "output_bytes": None,
                "skip_reason": f"zip_extract_failed:{exc}",
            }
        )
    return rows


def _extract_google_philippines(global_csv_path: Path, output_path: Path) -> dict[str, Any]:
    if not global_csv_path.exists():
        return {
            "source_member": global_csv_path.name,
            "rows_scanned": 0,
            "philippines_rows_written": 0,
            "output_path": "",
            "output_sha256": None,
            "output_bytes": None,
            "skip_reason": "source_missing",
        }
    with global_csv_path.open("rb") as handle:
        extracted = _extract_philippines_from_csv_stream(handle, output_path, source_member=global_csv_path.name)
    extracted["skip_reason"] = ""
    return extracted


def _ingest_source(spec: dict[str, Any], source_dir: Path, filtered_dir: Path, *, max_bytes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = source_dir / str(spec["source_family"]) / str(spec["filename"])
    download = _download_stream(str(spec["url"]), raw_path, max_bytes=max_bytes)
    extract_rows: list[dict[str, Any]] = []
    if download["download_status"] == "downloaded":
        if raw_path.suffix.lower() == ".zip":
            extract_rows = _extract_philippines_from_zip(raw_path, filtered_dir / str(spec["source_id"]), source_id=str(spec["source_id"]))
        elif str(spec["source_id"]) == "google_global_mobility_report":
            extract_rows = [
                _extract_google_philippines(
                    raw_path,
                    filtered_dir / str(spec["source_id"]) / "google_mobility_philippines.csv",
                )
            ]
    manifest = {
        **spec,
        "raw_path": raw_path.as_posix() if raw_path.exists() else "",
        "raw_sha256": download.get("sha256"),
        "raw_bytes": download.get("bytes_downloaded"),
        "download_status": download.get("download_status"),
        "download_error": download.get("error"),
        "extracted_member_count": len(extract_rows),
        "extracted_philippines_row_count": int(sum(int(row.get("philippines_rows_written") or 0) for row in extract_rows)),
        "extracted_output_bytes": int(sum(int(row.get("output_bytes") or 0) for row in extract_rows)),
        "oom_guard": "stream_to_disk_then_stream_filter; never load global source into memory",
    }
    return manifest, [{**row, "source_id": spec["source_id"], "source_family": spec["source_family"]} for row in extract_rows]


def _coverage_delta_rows(r66_report: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = {str(row.get("module_target") or ""): int(row.get("source_count") or 0) for row in r66_report.get("module_coverage_rows") or []}
    modules = sorted({module for row in manifest_rows for module in str(row.get("module_targets") or "").split("|") if module})
    rows: list[dict[str, Any]] = []
    for module in modules:
        added = [row for row in manifest_rows if module in str(row.get("module_targets") or "").split("|")]
        rows.append(
            {
                "module_target": module,
                "r66_previous_source_count": previous.get(module, 0),
                "r68_added_source_count": len(added),
                "r68_downloaded_count": sum(1 for row in added if row.get("download_status") == "downloaded"),
                "r68_extracted_philippines_row_count": sum(int(row.get("extracted_philippines_row_count") or 0) for row in added),
                "added_source_families": "|".join(sorted({str(row.get("source_family") or "") for row in added})),
            }
        )
    return rows


def _gate(manifest_rows: list[dict[str, Any]], extract_rows: list[dict[str, Any]]) -> dict[str, Any]:
    downloaded = sum(1 for row in manifest_rows if row.get("download_status") == "downloaded")
    failed = sum(1 for row in manifest_rows if row.get("download_status") != "downloaded")
    extracted = sum(int(row.get("philippines_rows_written") or 0) for row in extract_rows)
    return {
        "status": "bulk_external_sources_ingested" if downloaded > 0 and extracted > 0 else "bulk_external_sources_incomplete",
        "downloaded_source_count": downloaded,
        "failed_source_count": failed,
        "extracted_philippines_row_count": extracted,
        "contract": (
            "R68 may download bulk external files, but downstream modeling may use only the extracted Philippines tables "
            "with their allowed-use roles. Validation-only annual estimates remain weak/external evidence, and Google "
            "mobility is a 2020-2022 reporting/mobility covariate, not a transmission truth source."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("bulk_ingest_gate") or {})
    lines = [
        "# Phase 3 R68 Bulk External Source Ingest",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Downloaded sources: `{gate.get('downloaded_source_count')}`",
        f"- Failed/skipped sources: `{gate.get('failed_source_count')}`",
        f"- Extracted Philippines rows: `{gate.get('extracted_philippines_row_count')}`",
        "",
        "## Sources",
        "",
        "| Source | Status | PH rows | Allowed use |",
        "|---|---|---:|---|",
    ]
    for row in report.get("bulk_manifest_rows") or []:
        lines.append(
            f"| `{row.get('source_id')}` | `{row.get('download_status')}` | "
            f"{int(row.get('extracted_philippines_row_count') or 0)} | {row.get('allowed_use')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r68_bulk_external_source_ingest(
    *,
    run_id: str = R68_RUN_ID,
    r66_report_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    source_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "external_sources")
    filtered_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "filtered_philippines")
    r66_path = R66_DEFAULT_REPORT if r66_report_path is None else Path(r66_report_path)
    r66 = dict(read_json(r66_path, default={}) or {}) if r66_path.exists() else {}
    manifest_rows: list[dict[str, Any]] = []
    extract_rows: list[dict[str, Any]] = []
    for spec in _bulk_source_specs():
        manifest, extracted = _ingest_source(spec, source_dir, filtered_dir, max_bytes=max_bytes)
        manifest_rows.append(manifest)
        extract_rows.extend(extracted)
    coverage_rows = _coverage_delta_rows(r66, manifest_rows)
    gate = _gate(manifest_rows, extract_rows)
    report_path = analysis_dir / "r68_bulk_external_source_ingest_report.json"
    markdown_path = analysis_dir / "r68_bulk_external_source_ingest_report.md"
    manifest_csv = analysis_dir / "r68_bulk_manifest_rows.csv"
    extract_csv = analysis_dir / "r68_extracted_table_rows.csv"
    coverage_csv = analysis_dir / "r68_coverage_delta_rows.csv"
    report = {
        "schema_version": R68_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "bulk_ingest_gate": gate,
        "bulk_manifest_rows": manifest_rows,
        "extracted_table_rows": extract_rows,
        "coverage_delta_rows": coverage_rows,
        "source_artifacts": {
            "r66": {"path": r66_path.as_posix(), "sha256": _sha256(r66_path) if r66_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "bulk_manifest_csv": manifest_csv.as_posix(),
            "extracted_table_csv": extract_csv.as_posix(),
            "coverage_delta_csv": coverage_csv.as_posix(),
            "external_source_dir": source_dir.as_posix(),
            "filtered_philippines_dir": filtered_dir.as_posix(),
        },
    }
    _write_csv(manifest_csv, manifest_rows)
    _write_csv(extract_csv, extract_rows)
    _write_csv(coverage_csv, coverage_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R68 bulk external source ingest.")
    parser.add_argument("--run-id", default=R68_RUN_ID)
    parser.add_argument("--r66-report-path", default=None)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()
    run_r68_bulk_external_source_ingest(
        run_id=str(args.run_id),
        r66_report_path=None if args.r66_report_path is None else Path(args.r66_report_path),
        max_bytes=int(args.max_bytes),
    )


if __name__ == "__main__":
    _main()
