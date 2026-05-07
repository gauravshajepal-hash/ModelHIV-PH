from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import default_epigraph_root, sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r43_subnational_evidence_intake import _extract_regional_cascade_rows
from .runtime import ensure_dir, write_json


R44_SCHEMA_VERSION = "phase3_dynamic.r44_subnational_horizon_gate.v1"
R44_RUN_ID = "p3d-r44-subnational-horizon-gate-20260502-s00"


def _hiv_data_dir(epigraph_root: Path) -> Path:
    return Path(epigraph_root) / "src" / "epigraph_ph" / "harp_archive" / "HIV_Data"


def _candidate_pdfs(*, epigraph_root: Path, extra_pdf_roots: tuple[Path, ...] = ()) -> list[Path]:
    roots = [_hiv_data_dir(epigraph_root), *[Path(root) for root in extra_pdf_roots]]
    pdfs: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                pdfs[path.resolve().as_posix()] = path
    return [pdfs[key] for key in sorted(pdfs)]


def _source_metadata_by_pdf(*, epigraph_root: Path, extra_pdf_roots: tuple[Path, ...] = ()) -> dict[str, dict[str, Any]]:
    roots = [_hiv_data_dir(epigraph_root), *[Path(root) for root in extra_pdf_roots]]
    metadata: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("downloaded_hasp_manifest.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            for row in list(payload.get("rows") or []):
                filename = str(row.get("filename") or "")
                if not filename:
                    continue
                path = (manifest.parent / filename).resolve().as_posix()
                metadata[path] = dict(row)
                metadata[filename] = dict(row)
    return metadata


def _period_from_pdf_name(pdf_path: Path, fallback_time_end: str | None = None) -> dict[str, str]:
    text = pdf_path.stem.replace("-", "_").replace(" ", "_")
    match = re.search(r"(?P<year>20\d{2})_?Q(?P<quarter>[1-4])", text, flags=re.IGNORECASE)
    if match:
        year = int(match.group("year"))
        quarter = int(match.group("quarter"))
        month = quarter * 3
        return {
            "time_start": f"{year}-{month:02d}",
            "time_end": f"{year}-{month:02d}",
            "time_granularity": "quarterly_snapshot",
            "period_id": f"{year}-Q{quarter}",
        }
    match = re.search(r"Q(?P<quarter>[1-4])_?(?P<year>20\d{2})", text, flags=re.IGNORECASE)
    if match:
        year = int(match.group("year"))
        quarter = int(match.group("quarter"))
        month = quarter * 3
        return {
            "time_start": f"{year}-{month:02d}",
            "time_end": f"{year}-{month:02d}",
            "time_granularity": "quarterly_snapshot",
            "period_id": f"{year}-Q{quarter}",
        }
    match = re.search(
        r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)_?(?P<year>20\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month_lookup = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        month = month_lookup[str(match.group("month")).lower()]
        year = int(match.group("year"))
        return {
            "time_start": f"{year}-{month:02d}",
            "time_end": f"{year}-{month:02d}",
            "time_granularity": "monthly_snapshot",
            "period_id": f"{year}-{month:02d}",
        }
    if fallback_time_end:
        return {
            "time_start": str(fallback_time_end),
            "time_end": str(fallback_time_end),
            "time_granularity": "unknown_snapshot",
            "period_id": str(fallback_time_end),
        }
    return {
        "time_start": "unknown",
        "time_end": "unknown",
        "time_granularity": "unknown_snapshot",
        "period_id": "unknown",
    }


def _scan_regional_cascade_pdfs(
    *,
    epigraph_root: Path,
    extra_pdf_roots: tuple[Path, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pdf_reports: list[dict[str, Any]] = []
    seen_pdf_sha256: set[str] = set()
    source_metadata = _source_metadata_by_pdf(epigraph_root=epigraph_root, extra_pdf_roots=extra_pdf_roots)
    for pdf_path in _candidate_pdfs(epigraph_root=epigraph_root, extra_pdf_roots=extra_pdf_roots):
        pdf_sha256 = _sha256(pdf_path) if pdf_path.exists() else ""
        if pdf_sha256 and pdf_sha256 in seen_pdf_sha256:
            pdf_reports.append(
                {
                    "pdf_path": pdf_path.as_posix(),
                    "pdf_sha256": pdf_sha256,
                    "status": "duplicate_pdf_sha256_skipped",
                    "regional_row_count": 0,
                }
            )
            continue
        if pdf_sha256:
            seen_pdf_sha256.add(pdf_sha256)
        metadata = source_metadata.get(pdf_path.resolve().as_posix()) or source_metadata.get(pdf_path.name) or {}
        try:
            extracted = _extract_regional_cascade_rows(pdf_path=pdf_path)
        except Exception as exc:  # pragma: no cover - persisted as source-level diagnostic.
            pdf_reports.append(
                {
                    "pdf_path": pdf_path.as_posix(),
                    "pdf_sha256": pdf_sha256 or None,
                    "status": "parse_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "regional_row_count": 0,
                }
            )
            continue
        period = _period_from_pdf_name(
            pdf_path,
            fallback_time_end=str(metadata.get("time_period_hint") or extracted[0].get("time_end") or "") if extracted else None,
        )
        normalized_rows: list[dict[str, Any]] = []
        for row in extracted:
            normalized = dict(row)
            normalized.update(period)
            normalized["source_id"] = f"doh_hasp_{period['period_id']}_regional_cascade"
            normalized["source_path"] = pdf_path.as_posix()
            normalized["source_sha256"] = pdf_sha256
            normalized["source_url"] = metadata.get("url")
            normalized["source_page"] = metadata.get("source_page")
            normalized["source_tier"] = metadata.get("source_tier") or normalized.get("source_tier")
            normalized["source_access_route"] = "downloaded_manifest" if metadata else "local_or_direct_pdf_scan"
            normalized["observation_role"] = "auxiliary_likelihood"
            normalized["allowed_use"] = "subnational_anchor_and_horizon_gate_diagnostic"
            normalized["support_partition"] = "new_subnational_support"
            normalized["leakage_status"] = "not_in_active_phase3_training"
            normalized_rows.append(normalized)
        rows.extend(normalized_rows)
        pdf_reports.append(
            {
                "pdf_path": pdf_path.as_posix(),
                "pdf_sha256": pdf_sha256,
                "status": "regional_cascade_rows_extracted" if normalized_rows else "no_regional_cascade_table",
                "period_id": period["period_id"],
                "time_end": period["time_end"],
                "time_granularity": period["time_granularity"],
                "source_url": metadata.get("url"),
                "source_page": metadata.get("source_page"),
                "source_tier": metadata.get("source_tier"),
                "regional_row_count": len(normalized_rows),
            }
        )
    return rows, pdf_reports


def _subnational_horizon_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    periods = sorted({str(row.get("period_id") or row.get("time_end") or "") for row in rows if row.get("region")})
    regions_by_period: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        period = str(row.get("period_id") or row.get("time_end") or "")
        region = str(row.get("region") or "")
        if period and region:
            regions_by_period[period].add(region)
    overlapping_regions = sorted(set.intersection(*regions_by_period.values())) if len(regions_by_period) >= 2 else []
    blockers: list[str] = []
    if not rows:
        blockers.append("no_regional_cascade_rows_extracted")
    if len(periods) < 2:
        blockers.append("need_at_least_two_regional_cascade_periods_for_blocked_subnational_gate")
    if len(periods) >= 2 and not overlapping_regions:
        blockers.append("need_overlapping_regions_across_periods_for_blocked_subnational_gate")
    status = "multi_period_subnational_gate_ready" if not blockers else "single_period_auxiliary_only"
    if not rows:
        status = "blocked_no_regional_cascade_rows"
    return {
        "status": status,
        "blockers": blockers,
        "unique_period_count": len(periods),
        "periods": periods,
        "unique_region_count": len({str(row.get("region") or "") for row in rows if row.get("region")}),
        "overlapping_region_count": len(overlapping_regions),
        "overlapping_regions": overlapping_regions,
        "period_region_counts": [
            {"period_id": period, "region_count": len(regions_by_period[period])}
            for period in sorted(regions_by_period)
        ],
        "contract": (
            "A subnational blocked-time claim requires at least two comparable regional cascade periods and "
            "overlapping regions. Single-period regional rows may only be used as auxiliary anchors or "
            "partial-pooling sanity checks."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("subnational_horizon_gate") or {})
    lines = [
        "# Phase 3 R44 Subnational Horizon Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Periods: `{gate.get('unique_period_count')}`",
        f"- Regions: `{gate.get('unique_region_count')}`",
        f"- Overlapping regions: `{gate.get('overlapping_region_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## PDF Intake",
        "",
        "| PDF | Status | Period | Regional rows |",
        "|---|---|---:|---:|",
    ]
    for row in list(report.get("pdf_reports") or []):
        lines.append(
            f"| {Path(str(row.get('pdf_path') or '')).name} | {row.get('status')} | "
            f"{row.get('period_id') or ''} | {row.get('regional_row_count')} |"
        )
    lines.extend(
        [
            "",
            "## Scientific Contract",
            "",
            str(gate.get("contract") or ""),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    if not rows:
        path.write_text("no regional rows", encoding="utf-8")
        return
    periods = list(gate.get("periods") or [])
    regions = sorted({str(row.get("region") or "") for row in rows if row.get("region")})
    coverage = np.zeros((len(regions), len(periods)), dtype=np.float64)
    value_by_key = {
        (str(row.get("region") or ""), str(row.get("period_id") or row.get("time_end") or "")): 1.0
        for row in rows
    }
    for i, region in enumerate(regions):
        for j, period in enumerate(periods):
            coverage[i, j] = value_by_key.get((region, period), 0.0)

    latest_period = periods[-1] if periods else ""
    latest_rows = [
        row for row in rows if str(row.get("period_id") or row.get("time_end") or "") == latest_period
    ]
    latest_rows = sorted(latest_rows, key=lambda row: float(row.get("diagnosed_plhiv") or 0.0), reverse=True)
    labels = [str(row.get("region") or "") for row in latest_rows]
    diagnosed = np.asarray([float(row.get("diagnosed_plhiv") or 0.0) for row in latest_rows], dtype=np.float64)
    art = np.asarray([float(row.get("alive_on_art") or 0.0) for row in latest_rows], dtype=np.float64)
    suppressed = np.asarray([float(row.get("virally_suppressed") or 0.0) for row in latest_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("R44 subnational evidence gate", fontsize=14, fontweight="bold")
    axes[0].imshow(coverage, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    axes[0].set_title("Regional cascade period coverage")
    axes[0].set_xticks(np.arange(len(periods)))
    axes[0].set_xticklabels(periods, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(regions)))
    axes[0].set_yticklabels(regions)
    axes[0].set_xlabel("period")
    axes[0].set_ylabel("region")
    y = np.arange(len(labels), dtype=np.float64)
    axes[1].barh(y, diagnosed, color="#5c6f83", label="diagnosed")
    axes[1].barh(y, art, color="#2f6b4f", label="on ART")
    axes[1].barh(y, suppressed, color="#8a6f2a", label="suppressed")
    axes[1].set_title(f"Latest available cascade anchor: {latest_period}")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("people")
    axes[1].legend()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r44_subnational_horizon_gate(
    *,
    run_id: str = R44_RUN_ID,
    epigraph_root: Path | None = None,
    extra_pdf_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    rows, pdf_reports = _scan_regional_cascade_pdfs(epigraph_root=root, extra_pdf_roots=extra_pdf_roots)
    gate = _subnational_horizon_gate(rows)
    verdict = (
        "R44 found multi-period subnational cascade support; region-level blocked-time gates can now be attempted."
        if gate["status"] == "multi_period_subnational_gate_ready"
        else "R44 found only auxiliary subnational support. Region/province forecast superiority remains blocked until comparable multi-period regional cascade rows are onboarded."
    )
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    report = {
        "schema_version": R44_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "epigraph_root": root.as_posix(),
        "scanned_pdf_count": len(pdf_reports),
        "regional_cascade_row_count": len(rows),
        "pdf_reports": pdf_reports,
        "subnational_horizon_gate": gate,
        "regional_cascade_rows": rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r44_subnational_horizon_gate_report.json"
    md_path = analysis_dir / "r44_subnational_horizon_gate_report.md"
    csv_path = analysis_dir / "r44_regional_cascade_rows.csv"
    pdf_csv_path = analysis_dir / "r44_pdf_intake_summary.csv"
    dashboard_path = analysis_dir / "r44_subnational_horizon_gate_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "regional_cascade_csv": csv_path.as_posix(),
        "pdf_intake_csv": pdf_csv_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, rows)
    _write_csv(pdf_csv_path, pdf_reports)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, rows, gate)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R44 subnational horizon gate.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--run-id", default=R44_RUN_ID)
    parser.add_argument("--extra-pdf-root", action="append", default=[])
    args = parser.parse_args()
    run_r44_subnational_horizon_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        extra_pdf_roots=tuple(Path(item) for item in args.extra_pdf_root),
    )


if __name__ == "__main__":
    _main()
