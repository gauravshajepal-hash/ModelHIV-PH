from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Any

from .data import default_epigraph_root, sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .runtime import ensure_dir, write_json


R43_SCHEMA_VERSION = "phase3_dynamic.r43_subnational_evidence_intake.v1"
R43_RUN_ID = "p3d-r43-subnational-evidence-intake-20260502-s00"

REGIONAL_CASCADE_PATTERN = re.compile(
    r"^\s*(?P<region>[A-Z0-9]+(?:\s+[A-Z0-9]+)?)\s+"
    r"(?P<estimated_plhiv>[\d,]+)\s+"
    r"(?P<diagnosed_plhiv>[\d,]+)\s+"
    r"(?P<first_95_pct>\d+%)\s+"
    r"(?P<alive_on_art>[\d,]+)\s+"
    r"(?P<second_95_pct>\d+%)\s+"
    r"(?P<tested_for_viral_load>[\d,]+)\s+"
    r"(?P<vl_testing_coverage_pct>\d+%)\s+"
    r"(?P<virally_suppressed>[\d,]+)\s+"
    r"(?P<suppression_among_tested_pct>\d+%)\s+"
    r"(?P<third_95_pct>\d+%)\s*$"
)

REGIONAL_CASCADE_NO_VL_COVERAGE_PATTERN = re.compile(
    r"^\s*(?P<region>[A-Z0-9]+(?:\s+[A-Z0-9]+)?)\s+"
    r"(?P<estimated_plhiv>[\d,]+)\s+"
    r"(?P<diagnosed_plhiv>[\d,]+)\s+"
    r"(?P<first_95_pct>\d+%)\s+"
    r"(?P<alive_on_art>[\d,]+)\s+"
    r"(?P<second_95_pct>\d+%)\s+"
    r"(?P<tested_for_viral_load>[\d,]+)\s+"
    r"(?P<virally_suppressed>[\d,]+)\s+"
    r"(?P<suppression_among_tested_pct>\d+%)\s+"
    r"(?P<third_95_pct>\d+%)\s*$"
)


def _default_hasp_pdf(epigraph_root: Path) -> Path:
    return (
        Path(epigraph_root)
        / "src"
        / "epigraph_ph"
        / "harp_archive"
        / "HIV_Data"
        / "2025_Q2-HIV-AIDS-Surveillance-Report-of-the-Philippines-1.pdf"
    )


def _parse_int(text: str) -> int:
    return int(str(text).replace(",", ""))


def _parse_percent(text: str) -> float:
    return float(str(text).strip().removesuffix("%")) / 100.0


def _parse_regional_cascade_line(line: str) -> dict[str, Any] | None:
    match = REGIONAL_CASCADE_PATTERN.match(line)
    has_reported_vl_testing_coverage = True
    if not match:
        match = REGIONAL_CASCADE_NO_VL_COVERAGE_PATTERN.match(line)
        has_reported_vl_testing_coverage = False
    if not match:
        return None
    groups = match.groupdict()
    region = str(groups["region"]).strip()
    if region in {"REGION", "AGE GROUP", "KEY POPULATION"}:
        return None
    alive_on_art = _parse_int(groups["alive_on_art"])
    tested_for_viral_load = _parse_int(groups["tested_for_viral_load"])
    if has_reported_vl_testing_coverage:
        vl_testing_coverage = _parse_percent(groups["vl_testing_coverage_pct"])
        vl_testing_coverage_semantics = "reported_percent"
    else:
        vl_testing_coverage = float(tested_for_viral_load) / float(alive_on_art) if alive_on_art > 0 else 0.0
        vl_testing_coverage_semantics = "derived_vl_tested_over_art_no_reported_column"
    return {
        "region": region,
        "estimated_plhiv": _parse_int(groups["estimated_plhiv"]),
        "diagnosed_plhiv": _parse_int(groups["diagnosed_plhiv"]),
        "first_95": _parse_percent(groups["first_95_pct"]),
        "alive_on_art": alive_on_art,
        "second_95": _parse_percent(groups["second_95_pct"]),
        "tested_for_viral_load": tested_for_viral_load,
        "vl_testing_coverage": vl_testing_coverage,
        "vl_testing_coverage_semantics": vl_testing_coverage_semantics,
        "virally_suppressed": _parse_int(groups["virally_suppressed"]),
        "suppression_among_tested": _parse_percent(groups["suppression_among_tested_pct"]),
        "third_95": _parse_percent(groups["third_95_pct"]),
    }


def _pdf_layout_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        text=True,
        capture_output=True,
    )
    return str(result.stdout)


def _extract_regional_cascade_rows(*, pdf_path: Path) -> list[dict[str, Any]]:
    text = _pdf_layout_text(pdf_path)
    rows: list[dict[str, Any]] = []
    in_table = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "Care Cascade by Region" in line:
            in_table = True
            continue
        if in_table and "Care Cascade by Age Group" in line:
            break
        if not in_table:
            continue
        parsed = _parse_regional_cascade_line(line)
        if parsed is None:
            continue
        rows.append(
            {
                **parsed,
                "source_id": "doh_hasp_2025_q2_regional_cascade_annex",
                "source_path": pdf_path.as_posix(),
                "source_sha256": _sha256(pdf_path),
                "source_line_number": int(line_number),
                "time_start": "2025-06",
                "time_end": "2025-06",
                "time_granularity": "quarterly_snapshot",
                "geography": str(parsed["region"]),
                "population": "all",
                "source_tier": "official_doh_archive",
                "measurement_class": "program_observed_hasp_annex",
                "observation_role": "auxiliary_likelihood",
                "allowed_use": "subnational_anchor_and_partial_pooling_diagnostic",
                "support_partition": "new_subnational_support",
                "leakage_status": "not_in_active_phase3_training",
            }
        )
    return rows


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
    lines = [
        "# Phase 3 R43 Subnational Evidence Intake",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Regional Cascade Anchors",
        "",
        "| Region | Estimated PLHIV | Diagnosed | ART | VL tested | Suppressed | 1st 95 | 2nd 95 | 3rd 95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(report.get("regional_cascade_rows") or []):
        lines.append(
            f"| {row.get('region')} | {row.get('estimated_plhiv')} | {row.get('diagnosed_plhiv')} | "
            f"{row.get('alive_on_art')} | {row.get('tested_for_viral_load')} | {row.get('virally_suppressed')} | "
            f"{100.0 * float(row.get('first_95') or 0.0):.0f}% | {100.0 * float(row.get('second_95') or 0.0):.0f}% | "
            f"{100.0 * float(row.get('third_95') or 0.0):.0f}% |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- These rows are not injected into Phase 3 training yet.",
            "- They are subnational anchors for partial pooling and external geography diagnostics.",
            "- A region claim still requires multi-period support or an explicit auxiliary-only label.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    sorted_rows = sorted(rows, key=lambda row: float(row.get("diagnosed_plhiv") or 0.0), reverse=True)
    labels = [str(row.get("region") or "") for row in sorted_rows]
    diagnosed = np.asarray([float(row.get("diagnosed_plhiv") or 0.0) for row in sorted_rows], dtype=np.float64)
    art = np.asarray([float(row.get("alive_on_art") or 0.0) for row in sorted_rows], dtype=np.float64)
    suppressed = np.asarray([float(row.get("virally_suppressed") or 0.0) for row in sorted_rows], dtype=np.float64)
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Subnational HASP 2025-Q2 cascade anchors", fontsize=14, fontweight="bold")
    ax.barh(y, diagnosed, color="#5c6f83", label="diagnosed")
    ax.barh(y, art, color="#2f6b4f", label="on ART")
    ax.barh(y, suppressed, color="#8a6f2a", label="suppressed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("people")
    ax.legend()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r43_subnational_evidence_intake(
    *,
    run_id: str = R43_RUN_ID,
    epigraph_root: Path | None = None,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    pdf = Path(pdf_path) if pdf_path else _default_hasp_pdf(root)
    if not pdf.exists():
        raise FileNotFoundError(f"HASP PDF not found: {pdf}")
    rows = _extract_regional_cascade_rows(pdf_path=pdf)
    blockers: list[str] = []
    if not rows:
        blockers.append("no_regional_cascade_rows_extracted")
    status = "ready_as_auxiliary_subnational_anchor" if not blockers else "blocked"
    verdict = (
        "R43 extracted regional HASP cascade anchors for subnational partial-pooling diagnostics; these are not yet multi-period validation targets."
        if not blockers
        else "R43 could not extract regional HASP cascade anchors."
    )
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    report = {
        "schema_version": R43_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "verdict": verdict,
        "source_pdf": pdf.as_posix(),
        "source_pdf_sha256": _sha256(pdf),
        "regional_cascade_row_count": len(rows),
        "regional_cascade_rows": rows,
        "contract": (
            "Regional cascade rows from HASP 2025-Q2 are auxiliary subnational anchors. They may support "
            "partial-pooling diagnostics and scenario sanity checks, but they cannot establish blocked-time "
            "province or region forecast superiority until comparable multi-period regional rows are onboarded."
        ),
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r43_subnational_evidence_intake_report.json"
    md_path = analysis_dir / "r43_subnational_evidence_intake_report.md"
    csv_path = analysis_dir / "r43_regional_cascade_rows.csv"
    dashboard_path = analysis_dir / "r43_regional_cascade_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "regional_cascade_csv": csv_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R43 subnational evidence intake.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--pdf-path", default=None)
    parser.add_argument("--run-id", default=R43_RUN_ID)
    args = parser.parse_args()
    run_r43_subnational_evidence_intake(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        pdf_path=None if args.pdf_path is None else Path(args.pdf_path),
    )


if __name__ == "__main__":
    _main()
