"""Stream and audit a UNAIDS workbook without mistaking its URL for its vintage.

This is source intake, not a forecasting benchmark or a training-data loader.
Rounded/censored workbook cells are preserved; no precision is invented.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


METRICS = {
    "AIDS-related deaths among adults and children": "annual_aids_deaths",
    "Estimated adults and children living with HIV": "estimated_plhiv",
    "Adults and children newly infected with HIV": "annual_new_infections",
}


def parse_display(value) -> dict:
    raw = str(value).strip() if value is not None else ""
    if raw in ("", "...", "-", ".."):
        return {"raw": value, "numeric": None, "status": "missing"}
    text = re.sub(r"\s+", "", raw).replace(",", "")
    match = re.fullmatch(r"([<>]?)(\d+(?:\.\d+)?)([mk]?)", text, re.I)
    if not match:
        return {"raw": value, "numeric": None, "status": "unparsed"}
    sign, number, scale = match.groups()
    bound = float(number) * {"": 1, "k": 1000, "m": 1000000}[scale.lower()]
    return {"raw": value, "numeric": None if sign else bound,
            "status": "left_censored" if sign == "<" else "right_censored" if sign else "display_value",
            "censor_threshold": bound if sign else None}


def audit_workbook(path: Path, *, requested_release: int, source_url: str) -> dict:
    import openpyxl

    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    sheets, evidence, seen = [], [], set()
    with path.open("rb") as handle:
        workbook = openpyxl.load_workbook(handle, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                # ByArea duplicates ByYear; do not count both as independent support.
                if not sheet.title.endswith("_ByYear"):
                    continue
                iterator = iter(sheet.values)
                header = [next(iterator) for _ in range(6)]
                release_match = re.fullmatch(r"Source: UNAIDS (\d{4}) estimates", str(header[2][0]))
                if not release_match:
                    raise ValueError("Missing or ambiguous in-workbook source vintage")
                release = int(release_match[1])
                country_rows = []
                core_columns = {i: METRICS[label] for i, label in enumerate(header[4]) if label in METRICS}
                for index, cells in enumerate(iterator, start=7):
                    if len(cells) < 3 or cells[2] != "Philippines":
                        continue
                    if cells[1] != "PHL" or not isinstance(cells[0], int):
                        raise ValueError("Unexpected Philippines row identity/year")
                    year = cells[0]
                    country_rows.append({"excel_row": index, "year": year, "cells": list(cells)})
                    for column, metric in core_columns.items():
                        if tuple(header[5][column:column+3]) != ("Estimate", "Low", "High"):
                            raise ValueError("Unrecognized uncertainty-column semantics")
                        key = (year, metric)
                        if key in seen:
                            raise ValueError("Duplicate core annual metric/year")
                        seen.add(key)
                        row = {"year": year, "metric_id": metric, "population": "all_ages", "geography": "Philippines",
                               "unit": "people", "observation_role": "validation_only", "allowed_use": ["estimate_agreement_diagnostic"],
                               "measurement_semantics": "modeled_estimate", "source_id": digest, "source_url": source_url,
                               "source_release": release, "source_sheet": sheet.title, "source_excel_row": index,
                               "source_excel_columns": [column+1, column+2, column+3], "is_forecast": False,
                               "precision": "published_display_values_not_original_unrounded_spectrum_output",
                               "estimate": parse_display(cells[column]), "lower": parse_display(cells[column+1]),
                               "upper": parse_display(cells[column+2]),
                               "leakage_status": "not_new_independent_truth; compare vintage and prior use before scoring"}
                        row["row_hash"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                        evidence.append(row)
                if not country_rows:
                    raise ValueError("No Philippines data found")
                sheets.append({"sheet": sheet.title, "release": release, "header": [list(r) for r in header],
                               "min_year": min(r["year"] for r in country_rows), "max_year": max(r["year"] for r in country_rows),
                               "raw_philippines_rows": country_rows})
        finally:
            workbook.close()
    if not sheets or not evidence:
        raise ValueError("No supported annual sheets/core metrics")
    releases = sorted({s["release"] for s in sheets})
    return {"schema": "unaids_workbook_vintage_audit.v1", "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_url": source_url, "source_filename": path.name, "source_sha256": digest, "source_bytes": path.stat().st_size,
            "requested_release": requested_release, "actual_releases": releases,
            "status": "requested_vintage_verified" if releases == [requested_release] else "requested_vintage_not_present",
            "max_philippines_year": max(s["max_year"] for s in sheets), "sheets": sheets, "core_annual_rows": evidence,
            "is_new_external_validation": False, "is_official_model_forecast_benchmark": False,
            "claim_limit": "An estimates workbook is not independent truth or a matched official forecast; no training or champion promotion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--requested-release", type=int, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a source-vintage audit")
    report = audit_workbook(args.workbook, requested_release=args.requested_release, source_url=args.source_url)
    report["audit_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "actual_releases", "max_philippines_year", "source_sha256")}))


if __name__ == "__main__":
    main()
