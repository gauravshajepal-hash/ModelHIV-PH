"""HASP revision catalog, publication-delay challenge, and dated forecast lock."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone
import unicodedata
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .data import sandbox_repo_root
from .r44_subnational_horizon_gate import _period_from_pdf_name
from .r96_monthly_diagnosis_state import FAMILIES, Q2_PDF, _digest, _write, forecast


RUN_ID = "p3d-r97-report-vintages-20260911-s01"
FIRST_YEAR = 2023
CONTRACT = {
    "schema": "r97.report_vintages.v1",
    "claim": "report-vintage and declared-mirror-availability sensitivity",
    "source": "DOH HASP PDFs; SHIP WordPress posting metadata",
    "date_semantics": "mirror_posting_date is not first_publication_date or proof of historical file bytes",
    "training": "one entire eligible report vintage; never backfill its history from a newer report",
    "OCR": "right-column then whole-page fallback; one image and one Tesseract thread; unique PSM 3/6 agreement across layouts; verify AVG where present; mask printed future placeholders",
    "historical_origins": ["2026-01-01", "2026-04-01"],
    "score": "reported diagnosis count absolute error; propagate all unobserved gap months",
    "outcome_definition": "sum the three target-month cells in that report's monthly diagnosis table, not a substituted headline or later revision",
    "selection": "earlier target blocks whose report was posted by this origin; otherwise carry-forward",
    "prospective_target": "2026-Q4",
    "prospective_information_cutoff": "generation timestamp; target quarter must not have begun",
    "no_promotion": ["first release and historical bytes not verified", "only two target quarters",
                     "reporting and biological processes still confounded", "no regional or AEM comparison"],
}


def parse_rows(text: str, end_month: str) -> dict[int, dict]:
    end_year, last_month = map(int, end_month.split("-"))
    found = {}
    for line in text.splitlines():
        for match in re.finditer(r"(20\d{2})", line):
            year = int(match.group(1))
            if not FIRST_YEAR <= year <= end_year:
                continue
            suffix = line[match.end():]
            if re.search(r"[A-Za-z%]", suffix):
                continue
            suffix = re.sub(r"(?<=\d),(?=\d)", "", suffix)
            values = [int(v) for v in re.findall(r"\d+", suffix)]
            n = last_month if year == end_year else 12
            average = None
            if len(values) in (13, n + 1):
                average = values[-1]
                counts = values[:-1]
            elif len(values) in (12, n) and not re.search(r"\bAVG\b", text, re.I):
                counts = values
            else:
                continue
            if len(counts) > n and any(counts[n:]):
                continue
            counts = counts[:n]
            if len(counts) != n or (average is not None and abs(sum(counts) - n * average) > n / 2):
                continue
            row = {"year": year, "counts": counts, "printed_average": average, "source_line": line.strip()}
            if year in found and found[year]["counts"] != row["counts"]:
                raise ValueError(f"Conflicting numeric table rows for {year}")
            found[year] = row
    return found


def agreed_rows(first: dict, second: dict) -> dict:
    return {y: r for y, r in first.items() if y in second and r["counts"] == second[y]["counts"]
            and r["printed_average"] == second[y]["printed_average"]}


def layout_consensus(extractions: dict[int, list[dict]]) -> dict:
    """Two segmentation modes must identify one and only one identical row."""
    candidates = {}
    for mode, layouts in extractions.items():
        candidates[mode] = {}
        for rows in layouts:
            for year, row in rows.items():
                key = (tuple(row["counts"]), row["printed_average"])
                candidates[mode].setdefault(year, {})[key] = row
    result = {}
    for year, rows in candidates[3].items():
        common = set(rows) & set(candidates[6].get(year, {}))
        if len(common) == 1:
            result[year] = rows[common.pop()]
    return result


def _ocr_layout(pdf: Path, page_index: int, cache: Path, tesseract: str, layout: str) -> dict:
    suffix = "_right_column" if layout == "right_column" else ""
    stem = cache / (_digest(pdf)[:16] + suffix)
    if not all(stem.with_suffix(f".psm{psm}.txt").exists() for psm in (3, 6)):
        crop = []
        if layout == "right_column":
            bbox = ET.fromstring(subprocess.check_output([
                "pdftotext", "-f", str(page_index + 1), "-l", str(page_index + 1), "-bbox-layout", str(pdf), "-"
            ]))
            width = float(next(e for e in bbox.iter() if e.tag.endswith("page")).attrib["width"])
            left = int(width * 350 / 72 / 2)
            crop = ["-x", str(left), "-W", str(left + 1)]
        subprocess.run(["pdftoppm", "-f", str(page_index + 1), "-l", str(page_index + 1), *crop,
                        "-r", "350", "-png", "-singlefile", str(pdf), str(stem)], check=True, timeout=90)
    texts = {}
    try:
        for psm in (3, 6):
            target = stem.with_suffix(f".psm{psm}.txt")
            if not target.exists():
                result = subprocess.run([tesseract, str(stem.with_suffix(".png")), "stdout", "--psm", str(psm)],
                                        env={**os.environ, "OMP_THREAD_LIMIT": "1"}, check=True,
                                        capture_output=True, text=True, timeout=90)
                target.write_text(result.stdout, encoding="utf-8")
            texts[psm] = target.read_text()
    finally:
        stem.with_suffix(".png").unlink(missing_ok=True)
    return texts


def _monthly_extract(pdf: Path, end_month: str, cache: Path, tesseract: str | None) -> tuple:
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    text = unicodedata.normalize("NFKC", text)
    pages = text.split("\f")
    page_index = next((i for i, page in enumerate(pages) if "Number of monthly newly diagnosed HIV cases" in page), None)
    if page_index is None:
        raise ValueError("Monthly diagnosis figure caption missing")
    page = pages[page_index]
    page = page[page.index("Number of monthly newly diagnosed HIV cases"):]
    direct = parse_rows(page, end_month)
    expected = set(range(FIRST_YEAR, int(end_month[:4]) + 1))
    if set(direct) == expected:
        return direct, "pdf_text_numeric_rows_average_checked", text
    if not tesseract:
        raise ValueError("Image table requires Tesseract; it is unavailable")
    results = {3: [], 6: []}
    for layout in ("right_column", "whole_page"):
        texts = _ocr_layout(pdf, page_index, cache, tesseract, layout)
        for mode, content in texts.items():
            results[mode].append(parse_rows(content, end_month))
        agreed = layout_consensus(results)
        if set(agreed) == expected:
            return agreed, f"tesseract_psm3_psm6_consensus_through_{layout}_average_checked", text
    raise ValueError(f"OCR incomplete/disagrees: expected {sorted(expected)}, agreed {sorted(agreed)}")


def _title_period(title: str) -> str | None:
    title = html.unescape(title).lower()
    year = re.search(r"\b(202[4-6])\b", title)
    if year is None:
        return None
    for quarter, names in enumerate((("january", "march"), ("april", "june"), ("july", "september"), ("october", "december")), 1):
        if all(name in title for name in names):
            return f"{year.group(1)}-Q{quarter}"
    return None


def collect_posting_metadata(cache: Path) -> dict:
    records = {}
    for year in (2024, 2025, 2026):
        path = cache / f"ship_posts_{year}.json"
        url = "https://www.ship.ph/wp-json/wp/v2/posts?" + urlencode({"search": year, "per_page": 100,
                                                                   "_fields": "id,date_gmt,link,title,content"})
        if not path.exists():
            with urlopen(url, timeout=40) as response:
                # This is a small metadata query, not a corpus download.
                data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("Posting metadata response exceeds fixed resource cap")
            path.write_bytes(data)
        for post in json.loads(path.read_text()):
            period = _title_period(post["title"]["rendered"])
            if period is None:
                continue
            records[period] = {"mirror_posted_at": post["date_gmt"] + "Z", "source_page": post["link"],
                               "title": html.unescape(post["title"]["rendered"]), "metadata_url": url,
                               "metadata_snapshot_sha256": _digest(path), "first_publication_date": None,
                               "historical_bytes_verified": False}
    return records


def _month_index(label: str) -> int:
    return int(label[:4]) * 12 + int(label[5:7]) - 1


def _period_end(period: str) -> str:
    return f"{period[:4]}-{int(period[-1]) * 3:02d}"


def asof_vintage(catalog: list[dict], issued_at: str) -> dict:
    eligible = [r for r in catalog if r["status"] == "accepted" and r.get("mirror_posted_at")
                and _timestamp(r["mirror_posted_at"]) <= _timestamp(issued_at) and r["end_month"] < issued_at[:7]]
    if not eligible:
        raise ValueError("No eligible report vintage at the information cutoff")
    return max(eligible, key=lambda r: r["end_month"])


def predict_period(vintage: dict, family: str, target_start: str) -> dict:
    expected = _month_index(vintage["end_month"]) - _month_index(f"{FIRST_YEAR}-01") + 1
    if len(vintage["values"]) != expected:
        raise ValueError("Training vintage has an incomplete monthly calendar")
    gap = _month_index(target_start) - _month_index(vintage["end_month"]) - 1
    if gap < 0:
        raise ValueError("Target overlaps observed training months")
    result = forecast(vintage["values"], family, gap + 3)
    return {"prediction": sum(result["points"][gap:]), "monthly_points": result["points"][gap:],
            "unobserved_gap_months": gap, "training_end_month": vintage["end_month"], "family": family,
            "training_report_period": vintage["period"], "training_source_sha256": vintage["source_sha256"]}


def choose_at_origin(completed: list[dict], issued_at: str) -> str:
    available = [row for row in completed if _timestamp(row["target_posted_at"]) <= _timestamp(issued_at)]
    if not available:
        return "quarter_carry"
    for target in {r["target_posted_at"] for r in available}:
        if sorted(r["family"] for r in available if r["target_posted_at"] == target) != sorted(FAMILIES):
            raise ValueError("Selection requires every family on the same completed blocks")
    scores = {family: np.mean([r["absolute_error"] for r in available if r["family"] == family]) for family in FAMILIES}
    return min(FAMILIES, key=scores.get)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def report_period(pdf: Path) -> str:
    normalized = Path(re.sub(r"[_\s-]+", "_", pdf.stem))
    return _period_from_pdf_name(normalized)["period_id"]


def run(output: Path, *, tesseract: str | None = None) -> dict:
    if (output / "report.json").exists():
        raise FileExistsError("Completed R97 runs are immutable")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "source_snapshots"
    cache.mkdir(exist_ok=True)
    _write(output / "contract.json", CONTRACT)
    metadata = collect_posting_metadata(cache)
    catalog, observations, lab_rows = [], [], []
    pdfs = sorted(Q2_PDF.parent.parent.rglob("*.pdf"))
    for pdf in pdfs:
        period = report_period(pdf)
        if not re.fullmatch(r"202[4-6]-Q[1-4]", period):
            continue
        entry = {"period": period, "end_month": _period_end(period), "source_path": str(pdf),
                 "source_sha256": _digest(pdf), **metadata.get(period, {})}
        try:
            rows, method, text = _monthly_extract(pdf, entry["end_month"], cache, tesseract)
            entry.update(status="accepted", extraction_method=method,
                         values=[v for year in sorted(rows) for v in rows[year]["counts"]])
            for year, row in rows.items():
                for month, value in enumerate(row["counts"], 1):
                    observation = {"metric_id": "new_diagnosed_cases_period", "value": value,
                                   "month": f"{year}-{month:02d}", "report_period": period,
                                   "source_path": str(pdf), "source_sha256": entry["source_sha256"],
                                   "source_line": row["source_line"], "extraction_method": method,
                                   "observation_role": "direct_target", "allowed_use": "retrospective_vintage_diagnostic",
                                   "measurement_semantics": "flow_count", "unit": "people", "geography": "national",
                                   "first_publication_date": None, "mirror_posted_at": entry.get("mirror_posted_at")}
                    observation.update(source_id=pdf.stem, time_start=f"{year}-{month:02d}",
                                       time_end=f"{year}-{month:02d}", time_granularity="month",
                                       population="all_reported_diagnoses", source_tier="official_report_mirror",
                                       support_partition="report_vintage_monthly_table",
                                       leakage_status="asof_filter_required_first_release_unverified")
                    observation["row_hash"] = hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()
                    observations.append(observation)
            lab = re.search(r"\(CrCLs\)\s+in OHASIS increased from\s+\d+\s+facilities in 2021\s+to\s+(\d+)", text)
            if lab:
                lab_rows.append({"metric_id": "confirming_laboratories_asof_count", "period": period,
                                 "value": int(lab.group(1)), "source_path": str(pdf), "source_sha256": entry["source_sha256"],
                                 "observation_role": "prior_context", "measurement_semantics": "reporting_process_covariate",
                                 "allowed_use": "context_only_not_monthly_completeness_denominator", "source_span": lab.group(0)})
        except (ValueError, subprocess.SubprocessError) as error:
            entry.update(status="quarantined", reason=str(error))
        catalog.append(entry)
        print(f"{period}: {entry['status']} {entry.get('reason', '')}", flush=True)
    catalog.sort(key=lambda r: r["period"])
    accepted = [r for r in catalog if r["status"] == "accepted"]
    revisions = []
    for before, after in zip(accepted, accepted[1:]):
        for index, (a, b) in enumerate(zip(before["values"], after["values"])):
            revisions.append({"month": f"{FIRST_YEAR + index // 12}-{index % 12 + 1:02d}",
                              "earlier_period": before["period"], "later_period": after["period"],
                              "earlier_count": a, "later_count": b, "revision": b - a})
    replays, decisions = [], []
    for issued in CONTRACT["historical_origins"]:
        target_period = issued[:4] + f"-Q{(int(issued[5:7]) - 1) // 3 + 1}"
        target = next((r for r in accepted if r["period"] == target_period), None)
        if target is None or not target.get("mirror_posted_at"):
            continue
        vintage = asof_vintage(catalog, issued)
        choice = choose_at_origin(replays, issued)
        actual = sum(target["values"][-3:])
        for family in FAMILIES:
            result = predict_period(vintage, family, issued[:7])
            replays.append({"issued_at": issued, "target_period": target_period, "actual": actual,
                            "target_posted_at": target["mirror_posted_at"], **result,
                            "absolute_error": abs(result["prediction"] - actual), "selected": family == choice})
        decisions.append({"origin": issued, "family": choice, "target": target_period,
                          "reason": "only previously completed blocks posted by origin can select a family"})
    now = datetime.now(timezone.utc).isoformat()
    if now[:10] >= "2026-10-01":
        raise ValueError("The Q4 prospective window has started; a new preregistration is required")
    vintage = asof_vintage(catalog, now)
    choice = choose_at_origin(replays, now)
    forward = [{**predict_period(vintage, f, "2026-10"), "selected": f == choice} for f in FAMILIES]
    project = Q2_PDF.parents[5]
    frozen_path = project / "docs/r96_inputs/r95_frozen_report.json"
    frozen = json.loads(frozen_path.read_text())
    r41 = next(row for row in frozen["future_anchor_forecast_rows"] if row["quarter"] == "2026-Q4")
    stock = [float(r41[m]) for m in ("diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed")]
    assert all(a >= b for a, b in zip(stock, stock[1:])) and stock[-1] >= 0
    lock = {"generated_at": now, "target_period": "2026-Q4", "target_start": "2026-10-01", "target_end": "2026-12-31",
            "information_cutoff": now, "R41_reference_unchanged": r41, "monthly_candidates": forward,
            "selected_monthly_family": choice, "scoring_metric": "new_diagnosed_cases_period",
            "outcome_definition": CONTRACT["outcome_definition"],
            "R41_frozen_source_sha256": _digest(frozen_path),
            "outcome_policy": "first complete Q4 HASP report; later revisions scored separately; absent outcomes remain unscored",
            "promotion_policy": "no promotion from a single future quarter; retain existing full-cascade gates",
            "quarter_carry_definition": "repeat most recently observed quarter across the unobserved gap",
            "stock_cone_valid": True, "observed_outcome": None, "status": "prospective_forecasts_frozen_unscored"}
    _write(output / "prospective_Q4_lock.json", lock)
    report = {"run_id": output.name, "generated_at": now, "contract": CONTRACT, "catalog": catalog,
              "accepted_reports": len(accepted), "quarantined_reports": len(catalog) - len(accepted),
              "monthly_vintage_rows": len(observations), "revisions": revisions, "laboratory_context": lab_rows,
              "availability_replays": replays, "selection_decisions": decisions, "prospective_Q4": lock,
              "gate": {"status": "diagnostic_and_prospective_lock_only", "champion": None,
                       "first_release_confirmed": False, "historical_bytes_verified": False,
                       "publication_superiority_allowed": False, "R41_changed": False}}
    _write(output / "report.json", report)
    _write(output / "observation_ledger.json", observations)
    _write(output / "claim_card.json", report["gate"])
    _write(output / "manifest.json", {"generated_at": now, "source_pdf_hashes": {r["source_path"]: r["source_sha256"] for r in catalog},
                                      "code_sha256": _digest(Path(__file__)),
                                      "R96_forecast_code_sha256": _digest(Path(__file__).with_name("r96_monthly_diagnosis_state.py")),
                                      "R41_frozen_source_sha256": _digest(frozen_path),
                                      "outputs": {str(p.relative_to(output)): _digest(p) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=sandbox_repo_root() / "artifacts/runs" / RUN_ID)
    parser.add_argument("--tesseract", default=shutil.which("tesseract"))
    args = parser.parse_args()
    r = run(args.output_dir, tesseract=args.tesseract)
    print(json.dumps({k: r[k] for k in ("accepted_reports", "quarantined_reports", "monthly_vintage_rows", "gate")}, indent=2))


if __name__ == "__main__":
    main()
