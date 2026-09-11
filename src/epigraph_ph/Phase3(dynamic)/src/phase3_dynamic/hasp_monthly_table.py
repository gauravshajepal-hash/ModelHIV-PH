"""Strict Figure 3 parsing for the two-column 2026 HASP reports."""

from __future__ import annotations

import re


def parse_monthly_diagnosis_table(text: str, *, end_month: str) -> list[dict]:
    caption = re.search(
        r"Figure\s+3\.\s+Number of monthly newly diagnosed HIV cases,\s+Jan\s+(20\d{2})",
        text,
    )
    if caption is None:
        raise ValueError("Missing monthly diagnosis Figure 3 caption")
    first_year = int(caption.group(1))
    last_year, last_month = map(int, end_month.split("-"))
    if not 1 <= last_month <= 12 or first_year > last_year:
        raise ValueError("Invalid monthly table interval")
    block = text[caption.end():].split("Geographic Distribution", 1)[0]
    by_year: dict[int, dict] = {}
    for line in block.splitlines():
        # Left-column prose can precede the table. Only the numeric suffix belongs to it.
        match = re.search(r"\b(20\d{2})\s+((?:[\d,]+\s+)+[\d,]+)\s*$", line)
        if match is None:
            continue
        year = int(match.group(1))
        if not first_year <= year <= last_year:
            continue
        values = [int(token.replace(",", "")) for token in match.group(2).split()]
        expected = last_month if year == last_year else 12
        if len(values) != expected + 1:
            raise ValueError(f"Monthly table {year}: expected {expected} months and AVG, got {len(values)} cells")
        counts, average = values[:-1], values[-1]
        # AVG is printed to the nearest person, so the exact rounding tolerance is half a unit.
        if abs(sum(counts) - average * expected) > expected / 2:
            raise ValueError(f"Monthly table {year}: counts do not reconcile with printed AVG")
        if year in by_year:
            raise ValueError(f"Duplicate monthly table row for {year}")
        by_year[year] = {"year": year, "counts": counts, "average": average, "source_line": line.strip()}
    missing = set(range(first_year, last_year + 1)) - set(by_year)
    if missing:
        raise ValueError(f"Incomplete monthly diagnosis table, missing years: {sorted(missing)}")
    return [by_year[year] for year in sorted(by_year)]
