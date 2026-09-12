from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    R11_EVALUATION_METRICS,
    _candidate_predictions,
    _carry_forward_prediction,
    _finite_float,
    _generated_at,
    _sha256,
)
from .r75_bulk_unaids_annual_challenge import _write_csv
from .r94_2026_q1_hasp_intake_gate import R94_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R95_SCHEMA_VERSION = "phase3_dynamic.r95_2026_q2_hasp_intake_gate.v1"
R95_RUN_ID = "p3d-r95-2026-q2-hasp-intake-gate-20260910-s00"
R95_SOURCE_ID = "doh_hasp_2026_q2_user_provided_pdf"
R95_FAMILY = "r41_monotone_growth_component_process"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PDF_PATH = (
    PROJECT_ROOT
    / "src"
    / "epigraph_ph"
    / "harp_archive"
    / "HIV_Data"
    / "downloaded_hasp"
    / "2026_Q2-HIV-AIDS-Surveillance-of-the-Philippines.pdf"
)
DEFAULT_INPUT_PDF_PATH = Path("/media/gaurav/New_Volume/EpiGraph_PH/docs/harp_probe/2026_Q2 HIV & AIDS Surveillance of the Philippines.pdf")
DEFAULT_CANONICAL_EPIGRAPH_ROOT = Path("/media/gaurav/New_Volume/EpiGraph_PH")
R94_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R94_RUN_ID
    / "analysis"
    / "r94_2026_q1_hasp_intake_gate_report.json"
)
COMPARISON_METRICS: tuple[str, ...] = R11_EVALUATION_METRICS
STOCK_COMPARISON_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
DIRECT_Q2_MODEL_METRICS: frozenset[str] = frozenset(COMPARISON_METRICS + ("deaths_reported_period",))
VALIDATION_ONLY_METRICS: frozenset[str] = frozenset({"estimated_plhiv"})
AUXILIARY_METRICS: frozenset[str] = frozenset(
    {
        "advanced_hiv_cases_period",
        "median_cd4_at_enrollment",
        "art_ever_enrolled_cumulative",
        "art_ltfu_cumulative",
        "art_dead_cumulative",
        "art_transfer_out_cumulative",
        "art_stopped_cumulative",
        "prep_newly_enrolled_period",
        "prep_ever_enrolled_cumulative",
        "prep_refill_returned_period",
        "prep_non_returnees_cumulative",
        "prep_non_returnees_hiv_positive_cumulative",
    }
)
STOCK_METRICS: frozenset[str] = frozenset(
    {
        "estimated_plhiv",
        "diagnosed_plhiv",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
        "diagnosed_cases_cumulative",
        "deaths_reported_cumulative",
        "art_ever_enrolled_cumulative",
        "art_ltfu_cumulative",
        "art_dead_cumulative",
        "art_transfer_out_cumulative",
        "art_stopped_cumulative",
        "prep_ever_enrolled_cumulative",
        "prep_non_returnees_cumulative",
        "prep_non_returnees_hiv_positive_cumulative",
    }
)
FLOW_METRICS: frozenset[str] = frozenset(
    {
        "new_diagnosed_cases_period",
        "deaths_reported_period",
        "advanced_hiv_cases_period",
        "newly_enrolled_to_treatment",
        "prep_newly_enrolled_period",
    }
)
REGION_LABELS: tuple[str, ...] = (
    "NCR",
    "CARAGA",
    "BARMM",
    "CAR",
    "NIR",
    "4A",
    "4B",
    "1",
    "2",
    "3",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
)
MONTH_LABELS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _clean_number(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "").replace(" ", "")
    if text.endswith("%"):
        text = text[:-1]
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _capture_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return None if match is None else _clean_number(match.group(1))


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _row_role(metric_id: str) -> str:
    if metric_id in DIRECT_Q2_MODEL_METRICS:
        return "direct_target"
    if metric_id in VALIDATION_ONLY_METRICS:
        return "validation_only"
    if metric_id in AUXILIARY_METRICS:
        return "auxiliary_likelihood"
    return "prior_context"


def _row_semantics(metric_id: str, unit: str) -> str:
    if metric_id in VALIDATION_ONLY_METRICS:
        return "modeled_estimate"
    if metric_id in STOCK_METRICS:
        return "stock_anchor"
    if metric_id in FLOW_METRICS:
        return "flow_count"
    if str(unit).lower() == "percent" or metric_id.endswith("_percent") or metric_id.endswith("_coverage"):
        return "proportion"
    return "reporting_process_covariate"


def _row_hash(row: dict[str, Any]) -> str:
    payload = {
        key: row.get(key)
        for key in (
            "source_id",
            "source_path",
            "metric_id",
            "time_start",
            "time_end",
            "time_granularity",
            "geography",
            "population",
            "value",
            "unit",
            "extraction_method",
            "source_tier",
        )
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _add_row(
    rows: list[dict[str, Any]],
    *,
    metric_id: str,
    value: float | None,
    source_path: Path,
    time_start: str = "2026-04",
    time_end: str = "2026-06",
    time_granularity: str = "quarterly",
    quarter: str = "2026-Q2",
    geography: str = "national",
    population: str = "all",
    unit: str = "people",
    extraction_method: str = "pdftotext_regex_2026_q2_hasp",
    source_page: str = "",
    notes: str = "",
) -> None:
    if value is None:
        return
    role = _row_role(metric_id)
    row = {
        "source_id": R95_SOURCE_ID,
        "source_path": source_path.as_posix(),
        "metric_id": metric_id,
        "metric_name": metric_id,
        "time_start": time_start,
        "time_end": time_end,
        "time_granularity": time_granularity,
        "quarter": quarter,
        "geography": geography,
        "population": population,
        "value": float(value),
        "unit": unit,
        "extraction_method": extraction_method,
        "source_tier": "official_user_provided_pdf",
        "observation_role": role,
        "allowed_use": role,
        "measurement_semantics": _row_semantics(metric_id, unit),
        "support_partition": "expanded_support_2026_q2_hasp",
        "leakage_status": "post_2026_q1_holdout_for_q2_scoring",
        "source_page": source_page,
        "notes": notes,
    }
    row["row_hash"] = _row_hash(row)
    rows.append(row)


def extract_pdf_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise FileNotFoundError("pdftotext is required for R95 PDF extraction.")
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _extract_national_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    captures = {
        "estimated_plhiv": r"there would have been\s+([\d,]+)\s+estimated People Living with HIV",
        "diagnosed_plhiv": r"As of June 2026,\s+([\d,]+)\s+\(56%",
        "alive_on_art": r"Further,\s+([\d,]+)\s+\(69%.*?Antiretroviral Therapy",
        "tested_for_viral_load": r"of which\s+([\d,]+)\s+\(59%\).*?tested for viral load",
        "virally_suppressed": r"Among those tested for VL,\s+([\d,]+)\s+\(97%\)\s+were virally",
        "target_diagnosed_plhiv_95": r"Targets:\s*\n\s*([\d,]+)",
        "target_alive_on_art_95": r"\n\s*([\d,]+)\s+PLHIV on ART",
        "target_virally_suppressed_95": r"PLHIV on ART\s+([\d,]+)\s+PLHIV on ART who were tested",
        "gap_to_diagnosed_95": r"\n\s*([\d,]+)\s+PLHIV to be\s+PLHIV to",
        "gap_to_return_to_art": r"\n\s*([\d,]+)\s+PLHIV to be\s+tested for",
        "prep_newly_enrolled_period": r"From April to June 2026,\s+there were\s+([\d,]+)\s+clients newly enrolled",
        "prep_ever_enrolled_cumulative": r"Since the implementation of PrEP in March 2021,\s+a total of\s+([\d,]+)\s+clients have been",
        "prep_refill_returned_period": r"Among the total enrolled before 2026,\s+([\d,]+)\s+\(25%\)\s+returned",
        "prep_newly_enrolled_ytd": r"while\s+17%\s+\(([\d,]+)\)\s+were newly enrolled",
        "prep_non_returnees_cumulative": r"Of the\s+([\d,]+)\s+non-returnees,\s+[\d,]+\s+\(3%\)\s+tested positive",
        "prep_non_returnees_hiv_positive_cumulative": r"Of the\s+[\d,]+\s+non-returnees,\s+([\d,]+)\s+\(3%\)\s+tested positive",
        "new_diagnosed_cases_period": r"From April to June 2026,\s+there were\s+([\d,]+)\s+confirmed HIV-positive individuals",
        "advanced_hiv_cases_period": r"Of the recorded cases for this quarter,\s+([\d,]+)\s+\(28%\)\s+had an advanced",
        "diagnosed_cases_cumulative": r"Cumulatively,\s+([\d,]+)\s+confirmed HIV cases have been reported",
        "average_cases_per_day": r"quarter average cases per day are\s+([\d,]+)",
        "newly_enrolled_to_treatment": r"there were\s+([\d,]+)\s+people with HIV who were enrolled in treatment",
        "median_cd4_at_enrollment": r"median CD4.*?was\s+([\d,]+)\s+cells",
        "art_ever_enrolled_cumulative": r"A total of\s+([\d,]+)\s+people with HIV have ever been",
        "art_no_longer_receiving_cumulative": r"As of June 2026,\s+([\d,]+)\s+\(22%.*?were previously on ART",
        "art_ltfu_cumulative": r"This group includes\s+([\d,]+)\s+individuals\s+who were lost to follow-up",
        "art_stopped_cumulative": r"lost to follow-up,\s+([\d,]+)\s+who stopped ART",
        "art_transfer_out_cumulative": r"and\s+([\d,]+)\s+who reported migrating",
        "deaths_reported_period": r"From April to June 2026,\s+([\d,]+)\s+deaths from any cause",
        "deaths_reported_cumulative": r"Since January 1984,\s+a total of\s+([\d,]+)\s+deaths have been reported",
        "new_hiv_positive_pregnant_women_period": r"From April to June 2026,\s+there were\s+([\d,]+)\s+HIV-positive women",
        "new_hiv_positive_tgw_period": r"From April to June 2026,\s+there were\s+([\d,]+)\s+newly reported cases who identified\s+as transgender",
        "new_hiv_positive_migrant_workers_period": r"From April to June 2026,\s+([\d,]+)\s+migrant workers were reported",
        "new_transactional_sex_cases_period": r"From April to June 2026,\s+([\d,]+)\s+\(11%\)\s+of the newly diagnosed cases engaged",
    }
    for metric_id, pattern in captures.items():
        unit = "people_per_day" if metric_id == "average_cases_per_day" else "people"
        _add_row(rows, metric_id=metric_id, value=_capture_number(text, pattern), source_path=source_path, unit=unit, source_page="national_text")
    gap_art = _capture_number(text, r"\n\s*([\d,]+)\s+Strategic Information Unit")
    _add_row(rows, metric_id="gap_to_art_95", value=gap_art, source_path=source_path, source_page="figure_1")
    gap_pair = re.search(r"\n\s*([\d,]+)\s+([\d,]+)\s+Population Census", text, flags=re.IGNORECASE | re.DOTALL)
    if gap_pair:
        _add_row(rows, metric_id="gap_to_vl_tested_95", value=_clean_number(gap_pair.group(1)), source_path=source_path, source_page="figure_1")
        _add_row(rows, metric_id="gap_to_viral_suppression_95", value=_clean_number(gap_pair.group(2)), source_path=source_path, source_page="figure_1")
    _add_row(
        rows,
        metric_id="on_art_not_suppressed",
        value=_capture_number(text, r"while\s+([\d,]+)\s+\(3%\)\s+were not virally suppressed"),
        source_path=source_path,
        source_page="page_5",
    )
    return rows


def _extract_monthly_diagnosis_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    from .hasp_monthly_table import parse_monthly_diagnosis_table

    rows: list[dict[str, Any]] = []
    for table_row in parse_monthly_diagnosis_table(text, end_month="2026-06"):
        year = table_row["year"]
        monthly_values = table_row["counts"]
        values = monthly_values + [table_row["average"]]
        for month_index, value in enumerate(monthly_values[:12], start=1):
            _add_row(
                rows,
                metric_id="new_diagnosed_cases_period",
                value=float(value),
                source_path=source_path,
                time_start=f"{year:04d}-{month_index:02d}",
                time_end=f"{year:04d}-{month_index:02d}",
                time_granularity="monthly",
                quarter=f"{year:04d}-Q{((month_index - 1) // 3) + 1}",
                extraction_method="pdftotext_regex_figure_3_monthly_table",
                source_page="figure_3",
                notes=f"monthly_{MONTH_LABELS[month_index - 1]}",
            )
        _add_row(
            rows,
            metric_id="new_diagnosed_cases_monthly_average",
            value=float(values[-1]),
            source_path=source_path,
            time_start=f"{year:04d}-01",
            time_end=f"{year:04d}-{min(len(monthly_values), 12):02d}",
            time_granularity="annual_or_partial_year_average",
            quarter=f"{year:04d}-Q{((min(len(monthly_values), 12) - 1) // 3) + 1}",
            unit="people_per_month",
            extraction_method="pdftotext_regex_figure_3_monthly_table",
            source_page="figure_3",
        )
    return rows


def _extract_quarterly_prep_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = re.search(r"\b(202[3-6])\s+([\d,]+)\s+([\d,]+)(?:\s+([\d,]+)\s+([\d,]+))?", line)
        if not match or "," not in match.group(2):
            continue
        year = int(match.group(1))
        values = [value for value in match.groups()[1:] if value]
        for quarter_index, value in enumerate(values[:4], start=1):
            _add_row(
                rows,
                metric_id="prep_newly_enrolled_period",
                value=_clean_number(value),
                source_path=source_path,
                time_start=f"{year:04d}-{((quarter_index - 1) * 3) + 1:02d}",
                time_end=f"{year:04d}-{quarter_index * 3:02d}",
                time_granularity="quarterly",
                quarter=f"{year:04d}-Q{quarter_index}",
                extraction_method="pdftotext_regex_figure_2_quarterly_prep_table",
                source_page="figure_2",
            )
    return rows


def _extract_region_new_diagnosis_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = text.find("Figure 4. Distribution of newly diagnosed HIV")
    end = text.find("Sex and Age", start)
    block = text[start:end] if start >= 0 and end > start else ""
    region_pattern = "|".join(re.escape(region) for region in REGION_LABELS)
    for match in re.finditer(rf"^\s*({region_pattern})\s+([\d,]+)\s+(\d+)%", block, flags=re.MULTILINE):
        region, count, percent = match.groups()
        _add_row(rows, metric_id="new_diagnosed_cases_period", value=_clean_number(count), source_path=source_path, geography=region, extraction_method="pdftotext_regex_figure_4_region_table", source_page="figure_4")
        _add_row(rows, metric_id="new_diagnosed_cases_share_percent", value=_clean_number(percent), source_path=source_path, geography=region, unit="percent", extraction_method="pdftotext_regex_figure_4_region_table", source_page="figure_4")
    return rows


def _extract_region_cumulative_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = text.find("Table 1. Number of diagnosed HIV cases")
    end = text.find("Sex and Age", start)
    block = text[start:end] if start >= 0 and end > start else ""
    region_pattern = "|".join(re.escape(region) for region in REGION_LABELS)
    row_pattern = rf"^\s*({region_pattern})\s+([\d,]+)\s+\d+%?\s+([\d,]+)\s+(?:<\d+|\d+)%\s+(\d+)\s*$"
    for match in re.finditer(row_pattern, block, flags=re.MULTILINE):
        region, recent_count, cumulative_count, crcls = match.groups()
        for metric_id, value, unit in (
            ("diagnosed_cases_2020_to_2026q2_cumulative", recent_count, "people"),
            ("diagnosed_cases_cumulative", cumulative_count, "people"),
            ("crcl_facilities_count", crcls, "facilities"),
        ):
            _add_row(rows, metric_id=metric_id, value=_clean_number(value), source_path=source_path, geography=region, unit=unit, extraction_method="pdftotext_regex_table_1_region_cumulative", source_page="table_1")
    return rows


def _extract_art_outcome_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = text.find("Table 4.")
    end = text.find("Viral Load", start)
    block = text[start:end] if start >= 0 and end > start else ""
    region_pattern = "|".join(re.escape(region) for region in REGION_LABELS)
    row_pattern = rf"^\s*({region_pattern})\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+(\d+)%"
    for match in re.finditer(row_pattern, block, flags=re.MULTILINE):
        region, alive, ltfu, dead, transfer, stopped, total, ltfu_percent = match.groups()
        for metric_id, value, unit in (
            ("alive_on_art", alive, "people"),
            ("art_ltfu_cumulative", ltfu, "people"),
            ("art_dead_cumulative", dead, "people"),
            ("art_transfer_out_cumulative", transfer, "people"),
            ("art_stopped_cumulative", stopped, "people"),
            ("art_ever_enrolled_cumulative", total, "people"),
            ("art_ltfu_percent", ltfu_percent, "percent"),
        ):
            _add_row(rows, metric_id=metric_id, value=_clean_number(value), source_path=source_path, geography=region, unit=unit, extraction_method="pdftotext_regex_table_4_art_outcome", source_page="table_4")
    return rows


def _extract_vl_table_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = text.find("Table 5.")
    end = text.find("MORTALITY", start)
    block = text[start:end] if start >= 0 and end > start else ""
    region_pattern = "|".join(re.escape(region) for region in REGION_LABELS)
    row_pattern = rf"^\s*({region_pattern})\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%"
    for match in re.finditer(row_pattern, block, flags=re.MULTILINE):
        region, alive, tested, tested_pct, suppressed, suppressed_pct = match.groups()
        for metric_id, value, unit in (
            ("alive_on_art", alive, "people"),
            ("tested_for_viral_load", tested, "people"),
            ("vl_testing_given_art_percent", tested_pct, "percent"),
            ("virally_suppressed", suppressed, "people"),
            ("suppression_given_vl_tested_percent", suppressed_pct, "percent"),
        ):
            _add_row(rows, metric_id=metric_id, value=_clean_number(value), source_path=source_path, geography=region, unit=unit, extraction_method="pdftotext_regex_table_5_vl_suppression", source_page="table_5")
    return rows


def _add_cascade_table_row(
    rows: list[dict[str, Any]],
    *,
    label: str,
    values: list[str],
    source_path: Path,
    table_id: str,
    geography: str = "national",
    population: str = "all",
) -> None:
    specs = (
        ("estimated_plhiv", "people"),
        ("diagnosed_plhiv", "people"),
        ("diagnosed_coverage_percent", "percent"),
        ("alive_on_art", "people"),
        ("art_among_diagnosed_percent", "percent"),
        ("tested_for_viral_load", "people"),
        ("vl_testing_given_art_percent", "percent"),
        ("virally_suppressed", "people"),
        ("suppression_given_vl_tested_percent", "percent"),
        ("suppression_given_art_percent", "percent"),
    )
    for (metric_id, unit), value in zip(specs, values, strict=False):
        _add_row(
            rows,
            metric_id=metric_id,
            value=_clean_number(value),
            source_path=source_path,
            geography=geography,
            population=population,
            unit=unit,
            extraction_method=f"pdftotext_regex_{table_id}",
            source_page="annex",
            notes=label,
        )


def _extract_annex_cascade_rows(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    region_pattern = "|".join(re.escape(region) for region in REGION_LABELS)
    row_pattern = rf"^\s*({region_pattern})\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%"
    for match in re.finditer(row_pattern, text, flags=re.MULTILINE):
        _add_cascade_table_row(rows, label=match.group(1), values=list(match.groups()[1:]), source_path=source_path, table_id="annex_region_cascade", geography=match.group(1))
    normalized = _normalize_space(text)
    group_specs = {
        "children_lt10": r"CHILDREN\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%\s+\(<10\)",
        "adolescents_10_19": r"ADOLESCENTS\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%\s+\(10-19\)",
        "youth_15_24": r"YOUTH\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%\s+\(15-24\)",
        "adults_25_plus": r"ADULTS \(25\+\)\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%",
        "msm": r"MALES HAVING SEX WITH MALES\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%\s+\(MSM\)",
        "pwid": r"PERSONS WHO INJECT DRUGS\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%\s+\(PWID\)",
        "other_males": r"OTHER MALES\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%",
        "other_females": r"OTHER FEMALES\s+([\d,]+)\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+([\d,]+)\s+(\d+)%\s+(\d+)%",
    }
    for population, pattern in group_specs.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            _add_cascade_table_row(rows, label=population, values=list(match.groups()), source_path=source_path, table_id="annex_population_cascade", population=population)
    return rows


def _extract_vulnerable_population_cascades(text: str, source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized = _normalize_space(text)
    specs = {
        "pregnant_women_diagnosed_past_year": r"pregnant women diagnosed with HIV within the past year \(n=180\).*?only\s+([\d,]+).*?retained on ART.*?Among those on treatment,\s+([\d,]+).*?viral load testing.*?of whom\s+([\d,]+).*?virally suppressed",
        "tgw": r"Among the diagnosed cases of TGW,\s+([\d,]+).*?currently alive.*?only\s+([\d,]+).*?retained on ART.*?only\s+([\d,]+).*?tested for viral load.*?\(([\d,]+)\)\s+viral",
        "migrant_workers": r"Among the diagnosed cases of migrant workers,\s+([\d,]+).*?currently alive.*?only\s+([\d,]+).*?retained on ART.*?only\s+([\d,]+).*?tested for viral load.*?\(([\d,]+)\)\s+showing",
        "transactional_sex": r"Among the diagnosed cases who had a history of transactional sex,\s+([\d,]+).*?currently alive.*?only\s+([\d,]+).*?retained on ART.*?only\s+([\d,]+).*?tested for viral load.*?\(([\d,]+)\)\s+viral",
    }
    metric_specs = (
        ("diagnosed_plhiv", "people"),
        ("alive_on_art", "people"),
        ("tested_for_viral_load", "people"),
        ("virally_suppressed", "people"),
    )
    for population, pattern in specs.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        values = list(match.groups())
        if population == "pregnant_women_diagnosed_past_year":
            values = ["180", *values]
        for (metric_id, unit), value in zip(metric_specs, values, strict=False):
            _add_row(rows, metric_id=metric_id, value=_clean_number(value), source_path=source_path, population=population, unit=unit, extraction_method="pdftotext_regex_vulnerable_population_cascade", source_page="pages_5_6")
    return rows


def extract_hasp_2026_q2_rows(text: str, source_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_extract_national_rows(text, source_path))
    rows.extend(_extract_quarterly_prep_rows(text, source_path))
    rows.extend(_extract_monthly_diagnosis_rows(text, source_path))
    rows.extend(_extract_region_new_diagnosis_rows(text, source_path))
    rows.extend(_extract_region_cumulative_rows(text, source_path))
    rows.extend(_extract_art_outcome_rows(text, source_path))
    rows.extend(_extract_vl_table_rows(text, source_path))
    rows.extend(_extract_annex_cascade_rows(text, source_path))
    rows.extend(_extract_vulnerable_population_cascades(text, source_path))
    return rows, _quality_flags(rows)


def _quality_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    national = {
        str(row.get("metric_id")): float(row.get("value") or 0.0)
        for row in rows
        if str(row.get("geography")) == "national" and str(row.get("population")) == "all"
    }
    for metric in ("estimated_plhiv", "diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"):
        regional_sum = sum(
            float(row.get("value") or 0.0)
            for row in rows
            if str(row.get("metric_id")) == metric
            and str(row.get("geography")) in REGION_LABELS
            and str(row.get("source_page")) == "annex"
        )
        national_value = national.get(metric)
        if national_value is not None and regional_sum:
            tolerance = max(50.0, 0.001 * national_value)
            flags.append(
                {
                    "flag_id": f"annex_regional_sum_check_{metric}",
                    "status": "pass" if abs(regional_sum - national_value) <= tolerance else "warn",
                    "metric_id": metric,
                    "national_value": national_value,
                    "regional_sum": regional_sum,
                    "absolute_difference": abs(regional_sum - national_value),
                    "interpretation": "regional annex and national cascade are close enough for extraction use"
                    if abs(regional_sum - national_value) <= tolerance
                    else "regional annex does not exactly reconcile to national cascade; keep lineage explicit",
                }
            )
    return flags


def _direct_q2_target_row(extracted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target: dict[str, Any] = {"quarter": "2026-Q2", "metric_provenance": {}}
    for row in extracted_rows:
        if str(row.get("geography")) != "national" or str(row.get("population")) != "all":
            continue
        metric = str(row.get("metric_id") or "")
        if metric not in COMPARISON_METRICS:
            continue
        if str(row.get("time_granularity") or "") != "quarterly":
            continue
        if str(row.get("observation_role")) != "direct_target":
            continue
        target[metric] = float(row.get("value") or 0.0)
        target["metric_provenance"][metric] = {
            "tier": "exact_observed",
            "source_id": row.get("source_id"),
            "source_path": row.get("source_path"),
            "source_tier": row.get("source_tier"),
            "observation_role": row.get("observation_role"),
            "allowed_use": row.get("allowed_use"),
            "support_partition": row.get("support_partition"),
            "leakage_status": row.get("leakage_status"),
            "row_hash": row.get("row_hash"),
        }
    return target


def _metric_scale(train_rows: list[dict[str, Any]], metric: str, actual: float | None) -> float:
    values = [abs(float(value)) for value in (_finite_float(row.get(metric)) for row in train_rows) if value is not None]
    if actual is not None:
        values.append(abs(float(actual)))
    return max(values) if values else 1.0


def _comparison_rows(
    *,
    train_rows: list[dict[str, Any]],
    target_row: dict[str, Any],
    candidate_row: dict[str, Any],
    carry_row: dict[str, Any],
    comparison_label: str,
    metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        actual = _finite_float(target_row.get(metric))
        if actual is None:
            continue
        candidate = _finite_float(candidate_row.get(metric))
        carry = _finite_float(carry_row.get(metric))
        scale = _metric_scale(train_rows, metric, actual)
        candidate_abs_error = None if candidate is None else abs(float(candidate) - float(actual))
        carry_abs_error = None if carry is None else abs(float(carry) - float(actual))
        rows.append(
            {
                "comparison_label": comparison_label,
                "metric_name": metric,
                "quarter": "2026-Q2",
                "actual_value": float(actual),
                "candidate_family": R95_FAMILY,
                "candidate_value": candidate,
                "carry_forward_value": carry,
                "scale": float(scale),
                "candidate_abs_error": candidate_abs_error,
                "carry_forward_abs_error": carry_abs_error,
                "candidate_norm_error": None if candidate_abs_error is None else candidate_abs_error / max(scale, 1e-9),
                "carry_forward_norm_error": None if carry_abs_error is None else carry_abs_error / max(scale, 1e-9),
                "candidate_minus_carry_norm_error": None
                if candidate_abs_error is None or carry_abs_error is None
                else (candidate_abs_error - carry_abs_error) / max(scale, 1e-9),
                "candidate_relative_error": None
                if candidate_abs_error is None or abs(float(actual)) <= 1e-9
                else candidate_abs_error / abs(float(actual)),
                "carry_forward_relative_error": None
                if carry_abs_error is None or abs(float(actual)) <= 1e-9
                else carry_abs_error / abs(float(actual)),
            }
        )
    return rows


def _mean_error(rows: list[dict[str, Any]], key: str, metrics: tuple[str, ...] | None = None) -> float | None:
    allowed = set(metrics) if metrics is not None else None
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and (allowed is None or str(row.get("metric_name")) in allowed)
    ]
    return sum(values) / len(values) if values else None


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_mean = _mean_error(rows, "candidate_norm_error")
    carry_mean = _mean_error(rows, "carry_forward_norm_error")
    stock_candidate_mean = _mean_error(rows, "candidate_norm_error", STOCK_COMPARISON_METRICS)
    stock_carry_mean = _mean_error(rows, "carry_forward_norm_error", STOCK_COMPARISON_METRICS)
    blockers: list[str] = []
    if len(rows) < 5:
        blockers.append("fewer_than_five_q2_model_metrics_scored")
    if candidate_mean is None or carry_mean is None:
        blockers.append("missing_candidate_or_carry_scores")
    elif candidate_mean > carry_mean:
        blockers.append("q1_anchored_r41_does_not_beat_carry_forward_on_q2_mean")
    stock_rows = [row for row in rows if str(row.get("metric_name")) in STOCK_COMPARISON_METRICS]
    if any(float(row.get("candidate_norm_error") or 0.0) > float(row.get("carry_forward_norm_error") or 0.0) for row in stock_rows):
        blockers.append("stock_metric_regression_against_q1_carry_forward")
    flow_rows = [row for row in rows if str(row.get("metric_name")) == "new_diagnosed_cases_period"]
    flow_regression = any(float(row.get("candidate_norm_error") or 0.0) > float(row.get("carry_forward_norm_error") or 0.0) for row in flow_rows)
    if blockers:
        status = "r95_q2_hasp_intake_only"
    elif flow_regression:
        status = "r95_q2_stock_anchor_promoted_diagnosis_flow_shock_flagged"
    else:
        status = "r95_q2_hasp_anchor_promoted_for_future_initialization"
    return {
        "status": status,
        "blockers": blockers,
        "candidate_family": R95_FAMILY,
        "candidate_mean_norm_error": candidate_mean,
        "carry_forward_mean_norm_error": carry_mean,
        "candidate_minus_carry_mean_norm_error": None if candidate_mean is None or carry_mean is None else float(candidate_mean - carry_mean),
        "stock_candidate_mean_norm_error": stock_candidate_mean,
        "stock_carry_forward_mean_norm_error": stock_carry_mean,
        "diagnosis_flow_regresses_against_carry_forward": flow_regression,
        "scored_metric_count": len(rows),
        "contract": (
            "The 2026-Q2 HASP PDF is scored after the R94 2026-Q1 anchor. Stock anchors may initialize future "
            "forecasts if the stock cone beats Q1 carry-forward. A diagnosis-flow regression is retained as a "
            "reporting/service-shock flag, not tuned away with the Q2 outcome."
        ),
    }


def _default_evidence_root() -> Path:
    if (DEFAULT_CANONICAL_EPIGRAPH_ROOT / "artifacts" / "runs").exists():
        return DEFAULT_CANONICAL_EPIGRAPH_ROOT
    return PROJECT_ROOT


def _load_training_rows(epigraph_root: Path, source_run_id: str | None, baseline_source_run_id: str | None, r94_report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = resolve_active_source_run_id(epigraph_root, preferred=source_run_id)
    baseline = resolve_baseline_source_run_id(epigraph_root, source_run_id=source, preferred=baseline_source_run_id)
    rows = build_observation_rows(epigraph_root, source, baseline_source_run_id=baseline, include_validation_only=False)
    train_rows = [row for row in rows if quarter_sort_key(str(row.get("quarter") or "")) <= quarter_sort_key("2025-Q4")]
    q1_report = dict(read_json(r94_report_path, default={}) or {})
    q1_target = dict(q1_report.get("q1_target_row") or {})
    if q1_target.get("quarter") == "2026-Q1":
        train_rows.append(q1_target)
    train_rows = sorted(train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    return train_rows, {
        "epigraph_root": epigraph_root.as_posix(),
        "source_run_id": source,
        "baseline_source_run_id": baseline,
        "r94_q1_anchor_report": r94_report_path.as_posix(),
        "r94_q1_anchor_available": bool(q1_target),
        "training_row_count": len(train_rows),
        "latest_training_quarter": train_rows[-1].get("quarter") if train_rows else None,
    }


def _future_anchor_forecasts(train_rows: list[dict[str, Any]], target_row: dict[str, Any]) -> list[dict[str, Any]]:
    anchored_rows = sorted([*train_rows, dict(target_row)], key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    holdout_rows = [{"quarter": quarter} for quarter in ("2026-Q3", "2026-Q4", "2027-Q1", "2027-Q2")]
    predictions, _summary = _candidate_predictions(anchored_rows, holdout_rows, family=R95_FAMILY)
    return predictions


def _write_svg_dashboard(path: Path, comparison_rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    width = 1120
    row_height = 54
    top = 96
    left = 320
    chart_width = 700
    height = top + row_height * max(len(comparison_rows), 1) + 74
    max_error = max(
        [
            float(value)
            for row in comparison_rows
            for value in (row.get("candidate_abs_error"), row.get("carry_forward_abs_error"))
            if value is not None
        ]
        or [1.0]
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f3e8"/>',
        '<text x="42" y="44" font-family="Georgia, serif" font-size="26" fill="#1f2d2a">R95 2026-Q2 HASP Holdout Error</text>',
        '<text x="42" y="72" font-family="Arial, sans-serif" font-size="14" fill="#52615a">Absolute error. Green = Q1-anchored R41, gray = Q1 carry-forward. Orange marker flags diagnosis-flow shock.</text>',
        f'<line x1="{left}" x2="{left + chart_width}" y1="{top - 12}" y2="{top - 12}" stroke="#c8c0b2"/>',
    ]
    for index, row in enumerate(comparison_rows):
        y = top + index * row_height
        metric = str(row.get("metric_name") or "")
        actual = float(row.get("actual_value") or 0.0)
        candidate = row.get("candidate_abs_error")
        carry = row.get("carry_forward_abs_error")
        candidate_width = 0 if candidate is None else int((float(candidate) / max_error) * chart_width)
        carry_width = 0 if carry is None else int((float(carry) / max_error) * chart_width)
        candidate_color = "#b96b24" if metric == "new_diagnosed_cases_period" else "#2d6a4f"
        lines.extend(
            [
                f'<text x="42" y="{y + 20}" font-family="Arial, sans-serif" font-size="13" fill="#1e2a24">{metric}</text>',
                f'<text x="42" y="{y + 38}" font-family="Arial, sans-serif" font-size="11" fill="#6f766f">actual {actual:,.0f}</text>',
                f'<rect x="{left}" y="{y + 5}" width="{carry_width}" height="16" rx="3" fill="#b8b4aa"/>',
                f'<rect x="{left}" y="{y + 26}" width="{candidate_width}" height="16" rx="3" fill="{candidate_color}"/>',
                f'<text x="{left + max(carry_width, 4) + 8}" y="{y + 18}" font-family="Arial, sans-serif" font-size="11" fill="#555">carry {0.0 if carry is None else float(carry):,.0f}</text>',
                f'<text x="{left + max(candidate_width, 4) + 8}" y="{y + 39}" font-family="Arial, sans-serif" font-size="11" fill="#234">R41 {0.0 if candidate is None else float(candidate):,.0f}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("q2_holdout_gate") or {})
    lines = [
        "# Phase 3 R95 2026-Q2 HASP Intake Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Candidate family: `{gate.get('candidate_family')}`",
        f"- Candidate mean normalized error: `{gate.get('candidate_mean_norm_error')}`",
        f"- Carry-forward mean normalized error: `{gate.get('carry_forward_mean_norm_error')}`",
        f"- Stock candidate mean normalized error: `{gate.get('stock_candidate_mean_norm_error')}`",
        f"- Stock carry-forward mean normalized error: `{gate.get('stock_carry_forward_mean_norm_error')}`",
        f"- Diagnosis-flow shock flag: `{gate.get('diagnosis_flow_regresses_against_carry_forward')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Q2 Holdout Comparison",
        "",
        "| Metric | Actual | Q1-anchored R41 | Q1 carry-forward | R41 abs err | Carry abs err |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("q2_comparison_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{float(row.get('actual_value') or 0):,.0f}` | "
            f"`{float(row.get('candidate_value') or 0):,.0f}` | `{float(row.get('carry_forward_value') or 0):,.0f}` | "
            f"`{float(row.get('candidate_abs_error') or 0):,.0f}` | `{float(row.get('carry_forward_abs_error') or 0):,.0f}` |"
        )
    lines.extend(
        [
            "",
            "## Extraction Summary",
            "",
            f"- Extracted rows: `{report.get('extracted_row_count')}`",
            f"- Direct target rows: `{report.get('observation_role_counts', {}).get('direct_target', 0)}`",
            f"- Auxiliary rows: `{report.get('observation_role_counts', {}).get('auxiliary_likelihood', 0)}`",
            f"- Validation-only rows: `{report.get('observation_role_counts', {}).get('validation_only', 0)}`",
            "",
            "## Contract",
            "",
            str(gate.get("contract") or ""),
            "",
            "## Dashboard",
            "",
            f"![R95 dashboard]({report.get('artifact_paths', {}).get('svg_dashboard')})",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r95_2026_q2_hasp_intake_gate(
    *,
    run_id: str = R95_RUN_ID,
    pdf_path: Path | None = None,
    text_path: Path | None = None,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r94_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    pdf = DEFAULT_PDF_PATH if pdf_path is None else Path(pdf_path)
    text = Path(text_path).read_text(encoding="utf-8", errors="replace") if text_path is not None else extract_pdf_text(pdf)
    extracted_rows, quality_flags = extract_hasp_2026_q2_rows(text, pdf)
    target_row = _direct_q2_target_row(extracted_rows)
    evidence_root = _default_evidence_root() if epigraph_root is None else Path(epigraph_root)
    r94_path = R94_DEFAULT_REPORT if r94_report_path is None else Path(r94_report_path)
    train_rows, training_summary = _load_training_rows(evidence_root, source_run_id, baseline_source_run_id, r94_path)
    holdout_stub = [{"quarter": "2026-Q2"}]
    candidate_predictions, candidate_summary = _candidate_predictions(train_rows, holdout_stub, family=R95_FAMILY)
    carry_predictions = _carry_forward_prediction(train_rows, holdout_stub)
    candidate_row = candidate_predictions[0] if candidate_predictions else {"quarter": "2026-Q2"}
    carry_row = carry_predictions[0] if carry_predictions else {"quarter": "2026-Q2"}
    comparison_rows = _comparison_rows(
        train_rows=train_rows,
        target_row=target_row,
        candidate_row=candidate_row,
        carry_row=carry_row,
        comparison_label="q1_anchored_r41_vs_q1_carry_forward",
    )
    gate = _gate(comparison_rows)
    anchor_rows = [
        {
            **{key: value for key, value in target_row.items() if key != "metric_provenance"},
            "update_policy": "post_evaluation_latest_anchor_for_forecasts_after_2026_q2",
            "claim_boundary": "stock anchor update allowed; diagnosis-flow shock remains flagged",
        }
    ]
    future_forecasts = _future_anchor_forecasts(train_rows, target_row) if gate["status"] != "r95_q2_hasp_intake_only" else []
    role_counts = dict(Counter(str(row.get("observation_role") or "") for row in extracted_rows))
    metric_counts = dict(Counter(str(row.get("metric_id") or "") for row in extracted_rows))
    paths = {
        "json": analysis_dir / "r95_2026_q2_hasp_intake_gate_report.json",
        "markdown": analysis_dir / "r95_2026_q2_hasp_intake_gate_report.md",
        "extracted_rows_csv": analysis_dir / "r95_2026_q2_hasp_extracted_rows.csv",
        "comparison_rows_csv": analysis_dir / "r95_2026_q2_hasp_q2_comparison_rows.csv",
        "anchor_rows_csv": analysis_dir / "r95_2026_q2_hasp_anchor_rows.csv",
        "future_forecast_rows_csv": analysis_dir / "r95_2026_q2_hasp_anchor_future_forecast_rows.csv",
        "quality_flags_csv": analysis_dir / "r95_2026_q2_hasp_quality_flags.csv",
        "svg_dashboard": analysis_dir / "r95_2026_q2_hasp_q2_error_dashboard.svg",
    }
    report = {
        "schema_version": R95_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "candidate_family": R95_FAMILY,
        "source_pdf": {"path": pdf.as_posix(), "sha256": _sha256(pdf) if pdf.exists() else None},
        "training_summary": training_summary,
        "candidate_summary": candidate_summary,
        "q2_holdout_gate": gate,
        "q2_target_row": target_row,
        "q2_candidate_prediction_row": candidate_row,
        "q2_carry_forward_prediction_row": carry_row,
        "q2_comparison_rows": comparison_rows,
        "future_anchor_rows": anchor_rows,
        "future_anchor_forecast_rows": future_forecasts,
        "quality_flags": quality_flags,
        "extracted_row_count": len(extracted_rows),
        "observation_role_counts": role_counts,
        "metric_counts": metric_counts,
        "extracted_rows": extracted_rows,
        "artifact_paths": {key: value.as_posix() for key, value in paths.items()},
    }
    _write_csv(paths["extracted_rows_csv"], extracted_rows)
    _write_csv(paths["comparison_rows_csv"], comparison_rows)
    _write_csv(paths["anchor_rows_csv"], anchor_rows)
    _write_csv(paths["future_forecast_rows_csv"], future_forecasts)
    _write_csv(paths["quality_flags_csv"], quality_flags)
    _write_svg_dashboard(paths["svg_dashboard"], comparison_rows)
    _write_markdown(paths["markdown"], report)
    write_json(paths["json"], report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R95 2026-Q2 HASP intake and holdout gate.")
    parser.add_argument("--run-id", default=R95_RUN_ID)
    parser.add_argument("--pdf-path", default=None)
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r94-report-path", default=None)
    args = parser.parse_args()
    run_r95_2026_q2_hasp_intake_gate(
        run_id=str(args.run_id),
        pdf_path=None if args.pdf_path is None else Path(args.pdf_path),
        text_path=None if args.text_path is None else Path(args.text_path),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r94_report_path=None if args.r94_report_path is None else Path(args.r94_report_path),
    )


if __name__ == "__main__":
    _main()
