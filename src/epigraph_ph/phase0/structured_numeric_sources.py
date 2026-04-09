from __future__ import annotations

import csv
import itertools
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

import pdfplumber
import requests

from epigraph_ph.geography import infer_region_code, normalize_geo_label
from epigraph_ph.registry.models import LiteratureRefDetail, has_verifiable_locator
from epigraph_ph.runtime import ensure_dir, read_json, write_json

_WDI_SERIES_SPECS: tuple[dict[str, Any], ...] = (
    {
        "canonical_name": "poverty_rate",
        "indicator_code": "SI.POV.NAHC",
        "title_fragment": "Poverty Headcount Ratio",
        "parameter_text": "poverty headcount ratio at national poverty line",
        "unit": "percent",
        "confidence": 0.9,
        "soft_ontology_tags": ["economics", "official_series", "poverty"],
        "soft_subparameter_hints": ["poverty", "economic_access_constraint", "cash_instability"],
        "linkage_targets": ["testing_uptake", "linkage_to_care", "retention_adherence"],
        "measurement_type": "rate",
        "denominator_type": "population",
        "normalization_basis": "percent",
        "value_semantics": "direct_observed",
    },
    {
        "canonical_name": "education",
        "indicator_code": "SE.SEC.CMPT.LO.ZS",
        "title_fragment": "Lower Secondary Completion Rate",
        "parameter_text": "lower secondary completion rate",
        "unit": "percent",
        "confidence": 0.88,
        "soft_ontology_tags": ["education", "official_series", "human_capital"],
        "soft_subparameter_hints": ["education", "health_literacy", "social_capital"],
        "linkage_targets": ["testing_uptake", "prevention_access"],
        "measurement_type": "rate",
        "denominator_type": "population",
        "normalization_basis": "percent",
        "value_semantics": "direct_observed",
    },
    {
        "canonical_name": "cash_instability",
        "indicator_code": "FP.CPI.TOTL.ZG",
        "title_fragment": "",
        "parameter_text": "consumer price inflation",
        "unit": "percent",
        "confidence": 0.86,
        "soft_ontology_tags": ["economics", "official_series", "inflation"],
        "soft_subparameter_hints": ["cash_instability", "economic_access_constraint"],
        "linkage_targets": ["linkage_to_care", "retention_adherence"],
        "measurement_type": "rate",
        "denominator_type": "population",
        "normalization_basis": "percent",
        "value_semantics": "direct_observed",
    },
)

_GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
_LOCAL_STRUCTURED_DOC_ROOT = Path(__file__).resolve().parents[3] / "docs" / "Pdf"
_LOCAL_TMP_DOC_ROOT = Path(__file__).resolve().parents[3] / "tmp" / "philhealth_probe"
_PHILHEALTH_ANNUAL_REPORT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "report_year": 2024,
        "file_name": "ar2024.pdf",
        "url": "https://www.philhealth.gov.ph/about_us/annual_report/ar2024.pdf",
        "title": "PhilHealth Annual Report 2024",
    },
    {
        "report_year": 2023,
        "file_name": "AR2023.pdf",
        "url": "https://www.philhealth.gov.ph/about_us/annual_report/AR2023.pdf",
        "title": "PhilHealth Annual Report 2023",
    },
)
_PHILHEALTH_PORTAL_JSON_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "coverage",
        "url": "https://philhealth.open.gov.ph/data/coverage.json",
        "file_name": "philhealth_open_coverage.json",
        "title": "PhilHealth Open Portal Coverage Data",
    },
    {
        "name": "financials",
        "url": "https://philhealth.open.gov.ph/data/financials.json",
        "file_name": "philhealth_open_financials.json",
        "title": "PhilHealth Open Portal Financial Data",
    },
    {
        "name": "statistics_2025",
        "url": "https://philhealth.open.gov.ph/data/statistics-charts-2025.json",
        "file_name": "philhealth_open_statistics_2025.json",
        "title": "PhilHealth Open Portal Statistics and Charts 2025",
    },
)
_LOCAL_PSA_POVERTY_SPECS: tuple[dict[str, str], ...] = (
    {
        "kind": "city_municipal_csv",
        "file_name": "190710_poverty-statistics(Poverty_City_Mun_2009,2012,2015).csv",
        "title": "PSA City and Municipal Poverty Statistics 2009 2012 2015",
    },
    {
        "kind": "province_csv",
        "file_name": "200604_updated-annual-per-capita-poverty-threshold-poverty-incidence-and-magnitude-of-poor-fami(By Province).csv",
        "title": "PSA Updated Poverty Incidence and Magnitude of Poor Families by Province",
    },
    {
        "kind": "province_csv",
        "file_name": "200305_annual-per-capita-poverty-threshold-poverty-incidence-among-families_by-region-and-provi(By Province).csv",
        "title": "PSA Annual Per Capita Poverty Threshold and Poverty Incidence among Families by Region and Province",
    },
    {
        "kind": "city_municipal_xlsx",
        "file_name": "2_2023 SAE_with PSGC_noHUC_06Feb2026.xlsx",
        "title": "PSA City and Municipal Level Poverty Estimates 2018 2021 2023",
    },
)
_LOCAL_FIES_STRUCTURAL_SPECS: tuple[dict[str, str], ...] = (
    {
        "file_name": "2018 Philippines Statistical Yearbook (2019).pdf",
        "title": "2018 Philippines Statistical Yearbook Table 2.2 Number of Families Average Annual Income and Expenditure by Region",
    },
)
_PHILHEALTH_REGION_METADATA: dict[str, tuple[str, str]] = {
    "NCR": ("National Capital Region", "ncr"),
    "CAR": ("Cordillera Administrative Region", "car"),
    "I": ("Ilocos Region", "region_i"),
    "II": ("Cagayan Valley", "region_ii"),
    "III": ("Central Luzon", "region_iii"),
    "IV-A": ("CALABARZON", "region_iv_a"),
    "IV-B": ("MIMAROPA", "region_iv_b"),
    "V": ("Bicol Region", "region_v"),
    "VI": ("Western Visayas", "region_vi"),
    "VII": ("Central Visayas", "region_vii"),
    "VIII": ("Eastern Visayas", "region_viii"),
    "IX": ("Zamboanga Peninsula", "region_ix"),
    "X": ("Northern Mindanao", "region_x"),
    "XI": ("Davao Region", "region_xi"),
    "XII": ("SOCCSKSARGEN", "region_xii"),
    "BARMM": ("Bangsamoro Autonomous Region in Muslim Mindanao", "barmm"),
    "CARAGA": ("Caraga", "region_xiii"),
}
_PHILHEALTH_REGION_PATTERN = "|".join(
    re.escape(label)
    for label in sorted(
        ["Head Office", *_PHILHEALTH_REGION_METADATA.keys()],
        key=len,
        reverse=True,
    )
)
_XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_FIES_REGION_LABELS: tuple[str, ...] = (
    "NCR",
    "CAR",
    "I-Ilocos Region",
    "II-Cagayan Valley",
    "III-Central Luzon",
    "IV-A-CALABARZON",
    "IV-B-MIMAROPA",
    "V-Bicol Region",
    "VI-Western Visayas",
    "VII-Central Visayas",
    "VIII-Eastern Visayas",
    "IX-Zamboanga Peninsula",
    "X-Northern Mindanao",
    "XI-Davao Region",
    "XII-SOCCSKSARGEN",
    "XIII-Caraga",
    "ARMM",
)


def _candidate_source_row(
    source_rows: Mapping[str, dict[str, Any]],
    *,
    platform: str,
    year: int | None = None,
    title_fragment: str = "",
) -> dict[str, Any] | None:
    candidates = [
        dict(row)
        for row in source_rows.values()
        if str(row.get("platform") or "").strip().lower() == platform.strip().lower()
    ]
    if not candidates:
        return None
    if year is not None:
        exact_year = [row for row in candidates if int(row.get("year") or -1) == int(year)]
        if exact_year:
            candidates = exact_year
    if title_fragment:
        fragment = title_fragment.strip().lower()
        fragment_matches = [row for row in candidates if fragment in str(row.get("title") or "").strip().lower()]
        if fragment_matches:
            candidates = fragment_matches
    candidates.sort(key=lambda row: (str(row.get("collector_type") or ""), str(row.get("title") or "")))
    return candidates[0]


def _literature_ref_details(source_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    detail = LiteratureRefDetail(
        source_id=str(source_row.get("source_id") or ""),
        title=source_row.get("title"),
        year=source_row.get("year"),
        source_tier=source_row.get("source_tier"),
        url=source_row.get("url"),
        doi=source_row.get("doi"),
        pmid=source_row.get("pmid"),
        openalex_id=source_row.get("openalex_id"),
    ).to_dict()
    return [detail] if has_verifiable_locator(detail) else []


def _structured_candidate_row(
    *,
    source_row: Mapping[str, Any],
    canonical_name: str,
    parameter_text: str,
    value: float,
    unit: str,
    time_label: str,
    confidence: float,
    extraction_method: str,
    measurement_type: str,
    denominator_type: str,
    normalization_basis: str,
    value_semantics: str,
    soft_ontology_tags: Iterable[str],
    soft_subparameter_hints: Iterable[str],
    linkage_targets: Iterable[str],
    evidence_span: str,
    geo: str = "Philippines",
    region: str = "national",
    province: str = "",
) -> dict[str, Any]:
    token = f"{source_row.get('source_id') or 'structured'}_{canonical_name}_{time_label}"
    observation_id = f"obs-structured-{abs(hash(token))}"
    candidate_id = f"cand-structured-{abs(hash(token + '-candidate'))}"
    geo_label = str(geo or "Philippines").strip() or "Philippines"
    region_label = str(region or "national").strip() or "national"
    province_label = str(province or "").strip()
    details = _literature_ref_details(source_row)
    return {
        "candidate_id": candidate_id,
        "document_id": f"doc-{source_row.get('source_id') or 'structured'}",
        "source_id": str(source_row.get("source_id") or ""),
        "canonical_name": canonical_name,
        "candidate_text": f"{canonical_name} {geo_label} {time_label}",
        "parameter_text": parameter_text,
        "evidence_span": evidence_span,
        "extraction_method": extraction_method,
        "confidence": round(float(confidence), 6),
        "source_bank": "phase0_structured_numeric",
        "source_tier": str(source_row.get("source_tier") or ""),
        "source_title": str(source_row.get("title") or source_row.get("source_name") or ""),
        "platform": str(source_row.get("platform") or ""),
        "query_geo_focus": str(source_row.get("query_geo_focus") or "philippines"),
        "observation_id": observation_id,
        "geo": geo_label,
        "region": region_label,
        "province": province_label,
        "population": geo_label if region_label == "national" else region_label,
        "time": time_label,
        "sex": "",
        "age_band": "",
        "kp_group": "",
        "geo_mentions": [geo_label],
        "literature_ref_details": details,
        "linkage_targets": [str(value) for value in linkage_targets if str(value or "").strip()],
        "soft_ontology_tags": [str(value) for value in soft_ontology_tags if str(value or "").strip()],
        "soft_subparameter_hints": [str(value) for value in soft_subparameter_hints if str(value or "").strip()],
        "measurement_type": measurement_type,
        "denominator_type": denominator_type,
        "normalization_basis": normalization_basis,
        "value_semantics": value_semantics,
        "value": round(float(value), 6),
        "unit": unit,
        "is_anchor_eligible": False,
        "is_direct_measurement": True,
        "is_prior_only": False,
    }


def _numeric_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"candidate_id", "candidate_text", "literature_ref_details"}
    }


def _fallback_philhealth_portal_source_row(title: str) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-") or "philhealth-open-portal"
    return {
        "source_id": f"philhealth-open-{slug}",
        "platform": "philhealth_open_portal",
        "title": title,
        "source_tier": "tier1_official_anchor",
        "query_geo_focus": "philippines",
        "year": 2025,
        "url": "https://philhealth.open.gov.ph/",
    }


def _select_philhealth_portal_source_row(
    source_rows: Mapping[str, dict[str, Any]],
    *,
    title: str,
) -> dict[str, Any]:
    title_fragment = title.strip().lower()
    for platform, fallback_fragment in (
        ("philhealth_open_portal", title_fragment),
        ("philhealth", "open portal"),
        ("philhealth", "transparency"),
    ):
        matching = [
            dict(row)
            for row in source_rows.values()
            if str(row.get("platform") or "").strip().lower() == platform
            and fallback_fragment in str(row.get("title") or "").strip().lower()
        ]
        if matching:
            matching.sort(key=lambda row: (str(row.get("collector_type") or ""), str(row.get("title") or "")))
            return matching[0]
    return _fallback_philhealth_portal_source_row(title)


def _fallback_psa_poverty_source_row(title: str) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-") or "psa-poverty"
    return {
        "source_id": f"psa-{slug}",
        "platform": "psa_poverty",
        "title": title,
        "source_tier": "tier1_official_anchor",
        "query_geo_focus": "philippines",
        "year": 2024,
        "url": "",
    }


def _select_psa_poverty_source_row(
    source_rows: Mapping[str, dict[str, Any]],
    *,
    title: str,
) -> dict[str, Any]:
    return (
        _candidate_source_row(source_rows, platform="psa_poverty", title_fragment=title)
        or _candidate_source_row(source_rows, platform="psa", title_fragment="poverty")
        or _candidate_source_row(source_rows, platform="psa", title_fragment=title)
        or _fallback_psa_poverty_source_row(title)
    )


def _load_or_fetch_json(*, url: str, cache_path: Path, timeout: tuple[float, float] = (10.0, 45.0)) -> tuple[list[dict[str, Any]], bool]:
    if cache_path.exists():
        return list(read_json(cache_path, default=[]) or []), True
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = list(payload[1] or []) if isinstance(payload, list) and len(payload) >= 2 else []
    write_json(cache_path, rows)
    return rows, False


def _load_or_fetch_binary(
    *,
    url: str,
    cache_path: Path,
    timeout: tuple[float, float] = (20.0, 180.0),
) -> tuple[Path, bool]:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path, True
    ensure_dir(cache_path.parent)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with cache_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return cache_path, False


def _load_or_fetch_json_payload(
    *,
    url: str,
    cache_path: Path,
    default: Any,
    timeout: tuple[float, float] = (10.0, 45.0),
) -> tuple[Any, bool]:
    if cache_path.exists():
        return read_json(cache_path, default=default), True
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    write_json(cache_path, payload)
    return payload, False


def _google_mobility_monthly_rows(*, cache_path: Path) -> tuple[list[dict[str, Any]], bool]:
    if cache_path.exists():
        return list(read_json(cache_path, default=[]) or []), True

    response = requests.get(_GOOGLE_MOBILITY_URL, stream=True, timeout=(10.0, 60.0))
    response.raise_for_status()
    line_iter = response.iter_lines(decode_unicode=True)
    try:
        first_line = next(line_iter)
    except StopIteration:
        write_json(cache_path, [])
        return [], False
    reader = csv.DictReader(itertools.chain([str(first_line).lstrip("\ufeff")], line_iter))
    monthly_rollup: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in reader:
        if str(row.get("country_region_code") or "").strip().upper() != "PH":
            continue
        if str(row.get("sub_region_2") or "").strip():
            continue
        if str(row.get("metro_area") or "").strip():
            continue
        if str(row.get("census_fips_code") or "").strip():
            continue
        sub_region = str(row.get("sub_region_1") or "").strip()
        geo = "Philippines"
        region = "national"
        if sub_region:
            geo = normalize_geo_label(sub_region)
            region = infer_region_code(geo, sub_region) or infer_region_code(sub_region)
            if not region or region == "national":
                continue
        date_value = str(row.get("date") or "").strip()
        if len(date_value) < 7:
            continue
        month_label = date_value[:7]
        transit = _safe_float(row.get("transit_stations_percent_change_from_baseline"))
        retail = _safe_float(row.get("retail_and_recreation_percent_change_from_baseline"))
        workplaces = _safe_float(row.get("workplaces_percent_change_from_baseline"))
        rollup_key = (geo, region, month_label)
        if transit is not None:
            monthly_rollup[rollup_key]["congestion_travel_time"].append(transit)
        mixing_components = [value for value in (transit, retail, workplaces) if value is not None]
        if mixing_components:
            monthly_rollup[rollup_key]["mobility_network_mixing"].append(sum(mixing_components) / float(len(mixing_components)))

    rows: list[dict[str, Any]] = []
    for geo, region, month_label in sorted(monthly_rollup):
        month_metrics = monthly_rollup[(geo, region, month_label)]
        for canonical_name, samples in sorted(month_metrics.items()):
            if not samples:
                continue
            rows.append(
                {
                    "canonical_name": canonical_name,
                    "geo": geo,
                    "region": region,
                    "province": "",
                    "time": month_label,
                    "value": round(float(sum(samples) / float(len(samples))), 6),
                }
            )
    write_json(cache_path, rows)
    return rows, False


def _safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        text = str(value).replace(",", "").strip()
        if not text or text == "*":
            return None
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        return float(text)
    except Exception:
        return None


def _clean_geo_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    text = re.sub(r"\s*[a-z]/\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("*", "").strip(" ,-")
    return normalize_geo_label(text)


def _read_csv_dict_rows(path: Path) -> list[dict[str, Any]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"unable to decode {path}")


def _xlsx_column_index(cell_ref: str) -> int:
    token = "".join(ch for ch in str(cell_ref or "") if ch.isalpha()).upper()
    column = 0
    for char in token:
        column = column * 26 + (ord(char) - ord("A") + 1)
    return column


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for node in root.findall("main:si", _XLSX_NS):
        chunks = [str(text_node.text or "") for text_node in node.findall(".//main:t", _XLSX_NS)]
        values.append("".join(chunks))
    return values


def _xlsx_first_sheet_rows(path: Path) -> list[list[Any]]:
    with zipfile.ZipFile(path) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_node = workbook_root.find("main:sheets/main:sheet", _XLSX_NS)
        if sheet_node is None:
            return []
        rel_id = sheet_node.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = ""
        for rel_node in rel_root.findall("rel:Relationship", _XLSX_NS):
            if rel_node.attrib.get("Id") == rel_id:
                target = str(rel_node.attrib.get("Target") or "")
                break
        if not target:
            return []
        shared_strings = _xlsx_shared_strings(archive)
        sheet_path = f"xl/{target.lstrip('/')}"
        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[Any]] = []
        for row_node in sheet_root.findall(".//main:sheetData/main:row", _XLSX_NS):
            values: dict[int, Any] = {}
            max_index = 0
            for cell in row_node.findall("main:c", _XLSX_NS):
                index = _xlsx_column_index(cell.attrib.get("r", ""))
                if index <= 0:
                    continue
                cell_type = str(cell.attrib.get("t") or "")
                value_node = cell.find("main:v", _XLSX_NS)
                inline_node = cell.find("main:is/main:t", _XLSX_NS)
                value: Any = None
                if cell_type == "s" and value_node is not None:
                    shared_index = int(float(value_node.text or "0"))
                    value = shared_strings[shared_index] if 0 <= shared_index < len(shared_strings) else None
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None:
                    raw_text = str(value_node.text or "").strip()
                    if raw_text:
                        numeric_value = _safe_float(raw_text)
                        value = numeric_value if numeric_value is not None else raw_text
                values[index] = value
                max_index = max(max_index, index)
            rows.append([values.get(idx) for idx in range(1, max_index + 1)])
        return rows


def _poverty_rate_candidate(
    *,
    source_row: Mapping[str, Any],
    geo: str,
    region: str,
    province: str,
    time_label: str,
    value: float,
    confidence: float,
    extraction_method: str,
    parameter_text: str,
    evidence_span: str,
) -> dict[str, Any]:
    return _structured_candidate_row(
        source_row=source_row,
        canonical_name="poverty_rate",
        parameter_text=parameter_text,
        value=float(value),
        unit="percent",
        time_label=time_label,
        confidence=confidence,
        extraction_method=extraction_method,
        measurement_type="rate",
        denominator_type="population",
        normalization_basis="percent",
        value_semantics="direct_observed",
        soft_ontology_tags=["economics", "official_series", "poverty", "small_area_estimate"],
        soft_subparameter_hints=["poverty", "economic_access_constraint", "cash_instability"],
        linkage_targets=["testing_uptake", "linkage_to_care", "retention_adherence"],
        evidence_span=evidence_span,
        geo=geo,
        region=region,
        province=province,
    )


def _economic_access_constraint_candidate(
    *,
    source_row: Mapping[str, Any],
    geo: str,
    region: str,
    province: str,
    time_label: str,
    value: float,
    confidence: float,
    extraction_method: str,
    parameter_text: str,
    evidence_span: str,
) -> dict[str, Any]:
    return _structured_candidate_row(
        source_row=source_row,
        canonical_name="economic_access_constraint",
        parameter_text=parameter_text,
        value=float(value),
        unit="php_per_person_per_year",
        time_label=time_label,
        confidence=confidence,
        extraction_method=extraction_method,
        measurement_type="cost",
        denominator_type="population",
        normalization_basis="per_capita",
        value_semantics="direct_observed",
        soft_ontology_tags=["economics", "official_series", "poverty_threshold", "small_area_estimate"],
        soft_subparameter_hints=["economic_access_constraint", "poverty", "cash_instability"],
        linkage_targets=["testing_uptake", "linkage_to_care", "retention_adherence"],
        evidence_span=evidence_span,
        geo=geo,
        region=region,
        province=province,
    )


def _household_expenditure_burden_candidate(
    *,
    source_row: Mapping[str, Any],
    geo: str,
    region: str,
    time_label: str,
    value: float,
    confidence: float,
    extraction_method: str,
    parameter_text: str,
    evidence_span: str,
) -> dict[str, Any]:
    return _structured_candidate_row(
        source_row=source_row,
        canonical_name="household_expenditure_burden",
        parameter_text=parameter_text,
        value=float(value),
        unit="percent",
        time_label=time_label,
        confidence=confidence,
        extraction_method=extraction_method,
        measurement_type="rate",
        denominator_type="population",
        normalization_basis="percent",
        value_semantics="bounded_proxy",
        soft_ontology_tags=["economics", "official_series", "fies", "household_balance"],
        soft_subparameter_hints=["household_expenditure_burden", "economic_access_constraint", "cash_instability"],
        linkage_targets=["testing_uptake", "linkage_to_care", "retention_adherence"],
        evidence_span=evidence_span,
        geo=geo,
        region=region,
        province="",
    )


def _region_code_from_text(*, region_text: str, province_text: str = "", geo_text: str = "") -> str:
    return (
        infer_region_code(region_text, region_text)
        or infer_region_code(province_text, region_text)
        or infer_region_code(geo_text, region_text)
        or infer_region_code(region_text)
        or "national"
    )


def _matching_row_value(row: Mapping[str, Any], *patterns: str) -> Any:
    lowered_patterns = [pattern.strip().lower() for pattern in patterns if pattern.strip()]
    for key, value in row.items():
        normalized_key = re.sub(r"\s+", " ", str(key or "").strip().lower())
        if all(pattern in normalized_key for pattern in lowered_patterns):
            return value
    return None


def _normalized_table_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _select_yafs_source_row(source_rows: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    return _candidate_source_row(
        source_rows,
        platform="yafs",
        year=2021,
        title_fragment="Regional Tables",
    ) or _candidate_source_row(
        source_rows,
        platform="yafs",
        year=2021,
        title_fragment="YAFS 5",
    )


def _fallback_fies_source_row(title: str) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-") or "fies"
    return {
        "source_id": f"fies-{slug}",
        "platform": "fies",
        "title": title,
        "source_tier": "tier2_official_survey",
        "query_geo_focus": "philippines",
        "year": 2015,
        "url": "https://psa.gov.ph/statistics/income-expenditure/fies",
    }


def _select_fies_source_row(source_rows: Mapping[str, dict[str, Any]], *, title: str) -> dict[str, Any]:
    return (
        _candidate_source_row(source_rows, platform="fies", title_fragment=title)
        or _candidate_source_row(source_rows, platform="fies", title_fragment="Statistical Yearbook")
        or _candidate_source_row(source_rows, platform="fies", title_fragment="Family Income")
        or _fallback_fies_source_row(title)
    )


def _iter_yafs_profile_pdfs(doc_root: Path) -> list[Path]:
    if not doc_root.exists():
        return []
    return sorted(
        path
        for path in doc_root.glob("yafs5-regional-profiles-*.pdf")
        if "philippines" not in path.stem.lower()
    )


def _yafs_background_rows(pdf_path: Path) -> list[list[Any]]:
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return []
        page = pdf.pages[0]
        for table in page.extract_tables() or []:
            if not table:
                continue
            header = [str(cell or "").strip() for cell in table[0]]
            if header and str(header[0]).strip().lower() == "background characteristics":
                return [list(row or []) for row in table[1:]]
    return []


def _yafs_region_label(pdf_path: Path, background_rows: list[list[Any]]) -> tuple[str, str] | None:
    stem_label = pdf_path.stem.replace("yafs5-regional-profiles-", "").replace("-", " ").strip()
    geo = normalize_geo_label(stem_label)
    region = infer_region_code(geo, stem_label) or infer_region_code(stem_label)
    if region and region != "national":
        return geo, region
    for row in background_rows:
        row_label = str((row or [None])[0] or "").strip()
        if row_label and row_label.lower() not in {"background characteristics", "sex", "age", "educationalattainment", "socioeconomicstatuswealthquintile"}:
            continue
    return None


def _yafs_row_value_map(background_rows: list[list[Any]]) -> dict[str, float | None]:
    row_map: dict[str, float | None] = {}
    for row in background_rows:
        if not row:
            continue
        label = _normalized_table_token(row[0])
        if not label:
            continue
        value = _safe_float(row[1] if len(row) > 1 else None)
        row_map[label] = value
    return row_map


def _collect_local_yafs_structured_rows(
    *,
    doc_root: Path,
    source_rows: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_row = _select_yafs_source_row(source_rows)
    if source_row is None:
        return [], [
            {
                "collector": "local_yafs_profiles",
                "status": "source_missing",
                "row_count": 0,
                "cache_used": True,
            }
        ]
    pdf_paths = _iter_yafs_profile_pdfs(doc_root)
    if not pdf_paths:
        return [], [
            {
                "collector": "local_yafs_profiles",
                "status": "pdf_missing",
                "row_count": 0,
                "cache_used": True,
                "doc_root": str(doc_root),
            }
        ]

    candidate_rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []
    kept = 0
    for pdf_path in pdf_paths:
        try:
            background_rows = _yafs_background_rows(pdf_path)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "local_yafs_profiles",
                    "status": "parse_error",
                    "row_count": 0,
                    "cache_used": True,
                    "file": pdf_path.name,
                    "error": repr(exc),
                }
            )
            continue
        geo_region = _yafs_region_label(pdf_path, background_rows)
        if not background_rows or geo_region is None:
            collector_rows.append(
                {
                    "collector": "local_yafs_profiles",
                    "status": "table_missing",
                    "row_count": 0,
                    "cache_used": True,
                    "file": pdf_path.name,
                }
            )
            continue
        geo, region = geo_region
        row_map = _yafs_row_value_map(background_rows)
        some_college = row_map.get("somecollegeorhigher")
        lowest = row_map.get("lowestpoorest")
        second = row_map.get("second")
        middle = row_map.get("middle")
        fourth = row_map.get("fourth")
        highest = row_map.get("highestwealthiest")
        economic_access_constraint = None
        if lowest is not None and second is not None:
            economic_access_constraint = lowest + second
        elif middle is not None and fourth is not None and highest is not None:
            economic_access_constraint = max(0.0, 100.0 - (middle + fourth + highest))

        file_rows = 0
        if some_college is not None:
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=source_row,
                    canonical_name="education",
                    parameter_text="share of youth with some college or higher from YAFS regional profile",
                    value=float(some_college),
                    unit="percent",
                    time_label="2021",
                    confidence=0.82,
                    extraction_method="structured_yafs_regional_profile_pdf",
                    measurement_type="rate",
                    denominator_type="population",
                    normalization_basis="percent",
                    value_semantics="direct_observed",
                    soft_ontology_tags=["education", "official_series", "survey_wave", "regional_profile"],
                    soft_subparameter_hints=["education", "health_literacy", "social_capital"],
                    linkage_targets=["testing_uptake", "prevention_access", "retention_adherence"],
                    evidence_span=f"YAFS some college or higher ({geo}): {some_college}",
                    geo=geo,
                    region=region,
                )
            )
            file_rows += 1
        if economic_access_constraint is not None:
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=source_row,
                    canonical_name="economic_access_constraint",
                    parameter_text="share of youth in lowest or second wealth quintile from YAFS regional profile",
                    value=float(round(economic_access_constraint, 6)),
                    unit="percent",
                    time_label="2021",
                    confidence=0.8,
                    extraction_method="structured_yafs_regional_profile_pdf",
                    measurement_type="rate",
                    denominator_type="population",
                    normalization_basis="percent",
                    value_semantics="direct_observed",
                    soft_ontology_tags=["economics", "survey_wave", "wealth_quintile", "regional_profile"],
                    soft_subparameter_hints=["economic_access_constraint", "poverty", "cash_instability"],
                    linkage_targets=["testing_uptake", "linkage_to_care", "retention_adherence"],
                    evidence_span=f"YAFS lowest+second wealth quintile ({geo}): {economic_access_constraint}",
                    geo=geo,
                    region=region,
                )
            )
            file_rows += 1
        kept += file_rows
        collector_rows.append(
            {
                "collector": "local_yafs_profiles",
                "status": "ok" if file_rows > 0 else "no_numeric_rows",
                "row_count": file_rows,
                "cache_used": True,
                "file": pdf_path.name,
                "region": region,
            }
        )
    if kept == 0 and not collector_rows:
        collector_rows.append(
            {
                "collector": "local_yafs_profiles",
                "status": "no_rows",
                "row_count": 0,
                "cache_used": True,
                "doc_root": str(doc_root),
            }
        )
    return candidate_rows, collector_rows


def _collect_local_fies_structured_rows(
    *,
    doc_root: Path,
    source_rows: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []
    for spec in _LOCAL_FIES_STRUCTURAL_SPECS:
        path = doc_root / str(spec["file_name"])
        source_row = _select_fies_source_row(source_rows, title=str(spec["title"]))
        if not path.exists():
            collector_rows.append(
                {
                    "collector": "local_fies_yearbook",
                    "status": "file_missing",
                    "row_count": 0,
                    "cache_used": True,
                    "file": path.name,
                }
            )
            continue
        try:
            table_text = _extract_statistical_yearbook_table_2_2_text(path)
            table_rows = _parse_statistical_yearbook_family_rows(table_text)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "local_fies_yearbook",
                    "status": "parse_error",
                    "row_count": 0,
                    "cache_used": True,
                    "file": path.name,
                    "error": repr(exc),
                }
            )
            continue
        kept = 0
        for row in table_rows:
            income = float(row["average_income_thousands"])
            expenditure = float(row["average_expenditure_thousands"])
            savings = float(row["average_savings_thousands"])
            if income <= 0.0:
                continue
            expenditure_burden = 100.0 * expenditure / income
            evidence_span = (
                f"PSY table 2.2 household balance ({row['geo']}, {row['year']}): "
                f"income={income}, expenditure={expenditure}, savings={savings}"
            )
            candidate_rows.append(
                _household_expenditure_burden_candidate(
                    source_row=source_row,
                    geo=str(row["geo"]),
                    region=str(row["region"]),
                    time_label=str(row["year"]),
                    value=float(round(expenditure_burden, 6)),
                    confidence=0.81,
                    extraction_method="structured_local_fies_statistical_yearbook_pdf",
                    parameter_text="average annual family expenditure as a share of average annual family income from PSA Statistical Yearbook table 2.2",
                    evidence_span=evidence_span,
                )
            )
            kept += 1
        collector_rows.append(
            {
                "collector": "local_fies_yearbook",
                "status": "ok" if kept > 0 else "no_numeric_rows",
                "row_count": kept,
                "cache_used": True,
                "file": path.name,
            }
        )
    return candidate_rows, collector_rows


def _fallback_philhealth_source_row(report_year: int) -> dict[str, Any]:
    return {
        "source_id": f"philhealth-annual-report-{report_year}",
        "platform": "philhealth",
        "title": f"PhilHealth Annual Report {report_year}",
        "source_tier": "tier1_official_anchor",
        "query_geo_focus": "philippines",
        "year": report_year,
        "url": f"https://www.philhealth.gov.ph/about_us/annual_report/ar{report_year}.pdf",
    }


def _select_philhealth_source_row(
    source_rows: Mapping[str, dict[str, Any]],
    *,
    report_year: int,
    title: str,
) -> dict[str, Any]:
    return (
        _candidate_source_row(source_rows, platform="philhealth", year=report_year, title_fragment=title)
        or _candidate_source_row(source_rows, platform="philhealth", title_fragment="Annual Report")
        or _fallback_philhealth_source_row(report_year)
    )


def _existing_local_philhealth_report_path(file_name: str, cache_path: Path) -> Path | None:
    for candidate in (
        cache_path,
        _LOCAL_STRUCTURED_DOC_ROOT / file_name,
        _LOCAL_TMP_DOC_ROOT / file_name,
    ):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _extract_pdf_text(pdf_path: Path) -> str:
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _extract_statistical_yearbook_table_2_2_text(pdf_path: Path) -> str:
    target = "TABLE 2.2 Number of Families, Average Annual Income and Expenditure by Region"
    best_text = ""
    best_score = -1
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if target.lower() not in text.lower():
                continue
            score = sum(1 for label in _FIES_REGION_LABELS if label in text)
            if score > best_score:
                best_score = score
                best_text = text
    return best_text


def _parse_fies_family_count(tokens: list[str], start_idx: int) -> tuple[float | None, int]:
    if start_idx >= len(tokens):
        return None, start_idx
    first = str(tokens[start_idx] or "").strip()
    if not first:
        return None, start_idx + 1
    if "," in first or len(first) >= 3:
        return _safe_float(first), start_idx + 1
    if start_idx + 1 < len(tokens):
        second = str(tokens[start_idx + 1] or "").strip()
        if second.isdigit() and len(first) == 1 and len(second) == 2:
            return _safe_float(first + second), start_idx + 2
    return _safe_float(first), start_idx + 1


def _parse_statistical_yearbook_family_rows(table_text: str) -> list[dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in str(table_text or "").splitlines() if str(line or "").strip()]
    if not lines:
        return []
    rows: list[dict[str, Any]] = []
    year_pairs: tuple[int, int] | None = None
    for line in lines:
        normalized = line.lower()
        if normalized.startswith("table 2.2") or normalized.startswith("total number of") or normalized.startswith("(in thousands)") or normalized.startswith("note:") or normalized.startswith("source:"):
            continue
        year_match = re.match(r"^(?P<left_year>20\d{2})\s+[\d,]+\s+\d+\s+\d+\s+\d+\s+(?P<right_year>20\d{2})\s+[\d,]+\s+\d+\s+\d+\s+\d+$", line)
        if year_match:
            year_pairs = (int(year_match.group("left_year")), int(year_match.group("right_year")))
            continue
        if year_pairs is None:
            continue
        region_label = next((label for label in _FIES_REGION_LABELS if line.startswith(label)), "")
        if not region_label:
            continue
        tail = line[len(region_label) :].strip()
        tokens = tail.split()
        values: list[float] = []
        idx = 0
        family_200x, idx = _parse_fies_family_count(tokens, idx)
        if family_200x is None:
            continue
        values.append(float(family_200x))
        for _ in range(3):
            value = _safe_float(tokens[idx] if idx < len(tokens) else None)
            if value is None:
                values = []
                break
            values.append(float(value))
            idx += 1
        if not values:
            continue
        family_201x, idx = _parse_fies_family_count(tokens, idx)
        if family_201x is None:
            continue
        values.append(float(family_201x))
        for _ in range(3):
            value = _safe_float(tokens[idx] if idx < len(tokens) else None)
            if value is None:
                values = []
                break
            values.append(float(value))
            idx += 1
        if len(values) != 8:
            continue
        region_geo = region_label
        if region_geo in {"NCR", "CAR", "ARMM"}:
            region_geo = {
                "NCR": "National Capital Region",
                "CAR": "Cordillera Administrative Region",
                "ARMM": "Autonomous Region in Muslim Mindanao",
            }[region_geo]
        if region_label == "ARMM":
            region = "barmm"
        else:
            region = _region_code_from_text(region_text=region_geo, province_text=region_geo, geo_text=region_geo)
        for year, offset in ((year_pairs[0], 0), (year_pairs[1], 4)):
            rows.append(
                {
                    "year": int(year),
                    "geo": normalize_geo_label(region_geo),
                    "region": region,
                    "family_count_thousands": float(values[offset + 0]),
                    "average_income_thousands": float(values[offset + 1]),
                    "average_expenditure_thousands": float(values[offset + 2]),
                    "average_savings_thousands": float(values[offset + 3]),
                    "region_label": region_label,
                }
            )
    return rows


def _dedupe_philhealth_leave_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        deduped[(int(row["year"]), str(row["region"]))] = row
    return [deduped[key] for key in sorted(deduped)]


def _philhealth_rows_from_block(*, year: int, body: str) -> list[dict[str, Any]]:
    row_pattern = re.compile(
        rf"(?<![A-Z0-9-])(?P<label>{_PHILHEALTH_REGION_PATTERN})\s+"
        r"(?P<regular>[0-9,]+)\s+(?P<casual>[0-9,]+)\s+(?P<total>[0-9,]+)"
    )
    rows: list[dict[str, Any]] = []
    for row_match in row_pattern.finditer(body):
        label = str(row_match.group("label") or "").strip()
        if label in {"", "Head Office", "Total"}:
            continue
        if label not in _PHILHEALTH_REGION_METADATA:
            continue
        geo, region = _PHILHEALTH_REGION_METADATA[label]
        regular = _safe_float(row_match.group("regular"))
        casual = _safe_float(row_match.group("casual"))
        total = _safe_float(row_match.group("total"))
        if regular is None or casual is None or total is None or total <= 0.0:
            continue
        rows.append(
            {
                "year": year,
                "region_label": label,
                "geo": geo,
                "region": region,
                "regular_value": float(regular),
                "casual_value": float(casual),
                "total_value": float(total),
            }
        )
    return rows


def _parse_philhealth_leave_benefit_tables(report_text: str, *, report_year: int | None = None) -> list[dict[str, Any]]:
    block_pattern = re.compile(
        r"(?P<year>2024|2023)\s+Region Regular Casual Total(?P<body>.*?Total\s+[0-9,]+\s+[0-9,]+\s+[0-9,]+)",
        re.IGNORECASE | re.DOTALL,
    )
    rows: list[dict[str, Any]] = []
    for match in block_pattern.finditer(report_text):
        year = int(match.group("year"))
        body = match.group("body")
        rows.extend(_philhealth_rows_from_block(year=year, body=body))
    if rows:
        return _dedupe_philhealth_leave_rows(rows)

    if report_year is None:
        return []

    fallback_pattern = re.compile(
        r"Head Office\s+[0-9,]+\s+[0-9,]+\s+[0-9,]+.*?Total\s+[0-9,]+\s+[0-9,]+\s+[0-9,]+",
        re.DOTALL,
    )
    unique_blocks: list[str] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()
    for match in fallback_pattern.finditer(report_text):
        body = match.group(0)
        provisional_rows = _philhealth_rows_from_block(year=report_year, body=body)
        if not provisional_rows:
            continue
        signature = tuple((str(row["region"]), float(row["total_value"])) for row in provisional_rows[:5])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_blocks.append(body)
        if len(unique_blocks) >= 2:
            break
    for index, body in enumerate(unique_blocks):
        rows.extend(_philhealth_rows_from_block(year=report_year - index, body=body))
    return _dedupe_philhealth_leave_rows(rows)


def _collect_philhealth_leave_benefit_rows(
    *,
    cache_dir: Path,
    source_rows: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []
    emitted_region_years: set[tuple[int, str]] = set()
    for spec in _PHILHEALTH_ANNUAL_REPORT_SPECS:
        report_year = int(spec["report_year"])
        file_name = str(spec["file_name"])
        cache_path = cache_dir / file_name
        local_path = _existing_local_philhealth_report_path(file_name, cache_path)
        cache_used = False
        if local_path is None:
            try:
                local_path, cache_used = _load_or_fetch_binary(url=str(spec["url"]), cache_path=cache_path)
            except Exception as exc:
                collector_rows.append(
                    {
                        "collector": "philhealth_leave_benefit_proxy",
                        "status": "fetch_error",
                        "report_year": report_year,
                        "row_count": 0,
                        "cache_used": False,
                        "error": repr(exc),
                    }
                )
                continue
        else:
            cache_used = True
        try:
            report_text = _extract_pdf_text(local_path)
            parsed_rows = _parse_philhealth_leave_benefit_tables(report_text, report_year=report_year)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "philhealth_leave_benefit_proxy",
                    "status": "parse_error",
                    "report_year": report_year,
                    "row_count": 0,
                    "cache_used": cache_used,
                    "file": local_path.name,
                    "error": repr(exc),
                }
            )
            continue
        if not parsed_rows:
            collector_rows.append(
                {
                    "collector": "philhealth_leave_benefit_proxy",
                    "status": "no_rows",
                    "report_year": report_year,
                    "row_count": 0,
                    "cache_used": cache_used,
                    "file": local_path.name,
                }
            )
            continue

        rows_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in parsed_rows:
            rows_by_year[int(row["year"])].append(row)

        emitted = 0
        for year, year_rows in sorted(rows_by_year.items()):
            source_row = _select_philhealth_source_row(source_rows, report_year=report_year, title=str(spec["title"]))
            total_regional_value = sum(float(row["total_value"]) for row in year_rows)
            if total_regional_value <= 0.0:
                continue
            for row in year_rows:
                region_year_key = (int(year), str(row["region"]))
                if region_year_key in emitted_region_years:
                    continue
                casual_share = 100.0 * float(row["casual_value"]) / float(row["total_value"])
                reach_share = 100.0 * float(row["total_value"]) / float(total_regional_value)
                geo = str(row["geo"])
                region = str(row["region"])
                region_label = str(row["region_label"])
                candidate_rows.append(
                    _structured_candidate_row(
                        source_row=source_row,
                        canonical_name="health_system_reach",
                        parameter_text="PhilHealth regional leave-benefit payable share across regional offices",
                        value=reach_share,
                        unit="percent",
                        time_label=str(year),
                        confidence=0.72,
                        extraction_method="structured_philhealth_annual_report_leave_benefit_table",
                        measurement_type="rate",
                        denominator_type="none",
                        normalization_basis="percent",
                        value_semantics="bounded_proxy",
                        soft_ontology_tags=["philhealth", "official_report", "administrative_proxy", "regional_office"],
                        soft_subparameter_hints=["health_system_reach", "service_delivery_reach", "philhealth_admin_capacity"],
                        linkage_targets=["linkage_to_care", "retention_adherence", "viral_suppression_rate"],
                        evidence_span=(
                            f"PhilHealth leave-benefit regional share ({region_label}, {year}): "
                            f"{float(row['total_value']):.0f} / {total_regional_value:.0f} = {reach_share:.3f}%"
                        ),
                        geo=geo,
                        region=region,
                    )
                )
                candidate_rows.append(
                    _structured_candidate_row(
                        source_row=source_row,
                        canonical_name="policy_implementation_weakness",
                        parameter_text="PhilHealth casual-share proxy from regional leave-benefit payable table",
                        value=casual_share,
                        unit="percent",
                        time_label=str(year),
                        confidence=0.68,
                        extraction_method="structured_philhealth_annual_report_leave_benefit_table",
                        measurement_type="rate",
                        denominator_type="none",
                        normalization_basis="percent",
                        value_semantics="bounded_proxy",
                        soft_ontology_tags=["philhealth", "official_report", "administrative_proxy", "workforce"],
                        soft_subparameter_hints=["policy_implementation_weakness", "health_system_reach", "workforce_fragility"],
                        linkage_targets=["linkage_to_care", "retention_adherence", "viral_suppression_rate"],
                        evidence_span=(
                            f"PhilHealth casual leave-benefit share ({region_label}, {year}): "
                            f"{float(row['casual_value']):.0f} / {float(row['total_value']):.0f} = {casual_share:.3f}%"
                        ),
                        geo=geo,
                        region=region,
                    )
                )
                emitted_region_years.add(region_year_key)
                emitted += 2
        collector_rows.append(
            {
                "collector": "philhealth_leave_benefit_proxy",
                "status": "ok" if emitted > 0 else "no_rows",
                "report_year": report_year,
                "row_count": emitted,
                "cache_used": cache_used,
                "file": local_path.name,
                "years_covered": sorted(rows_by_year),
            }
        )
    return candidate_rows, collector_rows


def _portal_time_label(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    upper = token.upper()
    if upper.endswith("-H1"):
        return f"{upper[:4]}-06"
    return token


def _collect_philhealth_portal_rows(
    *,
    cache_dir: Path,
    source_rows: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads: dict[str, Any] = {}
    collector_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = defaultdict(int)
    for spec in _PHILHEALTH_PORTAL_JSON_SPECS:
        cache_path = cache_dir / str(spec["file_name"])
        try:
            payload, cache_used = _load_or_fetch_json_payload(
                url=str(spec["url"]),
                cache_path=cache_path,
                default={},
            )
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "philhealth_open_portal",
                    "dataset": str(spec["name"]),
                    "status": "fetch_error",
                    "row_count": 0,
                    "cache_used": False,
                    "error": repr(exc),
                }
            )
            continue
        payloads[str(spec["name"])] = payload
        collector_rows.append(
            {
                "collector": "philhealth_open_portal",
                "dataset": str(spec["name"]),
                "status": "ok",
                "row_count": 0,
                "cache_used": bool(cache_used),
            }
        )

    financials = dict(payloads.get("financials") or {})
    stats_2025 = dict(payloads.get("statistics_2025") or {})
    candidate_rows: list[dict[str, Any]] = []

    financial_source = _select_philhealth_portal_source_row(source_rows, title="PhilHealth Open Portal Financial Data")
    stats_source = _select_philhealth_portal_source_row(source_rows, title="PhilHealth Open Portal Statistics and Charts 2025")

    for report in list(financials.get("annualReports") or []):
        time_label = _portal_time_label(report.get("year"))
        if not time_label:
            continue
        coverage_rate = _safe_float(report.get("coverageRate"))
        if coverage_rate is not None:
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=financial_source,
                    canonical_name="philhealth_coverage",
                    parameter_text="PhilHealth national population coverage rate from transparency portal annual report series",
                    value=float(coverage_rate),
                    unit="percent",
                    time_label=time_label,
                    confidence=0.9,
                    extraction_method="structured_philhealth_open_portal_financials_json",
                    measurement_type="rate",
                    denominator_type="population",
                    normalization_basis="percent",
                    value_semantics="direct_observed",
                    soft_ontology_tags=["philhealth", "official_series", "coverage", "transparency_portal"],
                    soft_subparameter_hints=["philhealth_coverage", "service_delivery_reach"],
                    linkage_targets=["linkage_to_care", "retention_adherence"],
                    evidence_span=f"PhilHealth coverage rate ({time_label}): {coverage_rate}",
                )
            )
            dataset_counts["financials"] += 1
        beneficiaries = _safe_float(report.get("beneficiaries") or report.get("totalBeneficiaries"))
        claims_paid = _safe_float(report.get("claimsPaid"))
        if beneficiaries is not None and beneficiaries > 0.0 and claims_paid is not None:
            claims_paid_per_beneficiary = claims_paid / beneficiaries
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=financial_source,
                    canonical_name="health_expenditure",
                    parameter_text="PhilHealth claims paid per beneficiary from transparency portal annual report series",
                    value=float(round(claims_paid_per_beneficiary, 6)),
                    unit="php_per_beneficiary",
                    time_label=time_label,
                    confidence=0.84,
                    extraction_method="structured_philhealth_open_portal_financials_json",
                    measurement_type="cost",
                    denominator_type="population",
                    normalization_basis="per_capita",
                    value_semantics="direct_observed",
                    soft_ontology_tags=["philhealth", "official_series", "claims", "transparency_portal"],
                    soft_subparameter_hints=["health_expenditure", "service_delivery_reach"],
                    linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                    evidence_span=f"PhilHealth claims paid per beneficiary ({time_label}): {claims_paid} / {beneficiaries}",
                )
            )
            dataset_counts["financials"] += 1
        average_processing_days = _safe_float(report.get("averageProcessingDays"))
        if average_processing_days is not None:
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=financial_source,
                    canonical_name="policy_implementation_weakness",
                    parameter_text="PhilHealth average claims processing time from transparency portal annual report series",
                    value=float(average_processing_days),
                    unit="days",
                    time_label=time_label,
                    confidence=0.82,
                    extraction_method="structured_philhealth_open_portal_financials_json",
                    measurement_type="time_delay",
                    denominator_type="none",
                    normalization_basis="absolute",
                    value_semantics="bounded_proxy",
                    soft_ontology_tags=["philhealth", "official_series", "claims_processing", "transparency_portal"],
                    soft_subparameter_hints=["policy_implementation_weakness", "health_system_reach"],
                    linkage_targets=["linkage_to_care", "retention_adherence", "viral_suppression_rate"],
                    evidence_span=f"PhilHealth average claims processing days ({time_label}): {average_processing_days}",
                )
            )
            dataset_counts["financials"] += 1

    for year_key, metrics in sorted(dict(financials.get("keyMetrics") or {}).items()):
        time_label = _portal_time_label(year_key)
        processing_days = _safe_float(dict(metrics or {}).get("claimsProcessingTime"))
        if processing_days is not None:
            candidate_rows.append(
                _structured_candidate_row(
                    source_row=financial_source,
                    canonical_name="policy_implementation_weakness",
                    parameter_text="PhilHealth claims processing time from transparency portal key metrics",
                    value=float(processing_days),
                    unit="days",
                    time_label=time_label,
                    confidence=0.86,
                    extraction_method="structured_philhealth_open_portal_financials_json",
                    measurement_type="time_delay",
                    denominator_type="none",
                    normalization_basis="absolute",
                    value_semantics="bounded_proxy",
                    soft_ontology_tags=["philhealth", "official_series", "claims_processing", "transparency_portal"],
                    soft_subparameter_hints=["policy_implementation_weakness", "health_system_reach"],
                    linkage_targets=["linkage_to_care", "retention_adherence", "viral_suppression_rate"],
                    evidence_span=f"PhilHealth key-metric claims processing time ({time_label}): {processing_days}",
                )
            )
            dataset_counts["financials"] += 1

    stats_payload = dict(stats_2025.get("philhealth_transparency_data_2025") or {})
    stats_time_label = "2025-06"
    accreditation = dict(stats_payload.get("accreditation") or {})
    overview_payload = dict(payloads.get("coverage", {}).get("overview") or {})
    population_covered = _safe_float(overview_payload.get("populationCovered"))
    facilities_total = _safe_float(
        dict(accreditation.get("health_care_providers_institutions") or {}).get("grand_total")
    )
    if facilities_total is not None and facilities_total > 0.0 and population_covered is not None and population_covered > 0.0:
        facilities_per_100k = facilities_total * 100_000.0 / population_covered
        candidate_rows.append(
            _structured_candidate_row(
                source_row=stats_source,
                canonical_name="clinics_per_capita",
                parameter_text="PhilHealth accredited facilities per 100k covered population from transparency portal statistics",
                value=float(round(facilities_per_100k, 6)),
                unit="facilities_per_100k",
                time_label=stats_time_label,
                confidence=0.84,
                extraction_method="structured_philhealth_open_portal_statistics_json",
                measurement_type="capacity",
                denominator_type="population",
                normalization_basis="per_100k",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "facility_capacity", "transparency_portal"],
                soft_subparameter_hints=["clinics_per_capita", "health_system_reach", "service_delivery_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth accredited facilities per 100k ({stats_time_label}): {facilities_total} / {population_covered}",
            )
        )
        dataset_counts["statistics_2025"] += 1
    annual_reports = [dict(row or {}) for row in list(financials.get("annualReports") or [])]
    report_2025_h1 = next((row for row in annual_reports if str(row.get("year") or "").upper() == "2025-H1"), {})
    report_2023 = next((row for row in annual_reports if str(row.get("year") or "") == "2023"), {})
    program_coverage = dict(dict(report_2023.get("healthcareFacilities") or {}).get("programCoverage") or {})
    konsulta_coverage = _safe_float(dict(program_coverage.get("konsulta") or {}).get("coveragePercentage"))
    if konsulta_coverage is not None:
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name="service_delivery_reach_konsulta",
                parameter_text="PhilHealth Konsulta city coverage percentage from transparency portal annual report series",
                value=float(konsulta_coverage),
                unit="percent",
                time_label="2023",
                confidence=0.8,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="rate",
                denominator_type="facility",
                normalization_basis="percent",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "konsulta", "service_delivery"],
                soft_subparameter_hints=["service_delivery_reach", "health_system_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth Konsulta cities with providers coverage (2023): {konsulta_coverage}",
            )
        )
        dataset_counts["financials"] += 1

    for program_key, label in (("mcp", "MCP"), ("tbDots", "TB-DOTS")):
        coverage_value = _safe_float(dict(program_coverage.get(program_key) or {}).get("coveragePercentage"))
        if coverage_value is None:
            continue
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name=f"service_delivery_reach_{program_key.lower()}",
                parameter_text=f"PhilHealth {label} city coverage percentage from transparency portal annual report series",
                value=float(coverage_value),
                unit="percent",
                time_label="2023",
                confidence=0.78,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="rate",
                denominator_type="facility",
                normalization_basis="percent",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "service_delivery", "program_coverage"],
                soft_subparameter_hints=["service_delivery_reach", "health_system_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth {label} cities with providers coverage (2023): {coverage_value}",
            )
        )
        dataset_counts["financials"] += 1

    healthcare_facilities = dict(report_2023.get("healthcareFacilities") or {})
    covered_population_2023 = _safe_float(report_2023.get("populationCovered"))
    specific_facilities = dict(healthcare_facilities.get("specificFacilities") or {})
    for facility_key, label in (
        ("konsultaProviders", "PhilHealth Konsulta providers"),
        ("hivAidsCenters", "PhilHealth HIV/AIDS centers"),
    ):
        total_value = _safe_float(dict(specific_facilities.get(facility_key) or {}).get("total"))
        if total_value is None or total_value <= 0.0 or covered_population_2023 is None or covered_population_2023 <= 0.0:
            continue
        per_100k = total_value * 100_000.0 / covered_population_2023
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name=(
                    "clinics_per_capita_konsulta"
                    if facility_key == "konsultaProviders"
                    else "clinics_per_capita_hiv_aids_centers"
                ),
                parameter_text=f"{label} per 100k covered population from transparency portal annual report series",
                value=float(round(per_100k, 6)),
                unit="facilities_per_100k",
                time_label="2023",
                confidence=0.8,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="capacity",
                denominator_type="population",
                normalization_basis="per_100k",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "facility_capacity", "program_network"],
                soft_subparameter_hints=["clinics_per_capita", "service_delivery_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"{label} per 100k covered population (2023): {total_value} / {covered_population_2023}",
            )
        )
        dataset_counts["financials"] += 1

    total_hospitals = _safe_float(dict(healthcare_facilities.get("hospitals") or {}).get("total"))
    total_other_facilities = _safe_float(dict(healthcare_facilities.get("otherFacilities") or {}).get("total"))
    philhealth_infrastructure = dict(report_2023.get("philhealthInfrastructure") or {})
    facilities_with_cares = _safe_float(philhealth_infrastructure.get("facilitiesWithCARES"))
    total_facilities_2023 = (total_hospitals or 0.0) + (total_other_facilities or 0.0)
    if facilities_with_cares is not None and facilities_with_cares > 0.0 and total_facilities_2023 > 0.0:
        cares_reach = 100.0 * facilities_with_cares / total_facilities_2023
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name="service_delivery_reach_cares",
                parameter_text="PhilHealth CARES-supported facility share from transparency portal annual report series",
                value=float(round(cares_reach, 6)),
                unit="percent",
                time_label="2023",
                confidence=0.76,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="rate",
                denominator_type="facility",
                normalization_basis="percent",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "service_delivery", "cares_program"],
                soft_subparameter_hints=["service_delivery_reach", "health_system_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth CARES-supported facilities share (2023): {facilities_with_cares} / {total_facilities_2023}",
            )
        )
        dataset_counts["financials"] += 1

    yakap = dict(dict(report_2025_h1.get("breakdown") or {}).get("yakap") or {})
    covered_population_2025 = _safe_float(report_2025_h1.get("populationCovered"))
    accredited_clinics = _safe_float(yakap.get("accreditedClinics"))
    if accredited_clinics is not None and accredited_clinics > 0.0 and covered_population_2025 is not None and covered_population_2025 > 0.0:
        yakap_clinics_per_100k = accredited_clinics * 100_000.0 / covered_population_2025
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name="clinics_per_capita_yakap",
                parameter_text="PhilHealth YAKAP accredited clinics per 100k covered population from transparency portal annual report series",
                value=float(round(yakap_clinics_per_100k, 6)),
                unit="facilities_per_100k",
                time_label="2025-06",
                confidence=0.8,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="capacity",
                denominator_type="population",
                normalization_basis="per_100k",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "yakap", "facility_capacity"],
                soft_subparameter_hints=["clinics_per_capita", "service_delivery_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth YAKAP accredited clinics per 100k (2025-06): {accredited_clinics} / {covered_population_2025}",
            )
        )
        dataset_counts["financials"] += 1
    registrations = _safe_float(yakap.get("registrations"))
    first_encounters = _safe_float(yakap.get("firstEncounters"))
    if registrations is not None and registrations > 0.0 and first_encounters is not None:
        encounter_share = 100.0 * first_encounters / registrations
        candidate_rows.append(
            _structured_candidate_row(
                source_row=financial_source,
                canonical_name="service_delivery_reach_yakap",
                parameter_text="PhilHealth YAKAP first-encounter reach among registered beneficiaries from transparency portal annual report series",
                value=float(round(encounter_share, 6)),
                unit="percent",
                time_label="2025-06",
                confidence=0.79,
                extraction_method="structured_philhealth_open_portal_financials_json",
                measurement_type="rate",
                denominator_type="population",
                normalization_basis="percent",
                value_semantics="direct_observed",
                soft_ontology_tags=["philhealth", "official_series", "yakap", "service_delivery"],
                soft_subparameter_hints=["service_delivery_reach", "health_system_reach"],
                linkage_targets=["linkage_to_care", "viral_suppression_rate"],
                evidence_span=f"PhilHealth YAKAP first encounters share (2025-06): {first_encounters} / {registrations}",
            )
        )
        dataset_counts["financials"] += 1

    for row in collector_rows:
        if row.get("status") == "ok":
            row["row_count"] = int(dataset_counts.get(str(row.get("dataset")), 0))
    return candidate_rows, collector_rows


def _collect_local_psa_poverty_rows(
    *,
    doc_root: Path,
    source_rows: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []
    emitted_keys: set[tuple[str, str, str, str, str]] = set()
    if not doc_root.exists():
        return [], [
            {
                "collector": "local_psa_poverty",
                "status": "doc_root_missing",
                "row_count": 0,
                "cache_used": True,
                "doc_root": str(doc_root),
            }
        ]

    def _append_candidate(candidate: Mapping[str, Any]) -> bool:
        key = (
            str(candidate.get("canonical_name") or "").strip(),
            str(candidate.get("geo") or "").strip(),
            str(candidate.get("region") or "").strip(),
            str(candidate.get("province") or "").strip(),
            str(candidate.get("time") or "").strip(),
        )
        if key in emitted_keys:
            return False
        emitted_keys.add(key)
        candidate_rows.append(dict(candidate))
        return True

    for spec in _LOCAL_PSA_POVERTY_SPECS:
        path = doc_root / str(spec["file_name"])
        source_row = _select_psa_poverty_source_row(source_rows, title=str(spec["title"]))
        if not path.exists():
            collector_rows.append(
                {
                    "collector": "local_psa_poverty",
                    "status": "file_missing",
                    "row_count": 0,
                    "cache_used": True,
                    "file": path.name,
                }
            )
            continue
        try:
            if str(spec["kind"]) == "city_municipal_csv":
                raw_rows = _read_csv_dict_rows(path)
                kept = 0
                for row in raw_rows:
                    region_text = _clean_geo_token(row.get("Region"))
                    province_text = _clean_geo_token(row.get("Province  ") or row.get("Province"))
                    geo_text = _clean_geo_token(row.get("Municipality/City  ") or row.get("Municipality/City"))
                    if not region_text or not geo_text:
                        continue
                    region = _region_code_from_text(region_text=region_text, province_text=province_text, geo_text=geo_text)
                    for year in (2009, 2012, 2015):
                        value = _safe_float(_matching_row_value(row, "poverty incidence", str(year)))
                        if value is None:
                            continue
                        cv_value = _safe_float(_matching_row_value(row, "coefficient of variation", str(year)))
                        if _append_candidate(
                            _poverty_rate_candidate(
                                source_row=source_row,
                                geo=geo_text,
                                region=region,
                                province=province_text,
                                time_label=str(year),
                                value=float(value),
                                confidence=0.8,
                                extraction_method="structured_local_psa_city_municipal_poverty_csv",
                                parameter_text="city or municipal poverty incidence from PSA small-area estimate table",
                                evidence_span=(
                                    f"PSA city or municipal poverty incidence ({geo_text}, {year}): {value}"
                                    + (f"; CV={cv_value}" if cv_value is not None else "")
                                ),
                            )
                        ):
                            kept += 1
            elif str(spec["kind"]) == "province_csv":
                raw_rows = _read_csv_dict_rows(path)
                kept = 0
                for row in raw_rows:
                    region_text = _clean_geo_token(row.get("Region"))
                    geo_text = _clean_geo_token(row.get("Province"))
                    if not region_text or not geo_text:
                        continue
                    region = _region_code_from_text(region_text=region_text, province_text=geo_text, geo_text=geo_text)
                    for year in (2015, 2018):
                        threshold_value = _safe_float(
                            _matching_row_value(
                                row,
                                "annual per capita poverty threshold",
                                str(year),
                            )
                        )
                        if threshold_value is not None and _append_candidate(
                            _economic_access_constraint_candidate(
                                source_row=source_row,
                                geo=geo_text,
                                region=region,
                                province=geo_text,
                                time_label=str(year),
                                value=float(threshold_value),
                                confidence=0.86,
                                extraction_method="structured_local_psa_province_threshold_csv",
                                parameter_text="annual per capita poverty threshold from PSA province poverty table",
                                evidence_span=(
                                    f"PSA annual per capita poverty threshold ({geo_text}, {year}): {threshold_value}"
                                ),
                            )
                        ):
                            kept += 1
                        value = _safe_float(
                            _matching_row_value(
                                row,
                                "poverty incidence among families estimates",
                                str(year),
                            )
                        )
                        if value is None:
                            continue
                        cv_value = _safe_float(_matching_row_value(row, "coefficient of variation", str(year)))
                        if _append_candidate(
                            _poverty_rate_candidate(
                                source_row=source_row,
                                geo=geo_text,
                                region=region,
                                province=geo_text,
                                time_label=str(year),
                                value=float(value),
                                confidence=0.84,
                                extraction_method="structured_local_psa_province_poverty_csv",
                                parameter_text="province or district poverty incidence among families from PSA poverty table",
                                evidence_span=(
                                    f"PSA province poverty incidence ({geo_text}, {year}): {value}"
                                    + (f"; CV={cv_value}" if cv_value is not None else "")
                                ),
                            )
                        ):
                            kept += 1
            else:
                xlsx_rows = _xlsx_first_sheet_rows(path)
                kept = 0
                current_region = ""
                for row in xlsx_rows[4:]:
                    psgc = str((row[0] if len(row) > 0 else "") or "").strip()
                    region_or_province = _clean_geo_token(row[1] if len(row) > 1 else "")
                    geo_text = _clean_geo_token(row[2] if len(row) > 2 else "")
                    row_values = [
                        _safe_float(row[idx] if len(row) > idx else None)
                        for idx in (3, 4, 5)
                    ]
                    if region_or_province and not geo_text and all(value is None for value in row_values):
                        current_region = region_or_province
                        continue
                    if not psgc or not geo_text or not current_region:
                        continue
                    province_text = region_or_province
                    region = _region_code_from_text(
                        region_text=current_region,
                        province_text=province_text,
                        geo_text=geo_text,
                    )
                    for year, idx in ((2018, 3), (2021, 4), (2023, 5)):
                        value = _safe_float(row[idx] if len(row) > idx else None)
                        if value is None:
                            continue
                        if _append_candidate(
                            _poverty_rate_candidate(
                                source_row=source_row,
                                geo=geo_text,
                                region=region,
                                province=province_text,
                                time_label=str(year),
                                value=float(value),
                                confidence=0.82,
                                extraction_method="structured_local_psa_city_municipal_poverty_xlsx",
                                parameter_text="city or municipal poverty incidence from PSA 2018 2021 2023 small-area estimate workbook",
                                evidence_span=f"PSA city or municipal poverty incidence ({geo_text}, {year}): {value}",
                            )
                        ):
                            kept += 1
            collector_rows.append(
                {
                    "collector": "local_psa_poverty",
                    "status": "ok" if kept > 0 else "no_numeric_rows",
                    "row_count": kept,
                    "cache_used": True,
                    "file": path.name,
                }
            )
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "local_psa_poverty",
                    "status": "parse_error",
                    "row_count": 0,
                    "cache_used": True,
                    "file": path.name,
                    "error": repr(exc),
                }
            )
    return candidate_rows, collector_rows


def build_philhealth_portal_artifacts(
    *,
    candidate_rows: Iterable[Mapping[str, Any]],
    collector_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    portal_rows = [
        dict(row)
        for row in candidate_rows
        if str(row.get("platform") or "").strip().lower() == "philhealth_open_portal"
    ]
    portal_collectors = [
        dict(row)
        for row in collector_rows
        if str(row.get("collector") or "").strip().lower() == "philhealth_open_portal"
    ]

    by_source: dict[tuple[str, str], dict[str, Any]] = {}
    by_year: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for row in portal_rows:
        source_id = str(row.get("source_id") or "").strip()
        source_title = str(row.get("source_title") or "").strip()
        time_label = str(row.get("time") or "").strip()
        year = time_label[:4] if re.match(r"^\d{4}", time_label) else ""
        canonical_name = str(row.get("canonical_name") or "").strip()
        extraction_method = str(row.get("extraction_method") or "").strip()
        measurement_type = str(row.get("measurement_type") or "").strip()
        unit = str(row.get("unit") or "").strip()
        value = _safe_float(row.get("value"))

        source_bucket = by_source.setdefault(
            (source_id, source_title),
            {
                "source_id": source_id,
                "source_title": source_title,
                "row_count": 0,
                "years": set(),
                "time_labels": set(),
                "canonical_names": set(),
                "extraction_methods": set(),
            },
        )
        source_bucket["row_count"] += 1
        if year:
            source_bucket["years"].add(year)
        if time_label:
            source_bucket["time_labels"].add(time_label)
        if canonical_name:
            source_bucket["canonical_names"].add(canonical_name)
        if extraction_method:
            source_bucket["extraction_methods"].add(extraction_method)

        if year:
            year_bucket = by_year.setdefault(
                year,
                {
                    "year": year,
                    "row_count": 0,
                    "time_labels": set(),
                    "canonical_names": set(),
                    "source_titles": set(),
                },
            )
            year_bucket["row_count"] += 1
            if time_label:
                year_bucket["time_labels"].add(time_label)
            if canonical_name:
                year_bucket["canonical_names"].add(canonical_name)
            if source_title:
                year_bucket["source_titles"].add(source_title)

        canonical_bucket = by_canonical.setdefault(
            canonical_name,
            {
                "canonical_name": canonical_name,
                "row_count": 0,
                "years": set(),
                "source_titles": set(),
                "measurement_types": set(),
                "units": set(),
                "value_min": None,
                "value_max": None,
            },
        )
        canonical_bucket["row_count"] += 1
        if year:
            canonical_bucket["years"].add(year)
        if source_title:
            canonical_bucket["source_titles"].add(source_title)
        if measurement_type:
            canonical_bucket["measurement_types"].add(measurement_type)
        if unit:
            canonical_bucket["units"].add(unit)
        if value is not None:
            if canonical_bucket["value_min"] is None or value < canonical_bucket["value_min"]:
                canonical_bucket["value_min"] = value
            if canonical_bucket["value_max"] is None or value > canonical_bucket["value_max"]:
                canonical_bucket["value_max"] = value

    def _finalize_bucket(bucket: Mapping[str, Any], set_fields: Iterable[str]) -> dict[str, Any]:
        row = dict(bucket)
        for field in set_fields:
            row[field] = sorted(str(value) for value in row.get(field, set()) if str(value or "").strip())
        return row

    return {
        "candidate_rows": portal_rows,
        "summary": {
            "platform": "philhealth_open_portal",
            "candidate_count": len(portal_rows),
            "collector_rows": portal_collectors,
            "sources": [
                _finalize_bucket(bucket, ("years", "time_labels", "canonical_names", "extraction_methods"))
                for _, bucket in sorted(by_source.items(), key=lambda item: (item[0][1], item[0][0]))
            ],
            "years": [
                _finalize_bucket(bucket, ("time_labels", "canonical_names", "source_titles"))
                for _, bucket in sorted(by_year.items())
            ],
            "canonical_metrics": [
                _finalize_bucket(bucket, ("years", "source_titles", "measurement_types", "units"))
                for _, bucket in sorted(by_canonical.items())
            ],
        },
    }


def build_structured_numeric_candidates(
    *,
    raw_dir: Path,
    source_rows: Mapping[str, dict[str, Any]],
    plugin_id: str,
) -> dict[str, Any]:
    del plugin_id
    cache_dir = ensure_dir(Path(raw_dir) / "structured_numeric_cache")
    candidate_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []

    for spec in _WDI_SERIES_SPECS:
        cache_path = cache_dir / f"world_bank_{str(spec['indicator_code']).lower()}.json"
        source_row = _candidate_source_row(
            source_rows,
            platform="world_bank_wdi",
            title_fragment=str(spec.get("title_fragment") or ""),
        )
        if source_row is None:
            collector_rows.append(
                {
                    "collector": "world_bank_wdi",
                    "canonical_name": str(spec["canonical_name"]),
                    "status": "source_missing",
                    "row_count": 0,
                    "cache_used": False,
                }
            )
            continue
        url = f"https://api.worldbank.org/v2/country/PHL/indicator/{spec['indicator_code']}?format=json&per_page=20000"
        try:
            rows, cache_used = _load_or_fetch_json(url=url, cache_path=cache_path)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "world_bank_wdi",
                    "canonical_name": str(spec["canonical_name"]),
                    "status": "fetch_error",
                    "row_count": 0,
                    "cache_used": False,
                    "error": repr(exc),
                }
            )
            continue
        kept = 0
        for row in rows:
            year_text = str(row.get("date") or "").strip()
            value = _safe_float(row.get("value"))
            year = _safe_float(year_text)
            if value is None or year is None or int(year) < 2010:
                continue
            candidate = _structured_candidate_row(
                source_row=source_row,
                canonical_name=str(spec["canonical_name"]),
                parameter_text=str(spec["parameter_text"]),
                value=float(value),
                unit=str(spec["unit"]),
                time_label=str(int(year)),
                confidence=float(spec["confidence"]),
                extraction_method="structured_world_bank_api",
                measurement_type=str(spec["measurement_type"]),
                denominator_type=str(spec["denominator_type"]),
                normalization_basis=str(spec["normalization_basis"]),
                value_semantics=str(spec["value_semantics"]),
                soft_ontology_tags=spec["soft_ontology_tags"],
                soft_subparameter_hints=spec["soft_subparameter_hints"],
                linkage_targets=spec["linkage_targets"],
                evidence_span=f"{spec['parameter_text']}: {value}",
            )
            candidate_rows.append(candidate)
            numeric_rows.append(_numeric_projection(candidate))
            kept += 1
        collector_rows.append(
            {
                "collector": "world_bank_wdi",
                "canonical_name": str(spec["canonical_name"]),
                "status": "ok",
                "row_count": kept,
                "cache_used": bool(cache_used),
            }
        )

    google_source = _candidate_source_row(source_rows, platform="google_mobility")
    if google_source is None:
        collector_rows.append(
            {
                "collector": "google_mobility",
                "status": "source_missing",
                "row_count": 0,
                "cache_used": False,
            }
        )
    else:
        google_cache = cache_dir / "google_mobility_monthly_ph_v2.json"
        try:
            google_rows, cache_used = _google_mobility_monthly_rows(cache_path=google_cache)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "google_mobility",
                    "status": "fetch_error",
                    "row_count": 0,
                    "cache_used": False,
                    "error": repr(exc),
                }
            )
            google_rows = []
            cache_used = False
        kept = 0
        for row in google_rows:
            time_label = str(row.get("time") or "")
            if not time_label:
                continue
            year = _safe_float(time_label[:4])
            source_row = _candidate_source_row(source_rows, platform="google_mobility", year=int(year) if year is not None else None) or google_source
            canonical_name = str(row.get("canonical_name") or "")
            geo = str(row.get("geo") or "Philippines").strip() or "Philippines"
            region = str(row.get("region") or "national").strip() or "national"
            parameter_scope = "national rollup" if region == "national" else "regional rollup"
            parameter_text = f"google community mobility monthly {parameter_scope}"
            if canonical_name == "congestion_travel_time":
                parameter_text = f"transit-station mobility change from baseline ({parameter_scope})"
            candidate = _structured_candidate_row(
                source_row=source_row,
                canonical_name=canonical_name,
                parameter_text=parameter_text,
                value=float(row.get("value") or 0.0),
                unit="percent",
                time_label=time_label,
                confidence=0.82,
                extraction_method="structured_google_mobility_csv",
                measurement_type="rate",
                denominator_type="none",
                normalization_basis="percent",
                value_semantics="direct_observed",
                soft_ontology_tags=["mobility", "transport", "official_series"],
                soft_subparameter_hints=[canonical_name, "labor_migration", "transport_friction"],
                linkage_targets=["testing_uptake", "linkage_to_care", "retention_adherence"],
                evidence_span=f"{parameter_text} ({geo}): {row.get('value')}",
                geo=geo,
                region=region,
                province=str(row.get("province") or ""),
            )
            candidate_rows.append(candidate)
            numeric_rows.append(_numeric_projection(candidate))
            kept += 1
        collector_rows.append(
            {
                "collector": "google_mobility",
                "status": "ok",
                "row_count": kept,
                "cache_used": bool(cache_used),
            }
        )

    local_yafs_rows, local_yafs_collectors = _collect_local_yafs_structured_rows(
        doc_root=_LOCAL_STRUCTURED_DOC_ROOT,
        source_rows=source_rows,
    )
    if local_yafs_rows:
        candidate_rows.extend(local_yafs_rows)
        numeric_rows.extend(_numeric_projection(candidate) for candidate in local_yafs_rows)
    collector_rows.extend(local_yafs_collectors)

    local_fies_rows, local_fies_collectors = _collect_local_fies_structured_rows(
        doc_root=_LOCAL_STRUCTURED_DOC_ROOT,
        source_rows=source_rows,
    )
    if local_fies_rows:
        candidate_rows.extend(local_fies_rows)
        numeric_rows.extend(_numeric_projection(candidate) for candidate in local_fies_rows)
    collector_rows.extend(local_fies_collectors)

    philhealth_rows, philhealth_collectors = _collect_philhealth_leave_benefit_rows(
        cache_dir=cache_dir,
        source_rows=source_rows,
    )
    if philhealth_rows:
        candidate_rows.extend(philhealth_rows)
        numeric_rows.extend(_numeric_projection(candidate) for candidate in philhealth_rows)
    collector_rows.extend(philhealth_collectors)

    philhealth_portal_rows, philhealth_portal_collectors = _collect_philhealth_portal_rows(
        cache_dir=cache_dir,
        source_rows=source_rows,
    )
    if philhealth_portal_rows:
        candidate_rows.extend(philhealth_portal_rows)
        numeric_rows.extend(_numeric_projection(candidate) for candidate in philhealth_portal_rows)
    collector_rows.extend(philhealth_portal_collectors)

    local_psa_rows, local_psa_collectors = _collect_local_psa_poverty_rows(
        doc_root=_LOCAL_STRUCTURED_DOC_ROOT,
        source_rows=source_rows,
    )
    if local_psa_rows:
        candidate_rows.extend(local_psa_rows)
        numeric_rows.extend(_numeric_projection(candidate) for candidate in local_psa_rows)
    collector_rows.extend(local_psa_collectors)

    return {
        "candidate_rows": candidate_rows,
        "numeric_rows": numeric_rows,
        "summary": {
            "source_bank": "phase0_structured_numeric",
            "candidate_count": len(candidate_rows),
            "numeric_observation_count": len(numeric_rows),
            "cache_dir": str(cache_dir),
            "collectors": collector_rows,
        },
    }
