from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ensure_dir, sha256_file


ROOT_DIR = Path(__file__).resolve().parents[3]
_COUNTRY_FOCUS = "Philippines"
_SOURCE_KIND = "unaids_multinational_csv"

_MULTINATIONAL_HIV_SPECS: tuple[dict[str, str], ...] = (
    {
        "source_id": "unaids_estimated_plhiv_all_ages",
        "label": "UNAIDS People Living With HIV - All Ages",
        "filename": "People living with HIV_People living with HIV - All ages_Population_ All.csv",
        "metric_name": "estimated_plhiv",
        "unit": "count_people",
        "measurement_class": "model_estimate",
    },
    {
        "source_id": "unaids_annual_new_infections_all_ages",
        "label": "UNAIDS New HIV Infections - All Ages",
        "filename": "New HIV infections_New HIV infections - All ages_Population_ All ages.csv",
        "metric_name": "annual_new_infections",
        "unit": "count_people",
        "measurement_class": "model_estimate",
    },
    {
        "source_id": "unaids_annual_aids_deaths_all_ages",
        "label": "UNAIDS AIDS-Related Deaths - All Ages",
        "filename": "AIDS-related deaths_AIDS-related deaths - All ages_Population_ All ages.csv",
        "metric_name": "annual_aids_deaths",
        "unit": "count_people",
        "measurement_class": "model_estimate",
    },
    {
        "source_id": "unaids_alive_on_art_all_ages",
        "label": "UNAIDS People Living With HIV Receiving ART (#) - All Ages",
        "filename": "Treatment cascade_People living with HIV receiving ART (#)_Population_ All ages.csv",
        "metric_name": "alive_on_art",
        "unit": "count_people",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_known_status_share_all_ages",
        "label": "UNAIDS People Living With HIV Who Know Their Status (%) - All Ages",
        "filename": "Treatment cascade_People living with HIV who know their status (%)_Population_ All ages.csv",
        "metric_name": "unaids_known_status_share_percent",
        "unit": "percent",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_suppressed_share_all_ages",
        "label": "UNAIDS People Living With HIV Who Have Suppressed Viral Loads (%) - All Ages",
        "filename": "Treatment cascade_People living with HIV who have suppressed viral loads (%)_Population_ All ages.csv",
        "metric_name": "unaids_suppressed_share_percent",
        "unit": "percent",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_art_coverage_share_all_ages",
        "label": "UNAIDS People Living With HIV Receiving ART (%) - All Ages",
        "filename": "Treatment cascade_People living with HIV receiving ART (%)_Population_ All ages.csv",
        "metric_name": "unaids_art_coverage_percent",
        "unit": "percent",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_hiv_tests_volume_total",
        "label": "UNAIDS HIV Tests (Volume) - Total",
        "filename": "Combination prevention_HIV tests (volume)_Population_ Total.csv",
        "metric_name": "annual_hiv_tests_volume",
        "unit": "count_people",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_hiv_positivity_total",
        "label": "UNAIDS HIV Positivity Rate (%) - Total",
        "filename": "Combination prevention_HIV positivity rate (%)_Population_ Total.csv",
        "metric_name": "hiv_test_positivity_percent",
        "unit": "percent",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_prep_people_receiving",
        "label": "UNAIDS People Receiving PrEP",
        "filename": "Combination prevention_People receiving pre-exposure prophylaxis (PrEP).csv",
        "metric_name": "prep_people_receiving",
        "unit": "count_people",
        "measurement_class": "external_reference_unaids",
    },
    {
        "source_id": "unaids_late_hiv_diagnosis_all_ages",
        "label": "UNAIDS Late HIV Diagnosis - All Ages With CD4 <200",
        "filename": "Treatment cascade_Late HIV diagnosis_Population_ All ages with CD4 _200.csv",
        "metric_name": "late_hiv_diagnosis_percent",
        "unit": "percent",
        "measurement_class": "external_reference_unaids",
    },
)

_SPECS_BY_SOURCE_ID = {str(item["source_id"]): dict(item) for item in _MULTINATIONAL_HIV_SPECS}
_EXPLICIT_FILENAMES = {str(item["filename"]): dict(item) for item in _MULTINATIONAL_HIV_SPECS}

_AUTO_DISCOVERY_RULES: tuple[dict[str, Any], ...] = (
    {
        "coverage_tier": 1,
        "coverage_group": "treatment_cascade",
        "prefixes": ("Treatment cascade_",),
        "exclude_filenames": tuple(_EXPLICIT_FILENAMES.keys()),
    },
    {
        "coverage_tier": 2,
        "coverage_group": "epidemic_transition_metrics",
        "prefixes": ("Epidemic transition metrics_",),
    },
    {
        "coverage_tier": 3,
        "coverage_group": "age_stratified_core",
        "prefixes": (
            "New HIV infections_",
            "AIDS-related deaths_",
            "People living with HIV_",
        ),
        "exclude_filenames": tuple(_EXPLICIT_FILENAMES.keys()),
    },
    {
        "coverage_tier": 4,
        "coverage_group": "key_population_panels",
        "prefixes": (
            "Men who have sex with men_",
            "People who inject drugs_",
            "Prisoners_",
            "Sex workers_",
            "Transgender people_",
        ),
    },
    {
        "coverage_tier": 5,
        "coverage_group": "stigma_prevention_pmtct",
        "prefixes": (
            "Stigma and Discrimination_",
            "Combination prevention_",
            "Elimination of vertical transmission_",
            "Young people_",
        ),
        "exclude_filenames": tuple(_EXPLICIT_FILENAMES.keys()),
    },
)


def default_multinational_hiv_data_dir() -> Path:
    return Path(__file__).resolve().with_name("HIV_Data")


def _safe_ascii_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "document"


def _normalize_header_token(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _slugify_metric_token(value: str) -> str:
    token = (
        str(value or "")
        .replace("(%)", " percent ")
        .replace("%", " percent ")
        .replace("(#)", " number ")
        .replace("#", " number ")
        .replace("/", " ")
    )
    token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    token = re.sub(r"_+", "_", token)
    return token or "metric"


def _infer_unit_from_text(*values: str) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    if "per 1000" in text:
        return "count_per_1000_population"
    if "ratio" in text:
        return "ratio"
    if "percent" in text or "(%)" in text or "prevalence" in text or "coverage" in text or "knowledge" in text:
        return "percent"
    if "rate" in text:
        return "rate"
    if "expenditure" in text or "funding" in text:
        return "currency_value"
    if "size estimate" in text or "orphans" in text or "people" in text or "deaths" in text or "infections" in text or "kits" in text:
        return "count_people"
    return "annual_value"


def _find_auto_rule(filename: str) -> dict[str, Any] | None:
    for rule in _AUTO_DISCOVERY_RULES:
        prefixes = tuple(str(value) for value in tuple(rule.get("prefixes") or ()))
        if not filename.startswith(prefixes):
            continue
        excluded = {str(value) for value in tuple(rule.get("exclude_filenames") or ())}
        if filename in excluded:
            return None
        return dict(rule)
    return None


def _auto_source_row_for_file(path: Path, *, raw_dir: Path, rule: dict[str, Any]) -> dict[str, Any]:
    stem_slug = _slugify_metric_token(path.stem)
    source_id = f"unaids_auto_{stem_slug}"
    short_stem = _safe_ascii_label(path.stem)[:64]
    name_hash = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:10]
    copied_name = f"{source_id[:80]}_{short_stem}_{name_hash}{path.suffix.lower()}"
    copied_path = raw_dir / copied_name
    shutil.copy2(path, copied_path)
    return {
        "source_id": source_id,
        "label": f"UNAIDS {path.stem}",
        "source_kind": _SOURCE_KIND,
        "local_path": str(copied_path),
        "origin_path": str(path),
        "checksum": sha256_file(copied_path),
        "metric_name": stem_slug,
        "unit": _infer_unit_from_text(path.stem),
        "measurement_class": "external_reference_unaids_auto",
        "source_dataset": "UNAIDS_HIV_Data",
        "source_organization": "UNAIDS",
        "coverage_tier": int(rule.get("coverage_tier") or 99),
        "coverage_group": str(rule.get("coverage_group") or "auto"),
        "parse_mode": "auto",
    }


def _coerce_numeric_token(value: Any) -> tuple[float | None, str]:
    text = str(value or "").strip()
    if not text or text in {"...", "…"}:
        return None, text
    normalized = text.replace("\xa0", " ")
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = normalized.replace(",", "")
    if normalized.startswith("<") or normalized.startswith(">"):
        normalized = normalized[:1] + normalized[1:].strip()
    match = re.search(r"([<>]?)\s*(-?\d+(?:\.\d+)?)", normalized)
    if not match:
        return None, text
    operator = str(match.group(1) or "")
    try:
        return float(match.group(2)), operator
    except ValueError:
        return None, text


def _annual_column_map(header: list[str]) -> dict[int, dict[str, int]]:
    mapping: dict[int, dict[str, int]] = {}
    for index, cell in enumerate(header):
        token = _normalize_header_token(cell)
        match = re.match(r"^(?P<year>(?:19|20)\d{2})(?:_(?P<suffix>lower|upper|Footnote))?$", token, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group("year"))
        suffix = str(match.group("suffix") or "value").lower()
        mapping.setdefault(year, {})[suffix] = index
    return mapping


def _find_country_row(rows: list[list[str]], *, country_name: str) -> list[str] | None:
    target = str(country_name).strip().lower()
    for row in rows:
        if not row:
            continue
        if str(row[0] or "").strip().lower() == target:
            return row
    return None


def materialize_multinational_hiv_sources(run_dir: Path, *, data_dir: Path | None = None) -> list[dict[str, Any]]:
    source_dir = Path(data_dir) if data_dir is not None else default_multinational_hiv_data_dir()
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    raw_dir = ensure_dir(run_dir / "harp_archive" / "raw")
    rows: list[dict[str, Any]] = []
    for spec in _MULTINATIONAL_HIV_SPECS:
        origin = source_dir / str(spec["filename"])
        if not origin.exists() or not origin.is_file():
            continue
        copied_name = f"{spec['source_id']}_{_safe_ascii_label(origin.stem)}{origin.suffix.lower()}"
        copied_path = raw_dir / copied_name
        shutil.copy2(origin, copied_path)
        rows.append(
            {
                "source_id": str(spec["source_id"]),
                "label": str(spec["label"]),
                "source_kind": _SOURCE_KIND,
                "local_path": str(copied_path),
                "origin_path": str(origin),
                "checksum": sha256_file(copied_path),
                "metric_name": str(spec["metric_name"]),
                "unit": str(spec["unit"]),
                "measurement_class": str(spec["measurement_class"]),
                "source_dataset": "UNAIDS_HIV_Data",
                "source_organization": "UNAIDS",
                "coverage_tier": 0,
                "coverage_group": "curated_core",
                "parse_mode": "standard_panel",
            }
        )
    for origin in sorted(source_dir.glob("*.csv")):
        if origin.name in _EXPLICIT_FILENAMES:
            continue
        rule = _find_auto_rule(origin.name)
        if rule is None:
            continue
        rows.append(_auto_source_row_for_file(origin, raw_dir=raw_dir, rule=rule))
    return rows


def extract_multinational_hiv_rows(
    local_path: Path,
    *,
    source_id: str,
    source_label: str,
    metric_name: str,
    unit: str,
    measurement_class: str,
    country_name: str = _COUNTRY_FOCUS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with local_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], {
            "source_id": source_id,
            "source_label": source_label,
            "metric_name": metric_name,
            "imported_row_count": 0,
            "country_found": False,
            "available_years": [],
            "imported_years": [],
        }
    header = [str(cell or "") for cell in rows[0]]
    column_map = _annual_column_map(header)
    country_row = _find_country_row(rows[1:], country_name=country_name)
    metric_rows: list[dict[str, Any]] = []
    if country_row is not None:
        for year, columns in sorted(column_map.items()):
            value_index = columns.get("value")
            if value_index is None or value_index >= len(country_row):
                continue
            value, operator = _coerce_numeric_token(country_row[value_index])
            if value is None:
                continue
            lower_value = None
            upper_value = None
            if "lower" in columns and columns["lower"] < len(country_row):
                lower_value, _ = _coerce_numeric_token(country_row[columns["lower"]])
            if "upper" in columns and columns["upper"] < len(country_row):
                upper_value, _ = _coerce_numeric_token(country_row[columns["upper"]])
            metric_rows.append(
                {
                    "year": int(year),
                    "time": f"{int(year):04d}-01",
                    "metric_name": metric_name,
                    "value": float(value),
                    "reported_value": str(country_row[value_index]),
                    "bound_operator": operator,
                    "value_lower": lower_value,
                    "value_upper": upper_value,
                    "unit": unit,
                    "measurement_class": measurement_class,
                    "series_kind": "annual_multinational_series",
                    "temporal_precision": "annual_series",
                    "geo": country_name,
                    "region": "national",
                    "province": country_name,
                    "source_id": source_id,
                    "source_label": source_label,
                    "source_url": "",
                    "source_note": f"Imported from {local_path.name}",
                    "source_dataset": "UNAIDS_HIV_Data",
                    "source_organization": "UNAIDS",
                    "source_quality_tier": "external_multinational_hiv_panel",
                    "evidence_confidence": 0.86 if measurement_class == "model_estimate" else 0.82,
                    "value_semantics": "point_estimate",
                    "extraction_method": "unaids_multinational_csv_import",
                }
            )
    inventory_row = {
        "source_id": source_id,
        "source_label": source_label,
        "metric_name": metric_name,
        "measurement_class": measurement_class,
        "unit": unit,
        "country_name": country_name,
        "country_found": bool(country_row is not None),
        "available_years": sorted(int(year) for year in column_map),
        "imported_years": [int(row["year"]) for row in metric_rows],
        "imported_row_count": len(metric_rows),
        "origin_path": str(local_path),
    }
    return metric_rows, inventory_row


def _build_metric_row(
    *,
    year: int,
    metric_name: str,
    value: float,
    reported_value: str,
    operator: str,
    unit: str,
    measurement_class: str,
    source_id: str,
    source_label: str,
    source_note: str,
    local_path: Path,
    lower_value: float | None = None,
    upper_value: float | None = None,
    source_quality_tier: str = "external_multinational_hiv_panel_auto",
) -> dict[str, Any]:
    return {
        "year": int(year),
        "time": f"{int(year):04d}-01",
        "metric_name": metric_name,
        "value": float(value),
        "reported_value": reported_value,
        "bound_operator": operator,
        "value_lower": lower_value,
        "value_upper": upper_value,
        "unit": unit,
        "measurement_class": measurement_class,
        "series_kind": "annual_multinational_series",
        "temporal_precision": "annual_series",
        "geo": _COUNTRY_FOCUS,
        "region": "national",
        "province": _COUNTRY_FOCUS,
        "source_id": source_id,
        "source_label": source_label,
        "source_url": "",
        "source_note": source_note,
        "source_dataset": "UNAIDS_HIV_Data",
        "source_organization": "UNAIDS",
        "source_quality_tier": source_quality_tier,
        "evidence_confidence": 0.78,
        "value_semantics": "point_estimate",
        "extraction_method": "unaids_multinational_csv_import_auto",
    }


def _latest_year_from_header(header: list[str]) -> int | None:
    for token in header[1:]:
        match = re.search(r"((?:19|20)\d{2})", str(token or ""))
        if match:
            return int(match.group(1))
    return None


def _extract_latest_value_rows(
    rows: list[list[str]],
    *,
    local_path: Path,
    source_row: dict[str, Any],
    country_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    header = [str(cell or "") for cell in rows[0]]
    year = _latest_year_from_header(header)
    country_row = _find_country_row(rows[1:], country_name=country_name)
    if year is None or country_row is None or len(country_row) < 2:
        return [], [
            {
                "source_id": str(source_row.get("source_id") or ""),
                "source_label": str(source_row.get("label") or ""),
                "metric_name": str(source_row.get("metric_name") or ""),
                "measurement_class": str(source_row.get("measurement_class") or ""),
                "unit": str(source_row.get("unit") or ""),
                "country_name": country_name,
                "country_found": bool(country_row is not None),
                "available_years": [int(year)] if year is not None else [],
                "imported_years": [],
                "imported_row_count": 0,
                "origin_path": str(local_path),
            }
        ]
    value, operator = _coerce_numeric_token(country_row[1])
    if value is None:
        imported_years: list[int] = []
        imported_row_count = 0
        metric_rows: list[dict[str, Any]] = []
    else:
        metric_rows = [
            _build_metric_row(
                year=int(year),
                metric_name=str(source_row.get("metric_name") or ""),
                value=float(value),
                reported_value=str(country_row[1]),
                operator=operator,
                unit=str(source_row.get("unit") or "annual_value"),
                measurement_class=str(source_row.get("measurement_class") or "external_reference_unaids_auto"),
                source_id=str(source_row.get("source_id") or ""),
                source_label=str(source_row.get("label") or ""),
                source_note=f"Imported from {local_path.name}",
                local_path=local_path,
            )
        ]
        imported_years = [int(year)]
        imported_row_count = 1
    return metric_rows, [
        {
            "source_id": str(source_row.get("source_id") or ""),
            "source_label": str(source_row.get("label") or ""),
            "metric_name": str(source_row.get("metric_name") or ""),
            "measurement_class": str(source_row.get("measurement_class") or ""),
            "unit": str(source_row.get("unit") or ""),
            "country_name": country_name,
            "country_found": bool(country_row is not None),
            "available_years": [int(year)] if year is not None else [],
            "imported_years": imported_years,
            "imported_row_count": imported_row_count,
            "origin_path": str(local_path),
        }
    ]


def _strip_series_suffix(label: str) -> tuple[str, str]:
    text = str(label or "").strip()
    lowered = text.lower()
    for suffix, normalized in (("_footnote", "footnote"), ("_lower", "lower"), ("_upper", "upper")):
        if lowered.endswith(suffix):
            return text[: -len(suffix)].strip(), normalized
    return text, "value"


def _extract_multi_series_rows(
    rows: list[list[str]],
    *,
    local_path: Path,
    source_row: dict[str, Any],
    country_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 3:
        return [], []
    year_header = [str(cell or "") for cell in rows[0]]
    series_header = [str(cell or "") for cell in rows[1]]
    country_row = _find_country_row(rows[2:], country_name=country_name)
    if country_row is None:
        return [], []
    series_columns: dict[tuple[str, int], dict[str, int]] = {}
    for index in range(1, min(len(year_header), len(series_header))):
        year_match = re.search(r"((?:19|20)\d{2})", str(year_header[index] or ""))
        if year_match is None:
            continue
        base_label, suffix = _strip_series_suffix(series_header[index])
        if not base_label:
            continue
        series_columns.setdefault((base_label, int(year_match.group(1))), {})[suffix] = index
    metric_rows: list[dict[str, Any]] = []
    inventory_counts: Counter[tuple[str, str]] = Counter()
    metric_years: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    base_metric = str(source_row.get("metric_name") or _slugify_metric_token(local_path.stem))
    for (series_label, year), column_map in sorted(series_columns.items(), key=lambda item: (item[0][0].lower(), item[0][1])):
        value_index = column_map.get("value")
        if value_index is None or value_index >= len(country_row):
            continue
        value, operator = _coerce_numeric_token(country_row[value_index])
        if value is None:
            continue
        lower_value = None
        upper_value = None
        if "lower" in column_map and column_map["lower"] < len(country_row):
            lower_value, _ = _coerce_numeric_token(country_row[column_map["lower"]])
        if "upper" in column_map and column_map["upper"] < len(country_row):
            upper_value, _ = _coerce_numeric_token(country_row[column_map["upper"]])
        series_slug = _slugify_metric_token(series_label)
        metric_name = f"{base_metric}__{series_slug}"
        series_source_id = f"{str(source_row.get('source_id') or base_metric)}__{series_slug}"
        series_source_label = f"{str(source_row.get('label') or local_path.stem)} :: {series_label}"
        metric_rows.append(
            _build_metric_row(
                year=int(year),
                metric_name=metric_name,
                value=float(value),
                reported_value=str(country_row[value_index]),
                operator=operator,
                unit=_infer_unit_from_text(series_label, str(source_row.get("label") or ""), local_path.stem),
                measurement_class=str(source_row.get("measurement_class") or "external_reference_unaids_auto"),
                source_id=series_source_id,
                source_label=series_source_label,
                source_note=f"Imported from {local_path.name}",
                local_path=local_path,
                lower_value=lower_value,
                upper_value=upper_value,
            )
        )
        inventory_counts[(metric_name, series_source_id)] += 1
        metric_years[(metric_name, series_source_id)].append(int(year))
    inventory_rows: list[dict[str, Any]] = []
    for (metric_name, series_source_id), count in sorted(inventory_counts.items()):
        years = sorted(metric_years[(metric_name, series_source_id)])
        inventory_rows.append(
            {
                "source_id": series_source_id,
                "source_label": next(
                    (
                        str(row.get("source_label") or "")
                        for row in metric_rows
                        if str(row.get("source_id") or "") == series_source_id
                    ),
                    str(source_row.get("label") or ""),
                ),
                "metric_name": metric_name,
                "measurement_class": str(source_row.get("measurement_class") or ""),
                "unit": next(
                    (
                        str(row.get("unit") or "")
                        for row in metric_rows
                        if str(row.get("source_id") or "") == series_source_id
                    ),
                    str(source_row.get("unit") or ""),
                ),
                "country_name": country_name,
                "country_found": True,
                "available_years": years,
                "imported_years": years,
                "imported_row_count": int(count),
                "origin_path": str(local_path),
            }
        )
    return metric_rows, inventory_rows


def _detect_parse_mode(rows: list[list[str]]) -> str:
    if len(rows) >= 2:
        first_header = str(rows[0][0] or "").strip().lower() if rows[0] else ""
        second_header = str(rows[1][0] or "").strip().lower() if rows[1] else ""
        if first_header in {"", "country"} and second_header == "country":
            return "multi_series_panel"
    if rows and rows[0]:
        normalized_header = [_normalize_header_token(cell) for cell in rows[0]]
        if any(token.lower().startswith("mostrecentdataasof") for token in normalized_header[1:]):
            return "latest_value_panel"
    return "standard_panel"


def derive_multinational_hiv_rows(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_metric_year = {
        (str(row.get("metric_name") or ""), int(row.get("year") or 0)): dict(row)
        for row in metric_rows
    }
    derived_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    derived_specs = (
        {
            "metric_name": "diagnosed_plhiv",
            "source_id": "unaids_derived_diagnosed_plhiv",
            "source_label": "UNAIDS Derived Diagnosed PLHIV",
            "numerator_metric": "estimated_plhiv",
            "share_metric": "unaids_known_status_share_percent",
        },
        {
            "metric_name": "virally_suppressed",
            "source_id": "unaids_derived_virally_suppressed",
            "source_label": "UNAIDS Derived Virally Suppressed",
            "numerator_metric": "estimated_plhiv",
            "share_metric": "unaids_suppressed_share_percent",
        },
    )
    for spec in derived_specs:
        imported_years: list[int] = []
        for year in sorted({int(row.get("year") or 0) for row in metric_rows}):
            numerator_row = by_metric_year.get((str(spec["numerator_metric"]), year))
            share_row = by_metric_year.get((str(spec["share_metric"]), year))
            if numerator_row is None or share_row is None:
                continue
            derived_value = float(numerator_row["value"]) * float(share_row["value"]) / 100.0
            derived_rows.append(
                {
                    "year": int(year),
                    "time": f"{int(year):04d}-01",
                    "metric_name": str(spec["metric_name"]),
                    "value": round(float(derived_value), 6),
                    "unit": "count_people",
                    "measurement_class": "external_reference_unaids_derived",
                    "series_kind": "annual_multinational_derived_series",
                    "temporal_precision": "annual_series",
                    "geo": _COUNTRY_FOCUS,
                    "region": "national",
                    "province": _COUNTRY_FOCUS,
                    "source_id": str(spec["source_id"]),
                    "source_label": str(spec["source_label"]),
                    "source_url": "",
                    "source_note": (
                        f"Derived as {spec['numerator_metric']} × {spec['share_metric']} / 100 "
                        f"from UNAIDS annual multinational panels"
                    ),
                    "source_dataset": "UNAIDS_HIV_Data",
                    "source_organization": "UNAIDS",
                    "source_quality_tier": "external_multinational_hiv_panel_derived",
                    "evidence_confidence": round(
                        min(
                            float(numerator_row.get("evidence_confidence") or 0.0),
                            float(share_row.get("evidence_confidence") or 0.0),
                        )
                        * 0.95,
                        6,
                    ),
                    "value_semantics": "derived_point_estimate",
                    "derivation_inputs": [
                        str(spec["numerator_metric"]),
                        str(spec["share_metric"]),
                    ],
                    "extraction_method": "unaids_multinational_csv_derived",
                }
            )
            imported_years.append(int(year))
        inventory_rows.append(
            {
                "source_id": str(spec["source_id"]),
                "source_label": str(spec["source_label"]),
                "metric_name": str(spec["metric_name"]),
                "measurement_class": "external_reference_unaids_derived",
                "unit": "count_people",
                "country_name": _COUNTRY_FOCUS,
                "country_found": True,
                "available_years": imported_years,
                "imported_years": imported_years,
                "imported_row_count": len(imported_years),
                "origin_path": "",
            }
        )
    return derived_rows, inventory_rows


def extract_multinational_hiv_rows_from_sources(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for source in source_rows:
        if str(source.get("source_kind") or "") != _SOURCE_KIND:
            continue
        local_path = Path(str(source.get("local_path") or ""))
        spec = _SPECS_BY_SOURCE_ID.get(str(source.get("source_id") or ""))
        if not local_path.exists() or not local_path.is_file():
            continue
        if spec is not None:
            rows, inventory = extract_multinational_hiv_rows(
                local_path,
                source_id=str(source["source_id"]),
                source_label=str(source.get("label") or spec["label"]),
                metric_name=str(spec["metric_name"]),
                unit=str(spec["unit"]),
                measurement_class=str(spec["measurement_class"]),
            )
            metric_rows.extend(rows)
            inventory_rows.append(inventory)
            continue
        with local_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.reader(handle))
        parse_mode = str(source.get("parse_mode") or "").strip().lower()
        if not parse_mode or parse_mode == "auto":
            parse_mode = _detect_parse_mode(raw_rows)
        if parse_mode == "multi_series_panel":
            rows, inventories = _extract_multi_series_rows(
                raw_rows,
                local_path=local_path,
                source_row=source,
                country_name=_COUNTRY_FOCUS,
            )
        elif parse_mode == "latest_value_panel":
            rows, inventories = _extract_latest_value_rows(
                raw_rows,
                local_path=local_path,
                source_row=source,
                country_name=_COUNTRY_FOCUS,
            )
        else:
            rows, inventory = extract_multinational_hiv_rows(
                local_path,
                source_id=str(source["source_id"]),
                source_label=str(source.get("label") or source.get("source_id") or local_path.stem),
                metric_name=str(source.get("metric_name") or _slugify_metric_token(local_path.stem)),
                unit=str(source.get("unit") or _infer_unit_from_text(str(source.get("label") or ""), local_path.stem)),
                measurement_class=str(source.get("measurement_class") or "external_reference_unaids_auto"),
            )
            inventories = [inventory]
        metric_rows.extend(rows)
        inventory_rows.extend(inventories)
    derived_rows, derived_inventory = derive_multinational_hiv_rows(metric_rows)
    metric_rows.extend(derived_rows)
    inventory_rows.extend(derived_inventory)
    metric_rows.sort(key=lambda item: (str(item.get("metric_name") or ""), int(item.get("year") or 0), str(item.get("source_id") or "")))
    inventory_rows.sort(key=lambda item: (str(item.get("metric_name") or ""), str(item.get("source_id") or "")))
    return metric_rows, inventory_rows
