from __future__ import annotations

import csv
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from epigraph_ph.harp_archive.doh_hiv_sti_archive import extract_archive_metric_rows, extract_archive_pdf_pages
from epigraph_ph.runtime import RunContext, ensure_dir, read_json, sha256_file, utc_now_iso, write_json

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


BASE_URL = "https://www.aidsdatahub.org"
COUNTRY_LISTING_URLS = (
    "https://www.aidsdatahub.org/country-snapshot/philippines",
    "https://www.aidsdatahub.org/country-profiles/philippines",
    "https://new.aidsdatahub.org/country-snapshot/philippines",
)
CURATED_DETAIL_URLS = (
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2017",
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2018",
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2019",
    "https://www.aidsdatahub.org/resource/philippines-country-data-2020",
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2021",
    "https://new.aidsdatahub.org/resource/philippines-country-data-2022",
    "https://www.aidsdatahub.org/index.php/resource/philippines-country-data-2023",
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-january-2021",
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-july-2019",
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-october-december-2019",
    "https://www.aidsdatahub.org/resource/philippines-hiv-aids-registry-oct-2011",
    "https://www.aidsdatahub.org/resource/philippines-hiv-aids-surveillance-technical-report-2003",
)
NUMERIC_TOKEN_RE = re.compile(r"(?:<\s*\d[\d, ]*|\d[\d, ]*)")
TECHNICAL_NUMERIC_TOKEN_RE = re.compile(r"<\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?")
MONTH_NAME_TO_NUMBER = {
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
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
COUNTRY_ANCHOR_SERIES: dict[str, tuple[str, str]] = {
    "New HIV infections (all ages)": ("annual_new_infections", "count_people"),
    "AIDS-related deaths (all ages)": ("annual_aids_deaths", "count_people"),
    "People living with HIV (all ages)": ("estimated_plhiv", "count_people"),
}
CURATED_RESOURCE_OVERRIDES: dict[str, dict[str, Any]] = {
    "https://new.aidsdatahub.org/resource/country-snapshot-2019-philippines": {
        "title": "Country Snapshot 2019 Philippines",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-card-2019.pdf",
        "release_year": 2019,
        "resource_kind": "annual_country_summary",
    },
    "https://new.aidsdatahub.org/resource/philippines-country-snapshot-2017": {
        "title": "Philippines Country Snapshot 2017",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-snapshot-2017.pdf",
        "release_year": 2017,
        "resource_kind": "annual_country_summary",
    },
    "https://new.aidsdatahub.org/resource/philippines-country-snapshot-2018": {
        "title": "Philippines Country Snapshot 2018",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-snapshot-2018.pdf",
        "release_year": 2018,
        "resource_kind": "annual_country_summary",
    },
    "https://new.aidsdatahub.org/resource/philippines-country-snapshot-2019": {
        "title": "Philippines Country Snapshot 2019",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-card-nov-2019.pdf",
        "release_year": 2019,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2017": {
        "title": "Philippines Country Snapshot 2017",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-snapshot-2017.pdf",
        "release_year": 2017,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2018": {
        "title": "Philippines Country Snapshot 2018",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-snapshot-2018.pdf",
        "release_year": 2018,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-snapshot-2019": {
        "title": "Philippines Country Snapshot 2019",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-country-card-nov-2019.pdf",
        "release_year": 2019,
        "resource_kind": "annual_country_summary",
    },
    "https://new.aidsdatahub.org/resource/philippines-country-data-2020": {
        "title": "Philippines Country Data 2020",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/2020-aids-data-book-ph.pdf",
        "release_year": 2020,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-data-2020": {
        "title": "Philippines Country Data 2020",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/2020-aids-data-book-ph.pdf",
        "release_year": 2020,
        "resource_kind": "annual_country_summary",
    },
    "https://new.aidsdatahub.org/resource/philippines-country-data-2022": {
        "title": "Philippines Country Data 2022",
        "pdf_url": "https://new.aidsdatahub.org/sites/default/files/resource/philippines-data-book-2022-en.pdf",
        "release_year": 2022,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-data-2022": {
        "title": "Philippines Country Data 2022",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/philippines-data-book-2022-en.pdf",
        "release_year": 2022,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/index.php/resource/philippines-country-data-2023": {
        "title": "Philippines Country Data 2023",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/phillippines-data-book-2023.pdf",
        "release_year": 2023,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/philippines-country-data-2023": {
        "title": "Philippines Country Data 2023",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/phillippines-data-book-2023.pdf",
        "release_year": 2023,
        "resource_kind": "annual_country_summary",
    },
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-january-2021": {
        "title": "Hiv Aids And Art Registry Philippines January 2021",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/eb-harp-january-aidsreg2021.pdf",
        "release_year": 2021,
        "resource_kind": "harp_registry",
    },
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-july-2019": {
        "title": "Hiv Aids And Art Registry Philippines July 2019",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/eb-harp-july-aidsreg2019.pdf",
        "release_year": 2019,
        "resource_kind": "harp_registry",
    },
    "https://www.aidsdatahub.org/resource/hiv-aids-and-art-registry-philippines-october-december-2019": {
        "title": "Hiv Aids And Art Registry Philippines October December 2019",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/eb-harp-october-december-aidsreg2019.pdf",
        "release_year": 2019,
        "resource_kind": "harp_registry",
    },
    "https://www.aidsdatahub.org/resource/hiv-aids-surveillance-philippines-oct-dec-2023": {
        "title": "Hiv Aids Surveillance Philippines Oct Dec 2023",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/hiv-2023-oct-dec.pdf",
        "release_year": 2023,
        "resource_kind": "surveillance_report",
    },
    "https://www.aidsdatahub.org/resource/philippines-hiv-aids-surveillance-technical-report-2003": {
        "title": "Philippines HIV/AIDS Surveillance Technical Report 2003",
        "pdf_url": "https://www.aidsdatahub.org/sites/default/files/resource/philippines-hiv-aids-surveillance-technical-report-2003.pdf",
        "release_year": 2003,
        "resource_kind": "surveillance_technical_report",
    },
}


@dataclass(slots=True)
class ResourceRecord:
    detail_url: str
    title: str
    pdf_url: str
    release_year: int | None
    resource_kind: str
    published_label: str = ""
    local_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _download_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "EpiGraph-PH-AIDSDataHub-Extractor/1.0 (+https://github.com/openai/codex)",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _curated_override_record(detail_url: str) -> ResourceRecord | None:
    override = CURATED_RESOURCE_OVERRIDES.get(str(detail_url or "").strip())
    if not override:
        return None
    return ResourceRecord(
        detail_url=str(detail_url or "").strip(),
        title=str(override.get("title") or Path(str(detail_url or "")).name),
        pdf_url=str(override.get("pdf_url") or "").strip(),
        release_year=int(override.get("release_year")) if override.get("release_year") is not None else None,
        resource_kind=str(override.get("resource_kind") or classify_resource_kind(str(override.get("title") or ""), str(detail_url or ""))),
    )


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_ascii_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return cleaned or "document"


def _find_cached_download(target_name: str) -> Path | None:
    runs_root = Path(__file__).resolve().parents[3] / "artifacts" / "runs"
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for pattern in (f"*/aidsdatahub_philippines/downloads/{target_name}", f"*/downloads/{target_name}"):
        candidates.extend(path for path in runs_root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _canonical_url(base_url: str, href: str) -> str:
    return urljoin(base_url, str(href or "").strip())

def _slug_token(url: str) -> str:
    normalized = re.sub(r"[?#].*$", "", str(url or "").strip()).rstrip("/")
    token = normalized.rsplit("/", 1)[-1] if normalized else ""
    if token.lower().endswith(".pdf"):
        token = token[:-4]
    return token.strip()


def _display_title(title: str, *, detail_url: str, pdf_url: str = "") -> str:
    normalized = _normalize_whitespace(title)
    if normalized and normalized.lower() != "hiv/aids data hub for the asia pacific":
        return normalized
    slug = _slug_token(detail_url) or _slug_token(pdf_url)
    if not slug:
        return normalized or "AIDS Data Hub Resource"
    return _normalize_whitespace(re.sub(r"[-_]+", " ", slug)).title()




def _discover_detail_urls_from_html(html: str, *, base_url: str) -> list[str]:
    urls: set[str] = set()
    for match in re.finditer(r"""href=["']([^"'#]+)["']""", html, flags=re.I):
        href = str(match.group(1) or "").strip()
        if not href:
            continue
        full_url = _canonical_url(base_url, href)
        lowered = full_url.lower()
        if lowered.endswith(".pdf"):
            if "philipp" in lowered:
                urls.add(full_url)
            continue
        if "/resource/" in lowered or "/index.php/resource/" in lowered:
            if "philipp" in lowered:
                urls.add(full_url)
    return sorted(urls)


def _extract_title_from_html(html: str) -> str:
    for pattern in (
        r"""property=["']og:title["'][^>]*content=["']([^"']+)["']""",
        r"""<title>([^<]+)</title>""",
        r"""<h1[^>]*>(.*?)</h1>""",
    ):
        match = re.search(pattern, html, flags=re.I | re.S)
        if match:
            title = re.sub(r"<[^>]+>", " ", match.group(1))
            return _normalize_whitespace(title)
    return ""


def _extract_pdf_urls_from_html(html: str, *, base_url: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r"""href=["']([^"']+\.pdf(?:\?[^"']*)?)["']""", html, flags=re.I):
        urls.append(_canonical_url(base_url, match.group(1)))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def _extract_release_year(title: str, url: str) -> int | None:
    candidates = [int(token) for token in re.findall(r"(20\d{2})", f"{title} {url}")]
    return max(candidates) if candidates else None


def classify_resource_kind(title: str, url: str) -> str:
    lowered = f"{title} {url}".lower()
    if any(token in lowered for token in ("country data", "country-data", "country snapshot", "country-snapshot")):
        return "annual_country_summary"
    if ("registry" in lowered and "philippines" in lowered) or "aidsreg" in lowered or "art-registry" in lowered:
        return "harp_registry"
    if any(token in lowered for token in ("surveillance technical report", "surveillance-technical-report")):
        return "surveillance_technical_report"
    if "surveillance" in lowered and "philippines" in lowered:
        return "surveillance_report"
    return "other"


def _month_label(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _parse_temporal_metadata_from_title(title: str, release_year: int | None) -> dict[str, str]:
    lowered = _normalize_whitespace(title).lower()
    quarter_match = re.search(
        r"\b([a-z]{3,9})\s*[-/]\s*([a-z]{3,9})\s*(20\d{2})\b",
        lowered,
        flags=re.I,
    )
    if quarter_match:
        start_month = MONTH_NAME_TO_NUMBER.get(quarter_match.group(1).lower())
        end_month = MONTH_NAME_TO_NUMBER.get(quarter_match.group(2).lower())
        year = int(quarter_match.group(3))
        if start_month is not None and end_month is not None:
            return {
                "temporal_precision": "quarterly_snapshot",
                "start_month": _month_label(year, start_month),
                "end_month": _month_label(year, end_month),
                "effective_month": _month_label(year, end_month),
            }
    month_match = re.search(r"\b([a-z]{3,9})\s+(20\d{2})\b", lowered, flags=re.I)
    if month_match:
        month = MONTH_NAME_TO_NUMBER.get(month_match.group(1).lower())
        year = int(month_match.group(2))
        if month is not None:
            month_label = _month_label(year, month)
            return {
                "temporal_precision": "monthly_snapshot",
                "start_month": month_label,
                "end_month": month_label,
                "effective_month": month_label,
            }
    year = int(release_year or 0) if release_year else 0
    if year > 0:
        month_label = _month_label(year, 12)
        return {
            "temporal_precision": "annual_summary",
            "start_month": month_label,
            "end_month": month_label,
            "effective_month": month_label,
        }
    return {
        "temporal_precision": "unknown",
        "start_month": "",
        "end_month": "",
        "effective_month": "",
    }


def _harp_report_row(resource: ResourceRecord, *, local_path: Path) -> dict[str, Any]:
    temporal = _parse_temporal_metadata_from_title(resource.title, resource.release_year)
    source_year = int((temporal.get("effective_month") or "0000")[:4] or 0)
    return {
        "source_id": f"aidsdatahub_{_safe_ascii_label(resource.title)}",
        "label": resource.title,
        "source_kind": "aidsdatahub_pdf",
        "source_label": "AIDS Data Hub",
        "source_year": source_year,
        "source_url": resource.pdf_url or resource.detail_url,
        "local_path": str(local_path),
        "temporal_precision": temporal["temporal_precision"],
        "effective_month": temporal["effective_month"],
        "start_month": temporal["start_month"],
        "end_month": temporal["end_month"],
    }


def _pdf_text(local_path: Path, *, max_pages: int = 12) -> str:
    pages: list[str] = []
    if PdfReader is not None:
        try:
            reader = PdfReader(str(local_path))
            for page in list(reader.pages)[:max_pages]:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    pages.append(text)
        except Exception:
            pages = []
    joined = "\n".join(pages)
    if len(_normalize_whitespace(joined)) >= 200:
        return joined
    try:
        ocr_pages = extract_archive_pdf_pages(local_path, max_pages=max_pages)
    except Exception:
        return joined
    return "\n".join(str(page.get("text") or "") for page in ocr_pages)



def _phase0_ocr_fallback_blocks(
    local_path: Path,
    *,
    requested_backend: str = "auto",
    max_pages: int = 5,
    layout_hint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from epigraph_ph.phase0.pipeline import _lighton_ocr_sidecar_blocks

    blocks, meta = _lighton_ocr_sidecar_blocks(
        local_path,
        preferred_pages=None,
        max_pages=max_pages,
        requested_backend=requested_backend,
        layout_hint=layout_hint,
    )
    return [dict(block) for block in list(blocks or [])], dict(meta or {})
def _token_to_numeric(token: str) -> tuple[float | None, str]:
    raw = _normalize_whitespace(token).replace(",", "")
    if not raw or raw == "...":
        return None, "missing"
    if raw.startswith("<"):
        numeric = re.sub(r"[^0-9.]", "", raw)
        if not numeric:
            return None, "missing"
        return float(numeric), "upper_bound"
    numeric = re.sub(r"[^0-9.]", "", raw)
    if not numeric:
        return None, "missing"
    return float(numeric), "point_estimate"


def _country_anchor_rows(
    text: str,
    *,
    release_year: int,
    source_id: str,
    source_label: str,
    source_url: str,
) -> list[dict[str, Any]]:
    latest_year = int(release_year) - 1
    normalized_text = _normalize_whitespace(text)
    lines = [str(line or "").strip() for line in str(text or "").splitlines() if str(line or "").strip()]
    rows: list[dict[str, Any]] = []
    for label, (metric_name, unit) in COUNTRY_ANCHOR_SERIES.items():
        tokens: list[str] = []
        for idx, line in enumerate(lines):
            if label.lower() not in line.lower():
                continue
            for next_line in lines[idx + 1 :]:
                if any(other_label.lower() in next_line.lower() for other_label in COUNTRY_ANCHOR_SERIES.keys() if other_label != label):
                    break
                tokens.extend(NUMERIC_TOKEN_RE.findall(next_line))
                if len(tokens) >= 3:
                    break
            if tokens:
                break
        if len(tokens) < 3:
            start = normalized_text.lower().find(label.lower())
            if start >= 0:
                chunk = normalized_text[start + len(label) : start + len(label) + 250]
                tokens = NUMERIC_TOKEN_RE.findall(chunk)
        if len(tokens) < 3:
            continue
        for year, token in zip((2010, 2015, latest_year), tokens[:3], strict=True):
            value, semantics = _token_to_numeric(token)
            if value is None:
                continue
            rows.append(
                {
                    "year": int(year),
                    "time": f"{int(year):04d}-01",
                    "metric_name": metric_name,
                    "value": float(value),
                    "unit": unit,
                    "source_id": source_id,
                    "source_label": source_label,
                    "measurement_class": "model_estimate",
                    "series_kind": "annual_anchor",
                    "temporal_precision": "annual_anchor",
                    "period_start": f"{int(year):04d}-01",
                    "period_end": f"{int(year):04d}-12",
                    "geo": "Philippines",
                    "region": "national",
                    "province": "Philippines",
                    "evidence_confidence": 0.97,
                    "source_url": source_url,
                    "source_note": f"AIDS Data Hub annual country summary ({release_year})",
                    "source_quality_tier": "official_unaids_country_data",
                    "reported_value": _normalize_whitespace(token),
                    "value_semantics": semantics,
                    "extraction_method": "aidsdatahub_country_anchor_regex",
                }
            )
    return rows


def _country_snapshot_year(text: str, release_year: int) -> int:
    snapshot_match = re.search(r"country\s*snapshot[, ]+\s*((?:19|20)\d{2})", str(text or ""), flags=re.I)
    if snapshot_match:
        return int(snapshot_match.group(1))
    return max(int(release_year) - 1, 1900)


def _country_snapshot_card_rows(
    text: str,
    *,
    release_year: int,
    source_id: str,
    source_label: str,
    source_url: str,
) -> list[dict[str, Any]]:
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return []
    snapshot_year = _country_snapshot_year(text, release_year)
    rows: list[dict[str, Any]] = []
    num_pattern = r"(<\s*\d[\d, ]*|\d[\d, ]*)"
    point_metrics = (
        (
            r"People\s+living\s+with\s+HIV(?:\s*\(PLHIV\))?\s*"
            rf"(?P<point>{num_pattern})\s*Low\s*estimate\s*(?P<low>{num_pattern})\s*High\s*estimate\s*(?P<high>{num_pattern})",
            "estimated_plhiv",
            "count_people",
            "model_estimate",
            snapshot_year,
        ),
        (
            r"New\s+HIV\s+infections\s*"
            rf"(?P<point>{num_pattern})\s*Low\s*estimate\s*(?P<low>{num_pattern})\s*High\s*estimate\s*(?P<high>{num_pattern})",
            "annual_new_infections",
            "count_people",
            "model_estimate",
            snapshot_year,
        ),
        (
            r"AIDS[- ]?related\s*deaths\s*"
            rf"(?P<point>{num_pattern})\s*Low\s*estimate\s*(?P<low>{num_pattern})\s*High\s*estimate\s*(?P<high>{num_pattern})",
            "annual_aids_deaths",
            "count_people",
            "model_estimate",
            snapshot_year,
        ),
    )
    for pattern, metric_name, unit, measurement_class, year in point_metrics:
        match = re.search(pattern, normalized_text, flags=re.I)
        if not match:
            continue
        for value_key, semantics in (("point", "point_estimate"), ("low", "lower_bound"), ("high", "upper_bound")):
            value, _ = _token_to_numeric(str(match.group(value_key) or ""))
            if value is None:
                continue
            rows.append(
                {
                    "year": int(year),
                    "time": f"{int(year):04d}-01",
                    "metric_name": metric_name,
                    "value": float(value),
                    "unit": unit,
                    "source_id": source_id,
                    "source_label": source_label,
                    "measurement_class": measurement_class,
                    "series_kind": "annual_snapshot_card",
                    "temporal_precision": "annual_snapshot",
                    "period_start": f"{int(year):04d}-01",
                    "period_end": f"{int(year):04d}-12",
                    "geo": "Philippines",
                    "region": "national",
                    "province": "Philippines",
                    "evidence_confidence": 0.9,
                    "source_url": source_url,
                    "source_note": f"AIDS Data Hub country snapshot ({snapshot_year})",
                    "source_quality_tier": "official_unaids_country_snapshot",
                    "reported_value": _normalize_whitespace(str(match.group(value_key) or "")),
                    "value_semantics": semantics,
                    "extraction_method": "aidsdatahub_country_snapshot_card",
                }
            )
    art_match = re.search(
        r"People\s+on\s+ART\s*\((?P<year>(?:19|20)\d{2})\)\s*(?P<point><\s*\d[\d, ]*|\d[\d, ]*)",
        normalized_text,
        flags=re.I,
    )
    if art_match:
        art_year = int(art_match.group("year"))
        art_value, art_semantics = _token_to_numeric(str(art_match.group("point") or ""))
        if art_value is not None:
            rows.append(
                {
                    "year": art_year,
                    "time": f"{art_year:04d}-01",
                    "metric_name": "alive_on_art",
                    "value": float(art_value),
                    "unit": "count_people",
                    "source_id": source_id,
                    "source_label": source_label,
                    "measurement_class": "program_observed_harp",
                    "series_kind": "annual_snapshot_card",
                    "temporal_precision": "annual_snapshot",
                    "period_start": f"{art_year:04d}-01",
                    "period_end": f"{art_year:04d}-12",
                    "geo": "Philippines",
                    "region": "national",
                    "province": "Philippines",
                    "evidence_confidence": 0.9,
                    "source_url": source_url,
                    "source_note": f"AIDS Data Hub country snapshot ART panel ({art_year})",
                    "source_quality_tier": "official_unaids_country_snapshot",
                    "reported_value": _normalize_whitespace(str(art_match.group('point') or "")),
                    "value_semantics": art_semantics,
                    "extraction_method": "aidsdatahub_country_snapshot_card",
                }
            )
    lower_text = normalized_text.lower()
    cascade_start = lower_text.find("treatment cascade")
    if cascade_start >= 0:
        cascade_chunk = normalized_text[cascade_start : cascade_start + 450]
        cascade_chunk = re.split(r"Prevention\s+of\s+Mother|Key\s+Populations", cascade_chunk, maxsplit=1, flags=re.I)[0]
        cascade_year_match = re.search(r"Treatment\s+Cascade,\s*(?P<year>(?:19|20)\d{2})", cascade_chunk, flags=re.I)
        if cascade_year_match:
            cascade_year = int(cascade_year_match.group("year"))
            numeric_values: list[float] = []
            for token in TECHNICAL_NUMERIC_TOKEN_RE.findall(cascade_chunk):
                value, _ = _token_to_numeric(token)
                if value is None:
                    continue
                numeric_values.append(float(value))
            if numeric_values and abs(numeric_values[0] - float(cascade_year)) < 1e-6:
                numeric_values = numeric_values[1:]
            if numeric_values and numeric_values[-1] == 0.0:
                numeric_values = numeric_values[:-1]
            if len(numeric_values) >= 6 and abs(numeric_values[0] - numeric_values[1]) < 1e-6:
                numeric_values = numeric_values[1:]
            if len(numeric_values) >= 5:
                cascade_mapping = (
                    ("estimated_plhiv", numeric_values[0], "model_estimate"),
                    ("diagnosed_plhiv", numeric_values[1], "program_observed_harp"),
                    ("alive_on_art", numeric_values[2], "program_observed_harp"),
                    ("tested_for_viral_load", numeric_values[3], "program_observed_harp"),
                    ("virally_suppressed", numeric_values[4], "program_observed_harp"),
                )
                for metric_name, value, measurement_class in cascade_mapping:
                    rows.append(
                        {
                            "year": cascade_year,
                            "time": f"{cascade_year:04d}-01",
                            "metric_name": metric_name,
                            "value": float(value),
                            "unit": "count_people",
                            "source_id": source_id,
                            "source_label": source_label,
                            "measurement_class": measurement_class,
                            "series_kind": "annual_snapshot_card",
                            "temporal_precision": "annual_snapshot",
                            "period_start": f"{cascade_year:04d}-01",
                            "period_end": f"{cascade_year:04d}-12",
                            "geo": "Philippines",
                            "region": "national",
                            "province": "Philippines",
                            "evidence_confidence": 0.87,
                            "source_url": source_url,
                            "source_note": f"AIDS Data Hub treatment cascade panel ({cascade_year})",
                            "source_quality_tier": "official_unaids_country_snapshot",
                            "reported_value": f"{value:g}",
                            "value_semantics": "point_estimate",
                            "extraction_method": "aidsdatahub_country_snapshot_treatment_cascade",
                        }
                    )
    return rows


def _text_rich_for_native_technical_parsing(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if len(normalized) < 1200:
        return False
    alpha_count = sum(1 for char in normalized if char.isalpha())
    return alpha_count >= 600


def _technical_report_metric_name(caption: str) -> str:
    normalized = _normalize_whitespace(caption)
    normalized = re.sub(r"^(Figure|Table)\s*\d+[A-Za-z]*\s*\.?\s*", "", normalized, flags=re.I)
    normalized = re.sub(r"\b(?:BSS|HSS|HIV/AIDS Registry|Behavioral Sentinel Surveillance|HIV Serologic Surveillance)\b.*$", "", normalized, flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return f"technical_report_{slug}" if slug else "technical_report_series"


def _technical_report_unit(caption: str, block_text: str) -> str:
    lowered = f"{caption} {block_text}".lower()
    if "percent" in lowered or "proportion" in lowered or "%" in lowered:
        return "percent"
    if "frequency" in lowered or "no." in lowered or "number of" in lowered or "seropositives" in lowered:
        return "count_people"
    return "count_people"


def _technical_report_geo(caption: str) -> str:
    match = re.search(r"(?:BSS|HSS),\s*([^,]+?)\s*,\s*(?:19|20)\d{2}", caption, flags=re.I)
    if match:
        return _normalize_whitespace(match.group(1))
    return "Philippines"


def _parse_years_from_block(block_lines: list[str], caption: str) -> list[int]:
    del caption
    for line in block_lines:
        raw_tokens = re.findall(r"(?:19|20)\d{2}[A-Za-z]?", line)
        years = [int(token[:4]) for token in raw_tokens]
        if len(years) < 4:
            continue
        if years != sorted(years):
            continue
        unique_years = sorted(set(years))
        if len(unique_years) < 4:
            continue
        gaps = [b - a for a, b in zip(unique_years, unique_years[1:], strict=False)]
        if gaps and max(gaps) > 2:
            continue
        return years
    return []


def _technical_report_year_header(block_lines: list[str]) -> tuple[int, list[int]]:
    for idx, line in enumerate(block_lines):
        years = _parse_years_from_block([line], "")
        if years:
            return idx, years
    return -1, []


def _technical_report_year_series_rows(
    text: str,
    *,
    source_id: str,
    source_label: str,
    source_url: str,
) -> list[dict[str, Any]]:
    lines = [str(line or "").strip() for line in str(text or "").splitlines() if str(line or "").strip()]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^(Table)\s+\d+", line, flags=re.I) or re.match(r"^(Appendix)\s+[A-Z0-9]+", line, flags=re.I):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    rows: list[dict[str, Any]] = []
    for block in blocks:
        year_header_idx, years = _technical_report_year_header(block)
        if year_header_idx < 0 or not years:
            continue
        caption_lines = block[:year_header_idx]
        data_lines = block[year_header_idx + 1 :]
        caption = _normalize_whitespace(" ".join(caption_lines))
        metric_name = _technical_report_metric_name(caption)
        unit = _technical_report_unit(caption, " ".join(block[: min(len(block), 12)]))
        geo = _technical_report_geo(caption)
        parsed_row_count = 0
        for line in data_lines:
            if re.match(r"^(Figure|Table|Appendix)\s+", line, flags=re.I):
                break
            tokens = TECHNICAL_NUMERIC_TOKEN_RE.findall(line)
            label = TECHNICAL_NUMERIC_TOKEN_RE.sub("", line).strip(" :-")
            clean_label = _normalize_whitespace(label)
            lowered_label = clean_label.lower()
            if parsed_row_count > 0 and (not tokens or (len(tokens) <= 1 and len(clean_label.split()) > 5)):
                break
            if not label or len(tokens) < 3:
                continue
            if lowered_label in {"year", "percent", "risk group", "risk groups", "high risk group", "age", "range", "median", "mode of transmission", "round", "city"}:
                continue
            if not re.match(r"^[A-Za-z][A-Za-z0-9/ &'()._-]{0,80}$", clean_label):
                continue
            if len(clean_label.split()) > 5:
                continue
            if len(tokens) > len(years):
                continue
            numeric_values: list[float] = []
            semantics: list[str] = []
            for token in tokens:
                value, value_semantics = _token_to_numeric(token)
                if value is None:
                    continue
                numeric_values.append(float(value))
                semantics.append(value_semantics)
            if len(numeric_values) < 3:
                continue
            count = min(len(years), len(numeric_values), len(semantics))
            if count < 3:
                continue
            aligned_years = years[-count:] if count < len(years) else years[:count]
            aligned_values = numeric_values[-count:] if count < len(numeric_values) else numeric_values[:count]
            aligned_semantics = semantics[-count:] if count < len(semantics) else semantics[:count]
            for year, value, value_semantics in zip(aligned_years, aligned_values, aligned_semantics, strict=True):
                rows.append(
                    {
                        "year": int(year),
                        "time": f"{int(year):04d}-01",
                        "metric_name": metric_name,
                        "value": float(value),
                        "unit": unit,
                        "source_id": source_id,
                        "source_label": source_label,
                        "measurement_class": "survey_series",
                        "series_kind": "technical_report_year_series",
                        "temporal_precision": "annual_series",
                        "period_start": f"{int(year):04d}-01",
                        "period_end": f"{int(year):04d}-12",
                        "geo": geo,
                        "region": "",
                        "province": geo,
                        "evidence_confidence": 0.88,
                        "source_url": source_url,
                        "source_note": caption,
                        "source_quality_tier": "official_surveillance_technical_report",
                        "reported_value": str(value),
                        "value_semantics": value_semantics,
                        "subgroup": clean_label,
                        "group_kind": "technical_report_series",
                        "extraction_method": "aidsdatahub_technical_report_year_table",
                    }
                )
            parsed_row_count += 1
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = (
            str(row.get("metric_name") or ""),
            str(row.get("time") or ""),
            str(row.get("source_id") or ""),
            str(row.get("value") or ""),
            str(row.get("reported_value") or ""),
        )
        stable = "|".join(key)
        if stable in seen:
            continue
        seen.add(stable)
        deduped.append(dict(row))
    deduped.sort(key=lambda item: (int(item.get("year") or 0), str(item.get("time") or ""), str(item.get("metric_name") or ""), str(item.get("source_id") or "")))
    return deduped


def discover_philippines_resources(
    *,
    session: requests.Session | None = None,
    listing_urls: tuple[str, ...] = COUNTRY_LISTING_URLS,
    curated_detail_urls: tuple[str, ...] = CURATED_DETAIL_URLS,
) -> list[str]:
    close_session = False
    if session is None:
        session = _download_session()
        close_session = True
    detail_urls: set[str] = set(curated_detail_urls)
    try:
        for listing_url in listing_urls:
            try:
                response = session.get(listing_url, timeout=60)
                response.raise_for_status()
            except requests.RequestException:
                continue
            detail_urls.update(_discover_detail_urls_from_html(response.text, base_url=listing_url))
    finally:
        if close_session:
            session.close()
    return sorted(detail_urls)


def hydrate_resource_record(detail_url: str, *, session: requests.Session | None = None) -> ResourceRecord:
    close_session = False
    if session is None:
        session = _download_session()
        close_session = True
    try:
        override_record = _curated_override_record(detail_url)
        if override_record is not None:
            return override_record
        if detail_url.lower().endswith(".pdf"):
            title = Path(detail_url).name
            return ResourceRecord(
                detail_url=detail_url,
                title=title,
                pdf_url=detail_url,
                release_year=_extract_release_year(title, detail_url),
                resource_kind=classify_resource_kind(title, detail_url),
            )
        response = session.get(detail_url, timeout=60)
        response.raise_for_status()
        html = response.text
        title = _extract_title_from_html(html) or Path(detail_url).name
        pdf_urls = _extract_pdf_urls_from_html(html, base_url=detail_url)
        pdf_url = pdf_urls[0] if pdf_urls else ""
        title = _display_title(title, detail_url=detail_url, pdf_url=pdf_url)
        return ResourceRecord(
            detail_url=detail_url,
            title=title,
            pdf_url=pdf_url,
            release_year=_extract_release_year(title, detail_url),
            resource_kind=classify_resource_kind(title, detail_url),
        )
    finally:
        if close_session:
            session.close()


def _download_pdf(resource: ResourceRecord, *, session: requests.Session, download_dir: Path, refresh: bool) -> Path | None:
    if not resource.pdf_url:
        return None
    pdf_slug = _slug_token(resource.pdf_url)
    target_name = f"{_safe_ascii_label(pdf_slug or resource.title)}.pdf"
    local_path = download_dir / target_name
    if local_path.exists() and not refresh:
        return local_path
    try:
        response = session.get(resource.pdf_url, timeout=120)
        response.raise_for_status()
        ensure_dir(local_path.parent)
        local_path.write_bytes(response.content)
        return local_path
    except requests.RequestException:
        cached_path = _find_cached_download(target_name)
        if cached_path is None:
            raise
        ensure_dir(local_path.parent)
        shutil.copy2(cached_path, local_path)
        return local_path


def run_aidsdatahub_philippines_extract(
    *,
    run_id: str,
    plugin_id: str = "hiv",
    refresh: bool = False,
    max_resources: int | None = None,
    use_phase0_ocr_fallback: bool = False,
    phase0_ocr_backend: str = "auto",
    phase0_ocr_max_pages: int = 5,
) -> dict[str, str]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    output_dir = ensure_dir(ctx.run_dir / "aidsdatahub_philippines")
    download_dir = ensure_dir(output_dir / "downloads")
    session = _download_session()
    resource_urls = discover_philippines_resources(session=session)
    if max_resources is not None:
        resource_urls = resource_urls[: int(max_resources)]
    resource_records: list[ResourceRecord] = []
    metric_rows: list[dict[str, Any]] = []
    skipped_resources: list[dict[str, Any]] = []
    downloaded_manifest: list[dict[str, Any]] = []
    phase0_ocr_fallback_manifest: list[dict[str, Any]] = []
    try:
        for detail_url in resource_urls:
            try:
                resource = hydrate_resource_record(detail_url, session=session)
            except requests.RequestException as exc:
                skipped_resources.append({"detail_url": detail_url, "reason": f"hydrate_failed:{exc.__class__.__name__}"})
                continue
            resource_records.append(resource)
            if not resource.pdf_url:
                skipped_resources.append({"detail_url": detail_url, "reason": "missing_pdf_url", "title": resource.title})
                continue
            try:
                local_path = _download_pdf(resource, session=session, download_dir=download_dir, refresh=refresh)
            except requests.RequestException as exc:
                skipped_resources.append({"detail_url": detail_url, "title": resource.title, "reason": f"download_failed:{exc.__class__.__name__}"})
                continue
            if local_path is None:
                skipped_resources.append({"detail_url": detail_url, "title": resource.title, "reason": "no_local_pdf"})
                continue
            resource.local_path = str(local_path)
            downloaded_manifest.append(
                {
                    "detail_url": resource.detail_url,
                    "title": resource.title,
                    "pdf_url": resource.pdf_url,
                    "local_path": str(local_path),
                    "checksum": sha256_file(local_path),
                    "resource_kind": resource.resource_kind,
                }
            )

            resource_rows: list[dict[str, Any]] = []
            native_text = ""
            if resource.resource_kind in {"harp_registry", "surveillance_report"}:
                report_row = _harp_report_row(resource, local_path=local_path)
                page_texts = extract_archive_pdf_pages(local_path, max_pages=4)
                resource_rows = list(extract_archive_metric_rows(report_row, page_texts) or [])
            elif resource.resource_kind == "annual_country_summary" and resource.release_year:
                native_text = _pdf_text(local_path, max_pages=12)
                source_id = f"aidsdatahub_{_safe_ascii_label(resource.title)}"
                resource_rows = _country_anchor_rows(
                    native_text,
                    release_year=int(resource.release_year),
                    source_id=source_id,
                    source_label=resource.title,
                    source_url=resource.pdf_url,
                )
                resource_rows.extend(
                    _country_snapshot_card_rows(
                        native_text,
                        release_year=int(resource.release_year),
                        source_id=source_id,
                        source_label=resource.title,
                        source_url=resource.pdf_url,
                    )
                )
            elif resource.resource_kind == "surveillance_technical_report":
                native_text = _pdf_text(local_path, max_pages=120)
                source_id = f"aidsdatahub_{_safe_ascii_label(resource.title)}"
                resource_rows = _technical_report_year_series_rows(
                    native_text,
                    source_id=source_id,
                    source_label=resource.title,
                    source_url=resource.pdf_url,
                )
            if resource_rows:
                metric_rows.extend(resource_rows)
                continue

            if use_phase0_ocr_fallback:
                fallback_blocks: list[dict[str, Any]] = []
                fallback_meta: dict[str, Any] = {}
                fallback_rows: list[dict[str, Any]] = []
                fallback_text = ""
                fallback_text_path = output_dir / "phase0_ocr_fallback" / f"{_safe_ascii_label(resource.title)}.txt"
                layout_hint = "annual_country_summary" if resource.resource_kind == "annual_country_summary" else None
                should_attempt_ocr = not (
                    resource.resource_kind == "surveillance_technical_report"
                    and _text_rich_for_native_technical_parsing(native_text)
                )
                if should_attempt_ocr:
                    try:
                        fallback_blocks, fallback_meta = _phase0_ocr_fallback_blocks(
                            local_path,
                            requested_backend=phase0_ocr_backend,
                            max_pages=int(phase0_ocr_max_pages),
                            layout_hint=layout_hint,
                        )
                    except Exception as exc:
                        fallback_meta = {"status": f"failed:{exc.__class__.__name__}", "failure_notes": [str(exc)]}
                    fallback_text = "\n".join(str(block.get("text") or "") for block in fallback_blocks)
                    if fallback_text.strip():
                        ensure_dir(fallback_text_path.parent)
                        fallback_text_path.write_text(fallback_text, encoding="utf-8")
                else:
                    fallback_meta = {
                        "status": "skipped_text_rich_native_first",
                        "failure_notes": ["native_text_rich_technical_report"],
                        "layout_hint": str(layout_hint or ""),
                    }
                if resource.resource_kind == "annual_country_summary" and resource.release_year and fallback_text.strip():
                    source_id = f"aidsdatahub_{_safe_ascii_label(resource.title)}"
                    fallback_rows.extend(
                        _country_anchor_rows(
                            fallback_text,
                            release_year=int(resource.release_year),
                            source_id=source_id,
                            source_label=resource.title,
                            source_url=resource.pdf_url,
                        )
                    )
                    fallback_rows.extend(
                        _country_snapshot_card_rows(
                            fallback_text,
                            release_year=int(resource.release_year),
                            source_id=source_id,
                            source_label=resource.title,
                            source_url=resource.pdf_url,
                        )
                    )
                if resource.resource_kind == "surveillance_technical_report" and fallback_text.strip():
                    source_id = f"aidsdatahub_{_safe_ascii_label(resource.title)}"
                    fallback_rows.extend(
                        _technical_report_year_series_rows(
                            fallback_text,
                            source_id=source_id,
                            source_label=resource.title,
                            source_url=resource.pdf_url,
                        )
                    )
                if resource.resource_kind in {"harp_registry", "surveillance_report"} and fallback_blocks:
                    report_row = _harp_report_row(resource, local_path=local_path)
                    fallback_rows.extend(extract_archive_metric_rows(report_row, fallback_blocks))
                if resource.resource_kind == "surveillance_technical_report" and fallback_blocks and not fallback_rows:
                    report_row = _harp_report_row(resource, local_path=local_path)
                    fallback_rows.extend(extract_archive_metric_rows(report_row, fallback_blocks))
                phase0_ocr_fallback_manifest.append(
                    {
                        "detail_url": resource.detail_url,
                        "title": resource.title,
                        "resource_kind": resource.resource_kind,
                        "local_path": str(local_path),
                        "fallback_backend": str(fallback_meta.get("backend") or phase0_ocr_backend),
                        "fallback_status": str(fallback_meta.get("status") or ""),
                        "fallback_failure_notes": list(fallback_meta.get("failure_notes") or []),
                        "fallback_page_count": int(fallback_meta.get("page_count") or 0),
                        "fallback_parsed_page_count": int(fallback_meta.get("parsed_page_count") or 0),
                        "fallback_text_path": str(fallback_text_path) if fallback_text.strip() else "",
                        "fallback_row_count": len(fallback_rows),
                    }
                )
                if fallback_rows:
                    metric_rows.extend(fallback_rows)
                    continue

            skipped_resources.append({"detail_url": detail_url, "title": resource.title, "reason": f"unsupported_kind:{resource.resource_kind}"})
    finally:
        session.close()

    metric_rows = _deduplicate_rows(metric_rows)
    resource_inventory_path = output_dir / "resource_inventory.json"
    download_manifest_path = output_dir / "download_manifest.json"
    rows_json_path = output_dir / "historical_metric_rows.json"
    rows_csv_path = output_dir / "historical_metric_rows.csv"
    skipped_path = output_dir / "skipped_resources.json"
    phase0_ocr_fallback_path = output_dir / "phase0_ocr_fallback_manifest.json"
    manifest_path = output_dir / "aidsdatahub_philippines_manifest.json"

    write_json(resource_inventory_path, [resource.to_dict() for resource in resource_records])
    write_json(download_manifest_path, downloaded_manifest)
    write_json(rows_json_path, metric_rows)
    _write_csv(rows_csv_path, metric_rows)
    write_json(skipped_path, skipped_resources)
    write_json(phase0_ocr_fallback_path, phase0_ocr_fallback_manifest)
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "plugin_id": plugin_id,
            "generated_at": utc_now_iso(),
            "resource_count": len(resource_records),
            "downloaded_count": len(downloaded_manifest),
            "metric_row_count": len(metric_rows),
            "resource_inventory_path": str(resource_inventory_path),
            "download_manifest_path": str(download_manifest_path),
            "historical_metric_rows_path": str(rows_json_path),
            "historical_metric_rows_csv_path": str(rows_csv_path),
            "skipped_resources_path": str(skipped_path),
            "phase0_ocr_fallback_manifest_path": str(phase0_ocr_fallback_path),
        },
    )
    return {
        "resource_inventory_path": str(resource_inventory_path),
        "download_manifest_path": str(download_manifest_path),
        "historical_metric_rows_path": str(rows_json_path),
        "historical_metric_rows_csv_path": str(rows_csv_path),
        "skipped_resources_path": str(skipped_path),
        "phase0_ocr_fallback_manifest_path": str(phase0_ocr_fallback_path),
        "manifest_path": str(manifest_path),
    }

__all__ = [
    "COUNTRY_LISTING_URLS",
    "CURATED_DETAIL_URLS",
    "ResourceRecord",
    "classify_resource_kind",
    "discover_philippines_resources",
    "hydrate_resource_record",
    "run_aidsdatahub_philippines_extract",
    "_country_anchor_rows",
    "_discover_detail_urls_from_html",
    "_parse_temporal_metadata_from_title",
]



