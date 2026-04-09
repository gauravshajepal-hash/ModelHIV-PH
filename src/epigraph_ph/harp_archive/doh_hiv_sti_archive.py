from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, sha256_file, utc_now_iso, write_json

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover
    RapidOCR = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


_HIV_PLUGIN = get_disease_plugin("hiv")
_PHASE0_CFG = dict((_HIV_PLUGIN.constraint_settings or {}).get("phase0", {}) or {})
_PHASE0_OCR_CFG = dict(_PHASE0_CFG.get("ocr", {}) or {})
_ARCHIVE_OCR_CFG = dict(_PHASE0_OCR_CFG.get("archive_full_document", {}) or {})
_RENDER_DPI = int(float(_PHASE0_OCR_CFG.get("render_dpi") or 200.0))
_DIRECT_TEXT_CHAR_FLOOR = int(float(_PHASE0_OCR_CFG.get("pdf_requires_ocr_avg_chars") or 120.0))
_ARCHIVE_FORCE_OCR_EVERY_PAGE = bool(_ARCHIVE_OCR_CFG.get("force_ocr_every_page", True))
_ARCHIVE_MAX_PAGES_PER_DOCUMENT = int(float(_ARCHIVE_OCR_CFG.get("max_pages_per_document") or 0.0))
_ARCHIVE_CHECKPOINT_EVERY_DOCUMENTS = int(float(_ARCHIVE_OCR_CFG.get("checkpoint_every_documents") or 1.0))
_ARCHIVE_PAUSE_SECONDS = float(_ARCHIVE_OCR_CFG.get("pause_seconds") or 0.0)
_ARCHIVE_PREFER_CUDA = bool(_ARCHIVE_OCR_CFG.get("prefer_cuda", True))
_ARCHIVE_PREFER_DML = bool(_ARCHIVE_OCR_CFG.get("prefer_dml", True))
_ARCHIVE_SHARED_OCR_CACHE_VERSION = "harp_archive_shared_ocr_v1"

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MONTH_ABBREVIATIONS = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
ARCHIVE_APPENDIX_MARKERS = (
    "facilities designated as hiv treatment hubs",
    "primary hiv care facilities",
    "treatment hubs (outpatient and inpatient care",
    "other facilities providing outpatient hiv care",
)
MONTH_COLUMN_ORDER = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]

GDRIVE_EXPORT = "https://drive.google.com/uc?export=download&id={file_id}"
GDRIVE_CONFIRM = "https://drive.google.com/uc?export=download&confirm=t&id={file_id}"


def _available_ocr_providers() -> list[str]:
    if ort is None:  # pragma: no cover
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def archive_ocr_settings() -> dict[str, Any]:
    providers = _available_ocr_providers()
    return {
        "render_dpi": int(_RENDER_DPI),
        "pdf_requires_ocr_avg_chars": int(_DIRECT_TEXT_CHAR_FLOOR),
        "force_ocr_every_page": bool(_ARCHIVE_FORCE_OCR_EVERY_PAGE),
        "max_pages_per_document": int(_ARCHIVE_MAX_PAGES_PER_DOCUMENT),
        "checkpoint_every_documents": int(_ARCHIVE_CHECKPOINT_EVERY_DOCUMENTS),
        "pause_seconds": float(_ARCHIVE_PAUSE_SECONDS),
        "prefer_cuda": bool(_ARCHIVE_PREFER_CUDA),
        "prefer_dml": bool(_ARCHIVE_PREFER_DML),
        "available_providers": providers,
        "cuda_provider_available": "CUDAExecutionProvider" in providers,
        "dml_provider_available": "DmlExecutionProvider" in providers,
    }


def archive_ocr_cache_settings(
    *,
    force_ocr_every_page: bool | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    effective_force_ocr = bool(_ARCHIVE_FORCE_OCR_EVERY_PAGE) if force_ocr_every_page is None else bool(force_ocr_every_page)
    effective_max_pages = int(_ARCHIVE_MAX_PAGES_PER_DOCUMENT) if max_pages is None else int(max_pages)
    return {
        "render_dpi": int(_RENDER_DPI),
        "pdf_requires_ocr_avg_chars": int(_DIRECT_TEXT_CHAR_FLOOR),
        "force_ocr_every_page": effective_force_ocr,
        "max_pages_per_document": effective_max_pages,
    }


def shared_archive_ocr_dir() -> Path:
    override = str(os.getenv("EPIGRAPH_SHARED_OCR_DIR", "") or "").strip()
    if override:
        return ensure_dir(Path(override))
    return ensure_dir(ROOT_DIR / "artifacts" / "shared" / "harp_archive_ocr")


def shared_archive_ocr_cache_key(*, checksum: str, cache_settings: dict[str, Any]) -> str:
    stable_payload = {
        "cache_version": _ARCHIVE_SHARED_OCR_CACHE_VERSION,
        "checksum": str(checksum or ""),
        "cache_settings": dict(cache_settings or {}),
    }
    return hashlib.sha256(json.dumps(stable_payload, sort_keys=True).encode("utf-8")).hexdigest()


def shared_archive_ocr_artifact_path(cache_key: str) -> Path:
    normalized_key = str(cache_key or "").strip().lower()
    shard = normalized_key[:2] if len(normalized_key) >= 2 else "root"
    return ensure_dir(shared_archive_ocr_dir() / shard) / f"{normalized_key}.json"


def _shared_archive_ocr_base_payload_is_valid(
    payload: dict[str, Any],
    *,
    checksum: str,
    cache_settings: dict[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("checksum") or "") != str(checksum or ""):
        return False
    if dict(payload.get("cache_settings") or {}) != dict(cache_settings or {}):
        return False
    pages = list(payload.get("pages", []) or [])
    return bool(pages) or int(payload.get("page_count") or 0) == 0


def _archive_ocr_reference_payload(
    *,
    local_path: Path,
    checksum: str,
    cache_settings: dict[str, Any],
    shared_cache_key: str,
    shared_artifact_path: Path,
) -> dict[str, Any]:
    return {
        "local_path": str(local_path),
        "checksum": str(checksum or ""),
        "cache_settings": dict(cache_settings or {}),
        "shared_cache_key": str(shared_cache_key or ""),
        "shared_artifact_path": str(shared_artifact_path),
        "generated_at": utc_now_iso(),
    }


def _shared_archive_ocr_base_payload_from_payload(
    payload: dict[str, Any],
    *,
    checksum: str,
    cache_settings: dict[str, Any],
    shared_cache_key: str,
    shared_artifact_path: Path,
) -> dict[str, Any]:
    pages = [dict(page) for page in list(payload.get("pages", []) or [])]
    page_count = int(payload.get("page_count") or len(pages))
    return {
        "checksum": str(checksum or ""),
        "page_count": page_count,
        "pages": pages,
        "runtime": dict(payload.get("runtime") or {}),
        "force_ocr_every_page": bool(payload.get("force_ocr_every_page")),
        "max_pages": int(payload.get("max_pages") or 0),
        "generated_at": str(payload.get("generated_at") or utc_now_iso()),
        "cache_settings": dict(cache_settings or {}),
        "shared_cache_key": str(shared_cache_key or ""),
        "shared_artifact_path": str(shared_artifact_path),
    }


def promote_archive_ocr_artifact_to_shared(artifact_path: Path) -> dict[str, Any] | None:
    payload = read_json(artifact_path, default={})
    if not isinstance(payload, dict):
        return None
    if str(payload.get("shared_artifact_path") or "").strip():
        return {
            "shared_cache_key": str(payload.get("shared_cache_key") or ""),
            "shared_artifact_path": str(payload.get("shared_artifact_path") or ""),
            "cache_settings": dict(payload.get("cache_settings") or {}),
        }
    checksum = str(payload.get("checksum") or "")
    if not checksum:
        return None
    cache_settings = dict(payload.get("cache_settings") or {})
    if not cache_settings:
        cache_settings = archive_ocr_cache_settings(
            force_ocr_every_page=bool(payload.get("force_ocr_every_page")),
            max_pages=int(payload.get("max_pages") or 0),
        )
    shared_cache_key = shared_archive_ocr_cache_key(checksum=checksum, cache_settings=cache_settings)
    shared_artifact_path = shared_archive_ocr_artifact_path(shared_cache_key)
    if not shared_artifact_path.exists():
        shared_payload = _shared_archive_ocr_base_payload_from_payload(
            payload,
            checksum=checksum,
            cache_settings=cache_settings,
            shared_cache_key=shared_cache_key,
            shared_artifact_path=shared_artifact_path,
        )
        write_json(shared_artifact_path, shared_payload)
    return {
        "shared_cache_key": shared_cache_key,
        "shared_artifact_path": str(shared_artifact_path),
        "cache_settings": cache_settings,
    }


def _hydrate_archive_ocr_payload(
    base_payload: dict[str, Any],
    *,
    report_row: dict[str, Any],
    local_path: Path,
    run_artifact_path: Path,
    shared_artifact_path: Path,
    shared_cache_key: str,
    from_cache: bool,
    from_shared_cache: bool,
) -> dict[str, Any]:
    hydrated = dict(base_payload)
    hydrated.update(
        {
            "local_path": str(local_path),
            "source_id": str(report_row.get("source_id") or ""),
            "source_label": str(report_row.get("label") or ""),
            "source_year": int(report_row.get("source_year") or 0),
            "source_url": str(report_row.get("source_url") or ""),
            "artifact_path": str(shared_artifact_path),
            "run_artifact_path": str(run_artifact_path),
            "shared_artifact_path": str(shared_artifact_path),
            "shared_cache_key": str(shared_cache_key or ""),
            "from_cache": bool(from_cache),
            "from_shared_cache": bool(from_shared_cache),
        }
    )
    return hydrated


def _safe_ascii_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return cleaned or "document"


def _normalize_label(label: str) -> str:
    cleaned = str(label or "").replace("&#8211;", "-").replace("&amp;", "&")
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _month_label(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _month_ordinal(month_label: str) -> int | None:
    value = str(month_label or "")
    if len(value) >= 7 and value[:4].isdigit() and value[5:7].isdigit():
        return int(value[:4]) * 12 + int(value[5:7]) - 1
    return None


def _token_to_month(token: str) -> int | None:
    lowered = re.sub(r"[^A-Za-z]", "", str(token or "").strip()).lower()
    if not lowered:
        return None
    lowered = MONTH_ABBREVIATIONS.get(lowered, lowered)
    return MONTH_NAME_TO_NUMBER.get(lowered)


def _range_months(start_month: int, end_month: int) -> list[int]:
    if end_month < start_month:
        return [start_month]
    return list(range(start_month, end_month + 1))


def _parse_period_from_label(year: int, label: str) -> dict[str, Any]:
    normalized = _normalize_label(label)
    normalized = re.sub(rf"^{int(year)}\s+", "", normalized)
    normalized = re.sub(r"\bto\b", "-", normalized, flags=re.I)
    parts = [part.strip() for part in re.split(r"\s*-\s*", normalized) if part.strip()]
    month_numbers = [_token_to_month(part) for part in parts]
    month_numbers = [month for month in month_numbers if month is not None]
    if not month_numbers:
        effective_month = _month_label(year, 12)
        return {
            "period_label": normalized,
            "start_month": effective_month,
            "end_month": effective_month,
            "effective_month": effective_month,
            "temporal_precision": "annual_snapshot",
            "period_months": [effective_month],
        }
    start_month = int(month_numbers[0])
    end_month = int(month_numbers[-1])
    month_span = _range_months(start_month, end_month)
    temporal_precision = "monthly_snapshot"
    if len(month_span) == 3 and start_month in {1, 4, 7, 10}:
        temporal_precision = "quarterly_snapshot"
    elif len(month_span) > 1:
        temporal_precision = "multi_month_snapshot"
    period_months = [_month_label(year, month) for month in month_span]
    effective_month = period_months[-1]
    return {
        "period_label": normalized,
        "start_month": period_months[0],
        "end_month": effective_month,
        "effective_month": effective_month,
        "temporal_precision": temporal_precision,
        "period_months": period_months,
    }


def load_archive_seed_rows(seed_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seed_path.exists():
        return rows
    with seed_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            try:
                year = int(str(raw.get("year") or "").strip())
            except Exception:
                continue
            label = _normalize_label(str(raw.get("label") or ""))
            file_id = str(raw.get("file_id") or "").strip()
            if year <= 0 or not label or not file_id:
                continue
            period = _parse_period_from_label(year, label)
            source_id = f"doh_hiv_sti_{year}_{_safe_ascii_label(label)}"
            rows.append(
                {
                    "source_id": source_id,
                    "label": label,
                    "source_kind": "doh_hiv_sti_archive_pdf",
                    "source_year": year,
                    "file_id": file_id,
                    "source_url": f"https://drive.google.com/file/d/{file_id}/view",
                    "seed_path": str(seed_path),
                    **period,
                }
            )
    return rows


def _download_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def _looks_like_pdf(content: bytes) -> bool:
    return bytes(content[:16]).startswith(b"%PDF")


def download_archive_pdfs(
    *,
    archive_rows: list[dict[str, Any]],
    raw_dir: Path,
    request_timeout_seconds: float = 60.0,
    pause_seconds: float = 0.15,
) -> list[dict[str, Any]]:
    session = _download_session()
    hydrated: list[dict[str, Any]] = []
    for row in archive_rows:
        local_path = raw_dir / f"{row['source_id']}.pdf"
        failure_reason = ""
        if not local_path.exists() or local_path.stat().st_size < 1024:
            success = False
            for template in (GDRIVE_EXPORT, GDRIVE_CONFIRM):
                try:
                    response = session.get(
                        template.format(file_id=row["file_id"]),
                        timeout=request_timeout_seconds,
                        allow_redirects=True,
                    )
                except requests.RequestException as exc:
                    failure_reason = f"{exc.__class__.__name__}: {exc}"
                    continue
                if response.status_code != 200:
                    failure_reason = f"http_status:{response.status_code}"
                    continue
                if not _looks_like_pdf(response.content):
                    failure_reason = "non_pdf_response"
                    continue
                local_path.write_bytes(response.content)
                if local_path.stat().st_size >= 1024:
                    success = True
                    failure_reason = ""
                    break
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            if not success:
                hydrated.append(
                    {
                        **row,
                        "download_status": "failed",
                        "local_path": "",
                        "checksum": "",
                        "downloaded_at": utc_now_iso(),
                        "download_error": failure_reason,
                    }
                )
                continue
        hydrated.append(
            {
                **row,
                "download_status": "downloaded",
                "local_path": str(local_path),
                "checksum": sha256_file(local_path),
                "downloaded_at": utc_now_iso(),
                "download_error": "",
            }
        )
    return hydrated


@lru_cache(maxsize=1)
def _rapidocr_engine() -> Any:
    if RapidOCR is None:  # pragma: no cover
        return None
    settings = archive_ocr_settings()
    kwargs: dict[str, Any] = {}
    if settings["prefer_cuda"] and settings["cuda_provider_available"]:
        kwargs.update({"det_use_cuda": True, "cls_use_cuda": True, "rec_use_cuda": True})
    elif settings["prefer_dml"] and settings["dml_provider_available"]:
        kwargs.update({"det_use_dml": True, "cls_use_dml": True, "rec_use_dml": True})
    for key in ("intra_op_num_threads", "inter_op_num_threads"):
        if key in _ARCHIVE_OCR_CFG and _ARCHIVE_OCR_CFG.get(key) is not None:
            kwargs[key] = int(float(_ARCHIVE_OCR_CFG[key]))
    return RapidOCR(**kwargs)


def _normalize_page_text(text: str) -> str:
    normalized_text = (
        str(text or "")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized_text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _page_requires_ocr(text: str) -> bool:
    return len(_normalize_page_text(text)) < _DIRECT_TEXT_CHAR_FLOOR


def _render_page_png_bytes(page: Any) -> bytes:
    pix = page.get_pixmap(dpi=_RENDER_DPI, alpha=False)
    return bytes(pix.tobytes("png"))


def _ocr_page_text(page: Any) -> str:
    engine = _rapidocr_engine()
    if engine is None:  # pragma: no cover
        return ""
    image_bytes = _render_page_png_bytes(page)
    result, _ = engine(image_bytes)
    if not result:
        return ""
    lines = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lines.append(str(item[1]))
    return _normalize_page_text("\n".join(lines))


def _is_appendix_page(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ARCHIVE_APPENDIX_MARKERS)


def _page_markers(text: str) -> dict[str, bool]:
    lowered = str(text or "").lower()
    return {
        "mentions_table": "table" in lowered,
        "mentions_figure": "figure" in lowered,
        "mentions_care_cascade": "care cascade" in lowered or "continuum of care" in lowered,
        "mentions_art_registry": "aids and art registry" in lowered or "art registry" in lowered,
        "mentions_month_grid": all(month in lowered for month in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")),
    }


def _selected_page_text(direct_text: str, ocr_text: str) -> tuple[str, str]:
    normalized_direct = _normalize_page_text(direct_text)
    normalized_ocr = _normalize_page_text(ocr_text)
    if len(normalized_ocr) > len(normalized_direct):
        return normalized_ocr, "rapidocr"
    if normalized_direct:
        return normalized_direct, "pymupdf_text"
    if normalized_ocr:
        return normalized_ocr, "rapidocr"
    return "", "empty"


def ocr_archive_pdf_payload(
    local_path: Path,
    *,
    force_ocr_every_page: bool | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    if fitz is None or not local_path.exists():  # pragma: no cover
        return {
            "local_path": str(local_path),
            "checksum": "",
            "page_count": 0,
            "pages": [],
            "runtime": archive_ocr_settings(),
        }
    force_ocr = _ARCHIVE_FORCE_OCR_EVERY_PAGE if force_ocr_every_page is None else bool(force_ocr_every_page)
    page_limit_cfg = _ARCHIVE_MAX_PAGES_PER_DOCUMENT if max_pages is None else int(max_pages)
    payload_pages: list[dict[str, Any]] = []
    doc = fitz.open(str(local_path))
    try:
        doc_page_limit = int(doc.page_count) if page_limit_cfg <= 0 else min(int(doc.page_count), int(page_limit_cfg))
        for page_index in range(doc_page_limit):
            page = doc.load_page(page_index)
            direct_text = _normalize_page_text(page.get_text("text"))
            should_ocr = force_ocr or _page_requires_ocr(direct_text)
            ocr_text = _ocr_page_text(page) if should_ocr else ""
            selected_text, selected_source = _selected_page_text(direct_text, ocr_text)
            payload_pages.append(
                {
                    "page_number": int(page_index + 1),
                    "parser_used": selected_source,
                    "text_source": selected_source,
                    "text": selected_text,
                    "direct_text": direct_text,
                    "ocr_text": ocr_text,
                    "ocr_performed": bool(should_ocr),
                    "direct_char_count": int(len(direct_text)),
                    "ocr_char_count": int(len(ocr_text)),
                    "markers": _page_markers(selected_text or ocr_text or direct_text),
                }
            )
    finally:
        doc.close()
    return {
        "local_path": str(local_path),
        "checksum": sha256_file(local_path),
        "page_count": len(payload_pages),
        "pages": payload_pages,
        "runtime": archive_ocr_settings(),
        "force_ocr_every_page": bool(force_ocr),
        "max_pages": int(page_limit_cfg),
        "generated_at": utc_now_iso(),
    }


def materialize_archive_ocr_payload(
    report_row: dict[str, Any],
    *,
    ocr_dir: Path,
    force_refresh: bool = False,
    force_ocr_every_page: bool | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    local_path = Path(str(report_row.get("local_path") or ""))
    checksum = sha256_file(local_path) if local_path.exists() else ""
    artifact_path = ocr_dir / f"{_safe_ascii_label(str(report_row.get('source_id') or local_path.stem))}.json"
    cache_settings = archive_ocr_cache_settings(
        force_ocr_every_page=force_ocr_every_page,
        max_pages=max_pages,
    )
    shared_cache_key = shared_archive_ocr_cache_key(checksum=checksum, cache_settings=cache_settings)
    shared_artifact_path = shared_archive_ocr_artifact_path(shared_cache_key)
    if not force_refresh and artifact_path.exists():
        cached_payload = read_json(artifact_path, default={})
        shared_path = Path(str(cached_payload.get("shared_artifact_path") or shared_artifact_path))
        if str(cached_payload.get("shared_artifact_path") or "").strip() and shared_path.exists():
            shared_payload = read_json(shared_path, default={})
            if _shared_archive_ocr_base_payload_is_valid(shared_payload, checksum=checksum, cache_settings=cache_settings):
                return _hydrate_archive_ocr_payload(
                    dict(shared_payload),
                    report_row=report_row,
                    local_path=local_path,
                    run_artifact_path=artifact_path,
                    shared_artifact_path=shared_path,
                    shared_cache_key=str(cached_payload.get("shared_cache_key") or shared_cache_key),
                    from_cache=True,
                    from_shared_cache=True,
                )
        cached_cache_settings = dict(cached_payload.get("cache_settings") or {})
        if not cached_cache_settings and isinstance(cached_payload.get("runtime"), dict):
            cached_cache_settings = archive_ocr_cache_settings(
                force_ocr_every_page=bool(cached_payload.get("force_ocr_every_page")),
                max_pages=int(cached_payload.get("max_pages") or 0),
            )
        if (
            isinstance(cached_payload, dict)
            and str(cached_payload.get("checksum") or "") == checksum
            and str(cached_payload.get("local_path") or "") == str(local_path)
            and cached_cache_settings == cache_settings
        ):
            promoted = promote_archive_ocr_artifact_to_shared(artifact_path)
            if promoted is not None:
                write_json(
                    artifact_path,
                    _archive_ocr_reference_payload(
                        local_path=local_path,
                        checksum=checksum,
                        cache_settings=cache_settings,
                        shared_cache_key=str(promoted.get("shared_cache_key") or shared_cache_key),
                        shared_artifact_path=Path(str(promoted.get("shared_artifact_path") or shared_artifact_path)),
                    ),
                )
            cached = dict(cached_payload)
            cached["from_cache"] = True
            cached["from_shared_cache"] = False
            return cached
    if not force_refresh and shared_artifact_path.exists():
        shared_payload = read_json(shared_artifact_path, default={})
        if _shared_archive_ocr_base_payload_is_valid(shared_payload, checksum=checksum, cache_settings=cache_settings):
            write_json(
                artifact_path,
                _archive_ocr_reference_payload(
                    local_path=local_path,
                    checksum=checksum,
                    cache_settings=cache_settings,
                    shared_cache_key=shared_cache_key,
                    shared_artifact_path=shared_artifact_path,
                ),
            )
            return _hydrate_archive_ocr_payload(
                dict(shared_payload),
                report_row=report_row,
                local_path=local_path,
                run_artifact_path=artifact_path,
                shared_artifact_path=shared_artifact_path,
                shared_cache_key=shared_cache_key,
                from_cache=True,
                from_shared_cache=True,
            )
    payload = ocr_archive_pdf_payload(
        local_path,
        force_ocr_every_page=force_ocr_every_page,
        max_pages=max_pages,
    )
    shared_payload = dict(payload)
    shared_payload.update(
        {
            "cache_settings": cache_settings,
            "shared_cache_key": shared_cache_key,
            "shared_artifact_path": str(shared_artifact_path),
        }
    )
    write_json(shared_artifact_path, shared_payload)
    write_json(
        artifact_path,
        _archive_ocr_reference_payload(
            local_path=local_path,
            checksum=checksum,
            cache_settings=cache_settings,
            shared_cache_key=shared_cache_key,
            shared_artifact_path=shared_artifact_path,
        ),
    )
    if _ARCHIVE_PAUSE_SECONDS > 0.0:
        time.sleep(_ARCHIVE_PAUSE_SECONDS)
    return _hydrate_archive_ocr_payload(
        shared_payload,
        report_row=report_row,
        local_path=local_path,
        run_artifact_path=artifact_path,
        shared_artifact_path=shared_artifact_path,
        shared_cache_key=shared_cache_key,
        from_cache=False,
        from_shared_cache=False,
    )


def extract_archive_pdf_pages(local_path: Path, *, max_pages: int = 2) -> list[dict[str, Any]]:
    payload = ocr_archive_pdf_payload(local_path, force_ocr_every_page=False, max_pages=max_pages)
    return [dict(page) for page in list(payload.get("pages", []) or []) if str(page.get("text") or "").strip()]


def _to_float(value: str | None) -> float | None:
    token = str(value or "").replace(",", "").strip()
    if not token:
        return None
    try:
        return float(token)
    except Exception:
        return None


def _capture_first_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if not match:
            continue
        value = _to_float(match.group(1))
        if value is not None:
            return value
    return None


def _make_metric_row(
    *,
    report_row: dict[str, Any],
    metric_name: str,
    value: float,
    unit: str,
    measurement_class: str,
    extraction_method: str,
    page_number: int | None = None,
    series_kind: str | None = None,
    time_label: str | None = None,
) -> dict[str, Any]:
    return {
        "year": int(report_row["source_year"]),
        "time": str(time_label or report_row["effective_month"]),
        "metric_name": metric_name,
        "value": float(value),
        "unit": unit,
        "source_id": report_row["source_id"],
        "source_label": report_row["label"],
        "page_number": page_number,
        "measurement_class": measurement_class,
        "series_kind": str(series_kind or report_row["temporal_precision"]),
        "temporal_precision": str(report_row["temporal_precision"]),
        "period_start": str(report_row["start_month"]),
        "period_end": str(report_row["end_month"]),
        "geo": "Philippines",
        "region": "national",
        "province": "Philippines",
        "evidence_confidence": 0.92,
        "source_url": str(report_row.get("source_url") or ""),
        "source_note": f"DOH HIV/STI archive report {report_row['label']}",
        "source_quality_tier": "official_doh_archive",
        "extraction_method": extraction_method,
    }


def extract_continuum_metric_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined = "\n".join(str(page.get("text") or "") for page in page_texts)
    lowered = joined.lower()
    if "95-95-95" not in lowered and "continuum of care" not in lowered and "care cascade" not in lowered:
        return []
    estimated = _capture_first_number(
        joined,
        [
            r"there will be\s*(\d{1,3}(?:,\d{3})+)\s*estimated people living with hiv",
            r"estimated(?: number of)? people living with hiv[^\d]{0,120}(\d{1,3}(?:,\d{3})+)",
            r"estimated plhiv[^\d]{0,80}(\d{1,3}(?:,\d{3})+)",
        ],
    )
    diagnosed = _capture_first_number(
        joined,
        [
            r"of the estimated plhiv,\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)",
            r"number of diagnosed plhiv[^\d]{0,80}(\d{1,3}(?:,\d{3})+)",
            r"(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*of the estimated plhiv",
            r"as of\s+[a-z]+\s+\d{4},\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%[^)]*\)\s*plhiv\s+have\s+been\s+diagnosed",
        ],
    )
    on_art = _capture_first_number(
        joined,
        [
            r"further,\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*plhiv\s+are\s+currently\s+on\s+life[-\s]*saving",
            r"further,\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%[^)]*\)\s*plhiv\s+are\s+currently\s+on\s+life[-\s]*saving",
            r"further,\s*(\d{1,3}(?:,\d{3})+)\s*plhiv\s+are\s+currently\s+on\s+life[-\s]*saving",
            r"number of plhiv on art[^\d]{0,80}(\d{1,3}(?:,\d{3})+)",
            r"(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*plhiv\s+are\s+currently\s+on\s+life[-\s]*saving antiretroviral therapy",
            r"(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%[^)]*\)\s*plhiv\s+are\s+currently\s+on\s+life[-\s]*saving antiretroviral therapy",
            r"a total of\s*(\d{1,3}(?:,\d{3})+)\s*people living with hiv\s*\(plhiv\)\s*were presently\s*on art",
        ],
    )
    tested = _capture_first_number(
        joined,
        [
            r"of which,?\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*plhiv\s+have\s+been\s*tested for viral load",
            r"(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*plhiv\s+have\s+been\s*tested for viral load",
            r"number of vl[^\n]*plhiv on art[^\d]{0,80}(\d{1,3}(?:,\d{3})+)",
        ],
    )
    suppressed = _capture_first_number(
        joined,
        [
            r"among those tested for vl,\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*(?:were|are)\s+virally suppressed",
            r"vl,\s*(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)\s*(?:were|are)\s+virally suppressed",
            r"number of vl[^\n]*virally suppressed[^\d]{0,80}(\d{1,3}(?:,\d{3})+)",
        ],
    )
    rows: list[dict[str, Any]] = []
    if estimated is not None:
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name="estimated_plhiv",
                value=estimated,
                unit="count_people",
                measurement_class="model_estimate",
                extraction_method="rapidocr_continuum",
            )
        )
    if diagnosed is not None:
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name="diagnosed_plhiv",
                value=diagnosed,
                unit="count_people",
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_continuum",
            )
        )
    if on_art is not None:
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name="alive_on_art",
                value=on_art,
                unit="count_people",
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_continuum",
            )
        )
    if tested is not None:
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name="tested_for_viral_load",
                value=tested,
                unit="count_people",
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_continuum",
            )
        )
    if suppressed is not None:
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name="virally_suppressed",
                value=suppressed,
                unit="count_people",
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_continuum",
            )
        )
    return rows


def extract_diagnosis_summary_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not page_texts:
        return []
    page_text = str(page_texts[0].get("text") or "")
    metrics = {
        "new_diagnosed_cases_period": _capture_first_number(
            page_text,
            [
                r"there were\s*(\d{1,3}(?:,\d{3})+)\s*confirmed hiv-positive individuals reported to the one hiv",
                r"in\s+[a-z]+\s+\d{4},\s*there were\s*(\d{1,3}(?:,\d{3})+)\s*confirmed hiv-positive individuals reported",
                r"in\s+[a-z]+\s+to\s+[a-z]+\s+\d{4},\s*there were\s*(\d{1,3}(?:,\d{3})+)\s*confirmed hiv-positive individuals reported",
            ],
        ),
        "advanced_hiv_cases_period": _capture_first_number(
            page_text,
            [
                r"with advanced hiv disease[,:\s]+[a-z0-9\-\s]*\s(\d{1,3}(?:,\d{3})+)",
                r"(\d{1,3}(?:,\d{3})+)\s*\(\d{1,3}%\)[^\n]{0,40}had an advanced hiv infection",
            ],
        ),
        "deaths_reported_period": _capture_first_number(
            page_text,
            [
                r"total reported deaths[,:\s]+[a-z0-9\-\s]*\s(\d{1,3}(?:,\d{3})+)",
                r"there were (\d{1,3}(?:,\d{3})*) reported deaths due to any cause",
            ],
        ),
        "average_cases_per_day": _capture_first_number(
            page_text,
            [
                r"average cases per day[,:\s]+[a-z0-9\-\s]*\s(\d{1,3}(?:,\d{3})*)",
                r"daily average of (\d{1,3}(?:,\d{3})*) cases",
            ],
        ),
        "diagnosed_cases_cumulative": _capture_first_number(
            page_text,
            [
                r"total reported cases[,:\s]+jan 1984[a-z0-9\-\s]*\s(\d{1,3}(?:,\d{3})+)",
                r"cumulatively,\s*(\d{1,3}(?:,\d{3})+)\s*confirmed hiv cases",
            ],
        ),
        "new_hiv_positive_pregnant_women": _capture_first_number(
            "\n".join(str(page.get("text") or "") for page in page_texts[:2]),
            [r"there were (\d{1,3}(?:,\d{3})*) hiv-positive women who were pregnant"],
        ),
        "new_hiv_positive_ofw": _capture_first_number(
            "\n".join(str(page.get("text") or "") for page in page_texts[:2]),
            [r"new hiv\+ ofw[^\d]{0,20}(\d{1,3}(?:,\d{3})*)"],
        ),
        "transactional_sex_cases_period": _capture_first_number(
            "\n".join(str(page.get("text") or "") for page in page_texts[:2]),
            [r"in [a-z0-9\-\s,]+,\s*(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*of the newly reported cases engaged in transactional sex"],
        ),
        "youth_cases_15_24_period": _capture_first_number(
            "\n".join(str(page.get("text") or "") for page in page_texts[:2]),
            [
                r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*of the reported cases\s*this month were among the youth aged 15-24",
                r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*were\s*15-24\s*years?\s*old",
                r"\b\d{1,3}%\s*\((\d{1,3}(?:,\d{3})*)\)\s*were\s*15-24\s*years?\s*old",
                r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*were\s*youth aged 15-24",
                r"\b\d{1,3}%\s*\((\d{1,3}(?:,\d{3})*)\)\s*were\s*youth aged 15-24",
                r"Youth\s*15-24yo[^\d]{0,20}(\d{1,3}(?:,\d{3})*)",
                r"Youth\s*15-24(?:\s*years?\s*old)?[^\d]{0,20}(\d{1,3}(?:,\d{3})*)",
            ],
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, value in metrics.items():
        if value is None:
            continue
        unit = "count_people" if metric_name != "average_cases_per_day" else "count_per_day"
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name=metric_name,
                value=value,
                unit=unit,
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_summary",
                page_number=int(page_texts[0].get("page_number") or 1),
            )
        )
    return rows


def extract_art_summary_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined = "\n".join(str(page.get("text") or "") for page in page_texts[:2])
    metrics = {
        "newly_enrolled_to_treatment": _capture_first_number(
            joined,
            [
                r"newly enrolled[^\d]{0,40}(\d{1,3}(?:,\d{3})+)",
                r"in [a-z0-9\-\s,]+,\s*there were (\d{1,3}(?:,\d{3})+)\w* people with hiv who were enrolled to treatment",
                r"there were[\W_]*(\d{1,3}(?:,\d{3})+)\w*[\W_]*people with hiv[\W_]*who were enrolled to treatment",
            ],
        ),
        "alive_on_art": _capture_first_number(
            joined,
            [
                r"plhiv alive on art[^\d]{0,20}(\d{1,3}(?:,\d{3})+)",
                r"a total of\s*(\d{1,3}(?:,\d{3})+)\s*people living with hiv\s*\(plhiv\)\s*were presently on art",
                r"a total of[\W_]*(\d{1,3}(?:,\d{3})+)[\W_]*people living with hiv[\W_]*plhiv[\W_]*were presently[\W_]*on art",
            ],
        ),
        "median_cd4_at_enrollment": _capture_first_number(
            joined,
            [r"median cd4[^\d]{0,40}(\d{1,3}(?:,\d{3})*)"],
        ),
        "art_median_age": _capture_first_number(
            joined,
            [r"median age[^\d]{0,20}(\d{1,3}(?:,\d{3})*)"],
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, value in metrics.items():
        if value is None:
            continue
        unit = "count_people"
        if metric_name == "median_cd4_at_enrollment":
            unit = "cells_per_mm3"
        elif metric_name == "art_median_age":
            unit = "years"
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name=metric_name,
                value=value,
                unit=unit,
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_art_summary",
            )
        )
    return rows


def extract_prep_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined = "\n".join(str(page.get("text") or "") for page in page_texts[:2])
    if "pre-exposure prophylaxis" not in joined.lower() and "prep" not in joined.lower():
        return []
    metrics = {
        "prep_newly_enrolled_period": _capture_first_number(
            joined,
            [
                r"there were (\d{1,3}(?:,\d{3})+) clients newly enrolled to pre-exposure prophylaxis",
                r"there were (\d{1,3}(?:,\d{3})+) clients newly enrolled to prep",
            ],
        ),
        "prep_cumulative_enrolled": _capture_first_number(
            joined,
            [r"a total of (\d{1,3}(?:,\d{3})+) clients have been enrolled"],
        ),
        "prep_returned_for_refill": _capture_first_number(
            joined,
            [r"over half.*?\((\d{1,3}(?:,\d{3})+)\)\s*returned for a prep refill in \d{4}"],
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, value in metrics.items():
        if value is None:
            continue
        rows.append(
            _make_metric_row(
                report_row=report_row,
                metric_name=metric_name,
                value=value,
                unit="count_people",
                measurement_class="program_observed_harp",
                extraction_method="rapidocr_prep",
            )
        )
    return rows


def extract_monthly_case_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _rows_from_text(text: str, *, extraction_method: str) -> list[dict[str, Any]]:
        parsed_rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line).strip()
            match = re.match(r"^(20\d{2})\s+(.+)$", normalized)
            if not match:
                continue
            row_year = int(match.group(1))
            values = re.findall(r"\d{1,4}(?:,\d{3})*", match.group(2))
            if len(values) < 12:
                continue
            for month_index, raw_value in enumerate(values[:12], start=1):
                value = _to_float(raw_value)
                if value is None:
                    continue
                parsed_rows.append(
                    _make_metric_row(
                        report_row=report_row,
                        metric_name="new_diagnosed_cases_monthly",
                        value=value,
                        unit="count_people",
                        measurement_class="program_observed_harp",
                        extraction_method=extraction_method,
                        series_kind="monthly_snapshot",
                        time_label=_month_label(row_year, month_index),
                    )
                )
        return parsed_rows

    joined = "\n".join(str(page.get("text") or "") for page in page_texts[:2])
    rows: list[dict[str, Any]] = []
    rows.extend(_rows_from_text(joined, extraction_method="pdf_text_monthly_case_table"))
    if rows:
        return rows
    local_path = Path(str(report_row.get("local_path") or ""))
    if fitz is None or RapidOCR is None or not local_path.exists():
        return rows
    doc = fitz.open(str(local_path))
    try:
        if doc.page_count < 2:
            return rows
        ocr_text = _ocr_page_text(doc.load_page(1))
    finally:
        doc.close()
    rows.extend(_rows_from_text(ocr_text, extraction_method="rapidocr_monthly_case_table"))
    return rows


def extract_archive_metric_rows(report_row: dict[str, Any], page_texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(extract_continuum_metric_rows(report_row, page_texts))
    rows.extend(extract_diagnosis_summary_rows(report_row, page_texts))
    rows.extend(extract_art_summary_rows(report_row, page_texts))
    rows.extend(extract_prep_rows(report_row, page_texts))
    rows.extend(extract_monthly_case_rows(report_row, page_texts))
    return rows


def _estimated_plhiv_points(metric_rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return sorted(
        [
            (_month_ordinal(str(row.get("time") or "")), float(row.get("value") or 0.0))
            for row in metric_rows
            if str(row.get("metric_name") or "") == "estimated_plhiv"
            and _month_ordinal(str(row.get("time") or "")) is not None
        ],
        key=lambda item: int(item[0] or 0),
    )


def _interpolate_estimated_plhiv(estimate_points: list[tuple[int, float]], ordinals: np.ndarray) -> np.ndarray:
    if not estimate_points or ordinals.size == 0:
        return np.zeros_like(ordinals, dtype=np.float32)
    x = np.asarray([int(item[0]) for item in estimate_points], dtype=np.float32)
    y = np.asarray([float(item[1]) for item in estimate_points], dtype=np.float32)
    return np.interp(ordinals.astype(np.float32), x, y).astype(np.float32)


def build_harp_program_points(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    point_metrics = {
        "estimated_plhiv",
        "diagnosed_plhiv",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
    }
    rows_by_time: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        metric_name = str(row.get("metric_name") or "")
        time_label = str(row.get("time") or "")
        if metric_name not in point_metrics or not time_label:
            continue
        rows_by_time.setdefault(time_label, {})[metric_name] = row

    estimate_points = _estimated_plhiv_points(metric_rows)

    points: list[dict[str, Any]] = []
    for time_label, grouped in sorted(rows_by_time.items(), key=lambda item: _month_ordinal(item[0]) or -1):
        diagnosed = grouped.get("diagnosed_plhiv")
        on_art = grouped.get("alive_on_art")
        tested = grouped.get("tested_for_viral_load")
        suppressed = grouped.get("virally_suppressed")
        if diagnosed is None or on_art is None or tested is None or suppressed is None:
            continue
        estimate_row = grouped.get("estimated_plhiv")
        if estimate_row is not None:
            estimated_value = float(estimate_row.get("value") or 0.0)
        else:
            ordinal = _month_ordinal(time_label)
            estimate_array = _interpolate_estimated_plhiv(
                estimate_points,
                np.asarray([ordinal], dtype=np.float32) if ordinal is not None else np.zeros((0,), dtype=np.float32),
            )
            estimated_value = float(estimate_array[0]) if estimate_array.size else 0.0
        if estimated_value <= 0.0:
            continue
        source_row = diagnosed
        points.append(
            {
                "label": str(source_row.get("source_label") or f"HARP point {time_label}"),
                "month": time_label,
                "effective_month": time_label,
                "source_month": time_label,
                "temporal_precision": str(source_row.get("series_kind") or source_row.get("temporal_precision") or "monthly_snapshot"),
                "estimated_plhiv": round(float(estimated_value), 6),
                "diagnosed": round(float(diagnosed.get("value") or 0.0), 6),
                "on_art": round(float(on_art.get("value") or 0.0), 6),
                "viral_load_tested": round(float(tested.get("value") or 0.0), 6),
                "suppressed": round(float(suppressed.get("value") or 0.0), 6),
                "source_label": str(source_row.get("source_label") or ""),
                "source_url": str(source_row.get("source_url") or ""),
            }
        )
    return points


def build_diagnosis_flow_points(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    estimate_points = _estimated_plhiv_points(metric_rows)
    advanced_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        if str(row.get("metric_name") or "") != "advanced_hiv_cases_period":
            continue
        start = str(row.get("period_start") or row.get("time") or "")
        end = str(row.get("period_end") or row.get("time") or "")
        if not start or not end:
            continue
        advanced_rows[(start, end)] = dict(row)

    latest_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in {"new_diagnosed_cases_monthly", "new_diagnosed_cases_period"}:
            continue
        time_label = str(row.get("time") or "")
        if not time_label:
            continue
        if metric_name == "new_diagnosed_cases_monthly":
            start = end = time_label
        else:
            start = str(row.get("period_start") or time_label)
            end = str(row.get("period_end") or time_label)
        key = (start, end)
        incumbent = latest_rows.get(key)
        incumbent_sort = (
            _month_ordinal(str(incumbent.get("period_end") or incumbent.get("time") or "")) or -1,
            float(incumbent.get("evidence_confidence") or 0.0),
        ) if incumbent is not None else (-1, -1.0)
        candidate_sort = (
            _month_ordinal(str(row.get("period_end") or row.get("time") or "")) or -1,
            float(row.get("evidence_confidence") or 0.0),
        )
        if incumbent is None or candidate_sort >= incumbent_sort:
            latest_rows[key] = dict(row)

    points: list[dict[str, Any]] = []
    for (period_start, period_end), row in sorted(latest_rows.items(), key=lambda item: _month_ordinal(item[0][1]) or -1):
        start_ordinal = _month_ordinal(period_start)
        end_ordinal = _month_ordinal(period_end)
        if start_ordinal is None or end_ordinal is None or end_ordinal < start_ordinal:
            continue
        period_ordinals = np.arange(start_ordinal, end_ordinal + 1, dtype=np.float32)
        estimated_series = _interpolate_estimated_plhiv(estimate_points, period_ordinals)
        estimated_plhiv = float(np.mean(estimated_series)) if estimated_series.size else 0.0
        if estimated_plhiv <= 0.0:
            continue
        diagnosed_count = float(row.get("value") or 0.0)
        if diagnosed_count <= 0.0:
            continue
        advanced_row = advanced_rows.get((period_start, period_end))
        advanced_count = float(advanced_row.get("value") or 0.0) if advanced_row is not None else None
        point = {
            "label": str(row.get("source_label") or f"Diagnosis flow {period_end}"),
            "month": period_end,
            "effective_month": period_end,
            "source_month": period_end,
            "period_start": period_start,
            "period_end": period_end,
            "temporal_precision": str(row.get("series_kind") or row.get("temporal_precision") or "monthly_snapshot"),
            "series_kind": str(row.get("series_kind") or ""),
            "months_covered": int(end_ordinal - start_ordinal + 1),
            "diagnosed_count": round(diagnosed_count, 6),
            "estimated_plhiv": round(estimated_plhiv, 6),
            "diagnosed_share": round(diagnosed_count / estimated_plhiv, 6),
            "source_id": str(row.get("source_id") or ""),
            "source_label": str(row.get("source_label") or ""),
            "source_url": str(row.get("source_url") or ""),
            "evidence_confidence": float(row.get("evidence_confidence") or 0.0),
        }
        if advanced_count is not None and advanced_count > 0.0:
            point["advanced_hiv_cases"] = round(advanced_count, 6)
            point["advanced_hiv_share"] = round(advanced_count / diagnosed_count, 6)
        points.append(point)
    return points


def archive_seed_path() -> Path:
    return Path(__file__).resolve().with_name("seeds") / "doh_hiv_sti_drive_archive.tsv"
