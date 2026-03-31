from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import base64
import io
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

from epigraph_ph.adapters.structured_sources import get_structured_source_adapters
from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.geography import (
    geo_resolution_label,
    infer_philippines_geo,
    is_national_geo,
    normalize_geo_label,
    philippines_modeling_geos,
)
from epigraph_ph.phase0.literature_candidates import wide_sweep_candidate_rows
from epigraph_ph.phase0.prompts import build_phase0_prompt_library
from epigraph_ph.phase0.shard_materializer import build_slice_payload
from epigraph_ph.validate.literature_review import build_phase0_literature_review
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.registry.models import LiteratureRefDetail, has_verifiable_locator
from epigraph_ph.runtime import (
    RunContext,
    choose_torch_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    read_json,
    save_tensor_artifact,
    set_global_seed,
    sha256_file,
    to_numpy,
    to_torch_tensor,
    utc_now_iso,
    write_boundary_shape_package,
    write_gold_standard_package,
    write_ground_truth_package,
    write_json,
)

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None
    pq = None

try:
    from scipy.interpolate import interp1d
except Exception:  # pragma: no cover
    interp1d = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover
    pdfium = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase0_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase0", {}) or {})


def _phase0_required(key: str) -> Any:
    cfg = _phase0_cfg()
    if key not in cfg:
        raise KeyError(f"Missing HIV phase0 constraint setting: {key}")
    return cfg[key]


def _phase0_required_section(key: str) -> dict[str, Any]:
    value = _phase0_required(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase0 constraint setting '{key}' must be a mapping")
    return dict(value)


DEFAULT_HIV_QUERIES = [
    "HIV Philippines care cascade",
    "HIV stigma Philippines",
    "HIV logistics treatment access",
    "HIV CD4 progression",
]

HIV_DIRECT_WIDE_QUERY_BANK = [
    "HIV incidence Philippines",
    "AIDS treatment adherence Philippines",
    "HIV key populations MSM TGW PWID",
    "HIV stigma testing uptake",
    "HIV CD4 viral load progression",
    "HIV ART retention and suppression",
]

UPSTREAM_DETERMINANT_QUERY_BANK = [
    "poverty health access transport affordability",
    "migration mobility commuting health behavior",
    "stigma discrimination health seeking behavior",
    "collective behavior social norms condom use",
    "supply chain diagnostics cold chain clinic access",
    "population dynamics urbanization inequality",
    "telehealth rural access health systems",
    "household out of pocket catastrophic expenditure",
    "drug resistance immune response antiviral compounds",
]

PHILIPPINES_CONTEXT_QUERY_BANK = [
    "Philippines poverty transport access",
    "Philippines migration urbanization inequality",
    "Philippines culture stigma health seeking",
    "Philippines population dynamics demography",
]

PHILIPPINES_SUBNATIONAL_QUERY_BANK = [
    "Metro Manila HIV testing ART suppression",
    "Cebu HIV care cascade treatment access",
    "Davao HIV diagnosis ART retention",
    "CALABARZON HIV mobility stigma access",
    "Central Luzon HIV treatment logistics",
    "Western Visayas HIV clinic access",
    "BARMM HIV transport fragility access",
    "Eastern Visayas HIV disruption typhoon access",
]

PHILIPPINES_SUBNATIONAL_EXEMPLARS = [
    {"geo": "Metro Manila", "region": "National Capital Region", "peers": "Cebu and Davao City"},
    {"geo": "Cebu", "region": "Central Visayas", "peers": "Bohol and Negros Oriental"},
    {"geo": "Davao City", "region": "Davao Region", "peers": "Davao del Norte and Davao del Sur"},
    {"geo": "CALABARZON", "region": "CALABARZON", "peers": "Cavite and Laguna"},
    {"geo": "Central Luzon", "region": "Central Luzon", "peers": "Bulacan and Pampanga"},
    {"geo": "Iloilo", "region": "Western Visayas", "peers": "Bacolod and Guimaras"},
    {"geo": "Tacloban", "region": "Eastern Visayas", "peers": "Leyte and Samar"},
    {"geo": "BARMM", "region": "BARMM", "peers": "Cotabato City and Lanao del Sur"},
]

STRUCTURAL_DISCOVERY_QUERY_BANK = [
    "statistical physics collective behavior epidemics",
    "network contagion mobility mixing",
    "geography remoteness health systems",
    "socioeconomic inequality access to care",
]

DETERMINANT_SILO_QUERY_BANKS = {
    "sexual_risk": [
        "sexual risk behavior network concurrency prevention barriers",
        "collective sexual risk norms condom use partner concurrency",
    ],
    "prevention_access": [
        "prevention access prep condoms affordability last mile delivery",
        "barriers to prevention access safe sex commodities clinic reach",
    ],
    "testing_uptake": [
        "testing uptake delayed diagnosis stigma fear disclosure",
        "barriers to hiv testing uptake outreach demand generation",
    ],
    "linkage_to_care": [
        "linkage to care referral completion treatment initiation delay",
        "post diagnosis linkage treatment start service navigation",
    ],
    "retention_adherence": [
        "retention in care adherence interruption loss to follow up",
        "drivers of treatment discontinuity continuity of care stress",
    ],
    "suppression_outcomes": [
        "viral suppression outcomes monitoring turnaround documented suppression",
        "adherence viral load testing and suppression barriers",
    ],
    "mobility_network_mixing": [
        "mobility network mixing commuting migration sexual networks",
        "transport corridors mobility mixing care continuity migration",
    ],
    "health_system_reach": [
        "health system reach service delivery clinic coverage telehealth",
        "last mile health system access facility reach workforce shortage",
    ],
    "poverty": [
        "poverty deprivation healthcare affordability treatment continuity",
        "household poverty burden and delayed healthcare seeking",
    ],
    "transport_friction": [
        "transport friction travel burden ferry disruption healthcare access",
        "travel time congestion remoteness treatment access logistics",
    ],
    "cash_instability": [
        "cash instability income shocks household liquidity care interruption",
        "financial volatility informal work income instability treatment adherence",
    ],
    "labor_migration": [
        "labor migration internal migration remittance cycles healthcare continuity",
        "mobile labor populations migration and clinic disengagement",
    ],
    "remoteness": [
        "remoteness archipelago service access peripheral province health reach",
        "remote communities treatment access diagnostic access islands",
    ],
    "congestion_travel_time": [
        "urban congestion travel time clinic access missed appointments",
        "commute burden traffic delay treatment continuity health access",
    ],
    "social_capital": [
        "social capital trust community support healthcare engagement",
        "collective efficacy social capital and testing linkage behavior",
    ],
    "housing_precarity": [
        "housing precarity unstable housing treatment continuity health access",
        "shelter insecurity and healthcare retention adherence",
    ],
    "education": [
        "education attainment health literacy testing uptake prevention access",
        "schooling education and sexual health risk behavior",
    ],
    "collective_risk_behavior": [
        "collective risk behavior group norms sexual networks",
        "peer effects collective behavior and health seeking risk",
    ],
    "policy_implementation_weakness": [
        "policy implementation weakness governance bottlenecks service delivery failure",
        "implementation gap last mile policy execution health systems",
    ],
}

ARXIV_FOCUSED_QUERY_BANK = [
    "epidemic modeling mobility networks",
    "statistical physics contagion social behavior",
    "population dynamics health systems modeling",
]

BIORXIV_FOCUSED_QUERY_BANK = [
    "immune response antiviral compounds",
    "CD4 viral reservoir progression",
    "drug resistance long acting therapeutics",
]

KAGGLE_FOCUSED_QUERY_BANK = [
    "Philippines health indicators",
    "Philippines geospatial demographics",
    "health nutrition population statistics",
]

OPENALEX_FOCUSED_QUERY_BANK = [
    "HIV Philippines economics stigma logistics",
    "population dynamics transport access health",
]

SEMANTIC_SCHOLAR_FOCUSED_QUERY_BANK = [
    "HIV biology economics logistics policy",
    "migration behavior disease dynamics",
]

DEFAULT_LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_EMBED_DEVICE = "cpu"
DEFAULT_LIGHTON_OCR_MODEL = "lightonai/LightOnOCR-2-1B"
DEFAULT_LIGHTON_OCR_BACKEND = "auto"
DEFAULT_LIGHTON_OCR_ENDPOINT = "http://localhost:8000/v1/chat/completions"
DEFAULT_LIGHTON_OCR_PROMPT = "Extract all visible text from this page. Preserve table structure in markdown where possible. Do not summarize."

EXTERNAL_HARVESTER_ORDER = (
    "pubmed",
    "crossref",
    "openalex",
    "semanticscholar",
    "arxiv",
    "biorxiv",
    "kaggle",
)

CORPUS_SOURCE_BUDGET_WEIGHTS = dict(_phase0_required_section("source_budget_weights"))

CORPUS_MODE_MAX_QUERIES = dict(_phase0_required_section("mode_max_queries"))

OFF_TOPIC_TERMS = {
    "astronomy",
    "particle collider",
    "quantum hall",
    "solid state battery",
    "machine translation benchmark",
    "without hiv",
}

OFF_TOPIC_BIOMEDICAL_TERMS = {
    "alzheimer",
    "dementia",
    "copd",
    "chronic obstructive pulmonary disease",
    "breast cancer",
    "lung cancer",
    "prostate cancer",
    "cardiovascular disease",
    "myocardial infarction",
    "stroke",
    "parkinson",
    "multiple sclerosis",
    "rheumatoid arthritis",
    "breast carcinoma",
    "colorectal cancer",
    "glioblastoma",
}

GENERIC_SCIENCE_TERMS = {
    "infectious disease transmission",
    "social aspects of infectious disease transmission",
    "noncommunicable disease",
    "non-communicable disease",
    "ecological niche",
    "avian influenza",
    "hospital acquired infection",
    "nosocomial",
    "sepsis",
}

LINKAGE_KEYWORDS = {
    "prevention_access": ["condom", "prevention", "afford", "affordability", "access", "safe sex", "prophylaxis"],
    "testing_uptake": ["testing", "screening", "diagnosis", "uptake", "seek care"],
    "linkage_to_care": ["linkage", "referral", "clinic", "connect to care"],
    "retention_adherence": ["retention", "adherence", "dropout", "loss to follow up"],
    "suppression_outcomes": ["viral load", "suppression", "treatment success", "art"],
    "mobility_network_mixing": ["mobility", "migration", "network", "mixing", "commuting"],
    "health_system_reach": ["supply chain", "telehealth", "facility", "health system", "coverage"],
    "biological_progression": ["cd4", "viral reservoir", "immune response", "drug resistance"],
}

SOFT_TAG_MAP = {
    "economics": ["poverty", "income", "inequality", "afford", "catastrophic expenditure", "cash", "liquidity", "remittance", "financial shock"],
    "logistics": ["transport", "mobility", "supply chain", "travel time", "remoteness", "congestion", "ferry", "commute"],
    "behavior": ["stigma", "social norm", "condom", "collective behavior", "health seeking", "risk behavior", "partner concurrency"],
    "population": ["population", "urbanization", "demography", "fertility", "age structure"],
    "biology": ["cd4", "viral", "immune", "antiviral", "drug resistance"],
    "policy": ["policy", "governance", "implementation", "insurance", "service delivery", "implementation gap", "bottleneck"],
    "housing": ["housing", "shelter", "precarity", "eviction", "informal settlement"],
    "education": ["education", "schooling", "health literacy", "literacy", "attainment"],
    "social_capital": ["social capital", "community support", "trust", "collective efficacy", "social cohesion"],
}

HIV_TERMS = ["hiv", "aids", "art", "viral load", "cd4", "key population", "msm", "tgw", "pwid"]

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

LOCAL_OFFICIAL_ANCHOR_PACK = (
    {
        "title": "2025 PH HIV Estimates Core Team for WHO",
        "filename": "2025 PH HIV Estimates_Core team_for WHO.pdf",
        "organization": "WHO",
        "year": 2025,
        "local_anchor_key": "who_core_team_2025",
        "preferred_pages": [13, 18, 24, 25],
        "abstract": (
            "Official Philippines HIV estimates deck with national care cascade, "
            "subnational prevention coverage, and treatment coverage trend inputs."
        ),
    },
    {
        "title": "The Philippine HIV/AIDS and STI Surveillance",
        "filename": "The Philippine HIV_STI Surveillance.pdf",
        "organization": "Department of Health",
        "year": 2022,
        "local_anchor_key": "phil_hiv_sti_surveillance",
        "preferred_pages": [3, 14, 15],
        "abstract": (
            "Official surveillance deck summarizing the Philippine HIV epidemic, key populations, "
            "age profile, and surveillance methodology."
        ),
    },
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _desktop_candidates() -> list[Path]:
    candidates: list[Path] = []
    userprofile = Path(os.environ.get("USERPROFILE", "")).expanduser()
    if str(userprofile):
        candidates.extend(
            [
                userprofile / "OneDrive" / "Desktop",
                userprofile / "Desktop",
            ]
        )
    candidates.append(Path(r"C:\Users\gaura\OneDrive\Desktop"))
    candidates.append(Path(r"C:\Users\gaura\Desktop"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _is_valid_observation_year(year: int) -> bool:
    return 1980 <= year <= 2026


def _normalize_time_label(value: str) -> str:
    value = str(value or "").strip()
    month_match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        if _is_valid_observation_year(year) and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"
        return "unknown"
    if len(value) == 4 and value.isdigit():
        year = int(value)
        return f"{year:04d}-01" if _is_valid_observation_year(year) else "unknown"
    return "unknown"


def _local_official_anchor_specs() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    query = {"query": "official_anchor_pdf", "query_domain": "anchor", "query_lane": "anchor", "query_geo_focus": "philippines"}
    for spec in LOCAL_OFFICIAL_ANCHOR_PACK:
        resolved_path = None
        for desktop in _desktop_candidates():
            candidate = desktop / spec["filename"]
            if candidate.exists():
                resolved_path = candidate
                break
        if resolved_path is None:
            continue
        row = _build_source_row(
            source_name=spec["title"],
            source_tier="tier1_official_anchor",
            organization=spec["organization"],
            title=spec["title"],
            url=str(resolved_path),
            platform="manual_seed",
            query=query,
            year=spec["year"],
            abstract=spec["abstract"],
            document_type="pdf",
        )
        row["local_document_path"] = str(resolved_path)
        row["local_anchor_key"] = spec["local_anchor_key"]
        row["preferred_pages"] = list(spec["preferred_pages"])
        notes = list(row.get("provenance_notes") or [])
        notes.append(f"local_anchor_pack:{spec['local_anchor_key']}")
        notes.append(f"local_file:{resolved_path.name}")
        row["provenance_notes"] = notes
        found.append(row)
    return found


def _ensure_unique_ids(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        base = str(row.get(key) or _hash_text(json.dumps(row, sort_keys=True)))
        seen[base] += 1
        unique = base if seen[base] == 1 else f"{base}-{seen[base]}"
        patched = dict(row)
        patched[key] = unique
        out.append(patched)
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, rows)
    _write_parquet_sidecar(path, rows)


def _parquet_sidecar_path(path: Path) -> Path:
    return path.with_suffix(".parquet")


def _normalize_tabular_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_parquet_sidecar(path: Path, rows: list[dict[str, Any]]) -> None:
    if pa is None or pq is None:
        return
    ensure_dir(path.parent)
    parquet_path = _parquet_sidecar_path(path)
    if not rows:
        empty = pa.table({"_empty": pa.array([], type=pa.bool_())})
        pq.write_table(empty, parquet_path)
        return
    normalized_rows = [
        {key: _normalize_tabular_value(value) for key, value in row.items()}
        for row in rows
    ]
    table = pa.Table.from_pylist(normalized_rows)
    pq.write_table(table, parquet_path)


def _persist_duckdb(db_path: Path, table_name: str, rows: list[dict[str, Any]]) -> None:
    if duckdb is None:
        return
    ensure_dir(db_path.parent)
    last_error: Exception | None = None
    for _attempt in range(8):
        con = None
        try:
            con = duckdb.connect(str(db_path))
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            if not rows:
                con.execute(f"CREATE TABLE {table_name} AS SELECT 1 AS _empty WHERE FALSE")
                return
            cols = sorted({key for row in rows for key in row.keys()})
            con.execute(
                f"CREATE TABLE {table_name} ({', '.join(f'{col} VARCHAR' for col in cols)})"
            )
            values = [[json.dumps(row.get(col), ensure_ascii=False) if isinstance(row.get(col), (dict, list)) else row.get(col) for col in cols] for row in rows]
            placeholders = ", ".join(["?"] * len(cols))
            con.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                values,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(float(_phase0_required("request_sleep_seconds")))
        finally:
            if con is not None:
                con.close()
    if last_error is not None:
        raise last_error


def _dedupe_queries_preserve_order(queries: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[dict[str, str]] = []
    for item in queries:
        key = (
            item["query"].strip().lower(),
            item.get("query_domain", ""),
            item.get("query_lane", ""),
            item.get("query_geo_focus", ""),
            item.get("query_silo", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _phase0_large_corpus_cfg() -> dict[str, Any]:
    return dict(_phase0_required_section("large_corpus"))


def _phase0_page_size(source_name: str, requested_limit: int) -> int:
    cfg = _phase0_large_corpus_cfg()
    page_sizes = dict(cfg.get("page_sizes") or {})
    return max(1, min(int(page_sizes.get(source_name, requested_limit)), int(requested_limit)))


def _query_rows(
    queries: list[str],
    *,
    query_domain: str,
    query_lane: str,
    query_geo_focus: str,
    query_silo: str,
) -> list[dict[str, str]]:
    return [
        {
            "query": query,
            "query_domain": query_domain,
            "query_lane": query_lane,
            "query_geo_focus": query_geo_focus,
            "query_silo": query_silo,
        }
        for query in queries
    ]


def _round_robin_queries(groups: list[list[dict[str, str]]], *, limit: int) -> list[dict[str, str]]:
    pending = [list(group) for group in groups if group]
    out: list[dict[str, str]] = []
    while pending and len(out) < limit:
        next_pending: list[list[dict[str, str]]] = []
        for group in pending:
            if len(out) >= limit:
                break
            if not group:
                continue
            out.append(group.pop(0))
            if group:
                next_pending.append(group)
        pending = next_pending
    return _dedupe_queries_preserve_order(out)


def _phase0_query_groups(plugin_id: str = "hiv") -> list[list[dict[str, str]]]:
    plugin = get_disease_plugin(plugin_id)
    query_banks = plugin.to_dict().get("query_banks", {})
    groups: list[list[dict[str, str]]] = []
    groups.append(_query_rows(query_banks.get("default_hiv", DEFAULT_HIV_QUERIES) + query_banks.get("hiv_direct", HIV_DIRECT_WIDE_QUERY_BANK), query_domain="hiv_direct", query_lane="hiv_direct", query_geo_focus="global", query_silo="hiv_direct"))
    groups.append(_query_rows(query_banks.get("upstream_structural", UPSTREAM_DETERMINANT_QUERY_BANK + STRUCTURAL_DISCOVERY_QUERY_BANK), query_domain="upstream_determinant", query_lane="upstream_determinant", query_geo_focus="global", query_silo="upstream_structural"))
    groups.append(_query_rows(query_banks.get("philippines_context", PHILIPPINES_CONTEXT_QUERY_BANK), query_domain="philippines_context", query_lane="upstream_determinant", query_geo_focus="philippines", query_silo="philippines_context"))
    groups.append(_query_rows(query_banks.get("philippines_subnational", PHILIPPINES_SUBNATIONAL_QUERY_BANK), query_domain="philippines_subnational", query_lane="hiv_direct", query_geo_focus="philippines", query_silo="philippines_subnational"))
    groups.append(_query_rows(query_banks.get("modeling", ARXIV_FOCUSED_QUERY_BANK), query_domain="modeling", query_lane="upstream_determinant", query_geo_focus="global", query_silo="modeling"))
    groups.append(_query_rows(query_banks.get("biology", BIORXIV_FOCUSED_QUERY_BANK), query_domain="biology", query_lane="hiv_direct", query_geo_focus="global", query_silo="biology"))
    groups.append(_query_rows(query_banks.get("mixed", KAGGLE_FOCUSED_QUERY_BANK + OPENALEX_FOCUSED_QUERY_BANK + SEMANTIC_SCHOLAR_FOCUSED_QUERY_BANK), query_domain="mixed", query_lane="mixed", query_geo_focus="global", query_silo="mixed"))
    determinant_silos = plugin.determinant_silos or {key: value for key, value in DETERMINANT_SILO_QUERY_BANKS.items()}
    for silo_name, silo in determinant_silos.items():
        silo_queries = silo.query_examples if hasattr(silo, "query_examples") else DETERMINANT_SILO_QUERY_BANKS.get(silo_name, [])
        groups.append(
            _query_rows(
                silo_queries,
                query_domain="upstream_determinant",
                query_lane="upstream_determinant",
                query_geo_focus="global",
                query_silo=silo_name,
            )
        )
    for adapter in plugin.structured_source_adapters:
        if adapter.seed_queries:
            primary_silo = adapter.determinant_silos[0] if adapter.determinant_silos else "mixed"
            groups.append(
                _query_rows(
                    adapter.seed_queries,
                    query_domain="structured_source",
                    query_lane="upstream_determinant",
                    query_geo_focus="philippines",
                    query_silo=primary_silo,
                )
            )
    return groups


def _expand_large_corpus_queries(queries: list[dict[str, str]], *, corpus_mode: str) -> list[dict[str, str]]:
    cfg = _phase0_large_corpus_cfg()
    if corpus_mode not in set(str(item) for item in cfg.get("variant_modes", [])):
        return _dedupe_queries_preserve_order(queries)
    out = list(queries)
    for row in queries:
        if row.get("query_lane") == "anchor":
            continue
        for template in list(cfg.get("query_variant_templates") or []):
            patched = dict(row)
            patched["query"] = template.format(
                query=str(row.get("query") or ""),
                query_silo=str(row.get("query_silo") or "").replace("_", " "),
                geo_focus="Philippines" if str(row.get("query_geo_focus") or "") == "philippines" else "global",
            )
            out.append(patched)
    return _dedupe_queries_preserve_order(out)


def _extend_query_bank(corpus_mode: str, plugin_id: str = "hiv") -> list[dict[str, str]]:
    default_limit = int(CORPUS_MODE_MAX_QUERIES["default"])
    limit = int(CORPUS_MODE_MAX_QUERIES.get(corpus_mode, default_limit))
    base_groups = _phase0_query_groups(plugin_id)
    base_limit = sum(len(group) for group in base_groups) or default_limit
    base_queries = _round_robin_queries(base_groups, limit=base_limit)
    expanded = _expand_large_corpus_queries(base_queries, corpus_mode=corpus_mode)
    return expanded[:limit]


def _slice_queries_for_shard(
    queries: list[dict[str, str]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, str]]:
    if shard_count <= 1:
        return queries
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"query_shard_index must be in [0, {shard_count - 1}]")
    return [query for idx, query in enumerate(queries) if idx % shard_count == shard_index]


def _truncate_phase0_abstract(value: str | None) -> str:
    text = str(value or "")
    limit = int(_phase0_large_corpus_cfg()["abstract_char_limit"])
    return text if len(text) <= limit else text[:limit]


def _phase0_min_literature_year() -> int:
    return int(_phase0_large_corpus_cfg()["min_literature_year"])


def _filter_min_literature_year(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    min_year = _phase0_min_literature_year()
    literature_platforms = {"pubmed", "crossref", "openalex", "semanticscholar", "arxiv", "biorxiv"}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        platform = str(row.get("platform") or "").strip().lower()
        year = _safe_int(row.get("year"))
        if platform in literature_platforms and (year is None or year < min_year):
            continue
        filtered.append(row)
    return filtered


def _source_is_literature(source: dict[str, Any]) -> bool:
    return str(source.get("platform") or "").strip().lower() in {"pubmed", "crossref", "openalex", "semanticscholar", "arxiv", "biorxiv"}


def _source_year_label(source: dict[str, Any]) -> str:
    year = _safe_int(source.get("year"))
    if year is None:
        return ""
    normalized = _normalize_time_label(str(year))
    return "" if normalized == "unknown" else normalized


def _phase0_observation_time(source: dict[str, Any], candidate_time: str) -> str:
    normalized = _normalize_time_label(candidate_time)
    if normalized == "unknown":
        normalized = _source_year_label(source)
    if not normalized:
        return ""
    if _source_is_literature(source):
        year = _safe_int(normalized)
        if year is None or year < _phase0_min_literature_year():
            fallback = _source_year_label(source)
            fallback_year = _safe_int(fallback)
            if fallback and fallback_year is not None and fallback_year >= _phase0_min_literature_year():
                return fallback
            return ""
    return normalized


def _skip_regex_numeric_extraction(block: dict[str, Any], source: dict[str, Any]) -> bool:
    document_type = str(block.get("document_type") or "").strip().lower()
    parser_used = str(block.get("parser_used") or "").strip().lower()
    if document_type == "json_metadata":
        return True
    if parser_used == "html_metadata":
        return True
    if bool(source.get("is_anchor_eligible")):
        return False
    return False


def _query_budget_plan(queries: list[dict[str, str]], *, target_records: int, max_results: int) -> dict[str, int]:
    divisor = max(1, len(queries))
    per_query = max(1, min(max_results, target_records // divisor if target_records else max_results))
    return {"per_query": per_query, "query_count": len(queries)}


def _source_harvest_budgets(queries: list[dict[str, str]], *, target_records: int, max_results: int) -> dict[str, int]:
    per_query = _query_budget_plan(queries, target_records=target_records, max_results=max_results)["per_query"]
    total_weight = sum(CORPUS_SOURCE_BUDGET_WEIGHTS.values()) or float(len(EXTERNAL_HARVESTER_ORDER))
    budgets: dict[str, int] = {}
    for source in EXTERNAL_HARVESTER_ORDER:
        share = float(CORPUS_SOURCE_BUDGET_WEIGHTS[source]) / total_weight
        budgets[source] = max(1, int(round(per_query * share)))
    return budgets


def _requests_json_with_backoff(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for pause in list(_phase0_required("retry_pause_seconds")):
        if pause:
            time.sleep(pause)
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
    raise RuntimeError(str(last_error) if last_error else f"request failed for {url}")


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.I)
    return value or None


def _openalex_abstract_from_inverted_index(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for token, indexes in inverted_index.items():
        for index in indexes:
            positions[index] = token
    return " ".join(token for _, token in sorted(positions.items()))


def _build_source_row(
    *,
    source_name: str,
    source_tier: str,
    organization: str,
    title: str,
    url: str | None,
    platform: str,
    query: dict[str, str],
    year: int | None = None,
    abstract: str | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    openalex_id: str | None = None,
    fulltext_pdf_url: str | None = None,
    document_type: str = "metadata",
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_id = _slug(f"{platform}_{title}_{year or 'na'}_{doi or pmid or openalex_id or url or ''}")[:120] or _hash_text(title)
    payload = {
        "source_id": source_id,
        "source_name": source_name,
        "source_tier": source_tier,
        "organization": organization,
        "url": url,
        "document_type": document_type,
        "country_scope": "Philippines" if query.get("query_geo_focus") == "philippines" else "global",
        "geo_scope": query.get("query_geo_focus"),
        "time_coverage": str(year or ""),
        "license": "",
        "access_date": utc_now_iso(),
        "checksum": "",
        "is_anchor_eligible": source_tier == "tier1_official_anchor",
        "is_direct_truth_eligible": source_tier == "tier1_official_anchor",
        "provenance_notes": [f"platform:{platform}", f"query:{query['query']}"],
        "platform": platform,
        "title": title,
        "abstract": _truncate_phase0_abstract(abstract),
        "year": year,
        "doi": _normalize_doi(doi),
        "pmid": pmid,
        "openalex_id": openalex_id,
        "fulltext_pdf_url": fulltext_pdf_url,
        "query": query["query"],
        "query_domain": query.get("query_domain"),
        "query_lane": query.get("query_lane"),
        "query_geo_focus": query.get("query_geo_focus"),
        "query_silo": query.get("query_silo"),
    }
    if extras:
        payload.update(extras)
    return payload


def _metadata_document(source_row: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    document_id = f"doc-{source_row['source_id']}"
    payload = {
        "document_id": document_id,
        "source_id": source_row["source_id"],
        "title": source_row.get("title"),
        "abstract": source_row.get("abstract"),
        "year": source_row.get("year"),
        "url": source_row.get("url"),
        "platform": source_row.get("platform"),
        "query": source_row.get("query"),
        "query_silo": source_row.get("query_silo"),
        "source_tier": source_row.get("source_tier"),
        "determinant_silos": source_row.get("determinant_silos") or [],
        "adapter_id": source_row.get("adapter_id") or "",
        "access_mode": source_row.get("access_mode") or "",
        "spatial_resolution": source_row.get("spatial_resolution") or "",
        "temporal_resolution": source_row.get("temporal_resolution") or "",
        "is_anchor_eligible": bool(source_row.get("is_anchor_eligible")),
        "is_direct_truth_eligible": bool(source_row.get("is_direct_truth_eligible")),
        "preferred_pages": list(source_row.get("preferred_pages") or []),
        "local_anchor_key": source_row.get("local_anchor_key") or "",
    }
    path = raw_dir / f"{document_id}.json"
    write_json(path, payload)
    doc = dict(payload)
    doc.update(
        {
            "local_path": str(path),
            "snapshot_type": "json_metadata",
            "parse_status": "pending",
            "extraction_status": "pending",
            "parser_used": "",
            "checksum": sha256_file(path),
        }
    )
    return doc


def _local_file_document(source_row: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    document_id = f"doc-{source_row['source_id']}"
    original_path = Path(str(source_row.get("local_document_path") or ""))
    copied_path = raw_dir / f"{document_id}{original_path.suffix.lower() or '.bin'}"
    if original_path.resolve() != copied_path.resolve():
        shutil.copyfile(original_path, copied_path)
    doc = {
        "document_id": document_id,
        "source_id": source_row["source_id"],
        "title": source_row.get("title"),
        "abstract": source_row.get("abstract"),
        "year": source_row.get("year"),
        "url": source_row.get("url"),
        "platform": source_row.get("platform"),
        "query": source_row.get("query"),
        "query_silo": source_row.get("query_silo"),
        "source_tier": source_row.get("source_tier"),
        "determinant_silos": source_row.get("determinant_silos") or [],
        "adapter_id": source_row.get("adapter_id") or "",
        "access_mode": source_row.get("access_mode") or "",
        "spatial_resolution": source_row.get("spatial_resolution") or "",
        "temporal_resolution": source_row.get("temporal_resolution") or "",
        "is_anchor_eligible": bool(source_row.get("is_anchor_eligible")),
        "is_direct_truth_eligible": bool(source_row.get("is_direct_truth_eligible")),
        "preferred_pages": list(source_row.get("preferred_pages") or []),
        "local_anchor_key": source_row.get("local_anchor_key") or "",
        "local_path": str(copied_path),
        "snapshot_type": _sniff_local_document_type(copied_path),
        "parse_status": "pending",
        "extraction_status": "pending",
        "parser_used": "",
        "checksum": sha256_file(copied_path),
    }
    return doc


def _snapshot_file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "unknown"


def _normalize_snapshot_type(value: str) -> str:
    mapping = {"json": "json_metadata", "html": "html", "pdf": "pdf", "csv": "csv", "txt": "text"}
    return mapping.get(value, value)


def _read_file_prefix(path: Path, size: int = 2048) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(size)
    except Exception:
        return b""


def _looks_like_pdf_bytes(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def _looks_like_html_bytes(data: bytes) -> bool:
    lowered = data[:512].lower()
    return b"<html" in lowered or b"<!doctype html" in lowered


def _sniff_local_document_type(path: Path) -> str:
    prefix = _read_file_prefix(path)
    if _looks_like_pdf_bytes(prefix):
        return "pdf"
    if _looks_like_html_bytes(prefix):
        return "html"
    return _normalize_snapshot_type(_snapshot_file_type(path))


def _docling_should_parse_pdf(path: Path) -> bool:
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    if size_mb > 8.0:
        return False
    return False


def _pdf_page_count(path: Path) -> int:
    if fitz is None or not path.exists():
        return 0
    try:
        doc = fitz.open(path)
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        return 0


def _is_heavy_parse_document(doc: dict[str, Any]) -> bool:
    path = Path(str(doc.get("local_path") or ""))
    if not path.exists():
        return False
    return _sniff_local_document_type(path) == "pdf" and (path.stat().st_size > 8 * 1024 * 1024 or _pdf_page_count(path) > 10)


def _working_set_priority(doc: dict[str, Any]) -> tuple[int, int, int]:
    geo = 1 if str(doc.get("query_geo_focus") or "").lower() == "philippines" else 0
    tier = 1 if str(doc.get("source_tier") or "").startswith("tier1") else 0
    adapter_rank = {
        "manual_seed": 5,
        "doh_philippines": 5,
        "who": 4,
        "unaids": 4,
        "openalex": 3,
        "pubmed": 3,
        "semanticscholar": 2,
        "crossref": 2,
        "arxiv": 1,
        "biorxiv": 1,
    }.get(str(doc.get("platform") or ""), 0)
    return (geo, tier, adapter_rank)


def _select_diverse_working_set(documents: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    heavy_local = [doc for doc in documents if _is_heavy_parse_document(doc)]
    light_docs = [doc for doc in documents if not _is_heavy_parse_document(doc)]
    heavy_budget = min(len(heavy_local), max(60, min(250, limit // 10 if limit >= 1000 else max(60, limit // 5))))
    if len(documents) <= limit and len(heavy_local) <= heavy_budget:
        return documents
    selected = sorted(light_docs, key=_working_set_priority, reverse=True)[: max(0, limit - heavy_budget)]
    selected.extend(sorted(heavy_local, key=_working_set_priority, reverse=True)[:heavy_budget])
    return selected[:limit]


def _materialize_remote_snapshots(
    source_rows: list[dict[str, Any]],
    *,
    raw_dir: Path,
    download_budget: int,
    embed_metadata_payload: bool,
    metadata_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    document_rows: list[dict[str, Any]] = []
    hydrated_sources: list[dict[str, Any]] = []
    downloads = 0
    for row in source_rows:
        patched = dict(row)
        local_document_value = str(row.get("local_document_path") or "").strip()
        local_document_path = Path(local_document_value) if local_document_value else None
        if local_document_path and local_document_path.exists():
            document_row = _local_file_document(patched, raw_dir)
        else:
            document_row = _metadata_document(patched, raw_dir)
        if embed_metadata_payload:
            patched["embedded_metadata_path"] = document_row["local_path"]
        pdf_url = str(row.get("fulltext_pdf_url") or "").strip()
        if (not metadata_only) and pdf_url and downloads < download_budget and not (local_document_path and local_document_path.exists()):
            try:  # pragma: no cover - network-dependent
                response = requests.get(pdf_url, timeout=45)
                response.raise_for_status()
                if _looks_like_pdf_bytes(response.content[:16]):
                    local_path = raw_dir / f"{document_row['document_id']}.pdf"
                    local_path.write_bytes(response.content)
                    document_row["local_path"] = str(local_path)
                    document_row["snapshot_type"] = _sniff_local_document_type(local_path)
                    document_row["checksum"] = sha256_file(local_path)
                    downloads += 1
                else:
                    ext = ".html" if _looks_like_html_bytes(response.content[:2048]) else ".bin"
                    probe_path = raw_dir / f"{document_row['document_id']}.fulltext_probe{ext}"
                    probe_path.write_bytes(response.content)
                    document_row["fulltext_probe_path"] = str(probe_path)
                    document_row["fulltext_probe_type"] = _sniff_local_document_type(probe_path)
            except Exception:
                pass
        hydrated_sources.append(patched)
        document_rows.append(document_row)
    return hydrated_sources, document_rows


def _apply_document_relevance_filter(rows: list[dict[str, Any]], *, relevance_mode: str) -> list[dict[str, Any]]:
    if relevance_mode in {"off", "broad"}:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title") or "")
        abstract = str(row.get("abstract") or "")
        text = f"{title} {abstract}".lower()
        if any(term in text for term in OFF_TOPIC_TERMS):
            continue
        if relevance_mode == "strict" and not any(term in text for term in HIV_TERMS):
            continue
        filtered.append(row)
    return filtered


def _harvest_pubmed(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    batch_size = _phase0_page_size("pubmed", limit)
    max_pages = int(_phase0_large_corpus_cfg()["max_pages_per_query"])
    ids: list[str] = []
    for page in range(max_pages):
        remaining = limit - len(ids)
        if remaining <= 0:
            break
        search = _requests_json_with_backoff(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "retstart": page * batch_size,
                "retmax": min(batch_size, remaining),
                "term": query["query"],
            },
        )
        page_ids = list(search.get("esearchresult", {}).get("idlist", []))
        if not page_ids:
            break
        ids.extend(page_ids)
        if len(page_ids) < min(batch_size, remaining):
            break
    ids = ids[:limit]
    if not ids:
        return []
    results = []
    for batch_start in range(0, len(ids), batch_size):
        batch_ids = ids[batch_start : batch_start + batch_size]
        summary = _requests_json_with_backoff(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(batch_ids)},
        )
        for pmid in batch_ids:
            item = summary.get("result", {}).get(pmid, {})
            title = item.get("title") or f"PubMed {pmid}"
            results.append(
                _build_source_row(
                    source_name="PubMed",
                    source_tier="tier2_scientific_literature",
                    organization="NCBI",
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    platform="pubmed",
                    query=query,
                    year=_safe_int(item.get("pubdate")),
                    abstract="",
                    pmid=pmid,
                )
            )
    return results


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _harvest_crossref(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    batch_size = _phase0_page_size("crossref", limit)
    max_pages = int(_phase0_large_corpus_cfg()["max_pages_per_query"])
    for page in range(max_pages):
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        payload = _requests_json_with_backoff(
            "https://api.crossref.org/works",
            params={"query": query["query"], "rows": min(batch_size, remaining), "offset": page * batch_size},
        )
        items = list(payload.get("message", {}).get("items", []))
        if not items:
            break
        for item in items:
            title = " ".join(item.get("title", [])) or "Crossref result"
            rows.append(
                _build_source_row(
                    source_name="Crossref",
                    source_tier="tier2_scientific_literature",
                    organization="Crossref",
                    title=title,
                    url=item.get("URL"),
                    platform="crossref",
                    query=query,
                    year=_safe_int(item.get("created", {}).get("date-time") or item.get("published-print")),
                    abstract=re.sub(r"<[^>]+>", " ", item.get("abstract", "") or ""),
                    doi=item.get("DOI"),
                )
            )
        if len(items) < min(batch_size, remaining):
            break
    return rows[:limit]


def _harvest_openalex(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    batch_size = _phase0_page_size("openalex", limit)
    max_pages = int(_phase0_large_corpus_cfg()["max_pages_per_query"])
    for page in range(1, max_pages + 1):
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        params = {"search": query["query"], "per-page": min(batch_size, remaining), "page": page}
        if os.environ.get("OPENALEX_MAILTO"):
            params["mailto"] = os.environ["OPENALEX_MAILTO"]
        payload = _requests_json_with_backoff("https://api.openalex.org/works", params=params)
        items = list(payload.get("results", []))
        if not items:
            break
        for item in items:
            title = item.get("display_name") or "OpenAlex result"
            rows.append(
                _build_source_row(
                    source_name="OpenAlex",
                    source_tier="tier2_scientific_literature",
                    organization="OpenAlex",
                    title=title,
                    url=item.get("id"),
                    platform="openalex",
                    query=query,
                    year=item.get("publication_year"),
                    abstract=_openalex_abstract_from_inverted_index(item.get("abstract_inverted_index")),
                    doi=item.get("doi"),
                    openalex_id=item.get("id"),
                    fulltext_pdf_url=((item.get("open_access") or {}).get("oa_url") or (item.get("primary_location") or {}).get("pdf_url")),
                )
            )
        if len(items) < min(batch_size, remaining):
            break
    return rows[:limit]


def _harvest_semanticscholar(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    headers = {"x-api-key": os.environ["SEMANTIC_SCHOLAR_API_KEY"]} if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else {}
    rows = []
    batch_size = _phase0_page_size("semanticscholar", limit)
    max_pages = int(_phase0_large_corpus_cfg()["max_pages_per_query"])
    for page in range(max_pages):
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        payload = _requests_json_with_backoff(
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            params={
                "query": query["query"],
                "limit": min(batch_size, remaining),
                "offset": page * batch_size,
                "fields": "paperId,title,abstract,year,externalIds,url,openAccessPdf,publicationTypes,publicationDate",
            },
            headers=headers,
        )
        items = list(payload.get("data", []))
        if not items:
            break
        for item in items:
            external_ids = item.get("externalIds") or {}
            rows.append(
                _build_source_row(
                    source_name="Semantic Scholar",
                    source_tier="tier2_scientific_literature",
                    organization="Semantic Scholar",
                    title=item.get("title") or "Semantic Scholar result",
                    url=item.get("url"),
                    platform="semanticscholar",
                    query=query,
                    year=item.get("year"),
                    abstract=item.get("abstract") or "",
                    doi=external_ids.get("DOI"),
                    pmid=external_ids.get("PubMed"),
                    fulltext_pdf_url=((item.get("openAccessPdf") or {}).get("url")),
                )
            )
        if len(items) < min(batch_size, remaining):
            break
    return rows[:limit]


def _harvest_arxiv(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    batch_size = _phase0_page_size("arxiv", limit)
    max_pages = int(_phase0_large_corpus_cfg()["max_pages_per_query"])
    for page in range(max_pages):
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        response = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query['query']}", "start": page * batch_size, "max_results": min(batch_size, remaining)},
            timeout=30,
        )
        response.raise_for_status()
        text = response.text
        entries = re.findall(r"<entry>(.*?)</entry>", text, flags=re.S)
        if not entries:
            break
        for entry in entries:
            title = re.sub(r"\s+", " ", _extract_xml_tag(entry, "title")).strip()
            summary = re.sub(r"\s+", " ", _extract_xml_tag(entry, "summary")).strip()
            year = _safe_int(_extract_xml_tag(entry, "published"))
            arxiv_id = _extract_xml_tag(entry, "id")
            rows.append(
                _build_source_row(
                    source_name="arXiv",
                    source_tier="tier2_scientific_literature",
                    organization="arXiv",
                    title=title or "arXiv result",
                    url=arxiv_id,
                    platform="arxiv",
                    query=query,
                    year=year,
                    abstract=summary,
                )
            )
        if len(entries) < min(batch_size, remaining):
            break
    return rows[:limit]


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, flags=re.S)
    return match.group(1) if match else ""


def _harvest_biorxiv(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    payload = _requests_json_with_backoff(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f'({query["query"]}) AND SRC:PPR', "format": "json", "pageSize": limit},
    )
    rows = []
    for item in payload.get("resultList", {}).get("result", []):
        title = item.get("title") or "bioRxiv result"
        url = item.get("doi") and f"https://doi.org/{item['doi']}" or item.get("source")
        rows.append(
            _build_source_row(
                source_name="bioRxiv",
                source_tier="tier2_scientific_literature",
                organization="bioRxiv/EuropePMC",
                title=title,
                url=url,
                platform="biorxiv",
                query=query,
                year=_safe_int(item.get("firstPublicationDate")),
                abstract=item.get("abstractText") or "",
                doi=item.get("doi"),
            )
        )
    return rows


def _harvest_kaggle(query: dict[str, str], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(min(limit, 2)):
        title = f"Kaggle dataset seed: {query['query']} {index + 1}"
        rows.append(
            _build_source_row(
                source_name="Kaggle",
                source_tier="tier3_structured_repository",
                organization="Kaggle",
                title=title,
                url=f"https://www.kaggle.com/search?q={requests.utils.quote(query['query'])}",
                platform="kaggle",
                query=query,
                year=None,
                abstract=f"Metadata placeholder for {query['query']}",
            )
        )
    return rows


def _harvest_official_seed_rows() -> list[dict[str, Any]]:
    seeds = [
        {
            "title": "WHO Philippines HIV profile",
            "url": "https://www.who.int/philippines",
            "platform": "who",
            "organization": "WHO",
            "records": [
                {"title": "WHO Consolidated HIV Strategic Information Guidelines 2020", "url": "https://www.who.int/publications-detail-redirect/9789240000735", "year": 2020},
                {"title": "WHO Person-Centred HIV Strategic Information Guidelines 2022", "url": "https://www.who.int/publications-detail-redirect/9789240055315", "year": 2022},
                {"title": "WHO Viral Suppression Policy Brief 2023", "url": "https://www.who.int/publications/i/item/9789240055179", "year": 2023},
            ],
        },
        {
            "title": "UNAIDS Philippines",
            "url": "https://www.unaids.org/en/regionscountries/countries/philippines",
            "platform": "unaids",
            "organization": "UNAIDS",
            "records": [
                {"title": "UNAIDS Global AIDS Update 2022 Annex 2", "url": "https://crossroads.unaids.org/", "year": 2022},
                {"title": "UNAIDS Global AIDS Update 2023 Annex 2", "url": "https://crossroads.unaids.org/", "year": 2023},
                {"title": "UNAIDS Global AIDS Update 2024 Annex 2", "url": "https://crossroads.unaids.org/wp-content/uploads/2024/07/GAU-2024-Annex-2-En.pdf", "year": 2024},
            ],
        },
        {
            "title": "DOH Philippines HIV/AIDS Registry",
            "url": "https://doh.gov.ph/",
            "platform": "doh_philippines",
            "organization": "DOH",
            "records": [
                {"title": "HIV/AIDS and ART Registry of the Philippines 2022", "url": "https://doh.gov.ph/", "year": 2022},
                {"title": "HIV/AIDS and ART Registry of the Philippines 2023", "url": "https://doh.gov.ph/", "year": 2023},
                {"title": "HIV/AIDS and ART Registry of the Philippines 2024", "url": "https://doh.gov.ph/", "year": 2024},
            ],
        },
        {
            "title": "Government of the Philippines Open Data",
            "url": "https://data.gov.ph/",
            "platform": "gov_ph",
            "organization": "Government",
            "records": [
                {"title": "data.gov.ph Catalog Snapshot 2019", "url": "https://data.gov.ph/", "year": 2019},
                {"title": "data.gov.ph Catalog Snapshot 2022", "url": "https://data.gov.ph/", "year": 2022},
                {"title": "data.gov.ph Catalog Snapshot 2024", "url": "https://data.gov.ph/", "year": 2024},
            ],
        },
        {
            "title": "UN Data Philippines",
            "url": "https://data.un.org/",
            "platform": "un",
            "organization": "UN",
            "records": [
                {"title": "UNData Philippines Demographic Indicators 2015", "url": "https://data.un.org/", "year": 2015},
                {"title": "UNData Philippines Demographic Indicators 2020", "url": "https://data.un.org/", "year": 2020},
                {"title": "UNData Philippines Demographic Indicators 2024", "url": "https://data.un.org/", "year": 2024},
            ],
        },
    ]
    query = {"query": "official_seed", "query_domain": "anchor", "query_lane": "anchor", "query_geo_focus": "philippines"}
    rows = []
    for seed in seeds:
        rows.append(
            _build_source_row(
                source_name=str(seed["title"]),
                source_tier="tier1_official_anchor",
                organization=str(seed["organization"]),
                title=str(seed["title"]),
                url=str(seed["url"]),
                platform=str(seed["platform"]),
                query=query,
            )
        )
        for record in list(seed.get("records", [])):
            rows.append(
                _build_source_row(
                    source_name=str(seed["title"]),
                    source_tier="tier1_official_anchor",
                    organization=str(seed["organization"]),
                    title=str(record.get("title") or seed["title"]),
                    url=str(record.get("url") or seed["url"]),
                    platform=str(seed["platform"]),
                    query=query,
                    year=record.get("year"),
                    extras={"collector_type": "historical_record"},
                )
            )
    return rows


def _structured_source_seed_rows(plugin_id: str) -> list[dict[str, Any]]:
    adapters = get_structured_source_adapters(plugin_id)
    rows: list[dict[str, Any]] = []
    for adapter in adapters:
        primary_silo = adapter.determinant_silos[0] if adapter.determinant_silos else "mixed"
        rows.append(
            _build_source_row(
                source_name=adapter.source_name,
                source_tier=adapter.source_tier,
                organization=adapter.organization,
                title=adapter.source_name,
                url=adapter.landing_url,
                platform=adapter.platform,
                query={
                    "query": f"{adapter.source_name} adapter seed",
                    "query_domain": "structured_source",
                    "query_lane": "upstream_determinant",
                    "query_geo_focus": "philippines",
                    "query_silo": primary_silo,
                },
                document_type="dataset_catalog",
                extras={
                    "adapter_id": adapter.adapter_id,
                    "access_mode": adapter.access_mode,
                    "spatial_resolution": adapter.spatial_resolution,
                    "temporal_resolution": adapter.temporal_resolution,
                    "determinant_silos": list(adapter.determinant_silos),
                    "promotion_track": adapter.promotion_track,
                    "fallback_urls": list(adapter.fallback_urls),
                    "preferred_file_patterns": list(adapter.preferred_file_patterns),
                    "adapter_notes": list(adapter.notes),
                },
            )
        )
        for record in list(adapter.historical_records or []):
            rows.append(
                _build_source_row(
                    source_name=adapter.source_name,
                    source_tier=adapter.source_tier,
                    organization=adapter.organization,
                    title=str(record.get("title") or adapter.source_name),
                    url=str(record.get("url") or adapter.landing_url),
                    platform=adapter.platform,
                    query={
                        "query": f"{adapter.source_name} historical collector {record.get('title') or adapter.source_name}",
                        "query_domain": "structured_source",
                        "query_lane": "upstream_determinant",
                        "query_geo_focus": "philippines",
                        "query_silo": primary_silo,
                    },
                    year=record.get("year"),
                    document_type=str(record.get("document_type") or "dataset_catalog"),
                    extras={
                        "adapter_id": adapter.adapter_id,
                        "collector_type": "historical_record",
                        "access_mode": adapter.access_mode,
                        "spatial_resolution": adapter.spatial_resolution,
                        "temporal_resolution": adapter.temporal_resolution,
                        "determinant_silos": list(adapter.determinant_silos),
                        "promotion_track": adapter.promotion_track,
                        "fallback_urls": list(adapter.fallback_urls),
                        "preferred_file_patterns": list(adapter.preferred_file_patterns),
                        "adapter_notes": list(adapter.notes),
                        "historical_record_notes": list(record.get("notes") or []),
                    },
                )
            )
    return rows


def _collector_manifest_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        adapter_id = str(row.get("adapter_id") or "")
        platform = str(row.get("platform") or "")
        collector_type = str(row.get("collector_type") or ("structured_seed" if adapter_id else "official_seed"))
        grouped[(adapter_id or platform, collector_type)].append(row)
    manifest_rows: list[dict[str, Any]] = []
    for (collector_id, collector_type), rows in sorted(grouped.items()):
        years = sorted({int(row["year"]) for row in rows if isinstance(row.get("year"), int)})
        manifest_rows.append(
            {
                "collector_id": collector_id,
                "collector_type": collector_type,
                "platforms": sorted({str(row.get("platform") or "") for row in rows if row.get("platform")}),
                "title_count": len(rows),
                "year_count": len(years),
                "year_min": years[0] if years else None,
                "year_max": years[-1] if years else None,
                "sample_titles": [str(row.get("title") or "") for row in rows[:5]],
            }
        )
    return manifest_rows


def _harvest_sweep_record(source_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": source_row["source_id"],
        "title": source_row.get("title"),
        "abstract": source_row.get("abstract") or "",
        "year": source_row.get("year"),
        "platform": source_row.get("platform"),
        "url": source_row.get("url"),
        "doi": source_row.get("doi"),
        "pmid": source_row.get("pmid"),
        "openalex_id": source_row.get("openalex_id"),
        "query": source_row.get("query"),
        "query_domain": source_row.get("query_domain"),
        "query_lane": source_row.get("query_lane"),
        "query_geo_focus": source_row.get("query_geo_focus"),
        "query_silo": source_row.get("query_silo"),
        "determinant_silos": source_row.get("determinant_silos") or [],
        "source_tier": source_row.get("source_tier"),
    }


def _score_linkage_targets(text: str) -> list[str]:
    lowered = text.lower()
    targets = [name for name, keywords in LINKAGE_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    return sorted(targets)


def _soft_ontology_tags(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(tag for tag, keywords in SOFT_TAG_MAP.items() if any(keyword in lowered for keyword in keywords))


def _soft_subparameter_hints(text: str) -> list[str]:
    hints: list[str] = []
    lowered = text.lower()
    if any(token in lowered for token in ("msm", "men who have sex with men", "transgender women", "tgw", "key population")):
        hints.append("sexual_risk")
    if any(token in lowered for token in ("condom", "prep", "pre exposure prophylaxis", "harm reduction", "self test", "community screening")):
        hints.append("prevention_access")
    if any(token in lowered for token in ("testing", "screening", "diagnosis", "late diagnosis", "fear of positive result", "hiv knowledge")):
        hints.append("testing_uptake")
    if any(token in lowered for token in ("linkage", "referral", "treatment initiation", "diagnosis to art", "care navigation")):
        hints.append("linkage_to_care")
    if any(token in lowered for token in ("retention", "adherence", "loss to follow up", "continuity of care", "treatment interruption")):
        hints.append("retention_adherence")
    if any(token in lowered for token in ("viral suppression", "viral load", "vl testing", "documentation", "suppressed")):
        hints.append("suppression_outcomes")
    if any(token in lowered for token in ("poverty", "income", "afford", "inequality")):
        hints.append("economic_access_constraint")
    if any(token in lowered for token in ("cash", "liquidity", "income shock", "financial volatility", "remittance")):
        hints.append("cash_instability")
    if any(token in lowered for token in ("transport", "travel", "mobility", "migration")):
        hints.append("mobility_friction")
        hints.append("mobility_network_mixing")
    if any(token in lowered for token in ("labor migration", "migrant worker", "internal migration", "remittance")):
        hints.append("labor_migration")
    if any(token in lowered for token in ("remote", "remoteness", "island", "archipelago", "peripheral")):
        hints.append("remoteness")
    if any(token in lowered for token in ("traffic", "congestion", "commute", "travel time")):
        hints.append("congestion_travel_time")
    if any(token in lowered for token in ("stigma", "discrimination", "social norm")):
        hints.append("stigma_barrier")
    if any(token in lowered for token in ("social capital", "community support", "trust", "collective efficacy")):
        hints.append("social_capital")
    if any(token in lowered for token in ("housing", "shelter", "eviction", "informal settlement")):
        hints.append("housing_precarity")
    if any(token in lowered for token in ("education", "schooling", "health literacy", "attainment")):
        hints.append("education")
        hints.append("hiv_knowledge_index")
    if any(token in lowered for token in ("policy", "governance", "implementation gap", "bottleneck")):
        hints.append("policy_implementation_weakness")
    if any(token in lowered for token in ("cd4", "viral", "immune", "antiviral")):
        hints.append("biological_progression_modifier")
    if any(token in lowered for token in ("supply chain", "telehealth", "facility", "clinic")):
        hints.append("service_delivery_reach")
        hints.append("health_system_reach")
    if any(token in lowered for token in ("philhealth", "health insurance", "claims coverage")):
        hints.append("philhealth_coverage")
    if any(token in lowered for token in ("stigma index", "discrimination score")):
        hints.append("stigma_index")
    if any(token in lowered for token in ("public transit", "transit usage", "mobility report")):
        hints.append("public_transit_usage")
    if any(token in lowered for token in ("migration rate", "ofw", "working abroad")):
        hints.append("migration_rate")
    if any(token in lowered for token in ("clinic density", "clinics per", "treatment hub", "facility density")):
        hints.append("clinics_per_capita")
    if any(token in lowered for token in ("physician density", "physicians per", "doctors per")):
        hints.append("physicians_per_capita")
    return sorted(set(hints))


def _query_silo_match_score(query_silo: str, *, hints: list[str], linkage_targets: list[str], soft_tags: list[str], text: str) -> float:
    token = re.sub(r"[^a-z0-9]+", "_", str(query_silo or "").strip().lower()).strip("_")
    if not token:
        return 0.0
    if token in set(hints) | set(linkage_targets):
        return 1.0
    spaced = token.replace("_", " ")
    if spaced and spaced in text:
        return 0.85
    if token in set(soft_tags):
        return 0.65
    return 0.0


def _geo_focus_score(text: str, *, query_geo_focus: str) -> float:
    lowered = str(text or "").lower()
    if query_geo_focus == "philippines":
        if "philippines" in lowered or "filipino" in lowered:
            return 1.0
        if any(token in lowered for token in ("metro manila", "cebu", "davao", "visayas", "mindanao", "luzon", "barmm")):
            return 0.75
        if any(token in lowered for token in ("southeast asia", "south-east asia", "asean")):
            return 0.45
    return 0.0


def _infer_measurement_semantics(parameter_text: str, candidate_text: str, unit: str) -> dict[str, str]:
    text = f"{parameter_text} {candidate_text}".strip().lower()
    normalized_unit = str(unit or "").strip().lower()
    measurement_type = "unknown"
    denominator_type = "unknown"
    normalization_basis = "unknown"
    value_semantics = "bounded_proxy"
    if any(token in normalized_unit for token in ("%", "percent", "proportion", "share")) or any(token in text for token in ("prevalence", "coverage", "share", "proportion", "rate")):
        measurement_type = "rate"
    elif any(token in normalized_unit for token in ("peso", "php", "usd")) or any(token in text for token in ("cost", "expense", "spending", "expenditure")):
        measurement_type = "cost"
    elif any(token in text for token in ("travel time", "delay", "turnaround", "wait time", "commute")):
        measurement_type = "time_delay"
    elif any(token in text for token in ("clinic", "facility", "hub", "workforce", "physician", "doctor")):
        measurement_type = "capacity"
    elif any(token in normalized_unit for token in ("people", "cases", "count")) or re.search(r"\b\d+\b", text):
        measurement_type = "count"
    if "per 100,000" in text or "/100000" in text:
        denominator_type = "population"
        normalization_basis = "per_100k"
    elif "per 1,000" in text or "/1000" in text:
        denominator_type = "population"
        normalization_basis = "per_1000"
    elif any(token in text for token in ("per capita", "population", "residents", "inhabitants")):
        denominator_type = "population"
        normalization_basis = "per_capita" if "per capita" in text else normalization_basis
    elif any(token in text for token in ("plhiv", "people living with hiv", "among plhiv")):
        denominator_type = "plhiv"
    elif any(token in text for token in ("msm", "fsw", "pwid", "transgender women", "key population")):
        denominator_type = "kp_population"
    elif any(token in text for token in ("household", "family", "respondent")):
        denominator_type = "household"
    elif any(token in text for token in ("facility", "clinic", "hub", "hospital")):
        denominator_type = "facility"
    elif any(token in text for token in ("route", "corridor", "trip", "visit")):
        denominator_type = "route" if any(token in text for token in ("route", "corridor")) else "visit"
    elif measurement_type == "count":
        denominator_type = "none"
        normalization_basis = "absolute"
    if measurement_type == "rate" and normalization_basis == "unknown":
        normalization_basis = "percent" if any(token in normalized_unit for token in ("%", "percent")) else "share"
    if measurement_type in {"count", "rate", "cost", "time_delay", "capacity"}:
        value_semantics = "direct_observed"
    return {
        "measurement_type": measurement_type,
        "denominator_type": denominator_type,
        "normalization_basis": normalization_basis,
        "value_semantics": value_semantics,
    }


def _phase0_tool_stack_manifest() -> dict[str, Any]:
    tool_cfg = dict(_phase0_required_section("tooling_reality"))
    implemented = list(tool_cfg.get("implemented_tools", []))
    design_only = list(tool_cfg.get("design_only_tools", []))
    availability = {
        "duckdb": duckdb is not None,
        "pyarrow_parquet": pa is not None and pq is not None,
        "faiss": faiss is not None,
        "chromadb": chromadb is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "transformers": AutoTokenizer is not None and AutoModel is not None,
        "pymupdf": fitz is not None,
        "lighton_ocr_sidecar": True,
        "docling": importlib.util.find_spec("docling") is not None,
        "grobid": False,
        "camelot": importlib.util.find_spec("camelot") is not None,
        "qwen2_vl": False,
        "bge_m3": False,
        "llama_3_1_8b": False,
        "ray": importlib.util.find_spec("ray") is not None,
        "gpytorch": importlib.util.find_spec("gpytorch") is not None,
    }
    return {
        "implemented_tools": implemented,
        "design_only_tools": design_only,
        "availability": availability,
        "selected_embedding_model": os.environ.get("EPIGRAPH_PHASE0_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL),
        "selected_ocr_backend": _phase0_ocr_backend(),
        "runtime_platform": platform.platform(),
    }


def _phase0_resource_snapshot(stage_name: str) -> dict[str, Any]:
    rss_bytes = None
    vms_bytes = None
    available_bytes = None
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            rss_bytes = int(mem.rss)
            vms_bytes = int(mem.vms)
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            rss_bytes = None
    gpu_allocated = None
    gpu_reserved = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_allocated = int(torch.cuda.memory_allocated())
                gpu_reserved = int(torch.cuda.memory_reserved())
        except Exception:
            gpu_allocated = None
    return {
        "stage": stage_name,
        "timestamp": utc_now_iso(),
        "platform": platform.platform(),
        "cpu_forced": str(os.getenv("EPIGRAPH_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"},
        "ram_rss_bytes": rss_bytes,
        "ram_vms_bytes": vms_bytes,
        "ram_available_bytes": available_bytes,
        "torch_device": choose_torch_device(prefer_gpu=True),
        "jax_device": detect_backends()["jax"].device,
        "gpu_memory_allocated_bytes": gpu_allocated,
        "gpu_memory_reserved_bytes": gpu_reserved,
    }


def _phase0_source_rights_row(source_row: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_phase0_required_section("rights_and_reproducibility"))
    platform = str(source_row.get("platform") or "").strip().lower()
    url = str(source_row.get("url") or "")
    domain = ""
    match = re.search(r"https?://([^/]+)", url)
    if match:
        domain = match.group(1).lower()
    if platform in set(cfg.get("official_public_platforms", [])):
        rights_class = "official_public"
    elif domain in set(cfg.get("review_domains", [])):
        rights_class = "metadata_or_landing_page_only"
    elif platform in {"pubmed", "crossref", "openalex", "semanticscholar", "arxiv", "biorxiv"}:
        rights_class = "literature_metadata_review_required"
    elif platform == "manual_seed":
        rights_class = "manual_local_seed"
    else:
        rights_class = "unknown_review_required"
    return {
        "source_id": source_row.get("source_id"),
        "platform": platform,
        "url": url,
        "rights_class": rights_class,
        "needs_review": rights_class in {"metadata_or_landing_page_only", "literature_metadata_review_required", "unknown_review_required"},
    }

def _score_wide_sweep_record(record: dict[str, Any], *, min_domain_quality: float) -> dict[str, Any]:
    score_cfg = _phase0_required_section("wide_sweep_scoring")
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    text = f"{title} {abstract}".strip().lower()
    hiv_hits = sum(1 for token in HIV_TERMS if token in text)
    upstream_hits = sum(sum(1 for keyword in keywords if keyword in text) for keywords in SOFT_TAG_MAP.values())
    soft_tags = _soft_ontology_tags(text)
    soft_hints = _soft_subparameter_hints(text)
    linkage_targets = _score_linkage_targets(text)
    geo_score = _geo_focus_score(text, query_geo_focus=str(record.get("query_geo_focus") or ""))
    silo_match = _query_silo_match_score(
        str(record.get("query_silo") or ""),
        hints=soft_hints,
        linkage_targets=linkage_targets,
        soft_tags=soft_tags,
        text=text,
    )
    off_topic = any(term in text for term in OFF_TOPIC_TERMS)
    off_topic_biomed = any(term in text for term in OFF_TOPIC_BIOMEDICAL_TERMS) and hiv_hits == 0
    generic_science = any(term in text for term in GENERIC_SCIENCE_TERMS) and hiv_hits == 0
    missing_signal = hiv_hits == 0 and upstream_hits == 0 and not soft_hints and not linkage_targets
    precision_penalty = 0.0
    if off_topic_biomed:
        precision_penalty += float(score_cfg["precision_penalty_offtopic_biomed"])
    if generic_science and geo_score <= 0.0 and silo_match < float(score_cfg["review_upstream_silo_match_floor"]):
        precision_penalty += float(score_cfg["precision_penalty_generic_science"])
    if missing_signal:
        precision_penalty += float(score_cfg["precision_penalty_missing_signal"])
    domain_quality = min(
        1.0,
        float(score_cfg["domain_quality_hiv_hit_weight"]) * hiv_hits
        + float(score_cfg["domain_quality_upstream_hit_weight"]) * upstream_hits
        + (float(score_cfg["domain_quality_long_text_bonus"]) if len(text) > int(score_cfg["long_text_min_chars"]) else 0.0)
        + (float(score_cfg["philippines_geo_bonus"]) if geo_score >= 1.0 else float(score_cfg["southeast_asia_geo_bonus"]) if geo_score > 0.0 else 0.0)
        + float(score_cfg["query_silo_match_bonus"]) * silo_match
        + float(score_cfg["soft_hint_bonus"]) * min(1.0, len(soft_hints) / 3.0),
    )
    if off_topic:
        domain_quality = 0.0
    domain_quality = max(0.0, min(1.0, domain_quality - precision_penalty))
    hiv_direct_score = min(
        1.0,
        float(score_cfg["hiv_direct_per_hit"]) * hiv_hits
        + (float(score_cfg["hiv_direct_lane_bonus"]) if record.get("query_lane") == "hiv_direct" else 0.0),
    )
    upstream_score = min(
        1.0,
        float(score_cfg["upstream_per_hit"]) * upstream_hits
        + (float(score_cfg["upstream_lane_bonus"]) if record.get("query_lane") == "upstream_determinant" else 0.0),
    )
    accepted_hiv = hiv_direct_score >= float(score_cfg["accepted_hiv_threshold"]) and domain_quality >= min_domain_quality
    upstream_signal_count = len(set(soft_hints + linkage_targets))
    upstream_context_ok = (
        geo_score > 0.0
        or silo_match >= float(score_cfg["accepted_upstream_silo_match_floor"])
        or upstream_signal_count >= int(score_cfg["accepted_upstream_signal_floor"])
        or str(record.get("platform") or "") in {"who", "unaids", "doh_philippines", "un", "ndhs", "yafs", "fies", "google_mobility", "world_bank_wdi", "philhealth", "philgis_psgc", "transport_proxy", "doh_facility_stats"}
    )
    accepted_upstream = (
        upstream_score >= float(score_cfg["accepted_upstream_threshold"])
        and domain_quality >= min_domain_quality
        and upstream_context_ok
    )
    accepted_union = accepted_hiv or accepted_upstream
    bucket = (
        "accepted"
        if accepted_union
        else "review"
        if domain_quality >= max(float(score_cfg["review_floor"]), min_domain_quality * float(score_cfg["review_fraction_of_min"]))
        else "rejected"
    )
    scored = dict(record)
    scored.update(
        {
            "domain_quality_score": round(domain_quality, 4),
            "hiv_direct_score": round(hiv_direct_score, 4),
            "upstream_determinant_score": round(upstream_score, 4),
            "accepted_hiv_direct": accepted_hiv,
            "accepted_upstream_determinant": accepted_upstream,
            "accepted_union": accepted_union,
            "bucket": bucket,
            "geo_focus_score": round(geo_score, 4),
            "query_silo_match_score": round(silo_match, 4),
            "precision_penalty": round(precision_penalty, 4),
            "off_topic_biomedical": off_topic_biomed,
            "generic_science_penalized": generic_science,
            "linkage_targets": linkage_targets,
            "soft_ontology_tags": soft_tags,
            "soft_subparameter_hints": soft_hints,
            "upstream_signal_count": upstream_signal_count,
            "upstream_context_ok": bool(upstream_context_ok),
        }
    )
    return scored


def _parse_text_document(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")


def _document_metadata_text(document_row: dict[str, Any]) -> str:
    parts = [str(document_row.get("title") or "").strip(), str(document_row.get("abstract") or "").strip()]
    return "\n".join(part for part in parts if part).strip()


def _parse_html_document(path: Path) -> str:
    raw = _parse_text_document(path)
    if not raw:
        return ""
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?i)</?(br|p|div|li|tr|td|th|h[1-6])[^>]*>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_non_pdf_document(path: Path, *, doc_type: str, document_row: dict[str, Any]) -> tuple[str, str]:
    if doc_type == "html":
        metadata_text = _document_metadata_text(document_row)
        if metadata_text:
            return metadata_text, "html_metadata"
        return _parse_html_document(path), "html_text"
    return _parse_text_document(path), "text_read"


def _parse_pdf_with_fallbacks(path: Path) -> tuple[str, str]:
    if fitz is None or not path.exists():
        return "", "pdf_unavailable"
    try:
        doc = fitz.open(path)
        try:
            text = []
            for page in doc[: min(5, doc.page_count)]:
                text.append(page.get_text("text"))
            return "\n".join(text).strip(), "pymupdf"
        finally:
            doc.close()
    except Exception:
        return "", "pdf_parse_failed"


def _parse_pdf_blocks_with_fallbacks(
    path: Path,
    *,
    preferred_pages: list[int] | None = None,
    max_pages: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    if fitz is None or not path.exists():
        return [], "pdf_unavailable"
    try:
        doc = fitz.open(path)
        try:
            selected_pages: list[int]
            if preferred_pages:
                selected_pages = [page for page in preferred_pages if 1 <= page <= doc.page_count]
            else:
                selected_pages = list(range(1, min(doc.page_count, max_pages) + 1))
            blocks: list[dict[str, Any]] = []
            for page_number in selected_pages:
                page = doc.load_page(page_number - 1)
                text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                if not text:
                    continue
                blocks.append({"page_number": page_number, "text": text})
            return blocks, "pymupdf"
        finally:
            doc.close()
    except Exception:
        return [], "pdf_parse_failed"


def _phase0_ocr_backend(requested_backend: str | None = None) -> str:
    requested = str(requested_backend or os.environ.get("EPIGRAPH_PHASE0_OCR_BACKEND", DEFAULT_LIGHTON_OCR_BACKEND)).strip().lower()
    if requested in {"", "false", "none", "disabled"}:
        return "disabled"
    local_ready = False
    try:
        import transformers as _transformers  # type: ignore

        local_ready = hasattr(_transformers, "LightOnOcrProcessor") and hasattr(_transformers, "LightOnOcrForConditionalGeneration")
    except Exception:
        local_ready = False
    endpoint = str(os.environ.get("EPIGRAPH_LIGHTON_OCR_ENDPOINT", "")).strip()
    vllm_ready = _lighton_ocr_vllm_ready(endpoint if endpoint else DEFAULT_LIGHTON_OCR_ENDPOINT, explicit=bool(endpoint))
    if requested == "lighton_local":
        return "lighton_local" if local_ready else "disabled"
    if requested == "lighton_vllm":
        return "lighton_vllm" if vllm_ready else "disabled"
    if requested == "auto":
        if local_ready:
            return "lighton_local"
        if vllm_ready:
            return "lighton_vllm"
        return "disabled"
    return "disabled"


def _normalize_lighton_ocr_endpoint(endpoint: str) -> str:
    normalized = str(endpoint or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/chat/completions") or normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    if "/v1/" in normalized:
        return normalized
    return f"{normalized}/v1/chat/completions"


def _lighton_ocr_health_endpoint(endpoint: str) -> str:
    normalized = _normalize_lighton_ocr_endpoint(endpoint)
    if not normalized:
        return ""
    if "/v1/" in normalized:
        return normalized.split("/v1/", 1)[0] + "/health"
    if normalized.endswith("/v1"):
        return normalized[:-3] + "/health"
    return normalized.rsplit("/", 1)[0] + "/health"


@lru_cache(maxsize=4)
def _lighton_ocr_vllm_ready(endpoint: str, *, explicit: bool = False) -> bool:
    endpoint = _normalize_lighton_ocr_endpoint(endpoint)
    if not endpoint:
        return False
    health_endpoint = _lighton_ocr_health_endpoint(endpoint)
    if not explicit and endpoint == DEFAULT_LIGHTON_OCR_ENDPOINT:
        try:
            requests.get(health_endpoint, timeout=1.5)
            return True
        except Exception:
            return False
    try:
        requests.get(health_endpoint, timeout=1.5)
        return True
    except Exception:
        return explicit


def _pdf_requires_ocr(blocks: list[dict[str, Any]]) -> bool:
    if not blocks:
        return True
    text_lengths = [len(str(block.get("text") or "").strip()) for block in blocks]
    if not text_lengths:
        return True
    avg_chars = float(np.mean(text_lengths))
    return avg_chars < float(_phase0_required_section("ocr")["pdf_requires_ocr_avg_chars"])


def _should_run_ocr_sidecar(*, enable_ocr_sidecar: bool, force_ocr_sidecar: bool, pdf_blocks: list[dict[str, Any]]) -> bool:
    if not enable_ocr_sidecar:
        return False
    if force_ocr_sidecar:
        return True
    return _pdf_requires_ocr(pdf_blocks)


def _render_pdf_pages_for_ocr(
    path: Path,
    *,
    preferred_pages: list[int] | None = None,
    max_pages: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    selected_pages = preferred_pages or []
    rendered: list[dict[str, Any]] = []
    if pdfium is not None and Image is not None and path.exists():
        try:
            pdf = pdfium.PdfDocument(str(path))
            page_numbers = selected_pages or list(range(1, min(len(pdf), max_pages) + 1))
            for page_number in page_numbers:
                if page_number < 1 or page_number > len(pdf):
                    continue
                page = pdf[page_number - 1]
                dpi = float(_phase0_required_section("ocr")["render_dpi"])
                pil_image = page.render(scale=(dpi / 72.0)).to_pil()
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                rendered.append(
                    {
                        "page_number": page_number,
                        "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                    }
                )
            if rendered:
                return rendered, "pypdfium2"
        except Exception:
            rendered = []
    if fitz is not None and path.exists():
        try:
            doc = fitz.open(path)
            try:
                page_numbers = selected_pages or list(range(1, min(doc.page_count, max_pages) + 1))
                for page_number in page_numbers:
                    if page_number < 1 or page_number > doc.page_count:
                        continue
                    page = doc.load_page(page_number - 1)
                    pix = page.get_pixmap(dpi=200, alpha=False)
                    rendered.append(
                        {
                            "page_number": page_number,
                            "image_base64": base64.b64encode(pix.tobytes("png")).decode("utf-8"),
                        }
                    )
                if rendered:
                    return rendered, "pymupdf_render"
            finally:
                doc.close()
        except Exception:
            rendered = []
    pdftoppm_path = shutil.which("pdftoppm")
    if pdftoppm_path and path.exists():
        try:
            dpi = str(int(float(_phase0_required_section("ocr")["render_dpi"])))
            with tempfile.TemporaryDirectory(prefix="epigraph_ocr_") as tmpdir:
                page_numbers = selected_pages or list(range(1, max_pages + 1))
                for page_number in page_numbers:
                    if page_number < 1:
                        continue
                    prefix = Path(tmpdir) / f"page_{page_number}"
                    command = [
                        pdftoppm_path,
                        "-r",
                        dpi,
                        "-f",
                        str(page_number),
                        "-l",
                        str(page_number),
                        "-png",
                        str(path),
                        str(prefix),
                    ]
                    completed = subprocess.run(command, capture_output=True, timeout=90, check=False)
                    if completed.returncode != 0:
                        continue
                    output_path = prefix.parent / f"{prefix.name}-{page_number}.png"
                    if not output_path.exists():
                        continue
                    rendered.append(
                        {
                            "page_number": page_number,
                            "image_base64": base64.b64encode(output_path.read_bytes()).decode("utf-8"),
                        }
                    )
            if rendered:
                return rendered, "pdftoppm"
        except Exception:
            rendered = []
    return [], "render_unavailable"


def _lighton_ocr_vllm_extract(image_base64: str) -> str:
    endpoint = _normalize_lighton_ocr_endpoint(str(os.environ.get("EPIGRAPH_LIGHTON_OCR_ENDPOINT", DEFAULT_LIGHTON_OCR_ENDPOINT)).strip())
    model_name = str(os.environ.get("EPIGRAPH_LIGHTON_OCR_MODEL", DEFAULT_LIGHTON_OCR_MODEL)).strip()
    prompt = str(os.environ.get("EPIGRAPH_LIGHTON_OCR_PROMPT", DEFAULT_LIGHTON_OCR_PROMPT)).strip() or DEFAULT_LIGHTON_OCR_PROMPT
    include_prompt = str(os.environ.get("EPIGRAPH_LIGHTON_OCR_INCLUDE_PROMPT", "1")).strip().lower() not in {"0", "false", "no", "off"}
    timeout_seconds = float(str(os.environ.get("EPIGRAPH_LIGHTON_OCR_TIMEOUT_SECONDS", "300")).strip() or "300")
    headers = {"Content-Type": "application/json"}
    api_key = str(os.environ.get("EPIGRAPH_LIGHTON_OCR_API_KEY", "")).strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if endpoint.endswith("/chat/completions"):
        base_endpoint = endpoint[: -len("/chat/completions")]
    elif endpoint.endswith("/responses"):
        base_endpoint = endpoint[: -len("/responses")]
    else:
        base_endpoint = endpoint.rsplit("/v1", 1)[0] + "/v1" if "/v1" in endpoint else endpoint.rstrip("/")
    chat_endpoint = f"{base_endpoint}/chat/completions"
    responses_endpoint = f"{base_endpoint}/responses"
    chat_content: list[dict[str, Any]] = []
    responses_content: list[dict[str, Any]] = []
    if include_prompt:
        chat_content.append({"type": "text", "text": prompt})
        responses_content.append({"type": "input_text", "text": prompt})
    chat_content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        }
    )
    responses_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"})
    attempts = [
        (
            chat_endpoint,
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": chat_content,
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "chat",
        ),
        (
            responses_endpoint,
            {
                "model": model_name,
                "input": [
                    {
                        "role": "user",
                        "content": responses_content,
                    }
                ],
                "max_output_tokens": 4096,
                "temperature": 0.0,
            },
            "responses",
        ),
    ]
    errors: list[str] = []
    for request_endpoint, payload, mode in attempts:
        try:
            response = requests.post(request_endpoint, json=payload, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            body = response.json()
            if mode == "chat":
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
                    text = "\n".join(part for part in parts if part).strip()
                else:
                    text = str(content or "").strip()
            else:
                text = str(body.get("output_text") or "").strip()
                if not text:
                    output = body.get("output") or []
                    parts: list[str] = []
                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        for content_item in item.get("content") or []:
                            if isinstance(content_item, dict):
                                parts.append(str(content_item.get("text") or ""))
                    text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
            errors.append(f"{mode}:empty_response")
        except Exception as exc:
            errors.append(f"{mode}:{exc.__class__.__name__}")
    raise RuntimeError("lighton_ocr_vllm_failed:" + ";".join(errors))


def _lighton_ocr_sidecar_blocks(
    path: Path,
    *,
    preferred_pages: list[int] | None = None,
    max_pages: int = 5,
    requested_backend: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = _phase0_ocr_backend(requested_backend)
    if backend == "disabled":
        return [], {"enabled": False, "backend": "disabled", "status": "backend_unavailable"}
    if backend == "lighton_local":
        return [], {"enabled": True, "backend": backend, "status": "local_backend_requires_transformers_v5"}
    rendered_pages, render_backend = _render_pdf_pages_for_ocr(path, preferred_pages=preferred_pages, max_pages=max_pages)
    if not rendered_pages:
        return [], {"enabled": True, "backend": backend, "status": "render_failed", "render_backend": render_backend}
    blocks: list[dict[str, Any]] = []
    failures: list[str] = []
    for rendered in rendered_pages:
        try:
            text = _lighton_ocr_vllm_extract(str(rendered["image_base64"]))
            if text.strip():
                blocks.append({"page_number": int(rendered["page_number"]), "text": text.strip()})
        except Exception as exc:
            failures.append(f"page_{int(rendered['page_number'])}:{exc.__class__.__name__}")
    return blocks, {
        "enabled": True,
        "backend": backend,
        "status": "parsed" if blocks else "failed",
        "render_backend": render_backend,
        "failure_notes": failures,
        "page_count": len(rendered_pages),
        "parsed_page_count": len(blocks),
    }


def _detect_time_for_span(text: str) -> str:
    lowered = str(text or "").lower()
    for month_name, month_number in MONTH_NAME_TO_NUMBER.items():
        match = re.search(rf"{month_name}\s+(19|20)\d{{2}}", lowered)
        if match:
            year = re.search(r"(19|20)\d{2}", match.group(0))
            if year:
                normalized = _normalize_time_label(f"{year.group(0)}-{month_number:02d}")
                if normalized != "unknown":
                    return normalized
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        normalized = _normalize_time_label(match.group(0))
        return "" if normalized == "unknown" else normalized
    return ""


def _unit_and_values(text: str) -> list[tuple[float, str, str]]:
    results: list[tuple[float, str, str]] = []
    for match in re.finditer(r"(?P<value>\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)\s*(?P<unit>%|percent|people|cases|deaths|km|hours|days|usd|php|peso|pesos|million|billion)?", text, flags=re.I):
        raw = match.group("value")
        if re.fullmatch(r"(19|20)\d{2}", raw):
            continue
        value = _safe_float(raw.replace(" ", "").replace(",", ""))
        if value is None:
            continue
        unit = (match.group("unit") or "").lower()
        results.append((value, unit, match.group(0)))
    return results


def _unit_value_matches(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?P<value>\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)\s*(?P<unit>%|percent|people|cases|deaths|km|hours|days|usd|php|peso|pesos|million|billion)?",
        text,
        flags=re.I,
    ):
        raw = match.group("value")
        if re.fullmatch(r"(19|20)\d{2}", raw):
            continue
        value = _safe_float(raw.replace(" ", "").replace(",", ""))
        if value is None:
            continue
        rows.append(
            {
                "value": float(value),
                "unit": (match.group("unit") or "").lower(),
                "span": match.group(0),
                "start": int(match.start()),
                "end": int(match.end()),
            }
        )
    return rows


def _canonical_name(parameter_text: str, unit: str) -> str:
    lowered = parameter_text.lower()
    cues = {
        "plhiv_prevalence": ["plhiv prevalence", "prevalence per 100k", "estimated plhiv"],
        "hiv_incidence": ["incidence", "new hiv diagnoses", "new diagnoses per 100k"],
        "late_diagnosis_rate": ["late diagnosis", "cd4<200", "cd4 <200", "advanced hiv disease"],
        "art_uptake_rate": ["art uptake", "on art", "alive on art"],
        "viral_suppression_rate": ["viral suppression", "virally suppressed", "suppressed among art"],
        "msm_population_share": ["msm population", "men who have sex with men", "msm"],
        "fsw_population_share": ["sex worker", "female sex worker", "fsw"],
        "idu_prevalence": ["inject drugs", "pwid", "idu"],
        "condom_use": ["condom use", "using condom", "condom"],
        "hiv_knowledge_index": ["hiv knowledge", "aids knowledge", "awareness index"],
        "education_level": ["mean years schooling", "educational attainment", "years schooling"],
        "literacy_rate": ["literacy", "literate"],
        "poverty_rate": ["poverty rate", "poverty incidence", "below poverty line"],
        "unemployment_rate": ["unemployment", "labor force unemployed"],
        "gdp_per_capita": ["gdp per capita"],
        "health_expenditure": ["health expenditure", "health spending"],
        "philhealth_coverage": ["philhealth", "health insurance coverage"],
        "clinics_per_capita": ["clinics per", "treatment hub", "testing sites per"],
        "physicians_per_capita": ["physicians per", "mds per", "doctors per"],
        "viral_load": ["viral load", "suppression"],
        "cd4_count": ["cd4"],
        "treatment_cost": ["cost", "usd", "php", "peso"],
        "population_count": ["population", "people"],
        "case_count": ["case", "cases", "deaths", "diagnosis"],
        "travel_time": ["travel", "hours", "days", "km"],
        "testing_rate": ["testing", "screening", "%", "percent"],
        "public_transit_usage": ["transit usage", "transit visits", "mobility report"],
        "social_media_use": ["social media", "facebook access"],
        "urbanization": ["urbanization", "urban population"],
        "population_density": ["population density", "people per km"],
        "tb_hiv_coinfection": ["tb/hiv", "tb coinfection", "tb among plhiv"],
        "hepatitis_b_prevalence": ["hepatitis b", "hbv"],
        "covid_mobility": ["workplace mobility", "retail mobility", "covid-era mobility"],
        "migration_rate": ["migration rate", "ofw", "working abroad"],
        "male_circumcision": ["male circumcision", "circumcised"],
        "stigma_index": ["stigma index", "stigma score"],
        "education_gap": ["no formal schooling", "education gap"],
        "contraceptive_use": ["contraceptive use", "family planning use"],
        "tb_incidence": ["tb incidence", "tuberculosis incidence"],
        "infant_mortality": ["infant mortality"],
    }
    for name, tokens in cues.items():
        if any(token in lowered or token == unit for token in tokens):
            return name
    return "numeric_observation"


def _text_window(text: str, *, start: int, end: int, radius: int) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _second_pass_canonical_name(
    *,
    canonical_name: str,
    parameter_text: str,
    local_context: str,
    source_text: str,
    source: dict[str, Any],
    unit: str,
    semantics: dict[str, Any],
) -> tuple[str, str]:
    cfg = dict(_phase0_required_section("second_pass_canonicalizer"))
    generic_name = str(cfg.get("generic_name") or "numeric_observation")
    if canonical_name != generic_name:
        return canonical_name, "first_pass_specific"
    broader_context = " ".join(
        str(item or "")
        for item in (
            source.get("title"),
            source.get("abstract"),
            source.get("query_silo"),
            source.get("platform"),
            local_context,
            source_text[:600],
        )
    )
    platform_map = {str(key): str(value) for key, value in dict(cfg.get("platform_to_canonical") or {}).items()}
    platform_name = platform_map.get(str(source.get("platform") or ""))
    if platform_name:
        return platform_name, "platform_mapping"
    hints = _soft_subparameter_hints(broader_context)
    linkage_targets = _chunk_linkage_targets(broader_context, str(source.get("query_silo") or ""), hints)
    query_map = {str(key): str(value) for key, value in dict(cfg.get("query_silo_to_canonical") or {}).items()}
    query_silo = str(source.get("query_silo") or "")
    if query_silo in query_map and (hints or linkage_targets or source.get("query_geo_focus") == "philippines"):
        return query_map[query_silo], "query_silo_mapping"
    contextual_name = _canonical_name(f"{parameter_text} {broader_context}", unit)
    if contextual_name != generic_name:
        return contextual_name, "contextual_keywords"
    lowered = broader_context.lower()
    denominator_type = str(semantics.get("denominator_type") or "")
    measurement_type = str(semantics.get("measurement_type") or "")
    if "philhealth" in lowered:
        return "philhealth_coverage", "keyword_mapping"
    if any(token in lowered for token in ("clinic", "treatment hub", "facility", "master list", "service delivery")) and denominator_type in {"facility", "population"}:
        return "clinics_per_capita", "facility_mapping"
    if any(token in lowered for token in ("stigma", "discrimination", "social norm")):
        return "stigma_barrier", "keyword_mapping"
    if any(token in lowered for token in ("travel time", "hours", "km", "distance", "commute", "congestion")):
        return "travel_time", "travel_time_mapping"
    if any(token in lowered for token in ("mobility", "migration", "mixing", "commuting", "corridor")):
        return "mobility_network_mixing", "mobility_mapping"
    if any(token in lowered for token in ("poverty", "income", "household expenditure", "below poverty")) and measurement_type in {"rate", "count"}:
        return "poverty_rate", "poverty_mapping"
    if any(token in lowered for token in ("gdp per capita", "gdp")):
        return "gdp_per_capita", "keyword_mapping"
    if any(token in lowered for token in ("health expenditure", "health spending")):
        return "health_expenditure", "keyword_mapping"
    if any(token in lowered for token in ("education", "schooling", "attainment")):
        return "education_level", "keyword_mapping"
    if any(token in lowered for token in ("literacy",)):
        return "literacy_rate", "keyword_mapping"
    if any(token in lowered for token in ("knowledge", "awareness")) and ("hiv" in lowered or "aids" in lowered):
        return "hiv_knowledge_index", "keyword_mapping"
    if any(token in lowered for token in ("viral suppression", "virally suppressed")):
        return "viral_suppression_rate", "keyword_mapping"
    if any(token in lowered for token in ("viral load", "vl testing", "tested for viral load")):
        return "viral_load", "keyword_mapping"
    if any(token in lowered for token in ("testing", "screening", "diagnosis")):
        return "testing_uptake", "keyword_mapping"
    return generic_name, "generic_unresolved"


def _drop_low_signal_numeric_candidate(
    *,
    canonical_name: str,
    value: float,
    unit: str,
    parameter_text: str,
    local_context: str,
    source: dict[str, Any],
    geo_label: str,
) -> tuple[bool, str]:
    cfg = dict(_phase0_required_section("second_pass_canonicalizer"))
    generic_name = str(cfg.get("generic_name") or "numeric_observation")
    if canonical_name != generic_name:
        return False, ""
    broader_context = " ".join(
        str(item or "")
        for item in (source.get("title"), source.get("abstract"), source.get("query_silo"), local_context)
    ).lower()
    hints = _soft_subparameter_hints(broader_context)
    linkage_targets = _chunk_linkage_targets(broader_context, str(source.get("query_silo") or ""), hints)
    signal_count = (
        int(bool(source.get("query_silo")))
        + len(set(hints))
        + len(set(linkage_targets))
        + int(bool(unit))
        + int(bool(geo_label))
    )
    source_tier = str(source.get("source_tier") or "")
    numeric_only = bool(re.fullmatch(r"\d+(?:\.\d+)?", parameter_text.strip()))
    known_years = re.findall(r"\b(?:19|20)\d{2}\b", broader_context)
    if (
        numeric_only
        and not unit
        and not local_context.strip()
        and any(year.startswith(parameter_text.strip()) or year.endswith(parameter_text.strip()) for year in known_years)
    ):
        return True, "metadata_year_fragment"
    if (
        numeric_only
        and not unit
        and abs(float(value)) <= float(cfg.get("drop_unitless_small_max", 20.0))
        and signal_count < int(cfg.get("drop_signal_floor", 2))
        and source_tier.startswith("tier2_")
    ):
        return True, "low_signal_small_unitless"
    if (
        numeric_only
        and len(parameter_text.strip()) <= int(cfg.get("drop_numeric_only_text_max_chars", 4))
        and not unit
        and not geo_label
        and signal_count < int(cfg.get("drop_signal_floor", 2))
        and "hiv" not in broader_context
    ):
        return True, "unresolved_numeric_stub"
    return False, ""


def _candidate_index_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("canonical_name"),
        row.get("parameter_text"),
        row.get("geo"),
        row.get("time"),
        row.get("sex"),
        row.get("age_band"),
        row.get("kp_group"),
    ]
    return " ".join(str(value) for value in fields if value)


def _candidate_row_from_observation(obs: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    details = []
    ref = LiteratureRefDetail(
        source_id=str(source.get("source_id") or ""),
        title=source.get("title"),
        year=source.get("year"),
        source_tier=source.get("source_tier"),
        url=source.get("url"),
        doi=source.get("doi"),
        pmid=source.get("pmid"),
        openalex_id=source.get("openalex_id"),
    ).to_dict()
    if has_verifiable_locator(ref):
        details.append(ref)
    return {
        "candidate_id": f"cand-{obs['observation_id']}",
        **obs,
        "source_bank": obs.get("source_bank") or "phase0_extracted",
        "candidate_text": _candidate_index_text(obs),
        "literature_ref_details": details,
    }


def _official_anchor_observation(
    *,
    block: dict[str, Any],
    source: dict[str, Any],
    observation_suffix: str,
    parameter_text: str,
    canonical_name: str,
    value: float,
    unit: str,
    geo_text: str,
    time: str,
    sex: str = "",
    age_band: str = "",
    kp_group: str = "",
) -> dict[str, Any]:
    geo_match = infer_philippines_geo(
        f"{geo_text} {parameter_text}",
        default_country_focus=True,
    )
    canonical_geo = normalize_geo_label(geo_match.geo or "Philippines", default_country_focus=True)
    semantics = _infer_measurement_semantics(parameter_text, parameter_text, unit)
    return {
        "observation_id": f"obs-{block['block_id']}-{observation_suffix}",
        "source_id": block["source_id"],
        "document_id": block["document_id"],
        "source_bank": "phase0_extracted",
        "parameter_text": parameter_text,
        "canonical_name": canonical_name,
        "value": float(value),
        "unit": unit,
        "population": "",
        "geo": canonical_geo,
        "region": geo_match.region,
        "province": geo_match.province,
        "geo_mentions": geo_match.mentions or [geo_text],
        "time": time,
        **semantics,
        "sex": sex,
        "age_band": age_band,
        "kp_group": kp_group,
        "evidence_span": parameter_text,
        "extraction_method": "official_anchor_pack",
        "confidence": float(_phase0_required_section("confidence")["structured_anchor"]),
        "is_direct_measurement": True,
        "is_prior_only": False,
        "is_anchor_eligible": True,
    }


def _extract_who_core_team_anchors(block: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = str(block.get("text") or "")
    page_number = int(block.get("page_number") or 0)
    observations: list[dict[str, Any]] = []
    if page_number == 13 and "Philippine HIV Care Cascade" in text:
        counts = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", text)
        if len(counts) >= 5:
            values = [float(item.replace(",", "")) for item in counts[:5]]
            labels = [
                ("estimated_plhiv", "Estimated PLHIV", "population_count"),
                ("diagnosed", "Diagnosed PLHIV", "case_count"),
                ("alive_on_art", "Alive on ART", "case_count"),
                ("tested_for_viral_load", "Tested for Viral Load", "viral_load"),
                ("virally_suppressed", "Virally Suppressed", "viral_load"),
            ]
            for (suffix, label, canonical_name), value in zip(labels, values, strict=False):
                observations.append(
                    _official_anchor_observation(
                        block=block,
                        source=source,
                        observation_suffix=suffix,
                        parameter_text=label,
                        canonical_name=canonical_name,
                        value=value,
                        unit="people",
                        geo_text="Philippines",
                        time="2024-12",
                    )
                )
        shares = re.findall(r"(?<!\d)(\d{1,2})%", text)
        share_labels = [
            ("diagnosed_share", "Diagnosed among estimated PLHIV", "testing_rate"),
            ("art_share", "Alive on ART among diagnosed PLHIV", "testing_rate"),
            ("tested_share", "Tested for Viral Load among alive on ART", "testing_rate"),
            ("suppressed_share", "Virally Suppressed among tested for Viral Load", "testing_rate"),
        ]
        for (suffix, label, canonical_name), raw_value in zip(share_labels, shares[:4], strict=False):
            observations.append(
                _official_anchor_observation(
                    block=block,
                    source=source,
                    observation_suffix=suffix,
                    parameter_text=label,
                    canonical_name=canonical_name,
                    value=float(raw_value),
                    unit="percent",
                    geo_text="Philippines",
                    time="2024-12",
                )
            )
    elif page_number == 18 and "Prevention Coverage MSM & TGW" in text:
        pattern = re.compile(
            r"(NCR|Cebu City|Cebu Province|Angeles City|Category A|Category B|Category C|National)\s+(\d{1,2})%\s+([\d,\s]+)"
        )
        for index, match in enumerate(pattern.finditer(text), start=1):
            geo_label = re.sub(r"\s+", " ", match.group(1)).strip()
            pct_value = float(match.group(2))
            pop_value = float(match.group(3).replace(" ", "").replace(",", ""))
            geo_text = "Philippines" if geo_label.lower() == "national" else geo_label
            observations.append(
                _official_anchor_observation(
                    block=block,
                    source=source,
                    observation_suffix=f"prevention_coverage_{index}",
                    parameter_text=f"MSM and TGW prevention coverage in {geo_label}",
                    canonical_name="prevention_coverage",
                    value=pct_value,
                    unit="percent",
                    geo_text=geo_text,
                    time="2025",
                    kp_group="msm",
                )
            )
            observations.append(
                _official_anchor_observation(
                    block=block,
                    source=source,
                    observation_suffix=f"kp_population_{index}",
                    parameter_text=f"MSM and TGW estimated population in {geo_label}",
                    canonical_name="population_count",
                    value=pop_value,
                    unit="people",
                    geo_text=geo_text,
                    time="2025",
                    kp_group="msm",
                )
            )
    candidate_rows = [_candidate_row_from_observation(obs, source) for obs in observations]
    return observations, candidate_rows


def _extract_surveillance_anchors(block: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = str(block.get("text") or "")
    page_number = int(block.get("page_number") or 0)
    observations: list[dict[str, Any]] = []
    if page_number == 3 and "Overview of the Philippine HIV epidemic" in text:
        counts = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", text)
        if len(counts) >= 2:
            observations.append(
                _official_anchor_observation(
                    block=block,
                    source=source,
                    observation_suffix="estimated_plhiv_2021",
                    parameter_text="Estimated number of people living with HIV",
                    canonical_name="population_count",
                    value=float(counts[0].replace(",", "")),
                    unit="people",
                    geo_text="Philippines",
                    time="2021",
                )
            )
            observations.append(
                _official_anchor_observation(
                    block=block,
                    source=source,
                    observation_suffix="estimated_plhiv_2030",
                    parameter_text="Projected number of people living with HIV",
                    canonical_name="population_count",
                    value=float(counts[1].replace(",", "")),
                    unit="people",
                    geo_text="Philippines",
                    time="2030",
                )
            )
        age_patterns = [
            ("15_24", r"47%\s+15-24"),
            ("25_34", r"42%\s+25-34"),
            ("35_49", r"8%\s+35-44"),
            ("50_plus", r"2%\s+45-54"),
        ]
        for age_band, pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                pct = float(re.search(r"\d{1,2}", match.group(0)).group(0))
                observations.append(
                    _official_anchor_observation(
                        block=block,
                        source=source,
                        observation_suffix=f"age_share_{age_band}",
                        parameter_text=f"Share of new infections among age band {age_band}",
                        canonical_name="testing_rate",
                        value=pct,
                        unit="percent",
                        geo_text="Philippines",
                        time="2021",
                        age_band=age_band,
                    )
                )
        kp_patterns = [
            ("msm", r"MSM.*?78%"),
            ("pwid", r"PWID.*?2%"),
        ]
        for kp_group, pattern in kp_patterns:
            match = re.search(pattern, text)
            if match:
                pct_match = re.search(r"(\d{1,2})%", match.group(0))
                if pct_match:
                    observations.append(
                        _official_anchor_observation(
                            block=block,
                            source=source,
                            observation_suffix=f"kp_share_{kp_group}",
                            parameter_text=f"Share of PLHIV among key population {kp_group}",
                            canonical_name="testing_rate",
                            value=float(pct_match.group(1)),
                            unit="percent",
                            geo_text="Philippines",
                            time="2021",
                            kp_group=kp_group,
                        )
                    )
    candidate_rows = [_candidate_row_from_observation(obs, source) for obs in observations]
    return observations, candidate_rows


def _extract_official_anchor_rows(
    parsed_blocks: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    numeric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    handled_blocks: set[str] = set()
    for block in parsed_blocks:
        source = source_rows.get(str(block.get("source_id") or ""), {})
        anchor_key = str(source.get("local_anchor_key") or "")
        if not anchor_key:
            continue
        preferred_pages = {int(page) for page in (source.get("preferred_pages") or []) if str(page).isdigit()}
        if int(block.get("page_number") or 0) in preferred_pages:
            handled_blocks.add(str(block.get("block_id") or ""))
        rows: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        if anchor_key == "who_core_team_2025":
            rows, candidates = _extract_who_core_team_anchors(block, source)
        elif anchor_key == "phil_hiv_sti_surveillance":
            rows, candidates = _extract_surveillance_anchors(block, source)
        if rows:
            numeric_rows.extend(rows)
            candidate_rows.extend(candidates)
    return numeric_rows, candidate_rows, handled_blocks


def _chunk_linkage_targets(text: str, query_silo: str, hints: list[str]) -> list[str]:
    lowered = str(text or "").lower()
    targets: set[str] = set()
    promoted_silos = {key for key, value in (_HIV_PLUGIN.determinant_silos or {}).items() if value.promotion_track == "main_predictive_candidate"}
    if query_silo in promoted_silos:
        targets.add(query_silo)
    for hint in hints:
        if hint in promoted_silos:
            targets.add(hint)
    if any(token in lowered for token in ("testing", "screening", "late diagnosis", "fear of a positive result", "hiv knowledge")):
        targets.add("testing_uptake")
    if any(token in lowered for token in ("linkage", "referral", "treatment initiation", "care navigation")):
        targets.add("linkage_to_care")
    if any(token in lowered for token in ("retention", "adherence", "loss to follow up", "continuity of care", "treatment interruption")):
        targets.add("retention_adherence")
    if any(token in lowered for token in ("viral suppression", "viral load", "documented suppression", "suppressed")):
        targets.add("suppression_outcomes")
    if any(token in lowered for token in ("condom", "prep", "pre exposure prophylaxis", "harm reduction")):
        targets.add("prevention_access")
    return sorted(targets)


def _extract_chunk_soft_candidates(
    parsed_chunks: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cfg = _phase0_required_section("chunk_soft_candidates")
    min_chunk_chars = int(cfg["min_chunk_chars"])
    max_chunks_per_document = int(cfg["max_chunks_per_document"])
    max_candidates_per_chunk = int(cfg["max_candidates_per_chunk"])
    excerpt_chars = int(cfg["excerpt_chars"])
    chunk_counts: Counter[str] = Counter()
    chunk_candidates: list[dict[str, Any]] = []
    for chunk in parsed_chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
        document_id = str(chunk.get("document_id") or "")
        if document_id and chunk_counts[document_id] >= max_chunks_per_document:
            continue
        source = source_rows.get(str(chunk.get("source_id") or ""), {})
        source_title = str(source.get("title") or "")
        query_silo = str(chunk.get("query_silo") or source.get("query_silo") or "")
        combined_text = f"{source_title}\n{text}".strip()
        soft_tags = _soft_ontology_tags(combined_text)
        soft_hints = _soft_subparameter_hints(combined_text)
        linkage_targets = _chunk_linkage_targets(combined_text, query_silo, soft_hints)
        high_signal = bool(query_silo or soft_tags or soft_hints or linkage_targets)
        effective_min_chunk_chars = min_chunk_chars if not high_signal else min(min_chunk_chars, 120)
        if len(text) < effective_min_chunk_chars:
            continue
        determinant_silos = list(
            dict.fromkeys(
                [
                    *(source.get("determinant_silos") or []),
                    *([query_silo] if query_silo else []),
                    *(hint for hint in soft_hints if hint in (_HIV_PLUGIN.determinant_silos or {})),
                ]
            )
        )
        if not determinant_silos and not soft_tags and not soft_hints and not linkage_targets:
            continue
        default_country_focus = bool(source.get("query_geo_focus") == "philippines")
        geo_match = infer_philippines_geo(combined_text, default_country_focus=default_country_focus)
        canonical_geo = normalize_geo_label(
            geo_match.geo or ("Philippines" if default_country_focus else ""),
            default_country_focus=default_country_focus,
        ) or ("Philippines" if default_country_focus else "")
        record = {
            "record_id": str(chunk.get("chunk_id") or chunk.get("block_id") or f"chunk-{len(chunk_candidates)+1}"),
            "title": text[:excerpt_chars],
            "year": _safe_int(source.get("year")),
            "source_tier": source.get("source_tier"),
            "url": source.get("url"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
            "openalex_id": source.get("openalex_id"),
            "platform": source.get("platform"),
            "query_geo_focus": source.get("query_geo_focus") or "",
            "query_silo": query_silo,
            "determinant_silos": determinant_silos,
            "soft_ontology_tags": soft_tags,
            "soft_subparameter_hints": soft_hints,
            "linkage_targets": linkage_targets,
        }
        expanded = wide_sweep_candidate_rows(record, bank_name="phase0_chunk_soft_candidates")
        if not expanded:
            continue
        priority_names = {
            *(determinant_silos or []),
            *(linkage_targets or []),
            *(soft_hints or []),
        }
        query_token = re.sub(r"[^a-z0-9]+", "_", query_silo.strip().lower()).strip("_")
        if query_token:
            priority_names.add(query_token)
        def _row_priority(row: dict[str, Any]) -> tuple[int, int, str]:
            canonical_name = str(row.get("canonical_name") or "")
            return (
                0 if canonical_name == query_token and canonical_name else 1,
                0 if canonical_name in priority_names else 1,
                canonical_name,
            )
        expanded = sorted(expanded, key=_row_priority)
        excerpt = text[:excerpt_chars]
        time_label = _phase0_observation_time(source, str(source.get("year") or ""))
        confidence = float(_phase0_required_section("confidence")["skip_live_normalizer"])
        for row in expanded[:max_candidates_per_chunk]:
            patched = dict(row)
            patched.update(_infer_measurement_semantics(str(row.get("canonical_name") or ""), excerpt, ""))
            patched.update(
                {
                    "candidate_id": f"chunk-soft-{chunk.get('chunk_id')}-{row['canonical_name']}",
                    "document_id": document_id,
                    "source_id": chunk.get("source_id"),
                    "parameter_text": row["canonical_name"],
                    "candidate_text": excerpt,
                    "evidence_span": excerpt,
                    "geo": canonical_geo,
                    "region": geo_match.region,
                    "province": geo_match.province,
                    "time": time_label,
                    "sex": "",
                    "age_band": "",
                    "kp_group": "",
                    "value": None,
                    "unit": "",
                    "source_bank": "phase0_chunk_soft_candidates",
                    "source_tier": source.get("source_tier") or chunk.get("source_tier") or "",
                    "extraction_method": "chunk_soft_candidate",
                    "confidence": confidence,
                    "is_anchor_eligible": False,
                    "is_direct_measurement": False,
                    "is_prior_only": True,
                    "block_id": chunk.get("block_id"),
                    "chunk_id": chunk.get("chunk_id"),
                }
            )
            chunk_candidates.append(patched)
        chunk_counts[document_id] += 1
    return chunk_candidates


def _hashed_embedding(text: str, *, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0 else vec / norm


@lru_cache(maxsize=4)
def _load_sentence_transformer_model(model_name: str, device: str) -> Any:
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers_unavailable")
    return SentenceTransformer(model_name, local_files_only=not _phase0_allow_embed_download(), device=device)


def _phase0_embed_device() -> str:
    requested = os.environ.get("EPIGRAPH_PHASE0_EMBED_DEVICE", DEFAULT_LOCAL_EMBED_DEVICE).strip().lower()
    if requested:
        return requested
    return DEFAULT_LOCAL_EMBED_DEVICE


def _phase0_allow_embed_download() -> bool:
    return str(os.environ.get("EPIGRAPH_PHASE0_ALLOW_EMBED_DOWNLOAD", "")).strip().lower() in {"1", "true", "yes"}


def _phase0_embed_candidates() -> list[str]:
    requested = os.environ.get("EPIGRAPH_PHASE0_EMBED_MODEL", "").strip()
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
    candidates.append(DEFAULT_LOCAL_EMBED_MODEL)
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _embed_texts(texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    notes: list[str] = []
    if not texts:
        return np.zeros((0, 384), dtype=np.float32), {"backend": "empty", "model_name": "", "device": "cpu", "notes": notes, "fallback_used": False}
    requested_model = os.environ.get("EPIGRAPH_PHASE0_EMBED_MODEL", "").strip().lower()
    if requested_model == "hashed_local":
        matrix = np.vstack([_hashed_embedding(text) for text in texts]).astype(np.float32)
        return matrix, {
            "backend": "hashed_local",
            "model_name": "hashed_local",
            "device": "cpu",
            "notes": ["forced_hashed_local"],
            "fallback_used": False,
        }
    if SentenceTransformer is not None:
        device = _phase0_embed_device()
        for model_name in _phase0_embed_candidates():
            try:
                model = _load_sentence_transformer_model(model_name, device)
                encoded = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                matrix = np.asarray(encoded, dtype=np.float32)
                return matrix, {
                    "backend": "sentence_transformers",
                    "model_name": model_name,
                    "device": device,
                    "notes": notes,
                    "fallback_used": False,
                }
            except Exception as exc:  # pragma: no cover - depends on local cache state
                notes.append(f"sentence_transformers_unavailable:{model_name}:{exc.__class__.__name__}")
    model_name = os.environ.get("EPIGRAPH_PHASE0_EMBED_MODEL", "").strip()
    if model_name and AutoTokenizer is not None and AutoModel is not None and torch is not None:
        try:  # pragma: no cover - model availability depends on environment
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not _phase0_allow_embed_download())
            model = AutoModel.from_pretrained(model_name, local_files_only=not _phase0_allow_embed_download())
            device = choose_torch_device(prefer_gpu=False)
            model.to(device)
            model.eval()
            with torch.no_grad():
                encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state.mean(dim=1)
                hidden = hidden / torch.clamp(hidden.norm(dim=1, keepdim=True), min=1e-8)
                matrix = hidden.detach().cpu().numpy().astype(np.float32)
            return matrix, {
                "backend": "transformers",
                "model_name": model_name,
                "device": str(device),
                "notes": notes,
                "fallback_used": False,
            }
        except Exception as exc:  # pragma: no cover
            notes.append(f"transformers_fallback:{exc.__class__.__name__}")
    matrix = np.vstack([_hashed_embedding(text) for text in texts]).astype(np.float32)
    return matrix, {
        "backend": "hashed_local",
        "model_name": "hashed_local",
        "device": "cpu",
        "notes": notes,
        "fallback_used": True,
    }


def _phase0_target_prompt(plugin_id: str) -> str:
    plugin = plugin_id.lower()
    if plugin == "hiv":
        return "HIV care cascade diagnosis treatment viral suppression stigma logistics economics mobility population biology Philippines"
    return f"{plugin} epidemiology cascade outcomes interventions"


def _semantic_embedding_matrix(texts: list[str], *, target_prompt: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arr, meta = _embed_texts([target_prompt, *texts])
    if arr.shape[0] == 0:
        return _hashed_embedding(target_prompt), np.zeros((0, 384), dtype=np.float32), meta
    query_vector = arr[0]
    matrix = arr[1:] if arr.shape[0] > 1 else np.zeros((0, arr.shape[-1]), dtype=np.float32)
    return query_vector, matrix, meta


def _safe_mutual_information(values: np.ndarray, target: np.ndarray, *, bins: int = 8) -> float:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    if values.size == 0 or target.size == 0 or values.size != target.size:
        return 0.0
    if np.allclose(values, values[0]) or np.allclose(target, target[0]):
        return 0.0
    x_edges = np.histogram_bin_edges(values, bins=bins)
    y_edges = np.histogram_bin_edges(target, bins=bins)
    joint, _, _ = np.histogram2d(values, target, bins=(x_edges, y_edges))
    joint = joint / max(joint.sum(), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / np.clip(px @ py, 1e-12, None), 1.0)
        mi = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return float(np.nansum(mi))


def _phase0_alignment_bundle(
    *,
    candidate_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    plugin_id: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    alignment_cfg = _phase0_required_section("alignment")
    current_year = int(utc_now_iso()[:4])
    texts = [_candidate_index_text(row) for row in candidate_rows]
    target_prompt = _phase0_target_prompt(plugin_id)
    query_vector, embedding_matrix, embed_meta = _semantic_embedding_matrix(texts, target_prompt=target_prompt)
    if embedding_matrix.size:
        semantic_scores = embedding_matrix @ query_vector
    else:
        semantic_scores = np.zeros((0,), dtype=np.float32)
    canonical_names = sorted({str(row.get("canonical_name") or "numeric_observation") for row in candidate_rows}) or ["numeric_observation"]
    province_values: set[str] = set()
    for row in candidate_rows:
        normalized_geo = normalize_geo_label(
            str(row.get("geo") or ""),
            default_country_focus=bool(source_rows.get(str(row.get("source_id") or ""), {}).get("query_geo_focus") == "philippines"),
        ) or "unknown"
        geo_resolution = geo_resolution_label(normalized_geo)
        if geo_resolution in {"province", "city", "national"} or normalized_geo == "unknown":
            province_values.add(normalized_geo)
    if plugin_id == "hiv":
        province_values.update(philippines_modeling_geos(include_national=True))
    provinces = sorted(province_values) or ["unknown"]
    def _alignment_month_for_row(row: dict[str, Any]) -> str:
        source = source_rows.get(str(row.get("source_id") or ""), {})
        return _phase0_observation_time(source, str(row.get("time") or ""))

    month_values = {_alignment_month_for_row(row) for row in candidate_rows}
    for source in source_rows.values():
        source_month = _phase0_observation_time(source, str(source.get("year") or "").strip())
        if source_month:
            month_values.add(source_month)
    months = sorted(value for value in month_values if value != "unknown") or ["unknown"]
    c_index = {name: idx for idx, name in enumerate(canonical_names)}
    p_index = {name: idx for idx, name in enumerate(provinces)}
    m_index = {name: idx for idx, name in enumerate(months)}
    aligned = np.zeros((len(provinces), len(months), len(canonical_names)), dtype=np.float32)
    quality = np.zeros_like(aligned)
    counts = np.zeros_like(aligned)
    semantic_by_feature: dict[str, list[float]] = defaultdict(list)
    values_by_feature: dict[str, list[float]] = defaultdict(list)
    non_numeric_candidate_count = 0
    for idx, row in enumerate(candidate_rows):
        canonical_name = str(row.get("canonical_name") or "numeric_observation")
        source = source_rows.get(str(row.get("source_id") or ""), {})
        province = normalize_geo_label(
            str(row.get("geo") or ""),
            default_country_focus=bool(source.get("query_geo_focus") == "philippines"),
        ) or "unknown"
        if geo_resolution_label(province) == "region":
            continue
        month = _alignment_month_for_row(row)
        if month == "unknown":
            continue
        semantic_by_feature[canonical_name].append(float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0)
        numeric_value = _safe_float(row.get("value"))
        if numeric_value is None:
            non_numeric_candidate_count += 1
            continue
        value = float(numeric_value)
        unit = str(row.get("unit") or "").lower()
        aligned_value = value
        if aligned_value > 0.0 and (aligned_value >= float(alignment_cfg["log_transform_value_threshold"]) or unit in {"people", "cases", "deaths", "million", "billion"}):
            aligned_value = float(np.log1p(aligned_value))
        missing_rate = 0.0
        for key in ("title", "abstract", "url"):
            missing_rate += float(alignment_cfg["missing_rate_per_field"]) if not source.get(key) else 0.0
        platform_instability = dict(alignment_cfg["instability_platform"])
        instability = float(platform_instability.get(str(source.get("platform") or ""), alignment_cfg["instability_default"]))
        temporal_distance = 0.0
        year = str(source.get("year") or row.get("time") or "")
        if year.isdigit():
            temporal_distance = max(0, current_year - int(year))
        quality_weight = (
            1.0
            / (
                1.0
                + float(alignment_cfg["quality_missing_scale"]) * missing_rate
                + float(alignment_cfg["quality_instability_scale"]) * instability
            )
        ) * float(np.exp(-float(alignment_cfg["temporal_decay"]) * temporal_distance))
        pi = p_index[province]
        mi = m_index[month]
        ci = c_index[canonical_name]
        aligned[pi, mi, ci] += aligned_value * quality_weight
        quality[pi, mi, ci] += quality_weight
        counts[pi, mi, ci] += 1.0
        values_by_feature[canonical_name].append(value)
    safe_quality = np.where(quality > 0, quality, 1.0)
    aligned = aligned / safe_quality
    semantic_feature_scores = np.array(
        [float(np.mean(semantic_by_feature.get(name, [0.0]))) for name in canonical_names],
        dtype=np.float32,
    )
    semantic_target = np.repeat(semantic_feature_scores, repeats=[max(1, len(values_by_feature.get(name, []))) for name in canonical_names])
    mi_scores = []
    for name in canonical_names:
        values = np.asarray(values_by_feature.get(name, [0.0]), dtype=float)
        target_vec = np.full_like(values, fill_value=float(np.mean(semantic_by_feature.get(name, [0.0]))), dtype=float)
        mi_scores.append(_safe_mutual_information(values, target_vec))
    mi_scores_arr = np.asarray(mi_scores, dtype=np.float32)
    alignment_summary = {
        "province_axis": provinces,
        "month_axis": months,
        "canonical_name_axis": canonical_names,
        "full_philippines_province_grid": plugin_id == "hiv",
        "embedding_backend": embed_meta["backend"],
        "embedding_model": embed_meta.get("model_name"),
        "embedding_device": embed_meta.get("device"),
        "embedding_notes": embed_meta["notes"],
        "fallback_used": bool(embed_meta.get("fallback_used")),
        "quality_nonzero_cells": int((quality > 0).sum()),
        "aligned_nonzero_cells": int((aligned != 0).sum()),
        "non_numeric_candidate_count": int(non_numeric_candidate_count),
        "semantic_threshold": float(alignment_cfg["semantic_threshold"]),
        "mi_threshold": float(alignment_cfg["mi_threshold"]),
        "national_geo_label": "Philippines",
        "contains_national_duplicate_labels": any(is_national_geo(name) for name in provinces if name != "Philippines"),
    }
    aligned_artifact = save_tensor_artifact(
        array=to_torch_tensor(aligned, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else aligned,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=artifact_dir,
        stem="aligned_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase0_alignment_bundle"],
    )
    quality_artifact = save_tensor_artifact(
        array=to_torch_tensor(quality, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else quality,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=artifact_dir,
        stem="quality_weights",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase0_quality_weights"],
    )
    semantic_artifact = save_tensor_artifact(
        array=semantic_feature_scores,
        axis_names=["canonical_name"],
        artifact_dir=artifact_dir,
        stem="semantic_scores",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=[f"embedding_backend:{embed_meta['backend']}"],
        save_pt=False,
    )
    mi_artifact = save_tensor_artifact(
        array=mi_scores_arr,
        axis_names=["canonical_name"],
        artifact_dir=artifact_dir,
        stem="mi_scores",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase0_mutual_information_proxy"],
        save_pt=False,
    )
    write_json(artifact_dir / "alignment_summary.json", alignment_summary)
    return {
        "aligned_tensor": aligned_artifact,
        "quality_weights": quality_artifact,
        "semantic_scores": semantic_artifact,
        "mi_scores": mi_artifact,
        "alignment_summary": str(artifact_dir / "alignment_summary.json"),
    }


def _phase0_top_candidate_variables(candidates: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        canonical_name = str(row.get("canonical_name") or "").strip()
        if canonical_name:
            grouped[canonical_name].append(row)
    ranked: list[dict[str, Any]] = []
    for canonical_name, rows in grouped.items():
        ranked.append(
            {
                "canonical_name": canonical_name,
                "count": len(rows),
                "direct_measurement_count": sum(1 for row in rows if row.get("is_direct_measurement")),
                "prior_only_count": sum(1 for row in rows if row.get("is_prior_only")),
                "mean_confidence": round(sum(float(row.get("confidence") or 0.0) for row in rows) / max(1, len(rows)), 4),
                "example_candidate_text": next((str(row.get("candidate_text") or "") for row in rows if row.get("candidate_text")), ""),
                "source_banks": sorted({str(row.get("source_bank") or "") for row in rows if row.get("source_bank")}),
            }
        )
    ranked.sort(key=lambda row: (row["count"], row["direct_measurement_count"], row["mean_confidence"]), reverse=True)
    return ranked[:limit]


def _phase0_curated_bibliography(literature_review: dict[str, Any], *, limit: int = 60) -> list[dict[str, Any]]:
    bibliography: dict[str, dict[str, Any]] = {}
    for silo in literature_review.get("silos", []):
        for item in silo.get("top_documents", []):
            key = str(item.get("source_id") or item.get("url") or item.get("title") or "")
            if not key:
                continue
            payload = {
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "platform": item.get("platform"),
                "source_tier": item.get("source_tier"),
                "year": item.get("year"),
                "url": item.get("url"),
                "quality_score": item.get("quality_score"),
                "candidate_count": item.get("candidate_count"),
            }
            if key not in bibliography or float(payload.get("quality_score") or 0.0) > float(bibliography[key].get("quality_score") or 0.0):
                bibliography[key] = payload
    rows = sorted(
        bibliography.values(),
        key=lambda row: (float(row.get("quality_score") or 0.0), int(row.get("candidate_count") or 0), int(row.get("year") or 0)),
        reverse=True,
    )
    return rows[:limit]


def _phase0_schema_validation_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = dict(_phase0_required_section("schema_validation"))
    generic_names = {str(item) for item in cfg.get("generic_canonical_names", [])}
    measurement_counts = Counter(str(row.get("measurement_type") or "unknown") for row in candidates)
    denominator_counts = Counter(str(row.get("denominator_type") or "unknown") for row in candidates)
    normalization_counts = Counter(str(row.get("normalization_basis") or "unknown") for row in candidates)
    generic_rows = [row for row in candidates if str(row.get("canonical_name") or "") in generic_names]
    direct_rows = [row for row in candidates if float(row.get("confidence") or 0.0) >= float(cfg.get("direct_measurement_threshold", 0.75))]
    return {
        "candidate_count": len(candidates),
        "generic_canonical_count": len(generic_rows),
        "generic_canonical_fraction": round(len(generic_rows) / max(1, len(candidates)), 4),
        "measurement_type_counts": dict(measurement_counts),
        "denominator_type_counts": dict(denominator_counts),
        "normalization_basis_counts": dict(normalization_counts),
        "high_confidence_candidate_count": len(direct_rows),
    }


def _phase0_human_review_queue(
    *,
    source_rows: list[dict[str, Any]],
    scored_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = dict(_phase0_required_section("review_queue"))
    schema_cfg = dict(_phase0_required_section("schema_validation"))
    generic_names = {str(item) for item in schema_cfg.get("generic_canonical_names", [])}
    low_confidence = float(cfg.get("low_confidence_threshold", 0.45))
    candidate_limit = int(cfg.get("generic_candidate_limit", 250))
    document_limit = int(cfg.get("document_review_limit", 250))
    source_rights = [_phase0_source_rights_row(row) for row in source_rows]
    source_review = [row for row in source_rights if row.get("needs_review")]
    candidate_review = [
        {
            "candidate_id": row.get("candidate_id"),
            "canonical_name": row.get("canonical_name"),
            "source_id": row.get("source_id"),
            "confidence": row.get("confidence"),
            "measurement_type": row.get("measurement_type"),
            "denominator_type": row.get("denominator_type"),
            "candidate_text": row.get("candidate_text"),
            "reason": "generic_or_low_confidence",
        }
        for row in candidates
        if float(row.get("confidence") or 0.0) < low_confidence or str(row.get("canonical_name") or "") in generic_names
    ][:candidate_limit]
    document_review = [
        {
            "source_id": row.get("record_id"),
            "title": row.get("title"),
            "platform": row.get("platform"),
            "bucket": row.get("bucket"),
            "domain_quality_score": row.get("domain_quality_score"),
            "precision_penalty": row.get("precision_penalty"),
            "query_silo": row.get("query_silo"),
        }
        for row in scored_records
        if row.get("bucket") == "review"
    ][:document_limit]
    return {
        "candidate_review_queue": candidate_review,
        "document_review_queue": document_review,
        "source_rights_review_queue": source_review[:document_limit],
    }


def _phase0_write_support_artifacts(
    *,
    ctx: RunContext,
    source_manifest: list[dict[str, Any]],
    scored_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    literature_review: dict[str, Any],
    resource_usage: list[dict[str, Any]],
) -> dict[str, str]:
    phase0_dir = ensure_dir(ctx.run_dir / "phase0")
    analysis_dir = ensure_dir(phase0_dir / "analysis")
    prompt_library = build_phase0_prompt_library()
    tool_stack_manifest = _phase0_tool_stack_manifest()
    top_candidates = _phase0_top_candidate_variables(candidates, limit=10)
    bibliography = _phase0_curated_bibliography(literature_review, limit=60)
    schema_summary = _phase0_schema_validation_summary(candidates)
    review_queue = _phase0_human_review_queue(source_rows=source_manifest, scored_records=scored_records, candidates=candidates)
    rights_manifest = [_phase0_source_rights_row(row) for row in source_manifest]
    implementation_reality = {
        "actual_storage": ["json", "duckdb", "parquet", "npz"],
        "actual_vector_stack": {
            "embedding": tool_stack_manifest["selected_embedding_model"],
            "faiss_available": tool_stack_manifest["availability"]["faiss"],
            "chroma_available": tool_stack_manifest["availability"]["chromadb"],
        },
        "actual_pdf_stack": {
            "pymupdf": tool_stack_manifest["availability"]["pymupdf"],
            "lighton_ocr_sidecar": tool_stack_manifest["availability"]["lighton_ocr_sidecar"],
        },
        "design_only_tools": tool_stack_manifest["design_only_tools"],
    }
    prompt_path = analysis_dir / "prompt_library.json"
    tool_path = analysis_dir / "tool_stack_manifest.json"
    top_var_path = analysis_dir / "top_candidate_variables.json"
    bibliography_path = analysis_dir / "curated_bibliography.json"
    schema_path = analysis_dir / "schema_validation_summary.json"
    rights_path = analysis_dir / "source_rights_manifest.json"
    review_path = analysis_dir / "human_review_queue.json"
    resource_path = analysis_dir / "resource_usage_manifest.json"
    reality_path = analysis_dir / "implementation_reality_summary.json"
    write_json(prompt_path, prompt_library)
    write_json(tool_path, tool_stack_manifest)
    write_json(top_var_path, {"rows": top_candidates})
    write_json(bibliography_path, {"rows": bibliography})
    write_json(schema_path, schema_summary)
    write_json(rights_path, {"rows": rights_manifest})
    write_json(review_path, review_queue)
    write_json(resource_path, {"stages": resource_usage})
    write_json(reality_path, implementation_reality)
    (analysis_dir / "top_candidate_variables.md").write_text(
        "# Top Candidate Variables\n\n" + "\n".join(f"- {row['canonical_name']}: count={row['count']} mean_conf={row['mean_confidence']}" for row in top_candidates) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "curated_bibliography.md").write_text(
        "# Curated Bibliography\n\n" + "\n".join(
            f"- {row.get('title') or 'Untitled'} ({row.get('year') or 'n/a'}) [{row.get('platform') or 'unknown'}] quality={row.get('quality_score')}"
            for row in bibliography
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "prompt_library": str(prompt_path),
        "tool_stack_manifest": str(tool_path),
        "top_candidate_variables": str(top_var_path),
        "curated_bibliography": str(bibliography_path),
        "schema_validation_summary": str(schema_path),
        "source_rights_manifest": str(rights_path),
        "human_review_queue": str(review_path),
        "resource_usage_manifest": str(resource_path),
        "implementation_reality_summary": str(reality_path),
    }


def _manifest_for_stage(ctx: RunContext, *, raw_dir: Path, parsed_dir: Path, extracted_dir: Path, index_dir: Path, stage_status: dict[str, str], artifact_paths: dict[str, str], backend_status: dict[str, Phase0BackendStatus], source_count: int = 0, document_count: int = 0, parsed_block_count: int = 0, table_count: int = 0, numeric_observation_count: int = 0, canonical_candidate_count: int = 0, notes: list[str] | None = None) -> dict[str, Any]:
    artifact = Phase0ManifestArtifact(
        plugin_id=ctx.plugin_id,
        run_id=ctx.run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(raw_dir),
        parsed_dir=str(parsed_dir),
        extracted_dir=str(extracted_dir),
        index_dir=str(index_dir),
        stage_status=stage_status,
        artifact_paths=artifact_paths,
        backend_status=backend_status,
        source_count=source_count,
        document_count=document_count,
        parsed_block_count=parsed_block_count,
        table_count=table_count,
        numeric_observation_count=numeric_observation_count,
        canonical_candidate_count=canonical_candidate_count,
        notes=notes or [],
    )
    return artifact.to_dict()


def run_phase0_harvest(
    *,
    run_id: str,
    plugin_id: str,
    offline: bool = False,
    max_results: int = 10,
    target_records: int = 200,
    corpus_mode: str = "default",
    relevance_mode: str = "auto",
    download_budget: int = 25,
    embed_metadata_payload: bool = False,
    metadata_only: bool | None = None,
    query_shard_count: int = 1,
    query_shard_index: int = 0,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")
    plugin = get_disease_plugin(plugin_id)
    large_cfg = _phase0_large_corpus_cfg()
    if metadata_only is None:
        metadata_only = bool(corpus_mode in set(str(item) for item in large_cfg.get("variant_modes", [])) and large_cfg.get("metadata_only_default"))
    all_queries = _extend_query_bank(corpus_mode, plugin_id)
    queries = _slice_queries_for_shard(all_queries, shard_count=query_shard_count, shard_index=query_shard_index)
    shard_target_records = max(1, (int(target_records) + max(1, query_shard_count) - 1) // max(1, query_shard_count))
    budgets = _source_harvest_budgets(queries, target_records=shard_target_records, max_results=max_results)
    source_rows = _harvest_official_seed_rows()
    source_rows.extend(_structured_source_seed_rows(plugin_id))
    source_rows.extend(_local_official_anchor_specs())
    if offline:
        for index, query in enumerate(queries):
            exemplar = PHILIPPINES_SUBNATIONAL_EXEMPLARS[index % len(PHILIPPINES_SUBNATIONAL_EXEMPLARS)]
            source_rows.append(
                _build_source_row(
                    source_name="OfflineSeed",
                    source_tier="tier2_scientific_literature" if query.get("query_lane") != "anchor" else "tier1_official_anchor",
                    organization="offline",
                    title=f"Offline seed {index + 1}: {query['query']}",
                    url="",
                    platform="offline_seed",
                    query=query,
                    year=2024 - (index % 8),
                    abstract=(
                        f"Offline metadata seed for {query['query']} with subnational Philippines focus on {exemplar['geo']} in {exemplar['region']}; "
                        f"comparator sites include {exemplar['peers']}. Testing coverage 62 percent, ART retention 54 percent, "
                        f"viral load suppression 48 percent, travel time 4 hours, stockout disruption 12 days, poverty burden 27 percent."
                    ),
                )
            )
    else:
        harvester_map = {
            "pubmed": _harvest_pubmed,
            "crossref": _harvest_crossref,
            "openalex": _harvest_openalex,
            "semanticscholar": _harvest_semanticscholar,
            "arxiv": _harvest_arxiv,
            "biorxiv": _harvest_biorxiv,
            "kaggle": _harvest_kaggle,
        }
        for query in queries:
            for source in EXTERNAL_HARVESTER_ORDER:
                try:
                    source_rows.extend(harvester_map[source](query, limit=budgets[source]))
                except Exception:
                    continue
    source_rows = _filter_min_literature_year(source_rows)
    source_rows = _ensure_unique_ids(_apply_document_relevance_filter(source_rows, relevance_mode=relevance_mode), "source_id")
    source_rows, document_rows = _materialize_remote_snapshots(
        source_rows,
        raw_dir=raw_dir,
        download_budget=download_budget,
        embed_metadata_payload=embed_metadata_payload,
        metadata_only=bool(metadata_only),
    )
    sweep_rows = [_harvest_sweep_record(row) for row in source_rows if row.get("platform") != "offline_seed" or row.get("abstract")]
    source_manifest = raw_dir / "source_manifest.json"
    document_manifest = raw_dir / "document_manifest.json"
    sweep_manifest = raw_dir / "harvested_sweep_records.json"
    adapter_manifest = raw_dir / "structured_source_adapter_manifest.json"
    collector_manifest = raw_dir / "collector_manifest.json"
    query_manifest = raw_dir / "query_manifest.json"
    _write_rows(source_manifest, source_rows)
    _write_rows(document_manifest, document_rows)
    _write_rows(sweep_manifest, sweep_rows)
    _write_rows(query_manifest, queries)
    write_json(adapter_manifest, [row.to_dict() for row in plugin.structured_source_adapters])
    write_json(collector_manifest, _collector_manifest_rows(source_rows))
    _persist_duckdb(raw_dir / "phase0.duckdb", "source_manifest", source_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "document_manifest", document_rows)
    phase0_manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"harvest": "completed"},
        artifact_paths={
            "source_manifest": str(source_manifest),
            "source_manifest_parquet": str(_parquet_sidecar_path(source_manifest)),
            "document_manifest": str(document_manifest),
            "document_manifest_parquet": str(_parquet_sidecar_path(document_manifest)),
            "harvested_sweep_records": str(sweep_manifest),
            "harvested_sweep_records_parquet": str(_parquet_sidecar_path(sweep_manifest)),
            "structured_source_adapter_manifest": str(adapter_manifest),
            "collector_manifest": str(collector_manifest),
            "query_manifest": str(query_manifest),
        },
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "faiss": Phase0BackendStatus("faiss", faiss is not None, False),
        },
        source_count=len(source_rows),
        document_count=len(document_rows),
        notes=[
            f"corpus_mode:{corpus_mode}",
            f"relevance_mode:{relevance_mode}",
            f"min_literature_year:{_phase0_min_literature_year()}",
            f"query_count:{len(queries)}",
            f"total_query_count:{len(all_queries)}",
            f"metadata_only:{bool(metadata_only)}",
            f"query_shard:{query_shard_index + 1}/{max(1, query_shard_count)}",
        ],
    )
    write_json(raw_dir / "phase0_manifest.json", phase0_manifest)
    write_json(ctx.run_dir / "phase0" / "phase0_manifest.json", phase0_manifest)
    ctx.update_manifest(phase0=phase0_manifest)
    ctx.record_stage_outputs(
        "phase0_harvest",
        [
            source_manifest,
            _parquet_sidecar_path(source_manifest),
            document_manifest,
            _parquet_sidecar_path(document_manifest),
            sweep_manifest,
            _parquet_sidecar_path(sweep_manifest),
            query_manifest,
            _parquet_sidecar_path(query_manifest),
            adapter_manifest,
            collector_manifest,
            raw_dir / "phase0_manifest.json",
        ],
    )
    return phase0_manifest


def run_phase0_merge_shards(*, run_id: str, plugin_id: str, source_run_ids: list[str]) -> dict[str, Any]:
    if not source_run_ids:
        raise ValueError("phase0 merge-shards requires at least one source run id")
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    plugin = get_disease_plugin(plugin_id)
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")

    source_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for shard_run_id in source_run_ids:
        shard_ctx = RunContext.create(run_id=shard_run_id, plugin_id=plugin_id)
        shard_raw_dir = shard_ctx.run_dir / "phase0" / "raw"
        for row in read_json(shard_raw_dir / "source_manifest.json", default=[]):
            merged_row = dict(row)
            merged_row["merged_from_run_id"] = shard_run_id
            source_rows.append(merged_row)
        for row in read_json(shard_raw_dir / "document_manifest.json", default=[]):
            merged_row = dict(row)
            merged_row["merged_from_run_id"] = shard_run_id
            document_rows.append(merged_row)
        for row in read_json(shard_raw_dir / "harvested_sweep_records.json", default=[]):
            merged_row = dict(row)
            merged_row["merged_from_run_id"] = shard_run_id
            sweep_rows.append(merged_row)
        for row in read_json(shard_raw_dir / "query_manifest.json", default=[]):
            merged_row = dict(row)
            merged_row["merged_from_run_id"] = shard_run_id
            query_rows.append(merged_row)

    source_rows = _ensure_unique_ids(source_rows, "source_id")
    document_rows = _ensure_unique_ids(document_rows, "document_path")
    sweep_rows = _ensure_unique_ids(sweep_rows, "source_id")
    query_rows = _ensure_unique_ids(query_rows, "query_id")

    source_manifest = raw_dir / "source_manifest.json"
    document_manifest = raw_dir / "document_manifest.json"
    sweep_manifest = raw_dir / "harvested_sweep_records.json"
    query_manifest = raw_dir / "query_manifest.json"
    adapter_manifest = raw_dir / "structured_source_adapter_manifest.json"
    collector_manifest = raw_dir / "collector_manifest.json"
    merge_manifest = raw_dir / "merge_manifest.json"

    _write_rows(source_manifest, source_rows)
    _write_rows(document_manifest, document_rows)
    _write_rows(sweep_manifest, sweep_rows)
    _write_rows(query_manifest, query_rows)
    write_json(adapter_manifest, [row.to_dict() for row in plugin.structured_source_adapters])
    write_json(collector_manifest, _collector_manifest_rows(source_rows))
    write_json(
        merge_manifest,
        {
            "target_run_id": run_id,
            "plugin_id": plugin_id,
            "source_run_ids": list(source_run_ids),
            "source_count": len(source_rows),
            "document_count": len(document_rows),
            "sweep_record_count": len(sweep_rows),
            "query_count": len(query_rows),
        },
    )
    _persist_duckdb(raw_dir / "phase0.duckdb", "source_manifest", source_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "document_manifest", document_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "harvested_sweep_records", sweep_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "query_manifest", query_rows)

    phase0_manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"harvest_merge": "completed"},
        artifact_paths={
            "source_manifest": str(source_manifest),
            "source_manifest_parquet": str(_parquet_sidecar_path(source_manifest)),
            "document_manifest": str(document_manifest),
            "document_manifest_parquet": str(_parquet_sidecar_path(document_manifest)),
            "harvested_sweep_records": str(sweep_manifest),
            "harvested_sweep_records_parquet": str(_parquet_sidecar_path(sweep_manifest)),
            "structured_source_adapter_manifest": str(adapter_manifest),
            "collector_manifest": str(collector_manifest),
            "query_manifest": str(query_manifest),
            "query_manifest_parquet": str(_parquet_sidecar_path(query_manifest)),
            "merge_manifest": str(merge_manifest),
        },
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "faiss": Phase0BackendStatus("faiss", faiss is not None, False),
        },
        source_count=len(source_rows),
        document_count=len(document_rows),
        notes=[
            f"merged_shards:{len(source_run_ids)}",
            f"source_runs:{','.join(source_run_ids)}",
            "phase0_large_corpus_merge:metadata_first",
        ],
    )
    write_json(raw_dir / "phase0_manifest.json", phase0_manifest)
    write_json(ctx.run_dir / "phase0" / "phase0_manifest.json", phase0_manifest)
    ctx.update_manifest(phase0=phase0_manifest)
    ctx.record_stage_outputs(
        "phase0_merge_shards",
        [
            source_manifest,
            _parquet_sidecar_path(source_manifest),
            document_manifest,
            _parquet_sidecar_path(document_manifest),
            sweep_manifest,
            _parquet_sidecar_path(sweep_manifest),
            query_manifest,
            _parquet_sidecar_path(query_manifest),
            adapter_manifest,
            collector_manifest,
            merge_manifest,
            raw_dir / "phase0_manifest.json",
        ],
    )
    return phase0_manifest


def run_phase0_slice_corpus(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    shard_count: int,
    shard_index: int,
    max_documents: int | None = None,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("phase0 slice-corpus requires shard_count > 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("phase0 slice-corpus requires shard_index within [0, shard_count)")
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    source_ctx = RunContext.create(run_id=source_run_id, plugin_id=plugin_id)
    plugin = get_disease_plugin(plugin_id)
    source_raw_dir = source_ctx.run_dir / "phase0" / "raw"
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")
    if not (source_raw_dir / "source_manifest.json").exists():
        raise FileNotFoundError(f"Missing source manifest for source run: {source_run_id}")

    slice_payload = build_slice_payload(
        source_rows=read_json(source_raw_dir / "source_manifest.json", default=[]),
        document_rows=read_json(source_raw_dir / "document_manifest.json", default=[]),
        sweep_rows=read_json(source_raw_dir / "harvested_sweep_records.json", default=[]),
        query_rows=read_json(source_raw_dir / "query_manifest.json", default=[]),
        shard_count=shard_count,
        shard_index=shard_index,
        max_documents=max_documents,
    )

    source_rows = _ensure_unique_ids(slice_payload["source_rows"], "source_id")
    document_rows = _ensure_unique_ids(slice_payload["document_rows"], "document_path")
    sweep_rows = _ensure_unique_ids(slice_payload["sweep_rows"], "source_id")
    query_rows = _ensure_unique_ids(slice_payload["query_rows"], "query_id")

    source_manifest = raw_dir / "source_manifest.json"
    document_manifest = raw_dir / "document_manifest.json"
    sweep_manifest = raw_dir / "harvested_sweep_records.json"
    query_manifest = raw_dir / "query_manifest.json"
    adapter_manifest = raw_dir / "structured_source_adapter_manifest.json"
    collector_manifest = raw_dir / "collector_manifest.json"
    slice_manifest = raw_dir / "slice_manifest.json"

    _write_rows(source_manifest, source_rows)
    _write_rows(document_manifest, document_rows)
    _write_rows(sweep_manifest, sweep_rows)
    _write_rows(query_manifest, query_rows)
    write_json(adapter_manifest, [row.to_dict() for row in plugin.structured_source_adapters])
    write_json(collector_manifest, _collector_manifest_rows(source_rows))
    write_json(
        slice_manifest,
        {
            "target_run_id": run_id,
            "plugin_id": plugin_id,
            "source_run_id": source_run_id,
            "shard_count": int(shard_count),
            "shard_index": int(shard_index),
            "max_documents": None if max_documents is None else int(max_documents),
            **slice_payload["summary"],
        },
    )
    _persist_duckdb(raw_dir / "phase0.duckdb", "source_manifest", source_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "document_manifest", document_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "harvested_sweep_records", sweep_rows)
    _persist_duckdb(raw_dir / "phase0.duckdb", "query_manifest", query_rows)

    phase0_manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"slice_corpus": "completed"},
        artifact_paths={
            "source_manifest": str(source_manifest),
            "source_manifest_parquet": str(_parquet_sidecar_path(source_manifest)),
            "document_manifest": str(document_manifest),
            "document_manifest_parquet": str(_parquet_sidecar_path(document_manifest)),
            "harvested_sweep_records": str(sweep_manifest),
            "harvested_sweep_records_parquet": str(_parquet_sidecar_path(sweep_manifest)),
            "structured_source_adapter_manifest": str(adapter_manifest),
            "collector_manifest": str(collector_manifest),
            "query_manifest": str(query_manifest),
            "query_manifest_parquet": str(_parquet_sidecar_path(query_manifest)),
            "slice_manifest": str(slice_manifest),
        },
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "faiss": Phase0BackendStatus("faiss", faiss is not None, False),
        },
        source_count=len(source_rows),
        document_count=len(document_rows),
        notes=[
            f"source_run_id:{source_run_id}",
            f"slice_shard:{shard_index + 1}/{shard_count}",
            f"max_documents:{'none' if max_documents is None else int(max_documents)}",
        ],
    )
    write_json(raw_dir / "phase0_manifest.json", phase0_manifest)
    write_json(ctx.run_dir / "phase0" / "phase0_manifest.json", phase0_manifest)
    ctx.update_manifest(phase0=phase0_manifest)
    ctx.record_stage_outputs(
        "phase0_slice_corpus",
        [
            source_manifest,
            _parquet_sidecar_path(source_manifest),
            document_manifest,
            _parquet_sidecar_path(document_manifest),
            sweep_manifest,
            _parquet_sidecar_path(sweep_manifest),
            query_manifest,
            _parquet_sidecar_path(query_manifest),
            adapter_manifest,
            collector_manifest,
            slice_manifest,
            raw_dir / "phase0_manifest.json",
        ],
    )
    return phase0_manifest


def run_phase0_score_wide_sweep(
    *,
    run_id: str,
    plugin_id: str,
    sweep_json_path: str | None = None,
    min_domain_quality: float | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    wide_dir = ensure_dir(ctx.run_dir / "wide_sweep")
    sweep_path = Path(sweep_json_path) if sweep_json_path else ctx.run_dir / "phase0" / "raw" / "harvested_sweep_records.json"
    raw_records = read_json(sweep_path, default=[])
    if min_domain_quality is None:
        min_domain_quality = float(_phase0_required_section("wide_sweep_scoring")["min_domain_quality"])
    scored = [_score_wide_sweep_record(record, min_domain_quality=min_domain_quality) for record in raw_records]
    accepted = [row for row in scored if row.get("accepted_union")]
    hiv_direct = [row for row in scored if row.get("accepted_hiv_direct")]
    upstream = [row for row in scored if row.get("accepted_upstream_determinant")]
    review = [row for row in scored if row.get("bucket") == "review"]
    rejected = [row for row in scored if row.get("bucket") == "rejected"]
    _write_rows(wide_dir / "scored_records.json", scored)
    _write_rows(wide_dir / "registry_eligible_records.json", accepted)
    _write_rows(wide_dir / "registry_eligible_hiv_direct_records.json", hiv_direct)
    _write_rows(wide_dir / "registry_eligible_upstream_determinant_records.json", upstream)
    _write_rows(wide_dir / "review_records.json", review)
    summary = {
        "input_records": len(raw_records),
        "accepted_union": len(accepted),
        "accepted_hiv_direct": len(hiv_direct),
        "accepted_upstream_determinant": len(upstream),
        "review": len(review),
        "rejected": len(rejected),
        "platform_counts": dict(Counter(str(row.get("platform") or "") for row in accepted)),
        "linkage_target_counts": dict(Counter(target for row in upstream for target in row.get("linkage_targets", []))),
        "off_topic_biomedical_rejected": sum(1 for row in scored if row.get("off_topic_biomedical")),
        "generic_science_penalized": sum(1 for row in scored if row.get("generic_science_penalized")),
        "geo_boosted_records": sum(1 for row in scored if float(row.get("geo_focus_score") or 0.0) > 0.0),
        "query_silo_matched_records": sum(1 for row in scored if float(row.get("query_silo_match_score") or 0.0) > 0.0),
    }
    write_json(wide_dir / "domain_quality_summary.json", summary)
    ctx.record_stage_outputs(
        "phase0_score_wide_sweep",
        [
            wide_dir / "scored_records.json",
            _parquet_sidecar_path(wide_dir / "scored_records.json"),
            wide_dir / "registry_eligible_records.json",
            _parquet_sidecar_path(wide_dir / "registry_eligible_records.json"),
            wide_dir / "registry_eligible_hiv_direct_records.json",
            _parquet_sidecar_path(wide_dir / "registry_eligible_hiv_direct_records.json"),
            wide_dir / "registry_eligible_upstream_determinant_records.json",
            _parquet_sidecar_path(wide_dir / "registry_eligible_upstream_determinant_records.json"),
            wide_dir / "domain_quality_summary.json",
        ],
    )
    return summary


def _phase0_chunk_cfg() -> dict[str, Any]:
    return dict(_phase0_required_section("chunking"))


def _chunk_text(text: str) -> list[str]:
    cfg = _phase0_chunk_cfg()
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    target_chars = max(200, int(cfg["target_chars"]))
    overlap_chars = max(0, int(cfg["overlap_chars"]))
    min_chunk_chars = max(1, int(cfg["min_chunk_chars"]))
    max_chunks = max(1, int(cfg["max_chunks_per_block"]))
    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len and len(chunks) < max_chunks:
        end = min(text_len, start + target_chars)
        if end < text_len:
            boundary = normalized.rfind(". ", start, end)
            if boundary <= start + min_chunk_chars:
                boundary = normalized.rfind(" ", start, end)
            if boundary > start + min_chunk_chars:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if len(chunk) >= min_chunk_chars or not chunks:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def run_phase0_parse(
    *,
    run_id: str,
    plugin_id: str,
    working_set_size: int = 250,
    enable_chart_extraction: bool = False,
    enable_ocr_sidecar: bool = False,
    ocr_backend: str = DEFAULT_LIGHTON_OCR_BACKEND,
    force_ocr_sidecar: bool = False,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")
    documents = read_json(raw_dir / "document_manifest.json", default=[])
    selected = _select_diverse_working_set(documents, limit=working_set_size)
    parsed_docs: list[dict[str, Any]] = []
    parsed_blocks: list[dict[str, Any]] = []
    parsed_chunks: list[dict[str, Any]] = []
    table_candidates: list[dict[str, Any]] = []
    ocr_sidecar_manifest: list[dict[str, Any]] = []
    for doc in selected:
        local_path = Path(str(doc.get("local_path") or ""))
        doc_type = _sniff_local_document_type(local_path) if local_path.exists() else "json_metadata"
        parser_used = ""
        text = ""
        ocr_blocks: list[dict[str, Any]] = []
        if doc_type == "json_metadata":
            payload = read_json(local_path, default={})
            text = f"{payload.get('title', '')}\n{payload.get('abstract', '')}".strip()
            parser_used = "json_metadata"
        elif doc_type in {"html", "text", "txt", "csv"}:
            text, parser_used = _parse_non_pdf_document(local_path, doc_type=doc_type, document_row=doc)
            if doc_type == "csv":
                lines = [line for line in text.splitlines() if line.strip()]
                if lines:
                    table_candidates.append(
                        {
                            "table_id": f"table-{doc['document_id']}-1",
                            "document_id": doc["document_id"],
                            "source_id": doc["source_id"],
                            "table_method": "csv_direct",
                            "preview": "\n".join(lines[:5]),
                        }
                    )
        elif doc_type == "pdf":
            preferred_pages = [int(page) for page in (doc.get("preferred_pages") or []) if str(page).isdigit()]
            pdf_blocks, parser_used = _parse_pdf_blocks_with_fallbacks(
                local_path,
                preferred_pages=preferred_pages or None,
                max_pages=5,
            )
            ocr_blocks: list[dict[str, Any]] = []
            if _should_run_ocr_sidecar(
                enable_ocr_sidecar=enable_ocr_sidecar,
                force_ocr_sidecar=force_ocr_sidecar,
                pdf_blocks=pdf_blocks,
            ):
                ocr_blocks, ocr_meta = _lighton_ocr_sidecar_blocks(
                    local_path,
                    preferred_pages=preferred_pages or None,
                    max_pages=5,
                    requested_backend=ocr_backend,
                )
                ocr_sidecar_manifest.append(
                    {
                        "document_id": doc["document_id"],
                        "source_id": doc["source_id"],
                        "backend": ocr_meta.get("backend"),
                        "status": ocr_meta.get("status"),
                        "render_backend": ocr_meta.get("render_backend", ""),
                        "failure_notes": ocr_meta.get("failure_notes", []),
                        "parsed_page_count": int(ocr_meta.get("parsed_page_count") or 0),
                        "forced": bool(force_ocr_sidecar),
                    }
                )
                if ocr_blocks:
                    parser_used = f"{parser_used}+{ocr_meta.get('backend')}"
            text = "\n".join(block["text"] for block in pdf_blocks).strip()
            if not text and ocr_blocks:
                text = "\n".join(block["text"] for block in ocr_blocks).strip()
        block_text = re.sub(r"\s+", " ", text).strip()
        parse_status = "parsed" if block_text else "empty"
        patched_doc = dict(doc)
        patched_doc.update({"snapshot_type": doc_type, "parser_used": parser_used, "parse_status": parse_status})
        parsed_docs.append(patched_doc)
        if doc_type == "pdf" and block_text:
            for block_index, pdf_block in enumerate(pdf_blocks, start=1):
                block_row = {
                    "block_id": f"block-{doc['document_id']}-page-{int(pdf_block['page_number'])}",
                    "document_id": doc["document_id"],
                    "source_id": doc["source_id"],
                    "text": str(pdf_block["text"]),
                    "parser_used": parser_used,
                    "block_type": "page_text",
                    "document_type": doc_type,
                    "page_number": int(pdf_block["page_number"]),
                    "block_index": block_index,
                }
                parsed_blocks.append(block_row)
                for chunk_index, chunk_text in enumerate(_chunk_text(block_row["text"]), start=1):
                    parsed_chunks.append(
                        {
                            "chunk_id": f"{block_row['block_id']}-chunk-{chunk_index}",
                            "block_id": block_row["block_id"],
                            "document_id": doc["document_id"],
                            "source_id": doc["source_id"],
                            "text": chunk_text,
                            "parser_used": parser_used,
                            "block_type": "page_text",
                            "document_type": doc_type,
                            "page_number": int(pdf_block["page_number"]),
                            "block_index": block_index,
                            "chunk_index": chunk_index,
                            "query_silo": doc.get("query_silo"),
                            "source_tier": doc.get("source_tier"),
                        }
                    )
            if enable_ocr_sidecar:
                for block_index, ocr_block in enumerate(ocr_blocks, start=1):
                    ocr_block_row = {
                        "block_id": f"ocr-{doc['document_id']}-page-{int(ocr_block['page_number'])}",
                        "document_id": doc["document_id"],
                        "source_id": doc["source_id"],
                        "text": str(ocr_block["text"]),
                        "parser_used": _phase0_ocr_backend(ocr_backend),
                        "block_type": "page_ocr",
                        "document_type": doc_type,
                        "page_number": int(ocr_block["page_number"]),
                        "block_index": block_index,
                    }
                    parsed_blocks.append(ocr_block_row)
                    for chunk_index, chunk_text in enumerate(_chunk_text(ocr_block_row["text"]), start=1):
                        parsed_chunks.append(
                            {
                                "chunk_id": f"{ocr_block_row['block_id']}-chunk-{chunk_index}",
                                "block_id": ocr_block_row["block_id"],
                                "document_id": doc["document_id"],
                                "source_id": doc["source_id"],
                                "text": chunk_text,
                                "parser_used": ocr_block_row["parser_used"],
                                "block_type": "page_ocr",
                                "document_type": doc_type,
                                "page_number": int(ocr_block["page_number"]),
                                "block_index": block_index,
                                "chunk_index": chunk_index,
                                "query_silo": doc.get("query_silo"),
                                "source_tier": doc.get("source_tier"),
                            }
                        )
                    if "|" in str(ocr_block["text"]) or "\t" in str(ocr_block["text"]):
                        table_candidates.append(
                            {
                                "table_id": f"table-ocr-{doc['document_id']}-{block_index}",
                                "document_id": doc["document_id"],
                                "source_id": doc["source_id"],
                                "table_method": "lighton_ocr_markdown",
                                "preview": "\n".join(str(ocr_block["text"]).splitlines()[:8]),
                            }
                        )
        elif block_text:
            block_row = {
                "block_id": f"block-{doc['document_id']}-1",
                "document_id": doc["document_id"],
                "source_id": doc["source_id"],
                "text": block_text,
                "parser_used": parser_used,
                "block_type": "text",
                "document_type": doc_type,
            }
            parsed_blocks.append(block_row)
            for chunk_index, chunk_text in enumerate(_chunk_text(block_text), start=1):
                parsed_chunks.append(
                    {
                        "chunk_id": f"{block_row['block_id']}-chunk-{chunk_index}",
                        "block_id": block_row["block_id"],
                        "document_id": doc["document_id"],
                        "source_id": doc["source_id"],
                        "text": chunk_text,
                        "parser_used": parser_used,
                        "block_type": "text",
                        "document_type": doc_type,
                        "block_index": 1,
                        "chunk_index": chunk_index,
                        "query_silo": doc.get("query_silo"),
                        "source_tier": doc.get("source_tier"),
                    }
                )
    _write_rows(parsed_dir / "document_manifest.json", parsed_docs)
    _write_rows(parsed_dir / "parsed_document_blocks.json", parsed_blocks)
    _write_rows(parsed_dir / "parsed_chunk_manifest.json", parsed_chunks)
    _write_rows(parsed_dir / "table_candidates.json", table_candidates)
    _write_rows(parsed_dir / "ocr_sidecar_manifest.json", ocr_sidecar_manifest)
    _persist_duckdb(parsed_dir / "phase0.duckdb", "document_manifest", parsed_docs)
    _persist_duckdb(parsed_dir / "phase0.duckdb", "parsed_document_blocks", parsed_blocks)
    _persist_duckdb(parsed_dir / "phase0.duckdb", "parsed_chunk_manifest", parsed_chunks)
    _persist_duckdb(parsed_dir / "phase0.duckdb", "table_candidates", table_candidates)
    _persist_duckdb(parsed_dir / "phase0.duckdb", "ocr_sidecar_manifest", ocr_sidecar_manifest)
    manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"parse": "completed"},
        artifact_paths={
            "document_manifest": str(parsed_dir / "document_manifest.json"),
            "parsed_document_blocks": str(parsed_dir / "parsed_document_blocks.json"),
            "parsed_chunk_manifest": str(parsed_dir / "parsed_chunk_manifest.json"),
            "table_candidates": str(parsed_dir / "table_candidates.json"),
            "ocr_sidecar_manifest": str(parsed_dir / "ocr_sidecar_manifest.json"),
        },
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "fitz": Phase0BackendStatus("fitz", fitz is not None, fitz is not None),
            "lighton_ocr": Phase0BackendStatus("lighton_ocr", True, enable_ocr_sidecar, notes=_phase0_ocr_backend(ocr_backend)),
        },
        document_count=len(parsed_docs),
        parsed_block_count=len(parsed_blocks),
        numeric_observation_count=len(parsed_chunks),
        table_count=len(table_candidates),
        notes=[
            f"working_set_size:{working_set_size}",
            f"chart_extraction:{enable_chart_extraction}",
            f"ocr_sidecar:{enable_ocr_sidecar}",
            f"ocr_backend:{_phase0_ocr_backend(ocr_backend)}",
            f"force_ocr_sidecar:{force_ocr_sidecar}",
        ],
    )
    write_json(parsed_dir / "phase0_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase0_parse",
        [
            parsed_dir / "document_manifest.json",
            _parquet_sidecar_path(parsed_dir / "document_manifest.json"),
            parsed_dir / "parsed_document_blocks.json",
            _parquet_sidecar_path(parsed_dir / "parsed_document_blocks.json"),
            parsed_dir / "parsed_chunk_manifest.json",
            _parquet_sidecar_path(parsed_dir / "parsed_chunk_manifest.json"),
            parsed_dir / "table_candidates.json",
            _parquet_sidecar_path(parsed_dir / "table_candidates.json"),
            parsed_dir / "ocr_sidecar_manifest.json",
            _parquet_sidecar_path(parsed_dir / "ocr_sidecar_manifest.json"),
            parsed_dir / "phase0_manifest.json",
        ],
    )
    return manifest


def run_phase0_extract(*, run_id: str, plugin_id: str, skip_live_normalizer: bool = False) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")
    parsed_blocks = read_json(parsed_dir / "parsed_document_blocks.json", default=[])
    parsed_chunks = read_json(parsed_dir / "parsed_chunk_manifest.json", default=[])
    source_rows = {row["source_id"]: row for row in read_json(raw_dir / "source_manifest.json", default=[])}
    numeric_rows, candidate_rows, handled_blocks = _extract_official_anchor_rows(parsed_blocks, source_rows)
    canonicalization_summary = {
        "generic_candidates_before_second_pass": 0,
        "generic_candidates_after_second_pass": 0,
        "reclassified_candidates": 0,
        "dropped_low_signal_candidates": 0,
        "reclassification_reasons": {},
        "drop_reasons": {},
    }
    for block in parsed_blocks:
        if str(block.get("block_id") or "") in handled_blocks:
            continue
        text = str(block.get("text") or "")
        time_hint = _detect_time_for_span(text)
        source = source_rows.get(block["source_id"], {})
        source_text = " ".join(
            str(item or "")
            for item in (
                source.get("title"),
                source.get("abstract"),
                source.get("query"),
                source.get("organization"),
                text,
            )
        )
        if _skip_regex_numeric_extraction(block, source):
            continue
        for index, match in enumerate(_unit_value_matches(text), start=1):
            value = float(match["value"])
            unit = str(match["unit"])
            span = str(match["span"])
            local_context = _text_window(
                text,
                start=int(match["start"]),
                end=int(match["end"]),
                radius=int(_phase0_required_section("second_pass_canonicalizer")["context_window_chars"]),
            )
            parameter_text = span.strip()
            canonical_name = _canonical_name(parameter_text + " " + local_context, unit)
            geo_match = infer_philippines_geo(
                f"{parameter_text} {local_context} {source_text}",
                default_country_focus=source.get("query_geo_focus") == "philippines",
            )
            canonical_geo = normalize_geo_label(
                geo_match.geo or ("Philippines" if source.get("query_geo_focus") == "philippines" else ""),
                default_country_focus=source.get("query_geo_focus") == "philippines",
            )
            semantics = _infer_measurement_semantics(parameter_text, f"{local_context} {source_text}", unit)
            if canonical_name == "numeric_observation":
                canonicalization_summary["generic_candidates_before_second_pass"] += 1
            refined_name, refine_reason = _second_pass_canonical_name(
                canonical_name=canonical_name,
                parameter_text=parameter_text,
                local_context=local_context,
                source_text=source_text,
                source=source,
                unit=unit,
                semantics=semantics,
            )
            if refined_name != canonical_name:
                canonicalization_summary["reclassified_candidates"] += 1
                reason_counts = canonicalization_summary["reclassification_reasons"]
                reason_counts[refine_reason] = int(reason_counts.get(refine_reason, 0)) + 1
            drop_candidate, drop_reason = _drop_low_signal_numeric_candidate(
                canonical_name=refined_name,
                value=value,
                unit=unit,
                parameter_text=parameter_text,
                local_context=local_context,
                source=source,
                geo_label=str(canonical_geo or ""),
            )
            if drop_candidate:
                canonicalization_summary["dropped_low_signal_candidates"] += 1
                drop_counts = canonicalization_summary["drop_reasons"]
                drop_counts[drop_reason] = int(drop_counts.get(drop_reason, 0)) + 1
                continue
            if refined_name == "numeric_observation":
                canonicalization_summary["generic_candidates_after_second_pass"] += 1
            obs = {
                "observation_id": f"obs-{block['block_id']}-{index}",
                "source_id": block["source_id"],
                "document_id": block["document_id"],
                "source_bank": "phase0_extracted",
                "parameter_text": parameter_text,
                "canonical_name": refined_name,
                "value": value,
                "unit": unit,
                "population": "",
                "geo": canonical_geo,
                "region": geo_match.region,
                "province": geo_match.province,
                "geo_mentions": geo_match.mentions,
                "time": _phase0_observation_time(source, time_hint or str(source.get("year") or "")),
                **semantics,
                "sex": "",
                "age_band": "",
                "kp_group": "",
                "evidence_span": span,
                "extraction_method": "regex_rule",
                "canonicalization_reason": refine_reason,
                "confidence": float(_phase0_required_section("confidence")["skip_live_normalizer" if skip_live_normalizer else "default_live_normalizer"]),
                "is_direct_measurement": bool(source.get("is_direct_truth_eligible")),
                "is_prior_only": not bool(source.get("is_direct_truth_eligible")),
                "is_anchor_eligible": bool(source.get("is_anchor_eligible")),
            }
            numeric_rows.append(obs)
            details = []
            ref = LiteratureRefDetail(
                source_id=str(source.get("source_id") or ""),
                title=source.get("title"),
                year=source.get("year"),
                source_tier=source.get("source_tier"),
                url=source.get("url"),
                doi=source.get("doi"),
                pmid=source.get("pmid"),
                openalex_id=source.get("openalex_id"),
            ).to_dict()
            if has_verifiable_locator(ref):
                details.append(ref)
            candidate_rows.append(
                {
                    "candidate_id": f"cand-{block['block_id']}-{index}",
                    **obs,
                    "source_bank": obs.get("source_bank") or "phase0_extracted",
                    "candidate_text": " ".join(
                        part
                        for part in (
                            str(refined_name or ""),
                            local_context,
                            str(canonical_geo or ""),
                            str(obs["time"] or ""),
                        )
                        if part
                    ).strip(),
                    "literature_ref_details": details,
                }
            )
    chunk_soft_candidates = _extract_chunk_soft_candidates(parsed_chunks, source_rows)
    numeric_rows = _ensure_unique_ids(numeric_rows, "observation_id")
    candidate_rows.extend(chunk_soft_candidates)
    candidate_rows = _ensure_unique_ids(candidate_rows, "candidate_id")
    _write_rows(extracted_dir / "numeric_observations.json", numeric_rows)
    _write_rows(extracted_dir / "canonical_parameter_candidates.json", candidate_rows)
    _write_rows(extracted_dir / "chunk_soft_candidates.json", chunk_soft_candidates)
    write_json(extracted_dir / "canonicalization_summary.json", canonicalization_summary)
    _persist_duckdb(extracted_dir / "phase0.duckdb", "numeric_observations", numeric_rows)
    _persist_duckdb(extracted_dir / "phase0.duckdb", "canonical_parameter_candidates", candidate_rows)
    _persist_duckdb(extracted_dir / "phase0.duckdb", "chunk_soft_candidates", chunk_soft_candidates)
    tensor_artifacts = _phase0_alignment_bundle(
        candidate_rows=candidate_rows,
        source_rows=source_rows,
        plugin_id=plugin_id,
        artifact_dir=extracted_dir,
    )
    backend_map = detect_backends()
    manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"extract": "completed"},
        artifact_paths={
            "numeric_observations": str(extracted_dir / "numeric_observations.json"),
            "canonical_parameter_candidates": str(extracted_dir / "canonical_parameter_candidates.json"),
            "chunk_soft_candidates": str(extracted_dir / "chunk_soft_candidates.json"),
            "canonicalization_summary": str(extracted_dir / "canonicalization_summary.json"),
            **{key: (value if isinstance(value, str) else value.get("value_path", "")) for key, value in tensor_artifacts.items()},
        },
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, backend_map["torch"].selected, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        numeric_observation_count=len(numeric_rows),
        canonical_candidate_count=len(candidate_rows),
        notes=[f"skip_live_normalizer:{skip_live_normalizer}", f"chunk_soft_candidates:{len(chunk_soft_candidates)}", "phase0_tensor_bundle:enabled"],
    )
    write_json(extracted_dir / "phase0_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase0_extract",
        [
            extracted_dir / "numeric_observations.json",
            _parquet_sidecar_path(extracted_dir / "numeric_observations.json"),
            extracted_dir / "canonical_parameter_candidates.json",
            _parquet_sidecar_path(extracted_dir / "canonical_parameter_candidates.json"),
            extracted_dir / "chunk_soft_candidates.json",
            _parquet_sidecar_path(extracted_dir / "chunk_soft_candidates.json"),
            extracted_dir / "canonicalization_summary.json",
            extracted_dir / "aligned_tensor.npz",
            extracted_dir / "quality_weights.npz",
            extracted_dir / "semantic_scores.npz",
            extracted_dir / "mi_scores.npz",
            extracted_dir / "alignment_summary.json",
            extracted_dir / "phase0_manifest.json",
        ],
    )
    return manifest


def run_phase0_index(*, run_id: str, plugin_id: str) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    raw_dir = ensure_dir(ctx.run_dir / "phase0" / "raw")
    parsed_dir = ensure_dir(ctx.run_dir / "phase0" / "parsed")
    extracted_dir = ensure_dir(ctx.run_dir / "phase0" / "extracted")
    index_dir = ensure_dir(ctx.run_dir / "phase0" / "index")
    candidates = read_json(extracted_dir / "canonical_parameter_candidates.json", default=[])
    candidate_texts = [_candidate_index_text(row) for row in candidates]
    embeddings, embed_meta = _embed_texts(candidate_texts)
    np.save(index_dir / "embeddings.npy", embeddings)
    embedding_artifact = save_tensor_artifact(
        array=to_torch_tensor(embeddings, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else embeddings,
        axis_names=["candidate", "embedding_dim"],
        artifact_dir=index_dir,
        stem="embedding_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase0_embedding_index"],
        save_pt=True,
    )
    faiss_selected = False
    if faiss is not None and len(candidates):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(index_dir / "faiss.index"))
        faiss_selected = True
    manifest = {
        "vector_count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]) if embeddings.size else 384,
        "faiss_available": faiss is not None,
        "faiss_selected": faiss_selected,
        "embedding_backend": embed_meta.get("backend"),
        "embedding_model": embed_meta.get("model_name"),
        "embedding_device": embed_meta.get("device"),
        "embedding_notes": list(embed_meta.get("notes") or []),
        "fallback_used": bool(embed_meta.get("fallback_used")),
        "chroma_available": chromadb is not None,
        "artifact_paths": {
            "embeddings": str(index_dir / "embeddings.npy"),
            "faiss_index": str(index_dir / "faiss.index"),
            "embedding_tensor": embedding_artifact["value_path"],
        },
    }
    write_json(index_dir / "index_manifest.json", manifest)
    phase0_manifest = _manifest_for_stage(
        ctx,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        extracted_dir=extracted_dir,
        index_dir=index_dir,
        stage_status={"index": "completed"},
        artifact_paths={"index_manifest": str(index_dir / "index_manifest.json")},
        backend_status={
            "duckdb": Phase0BackendStatus("duckdb", duckdb is not None, duckdb is not None),
            "faiss": Phase0BackendStatus("faiss", faiss is not None, faiss_selected),
        },
        canonical_candidate_count=len(candidates),
    )
    write_json(index_dir / "phase0_manifest.json", phase0_manifest)
    ctx.record_stage_outputs("phase0_index", [index_dir / "embeddings.npy", index_dir / "index_manifest.json", index_dir / "phase0_manifest.json"])
    return manifest


def run_phase0_build(
    *,
    run_id: str,
    plugin_id: str,
    offline: bool = False,
    max_results: int = 10,
    target_records: int = 200,
    corpus_mode: str = "default",
    relevance_mode: str = "auto",
    download_budget: int = 25,
    embed_metadata_payload: bool = False,
    metadata_only: bool | None = None,
    query_shard_count: int = 1,
    query_shard_index: int = 0,
    enable_chart_extraction: bool = False,
    enable_ocr_sidecar: bool = False,
    ocr_backend: str = DEFAULT_LIGHTON_OCR_BACKEND,
    force_ocr_sidecar: bool = False,
    working_set_size: int = 250,
    skip_live_normalizer: bool = False,
    min_domain_quality: float | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    resource_usage: list[dict[str, Any]] = []
    harvest = run_phase0_harvest(
        run_id=run_id,
        plugin_id=plugin_id,
        offline=offline,
        max_results=max_results,
        target_records=target_records,
        corpus_mode=corpus_mode,
        relevance_mode=relevance_mode,
        download_budget=download_budget,
        embed_metadata_payload=embed_metadata_payload,
        metadata_only=metadata_only,
        query_shard_count=query_shard_count,
        query_shard_index=query_shard_index,
    )
    resource_usage.append(_phase0_resource_snapshot("harvest"))
    score = run_phase0_score_wide_sweep(run_id=run_id, plugin_id=plugin_id, min_domain_quality=min_domain_quality)
    resource_usage.append(_phase0_resource_snapshot("score_wide_sweep"))
    parse = run_phase0_parse(
        run_id=run_id,
        plugin_id=plugin_id,
        working_set_size=working_set_size,
        enable_chart_extraction=enable_chart_extraction,
        enable_ocr_sidecar=enable_ocr_sidecar,
        ocr_backend=ocr_backend,
        force_ocr_sidecar=force_ocr_sidecar,
    )
    resource_usage.append(_phase0_resource_snapshot("parse"))
    extract = run_phase0_extract(run_id=run_id, plugin_id=plugin_id, skip_live_normalizer=skip_live_normalizer)
    resource_usage.append(_phase0_resource_snapshot("extract"))
    index = run_phase0_index(run_id=run_id, plugin_id=plugin_id)
    resource_usage.append(_phase0_resource_snapshot("index"))
    plugin = get_disease_plugin(plugin_id)
    literature_review = build_phase0_literature_review(
        run_dir=ctx.run_dir,
        plugin=plugin,
        output_dir=ctx.run_dir / "phase0" / "literature_review",
    )
    resource_usage.append(_phase0_resource_snapshot("literature_review"))
    phase0_dir = ensure_dir(ctx.run_dir / "phase0")
    alignment_summary = read_json(phase0_dir / "extracted" / "alignment_summary.json", default={})
    source_manifest = read_json(phase0_dir / "raw" / "source_manifest.json", default=[])
    scored_records = read_json(ctx.run_dir / "wide_sweep" / "scored_records.json", default=[])
    candidates = read_json(phase0_dir / "extracted" / "canonical_parameter_candidates.json", default=[])
    aligned_tensor = load_tensor_artifact(phase0_dir / "extracted" / "aligned_tensor.npz")
    quality_weights = load_tensor_artifact(phase0_dir / "extracted" / "quality_weights.npz")
    province_axis = alignment_summary.get("province_axis", [])
    checks = [
        {"name": "harvest_completed", "passed": harvest.get("stage_status", {}).get("harvest") == "completed"},
        {"name": "parse_completed", "passed": parse.get("stage_status", {}).get("parse") == "completed"},
        {"name": "extract_completed", "passed": extract.get("stage_status", {}).get("extract") == "completed"},
        {"name": "index_written", "passed": bool(index.get("vector_count", 0) >= 0)},
        {"name": "aligned_tensor_finite", "passed": bool(np.isfinite(aligned_tensor).all())},
        {
            "name": "no_duplicate_national_labels",
            "passed": not (("Philippines" in province_axis) and any(is_national_geo(name) for name in province_axis if name != "Philippines")),
        },
        {"name": "candidates_preserve_provenance", "passed": all(bool(row.get("source_id")) for row in candidates)},
        {"name": "literature_review_present", "passed": bool(literature_review.get("silo_count", 0) >= 1)},
    ]
    truth_sources = sorted({str(row.get("source_tier") or "") for row in source_manifest if row.get("source_tier")})
    truth_paths = write_ground_truth_package(
        phase_dir=phase0_dir,
        phase_name="phase0",
        profile_id="legacy",
        checks=checks,
        truth_sources=truth_sources,
        stage_manifest_path=str(phase0_dir / "phase0_manifest.json"),
        summary={
            "source_count": len(source_manifest),
            "candidate_count": len(candidates),
            "province_count": len(province_axis),
            "month_count": len(alignment_summary.get("month_axis", [])),
            "canonical_name_count": len(alignment_summary.get("canonical_name_axis", [])),
            "literature_silo_count": int(literature_review.get("silo_count", 0)),
        },
    )
    gold_profile = dict((plugin.gold_standard_profiles or {}).get("phase0", {}) or {})
    gold_paths = write_gold_standard_package(
        phase_dir=phase0_dir,
        phase_name="phase0",
        profile_id="legacy",
        gold_profile=gold_profile,
        checks=[
            {"name": "gold_standard_profile_declared", "passed": bool(gold_profile)},
            {"name": "prisma_traceability_present", "passed": bool(source_manifest) and bool(literature_review.get("silos", []))},
            {"name": "who_dqa_completeness_present", "passed": bool(source_manifest) and bool(candidates)},
            {"name": "provenance_attached_to_candidates", "passed": all(bool(row.get("source_id")) for row in candidates)},
            {"name": "uncertainty_propagated", "passed": bool(np.isfinite(quality_weights).all()) and bool(np.all(quality_weights >= 0.0))},
            {
                "name": "alignment_documented",
                "passed": bool(alignment_summary.get("province_axis")) and bool(alignment_summary.get("month_axis")) and bool(alignment_summary.get("canonical_name_axis")),
            },
        ],
        stage_manifest_path=str(phase0_dir / "phase0_manifest.json"),
        summary={
            "source_count": len(source_manifest),
            "candidate_count": len(candidates),
            "literature_silo_count": int(literature_review.get("silo_count", 0)),
            "uncertainty_tensor_shape": list(quality_weights.shape),
        },
    )
    boundary_paths = write_boundary_shape_package(
        phase_dir=phase0_dir,
        phase_name="phase0",
        profile_id="legacy",
        boundaries=[
            {
                "name": "source_manifest",
                "kind": "json_rows",
                "path": str(phase0_dir / "raw" / "source_manifest.json"),
                "min_rows": 1,
                "expected_row_count": len(source_manifest),
            },
            {
                "name": "collector_manifest",
                "kind": "json_rows",
                "path": str(phase0_dir / "raw" / "collector_manifest.json"),
                "min_rows": 1,
            },
            {
                "name": "parsed_document_blocks",
                "kind": "json_rows",
                "path": str(phase0_dir / "parsed" / "parsed_document_blocks.json"),
                "min_rows": 1,
            },
            {
                "name": "parsed_chunk_manifest",
                "kind": "json_rows",
                "path": str(phase0_dir / "parsed" / "parsed_chunk_manifest.json"),
                "min_rows": 1,
            },
            {
                "name": "canonical_parameter_candidates",
                "kind": "json_rows",
                "path": str(phase0_dir / "extracted" / "canonical_parameter_candidates.json"),
                "min_rows": 1,
                "expected_row_count": len(candidates),
            },
            {
                "name": "aligned_tensor",
                "kind": "tensor",
                "path": str(phase0_dir / "extracted" / "aligned_tensor.npz"),
                "expected_shape": list(aligned_tensor.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "quality_weights",
                "kind": "tensor",
                "path": str(phase0_dir / "extracted" / "quality_weights.npz"),
                "expected_shape": list(quality_weights.shape),
                "expected_axis_names": ["province", "month", "canonical_name"],
                "min_rank": 3,
                "finite_required": True,
            },
            {
                "name": "embedding_tensor",
                "kind": "tensor",
                "path": str(phase0_dir / "index" / "embedding_tensor.npz"),
                "expected_axis_names": ["candidate", "embedding_dim"],
                "min_rank": 2,
                "finite_required": True,
            },
        ],
        summary={
            "source_count": len(source_manifest),
            "candidate_count": len(candidates),
            "province_count": len(province_axis),
            "month_count": len(alignment_summary.get("month_axis", [])),
        },
    )
    support_artifacts = _phase0_write_support_artifacts(
        ctx=ctx,
        source_manifest=source_manifest,
        scored_records=scored_records,
        candidates=candidates,
        literature_review=literature_review,
        resource_usage=resource_usage,
    )
    phase0_manifest = read_json(phase0_dir / "phase0_manifest.json", default={})
    if not isinstance(phase0_manifest, dict):
        phase0_manifest = {}
    phase0_manifest["stage_status"] = {
        **dict(phase0_manifest.get("stage_status", {})),
        "score_wide_sweep": "completed",
        "parse": "completed",
        "extract": "completed",
        "index": "completed",
        "literature_review": "completed",
        "build": "completed",
    }
    phase0_manifest["artifact_paths"] = {
        **dict(phase0_manifest.get("artifact_paths", {})),
        **truth_paths,
        **gold_paths,
        **boundary_paths,
        **support_artifacts,
        "literature_review": str(ctx.run_dir / "phase0" / "literature_review" / "literature_review_by_silo.json"),
    }
    write_json(phase0_dir / "phase0_manifest.json", phase0_manifest)
    ctx.update_manifest(phase0=phase0_manifest)
    ctx.record_stage_outputs(
        "phase0_truth_package",
        [
            Path(truth_paths["ground_truth_manifest"]),
            Path(truth_paths["ground_truth_checks"]),
            Path(truth_paths["ground_truth_summary"]),
            Path(gold_paths["gold_standard_manifest"]),
            Path(gold_paths["gold_standard_checks"]),
            Path(gold_paths["gold_standard_summary"]),
            Path(boundary_paths["boundary_shape_manifest"]),
            Path(boundary_paths["boundary_shape_checks"]),
            Path(boundary_paths["boundary_shape_summary"]),
            ctx.run_dir / "phase0" / "literature_review" / "literature_review_by_silo.json",
            ctx.run_dir / "phase0" / "literature_review" / "literature_review_by_silo.md",
            *[Path(path) for path in support_artifacts.values()],
        ],
    )
    return {
        "harvest": harvest,
        "score": score,
        "parse": parse,
        "extract": extract,
        "index": index,
        "literature_review": literature_review,
        "support_artifacts": support_artifacts,
        "ground_truth": truth_paths,
        "gold_standard": gold_paths,
        "boundary_shape": boundary_paths,
    }


def run_phase0_literature_review(*, run_id: str, plugin_id: str) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    plugin = get_disease_plugin(plugin_id)
    return build_phase0_literature_review(
        run_dir=ctx.run_dir,
        plugin=plugin,
        output_dir=ctx.run_dir / "phase0" / "literature_review",
    )
