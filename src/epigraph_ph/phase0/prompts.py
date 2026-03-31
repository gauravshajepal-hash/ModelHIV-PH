from __future__ import annotations

from typing import Any


PHASE0_PROMPT_LIBRARY: dict[str, dict[str, Any]] = {
    "canonicalize_subparameter": {
        "id": "canonicalize_subparameter",
        "task": "Map a raw extracted field into a canonical Phase 0 subparameter.",
        "inputs": ["parameter_text", "candidate_text", "unit", "geo", "time", "source_tier", "query_silo"],
        "output_schema": {
            "canonical_name": "snake_case canonical name",
            "measurement_type": "count|rate|share|cost|capacity|binary|ordinal|qualitative_support",
            "denominator_type": "population|plhiv|kp_population|household|facility|route|visit|none|unknown",
            "value_semantics": "direct_observed|bounded_proxy|supporting_context|prior_only",
            "confidence": "0-1",
            "reasoning": "brief evidence-backed explanation",
        },
        "prompt": (
            "You are canonicalizing epidemiologic extraction candidates for a probabilistic HIV modeling pipeline. "
            "Given a raw field, map it to the narrowest valid canonical subparameter already used by the pipeline when possible. "
            "Do not invent a novel canonical name when an existing one is a better match. "
            "Return measurement_type, denominator_type, and whether the value is a direct observation, proxy, or prior-only support."
        ),
    },
    "infer_denominator": {
        "id": "infer_denominator",
        "task": "Infer denominator semantics for a numeric extraction.",
        "inputs": ["parameter_text", "candidate_text", "unit", "source_title", "query_silo"],
        "output_schema": {
            "denominator_type": "population|plhiv|kp_population|household|facility|route|visit|none|unknown",
            "normalization_basis": "per_capita|per_100k|percent|share|absolute|currency|time_delay|unknown",
            "confidence": "0-1",
            "reasoning": "brief explanation",
        },
        "prompt": (
            "Infer the denominator and normalization basis from the surrounding text. "
            "Prefer explicit denominator semantics such as population, PLHIV, key population, household, facility, route, or visit. "
            "If the evidence is weak, return unknown rather than forcing a denominator."
        ),
    },
    "table_to_rows": {
        "id": "table_to_rows",
        "task": "Convert OCR or PDF table text into row-wise candidate observations.",
        "inputs": ["table_markdown", "source_title", "page_number", "query_silo", "source_tier"],
        "output_schema": {
            "rows": [
                {
                    "parameter_text": "raw label",
                    "value": "numeric or null",
                    "unit": "raw unit",
                    "geo": "raw geography",
                    "time": "raw time",
                    "notes": "short extraction note",
                }
            ]
        },
        "prompt": (
            "Convert the table into row-wise structured observations. Preserve units, geography, and time labels exactly when visible. "
            "Do not summarize or aggregate across rows."
        ),
    },
    "chunk_support_summary": {
        "id": "chunk_support_summary",
        "task": "Summarize a literature chunk into soft candidate support signals.",
        "inputs": ["chunk_text", "query_silo", "source_title", "year", "source_tier"],
        "output_schema": {
            "canonical_supports": ["ordered list of canonical subparameters"],
            "soft_ontology_tags": ["ordered list of soft ontology tags"],
            "linkage_targets": ["ordered list of cascade linkage targets"],
            "confidence": "0-1",
        },
        "prompt": (
            "Read the chunk as supporting-context evidence for a probabilistic HIV model. "
            "Return only canonical support signals and cascade linkage targets that are actually supported by the text."
        ),
    },
}


def build_phase0_prompt_library() -> dict[str, Any]:
    return {
        "prompt_count": len(PHASE0_PROMPT_LIBRARY),
        "prompts": [dict(value) for value in PHASE0_PROMPT_LIBRARY.values()],
    }
