from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import RunContext, ensure_dir, read_json, write_json

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from .pipeline import (
    _candidate_index_text,
    _embed_texts,
    _hashed_embedding,
    _score_linkage_targets,
    _soft_ontology_tags,
    _soft_subparameter_hints,
)


SEMANTIC_BENCHMARK_QUERIES = [
    {
        "query_id": "q_sexual_risk",
        "query": "peer-driven sexual risk norms and partner concurrency that raise exposure pressure",
        "preferred_silo": "sexual_risk",
        "expected_linkage_targets": ["prevention_access"],
        "expected_soft_tags": ["behavior"],
        "expected_hints": [],
        "keywords": ["sexual", "risk", "condom", "partner", "norm", "behavior"],
    },
    {
        "query_id": "q_prevention_access",
        "query": "last-mile access to condoms prep and prevention commodities",
        "preferred_silo": "prevention_access",
        "expected_linkage_targets": ["prevention_access"],
        "expected_soft_tags": ["behavior", "policy"],
        "expected_hints": ["service_delivery_reach"],
        "keywords": ["prevention", "access", "prep", "condom", "commodities"],
    },
    {
        "query_id": "q_testing_uptake",
        "query": "fear, stigma, and weak outreach reducing willingness to get tested",
        "preferred_silo": "testing_uptake",
        "expected_linkage_targets": ["testing_uptake"],
        "expected_soft_tags": ["behavior"],
        "expected_hints": ["stigma_barrier"],
        "keywords": ["testing", "uptake", "screening", "diagnosis", "stigma"],
    },
    {
        "query_id": "q_linkage",
        "query": "delays between diagnosis referral and treatment start",
        "preferred_silo": "linkage_to_care",
        "expected_linkage_targets": ["linkage_to_care"],
        "expected_soft_tags": ["policy"],
        "expected_hints": ["service_delivery_reach"],
        "keywords": ["linkage", "referral", "clinic", "treatment", "start"],
    },
    {
        "query_id": "q_retention",
        "query": "loss to follow up and interruptions in continuity of care",
        "preferred_silo": "retention_adherence",
        "expected_linkage_targets": ["retention_adherence"],
        "expected_soft_tags": ["behavior", "policy"],
        "expected_hints": ["service_delivery_reach"],
        "keywords": ["retention", "adherence", "loss", "follow", "continuity"],
    },
    {
        "query_id": "q_suppression",
        "query": "documented viral suppression and viral load monitoring turnaround",
        "preferred_silo": "suppression_outcomes",
        "expected_linkage_targets": ["suppression_outcomes"],
        "expected_soft_tags": ["biology"],
        "expected_hints": ["biological_progression_modifier"],
        "keywords": ["suppression", "viral", "load", "monitoring", "documented"],
    },
    {
        "query_id": "q_mobility",
        "query": "network mixing from commuting and migration between places",
        "preferred_silo": "mobility_network_mixing",
        "expected_linkage_targets": ["mobility_network_mixing"],
        "expected_soft_tags": ["logistics", "population"],
        "expected_hints": ["mobility_friction", "labor_migration"],
        "keywords": ["mobility", "network", "mixing", "migration", "commuting"],
    },
    {
        "query_id": "q_health_system",
        "query": "weak health-system reach and thin clinic coverage in hard-to-serve areas",
        "preferred_silo": "health_system_reach",
        "expected_linkage_targets": ["health_system_reach"],
        "expected_soft_tags": ["policy", "logistics"],
        "expected_hints": ["service_delivery_reach"],
        "keywords": ["health", "system", "reach", "facility", "coverage", "clinic"],
    },
    {
        "query_id": "q_poverty",
        "query": "poverty and affordability barriers that interrupt care seeking",
        "preferred_silo": "poverty",
        "expected_linkage_targets": ["prevention_access"],
        "expected_soft_tags": ["economics"],
        "expected_hints": ["economic_access_constraint"],
        "keywords": ["poverty", "income", "affordability", "expenditure"],
    },
    {
        "query_id": "q_transport",
        "query": "transport friction and travel burden to treatment hubs",
        "preferred_silo": "transport_friction",
        "expected_linkage_targets": ["health_system_reach", "mobility_network_mixing"],
        "expected_soft_tags": ["logistics"],
        "expected_hints": ["mobility_friction", "remoteness", "congestion_travel_time"],
        "keywords": ["transport", "travel", "time", "ferry", "congestion", "remote"],
    },
    {
        "query_id": "q_cash_instability",
        "query": "income shocks and unstable cash flow disrupting treatment continuity",
        "preferred_silo": "cash_instability",
        "expected_linkage_targets": ["retention_adherence"],
        "expected_soft_tags": ["economics"],
        "expected_hints": ["cash_instability"],
        "keywords": ["cash", "income", "liquidity", "shock", "financial"],
    },
    {
        "query_id": "q_labor_migration",
        "query": "migrant work and remittance cycles affecting clinic continuity",
        "preferred_silo": "labor_migration",
        "expected_linkage_targets": ["mobility_network_mixing"],
        "expected_soft_tags": ["population", "economics", "logistics"],
        "expected_hints": ["labor_migration", "mobility_friction"],
        "keywords": ["labor", "migration", "migrant", "worker", "remittance"],
    },
    {
        "query_id": "q_social_capital",
        "query": "community trust and social capital supporting care engagement",
        "preferred_silo": "social_capital",
        "expected_linkage_targets": ["testing_uptake", "retention_adherence"],
        "expected_soft_tags": ["social_capital", "behavior"],
        "expected_hints": ["social_capital"],
        "keywords": ["social", "capital", "trust", "community", "support"],
    },
    {
        "query_id": "q_housing",
        "query": "unstable housing and shelter insecurity worsening retention",
        "preferred_silo": "housing_precarity",
        "expected_linkage_targets": ["retention_adherence"],
        "expected_soft_tags": ["housing", "economics"],
        "expected_hints": ["housing_precarity"],
        "keywords": ["housing", "shelter", "precarity", "eviction"],
    },
    {
        "query_id": "q_education",
        "query": "education and health literacy shaping testing and prevention",
        "preferred_silo": "education",
        "expected_linkage_targets": ["testing_uptake", "prevention_access"],
        "expected_soft_tags": ["education", "behavior"],
        "expected_hints": ["education"],
        "keywords": ["education", "schooling", "literacy", "health literacy"],
    },
    {
        "query_id": "q_policy_implementation",
        "query": "implementation weakness and governance bottlenecks in service delivery",
        "preferred_silo": "policy_implementation_weakness",
        "expected_linkage_targets": ["health_system_reach", "linkage_to_care"],
        "expected_soft_tags": ["policy"],
        "expected_hints": ["policy_implementation_weakness"],
        "keywords": ["policy", "implementation", "governance", "bottleneck"],
    },
]


def _dcg(scores: list[float]) -> float:
    total = 0.0
    for idx, score in enumerate(scores, start=1):
        total += (2.0**score - 1.0) / math.log2(idx + 1.0)
    return total


def _ndcg_at_k(scores: list[float], *, k: int) -> float:
    observed = scores[:k]
    ideal = sorted(scores, reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0.0:
        return 0.0
    return _dcg(observed) / ideal_dcg


def _average_precision_at_k(scores: list[float], *, k: int, threshold: float = 1.0) -> float:
    hits = 0
    precision_total = 0.0
    for idx, score in enumerate(scores[:k], start=1):
        if score >= threshold:
            hits += 1
            precision_total += hits / idx
    if hits == 0:
        return 0.0
    return precision_total / hits


def _candidate_relevance(
    row: dict[str, Any],
    source_row: dict[str, Any] | None,
    query_spec: dict[str, Any],
) -> float:
    text = str(row.get("candidate_text") or row.get("text") or _candidate_index_text(row)).lower()
    linkage_targets = set(_score_linkage_targets(text))
    soft_tags = set(_soft_ontology_tags(text))
    hints = set(_soft_subparameter_hints(text))
    score = 0.0
    for target in query_spec["expected_linkage_targets"]:
        if target in linkage_targets:
            score += 1.4
    for tag in query_spec["expected_soft_tags"]:
        if tag in soft_tags:
            score += 0.9
    for hint in query_spec["expected_hints"]:
        if hint in hints:
            score += 1.0
    keyword_hits = sum(1 for keyword in query_spec["keywords"] if keyword in text)
    score += min(1.2, 0.25 * keyword_hits)
    if source_row and source_row.get("query_silo") == query_spec.get("preferred_silo"):
        score += 0.8
    if source_row and str(source_row.get("source_tier") or "").startswith("tier1"):
        score += 0.2
    return round(min(score, 4.0), 4)


def _build_relevance_labels(
    candidates: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    query_spec: dict[str, Any],
) -> list[float]:
    return [
        _candidate_relevance(row, source_rows.get(str(row.get("source_id") or "")), query_spec)
        for row in candidates
    ]


def _rank_by_inner_product(
    query_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    if candidate_matrix.size == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    matrix = np.asarray(candidate_matrix, dtype=np.float32)
    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        _, ids = index.search(query, top_k)
        return [int(idx) for idx in ids[0] if int(idx) >= 0]
    sims = (matrix @ query_vector).astype(np.float32)
    ranked = np.argsort(-sims)
    return [int(idx) for idx in ranked[:top_k]]


def _rank_with_chroma(
    query_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    *,
    candidate_ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    top_k: int,
    preferred_silo: str | None,
    filtered: bool,
) -> list[int]:
    if chromadb is None or candidate_matrix.size == 0:
        return []
    batch_size = 2048
    if hasattr(chromadb, "EphemeralClient"):
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(name="phase0_semantic_quality")
        embedding_rows = np.asarray(candidate_matrix, dtype=np.float32).tolist()
        for start in range(0, len(candidate_ids), batch_size):
            stop = min(len(candidate_ids), start + batch_size)
            collection.add(
                ids=candidate_ids[start:stop],
                documents=documents[start:stop],
                embeddings=embedding_rows[start:stop],
                metadatas=metadatas[start:stop],
            )
        where = {"query_silo": preferred_silo} if filtered and preferred_silo else None
        result = collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=min(top_k, len(candidate_ids)),
            where=where,
        )
        ranked_ids = result.get("ids", [[]])[0]
    else:  # pragma: no cover - compatibility fallback
        with tempfile.TemporaryDirectory(prefix="epigraph_chroma_") as tmpdir:
            client = chromadb.PersistentClient(path=tmpdir)
            collection = client.get_or_create_collection(name="phase0_semantic_quality")
            embedding_rows = np.asarray(candidate_matrix, dtype=np.float32).tolist()
            for start in range(0, len(candidate_ids), batch_size):
                stop = min(len(candidate_ids), start + batch_size)
                collection.add(
                    ids=candidate_ids[start:stop],
                    documents=documents[start:stop],
                    embeddings=embedding_rows[start:stop],
                    metadatas=metadatas[start:stop],
                )
            where = {"query_silo": preferred_silo} if filtered and preferred_silo else None
            result = collection.query(
                query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
                n_results=min(top_k, len(candidate_ids)),
                where=where,
            )
            ranked_ids = result.get("ids", [[]])[0]
    id_to_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}
    return [id_to_index[candidate_id] for candidate_id in ranked_ids if candidate_id in id_to_index]


def run_phase0_semantic_benchmark(
    *,
    run_id: str,
    plugin_id: str,
    candidate_json_path: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase0_dir = ensure_dir(ctx.run_dir / "phase0")
    extracted_dir = ensure_dir(phase0_dir / "extracted")
    analysis_dir = ensure_dir(phase0_dir / "analysis")
    candidate_path = Path(candidate_json_path) if candidate_json_path else extracted_dir / "canonical_parameter_candidates.json"
    source_path = phase0_dir / "raw" / "source_manifest.json"
    candidates = read_json(candidate_path, default=[])
    source_rows = {str(row.get("source_id") or ""): row for row in read_json(source_path, default=[])}
    candidate_texts = [str(row.get("candidate_text") or row.get("text") or _candidate_index_text(row)) for row in candidates]
    candidate_ids = [str(row.get("candidate_id") or row.get("chunk_id") or row.get("block_id") or f"candidate-{idx}") for idx, row in enumerate(candidates)]
    metadatas = [
        {
            "query_silo": str(source_rows.get(str(row.get("source_id") or ""), {}).get("query_silo") or "unknown"),
            "source_tier": str(source_rows.get(str(row.get("source_id") or ""), {}).get("source_tier") or "unknown"),
            "canonical_name": str(row.get("canonical_name") or "unknown"),
        }
        for row in candidates
    ]

    hashed_matrix = np.vstack([_hashed_embedding(text) for text in candidate_texts]) if candidate_texts else np.zeros((0, 384), dtype=np.float32)
    local_matrix, local_meta = _embed_texts(candidate_texts)
    if not candidate_texts:
        local_meta = dict(local_meta)
        local_meta["fallback_used"] = True

    systems = {
        "hashed_local_faiss": {"matrix": hashed_matrix, "kind": "faiss", "backend": "hashed_local", "filtered": False},
        "local_embedder_faiss": {"matrix": local_matrix, "kind": "faiss", "backend": local_meta.get("backend"), "filtered": False},
    }
    if chromadb is not None:
        systems["local_embedder_chroma"] = {"matrix": local_matrix, "kind": "chroma", "backend": local_meta.get("backend"), "filtered": False}
        systems["local_embedder_chroma_filtered"] = {"matrix": local_matrix, "kind": "chroma", "backend": local_meta.get("backend"), "filtered": True}

    per_query_results: list[dict[str, Any]] = []
    system_scores: dict[str, dict[str, list[float]]] = {
        name: {"ndcg": [], "map": [], "avg_relevance": [], "top1": []}
        for name in systems
    }

    for query_spec in SEMANTIC_BENCHMARK_QUERIES:
        hashed_query = _hashed_embedding(query_spec["query"])
        local_query, local_query_meta = _embed_texts([query_spec["query"]])
        local_query_vector = local_query[0] if local_query.shape[0] else hashed_query
        relevance_scores = _build_relevance_labels(candidates, source_rows, query_spec)
        query_result = {
            "query_id": query_spec["query_id"],
            "query": query_spec["query"],
            "preferred_silo": query_spec.get("preferred_silo"),
            "relevant_candidate_count": int(sum(1 for score in relevance_scores if score > 0.0)),
            "systems": {},
        }
        for system_name, system in systems.items():
            if system_name == "hashed_local_faiss":
                query_vector = hashed_query
            else:
                query_vector = local_query_vector
            if system["kind"] == "faiss":
                ranked = _rank_by_inner_product(query_vector, system["matrix"], top_k=top_k)
            else:
                ranked = _rank_with_chroma(
                    query_vector,
                    system["matrix"],
                    candidate_ids=candidate_ids,
                    documents=candidate_texts,
                    metadatas=metadatas,
                    top_k=top_k,
                    preferred_silo=query_spec.get("preferred_silo"),
                    filtered=bool(system["filtered"]),
                )
            ranked_scores = [relevance_scores[idx] for idx in ranked]
            ndcg = _ndcg_at_k(ranked_scores, k=top_k)
            avg_precision = _average_precision_at_k(ranked_scores, k=top_k)
            avg_relevance = float(np.mean(ranked_scores)) if ranked_scores else 0.0
            top1_relevance = float(ranked_scores[0]) if ranked_scores else 0.0
            system_scores[system_name]["ndcg"].append(ndcg)
            system_scores[system_name]["map"].append(avg_precision)
            system_scores[system_name]["avg_relevance"].append(avg_relevance)
            system_scores[system_name]["top1"].append(top1_relevance)
            query_result["systems"][system_name] = {
                "ndcg_at_k": round(ndcg, 4),
                "average_precision_at_k": round(avg_precision, 4),
                "average_relevance_at_k": round(avg_relevance, 4),
                "top1_relevance": round(top1_relevance, 4),
                "top_hits": [
                    {
                        "candidate_id": candidate_ids[idx],
                        "canonical_name": candidates[idx].get("canonical_name"),
                        "parameter_text": candidates[idx].get("parameter_text"),
                        "text_excerpt": str(candidate_texts[idx])[:200],
                        "source_tier": source_rows.get(str(candidates[idx].get("source_id") or ""), {}).get("source_tier"),
                        "query_silo": source_rows.get(str(candidates[idx].get("source_id") or ""), {}).get("query_silo"),
                        "relevance": relevance_scores[idx],
                    }
                    for idx in ranked[: min(5, len(ranked))]
                ],
            }
        query_result["local_query_backend"] = local_query_meta.get("backend")
        per_query_results.append(query_result)

    summary = {
        "candidate_count": len(candidates),
        "top_k": top_k,
        "systems": {},
        "query_results": per_query_results,
        "local_embedder": {
            "backend": local_meta.get("backend"),
            "model_name": local_meta.get("model_name"),
            "device": local_meta.get("device"),
            "fallback_used": bool(local_meta.get("fallback_used")),
            "notes": list(local_meta.get("notes") or []),
        },
        "chroma_available": chromadb is not None,
        "faiss_available": faiss is not None,
        "interpretation": {},
    }
    for system_name, metrics in system_scores.items():
        summary["systems"][system_name] = {
            "mean_ndcg_at_k": round(float(np.mean(metrics["ndcg"])) if metrics["ndcg"] else 0.0, 4),
            "mean_average_precision_at_k": round(float(np.mean(metrics["map"])) if metrics["map"] else 0.0, 4),
            "mean_average_relevance_at_k": round(float(np.mean(metrics["avg_relevance"])) if metrics["avg_relevance"] else 0.0, 4),
            "mean_top1_relevance": round(float(np.mean(metrics["top1"])) if metrics["top1"] else 0.0, 4),
        }
    baseline = summary["systems"].get("hashed_local_faiss", {})
    upgraded = summary["systems"].get("local_embedder_faiss", {})
    chroma_same = summary["systems"].get("local_embedder_chroma", {})
    chroma_filtered = summary["systems"].get("local_embedder_chroma_filtered", {})
    summary["interpretation"] = {
        "embedder_gain_over_hashed_ndcg": round(float(upgraded.get("mean_ndcg_at_k", 0.0) - baseline.get("mean_ndcg_at_k", 0.0)), 4),
        "embedder_gain_over_hashed_avg_relevance": round(float(upgraded.get("mean_average_relevance_at_k", 0.0) - baseline.get("mean_average_relevance_at_k", 0.0)), 4),
        "chroma_delta_vs_faiss_ndcg": round(float(chroma_same.get("mean_ndcg_at_k", 0.0) - upgraded.get("mean_ndcg_at_k", 0.0)), 4),
        "chroma_filtered_delta_vs_faiss_ndcg": round(float(chroma_filtered.get("mean_ndcg_at_k", 0.0) - upgraded.get("mean_ndcg_at_k", 0.0)), 4),
        "semantic_winner": max(summary["systems"].items(), key=lambda item: item[1].get("mean_ndcg_at_k", 0.0))[0] if summary["systems"] else "none",
    }

    write_json(analysis_dir / "semantic_quality_benchmark.json", summary)
    markdown_lines = [
        "# Phase 0 Semantic Quality Benchmark",
        "",
        f"- candidate count: `{len(candidates)}`",
        f"- top-k: `{top_k}`",
        f"- local embedder: `{summary['local_embedder']['backend']}` / `{summary['local_embedder']['model_name']}`",
        "",
        "## System Summary",
        "",
    ]
    for system_name, metrics in summary["systems"].items():
        markdown_lines.append(
            f"- `{system_name}`: ndcg@{top_k} `{metrics['mean_ndcg_at_k']}`, map@{top_k} `{metrics['mean_average_precision_at_k']}`, avg relevance `{metrics['mean_average_relevance_at_k']}`"
        )
    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- semantic winner: `{summary['interpretation']['semantic_winner']}`",
            f"- local embedder gain over hashed ndcg: `{summary['interpretation']['embedder_gain_over_hashed_ndcg']}`",
            f"- local embedder gain over hashed avg relevance: `{summary['interpretation']['embedder_gain_over_hashed_avg_relevance']}`",
            f"- chroma delta vs faiss ndcg: `{summary['interpretation']['chroma_delta_vs_faiss_ndcg']}`",
            f"- chroma filtered delta vs faiss ndcg: `{summary['interpretation']['chroma_filtered_delta_vs_faiss_ndcg']}`",
        ]
    )
    (analysis_dir / "semantic_quality_benchmark.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    ctx.record_stage_outputs(
        "phase0_semantic_benchmark",
        [
            analysis_dir / "semantic_quality_benchmark.json",
            analysis_dir / "semantic_quality_benchmark.md",
        ],
    )
    return summary
