from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import json

from epigraph_ph.phase0.semantic_benchmark import run_phase0_semantic_benchmark
from epigraph_ph.runtime import RunContext, ensure_dir, read_json, write_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _safe_year(value: Any) -> int | None:
    try:
        year = int(value)
    except Exception:
        return None
    return year if 1900 <= year <= 2100 else None


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _plot_bar(counter: Counter[str], *, title: str, output_path: Path, rotate: bool = False) -> str:
    if plt is None or not counter:
        return ""
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2a6f97")
    ax.set_title(title)
    ax.set_ylabel("Count")
    if rotate:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _plot_retrieval(candidate_summary: dict[str, Any], chunk_summary: dict[str, Any], *, output_path: Path) -> str:
    if plt is None:
        return ""
    candidate_systems = candidate_summary.get("systems", {})
    chunk_systems = chunk_summary.get("systems", {})
    systems = sorted(set(candidate_systems) | set(chunk_systems))
    if not systems:
        return ""
    candidate_values = [float(candidate_systems.get(system, {}).get("mean_ndcg_at_k", 0.0)) for system in systems]
    chunk_values = [float(chunk_systems.get(system, {}).get("mean_ndcg_at_k", 0.0)) for system in systems]
    x = range(len(systems))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.36
    ax.bar([idx - width / 2 for idx in x], candidate_values, width=width, label="Numeric candidates", color="#457b9d")
    ax.bar([idx + width / 2 for idx in x], chunk_values, width=width, label="Parsed chunks", color="#e76f51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, rotation=25, ha="right")
    ax.set_ylabel("Mean nDCG@k")
    ax.set_title("Phase 0 Retrieval Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def run_phase0_pilot_report(*, run_id: str, plugin_id: str, top_k: int = 10) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase0_dir = ensure_dir(ctx.run_dir / "phase0")
    analysis_dir = ensure_dir(phase0_dir / "analysis")
    raw_dir = ensure_dir(phase0_dir / "raw")
    parsed_dir = ensure_dir(phase0_dir / "parsed")
    extracted_dir = ensure_dir(phase0_dir / "extracted")
    registry_dir = ensure_dir(ctx.run_dir / "registry")

    candidate_summary = run_phase0_semantic_benchmark(run_id=run_id, plugin_id=plugin_id, top_k=top_k)
    chunk_manifest_path = parsed_dir / "parsed_chunk_manifest.json"
    chunk_summary = run_phase0_semantic_benchmark(
        run_id=run_id,
        plugin_id=plugin_id,
        candidate_json_path=str(chunk_manifest_path),
        top_k=top_k,
    )
    candidate_benchmark_path = analysis_dir / "candidate_semantic_quality_benchmark.json"
    chunk_benchmark_path = analysis_dir / "chunk_semantic_quality_benchmark.json"
    write_json(candidate_benchmark_path, candidate_summary)
    write_json(chunk_benchmark_path, chunk_summary)

    source_rows = read_json(raw_dir / "source_manifest.json", default=[])
    parsed_blocks = read_json(parsed_dir / "parsed_document_blocks.json", default=[])
    ocr_rows = read_json(parsed_dir / "ocr_sidecar_manifest.json", default=[])
    candidates = read_json(extracted_dir / "canonical_parameter_candidates.json", default=[])
    literature_review = read_json(phase0_dir / "literature_review" / "literature_review_by_silo.json", default={})
    subparameter_registry = read_json(registry_dir / "subparameter_registry.json", default={})
    top_candidate_variables = read_json(analysis_dir / "top_candidate_variables.json", default={})
    curated_bibliography = read_json(analysis_dir / "curated_bibliography.json", default={})
    schema_summary = read_json(analysis_dir / "schema_validation_summary.json", default={})
    canonicalization_summary = read_json(extracted_dir / "canonicalization_summary.json", default={})
    tool_stack_manifest = read_json(analysis_dir / "tool_stack_manifest.json", default={})
    resource_usage = read_json(analysis_dir / "resource_usage_manifest.json", default={})

    platform_counts = Counter(str(row.get("platform") or "unknown") for row in source_rows)
    silo_counts = Counter(str(row.get("query_silo") or "unknown") for row in source_rows)
    parser_counts = Counter(str(row.get("parser_used") or "unknown") for row in parsed_blocks)
    candidate_name_counts = Counter(str(row.get("canonical_name") or "unknown") for row in candidates)
    ocr_status_counts = Counter(str(row.get("status") or "unknown") for row in ocr_rows)
    year_counts = Counter(str(year) for year in (_safe_year(row.get("year")) for row in source_rows) if year is not None and year >= 2010)

    coverage_platform_path = _plot_bar(platform_counts, title="Phase 0 Source Coverage by Platform", output_path=analysis_dir / "phase0_platform_coverage.png", rotate=True)
    coverage_silo_path = _plot_bar(silo_counts, title="Phase 0 Source Coverage by Silo", output_path=analysis_dir / "phase0_silo_coverage.png", rotate=True)
    parser_path = _plot_bar(parser_counts, title="Phase 0 Parsed Blocks by Parser", output_path=analysis_dir / "phase0_parser_counts.png", rotate=True)
    ocr_path = _plot_bar(ocr_status_counts, title="Phase 0 OCR Sidecar Status", output_path=analysis_dir / "phase0_ocr_status.png", rotate=True)
    year_path = _plot_bar(year_counts, title="Phase 0 Literature Year Distribution", output_path=analysis_dir / "phase0_year_distribution.png", rotate=True)
    retrieval_path = _plot_retrieval(candidate_summary, chunk_summary, output_path=analysis_dir / "phase0_retrieval_quality.png")
    top_variable_counter = Counter({str(row.get("canonical_name") or ""): int(row.get("count") or 0) for row in top_candidate_variables.get("rows", [])})
    top_variable_path = _plot_bar(top_variable_counter, title="Top Candidate Variables", output_path=analysis_dir / "phase0_top_candidate_variables.png", rotate=True)

    report = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "coverage": {
            "source_count": len(source_rows),
            "platform_counts": dict(platform_counts),
            "silo_counts": dict(silo_counts),
            "year_counts": dict(year_counts),
        },
        "parsing": {
            "parsed_block_count": len(parsed_blocks),
            "parser_counts": dict(parser_counts),
            "ocr_status_counts": dict(ocr_status_counts),
            "ocr_render_backends": dict(Counter(str(row.get("render_backend") or "unknown") for row in ocr_rows)),
        },
        "extraction": {
            "candidate_count": len(candidates),
            "top_canonical_names": dict(candidate_name_counts.most_common(15)),
            "top_candidate_variables": top_candidate_variables.get("rows", []),
            "schema_validation": schema_summary,
            "canonicalization_summary": canonicalization_summary,
        },
        "retrieval": {
            "candidate_benchmark_path": str(candidate_benchmark_path),
            "chunk_benchmark_path": str(chunk_benchmark_path),
            "candidate_systems": candidate_summary.get("systems", {}),
            "chunk_systems": chunk_summary.get("systems", {}),
        },
        "literature_review": {
            "silo_count": int(literature_review.get("silo_count") or 0),
            "subparameter_count": int(subparameter_registry.get("subparameter_count") or 0),
            "curated_bibliography_count": len(curated_bibliography.get("rows", [])),
        },
        "tooling": tool_stack_manifest,
        "resource_usage": resource_usage,
        "graph_paths": {
            "platform_coverage": coverage_platform_path,
            "silo_coverage": coverage_silo_path,
            "year_distribution": year_path,
            "parser_counts": parser_path,
            "ocr_status": ocr_path,
            "retrieval_quality": retrieval_path,
            "top_candidate_variables": top_variable_path,
        },
        "scale_strategy": {
            "harvest": "Use metadata-only shard harvests first for 5k+ and merge shards before downloads.",
            "parse": "Keep a bounded diverse working set for local parse/OCR and avoid full-corpus PDF parsing on Windows.",
            "extract": "Sanitize HTML and extract on parsed working sets before any large-corpus expansion.",
            "index": "Use Chroma for chunk retrieval evaluation and FAISS or hashed baselines as comparators.",
        },
    }

    report_path = analysis_dir / "phase0_pilot_report.json"
    write_json(report_path, report)

    md_lines = [
        f"# Phase 0 Pilot Report: {run_id}",
        "",
        f"- sources: `{len(source_rows)}`",
        f"- parsed blocks: `{len(parsed_blocks)}`",
        f"- extracted numeric candidates: `{len(candidates)}`",
        f"- OCR sidecar rows: `{len(ocr_rows)}`",
        f"- literature silos reviewed: `{int(literature_review.get('silo_count') or 0)}`",
        f"- registry subparameters: `{int(subparameter_registry.get('subparameter_count') or 0)}`",
        f"- curated bibliography rows: `{len(curated_bibliography.get('rows', []))}`",
        f"- generic numeric candidates reclassified: `{int(canonicalization_summary.get('reclassified_candidates') or 0)}`",
        f"- low-signal generic candidates dropped: `{int(canonicalization_summary.get('dropped_low_signal_candidates') or 0)}`",
        "",
        "## Retrieval Quality",
        "",
        "| system | numeric candidates mean nDCG@k | parsed chunks mean nDCG@k |",
        "|---|---:|---:|",
    ]
    candidate_systems = candidate_summary.get("systems", {})
    chunk_systems = chunk_summary.get("systems", {})
    for system in sorted(set(candidate_systems) | set(chunk_systems)):
        md_lines.append(
            f"| {system} | {float(candidate_systems.get(system, {}).get('mean_ndcg_at_k', 0.0)):.4f} | {float(chunk_systems.get(system, {}).get('mean_ndcg_at_k', 0.0)):.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## Graphs",
            "",
            f"- platform coverage: `{report['graph_paths']['platform_coverage']}`",
            f"- silo coverage: `{report['graph_paths']['silo_coverage']}`",
            f"- year distribution: `{report['graph_paths']['year_distribution']}`",
            f"- parser counts: `{report['graph_paths']['parser_counts']}`",
            f"- OCR status: `{report['graph_paths']['ocr_status']}`",
            f"- retrieval quality: `{report['graph_paths']['retrieval_quality']}`",
            f"- top candidate variables: `{report['graph_paths']['top_candidate_variables']}`",
            "",
            "## Tooling Reality",
            "",
            f"- selected embedding model: `{tool_stack_manifest.get('selected_embedding_model', '')}`",
            f"- selected OCR backend: `{tool_stack_manifest.get('selected_ocr_backend', '')}`",
        ]
    )
    _write_markdown(analysis_dir / "phase0_pilot_report.md", "\n".join(md_lines))
    return report
