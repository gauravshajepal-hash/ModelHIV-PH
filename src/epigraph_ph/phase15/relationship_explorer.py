from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ensure_dir, write_json

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _relationship_rows(profiles: list[Any], similarity_edges: list[dict[str, Any]], *, limit: int = 250) -> list[dict[str, Any]]:
    profile_by_name = {str(profile.canonical_name): profile for profile in profiles}
    scored_rows: list[dict[str, Any]] = []
    for edge in similarity_edges:
        left = profile_by_name.get(str(edge.get("left") or ""))
        right = profile_by_name.get(str(edge.get("right") or ""))
        if left is None or right is None:
            continue
        semantic_index = (
            0.45 * float(edge.get("semantic_sim") or 0.0)
            + 0.30 * float(edge.get("corr_sim") or 0.0)
            + 0.15 * float(edge.get("target_sim") or 0.0)
            + 0.10 * float(edge.get("geo_sim") or 0.0)
        )
        support_mass = (
            float(edge.get("left_numeric_support") or 0.0)
            + float(edge.get("right_numeric_support") or 0.0)
            + float(edge.get("left_anchor_support") or 0.0)
            + float(edge.get("right_anchor_support") or 0.0)
        )
        scored_rows.append(
            {
                "left": str(edge.get("left") or ""),
                "right": str(edge.get("right") or ""),
                "block_name": str(edge.get("block_name") or ""),
                "similarity_score": round(float(edge.get("similarity_score") or 0.0), 6),
                "semantic_index": round(float(semantic_index), 6),
                "corr_sim": round(float(edge.get("corr_sim") or 0.0), 6),
                "semantic_sim": round(float(edge.get("semantic_sim") or 0.0), 6),
                "target_sim": round(float(edge.get("target_sim") or 0.0), 6),
                "geo_sim": round(float(edge.get("geo_sim") or 0.0), 6),
                "lane_sim": round(float(edge.get("lane_sim") or 0.0), 6),
                "signature_cosine": round(float(edge.get("signature_cosine") or 0.0), 6),
                "support_mass": round(float(support_mass), 6),
                "left_block": str(getattr(left, "block_name", "")),
                "right_block": str(getattr(right, "block_name", "")),
                "left_source_mix": dict(getattr(left, "source_mix", {}) or {}),
                "right_source_mix": dict(getattr(right, "source_mix", {}) or {}),
                "left_geo_resolutions": dict(getattr(left, "geo_resolutions", {}) or {}),
                "right_geo_resolutions": dict(getattr(right, "geo_resolutions", {}) or {}),
            }
        )
    scored_rows.sort(
        key=lambda row: (
            float(row["semantic_index"]),
            float(row["similarity_score"]),
            float(row["support_mass"]),
        ),
        reverse=True,
    )
    return scored_rows[:limit]


def _write_bubble_chart(rows: list[dict[str, Any]], *, output_path: Path) -> str | None:
    if plt is None or not rows:
        return None
    blocks = sorted({str(row.get("block_name") or "unknown") for row in rows})
    color_map = {block: idx for idx, block in enumerate(blocks)}
    xs = [float(row.get("semantic_sim") or 0.0) for row in rows]
    ys = [float(row.get("corr_sim") or 0.0) for row in rows]
    sizes = [120.0 + 900.0 * float(row.get("similarity_score") or 0.0) + 30.0 * float(row.get("support_mass") or 0.0) for row in rows]
    colors = [color_map[str(row.get("block_name") or "unknown")] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="tab20", alpha=0.65, edgecolors="black", linewidths=0.4)
    ax.set_title("Phase 1.5 Subparameter Relationship Bubble Chart")
    ax.set_xlabel("Semantic Similarity")
    ax.set_ylabel("Observed Surface Correlation")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    top_rows = rows[:12]
    for row in top_rows:
        ax.annotate(
            f"{row['left']} <-> {row['right']}",
            (float(row.get("semantic_sim") or 0.0), float(row.get("corr_sim") or 0.0)),
            fontsize=7,
            alpha=0.85,
        )
    handles = []
    for block, idx in color_map.items():
        handles.append(ax.scatter([], [], s=120, color=scatter.cmap(scatter.norm(idx)), label=block))
    if handles:
        ax.legend(handles=handles[:12], title="Block", fontsize=7, title_fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def build_relationship_explorer_artifacts(
    *,
    profiles: list[Any],
    similarity_edges: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    phase15_dir = ensure_dir(output_dir)
    relationship_rows = _relationship_rows(profiles, similarity_edges)
    block_counts = Counter(str(row.get("block_name") or "unknown") for row in relationship_rows)
    payload = {
        "relationship_count": len(relationship_rows),
        "block_counts": dict(block_counts),
        "top_relationships": relationship_rows[:25],
        "rows": relationship_rows,
    }
    index_path = phase15_dir / "semantic_relationship_index.json"
    write_json(index_path, payload)
    chart_path = _write_bubble_chart(relationship_rows, output_path=phase15_dir / "semantic_relationship_bubble_chart.png")
    return {
        "semantic_relationship_index": str(index_path),
        "semantic_relationship_bubble_chart": chart_path,
        "relationship_count": len(relationship_rows),
    }
