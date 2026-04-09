from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epigraph_ph.phase0.pipeline import _extend_query_bank
from epigraph_ph.runtime import ensure_dir, read_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _query_rows(run_dir: Path) -> list[dict[str, Any]]:
    return read_json(run_dir / "phase0" / "raw" / "query_manifest.json", default=[])


def _survivor_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pool = read_json(run_dir / "phase15" / "factor_survival_pool.json", default=[])
    catalog = read_json(run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    catalog_by_id = {str(row.get("factor_id") or ""): row for row in catalog}
    survivors = [
        row
        for row in pool
        if str(row.get("promotion_class") or "") in {"survivor_primary", "survivor_secondary"}
    ]
    survivors.sort(
        key=lambda row: (
            _safe_float(row.get("survival_score")),
            _safe_float(row.get("holdout_mae_improvement")),
            _safe_float(row.get("holdout_smape_improvement")),
        ),
        reverse=True,
    )
    return survivors, catalog, catalog_by_id


def _relationship_rows(run_dir: Path) -> list[dict[str, Any]]:
    relationship_index = read_json(run_dir / "phase15" / "semantic_relationship_index.json", default={})
    return list(relationship_index.get("rows") or [])


def _phase3_tournament(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "phase3_frozen_backtest_tournament" / "representation_tournament.json", default={})


def _plot_survivor_scores(survivors: list[dict[str, Any]], *, output_path: Path) -> str:
    if plt is None or not survivors:
        return ""
    labels = [str(row.get("factor_name") or row.get("factor_id") or "") for row in survivors]
    values = [_safe_float(row.get("survival_score")) for row in survivors]
    colors = []
    for row in survivors:
        representation = str(row.get("representation_type") or "")
        if representation == "unclumped":
            colors.append("#1d3557")
        elif representation == "clumped":
            colors.append("#2a9d8f")
        elif representation == "network":
            colors.append("#e76f51")
        else:
            colors.append("#6c757d")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(labels))))
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Survival Score")
    ax.set_title("Phase 1.5 Surviving Factors")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _factor_member_map(
    survivors: list[dict[str, Any]],
    *,
    catalog_by_id: dict[str, dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relationship_rows:
        by_block[str(row.get("block_name") or "")].append(row)
    for rows in by_block.values():
        rows.sort(key=lambda row: (_safe_float(row.get("semantic_index")), _safe_float(row.get("similarity_score"))), reverse=True)

    member_map: dict[str, list[str]] = {}
    for survivor in survivors:
        factor_id = str(survivor.get("factor_id") or "")
        catalog_row = catalog_by_id.get(factor_id, {})
        members = [str(item) for item in catalog_row.get("member_canonical_names") or [] if str(item)]
        if members:
            member_map[factor_id] = members[:6]
            continue
        block_name = str(survivor.get("block_name") or "")
        anchors: list[str] = []
        for row in by_block.get(block_name, []):
            for key in ("left", "right"):
                value = str(row.get(key) or "")
                if value and value not in anchors:
                    anchors.append(value)
                if len(anchors) >= 2:
                    break
            if len(anchors) >= 2:
                break
        member_map[factor_id] = anchors[:2] if anchors else [str(survivor.get("factor_name") or factor_id)]
    return member_map


def _survivor_relationship_subset(
    survivors: list[dict[str, Any]],
    *,
    member_map: dict[str, list[str]],
    relationship_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed = {item for members in member_map.values() for item in members}
    subset = [
        row
        for row in relationship_rows
        if str(row.get("left") or "") in allowed and str(row.get("right") or "") in allowed
    ]
    subset.sort(
        key=lambda row: (
            _safe_float(row.get("semantic_index")),
            _safe_float(row.get("similarity_score")),
            _safe_float(row.get("support_mass")),
        ),
        reverse=True,
    )
    return subset[:12]


def _plot_survivor_graph(
    survivors: list[dict[str, Any]],
    *,
    member_map: dict[str, list[str]],
    relationship_rows: list[dict[str, Any]],
    output_path: Path,
) -> str:
    if plt is None or not survivors:
        return ""
    fig, ax = plt.subplots(figsize=(14, 8))
    factor_positions: dict[str, tuple[float, float]] = {}
    member_positions: dict[str, tuple[float, float]] = {}
    factor_xs = [idx for idx in range(len(survivors))]
    top_y = 0.85
    member_y = 0.3

    for idx, survivor in enumerate(survivors):
        factor_id = str(survivor.get("factor_id") or "")
        factor_label = str(survivor.get("factor_name") or factor_id)
        x = float(factor_xs[idx])
        factor_positions[factor_id] = (x, top_y)
        ax.scatter([x], [top_y], s=1100, color="#264653", alpha=0.9, zorder=3)
        ax.text(x, top_y + 0.06, factor_label, ha="center", va="bottom", fontsize=8, wrap=True)
        members = member_map.get(factor_id, [])
        if not members:
            continue
        offsets = [0.0] if len(members) == 1 else [((j / max(1, len(members) - 1)) - 0.5) * 0.8 for j in range(len(members))]
        for offset, member in zip(offsets, members, strict=False):
            mx = x + float(offset)
            member_positions[member] = (mx, member_y)
            ax.scatter([mx], [member_y], s=450, color="#8ecae6", alpha=0.9, zorder=3)
            ax.text(mx, member_y - 0.06, member, ha="center", va="top", fontsize=8, wrap=True)
            ax.plot([x, mx], [top_y - 0.03, member_y + 0.03], color="#6c757d", linewidth=1.3, alpha=0.7, zorder=1)

    for row in relationship_rows:
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if left not in member_positions or right not in member_positions:
            continue
        x1, y1 = member_positions[left]
        x2, y2 = member_positions[right]
        width = 0.6 + 2.0 * _safe_float(row.get("semantic_index"))
        ax.plot([x1, x2], [y1, y2], color="#e63946", linestyle="--", linewidth=width, alpha=0.45, zorder=2)

    ax.set_title("Surviving Factors and Their Subparameter Relationships")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_representation_mae(tournament: dict[str, Any], *, output_path: Path) -> str:
    if plt is None:
        return ""
    trial_rows = list(tournament.get("trial_rows") or [])
    if not trial_rows:
        return ""
    labels = [str(row.get("representation") or "") for row in trial_rows]
    model_mae = [_safe_float(row.get("model_mean_absolute_error")) for row in trial_rows]
    carry_forward = [_safe_float(row.get("carry_forward_mean_absolute_error")) for row in trial_rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    ax.bar([idx - width / 2 for idx in x], model_mae, width=width, label="Model MAE", color="#457b9d")
    ax.bar([idx + width / 2 for idx in x], carry_forward, width=width, label="Carry-forward MAE", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Frozen Holdout: Model vs Carry-Forward")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def run_query_survivor_dossier(
    *,
    run_id: str,
    root_dir: str | Path,
    plugin_id: str = "hiv",
    corpus_mode: str = "megacrawl",
) -> dict[str, Any]:
    run_dir = Path(root_dir) / "artifacts" / "runs" / run_id
    analysis_dir = ensure_dir(run_dir / "analysis")

    executed_query_rows = _query_rows(run_dir)
    full_query_rows = _extend_query_bank(corpus_mode, plugin_id)
    boundary_report = read_json(run_dir / "phase0" / "extracted" / "boundary_validation_report.json", default={})
    survivors, _catalog, catalog_by_id = _survivor_rows(run_dir)
    relationship_rows = _relationship_rows(run_dir)
    tournament = _phase3_tournament(run_dir)
    trial_rows = list(tournament.get("trial_rows") or [])
    summary = dict(tournament.get("summary") or {})

    query_domain_counts = Counter(str(row.get("query_domain") or "unknown") for row in full_query_rows)
    query_geo_counts = Counter(str(row.get("query_geo_focus") or "unknown") for row in full_query_rows)
    query_silo_counts = Counter(str(row.get("query_silo") or "unknown") for row in full_query_rows)
    executed_domain_counts = Counter(str(row.get("query_domain") or "unknown") for row in executed_query_rows)
    executed_geo_counts = Counter(str(row.get("query_geo_focus") or "unknown") for row in executed_query_rows)

    survivor_chart_path = _plot_survivor_scores(survivors, output_path=analysis_dir / "survivor_factor_scores.png")
    member_map = _factor_member_map(survivors, catalog_by_id=catalog_by_id, relationship_rows=relationship_rows)
    relationship_subset = _survivor_relationship_subset(survivors, member_map=member_map, relationship_rows=relationship_rows)
    survivor_graph_path = _plot_survivor_graph(
        survivors,
        member_map=member_map,
        relationship_rows=relationship_subset,
        output_path=analysis_dir / "survivor_subparameter_graph.png",
    )
    representation_chart_path = _plot_representation_mae(
        tournament,
        output_path=analysis_dir / "phase3_representation_vs_carry_forward.png",
    )

    query_md_lines = [
        f"# Phase 0 Query Database: {run_id}",
        "",
        f"- total generated queries in `{corpus_mode}` bank: `{len(full_query_rows)}`",
        f"- executed shard queries in this run: `{len(executed_query_rows)}`",
        f"- accepted Phase 0 candidates: `{int(boundary_report.get('candidate_count_accepted') or 0)}`",
        f"- rejected Phase 0 candidates: `{int(boundary_report.get('candidate_count_rejected') or 0)}`",
        f"- acceptance rate: `{_safe_float(boundary_report.get('acceptance_rate')):.4f}`",
        "",
        "## Executed Shard Query Counts By Domain",
        "",
        "| domain | count |",
        "|---|---:|",
    ]
    for key, value in executed_domain_counts.most_common():
        query_md_lines.append(f"| {key} | {value} |")
    query_md_lines.extend(["", "## Executed Shard Query Counts By Geo Focus", "", "| geo focus | count |", "|---|---:|"])
    for key, value in executed_geo_counts.most_common():
        query_md_lines.append(f"| {key} | {value} |")
    query_md_lines.extend(
        [
            "",
        "## Query Counts By Domain",
        "",
        "| domain | count |",
        "|---|---:|",
        ]
    )
    for key, value in query_domain_counts.most_common():
        query_md_lines.append(f"| {key} | {value} |")
    query_md_lines.extend(["", "## Query Counts By Geo Focus", "", "| geo focus | count |", "|---|---:|"])
    for key, value in query_geo_counts.most_common():
        query_md_lines.append(f"| {key} | {value} |")
    query_md_lines.extend(["", "## Query Counts By Silo", "", "| silo | count |", "|---|---:|"])
    for key, value in query_silo_counts.most_common():
        query_md_lines.append(f"| {key} | {value} |")
    query_md_lines.extend(
        [
            "",
            "## Full Query Catalogue",
            "",
            "| # | query | domain | geo focus | lane | silo |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(full_query_rows, start=1):
        query_md_lines.append(
            "| {idx} | {query} | {domain} | {geo} | {lane} | {silo} |".format(
                idx=idx,
                query=str(row.get("query") or "").replace("|", "\\|"),
                domain=str(row.get("query_domain") or ""),
                geo=str(row.get("query_geo_focus") or ""),
                lane=str(row.get("query_lane") or ""),
                silo=str(row.get("query_silo") or ""),
            )
        )

    query_md_path = analysis_dir / "phase0_query_database.md"
    _write_markdown(query_md_path, "\n".join(query_md_lines) + "\n")

    carry_forward_mae = min((_safe_float(row.get("carry_forward_mean_absolute_error"), 1.0) for row in trial_rows), default=1.0)
    survivor_md_lines = [
        f"# Survivor And Relationship Dossier: {run_id}",
        "",
        f"- query database: [{query_md_path.name}]({query_md_path.as_posix()})",
        f"- total surviving factors: `{len(survivors)}`",
        f"- total relationship rows considered: `{len(relationship_rows)}`",
        f"- Phase 3 winner: `{summary.get('winner_representation')}`",
        f"- winner MAE: `{_safe_float(summary.get('winner_model_mean_absolute_error')):.6f}`",
        f"- winner SMAPE: `{_safe_float(summary.get('winner_model_smape')):.6f}`",
        f"- best carry-forward MAE in tournament: `{carry_forward_mae:.6f}`",
        f"- winner beats carry-forward: `{bool(summary.get('winner_beats_carry_forward'))}`",
        "",
        "## Survivor Chart",
        "",
        f"![Surviving factors]({Path(survivor_chart_path).as_posix()})" if survivor_chart_path else "- chart unavailable",
        "",
        "## Survivor Graph",
        "",
        f"![Survivor subparameter graph]({Path(survivor_graph_path).as_posix()})" if survivor_graph_path else "- graph unavailable",
        "",
        "## Relationship Bubble Chart",
        "",
        f"![Relationship bubble chart]({(run_dir / 'phase15' / 'semantic_relationship_bubble_chart.png').as_posix()})",
        "",
        "## Frozen Holdout Comparison",
        "",
        f"![Representation MAE comparison]({Path(representation_chart_path).as_posix()})" if representation_chart_path else "- chart unavailable",
        "",
        "| representation | model MAE | carry-forward MAE | delta vs carry-forward | model SMAPE | beats carry-forward |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in trial_rows:
        model_mae = _safe_float(row.get("model_mean_absolute_error"))
        cf_mae = _safe_float(row.get("carry_forward_mean_absolute_error"))
        survivor_md_lines.append(
            f"| {row.get('representation')} | {model_mae:.6f} | {cf_mae:.6f} | {model_mae - cf_mae:+.6f} | {_safe_float(row.get('model_smape')):.6f} | {bool(row.get('model_beats_carry_forward'))} |"
        )
    survivor_md_lines.extend(
        [
            "",
            "## Surviving Factors",
            "",
            "| factor | block | representation | members | holdout MAE improvement | holdout SMAPE improvement | survival score |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for survivor in survivors:
        factor_id = str(survivor.get("factor_id") or "")
        catalog_row = catalog_by_id.get(factor_id, {})
        survivor_md_lines.append(
            f"| {survivor.get('factor_name')} | {survivor.get('block_name')} | {survivor.get('representation_type')} | {int(catalog_row.get('member_count') or 0)} | {_safe_float(survivor.get('holdout_mae_improvement')):.6f} | {_safe_float(survivor.get('holdout_smape_improvement')):.6f} | {_safe_float(survivor.get('survival_score')):.6f} |"
        )
    survivor_md_lines.extend(["", "## Survivor Member Map", ""])
    for survivor in survivors:
        factor_id = str(survivor.get("factor_id") or "")
        members = member_map.get(factor_id, [])
        survivor_md_lines.append(f"### {survivor.get('factor_name')}")
        survivor_md_lines.append("")
        survivor_md_lines.append(f"- block: `{survivor.get('block_name')}`")
        survivor_md_lines.append(f"- representation: `{survivor.get('representation_type')}`")
        survivor_md_lines.append(f"- survival score: `{_safe_float(survivor.get('survival_score')):.6f}`")
        survivor_md_lines.append(f"- member or anchor subparameters: `{', '.join(members)}`")
        survivor_md_lines.append("")
    survivor_md_lines.extend(
        [
            "## Strongest Subparameter Relationships In The Survivor Graph",
            "",
            "| left | right | block | semantic index | similarity score | support mass |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in relationship_subset:
        survivor_md_lines.append(
            f"| {row.get('left')} | {row.get('right')} | {row.get('block_name')} | {_safe_float(row.get('semantic_index')):.6f} | {_safe_float(row.get('similarity_score')):.6f} | {_safe_float(row.get('support_mass')):.6f} |"
        )
    survivor_md_lines.extend(
        [
            "",
            "## What Happened",
            "",
            f"- The winner was `{summary.get('winner_representation')}`, which means that representation matched the 2024 holdout better than the other model variants.",
            f"- It still lost to carry-forward by `{_safe_float(summary.get('winner_model_mean_absolute_error')) - carry_forward_mae:+.6f}` MAE, so the representation is useful for structure inspection but not yet better than persistence forecasting on this holdout.",
            f"- Phase 0 geo binding was strong on this run: `{boundary_report.get('geo_binding_class_counts', {}).get('explicit_geo', 0)}` explicit geo rows and `{boundary_report.get('rejection_reason_counts', {}).get('missing_geo_binding', 0)}` geo-binding failures.",
            "",
        ]
    )

    survivor_md_path = analysis_dir / "survivor_relationship_dossier.md"
    _write_markdown(survivor_md_path, "\n".join(survivor_md_lines) + "\n")

    return {
        "run_id": run_id,
        "query_database_markdown": str(query_md_path),
        "survivor_dossier_markdown": str(survivor_md_path),
        "survivor_chart": survivor_chart_path,
        "survivor_graph": survivor_graph_path,
        "representation_chart": representation_chart_path,
    }
