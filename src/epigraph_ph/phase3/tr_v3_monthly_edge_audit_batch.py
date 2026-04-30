from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, utc_now_iso, write_json


DEFAULT_BASELINE_RUN_ID = "tr-v3-monthly-phase2-lane-20260416-s01"
DEFAULT_CANDIDATE_RUN_ID = "tr-v3-monthly-phase2-lane-20260418-s01"
KEY_METRICS = (
    "estimated_plhiv",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
    "new_diagnosed_cases_period",
)
TIME_RESOLUTIONS = ("monthly", "quarterly", "annual")


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _run_dir(run_id: str) -> Path:
    return ROOT_DIR / "artifacts" / "runs" / str(run_id)


def _edge_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("source") or ""),
        str(row.get("target") or ""),
        int(row.get("lag") or 0),
    )


def _load_edge_rows(run_id: str) -> list[dict[str, Any]]:
    payload = dict(read_json(_run_dir(run_id) / "phase2" / "phase2_structural_payload.json", default={}))
    return [dict(row) for row in list(payload.get("direct_temporal_edge_rows") or [])]


def _edge_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {_edge_key(row): dict(row) for row in rows}


def _edge_score(row: dict[str, Any]) -> float:
    weight = float(row.get("weight") or 0.0)
    stability = float(row.get("stability") or 0.0)
    return float(weight * stability)


def _load_report_summary(run_id: str) -> dict[str, Any]:
    return dict(read_json(_run_dir(run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json", default={}))


def _load_archive_support(run_id: str) -> dict[str, Any]:
    return dict(read_json(_run_dir(run_id) / "phase15" / "archive_observation_target_support.json", default={}))


def _load_axis_catalog(run_id: str) -> list[str]:
    return [str(value) for value in list(dict(read_json(_run_dir(run_id) / "phase1" / "axis_catalogs.json", default={})).get("canonical_name") or [])]


def _key_metric_support(run_id: str) -> dict[str, dict[str, int]]:
    rows = list(read_json(_run_dir(run_id) / "phase1" / "normalized_subparameters.json", default=[]))
    counts: dict[str, dict[str, int]] = {metric: {resolution: 0 for resolution in TIME_RESOLUTIONS} for metric in KEY_METRICS}
    for row in rows:
        if str(row.get("geo") or "") != "Philippines":
            continue
        canonical_name = str(row.get("canonical_name") or "")
        time_resolution = str(row.get("time_resolution") or "")
        if canonical_name not in counts or time_resolution not in counts[canonical_name]:
            continue
        counts[canonical_name][time_resolution] += 1
    return counts


def _diagnosis_flow_file_count(run_id: str) -> int:
    payload = dict(read_json(_run_dir(run_id) / "harp_archive" / "diagnosis_flow_points.json", default={}))
    return int(len(list(payload.get("points") or [])))


def _historical_panel_count(run_id: str) -> int:
    payload = dict(read_json(_run_dir(run_id) / "harp_archive" / "historical_harp_panel.json", default={}))
    return int(len(list(payload.get("rows") or [])))


def _summary_table(baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
    baseline_report = _load_report_summary(baseline_run_id)
    candidate_report = _load_report_summary(candidate_run_id)
    baseline_archive = _load_archive_support(baseline_run_id)
    candidate_archive = _load_archive_support(candidate_run_id)
    baseline_canonical_axis = _load_axis_catalog(baseline_run_id)
    candidate_canonical_axis = _load_axis_catalog(candidate_run_id)
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "baseline_monthly_row_count": int(baseline_report.get("monthly_row_count") or 0),
        "candidate_monthly_row_count": int(candidate_report.get("monthly_row_count") or 0),
        "baseline_harp_phase1_row_count": int(baseline_report.get("harp_phase1_row_count") or 0),
        "candidate_harp_phase1_row_count": int(candidate_report.get("harp_phase1_row_count") or 0),
        "candidate_historical_panel_row_count": int(candidate_report.get("harp_historical_panel_row_count") or 0),
        "baseline_canonical_count": int(len(baseline_canonical_axis)),
        "candidate_canonical_count": int(len(candidate_canonical_axis)),
        "baseline_direct_edge_count": int(dict(baseline_report.get("rebuilt_phase2_summary") or {}).get("direct_temporal_edge_count") or 0),
        "candidate_direct_edge_count": int(dict(candidate_report.get("rebuilt_phase2_summary") or {}).get("direct_temporal_edge_count") or 0),
        "baseline_diagnosis_flow_point_count": int(baseline_archive.get("diagnosis_flow_point_count") or 0),
        "candidate_diagnosis_flow_point_count": int(candidate_archive.get("diagnosis_flow_point_count") or 0),
        "baseline_harp_program_point_count": int(baseline_archive.get("harp_program_point_count") or 0),
        "candidate_harp_program_point_count": int(candidate_archive.get("harp_program_point_count") or 0),
        "baseline_diagnosis_flow_file_count": _diagnosis_flow_file_count(baseline_run_id),
        "candidate_diagnosis_flow_file_count": _diagnosis_flow_file_count(candidate_run_id),
        "baseline_historical_panel_count": _historical_panel_count(baseline_run_id),
        "candidate_historical_panel_count": _historical_panel_count(candidate_run_id),
    }


def _edge_delta_rows(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_lookup = _edge_lookup(baseline_rows)
    candidate_lookup = _edge_lookup(candidate_rows)
    keys = sorted(set(baseline_lookup) | set(candidate_lookup))
    rows: list[dict[str, Any]] = []
    for key in keys:
        baseline = baseline_lookup.get(key, {})
        candidate = candidate_lookup.get(key, {})
        baseline_weight = float(baseline.get("weight") or 0.0)
        candidate_weight = float(candidate.get("weight") or 0.0)
        baseline_stability = float(baseline.get("stability") or 0.0)
        candidate_stability = float(candidate.get("stability") or 0.0)
        rows.append(
            {
                "source": key[0],
                "target": key[1],
                "lag": key[2],
                "baseline_weight": baseline_weight,
                "candidate_weight": candidate_weight,
                "baseline_stability": baseline_stability,
                "candidate_stability": candidate_stability,
                "baseline_score": round(baseline_weight * baseline_stability, 6),
                "candidate_score": round(candidate_weight * candidate_stability, 6),
                "score_delta": round((candidate_weight * candidate_stability) - (baseline_weight * baseline_stability), 6),
                "edge_status": (
                    "new"
                    if key not in baseline_lookup and key in candidate_lookup
                    else "dropped"
                    if key in baseline_lookup and key not in candidate_lookup
                    else "retained"
                ),
            }
        )
    return rows


def _confounder_flags(summary: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(summary["candidate_diagnosis_flow_point_count"]) < int(summary["baseline_diagnosis_flow_point_count"]):
        flags.append(
            f"Phase15 diagnosis-flow support dropped from {summary['baseline_diagnosis_flow_point_count']} to {summary['candidate_diagnosis_flow_point_count']} points."
        )
    if int(summary["candidate_harp_program_point_count"]) < int(summary["baseline_harp_program_point_count"]):
        flags.append(
            f"Phase15 HARP program-point support dropped from {summary['baseline_harp_program_point_count']} to {summary['candidate_harp_program_point_count']} points."
        )
    if int(summary["candidate_diagnosis_flow_file_count"]) < int(summary["baseline_diagnosis_flow_file_count"]):
        flags.append(
            f"Target HARP archive diagnosis-flow file collapsed from {summary['baseline_diagnosis_flow_file_count']} to {summary['candidate_diagnosis_flow_file_count']} rows during the rerun."
        )
    return flags


def _plot_network_compare(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    positions = {
        "testing_engagement": (0.15, 0.80),
        "care_access_continuity": (0.55, 0.75),
        "suppression_capacity": (0.85, 0.35),
        "mobility_exposure_pressure": (0.35, 0.20),
    }

    def draw_panel(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
        ax.set_title(title)
        for node, (x_pos, y_pos) in positions.items():
            ax.scatter([x_pos], [y_pos], s=1800, color="#f4f4f0", edgecolor="#202020", linewidth=1.0, zorder=3)
            ax.text(x_pos, y_pos, node.replace("_", "\n"), ha="center", va="center", fontsize=10, zorder=4)
        for row in rows:
            source = str(row.get("source") or "")
            target = str(row.get("target") or "")
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            weight = float(row.get("weight") or 0.0)
            stability = float(row.get("stability") or 0.0)
            color = "#0b6e4f" if weight >= 0.0 else "#8b1e3f"
            width = max(1.0, 8.0 * abs(weight))
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", linewidth=width, color=color, alpha=max(0.35, stability)),
                zorder=2,
            )
            xm = (x0 + x1) / 2.0
            ym = (y0 + y1) / 2.0
            ax.text(
                xm,
                ym,
                f"lag {int(row.get('lag') or 0)}\nw={weight:.3f}\ns={stability:.3f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="#cccccc"),
                zorder=5,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_panel(axes[0], baseline_rows, "Baseline monthly lane")
    draw_panel(axes[1], candidate_rows, "Annual-anchor monthly lane")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_edge_delta_heatmap(edge_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    blocks = sorted({str(row.get("source") or "") for row in edge_rows} | {str(row.get("target") or "") for row in edge_rows})
    matrix = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    for row in edge_rows:
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if source not in blocks or target not in blocks:
            continue
        matrix[blocks.index(source), blocks.index(target)] = float(row.get("score_delta") or 0.0)
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = vmax if vmax > 0.0 else 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(blocks)), labels=[block.replace("_", "\n") for block in blocks], rotation=0)
    ax.set_yticks(range(len(blocks)), labels=[block.replace("_", "\n") for block in blocks])
    ax.set_title("Edge score delta (candidate - baseline)")
    for row_idx in range(len(blocks)):
        for col_idx in range(len(blocks)):
            value = float(matrix[row_idx, col_idx])
            if abs(value) < 1e-6:
                continue
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8, label="weight x stability delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_testing_edge_bars(edge_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    testing_rows = [row for row in edge_rows if str(row.get("source") or "") == "testing_engagement"]
    labels = [f"{row['source']} -> {row['target']} @ lag{row['lag']}" for row in testing_rows]
    baseline_scores = [float(row.get("baseline_score") or 0.0) for row in testing_rows]
    candidate_scores = [float(row.get("candidate_score") or 0.0) for row in testing_rows]
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(3.5, 1.1 * len(labels))))
    ax.barh(y_pos - 0.18, baseline_scores, height=0.35, color="#7f8c8d", label="baseline")
    ax.barh(y_pos + 0.18, candidate_scores, height=0.35, color="#0b6e4f", label="candidate")
    ax.set_yticks(y_pos, labels=labels)
    ax.axvline(0.0, color="#303030", linewidth=1.0)
    ax.set_xlabel("edge score (weight x stability)")
    ax.set_title("Testing-led edge score comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_support_context(summary: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [
        "monthly rows",
        "HARP Phase1 rows",
        "direct edges",
        "diagnosis-flow points\n(phase15 support)",
        "HARP program points\n(phase15 support)",
        "diagnosis-flow file\n(harp_archive)",
    ]
    baseline_values = [
        int(summary["baseline_monthly_row_count"]),
        int(summary["baseline_harp_phase1_row_count"]),
        int(summary["baseline_direct_edge_count"]),
        int(summary["baseline_diagnosis_flow_point_count"]),
        int(summary["baseline_harp_program_point_count"]),
        int(summary["baseline_diagnosis_flow_file_count"]),
    ]
    candidate_values = [
        int(summary["candidate_monthly_row_count"]),
        int(summary["candidate_harp_phase1_row_count"]),
        int(summary["candidate_direct_edge_count"]),
        int(summary["candidate_diagnosis_flow_point_count"]),
        int(summary["candidate_harp_program_point_count"]),
        int(summary["candidate_diagnosis_flow_file_count"]),
    ]
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x_pos - 0.18, baseline_values, width=0.35, color="#7f8c8d", label="baseline")
    ax.bar(x_pos + 0.18, candidate_values, width=0.35, color="#1f77b4", label="candidate")
    ax.set_xticks(x_pos, labels=labels, rotation=15, ha="right")
    ax.set_title("Support context before and after annual-anchor injection")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_key_metric_support(
    baseline_support: dict[str, dict[str, int]],
    candidate_support: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels: list[str] = []
    baseline_values: list[int] = []
    candidate_values: list[int] = []
    for metric in KEY_METRICS:
        for resolution in TIME_RESOLUTIONS:
            labels.append(f"{metric}\n{resolution}")
            baseline_values.append(int(dict(baseline_support.get(metric) or {}).get(resolution) or 0))
            candidate_values.append(int(dict(candidate_support.get(metric) or {}).get(resolution) or 0))
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x_pos - 0.18, baseline_values, width=0.35, color="#7f8c8d", label="baseline")
    ax.bar(x_pos + 0.18, candidate_values, width=0.35, color="#0b6e4f", label="candidate")
    ax.set_xticks(x_pos, labels=labels, rotation=55, ha="right")
    ax.set_title("Key metric support by time resolution")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _decision(summary: dict[str, Any], edge_rows: list[dict[str, Any]]) -> str:
    flags = _confounder_flags(summary)
    testing_to_care = next((row for row in edge_rows if row["source"] == "testing_engagement" and row["target"] == "care_access_continuity" and int(row["lag"]) == 1), None)
    if flags:
        return (
            "Do not promote yet. The testing-centered graph strengthened, but the rerun simultaneously changed Phase15 support geometry. "
            + " ".join(flags)
        )
    if testing_to_care and float(testing_to_care.get("score_delta") or 0.0) > 0.0:
        return "Promotable for narrow follow-up. The testing-to-care edge strengthened without obvious support collapse."
    return "Do not promote. The graph changed, but not in a direction that clears the audit gate."


def _report_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload["summary"])
    edge_rows = list(payload["edge_deltas"])
    flags = list(payload["confounder_flags"])
    lines = [
        "# Monthly Testing-Edge Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline run: `{summary['baseline_run_id']}`",
        f"- Candidate run: `{summary['candidate_run_id']}`",
        "",
        "## Headline",
        "",
        payload["decision"],
        "",
        "## Structural Delta",
        "",
        f"- Total monthly-aligned Phase1 rows: `{summary['baseline_monthly_row_count']} -> {summary['candidate_monthly_row_count']}`",
        f"- Injected HARP Phase1 rows: `{summary['baseline_harp_phase1_row_count']} -> {summary['candidate_harp_phase1_row_count']}`",
        f"- Historical panel rows in candidate: `{summary['candidate_historical_panel_row_count']}`",
        f"- Canonical count: `{summary['baseline_canonical_count']} -> {summary['candidate_canonical_count']}`",
        f"- Direct temporal edges: `{summary['baseline_direct_edge_count']} -> {summary['candidate_direct_edge_count']}`",
        "",
        "## Support Confounders",
        "",
    ]
    if flags:
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- No hard support-collapse flag fired.")
    lines.extend(
        [
            "",
            "## Edge Table",
            "",
            "| Source | Target | Lag | Status | Baseline Score | Candidate Score | Delta |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in edge_rows:
        lines.append(
            f"| {row['source']} | {row['target']} | {int(row['lag'])} | {row['edge_status']} | "
            f"{float(row['baseline_score']):.6f} | {float(row['candidate_score']):.6f} | {float(row['score_delta']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Graph Pack",
            "",
            "- `analysis/edge_network_compare.png`",
            "- `analysis/edge_delta_heatmap.png`",
            "- `analysis/testing_edge_bars.png`",
            "- `analysis/support_context_compare.png`",
            "- `analysis/key_metric_support_compare.png`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_monthly_edge_audit_batch(
    *,
    run_id: str,
    baseline_run_id: str = DEFAULT_BASELINE_RUN_ID,
    candidate_run_id: str = DEFAULT_CANDIDATE_RUN_ID,
) -> dict[str, Any]:
    target_run_dir = _run_dir(run_id)
    analysis_dir = ensure_dir(target_run_dir / "analysis")

    baseline_edges = _load_edge_rows(baseline_run_id)
    candidate_edges = _load_edge_rows(candidate_run_id)
    summary = _summary_table(baseline_run_id, candidate_run_id)
    edge_deltas = _edge_delta_rows(baseline_edges, candidate_edges)
    baseline_support = _key_metric_support(baseline_run_id)
    candidate_support = _key_metric_support(candidate_run_id)
    confounder_flags = _confounder_flags(summary)

    _plot_network_compare(baseline_edges, candidate_edges, analysis_dir / "edge_network_compare.png")
    _plot_edge_delta_heatmap(edge_deltas, analysis_dir / "edge_delta_heatmap.png")
    _plot_testing_edge_bars(edge_deltas, analysis_dir / "testing_edge_bars.png")
    _plot_support_context(summary, analysis_dir / "support_context_compare.png")
    _plot_key_metric_support(baseline_support, candidate_support, analysis_dir / "key_metric_support_compare.png")

    payload = {
        "generated_at": utc_now_iso(),
        "run_id": str(run_id),
        "summary": summary,
        "edge_deltas": edge_deltas,
        "baseline_key_metric_support": baseline_support,
        "candidate_key_metric_support": candidate_support,
        "confounder_flags": confounder_flags,
        "decision": _decision(summary, edge_deltas),
        "artifacts": {
            "edge_network_compare": str(analysis_dir / "edge_network_compare.png"),
            "edge_delta_heatmap": str(analysis_dir / "edge_delta_heatmap.png"),
            "testing_edge_bars": str(analysis_dir / "testing_edge_bars.png"),
            "support_context_compare": str(analysis_dir / "support_context_compare.png"),
            "key_metric_support_compare": str(analysis_dir / "key_metric_support_compare.png"),
        },
    }
    write_json(analysis_dir / "tr_v3_monthly_edge_audit_batch_report.json", payload)
    (analysis_dir / "tr_v3_monthly_edge_audit_batch_report.md").write_text(_report_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit testing-centered edge changes in the monthly HARP-enriched Phase2 lane.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--candidate-run-id", default=DEFAULT_CANDIDATE_RUN_ID)
    args = parser.parse_args()
    run_tr_v3_monthly_edge_audit_batch(
        run_id=str(args.run_id),
        baseline_run_id=str(args.baseline_run_id),
        candidate_run_id=str(args.candidate_run_id),
    )


if __name__ == "__main__":
    main()
