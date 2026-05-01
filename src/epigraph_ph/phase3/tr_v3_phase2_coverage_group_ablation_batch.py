from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as monthly_lane
from epigraph_ph.phase3 import tr_v3_phase2_coverage_indicator_effect_batch as effect
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, write_json


DEFAULT_BASELINE_RUN_ID = effect.DEFAULT_BASELINE_RUN_ID
DEFAULT_SOURCE_RUN_ID = effect.DEFAULT_SOURCE_RUN_ID
DEFAULT_COVERAGE_RUN_ID = effect.DEFAULT_COVERAGE_RUN_ID
LOAD_BEARING_TRIO: tuple[str, ...] = (
    "prep_people_receiving_per_100k",
    "hiv_test_positivity_percent",
    "late_hiv_diagnosis_percent",
)
ALL_ADDED_COVERAGE_CANONICALS: tuple[str, ...] = effect.ADDED_COVERAGE_CANONICALS


def _plot_variant_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    values = [float(row.get("direct_temporal_edge_count") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylabel("Direct temporal edge count")
    ax.set_title("Grouped coverage-ablation edge counts")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_structural_support(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    values = [float(row.get("structural_monthly_row_count") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color="#dd8452")
    ax.set_ylabel("Structural monthly row count")
    ax.set_title("Grouped coverage-ablation support")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Coverage Group Ablation Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline monthly run: `{payload['baseline_run_id']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Coverage run: `{payload['coverage_run_id']}`",
        "",
        "## Variant summary",
        "",
        "| Variant | Excluded canonicals | Structural rows | Canonicals | Direct edges | Edge signature | Baseline overlap | Restores testing edge |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in list(payload.get("summary_rows") or []):
        excluded = ", ".join(list(row.get("excluded_canonicals") or [])) or "none"
        lines.append(
            f"| `{row['variant']}` | `{excluded}` | `{int(row['structural_monthly_row_count'])}` | "
            f"`{int(row['canonical_count'])}` | `{int(row['direct_temporal_edge_count'])}` | "
            f"`{row['edge_signature']}` | `{int(row['baseline_overlap_count'])}` | "
            f"`{bool(row['restores_baseline_testing_edge'])}` |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `edge_count_plot`: `{payload['artifacts']['edge_count_plot']}`",
            f"- `support_plot`: `{payload['artifacts']['support_plot']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_phase2_coverage_group_ablation_batch(
    *,
    run_id: str,
    baseline_run_id: str = DEFAULT_BASELINE_RUN_ID,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    coverage_run_id: str = DEFAULT_COVERAGE_RUN_ID,
    plugin_id: str = "hiv",
    start_month: str = "2010-01",
) -> dict[str, Any]:
    baseline_phase2 = effect._read_phase2_payload(baseline_run_id)
    baseline_edge_set = effect._edge_set(effect._edge_signature_rows(baseline_phase2))

    variant_specs: list[tuple[str, tuple[str, ...]]] = [
        ("merged_all", ()),
        ("exclude_load_bearing_trio", LOAD_BEARING_TRIO),
        ("exclude_all_added_coverage", ALL_ADDED_COVERAGE_CANONICALS),
    ]

    summary_rows: list[dict[str, Any]] = []
    merged_all_edge_set: set[tuple[str, str, int]] | None = None

    for label, exclusions in variant_specs:
        variant_run_id = f"{run_id}-{label}"
        monthly_lane.run_tr_v3_monthly_phase2_lane_batch(
            run_id=variant_run_id,
            source_run_id=source_run_id,
            coverage_run_id=coverage_run_id,
            plugin_id=plugin_id,
            start_month=start_month,
            structural_excluded_canonicals=tuple(exclusions),
        )
        report = effect._read_monthly_lane_report(variant_run_id)
        phase2_payload = effect._read_phase2_payload(variant_run_id)
        edge_rows = effect._edge_signature_rows(phase2_payload)
        if label == "merged_all":
            merged_all_edge_set = effect._edge_set(edge_rows)
        summary_rows.append(
            effect._score_variant(
                label=label,
                excluded_canonicals=tuple(exclusions),
                report=report,
                phase2_payload=phase2_payload,
                baseline_edge_set=baseline_edge_set,
                merged_all_edge_set=merged_all_edge_set or set(),
            )
        )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    edge_count_plot = analysis_dir / "coverage_group_ablation_edge_counts.png"
    support_plot = analysis_dir / "coverage_group_ablation_support.png"
    _plot_variant_bars(summary_rows, edge_count_plot)
    _plot_structural_support(summary_rows, support_plot)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_run_id": str(baseline_run_id),
        "source_run_id": str(source_run_id),
        "coverage_run_id": str(coverage_run_id),
        "baseline_edge_signature": effect._edge_signature(effect._edge_signature_rows(baseline_phase2)),
        "summary_rows": summary_rows,
        "artifacts": {
            "edge_count_plot": edge_count_plot.name,
            "support_plot": support_plot.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_coverage_group_ablation_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_coverage_group_ablation_batch_report.md").write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grouped structural ablations for the added coverage canonicals.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--coverage-run-id", default=DEFAULT_COVERAGE_RUN_ID)
    parser.add_argument("--plugin", default="hiv")
    parser.add_argument("--start-month", default="2010-01")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_coverage_group_ablation_batch(
        run_id=str(args.run_id),
        baseline_run_id=str(args.baseline_run_id),
        source_run_id=str(args.source_run_id),
        coverage_run_id=str(args.coverage_run_id),
        plugin_id=str(args.plugin),
        start_month=str(args.start_month),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
