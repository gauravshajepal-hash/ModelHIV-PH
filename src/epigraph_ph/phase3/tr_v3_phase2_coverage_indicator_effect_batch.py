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
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_BASELINE_RUN_ID = "tr-v3-monthly-phase2-lane-20260418-s02"
DEFAULT_SOURCE_RUN_ID = "phase2-replay-source-hiv-anchors-20260418-s00"
DEFAULT_COVERAGE_RUN_ID = "harp-archive-hiv-data-coverage-20260419-s00"
ADDED_COVERAGE_CANONICALS: tuple[str, ...] = (
    "annual_hiv_tests_volume_per_100k",
    "prep_people_receiving_per_100k",
    "hiv_test_positivity_percent",
    "late_hiv_diagnosis_percent",
    "unaids_known_status_share_percent",
)


def _read_monthly_lane_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"missing monthly lane report: {path}")
    return payload


def _read_phase2_payload(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "phase2" / "phase2_structural_payload.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"missing phase2 payload: {path}")
    return payload


def _edge_signature_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in list(payload.get("direct_temporal_edge_rows") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source": str(item.get("source") or ""),
                "target": str(item.get("target") or ""),
                "lag": int(item.get("lag") or 0),
                "weight": float(item.get("weight") or 0.0),
                "stability": float(item.get("stability") or 0.0),
            }
        )
    rows.sort(key=lambda row: (row["source"], row["target"], row["lag"]))
    return rows


def _edge_signature(edge_rows: list[dict[str, Any]]) -> str:
    if not edge_rows:
        return "none"
    return "; ".join(
        f"{row['source']}->{row['target']}@lag{row['lag']}" for row in edge_rows
    )


def _edge_set(edge_rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    return {(row["source"], row["target"], int(row["lag"])) for row in edge_rows}


def _score_variant(
    *,
    label: str,
    excluded_canonicals: tuple[str, ...],
    report: dict[str, Any],
    phase2_payload: dict[str, Any],
    baseline_edge_set: set[tuple[str, str, int]],
    merged_all_edge_set: set[tuple[str, str, int]],
) -> dict[str, Any]:
    edge_rows = _edge_signature_rows(phase2_payload)
    variant_edge_set = _edge_set(edge_rows)
    return {
        "variant": label,
        "excluded_canonicals": list(excluded_canonicals),
        "structural_monthly_row_count": int(report.get("structural_monthly_row_count") or 0),
        "canonical_count": int(dict(report.get("phase1_summary") or {}).get("canonical_count") or 0),
        "block_count": int(dict(report.get("rebuilt_phase2_summary") or {}).get("block_count") or 0),
        "direct_temporal_edge_count": int(dict(report.get("rebuilt_phase2_summary") or {}).get("direct_temporal_edge_count") or 0),
        "edge_signature": _edge_signature(edge_rows),
        "edge_rows": edge_rows,
        "testing_edge_present": any(row["source"].startswith("testing") for row in edge_rows),
        "care_to_suppression_present": ("care_access_continuity", "suppression_capacity", 1) in variant_edge_set,
        "baseline_overlap_count": int(len(variant_edge_set & baseline_edge_set)),
        "merged_all_overlap_count": int(len(variant_edge_set & merged_all_edge_set)),
        "restores_baseline_testing_edge": ("testing_engagement", "care_access_continuity", 1) in variant_edge_set
        or ("testing_prevention_reach", "care_access_continuity", 1) in variant_edge_set,
    }


def _plot_edge_overlap(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    baseline = [float(row.get("baseline_overlap_count") or 0.0) for row in rows]
    merged = [float(row.get("merged_all_overlap_count") or 0.0) for row in rows]
    positions = np.arange(len(labels), dtype=np.int32)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(positions - width / 2.0, baseline, width=width, label="overlap vs baseline")
    ax.bar(positions + width / 2.0, merged, width=width, label="overlap vs merged_all")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Direct-edge overlap count")
    ax.set_title("Coverage-canonical exclusion edge overlap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_structural_row_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    values = [float(row.get("structural_monthly_row_count") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylabel("Structural monthly row count")
    ax.set_title("Coverage-canonical exclusion support footprint")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Coverage Indicator Effect Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline monthly run: `{payload['baseline_run_id']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Coverage run: `{payload['coverage_run_id']}`",
        "",
        "## Variant summary",
        "",
        "| Variant | Excluded canonical | Structural rows | Canonicals | Direct edges | Edge signature | Baseline overlap | Merged overlap | Restores testing edge |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in list(payload.get("summary_rows") or []):
        excluded = ", ".join(list(row.get("excluded_canonicals") or [])) or "none"
        lines.append(
            f"| `{row['variant']}` | `{excluded}` | `{int(row['structural_monthly_row_count'])}` | "
            f"`{int(row['canonical_count'])}` | `{int(row['direct_temporal_edge_count'])}` | "
            f"`{row['edge_signature']}` | `{int(row['baseline_overlap_count'])}` | "
            f"`{int(row['merged_all_overlap_count'])}` | `{bool(row['restores_baseline_testing_edge'])}` |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `edge_overlap_plot`: `{payload['artifacts']['edge_overlap_plot']}`",
            f"- `row_delta_plot`: `{payload['artifacts']['row_delta_plot']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_phase2_coverage_indicator_effect_batch(
    *,
    run_id: str,
    baseline_run_id: str = DEFAULT_BASELINE_RUN_ID,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    coverage_run_id: str = DEFAULT_COVERAGE_RUN_ID,
    plugin_id: str = "hiv",
    start_month: str = "2010-01",
) -> dict[str, Any]:
    baseline_report = _read_monthly_lane_report(baseline_run_id)
    baseline_phase2 = _read_phase2_payload(baseline_run_id)
    baseline_edge_set = _edge_set(_edge_signature_rows(baseline_phase2))

    variant_specs: list[tuple[str, tuple[str, ...]]] = [("merged_all", ())]
    variant_specs.extend((f"exclude_{canonical}", (canonical,)) for canonical in ADDED_COVERAGE_CANONICALS)

    variant_rows: list[dict[str, Any]] = []
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
        report = _read_monthly_lane_report(variant_run_id)
        phase2_payload = _read_phase2_payload(variant_run_id)
        edge_rows = _edge_signature_rows(phase2_payload)
        if label == "merged_all":
            merged_all_edge_set = _edge_set(edge_rows)
        variant_rows.append(
            {
                "label": label,
                "excluded_canonicals": list(exclusions),
                "report": report,
                "phase2_payload": phase2_payload,
            }
        )

    merged_all_edge_set = merged_all_edge_set or set()
    summary_rows = [
        _score_variant(
            label=str(item["label"]),
            excluded_canonicals=tuple(str(value) for value in list(item["excluded_canonicals"] or [])),
            report=dict(item["report"]),
            phase2_payload=dict(item["phase2_payload"]),
            baseline_edge_set=baseline_edge_set,
            merged_all_edge_set=merged_all_edge_set,
        )
        for item in variant_rows
    ]

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    edge_overlap_plot = analysis_dir / "coverage_indicator_edge_overlap.png"
    row_delta_plot = analysis_dir / "coverage_indicator_structural_rows.png"
    _plot_edge_overlap(summary_rows, edge_overlap_plot)
    _plot_structural_row_deltas(summary_rows, row_delta_plot)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_run_id": str(baseline_run_id),
        "source_run_id": str(source_run_id),
        "coverage_run_id": str(coverage_run_id),
        "baseline_edge_signature": _edge_signature(_edge_signature_rows(baseline_phase2)),
        "summary_rows": summary_rows,
        "artifacts": {
            "edge_overlap_plot": edge_overlap_plot.name,
            "row_delta_plot": row_delta_plot.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_coverage_indicator_effect_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_coverage_indicator_effect_batch_report.md").write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit which added coverage canonicals are moving the retained Phase2 monthly structure.")
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
    run_tr_v3_phase2_coverage_indicator_effect_batch(
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
