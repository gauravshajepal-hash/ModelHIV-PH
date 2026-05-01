from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_testing_prevention_rebuild_batch as rebuild
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


ABLATION_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full_block", ()),
    ("exclude_prep", ("prep_people_receiving_per_100k",)),
    ("exclude_tests", ("annual_hiv_tests_volume_per_100k",)),
)


def _read_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_testing_prevention_rebuild_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing testing-prevention rebuild report: {path}")
    return payload


def _score_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    support = dict(payload.get("support_gate") or {})
    edge = dict(payload.get("edge_gate") or {})
    circularity = dict(payload.get("circularity_gate") or {})
    baseline = dict(payload.get("baseline_preservation_gate") or {})
    testing = dict(payload.get("testing_scenario_gate") or {})
    archive_gate = dict(payload.get("archive_alignment_gate") or {})
    edge_row = dict(edge.get("edge_row") or {})
    return {
        "variant": label,
        "overall_decision": str(payload.get("overall_decision") or ""),
        "support_decision": str(support.get("decision") or ""),
        "block_retained": bool(support.get("block_retained")),
        "indicator_count": int(support.get("indicator_count") or 0),
        "max_indicator_share": float(support.get("max_indicator_share") or 0.0),
        "edge_decision": str(edge.get("decision") or ""),
        "edge_target": str(edge_row.get("target") or ""),
        "edge_score": float(edge_row.get("mean_score") or 0.0),
        "edge_survive_top2_count": int(edge_row.get("survive_top2_count") or 0),
        "circularity_decision": str(circularity.get("decision") or ""),
        "circularity_sign_agreement": float(circularity.get("sign_agreement_rate") or 0.0),
        "circularity_mean_abs_ratio": float(circularity.get("mean_abs_ratio") or 0.0),
        "baseline_decision": str(baseline.get("decision") or ""),
        "baseline_sign_agreement": float(baseline.get("sign_agreement_rate") or 0.0),
        "baseline_mean_abs_delta": float(baseline.get("mean_abs_delta_of_delta") or 0.0),
        "testing_decision": str(testing.get("decision") or ""),
        "testing_nonnull_count": int(testing.get("nonnull_terminal_count") or 0),
        "testing_max_abs_delta": float(testing.get("max_abs_terminal_delta") or 0.0),
        "archive_decision": str(archive_gate.get("decision") or ""),
    }


def _plot_gate_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = [
        ("block_retained", 1.0),
        ("edge_survive_top2_count", 1.0),
        ("circularity_sign_agreement", 1.0),
        ("baseline_sign_agreement", 1.0),
        ("testing_nonnull_count", 1.0),
    ]
    labels = [str(row["variant"]) for row in rows]
    matrix = np.asarray(
        [[float(row.get(metric) or 0.0) for metric, _ in metrics] for row in rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(labels) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([metric for metric, _ in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Testing indicator ablation gate matrix")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_terminal_delta_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    values = [float(row.get("testing_max_abs_delta") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylabel("Max abs testing terminal delta")
    ax.set_title("Testing scenario amplitude by indicator ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Testing Indicator Ablation Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline two-block run: `{payload['baseline_two_block_run_id']}`",
        f"- Base monthly run: `{payload['base_monthly_run_id']}`",
        "",
        "## Variant summary",
        "",
        "| Variant | Overall | Support | Edge target | Edge gate | Circularity | Baseline preserve | Testing gate | Testing max abs delta |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for row in list(payload.get("summary_rows") or []):
        lines.append(
            f"| `{row['variant']}` | `{row['overall_decision']}` | `{row['support_decision']}` | "
            f"`{row['edge_target'] or 'none'}` | `{row['edge_decision']}` | `{row['circularity_decision']}` | "
            f"`{row['baseline_decision']}` | `{row['testing_decision']}` | `{float(row['testing_max_abs_delta']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `gate_matrix`: `{payload['artifacts']['gate_matrix']}`",
            f"- `terminal_delta_bars`: `{payload['artifacts']['terminal_delta_bars']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_phase2_testing_indicator_ablation_batch(
    *,
    run_id: str,
    baseline_two_block_run_id: str | None = None,
    base_monthly_run_id: str | None = None,
    legacy_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
) -> dict[str, Any]:
    variant_reports: list[dict[str, Any]] = []
    baseline_two_block = None
    base_monthly = None
    for label, exclusions in ABLATION_VARIANTS:
        variant_run_id = f"{run_id}-{label}"
        rebuild.run_tr_v3_phase2_testing_prevention_rebuild_batch(
            run_id=variant_run_id,
            baseline_two_block_run_id=baseline_two_block_run_id,
            base_monthly_run_id=base_monthly_run_id,
            legacy_archive_run_id=legacy_archive_run_id,
            forecast_horizon_quarters=int(forecast_horizon_quarters),
            additional_excluded_canonicals=tuple(exclusions),
        )
        payload = _read_report(variant_run_id)
        if baseline_two_block is None:
            baseline_two_block = str(payload.get("baseline_two_block_run_id") or "")
        if base_monthly is None:
            base_monthly = str(payload.get("base_monthly_run_id") or "")
        variant_reports.append({"label": label, "exclusions": list(exclusions), "payload": payload})

    summary_rows = [_score_row(str(item["label"]), dict(item["payload"])) for item in variant_reports]

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    gate_matrix = analysis_dir / "testing_indicator_ablation_gate_matrix.png"
    delta_bars = analysis_dir / "testing_indicator_ablation_terminal_deltas.png"
    _plot_gate_matrix(summary_rows, gate_matrix)
    _plot_terminal_delta_bars(summary_rows, delta_bars)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_two_block_run_id": str(baseline_two_block or ""),
        "base_monthly_run_id": str(base_monthly or ""),
        "variant_rows": variant_reports,
        "summary_rows": summary_rows,
        "artifacts": {
            "gate_matrix": gate_matrix.name,
            "terminal_delta_bars": delta_bars.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_testing_indicator_ablation_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_testing_indicator_ablation_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run indicator-level falsification for the testing_prevention_reach block.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-two-block-run-id", default=None)
    parser.add_argument("--base-monthly-run-id", default=None)
    parser.add_argument("--legacy-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_testing_indicator_ablation_batch(
        run_id=str(args.run_id),
        baseline_two_block_run_id=args.baseline_two_block_run_id,
        base_monthly_run_id=args.base_monthly_run_id,
        legacy_archive_run_id=args.legacy_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
