from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_phase2_archive_alignment_batch as archive_alignment
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_phase2_sidecar_ablation_batch as sidecar_ablation
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_ACTIVE_BLOCK_SUBSET: tuple[str, ...] = (
    "care_access_continuity",
    "mobility_exposure_pressure",
)
ACTIVE_SCENARIOS: tuple[str, ...] = (
    "disruption_recovery",
    "mobility_spike",
)


def _seeded_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing seeded report: {path}")
    return payload


def _alignment_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing archive alignment report: {path}")
    return payload


def _active_terminal_rows(seed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in list(seed_payload.get("terminal_delta_rows") or [])
        if str(row.get("scenario") or "") in set(ACTIVE_SCENARIOS)
    ]


def _active_scenario_summary(seed_payload: dict[str, Any]) -> dict[str, Any]:
    rows = _active_terminal_rows(seed_payload)
    abs_values: list[float] = []
    nonnull_count = 0
    for row in rows:
        for metric_name in seeded.METRIC_PLOT_ORDER:
            value = float(row.get(f"{metric_name}_delta") or 0.0)
            abs_values.append(abs(value))
            if abs(value) > 1e-6:
                nonnull_count += 1
    max_abs_delta = float(max(abs_values) if abs_values else 0.0)
    return {
        "row_count": int(len(rows)),
        "nonnull_terminal_count": int(nonnull_count),
        "max_abs_terminal_delta": max_abs_delta,
        "mean_abs_terminal_delta": float(np.mean(abs_values)) if abs_values else 0.0,
        "decision": "keep" if max_abs_delta >= 1.0 and nonnull_count > 0 else "revert",
    }


def _plot_active_terminal_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Two-block kernel", body="No active terminal rows.")
        return
    labels = [f"{row['contract']}:{row['scenario']}" for row in rows]
    matrix = np.asarray(
        [
            [float(row.get(f"{metric_name}_delta") or 0.0) for metric_name in seeded.METRIC_PLOT_ORDER]
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(labels) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="PiYG")
    ax.set_xticks(range(len(seeded.METRIC_PLOT_ORDER)))
    ax.set_xticklabels([metric.replace("_", " ") for metric in seeded.METRIC_PLOT_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Two-block kernel active-scenario terminal deltas")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    active = dict(payload.get("active_scenario_summary") or {})
    archive_gate = dict(payload.get("archive_alignment_gate") or {})
    lines = [
        "# TR-V3 Phase 2 Two-Block Kernel Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Monthly run: `{payload['monthly_phase2_run_id']}`",
        f"- Legacy archive: `{payload['legacy_archive_run_id']}`",
        f"- Aligned archive: `{payload['aligned_archive_run_id']}`",
        f"- Active block subset: `{list(payload.get('active_block_subset') or [])}`",
        "",
        "## Active scenarios",
        "",
        f"- Scenarios: `{list(payload.get('active_scenarios') or [])}`",
        f"- Terminal row count: `{int(active.get('row_count') or 0)}`",
        f"- Non-null terminal deltas: `{int(active.get('nonnull_terminal_count') or 0)}`",
        f"- Max abs terminal delta: `{float(active.get('max_abs_terminal_delta') or 0.0):.3f}`",
        f"- Mean abs terminal delta: `{float(active.get('mean_abs_terminal_delta') or 0.0):.3f}`",
        f"- Decision: `{active.get('decision')}`",
        "",
        "## Frozen-readout archive alignment",
        "",
        f"- Active sign agreement: `{float(archive_gate.get('active_sign_agreement') or 0.0):.3f}`",
        f"- Active mean abs delta-of-delta: `{float(archive_gate.get('active_mean_abs_delta') or 0.0):.3f}`",
        f"- Decision: `{archive_gate.get('decision')}`",
        "",
        "## Overall",
        "",
        f"- Decision: `{payload.get('overall_decision')}`",
        "",
        "## Artifacts",
        "",
        f"- `seeded_report`: `{payload['artifacts']['seeded_report']}`",
        f"- `alignment_report`: `{payload['artifacts']['alignment_report']}`",
        f"- `active_terminal_deltas`: `{payload['artifacts']['active_terminal_deltas']}`",
        f"- `aligned_structural_graph`: `{payload['artifacts']['aligned_structural_graph']}`",
        f"- `aligned_terminal_delta_heatmap`: `{payload['artifacts']['aligned_terminal_delta_heatmap']}`",
        f"- `archive_alignment_compare`: `{payload['artifacts']['archive_alignment_compare']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_phase2_two_block_kernel_batch(
    *,
    run_id: str,
    monthly_phase2_run_id: str | None = None,
    legacy_archive_run_id: str | None = None,
    aligned_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
    active_block_subset: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    monthly_run = str(monthly_phase2_run_id or sidecar_ablation._latest_outcome_lite_monthly_run())
    legacy_archive_run = str(legacy_archive_run_id or suite._latest_standard_archive_run())
    aligned_archive_run = str(aligned_archive_run_id or monthly_run)
    active_blocks = tuple(str(name) for name in list(active_block_subset or DEFAULT_ACTIVE_BLOCK_SUBSET))

    seeded_run = f"{run_id}-aligned-seeded"
    alignment_run = f"{run_id}-alignment"
    seeded.run_tr_v3_phase2_seeded_champion_batch(
        run_id=seeded_run,
        archive_run_id=aligned_archive_run,
        readout_source_archive_run_id=aligned_archive_run,
        monthly_phase2_run_id=monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_blocks,
    )
    archive_alignment.run_tr_v3_phase2_archive_alignment_batch(
        run_id=alignment_run,
        monthly_phase2_run_id=monthly_run,
        legacy_archive_run_id=legacy_archive_run,
        aligned_archive_run_id=aligned_archive_run,
        readout_source_archive_run_id=aligned_archive_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_blocks,
    )

    seeded_payload = _seeded_report(seeded_run)
    alignment_payload = _alignment_report(alignment_run)
    active_summary = _active_scenario_summary(seeded_payload)

    overall_decision = (
        "keep_minimal_two_block_kernel"
        if str(active_summary.get("decision") or "") == "keep"
        and str(dict(alignment_payload.get("archive_alignment_gate") or {}).get("decision") or "") == "keep"
        else "revert_two_block_kernel"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    active_terminal_path = analysis_dir / "two_block_active_terminal_deltas.png"
    _plot_active_terminal_deltas(_active_terminal_rows(seeded_payload), active_terminal_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "monthly_phase2_run_id": monthly_run,
        "legacy_archive_run_id": legacy_archive_run,
        "aligned_archive_run_id": aligned_archive_run,
        "active_block_subset": list(active_blocks),
        "active_scenarios": list(ACTIVE_SCENARIOS),
        "active_scenario_summary": active_summary,
        "archive_alignment_gate": dict(alignment_payload.get("archive_alignment_gate") or {}),
        "overall_decision": overall_decision,
        "artifacts": {
            "seeded_report": str(Path(seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "alignment_report": str(Path(alignment_run) / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.md"),
            "active_terminal_deltas": active_terminal_path.name,
            "aligned_structural_graph": str(
                Path(seeded_run) / "analysis" / str(dict(seeded_payload.get("artifacts") or {}).get("structural_graph") or "")
            ),
            "aligned_terminal_delta_heatmap": str(
                Path(seeded_run) / "analysis" / str(dict(seeded_payload.get("artifacts") or {}).get("terminal_delta_heatmap") or "")
            ),
            "archive_alignment_compare": str(
                Path(alignment_run) / "analysis" / str(dict(alignment_payload.get("artifacts") or {}).get("archive_alignment_compare") or "")
            ),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_two_block_kernel_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_two_block_kernel_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal 2-block seeded-kernel falsification pass.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    parser.add_argument("--legacy-archive-run-id", default=None)
    parser.add_argument("--aligned-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    parser.add_argument("--active-block", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_two_block_kernel_batch(
        run_id=str(args.run_id),
        monthly_phase2_run_id=args.monthly_phase2_run_id,
        legacy_archive_run_id=args.legacy_archive_run_id,
        aligned_archive_run_id=args.aligned_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
        active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
