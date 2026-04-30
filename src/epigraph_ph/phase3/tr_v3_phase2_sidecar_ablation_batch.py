from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_monthly_loading_sanity_batch as loading_sanity
from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as monthly_lane
from epigraph_ph.phase3 import tr_v3_phase2_seeded_gate_batch as seeded_gate
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_ADDITIONAL_EXCLUDED_CANONICALS: tuple[str, ...] = (
    "tested_for_viral_load",
    "virally_suppressed",
    "suppression_outcomes",
    "viral_suppression_rate",
)


def _latest_outcome_lite_monthly_run() -> str:
    candidates = sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-monthly-phase2-lane-*-outcome-lite"))
    if not candidates:
        raise FileNotFoundError("No outcome-lite monthly Phase 2 lane run found.")
    return str(candidates[-1].name)


def _latest_outcome_lite_seeded_run() -> str:
    candidates = sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-phase2-seeded-gates-*-outcome-lite-aligned-seeded"))
    if not candidates:
        raise FileNotFoundError("No outcome-lite aligned seeded run found.")
    return str(candidates[-1].name)


def _monthly_lane_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing monthly lane report: {path}")
    return payload


def _loading_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing monthly loading report: {path}")
    return payload


def _seeded_gate_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_gate_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing seeded gate report: {path}")
    return payload


def _phase2_payload(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "phase2" / "phase2_structural_payload.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing Phase 2 structural payload: {path}")
    return payload


def _find_loading_run_for_monthly_run(monthly_run_id: str) -> str | None:
    for candidate in sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-monthly-loading-sanity-*"), reverse=True):
        payload = read_json(candidate / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json", default={})
        if isinstance(payload, dict) and str(payload.get("monthly_phase2_run_id") or "") == str(monthly_run_id):
            return str(candidate.name)
    return None


def _merge_exclusions(base_report: dict[str, Any], additional: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    base = list(dict(base_report.get("structural_exclusion_summary") or {}).get("excluded_canonicals") or [])
    merged = sorted({str(name).strip() for name in list(base) + list(additional) if str(name).strip()})
    return tuple(merged)


def _block_summary_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["block_id"]): dict(row) for row in list(payload.get("block_summary_rows") or []) if row.get("block_id")}


def _edge_lookup(phase2_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["source"]), str(row["target"])): dict(row)
        for row in list(phase2_payload.get("direct_temporal_edge_rows") or [])
        if row.get("source") and row.get("target")
    }


def _scenario_filtered_sign_agreement(rows: list[dict[str, Any]], *, scenarios: set[str]) -> float:
    filtered = [dict(row) for row in rows if str(row.get("scenario") or "") in scenarios]
    if not filtered:
        return 0.0
    return float(np.mean([1.0 if bool(row.get("sign_agrees")) else 0.0 for row in filtered]))


def _keep_decision(
    *,
    baseline_loading: dict[str, Any],
    ablated_loading: dict[str, Any],
    gate_payload: dict[str, Any],
    ablated_phase2: dict[str, Any],
) -> dict[str, Any]:
    baseline_blocks = _block_summary_lookup(baseline_loading)
    ablated_blocks = _block_summary_lookup(ablated_loading)
    suppression_before = float(dict(baseline_blocks.get("suppression_capacity") or {}).get("cascade_loading_share") or 0.0)
    suppression_after = float(dict(ablated_blocks.get("suppression_capacity") or {}).get("cascade_loading_share") or 0.0)
    annual_before = float(dict(baseline_blocks.get("suppression_capacity") or {}).get("annual_only_loading_share") or 0.0)
    annual_after = float(dict(ablated_blocks.get("suppression_capacity") or {}).get("annual_only_loading_share") or 0.0)
    champion_clean = all(float(row.get("champion_loading_share") or 0.0) == 0.0 for row in list(ablated_loading.get("block_summary_rows") or []))
    edge_rows = {
        (str(row["source"]), str(row["target"])): dict(row)
        for row in list(dict(gate_payload.get("edge_stability") or {}).get("edge_summary_rows") or [])
        if row.get("source") and row.get("target")
    }
    key_edge = dict(edge_rows.get(("care_access_continuity", "suppression_capacity")) or {})
    active_sign_agreement = _scenario_filtered_sign_agreement(
        list(dict(gate_payload.get("archive_alignment") or {}).get("rows") or []),
        scenarios={"disruption_recovery", "mobility_spike"},
    )
    keep = (
        bool(key_edge.get("pass_keep_gate"))
        and champion_clean
        and suppression_after < suppression_before
        and annual_after <= annual_before
        and active_sign_agreement >= 0.75
        and str(dict(gate_payload.get("gate_summary") or {}).get("gate_02_status") or "") == "keep"
        and str(dict(gate_payload.get("gate_summary") or {}).get("gate_03_status") or "") == "keep"
    )
    reasons = {
        "key_edge_pass_keep_gate": bool(key_edge.get("pass_keep_gate")),
        "champion_loading_share_zero": champion_clean,
        "suppression_cascade_loading_share_before": suppression_before,
        "suppression_cascade_loading_share_after": suppression_after,
        "suppression_annual_only_share_before": annual_before,
        "suppression_annual_only_share_after": annual_after,
        "active_sign_agreement": active_sign_agreement,
        "readout_gate": str(dict(gate_payload.get("gate_summary") or {}).get("gate_02_status") or ""),
        "boundedness_gate": str(dict(gate_payload.get("gate_summary") or {}).get("gate_03_status") or ""),
        "retained_block_count": int(len(list(ablated_phase2.get("block_axis") or []))),
    }
    return {
        "decision": "keep" if keep else "revert",
        "reasons": reasons,
    }


def _plot_block_share_compare(
    *,
    baseline_payload: dict[str, Any],
    ablated_payload: dict[str, Any],
    path: Path,
) -> None:
    baseline = _block_summary_lookup(baseline_payload)
    ablated = _block_summary_lookup(ablated_payload)
    blocks = sorted(set(baseline) | set(ablated))
    metrics = [
        ("cascade_loading_share", "Cascade sidecar share"),
        ("annual_only_loading_share", "Annual-only share"),
        ("champion_loading_share", "Champion share"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, max(4.5, len(blocks) * 0.75)))
    axes_list = list(np.asarray(axes).reshape(-1))
    y = np.arange(len(blocks), dtype=np.float64)
    for ax, (field, title) in zip(axes_list, metrics, strict=False):
        before = [float(dict(baseline.get(block) or {}).get(field) or 0.0) for block in blocks]
        after = [float(dict(ablated.get(block) or {}).get(field) or 0.0) for block in blocks]
        ax.barh(y - 0.18, before, height=0.32, label="baseline", color="#4c72b0")
        ax.barh(y + 0.18, after, height=0.32, label="ablated", color="#dd8452")
        ax.set_yticks(y)
        ax.set_yticklabels(blocks)
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", alpha=0.25)
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_active_scenario_compare(gate_payload: dict[str, Any], path: Path) -> None:
    rows = [dict(row) for row in list(dict(gate_payload.get("archive_alignment") or {}).get("rows") or []) if str(row.get("scenario") or "") in {"disruption_recovery", "mobility_spike"}]
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No active scenario rows.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    before = [float(row.get("baseline_terminal_delta") or 0.0) for row in rows]
    after = [float(row.get("aligned_terminal_delta") or 0.0) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.45)))
    ax.barh(y - 0.18, before, height=0.32, label="baseline", color="#4c72b0")
    ax.barh(y + 0.18, after, height=0.32, label="ablated", color="#dd8452")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Active scenario terminal deltas: baseline vs sidecar ablation")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("keep_revert") or {})
    reasons = dict(decision.get("reasons") or {})
    lines = [
        "# TR-V3 Phase 2 Sidecar Ablation Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Base monthly run: `{payload['base_monthly_run_id']}`",
        f"- Ablated monthly run: `{payload['ablated_monthly_run_id']}`",
        f"- Baseline seeded run: `{payload['baseline_seeded_run_id']}`",
        f"- Source run reused: `{payload['source_run_id']}`",
        "",
        "## Exclusions",
        "",
        f"- Base exclusions: `{', '.join(list(payload.get('base_excluded_canonicals') or [])) or 'none'}`",
        f"- Additional exclusions: `{', '.join(list(payload.get('additional_excluded_canonicals') or [])) or 'none'}`",
        f"- Final exclusions: `{', '.join(list(payload.get('final_excluded_canonicals') or [])) or 'none'}`",
        "",
        "## Structural Change",
        "",
        f"- Retained block count: `{int(payload.get('ablated_phase2_summary', {}).get('block_count') or 0)}`",
        f"- Retained direct edge count: `{int(payload.get('ablated_phase2_summary', {}).get('direct_temporal_edge_count') or 0)}`",
        "",
        "## Loading Change",
        "",
        f"- Suppression cascade share: `{float(reasons.get('suppression_cascade_loading_share_before') or 0.0):.3f} -> {float(reasons.get('suppression_cascade_loading_share_after') or 0.0):.3f}`",
        f"- Suppression annual-only share: `{float(reasons.get('suppression_annual_only_share_before') or 0.0):.3f} -> {float(reasons.get('suppression_annual_only_share_after') or 0.0):.3f}`",
        f"- Champion share remains zero: `{bool(reasons.get('champion_loading_share_zero'))}`",
        "",
        "## Gate Readout",
        "",
        f"- Active-scenario sign agreement: `{float(reasons.get('active_sign_agreement') or 0.0):.3f}`",
        f"- Key edge keep gate: `{bool(reasons.get('key_edge_pass_keep_gate'))}`",
        f"- Readout gate: `{reasons.get('readout_gate')}`",
        f"- Boundedness gate: `{reasons.get('boundedness_gate')}`",
        "",
        "## Decision",
        "",
        f"- Keep / revert: `{decision.get('decision')}`",
        "",
        "## Artifacts",
        "",
        "- `analysis/block_share_compare.png`",
        "- `analysis/active_scenario_terminal_delta_compare.png`",
        f"- `analysis/seeded_gate_report.md`: `{payload['artifacts']['gate_report']}`",
        f"- `analysis/loading_report.md`: `{payload['artifacts']['loading_report']}`",
        f"- `analysis/monthly_lane_report.md`: `{payload['artifacts']['monthly_lane_report']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_phase2_sidecar_ablation_batch(
    *,
    run_id: str,
    base_monthly_run_id: str | None = None,
    baseline_seeded_run_id: str | None = None,
    additional_excluded_canonicals: tuple[str, ...] | list[str] = DEFAULT_ADDITIONAL_EXCLUDED_CANONICALS,
    forecast_horizon_quarters: int = 8,
) -> dict[str, Any]:
    base_monthly_run = str(base_monthly_run_id or _latest_outcome_lite_monthly_run())
    baseline_seeded_run = str(baseline_seeded_run_id or _latest_outcome_lite_seeded_run())
    base_monthly_report = _monthly_lane_report(base_monthly_run)
    source_run_id = str(base_monthly_report.get("source_run_id") or "")
    if not source_run_id:
        raise ValueError(f"Missing source_run_id in monthly lane report for {base_monthly_run}")
    final_exclusions = _merge_exclusions(base_monthly_report, additional_excluded_canonicals)

    ablated_monthly_run = f"{run_id}-monthly"
    monthly_lane.run_tr_v3_monthly_phase2_lane_batch(
        run_id=ablated_monthly_run,
        source_run_id=source_run_id,
        plugin_id="hiv",
        start_month=str(base_monthly_report.get("start_month") or monthly_lane.DEFAULT_START_MONTH),
        structural_excluded_canonicals=final_exclusions,
    )

    loading_run = f"{run_id}-loading"
    loading_sanity.run_tr_v3_monthly_loading_sanity_batch(
        run_id=loading_run,
        monthly_phase2_run_id=ablated_monthly_run,
    )

    gate_run = f"{run_id}-gates"
    seeded_gate.run_tr_v3_phase2_seeded_gate_batch(
        run_id=gate_run,
        monthly_phase2_run_id=ablated_monthly_run,
        baseline_seeded_run_id=baseline_seeded_run,
        aligned_archive_run_id=ablated_monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )

    baseline_loading_run = _find_loading_run_for_monthly_run(base_monthly_run)
    if baseline_loading_run is None:
        raise FileNotFoundError(f"No monthly loading sanity run found for baseline monthly run {base_monthly_run}")
    baseline_loading = _loading_report(baseline_loading_run)
    ablated_loading = _loading_report(loading_run)
    ablated_gate = _seeded_gate_report(gate_run)
    ablated_monthly_report = _monthly_lane_report(ablated_monthly_run)
    ablated_phase2 = _phase2_payload(ablated_monthly_run)

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    block_share_path = analysis_dir / "block_share_compare.png"
    _plot_block_share_compare(baseline_payload=baseline_loading, ablated_payload=ablated_loading, path=block_share_path)
    scenario_compare_path = analysis_dir / "active_scenario_terminal_delta_compare.png"
    _plot_active_scenario_compare(ablated_gate, scenario_compare_path)

    keep_revert = _keep_decision(
        baseline_loading=baseline_loading,
        ablated_loading=ablated_loading,
        gate_payload=ablated_gate,
        ablated_phase2=ablated_phase2,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "base_monthly_run_id": base_monthly_run,
        "ablated_monthly_run_id": ablated_monthly_run,
        "baseline_seeded_run_id": baseline_seeded_run,
        "source_run_id": source_run_id,
        "base_excluded_canonicals": list(dict(base_monthly_report.get("structural_exclusion_summary") or {}).get("excluded_canonicals") or []),
        "additional_excluded_canonicals": list(additional_excluded_canonicals),
        "final_excluded_canonicals": list(final_exclusions),
        "ablated_phase2_summary": dict(ablated_monthly_report.get("rebuilt_phase2_summary") or {}),
        "keep_revert": keep_revert,
        "artifacts": {
            "monthly_lane_report": str(Path(ablated_monthly_run) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.md"),
            "loading_report": str(Path(loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "gate_report": str(Path(gate_run) / "analysis" / "tr_v3_phase2_seeded_gate_batch_report.md"),
            "block_share_compare": str(block_share_path.name),
            "active_scenario_terminal_delta_compare": str(scenario_compare_path.name),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_sidecar_ablation_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_sidecar_ablation_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an archive-matched cascade-sidecar ablation on the outcome-lite Phase 2 seeded substrate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-monthly-run-id", default=None)
    parser.add_argument("--baseline-seeded-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    parser.add_argument("--exclude-canonical", action="append", default=list(DEFAULT_ADDITIONAL_EXCLUDED_CANONICALS))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_sidecar_ablation_batch(
        run_id=str(args.run_id),
        base_monthly_run_id=args.base_monthly_run_id,
        baseline_seeded_run_id=args.baseline_seeded_run_id,
        additional_excluded_canonicals=tuple(str(value) for value in list(args.exclude_canonical or [])),
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
