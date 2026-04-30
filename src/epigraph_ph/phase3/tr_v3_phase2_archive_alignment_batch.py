from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_monthly_loading_sanity_batch as loading_sanity
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


def _latest_sidecar_ablation_monthly_run() -> str:
    candidates = sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-phase2-sidecar-ablation-*-monthly"))
    if not candidates:
        raise FileNotFoundError("No sidecar-ablation monthly run found.")
    return str(candidates[-1].name)


def _find_loading_run_for_monthly_run(monthly_run_id: str) -> str | None:
    for candidate in sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-monthly-loading-sanity-*"), reverse=True):
        payload = read_json(candidate / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json", default={})
        if isinstance(payload, dict) and str(payload.get("monthly_phase2_run_id") or "") == str(monthly_run_id):
            return str(candidate.name)
    return None


def _seeded_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing seeded champion report: {path}")
    return payload


def _loading_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing loading sanity report: {path}")
    return payload


def _terminal_delta_lookup(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in list(payload.get("terminal_delta_rows") or []):
        contract = str(row.get("contract") or "")
        scenario = str(row.get("scenario") or "")
        for metric_name in seeded.METRIC_PLOT_ORDER:
            lookup[(contract, scenario, metric_name)] = float(row.get(f"{metric_name}_delta") or 0.0)
    return lookup


def _alignment_rows(
    *,
    legacy_payload: dict[str, Any],
    aligned_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    legacy_lookup = _terminal_delta_lookup(legacy_payload)
    aligned_lookup = _terminal_delta_lookup(aligned_payload)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(legacy_lookup) | set(aligned_lookup)):
        legacy_value = float(legacy_lookup.get(key) or 0.0)
        aligned_value = float(aligned_lookup.get(key) or 0.0)
        rows.append(
            {
                "contract": str(key[0]),
                "scenario": str(key[1]),
                "metric": str(key[2]),
                "legacy_terminal_delta": legacy_value,
                "aligned_terminal_delta": aligned_value,
                "delta_of_delta": float(aligned_value - legacy_value),
                "sign_agrees": bool(np.sign(legacy_value) == np.sign(aligned_value)),
            }
        )
    return rows


def _scenario_sign_agreement(rows: list[dict[str, Any]], *, scenarios: set[str] | None = None) -> float:
    filtered = [row for row in rows if scenarios is None or str(row.get("scenario") or "") in scenarios]
    if not filtered:
        return 0.0
    return float(np.mean([1.0 if bool(row.get("sign_agrees")) else 0.0 for row in filtered]))


def _mean_abs_delta(rows: list[dict[str, Any]], *, scenarios: set[str] | None = None) -> float:
    filtered = [row for row in rows if scenarios is None or str(row.get("scenario") or "") in scenarios]
    if not filtered:
        return 0.0
    return float(np.mean([abs(float(row.get("delta_of_delta") or 0.0)) for row in filtered]))


def _context_proxy_share(loading_payload: dict[str, Any], *, block_id: str) -> float:
    rows = [dict(row) for row in list(loading_payload.get("audit_rows") or []) if str(row.get("block_id") or "") == str(block_id)]
    total = float(sum(abs(float(row.get("loading") or 0.0)) for row in rows))
    if total <= 0.0:
        return 0.0
    return float(sum(abs(float(row.get("loading") or 0.0)) for row in rows if str(row.get("category") or "") == "context_or_proxy") / total)


def _adequacy_gate(loading_payload: dict[str, Any], *, active_block_subset: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    active_blocks = [str(name).strip() for name in list(active_block_subset or []) if str(name).strip()]
    if active_blocks and "suppression_capacity" not in active_blocks:
        return {
            "decision": "not_applicable",
            "reasons": {
                "indicator_count": 0,
                "singleton_loading_share": 0.0,
                "mean_ppc_corr": 0.0,
                "context_or_proxy_share": 0.0,
                "suppression_inactive": True,
                "active_block_subset": list(active_blocks),
            },
        }
    block_rows = {str(row["block_id"]): dict(row) for row in list(loading_payload.get("block_summary_rows") or []) if row.get("block_id")}
    suppression = dict(block_rows.get("suppression_capacity") or {})
    reasons = {
        "indicator_count": int(suppression.get("indicator_count") or 0),
        "singleton_loading_share": float(suppression.get("singleton_loading_share") or 0.0),
        "mean_ppc_corr": float(suppression.get("mean_ppc_corr") or 0.0),
        "context_or_proxy_share": float(_context_proxy_share(loading_payload, block_id="suppression_capacity")),
    }
    keep = (
        int(reasons["indicator_count"]) >= 2
        and float(reasons["singleton_loading_share"]) <= 0.5
        and float(reasons["mean_ppc_corr"]) >= 0.25
        and float(reasons["context_or_proxy_share"]) < 0.8
    )
    return {
        "decision": "keep" if keep else "revert",
        "reasons": reasons,
    }


def _plot_alignment_compare(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    legacy = [float(row.get("legacy_terminal_delta") or 0.0) for row in rows]
    aligned = [float(row.get("aligned_terminal_delta") or 0.0) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.42)))
    ax.barh(y - 0.18, legacy, height=0.32, label="legacy archive", color="#4c72b0")
    ax.barh(y + 0.18, aligned, height=0.32, label="aligned archive", color="#dd8452")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Frozen-readout archive alignment: terminal deltas")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_adequacy(reasons: dict[str, Any], path: Path) -> None:
    labels = ["indicator_count", "singleton_share", "mean_ppc_corr", "context_proxy_share"]
    values = [
        float(reasons.get("indicator_count") or 0.0),
        float(reasons.get("singleton_loading_share") or 0.0),
        float(reasons.get("mean_ppc_corr") or 0.0),
        float(reasons.get("context_or_proxy_share") or 0.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, values, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    ax.set_title("Suppression-capacity adequacy gate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    archive_gate = dict(payload.get("archive_alignment_gate") or {})
    adequacy = dict(payload.get("adequacy_gate") or {})
    lines = [
        "# TR-V3 Phase 2 Archive Alignment Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Monthly run: `{payload['monthly_phase2_run_id']}`",
        f"- Legacy archive: `{payload['legacy_archive_run_id']}`",
        f"- Aligned archive: `{payload['aligned_archive_run_id']}`",
        f"- Frozen readout source archive: `{payload['readout_source_archive_run_id']}`",
        f"- Active block subset: `{list(payload.get('active_block_subset') or []) or 'full block axis'}`",
        "",
        "## Archive Alignment Gate",
        "",
        f"- Full sign agreement: `{float(archive_gate.get('full_sign_agreement') or 0.0):.3f}`",
        f"- Full mean abs delta-of-delta: `{float(archive_gate.get('full_mean_abs_delta') or 0.0):.3f}`",
        f"- Active-scenario sign agreement: `{float(archive_gate.get('active_sign_agreement') or 0.0):.3f}`",
        f"- Active-scenario mean abs delta-of-delta: `{float(archive_gate.get('active_mean_abs_delta') or 0.0):.3f}`",
        f"- Decision: `{archive_gate.get('decision')}`",
        "",
        "## Adequacy Gate",
        "",
        f"- Suppression indicator count: `{int(dict(adequacy.get('reasons') or {}).get('indicator_count') or 0)}`",
        f"- Suppression singleton loading share: `{float(dict(adequacy.get('reasons') or {}).get('singleton_loading_share') or 0.0):.3f}`",
        f"- Suppression mean PPC corr: `{float(dict(adequacy.get('reasons') or {}).get('mean_ppc_corr') or 0.0):.3f}`",
        f"- Suppression context/proxy share: `{float(dict(adequacy.get('reasons') or {}).get('context_or_proxy_share') or 0.0):.3f}`",
        f"- Suppression inactive: `{bool(dict(adequacy.get('reasons') or {}).get('suppression_inactive'))}`",
        f"- Decision: `{adequacy.get('decision')}`",
        "",
        "## Overall",
        "",
        f"- Decision: `{payload.get('overall_decision')}`",
        "",
        "## Artifacts",
        "",
        "- `analysis/archive_alignment_compare.png`",
        "- `analysis/suppression_capacity_adequacy.png`",
        f"- `analysis/legacy_seeded_report.md`: `{payload['artifacts']['legacy_seeded_report']}`",
        f"- `analysis/aligned_seeded_report.md`: `{payload['artifacts']['aligned_seeded_report']}`",
        f"- `analysis/loading_report.md`: `{payload['artifacts']['loading_report']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_phase2_archive_alignment_batch(
    *,
    run_id: str,
    monthly_phase2_run_id: str | None = None,
    legacy_archive_run_id: str | None = None,
    aligned_archive_run_id: str | None = None,
    readout_source_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
    active_block_subset: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    monthly_run = str(monthly_phase2_run_id or _latest_sidecar_ablation_monthly_run())
    legacy_archive_run = str(legacy_archive_run_id or suite._latest_standard_archive_run())
    aligned_archive_run = str(aligned_archive_run_id or monthly_run)
    readout_source_archive_run = str(readout_source_archive_run_id or aligned_archive_run)

    loading_run = _find_loading_run_for_monthly_run(monthly_run)
    if loading_run is None:
        loading_run = f"{run_id}-loading"
        loading_sanity.run_tr_v3_monthly_loading_sanity_batch(
            run_id=loading_run,
            monthly_phase2_run_id=monthly_run,
        )

    legacy_seeded_run = f"{run_id}-legacy"
    aligned_seeded_run = f"{run_id}-aligned"
    seeded.run_tr_v3_phase2_seeded_champion_batch(
        run_id=legacy_seeded_run,
        archive_run_id=legacy_archive_run,
        readout_source_archive_run_id=readout_source_archive_run,
        monthly_phase2_run_id=monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_block_subset,
    )
    seeded.run_tr_v3_phase2_seeded_champion_batch(
        run_id=aligned_seeded_run,
        archive_run_id=aligned_archive_run,
        readout_source_archive_run_id=readout_source_archive_run,
        monthly_phase2_run_id=monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_block_subset,
    )

    legacy_payload = _seeded_report(legacy_seeded_run)
    aligned_payload = _seeded_report(aligned_seeded_run)
    loading_payload = _loading_report(loading_run)

    rows = _alignment_rows(legacy_payload=legacy_payload, aligned_payload=aligned_payload)
    archive_alignment_gate = {
        "full_sign_agreement": _scenario_sign_agreement(rows, scenarios=None),
        "full_mean_abs_delta": _mean_abs_delta(rows, scenarios=None),
        "active_sign_agreement": _scenario_sign_agreement(rows, scenarios={"disruption_recovery", "mobility_spike"}),
        "active_mean_abs_delta": _mean_abs_delta(rows, scenarios={"disruption_recovery", "mobility_spike"}),
    }
    archive_alignment_gate["decision"] = (
        "keep"
        if archive_alignment_gate["active_sign_agreement"] >= 0.75 and archive_alignment_gate["active_mean_abs_delta"] <= 132.533
        else "revisit"
    )
    adequacy_gate = _adequacy_gate(loading_payload, active_block_subset=active_block_subset)
    overall_decision = (
        "proceed_to_testing_branch"
        if archive_alignment_gate["decision"] == "keep" and adequacy_gate["decision"] == "keep"
        else "keep_active_subset_kernel"
        if archive_alignment_gate["decision"] == "keep" and adequacy_gate["decision"] == "not_applicable"
        else "stop_before_testing_rebuild"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    alignment_path = analysis_dir / "archive_alignment_compare.png"
    _plot_alignment_compare(rows, alignment_path)
    adequacy_path = analysis_dir / "suppression_capacity_adequacy.png"
    _plot_adequacy(dict(adequacy_gate.get("reasons") or {}), adequacy_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "monthly_phase2_run_id": monthly_run,
        "legacy_archive_run_id": legacy_archive_run,
        "aligned_archive_run_id": aligned_archive_run,
        "readout_source_archive_run_id": readout_source_archive_run,
        "active_block_subset": list(active_block_subset or []),
        "archive_alignment_gate": archive_alignment_gate,
        "adequacy_gate": adequacy_gate,
        "overall_decision": overall_decision,
        "artifacts": {
            "legacy_seeded_report": str(Path(legacy_seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "aligned_seeded_report": str(Path(aligned_seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "loading_report": str(Path(loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "archive_alignment_compare": alignment_path.name,
            "suppression_capacity_adequacy": adequacy_path.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_archive_alignment_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_archive_alignment_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen-readout archive alignment and adequacy gates on the current Phase 2 seeded substrate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    parser.add_argument("--legacy-archive-run-id", default=None)
    parser.add_argument("--aligned-archive-run-id", default=None)
    parser.add_argument("--readout-source-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    parser.add_argument("--active-block", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_archive_alignment_batch(
        run_id=str(args.run_id),
        monthly_phase2_run_id=args.monthly_phase2_run_id,
        legacy_archive_run_id=args.legacy_archive_run_id,
        aligned_archive_run_id=args.aligned_archive_run_id,
        readout_source_archive_run_id=args.readout_source_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
        active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
