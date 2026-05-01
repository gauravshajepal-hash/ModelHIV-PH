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
from epigraph_ph.phase3 import tr_v3_monthly_loading_sanity_batch as loading_sanity
from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as monthly_lane
from epigraph_ph.phase3 import tr_v3_phase2_archive_alignment_batch as archive_alignment
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_phase2_seeded_gate_batch as seeded_gate
from epigraph_ph.phase3 import tr_v3_phase2_two_block_kernel_batch as two_block
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_ACTIVE_BLOCK_ORDER: tuple[str, ...] = (
    "testing_prevention_reach",
    "care_access_continuity",
    "mobility_exposure_pressure",
)
BASELINE_SCENARIOS: tuple[str, ...] = (
    "disruption_recovery",
    "mobility_spike",
)
TESTING_SCENARIOS: tuple[str, ...] = (
    "testing_pulse",
    "testing_plateau",
)


def _latest_two_block_kernel_run() -> str:
    candidates = sorted(
        candidate
        for candidate in (ROOT_DIR / "artifacts" / "runs").glob("tr-v3-phase2-two-block-kernel-*")
        if (candidate / "analysis" / "tr_v3_phase2_two_block_kernel_batch_report.json").exists()
    )
    if not candidates:
        raise FileNotFoundError("No Phase 2 two-block kernel run found.")
    return str(candidates[-1].name)


def _read_report(path: Path) -> dict[str, Any]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing report payload: {path}")
    return payload


def _two_block_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_two_block_kernel_batch_report.json")


def _monthly_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json")


def _loading_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json")


def _seeded_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json")


def _alignment_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.json")


def _phase2_payload(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "phase2" / "phase2_structural_payload.json")


def _block_summary_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["block_id"]): dict(row) for row in list(payload.get("block_summary_rows") or []) if row.get("block_id")}


def _testing_block_support_gate(loading_payload: dict[str, Any], phase2_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(_block_summary_lookup(loading_payload).get("testing_prevention_reach") or {})
    audit_rows = [dict(row) for row in list(loading_payload.get("audit_rows") or []) if str(row.get("block_id") or "") == "testing_prevention_reach"]
    total_abs_loading = float(sum(float(row.get("abs_loading") or 0.0) for row in audit_rows))
    max_indicator_share = float(
        max((float(row.get("abs_loading") or 0.0) / max(total_abs_loading, 1e-9) for row in audit_rows), default=0.0)
    )
    retained = "testing_prevention_reach" in set(str(name) for name in list(phase2_payload.get("block_axis") or []))
    champion_share = float(summary.get("champion_loading_share") or 0.0)
    cascade_share = float(summary.get("cascade_loading_share") or 0.0)
    indicator_count = int(summary.get("indicator_count") or 0)
    decision = (
        "keep"
        if retained
        and indicator_count >= 2
        and max_indicator_share <= 0.70
        and champion_share <= 1e-9
        and cascade_share <= 1e-9
        else "revert"
    )
    return {
        "decision": decision,
        "block_retained": retained,
        "indicator_count": indicator_count,
        "champion_loading_share": champion_share,
        "cascade_loading_share": cascade_share,
        "max_indicator_share": max_indicator_share,
        "indicator_rows": audit_rows,
    }


def _edge_gate(edge_stability: dict[str, Any]) -> dict[str, Any]:
    row = next(
        (
            dict(item)
            for item in list(edge_stability.get("edge_summary_rows") or [])
            if str(item.get("source") or "") == "testing_prevention_reach" and str(item.get("target") or "") == "care_access_continuity"
        ),
        {},
    )
    return {
        "decision": "keep" if bool(row.get("pass_keep_gate")) else "revert",
        "edge_row": row,
    }


def _circularity_gate(
    *,
    monthly_run_id: str,
    aligned_archive_run_id: str,
    phase2_state: seeded.Phase2QuarterState,
    seeded_payload: dict[str, Any],
    forecast_horizon_quarters: int,
) -> dict[str, Any]:
    loading_rows = seeded_gate._load_block_loading_rows(monthly_run_id)
    outcome_rows = seeded_gate._summarize_outcome_circularity_rows(
        loading_rows=loading_rows,
        outcome_canonical_names=set(seeded_gate.OUTCOME_CANONICALS),
    )
    orthogonalized_state, orthogonalization_rows = seeded_gate._orthogonalize_phase2_state_against_outcomes(
        phase2_state=phase2_state,
        archive_run_id=aligned_archive_run_id,
        loading_rows=loading_rows,
    )
    orthogonalized_payload = seeded_gate._seeded_payload_from_phase2_state(
        archive_run_id=aligned_archive_run_id,
        phase2_state=orthogonalized_state,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )
    alignment_rows = seeded_gate._archive_alignment_rows(seeded_payload, orthogonalized_payload)
    sign_agreement = float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in alignment_rows])) if alignment_rows else 0.0
    mean_abs_delta = float(np.mean([abs(float(row["delta_of_delta"])) for row in alignment_rows])) if alignment_rows else 0.0
    ratios = [
        abs(float(row["aligned_terminal_delta"])) / max(abs(float(row["baseline_terminal_delta"])), 1.0)
        for row in alignment_rows
    ]
    mean_abs_ratio = float(np.mean(ratios)) if ratios else 0.0
    return {
        "decision": "keep" if sign_agreement >= 0.75 and 0.4 <= mean_abs_ratio <= 2.0 else "revert",
        "loading_rows": outcome_rows,
        "orthogonalization_rows": orthogonalization_rows,
        "ablation_rows": alignment_rows,
        "sign_agreement_rate": sign_agreement,
        "mean_abs_delta_of_delta": mean_abs_delta,
        "mean_abs_ratio": mean_abs_ratio,
    }


def _scenario_rows(payload: dict[str, Any], scenarios: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(payload.get("terminal_delta_rows") or []) if str(row.get("scenario") or "") in scenarios]


def _baseline_preservation_gate(baseline_seeded_payload: dict[str, Any], candidate_seeded_payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in seeded_gate._archive_alignment_rows(baseline_seeded_payload, candidate_seeded_payload)
        if str(row.get("scenario") or "") in set(BASELINE_SCENARIOS)
    ]
    sign_agreement = float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in rows])) if rows else 0.0
    mean_abs_delta = float(np.mean([abs(float(row["delta_of_delta"])) for row in rows])) if rows else 0.0
    return {
        "decision": "keep" if sign_agreement >= 0.999 and mean_abs_delta <= 30.0 else "revert",
        "rows": rows,
        "sign_agreement_rate": sign_agreement,
        "mean_abs_delta_of_delta": mean_abs_delta,
    }


def _testing_scenario_gate(candidate_seeded_payload: dict[str, Any]) -> dict[str, Any]:
    rows = _scenario_rows(candidate_seeded_payload, set(TESTING_SCENARIOS))
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
        "decision": "keep" if nonnull_count > 0 and max_abs_delta >= 10.0 else "revert",
        "rows": rows,
        "nonnull_terminal_count": int(nonnull_count),
        "max_abs_terminal_delta": max_abs_delta,
    }


def _archive_gate(candidate_payload: dict[str, Any], baseline_two_block_payload: dict[str, Any]) -> dict[str, Any]:
    baseline_gate = dict(baseline_two_block_payload.get("archive_alignment_gate") or {})
    candidate_gate = dict(candidate_payload.get("archive_alignment_gate") or {})
    allowed_mean_abs_delta = float(baseline_gate.get("active_mean_abs_delta") or 0.0) + 10.0
    decision = (
        "keep"
        if float(candidate_gate.get("active_sign_agreement") or 0.0) >= 0.75
        and float(candidate_gate.get("active_mean_abs_delta") or 0.0) <= allowed_mean_abs_delta
        else "revert"
    )
    return {
        **candidate_gate,
        "decision": decision,
        "baseline_active_mean_abs_delta": float(baseline_gate.get("active_mean_abs_delta") or 0.0),
        "allowed_active_mean_abs_delta": allowed_mean_abs_delta,
    }


def _plot_testing_block_loadings(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Testing prevention block", body="Testing block not retained.")
        return
    labels = [str(row["canonical_name"]) for row in rows]
    values = [float(row.get("loading") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(10, max(4.0, len(labels) * 0.55)))
    ax.barh(np.arange(len(labels)), values, color="#4c72b0")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_title("Testing prevention reach block loadings")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_terminal_delta_compare(rows: list[dict[str, Any]], *, title: str, path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title=title, body="No rows to compare.")
        return
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    baseline = [float(row.get("baseline_terminal_delta") or 0.0) for row in rows]
    candidate = [float(row.get("aligned_terminal_delta") or 0.0) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.42)))
    ax.barh(y - 0.18, baseline, height=0.32, label="baseline", color="#4c72b0")
    ax.barh(y + 0.18, candidate, height=0.32, label="candidate", color="#dd8452")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    support = dict(payload.get("support_gate") or {})
    edge = dict(payload.get("edge_gate") or {})
    circularity = dict(payload.get("circularity_gate") or {})
    baseline = dict(payload.get("baseline_preservation_gate") or {})
    testing = dict(payload.get("testing_scenario_gate") or {})
    archive_gate = dict(payload.get("archive_alignment_gate") or {})
    lines = [
        "# TR-V3 Phase 2 Testing Prevention Rebuild Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline two-block run: `{payload['baseline_two_block_run_id']}`",
        f"- Base monthly run: `{payload['base_monthly_run_id']}`",
        f"- Candidate monthly run: `{payload['candidate_monthly_run_id']}`",
        f"- Source run reused: `{payload['source_run_id']}`",
        f"- Active block subset: `{list(payload.get('active_block_subset') or [])}`",
        f"- Additional exclusions: `{', '.join(list(payload.get('structural_exclusions') or [])) or 'none'}`",
        "",
        "## Support gate",
        "",
        f"- Decision: `{support.get('decision')}`",
        f"- Block retained: `{bool(support.get('block_retained'))}`",
        f"- Indicator count: `{int(support.get('indicator_count') or 0)}`",
        f"- Champion loading share: `{float(support.get('champion_loading_share') or 0.0):.3f}`",
        f"- Cascade loading share: `{float(support.get('cascade_loading_share') or 0.0):.3f}`",
        f"- Max indicator share: `{float(support.get('max_indicator_share') or 0.0):.3f}`",
        "",
        "## Structural and circularity gates",
        "",
        f"- Edge gate: `{edge.get('decision')}`",
        f"- Circularity gate: `{circularity.get('decision')}`",
        f"- Circularity sign agreement: `{float(circularity.get('sign_agreement_rate') or 0.0):.3f}`",
        f"- Circularity mean abs ratio: `{float(circularity.get('mean_abs_ratio') or 0.0):.3f}`",
        "",
        "## Scenario gates",
        "",
        f"- Baseline preservation: `{baseline.get('decision')}`",
        f"- Baseline sign agreement: `{float(baseline.get('sign_agreement_rate') or 0.0):.3f}`",
        f"- Baseline mean abs delta-of-delta: `{float(baseline.get('mean_abs_delta_of_delta') or 0.0):.3f}`",
        f"- Testing scenario gate: `{testing.get('decision')}`",
        f"- Testing non-null terminal count: `{int(testing.get('nonnull_terminal_count') or 0)}`",
        f"- Testing max abs terminal delta: `{float(testing.get('max_abs_terminal_delta') or 0.0):.3f}`",
        "",
        "## Archive alignment",
        "",
        f"- Decision: `{archive_gate.get('decision')}`",
        f"- Active sign agreement: `{float(archive_gate.get('active_sign_agreement') or 0.0):.3f}`",
        f"- Active mean abs delta-of-delta: `{float(archive_gate.get('active_mean_abs_delta') or 0.0):.3f}`",
        f"- Allowed active mean abs delta-of-delta: `{float(archive_gate.get('allowed_active_mean_abs_delta') or 0.0):.3f}`",
        "",
        "## Overall",
        "",
        f"- Decision: `{payload.get('overall_decision')}`",
        "",
        "## Artifacts",
        "",
        f"- `candidate_seeded_report`: `{payload['artifacts']['candidate_seeded_report']}`",
        f"- `candidate_alignment_report`: `{payload['artifacts']['candidate_alignment_report']}`",
        f"- `candidate_loading_report`: `{payload['artifacts']['candidate_loading_report']}`",
        f"- `testing_block_loadings`: `{payload['artifacts']['testing_block_loadings']}`",
        f"- `testing_terminal_deltas`: `{payload['artifacts']['testing_terminal_deltas']}`",
        f"- `baseline_preservation_compare`: `{payload['artifacts']['baseline_preservation_compare']}`",
        f"- `circularity_ablation_compare`: `{payload['artifacts']['circularity_ablation_compare']}`",
        f"- `edge_stability_graph`: `{payload['artifacts']['edge_stability_graph']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_phase2_testing_prevention_rebuild_batch(
    *,
    run_id: str,
    baseline_two_block_run_id: str | None = None,
    base_monthly_run_id: str | None = None,
    legacy_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
    additional_excluded_canonicals: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    baseline_two_block_run = str(baseline_two_block_run_id or _latest_two_block_kernel_run())
    baseline_two_block_payload = _two_block_report(baseline_two_block_run)
    base_monthly_run = str(base_monthly_run_id or baseline_two_block_payload.get("monthly_phase2_run_id") or "")
    if not base_monthly_run:
        raise ValueError(f"Missing base monthly run in baseline two-block report for {baseline_two_block_run}")
    base_monthly_payload = _monthly_report(base_monthly_run)
    source_run_id = str(base_monthly_payload.get("source_run_id") or "")
    if not source_run_id:
        raise ValueError(f"Missing source_run_id in monthly lane report for {base_monthly_run}")
    base_structural_exclusions = tuple(
        str(name)
        for name in list(dict(base_monthly_payload.get("structural_exclusion_summary") or {}).get("excluded_canonicals") or [])
    )
    structural_exclusions = tuple(
        sorted(
            {
                *[str(name).strip() for name in list(base_structural_exclusions) if str(name).strip()],
                *[str(name).strip() for name in list(additional_excluded_canonicals) if str(name).strip()],
            }
        )
    )
    legacy_archive_run = str(
        legacy_archive_run_id
        or baseline_two_block_payload.get("legacy_archive_run_id")
        or suite._latest_standard_archive_run()
    )

    candidate_monthly_run = f"{run_id}-monthly"
    monthly_lane.run_tr_v3_monthly_phase2_lane_batch(
        run_id=candidate_monthly_run,
        source_run_id=source_run_id,
        plugin_id="hiv",
        start_month=str(base_monthly_payload.get("start_month") or monthly_lane.DEFAULT_START_MONTH),
        structural_excluded_canonicals=structural_exclusions,
    )

    candidate_loading_run = f"{run_id}-loading"
    loading_sanity.run_tr_v3_monthly_loading_sanity_batch(
        run_id=candidate_loading_run,
        monthly_phase2_run_id=candidate_monthly_run,
    )

    candidate_phase2_payload = _phase2_payload(candidate_monthly_run)
    active_blocks = tuple(name for name in DEFAULT_ACTIVE_BLOCK_ORDER if name in set(str(item) for item in list(candidate_phase2_payload.get("block_axis") or [])))
    if not active_blocks:
        active_blocks = tuple(two_block.DEFAULT_ACTIVE_BLOCK_SUBSET)

    candidate_seeded_run = f"{run_id}-seeded"
    seeded.run_tr_v3_phase2_seeded_champion_batch(
        run_id=candidate_seeded_run,
        archive_run_id=candidate_monthly_run,
        readout_source_archive_run_id=candidate_monthly_run,
        monthly_phase2_run_id=candidate_monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_blocks,
    )

    candidate_alignment_run = f"{run_id}-alignment"
    archive_alignment.run_tr_v3_phase2_archive_alignment_batch(
        run_id=candidate_alignment_run,
        monthly_phase2_run_id=candidate_monthly_run,
        legacy_archive_run_id=legacy_archive_run,
        aligned_archive_run_id=candidate_monthly_run,
        readout_source_archive_run_id=candidate_monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_blocks,
    )

    candidate_loading_payload = _loading_report(candidate_loading_run)
    candidate_seeded_payload = _seeded_report(candidate_seeded_run)
    candidate_alignment_payload = _alignment_report(candidate_alignment_run)
    baseline_seeded_payload = _seeded_report(f"{baseline_two_block_run}-aligned-seeded")

    phase2_state = seeded._filter_phase2_quarter_state(
        seeded._load_phase2_quarter_state(candidate_monthly_run),
        active_block_subset=active_blocks,
    )
    edge_stability = seeded_gate._retained_edge_stability(phase2_state=phase2_state)

    support_gate = _testing_block_support_gate(candidate_loading_payload, candidate_phase2_payload)
    edge_gate = _edge_gate(edge_stability)
    circularity_gate = _circularity_gate(
        monthly_run_id=candidate_monthly_run,
        aligned_archive_run_id=candidate_monthly_run,
        phase2_state=phase2_state,
        seeded_payload=candidate_seeded_payload,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )
    baseline_preservation_gate = _baseline_preservation_gate(baseline_seeded_payload, candidate_seeded_payload)
    testing_scenario_gate = _testing_scenario_gate(candidate_seeded_payload)
    archive_gate = _archive_gate(candidate_alignment_payload, baseline_two_block_payload)

    overall_decision = (
        "keep_testing_prevention_reentry"
        if all(
            str(gate.get("decision") or "") == "keep"
            for gate in (
                support_gate,
                edge_gate,
                circularity_gate,
                baseline_preservation_gate,
                testing_scenario_gate,
                archive_gate,
            )
        )
        else "revert_to_two_block_kernel"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    testing_loadings_graph = analysis_dir / "testing_prevention_block_loadings.png"
    testing_delta_graph = analysis_dir / "testing_prevention_terminal_deltas.png"
    baseline_graph = analysis_dir / "baseline_preservation_compare.png"
    circularity_graph = analysis_dir / "circularity_ablation_compare.png"
    edge_graph = analysis_dir / "testing_edge_stability_heatmap.png"

    _plot_testing_block_loadings(list(support_gate.get("indicator_rows") or []), testing_loadings_graph)
    _plot_terminal_delta_compare(
        [
            {
                "contract": row["contract"],
                "scenario": row["scenario"],
                "metric": metric_name,
                "baseline_terminal_delta": 0.0,
                "aligned_terminal_delta": float(row.get(f"{metric_name}_delta") or 0.0),
            }
            for row in list(testing_scenario_gate.get("rows") or [])
            for metric_name in seeded.METRIC_PLOT_ORDER
        ],
        title="Testing scenario terminal deltas",
        path=testing_delta_graph,
    )
    _plot_terminal_delta_compare(
        list(baseline_preservation_gate.get("rows") or []),
        title="Baseline preservation vs frozen 2-block kernel",
        path=baseline_graph,
    )
    _plot_terminal_delta_compare(
        list(circularity_gate.get("ablation_rows") or []),
        title="Outcome-circularity ablation compare",
        path=circularity_graph,
    )
    edge_rows = list(edge_stability.get("edge_summary_rows") or [])
    if edge_rows:
        seeded_gate._plot_edge_stability_heatmap(edge_rows, edge_graph)
    else:
        suite._plot_placeholder(edge_graph, title="Testing edge stability", body="No retained testing edges under this variant.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_two_block_run_id": baseline_two_block_run,
        "base_monthly_run_id": base_monthly_run,
        "candidate_monthly_run_id": candidate_monthly_run,
        "candidate_loading_run_id": candidate_loading_run,
        "candidate_seeded_run_id": candidate_seeded_run,
        "candidate_alignment_run_id": candidate_alignment_run,
        "source_run_id": source_run_id,
        "base_structural_exclusions": list(base_structural_exclusions),
        "structural_exclusions": list(structural_exclusions),
        "active_block_subset": list(active_blocks),
        "support_gate": support_gate,
        "edge_gate": edge_gate,
        "circularity_gate": circularity_gate,
        "baseline_preservation_gate": baseline_preservation_gate,
        "testing_scenario_gate": testing_scenario_gate,
        "archive_alignment_gate": archive_gate,
        "overall_decision": overall_decision,
        "artifacts": {
            "candidate_seeded_report": str(Path(candidate_seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "candidate_alignment_report": str(Path(candidate_alignment_run) / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.md"),
            "candidate_loading_report": str(Path(candidate_loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "testing_block_loadings": testing_loadings_graph.name,
            "testing_terminal_deltas": testing_delta_graph.name,
            "baseline_preservation_compare": baseline_graph.name,
            "circularity_ablation_compare": circularity_graph.name,
            "edge_stability_graph": edge_graph.name,
            "candidate_structural_scenarios": str(Path(candidate_seeded_run) / "analysis" / "phase2_seeded_structural_scenarios.png"),
            "candidate_terminal_delta_heatmap": str(Path(candidate_seeded_run) / "analysis" / "phase2_seeded_terminal_delta_heatmap.png"),
            "archive_alignment_compare": str(Path(candidate_alignment_run) / "analysis" / "archive_alignment_compare.png"),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_testing_prevention_rebuild_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_testing_prevention_rebuild_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild a non-outcome testing_prevention_reach block and compare it against the frozen 2-block seeded kernel.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-two-block-run-id", default=None)
    parser.add_argument("--base-monthly-run-id", default=None)
    parser.add_argument("--legacy-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_testing_prevention_rebuild_batch(
        run_id=str(args.run_id),
        baseline_two_block_run_id=args.baseline_two_block_run_id,
        base_monthly_run_id=args.base_monthly_run_id,
        legacy_archive_run_id=args.legacy_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
