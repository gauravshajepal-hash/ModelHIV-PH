from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.phase3_target_contract import PHASE3_MODULE_CONTRACT
from epigraph_ph.phase2.latent_temporal_graph import (
    _indicator_names_by_block,
    _load_uncertainty_tensors,
    _phase15_phi_by_block,
    estimate_latent_temporal_scale_graph,
)
from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json


FALSIFICATION_SCHEMA_VERSION = "phase2_edge_source_ablation_time_window_falsification.v1"
PRIOR_SUPPORT_ALLOWED_USES = {
    "",
    "anchor_context_only",
    "direct_determinant_covariate_candidate",
    "prior_context_only",
    "reporting_process_context_only",
}


@dataclass(frozen=True, slots=True)
class TimeWindowSpec:
    window_id: str
    start_year: int
    end_year: int
    falsification_window: bool
    scientific_role: str


DEFAULT_TIME_WINDOWS: tuple[TimeWindowSpec, ...] = (
    TimeWindowSpec(
        "pre_monthly_2010_2019",
        2010,
        2019,
        True,
        "Early determinant and annual-support era; often not estimable for month-lag edges.",
    ),
    TimeWindowSpec(
        "monthly_rollout_2020_2021",
        2020,
        2021,
        True,
        "Dense monthly HARP/HASP-era support window.",
    ),
    TimeWindowSpec(
        "expanded_support_2022_2025",
        2022,
        2025,
        True,
        "Expanded archive holdout/support window.",
    ),
    TimeWindowSpec(
        "full_2010_2025",
        2010,
        2025,
        False,
        "Full pre-2026 diagnostic refit; not counted as an independent falsification window.",
    ),
)


def _phase2_latent_cfg() -> dict[str, Any]:
    plugin = get_disease_plugin("hiv")
    return dict((((plugin.constraint_settings or {}).get("phase2") or {}).get("latent_temporal_graph") or {}))


def _known_latent_blocks() -> set[str]:
    blocks: set[str] = set()
    for module in PHASE3_MODULE_CONTRACT.values():
        blocks.update(str(value) for value in list(module.get("latent_blocks") or []) if str(value))
    return blocks


def _load_tensor(path: Path) -> np.ndarray:
    payload = np.load(path)
    key = payload.files[0]
    return np.asarray(payload[key], dtype=np.float32)


def _month_year(label: str) -> int | None:
    try:
        return int(str(label)[:4])
    except Exception:
        return None


def _month_indices(month_axis: list[str], spec: TimeWindowSpec) -> list[int]:
    indices: list[int] = []
    for idx, label in enumerate(month_axis):
        year = _month_year(label)
        if year is not None and spec.start_year <= year <= spec.end_year:
            indices.append(idx)
    return indices


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _edge_key(row: Mapping[str, Any], *, kind: str) -> str:
    return f"{kind}:{row.get('source')}->{row.get('target')}:lag{int(row.get('lag') or 0)}"


def _edge_sign(value: Any) -> int:
    try:
        weight = float(value)
    except Exception:
        return 0
    if weight > 0:
        return 1
    if weight < 0:
        return -1
    return 0


def _match_edge(rows: Iterable[Mapping[str, Any]], edge: Mapping[str, Any]) -> dict[str, Any] | None:
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    lag = int(edge.get("lag") or 0)
    for row in rows:
        if str(row.get("source") or "") == source and str(row.get("target") or "") == target and int(row.get("lag") or 0) == lag:
            return dict(row)
    return None


def _ledger_block_support(ledger_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    known_blocks = _known_latent_blocks()
    block_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    block_family_counts_all: dict[str, Counter[str]] = defaultdict(Counter)
    block_allowed_use_counts: dict[str, Counter[str]] = defaultdict(Counter)
    block_years: dict[str, set[str]] = defaultdict(set)
    block_rows: Counter[str] = Counter()
    block_rows_all: Counter[str] = Counter()
    excluded_allowed_use_counts: Counter[str] = Counter()
    for row in ledger_rows:
        blocks = []
        candidate_block = str(row.get("candidate_block") or "").strip()
        if candidate_block in known_blocks:
            blocks.append(candidate_block)
        if not blocks:
            blocks.extend(str(value) for value in list(row.get("contract_latent_blocks") or []) if str(value) in known_blocks)
        if not blocks:
            continue
        source_family = str(row.get("source_family") or "unknown")
        source_count = max(int(row.get("row_count") or 1), 1)
        if not bool(row.get("has_verifiable_locator")):
            continue
        allowed_use = str(row.get("allowed_use") or "").strip()
        for block in blocks:
            block_family_counts_all[block][source_family] += source_count
            block_allowed_use_counts[block][allowed_use or "unspecified"] += source_count
            block_rows_all[block] += source_count
            if allowed_use not in PRIOR_SUPPORT_ALLOWED_USES:
                excluded_allowed_use_counts[allowed_use or "unspecified"] += source_count
                continue
            block_family_counts[block][source_family] += source_count
            block_rows[block] += source_count
            for year in list(row.get("years") or []):
                if str(year):
                    block_years[block].add(str(year)[:4])
    return {
        "block_family_counts": {block: dict(sorted(counter.items())) for block, counter in sorted(block_family_counts.items())},
        "block_family_counts_all_verifiable": {block: dict(sorted(counter.items())) for block, counter in sorted(block_family_counts_all.items())},
        "block_allowed_use_counts": {block: dict(sorted(counter.items())) for block, counter in sorted(block_allowed_use_counts.items())},
        "block_years": {block: sorted(years) for block, years in sorted(block_years.items())},
        "block_row_counts": dict(sorted(block_rows.items())),
        "block_row_counts_all_verifiable": dict(sorted(block_rows_all.items())),
        "excluded_allowed_use_counts": dict(sorted(excluded_allowed_use_counts.items())),
        "prior_support_allowed_uses": sorted(PRIOR_SUPPORT_ALLOWED_USES - {""}),
    }


def _source_ablation_for_edge(edge: Mapping[str, Any], block_support: Mapping[str, Any]) -> dict[str, Any]:
    source_block = str(edge.get("source") or "")
    target_block = str(edge.get("target") or "")
    family_counts = dict(block_support.get("block_family_counts") or {})
    source_counts = Counter(dict(family_counts.get(source_block) or {}))
    target_counts = Counter(dict(family_counts.get(target_block) or {}))
    source_total = int(sum(source_counts.values()))
    target_total = int(sum(target_counts.values()))
    families = sorted(set(source_counts) | set(target_counts))
    rows: list[dict[str, Any]] = []
    for family in families:
        source_after = source_total - int(source_counts.get(family, 0))
        target_after = target_total - int(target_counts.get(family, 0))
        survives = source_after > 0 and target_after > 0
        rows.append(
            {
                "ablated_source_family": family,
                "source_block_count_after_ablation": source_after,
                "target_block_count_after_ablation": target_after,
                "survives": survives,
                "failure_reason": "" if survives else "source_or_target_block_loses_all_verifiable_evidence",
            }
        )
    passed = bool(source_total > 0 and target_total > 0 and all(bool(row["survives"]) for row in rows))
    return {
        "source_block": source_block,
        "target_block": target_block,
        "source_block_verifiable_row_count": source_total,
        "target_block_verifiable_row_count": target_total,
        "ablation_rows": rows,
        "passed": passed,
        "failure_reason": "" if passed else "edge evidentiary support is not robust to source-family ablation",
    }


def _source_reestimated_ablation_for_edge(
    *,
    edge_key: str,
    source_reestimated_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not source_reestimated_report:
        return {
            "available": False,
            "passed": False,
            "failure_reason": "source-family re-estimation ablation report not provided",
            "evaluation_rows": [],
        }
    edge_summary = dict((source_reestimated_report.get("edge_summary") or {}).get(edge_key) or {})
    evaluation_rows = [
        dict(row)
        for row in list(source_reestimated_report.get("edge_family_rows") or [])
        if str(row.get("edge_key") or "") == edge_key
    ]
    completed_rows = [row for row in evaluation_rows if str(row.get("ablation_status") or "") == "completed"]
    passed = bool(edge_summary.get("passed")) and bool(completed_rows)
    return {
        "available": True,
        "passed": passed,
        "evaluated_family_count": int(edge_summary.get("evaluated_family_count") or 0),
        "survived_family_count": int(edge_summary.get("survived_family_count") or 0),
        "absent_family_count": int(edge_summary.get("absent_family_count") or 0),
        "sign_conflict_family_count": int(edge_summary.get("sign_conflict_family_count") or 0),
        "failure_reason": "" if passed else "edge did not survive every completed source-family re-estimation ablation",
        "evaluation_rows": evaluation_rows,
    }


def _fit_time_windows(run_dir: Path, *, cfg: Mapping[str, Any]) -> dict[str, Any]:
    phase15_dir = run_dir / "phase15"
    block_axis, phi_by_block = _phase15_phi_by_block(phase15_dir)
    if not block_axis:
        return {
            "available": False,
            "reason": "missing_phase15_v2_fit_summary",
            "window_scale_rows": [],
            "window_edge_rows": {},
        }
    uncertainty_payload = read_json(phase15_dir / "phase15_v2_uncertainty.json", default={})
    month_axis = [str(value) for value in list((uncertainty_payload or {}).get("month_axis") or [])]
    if not month_axis:
        return {
            "available": False,
            "reason": "missing_phase15_month_axis",
            "window_scale_rows": [],
            "window_edge_rows": {},
        }
    uncertainty_tensors = _load_uncertainty_tensors(phase15_dir, block_axis)
    indicator_names_by_block = _indicator_names_by_block(phase15_dir, block_axis)
    target_block_ids = [block_id for block_id in list(cfg.get("phase3_target_block_ids") or []) if block_id in set(block_axis)] or list(block_axis)
    paths = {
        "province": phase15_dir / "phase15_v2_province_state_tensor.npz",
        "region": phase15_dir / "phase15_v2_region_state_tensor.npz",
        "national": phase15_dir / "phase15_v2_national_state_tensor.npz",
    }
    window_scale_rows: list[dict[str, Any]] = []
    edge_rows_by_window_scale: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for spec in DEFAULT_TIME_WINDOWS:
        indices = _month_indices(month_axis, spec)
        window_labels = [month_axis[idx] for idx in indices]
        edge_rows_by_window_scale[spec.window_id] = {}
        for scale, path in paths.items():
            if not path.exists():
                row = {
                    "window_id": spec.window_id,
                    "scale": scale,
                    "status": "unavailable",
                    "reason": "missing_phase15_state_tensor",
                    "falsification_window": spec.falsification_window,
                    "scientific_role": spec.scientific_role,
                    "month_count": len(indices),
                    "month_axis": window_labels,
                    "edge_count": 0,
                    "hidden_driver_count": 0,
                    "effective_sample_count": 0,
                }
                window_scale_rows.append(row)
                edge_rows_by_window_scale[spec.window_id][scale] = []
                continue
            if len(indices) < 2:
                row = {
                    "window_id": spec.window_id,
                    "scale": scale,
                    "status": "unavailable",
                    "reason": "insufficient_month_labels_in_window",
                    "falsification_window": spec.falsification_window,
                    "scientific_role": spec.scientific_role,
                    "month_count": len(indices),
                    "month_axis": window_labels,
                    "edge_count": 0,
                    "hidden_driver_count": 0,
                    "effective_sample_count": 0,
                }
                window_scale_rows.append(row)
                edge_rows_by_window_scale[spec.window_id][scale] = []
                continue
            tensor = _load_tensor(path)[:, indices, :]
            uncertainty_tensor = uncertainty_tensors.get(scale)
            if uncertainty_tensor is not None:
                uncertainty_tensor = np.asarray(uncertainty_tensor, dtype=np.float32)[:, indices, :]
            bundle, _blanket = estimate_latent_temporal_scale_graph(
                scale_name=f"{scale}_{spec.window_id}",
                state_tensor=tensor,
                uncertainty_tensor=uncertainty_tensor,
                block_axis=block_axis,
                target_block_ids=target_block_ids,
                indicator_names_by_block=indicator_names_by_block,
                cfg=dict(cfg),
                phi_by_block=phi_by_block,
                month_axis=window_labels,
            )
            direct_edges = list(bundle.get("edges") or [])
            hidden_edges = list(bundle.get("hidden_driver_rows") or [])
            edge_rows_by_window_scale[spec.window_id][scale] = [
                {**dict(row), "edge_kind": "direct"} for row in direct_edges
            ] + [{**dict(row), "edge_kind": "hidden"} for row in hidden_edges]
            window_scale_rows.append(
                {
                    "window_id": spec.window_id,
                    "scale": scale,
                    "status": str(bundle.get("status") or "unknown"),
                    "reason": str(bundle.get("reason") or ""),
                    "falsification_window": spec.falsification_window,
                    "scientific_role": spec.scientific_role,
                    "month_count": len(indices),
                    "month_axis": window_labels,
                    "edge_count": len(direct_edges),
                    "hidden_driver_count": len(hidden_edges),
                    "effective_sample_count": int(bundle.get("effective_sample_count") or 0),
                    "selected_hyperparameters": dict(bundle.get("selected_hyperparameters") or {}),
                }
            )
    return {
        "available": True,
        "block_axis": block_axis,
        "month_axis": month_axis,
        "window_scale_rows": window_scale_rows,
        "window_edge_rows": edge_rows_by_window_scale,
    }


def _time_window_for_edge(
    *,
    edge: Mapping[str, Any],
    kind: str,
    original_scales: list[str],
    time_window_payload: Mapping[str, Any],
) -> dict[str, Any]:
    scale_rows = [dict(row) for row in list(time_window_payload.get("window_scale_rows") or [])]
    rows_by_window_scale = {
        (str(row.get("window_id")), str(row.get("scale"))): row
        for row in scale_rows
        if bool(row.get("falsification_window"))
    }
    edge_rows = dict(time_window_payload.get("window_edge_rows") or {})
    original_sign = _edge_sign(edge.get("weight"))
    evaluations: list[dict[str, Any]] = []
    for (window_id, scale), scale_row in sorted(rows_by_window_scale.items()):
        if original_scales and scale not in set(original_scales):
            continue
        if str(scale_row.get("status")) != "completed":
            evaluations.append(
                {
                    "window_id": window_id,
                    "scale": scale,
                    "status": "not_estimable",
                    "reason": str(scale_row.get("reason") or "window refit did not complete"),
                    "effective_sample_count": int(scale_row.get("effective_sample_count") or 0),
                }
            )
            continue
        rows = [
            row
            for row in list(dict(edge_rows.get(window_id) or {}).get(scale) or [])
            if str(row.get("edge_kind") or "direct") == kind
        ]
        matched = _match_edge(rows, edge)
        if matched is None:
            evaluations.append(
                {
                    "window_id": window_id,
                    "scale": scale,
                    "status": "absent",
                    "reason": "edge not recovered in refit",
                    "effective_sample_count": int(scale_row.get("effective_sample_count") or 0),
                }
            )
            continue
        matched_sign = _edge_sign(matched.get("weight"))
        status = "survived" if matched_sign == original_sign or original_sign == 0 else "sign_conflict"
        evaluations.append(
            {
                "window_id": window_id,
                "scale": scale,
                "status": status,
                "reason": "" if status == "survived" else "refit recovered the same edge with opposite sign",
                "original_weight": edge.get("weight"),
                "refit_weight": matched.get("weight"),
                "refit_stability": matched.get("stability"),
                "effective_sample_count": int(scale_row.get("effective_sample_count") or 0),
            }
        )
    counts = Counter(str(row.get("status")) for row in evaluations)
    estimable = int(counts.get("survived", 0) + counts.get("absent", 0) + counts.get("sign_conflict", 0))
    passed = bool(evaluations and estimable == len(evaluations) and counts.get("survived", 0) == len(evaluations))
    if not evaluations:
        reason = "no applicable falsification windows for this edge scale"
    elif not passed:
        reason = "edge did not survive every applicable blocked time-window refit"
    else:
        reason = ""
    return {
        "evaluation_rows": evaluations,
        "status_counts": dict(sorted(counts.items())),
        "estimable_window_scale_count": estimable,
        "passed": passed,
        "failure_reason": reason,
    }


def build_phase2_edge_falsification_report(
    *,
    run_dir: Path,
    ledger_path: Path | None = None,
    out_dir: Path | None = None,
    source_reestimated_ablation_path: Path | None = None,
) -> dict[str, Any]:
    phase2_dir = run_dir / "phase2"
    structural = read_json(phase2_dir / "phase2_structural_payload.json", default={}) or {}
    direct_edges = [dict(row) for row in list(structural.get("direct_temporal_edge_rows") or [])]
    hidden_edges = [dict(row) for row in list(structural.get("hidden_driver_rows") or [])]
    ledger_rows = _read_jsonl(ledger_path or (run_dir / "phase0" / "evidence_ledger" / "phase3_determinant_evidence_ledger.jsonl"))
    block_support = _ledger_block_support(ledger_rows)
    cfg = _phase2_latent_cfg()
    time_window_payload = _fit_time_windows(run_dir, cfg=cfg)
    source_reestimated_report = (
        read_json(source_reestimated_ablation_path, default={}) or {}
        if source_reestimated_ablation_path is not None
        else {}
    )

    edge_reports: list[dict[str, Any]] = []
    for kind, rows in (("direct", direct_edges), ("hidden", hidden_edges)):
        for edge in rows:
            scales = [str(value) for value in list(edge.get("scales") or [])]
            edge_key = _edge_key(edge, kind=kind)
            source_ablation = _source_ablation_for_edge(edge, block_support)
            source_reestimated_ablation = _source_reestimated_ablation_for_edge(
                edge_key=edge_key,
                source_reestimated_report=source_reestimated_report,
            )
            time_window = _time_window_for_edge(
                edge=edge,
                kind=kind,
                original_scales=scales,
                time_window_payload=time_window_payload,
            )
            prior_eligible = bool(
                kind == "direct"
                and source_ablation["passed"]
                and source_reestimated_ablation["passed"]
                and time_window["passed"]
            )
            edge_reports.append(
                {
                    "edge_key": edge_key,
                    "edge_kind": kind,
                    "source": str(edge.get("source") or ""),
                    "target": str(edge.get("target") or ""),
                    "lag": int(edge.get("lag") or 0),
                    "weight": edge.get("weight"),
                    "stability": edge.get("stability"),
                    "support_count": int(edge.get("support_count") or 0),
                    "scales": scales,
                    "source_ablation": source_ablation,
                    "source_reestimated_ablation": source_reestimated_ablation,
                    "time_window_falsification": time_window,
                    "phase3_prior_eligible": prior_eligible,
                    "promotion_decision": "eligible_as_direct_prior" if prior_eligible else "blocked_from_phase3_prior_use",
                    "promotion_blockers": [
                        value
                        for value in [
                            "" if kind == "direct" else "hidden edges remain shared latent-shock diagnostics, not direct Phase 3 priors",
                            "" if source_ablation["passed"] else source_ablation["failure_reason"],
                            "" if source_reestimated_ablation["passed"] else source_reestimated_ablation["failure_reason"],
                            "" if time_window["passed"] else time_window["failure_reason"],
                        ]
                        if value
                    ],
                }
            )

    destination = ensure_dir(out_dir or (phase2_dir / "falsification"))
    report = {
        "schema_version": FALSIFICATION_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "generated_at": utc_now_iso(),
        "contract": {
            "source_ablation": "Leave one source family out of the verifiable determinant ledger and require both edge endpoint blocks to retain support.",
            "source_reestimated_ablation": "Rebuild Phase1, Phase15, and Phase2 after source-family exclusion from Phase0 canonical candidates, then require the same edge/sign to survive every completed source-family re-estimation.",
            "time_window": "Refit the same Phase 2 latent temporal graph on blocked calendar windows and require the same edge/sign to survive every applicable falsification window.",
            "use_constraint": "Only direct edges passing support, source-reestimated, and blocked-time gates may be used as Phase 3 priors. Validation-only, auxiliary-only, and observation-head rows are visible in the ledger but excluded from prior-support promotion.",
            "limitation": "" if source_reestimated_report else "Source-reestimated ablation was not provided; no edge can be promoted as a publication-grade Phase 3 prior.",
        },
        "input_artifacts": {
            "structural_payload": str(phase2_dir / "phase2_structural_payload.json"),
            "ledger_jsonl": str(ledger_path or (run_dir / "phase0" / "evidence_ledger" / "phase3_determinant_evidence_ledger.jsonl")),
            "source_reestimated_ablation_report": "" if source_reestimated_ablation_path is None else str(source_reestimated_ablation_path),
        },
        "direct_edge_count": len(direct_edges),
        "hidden_edge_count": len(hidden_edges),
        "edge_reports": edge_reports,
        "prior_eligible_direct_edge_count": int(sum(1 for row in edge_reports if bool(row.get("phase3_prior_eligible")))),
        "blocked_edge_count": int(sum(1 for row in edge_reports if not bool(row.get("phase3_prior_eligible")))),
        "ledger_block_support": block_support,
        "source_reestimated_ablation": {
            "available": bool(source_reestimated_report),
            "schema_version": str(source_reestimated_report.get("schema_version") or ""),
            "source_families": list(source_reestimated_report.get("source_families") or []),
            "family_rows": list(source_reestimated_report.get("family_rows") or []),
        },
        "time_window_refits": {
            "available": bool(time_window_payload.get("available")),
            "reason": str(time_window_payload.get("reason") or ""),
            "block_axis": list(time_window_payload.get("block_axis") or []),
            "window_scale_rows": list(time_window_payload.get("window_scale_rows") or []),
        },
    }
    report_path = destination / "phase2_edge_falsification_report.json"
    write_json(report_path, report)
    md_path = destination / "phase2_edge_falsification_report.md"
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    dashboard_path = destination / "phase2_edge_falsification_dashboard.png"
    dashboard_written = _write_dashboard(report, dashboard_path)
    report["artifact_paths"] = {
        "report_json": str(report_path),
        "report_md": str(md_path),
        "dashboard_png": str(dashboard_path) if dashboard_written else "",
    }
    write_json(report_path, report)
    return report


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Edge Falsification",
        "",
        f"- Run: `{report.get('run_id')}`",
        f"- Direct edges: `{report.get('direct_edge_count')}`",
        f"- Hidden edges: `{report.get('hidden_edge_count')}`",
        f"- Prior-eligible direct edges: `{report.get('prior_eligible_direct_edge_count')}`",
        "",
        "## Edge Decisions",
        "",
        "| Edge | Kind | Support Count | Support Ablation | Re-estimation | Time Window | Promotion |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in list(report.get("edge_reports") or []):
        source_status = "pass" if bool((row.get("source_ablation") or {}).get("passed")) else "fail"
        reestimated_status = "pass" if bool((row.get("source_reestimated_ablation") or {}).get("passed")) else "fail"
        time_status = "pass" if bool((row.get("time_window_falsification") or {}).get("passed")) else "fail"
        lines.append(
            f"| `{row.get('edge_key')}` | `{row.get('edge_kind')}` | {int(row.get('support_count') or 0)} | "
            f"{source_status} | {reestimated_status} | {time_status} | `{row.get('promotion_decision')}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            str((report.get("contract") or {}).get("use_constraint") or ""),
            "",
            str((report.get("contract") or {}).get("limitation") or ""),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_dashboard(report: Mapping[str, Any], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception:
        return False
    ensure_dir(path.parent)
    edge_reports = list(report.get("edge_reports") or [])
    block_counts = dict((report.get("ledger_block_support") or {}).get("block_row_counts") or {})
    window_rows = list((report.get("time_window_refits") or {}).get("window_scale_rows") or [])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Phase 2 Prior Falsification Before Phase 3 Use", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    blocks = sorted(block_counts)
    values = [int(block_counts[block]) for block in blocks]
    ax.barh(blocks, values, color="#2f6f73")
    ax.set_title("Prior-Admissible Determinant Ledger Support By Latent Block")
    ax.set_xlabel("Ledger-backed source rows")

    ax = axes[0, 1]
    status_counts = Counter("eligible" if bool(row.get("phase3_prior_eligible")) else "blocked" for row in edge_reports)
    ax.bar(list(status_counts), [status_counts[key] for key in status_counts], color=["#9ac36a" if key == "eligible" else "#c6503d" for key in status_counts])
    ax.set_title("Phase 3 Prior Promotion Gate")
    ax.set_ylabel("Edge count")

    ax = axes[1, 0]
    window_ids = sorted({str(row.get("window_id")) for row in window_rows if bool(row.get("falsification_window"))})
    row_labels = [str(row.get("edge_key")) for row in edge_reports]
    status_to_value = {"survived": 3, "absent": 2, "sign_conflict": 1, "not_estimable": 0}
    matrix = np.zeros((len(row_labels), len(window_ids)), dtype=float)
    for edge_idx, edge in enumerate(edge_reports):
        rows = list(((edge.get("time_window_falsification") or {}).get("evaluation_rows") or []))
        by_window: dict[str, list[str]] = defaultdict(list)
        for item in rows:
            by_window[str(item.get("window_id"))].append(str(item.get("status")))
        for window_idx, window_id in enumerate(window_ids):
            statuses = by_window.get(window_id) or ["not_estimable"]
            if "sign_conflict" in statuses:
                value = status_to_value["sign_conflict"]
            elif "absent" in statuses:
                value = status_to_value["absent"]
            elif all(status == "survived" for status in statuses):
                value = status_to_value["survived"]
            else:
                value = status_to_value["not_estimable"]
            matrix[edge_idx, window_idx] = value
    cmap = ListedColormap(["#b7b7b7", "#9e2f2f", "#f0a23a", "#4b8f4a"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(window_ids)), window_ids, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_title("Blocked Time-Window Edge Survival")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = {3: "survived", 2: "absent", 1: "sign", 0: "not est."}.get(int(matrix[i, j]), "")
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white" if matrix[i, j] == 1 else "black")

    ax = axes[1, 1]
    completed = Counter()
    for row in window_rows:
        if not bool(row.get("falsification_window")):
            continue
        completed[str(row.get("status") or "unknown")] += 1
    ax.bar(list(completed), [completed[key] for key in completed], color="#506c9a")
    ax.set_title("Window Refit Identifiability")
    ax.set_ylabel("Scale-window fits")
    ax.tick_params(axis="x", rotation=20)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run source-ablation and time-window falsification for Phase 2 edges.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-reestimated-ablation-path", type=Path, default=None)
    args = parser.parse_args()
    report = build_phase2_edge_falsification_report(
        run_dir=args.run_dir,
        ledger_path=args.ledger_path,
        out_dir=args.out_dir,
        source_reestimated_ablation_path=args.source_reestimated_ablation_path,
    )
    print(json.dumps(report.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
