from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    R10_COMPARABLE_METRICS,
    _candidate_predictions,
    _finite_float,
    _generated_at,
)
from .r28_r10_contract_lineage_audit import _collect_r10_records
from .r29_strict_ledger_matched_r10_gate import (
    R29_RUN_ID,
    _is_program_lineage,
    _reference_by_scope_horizon,
    _strict_reference_rows,
    run_r29_strict_ledger_matched_r10_gate,
)
from .r31_strict_policy_probe import _history_by_metric, _policy_prediction
from .runtime import ensure_dir, read_json, write_json


R32_SCHEMA_VERSION = "phase3_dynamic.r32_strict_composite_replay.v1"
R32_RUN_ID = "p3d-r32-r18-flow-composite-replay-20260502-s00"
R32_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "candidate_id": "r18_exact_replay",
        "base_family": "r18_evidence_backed_art_process",
        "flow_policy": "base",
    },
    {
        "candidate_id": "r18_plus_r31_positive_flow",
        "base_family": "r18_evidence_backed_art_process",
        "flow_policy": "positive_velocity",
    },
)


def _source_rows_by_quarter(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("quarter") or ""): dict(row)
        for row in rows
        if row.get("quarter")
    }


def _record_scope(record: dict[str, Any], scope: str) -> bool:
    if bool(record.get("unknown_provenance")):
        return False
    if scope == "all_mapped":
        return True
    if scope == "program_mapped":
        return _is_program_lineage(record)
    raise ValueError(f"Unknown R32 scope: {scope}")


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record.get("unknown_provenance")):
            continue
        grouped[(int(record.get("train_end_year") or 0), int(record.get("horizon_years") or 0))].append(dict(record))
    return grouped


def _candidate_prediction_rows(
    *,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    base_family: str,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    output: dict[tuple[int, int, str], dict[str, Any]] = {}
    for (train_end_year, horizon), grouped_records in sorted(_group_records(records).items()):
        train_rows = [
            dict(row)
            for row in source_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        holdout_quarters = sorted(
            {str(record.get("quarter") or "") for record in grouped_records if record.get("quarter")}
        )
        holdout_rows = [
            dict(source_by_quarter[quarter])
            for quarter in holdout_quarters
            if quarter in source_by_quarter
        ]
        if not train_rows or not holdout_rows:
            continue
        predictions, _summary = _candidate_predictions(train_rows, holdout_rows, family=base_family)
        for row in predictions:
            quarter = str(row.get("quarter") or "")
            if quarter:
                output[(int(train_end_year), int(horizon), quarter)] = dict(row)
    return output


def _candidate_value(
    *,
    record: dict[str, Any],
    candidate_row: dict[str, Any],
    candidate: dict[str, str],
    history_by_metric: dict[str, list[tuple[int, int, float]]],
) -> float | None:
    metric_name = str(record.get("metric_name") or "")
    if metric_name == "new_diagnosed_cases_period" and str(candidate.get("flow_policy") or "") != "base":
        return _policy_prediction(
            metric_name=metric_name,
            train_end_year=int(record.get("train_end_year") or 0),
            target_quarter=str(record.get("quarter") or ""),
            policy_id=str(candidate.get("flow_policy") or "carry_forward"),
            history_by_metric=history_by_metric,
        )
    return _finite_float(candidate_row.get(metric_name))


def _score_candidate_records(
    *,
    records: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    candidate: dict[str, str],
    prediction_rows: dict[tuple[int, int, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if prediction_rows is None:
        prediction_rows = _candidate_prediction_rows(
            source_rows=source_rows,
            records=records,
            base_family=str(candidate["base_family"]),
        )
    history = _history_by_metric(source_rows)
    scored_rows: list[dict[str, Any]] = []
    for record in records:
        key = (
            int(record.get("train_end_year") or 0),
            int(record.get("horizon_years") or 0),
            str(record.get("quarter") or ""),
        )
        candidate_row = prediction_rows.get(key)
        if not candidate_row:
            continue
        value = _candidate_value(
            record=record,
            candidate_row=candidate_row,
            candidate=candidate,
            history_by_metric=history,
        )
        target = _finite_float(record.get("target_value"))
        scale = _finite_float(record.get("scale"))
        if value is None or target is None or scale is None:
            continue
        scale = max(float(scale), float(np.finfo(np.float32).eps))
        error = abs(float(value) - float(target)) / scale
        scored_rows.append(
            {
                **{
                    key_name: record.get(key_name)
                    for key_name in (
                        "horizon_years",
                        "train_end_year",
                        "target_year",
                        "quarter",
                        "metric_name",
                        "source_lineage",
                        "unknown_provenance",
                        "target_value",
                        "scale",
                        "r10_norm_error",
                        "carry_forward_norm_error",
                    )
                },
                "candidate_id": str(candidate["candidate_id"]),
                "base_family": str(candidate["base_family"]),
                "flow_policy": str(candidate["flow_policy"]),
                "candidate_value": float(value),
                "candidate_norm_error": float(error),
                "candidate_minus_r10_norm_error": float(error - float(record["r10_norm_error"])),
                "candidate_minus_carry_forward_norm_error": float(error - float(record["carry_forward_norm_error"])),
            }
        )
    return scored_rows


def _mean_error(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if _finite_float(row.get(key)) is not None
    ]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _candidate_gate_rows(
    *,
    scored_rows_by_candidate: dict[str, list[dict[str, Any]]],
    strict_reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_by_scope_horizon(strict_reference_rows)
    rows: list[dict[str, Any]] = []
    required = (
        ("all_mapped", 1),
        ("all_mapped", 3),
        ("all_mapped", 5),
        ("program_mapped", 3),
        ("program_mapped", 5),
    )
    for candidate_id, candidate_rows in sorted(scored_rows_by_candidate.items()):
        for scope, horizon in required:
            scoped = [
                row
                for row in candidate_rows
                if int(row.get("horizon_years") or 0) == int(horizon)
                and _record_scope(row, scope)
            ]
            ref = reference.get((scope, int(horizon)), {})
            candidate_mean = _mean_error(scoped, "candidate_norm_error")
            matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
            carry = _mean_error(scoped, "carry_forward_norm_error")
            blockers: list[str] = []
            if not scoped:
                blockers.append("no_scored_rows")
            if candidate_mean is None or matched_r10 is None:
                blockers.append("candidate_or_strict_r10_missing")
            elif candidate_mean >= matched_r10:
                blockers.append("candidate_not_better_than_strict_mapped_r10")
            if candidate_mean is None or carry is None:
                blockers.append("candidate_or_carry_missing")
            elif candidate_mean >= carry:
                blockers.append("candidate_not_better_than_carry_forward")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scope": scope,
                    "horizon_years": int(horizon),
                    "entry_count": len(scoped),
                    "candidate_mean_mae": candidate_mean,
                    "strict_matched_r10_mean_mae": matched_r10,
                    "candidate_minus_strict_r10": None
                    if candidate_mean is None or matched_r10 is None
                    else float(candidate_mean - matched_r10),
                    "carry_forward_mean_mae": carry,
                    "candidate_minus_carry_forward": None
                    if candidate_mean is None or carry is None
                    else float(candidate_mean - carry),
                    "status": "pass" if not blockers else "fail",
                    "blockers": blockers,
                }
            )
    return rows


def _metric_anatomy_rows(scored_rows_by_candidate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate_id, rows in sorted(scored_rows_by_candidate.items()):
        for scope in ("all_mapped", "program_mapped"):
            for horizon in (1, 3, 5):
                for metric_name in R10_COMPARABLE_METRICS:
                    scoped = [
                        row
                        for row in rows
                        if int(row.get("horizon_years") or 0) == int(horizon)
                        and str(row.get("metric_name") or "") == metric_name
                        and _record_scope(row, scope)
                    ]
                    if not scoped:
                        continue
                    candidate_mean = _mean_error(scoped, "candidate_norm_error")
                    r10_mean = _mean_error(scoped, "r10_norm_error")
                    carry_mean = _mean_error(scoped, "carry_forward_norm_error")
                    output.append(
                        {
                            "candidate_id": candidate_id,
                            "scope": scope,
                            "horizon_years": int(horizon),
                            "metric_name": metric_name,
                            "entry_count": len(scoped),
                            "candidate_mean_mae": candidate_mean,
                            "strict_matched_r10_mean_mae": r10_mean,
                            "carry_forward_mean_mae": carry_mean,
                            "candidate_minus_strict_r10": None
                            if candidate_mean is None or r10_mean is None
                            else float(candidate_mean - r10_mean),
                        }
                    )
    return output


def _r18_r13_report_path(phase3_root: Path) -> Path:
    candidates = [
        phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r18-art-process-r13-queue-20260501-final"
        / "analysis"
        / "r13_priority_experiment_results.json",
        Path("/media/gaurav/New_Volume/EpiGraph_PH")
        / "src"
        / "epigraph_ph"
        / "Phase3(dynamic)"
        / "artifacts"
        / "runs"
        / "p3d-r18-art-process-r13-queue-20260501-final"
        / "analysis"
        / "r13_priority_experiment_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _r31_report_path(phase3_root: Path) -> Path:
    return phase3_root / "artifacts" / "runs" / "p3d-r31-strict-policy-probe-20260502-s00" / "analysis" / "r31_strict_policy_probe.json"


def _aggregate_r18_metric_lookup(r18_report: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    raw: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in list(r18_report.get("results") or []):
        if not isinstance(result, dict):
            continue
        experiment_id = str(result.get("experiment_id") or "")
        if experiment_id == "R13-050":
            scope = "all_mapped"
        elif experiment_id == "R13-006":
            scope = "program_mapped"
        else:
            continue
        for row in list(result.get("metric_rows") or []):
            if not isinstance(row, dict):
                continue
            metric_name = str(row.get("metric_name") or "")
            if metric_name not in R10_COMPARABLE_METRICS:
                continue
            if _finite_float(row.get("candidate_mean_norm_error")) is None:
                continue
            raw[(scope, int(row.get("horizon_years") or 0), metric_name)].append(dict(row))
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key, rows in sorted(raw.items()):
        weights = np.asarray([max(int(row.get("entry_count") or 0), 1) for row in rows], dtype=np.float64)
        candidate = np.asarray([float(row["candidate_mean_norm_error"]) for row in rows], dtype=np.float64)
        carry_values = [
            float(row["carry_forward_mean_norm_error"])
            for row in rows
            if _finite_float(row.get("carry_forward_mean_norm_error")) is not None
        ]
        output[key] = {
            "candidate_mean_mae": float(np.average(candidate, weights=weights)),
            "carry_forward_mean_mae": None
            if not carry_values
            else float(np.mean(np.asarray(carry_values, dtype=np.float64))),
            "entry_count": int(sum(int(row.get("entry_count") or 0) for row in rows)),
            "source": "r18_r13_aggregate_metric_rows",
        }
    return output


def _r31_positive_flow_lookup(r31_report: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    output: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in list(r31_report.get("best_policy_rows") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("metric_name") or "") != "new_diagnosed_cases_period":
            continue
        if str(row.get("policy_id") or "") != "positive_velocity":
            continue
        value = _finite_float(row.get("policy_mean_mae"))
        if value is None:
            continue
        output[(str(row.get("scope") or ""), int(row.get("horizon_years") or 0), "new_diagnosed_cases_period")] = {
            "candidate_mean_mae": float(value),
            "carry_forward_mean_mae": None,
            "entry_count": int(row.get("entry_count") or 0),
            "source": "r31_positive_velocity_policy",
            "strict_r10_mean_mae": row.get("strict_r10_mean_mae"),
            "policy_minus_strict_r10": row.get("policy_minus_strict_r10"),
        }
    return output


def _aggregate_composite_rows(
    *,
    r18_lookup: dict[tuple[str, int, str], dict[str, Any]],
    r31_flow_lookup: dict[tuple[str, int, str], dict[str, Any]],
    strict_reference_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _reference_by_scope_horizon(strict_reference_rows)
    required = (
        ("all_mapped", 1),
        ("all_mapped", 3),
        ("all_mapped", 5),
        ("program_mapped", 3),
        ("program_mapped", 5),
    )
    candidate_specs = (
        ("r18_aggregate_replay", False),
        ("r18_plus_r31_positive_flow_aggregate", True),
    )
    gate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for candidate_id, use_r31_flow in candidate_specs:
        for scope, horizon in required:
            weighted_error = 0.0
            entry_count = 0
            missing_metrics: list[str] = []
            for metric_name in R10_COMPARABLE_METRICS:
                key = (scope, int(horizon), metric_name)
                metric = r18_lookup.get(key)
                if use_r31_flow and metric_name == "new_diagnosed_cases_period":
                    metric = r31_flow_lookup.get(key) or metric
                if metric is None:
                    missing_metrics.append(metric_name)
                    continue
                metric_entry_count = max(int(metric.get("entry_count") or 0), 1)
                metric_value = _finite_float(metric.get("candidate_mean_mae"))
                if metric_value is None:
                    missing_metrics.append(metric_name)
                    continue
                weighted_error += float(metric_value) * float(metric_entry_count)
                entry_count += int(metric_entry_count)
                metric_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scope": scope,
                        "horizon_years": int(horizon),
                        "metric_name": metric_name,
                        "candidate_mean_mae": float(metric_value),
                        "entry_count": int(metric_entry_count),
                        "source": str(metric.get("source") or ""),
                    }
                )
            candidate_mean = None if entry_count == 0 else float(weighted_error / float(entry_count))
            ref = reference.get((scope, int(horizon)), {})
            matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
            carry = _finite_float(ref.get("carry_forward_mean_mae"))
            blockers: list[str] = []
            if missing_metrics:
                blockers.append("missing_aggregate_metrics_" + "_".join(missing_metrics))
            if candidate_mean is None or matched_r10 is None:
                blockers.append("candidate_or_strict_r10_missing")
            elif candidate_mean >= matched_r10:
                blockers.append("candidate_not_better_than_strict_mapped_r10")
            if candidate_mean is None or carry is None:
                blockers.append("candidate_or_carry_missing")
            elif candidate_mean >= carry:
                blockers.append("candidate_not_better_than_carry_forward")
            gate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "scope": scope,
                    "horizon_years": int(horizon),
                    "entry_count": int(entry_count),
                    "candidate_mean_mae": candidate_mean,
                    "strict_matched_r10_mean_mae": matched_r10,
                    "candidate_minus_strict_r10": None
                    if candidate_mean is None or matched_r10 is None
                    else float(candidate_mean - matched_r10),
                    "carry_forward_mean_mae": carry,
                    "candidate_minus_carry_forward": None
                    if candidate_mean is None or carry is None
                    else float(candidate_mean - carry),
                    "status": "pass" if not blockers else "fail",
                    "blockers": blockers,
                }
            )
    return gate_rows, metric_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if key != "blockers"})
    if any("blockers" in row for row in rows):
        fieldnames.append("blockers")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if "blockers" in output:
                output["blockers"] = ";".join(str(item) for item in list(output.get("blockers") or []))
            writer.writerow(output)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R32 Strict Composite Aggregate Replay",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Candidate Gate Rows",
        "",
        "| Candidate | Scope | Horizon | Entries | Candidate | Strict R10 | Delta | Carry | Status | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("candidate_gate_rows") or []):
        lines.append(
            f"| `{row.get('candidate_id')}` | {row.get('scope')} | {row.get('horizon_years')} | "
            f"{row.get('entry_count')} | {_format_float(row.get('candidate_mean_mae'))} | "
            f"{_format_float(row.get('strict_matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('candidate_minus_strict_r10'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | `{row.get('status')}` | "
            f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R32 is an aggregate non-promotional replay because exact R18 refitting across all strict origins is compute-heavy.",
            "- R18 D/A and full-sentinel metrics come from the frozen R18 R13 aggregate metric rows.",
            "- The `positive_velocity` flow policy is the R31 strict train-only policy and mutates only `new_diagnosed_cases_period`.",
            "- R19 already has a strict R29 evaluation; R32 isolates the R18 ART backbone plus R31-flow repair question.",
            "- This is a composite replay audit; a full process claim requires feeding a passing policy back into conserved state dynamics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    selected = [
        row
        for row in rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    labels = [f"{row.get('candidate_id')}\\nh{row.get('horizon_years')}" for row in selected]
    deltas = np.asarray([float(row.get("candidate_minus_strict_r10") or np.nan) for row in selected], dtype=np.float64)
    x = np.arange(len(selected), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("R32 strict replay: candidate minus strict program R10", fontsize=14, fontweight="bold")
    ax.bar(x, deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in deltas])
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("normalized MAE delta")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r32_strict_composite_replay(
    *,
    run_id: str = R32_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    source_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    phase3_root = sandbox_repo_root()
    r29_path = phase3_root / "artifacts" / "runs" / R29_RUN_ID / "analysis" / "r29_strict_ledger_matched_r10_gate.json"
    if r29_path.exists():
        r29_report = read_json(r29_path, default={})
    else:
        r29_report = run_r29_strict_ledger_matched_r10_gate(epigraph_root=root)
    strict_rows = list(dict(r29_report).get("strict_reference_rows") or [])
    if not strict_rows:
        records, _reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
        strict_rows = _strict_reference_rows(
            [record for record in records if not bool(record.get("unknown_provenance"))],
            horizons,
        )
    r18_path = _r18_r13_report_path(phase3_root)
    r31_path = _r31_report_path(phase3_root)
    r18_report = read_json(r18_path, default={})
    r31_report = read_json(r31_path, default={})
    if not isinstance(r18_report, dict) or not r18_report:
        raise FileNotFoundError(f"Missing R18 R13 report: {r18_path}")
    if not isinstance(r31_report, dict) or not r31_report:
        raise FileNotFoundError(f"Missing R31 strict policy report: {r31_path}")
    candidate_rows, metric_rows = _aggregate_composite_rows(
        r18_lookup=_aggregate_r18_metric_lookup(r18_report),
        r31_flow_lookup=_r31_positive_flow_lookup(r31_report),
        strict_reference_rows=strict_rows,
    )
    candidate_ids = sorted({str(row.get("candidate_id") or "") for row in candidate_rows})
    candidate_status: dict[str, str] = {}
    for candidate_id in candidate_ids:
        scoped = [row for row in candidate_rows if str(row.get("candidate_id") or "") == candidate_id]
        candidate_status[candidate_id] = "pass" if scoped and all(str(row.get("status") or "") == "pass" for row in scoped) else "fail"
    promoted = [candidate_id for candidate_id, status in candidate_status.items() if status == "pass"]
    verdict = (
        f"R32 strict composite replay passes for {promoted[0]}."
        if promoted
        else "R32 strict composite replay fails: no R18 plus R31-flow composite beats strict mapped R10 on every required route."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R32_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "horizons": list(horizons),
        "candidate_contracts": [
            {
                "candidate_id": "r18_aggregate_replay",
                "source": "R18 R13 aggregate metric rows",
            },
            {
                "candidate_id": "r18_plus_r31_positive_flow_aggregate",
                "source": "R18 R13 aggregate D/A plus R31 strict positive-velocity diagnosis-flow policy",
            },
        ],
        "r18_r13_report_path": r18_path.as_posix(),
        "r31_report_path": r31_path.as_posix(),
        "replay_mode": "aggregate_nonpromotional",
        "promotion_eligible": bool(promoted),
        "promoted_candidate_id": None if not promoted else promoted[0],
        "candidate_status": candidate_status,
        "verdict": verdict,
        "strict_reference_rows": strict_rows,
        "candidate_gate_rows": candidate_rows,
        "metric_anatomy_rows": metric_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r32_strict_composite_replay.json"
    md_path = analysis_dir / "r32_strict_composite_replay.md"
    gate_csv = analysis_dir / "r32_candidate_gate_rows.csv"
    metric_csv = analysis_dir / "r32_metric_anatomy_rows.csv"
    dashboard_path = analysis_dir / "r32_strict_composite_replay_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "candidate_gate_csv": gate_csv.as_posix(),
        "metric_anatomy_csv": metric_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(gate_csv, candidate_rows)
    _write_csv(metric_csv, metric_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, candidate_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R32 strict composite replay.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R32_RUN_ID)
    args = parser.parse_args()
    run_r32_strict_composite_replay(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
