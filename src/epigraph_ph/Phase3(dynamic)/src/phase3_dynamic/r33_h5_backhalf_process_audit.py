from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_ordinal
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import BACK_HALF_RATE_SPECS, CASCADE_STOCK_METRICS, _finite_float, _generated_at
from .r28_r10_contract_lineage_audit import _collect_r10_records, _lineage_id, _provenance_for_metric
from .r29_strict_ledger_matched_r10_gate import _is_program_lineage
from .r31_strict_policy_probe import _history_by_metric, _policy_prediction
from .runtime import ensure_dir, write_json


R33_SCHEMA_VERSION = "phase3_dynamic.r33_h5_backhalf_process_audit.v1"
R33_RUN_ID = "p3d-r33-h5-backhalf-process-audit-20260502-s00"
R33_ART_POLICY_IDS: tuple[str, ...] = (
    "direct_carry_forward",
    "direct_positive_velocity",
    "direct_median_velocity",
    "direct_recent_log_linear",
    "diagnosed_ratio_carry_forward__rate_carry_forward",
    "diagnosed_ratio_carry_forward__rate_logit_velocity",
    "diagnosed_ratio_carry_forward__rate_logit_trend",
    "diagnosed_ratio_positive_velocity__rate_carry_forward",
    "diagnosed_ratio_positive_velocity__rate_logit_velocity",
    "diagnosed_ratio_positive_velocity__rate_logit_trend",
    "diagnosed_ratio_median_velocity__rate_carry_forward",
    "diagnosed_ratio_median_velocity__rate_logit_velocity",
    "diagnosed_ratio_median_velocity__rate_logit_trend",
    "diagnosed_ratio_recent_log_linear__rate_carry_forward",
    "diagnosed_ratio_recent_log_linear__rate_logit_velocity",
    "diagnosed_ratio_recent_log_linear__rate_logit_trend",
)
R33_RATE_POLICY_IDS: tuple[str, ...] = (
    "carry_forward",
    "logit_velocity",
    "logit_trend",
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
    raise ValueError(f"Unknown R33 scope: {scope}")


def _safe_rate(numerator: Any, denominator: Any) -> float | None:
    num = _finite_float(numerator)
    den = _finite_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return float(min(max(float(num) / float(den), 0.0), 1.0))


def _clip_rate(value: float) -> float:
    eps = float(np.finfo(np.float64).eps)
    return float(min(max(float(value), eps), 1.0 - eps))


def _logit(value: float) -> float:
    clipped = _clip_rate(value)
    return float(math.log(clipped / (1.0 - clipped)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = math.exp(float(value))
    return float(z / (1.0 + z))


def _rate_from_row(row: dict[str, Any], *, numerator_metric: str, denominator_metric: str) -> float | None:
    return _safe_rate(row.get(numerator_metric), row.get(denominator_metric))


def _rate_history(
    rows: list[dict[str, Any]],
    *,
    numerator_metric: str,
    denominator_metric: str,
) -> list[tuple[int, int, float]]:
    history: list[tuple[int, int, float]] = []
    for row in rows:
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        rate = _rate_from_row(row, numerator_metric=numerator_metric, denominator_metric=denominator_metric)
        if rate is None:
            continue
        year_text = quarter.split("-Q", maxsplit=1)[0]
        try:
            year = int(year_text)
        except ValueError:
            continue
        history.append((quarter_ordinal(quarter), year, float(rate)))
    history.sort()
    return history


def _ratio_history(rows: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    return _rate_history(rows, numerator_metric="alive_on_art", denominator_metric="diagnosed_plhiv")


def _rate_policy_prediction(
    *,
    history: list[tuple[int, int, float]],
    train_end_year: int,
    target_quarter: str,
    policy_id: str,
) -> float | None:
    train_history = [row for row in history if int(row[1]) <= int(train_end_year)]
    if not train_history:
        return None
    target_ordinal = quarter_ordinal(target_quarter)
    last_ordinal, _last_year, last_value = max(train_history, key=lambda row: row[0])
    if policy_id == "carry_forward":
        return float(last_value)
    if policy_id == "logit_velocity":
        if len(train_history) < 2:
            return float(last_value)
        diffs = [
            (_logit(b[2]) - _logit(a[2])) / (b[0] - a[0])
            for a, b in zip(train_history, train_history[1:])
            if b[0] > a[0]
        ]
        if not diffs:
            return float(last_value)
        velocity = float(np.median(np.asarray(diffs, dtype=np.float64)))
        return _sigmoid(_logit(last_value) + velocity * (target_ordinal - last_ordinal))
    if policy_id == "logit_trend":
        if len(train_history) < 2:
            return float(last_value)
        x = np.asarray([row[0] for row in train_history], dtype=np.float64)
        y = np.asarray([_logit(row[2]) for row in train_history], dtype=np.float64)
        slope, intercept = np.polyfit(x - x[-1], y, 1)
        return _sigmoid(float(intercept + slope * (target_ordinal - x[-1])))
    raise ValueError(f"Unknown R33 rate policy: {policy_id}")


def _art_policy_prediction(
    *,
    record: dict[str, Any],
    policy_id: str,
    history_by_metric: dict[str, list[tuple[int, int, float]]],
    art_diagnosed_ratio_history: list[tuple[int, int, float]],
) -> float | None:
    train_end_year = int(record.get("train_end_year") or 0)
    target_quarter = str(record.get("quarter") or "")
    if policy_id.startswith("direct_"):
        return _policy_prediction(
            metric_name="alive_on_art",
            train_end_year=train_end_year,
            target_quarter=target_quarter,
            policy_id=policy_id.removeprefix("direct_"),
            history_by_metric=history_by_metric,
        )
    if not policy_id.startswith("diagnosed_ratio_"):
        raise ValueError(f"Unknown R33 ART policy: {policy_id}")
    forecast_part, rate_part = policy_id.removeprefix("diagnosed_ratio_").split("__rate_", maxsplit=1)
    diagnosed = _policy_prediction(
        metric_name="diagnosed_plhiv",
        train_end_year=train_end_year,
        target_quarter=target_quarter,
        policy_id=forecast_part,
        history_by_metric=history_by_metric,
    )
    ratio = _rate_policy_prediction(
        history=art_diagnosed_ratio_history,
        train_end_year=train_end_year,
        target_quarter=target_quarter,
        policy_id=rate_part,
    )
    if diagnosed is None or ratio is None:
        return None
    diagnosed = max(float(diagnosed), 0.0)
    return float(min(max(diagnosed * float(ratio), 0.0), diagnosed))


def _score_art_policy_records(
    *,
    records: list[dict[str, Any]],
    policy_id: str,
    history_by_metric: dict[str, list[tuple[int, int, float]]],
    art_diagnosed_ratio_history: list[tuple[int, int, float]],
) -> dict[str, Any]:
    policy_errors: list[float] = []
    r10_errors: list[float] = []
    carry_errors: list[float] = []
    stock_cone_violations = 0
    for record in records:
        prediction = _art_policy_prediction(
            record=record,
            policy_id=policy_id,
            history_by_metric=history_by_metric,
            art_diagnosed_ratio_history=art_diagnosed_ratio_history,
        )
        if prediction is None:
            continue
        target = _finite_float(record.get("target_value"))
        scale = _finite_float(record.get("scale"))
        if target is None or scale is None:
            continue
        scale = max(float(scale), float(np.finfo(np.float32).eps))
        policy_errors.append(abs(float(prediction) - float(target)) / scale)
        r10_errors.append(float(record["r10_norm_error"]))
        carry_errors.append(float(record["carry_forward_norm_error"]))
        diagnosed_target = _finite_float(record.get("diagnosed_target_value"))
        if diagnosed_target is not None and float(prediction) > float(diagnosed_target) + float(np.finfo(np.float32).eps):
            stock_cone_violations += 1
    if not policy_errors:
        return {
            "entry_count": 0,
            "policy_mean_mae": None,
            "strict_r10_mean_mae": None,
            "carry_forward_mean_mae": None,
            "policy_minus_strict_r10": None,
            "policy_minus_carry_forward": None,
            "stock_cone_violation_count": int(stock_cone_violations),
            "status": "not_evaluable",
        }
    policy_array = np.asarray(policy_errors, dtype=np.float64)
    r10_array = np.asarray(r10_errors, dtype=np.float64)
    carry_array = np.asarray(carry_errors, dtype=np.float64)
    policy_mean = float(np.mean(policy_array))
    r10_mean = float(np.mean(r10_array))
    carry_mean = float(np.mean(carry_array))
    blockers: list[str] = []
    if policy_mean >= r10_mean:
        blockers.append("policy_not_better_than_strict_r10")
    if policy_mean >= carry_mean:
        blockers.append("policy_not_better_than_carry_forward")
    if stock_cone_violations:
        blockers.append("stock_cone_violation")
    return {
        "entry_count": len(policy_errors),
        "policy_mean_mae": policy_mean,
        "strict_r10_mean_mae": r10_mean,
        "carry_forward_mean_mae": carry_mean,
        "policy_minus_strict_r10": float(policy_mean - r10_mean),
        "policy_minus_carry_forward": float(policy_mean - carry_mean),
        "policy_worst_mae": float(np.max(policy_array)),
        "strict_r10_worst_mae": float(np.max(r10_array)),
        "carry_forward_worst_mae": float(np.max(carry_array)),
        "stock_cone_violation_count": int(stock_cone_violations),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _art_policy_probe_rows(
    *,
    records: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    history = _history_by_metric(source_rows)
    ratio_history = _ratio_history(source_rows)
    diagnosed_by_key = {
        (int(row.get("train_end_year") or 0), int(row.get("horizon_years") or 0), str(row.get("quarter") or "")): float(row["target_value"])
        for row in records
        if str(row.get("metric_name") or "") == "diagnosed_plhiv"
        and _finite_float(row.get("target_value")) is not None
    }
    art_records = []
    for record in records:
        if str(record.get("metric_name") or "") != "alive_on_art":
            continue
        output = dict(record)
        output["diagnosed_target_value"] = diagnosed_by_key.get(
            (
                int(record.get("train_end_year") or 0),
                int(record.get("horizon_years") or 0),
                str(record.get("quarter") or ""),
            )
        )
        art_records.append(output)
    rows: list[dict[str, Any]] = []
    for scope in ("all_mapped", "program_mapped"):
        for horizon in horizons:
            scoped = [
                record
                for record in art_records
                if int(record.get("horizon_years") or 0) == int(horizon)
                and _record_scope(record, scope)
            ]
            for policy_id in R33_ART_POLICY_IDS:
                rows.append(
                    {
                        "scope": scope,
                        "horizon_years": int(horizon),
                        "metric_name": "alive_on_art",
                        "policy_id": policy_id,
                        **_score_art_policy_records(
                            records=scoped,
                            policy_id=policy_id,
                            history_by_metric=history,
                            art_diagnosed_ratio_history=ratio_history,
                        ),
                    }
                )
    return rows


def _best_policy_rows(rows: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _finite_float(row.get("policy_mean_mae")) is None:
            continue
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    for _key, values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        output.append(
            min(
                values,
                key=lambda row: (
                    float(row.get("policy_mean_mae") or float("inf")),
                    str(row.get("policy_id") or ""),
                ),
            )
        )
    return output


def _unique_split_quarters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        key = (
            int(record.get("train_end_year") or 0),
            int(record.get("horizon_years") or 0),
            str(record.get("quarter") or ""),
        )
        if key in seen or not key[2]:
            continue
        seen.add(key)
        output.append(
            {
                "train_end_year": key[0],
                "horizon_years": key[1],
                "quarter": key[2],
            }
        )
    return output


def _rate_target_scope(row: dict[str, Any], numerator_metric: str) -> tuple[str, bool]:
    provenance = _provenance_for_metric(row, numerator_metric)
    unknown = not provenance
    lineage = _lineage_id(provenance)
    return lineage, bool(unknown)


def _rate_policy_probe_rows(
    *,
    records: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    split_quarters = _unique_split_quarters(records)
    rows: list[dict[str, Any]] = []
    for spec in BACK_HALF_RATE_SPECS:
        rate_id = str(spec["rate_id"])
        numerator = str(spec["numerator_metric"])
        denominator = str(spec["denominator_metric"])
        history = _rate_history(source_rows, numerator_metric=numerator, denominator_metric=denominator)
        target_records: list[dict[str, Any]] = []
        for split in split_quarters:
            if int(split["horizon_years"]) not in horizons:
                continue
            target_row = source_by_quarter.get(str(split["quarter"]) or "", {})
            target_rate = _rate_from_row(target_row, numerator_metric=numerator, denominator_metric=denominator)
            if target_rate is None:
                continue
            lineage, unknown = _rate_target_scope(target_row, numerator)
            target_records.append(
                {
                    **split,
                    "rate_id": rate_id,
                    "target_rate": float(target_rate),
                    "source_lineage": lineage,
                    "unknown_provenance": bool(unknown),
                }
            )
        for scope in ("all_mapped", "program_mapped"):
            for horizon in horizons:
                scoped = [
                    record
                    for record in target_records
                    if int(record.get("horizon_years") or 0) == int(horizon)
                    and _record_scope(record, scope)
                ]
                for policy_id in R33_RATE_POLICY_IDS:
                    rows.append(
                        {
                            "scope": scope,
                            "horizon_years": int(horizon),
                            "rate_id": rate_id,
                            "policy_id": policy_id,
                            **_score_rate_policy_records(records=scoped, history=history, policy_id=policy_id),
                        }
                    )
    return rows


def _score_rate_policy_records(
    *,
    records: list[dict[str, Any]],
    history: list[tuple[int, int, float]],
    policy_id: str,
) -> dict[str, Any]:
    policy_errors: list[float] = []
    carry_errors: list[float] = []
    for record in records:
        prediction = _rate_policy_prediction(
            history=history,
            train_end_year=int(record.get("train_end_year") or 0),
            target_quarter=str(record.get("quarter") or ""),
            policy_id=policy_id,
        )
        carry = _rate_policy_prediction(
            history=history,
            train_end_year=int(record.get("train_end_year") or 0),
            target_quarter=str(record.get("quarter") or ""),
            policy_id="carry_forward",
        )
        target = _finite_float(record.get("target_rate"))
        if prediction is None or carry is None or target is None:
            continue
        policy_errors.append(abs(float(prediction) - float(target)))
        carry_errors.append(abs(float(carry) - float(target)))
    if not policy_errors:
        return {
            "entry_count": 0,
            "policy_mean_mae": None,
            "conditional_carry_forward_mean_mae": None,
            "policy_minus_conditional_carry_forward": None,
            "status": "not_evaluable",
        }
    policy_array = np.asarray(policy_errors, dtype=np.float64)
    carry_array = np.asarray(carry_errors, dtype=np.float64)
    policy_mean = float(np.mean(policy_array))
    carry_mean = float(np.mean(carry_array))
    blockers: list[str] = []
    if policy_id == "carry_forward":
        blockers.append("conditional_carry_forward_is_baseline_not_promotional")
    elif policy_mean >= carry_mean:
        blockers.append("policy_not_better_than_conditional_carry_forward")
    return {
        "entry_count": len(policy_errors),
        "policy_mean_mae": policy_mean,
        "conditional_carry_forward_mean_mae": carry_mean,
        "policy_minus_conditional_carry_forward": float(policy_mean - carry_mean),
        "policy_worst_mae": float(np.max(policy_array)),
        "conditional_carry_forward_worst_mae": float(np.max(carry_array)),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "blockers"})
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
        "# Phase 3 R33 H5 Back-Half Process Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Best ART Stock Policies",
        "",
        "| Scope | Horizon | Policy | Entries | Policy | Strict R10 | Delta | Carry | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(report.get("best_art_policy_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | `{row.get('policy_id')}` | "
            f"{row.get('entry_count')} | {_format_float(row.get('policy_mean_mae'))} | "
            f"{_format_float(row.get('strict_r10_mean_mae'))} | "
            f"{_format_float(row.get('policy_minus_strict_r10'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | `{row.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Best Conditional Back-Half Rate Policies",
            "",
            "| Scope | Horizon | Rate | Policy | Entries | Policy | Rate Carry | Delta | Status |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in list(report.get("best_rate_policy_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | `{row.get('rate_id')}` | "
            f"`{row.get('policy_id')}` | {row.get('entry_count')} | "
            f"{_format_float(row.get('policy_mean_mae'))} | "
            f"{_format_float(row.get('conditional_carry_forward_mean_mae'))} | "
            f"{_format_float(row.get('policy_minus_conditional_carry_forward'))} | `{row.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R33 is a strict train-only back-half audit, not a promoted model.",
            "- ART policies are scored on strict mapped R10 `alive_on_art` cells and must beat both strict R10 and carry-forward.",
            "- Conditional VL/suppression rates are scored against last-observed conditional-rate carry-forward because R10 does not supply a comparable VL/suppression process target.",
            "- No policy mutates `new_diagnosed_cases_period`; R32 already isolated that diagnosis-flow repair.",
            "- Promotion requires wiring any passing policy into conserved state dynamics and re-running the full R13/R29/annual gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, best_art_rows: list[dict[str, Any]], best_rate_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    program_art = [
        row
        for row in best_art_rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    program_rates = [
        row
        for row in best_rate_rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("R33 train-only back-half audit", fontsize=14, fontweight="bold")
    art_labels = [f"h{row.get('horizon_years')}" for row in program_art]
    art_deltas = np.asarray([float(row.get("policy_minus_strict_r10") or np.nan) for row in program_art], dtype=np.float64)
    axes[0].bar(np.arange(len(program_art)), art_deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in art_deltas])
    axes[0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_xticks(np.arange(len(program_art)))
    axes[0].set_xticklabels(art_labels)
    axes[0].set_title("ART stock: best policy minus strict R10")
    axes[0].set_ylabel("normalized MAE delta")
    rate_labels = [f"h{row.get('horizon_years')} {row.get('rate_id')}" for row in program_rates]
    rate_deltas = np.asarray([float(row.get("policy_minus_conditional_carry_forward") or np.nan) for row in program_rates], dtype=np.float64)
    axes[1].bar(np.arange(len(program_rates)), rate_deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in rate_deltas])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(program_rates)))
    axes[1].set_xticklabels(rate_labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Back-half rates: best policy minus rate carry")
    axes[1].set_ylabel("absolute rate MAE delta")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r33_h5_backhalf_process_audit(
    *,
    run_id: str = R33_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (3, 5),
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
    records, reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    art_rows = _art_policy_probe_rows(records=records, source_rows=source_rows, horizons=horizons)
    rate_rows = _rate_policy_probe_rows(records=records, source_rows=source_rows, horizons=horizons)
    best_art_rows = _best_policy_rows(art_rows, key_fields=("scope", "horizon_years", "metric_name"))
    best_rate_rows = _best_policy_rows(rate_rows, key_fields=("scope", "horizon_years", "rate_id"))
    program_art = [
        row
        for row in best_art_rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    program_rates = [
        row
        for row in best_rate_rows
        if str(row.get("scope") or "") == "program_mapped"
        and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    blockers = [
        f"h{row.get('horizon_years')}_art_policy_not_better_than_strict_r10"
        for row in program_art
        if str(row.get("status") or "") != "pass"
    ]
    blockers.extend(
        f"h{row.get('horizon_years')}_{row.get('rate_id')}_policy_not_better_than_rate_carry"
        for row in program_rates
        if str(row.get("status") or "") != "pass"
    )
    if not program_art:
        blockers.append("program_art_cells_not_evaluable")
    if not program_rates:
        blockers.append("program_rate_cells_not_evaluable")
    verdict = (
        "R33 finds a train-only back-half policy set that clears the ART and conditional-rate long-horizon blockers."
        if not blockers
        else "R33 blocks promotion: train-only ART/back-half policies do not clear every h3/h5 program blocker."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R33_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "horizons": list(horizons),
        "stock_metrics": list(CASCADE_STOCK_METRICS),
        "art_policy_ids": list(R33_ART_POLICY_IDS),
        "rate_policy_ids": list(R33_RATE_POLICY_IDS),
        "reference_rows": reference_rows,
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "verdict": verdict,
        "art_policy_rows": art_rows,
        "best_art_policy_rows": best_art_rows,
        "rate_policy_rows": rate_rows,
        "best_rate_policy_rows": best_rate_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r33_h5_backhalf_process_audit.json"
    md_path = analysis_dir / "r33_h5_backhalf_process_audit.md"
    art_csv = analysis_dir / "r33_art_policy_rows.csv"
    best_art_csv = analysis_dir / "r33_best_art_policy_rows.csv"
    rate_csv = analysis_dir / "r33_rate_policy_rows.csv"
    best_rate_csv = analysis_dir / "r33_best_rate_policy_rows.csv"
    dashboard_path = analysis_dir / "r33_h5_backhalf_process_audit_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "art_policy_csv": art_csv.as_posix(),
        "best_art_policy_csv": best_art_csv.as_posix(),
        "rate_policy_csv": rate_csv.as_posix(),
        "best_rate_policy_csv": best_rate_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(art_csv, art_rows)
    _write_csv(best_art_csv, best_art_rows)
    _write_csv(rate_csv, rate_rows)
    _write_csv(best_rate_csv, best_rate_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, best_art_rows, best_rate_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R33 h5 back-half process audit.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R33_RUN_ID)
    args = parser.parse_args()
    run_r33_h5_backhalf_process_audit(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
