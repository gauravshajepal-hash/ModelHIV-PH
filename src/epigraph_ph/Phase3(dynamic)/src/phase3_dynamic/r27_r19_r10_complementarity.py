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
    FLOAT_NONREGRESSION_TOLERANCE,
    R10_COMPARABLE_METRICS,
    _candidate_predictions,
    _finite_float,
    _generated_at,
    _r10_horizon_replay_paths,
)
from .r26_r10_teacher_fusion import (
    _metric_scales,
    _rows_by_quarter,
    _score_records,
    _select_reference_result,
)
from .runtime import ensure_dir, read_json, write_json


R27_SCHEMA_VERSION = "phase3_dynamic.r27_r19_r10_complementarity.v1"
R27_FAMILY = "r27_r19_r10_train_origin_complementarity"


def _alpha_candidates(records: list[dict[str, Any]]) -> list[float]:
    candidates = {0.0, 1.0}
    for record in records:
        r10 = float(record["r10_value"])
        r19 = float(record["r19_value"])
        target = float(record["target_value"])
        delta = r19 - r10
        if abs(delta) <= float(np.finfo(np.float64).eps):
            continue
        for raw_error in (0.0, abs(r10 - target), abs(r19 - target)):
            for signed_error in (-raw_error, raw_error):
                alpha = (target + signed_error - r10) / delta
                if -FLOAT_NONREGRESSION_TOLERANCE <= alpha <= 1.0 + FLOAT_NONREGRESSION_TOLERANCE:
                    candidates.add(float(min(max(alpha, 0.0), 1.0)))
        alpha_target = (target - r10) / delta
        if -FLOAT_NONREGRESSION_TOLERANCE <= alpha_target <= 1.0 + FLOAT_NONREGRESSION_TOLERANCE:
            candidates.add(float(min(max(alpha_target, 0.0), 1.0)))
    return sorted(candidates)


def _predictions_for_policy(records: list[dict[str, Any]], policy_by_metric: dict[str, dict[str, Any]]) -> list[float]:
    output: list[float] = []
    for record in records:
        metric_name = str(record["metric_name"])
        policy = dict(policy_by_metric.get(metric_name) or {"kind": "r10"})
        kind = str(policy.get("kind") or "r10")
        if kind == "r19":
            output.append(float(record["r19_value"]))
        elif kind == "carry":
            output.append(float(record["carry_forward_value"]))
        elif kind == "r10_r19_blend":
            alpha = float(policy.get("alpha") or 0.0)
            output.append(float((1.0 - alpha) * float(record["r10_value"]) + alpha * float(record["r19_value"])))
        else:
            output.append(float(record["r10_value"]))
    return output


def _fit_metric_policy(previous_records: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    metric_records = [record for record in previous_records if str(record.get("metric_name") or "") == metric_name]
    if not metric_records:
        return {"kind": "r10"}
    r10_score = _score_records(metric_records, [float(record["r10_value"]) for record in metric_records])
    r10_mean = _finite_float(r10_score.get("mean_mae"))
    r10_worst = _finite_float(r10_score.get("worst_mae"))
    candidates: list[dict[str, Any]] = [{"kind": "r10", "mean_mae": r10_mean, "worst_mae": r10_worst}]
    for kind, field_name in (("r19", "r19_value"), ("carry", "carry_forward_value")):
        score = _score_records(metric_records, [float(record[field_name]) for record in metric_records])
        mean = _finite_float(score.get("mean_mae"))
        worst = _finite_float(score.get("worst_mae"))
        if (
            mean is not None
            and worst is not None
            and r10_mean is not None
            and r10_worst is not None
            and mean < r10_mean
            and worst <= r10_worst + FLOAT_NONREGRESSION_TOLERANCE
        ):
            candidates.append({"kind": kind, "mean_mae": mean, "worst_mae": worst})
    best_alpha: float | None = None
    best_score: dict[str, Any] | None = None
    for alpha in _alpha_candidates(metric_records):
        score = _score_records(
            metric_records,
            [
                float((1.0 - alpha) * float(record["r10_value"]) + alpha * float(record["r19_value"]))
                for record in metric_records
            ],
        )
        mean = _finite_float(score.get("mean_mae"))
        if mean is None:
            continue
        if best_score is None or mean < float(best_score["mean_mae"]):
            best_alpha = float(alpha)
            best_score = score
    if best_alpha is not None and best_score is not None:
        mean = _finite_float(best_score.get("mean_mae"))
        worst = _finite_float(best_score.get("worst_mae"))
        if (
            mean is not None
            and worst is not None
            and r10_mean is not None
            and r10_worst is not None
            and mean < r10_mean
            and worst <= r10_worst + FLOAT_NONREGRESSION_TOLERANCE
        ):
            candidates.append({"kind": "r10_r19_blend", "alpha": best_alpha, "mean_mae": mean, "worst_mae": worst})
    return min(
        candidates,
        key=lambda row: (
            float("inf") if _finite_float(row.get("mean_mae")) is None else float(row["mean_mae"]),
            str(row.get("kind") or ""),
        ),
    )


def _fit_policy(previous_records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_policy = {
        metric_name: _fit_metric_policy(previous_records, metric_name)
        for metric_name in R10_COMPARABLE_METRICS
    }
    selected_count = sum(1 for row in metric_policy.values() if str(row.get("kind") or "") != "r10")
    return {
        "policy_id": "r19_r10_complementarity" if selected_count else "identity_r10",
        "metric_policy": metric_policy,
        "selected_metric_count": selected_count,
        "previous_record_count": len(previous_records),
    }


def _records_from_split(
    split_row: dict[str, Any],
    *,
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_end_year = int(split_row.get("train_end_year") or 0)
    train_rows = [
        dict(row)
        for row in source_rows
        if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) <= train_end_year
    ]
    target_rows = [dict(row) for row in list(split_row.get("holdout_target_rows") or []) if isinstance(row, dict)]
    if not train_rows or not target_rows:
        return []
    r19_predictions, _summary = _candidate_predictions(
        train_rows,
        target_rows,
        family="r19_joint_service_cascade_process",
    )
    scales = _metric_scales(split_row)
    r10_by_quarter = _rows_by_quarter(list(split_row.get("candidate_prediction_rows") or []))
    carry_by_quarter = _rows_by_quarter(list(split_row.get("carry_forward_prediction_rows") or []))
    r19_by_quarter = _rows_by_quarter(r19_predictions)
    records: list[dict[str, Any]] = []
    for target_row in target_rows:
        quarter = str(target_row.get("quarter") or "")
        tiers = dict(target_row.get("metric_tiers") or {})
        r10_row = r10_by_quarter.get(quarter, {})
        r19_row = r19_by_quarter.get(quarter, {})
        carry_row = carry_by_quarter.get(quarter, {})
        for metric_name in R10_COMPARABLE_METRICS:
            if str(tiers.get(metric_name) or "") not in {"exact_observed", "bridge_observed"}:
                continue
            target = _finite_float(target_row.get(metric_name))
            r10_value = _finite_float(r10_row.get(metric_name))
            r19_value = _finite_float(r19_row.get(metric_name))
            carry_value = _finite_float(carry_row.get(metric_name))
            if target is None or r10_value is None or r19_value is None or carry_value is None:
                continue
            records.append(
                {
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "target_value": float(max(target, 0.0)),
                    "r10_value": float(max(r10_value, 0.0)),
                    "r19_value": float(max(r19_value, 0.0)),
                    "carry_forward_value": float(max(carry_value, 0.0)),
                    "scale": max(float(scales.get(metric_name, 1.0)), float(np.finfo(np.float32).eps)),
                    "tier": str(tiers.get(metric_name) or ""),
                }
            )
    return records


def _evaluate_horizon(path: Path, horizon: int, *, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = read_json(path, default={})
    if not isinstance(report, dict):
        raise ValueError(f"Invalid R10 replay report: {path}")
    result = _select_reference_result(report)
    previous_records: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_row in sorted(
        list(result.get("quarterly_rows") or []),
        key=lambda row: int(dict(row).get("train_end_year") or 0),
    ):
        if not isinstance(split_row, dict):
            continue
        records = _records_from_split(dict(split_row), source_rows=source_rows)
        if not records:
            continue
        policy = _fit_policy(previous_records)
        r10_score = _score_records(records, [float(record["r10_value"]) for record in records])
        r19_score = _score_records(records, [float(record["r19_value"]) for record in records])
        carry_score = _score_records(records, [float(record["carry_forward_value"]) for record in records])
        fusion_predictions = _predictions_for_policy(records, dict(policy.get("metric_policy") or {}))
        fusion_score = _score_records(records, fusion_predictions)
        fusion_mean = _finite_float(fusion_score.get("mean_mae"))
        r10_mean = _finite_float(r10_score.get("mean_mae"))
        r19_mean = _finite_float(r19_score.get("mean_mae"))
        carry_mean = _finite_float(carry_score.get("mean_mae"))
        split_rows.append(
            {
                "horizon_years": int(horizon),
                "train_end_year": int(split_row.get("train_end_year") or 0),
                "record_count": len(records),
                "policy": policy,
                "fusion_mean_mae": fusion_mean,
                "matched_r10_mean_mae": r10_mean,
                "r19_mean_mae": r19_mean,
                "carry_forward_mean_mae": carry_mean,
                "fusion_minus_r10": None if fusion_mean is None or r10_mean is None else float(fusion_mean - r10_mean),
                "fusion_minus_r19": None if fusion_mean is None or r19_mean is None else float(fusion_mean - r19_mean),
                "fusion_minus_carry_forward": None if fusion_mean is None or carry_mean is None else float(fusion_mean - carry_mean),
            }
        )
        previous_records.extend(records)
    values: dict[str, list[float]] = defaultdict(list)
    for row in split_rows:
        for key in ("fusion_mean_mae", "matched_r10_mean_mae", "r19_mean_mae", "carry_forward_mean_mae"):
            value = _finite_float(row.get(key))
            if value is not None:
                values[key].append(value)
    fusion = None if not values["fusion_mean_mae"] else float(np.mean(np.asarray(values["fusion_mean_mae"], dtype=np.float64)))
    r10 = None if not values["matched_r10_mean_mae"] else float(np.mean(np.asarray(values["matched_r10_mean_mae"], dtype=np.float64)))
    r19 = None if not values["r19_mean_mae"] else float(np.mean(np.asarray(values["r19_mean_mae"], dtype=np.float64)))
    carry = None if not values["carry_forward_mean_mae"] else float(np.mean(np.asarray(values["carry_forward_mean_mae"], dtype=np.float64)))
    blockers: list[str] = []
    if fusion is None or r10 is None:
        blockers.append("missing_fusion_or_r10_score")
    elif fusion >= r10:
        blockers.append("fusion_not_better_than_matched_r10")
    if fusion is None or carry is None:
        blockers.append("missing_fusion_or_carry_score")
    elif fusion >= carry:
        blockers.append("fusion_not_better_than_carry_forward")
    return {
        "horizon_years": int(horizon),
        "artifact_path": path.as_posix(),
        "reference_experiment_id": str(result.get("experiment_id") or ""),
        "split_count": len(split_rows),
        "fusion_mean_mae": fusion,
        "matched_r10_mean_mae": r10,
        "r19_mean_mae": r19,
        "carry_forward_mean_mae": carry,
        "fusion_minus_r10": None if fusion is None or r10 is None else float(fusion - r10),
        "fusion_minus_r19": None if fusion is None or r19 is None else float(fusion - r19),
        "fusion_minus_carry_forward": None if fusion is None or carry is None else float(fusion - carry),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "split_rows": split_rows,
    }


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_csv(path: Path, horizon_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for horizon_row in horizon_rows:
        split_rows = list(horizon_row.get("split_rows") or [])
        if not split_rows:
            rows.append(
                {
                    "horizon_years": horizon_row.get("horizon_years"),
                    "row_scope": horizon_row.get("row_scope"),
                    "train_end_year": None,
                    "fusion_mean_mae": horizon_row.get("fusion_mean_mae"),
                    "matched_r10_mean_mae": horizon_row.get("matched_r10_mean_mae"),
                    "r19_mean_mae": horizon_row.get("r19_mean_mae"),
                    "carry_forward_mean_mae": horizon_row.get("carry_forward_mean_mae"),
                    "fusion_minus_r10": horizon_row.get("fusion_minus_r10"),
                    "fusion_minus_r19": horizon_row.get("fusion_minus_r19"),
                    "policy_id": "aggregate_oracle_upper_bound",
                    "selected_metric_count": None,
                }
            )
            continue
        for split_row in split_rows:
            rows.append(
                {
                    "horizon_years": horizon_row.get("horizon_years"),
                    "row_scope": horizon_row.get("row_scope"),
                    "train_end_year": split_row.get("train_end_year"),
                    "fusion_mean_mae": split_row.get("fusion_mean_mae"),
                    "matched_r10_mean_mae": split_row.get("matched_r10_mean_mae"),
                    "r19_mean_mae": split_row.get("r19_mean_mae"),
                    "carry_forward_mean_mae": split_row.get("carry_forward_mean_mae"),
                    "fusion_minus_r10": split_row.get("fusion_minus_r10"),
                    "fusion_minus_r19": split_row.get("fusion_minus_r19"),
                    "policy_id": dict(split_row.get("policy") or {}).get("policy_id"),
                    "selected_metric_count": dict(split_row.get("policy") or {}).get("selected_metric_count"),
                }
            )
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R27 R19/R10 Complementarity Diagnostic",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Horizon Scores",
        "",
        "| Scope | Horizon | Fusion | Matched R10 | Phase3/R19 | Carry-forward | Fusion minus R10 | Fusion minus Phase3 | Status | Blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("horizon_rows") or []):
        lines.append(
            f"| {row.get('row_scope', 'dense_identical_rows')} | {row.get('horizon_years')} | "
            f"{_format_float(row.get('fusion_mean_mae'))} | "
            f"{_format_float(row.get('matched_r10_mean_mae'))} | {_format_float(row.get('r19_mean_mae'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | {_format_float(row.get('fusion_minus_r10'))} | "
            f"{_format_float(row.get('fusion_minus_r19'))} | `{row.get('status')}` | "
            f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(["", "## Contract", ""])
    if str(report.get("mode") or "") == "aggregate_oracle_upper_bound_from_r24":
        lines.extend(
            [
                "- R27 fast mode reads the locked R24 fairness audit and computes an oracle upper bound over Phase3/R19-family versus matched R10.",
                "- Because this oracle sees horizon-level outcomes, it is not a promotable model; it only decides whether a dense learned selector is worth running.",
                "- If the oracle cannot strictly beat matched R10, a train-origin selector using less information should not be expected to beat it.",
                "- The expensive identical-row replay remains available behind `--dense-identical-row-replay` but is not the default path.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- R27 scores R19, matched R10, carry-forward, and a train-origin complementarity selector on identical frozen R10 replay holdout rows.",
                "- The selector sees only previous blocked splits for the same horizon before choosing a metric policy.",
                "- Allowed policies are R10, R19, carry-forward, or an exact R10/R19 convex blend selected from previous split geometry.",
                "- This is a hybrid readout diagnostic, not proof that R19 transition dynamics alone beat matched R10.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, horizon_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    labels = [
        f"{str(row.get('row_scope') or 'dense').replace('_', ' ')} h{int(row['horizon_years'])}"
        for row in horizon_rows
    ]
    fusion = np.asarray([float(row.get("fusion_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    r10 = np.asarray([float(row.get("matched_r10_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    r19 = np.asarray([float(row.get("r19_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("R27 R19/R10 Train-Origin Complementarity", fontsize=15, fontweight="bold")
    axes[0].bar(x - 1.5 * width, carry, width=width, color="#c9b879", label="carry-forward")
    axes[0].bar(x - 0.5 * width, r19, width=width, color="#637a91", label="R19 same rows")
    axes[0].bar(x + 0.5 * width, fusion, width=width, color="#20639b", label="R27 fusion")
    axes[0].bar(x + 1.5 * width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(loc="upper left")
    delta = fusion - r10
    axes[1].bar(x, delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("fusion minus matched R10")
    axes[1].set_title("Negative values beat matched R10")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _r24_audit_path(phase3_root: Path) -> Path:
    return (
        phase3_root
        / "artifacts"
        / "scientific_audits"
        / "phase3_r24_matched_r10_fairness_audit_results_20260502.json"
    )


def _summary_row(
    *,
    row_scope: str,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    phase3_value = _finite_float(source_row.get("r10_comparable_candidate_mean_mae"))
    r10_value = _finite_float(source_row.get("r10_horizon_reference_mae"))
    carry_value = _finite_float(source_row.get("carry_forward_mean_mae"))
    if phase3_value is None or r10_value is None:
        oracle_value = None
    else:
        oracle_value = float(min(phase3_value, r10_value))
    blockers: list[str] = []
    if oracle_value is None or r10_value is None:
        blockers.append("missing_oracle_or_r10_score")
    elif oracle_value >= r10_value:
        blockers.append("oracle_not_strictly_better_than_matched_r10")
    return {
        "row_scope": row_scope,
        "horizon_years": int(source_row.get("horizon_years") or 0),
        "phase3_family": str(source_row.get("family") or "r19_joint_service_cascade_process"),
        "fusion_mean_mae": oracle_value,
        "matched_r10_mean_mae": r10_value,
        "r19_mean_mae": phase3_value,
        "carry_forward_mean_mae": carry_value,
        "fusion_minus_r10": None if oracle_value is None or r10_value is None else float(oracle_value - r10_value),
        "fusion_minus_r19": None if oracle_value is None or phase3_value is None else float(oracle_value - phase3_value),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "split_rows": [],
    }


def run_r27_r19_r10_complementarity_summary(
    *,
    run_id: str = "p3d-r27-r19-r10-complementarity-20260502-s00",
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    audit_path = _r24_audit_path(phase3_root)
    audit = read_json(audit_path, default={})
    if not isinstance(audit, dict) or not audit:
        raise FileNotFoundError(f"Missing R24 fairness audit artifact: {audit_path}")
    horizon_rows: list[dict[str, Any]] = []
    for row in list(audit.get("all_support_r19_rows") or []):
        if isinstance(row, dict):
            horizon_rows.append(_summary_row(row_scope="full_sentinel_r19", source_row=dict(row)))
    best_program_by_horizon: dict[int, dict[str, Any]] = {}
    for row in list(audit.get("program_route_rows") or []):
        if not isinstance(row, dict):
            continue
        horizon = int(row.get("horizon_years") or 0)
        value = _finite_float(row.get("r10_comparable_candidate_mean_mae"))
        if value is None:
            continue
        previous = best_program_by_horizon.get(horizon)
        if previous is None or value < float(previous["r10_comparable_candidate_mean_mae"]):
            best_program_by_horizon[horizon] = dict(row)
    for horizon in sorted(best_program_by_horizon):
        horizon_rows.append(_summary_row(row_scope="program_best_phase3", source_row=best_program_by_horizon[horizon]))
    blockers = sorted({blocker for row in horizon_rows for blocker in list(row.get("blockers") or [])})
    promotion_eligible = bool(horizon_rows) and all(str(row.get("status") or "") == "pass" for row in horizon_rows)
    if promotion_eligible:
        verdict = "R27 oracle complementarity beats matched R10 in every audited route; a train-origin selector is worth implementing."
    else:
        verdict = (
            "R27 oracle complementarity does not beat matched R10 in every audited route. "
            "Even choosing the better of Phase3/R19-family and R10 per horizon cannot clear the program route or full h5 gate."
        )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R27_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "family": R27_FAMILY,
        "mode": "aggregate_oracle_upper_bound_from_r24",
        "source_artifact": audit_path.as_posix(),
        "promotion_eligible": promotion_eligible,
        "blockers": blockers,
        "verdict": verdict,
        "horizon_rows": horizon_rows,
        "artifact_paths": {},
        "contract": (
            "This fast R27 mode is an upper-bound diagnostic over locked R24 fairness rows. "
            "It asks whether any non-leaky train-origin R19/R10 selector could be worth a dense replay; "
            "if the oracle itself cannot strictly beat matched R10, a learned selector cannot be expected to do so."
        ),
    }
    json_path = analysis_dir / "r27_r19_r10_complementarity.json"
    csv_path = analysis_dir / "r27_r19_r10_complementarity_split_rows.csv"
    md_path = analysis_dir / "r27_r19_r10_complementarity.md"
    dashboard_path = analysis_dir / "r27_r19_r10_complementarity_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "csv": csv_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, horizon_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, horizon_rows)
    write_json(json_path, report)
    return report


def run_r27_r19_r10_complementarity(
    *,
    run_id: str = "p3d-r27-r19-r10-complementarity-20260502-s00",
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
    paths = _r10_horizon_replay_paths(root, horizons)
    horizon_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        path = paths.get(int(horizon))
        if path is None:
            horizon_rows.append(
                {
                    "horizon_years": int(horizon),
                    "status": "fail",
                    "blockers": ["missing_r10_replay_artifact"],
                    "split_rows": [],
                }
            )
            continue
        horizon_rows.append(_evaluate_horizon(Path(path), int(horizon), source_rows=source_rows))
    blockers = sorted({blocker for row in horizon_rows for blocker in list(row.get("blockers") or [])})
    promotion_eligible = bool(horizon_rows) and all(str(row.get("status") or "") == "pass" for row in horizon_rows)
    if promotion_eligible:
        verdict = "R27 beats matched R10 and carry-forward on every tested horizon, but remains a hybrid predictive readout diagnostic."
    else:
        verdict = "R27 does not beat matched R10 on every tested horizon. R19/R10 complementarity is diagnostic only."
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R27_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "family": R27_FAMILY,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "horizons": list(horizons),
        "promotion_eligible": promotion_eligible,
        "blockers": blockers,
        "verdict": verdict,
        "horizon_rows": horizon_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r27_r19_r10_complementarity.json"
    csv_path = analysis_dir / "r27_r19_r10_complementarity_split_rows.csv"
    md_path = analysis_dir / "r27_r19_r10_complementarity.md"
    dashboard_path = analysis_dir / "r27_r19_r10_complementarity_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "csv": csv_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, horizon_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, horizon_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R27 R19/R10 complementarity diagnostic.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default="p3d-r27-r19-r10-complementarity-20260502-s00")
    parser.add_argument(
        "--dense-identical-row-replay",
        action="store_true",
        help="Run the expensive identical-row R19/R10 replay instead of the R24 aggregate upper-bound audit.",
    )
    args = parser.parse_args()
    if args.dense_identical_row_replay:
        run_r27_r19_r10_complementarity(
            run_id=str(args.run_id),
            epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
        )
    else:
        run_r27_r19_r10_complementarity_summary(run_id=str(args.run_id))


if __name__ == "__main__":
    _main()
