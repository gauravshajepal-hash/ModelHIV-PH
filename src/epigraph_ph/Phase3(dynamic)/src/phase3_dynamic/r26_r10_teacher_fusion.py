from __future__ import annotations

import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import default_epigraph_root, sandbox_repo_root
from .r11_sparse_state_space import (
    FLOAT_NONREGRESSION_TOLERANCE,
    R10_COMPARABLE_METRICS,
    _finite_float,
    _generated_at,
    _r10_horizon_replay_paths,
    _select_horizon_matched_r10_reference,
)
from .runtime import ensure_dir, read_json, write_json


R26_SCHEMA_VERSION = "phase3_dynamic.r26_r10_teacher_fusion.v1"
R26_POLICIES: tuple[str, ...] = (
    "identity_r10",
    "metric_carry_selector",
    "metric_log_residual",
    "metric_alpha_blend",
)


def _metric_scales(split_row: dict[str, Any]) -> dict[str, float]:
    scales: dict[str, float] = {}
    audit = dict(split_row.get("endpoint_audit") or {})
    for source_name in ("candidate", "carry_forward"):
        by_metric = dict(dict(audit.get(source_name) or {}).get("by_metric") or {})
        for metric_name, metric_payload in by_metric.items():
            payload = dict(metric_payload or {})
            raw = _finite_float(payload.get("raw_mae"))
            norm = _finite_float(payload.get("normalized_mae"))
            if raw is not None and norm is not None and norm > 0.0:
                scales[str(metric_name)] = float(raw / norm)
    for metric_name in R10_COMPARABLE_METRICS:
        scales.setdefault(metric_name, 1.0)
    return scales


def _rows_by_quarter(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("quarter") or ""): dict(row)
        for row in rows
        if row.get("quarter")
    }


def _scored_records_from_split(split_row: dict[str, Any]) -> list[dict[str, Any]]:
    scales = _metric_scales(split_row)
    r10_by_quarter = _rows_by_quarter(list(split_row.get("candidate_prediction_rows") or []))
    carry_by_quarter = _rows_by_quarter(list(split_row.get("carry_forward_prediction_rows") or []))
    records: list[dict[str, Any]] = []
    for target_row in list(split_row.get("holdout_target_rows") or []):
        if not isinstance(target_row, dict):
            continue
        quarter = str(target_row.get("quarter") or "")
        tiers = dict(target_row.get("metric_tiers") or {})
        r10_row = r10_by_quarter.get(quarter, {})
        carry_row = carry_by_quarter.get(quarter, {})
        for metric_name in R10_COMPARABLE_METRICS:
            if str(tiers.get(metric_name) or "") not in {"exact_observed", "bridge_observed"}:
                continue
            target = _finite_float(target_row.get(metric_name))
            r10_value = _finite_float(r10_row.get(metric_name))
            carry_value = _finite_float(carry_row.get(metric_name))
            if target is None or r10_value is None or carry_value is None:
                continue
            scale = max(float(scales.get(metric_name, 1.0)), float(np.finfo(np.float32).eps))
            records.append(
                {
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "target_value": float(max(target, 0.0)),
                    "r10_value": float(max(r10_value, 0.0)),
                    "carry_forward_value": float(max(carry_value, 0.0)),
                    "scale": scale,
                    "tier": str(tiers.get(metric_name) or ""),
                }
            )
    return records


def _score_records(records: list[dict[str, Any]], predicted_by_record: list[float]) -> dict[str, Any]:
    errors: list[float] = []
    by_metric: dict[str, list[float]] = defaultdict(list)
    for record, prediction in zip(records, predicted_by_record):
        target = float(record["target_value"])
        scale = max(float(record["scale"]), float(np.finfo(np.float32).eps))
        error = abs(float(prediction) - target) / scale
        errors.append(float(error))
        by_metric[str(record["metric_name"])].append(float(error))
    return {
        "mean_mae": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
        "worst_mae": None if not errors else float(np.max(np.asarray(errors, dtype=np.float64))),
        "entry_count": len(errors),
        "metric_rows": [
            {
                "metric_name": metric_name,
                "entry_count": len(values),
                "mean_norm_error": float(np.mean(np.asarray(values, dtype=np.float64))),
                "worst_norm_error": float(np.max(np.asarray(values, dtype=np.float64))),
            }
            for metric_name, values in sorted(by_metric.items())
            if values
        ],
    }


def _alpha_candidates(records: list[dict[str, Any]]) -> list[float]:
    candidates = {0.0, 1.0}
    for record in records:
        r10 = float(record["r10_value"])
        carry = float(record["carry_forward_value"])
        target = float(record["target_value"])
        delta = carry - r10
        if abs(delta) <= float(np.finfo(np.float64).eps):
            continue
        reference_errors = {
            0.0,
            abs(r10 - target),
            abs(carry - target),
        }
        for raw_error in reference_errors:
            for signed_error in (-raw_error, raw_error):
                alpha = (target + signed_error - r10) / delta
                if -FLOAT_NONREGRESSION_TOLERANCE <= alpha <= 1.0 + FLOAT_NONREGRESSION_TOLERANCE:
                    candidates.add(float(min(max(alpha, 0.0), 1.0)))
        alpha_target = (target - r10) / delta
        if -FLOAT_NONREGRESSION_TOLERANCE <= alpha_target <= 1.0 + FLOAT_NONREGRESSION_TOLERANCE:
            candidates.add(float(min(max(alpha_target, 0.0), 1.0)))
    return sorted(candidates)


def _policy_predictions(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[float]:
    policy_id = str(policy.get("policy_id") or "identity_r10")
    metric_policy = dict(policy.get("metric_policy") or {})
    output: list[float] = []
    for record in records:
        metric_name = str(record["metric_name"])
        r10_value = float(record["r10_value"])
        carry_value = float(record["carry_forward_value"])
        metric_payload = dict(metric_policy.get(metric_name) or {})
        if policy_id == "identity_r10" or not metric_payload:
            output.append(r10_value)
        elif str(metric_payload.get("kind") or "") == "carry":
            output.append(carry_value)
        elif str(metric_payload.get("kind") or "") == "log_residual":
            residual = float(metric_payload.get("log_residual") or 0.0)
            output.append(float(max(np.expm1(np.log1p(max(r10_value, 0.0)) + residual), 0.0)))
        elif str(metric_payload.get("kind") or "") == "alpha_blend":
            alpha = float(metric_payload.get("alpha") or 0.0)
            output.append(float((1.0 - alpha) * r10_value + alpha * carry_value))
        else:
            output.append(r10_value)
    return output


def _fit_metric_policy(previous_records: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    metric_records = [record for record in previous_records if str(record.get("metric_name") or "") == metric_name]
    if not metric_records:
        return {"kind": "identity"}
    r10_score = _score_records(metric_records, [float(record["r10_value"]) for record in metric_records])
    carry_score = _score_records(metric_records, [float(record["carry_forward_value"]) for record in metric_records])
    r10_mean = _finite_float(r10_score.get("mean_mae"))
    r10_worst = _finite_float(r10_score.get("worst_mae"))
    carry_mean = _finite_float(carry_score.get("mean_mae"))
    carry_worst = _finite_float(carry_score.get("worst_mae"))
    candidates: list[dict[str, Any]] = [{"kind": "identity", "mean_mae": r10_mean, "worst_mae": r10_worst}]
    if (
        carry_mean is not None
        and carry_worst is not None
        and r10_mean is not None
        and r10_worst is not None
        and carry_mean < r10_mean
        and carry_worst <= r10_worst + FLOAT_NONREGRESSION_TOLERANCE
    ):
        candidates.append({"kind": "carry", "mean_mae": carry_mean, "worst_mae": carry_worst})
    residual = float(
        np.median(
            np.asarray(
                [
                    np.log1p(float(record["target_value"])) - np.log1p(float(record["r10_value"]))
                    for record in metric_records
                ],
                dtype=np.float64,
            )
        )
    )
    residual_score = _score_records(
        metric_records,
        [
            float(max(np.expm1(np.log1p(float(record["r10_value"])) + residual), 0.0))
            for record in metric_records
        ],
    )
    residual_mean = _finite_float(residual_score.get("mean_mae"))
    residual_worst = _finite_float(residual_score.get("worst_mae"))
    if (
        residual_mean is not None
        and residual_worst is not None
        and r10_mean is not None
        and r10_worst is not None
        and residual_mean < r10_mean
        and residual_worst <= r10_worst + FLOAT_NONREGRESSION_TOLERANCE
    ):
        candidates.append(
            {
                "kind": "log_residual",
                "log_residual": residual,
                "mean_mae": residual_mean,
                "worst_mae": residual_worst,
            }
        )
    best_alpha = 0.0
    best_alpha_score: dict[str, Any] | None = None
    for alpha in _alpha_candidates(metric_records):
        alpha_score = _score_records(
            metric_records,
            [
                float((1.0 - float(alpha)) * float(record["r10_value"]) + float(alpha) * float(record["carry_forward_value"]))
                for record in metric_records
            ],
        )
        alpha_mean = _finite_float(alpha_score.get("mean_mae"))
        if alpha_mean is None:
            continue
        if best_alpha_score is None or alpha_mean < float(best_alpha_score["mean_mae"]):
            best_alpha = float(alpha)
            best_alpha_score = alpha_score
    if best_alpha_score is not None:
        alpha_mean = _finite_float(best_alpha_score.get("mean_mae"))
        alpha_worst = _finite_float(best_alpha_score.get("worst_mae"))
        if (
            alpha_mean is not None
            and alpha_worst is not None
            and r10_mean is not None
            and r10_worst is not None
            and alpha_mean < r10_mean
            and alpha_worst <= r10_worst + FLOAT_NONREGRESSION_TOLERANCE
        ):
            candidates.append(
                {
                    "kind": "alpha_blend",
                    "alpha": best_alpha,
                    "mean_mae": alpha_mean,
                    "worst_mae": alpha_worst,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            float("inf") if _finite_float(row.get("mean_mae")) is None else float(row["mean_mae"]),
            str(row.get("kind") or ""),
        ),
    )
    return selected


def _fit_fusion_policy(previous_records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_policy = {
        metric_name: _fit_metric_policy(previous_records, metric_name)
        for metric_name in R10_COMPARABLE_METRICS
    }
    selected_count = sum(1 for payload in metric_policy.values() if str(payload.get("kind") or "") != "identity")
    return {
        "policy_id": "metric_safe_fusion" if selected_count else "identity_r10",
        "metric_policy": metric_policy,
        "selected_metric_count": selected_count,
        "previous_record_count": len(previous_records),
        "contract": (
            "For each metric, previous blocked replay rows can select carry-forward, log-residual correction, "
            "or exact convex blend only if that policy improves previous mean error without worsening previous "
            "worst-case error versus frozen matched R10."
        ),
    }


def _select_reference_result(report: dict[str, Any]) -> dict[str, Any]:
    reference = _select_horizon_matched_r10_reference(report)
    reference_id = str(reference.get("reference_experiment_id") or "")
    for result in list(report.get("results") or []):
        if isinstance(result, dict) and str(result.get("experiment_id") or "") == reference_id:
            return result
    raise ValueError(f"Could not locate R10 reference result {reference_id}")


def _evaluate_horizon(path: Path, horizon: int) -> dict[str, Any]:
    report = read_json(path, default={})
    if not isinstance(report, dict):
        raise ValueError(f"Invalid R10 replay report: {path}")
    reference = _select_horizon_matched_r10_reference(report)
    result = _select_reference_result(report)
    previous_records: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_row in sorted(
        list(result.get("quarterly_rows") or []),
        key=lambda row: int(dict(row).get("train_end_year") or 0),
    ):
        if not isinstance(split_row, dict):
            continue
        records = _scored_records_from_split(split_row)
        if not records:
            continue
        policy = _fit_fusion_policy(previous_records)
        r10_score = _score_records(records, [float(record["r10_value"]) for record in records])
        carry_score = _score_records(records, [float(record["carry_forward_value"]) for record in records])
        fusion_predictions = _policy_predictions(records, policy)
        fusion_score = _score_records(records, fusion_predictions)
        split_rows.append(
            {
                "horizon_years": int(horizon),
                "train_end_year": int(split_row.get("train_end_year") or 0),
                "record_count": len(records),
                "policy": policy,
                "r10_mean_mae": _finite_float(r10_score.get("mean_mae")),
                "r10_worst_mae": _finite_float(r10_score.get("worst_mae")),
                "carry_forward_mean_mae": _finite_float(carry_score.get("mean_mae")),
                "fusion_mean_mae": _finite_float(fusion_score.get("mean_mae")),
                "fusion_worst_mae": _finite_float(fusion_score.get("worst_mae")),
                "fusion_minus_r10": None
                if _finite_float(fusion_score.get("mean_mae")) is None or _finite_float(r10_score.get("mean_mae")) is None
                else float(float(fusion_score["mean_mae"]) - float(r10_score["mean_mae"])),
                "fusion_minus_carry_forward": None
                if _finite_float(fusion_score.get("mean_mae")) is None or _finite_float(carry_score.get("mean_mae")) is None
                else float(float(fusion_score["mean_mae"]) - float(carry_score["mean_mae"])),
            }
        )
        previous_records.extend(records)
    fusion_values = [float(row["fusion_mean_mae"]) for row in split_rows if _finite_float(row.get("fusion_mean_mae")) is not None]
    r10_values = [float(row["r10_mean_mae"]) for row in split_rows if _finite_float(row.get("r10_mean_mae")) is not None]
    carry_values = [float(row["carry_forward_mean_mae"]) for row in split_rows if _finite_float(row.get("carry_forward_mean_mae")) is not None]
    fusion_mean = None if not fusion_values else float(np.mean(np.asarray(fusion_values, dtype=np.float64)))
    r10_mean = None if not r10_values else float(np.mean(np.asarray(r10_values, dtype=np.float64)))
    carry_mean = None if not carry_values else float(np.mean(np.asarray(carry_values, dtype=np.float64)))
    replay_reference_mean = _finite_float(reference.get("reference_quarterly_mean_mae"))
    reproduction_abs_delta = None if r10_mean is None or replay_reference_mean is None else float(abs(r10_mean - replay_reference_mean))
    blockers: list[str] = []
    if reproduction_abs_delta is None:
        blockers.append("missing_r10_reproduction_check")
    elif reproduction_abs_delta > 1.0e-9:
        blockers.append("r10_reproduction_check_failed")
    if fusion_mean is None or r10_mean is None:
        blockers.append("missing_fusion_or_r10_score")
    elif fusion_mean >= r10_mean:
        blockers.append("fusion_not_better_than_matched_r10")
    if fusion_mean is None or carry_mean is None:
        blockers.append("missing_fusion_or_carry_score")
    elif fusion_mean >= carry_mean:
        blockers.append("fusion_not_better_than_carry_forward")
    return {
        "horizon_years": int(horizon),
        "artifact_path": path.as_posix(),
        "reference_experiment_id": str(result.get("experiment_id") or ""),
        "split_count": len(split_rows),
        "fusion_mean_mae": fusion_mean,
        "matched_r10_mean_mae": r10_mean,
        "replay_reference_mean_mae": replay_reference_mean,
        "reproduction_abs_delta": reproduction_abs_delta,
        "carry_forward_mean_mae": carry_mean,
        "fusion_minus_r10": None if fusion_mean is None or r10_mean is None else float(fusion_mean - r10_mean),
        "fusion_minus_carry_forward": None if fusion_mean is None or carry_mean is None else float(fusion_mean - carry_mean),
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "split_rows": split_rows,
    }


def _write_csv(path: Path, horizon_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for horizon_row in horizon_rows:
        for split_row in list(horizon_row.get("split_rows") or []):
            rows.append(
                {
                    "horizon_years": horizon_row.get("horizon_years"),
                    "train_end_year": split_row.get("train_end_year"),
                    "record_count": split_row.get("record_count"),
                    "fusion_mean_mae": split_row.get("fusion_mean_mae"),
                    "matched_r10_mean_mae": split_row.get("r10_mean_mae"),
                    "carry_forward_mean_mae": split_row.get("carry_forward_mean_mae"),
                    "fusion_minus_r10": split_row.get("fusion_minus_r10"),
                    "fusion_minus_carry_forward": split_row.get("fusion_minus_carry_forward"),
                    "reproduction_abs_delta": horizon_row.get("reproduction_abs_delta"),
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


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R26 R10-Teacher Fusion Diagnostic",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Horizon Scores",
        "",
        "| Horizon | Fusion | Matched R10 | Carry-forward | Fusion minus R10 | Fusion minus carry | Status | Blockers |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("horizon_rows") or []):
        lines.append(
            f"| {row.get('horizon_years')} | {_format_float(row.get('fusion_mean_mae'))} | "
            f"{_format_float(row.get('matched_r10_mean_mae'))} | {_format_float(row.get('carry_forward_mean_mae'))} | "
            f"{_format_float(row.get('fusion_minus_r10'))} | {_format_float(row.get('fusion_minus_carry_forward'))} | "
            f"`{row.get('status')}` | `{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## R10 Replay Reproduction Check",
            "",
            "| Horizon | Rescored R10 | Replay reference | Absolute delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in list(report.get("horizon_rows") or []):
        lines.append(
            f"| {row.get('horizon_years')} | {_format_float(row.get('matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('replay_reference_mean_mae'))} | "
            f"{_format_float(row.get('reproduction_abs_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R26 uses frozen horizon-matched R10 replay artifacts and never reads future splits when choosing a policy for a split.",
            "- Policies are selected from previous blocked replay rows only.",
            "- This is a predictive teacher-fusion diagnostic, not a mechanistic cascade claim.",
            "- If it fails to beat matched R10, the remaining blocker is the R10 endpoint frontier itself, not only R19 implementation details.",
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
    labels = [f"{int(row['horizon_years'])}y" for row in horizon_rows]
    fusion = np.asarray([float(row.get("fusion_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    r10 = np.asarray([float(row.get("matched_r10_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("R26 Train-Origin Fusion vs Matched R10", fontsize=15, fontweight="bold")
    axes[0].bar(x - width, carry, width=width, color="#c9b879", label="carry-forward")
    axes[0].bar(x, fusion, width=width, color="#20639b", label="R26 fusion")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(loc="upper left")
    delta = fusion - r10
    axes[1].bar(x, delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("fusion minus matched R10")
    axes[1].set_title("Negative values beat matched R10")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r26_r10_teacher_fusion(
    *,
    run_id: str = "p3d-r26-r10-teacher-fusion-20260502-s00",
    epigraph_root: Path | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
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
        horizon_rows.append(_evaluate_horizon(Path(path), int(horizon)))
    blockers = sorted({blocker for row in horizon_rows for blocker in list(row.get("blockers") or [])})
    promotion_eligible = bool(horizon_rows) and all(str(row.get("status") or "") == "pass" for row in horizon_rows)
    if promotion_eligible:
        verdict = "R26 beats matched R10 and carry-forward on every tested horizon, but it is a predictive teacher-fusion result, not a mechanistic process claim."
    else:
        verdict = "R26 does not beat matched R10 on every tested horizon. Teacher fusion is diagnostic only and should not replace the R19 mechanistic reference."
    report = {
        "schema_version": R26_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "horizons": list(horizons),
        "policy_candidates": list(R26_POLICIES),
        "promotion_eligible": promotion_eligible,
        "blockers": blockers,
        "verdict": verdict,
        "horizon_rows": horizon_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r26_r10_teacher_fusion.json"
    csv_path = analysis_dir / "r26_r10_teacher_fusion_split_rows.csv"
    md_path = analysis_dir / "r26_r10_teacher_fusion.md"
    dashboard_path = analysis_dir / "r26_r10_teacher_fusion_dashboard.png"
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
    parser = argparse.ArgumentParser(description="Run Phase3 R26 R10-teacher fusion diagnostic.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--run-id", default="p3d-r26-r10-teacher-fusion-20260502-s00")
    args = parser.parse_args()
    run_r26_r10_teacher_fusion(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
    )


if __name__ == "__main__":
    _main()
