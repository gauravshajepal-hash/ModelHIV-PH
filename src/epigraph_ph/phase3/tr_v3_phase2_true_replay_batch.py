from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_phase2_structure_batch as phase2_batch
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.frontier.phase2_structural_inputs import load_phase2_structural_inputs
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, build_annual_anchor_rows
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, write_json


EXACT_BASE_EXPERIMENT_ID = "EXP-R10-M1-F1-C1"
DENSE_BASE_EXPERIMENT_ID = "EXP-R10-DENSE-M1-C1-H1"
LOCKBOX_YEARS: list[int] = [2025]
MIN_TRAIN_ROWS = 5


@dataclass(frozen=True, slots=True)
class TrueReplayConfig:
    config_id: str
    use_blocks: bool
    use_hidden: bool
    correction_weight: float
    ridge_penalty: float
    lag_quarters: int = 1
    clip_multiplier: float = 1.25


TRUE_REPLAY_CANDIDATES: tuple[TrueReplayConfig, ...] = (
    TrueReplayConfig("P2-TRUE-BLOCK-r1-w25", use_blocks=True, use_hidden=False, correction_weight=0.25, ridge_penalty=1.0),
    TrueReplayConfig("P2-TRUE-BLOCK-r1-w50", use_blocks=True, use_hidden=False, correction_weight=0.50, ridge_penalty=1.0),
    TrueReplayConfig("P2-TRUE-HIDDEN-r1-w25", use_blocks=False, use_hidden=True, correction_weight=0.25, ridge_penalty=1.0),
    TrueReplayConfig("P2-TRUE-BOTH-r1-w25", use_blocks=True, use_hidden=True, correction_weight=0.25, ridge_penalty=1.0),
    TrueReplayConfig("P2-TRUE-BOTH-r1-w50", use_blocks=True, use_hidden=True, correction_weight=0.50, ridge_penalty=1.0),
)


def _load_structural_inputs(source_run_id: str) -> Any:
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / str(source_run_id)
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run does not exist: {source_run_dir}")
    phase15_dir = source_run_dir / "phase15"
    phase2_dir = source_run_dir / "phase2"
    if not phase15_dir.exists():
        raise FileNotFoundError(f"phase15 directory missing: {phase15_dir}")
    if not phase2_dir.exists():
        raise FileNotFoundError(f"phase2 directory missing: {phase2_dir}")
    return load_phase2_structural_inputs(SimpleNamespace(phase15_dir=phase15_dir, phase2_dir=phase2_dir))


def _lagged_feature_map(structural_inputs: Any, cfg: TrueReplayConfig) -> dict[str, np.ndarray]:
    quarter_axis = [str(value) for value in list(structural_inputs.quarter_axis)]
    block_tensor = np.asarray(structural_inputs.national_quarter_tensor, dtype=np.float32)
    if block_tensor.ndim == 3:
        block_tensor = block_tensor[0]
    hidden_tensor = np.asarray(getattr(structural_inputs, "hidden_mode_quarter_tensor", np.zeros((1, len(quarter_axis), 0), dtype=np.float32)), dtype=np.float32)
    if hidden_tensor.ndim == 3:
        hidden_tensor = hidden_tensor[0]
    feature_map: dict[str, np.ndarray] = {}
    for quarter_idx, quarter in enumerate(quarter_axis):
        source_idx = quarter_idx - int(cfg.lag_quarters)
        if source_idx < 0:
            continue
        parts: list[np.ndarray] = []
        if cfg.use_blocks and block_tensor.size:
            parts.append(np.asarray(block_tensor[source_idx], dtype=np.float32).reshape(-1))
        if cfg.use_hidden and hidden_tensor.size:
            parts.append(np.asarray(hidden_tensor[source_idx], dtype=np.float32).reshape(-1))
        if parts:
            feature_map[str(quarter)] = np.concatenate(parts).astype(np.float32)
    return feature_map


def _fit_ridge_multioutput(design: np.ndarray, target: np.ndarray, ridge_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    if design.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, target.shape[1]), dtype=np.float32)
    intercept = np.ones((design.shape[0], 1), dtype=np.float32)
    augmented = np.hstack([intercept, np.asarray(design, dtype=np.float32)])
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0], dtype=np.float32) * float(ridge_penalty)
    penalty[0, 0] = 0.0
    rhs = augmented.T @ np.asarray(target, dtype=np.float32)
    try:
        beta = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + penalty) @ rhs
    return np.asarray(beta[0], dtype=np.float32), np.asarray(beta[1:], dtype=np.float32)


def _fit_true_replay_model(
    transition_diagnostics: dict[str, Any],
    structural_inputs: Any,
    cfg: TrueReplayConfig,
) -> dict[str, Any] | None:
    residual_state = phase2_batch._build_train_residual_state(transition_diagnostics)
    quarters = [str(value) for value in list(residual_state["quarters"])]
    residual_matrix = np.asarray(residual_state["matrix"], dtype=np.float32)
    feature_lookup = _lagged_feature_map(structural_inputs, cfg)
    design_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    train_quarters: list[str] = []
    for quarter, target_row in zip(quarters, residual_matrix, strict=False):
        feature_row = feature_lookup.get(str(quarter))
        if feature_row is None:
            continue
        design_rows.append(np.asarray(feature_row, dtype=np.float32))
        target_rows.append(np.asarray(target_row, dtype=np.float32))
        train_quarters.append(str(quarter))
    if len(design_rows) < int(MIN_TRAIN_ROWS):
        return None
    design_matrix = np.asarray(design_rows, dtype=np.float32)
    target_matrix = np.asarray(target_rows, dtype=np.float32)
    center = np.mean(design_matrix, axis=0)
    scale = np.std(design_matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    standardized = (design_matrix - center) / scale
    intercept, coefficients = _fit_ridge_multioutput(standardized, target_matrix, cfg.ridge_penalty)
    target_min = np.min(target_matrix, axis=0)
    target_max = np.max(target_matrix, axis=0)
    spread = np.maximum(target_max - target_min, 1e-3)
    lower = target_min - float(cfg.clip_multiplier) * spread
    upper = target_max + float(cfg.clip_multiplier) * spread
    return {
        "config_id": str(cfg.config_id),
        "quarters": train_quarters,
        "center": np.asarray(center, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "intercept": np.asarray(intercept, dtype=np.float32),
        "coefficients": np.asarray(coefficients, dtype=np.float32),
        "clip_lower": np.asarray(lower, dtype=np.float32),
        "clip_upper": np.asarray(upper, dtype=np.float32),
        "feature_count": int(standardized.shape[1]),
        "block_feature_count": int(np.asarray(structural_inputs.national_quarter_tensor, dtype=np.float32).shape[-1]) if cfg.use_blocks else 0,
        "hidden_feature_count": int(np.asarray(getattr(structural_inputs, "hidden_mode_quarter_tensor", np.zeros((1, 0, 0), dtype=np.float32)), dtype=np.float32).shape[-1]) if cfg.use_hidden else 0,
        "train_row_count": int(len(train_quarters)),
        "feature_lookup": feature_lookup,
    }


def _corrected_prediction_rows(
    base_prediction_rows: list[dict[str, Any]],
    model: dict[str, Any] | None,
    cfg: TrueReplayConfig | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model is None or cfg is None:
        return [dict(row) for row in base_prediction_rows], []
    center = np.asarray(model["center"], dtype=np.float32)
    scale = np.asarray(model["scale"], dtype=np.float32)
    intercept = np.asarray(model["intercept"], dtype=np.float32)
    coefficients = np.asarray(model["coefficients"], dtype=np.float32)
    lower = np.asarray(model["clip_lower"], dtype=np.float32)
    upper = np.asarray(model["clip_upper"], dtype=np.float32)
    feature_lookup = dict(model["feature_lookup"])
    weight = float(cfg.correction_weight)
    corrected_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    for base_row in list(base_prediction_rows):
        quarter = str(base_row["quarter"])
        feature = np.asarray(feature_lookup.get(quarter), dtype=np.float32) if quarter in feature_lookup else None
        if feature is None or feature.size == 0:
            residual = np.zeros((len(PRIMARY_METRICS),), dtype=np.float32)
        else:
            standardized = (feature - center) / scale
            residual = np.clip(intercept + standardized @ coefficients, lower, upper)
        diagnosed_correction = float(weight * residual[0])
        art_correction = float(weight * residual[1])
        flow_correction = float(weight * residual[2])
        diagnosed_value = float(max(float(base_row.get("diagnosed_plhiv") or 0.0) + diagnosed_correction, 0.0))
        art_value = float(np.clip(float(base_row.get("alive_on_art") or 0.0) + art_correction, 0.0, diagnosed_value))
        flow_value = float(max(float(base_row.get("new_diagnosed_cases_period") or 0.0) + flow_correction, 0.0))
        corrected_row = dict(base_row)
        corrected_row["diagnosed_plhiv"] = diagnosed_value
        corrected_row["alive_on_art"] = art_value
        corrected_row["new_diagnosed_cases_period"] = flow_value
        if corrected_row.get("virally_suppressed") is not None:
            corrected_row["virally_suppressed"] = float(np.clip(float(corrected_row["virally_suppressed"]), 0.0, art_value))
        corrected_rows.append(corrected_row)
        correction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_correction": diagnosed_correction,
                "art_correction": art_correction,
                "flow_correction": flow_correction,
                "feature_available": bool(feature is not None and feature.size > 0),
            }
        )
    return corrected_rows, correction_rows


def _evaluate_split_overlay(
    split: dict[str, Any],
    *,
    allowed_tiers: set[str],
    structural_inputs: Any,
    cfg: TrueReplayConfig,
) -> dict[str, Any] | None:
    model = _fit_true_replay_model(dict(split.get("transition_diagnostics") or {}), structural_inputs, cfg)
    if model is None:
        return None
    corrected_rows, correction_rows = _corrected_prediction_rows(list(split.get("candidate_prediction_rows") or []), model, cfg)
    holdout_rows = list(split.get("holdout_target_rows") or [])
    metric_scales = dict(split.get("candidate", {}).get("metric_scales") or split.get("metric_scales") or {})
    if not metric_scales:
        metric_scales = {metric_name: 1.0 for metric_name in PRIMARY_METRICS}
        for metric_name in PRIMARY_METRICS:
            targets = [abs(float(row[metric_name])) for row in holdout_rows if row.get(metric_name) is not None]
            if targets:
                metric_scales[metric_name] = max(float(np.median(np.asarray(targets, dtype=np.float64))), 1.0)
    eps = float(split.get("candidate", {}).get("eps") or 1e-6)
    split_mae = phase2_batch.suite._filtered_normalized_mae(
        corrected_rows,
        holdout_rows,
        metric_scales,
        allowed_tiers=allowed_tiers,
        eps=eps,
    )
    raw_metrics = phase2_batch._raw_endpoint_row(corrected_rows, holdout_rows, allowed_tiers=allowed_tiers, metric_scales=metric_scales, eps=eps)
    return {
        "holdout_years": list(split.get("holdout_years") or []),
        "quarterly_mean_mae": float(split_mae),
        "prediction_rows": corrected_rows,
        "correction_rows": correction_rows,
        "raw_metrics": raw_metrics,
        "feature_count": int(model["feature_count"]),
        "block_feature_count": int(model["block_feature_count"]),
        "hidden_feature_count": int(model["hidden_feature_count"]),
        "train_row_count": int(model["train_row_count"]),
    }


def _aggregate_overlay_rows(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not split_rows:
        return {
            "quarterly_mean_mae": None,
            "diagnosed_raw_mae": None,
            "art_raw_mae": None,
            "flow_raw_mae": None,
            "feature_count_mean": None,
            "train_row_count_mean": None,
            "split_count": 0,
        }
    return {
        "quarterly_mean_mae": float(np.mean(np.asarray([float(row["quarterly_mean_mae"]) for row in split_rows], dtype=np.float64))),
        "diagnosed_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["diagnosed_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "art_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["art_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "flow_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["flow_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "feature_count_mean": float(np.mean(np.asarray([float(row["feature_count"]) for row in split_rows], dtype=np.float64))),
        "train_row_count_mean": float(np.mean(np.asarray([float(row["train_row_count"]) for row in split_rows], dtype=np.float64))),
        "split_count": int(len(split_rows)),
    }


def _evaluate_contract_candidates(
    base_result: dict[str, Any],
    *,
    allowed_tiers: set[str],
    structural_inputs: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_summary = dict(base_result.get("quarterly_summary") or {})
    base_endpoint = dict(dict(base_summary.get("endpoint_audit_summary") or {}).get("candidate") or {})
    base_by_metric = dict(base_endpoint.get("by_metric") or {})
    rows.append(
        {
            "config_id": "BASE",
            "status": "baseline",
            "quarterly_mean_mae": float(base_summary.get("candidate_mean_mae") or 0.0),
            "diagnosed_raw_mae": float(dict(base_by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
            "art_raw_mae": float(dict(base_by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
            "flow_raw_mae": float(dict(base_by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
            "feature_count_mean": None,
            "train_row_count_mean": None,
            "split_count": int(len(list(base_result.get("quarterly_rows") or []))),
            "split_rows": [],
        }
    )
    for cfg in TRUE_REPLAY_CANDIDATES:
        split_rows = []
        for split in list(base_result.get("quarterly_rows") or []):
            split_payload = _evaluate_split_overlay(dict(split), allowed_tiers=allowed_tiers, structural_inputs=structural_inputs, cfg=cfg)
            if split_payload is not None:
                split_rows.append(split_payload)
        aggregate = _aggregate_overlay_rows(split_rows)
        rows.append(
            {
                "config_id": str(cfg.config_id),
                "status": "executed" if split_rows else "insufficient_history",
                **aggregate,
                "split_rows": split_rows,
            }
        )
    return rows


def _evaluate_fixed_holdout_overlay(
    *,
    archive_run_id: str,
    contract_name: str,
    base_experiment_id: str,
    frozen_config: dict[str, Any],
    holdout_years: list[int],
    structural_inputs: Any,
    cfg: TrueReplayConfig,
) -> dict[str, Any] | None:
    dataset, base_candidate, scoring_tiers = phase2_batch._custom_fixed_holdout_base_candidate(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        base_experiment_id=base_experiment_id,
        frozen_config=frozen_config,
        holdout_years=holdout_years,
    )
    model = _fit_true_replay_model(dict(base_candidate.get("transition_diagnostics") or {}), structural_inputs, cfg)
    if model is None:
        return None
    corrected_rows, correction_rows = _corrected_prediction_rows(list(base_candidate.get("prediction_rows") or []), model, cfg)
    mae = phase2_batch.suite._filtered_normalized_mae(
        corrected_rows,
        dataset.holdout_rows,
        dataset.metric_scales,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    raw_metrics = phase2_batch._raw_endpoint_row(corrected_rows, dataset.holdout_rows, allowed_tiers=scoring_tiers, metric_scales=dataset.metric_scales, eps=dataset.eps)
    return {
        "config_id": str(cfg.config_id),
        "quarterly_mean_mae": float(mae),
        "raw_metrics": raw_metrics,
        "prediction_rows": corrected_rows,
        "correction_rows": correction_rows,
    }


def _save_overview(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    executed = [row for row in rows if row.get("quarterly_mean_mae") is not None]
    if not executed:
        phase2_batch.suite._plot_placeholder(path, title=title, body="No executed true-replay candidates.")
        return
    labels = [str(row["config_id"]) for row in executed]
    values = [float(row["quarterly_mean_mae"]) for row in executed]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.45)))
    ax.barh(y, values, color=["#999999" if label == "BASE" else "#4c72b0" for label in labels])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Quarterly normalized MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pick_best_nonbaseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("status") == "executed" and row.get("config_id") != "BASE" and row.get("quarterly_mean_mae") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["quarterly_mean_mae"]))


def _lane_report_payload(
    *,
    lane_name: str,
    contract_name: str,
    base_experiment_id: str,
    base_result: dict[str, Any],
    archive_run_id: str,
    structural_inputs: Any,
) -> dict[str, Any]:
    allowed_tiers = publish._scoring_tiers(contract_name)
    rows = _evaluate_contract_candidates(base_result, allowed_tiers=allowed_tiers, structural_inputs=structural_inputs)
    base_row = next(row for row in rows if row["config_id"] == "BASE")
    base_lockbox = phase2_batch._evaluate_fixed_holdout_base(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        base_experiment_id=base_experiment_id,
        frozen_config=dict(base_result.get("best_candidate") or {}),
        holdout_years=list(LOCKBOX_YEARS),
    )
    best_row = _pick_best_nonbaseline(rows)
    lockbox = None
    if best_row is None or float(best_row["quarterly_mean_mae"]) >= float(base_row["quarterly_mean_mae"]):
        decision = {
            "status": "revert",
            "winner_id": "BASE",
            "why": "No true Phase 2 replay readout beat the frozen champion on the primary rolling-origin contract.",
        }
    else:
        cfg = next(candidate for candidate in TRUE_REPLAY_CANDIDATES if candidate.config_id == best_row["config_id"])
        lockbox = _evaluate_fixed_holdout_overlay(
            archive_run_id=archive_run_id,
            contract_name=contract_name,
            base_experiment_id=base_experiment_id,
            frozen_config=dict(base_result.get("best_candidate") or {}),
            holdout_years=list(LOCKBOX_YEARS),
            structural_inputs=structural_inputs,
            cfg=cfg,
        )
        if lockbox is not None and float(lockbox["quarterly_mean_mae"]) <= float(base_lockbox["quarterly_mean_mae"]) + 1e-6:
            status = "sensitivity_only"
            why = "The true Phase 2 replay improved rolling-origin performance and was competitive on the fixed holdout, but it remains a full-history structural sensitivity rather than a promotion-grade train-only benchmark."
        else:
            status = "sensitivity_only"
            why = "The true Phase 2 replay improved rolling-origin performance, but it did not improve the fixed holdout cleanly; keep it as a structural sensitivity only."
        decision = {
            "status": status,
            "winner_id": str(best_row["config_id"]),
            "why": why,
        }
    return {
        "lane_name": lane_name,
        "contract_name": contract_name,
        "base_experiment_id": base_experiment_id,
        "rows": rows,
        "decision": decision,
        "base_lockbox": base_lockbox,
        "lockbox": lockbox,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase2 True Replay Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Source run: `{payload['source_run_id']}`",
        "- Note: this batch uses the regenerated full-history `phase15` and `phase2` artifacts as a structural sensitivity replay. It is not promotion-grade because the structural tensors are not rebuilt inside each split.",
        "",
    ]
    for lane_name in ("exact", "dense"):
        lane = dict(payload[lane_name])
        lines.extend(
            [
                f"## {lane_name.title()} lane",
                "",
                f"- Base experiment: `{lane['base_experiment_id']}`",
                f"- Contract: `{lane['contract_name']}`",
                f"- Decision: `{lane['decision']['status']}`",
                f"- Winner: `{lane['decision']['winner_id']}`",
                f"- Why: {lane['decision']['why']}",
                "",
                "| Config | Status | Quarterly MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Feature count | Train rows |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in list(lane.get("rows") or []):
            q_mae = "" if row.get("quarterly_mean_mae") is None else f"{float(row['quarterly_mean_mae']):.6f}"
            d_mae = "" if row.get("diagnosed_raw_mae") is None else f"{float(row['diagnosed_raw_mae']):.3f}"
            a_mae = "" if row.get("art_raw_mae") is None else f"{float(row['art_raw_mae']):.3f}"
            f_mae = "" if row.get("flow_raw_mae") is None else f"{float(row['flow_raw_mae']):.3f}"
            feat = "" if row.get("feature_count_mean") is None else f"{float(row['feature_count_mean']):.2f}"
            train = "" if row.get("train_row_count_mean") is None else f"{float(row['train_row_count_mean']):.2f}"
            lines.append(f"| {row['config_id']} | {row['status']} | {q_mae} | {d_mae} | {a_mae} | {f_mae} | {feat} | {train} |")
        if lane.get("lockbox"):
            lockbox = dict(lane["lockbox"])
            lines.extend(
                [
                    "",
                    f"- Base lockbox MAE (`{','.join(str(year) for year in LOCKBOX_YEARS)}`): `{float(dict(lane['base_lockbox'])['quarterly_mean_mae']):.6f}`",
                    f"- Replay lockbox MAE (`{','.join(str(year) for year in LOCKBOX_YEARS)}`): `{float(lockbox['quarterly_mean_mae']):.6f}`",
                    f"- Replay lockbox diagnosed raw MAE: `{float(lockbox['raw_metrics']['diagnosed_raw_mae']):.3f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_tr_v3_phase2_true_replay_batch(
    *,
    run_id: str,
    source_run_id: str,
    archive_run_id: str | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or source_run_id)
    structural_inputs = _load_structural_inputs(str(source_run_id))
    analysis_dir = ensure_dir(phase2_batch.suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")

    exact_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="exact_only",
        experiment_ids=[EXACT_BASE_EXPERIMENT_ID],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_ids=[DENSE_BASE_EXPERIMENT_ID],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    exact_result = phase2_batch._suite_result_map(exact_payload)[EXACT_BASE_EXPERIMENT_ID]
    dense_result = phase2_batch._suite_result_map(dense_payload)[DENSE_BASE_EXPERIMENT_ID]

    exact_lane = _lane_report_payload(
        lane_name="exact",
        contract_name="exact_only",
        base_experiment_id=EXACT_BASE_EXPERIMENT_ID,
        base_result=exact_result,
        archive_run_id=archive_run,
        structural_inputs=structural_inputs,
    )
    dense_lane = _lane_report_payload(
        lane_name="dense",
        contract_name="purged_dense",
        base_experiment_id=DENSE_BASE_EXPERIMENT_ID,
        base_result=dense_result,
        archive_run_id=archive_run,
        structural_inputs=structural_inputs,
    )

    _save_overview(list(exact_lane["rows"]), analysis_dir / "phase2_true_replay_exact_overview.png", title="True Phase2 replay readout | exact lane")
    _save_overview(list(dense_lane["rows"]), analysis_dir / "phase2_true_replay_dense_overview.png", title="True Phase2 replay readout | dense lane")

    for lane_name, lane, base_result in (("exact", exact_lane, exact_result), ("dense", dense_lane, dense_result)):
        if not lane.get("lockbox"):
            continue
        base_dataset, base_candidate, _ = phase2_batch._custom_fixed_holdout_base_candidate(
            archive_run_id=archive_run,
            contract_name=str(lane["contract_name"]),
            base_experiment_id=str(lane["base_experiment_id"]),
            frozen_config=dict(base_result.get("best_candidate") or {}),
            holdout_years=list(LOCKBOX_YEARS),
        )
        target_rows = [
            {
                "quarter": str(row["quarter"]),
                **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
            }
            for row in list(base_dataset.holdout_rows)
        ]
        phase2_batch._save_lockbox_curve(
            analysis_dir / f"phase2_true_replay_{lane_name}_lockbox.png",
            title=f"True Phase2 replay readout | {lane_name} lockbox",
            target_rows=target_rows,
            base_rows=list(base_candidate.get("prediction_rows") or []),
            corrected_rows=list(dict(lane["lockbox"]).get("prediction_rows") or []),
        )

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "source_run_id": str(source_run_id),
        "exact": exact_lane,
        "dense": dense_lane,
        "artifacts": {
            "exact_overview": "phase2_true_replay_exact_overview.png",
            "dense_overview": "phase2_true_replay_dense_overview.png",
            "exact_lockbox_curve": "phase2_true_replay_exact_lockbox.png",
            "dense_lockbox_curve": "phase2_true_replay_dense_lockbox.png",
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_true_replay_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_phase2_true_replay_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a true Phase 2 replay readout on the frozen TR-V3 champions using regenerated phase15/phase2 artifacts.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    args = parser.parse_args()
    run_tr_v3_phase2_true_replay_batch(
        run_id=str(args.run_id),
        source_run_id=str(args.source_run_id),
        archive_run_id=args.archive_run_id,
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
    )


if __name__ == "__main__":
    main()
