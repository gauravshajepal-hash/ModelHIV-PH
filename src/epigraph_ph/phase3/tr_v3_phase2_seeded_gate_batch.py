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

from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3.tr_v3_05_autoresearch import build_annual_anchor_rows
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


FOLD_SPECS: tuple[tuple[str, str, str], ...] = (
    ("2010_2015", "2010-Q1", "2015-Q4"),
    ("2016_2020", "2016-Q1", "2020-Q4"),
    ("2021_2025", "2021-Q1", "2025-Q4"),
)
OUTCOME_CANONICALS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)


def _latest_seeded_batch_run() -> str:
    candidates = sorted((ROOT_DIR / "artifacts" / "runs").glob("tr-v3-phase2-seeded-champions-*"))
    if not candidates:
        raise FileNotFoundError("No Phase 2 seeded champion batch run found.")
    return str(candidates[-1].name)


def _latest_monthly_phase2_run() -> str:
    return seeded._latest_monthly_phase2_run()


def _seeded_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing seeded champion batch report: {path}")
    return payload


def _archive_support_summary(archive_run_id: str) -> dict[str, Any]:
    run_dir = ROOT_DIR / "artifacts" / "runs" / str(archive_run_id)
    harp_dir = run_dir / "harp_archive"
    observed_panel = read_json(harp_dir / "observed_program_panel.json", default={})
    diagnosis_flow = read_json(harp_dir / "diagnosis_flow_points.json", default={})
    historical_panel = read_json(harp_dir / "historical_harp_panel.json", default={})
    multinational_rows = read_json(harp_dir / "multinational_hiv_metric_rows.json", default=[])
    return {
        "archive_run_id": str(archive_run_id),
        "observed_program_row_count": len(list((observed_panel or {}).get("rows") or [])),
        "diagnosis_flow_point_count": len(list((diagnosis_flow or {}).get("points") or [])),
        "historical_panel_row_count": len(list((historical_panel or {}).get("rows") or [])),
        "multinational_metric_row_count": len(list(multinational_rows or [])) if isinstance(multinational_rows, list) else 0,
    }


def _sign(value: float, *, tol: float = 1e-9) -> int:
    if float(value) > tol:
        return 1
    if float(value) < -tol:
        return -1
    return 0


def _terminal_delta_lookup(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in list(payload.get("terminal_delta_rows") or []):
        contract = str(row["contract"])
        scenario = str(row["scenario"])
        for metric_name in seeded.METRIC_PLOT_ORDER:
            lookup[(contract, scenario, metric_name)] = float(row.get(f"{metric_name}_delta") or 0.0)
    return lookup


def _archive_alignment_rows(baseline_payload: dict[str, Any], aligned_payload: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_lookup = _terminal_delta_lookup(baseline_payload)
    aligned_lookup = _terminal_delta_lookup(aligned_payload)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline_lookup) | set(aligned_lookup)):
        baseline_value = float(baseline_lookup.get(key) or 0.0)
        aligned_value = float(aligned_lookup.get(key) or 0.0)
        rows.append(
            {
                "contract": str(key[0]),
                "scenario": str(key[1]),
                "metric": str(key[2]),
                "baseline_terminal_delta": baseline_value,
                "aligned_terminal_delta": aligned_value,
                "delta_of_delta": float(aligned_value - baseline_value),
                "sign_agrees": _sign(baseline_value) == _sign(aligned_value),
            }
        )
    return rows


def _edge_score_rows_for_slice(
    *,
    quarter_axis: list[str],
    quarter_states: np.ndarray,
    block_axis: list[str],
    start_quarter: str,
    end_quarter: str,
) -> list[dict[str, Any]]:
    start_key = suite.quarter_sort_key(str(start_quarter))
    end_key = suite.quarter_sort_key(str(end_quarter))
    rows: list[dict[str, Any]] = []
    values = np.asarray(quarter_states, dtype=np.float64)
    quarter_keys = [suite.quarter_sort_key(value) for value in quarter_axis]
    for source_idx, source_name in enumerate(block_axis):
        for target_idx, target_name in enumerate(block_axis):
            if source_idx == target_idx:
                continue
            source_series: list[float] = []
            target_series: list[float] = []
            for idx in range(1, len(quarter_axis)):
                quarter = str(quarter_axis[idx])
                quarter_key = quarter_keys[idx]
                if quarter_key < start_key or quarter_key > end_key:
                    continue
                source_series.append(float(values[idx - 1, source_idx]))
                target_series.append(float(values[idx, target_idx]))
            if len(source_series) < 6:
                continue
            x = np.asarray(source_series, dtype=np.float64)
            y = np.asarray(target_series, dtype=np.float64)
            design = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            slope = float(coef[1])
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else 0.0
            rows.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "slope": slope,
                    "corr": corr,
                    "score": float(abs(slope * corr)),
                    "sign": _sign(slope),
                }
            )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _retained_edge_stability(
    *,
    phase2_state: seeded.Phase2QuarterState,
) -> dict[str, Any]:
    retained_edges = [
        {
            "source": str(row["source"]),
            "target": str(row["target"]),
            "weight": float(row.get("weight") or 0.0),
            "sign": _sign(float(row.get("weight") or 0.0)),
        }
        for row in list(phase2_state.edge_rows)
    ]
    fold_rows: list[dict[str, Any]] = []
    for fold_label, start_quarter, end_quarter in FOLD_SPECS:
        edge_rows = _edge_score_rows_for_slice(
            quarter_axis=phase2_state.quarter_axis,
            quarter_states=phase2_state.quarter_states,
            block_axis=phase2_state.block_axis,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        )
        edge_lookup = {(str(row["source"]), str(row["target"])): dict(row) for row in edge_rows}
        for retained in retained_edges:
            fold_edge = edge_lookup.get((retained["source"], retained["target"]))
            fold_rows.append(
                {
                    "fold": fold_label,
                    "source": retained["source"],
                    "target": retained["target"],
                    "expected_sign": int(retained["sign"]),
                    "fold_sign": int(fold_edge.get("sign") or 0) if fold_edge else 0,
                    "score": float(fold_edge.get("score") or 0.0) if fold_edge else 0.0,
                    "rank": int(fold_edge.get("rank") or 999) if fold_edge else 999,
                    "survives_top2": bool(fold_edge and int(fold_edge.get("rank") or 999) <= 2 and int(fold_edge.get("sign") or 0) == int(retained["sign"])),
                }
            )
    by_edge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_edge[(str(row["source"]), str(row["target"]))].append(row)
    edge_summary_rows: list[dict[str, Any]] = []
    for (source, target), rows in sorted(by_edge.items()):
        survive_count = sum(1 for row in rows if bool(row["survives_top2"]))
        edge_summary_rows.append(
            {
                "source": source,
                "target": target,
                "fold_count": len(rows),
                "survive_top2_count": survive_count,
                "consistent_sign_count": sum(1 for row in rows if int(row["fold_sign"]) == int(row["expected_sign"])),
                "mean_score": float(np.mean([float(row["score"]) for row in rows])) if rows else 0.0,
                "pass_keep_gate": bool(survive_count >= 2),
            }
        )
    return {
        "fold_rows": fold_rows,
        "edge_summary_rows": edge_summary_rows,
        "keep_gate_passes": all(bool(row["pass_keep_gate"]) for row in edge_summary_rows),
    }


def _unique_quarter_residual_points(
    *,
    result: dict[str, Any],
    allowed_tiers: set[str],
    quarter_feature_map: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = {metric: defaultdict(list) for metric in seeded.METRIC_PLOT_ORDER}
    for split in list(result.get("quarterly_rows") or []):
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            quarter = str(target_row["quarter"])
            if quarter not in quarter_feature_map:
                continue
            for metric_name in seeded.METRIC_PLOT_ORDER:
                if seeded._metric_tier(target_row, metric_name) not in allowed_tiers:
                    continue
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                grouped[metric_name][quarter].append(float(prediction_value) - float(target_value))
    points: dict[str, list[dict[str, Any]]] = {}
    for metric_name, quarter_map in grouped.items():
        metric_points = []
        for quarter, values in quarter_map.items():
            metric_points.append(
                {
                    "quarter": quarter,
                    "features": np.asarray(quarter_feature_map[quarter], dtype=np.float64),
                    "residual": float(np.mean(values)),
                }
            )
        metric_points.sort(key=lambda row: suite.quarter_sort_key(str(row["quarter"])))
        points[metric_name] = metric_points
    return points


def _fit_readout_from_points(points: list[dict[str, Any]], *, feature_dim: int, ridge_penalty: float = 1.0) -> dict[str, Any]:
    if len(points) < max(6, feature_dim + 1):
        return {"beta": np.zeros(feature_dim, dtype=np.float64), "scale": 0.0, "cap_abs": 0.0}
    design = np.asarray([row["features"] for row in points], dtype=np.float64)
    target = np.asarray([float(row["residual"]) for row in points], dtype=np.float64)
    centered = target - float(np.mean(target))
    lhs = design.T @ design
    rhs = design.T @ centered
    beta = np.linalg.solve(lhs + (float(ridge_penalty) * np.eye(feature_dim, dtype=np.float64)), rhs)
    raw = design @ beta
    raw_p90 = float(np.quantile(np.abs(raw), 0.9)) if raw.size else 0.0
    residual_p90 = float(np.quantile(np.abs(centered), 0.9)) if centered.size else 0.0
    scale = float(np.clip(0.35 * (residual_p90 / max(raw_p90, 1e-6)), 0.0, 0.5))
    return {"beta": beta, "scale": scale, "cap_abs": residual_p90}


def _bootstrap_readout_stability(
    *,
    points_by_metric: dict[str, list[dict[str, Any]]],
    reference_readout: dict[str, Any],
    bootstrap_draws: int = 200,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260418)
    feature_names = list(reference_readout.get("feature_names") or [])
    feature_dim = len(feature_names)
    sign_rows: list[dict[str, Any]] = []
    for metric_name in seeded.METRIC_PLOT_ORDER:
        metric_points = list(points_by_metric.get(metric_name) or [])
        ref_metric = dict((reference_readout.get("metrics") or {}).get(metric_name) or {})
        ref_beta = np.asarray(ref_metric.get("beta") or [0.0] * feature_dim, dtype=np.float64)
        if not metric_points:
            for feature_name in feature_names:
                sign_rows.append(
                    {
                        "metric": metric_name,
                        "feature": feature_name,
                        "sign_stability": 0.0,
                        "nonzero_reference": False,
                    }
                )
            continue
        draw_betas = []
        for _ in range(int(bootstrap_draws)):
            indices = rng.integers(0, len(metric_points), len(metric_points))
            sampled = [metric_points[int(idx)] for idx in indices]
            fitted = _fit_readout_from_points(sampled, feature_dim=feature_dim)
            draw_betas.append(np.asarray(fitted["beta"], dtype=np.float64))
        beta_draws = np.asarray(draw_betas, dtype=np.float64)
        for feature_idx, feature_name in enumerate(feature_names):
            ref_sign = _sign(ref_beta[feature_idx])
            if ref_sign == 0:
                stability = 0.0
            else:
                stability = float(np.mean([_sign(value) == ref_sign for value in beta_draws[:, feature_idx]]))
            sign_rows.append(
                {
                    "metric": metric_name,
                    "feature": feature_name,
                    "sign_stability": float(stability),
                    "nonzero_reference": bool(ref_sign != 0),
                }
            )
    active_rows = [row for row in sign_rows if bool(row["nonzero_reference"])]
    return {
        "sign_rows": sign_rows,
        "active_feature_count": len(active_rows),
        "mean_active_sign_stability": float(np.mean([float(row["sign_stability"]) for row in active_rows])) if active_rows else 0.0,
        "keep_gate_passes": bool(active_rows) and float(np.mean([float(row["sign_stability"]) for row in active_rows])) >= 0.7,
    }


def _boundedness_audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    violation_count = 0
    for contract_name, contract_payload in dict(payload.get("contracts") or {}).items():
        readout_metrics = dict((contract_payload.get("readout") or {}).get("metrics") or {})
        for scenario_name, scenario_rows in dict(contract_payload.get("scenario_rows") or {}).items():
            max_relative = {metric: 0.0 for metric in seeded.METRIC_PLOT_ORDER}
            scenario_violation_count = 0
            base_rows = list(contract_payload.get("base_forecast_rows") or [])
            for base_row, scenario_row in zip(base_rows, list(scenario_rows or []), strict=False):
                diagnosed = float(scenario_row.get("diagnosed_plhiv") or 0.0)
                art = float(scenario_row.get("alive_on_art") or 0.0)
                suppressed = float(scenario_row.get("virally_suppressed") or 0.0) if scenario_row.get("virally_suppressed") is not None else None
                if diagnosed < 0.0 or art < 0.0 or float(scenario_row.get("new_diagnosed_cases_period") or 0.0) < 0.0:
                    scenario_violation_count += 1
                if art > diagnosed + 1e-6:
                    scenario_violation_count += 1
                if suppressed is not None and suppressed > art + 1e-6:
                    scenario_violation_count += 1
                for metric_name in seeded.METRIC_PLOT_ORDER:
                    base_value = float(base_row.get(metric_name) or 0.0)
                    delta_value = float(scenario_row.get(f"{metric_name}_scenario_delta") or 0.0)
                    rel = abs(delta_value) / max(abs(base_value), 1.0)
                    max_relative[metric_name] = max(float(max_relative[metric_name]), float(rel))
                    cap_abs = float(dict(readout_metrics.get(metric_name) or {}).get("cap_abs") or 0.0)
                    if abs(delta_value) > cap_abs + 1e-6:
                        scenario_violation_count += 1
            rows.append(
                {
                    "contract": str(contract_name),
                    "scenario": str(scenario_name),
                    "violation_count": int(scenario_violation_count),
                    "diagnosed_max_relative_delta": float(max_relative["diagnosed_plhiv"]),
                    "art_max_relative_delta": float(max_relative["alive_on_art"]),
                    "flow_max_relative_delta": float(max_relative["new_diagnosed_cases_period"]),
                    "bounded_keep": bool(
                        scenario_violation_count == 0
                        and max_relative["diagnosed_plhiv"] <= 0.1
                        and max_relative["alive_on_art"] <= 0.1
                        and max_relative["new_diagnosed_cases_period"] <= 0.2
                    ),
                }
            )
            violation_count += int(scenario_violation_count)
    return {
        "rows": rows,
        "total_violation_count": int(violation_count),
        "keep_gate_passes": all(bool(row["bounded_keep"]) for row in rows),
    }


def _load_block_loading_rows(monthly_run_id: str) -> list[dict[str, Any]]:
    path = ROOT_DIR / "artifacts" / "runs" / str(monthly_run_id) / "phase15" / "national_block_loadings.json"
    payload = read_json(path, default={})
    rows = list((payload or {}).get("rows") or [])
    return [dict(row) for row in rows]


def _summarize_outcome_circularity_rows(
    *,
    loading_rows: list[dict[str, Any]],
    outcome_canonical_names: set[str],
) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loading_rows:
        by_block[str(row.get("block_id") or "")].append(dict(row))
    summary_rows: list[dict[str, Any]] = []
    for block_id, rows in sorted(by_block.items()):
        total_abs_loading = float(sum(abs(float(row.get("loading") or 0.0)) for row in rows))
        outcome_rows = [row for row in rows if str(row.get("canonical_name") or "") in outcome_canonical_names]
        outcome_abs_loading = float(sum(abs(float(row.get("loading") or 0.0)) for row in outcome_rows))
        summary_rows.append(
            {
                "block_id": block_id,
                "indicator_count": int(len(rows)),
                "outcome_indicator_count": int(len(outcome_rows)),
                "outcome_canonical_names": [str(row.get("canonical_name") or "") for row in outcome_rows],
                "outcome_loading_share": float(outcome_abs_loading / max(total_abs_loading, 1e-9)),
                "outcome_support_count": int(sum(int(row.get("direct_indicator_count") or 0) for row in outcome_rows)),
                "total_support_count": int(sum(int(row.get("direct_indicator_count") or 0) for row in rows)),
            }
        )
    return summary_rows


def _quarter_outcome_matrix(
    *,
    archive_run_id: str,
    quarter_axis: list[str],
) -> tuple[np.ndarray, dict[str, list[str]]]:
    observation_rows, _, _ = seeded.champion_forecast._contract_payload(str(archive_run_id), "dense_train_observed_score")
    grouped: dict[str, dict[str, list[float]]] = {metric: defaultdict(list) for metric in OUTCOME_CANONICALS}
    for row in list(observation_rows or []):
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        for metric_name in OUTCOME_CANONICALS:
            value = row.get(metric_name)
            if value is not None:
                grouped[metric_name][quarter].append(float(value))
    matrix = np.zeros((len(quarter_axis), len(OUTCOME_CANONICALS)), dtype=np.float64)
    availability: dict[str, list[str]] = {metric: [] for metric in OUTCOME_CANONICALS}
    for metric_idx, metric_name in enumerate(OUTCOME_CANONICALS):
        raw = np.full(len(quarter_axis), np.nan, dtype=np.float64)
        for idx, quarter in enumerate(quarter_axis):
            values = list(grouped[metric_name].get(quarter) or [])
            if values:
                raw[idx] = float(np.mean(values))
                availability[metric_name].append(quarter)
        if np.isnan(raw).all():
            continue
        mean_value = float(np.nanmean(raw))
        raw = np.where(np.isnan(raw), mean_value, raw)
        std_value = float(np.nanstd(raw))
        matrix[:, metric_idx] = (raw - mean_value) / max(std_value, 1e-6)
    return matrix, availability


def _orthogonalize_series_against_matrix(
    *,
    series: np.ndarray,
    design: np.ndarray,
    ridge_penalty: float = 1e-6,
) -> dict[str, Any]:
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    x = np.asarray(design, dtype=np.float64)
    if y.shape[0] != x.shape[0]:
        raise ValueError("series and design must have the same row count.")
    if x.size == 0 or x.shape[0] < max(6, x.shape[1] + 2):
        return {
            "orthogonalized": y.copy(),
            "beta": np.zeros(x.shape[1], dtype=np.float64),
            "intercept": float(np.mean(y)) if y.size else 0.0,
            "r2": 0.0,
            "count": int(y.shape[0]),
        }
    centered_x = x - np.mean(x, axis=0, keepdims=True)
    centered_y = y - float(np.mean(y))
    lhs = centered_x.T @ centered_x
    rhs = centered_x.T @ centered_y
    beta = np.linalg.solve(lhs + (float(ridge_penalty) * np.eye(centered_x.shape[1], dtype=np.float64)), rhs)
    fitted = float(np.mean(y)) + (centered_x @ beta)
    orthogonalized = (y - fitted) + float(np.mean(y))
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return {
        "orthogonalized": orthogonalized,
        "beta": beta,
        "intercept": float(np.mean(y)),
        "r2": 0.0 if ss_tot <= 1e-9 else float(max(0.0, 1.0 - (ss_res / ss_tot))),
        "count": int(y.shape[0]),
    }


def _orthogonalize_phase2_state_against_outcomes(
    *,
    phase2_state: seeded.Phase2QuarterState,
    archive_run_id: str,
    loading_rows: list[dict[str, Any]],
) -> tuple[seeded.Phase2QuarterState, list[dict[str, Any]]]:
    quarter_outcomes, availability = _quarter_outcome_matrix(archive_run_id=str(archive_run_id), quarter_axis=phase2_state.quarter_axis)
    by_block: dict[str, list[str]] = defaultdict(list)
    for row in loading_rows:
        canonical_name = str(row.get("canonical_name") or "")
        block_id = str(row.get("block_id") or "")
        if canonical_name in OUTCOME_CANONICALS:
            by_block[block_id].append(canonical_name)
    block_index = {name: idx for idx, name in enumerate(phase2_state.block_axis)}
    metric_index = {name: idx for idx, name in enumerate(OUTCOME_CANONICALS)}
    new_states = np.asarray(phase2_state.quarter_states, dtype=np.float64).copy()
    summary_rows: list[dict[str, Any]] = []
    for block_name in phase2_state.block_axis:
        metric_names = sorted(set(by_block.get(block_name) or []))
        if not metric_names:
            summary_rows.append(
                {
                    "block_id": block_name,
                    "metric_names": [],
                    "metric_count": 0,
                    "r2": 0.0,
                    "row_count": int(len(phase2_state.quarter_axis)),
                }
            )
            continue
        design = np.column_stack([quarter_outcomes[:, metric_index[name]] for name in metric_names])
        fit = _orthogonalize_series_against_matrix(series=new_states[:, block_index[block_name]], design=design)
        new_states[:, block_index[block_name]] = np.asarray(fit["orthogonalized"], dtype=np.float64)
        summary_rows.append(
            {
                "block_id": block_name,
                "metric_names": metric_names,
                "metric_count": int(len(metric_names)),
                "r2": float(fit["r2"]),
                "row_count": int(fit["count"]),
                "available_quarters": {name: len(list(availability.get(name) or [])) for name in metric_names},
            }
        )
    edge_scores = _edge_score_rows_for_slice(
        quarter_axis=phase2_state.quarter_axis,
        quarter_states=new_states,
        block_axis=phase2_state.block_axis,
        start_quarter=phase2_state.quarter_axis[1],
        end_quarter=phase2_state.quarter_axis[-1],
    )
    score_lookup = {(str(row["source"]), str(row["target"])): dict(row) for row in edge_scores}
    edge_rows = []
    for row in list(phase2_state.edge_rows):
        key = (str(row.get("source") or ""), str(row.get("target") or ""))
        score_row = score_lookup.get(key)
        edge_rows.append(
            {
                **dict(row),
                "weight": float(score_row.get("slope") if score_row else row.get("weight") or 0.0),
                "ablation_score": float(score_row.get("score") or 0.0) if score_row else 0.0,
            }
        )
    return (
        seeded.Phase2QuarterState(
            source_run_id=f"{phase2_state.source_run_id}:outcome_orthogonalized",
            month_axis=list(phase2_state.month_axis),
            quarter_axis=list(phase2_state.quarter_axis),
            block_axis=list(phase2_state.block_axis),
            quarter_states=new_states,
            edge_rows=edge_rows,
        ),
        summary_rows,
    )


def _seeded_payload_from_phase2_state(
    *,
    archive_run_id: str,
    phase2_state: seeded.Phase2QuarterState,
    forecast_horizon_quarters: int,
) -> dict[str, Any]:
    annual_rows = build_annual_anchor_rows(str(archive_run_id))
    kernel = seeded._fit_structural_kernel(phase2_state.quarter_states, phase2_state.block_axis, phase2_state.edge_rows)
    block_scales = seeded._recent_block_scales(phase2_state.quarter_states)
    last_state = np.asarray(phase2_state.quarter_states[-1, :], dtype=np.float64)
    scenario_paths = {
        scenario_name: seeded._propagate_structural_states(
            kernel=kernel,
            last_state=last_state,
            steps=int(forecast_horizon_quarters),
            shock_matrix=seeded._scenario_shock_matrix(
                scenario_name,
                steps=int(forecast_horizon_quarters),
                block_axis=phase2_state.block_axis,
                block_scales=block_scales,
            ),
        )
        for scenario_name in seeded.SCENARIO_ORDER
    }
    history_feature_matrix = seeded._feature_matrix(phase2_state.quarter_states)
    feature_mean = history_feature_matrix.mean(axis=0)
    feature_std = history_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    quarter_feature_map = {
        quarter: (history_feature_matrix[idx, :] - feature_mean) / feature_std
        for idx, quarter in enumerate(phase2_state.quarter_axis)
    }
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    contract_payloads: dict[str, Any] = {}
    terminal_delta_rows: list[dict[str, Any]] = []
    for contract_name, config in seeded.WINNER_CONFIGS.items():
        result = seeded._collect_contract_result(
            archive_run_id=str(archive_run_id),
            contract_name=str(config["suite_contract"]),
            experiment_id=str(config["winner_id"]),
            quarterly_start_year=2010,
            quarterly_end_year=2025,
            quarterly_min_train_years=3,
            annual_start_year=2010,
            annual_end_year=2024,
            annual_min_train_years=5,
            horizon_years=1,
        )
        readout = seeded._fit_residual_readout(
            result=result,
            allowed_tiers=set(config["allowed_tiers"]),
            quarter_feature_map=quarter_feature_map,
            feature_names=seeded._feature_names(phase2_state.block_axis),
        )
        observation_rows, _, _ = seeded.champion_forecast._contract_payload(str(archive_run_id), str(config["forecast_contract"]))
        latest_quarter = max((str(row["quarter"]) for row in observation_rows), key=suite.quarter_sort_key)
        forecast_quarters = seeded.champion_forecast._future_quarters(latest_quarter, int(forecast_horizon_quarters))
        forecast_candidate = seeded.champion_forecast._forecast_future_rows(
            observation_rows,
            annual_rows=annual_rows,
            spec=spec_map[str(config["winner_id"])],
            best_candidate=dict(result["best_candidate"]),
            forecast_quarters=forecast_quarters,
        )
        base_rows = [dict(row) for row in list(forecast_candidate.get("prediction_rows") or [])]
        base_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], scenario_paths["status_quo"]]))[1:, :] - feature_mean) / feature_std
        scenario_rows: dict[str, list[dict[str, Any]]] = {"status_quo": [dict(row) for row in base_rows]}
        for scenario_name in seeded.SCENARIO_ORDER[1:]:
            scenario_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], scenario_paths[scenario_name]]))[1:, :] - feature_mean) / feature_std
            scenario_rows[scenario_name] = seeded._apply_scenario_to_forecast(
                base_rows=base_rows,
                base_features=base_feature_path,
                scenario_features=scenario_feature_path,
                readout=readout,
            )
        for scenario_name in seeded.SCENARIO_ORDER[1:]:
            final_row = dict(scenario_rows[scenario_name][-1])
            terminal_delta_rows.append(
                {
                    "contract": contract_name,
                    "scenario": scenario_name,
                    "diagnosed_plhiv_delta": float(final_row.get("diagnosed_plhiv_scenario_delta") or 0.0),
                    "alive_on_art_delta": float(final_row.get("alive_on_art_scenario_delta") or 0.0),
                    "new_diagnosed_cases_period_delta": float(final_row.get("new_diagnosed_cases_period_scenario_delta") or 0.0),
                }
            )
        contract_payloads[contract_name] = {
            "winner_id": str(config["winner_id"]),
            "readout": readout,
            "base_forecast_rows": base_rows,
            "scenario_rows": scenario_rows,
        }
    return {
        "archive_run_id": str(archive_run_id),
        "phase2_seed": {
            "block_axis": list(phase2_state.block_axis),
            "quarter_axis": list(phase2_state.quarter_axis),
            "edge_rows": [dict(row) for row in phase2_state.edge_rows],
            "edge_count": int(len(phase2_state.edge_rows)),
        },
        "contracts": contract_payloads,
        "terminal_delta_rows": terminal_delta_rows,
    }


def _plot_archive_support_compare(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = [
        "observed_program_row_count",
        "diagnosis_flow_point_count",
        "historical_panel_row_count",
        "multinational_metric_row_count",
    ]
    labels = [str(row["archive_run_id"]) for row in rows]
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, row in enumerate(rows):
        values = [float(row[metric]) for metric in metrics]
        ax.bar(x + ((idx - 0.5) * width), values, width=width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", " ") for metric in metrics], rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Archive support comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_archive_alignment_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    matrix = np.asarray([[float(row["delta_of_delta"])] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, max(4.0, len(labels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks([0])
    ax.set_xticklabels(["aligned - baseline"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Archive-alignment terminal delta change")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_edge_stability_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    edge_labels = [f"{row['source']}→{row['target']}" for row in rows]
    matrix = np.asarray([[float(row["survive_top2_count"]), float(row["consistent_sign_count"]), float(row["mean_score"])] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, max(4.0, len(edge_labels) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["top2 folds", "sign-consistent folds", "mean score"], rotation=20, ha="right")
    ax.set_yticks(range(len(edge_labels)))
    ax.set_yticklabels(edge_labels)
    ax.set_title("Retained-edge blocked stability")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_readout_stability_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    active_rows = [row for row in rows if bool(row["nonzero_reference"])]
    if not active_rows:
        suite._plot_placeholder(path, title="Readout bootstrap stability", body="No active reference coefficients.")
        return
    row_labels = [f"{row['metric']}:{row['feature']}" for row in active_rows]
    matrix = np.asarray([[float(row["sign_stability"])] for row in active_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, max(4.0, len(row_labels) * 0.32)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks([0])
    ax.set_xticklabels(["sign stability"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Readout bootstrap sign stability")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_boundedness_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['contract']}:{row['scenario']}" for row in rows]
    diagnosed = [float(row["diagnosed_max_relative_delta"]) for row in rows]
    art = [float(row["art_max_relative_delta"]) for row in rows]
    flow = [float(row["flow_max_relative_delta"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.45)))
    ax.barh(y - 0.25, diagnosed, height=0.22, label="Diagnosed")
    ax.barh(y, art, height=0.22, label="ART")
    ax.barh(y + 0.25, flow, height=0.22, label="Flow")
    ax.axvline(0.1, color="#aa3333", linestyle="--", linewidth=1.0, label="Stock cap")
    ax.axvline(0.2, color="#dd8844", linestyle=":", linewidth=1.0, label="Flow cap")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Max absolute relative delta")
    ax.set_title("Scenario boundedness audit")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_outcome_circularity_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["block_id"]) for row in rows]
    shares = [float(row["outcome_loading_share"]) for row in rows]
    counts = [float(row["outcome_indicator_count"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, max(4.0, len(labels) * 0.6)))
    ax.barh(y, shares, color="#4477aa", alpha=0.85, label="Outcome loading share")
    ax.scatter(counts, y, color="#cc3311", s=45, label="Outcome indicator count")
    ax.axvline(0.25, color="#aa3333", linestyle="--", linewidth=1.0, label="share 0.25")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share / count")
    ax.set_title("Outcome-circularity pressure by retained block")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_circularity_ablation_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    matrix = np.asarray([[float(row["delta_of_delta"])] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, max(4.0, len(labels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks([0])
    ax.set_xticklabels(["orthogonalized - aligned"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Outcome-circularity ablation terminal delta change")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    gate_summary = dict(payload.get("gate_summary") or {})
    lines = [
        "# TR-V3 Phase 2 Seeded Gate Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Monthly Phase 2 run: `{payload['monthly_phase2_run_id']}`",
        f"- Baseline seeded run: `{payload['baseline_seeded_run_id']}`",
        f"- Archive-aligned seeded run: `{payload['aligned_seeded_run_id']}`",
        f"- Baseline archive: `{payload['baseline_archive_run_id']}`",
        f"- Aligned archive: `{payload['aligned_archive_run_id']}`",
        "",
        "## Gate summary",
        "",
        f"- `EXP-P2-GATE-00` archive alignment: `{gate_summary.get('gate_00_status', '')}`",
        f"- `EXP-P2-GATE-01` edge stability: `{gate_summary.get('gate_01_status', '')}`",
        f"- `EXP-P2-GATE-02` readout stability: `{gate_summary.get('gate_02_status', '')}`",
        f"- `EXP-P2-GATE-03` boundedness: `{gate_summary.get('gate_03_status', '')}`",
        f"- `EXP-P2-GATE-04` outcome circularity: `{gate_summary.get('gate_04_status', '')}`",
        f"- Overall decision: `{gate_summary.get('overall_decision', '')}`",
        "",
        "## Archive support",
        "",
        "| Archive run | Observed rows | Diagnosis-flow points | Historical-panel rows | Multinational rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("archive_support_rows") or []):
        lines.append(
            f"| {row['archive_run_id']} | `{int(row['observed_program_row_count'])}` | `{int(row['diagnosis_flow_point_count'])}` | "
            f"`{int(row['historical_panel_row_count'])}` | `{int(row['multinational_metric_row_count'])}` |"
        )
    lines.extend(
        [
            "",
            "## Archive alignment",
            "",
            f"- Terminal-delta sign agreement: `{float(payload['archive_alignment']['sign_agreement_rate']):.3f}`",
            f"- Mean absolute delta-of-delta: `{float(payload['archive_alignment']['mean_abs_delta_of_delta']):.3f}`",
            "",
            "## Edge stability",
            "",
            "| Edge | Top-2 folds | Sign-consistent folds | Mean score | Keep |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in list(payload.get("edge_stability", {}).get("edge_summary_rows") or []):
        lines.append(
            f"| `{row['source']} -> {row['target']}` | `{int(row['survive_top2_count'])}` | `{int(row['consistent_sign_count'])}` | "
            f"`{float(row['mean_score']):.4f}` | `{bool(row['pass_keep_gate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Readout stability",
            "",
            f"- Active coefficient count: `{int(payload['readout_stability']['active_feature_count'])}`",
            f"- Mean active sign stability: `{float(payload['readout_stability']['mean_active_sign_stability']):.3f}`",
            "",
            "## Boundedness",
            "",
            f"- Total violation count: `{int(payload['boundedness']['total_violation_count'])}`",
            "",
            "| Contract | Scenario | Violations | Diagnosed rel cap | ART rel cap | Flow rel cap | Keep |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in list(payload.get("boundedness", {}).get("rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | `{int(row['violation_count'])}` | "
            f"`{float(row['diagnosed_max_relative_delta']):.3f}` | `{float(row['art_max_relative_delta']):.3f}` | "
            f"`{float(row['flow_max_relative_delta']):.3f}` | `{bool(row['bounded_keep'])}` |"
        )
    lines.extend(
        [
            "",
            "## Outcome circularity",
            "",
            "| Block | Outcome indicators | Outcome loading share | Outcome support count | Outcome metrics |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in list(payload.get("outcome_circularity", {}).get("loading_rows") or []):
        lines.append(
            f"| `{row['block_id']}` | `{int(row['outcome_indicator_count'])}` | `{float(row['outcome_loading_share']):.3f}` | "
            f"`{int(row['outcome_support_count'])}` | `{', '.join(list(row.get('outcome_canonical_names') or []))}` |"
        )
    ablation = dict(payload.get("outcome_circularity", {}).get("ablation_summary") or {})
    lines.extend(
        [
            "",
            f"- Ablation sign agreement: `{float(ablation.get('sign_agreement_rate') or 0.0):.3f}`",
            f"- Ablation mean absolute delta-of-delta: `{float(ablation.get('mean_abs_delta_of_delta') or 0.0):.3f}`",
            f"- Ablation mean absolute ratio: `{float(ablation.get('mean_abs_ratio') or 0.0):.3f}`",
            "",
            "| Block | Orthogonalized metrics | R² removed | Row count |",
            "|---|---|---:|---:|",
        ]
    )
    for row in list(payload.get("outcome_circularity", {}).get("orthogonalization_rows") or []):
        lines.append(
            f"| `{row['block_id']}` | `{', '.join(list(row.get('metric_names') or []))}` | "
            f"`{float(row.get('r2') or 0.0):.3f}` | `{int(row.get('row_count') or 0)}` |"
        )
    lines.extend(["", "## Graphs", ""])
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run_tr_v3_phase2_seeded_gate_batch(
    *,
    run_id: str,
    monthly_phase2_run_id: str | None = None,
    baseline_seeded_run_id: str | None = None,
    aligned_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
) -> dict[str, Any]:
    monthly_run = str(monthly_phase2_run_id or _latest_monthly_phase2_run())
    baseline_seeded_run = str(baseline_seeded_run_id or _latest_seeded_batch_run())
    baseline_payload = _seeded_report(baseline_seeded_run)
    baseline_archive_run = str(baseline_payload["archive_run_id"])
    aligned_archive_run = str(aligned_archive_run_id or monthly_run)

    aligned_seeded_run = f"{run_id}-aligned-seeded"
    seeded.run_tr_v3_phase2_seeded_champion_batch(
        run_id=aligned_seeded_run,
        archive_run_id=aligned_archive_run,
        monthly_phase2_run_id=monthly_run,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )
    aligned_payload = _seeded_report(aligned_seeded_run)

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    archive_support_rows = [
        _archive_support_summary(baseline_archive_run),
        _archive_support_summary(aligned_archive_run),
    ]
    alignment_rows = _archive_alignment_rows(baseline_payload, aligned_payload)
    sign_agreement_rate = float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in alignment_rows])) if alignment_rows else 0.0
    mean_abs_delta_of_delta = float(np.mean([abs(float(row["delta_of_delta"])) for row in alignment_rows])) if alignment_rows else 0.0

    phase2_state = seeded._load_phase2_quarter_state(monthly_run)
    edge_stability = _retained_edge_stability(phase2_state=phase2_state)

    history_feature_matrix = seeded._feature_matrix(phase2_state.quarter_states)
    feature_mean = history_feature_matrix.mean(axis=0)
    feature_std = history_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    quarter_feature_map = {
        quarter: (history_feature_matrix[idx, :] - feature_mean) / feature_std
        for idx, quarter in enumerate(phase2_state.quarter_axis)
    }

    readout_stability_rows = []
    active_count = 0
    active_means = []
    for contract_name, config in seeded.WINNER_CONFIGS.items():
        result = seeded._collect_contract_result(
            archive_run_id=aligned_archive_run,
            contract_name=str(config["suite_contract"]),
            experiment_id=str(config["winner_id"]),
            quarterly_start_year=2010,
            quarterly_end_year=2025,
            quarterly_min_train_years=3,
            annual_start_year=2010,
            annual_end_year=2024,
            annual_min_train_years=5,
            horizon_years=1,
        )
        contract_readout = dict(dict(aligned_payload["contracts"][contract_name]).get("readout") or {})
        unique_points = _unique_quarter_residual_points(
            result=result,
            allowed_tiers=set(config["allowed_tiers"]),
            quarter_feature_map=quarter_feature_map,
        )
        stability = _bootstrap_readout_stability(points_by_metric=unique_points, reference_readout=contract_readout)
        for row in list(stability["sign_rows"]):
            readout_stability_rows.append({**row, "contract": contract_name})
        active_count += int(stability["active_feature_count"])
        if stability["active_feature_count"]:
            active_means.append(float(stability["mean_active_sign_stability"]))
    readout_stability = {
        "sign_rows": readout_stability_rows,
        "active_feature_count": int(active_count),
        "mean_active_sign_stability": float(np.mean(active_means)) if active_means else 0.0,
        "keep_gate_passes": bool(active_means) and float(np.mean(active_means)) >= 0.7,
    }

    boundedness = _boundedness_audit(aligned_payload)

    loading_rows = _load_block_loading_rows(monthly_run)
    outcome_circularity_rows = _summarize_outcome_circularity_rows(
        loading_rows=loading_rows,
        outcome_canonical_names=set(OUTCOME_CANONICALS),
    )
    orthogonalized_state, orthogonalization_rows = _orthogonalize_phase2_state_against_outcomes(
        phase2_state=phase2_state,
        archive_run_id=aligned_archive_run,
        loading_rows=loading_rows,
    )
    orthogonalized_payload = _seeded_payload_from_phase2_state(
        archive_run_id=aligned_archive_run,
        phase2_state=orthogonalized_state,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )
    circularity_alignment_rows = _archive_alignment_rows(aligned_payload, orthogonalized_payload)
    circularity_sign_agreement = (
        float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in circularity_alignment_rows]))
        if circularity_alignment_rows
        else 0.0
    )
    circularity_mean_abs_delta = (
        float(np.mean([abs(float(row["delta_of_delta"])) for row in circularity_alignment_rows]))
        if circularity_alignment_rows
        else 0.0
    )
    abs_ratios = []
    for row in circularity_alignment_rows:
        aligned_value = 0.0
        orthogonalized_value = 0.0
        if row["metric"] == "diagnosed_plhiv":
            aligned_value = float(row["baseline_terminal_delta"])
            orthogonalized_value = float(row["aligned_terminal_delta"])
        elif row["metric"] == "alive_on_art":
            aligned_value = float(row["baseline_terminal_delta"])
            orthogonalized_value = float(row["aligned_terminal_delta"])
        else:
            aligned_value = float(row["baseline_terminal_delta"])
            orthogonalized_value = float(row["aligned_terminal_delta"])
        abs_ratios.append(abs(orthogonalized_value) / max(abs(aligned_value), 1.0))
    circularity_mean_abs_ratio = float(np.mean(abs_ratios)) if abs_ratios else 0.0

    support_graph = analysis_dir / "archive_support_compare.png"
    _plot_archive_support_compare(archive_support_rows, support_graph)
    alignment_graph = analysis_dir / "archive_alignment_terminal_delta_compare.png"
    _plot_archive_alignment_heatmap(alignment_rows, alignment_graph)
    edge_graph = analysis_dir / "edge_stability_heatmap.png"
    _plot_edge_stability_heatmap(list(edge_stability["edge_summary_rows"]), edge_graph)
    readout_graph = analysis_dir / "readout_stability_heatmap.png"
    _plot_readout_stability_heatmap(readout_stability_rows, readout_graph)
    boundedness_graph = analysis_dir / "boundedness_summary.png"
    _plot_boundedness_bars(list(boundedness["rows"]), boundedness_graph)
    circularity_graph = analysis_dir / "outcome_circularity_bars.png"
    _plot_outcome_circularity_bars(outcome_circularity_rows, circularity_graph)
    circularity_ablation_graph = analysis_dir / "outcome_circularity_ablation_compare.png"
    _plot_circularity_ablation_heatmap(circularity_alignment_rows, circularity_ablation_graph)

    gate_summary = {
        "gate_00_status": "keep" if sign_agreement_rate >= 0.75 else "revisit_archive_alignment",
        "gate_01_status": "keep" if bool(edge_stability["keep_gate_passes"]) else "revert_edge_instability",
        "gate_02_status": "keep" if bool(readout_stability["keep_gate_passes"]) else "revert_readout_instability",
        "gate_03_status": "keep" if bool(boundedness["keep_gate_passes"]) else "revert_boundedness_failure",
        "gate_04_status": "keep" if circularity_sign_agreement >= 0.75 and 0.4 <= circularity_mean_abs_ratio <= 2.0 else "revert_outcome_circularity",
    }
    if all(value == "keep" for value in gate_summary.values()):
        gate_summary["overall_decision"] = "proceed_to_admissibility_and_scenario_expansion"
    else:
        gate_summary["overall_decision"] = "stop_before_scenario_expansion"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "monthly_phase2_run_id": monthly_run,
        "baseline_seeded_run_id": baseline_seeded_run,
        "aligned_seeded_run_id": aligned_seeded_run,
        "baseline_archive_run_id": baseline_archive_run,
        "aligned_archive_run_id": aligned_archive_run,
        "archive_support_rows": archive_support_rows,
        "archive_alignment": {
            "rows": alignment_rows,
            "sign_agreement_rate": float(sign_agreement_rate),
            "mean_abs_delta_of_delta": float(mean_abs_delta_of_delta),
        },
        "edge_stability": edge_stability,
        "readout_stability": readout_stability,
        "boundedness": boundedness,
        "outcome_circularity": {
            "loading_rows": outcome_circularity_rows,
            "orthogonalization_rows": orthogonalization_rows,
            "ablation_summary": {
                "rows": circularity_alignment_rows,
                "sign_agreement_rate": float(circularity_sign_agreement),
                "mean_abs_delta_of_delta": float(circularity_mean_abs_delta),
                "mean_abs_ratio": float(circularity_mean_abs_ratio),
            },
        },
        "gate_summary": gate_summary,
        "artifacts": {
            "archive_support_graph": support_graph.name,
            "archive_alignment_graph": alignment_graph.name,
            "edge_stability_graph": edge_graph.name,
            "readout_stability_graph": readout_graph.name,
            "boundedness_graph": boundedness_graph.name,
            "outcome_circularity_graph": circularity_graph.name,
            "outcome_circularity_ablation_graph": circularity_ablation_graph.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_seeded_gate_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_seeded_gate_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first Phase 2 seeded champion falsification/stability gates.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    parser.add_argument("--baseline-seeded-run-id", default=None)
    parser.add_argument("--aligned-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_seeded_gate_batch(
        run_id=str(args.run_id),
        monthly_phase2_run_id=args.monthly_phase2_run_id,
        baseline_seeded_run_id=args.baseline_seeded_run_id,
        aligned_archive_run_id=args.aligned_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
