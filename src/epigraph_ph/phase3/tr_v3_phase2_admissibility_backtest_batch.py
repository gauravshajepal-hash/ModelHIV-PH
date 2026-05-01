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
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_phase2_seeded_gate_batch as seeded_gate
from epigraph_ph.phase3 import tr_v3_phase2_two_block_kernel_batch as two_block
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


ACTIVE_SCENARIOS: tuple[str, ...] = two_block.ACTIVE_SCENARIOS
PLACEBO_SCENARIO_MAP: dict[str, str] = {
    "disruption_recovery": "disruption_recovery_placebo",
    "mobility_spike": "mobility_spike_placebo",
}
PLACEBO_SCENARIOS: tuple[str, ...] = tuple(PLACEBO_SCENARIO_MAP.values())
ALL_REPLAY_SCENARIOS: tuple[str, ...] = ACTIVE_SCENARIOS + PLACEBO_SCENARIOS
REPLAY_KEEP_DIRECTIONAL_RATE = 0.75


def _read_report(path: Path) -> dict[str, Any]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Missing report payload: {path}")
    return payload


def _seeded_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json")


def _alignment_report(run_id: str) -> dict[str, Any]:
    return _read_report(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.json")


def _quarter_sort(quarter: str) -> tuple[int, int]:
    return suite.quarter_sort_key(str(quarter))


def _previous_quarter(quarter: str) -> str:
    year_text, quarter_text = str(quarter).split("-Q", 1)
    year = int(year_text)
    q = int(quarter_text)
    if q <= 1:
        return f"{year - 1:04d}-Q4"
    return f"{year:04d}-Q{q - 1}"


def _metric_sign(value: float, *, tol: float = 1e-9) -> int:
    if float(value) > tol:
        return 1
    if float(value) < -tol:
        return -1
    return 0


def _history_feature_stats(quarter_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_matrix = seeded._feature_matrix(np.asarray(quarter_states, dtype=np.float64))
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    return np.asarray(feature_mean, dtype=np.float64), np.asarray(feature_std, dtype=np.float64)


def _placebo_shock_matrix(
    *,
    scenario_name: str,
    steps: int,
    block_axis: list[str],
    block_scales: np.ndarray,
) -> np.ndarray:
    if scenario_name not in set(PLACEBO_SCENARIOS):
        raise ValueError(f"Unsupported placebo scenario: {scenario_name}")
    if scenario_name == "disruption_recovery_placebo":
        real = seeded._scenario_shock_matrix(
            "disruption_recovery",
            steps=int(steps),
            block_axis=block_axis,
            block_scales=block_scales,
        )
        placebo = np.zeros_like(real)
        block_index = {name: idx for idx, name in enumerate(block_axis)}
        care_idx = block_index.get("care_access_continuity")
        mobility_idx = block_index.get("mobility_exposure_pressure")
        if care_idx is None:
            return real[::-1, :]
        source = int(care_idx)
        target = int(mobility_idx) if mobility_idx is not None else source
        placebo[:, target] = real[::-1, source]
        return placebo
    real = seeded._scenario_shock_matrix(
        "mobility_spike",
        steps=int(steps),
        block_axis=block_axis,
        block_scales=block_scales,
    )
    placebo = np.zeros_like(real)
    block_index = {name: idx for idx, name in enumerate(block_axis)}
    care_idx = block_index.get("care_access_continuity")
    mobility_idx = block_index.get("mobility_exposure_pressure")
    if mobility_idx is None:
        return real[::-1, :]
    source = int(mobility_idx)
    target = int(care_idx) if care_idx is not None else source
    placebo[:, target] = real[:, source]
    return placebo


def _shock_matrix_for_replay(
    *,
    scenario_name: str,
    steps: int,
    block_axis: list[str],
    block_scales: np.ndarray,
) -> np.ndarray:
    if scenario_name in set(PLACEBO_SCENARIOS):
        return _placebo_shock_matrix(
            scenario_name=scenario_name,
            steps=int(steps),
            block_axis=block_axis,
            block_scales=block_scales,
        )
    return seeded._scenario_shock_matrix(
        scenario_name,
        steps=int(steps),
        block_axis=block_axis,
        block_scales=block_scales,
    )


def _split_exclusive_points(
    *,
    result: dict[str, Any],
    excluded_split_idx: int,
    allowed_tiers: set[str],
    quarter_feature_map: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    points_by_metric = {metric_name: [] for metric_name in seeded.METRIC_PLOT_ORDER}
    for split_idx, split in enumerate(list(result.get("quarterly_rows") or [])):
        if int(split_idx) == int(excluded_split_idx):
            continue
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            quarter = str(target_row.get("quarter") or "")
            features = quarter_feature_map.get(quarter)
            if features is None:
                continue
            for metric_name in seeded.METRIC_PLOT_ORDER:
                if seeded._metric_tier(target_row, metric_name) not in allowed_tiers:
                    continue
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                points_by_metric[metric_name].append(
                    {
                        "features": np.asarray(features, dtype=np.float64),
                        "residual": float(prediction_value) - float(target_value),
                    }
                )
    return points_by_metric


def _fit_split_exclusive_readout(
    *,
    result: dict[str, Any],
    excluded_split_idx: int,
    allowed_tiers: set[str],
    quarter_feature_map: dict[str, np.ndarray],
    feature_names: list[str],
) -> dict[str, Any]:
    points_by_metric = _split_exclusive_points(
        result=result,
        excluded_split_idx=int(excluded_split_idx),
        allowed_tiers=allowed_tiers,
        quarter_feature_map=quarter_feature_map,
    )
    feature_dim = len(feature_names)
    metrics: dict[str, Any] = {}
    for metric_name in seeded.METRIC_PLOT_ORDER:
        fitted = seeded_gate._fit_readout_from_points(
            list(points_by_metric.get(metric_name) or []),
            feature_dim=feature_dim,
        )
        beta_values = fitted.get("beta")
        if beta_values is None:
            beta_array = np.zeros(feature_dim, dtype=np.float64)
        else:
            beta_array = np.asarray(beta_values, dtype=np.float64)
        metrics[metric_name] = {
            "count": int(len(list(points_by_metric.get(metric_name) or []))),
            "beta": [float(value) for value in beta_array],
            "scale": float(fitted.get("scale") or 0.0),
            "cap_abs": float(fitted.get("cap_abs") or 0.0),
        }
    return {
        "feature_names": list(feature_names),
        "metrics": metrics,
    }


def _scenario_rows_for_split(
    *,
    phase2_state: seeded.Phase2QuarterState,
    seed_quarter: str,
    steps: int,
    scenario_name: str,
    base_rows: list[dict[str, Any]],
    readout: dict[str, Any],
) -> list[dict[str, Any]]:
    quarter_to_index = {str(quarter): idx for idx, quarter in enumerate(phase2_state.quarter_axis)}
    if seed_quarter not in quarter_to_index:
        raise KeyError(f"Seed quarter {seed_quarter} is not present in Phase 2 quarter axis.")
    seed_idx = int(quarter_to_index[seed_quarter])
    history_states = np.asarray(phase2_state.quarter_states[: seed_idx + 1, :], dtype=np.float64)
    if history_states.shape[0] < 2:
        raise ValueError(f"Insufficient Phase 2 history through seed quarter {seed_quarter}.")
    kernel = seeded._fit_structural_kernel(history_states, phase2_state.block_axis, phase2_state.edge_rows)
    block_scales = seeded._recent_block_scales(history_states)
    last_state = np.asarray(history_states[-1, :], dtype=np.float64)
    feature_mean, feature_std = _history_feature_stats(history_states)
    status_quo_path = seeded._propagate_structural_states(
        kernel=kernel,
        last_state=last_state,
        steps=int(steps),
        shock_matrix=_shock_matrix_for_replay(
            scenario_name="status_quo",
            steps=int(steps),
            block_axis=phase2_state.block_axis,
            block_scales=block_scales,
        ),
    )
    scenario_path = seeded._propagate_structural_states(
        kernel=kernel,
        last_state=last_state,
        steps=int(steps),
        shock_matrix=_shock_matrix_for_replay(
            scenario_name=scenario_name,
            steps=int(steps),
            block_axis=phase2_state.block_axis,
            block_scales=block_scales,
        ),
    )
    base_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], status_quo_path]))[1:, :] - feature_mean) / feature_std
    scenario_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], scenario_path]))[1:, :] - feature_mean) / feature_std
    return seeded._apply_scenario_to_forecast(
        base_rows=[dict(row) for row in list(base_rows)],
        base_features=base_feature_path,
        scenario_features=scenario_feature_path,
        readout=readout,
    )


def _evaluate_scenario_rows(
    *,
    contract: str,
    split_idx: int,
    scenario_name: str,
    target_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    allowed_tiers: set[str],
) -> dict[str, Any]:
    base_errors: list[float] = []
    scenario_errors: list[float] = []
    improved_count = 0
    directional_matches = 0
    directional_total = 0
    terminal_abs_delta = 0.0
    for target_row, base_row, scenario_row in zip(target_rows, base_rows, scenario_rows, strict=False):
        for metric_name in seeded.METRIC_PLOT_ORDER:
            if seeded._metric_tier(target_row, metric_name) not in allowed_tiers:
                continue
            target_value = target_row.get(metric_name)
            base_value = base_row.get(metric_name)
            scenario_value = scenario_row.get(metric_name)
            if target_value is None or base_value is None or scenario_value is None:
                continue
            base_residual = float(base_value) - float(target_value)
            scenario_residual = float(scenario_value) - float(target_value)
            correction = float(scenario_value) - float(base_value)
            base_abs = abs(base_residual)
            scenario_abs = abs(scenario_residual)
            base_errors.append(base_abs)
            scenario_errors.append(scenario_abs)
            if scenario_abs + 1e-9 < base_abs:
                improved_count += 1
            if abs(base_residual) <= 1e-9 and abs(correction) <= 1e-9:
                directional_matches += 1
                directional_total += 1
            elif abs(correction) > 1e-9:
                directional_matches += int(_metric_sign(correction) == -_metric_sign(base_residual))
                directional_total += 1
            terminal_abs_delta = max(
                float(terminal_abs_delta),
                abs(float(scenario_row.get(f"{metric_name}_scenario_delta") or 0.0)),
            )
    row_count = len(base_errors)
    return {
        "contract": str(contract),
        "split_idx": int(split_idx),
        "scenario": str(scenario_name),
        "row_count": int(row_count),
        "base_mae": float(np.mean(base_errors)) if base_errors else 0.0,
        "scenario_mae": float(np.mean(scenario_errors)) if scenario_errors else 0.0,
        "mae_delta": float(np.mean(scenario_errors) - np.mean(base_errors)) if base_errors else 0.0,
        "improved_fraction": float(improved_count / row_count) if row_count else 0.0,
        "directional_fit_rate": float(directional_matches / directional_total) if directional_total else 0.0,
        "terminal_max_abs_delta": float(terminal_abs_delta),
    }


def _replay_contract_scenarios(
    *,
    archive_run_id: str,
    contract_name: str,
    config: dict[str, Any],
    phase2_state: seeded.Phase2QuarterState,
    scenarios: tuple[str, ...],
) -> list[dict[str, Any]]:
    feature_names = seeded._feature_names(phase2_state.block_axis)
    history_feature_matrix = seeded._feature_matrix(np.asarray(phase2_state.quarter_states, dtype=np.float64))
    feature_mean = history_feature_matrix.mean(axis=0)
    feature_std = history_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    quarter_feature_map = {
        quarter: (history_feature_matrix[idx, :] - feature_mean) / feature_std
        for idx, quarter in enumerate(phase2_state.quarter_axis)
    }
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
    replay_rows: list[dict[str, Any]] = []
    for split_idx, split in enumerate(list(result.get("quarterly_rows") or [])):
        holdout_rows = [dict(row) for row in list(split.get("holdout_target_rows") or [])]
        base_rows = [dict(row) for row in list(split.get("candidate_prediction_rows") or [])]
        if not holdout_rows or not base_rows:
            continue
        holdout_rows.sort(key=lambda row: _quarter_sort(str(row["quarter"])))
        base_rows.sort(key=lambda row: _quarter_sort(str(row["quarter"])))
        first_quarter = str(holdout_rows[0]["quarter"])
        seed_quarter = _previous_quarter(first_quarter)
        readout = _fit_split_exclusive_readout(
            result=result,
            excluded_split_idx=int(split_idx),
            allowed_tiers=set(config["allowed_tiers"]),
            quarter_feature_map=quarter_feature_map,
            feature_names=feature_names,
        )
        for scenario_name in scenarios:
            try:
                scenario_prediction_rows = _scenario_rows_for_split(
                    phase2_state=phase2_state,
                    seed_quarter=seed_quarter,
                    steps=len(holdout_rows),
                    scenario_name=str(scenario_name),
                    base_rows=base_rows,
                    readout=readout,
                )
            except (KeyError, ValueError):
                continue
            replay_rows.append(
                _evaluate_scenario_rows(
                    contract=str(contract_name),
                    split_idx=int(split_idx),
                    scenario_name=str(scenario_name),
                    target_rows=holdout_rows,
                    base_rows=base_rows,
                    scenario_rows=scenario_prediction_rows,
                    allowed_tiers=set(config["allowed_tiers"]),
                )
            )
    return replay_rows


def _summarize_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["contract"]), str(row["scenario"]))].append(dict(row))
    summary_rows: list[dict[str, Any]] = []
    for (contract, scenario), group_rows in sorted(grouped.items()):
        mean_mae_delta = float(np.mean([float(row["mae_delta"]) for row in group_rows])) if group_rows else 0.0
        mean_directional_fit = float(np.mean([float(row["directional_fit_rate"]) for row in group_rows])) if group_rows else 0.0
        mean_improved_fraction = float(np.mean([float(row["improved_fraction"]) for row in group_rows])) if group_rows else 0.0
        mean_terminal_abs_delta = float(np.mean([float(row["terminal_max_abs_delta"]) for row in group_rows])) if group_rows else 0.0
        summary_rows.append(
            {
                "contract": contract,
                "scenario": scenario,
                "split_count": int(len(group_rows)),
                "mean_mae_delta": mean_mae_delta,
                "mean_directional_fit_rate": mean_directional_fit,
                "mean_improved_fraction": mean_improved_fraction,
                "mean_terminal_max_abs_delta": mean_terminal_abs_delta,
                "replay_keep": bool(mean_mae_delta <= 0.0 and mean_directional_fit >= REPLAY_KEEP_DIRECTIONAL_RATE),
            }
        )
    return summary_rows


def _placebo_separation_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(str(row["contract"]), str(row["scenario"])): dict(row) for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for real_scenario, placebo_scenario in PLACEBO_SCENARIO_MAP.items():
        for contract_name in seeded.WINNER_CONFIGS:
            real_row = dict(lookup.get((contract_name, real_scenario)) or {})
            placebo_row = dict(lookup.get((contract_name, placebo_scenario)) or {})
            if not real_row or not placebo_row:
                continue
            mae_margin = float(placebo_row["mean_mae_delta"]) - float(real_row["mean_mae_delta"])
            directional_margin = float(real_row["mean_directional_fit_rate"]) - float(placebo_row["mean_directional_fit_rate"])
            rows.append(
                {
                    "contract": contract_name,
                    "scenario": real_scenario,
                    "placebo_scenario": placebo_scenario,
                    "real_mean_mae_delta": float(real_row["mean_mae_delta"]),
                    "placebo_mean_mae_delta": float(placebo_row["mean_mae_delta"]),
                    "mae_margin_vs_placebo": mae_margin,
                    "real_directional_fit_rate": float(real_row["mean_directional_fit_rate"]),
                    "placebo_directional_fit_rate": float(placebo_row["mean_directional_fit_rate"]),
                    "directional_margin_vs_placebo": directional_margin,
                    "separation_keep": bool(mae_margin > 0.0 and directional_margin > 0.0),
                }
            )
    return rows


def _plot_replay_mae_delta(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Replay MAE deltas", body="No replay rows.")
        return
    labels = [f"{row['contract']}:{row['scenario']}" for row in rows]
    values = [float(row["mean_mae_delta"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    colors = ["#4c72b0" if str(row["scenario"]) in set(ACTIVE_SCENARIOS) else "#c44e52" for row in rows]
    fig, ax = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.45)))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Scenario MAE - baseline MAE")
    ax.set_title("Blocked replay mean MAE deltas")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_directional_fit_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Replay directional fit", body="No replay rows.")
        return
    row_labels = [f"{row['contract']}:{row['scenario']}" for row in rows]
    matrix = np.asarray([[float(row["mean_directional_fit_rate"])] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, max(4.0, len(row_labels) * 0.38)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks([0])
    ax.set_xticklabels(["directional fit"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Blocked replay directional-fit rates")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_placebo_separation(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Placebo separation", body="No placebo rows.")
        return
    labels = [f"{row['contract']}:{row['scenario']}" for row in rows]
    matrix = np.asarray(
        [
            [float(row["mae_margin_vs_placebo"]), float(row["directional_margin_vs_placebo"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7, max(4.0, len(labels) * 0.42)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MAE margin", "directional margin"], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Real scenario separation from matched placebo")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_replay_split_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        suite._plot_placeholder(path, title="Replay split audit", body="No split rows.")
        return
    labels = [f"{row['contract']}:{row['scenario']}:split{int(row['split_idx'])}" for row in rows]
    matrix = np.asarray(
        [
            [float(row["mae_delta"]), float(row["directional_fit_rate"]), float(row["terminal_max_abs_delta"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(labels) * 0.32)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["MAE delta", "directional fit", "terminal |delta|"], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Blocked replay split-level audit")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_orthogonalized_alignment(rows: list[dict[str, Any]], path: Path) -> None:
    seeded_gate._plot_circularity_ablation_heatmap(rows, path)


def _markdown_report(payload: dict[str, Any]) -> str:
    orth_gate = dict(payload.get("orthogonalized_gate") or {})
    lines = [
        "# TR-V3 Phase 2 Admissibility Backtest Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Two-block run: `{payload['two_block_run_id']}`",
        f"- Monthly Phase 2 run: `{payload['monthly_phase2_run_id']}`",
        f"- Aligned archive run: `{payload['aligned_archive_run_id']}`",
        f"- Active block subset: `{list(payload.get('active_block_subset') or [])}`",
        "",
        "## Fixed context",
        "",
        f"- Champion context: `R10 exact + dense observation-first benchmark track`",
        f"- Active scenarios: `{list(payload.get('active_scenarios') or [])}`",
        f"- Placebo scenarios: `{list(payload.get('placebo_scenarios') or [])}`",
        "",
        "## Two-block baseline",
        "",
        f"- Decision: `{payload['two_block_baseline']['decision']}`",
        f"- Active sign agreement: `{float(payload['two_block_baseline']['active_sign_agreement']):.3f}`",
        f"- Active mean abs delta-of-delta: `{float(payload['two_block_baseline']['active_mean_abs_delta']):.3f}`",
        f"- Max abs terminal delta: `{float(payload['two_block_baseline']['max_abs_terminal_delta']):.3f}`",
        "",
        "## Orthogonalized-state admissibility",
        "",
        f"- Direct outcome-loaded active blocks: `{int(orth_gate.get('active_outcome_loaded_block_count') or 0)}`",
        f"- Sign agreement: `{float(orth_gate.get('sign_agreement_rate') or 0.0):.3f}`",
        f"- Mean abs delta-of-delta: `{float(orth_gate.get('mean_abs_delta_of_delta') or 0.0):.3f}`",
        f"- Mean abs ratio: `{float(orth_gate.get('mean_abs_ratio') or 0.0):.3f}`",
        f"- Decision: `{orth_gate.get('decision')}`",
        "",
        "## Blocked replay summary",
        "",
        "| Contract | Scenario | Splits | Mean MAE delta | Directional fit | Improved fraction | Keep |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("replay_summary_rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | `{int(row['split_count'])}` | "
            f"`{float(row['mean_mae_delta']):.3f}` | `{float(row['mean_directional_fit_rate']):.3f}` | "
            f"`{float(row['mean_improved_fraction']):.3f}` | `{bool(row['replay_keep'])}` |"
        )
    lines.extend(
        [
            "",
            "## Placebo separation",
            "",
            "| Contract | Scenario | Placebo | MAE margin | Directional margin | Keep |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in list(payload.get("placebo_separation_rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | {row['placebo_scenario']} | "
            f"`{float(row['mae_margin_vs_placebo']):.3f}` | `{float(row['directional_margin_vs_placebo']):.3f}` | "
            f"`{bool(row['separation_keep'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Replay gate: `{payload['gate_summary']['replay_status']}`",
            f"- Placebo gate: `{payload['gate_summary']['placebo_status']}`",
            f"- Overall: `{payload['gate_summary']['overall_decision']}`",
            "",
            "## Graphs",
            "",
        ]
    )
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run_tr_v3_phase2_admissibility_backtest_batch(
    *,
    run_id: str,
    monthly_phase2_run_id: str | None = None,
    legacy_archive_run_id: str | None = None,
    aligned_archive_run_id: str | None = None,
    forecast_horizon_quarters: int = 8,
    active_block_subset: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    active_blocks = tuple(str(name) for name in list(active_block_subset or two_block.DEFAULT_ACTIVE_BLOCK_SUBSET))
    two_block_run_id = f"{run_id}-two-block"
    two_block_payload = two_block.run_tr_v3_phase2_two_block_kernel_batch(
        run_id=two_block_run_id,
        monthly_phase2_run_id=monthly_phase2_run_id,
        legacy_archive_run_id=legacy_archive_run_id,
        aligned_archive_run_id=aligned_archive_run_id,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        active_block_subset=active_blocks,
    )
    monthly_run = str(two_block_payload["monthly_phase2_run_id"])
    aligned_archive_run = str(two_block_payload["aligned_archive_run_id"])
    aligned_seeded_run = f"{two_block_run_id}-aligned-seeded"
    aligned_seeded_payload = _seeded_report(aligned_seeded_run)

    phase2_state = seeded._filter_phase2_quarter_state(
        seeded._load_phase2_quarter_state(monthly_run),
        active_block_subset=active_blocks,
    )
    loading_rows = seeded_gate._load_block_loading_rows(monthly_run)
    outcome_loading_rows = seeded_gate._summarize_outcome_circularity_rows(
        loading_rows=loading_rows,
        outcome_canonical_names=set(seeded_gate.OUTCOME_CANONICALS),
    )
    active_outcome_rows = [
        dict(row)
        for row in outcome_loading_rows
        if str(row.get("block_id") or "") in set(active_blocks)
    ]
    orthogonalized_state, orthogonalization_rows = seeded_gate._orthogonalize_phase2_state_against_outcomes(
        phase2_state=phase2_state,
        archive_run_id=aligned_archive_run,
        loading_rows=loading_rows,
    )
    orthogonalized_payload = seeded_gate._seeded_payload_from_phase2_state(
        archive_run_id=aligned_archive_run,
        phase2_state=orthogonalized_state,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
    )
    orthogonalized_alignment_rows = seeded_gate._archive_alignment_rows(aligned_seeded_payload, orthogonalized_payload)
    sign_agreement = (
        float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in orthogonalized_alignment_rows]))
        if orthogonalized_alignment_rows
        else 0.0
    )
    mean_abs_delta = (
        float(np.mean([abs(float(row["delta_of_delta"])) for row in orthogonalized_alignment_rows]))
        if orthogonalized_alignment_rows
        else 0.0
    )
    abs_ratios = [
        abs(float(row["aligned_terminal_delta"])) / max(abs(float(row["baseline_terminal_delta"])), 1.0)
        for row in orthogonalized_alignment_rows
    ]
    mean_abs_ratio = float(np.mean(abs_ratios)) if abs_ratios else 0.0
    orth_gate = {
        "active_outcome_loaded_block_count": int(sum(1 for row in active_outcome_rows if float(row.get("outcome_loading_share") or 0.0) > 1e-9)),
        "sign_agreement_rate": sign_agreement,
        "mean_abs_delta_of_delta": mean_abs_delta,
        "mean_abs_ratio": mean_abs_ratio,
        "decision": "keep" if sign_agreement >= 0.75 and 0.4 <= mean_abs_ratio <= 2.0 else "revert",
        "loading_rows": active_outcome_rows,
        "orthogonalization_rows": orthogonalization_rows,
        "alignment_rows": orthogonalized_alignment_rows,
    }

    replay_rows: list[dict[str, Any]] = []
    for contract_name, config in seeded.WINNER_CONFIGS.items():
        replay_rows.extend(
            _replay_contract_scenarios(
                archive_run_id=aligned_archive_run,
                contract_name=contract_name,
                config=config,
                phase2_state=phase2_state,
                scenarios=ALL_REPLAY_SCENARIOS,
            )
        )
    replay_summary_rows = _summarize_replay_rows(replay_rows)
    placebo_rows = _placebo_separation_rows(replay_summary_rows)
    real_summary_rows = [row for row in replay_summary_rows if str(row["scenario"]) in set(ACTIVE_SCENARIOS)]
    replay_status = (
        "keep"
        if real_summary_rows and all(bool(row["replay_keep"]) for row in real_summary_rows)
        else "revert_replay_instability"
    )
    placebo_status = (
        "keep"
        if placebo_rows and all(bool(row["separation_keep"]) for row in placebo_rows)
        else "revert_placebo_nonseparation"
    )
    overall_decision = (
        "keep_provisional_two_block_sidecar"
        if str(two_block_payload.get("overall_decision") or "") == "keep_minimal_two_block_kernel"
        and orth_gate["decision"] == "keep"
        and replay_status == "keep"
        and placebo_status == "keep"
        else "revert_to_no_live_phase2_sidecar"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    orth_graph = analysis_dir / "two_block_orthogonalized_alignment_compare.png"
    _plot_orthogonalized_alignment(orthogonalized_alignment_rows, orth_graph)
    replay_mae_graph = analysis_dir / "two_block_replay_mae_delta_compare.png"
    _plot_replay_mae_delta(replay_summary_rows, replay_mae_graph)
    replay_directional_graph = analysis_dir / "two_block_replay_directional_fit_heatmap.png"
    _plot_directional_fit_heatmap(replay_summary_rows, replay_directional_graph)
    placebo_graph = analysis_dir / "two_block_placebo_separation_heatmap.png"
    _plot_placebo_separation(placebo_rows, placebo_graph)
    replay_split_graph = analysis_dir / "two_block_replay_split_heatmap.png"
    _plot_replay_split_heatmap(replay_rows, replay_split_graph)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "two_block_run_id": str(two_block_run_id),
        "monthly_phase2_run_id": monthly_run,
        "aligned_archive_run_id": aligned_archive_run,
        "active_block_subset": list(active_blocks),
        "active_scenarios": list(ACTIVE_SCENARIOS),
        "placebo_scenarios": list(PLACEBO_SCENARIOS),
        "two_block_baseline": {
            "decision": str(two_block_payload.get("overall_decision") or ""),
            "active_sign_agreement": float(dict(two_block_payload.get("archive_alignment_gate") or {}).get("active_sign_agreement") or 0.0),
            "active_mean_abs_delta": float(dict(two_block_payload.get("archive_alignment_gate") or {}).get("active_mean_abs_delta") or 0.0),
            "max_abs_terminal_delta": float(dict(two_block_payload.get("active_scenario_summary") or {}).get("max_abs_terminal_delta") or 0.0),
        },
        "orthogonalized_gate": orth_gate,
        "replay_rows": replay_rows,
        "replay_summary_rows": replay_summary_rows,
        "placebo_separation_rows": placebo_rows,
        "gate_summary": {
            "replay_status": replay_status,
            "placebo_status": placebo_status,
            "overall_decision": overall_decision,
        },
        "artifacts": {
            "two_block_report": str(Path(two_block_run_id) / "analysis" / "tr_v3_phase2_two_block_kernel_batch_report.md"),
            "two_block_seeded_report": str(Path(f"{two_block_run_id}-aligned-seeded") / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "two_block_alignment_report": str(Path(f"{two_block_run_id}-alignment") / "analysis" / "tr_v3_phase2_archive_alignment_batch_report.md"),
            "orthogonalized_alignment_graph": orth_graph.name,
            "replay_mae_delta_graph": replay_mae_graph.name,
            "replay_directional_fit_graph": replay_directional_graph.name,
            "placebo_separation_graph": placebo_graph.name,
            "replay_split_graph": replay_split_graph.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_admissibility_backtest_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_admissibility_backtest_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the combined Phase 2 sidecar admissibility backtest on the frozen two-block kernel.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    parser.add_argument("--legacy-archive-run-id", default=None)
    parser.add_argument("--aligned-archive-run-id", default=None)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    parser.add_argument("--active-block", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_admissibility_backtest_batch(
        run_id=str(args.run_id),
        monthly_phase2_run_id=args.monthly_phase2_run_id,
        legacy_archive_run_id=args.legacy_archive_run_id,
        aligned_archive_run_id=args.aligned_archive_run_id,
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
        active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
