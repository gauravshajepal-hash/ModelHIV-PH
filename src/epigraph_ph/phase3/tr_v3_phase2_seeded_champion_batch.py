from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_champion_forecast as champion_forecast
from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3.tr_v3_05_autoresearch import build_annual_anchor_rows
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, load_tensor_artifact, read_json, write_json


WINNER_CONFIGS: dict[str, dict[str, Any]] = {
    "exact_only": {
        "winner_id": "EXP-R10-M1-F1-C1",
        "suite_contract": "exact_only",
        "forecast_contract": "exact_only",
        "allowed_tiers": {"exact_observed"},
        "title": "Exact winner",
    },
    "purged_dense": {
        "winner_id": "EXP-R10-DENSE-M1-C1-H1",
        "suite_contract": "purged_dense",
        "forecast_contract": "dense_train_observed_score",
        "allowed_tiers": {"exact_observed", "bridge_observed"},
        "title": "Dense winner",
    },
}
SCENARIO_ORDER: tuple[str, ...] = (
    "status_quo",
    "testing_pulse",
    "testing_plateau",
    "disruption_recovery",
    "mobility_spike",
)
SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "status_quo": "No additional structural shock. Base champion path plus the fitted structural persistence kernel only.",
    "testing_pulse": "Positive testing or prevention-reach pulse that decays over three quarters and feeds into care through the learned lag structure when an upstream testing block is present.",
    "testing_plateau": "Sustained moderate testing or prevention pressure that holds the structural state on an elevated plateau rather than allowing a quick mean reversion.",
    "disruption_recovery": "Short testing or care disruption followed by a delayed recovery pulse, intended to emulate program interruption and catch-up.",
    "mobility_spike": "Temporary mobility/exposure spike with no direct mean-model claim; kept as a structural stress test.",
}
METRIC_PLOT_ORDER: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)


@dataclass(slots=True)
class Phase2QuarterState:
    source_run_id: str
    month_axis: list[str]
    quarter_axis: list[str]
    block_axis: list[str]
    quarter_states: np.ndarray
    edge_rows: list[dict[str, Any]]


def _latest_monthly_phase2_run() -> str:
    run_root = ROOT_DIR / "artifacts" / "runs"
    candidates = sorted(run_root.glob("tr-v3-monthly-phase2-lane-*"))
    if not candidates:
        raise FileNotFoundError("No monthly Phase 2 lane run found under artifacts/runs.")
    return str(candidates[-1].name)


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    quarter = ((int(month_text[:2]) - 1) // 3) + 1
    return f"{int(year_text):04d}-Q{quarter}"


def _quarterize_tensor(tensor: np.ndarray, month_axis: list[str]) -> tuple[list[str], np.ndarray]:
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    grouped: dict[str, list[int]] = {}
    for idx, month_label in enumerate(month_axis):
        grouped.setdefault(_quarter_from_month(str(month_label)), []).append(int(idx))
    quarter_axis = sorted(grouped, key=suite.quarter_sort_key)
    quarter_values = np.zeros((len(quarter_axis), values.shape[2]), dtype=np.float64)
    for quarter_idx, quarter in enumerate(quarter_axis):
        quarter_values[quarter_idx, :] = values[0, grouped[quarter], :].mean(axis=0)
    return quarter_axis, quarter_values


def _load_phase2_quarter_state(run_id: str) -> Phase2QuarterState:
    run_dir = ROOT_DIR / "artifacts" / "runs" / str(run_id)
    payload = read_json(run_dir / "phase2" / "phase2_structural_payload.json", default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"Phase 2 structural payload missing for run {run_id}.")
    month_axis = [str(value) for value in list(payload.get("month_axis") or [])]
    block_axis = [str(value) for value in list(payload.get("block_axis") or [])]
    artifact_paths = dict(payload.get("artifact_paths") or {})
    tensor_path = Path(str(artifact_paths.get("phase15_national_state_tensor") or run_dir / "phase15" / "phase15_v2_national_state_tensor.npz"))
    quarter_axis, quarter_states = _quarterize_tensor(np.asarray(load_tensor_artifact(tensor_path), dtype=np.float64), month_axis)
    edge_rows = [dict(row) for row in list(payload.get("direct_temporal_edge_rows") or []) if int(row.get("lag") or 0) == 1]
    return Phase2QuarterState(
        source_run_id=str(run_id),
        month_axis=month_axis,
        quarter_axis=quarter_axis,
        block_axis=block_axis,
        quarter_states=quarter_states,
        edge_rows=edge_rows,
    )


def _filter_phase2_quarter_state(
    phase2_state: Phase2QuarterState,
    *,
    active_block_subset: tuple[str, ...] | list[str] | None,
) -> Phase2QuarterState:
    requested = [str(name).strip() for name in list(active_block_subset or []) if str(name).strip()]
    if not requested:
        return phase2_state
    missing = [name for name in requested if name not in phase2_state.block_axis]
    if missing:
        raise ValueError(f"Requested active blocks are not present in Phase 2 state: {missing}")
    index_lookup = {name: idx for idx, name in enumerate(phase2_state.block_axis)}
    keep_indices = [int(index_lookup[name]) for name in requested]
    filtered_edges = [
        dict(row)
        for row in list(phase2_state.edge_rows)
        if str(row.get("source") or "") in requested and str(row.get("target") or "") in requested
    ]
    return Phase2QuarterState(
        source_run_id=phase2_state.source_run_id,
        month_axis=list(phase2_state.month_axis),
        quarter_axis=list(phase2_state.quarter_axis),
        block_axis=list(requested),
        quarter_states=np.asarray(phase2_state.quarter_states[:, keep_indices], dtype=np.float64),
        edge_rows=filtered_edges,
    )


def _feature_names(block_axis: list[str]) -> list[str]:
    return list(block_axis) + [f"{name}_delta" for name in block_axis]


def _feature_matrix(state_rows: np.ndarray) -> np.ndarray:
    values = np.asarray(state_rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("state_rows must be 2D.")
    deltas = np.zeros_like(values)
    if values.shape[0] > 1:
        deltas[1:, :] = values[1:, :] - values[:-1, :]
    return np.concatenate([values, deltas], axis=1)


def _metric_tier(target_row: dict[str, Any], metric_name: str) -> str:
    metric_tiers = target_row.get("metric_tiers")
    if isinstance(metric_tiers, dict) and metric_name in metric_tiers:
        return str(metric_tiers[metric_name])
    tier_name = target_row.get(f"{metric_name}_tier")
    if isinstance(tier_name, str):
        return str(tier_name)
    return "exact_observed"


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _collect_contract_result(
    *,
    archive_run_id: str,
    contract_name: str,
    experiment_id: str,
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        experiment_ids=[experiment_id],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    result_map = _suite_result_map(payload)
    if experiment_id not in result_map:
        raise LookupError(f"Experiment {experiment_id} did not execute for contract {contract_name}.")
    return result_map[experiment_id]


def _fit_structural_kernel(quarter_states: np.ndarray, block_axis: list[str], edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    states = np.asarray(quarter_states, dtype=np.float64)
    block_index = {name: idx for idx, name in enumerate(block_axis)}
    edge_matrix = np.zeros((len(block_axis), len(block_axis)), dtype=np.float64)
    for row in edge_rows:
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if source not in block_index or target not in block_index:
            continue
        edge_matrix[block_index[target], block_index[source]] = float(row.get("weight") or 0.0)
    intercept = np.zeros(len(block_axis), dtype=np.float64)
    persistence = np.zeros(len(block_axis), dtype=np.float64)
    for block_idx in range(len(block_axis)):
        current = states[:-1, block_idx]
        incoming = states[:-1, :] @ edge_matrix[block_idx, :]
        target = states[1:, block_idx] - incoming
        design = np.column_stack([np.ones_like(current), current])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        intercept[block_idx] = float(coef[0])
        persistence[block_idx] = float(np.clip(coef[1], 0.0, 0.98))
    return {
        "block_axis": list(block_axis),
        "edge_matrix": edge_matrix,
        "intercept": intercept,
        "persistence": persistence,
    }


def _recent_block_scales(quarter_states: np.ndarray) -> np.ndarray:
    values = np.asarray(quarter_states, dtype=np.float64)
    recent = values[-8:, :] if values.shape[0] >= 8 else values
    scale = np.std(recent, axis=0)
    global_scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, global_scale)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.asarray(scale, dtype=np.float64)


def _testing_block_name(block_axis: list[str]) -> str | None:
    for name in ("testing_prevention_reach", "testing_engagement"):
        if name in set(block_axis):
            return name
    return None


def _scenario_shock_matrix(
    scenario_name: str,
    *,
    steps: int,
    block_axis: list[str],
    block_scales: np.ndarray,
) -> np.ndarray:
    shocks = np.zeros((int(steps), len(block_axis)), dtype=np.float64)
    block_index = {name: idx for idx, name in enumerate(block_axis)}
    testing_block = _testing_block_name(block_axis)

    def add(block_name: str, quarter_idx: int, magnitude: float) -> None:
        if block_name not in block_index or quarter_idx < 0 or quarter_idx >= steps:
            return
        shocks[quarter_idx, block_index[block_name]] += magnitude * float(block_scales[block_index[block_name]])

    if scenario_name == "status_quo":
        return shocks
    if scenario_name == "testing_pulse":
        for idx, weight in enumerate((1.0, 0.55, 0.2)):
            if testing_block is not None:
                add(testing_block, idx, weight)
        return shocks
    if scenario_name == "testing_plateau":
        for idx in range(steps):
            if testing_block is not None:
                add(testing_block, idx, 0.45)
        return shocks
    if scenario_name == "disruption_recovery":
        for idx, weight in enumerate((-0.9, -0.7, -0.35)):
            if testing_block is not None:
                add(testing_block, idx, weight)
        for idx, weight in zip((1, 2), (-0.35, -0.2), strict=False):
            add("care_access_continuity", idx, weight)
        for idx, weight in zip((3, 4, 5), (0.25, 0.35, 0.2), strict=False):
            if testing_block is not None:
                add(testing_block, idx, weight)
            add("care_access_continuity", idx, weight * 0.5)
        return shocks
    if scenario_name == "mobility_spike":
        for idx, weight in enumerate((1.0, 0.65, 0.3)):
            add("mobility_exposure_pressure", idx, weight)
        return shocks
    raise ValueError(f"Unsupported scenario_name: {scenario_name}")


def _propagate_structural_states(
    *,
    kernel: dict[str, Any],
    last_state: np.ndarray,
    steps: int,
    shock_matrix: np.ndarray,
) -> np.ndarray:
    intercept = np.asarray(kernel["intercept"], dtype=np.float64)
    persistence = np.asarray(kernel["persistence"], dtype=np.float64)
    edge_matrix = np.asarray(kernel["edge_matrix"], dtype=np.float64)
    prev = np.asarray(last_state, dtype=np.float64)
    path = np.zeros((int(steps), prev.shape[0]), dtype=np.float64)
    for step_idx in range(int(steps)):
        next_state = intercept + (persistence * prev) + (edge_matrix @ prev) + shock_matrix[step_idx, :]
        path[step_idx, :] = next_state
        prev = next_state
    return path


def _fit_residual_readout(
    *,
    result: dict[str, Any],
    allowed_tiers: set[str],
    quarter_feature_map: dict[str, np.ndarray],
    feature_names: list[str],
    ridge_penalty: float = 1.0,
) -> dict[str, Any]:
    feature_dim = len(feature_names)
    metrics: dict[str, Any] = {}
    for metric_name in METRIC_PLOT_ORDER:
        rows: list[tuple[np.ndarray, float]] = []
        for split in list(result.get("quarterly_rows") or []):
            for target_row, prediction_row in zip(
                list(split.get("holdout_target_rows") or []),
                list(split.get("candidate_prediction_rows") or []),
                strict=False,
            ):
                if _metric_tier(target_row, metric_name) not in allowed_tiers:
                    continue
                quarter = str(target_row["quarter"])
                features = quarter_feature_map.get(quarter)
                if features is None:
                    continue
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                rows.append((features, float(prediction_value) - float(target_value)))
        if len(rows) < max(8, feature_dim + 2):
            metrics[metric_name] = {
                "count": len(rows),
                "beta": [0.0] * feature_dim,
                "scale": 0.0,
                "cap_abs": 0.0,
                "residual_p90": 0.0,
                "residual_mean": float(np.mean([value for _, value in rows])) if rows else 0.0,
            }
            continue
        design = np.asarray([features for features, _ in rows], dtype=np.float64)
        target = np.asarray([value for _, value in rows], dtype=np.float64)
        centered = target - float(np.mean(target))
        lhs = design.T @ design
        rhs = design.T @ centered
        beta = np.linalg.solve(lhs + (float(ridge_penalty) * np.eye(feature_dim, dtype=np.float64)), rhs)
        raw = design @ beta
        raw_p90 = float(np.quantile(np.abs(raw), 0.9)) if raw.size else 0.0
        residual_p90 = float(np.quantile(np.abs(centered), 0.9)) if centered.size else 0.0
        scale = float(np.clip(0.35 * (residual_p90 / max(raw_p90, 1e-6)), 0.0, 0.5))
        metrics[metric_name] = {
            "count": int(len(rows)),
            "beta": [float(value) for value in beta],
            "scale": float(scale),
            "cap_abs": float(residual_p90),
            "residual_p90": float(residual_p90),
            "residual_mean": float(np.mean(target)),
        }
    return {"feature_names": list(feature_names), "metrics": metrics}


def _bounded_metric_correction(readout_metric: dict[str, Any], feature_delta: np.ndarray) -> float:
    beta = np.asarray(list(readout_metric.get("beta") or []), dtype=np.float64)
    if beta.size == 0:
        return 0.0
    raw = float(readout_metric.get("scale") or 0.0) * float(np.dot(beta, np.asarray(feature_delta, dtype=np.float64)))
    cap_abs = float(readout_metric.get("cap_abs") or 0.0)
    if cap_abs <= 0.0:
        return 0.0
    return float(np.clip(raw, -cap_abs, cap_abs))


def _apply_scenario_to_forecast(
    *,
    base_rows: list[dict[str, Any]],
    base_features: np.ndarray,
    scenario_features: np.ndarray,
    readout: dict[str, Any],
) -> list[dict[str, Any]]:
    adjusted_rows: list[dict[str, Any]] = []
    for row, base_feature_row, scenario_feature_row in zip(base_rows, base_features, scenario_features, strict=False):
        feature_delta = np.asarray(scenario_feature_row, dtype=np.float64) - np.asarray(base_feature_row, dtype=np.float64)
        adjusted = dict(row)
        for metric_name in METRIC_PLOT_ORDER:
            correction = _bounded_metric_correction(dict(readout["metrics"][metric_name]), feature_delta)
            base_value = float(adjusted.get(metric_name) or 0.0)
            adjusted[metric_name] = float(max(base_value + correction, 0.0))
            adjusted[f"{metric_name}_scenario_delta"] = float(correction)
        adjusted["alive_on_art"] = float(min(float(adjusted.get("alive_on_art") or 0.0), float(adjusted.get("diagnosed_plhiv") or 0.0)))
        if adjusted.get("virally_suppressed") is not None:
            adjusted["virally_suppressed"] = float(min(float(adjusted.get("virally_suppressed") or 0.0), float(adjusted.get("alive_on_art") or 0.0)))
        adjusted_rows.append(adjusted)
    return adjusted_rows


def _plot_structural_history_and_scenarios(
    *,
    quarter_axis: list[str],
    block_axis: list[str],
    history_states: np.ndarray,
    future_quarters: list[str],
    scenario_paths: dict[str, np.ndarray],
    path: Path,
) -> None:
    rows = int(np.ceil(len(block_axis) / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(4.5, rows * 3.6)))
    axes_list = list(np.asarray(axes).reshape(-1))
    colors = {
        "status_quo": "#1f77b4",
        "testing_pulse": "#d62728",
        "testing_plateau": "#2ca02c",
        "disruption_recovery": "#ff7f0e",
        "mobility_spike": "#9467bd",
    }
    for ax, block_name in zip(axes_list, block_axis, strict=False):
        block_idx = block_axis.index(block_name)
        ax.plot(list(quarter_axis), history_states[:, block_idx], color="#111111", linewidth=1.8, label="History")
        for scenario_name in SCENARIO_ORDER:
            ax.plot(list(future_quarters), scenario_paths[scenario_name][:, block_idx], color=colors[scenario_name], linewidth=1.8, label=scenario_name)
        ax.set_title(block_name.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
    for ax in axes_list[len(block_axis):]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    fig.suptitle("Quarterized Phase 2 block states with seeded future scenarios")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_contract_scenarios(
    *,
    title: str,
    observation_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    scenario_rows: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_list = list(np.asarray(axes).reshape(-1))
    colors = {
        "status_quo": "#1f77b4",
        "testing_pulse": "#d62728",
        "testing_plateau": "#2ca02c",
        "disruption_recovery": "#ff7f0e",
        "mobility_spike": "#9467bd",
    }
    observed_lookup = {metric_name: [row for row in observation_rows if row.get(metric_name) is not None] for metric_name in METRIC_PLOT_ORDER}
    for ax, metric_name in zip(axes_list[:3], METRIC_PLOT_ORDER, strict=False):
        observed = observed_lookup[metric_name]
        ax.plot([str(row["quarter"]) for row in observed], [float(row[metric_name]) for row in observed], color="#111111", linewidth=1.8, label="Observed history")
        for scenario_name in SCENARIO_ORDER:
            rows = scenario_rows[scenario_name] if scenario_name != "status_quo" else base_rows
            ax.plot([str(row["quarter"]) for row in rows], [float(row[metric_name]) for row in rows], color=colors[scenario_name], linewidth=1.8, label=scenario_name)
        ax.set_title(metric_name.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
    delta_ax = axes_list[3]
    quarters = [str(row["quarter"]) for row in base_rows]
    for scenario_name in SCENARIO_ORDER[1:]:
        delta_ax.plot(
            quarters,
            [float(row.get("diagnosed_plhiv_scenario_delta") or 0.0) for row in scenario_rows[scenario_name]],
            linewidth=1.8,
            label=f"{scenario_name}: diagnosed delta",
            color=colors[scenario_name],
        )
    delta_ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    delta_ax.set_title("Diagnosed structural deltas vs base")
    delta_ax.tick_params(axis="x", rotation=45)
    delta_ax.grid(True, alpha=0.2)
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_readout_heatmap(
    *,
    payloads: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    feature_names = list(next(iter(payloads.values())).get("feature_names") or [])
    row_labels: list[str] = []
    rows: list[list[float]] = []
    for contract_name, payload in payloads.items():
        metrics = dict(payload.get("metrics") or {})
        for metric_name in METRIC_PLOT_ORDER:
            metric_payload = dict(metrics.get(metric_name) or {})
            beta = np.asarray(metric_payload.get("beta") or [0.0] * len(feature_names), dtype=np.float64)
            scale = float(metric_payload.get("scale") or 0.0)
            rows.append(list(scale * beta))
            row_labels.append(f"{contract_name}:{metric_name}")
    matrix = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8.0, len(feature_names) * 0.8), max(4.0, len(row_labels) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Bounded Phase 2 residual readout coefficients")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_terminal_delta_heatmap(
    *,
    scenario_terminal_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not scenario_terminal_rows:
        suite._plot_placeholder(path, title="Phase 2 seeded champion scenarios", body="No terminal deltas.")
        return
    row_labels = [f"{row['contract']}:{row['scenario']}" for row in scenario_terminal_rows]
    matrix = np.asarray(
        [
            [
                float(row.get("diagnosed_plhiv_delta") or 0.0),
                float(row.get("alive_on_art_delta") or 0.0),
                float(row.get("new_diagnosed_cases_period_delta") or 0.0),
            ]
            for row in scenario_terminal_rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(row_labels) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="PiYG")
    ax.set_xticks(range(len(METRIC_PLOT_ORDER)))
    ax.set_xticklabels([metric.replace("_", " ") for metric in METRIC_PLOT_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Final-quarter scenario deltas vs base champion")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    active_subset = list(payload.get("active_block_subset") or [])
    lines = [
        "# TR-V3 Phase 2 Seeded Champion Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Readout source archive run: `{payload['readout_source_archive_run_id']}`",
        f"- Phase 2 monthly lane run: `{payload['phase2_monthly_run_id']}`",
        f"- Active block subset: `{active_subset if active_subset else 'full block axis'}`",
        f"- Forecast horizon (quarters): `{payload['forecast_horizon_quarters']}`",
        "",
        "## Structural seed",
        "",
        f"- Source block axis: `{payload['phase2_seed']['source_block_axis']}`",
        f"- Blocks: `{payload['phase2_seed']['block_axis']}`",
        f"- Lag-1 direct edges: `{payload['phase2_seed']['edge_count']}`",
        f"- Quarter range: `{payload['phase2_seed']['quarter_axis'][0]}` to `{payload['phase2_seed']['quarter_axis'][-1]}`",
        "",
        "## Scenarios",
        "",
    ]
    for scenario_name in SCENARIO_ORDER:
        lines.append(f"- `{scenario_name}`: {SCENARIO_DESCRIPTIONS[scenario_name]}")
    lines.extend(
        [
            "",
            "## Readout fit",
            "",
            "| Contract | Metric | Residual count | Readout scale | Residual p90 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for contract_name, contract_payload in dict(payload.get("contracts") or {}).items():
        readout_metrics = dict(contract_payload.get("readout") or {}).get("metrics") or {}
        for metric_name in METRIC_PLOT_ORDER:
            metric_payload = dict(readout_metrics.get(metric_name) or {})
            lines.append(
                f"| {contract_name} | `{metric_name}` | `{int(metric_payload.get('count') or 0)}` | "
                f"`{float(metric_payload.get('scale') or 0.0):.4f}` | `{float(metric_payload.get('residual_p90') or 0.0):.3f}` |"
            )
    lines.extend(
        [
            "",
            "## Final-quarter scenario deltas",
            "",
            "| Contract | Scenario | Diagnosed delta | ART delta | Flow delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in list(payload.get("terminal_delta_rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | `{float(row['diagnosed_plhiv_delta']):.1f}` | "
            f"`{float(row['alive_on_art_delta']):.1f}` | `{float(row['new_diagnosed_cases_period_delta']):.1f}` |"
        )
    lines.extend(["", "## Graphs", ""])
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run_tr_v3_phase2_seeded_champion_batch(
    *,
    run_id: str,
    archive_run_id: str | None = None,
    readout_source_archive_run_id: str | None = None,
    monthly_phase2_run_id: str | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
    forecast_horizon_quarters: int = 8,
    active_block_subset: tuple[str, ...] | list[str] | None = None,
    winner_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or suite._latest_standard_archive_run())
    readout_source_archive_run = str(readout_source_archive_run_id or archive_run)
    phase2_run = str(monthly_phase2_run_id or _latest_monthly_phase2_run())
    source_phase2_state = _load_phase2_quarter_state(phase2_run)
    phase2_state = _filter_phase2_quarter_state(source_phase2_state, active_block_subset=active_block_subset)
    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    annual_rows = build_annual_anchor_rows(archive_run)

    kernel = _fit_structural_kernel(phase2_state.quarter_states, phase2_state.block_axis, phase2_state.edge_rows)
    block_scales = _recent_block_scales(phase2_state.quarter_states)
    last_state = np.asarray(phase2_state.quarter_states[-1, :], dtype=np.float64)
    scenario_paths = {
        scenario_name: _propagate_structural_states(
            kernel=kernel,
            last_state=last_state,
            steps=int(forecast_horizon_quarters),
            shock_matrix=_scenario_shock_matrix(
                scenario_name,
                steps=int(forecast_horizon_quarters),
                block_axis=phase2_state.block_axis,
                block_scales=block_scales,
            ),
        )
        for scenario_name in SCENARIO_ORDER
    }
    history_feature_matrix = _feature_matrix(phase2_state.quarter_states)
    feature_mean = history_feature_matrix.mean(axis=0)
    feature_std = history_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    quarter_feature_map = {
        quarter: (history_feature_matrix[idx, :] - feature_mean) / feature_std
        for idx, quarter in enumerate(phase2_state.quarter_axis)
    }

    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    active_winner_configs = dict(winner_configs or WINNER_CONFIGS)
    contract_payloads: dict[str, Any] = {}
    terminal_delta_rows: list[dict[str, Any]] = []
    final_forecast_quarters: list[str] = []
    for contract_name, config in active_winner_configs.items():
        result = _collect_contract_result(
            archive_run_id=readout_source_archive_run,
            contract_name=str(config["suite_contract"]),
            experiment_id=str(config["winner_id"]),
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
        )
        readout = _fit_residual_readout(
            result=result,
            allowed_tiers=set(config["allowed_tiers"]),
            quarter_feature_map=quarter_feature_map,
            feature_names=_feature_names(phase2_state.block_axis),
        )
        observation_rows, _, _ = champion_forecast._contract_payload(archive_run, str(config["forecast_contract"]))
        latest_quarter = max((str(row["quarter"]) for row in observation_rows), key=suite.quarter_sort_key)
        forecast_quarters = champion_forecast._future_quarters(latest_quarter, int(forecast_horizon_quarters))
        final_forecast_quarters = list(forecast_quarters)
        forecast_candidate = champion_forecast._forecast_future_rows(
            observation_rows,
            annual_rows=annual_rows,
            spec=spec_map[str(config["winner_id"])],
            best_candidate=dict(result["best_candidate"]),
            forecast_quarters=forecast_quarters,
        )
        base_rows = [dict(row) for row in list(forecast_candidate.get("prediction_rows") or [])]
        base_feature_path = (_feature_matrix(np.vstack([last_state[None, :], scenario_paths["status_quo"]]))[1:, :] - feature_mean) / feature_std
        scenario_rows: dict[str, list[dict[str, Any]]] = {"status_quo": [dict(row) for row in base_rows]}
        for scenario_name in SCENARIO_ORDER[1:]:
            scenario_feature_path = (_feature_matrix(np.vstack([last_state[None, :], scenario_paths[scenario_name]]))[1:, :] - feature_mean) / feature_std
            adjusted_rows = _apply_scenario_to_forecast(
                base_rows=base_rows,
                base_features=base_feature_path,
                scenario_features=scenario_feature_path,
                readout=readout,
            )
            scenario_rows[scenario_name] = adjusted_rows
        graph_path = analysis_dir / f"{contract_name}_phase2_seeded_scenarios.png"
        _plot_contract_scenarios(
            title=f"{config['title']} with Phase 2 seeded scenarios",
            observation_rows=observation_rows,
            base_rows=base_rows,
            scenario_rows=scenario_rows,
            path=graph_path,
        )
        for scenario_name in SCENARIO_ORDER[1:]:
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
            "suite_contract": str(config["suite_contract"]),
            "forecast_contract": str(config["forecast_contract"]),
            "quarterly_mean_mae": float(result["quarterly_summary"]["candidate_mean_mae"]),
            "readout": readout,
            "base_forecast_rows": base_rows,
            "scenario_rows": scenario_rows,
            "graph_file": graph_path.name,
        }

    structural_path = analysis_dir / "phase2_seeded_structural_scenarios.png"
    _plot_structural_history_and_scenarios(
        quarter_axis=phase2_state.quarter_axis,
        block_axis=phase2_state.block_axis,
        history_states=phase2_state.quarter_states,
        future_quarters=final_forecast_quarters,
        scenario_paths=scenario_paths,
        path=structural_path,
    )
    readout_path = analysis_dir / "phase2_seeded_readout_heatmap.png"
    _plot_readout_heatmap(payloads={name: dict(payload["readout"]) for name, payload in contract_payloads.items()}, path=readout_path)
    terminal_path = analysis_dir / "phase2_seeded_terminal_delta_heatmap.png"
    _plot_terminal_delta_heatmap(scenario_terminal_rows=terminal_delta_rows, path=terminal_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "readout_source_archive_run_id": readout_source_archive_run,
        "phase2_monthly_run_id": phase2_run,
        "active_block_subset": list(phase2_state.block_axis),
        "forecast_horizon_quarters": int(forecast_horizon_quarters),
        "phase2_seed": {
            "source_block_axis": list(source_phase2_state.block_axis),
            "block_axis": list(phase2_state.block_axis),
            "quarter_axis": list(phase2_state.quarter_axis),
            "edge_rows": [dict(row) for row in phase2_state.edge_rows],
            "edge_count": int(len(phase2_state.edge_rows)),
        },
        "winner_configs": {
            str(contract_name): {
                **dict(config),
                "allowed_tiers": sorted(str(value) for value in list(config.get("allowed_tiers") or [])),
            }
            for contract_name, config in active_winner_configs.items()
        },
        "scenarios": {name: {"description": SCENARIO_DESCRIPTIONS[name]} for name in SCENARIO_ORDER},
        "contracts": contract_payloads,
        "terminal_delta_rows": terminal_delta_rows,
        "artifacts": {
            "structural_graph": structural_path.name,
            "readout_heatmap": readout_path.name,
            "terminal_delta_heatmap": terminal_path.name,
            **{f"{name}_scenario_graph": str(payload["graph_file"]) for name, payload in contract_payloads.items()},
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_seeded_champion_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_seeded_champion_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Phase 2 seeded scenario layer on the frozen Phase 3 champions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--readout-source-archive-run-id", default=None)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    parser.add_argument("--active-block", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_seeded_champion_batch(
        run_id=str(args.run_id),
        archive_run_id=args.archive_run_id,
        readout_source_archive_run_id=args.readout_source_archive_run_id,
        monthly_phase2_run_id=args.monthly_phase2_run_id,
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
        active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
