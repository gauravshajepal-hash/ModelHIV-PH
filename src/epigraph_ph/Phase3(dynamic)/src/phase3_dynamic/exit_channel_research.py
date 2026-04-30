from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    EXTERNAL_EXIT_CHANNEL_NAMES,
    BlockedTimeDataset,
    build_blocked_time_dataset,
    build_observation_rows,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
)
from .incidence import fit_incidence_flow_paths
from .metrics import normalized_mae, quarter_sort_key
from .model import fit_dynamic_hazard_paths, simulate_holdout
from .runtime import ensure_dir, write_json
from .scenario_lab import (
    DEFAULT_ACTIVE_SOURCE_RUN_ID,
    DEFAULT_BASELINE_SOURCE_RUN_ID,
    _load_reference_config,
)


EXIT_CHANNEL_RESEARCH_SCHEMA_VERSION = "phase3_dynamic_exit_channel_research.v1"
CHANNEL_ABLATIONS: tuple[str, ...] = (
    "mortality_removal",
    "treatment_non_initiation",
    "unresolved_external_removal",
    "art_ltfu",
    "reengagement",
    "vl_testing_loss",
)


def _zero_external_channel(
    channel_state_map: dict[str, dict[str, dict[str, float]]],
    channel_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        quarter: {
            channel: {
                state: 0.0 if channel == channel_name else max(float(value), 0.0)
                for state, value in dict(states or {}).items()
            }
            for channel, states in dict(channels or {}).items()
        }
        for quarter, channels in channel_state_map.items()
    }


def _zero_transition(
    hazard_map: dict[str, dict[str, float]],
    transition_name: str,
) -> dict[str, dict[str, float]]:
    return {
        quarter: {
            transition: 0.0 if transition == transition_name else float(value)
            for transition, value in dict(values or {}).items()
        }
        for quarter, values in hazard_map.items()
    }


def _simulate_with_paths(
    dataset: BlockedTimeDataset,
    hazard_map: dict[str, dict[str, float]],
    incidence_paths: dict[str, Any],
    *,
    exit_channel_state_outflow_map: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    return simulate_holdout(
        dataset,
        hazard_map,
        incidence_inflow_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
        incidence_hazard_map=dict(incidence_paths.get("holdout_incidence_hazard_map") or {}),
        incidence_cap_map=dict(incidence_paths.get("holdout_incidence_cap_map") or {}),
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=(
            dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {})
            if exit_channel_state_outflow_map is None
            else exit_channel_state_outflow_map
        ),
    )


def _vl_testing_loss_summary(dataset: BlockedTimeDataset) -> dict[str, Any]:
    losses = []
    for row in dataset.holdout_rows:
        tested = row.get("tested_for_viral_load")
        art = row.get("alive_on_art")
        if tested is None or art is None:
            continue
        losses.append(max(float(art) - float(tested), 0.0))
    return {
        "status": "observed_auxiliary_channel" if losses else "not_observed_in_split",
        "holdout_observed_mean_loss": float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else None,
        "primary_gate_effect": "not_evaluated_in_primary_mae",
        "reason": "tested_for_viral_load is not part of PRIMARY_METRICS; it is an observation-channel diagnostic.",
    }


def _evaluate_split(
    *,
    observation_rows: list[dict[str, Any]],
    split: dict[str, Any],
    reference_config: dict[str, Any],
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
    base_hazard_map = dict(hazard_paths.get("holdout_hazard_map") or {})
    base_exit_map = dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {})
    all_channels = _simulate_with_paths(dataset, base_hazard_map, incidence_paths)
    all_mae = float(all_channels.get("mae") or float("inf"))
    rows: list[dict[str, Any]] = []
    for channel_name in CHANNEL_ABLATIONS:
        if channel_name in EXTERNAL_EXIT_CHANNEL_NAMES:
            candidate = _simulate_with_paths(
                dataset,
                base_hazard_map,
                incidence_paths,
                exit_channel_state_outflow_map=_zero_external_channel(base_exit_map, channel_name),
            )
            ablated_mae = float(candidate.get("mae") or float("inf"))
            rows.append(
                {
                    "channel": channel_name,
                    "channel_class": "external_stock_exit",
                    "all_channel_mae": all_mae,
                    "ablated_mae": ablated_mae,
                    "ablation_minus_all_mae": float(ablated_mae - all_mae),
                    "included_channel_helped_primary_gate": bool(ablated_mae >= all_mae),
                }
            )
        elif channel_name == "art_ltfu":
            candidate = _simulate_with_paths(dataset, _zero_transition(base_hazard_map, "A_to_L"), incidence_paths)
            ablated_mae = float(candidate.get("mae") or float("inf"))
            rows.append(
                {
                    "channel": channel_name,
                    "channel_class": "care_state_transition",
                    "all_channel_mae": all_mae,
                    "ablated_mae": ablated_mae,
                    "ablation_minus_all_mae": float(ablated_mae - all_mae),
                    "included_channel_helped_primary_gate": bool(ablated_mae >= all_mae),
                }
            )
        elif channel_name == "reengagement":
            candidate = _simulate_with_paths(dataset, _zero_transition(base_hazard_map, "L_to_A"), incidence_paths)
            ablated_mae = float(candidate.get("mae") or float("inf"))
            rows.append(
                {
                    "channel": channel_name,
                    "channel_class": "care_state_transition",
                    "all_channel_mae": all_mae,
                    "ablated_mae": ablated_mae,
                    "ablation_minus_all_mae": float(ablated_mae - all_mae),
                    "included_channel_helped_primary_gate": bool(ablated_mae >= all_mae),
                }
            )
        else:
            rows.append(
                {
                    "channel": channel_name,
                    "channel_class": "observation_channel",
                    "all_channel_mae": all_mae,
                    "ablated_mae": None,
                    "ablation_minus_all_mae": None,
                    "included_channel_helped_primary_gate": None,
                    "auxiliary_summary": _vl_testing_loss_summary(dataset),
                }
            )
    return {
        "train_end_year": int(split["train_end_year"]),
        "holdout_years": list(split["holdout_years"]),
        "all_channel_mae": all_mae,
        "rows": rows,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for channel_name in CHANNEL_ABLATIONS:
        channel_rows = [row for row in rows if row["channel"] == channel_name and row.get("ablation_minus_all_mae") is not None]
        if not channel_rows:
            aux_rows = [row for row in rows if row["channel"] == channel_name]
            summaries.append(
                {
                    "channel": channel_name,
                    "evaluated_in_primary_gate": False,
                    "reason": str((aux_rows[0].get("auxiliary_summary") or {}).get("reason") or "not evaluated") if aux_rows else "not evaluated",
                }
            )
            continue
        deltas = np.asarray([float(row["ablation_minus_all_mae"]) for row in channel_rows], dtype=np.float64)
        summaries.append(
            {
                "channel": channel_name,
                "evaluated_in_primary_gate": True,
                "mean_ablation_minus_all_mae": float(np.mean(deltas)),
                "median_ablation_minus_all_mae": float(np.median(deltas)),
                "helped_split_count": int(sum(1 for value in deltas if value >= 0.0)),
                "regressed_split_count": int(sum(1 for value in deltas if value < 0.0)),
                "split_count": int(deltas.size),
            }
        )
    return summaries


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    summaries = [row for row in payload.get("summary", []) if row.get("evaluated_in_primary_gate")]
    labels = [str(row["channel"]) for row in summaries]
    deltas = [float(row.get("mean_ablation_minus_all_mae") or 0.0) for row in summaries]
    colors = ["#2f6f73" if value >= 0.0 else "#b23a48" for value in deltas]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, deltas, color=colors, width=0.68)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Ablated MAE - all-channel MAE")
    ax.set_title("Typed Exit/Care Channel Ablation", loc="left", fontweight="bold")
    ax.text(
        0.01,
        0.98,
        "Positive = channel helps primary blocked-time fit",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def run_exit_channel_research(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run = source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID
    baseline_run = baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run,
        baseline_source_run_id=baseline_run,
    )
    reference_config = _load_reference_config(None if reference_report_path is None else Path(reference_report_path))
    splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    split_reports = [
        report
        for split in splits
        if (report := _evaluate_split(observation_rows=observation_rows, split=split, reference_config=reference_config)) is not None
    ]
    flat_rows = [row for report in split_reports for row in list(report.get("rows") or [])]
    summary = _summarize(flat_rows)
    analysis_dir = sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis"
    ensure_dir(analysis_dir)
    report_path = analysis_dir / "exit_channel_research_report.json"
    dashboard_path = analysis_dir / "exit_channel_ablation_dashboard.png"
    payload = {
        "schema_version": EXIT_CHANNEL_RESEARCH_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run,
        "baseline_source_run_id": baseline_run,
        "reference_config": {
            "source": reference_config.get("source"),
            "path": reference_config.get("path"),
        },
        "contract": {
            "primary_question": "Does including each typed channel improve blocked-time primary endpoint fit versus channel ablation?",
            "positive_delta_interpretation": "ablated_mae - all_channel_mae >= 0 means the included channel helped or did not harm the primary blocked-time fit",
            "vl_testing_loss_scope": "auxiliary observation channel; not scored in PRIMARY_METRICS",
        },
        "split_count": len(split_reports),
        "summary": summary,
        "splits": split_reports,
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_dashboard(payload, dashboard_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="phase3-dynamic-exit-channel-research")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    args = parser.parse_args()
    payload = run_exit_channel_research(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        reference_report_path=args.reference_report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
        horizon_years=args.horizon_years,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
