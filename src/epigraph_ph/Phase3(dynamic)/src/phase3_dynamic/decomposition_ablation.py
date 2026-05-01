from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import build_observation_rows, default_epigraph_root, rolling_origin_splits, sandbox_repo_root
from .decomposition import DecompositionControlConfig, STREAM_TO_SUPPORT_METRIC
from .decomposition_research import (
    _evaluate_split,
    _long_horizon_status,
    _promotion_gate,
    _score_gate,
    _score_supported_back_half_gate,
    _score_supported_rate_gate,
)
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .runtime import ensure_dir, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID, _load_reference_config


DECOMPOSITION_ABLATION_SCHEMA_VERSION = "phase3_dynamic_decomposition_ablation.v1"


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "full",
            "variant_class": "full_reference",
            "config": DecompositionControlConfig(),
            "hypothesis": "all train-origin decomposition controls are active",
        },
        {
            "variant_id": "trend_only",
            "variant_class": "component_only",
            "config": DecompositionControlConfig(use_reporting_support_shift=False, use_residual_shock=False),
            "hypothesis": "slow trajectory drift alone carries the gain",
        },
        {
            "variant_id": "support_shift_only",
            "variant_class": "component_only",
            "config": DecompositionControlConfig(use_trend=False, use_residual_shock=False),
            "hypothesis": "observation support/reporting shifts alone carry the gain",
        },
        {
            "variant_id": "shock_only",
            "variant_class": "component_only",
            "config": DecompositionControlConfig(use_trend=False, use_reporting_support_shift=False),
            "hypothesis": "residual shock continuation alone carries the gain",
        },
        {
            "variant_id": "lean_shock_art",
            "variant_class": "lean_promoted_diagnostic",
            "config": DecompositionControlConfig(
                use_trend=False,
                use_reporting_support_shift=False,
                enabled_streams=("art",),
            ),
            "hypothesis": "the ablation-promoted minimal branch: ART stream residual shocks only",
        },
        {
            "variant_id": "lean_shock_art_back_half",
            "variant_class": "lean_back_half_probe",
            "config": DecompositionControlConfig(
                use_trend=False,
                use_reporting_support_shift=False,
                enabled_streams=("art", "vl", "suppression"),
            ),
            "hypothesis": "ART residual shocks plus explicit VL/suppression residual-shock controls",
        },
        {
            "variant_id": "without_trend",
            "variant_class": "component_removal",
            "config": DecompositionControlConfig(use_trend=False),
            "hypothesis": "drop only the slow trend component",
        },
        {
            "variant_id": "without_support_shift",
            "variant_class": "component_removal",
            "config": DecompositionControlConfig(use_reporting_support_shift=False),
            "hypothesis": "drop only the reporting/support-shift component",
        },
        {
            "variant_id": "without_residual_shock",
            "variant_class": "component_removal",
            "config": DecompositionControlConfig(use_residual_shock=False),
            "hypothesis": "drop only the residual-shock component",
        },
        *[
            {
                "variant_id": f"without_{stream_name}",
                "variant_class": "stream_removal",
                "removed_stream": stream_name,
                "config": DecompositionControlConfig(disabled_streams=(stream_name,)),
                "hypothesis": f"drop the {stream_name} stream while keeping other controls active",
            }
            for stream_name in STREAM_TO_SUPPORT_METRIC
        ],
    ]


def _config_payload(cfg: DecompositionControlConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    if payload.get("enabled_streams") is not None:
        payload["enabled_streams"] = list(payload["enabled_streams"])
    payload["disabled_streams"] = list(payload.get("disabled_streams") or [])
    return payload


def _evaluate_variant(
    *,
    variant: dict[str, Any],
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    one_year_splits: list[dict[str, Any]],
    five_year_splits: list[dict[str, Any]],
    scenario_start_year: int,
    scenario_end_year: int,
) -> dict[str, Any]:
    cfg = variant["config"]
    one_year_rows = [
        row
        for split in one_year_splits
        if (row := _evaluate_split(
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            split=split,
            reference_config=reference_config,
            decomposition_cfg=cfg,
        )) is not None
    ]
    five_year_rows = [
        row
        for split in five_year_splits
        if (row := _evaluate_split(
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            split=split,
            reference_config=reference_config,
            decomposition_cfg=cfg,
        )) is not None
    ]
    one_year_gate = _score_gate(one_year_rows, gate_name="one_year_blocked_time")
    five_year_gate = _score_gate(five_year_rows, gate_name="five_year_bounded_analog")
    one_year_back_half_gate = _score_supported_back_half_gate(
        one_year_rows,
        gate_name="one_year_support_aware_vl_suppression",
    )
    five_year_back_half_gate = _score_supported_back_half_gate(
        five_year_rows,
        gate_name="five_year_support_aware_vl_suppression",
    )
    one_year_rate_gate = _score_supported_rate_gate(
        one_year_rows,
        gate_name="one_year_support_aware_back_half_rates",
    )
    five_year_rate_gate = _score_supported_rate_gate(
        five_year_rows,
        gate_name="five_year_support_aware_back_half_rates",
    )
    long_horizon = _long_horizon_status(
        observation_rows=observation_rows,
        constraint_rows=constraint_rows,
        reference_config=reference_config,
        decomposition_cfg=cfg,
        scenario_start_year=scenario_start_year,
        scenario_end_year=scenario_end_year,
    )
    promotion = _promotion_gate(
        one_year_gate,
        five_year_gate,
        long_horizon,
        back_half_gate=one_year_back_half_gate,
        conditional_rate_gate=one_year_rate_gate,
    )
    return {
        "variant_id": variant["variant_id"],
        "variant_class": variant["variant_class"],
        "removed_stream": variant.get("removed_stream"),
        "hypothesis": variant["hypothesis"],
        "config": _config_payload(cfg),
        "one_year_gate": one_year_gate,
        "five_year_gate": five_year_gate,
        "one_year_support_aware_back_half_gate": one_year_back_half_gate,
        "five_year_support_aware_back_half_gate": five_year_back_half_gate,
        "one_year_support_aware_rate_gate": one_year_rate_gate,
        "five_year_support_aware_rate_gate": five_year_rate_gate,
        "long_horizon_status": long_horizon,
        "promotion_gate": promotion,
    }


def _attach_full_deltas(variant_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = next((row for row in variant_reports if row.get("variant_id") == "full"), None)
    if full is None:
        return variant_reports
    full_one = float((full.get("one_year_gate") or {}).get("candidate_mean_mae") or np.nan)
    full_five = float((full.get("five_year_gate") or {}).get("candidate_mean_mae") or np.nan)
    for row in variant_reports:
        one = float((row.get("one_year_gate") or {}).get("candidate_mean_mae") or np.nan)
        five = float((row.get("five_year_gate") or {}).get("candidate_mean_mae") or np.nan)
        one_delta = float(one - full_one)
        five_delta = float(five - full_five)
        row["ablation_summary"] = {
            "one_year_delta_vs_full": one_delta,
            "five_year_delta_vs_full": five_delta,
            "positive_delta_means_full_control_helped": True,
            "interpretation": _interpret_delta(row, one_delta=one_delta, five_delta=five_delta),
        }
    return variant_reports


def _interpret_delta(row: dict[str, Any], *, one_delta: float, five_delta: float) -> str:
    variant_class = str(row.get("variant_class") or "")
    tolerance = 1e-6
    one_sign = 1 if one_delta > tolerance else -1 if one_delta < -tolerance else 0
    five_sign = 1 if five_delta > tolerance else -1 if five_delta < -tolerance else 0
    if str(row.get("variant_id") or "") == "full":
        return "reference decomposition branch"
    if variant_class in {"stream_removal", "component_removal"}:
        if one_sign == 0 and five_sign == 0:
            return "removed control is numerically indistinguishable from full"
        if one_sign > 0 and five_sign > 0:
            return "removed control carries gain on both gates"
        if one_sign < 0 and five_sign < 0:
            return "removed control appears harmful on both gates"
        return "removed control has horizon-dependent effect"
    if variant_class == "component_only":
        gate = dict(row.get("promotion_gate") or {})
        return "standalone component passes all gates" if gate.get("promotion_eligible") else "standalone component is diagnostic only"
    return "diagnostic"


def _summary_rows(variant_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in variant_reports:
        one = dict(row.get("one_year_gate") or {})
        five = dict(row.get("five_year_gate") or {})
        summary = dict(row.get("ablation_summary") or {})
        promotion = dict(row.get("promotion_gate") or {})
        one_back = dict(row.get("one_year_support_aware_back_half_gate") or {})
        five_back = dict(row.get("five_year_support_aware_back_half_gate") or {})
        one_rate = dict(row.get("one_year_support_aware_rate_gate") or {})
        five_rate = dict(row.get("five_year_support_aware_rate_gate") or {})
        rows.append(
            {
                "variant_id": row.get("variant_id"),
                "variant_class": row.get("variant_class"),
                "removed_stream": row.get("removed_stream"),
                "one_year_candidate_mean_mae": one.get("candidate_mean_mae"),
                "five_year_candidate_mean_mae": five.get("candidate_mean_mae"),
                "one_year_delta_vs_full": summary.get("one_year_delta_vs_full"),
                "five_year_delta_vs_full": summary.get("five_year_delta_vs_full"),
                "one_year_back_half_candidate_mean_mae": one_back.get("candidate_mean_mae"),
                "one_year_back_half_delta_vs_reference": one_back.get("candidate_minus_baseline_mean_mae"),
                "one_year_back_half_status": one_back.get("status"),
                "one_year_back_half_claim_status": one_back.get("claim_status"),
                "five_year_back_half_candidate_mean_mae": five_back.get("candidate_mean_mae"),
                "five_year_back_half_delta_vs_reference": five_back.get("candidate_minus_baseline_mean_mae"),
                "five_year_back_half_status": five_back.get("status"),
                "five_year_back_half_claim_status": five_back.get("claim_status"),
                "one_year_conditional_rate_candidate_mean_mae": one_rate.get("candidate_mean_mae"),
                "one_year_conditional_rate_delta_vs_reference": one_rate.get("candidate_minus_baseline_mean_mae"),
                "one_year_conditional_rate_status": one_rate.get("status"),
                "one_year_conditional_rate_claim_status": one_rate.get("claim_status"),
                "five_year_conditional_rate_candidate_mean_mae": five_rate.get("candidate_mean_mae"),
                "five_year_conditional_rate_delta_vs_reference": five_rate.get("candidate_minus_baseline_mean_mae"),
                "five_year_conditional_rate_status": five_rate.get("status"),
                "five_year_conditional_rate_claim_status": five_rate.get("claim_status"),
                "promotion_status": promotion.get("status"),
                "interpretation": summary.get("interpretation"),
            }
        )
    return rows


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    summaries = list(payload.get("summary") or [])
    component_rows = [row for row in summaries if row.get("variant_id") in {"full", "trend_only", "support_shift_only", "shock_only"}]
    removal_rows = [row for row in summaries if row.get("variant_class") in {"stream_removal", "component_removal"}]
    fig, axis_grid = plt.subplots(2, 2, figsize=(14.8, 9.0), constrained_layout=True)
    axes = axis_grid.ravel()

    labels = [str(row["variant_id"]) for row in component_rows]
    x = np.arange(len(labels))
    one_values = [float(row.get("one_year_candidate_mean_mae") or 0.0) for row in component_rows]
    five_values = [float(row.get("five_year_candidate_mean_mae") or 0.0) for row in component_rows]
    axes[0].bar(x - 0.18, one_values, width=0.36, color="#b23a48", label="1-year")
    axes[0].bar(x + 0.18, five_values, width=0.36, color="#2f6f73", label="5-year")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Mean normalized MAE")
    axes[0].set_title("A. Component-Only Controls", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    removal_labels = [str(row["variant_id"]).replace("without_", "-") for row in removal_rows]
    y = np.arange(len(removal_labels))
    one_delta = [float(row.get("one_year_delta_vs_full") or 0.0) for row in removal_rows]
    five_delta = [float(row.get("five_year_delta_vs_full") or 0.0) for row in removal_rows]
    axes[1].barh(y - 0.18, one_delta, height=0.36, color="#b23a48", label="1-year")
    axes[1].barh(y + 0.18, five_delta, height=0.36, color="#2f6f73", label="5-year")
    axes[1].axvline(0.0, color="#111827", linewidth=0.9)
    axes[1].set_yticks(y, removal_labels)
    axes[1].set_xlabel("MAE delta vs full decomposition")
    axes[1].set_title("B. Removal Ablations", loc="left", fontweight="bold")
    axes[1].text(
        0.01,
        0.02,
        "Positive = removed control was useful",
        transform=axes[1].transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )
    axes[1].legend(frameon=False, loc="lower right")

    support_rows = [
        row
        for row in summaries
        if row.get("variant_id") in {"full", "shock_only", "lean_shock_art", "lean_shock_art_back_half"}
    ]
    support_labels = [str(row["variant_id"]).replace("lean_", "") for row in support_rows]
    support_x = np.arange(len(support_labels))
    back_one = [float(row.get("one_year_back_half_delta_vs_reference") or 0.0) for row in support_rows]
    back_five = [float(row.get("five_year_back_half_delta_vs_reference") or 0.0) for row in support_rows]
    axes[2].bar(support_x - 0.18, back_one, width=0.36, color="#6f4e37", label="1-year")
    axes[2].bar(support_x + 0.18, back_five, width=0.36, color="#3f7cac", label="5-year")
    axes[2].axhline(0.0, color="#111827", linewidth=0.9)
    axes[2].set_xticks(support_x, support_labels, rotation=30, ha="right")
    axes[2].set_ylabel("Support-aware delta vs reference")
    axes[2].set_title("C. VL/Suppression Gate", loc="left", fontweight="bold")
    axes[2].text(
        0.01,
        0.02,
        "Negative = supported back-half improvement",
        transform=axes[2].transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )
    axes[2].legend(frameon=False)

    rate_rows = support_rows
    rate_labels = [str(row["variant_id"]).replace("lean_", "") for row in rate_rows]
    rate_x = np.arange(len(rate_labels))
    rate_one = [float(row.get("one_year_conditional_rate_delta_vs_reference") or 0.0) for row in rate_rows]
    rate_five = [float(row.get("five_year_conditional_rate_delta_vs_reference") or 0.0) for row in rate_rows]
    axes[3].bar(rate_x - 0.18, rate_one, width=0.36, color="#6f4e37", label="1-year")
    axes[3].bar(rate_x + 0.18, rate_five, width=0.36, color="#3f7cac", label="5-year")
    axes[3].axhline(0.0, color="#111827", linewidth=0.9)
    axes[3].set_xticks(rate_x, rate_labels, rotation=30, ha="right")
    axes[3].set_ylabel("Conditional-rate delta vs reference")
    axes[3].set_title("D. VL/ART And Suppressed/VL Rates", loc="left", fontweight="bold")
    axes[3].text(
        0.01,
        0.02,
        "Negative = true back-half rate improvement",
        transform=axes[3].transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(axis="x" if ax is axes[1] else "y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 HIV Decomposition Ablation", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def run_decomposition_ablation(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    scenario_start_year: int = 2026,
    scenario_end_year: int = 2035,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_run = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    observation_rows = build_observation_rows(epigraph_root, source_run_id=source_run, baseline_source_run_id=baseline_run)
    constraint_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run,
        baseline_source_run_id=baseline_run,
        include_validation_only=True,
    )
    reference_config = _load_reference_config(None if reference_report_path is None else Path(reference_report_path))
    one_year_splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=1,
    )
    five_year_splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=5,
    )
    variant_reports = [
        _evaluate_variant(
            variant=variant,
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            reference_config=reference_config,
            one_year_splits=one_year_splits,
            five_year_splits=five_year_splits,
            scenario_start_year=scenario_start_year,
            scenario_end_year=scenario_end_year,
        )
        for variant in _variant_specs()
    ]
    variant_reports = _attach_full_deltas(variant_reports)
    summary = _summary_rows(variant_reports)
    promoted = [row["variant_id"] for row in variant_reports if (row.get("promotion_gate") or {}).get("promotion_eligible")]
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "decomposition_ablation_report.json"
    dashboard_path = analysis_dir / "decomposition_ablation_dashboard.png"
    payload = {
        "schema_version": DECOMPOSITION_ABLATION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run,
        "baseline_source_run_id": baseline_run,
        "reference_config": {
            "source": reference_config.get("source"),
            "path": reference_config.get("path"),
        },
        "contract": {
            "positive_removal_delta": "variant MAE minus full MAE; positive means the removed stream/component carried predictive gain",
            "promotion_rule": "each variant is promoted only if it passes one-year, five-year, long-horizon, supported back-half count, and supported conditional-rate gates against current reference and carry-forward where comparable",
            "conditional_rate_gate": "scores tested_for_viral_load / alive_on_art and virally_suppressed / tested_for_viral_load only when numerator and denominator have direct exact/bridge support with train-origin support",
        },
        "promoted_variants": promoted,
        "summary": summary,
        "variants": variant_reports,
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_dashboard(payload, dashboard_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="phase3-dynamic-decomposition-ablation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--scenario-start-year", type=int, default=2026)
    parser.add_argument("--scenario-end-year", type=int, default=2035)
    args = parser.parse_args()
    payload = run_decomposition_ablation(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        reference_report_path=args.reference_report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
        scenario_start_year=args.scenario_start_year,
        scenario_end_year=args.scenario_end_year,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
