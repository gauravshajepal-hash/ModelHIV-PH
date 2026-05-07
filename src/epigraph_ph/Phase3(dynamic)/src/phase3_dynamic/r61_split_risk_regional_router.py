from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .r52_subnational_coherence_gate import _coherence_rows
from .r54_national_total_regional_adapter import (
    R48_DEFAULT_REPORT,
    R49_DEFAULT_REPORT,
    R51_DEFAULT_REPORT,
    _base_prediction_rows,
    _load_report,
)
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
)
from .r56_split_guarded_regional_selector import (
    _prior_candidate_score_rows,
    _select_family_from_prior_scores,
)
from .r60_regional_experiment_queue import (
    CARRY_FORWARD_FAMILY,
    REFERENCE_FAMILY,
    R54_DEFAULT_REPORT,
    R59_DEFAULT_REPORT,
    R60_DEFAULT_REPORT,
    _candidate_mean_gate,
    _family_row,
    _prediction_rows_for_spec,
    _score_candidate,
    _write_csv,
)
from .runtime import ensure_dir, read_json, write_json


R61_SCHEMA_VERSION = "phase3_dynamic.r61_split_risk_regional_router.v1"
R61_RUN_ID = "p3d-r61-split-risk-regional-router-20260503-s00"
R61_FAMILY_PREFIX = "split_risk_regional_router"


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _rename_family(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        output["candidate_family"] = family
        renamed.append(output)
    return renamed


def _copy_metric_from_family(
    base_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    *,
    metric_family: dict[str, str],
    candidate_family: str,
) -> list[dict[str, Any]]:
    indexed = _prediction_index(source_rows)
    output_rows: list[dict[str, Any]] = []
    for row in base_rows:
        output = dict(row)
        period = str(row.get("holdout_period") or "")
        region = str(row.get("region") or "")
        output["candidate_family"] = candidate_family
        for metric, family in metric_family.items():
            value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
            if value is not None:
                output[metric] = float(value)
                output[f"{metric}_source_family"] = family
        output_rows.append(_projection_cascade(output))
    return output_rows


def _periods(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")}, key=quarter_sort_key)


def _families_complete_for_period(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    *,
    period: str,
    regions: list[str],
    families: list[str],
) -> list[str]:
    return [
        family
        for family in families
        if all((family, period, region) in indexed for region in regions)
    ]


def _train_prior_metric_router_rows(
    candidate_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    candidate_families: list[str],
    output_family: str,
    default_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(candidate_rows)
    periods = _periods(candidate_rows)
    _raw, regional_score_rows = _score_predictions(rows_by_key, candidate_rows)
    coherence_rows = _coherence_rows(rows_by_key, candidate_rows)
    selection_rows: list[dict[str, Any]] = []
    prior_score_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if candidate_period == period and family == default_family})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        available_families = _families_complete_for_period(
            indexed,
            period=period,
            regions=regions,
            families=candidate_families,
        )
        period_rows: dict[str, dict[str, Any]] = {
            region: {
                "candidate_family": output_family,
                "holdout_period": period,
                "region": region,
            }
            for region in regions
        }
        for metric in COUNT_METRICS:
            priors = _prior_candidate_score_rows(
                candidate_families=available_families,
                regional_score_rows=regional_score_rows,
                coherence_rows=coherence_rows,
                holdout_period=period,
                metric=metric,
                carry_family=CARRY_FORWARD_FAMILY,
                reference_family=REFERENCE_FAMILY,
            )
            chosen, reason = _select_family_from_prior_scores(priors, default_family=default_family)
            if chosen not in available_families and available_families:
                chosen = default_family if default_family in available_families else available_families[0]
                reason = "fallback_to_available_default"
            prior_score_rows.extend(priors)
            selection_rows.append(
                {
                    "candidate_family": output_family,
                    "holdout_period": period,
                    "metric_name": metric,
                    "selected_source_family": chosen,
                    "selection_reason": reason,
                    "available_family_count": len(available_families),
                    "prior_score_row_count": len(priors),
                }
            )
            for region in regions:
                value = _finite_float((indexed.get((chosen, period, region)) or {}).get(metric))
                if value is not None:
                    period_rows[region][metric] = float(value)
                    period_rows[region][f"{metric}_source_family"] = chosen
        for region in regions:
            output_rows.append(_projection_cascade(period_rows[region]))
    return output_rows, selection_rows, prior_score_rows


def _r61_experiment_specs(r60_best_family: str, r54_promoted_family: str) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "R61-001",
            "mode": "best_r60_replay",
            "description": "Replay the R60 best mean-promoted candidate under the stricter R61 accounting.",
        },
        {
            "experiment_id": "R61-002",
            "mode": "metric_backoff",
            "metric_family": {
                "tested_for_viral_load": CARRY_FORWARD_FAMILY,
                "virally_suppressed": CARRY_FORWARD_FAMILY,
            },
            "description": "Keep R60 best except force VL-tested and suppressed counts to locked regional carry-forward.",
        },
        {
            "experiment_id": "R61-003",
            "mode": "metric_backoff",
            "metric_family": {
                "alive_on_art": CARRY_FORWARD_FAMILY,
                "tested_for_viral_load": CARRY_FORWARD_FAMILY,
                "virally_suppressed": CARRY_FORWARD_FAMILY,
            },
            "description": "Keep R60 front-half streams but back off the ART/VL/suppression service stream to carry-forward.",
        },
        {
            "experiment_id": "R61-004",
            "mode": "metric_backoff",
            "metric_family": {
                "diagnosed_plhiv": REFERENCE_FAMILY,
                "alive_on_art": CARRY_FORWARD_FAMILY,
                "tested_for_viral_load": CARRY_FORWARD_FAMILY,
                "virally_suppressed": CARRY_FORWARD_FAMILY,
            },
            "description": "Use module-local diagnosed stock plus carry-forward service counts; keep only R60 estimated PLHIV.",
        },
        {
            "experiment_id": "R61-005",
            "mode": "train_prior_metric_router",
            "default_family": REFERENCE_FAMILY,
            "candidate_families": [
                r60_best_family,
                "anchor_stable_pareto_ensemble",
                r54_promoted_family,
                REFERENCE_FAMILY,
                CARRY_FORWARD_FAMILY,
                "R61-002__metric_backoff",
                "R61-003__metric_backoff",
                "R61-004__metric_backoff",
            ],
            "description": "Train-origin metric router with a conservative module-local default.",
        },
        {
            "experiment_id": "R61-006",
            "mode": "train_prior_metric_router",
            "default_family": r60_best_family,
            "candidate_families": [
                r60_best_family,
                "anchor_stable_pareto_ensemble",
                r54_promoted_family,
                REFERENCE_FAMILY,
                CARRY_FORWARD_FAMILY,
                "R61-002__metric_backoff",
                "R61-003__metric_backoff",
                "R61-004__metric_backoff",
            ],
            "description": "Train-origin metric router with R60 best as the no-prior default.",
        },
        {
            "experiment_id": "R61-007",
            "mode": "train_prior_metric_router",
            "default_family": REFERENCE_FAMILY,
            "candidate_families": [
                r60_best_family,
                REFERENCE_FAMILY,
                CARRY_FORWARD_FAMILY,
            ],
            "description": "Minimal no-regret router over R60 best, module-local, and carry-forward only.",
        },
        {
            "experiment_id": "R61-008",
            "mode": "metric_backoff",
            "metric_family": {
                "diagnosed_plhiv": r54_promoted_family,
                "alive_on_art": r54_promoted_family,
                "tested_for_viral_load": CARRY_FORWARD_FAMILY,
                "virally_suppressed": CARRY_FORWARD_FAMILY,
            },
            "description": "Use the R54 national-total adapter for diagnosed/ART stocks and carry-forward conditional back-half counts.",
        },
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("router_gate") or {})
    lines = [
        "# Phase 3 R61 Split-Risk Regional Router",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Experiment count: `{gate.get('experiment_count')}`",
        f"- Mean-pass count: `{gate.get('mean_pass_count')}`",
        f"- Mean-and-R60-nonregression count: `{gate.get('mean_and_r60_nonregression_count')}`",
        f"- Strict-pass count: `{gate.get('strict_pass_count')}`",
        f"- Best candidate: `{gate.get('best_candidate_family')}`",
        f"- Best regional / mass / share errors: `{gate.get('best_mean_regional_normalized_absolute_error')}` / `{gate.get('best_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('best_mean_regional_share_half_l1_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Experiments",
        "",
        "| Rank | Experiment | Mode | Regional NAE | Mass NAE | Share half-L1 | Mean gate | R60 gate | Split gate |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(list(report.get("experiment_score_rows") or []), start=1):
        lines.append(
            f"| {index} | `{row.get('experiment_id')}` | `{row.get('mode')}` | "
            f"{float(row.get('mean_regional_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_aggregate_mass_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_regional_share_half_l1_error') or 0.0):.6f} | "
            f"`{row.get('mean_gate_status')}` | `{row.get('r60_nonregression_status')}` | `{row.get('split_gate_status')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    labels = [str(row.get("experiment_id") or "") for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#4f6f61")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#476f9f")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#9a7a2c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R61 split-risk regional router experiments")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _r60_nonregression(row: dict[str, Any], r60_gate: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    checks = [
        (
            "r60_regional",
            _finite_float(row.get("mean_regional_normalized_absolute_error")),
            _finite_float(r60_gate.get("best_mean_regional_normalized_absolute_error")),
        ),
        (
            "r60_mass",
            _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error")),
            _finite_float(r60_gate.get("best_mean_aggregate_mass_normalized_absolute_error")),
        ),
        (
            "r60_share",
            _finite_float(row.get("mean_regional_share_half_l1_error")),
            _finite_float(r60_gate.get("best_mean_regional_share_half_l1_error")),
        ),
    ]
    for name, left, right in checks:
        if left is None or right is None:
            blockers.append(f"{name}_missing")
        elif float(left) > float(right):
            blockers.append(f"{name}_regressed")
    return not blockers, blockers


def run_r61_split_risk_regional_router(
    *,
    run_id: str = R61_RUN_ID,
    r54_report_path: Path | None = None,
    r59_report_path: Path | None = None,
    r60_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r59_path = Path(r59_report_path) if r59_report_path is not None else R59_DEFAULT_REPORT
    r60_path = Path(r60_report_path) if r60_report_path is not None else R60_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r59 = _read_report(r59_path)
    r60 = _read_report(r60_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    source_rows = (
        _base_prediction_rows(r48, r49, r51)
        + [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
        + [dict(row) for row in list(r59.get("prediction_rows") or [])]
    )
    r54_promoted = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or REFERENCE_FAMILY)
    r60_gate = dict(r60.get("queue_gate") or {})
    r60_best_id = str(r60_gate.get("best_experiment_id") or "")
    r60_best_spec = next((dict(spec) for spec in list(r60.get("experiment_specs") or []) if str(spec.get("experiment_id") or "") == r60_best_id), {})
    r60_best_rows, _weights = _prediction_rows_for_spec(
        r60_best_spec,
        source_rows=source_rows,
        rows_by_key=rows_by_key,
        combined_rows=[dict(row) for row in list(r54.get("combined_candidate_table") or [])],
        r54_promoted_family=r54_promoted,
        r59_report=r59,
    )
    r60_best_family = str(r60_best_rows[0].get("candidate_family") or r60_gate.get("best_candidate_family") or "r60_best") if r60_best_rows else "r60_best"
    replay_rows = _rename_family(r60_best_rows, "R61-001__best_r60_replay")
    candidate_pool_rows = source_rows + _rename_family(r60_best_rows, r60_best_family)
    experiment_specs = _r61_experiment_specs(r60_best_family, r54_promoted)
    experiment_prediction_rows: dict[str, list[dict[str, Any]]] = {"R61-001": replay_rows}
    selection_rows: list[dict[str, Any]] = []
    prior_score_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    split_comparison_rows: list[dict[str, Any]] = []
    carry = _family_row(list(r54.get("combined_candidate_table") or []), CARRY_FORWARD_FAMILY)
    reference = _family_row(list(r54.get("combined_candidate_table") or []), REFERENCE_FAMILY)
    for spec in experiment_specs:
        experiment_id = str(spec.get("experiment_id") or "")
        mode = str(spec.get("mode") or "")
        family = f"{experiment_id}__{mode}"
        if experiment_id == "R61-001":
            prediction_rows = replay_rows
        elif mode == "metric_backoff":
            prediction_rows = _copy_metric_from_family(
                _rename_family(r60_best_rows, family),
                candidate_pool_rows,
                metric_family={str(k): str(v) for k, v in dict(spec.get("metric_family") or {}).items()},
                candidate_family=family,
            )
        elif mode == "train_prior_metric_router":
            router_pool = candidate_pool_rows
            for existing_rows in experiment_prediction_rows.values():
                router_pool = router_pool + existing_rows
            prediction_rows, router_selection, router_priors = _train_prior_metric_router_rows(
                router_pool,
                rows_by_key,
                candidate_families=[str(item) for item in list(spec.get("candidate_families") or [])],
                output_family=family,
                default_family=str(spec.get("default_family") or REFERENCE_FAMILY),
            )
            selection_rows.extend(router_selection)
            prior_score_rows.extend(router_priors)
        else:
            raise ValueError(f"unknown R61 mode: {mode}")
        experiment_prediction_rows[experiment_id] = prediction_rows
        split_rows, coherence_rows, family_score = _score_candidate(rows_by_key, prediction_rows)
        comparison = _split_comparison_rows(
            regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + split_rows,
            coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])] + coherence_rows,
            promoted_family=str(family_score.get("candidate_family") or family),
        )
        split_gate = _split_stability_gate(comparison)
        split_comparison_rows.extend(comparison)
        mean_pass, mean_blockers = _candidate_mean_gate(
            family_score,
            carry=carry,
            reference=reference,
            r54_gate=dict(r54.get("adapter_gate") or {}),
            r59_gate=dict(r59.get("ensemble_gate") or {}),
        )
        r60_pass, r60_blockers = _r60_nonregression(family_score, r60_gate)
        score_rows.append(
            {
                **dict(spec),
                "candidate_family": family_score.get("candidate_family"),
                "mean_regional_normalized_absolute_error": family_score.get("mean_regional_normalized_absolute_error"),
                "mean_aggregate_mass_normalized_absolute_error": family_score.get("mean_aggregate_mass_normalized_absolute_error"),
                "mean_regional_share_half_l1_error": family_score.get("mean_regional_share_half_l1_error"),
                "worst_regional_normalized_absolute_error": family_score.get("worst_regional_normalized_absolute_error"),
                "mean_gate_status": "pass" if mean_pass else "fail",
                "mean_gate_blockers": mean_blockers,
                "r60_nonregression_status": "pass" if r60_pass else "fail",
                "r60_nonregression_blockers": r60_blockers,
                "split_gate_status": split_gate.get("status"),
                "split_gate_failure_counts": split_gate.get("failure_counts"),
                "split_metric_count": split_gate.get("split_metric_count"),
            }
        )
    score_rows.sort(
        key=lambda row: (
            0 if row.get("mean_gate_status") == "pass" and row.get("r60_nonregression_status") == "pass" else 1,
            0 if row.get("split_gate_status") == "strict_split_stable_adapter_promoted" else 1,
            float(row.get("mean_regional_normalized_absolute_error") or float("inf")),
            float(row.get("mean_aggregate_mass_normalized_absolute_error") or float("inf")),
            float(row.get("mean_regional_share_half_l1_error") or float("inf")),
            str(row.get("experiment_id") or ""),
        )
    )
    mean_pass_rows = [row for row in score_rows if row.get("mean_gate_status") == "pass"]
    mean_and_r60_rows = [row for row in mean_pass_rows if row.get("r60_nonregression_status") == "pass"]
    strict_pass_rows = [
        row
        for row in mean_and_r60_rows
        if row.get("split_gate_status") == "strict_split_stable_adapter_promoted"
    ]
    best = strict_pass_rows[0] if strict_pass_rows else (mean_and_r60_rows[0] if mean_and_r60_rows else (mean_pass_rows[0] if mean_pass_rows else score_rows[0]))
    blockers: list[str] = []
    if not strict_pass_rows:
        blockers.append("no_r61_candidate_passed_strict_split_gate")
    if not mean_and_r60_rows:
        blockers.append("no_r61_candidate_preserved_r60_mean_contract")
    if strict_pass_rows:
        status = "split_risk_regional_router_strict_champion_promoted"
    elif mean_and_r60_rows:
        status = "split_risk_regional_router_mean_promoted_split_limited"
    elif mean_pass_rows:
        status = "split_risk_regional_router_mean_vs_r59_only"
    else:
        status = "split_risk_regional_router_diagnostic_only"
    gate = {
        "status": status,
        "blockers": blockers,
        "experiment_count": len(score_rows),
        "mean_pass_count": len(mean_pass_rows),
        "mean_and_r60_nonregression_count": len(mean_and_r60_rows),
        "strict_pass_count": len(strict_pass_rows),
        "best_experiment_id": best.get("experiment_id"),
        "best_candidate_family": best.get("candidate_family"),
        "best_mean_regional_normalized_absolute_error": best.get("mean_regional_normalized_absolute_error"),
        "best_mean_aggregate_mass_normalized_absolute_error": best.get("mean_aggregate_mass_normalized_absolute_error"),
        "best_mean_regional_share_half_l1_error": best.get("mean_regional_share_half_l1_error"),
        "best_split_gate_status": best.get("split_gate_status"),
        "r60_best_candidate_family": r60_gate.get("best_candidate_family"),
        "r60_best_mean_regional_normalized_absolute_error": r60_gate.get("best_mean_regional_normalized_absolute_error"),
        "r60_best_mean_aggregate_mass_normalized_absolute_error": r60_gate.get("best_mean_aggregate_mass_normalized_absolute_error"),
        "r60_best_mean_regional_share_half_l1_error": r60_gate.get("best_mean_regional_share_half_l1_error"),
        "contract": (
            "R61 may only route whole metric streams using locked candidate families and train-origin prior split scores. "
            "It may not use same-holdout residuals. Promotion requires the R60 mean contract plus the R55 split-stability gate."
        ),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r61_split_risk_regional_router_report.json"
    md_path = analysis_dir / "r61_split_risk_regional_router_report.md"
    score_csv = analysis_dir / "r61_experiment_score_rows.csv"
    spec_csv = analysis_dir / "r61_experiment_specs.csv"
    selection_csv = analysis_dir / "r61_router_selection_rows.csv"
    prior_csv = analysis_dir / "r61_prior_score_rows.csv"
    comparison_csv = analysis_dir / "r61_split_comparison_rows.csv"
    dashboard_path = analysis_dir / "r61_split_risk_regional_router_dashboard.png"
    report = {
        "schema_version": R61_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "verdict": (
            "R61 found a strict split-stable regional router."
            if strict_pass_rows
            else (
                "R61 preserved the R60 mean contract but remains split-limited."
                if mean_and_r60_rows
                else "R61 did not preserve the R60 mean contract under conservative routing."
            )
        ),
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "r59_report_path": r59_path.as_posix(),
        "r60_report_path": r60_path.as_posix(),
        "router_gate": gate,
        "experiment_specs": experiment_specs,
        "experiment_score_rows": score_rows,
        "selection_rows": selection_rows,
        "prior_score_rows": prior_score_rows,
        "split_comparison_rows": split_comparison_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "experiment_score_csv": score_csv.as_posix(),
            "experiment_specs_csv": spec_csv.as_posix(),
            "selection_csv": selection_csv.as_posix(),
            "prior_score_csv": prior_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(score_csv, score_rows)
    _write_csv(spec_csv, experiment_specs)
    _write_csv(selection_csv, selection_rows)
    _write_csv(prior_csv, prior_score_rows)
    _write_csv(comparison_csv, split_comparison_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, score_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R61 split-risk regional router.")
    parser.add_argument("--run-id", default=R61_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r59-report-path", default=None)
    parser.add_argument("--r60-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r61_split_risk_regional_router(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r59_report_path=None if args.r59_report_path is None else Path(args.r59_report_path),
        r60_report_path=None if args.r60_report_path is None else Path(args.r60_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
