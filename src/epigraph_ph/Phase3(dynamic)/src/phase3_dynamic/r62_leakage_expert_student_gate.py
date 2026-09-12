from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_ordinal, quarter_sort_key
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .r52_subnational_coherence_gate import (
    _coherence_rows,
    _coherence_summary_table,
    _combined_table,
    _regional_summary_table,
)
from .r54_national_total_regional_adapter import (
    R48_DEFAULT_REPORT,
    R49_DEFAULT_REPORT,
    R51_DEFAULT_REPORT,
    _base_prediction_rows,
    _family_period_metric_shares,
    _family_period_metric_total,
    _load_report,
)
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
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
)
from .r61_split_risk_regional_router import _r60_nonregression
from .runtime import ensure_dir, read_json, write_json


R62_SCHEMA_VERSION = "phase3_dynamic.r62_leakage_expert_student_gate.v1"
R62_RUN_ID = "p3d-r62-leakage-expert-student-gate-20260503-s00"
R57_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r57-regional-candidate-ceiling-diagnostic-20260503-s00"
    / "analysis"
    / "r57_regional_candidate_ceiling_diagnostic_report.json"
)
R62_EXPERT_FAMILY = "leakage_expert_oracle_not_promotable"


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): dict(row)
        for row in rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("metric_name")
    }


def _periods(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")}, key=quarter_sort_key)


def _rename_family(rows: list[dict[str, Any]], family: str, *, leakage_status: str | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        renamed = dict(row)
        renamed["candidate_family"] = family
        if leakage_status is not None:
            renamed["leakage_status"] = leakage_status
        output.append(renamed)
    return output


def _prior_oracle_rows(selection_rows: list[dict[str, Any]], *, period: str, metric: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in selection_rows
        if str(row.get("metric_name") or "") == metric
        and row.get("selected_candidate_family")
        and quarter_sort_key(str(row.get("holdout_period") or "")) < quarter_sort_key(period)
    ]


def _most_recent_prior_family(selection_rows: list[dict[str, Any]], *, period: str, metric: str, default_family: str) -> tuple[str, str, int]:
    priors = _prior_oracle_rows(selection_rows, period=period, metric=metric)
    if not priors:
        return default_family, "default_no_prior_oracle_label", 0
    priors.sort(key=lambda row: quarter_sort_key(str(row.get("holdout_period") or "")), reverse=True)
    return str(priors[0].get("selected_candidate_family") or default_family), "most_recent_prior_oracle_label", len(priors)


def _frequency_prior_family(selection_rows: list[dict[str, Any]], *, period: str, metric: str, default_family: str) -> tuple[str, str, int]:
    priors = _prior_oracle_rows(selection_rows, period=period, metric=metric)
    if not priors:
        return default_family, "default_no_prior_oracle_label", 0
    counts: Counter[str] = Counter(str(row.get("selected_candidate_family") or "") for row in priors)
    last_seen = {
        family: max(quarter_ordinal(str(row.get("holdout_period") or "")) for row in priors if str(row.get("selected_candidate_family") or "") == family)
        for family in counts
    }
    family = sorted(counts, key=lambda item: (-counts[item], -last_seen[item], item))[0]
    return family, "most_frequent_prior_oracle_label", len(priors)


def _error_tuple(
    regional_index: dict[tuple[str, str, str], dict[str, Any]],
    coherence_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    family: str,
    period: str,
    metric: str,
) -> tuple[float, float, float] | None:
    regional = _finite_float((regional_index.get((family, period, metric)) or {}).get("normalized_absolute_error"))
    mass = _finite_float((coherence_index.get((family, period, metric)) or {}).get("aggregate_mass_normalized_absolute_error"))
    share = _finite_float((coherence_index.get((family, period, metric)) or {}).get("regional_share_half_l1_error"))
    if regional is None or mass is None or share is None:
        return None
    return float(regional), float(mass), float(share)


def _regret_prior_family(
    selection_rows: list[dict[str, Any]],
    candidate_families: list[str],
    regional_index: dict[tuple[str, str, str], dict[str, Any]],
    coherence_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    period: str,
    metric: str,
    default_family: str,
) -> tuple[str, str, int]:
    priors = _prior_oracle_rows(selection_rows, period=period, metric=metric)
    if not priors:
        return default_family, "default_no_prior_oracle_label", 0
    scored: list[tuple[float, float, float, str]] = []
    for family in candidate_families:
        regional_regrets: list[float] = []
        mass_regrets: list[float] = []
        share_regrets: list[float] = []
        for oracle in priors:
            prior_period = str(oracle.get("holdout_period") or "")
            candidate_errors = _error_tuple(
                regional_index,
                coherence_index,
                family=family,
                period=prior_period,
                metric=metric,
            )
            if candidate_errors is None:
                continue
            oracle_regional = _finite_float(oracle.get("selected_regional_normalized_absolute_error"))
            oracle_mass = _finite_float(oracle.get("selected_aggregate_mass_normalized_absolute_error"))
            oracle_share = _finite_float(oracle.get("selected_regional_share_half_l1_error"))
            if oracle_regional is None or oracle_mass is None or oracle_share is None:
                continue
            regional_regrets.append(max(float(candidate_errors[0]) - float(oracle_regional), 0.0))
            mass_regrets.append(max(float(candidate_errors[1]) - float(oracle_mass), 0.0))
            share_regrets.append(max(float(candidate_errors[2]) - float(oracle_share), 0.0))
        if not regional_regrets:
            continue
        scored.append(
            (
                float(np.mean(np.asarray(regional_regrets, dtype=np.float64))),
                float(np.mean(np.asarray(mass_regrets, dtype=np.float64))),
                float(np.mean(np.asarray(share_regrets, dtype=np.float64))),
                family,
            )
        )
    if not scored:
        return default_family, "default_no_prior_regret_scores", len(priors)
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return scored[0][3], "minimum_prior_oracle_regret", len(priors)


def _adapter_components(family: str) -> tuple[str, str]:
    prefix = "national_total_adapter__total="
    marker = "__share="
    if family.startswith(prefix) and marker in family:
        total, share = family[len(prefix) :].split(marker, 1)
        return total, share
    return family, family


def _component_prior_family(
    selection_rows: list[dict[str, Any]],
    *,
    period: str,
    metric: str,
    default_family: str,
    policy: str,
) -> tuple[str, str, str, int]:
    priors = _prior_oracle_rows(selection_rows, period=period, metric=metric)
    if not priors:
        total, share = _adapter_components(default_family)
        return total, share, "default_no_prior_oracle_label", 0
    if policy == "component_recency":
        priors.sort(key=lambda row: quarter_sort_key(str(row.get("holdout_period") or "")), reverse=True)
        total, share = _adapter_components(str(priors[0].get("selected_candidate_family") or default_family))
        return total, share, "most_recent_prior_oracle_components", len(priors)
    total_counts: Counter[str] = Counter()
    share_counts: Counter[str] = Counter()
    total_last: dict[str, int] = {}
    share_last: dict[str, int] = {}
    for row in priors:
        family = str(row.get("selected_candidate_family") or default_family)
        ordinal = quarter_ordinal(str(row.get("holdout_period") or ""))
        total, share = _adapter_components(family)
        total_counts[total] += 1
        share_counts[share] += 1
        total_last[total] = max(total_last.get(total, -1), ordinal)
        share_last[share] = max(share_last.get(share, -1), ordinal)
    total = sorted(total_counts, key=lambda item: (-total_counts[item], -total_last[item], item))[0]
    share = sorted(share_counts, key=lambda item: (-share_counts[item], -share_last[item], item))[0]
    return total, share, "most_frequent_prior_oracle_components", len(priors)


def _candidate_families_for_period(
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


def _build_family_student_rows(
    source_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    regional_score_rows: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    *,
    candidate_family: str,
    policy: str,
    default_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(source_rows)
    regional_index = _score_index(regional_score_rows)
    coherence_index = _score_index(coherence_rows)
    all_families = sorted({family for family, _period, _region in indexed})
    output_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    for period in _periods(source_rows):
        regions = sorted({region for family, candidate_period, region in indexed if family == default_family and candidate_period == period})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        available = _candidate_families_for_period(indexed, period=period, regions=regions, families=all_families)
        per_region: dict[str, dict[str, Any]] = {
            region: {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
                "leakage_status": "blocked_time_student_uses_prior_oracle_labels_only",
            }
            for region in regions
        }
        for metric in COUNT_METRICS:
            if policy == "recency":
                selected, reason, prior_count = _most_recent_prior_family(
                    selection_rows,
                    period=period,
                    metric=metric,
                    default_family=default_family,
                )
            elif policy == "frequency":
                selected, reason, prior_count = _frequency_prior_family(
                    selection_rows,
                    period=period,
                    metric=metric,
                    default_family=default_family,
                )
            elif policy == "regret":
                selected, reason, prior_count = _regret_prior_family(
                    selection_rows,
                    available,
                    regional_index,
                    coherence_index,
                    period=period,
                    metric=metric,
                    default_family=default_family,
                )
            else:
                raise ValueError(f"unknown leakage student policy: {policy}")
            if selected not in available:
                selected = default_family if default_family in available else (available[0] if available else selected)
                reason = f"{reason}_fallback_available_family"
            router_rows.append(
                {
                    "candidate_family": candidate_family,
                    "holdout_period": period,
                    "metric_name": metric,
                    "policy": policy,
                    "selected_source_family": selected,
                    "selection_reason": reason,
                    "prior_oracle_label_count": prior_count,
                    "available_family_count": len(available),
                    "leakage_training_status": (
                        "no_prior_oracle_label_default_only"
                        if prior_count == 0
                        else "uses_prior_holdout_oracle_labels_only"
                    ),
                }
            )
            for region in regions:
                value = _finite_float((indexed.get((selected, period, region)) or {}).get(metric))
                if value is not None:
                    per_region[region][metric] = float(value)
                    per_region[region][f"{metric}_source_family"] = selected
        for region in regions:
            output_rows.append(_projection_cascade(per_region[region]))
    return output_rows, router_rows


def _build_component_student_rows(
    source_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    candidate_family: str,
    policy: str,
    default_family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(source_rows)
    output_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    for period in _periods(source_rows):
        default_regions = sorted({region for family, candidate_period, region in indexed if family == default_family and candidate_period == period})
        regions = default_regions or sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        per_region: dict[str, dict[str, Any]] = {
            region: {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
                "leakage_status": "blocked_time_student_uses_prior_oracle_components_only",
            }
            for region in regions
        }
        for metric in COUNT_METRICS:
            total_family, share_family, reason, prior_count = _component_prior_family(
                selection_rows,
                period=period,
                metric=metric,
                default_family=default_family,
                policy=policy,
            )
            total_regions = set(region for family, candidate_period, region in indexed if family == total_family and candidate_period == period)
            share_regions = set(region for family, candidate_period, region in indexed if family == share_family and candidate_period == period)
            common_regions = sorted(set(regions) & total_regions & share_regions)
            if not common_regions:
                total_family, share_family = _adapter_components(default_family)
                common_regions = sorted(set(regions) & {region for family, candidate_period, region in indexed if family == total_family and candidate_period == period})
                reason = f"{reason}_fallback_default_components"
            total = _family_period_metric_total(
                indexed,
                family=total_family,
                period=period,
                metric=metric,
                regions=common_regions,
            )
            shares = _family_period_metric_shares(
                indexed,
                family=share_family,
                period=period,
                metric=metric,
                regions=common_regions,
            )
            router_rows.append(
                {
                    "candidate_family": candidate_family,
                    "holdout_period": period,
                    "metric_name": metric,
                    "policy": policy,
                    "selected_total_family": total_family,
                    "selected_share_family": share_family,
                    "selection_reason": reason,
                    "prior_oracle_label_count": prior_count,
                    "region_count": len(common_regions),
                    "leakage_training_status": (
                        "no_prior_oracle_label_default_only"
                        if prior_count == 0
                        else "uses_prior_holdout_oracle_components_only"
                    ),
                }
            )
            if total is None or shares is None:
                continue
            for region in common_regions:
                per_region[region][metric] = float(total * float(shares.get(region, 0.0)))
                per_region[region][f"{metric}_total_source_family"] = total_family
                per_region[region][f"{metric}_share_source_family"] = share_family
        for region in regions:
            output_rows.append(_projection_cascade(per_region[region]))
    return output_rows, router_rows


def _score_candidate(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence))
    family = str(prediction_rows[0].get("candidate_family") or "") if prediction_rows else ""
    return split_rows, coherence, _family_row(combined, family)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("leakage_student_gate") or {})
    lines = [
        "# Phase 3 R62 Leakage Expert / Student Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Expert family: `{gate.get('expert_family')}`",
        f"- Best student: `{gate.get('best_student_family')}`",
        f"- Student strict-pass count: `{gate.get('student_strict_pass_count')}`",
        f"- Student mean-pass count: `{gate.get('student_mean_pass_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Scores",
        "",
        "| Candidate | Type | Regional NAE | Mass NAE | Share half-L1 | Mean gate | R60 gate | Split gate |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in list(report.get("candidate_score_rows") or []):
        lines.append(
            f"| `{row.get('candidate_family')}` | `{row.get('candidate_type')}` | "
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
    labels = [str(row.get("candidate_family") or "") for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#5b6f62")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#4b709b")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#94772d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R62 leakage expert versus blocked-time students")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r62_leakage_expert_student_gate(
    *,
    run_id: str = R62_RUN_ID,
    r54_report_path: Path | None = None,
    r57_report_path: Path | None = None,
    r59_report_path: Path | None = None,
    r60_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r57_path = Path(r57_report_path) if r57_report_path is not None else R57_DEFAULT_REPORT
    r59_path = Path(r59_report_path) if r59_report_path is not None else R59_DEFAULT_REPORT
    r60_path = Path(r60_report_path) if r60_report_path is not None else R60_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r57 = _read_report(r57_path)
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
    r54_promoted = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or REFERENCE_FAMILY)
    source_rows = (
        _base_prediction_rows(r48, r49, r51)
        + [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
        + [dict(row) for row in list(r59.get("prediction_rows") or [])]
    )
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
    r60_best_family = str(r60_gate.get("best_candidate_family") or (r60_best_rows[0].get("candidate_family") if r60_best_rows else "") or r54_promoted)
    source_rows = source_rows + _rename_family(r60_best_rows, r60_best_family)
    _raw, source_split_rows = _score_predictions(rows_by_key, source_rows)
    source_coherence_rows = _coherence_rows(rows_by_key, source_rows)
    selection_rows = [dict(row) for row in list(r57.get("selection_rows") or [])]
    expert_rows = _rename_family(
        [dict(row) for row in list(r57.get("prediction_rows") or [])],
        R62_EXPERT_FAMILY,
        leakage_status="same_split_oracle_expert_not_promotable",
    )
    student_specs = [
        ("R62-001__prior_oracle_recency_student", "family", "recency"),
        ("R62-002__prior_oracle_frequency_student", "family", "frequency"),
        ("R62-003__prior_oracle_regret_student", "family", "regret"),
        ("R62-004__prior_oracle_component_recency_student", "component", "component_recency"),
        ("R62-005__prior_oracle_component_frequency_student", "component", "component_frequency"),
    ]
    candidate_rows_by_family: dict[str, list[dict[str, Any]]] = {R62_EXPERT_FAMILY: expert_rows}
    router_rows: list[dict[str, Any]] = []
    for family, mode, policy in student_specs:
        if mode == "family":
            rows, selections = _build_family_student_rows(
                source_rows,
                selection_rows,
                source_split_rows,
                source_coherence_rows,
                candidate_family=family,
                policy=policy,
                default_family=r60_best_family,
            )
        else:
            rows, selections = _build_component_student_rows(
                source_rows,
                selection_rows,
                candidate_family=family,
                policy=policy,
                default_family=r60_best_family,
            )
        candidate_rows_by_family[family] = rows
        router_rows.extend(selections)
    carry = _family_row(list(r54.get("combined_candidate_table") or []), CARRY_FORWARD_FAMILY)
    reference = _family_row(list(r54.get("combined_candidate_table") or []), REFERENCE_FAMILY)
    score_rows: list[dict[str, Any]] = []
    split_comparison_rows: list[dict[str, Any]] = []
    for family, prediction_rows in candidate_rows_by_family.items():
        split_rows, coherence_rows, family_score = _score_candidate(rows_by_key, prediction_rows)
        comparison = _split_comparison_rows(
            regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + split_rows,
            coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])] + coherence_rows,
            promoted_family=family,
        )
        split_gate = _split_stability_gate(comparison)
        split_comparison_rows.extend(comparison)
        if family == R62_EXPERT_FAMILY:
            mean_pass, mean_blockers = False, ["same_split_oracle_expert_not_promotable"]
            r60_pass, r60_blockers = False, ["same_split_oracle_expert_not_promotable"]
            candidate_type = "leakage_expert_oracle"
        else:
            mean_pass, mean_blockers = _candidate_mean_gate(
                family_score,
                carry=carry,
                reference=reference,
                r54_gate=dict(r54.get("adapter_gate") or {}),
                r59_gate=dict(r59.get("ensemble_gate") or {}),
            )
            r60_pass, r60_blockers = _r60_nonregression(family_score, r60_gate)
            candidate_type = "blocked_time_leakage_student"
        score_rows.append(
            {
                "candidate_family": family,
                "candidate_type": candidate_type,
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
            0 if row.get("candidate_type") == "blocked_time_leakage_student" else 1,
            0 if row.get("mean_gate_status") == "pass" and row.get("r60_nonregression_status") == "pass" else 1,
            0 if row.get("split_gate_status") == "strict_split_stable_adapter_promoted" else 1,
            float(row.get("mean_regional_normalized_absolute_error") or float("inf")),
            float(row.get("mean_aggregate_mass_normalized_absolute_error") or float("inf")),
            float(row.get("mean_regional_share_half_l1_error") or float("inf")),
            str(row.get("candidate_family") or ""),
        )
    )
    student_rows = [row for row in score_rows if row.get("candidate_type") == "blocked_time_leakage_student"]
    mean_pass_students = [row for row in student_rows if row.get("mean_gate_status") == "pass"]
    r60_safe_students = [row for row in mean_pass_students if row.get("r60_nonregression_status") == "pass"]
    strict_students = [
        row
        for row in r60_safe_students
        if row.get("split_gate_status") == "strict_split_stable_adapter_promoted"
    ]
    best_student = strict_students[0] if strict_students else (r60_safe_students[0] if r60_safe_students else (mean_pass_students[0] if mean_pass_students else (student_rows[0] if student_rows else {})))
    blockers: list[str] = []
    if not strict_students:
        blockers.append("no_blocked_time_leakage_student_passed_strict_split_gate")
    if not r60_safe_students:
        blockers.append("no_blocked_time_leakage_student_preserved_r60_mean_contract")
    if strict_students:
        status = "leakage_student_strict_champion_promoted"
    elif r60_safe_students:
        status = "leakage_student_mean_promoted_split_limited"
    else:
        status = "leakage_expert_only_diagnostic"
    expert_score = next((row for row in score_rows if row.get("candidate_family") == R62_EXPERT_FAMILY), {})
    gate = {
        "status": status,
        "blockers": blockers,
        "expert_family": R62_EXPERT_FAMILY,
        "expert_mean_regional_normalized_absolute_error": expert_score.get("mean_regional_normalized_absolute_error"),
        "expert_mean_aggregate_mass_normalized_absolute_error": expert_score.get("mean_aggregate_mass_normalized_absolute_error"),
        "expert_mean_regional_share_half_l1_error": expert_score.get("mean_regional_share_half_l1_error"),
        "best_student_family": best_student.get("candidate_family"),
        "best_student_mean_regional_normalized_absolute_error": best_student.get("mean_regional_normalized_absolute_error"),
        "best_student_mean_aggregate_mass_normalized_absolute_error": best_student.get("mean_aggregate_mass_normalized_absolute_error"),
        "best_student_mean_regional_share_half_l1_error": best_student.get("mean_regional_share_half_l1_error"),
        "student_mean_pass_count": len(mean_pass_students),
        "student_r60_safe_count": len(r60_safe_students),
        "student_strict_pass_count": len(strict_students),
        "r60_best_candidate_family": r60_best_family,
        "contract": (
            "R62 separates leakage into two roles. The Leakage Expert is an oracle/teacher that sees same-split "
            "holdout outcomes and is never promotable. Leakage Students may use only prior-period oracle labels "
            "inside a blocked-time replay; promotion still requires the R60 mean contract and R55 split-stability gate."
        ),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r62_leakage_expert_student_gate_report.json"
    md_path = analysis_dir / "r62_leakage_expert_student_gate_report.md"
    score_csv = analysis_dir / "r62_candidate_score_rows.csv"
    router_csv = analysis_dir / "r62_router_rows.csv"
    comparison_csv = analysis_dir / "r62_split_comparison_rows.csv"
    dashboard_path = analysis_dir / "r62_leakage_expert_student_gate_dashboard.png"
    report = {
        "schema_version": R62_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "verdict": (
            "R62 found a strict blocked-time leakage student."
            if strict_students
            else (
                "R62 found a mean-promoted leakage student, but split-stability remains limited."
                if r60_safe_students
                else "R62 confirms leakage is useful as an expert oracle, but current students do not preserve the R60 contract."
            )
        ),
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "r57_report_path": r57_path.as_posix(),
        "r59_report_path": r59_path.as_posix(),
        "r60_report_path": r60_path.as_posix(),
        "leakage_student_gate": gate,
        "candidate_score_rows": score_rows,
        "router_rows": router_rows,
        "split_comparison_rows": split_comparison_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "candidate_score_csv": score_csv.as_posix(),
            "router_csv": router_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(score_csv, score_rows)
    _write_csv(router_csv, router_rows)
    _write_csv(comparison_csv, split_comparison_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, score_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R62 leakage expert/student gate.")
    parser.add_argument("--run-id", default=R62_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r57-report-path", default=None)
    parser.add_argument("--r59-report-path", default=None)
    parser.add_argument("--r60-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r62_leakage_expert_student_gate(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r57_report_path=None if args.r57_report_path is None else Path(args.r57_report_path),
        r59_report_path=None if args.r59_report_path is None else Path(args.r59_report_path),
        r60_report_path=None if args.r60_report_path is None else Path(args.r60_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
