from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import _default_evidence_root
from .r75_bulk_unaids_annual_challenge import (
    _annual_carry_forward_prediction,
    _annual_score_row,
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _rolling_annual_splits,
    _score_summary_by_fields,
    _write_csv,
)
from .r89_incidence_mortality_mechanism_support_gate import _metric_support_summary
from .runtime import ensure_dir, read_json, write_json


R91_SCHEMA_VERSION = "phase3_dynamic.r91_mechanism_support_expansion_gate.v1"
R91_RUN_ID = "p3d-r91-mechanism-support-expansion-gate-20260509-s00"
R91_FAMILY = "mechanism_support_expansion_bridge_gate"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)

BRIDGE_POLICIES: tuple[str, ...] = (
    "median_ratio_last_any_proxy",
    "last_ratio_last_any_proxy",
    "mean_ratio_last_any_proxy",
    "median_ratio_last_complete_proxy",
)
BRIDGE_SPECS: tuple[dict[str, str], ...] = (
    {
        "bridge_id": "incidence_proxy_diagnosis_flow_bridge",
        "target_metric": "annual_new_infections",
        "proxy_metric": "new_diagnosed_cases_period",
        "direct_process_metric": "incident_infections_period",
        "claim_role": "proxy_only_incidence_support",
    },
    {
        "bridge_id": "mortality_reported_death_bridge",
        "target_metric": "annual_aids_deaths",
        "proxy_metric": "deaths_reported_period",
        "direct_process_metric": "deaths_reported_period",
        "claim_role": "reported_death_process_support",
    },
)


def _source_family_from_provenance(provenance: dict[str, Any] | None) -> str:
    item = dict(provenance or {})
    return "|".join(
        [
            str(item.get("source_tier") or item.get("source_quality_tier") or "unknown_source_tier"),
            str(item.get("measurement_class") or "unknown_measurement_class"),
            str(item.get("series_kind") or "unknown_series_kind"),
        ]
    )


def _metric_provenance(row: dict[str, Any], metric_name: str) -> dict[str, Any]:
    provenance = dict(row.get("metric_provenance") or {}).get(metric_name)
    return dict(provenance) if isinstance(provenance, dict) else {}


def _metric_source_family(row: dict[str, Any], metric_name: str) -> str:
    return _source_family_from_provenance(_metric_provenance(row, metric_name))


def _proxy_annual_series(rows: list[dict[str, Any]], proxy_metric: str) -> dict[int, dict[str, Any]]:
    by_year: dict[int, list[float]] = defaultdict(list)
    family_by_year: dict[int, Counter[str]] = defaultdict(Counter)
    source_by_year: dict[int, Counter[str]] = defaultdict(Counter)
    role_by_year: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = _finite_float(row.get(proxy_metric))
        quarter = str(row.get("quarter") or "")
        if value is None or "-Q" not in quarter:
            continue
        year = quarter_year(quarter)
        provenance = _metric_provenance(row, proxy_metric)
        family = _source_family_from_provenance(provenance)
        by_year[int(year)].append(max(float(value), 0.0))
        family_by_year[int(year)][family] += 1
        source_by_year[int(year)][str(provenance.get("source_id") or "")] += 1
        role_by_year[int(year)][str(provenance.get("observation_role") or "")] += 1
    output: dict[int, dict[str, Any]] = {}
    for year, values in sorted(by_year.items()):
        observed_count = len(values)
        annualized = None if observed_count <= 0 else float(sum(values) * 4.0 / float(observed_count))
        output[int(year)] = {
            "year": int(year),
            "proxy_metric": proxy_metric,
            "observed_quarter_count": int(observed_count),
            "annualized_proxy_value": annualized,
            "proxy_sum": float(sum(values)),
            "coverage_status": "complete_year" if observed_count >= 4 else "partial_year_annualized",
            "source_family_counts": dict(family_by_year[int(year)].most_common()),
            "source_counts": dict(source_by_year[int(year)].most_common()),
            "observation_role_counts": dict(role_by_year[int(year)].most_common()),
            "contract": "annualized proxy = sum(observed direct/proxy quarters) * 4 / observed quarter count",
        }
    return output


def _bridge_pairs(
    rows: list[dict[str, Any]],
    *,
    target_metric: str,
    proxy_by_year: dict[int, dict[str, Any]],
    train_end_year: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        if not quarter.endswith("-Q4"):
            continue
        year = quarter_year(quarter)
        if int(year) > int(train_end_year):
            continue
        target = _finite_float(row.get(target_metric))
        proxy = _finite_float((proxy_by_year.get(int(year)) or {}).get("annualized_proxy_value"))
        if target is None or proxy is None or proxy <= 0.0:
            continue
        proxy_row = dict(proxy_by_year.get(int(year)) or {})
        pairs.append(
            {
                "year": int(year),
                "target_metric": target_metric,
                "target_value": float(target),
                "annualized_proxy_value": float(proxy),
                "target_to_proxy_ratio": float(target) / float(proxy),
                "observed_quarter_count": proxy_row.get("observed_quarter_count"),
                "coverage_status": proxy_row.get("coverage_status"),
                "source_family_counts": proxy_row.get("source_family_counts"),
            }
        )
    return pairs


def _last_proxy_year(proxy_by_year: dict[int, dict[str, Any]], train_end_year: int, *, complete_only: bool) -> int | None:
    eligible = []
    for year, row in proxy_by_year.items():
        if int(year) > int(train_end_year):
            continue
        if complete_only and int(row.get("observed_quarter_count") or 0) < 4:
            continue
        if _finite_float(row.get("annualized_proxy_value")) is None:
            continue
        eligible.append(int(year))
    return max(eligible) if eligible else None


def _fit_bridge_policy(
    *,
    pairs: list[dict[str, Any]],
    proxy_by_year: dict[int, dict[str, Any]],
    train_end_year: int,
    policy: str,
) -> dict[str, Any]:
    if policy not in BRIDGE_POLICIES:
        raise ValueError(f"unknown bridge policy: {policy}")
    train_pairs = [dict(row) for row in pairs if int(row.get("year") or 0) <= int(train_end_year)]
    ratios = [
        float(row["target_to_proxy_ratio"])
        for row in train_pairs
        if _finite_float(row.get("target_to_proxy_ratio")) is not None
    ]
    if not ratios:
        return {"status": "not_estimable", "policy": policy, "reason": "no_train_target_proxy_pairs", "pair_count": 0}
    complete_only = policy.endswith("_complete_proxy")
    proxy_year = _last_proxy_year(proxy_by_year, train_end_year, complete_only=complete_only)
    if proxy_year is None:
        return {"status": "not_estimable", "policy": policy, "reason": "no_train_proxy_year", "pair_count": len(train_pairs)}
    proxy_value = _finite_float((proxy_by_year.get(int(proxy_year)) or {}).get("annualized_proxy_value"))
    if proxy_value is None:
        return {"status": "not_estimable", "policy": policy, "reason": "missing_selected_proxy_value", "pair_count": len(train_pairs)}
    if policy.startswith("median_ratio"):
        ratio = float(median(ratios))
    elif policy.startswith("mean_ratio"):
        ratio = float(np.mean(np.asarray(ratios, dtype=np.float64)))
    elif policy.startswith("last_ratio"):
        ratio = float(sorted(train_pairs, key=lambda row: int(row.get("year") or 0))[-1]["target_to_proxy_ratio"])
    else:
        raise ValueError(f"unsupported bridge policy: {policy}")
    return {
        "status": "completed",
        "policy": policy,
        "pair_count": len(train_pairs),
        "first_pair_year": int(train_pairs[0]["year"]),
        "last_pair_year": int(train_pairs[-1]["year"]),
        "selected_proxy_year": int(proxy_year),
        "selected_proxy_value": float(proxy_value),
        "ratio": float(ratio),
        "candidate_value": float(proxy_value) * float(ratio),
    }


def _internal_policy_score_rows(
    rows: list[dict[str, Any]],
    *,
    target_metric: str,
    proxy_by_year: dict[int, dict[str, Any]],
    train_end_year: int,
    bridge_id: str,
) -> list[dict[str, Any]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    output: list[dict[str, Any]] = []
    years = sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in q4_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
            and _finite_float(row.get(target_metric)) is not None
        }
    )
    for validation_year in years:
        prior_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) < int(validation_year)]
        if not prior_rows:
            continue
        holdout_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) == int(validation_year)]
        if not holdout_rows:
            continue
        prior_pairs = _bridge_pairs(
            rows,
            target_metric=target_metric,
            proxy_by_year=proxy_by_year,
            train_end_year=int(validation_year) - 1,
        )
        for policy in BRIDGE_POLICIES:
            model = _fit_bridge_policy(
                pairs=prior_pairs,
                proxy_by_year=proxy_by_year,
                train_end_year=int(validation_year) - 1,
                policy=policy,
            )
            candidate_value = _finite_float(model.get("candidate_value"))
            for holdout_row in holdout_rows:
                carry_value = _annual_carry_forward_prediction(prior_rows, holdout_row, target_metric)
                score = _annual_score_row(
                    family=f"{bridge_id}:{policy}:internal_selector",
                    horizon=1,
                    train_end_year=int(validation_year) - 1,
                    holdout_years=[int(validation_year)],
                    holdout_row=holdout_row,
                    metric_name=target_metric,
                    candidate_value=None if candidate_value is None else float(candidate_value),
                    carry_value=None if carry_value is None else float(carry_value),
                    scale=_annual_target_scale(prior_rows, target_metric),
                )
                score["bridge_id"] = bridge_id
                score["bridge_policy"] = policy
                score["training_use"] = "internal_train_only_policy_selection"
                output.append(score)
    return output


def _select_policy(internal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = _score_summary_by_fields(internal_rows, group_fields=("bridge_policy",))
    evaluable = [
        row
        for row in summaries
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    if not evaluable:
        return {
            "selected_policy": BRIDGE_POLICIES[0],
            "selection_status": "default_no_internal_scores",
            "internal_summary_rows": summaries,
        }
    order = {policy: index for index, policy in enumerate(BRIDGE_POLICIES)}
    evaluable.sort(
        key=lambda row: (
            float(row.get("candidate_mean_norm_error") or np.inf),
            order.get(str(row.get("bridge_policy") or ""), len(order)),
        )
    )
    selected = dict(evaluable[0])
    return {
        "selected_policy": str(selected.get("bridge_policy") or BRIDGE_POLICIES[0]),
        "selection_status": "train_internal_min_candidate_mean_norm_error",
        "selected_internal_candidate_mean_norm_error": selected.get("candidate_mean_norm_error"),
        "selected_internal_carry_forward_mean_norm_error": selected.get("carry_forward_mean_norm_error"),
        "internal_summary_rows": summaries,
    }


def _score_train_selected_bridge(
    rows: list[dict[str, Any]],
    *,
    bridge_id: str,
    target_metric: str,
    proxy_metric: str,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    proxy_by_year = _proxy_annual_series(rows, proxy_metric)
    score_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_rows:
            continue
        split_internal_rows = _internal_policy_score_rows(
            rows,
            target_metric=target_metric,
            proxy_by_year=proxy_by_year,
            train_end_year=train_end_year,
            bridge_id=bridge_id,
        )
        selection = _select_policy(split_internal_rows)
        selected_policy = str(selection["selected_policy"])
        pairs = _bridge_pairs(rows, target_metric=target_metric, proxy_by_year=proxy_by_year, train_end_year=train_end_year)
        model = _fit_bridge_policy(
            pairs=pairs,
            proxy_by_year=proxy_by_year,
            train_end_year=train_end_year,
            policy=selected_policy,
        )
        model_rows.append(
            {
                "candidate_family": f"{bridge_id}:train_selected_proxy_bridge",
                "bridge_id": bridge_id,
                "target_metric": target_metric,
                "proxy_metric": proxy_metric,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "selected_policy": selected_policy,
                "selection_status": selection.get("selection_status"),
                "selected_internal_candidate_mean_norm_error": selection.get("selected_internal_candidate_mean_norm_error"),
                "selected_internal_carry_forward_mean_norm_error": selection.get("selected_internal_carry_forward_mean_norm_error"),
                **{key: value for key, value in model.items() if key != "pairs"},
            }
        )
        for internal_row in split_internal_rows:
            row = dict(internal_row)
            row["outer_train_end_year"] = train_end_year
            row["outer_horizon_years"] = horizon
            internal_rows.append(row)
        candidate_value = _finite_float(model.get("candidate_value"))
        for holdout_row in holdout_rows:
            if _finite_float(holdout_row.get(target_metric)) is None:
                continue
            carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, target_metric)
            score = _annual_score_row(
                family=f"{bridge_id}:train_selected_proxy_bridge",
                horizon=horizon,
                train_end_year=train_end_year,
                holdout_years=holdout_years,
                holdout_row=holdout_row,
                metric_name=target_metric,
                candidate_value=None if candidate_value is None else float(candidate_value),
                carry_value=None if carry_value is None else float(carry_value),
                scale=_annual_target_scale(raw_train_rows, target_metric),
            )
            score["bridge_id"] = bridge_id
            score["proxy_metric"] = proxy_metric
            score["selected_policy"] = selected_policy
            score["training_use"] = "train_origin_proxy_bridge_policy_selected_inside_train_only"
            score_rows.append(score)
    return score_rows, model_rows, internal_rows


def _ablate_proxy_metric_source_family(rows: list[dict[str, Any]], *, proxy_metric: str, source_family: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        provenance = dict(next_row.get("metric_provenance") or {})
        if _finite_float(next_row.get(proxy_metric)) is not None and _metric_source_family(next_row, proxy_metric) == source_family:
            next_row[proxy_metric] = None
            provenance.pop(proxy_metric, None)
            next_row["metric_provenance"] = provenance
        output.append(next_row)
    return output


def _source_family_rows(rows: list[dict[str, Any]], proxy_metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    roles: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    sources: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric in proxy_metrics:
            if _finite_float(row.get(metric)) is None:
                continue
            provenance = _metric_provenance(row, metric)
            family = _source_family_from_provenance(provenance)
            key = (metric, family)
            counts[key] += 1
            roles[key][str(provenance.get("observation_role") or "")] += 1
            sources[key][str(provenance.get("source_id") or "")] += 1
    return [
        {
            "proxy_metric": metric,
            "source_family": family,
            "entry_count": int(count),
            "observation_role_counts": dict(roles[(metric, family)].most_common()),
            "top_sources": dict(sources[(metric, family)].most_common(10)),
        }
        for (metric, family), count in sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
    ]


def _ablation_rows(
    rows: list[dict[str, Any]],
    *,
    source_family_rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in BRIDGE_SPECS:
        bridge_id = spec["bridge_id"]
        target_metric = spec["target_metric"]
        proxy_metric = spec["proxy_metric"]
        full_score_rows, _, _ = _score_train_selected_bridge(
            rows,
            bridge_id=bridge_id,
            target_metric=target_metric,
            proxy_metric=proxy_metric,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
        )
        full_summary = (_score_summary_by_fields(full_score_rows, group_fields=("candidate_family",)) or [{}])[0]
        for family_row in source_family_rows:
            if str(family_row.get("proxy_metric") or "") != proxy_metric:
                continue
            source_family = str(family_row.get("source_family") or "")
            ablated = _ablate_proxy_metric_source_family(rows, proxy_metric=proxy_metric, source_family=source_family)
            ablated_score_rows, _, _ = _score_train_selected_bridge(
                ablated,
                bridge_id=bridge_id,
                target_metric=target_metric,
                proxy_metric=proxy_metric,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                horizons=horizons,
            )
            ablated_summary = (_score_summary_by_fields(ablated_score_rows, group_fields=("candidate_family",)) or [{}])[0]
            output.append(
                {
                    "bridge_id": bridge_id,
                    "target_metric": target_metric,
                    "proxy_metric": proxy_metric,
                    "ablated_source_family": source_family,
                    "ablated_entry_count": family_row.get("entry_count"),
                    "full_candidate_mean_norm_error": full_summary.get("candidate_mean_norm_error"),
                    "full_carry_forward_mean_norm_error": full_summary.get("carry_forward_mean_norm_error"),
                    "ablated_candidate_mean_norm_error": ablated_summary.get("candidate_mean_norm_error"),
                    "ablated_carry_forward_mean_norm_error": ablated_summary.get("carry_forward_mean_norm_error"),
                    "ablation_delta_candidate_mean_norm_error": None
                    if _finite_float(ablated_summary.get("candidate_mean_norm_error")) is None
                    or _finite_float(full_summary.get("candidate_mean_norm_error")) is None
                    else float(ablated_summary["candidate_mean_norm_error"]) - float(full_summary["candidate_mean_norm_error"]),
                    "ablation_status": "evaluable" if ablated_score_rows else "not_evaluable_after_ablation",
                }
            )
    return output


def _bridge_gate_row(summary: dict[str, Any], *, bridge_id: str) -> dict[str, Any]:
    candidate_mean = _finite_float(summary.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(summary.get("carry_forward_mean_norm_error"))
    candidate_coverage = _finite_float(summary.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(summary.get("carry_forward_interval_coverage"))
    return {
        "bridge_id": bridge_id,
        "candidate_mean_norm_error": candidate_mean,
        "carry_forward_mean_norm_error": carry_mean,
        "candidate_minus_carry_forward_mean_norm_error": None
        if candidate_mean is None or carry_mean is None
        else float(candidate_mean - carry_mean),
        "candidate_interval_coverage": candidate_coverage,
        "carry_forward_interval_coverage": carry_coverage,
        "beats_carry_forward_mean": bool(candidate_mean is not None and carry_mean is not None and candidate_mean < carry_mean),
        "coverage_nonregression": bool(
            candidate_coverage is not None and carry_coverage is not None and candidate_coverage >= carry_coverage
        ),
    }


def _gate(
    *,
    support_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    support_by_metric = {str(row.get("metric_name") or ""): dict(row) for row in support_rows}
    direct_incidence_count = int((support_by_metric.get("incident_infections_period") or {}).get("count") or 0)
    diagnosis_flow_proxy_count = int((support_by_metric.get("new_diagnosed_cases_period") or {}).get("count") or 0)
    direct_death_count = int((support_by_metric.get("deaths_reported_period") or {}).get("count") or 0)
    by_bridge = {str(row.get("candidate_family") or "").split(":")[0]: dict(row) for row in family_rows}
    incidence_bridge = _bridge_gate_row(by_bridge.get("incidence_proxy_diagnosis_flow_bridge") or {}, bridge_id="incidence_proxy_diagnosis_flow_bridge")
    mortality_bridge = _bridge_gate_row(by_bridge.get("mortality_reported_death_bridge") or {}, bridge_id="mortality_reported_death_bridge")
    blockers: list[str] = []
    if direct_incidence_count <= 0:
        blockers.append("direct_incidence_process_support_absent")
    if diagnosis_flow_proxy_count <= 0:
        blockers.append("diagnosis_flow_proxy_support_absent")
    if direct_death_count <= 0:
        blockers.append("direct_reported_death_support_absent")
    if not incidence_bridge["beats_carry_forward_mean"]:
        blockers.append("diagnosis_flow_proxy_bridge_not_better_than_carry_forward")
    if not incidence_bridge["coverage_nonregression"]:
        blockers.append("diagnosis_flow_proxy_bridge_interval_coverage_worse_than_carry_forward")
    if not mortality_bridge["beats_carry_forward_mean"]:
        blockers.append("reported_death_bridge_not_better_than_carry_forward")
    if not mortality_bridge["coverage_nonregression"]:
        blockers.append("reported_death_bridge_interval_coverage_worse_than_carry_forward")
    mortality_ablations = [row for row in ablation_rows if str(row.get("bridge_id") or "") == "mortality_reported_death_bridge"]
    if not mortality_ablations:
        blockers.append("mortality_source_family_ablation_absent")
    elif any(str(row.get("ablation_status") or "") != "evaluable" for row in mortality_ablations):
        blockers.append("mortality_source_family_ablation_not_evaluable")
    elif any(
        _finite_float(row.get("ablated_candidate_mean_norm_error")) is None
        or _finite_float(row.get("ablated_carry_forward_mean_norm_error")) is None
        or float(row["ablated_candidate_mean_norm_error"]) >= float(row["ablated_carry_forward_mean_norm_error"])
        for row in mortality_ablations
    ):
        blockers.append("mortality_bridge_not_source_family_stable")
    if not blockers:
        status = "mechanism_support_expansion_ready"
    elif incidence_bridge["beats_carry_forward_mean"] or mortality_bridge["beats_carry_forward_mean"]:
        status = "proxy_bridge_signal_detected_but_mechanism_claim_blocked"
    else:
        status = "mechanism_support_expansion_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "direct_incidence_process_support_count": direct_incidence_count,
        "diagnosis_flow_proxy_support_count": diagnosis_flow_proxy_count,
        "direct_reported_death_support_count": direct_death_count,
        "incidence_proxy_bridge": incidence_bridge,
        "mortality_reported_death_bridge": mortality_bridge,
        "contract": (
            "R91 expands R89 by testing train-origin proxy bridges and source-family ablation. Diagnosis flow may be a "
            "proxy/nowcast support signal, but it is not direct incidence evidence. Mechanism claims remain blocked unless "
            "direct incidence process support exists and mortality bridge performance is stable under source-family ablation."
        ),
    }


def run_r91_mechanism_support_expansion_gate(
    *,
    run_id: str = R91_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    external_start_year: int = 2010,
    start_year: int = 2019,
    end_year: int = 2024,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    root = Path(epigraph_root) if epigraph_root is not None else _default_evidence_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(root, source_run_id=source_run_id, preferred=baseline_source_run_id)
    observation_rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    rows = _merge_external_targets_into_observations(observation_rows, target_rows)
    support_metrics = (
        "incident_infections_period",
        "new_diagnosed_cases_period",
        "annual_new_infections",
        "deaths_reported_period",
        "annual_aids_deaths",
    )
    support_rows = [_metric_support_summary(rows, metric) for metric in support_metrics]
    proxy_series_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    all_model_rows: list[dict[str, Any]] = []
    all_internal_rows: list[dict[str, Any]] = []
    for spec in BRIDGE_SPECS:
        proxy_series_rows.extend(list(_proxy_annual_series(rows, spec["proxy_metric"]).values()))
        score_rows, model_rows, internal_rows = _score_train_selected_bridge(
            rows,
            bridge_id=spec["bridge_id"],
            target_metric=spec["target_metric"],
            proxy_metric=spec["proxy_metric"],
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
        )
        all_score_rows.extend(score_rows)
        all_model_rows.extend(model_rows)
        all_internal_rows.extend(internal_rows)
    metric_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family",))
    source_family_rows = _source_family_rows(rows, tuple(spec["proxy_metric"] for spec in BRIDGE_SPECS))
    bridge_ablation_rows = _ablation_rows(
        rows,
        source_family_rows=source_family_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    gate = _gate(support_rows=support_rows, family_rows=family_rows, ablation_rows=bridge_ablation_rows)
    report_path = analysis_dir / "r91_mechanism_support_expansion_gate_report.json"
    markdown_path = analysis_dir / "r91_mechanism_support_expansion_gate_report.md"
    support_csv = analysis_dir / "r91_support_rows.csv"
    proxy_series_csv = analysis_dir / "r91_proxy_series_rows.csv"
    score_csv = analysis_dir / "r91_bridge_score_rows.csv"
    model_csv = analysis_dir / "r91_bridge_model_rows.csv"
    internal_csv = analysis_dir / "r91_internal_policy_score_rows.csv"
    metric_csv = analysis_dir / "r91_bridge_metric_rows.csv"
    horizon_csv = analysis_dir / "r91_bridge_horizon_rows.csv"
    family_csv = analysis_dir / "r91_bridge_family_rows.csv"
    source_family_csv = analysis_dir / "r91_source_family_rows.csv"
    ablation_csv = analysis_dir / "r91_source_family_ablation_rows.csv"
    report = {
        "schema_version": R91_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "candidate_family": R91_FAMILY,
        "mechanism_support_expansion_gate": gate,
        "support_rows": support_rows,
        "proxy_series_rows": proxy_series_rows,
        "bridge_score_rows": all_score_rows,
        "bridge_model_rows": all_model_rows,
        "internal_policy_score_rows": all_internal_rows,
        "bridge_metric_rows": metric_rows,
        "bridge_horizon_rows": horizon_rows,
        "bridge_family_rows": family_rows,
        "source_family_rows": source_family_rows,
        "source_family_ablation_rows": bridge_ablation_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
            "epigraph_root": root.as_posix(),
            "source_run_id": source_run_id,
            "baseline_source_run_id": baseline_source_run_id,
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "support_rows_csv": support_csv.as_posix(),
            "proxy_series_rows_csv": proxy_series_csv.as_posix(),
            "bridge_score_rows_csv": score_csv.as_posix(),
            "bridge_model_rows_csv": model_csv.as_posix(),
            "internal_policy_score_rows_csv": internal_csv.as_posix(),
            "bridge_metric_rows_csv": metric_csv.as_posix(),
            "bridge_horizon_rows_csv": horizon_csv.as_posix(),
            "bridge_family_rows_csv": family_csv.as_posix(),
            "source_family_rows_csv": source_family_csv.as_posix(),
            "source_family_ablation_rows_csv": ablation_csv.as_posix(),
        },
    }
    _write_csv(support_csv, support_rows)
    _write_csv(proxy_series_csv, proxy_series_rows)
    _write_csv(score_csv, all_score_rows)
    _write_csv(model_csv, all_model_rows)
    _write_csv(internal_csv, all_internal_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(source_family_csv, source_family_rows)
    _write_csv(ablation_csv, bridge_ablation_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("mechanism_support_expansion_gate") or {})
    lines = [
        "# Phase 3 R91 Mechanism Support Expansion Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        f"- Direct incidence support count: `{gate.get('direct_incidence_process_support_count')}`",
        f"- Diagnosis-flow proxy support count: `{gate.get('diagnosis_flow_proxy_support_count')}`",
        f"- Direct reported-death support count: `{gate.get('direct_reported_death_support_count')}`",
        "",
        "## Bridge Results",
        "",
        "| Bridge | Candidate Mean Error | Carry-Forward Mean Error | Candidate Coverage | Carry Coverage | Mean Beats Carry | Coverage Nonregression |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for key in ("incidence_proxy_bridge", "mortality_reported_death_bridge"):
        row = dict(gate.get(key) or {})
        lines.append(
            f"| `{row.get('bridge_id')}` | `{row.get('candidate_mean_norm_error')}` | "
            f"`{row.get('carry_forward_mean_norm_error')}` | `{row.get('candidate_interval_coverage')}` | "
            f"`{row.get('carry_forward_interval_coverage')}` | `{row.get('beats_carry_forward_mean')}` | "
            f"`{row.get('coverage_nonregression')}` |"
        )
    lines.extend(["", "## Support Rows", "", "| Metric | Count | Year Count | Roles | Allowed Use |", "|---|---:|---:|---|---|"])
    for row in report.get("support_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('count')}` | `{row.get('year_count')}` | "
            f"`{row.get('observation_roles')}` | `{row.get('allowed_use')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R91 mechanism-support expansion gate.")
    parser.add_argument("--run-id", default=R91_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r91_mechanism_support_expansion_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
