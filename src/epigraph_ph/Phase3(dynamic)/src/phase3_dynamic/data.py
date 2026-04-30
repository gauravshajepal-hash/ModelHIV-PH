from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import PRIMARY_METRICS, quarter_sort_key, quarter_year
from .observation_ledger import (
    AUXILIARY_LIKELIHOOD_METRICS,
    DIRECT_TARGET_METRICS,
    VALIDATION_ONLY_METRICS,
    build_contract_row_hash,
    build_observation_contract_lookup,
    build_source_row_path,
)
from .runtime import read_json

STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "T", "V", "L", "R")
STATE_LABELS: dict[str, str] = {
    "U": "undiagnosed",
    "D": "diagnosed_not_on_art",
    "A": "active_art_without_recent_vl",
    "T": "vl_tested_unsuppressed",
    "V": "suppressed",
    "L": "interrupted_art",
    "R": "recently_reengaged_art",
}
DIAGNOSED_STATE_NAMES: tuple[str, ...] = ("D", "A", "T", "V", "L", "R")
ART_STATE_NAMES: tuple[str, ...] = ("A", "T", "V", "R")
VL_TESTED_STATE_NAMES: tuple[str, ...] = ("T", "V")
ART_INTERRUPTION_SOURCE_STATE_NAMES: tuple[str, ...] = ("A", "T", "V")
TRANSITION_NAMES: tuple[str, ...] = (
    "U_to_D",
    "D_to_A",
    "A_to_T",
    "T_to_V",
    "A_to_L",
    "T_to_L",
    "V_to_L",
    "L_to_R",
    "R_to_A",
)
EXTERNAL_EXIT_CHANNEL_NAMES: tuple[str, ...] = (
    "mortality_removal",
    "treatment_non_initiation",
    "unresolved_external_removal",
)
CASCADE_LEAKAGE_CHANNEL_NAMES: tuple[str, ...] = (
    "art_ltfu",
    "reengagement",
    "vl_testing_loss",
)
EXIT_CHANNEL_NAMES: tuple[str, ...] = EXTERNAL_EXIT_CHANNEL_NAMES + CASCADE_LEAKAGE_CHANNEL_NAMES
SNAPSHOT_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
    "estimated_plhiv",
)
FLOW_METRICS: tuple[str, ...] = (
    "new_diagnosed_cases_period",
    "deaths_reported_period",
    "prep_newly_enrolled_period",
)
AUXILIARY_METRICS: tuple[str, ...] = (
    "median_cd4_at_enrollment",
    "on_art_not_suppressed",
    "annual_new_infections",
    "annual_aids_deaths",
    "prep_people_receiving",
)
DENOMINATOR_CONTEXT_METRICS: tuple[str, ...] = ("population_total",)
OBSERVATION_METRICS: tuple[str, ...] = SNAPSHOT_METRICS + FLOW_METRICS + AUXILIARY_METRICS + DENOMINATOR_CONTEXT_METRICS

_PRIMARY_PRIORITY: dict[str, int] = {
    "official_user_provided_slide": 0,
    "official_local_corpus": 0,
    "official_mirror": 1,
    "official": 2,
    "model_estimate": 3,
    "derived": 4,
    "other": 5,
}

MISSING_DATA_LADDER: tuple[str, ...] = (
    "exact_observed",
    "bridge_observed",
    "rule_based_extrapolated",
    "latent_imputed",
    "rejected_or_quarantined",
)

_MISSING_DATA_RANK: dict[str, int] = {
    tier: index for index, tier in enumerate(MISSING_DATA_LADDER)
}


@dataclass(slots=True)
class BlockedTimeDataset:
    holdout_years: list[int]
    observation_rows: list[dict[str, Any]]
    train_rows: list[dict[str, Any]]
    holdout_rows: list[dict[str, Any]]
    train_state_rows: list[dict[str, Any]]
    holdout_state_rows: list[dict[str, Any]]
    train_transition_rows: list[dict[str, Any]]
    metric_scales: dict[str, float]
    eps: float
    provenance_summary: dict[str, Any]


def sandbox_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_epigraph_root() -> Path:
    sandbox_root = sandbox_repo_root()
    for candidate in (sandbox_root, *sandbox_root.parents):
        if (
            (candidate / "artifacts" / "runs").exists()
            and (candidate / "src" / "epigraph_ph").exists()
            and (candidate / "src" / "epigraph_ph" / "phase0").exists()
        ):
            return candidate
    return sandbox_root.parents[0]


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    return f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}"


def _month_from_time_label(time_label: str) -> int | None:
    try:
        return int(str(time_label).split("-", 1)[1][:2])
    except (IndexError, ValueError):
        return None


def _is_quarter_end_time_label(time_label: str) -> bool:
    return _month_from_time_label(time_label) in {3, 6, 9, 12}


def _priority_key(row: dict[str, Any]) -> tuple[float, float, str]:
    quality = str(row.get("source_quality_tier") or row.get("measurement_class") or "other")
    quality_rank = _PRIMARY_PRIORITY.get(quality, _PRIMARY_PRIORITY["other"])
    confidence = -float(row.get("evidence_confidence") or 0.0)
    source_id = str(row.get("source_id") or "")
    return float(quality_rank), confidence, source_id


def _empty_tier_counts() -> dict[str, int]:
    return {tier: 0 for tier in MISSING_DATA_LADDER}


def _increment_tier(counts: dict[str, int], tier: str | None) -> None:
    if tier is None:
        return
    counts.setdefault(tier, 0)
    counts[tier] += 1


def _worst_tier(tiers: list[str]) -> str | None:
    ranked = [tier for tier in tiers if tier in _MISSING_DATA_RANK]
    if not ranked:
        return None
    return max(ranked, key=lambda tier: _MISSING_DATA_RANK[tier])


def _base_missing_data_tier(row: dict[str, Any]) -> str:
    quality = str(row.get("source_quality_tier") or "").strip().lower()
    measurement = str(row.get("measurement_class") or "").strip().lower()
    source_tier = str(row.get("source_tier") or "").strip().lower()
    overlap_status = str(row.get("overlap_validation_status") or "").strip().lower()
    integration_role = str(row.get("integration_role") or "").strip().lower()
    if quality in {"official_user_provided_slide", "official_local_corpus", "official_mirror", "official"}:
        return "exact_observed"
    if measurement.startswith("program_observed") or measurement.startswith("surveillance_observed"):
        return "exact_observed"
    if quality == "model_estimate" or measurement == "model_estimate":
        return "rule_based_extrapolated"
    if quality == "derived" or measurement == "derived":
        return "rule_based_extrapolated"
    if source_tier in {"overlap_validated_external_reference", "canonical_external_reference_no_local_overlap", "reference_only_external_series"}:
        return "rule_based_extrapolated"
    if overlap_status or integration_role in {"external_reference", "supplemental_reference"}:
        return "rule_based_extrapolated"
    return "rule_based_extrapolated"


def _metric_provenance(
    row: dict[str, Any],
    *,
    aggregation_mode: str,
) -> dict[str, Any]:
    time_label = str(row.get("time") or row.get("period_end") or "")
    contract = dict(row.get("_contract") or {})
    tier = _base_missing_data_tier(row)
    if aggregation_mode == "monthly_to_quarter_sum":
        tier = "bridge_observed"
    elif aggregation_mode == "intraquarter_snapshot_bridge":
        tier = "bridge_observed"
    elif aggregation_mode == "annual_anchor_to_q4" and tier == "exact_observed":
        tier = "rule_based_extrapolated"
    return {
        "tier": tier,
        "aggregation_mode": aggregation_mode,
        "source_id": str(row.get("source_id") or ""),
        "source_quality_tier": str(row.get("source_quality_tier") or ""),
        "measurement_class": str(row.get("measurement_class") or ""),
        "series_kind": str(row.get("series_kind") or ""),
        "time": time_label,
        "source_path": str(contract.get("source_path") or ""),
        "source_tier": str(contract.get("source_tier") or row.get("source_tier") or ""),
        "extraction_method": str(contract.get("extraction_method") or row.get("extraction_method") or ""),
        "observation_role": str(contract.get("observation_role") or "prior_context"),
        "allowed_use": str(contract.get("allowed_use") or "prior_context"),
        "support_partition": str(contract.get("support_partition") or "common_support"),
        "leakage_status": str(contract.get("leakage_status") or "split_unassigned"),
        "measurement_semantics": str(contract.get("measurement_semantics") or ""),
        "row_hash": str(contract.get("row_hash") or ""),
    }


def _snapshot_aggregation_mode(row: dict[str, Any]) -> str:
    series_kind = str(row.get("series_kind") or "")
    time_label = str(row.get("time") or row.get("period_end") or "")
    if series_kind.startswith("quarterly"):
        return "quarterly_observed"
    if series_kind.startswith("annual"):
        return "annual_anchor_to_q4"
    if _is_quarter_end_time_label(time_label):
        return "quarter_end_snapshot"
    return "intraquarter_snapshot_bridge"


def _row_observation_role(row: dict[str, Any]) -> str:
    return str((row.get("_contract") or {}).get("observation_role") or "")


def _role_allowed_for_metric(
    *,
    metric_name: str,
    observation_role: str,
    include_validation_only: bool,
) -> bool:
    if metric_name in DENOMINATOR_CONTEXT_METRICS:
        return observation_role in {"prior_context", "auxiliary_likelihood", "validation_only"}
    if metric_name in DIRECT_TARGET_METRICS:
        return observation_role == "direct_target"
    if metric_name in AUXILIARY_LIKELIHOOD_METRICS:
        return observation_role == "auxiliary_likelihood"
    if metric_name in VALIDATION_ONLY_METRICS:
        return include_validation_only and observation_role == "validation_only"
    return False


def _metric_row_candidates(
    rows: list[dict[str, Any]],
    metric_name: str,
    *,
    include_validation_only: bool,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("region") or "").lower() != "national":
            continue
        if str(row.get("metric_name") or "") != metric_name:
            continue
        if not _role_allowed_for_metric(
            metric_name=metric_name,
            observation_role=_row_observation_role(row),
            include_validation_only=include_validation_only,
        ):
            continue
        time_label = str(row.get("time") or row.get("period_end") or "")
        if not time_label:
            continue
        if str(row.get("series_kind") or "").startswith("annual"):
            quarter = f"{int(str(time_label)[:4]):04d}-Q4"
        else:
            quarter = _quarter_from_month(time_label)
        grouped.setdefault(quarter, []).append(dict(row))
    return grouped


def _choose_primary_rows(candidates_by_quarter: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for quarter, rows in candidates_by_quarter.items():
        if not rows:
            continue
        chosen[quarter] = min(rows, key=_priority_key)
    return chosen


def build_observation_rows(
    epigraph_root: Path,
    source_run_id: str = "smoke-latent-blocks",
    *,
    baseline_source_run_id: str | None = None,
    allow_quarantined: bool = False,
    include_validation_only: bool = False,
) -> list[dict[str, Any]]:
    archive_path = epigraph_root / "artifacts" / "runs" / source_run_id / "harp_archive" / "historical_metric_rows.json"
    contract_lookup, _ledger = build_observation_contract_lookup(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    raw_rows = list(read_json(archive_path, default=[]) or [])
    metric_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = contract_lookup.get(row_hash)
        if contract is None:
            continue
        observation_role = str(contract.get("observation_role") or "")
        if observation_role == "quarantined" and not allow_quarantined:
            continue
        if observation_role == "validation_only" and not include_validation_only:
            continue
        metric_name = str(row.get("metric_name") or "")
        if observation_role not in {"direct_target", "auxiliary_likelihood", "validation_only"} and metric_name not in DENOMINATOR_CONTEXT_METRICS:
            continue
        enriched_row = dict(row)
        enriched_row["_contract"] = dict(contract)
        metric_rows.append(enriched_row)
    snapshot_maps = {
        metric_name: _choose_primary_rows(
            _metric_row_candidates(
                metric_rows,
                metric_name,
                include_validation_only=include_validation_only,
            )
        )
        for metric_name in SNAPSHOT_METRICS
    }
    flow_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for metric_name in FLOW_METRICS:
        flow_candidates = _metric_row_candidates(
            metric_rows,
            metric_name,
            include_validation_only=include_validation_only,
        )
        flow_map: dict[str, dict[str, Any]] = {}
        for quarter, rows in flow_candidates.items():
            quarterly = [row for row in rows if str(row.get("series_kind") or "").startswith("quarterly")]
            if quarterly:
                chosen = min(quarterly, key=_priority_key)
                flow_map[quarter] = {
                    "value": float(chosen.get("value") or 0.0),
                    "provenance": _metric_provenance(chosen, aggregation_mode="quarterly_observed"),
                }
                continue
            monthly = [row for row in rows if str(row.get("series_kind") or "").startswith("monthly")]
            if monthly:
                anchor = min(monthly, key=_priority_key)
                flow_map[quarter] = {
                    "value": float(sum(float(row.get("value") or 0.0) for row in monthly)),
                    "provenance": _metric_provenance(anchor, aggregation_mode="monthly_to_quarter_sum"),
                }
        flow_maps[metric_name] = flow_map
    auxiliary_maps = {
        metric_name: _choose_primary_rows(
            _metric_row_candidates(
                metric_rows,
                metric_name,
                include_validation_only=include_validation_only,
            )
        )
        for metric_name in AUXILIARY_METRICS
    }
    denominator_maps = {
        metric_name: _choose_primary_rows(
            _metric_row_candidates(
                metric_rows,
                metric_name,
                include_validation_only=include_validation_only,
            )
        )
        for metric_name in DENOMINATOR_CONTEXT_METRICS
    }
    quarter_set = sorted(
        {
            quarter
            for snapshot in snapshot_maps.values()
            for quarter in snapshot.keys()
        },
        key=quarter_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for quarter in quarter_set:
        diagnosed_row = snapshot_maps["diagnosed_plhiv"].get(quarter)
        alive_row = snapshot_maps["alive_on_art"].get(quarter)
        if diagnosed_row is None or alive_row is None:
            continue
        metric_provenance: dict[str, dict[str, Any] | None] = {}
        diagnosed_provenance = _metric_provenance(diagnosed_row, aggregation_mode=_snapshot_aggregation_mode(diagnosed_row))
        alive_provenance = _metric_provenance(alive_row, aggregation_mode=_snapshot_aggregation_mode(alive_row))
        row: dict[str, Any] = {
            "quarter": quarter,
            "diagnosed_plhiv": float(diagnosed_row.get("value") or 0.0),
            "alive_on_art": float(alive_row.get("value") or 0.0),
        }
        metric_provenance["diagnosed_plhiv"] = diagnosed_provenance
        metric_provenance["alive_on_art"] = alive_provenance
        for metric_name in ("tested_for_viral_load", "virally_suppressed", "estimated_plhiv"):
            candidate = snapshot_maps[metric_name].get(quarter)
            row[metric_name] = float(candidate.get("value") or 0.0) if candidate is not None else None
            metric_provenance[metric_name] = None if candidate is None else _metric_provenance(candidate, aggregation_mode=_snapshot_aggregation_mode(candidate))
        for metric_name in FLOW_METRICS:
            flow_entry = dict((flow_maps.get(metric_name) or {}).get(quarter) or {})
            row[metric_name] = None if not flow_entry else float(flow_entry["value"])
            metric_provenance[metric_name] = None if not flow_entry else dict(flow_entry["provenance"])
        for metric_name in AUXILIARY_METRICS:
            candidate = auxiliary_maps[metric_name].get(quarter)
            row[metric_name] = float(candidate.get("value") or 0.0) if candidate is not None else None
            metric_provenance[metric_name] = None if candidate is None else _metric_provenance(candidate, aggregation_mode=_snapshot_aggregation_mode(candidate))
        for metric_name in DENOMINATOR_CONTEXT_METRICS:
            candidate = denominator_maps[metric_name].get(quarter)
            row[metric_name] = float(candidate.get("value") or 0.0) if candidate is not None else None
            metric_provenance[metric_name] = None if candidate is None else _metric_provenance(candidate, aggregation_mode=_snapshot_aggregation_mode(candidate))
        row_tier = _worst_tier(
            [str(value.get("tier")) for value in metric_provenance.values() if isinstance(value, dict)]
        )
        row["metric_provenance"] = metric_provenance
        row["row_provenance_tier"] = row_tier
        rows.append(row)
    return _fill_population_denominators(rows)


def _fill_population_denominators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted([dict(row) for row in rows], key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    anchors = [
        (str(row.get("quarter") or ""), float(row.get("population_total") or 0.0), dict((row.get("metric_provenance") or {}).get("population_total") or {}))
        for row in sorted_rows
        if row.get("population_total") is not None and float(row.get("population_total") or 0.0) > 0.0
    ]
    if not anchors:
        return sorted_rows
    anchor_quarters = [quarter for quarter, _value, _provenance in anchors]
    for row in sorted_rows:
        if row.get("population_total") is not None and float(row.get("population_total") or 0.0) > 0.0:
            continue
        quarter = str(row.get("quarter") or "")
        prior = [(aq, value, provenance) for aq, value, provenance in anchors if quarter_sort_key(aq) <= quarter_sort_key(quarter)]
        if prior:
            source_quarter, value, provenance = prior[-1]
            fill_mode = "carry_forward_denominator_context"
        else:
            source_quarter, value, provenance = anchors[0]
            fill_mode = "backfill_denominator_context"
        filled_provenance = dict(provenance)
        filled_provenance["aggregation_mode"] = fill_mode
        filled_provenance["source_anchor_quarter"] = source_quarter
        row["population_total"] = float(value)
        metric_provenance = dict(row.get("metric_provenance") or {})
        metric_provenance["population_total"] = filled_provenance
        row["metric_provenance"] = metric_provenance
    return sorted_rows


def _coverage_mean(train_rows: list[dict[str, Any]]) -> float:
    coverages = []
    for row in train_rows:
        estimated = row.get("estimated_plhiv")
        diagnosed = row.get("diagnosed_plhiv")
        if estimated is None or diagnosed is None:
            continue
        estimated_value = float(estimated)
        if estimated_value <= 0.0:
            continue
        coverages.append(float(diagnosed) / estimated_value)
    if not coverages:
        raise ValueError("Need at least one train row with estimated_plhiv to build states")
    return float(np.mean(coverages))


def _share_mean(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float:
    shares = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            continue
        shares.append(float(numerator) / denominator_value)
    return float(np.mean(shares)) if shares else 0.0


def state_value(values: dict[str, Any], state_name: str) -> float:
    return max(float(values.get(state_name) or 0.0), 0.0)


def state_sum(values: dict[str, Any], state_names: tuple[str, ...]) -> float:
    return float(sum(state_value(values, state_name) for state_name in state_names))


def canonical_state_values(values: dict[str, Any]) -> dict[str, float]:
    return {state_name: state_value(values, state_name) for state_name in STATE_NAMES}


def _lost_gap_share(train_rows: list[dict[str, Any]]) -> float:
    shares = []
    for previous, current in zip(train_rows[:-1], train_rows[1:]):
        diagnosed_gap = max(float(previous.get("diagnosed_plhiv") or 0.0) - float(previous.get("alive_on_art") or 0.0), 0.0)
        art_drop = max(float(previous.get("alive_on_art") or 0.0) - float(current.get("alive_on_art") or 0.0), 0.0)
        if diagnosed_gap > 0.0 and art_drop > 0.0:
            shares.append(art_drop / diagnosed_gap)
    return float(np.mean(shares)) if shares else 0.0


def _state_row(
    row: dict[str, Any],
    *,
    coverage_mean: float,
    suppression_share_mean: float,
    vl_testing_share_mean: float,
    lost_gap_share: float,
) -> dict[str, Any]:
    quarter = str(row.get("quarter") or "")
    diagnosed = max(float(row.get("diagnosed_plhiv") or 0.0), 0.0)
    alive_on_art = max(float(row.get("alive_on_art") or 0.0), 0.0)
    observed_tested = row.get("tested_for_viral_load")
    observed_suppressed = row.get("virally_suppressed")
    metric_provenance = dict(row.get("metric_provenance") or {})
    if observed_tested is None:
        tested_for_viral_load = min(alive_on_art, max(vl_testing_share_mean * alive_on_art, 0.0))
        testing_tier = "latent_imputed"
    else:
        tested_for_viral_load = min(alive_on_art, max(float(observed_tested), 0.0))
        testing_provenance = metric_provenance.get("tested_for_viral_load")
        testing_tier = str((testing_provenance or {}).get("tier") or "rule_based_extrapolated")
    if observed_suppressed is None:
        virally_suppressed = min(alive_on_art, max(suppression_share_mean * alive_on_art, 0.0))
        suppression_tier = "latent_imputed"
    else:
        virally_suppressed = min(alive_on_art, max(float(observed_suppressed), 0.0))
        suppression_provenance = metric_provenance.get("virally_suppressed")
        suppression_tier = str((suppression_provenance or {}).get("tier") or "rule_based_extrapolated")
    tested_for_viral_load = min(alive_on_art, max(tested_for_viral_load, virally_suppressed))
    diagnosed_gap = max(diagnosed - alive_on_art, 0.0)
    lost = min(diagnosed_gap, max(lost_gap_share * diagnosed_gap, 0.0))
    diagnosed_not_on_art = max(diagnosed_gap - lost, 0.0)
    vl_tested_unsuppressed = max(tested_for_viral_load - virally_suppressed, 0.0)
    active_art_without_recent_vl = max(alive_on_art - tested_for_viral_load, 0.0)
    estimated_total = row.get("estimated_plhiv")
    estimated_tier = str(((metric_provenance.get("estimated_plhiv") or {}).get("tier")) or "rule_based_extrapolated")
    if estimated_total is None or float(estimated_total) <= 0.0:
        estimated_total = diagnosed / max(coverage_mean, 1e-6)
        estimated_tier = "latent_imputed"
    estimated_total = max(float(estimated_total), diagnosed)
    undiagnosed = max(float(estimated_total) - diagnosed, 0.0)
    state_provenance = {
        "U": estimated_tier,
        "D": "rule_based_extrapolated",
        "A": testing_tier,
        "T": _worst_tier([testing_tier, suppression_tier]) or "rule_based_extrapolated",
        "V": suppression_tier,
        "L": "rule_based_extrapolated",
        "R": "rule_based_extrapolated",
    }
    return {
        "quarter": quarter,
        "diagnosed_plhiv": diagnosed,
        "alive_on_art": alive_on_art,
        "new_diagnosed_cases_period": row.get("new_diagnosed_cases_period"),
        "tested_for_viral_load": float(tested_for_viral_load),
        "virally_suppressed": float(virally_suppressed),
        "estimated_plhiv": estimated_total,
        "population_total": row.get("population_total"),
        "deaths_reported_period": row.get("deaths_reported_period"),
        "annual_aids_deaths": row.get("annual_aids_deaths"),
        "on_art_not_suppressed": row.get("on_art_not_suppressed"),
        "metric_provenance": metric_provenance,
        "row_provenance_tier": row.get("row_provenance_tier"),
        "state_provenance": state_provenance,
        "state_row_tier": _worst_tier(list(state_provenance.values())),
        "state_values": {
            "U": float(undiagnosed),
            "D": float(diagnosed_not_on_art),
            "A": float(active_art_without_recent_vl),
            "T": float(vl_tested_unsuppressed),
            "V": float(virally_suppressed),
            "L": float(lost),
            "R": 0.0,
        },
    }


def build_state_rows(observation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_params = _state_row_params(observation_rows)
    return build_state_rows_with_params(observation_rows, state_params=state_params)


def _state_row_params(train_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "coverage_mean": _coverage_mean(train_rows),
        "suppression_share_mean": _share_mean(train_rows, "virally_suppressed", "alive_on_art"),
        "vl_testing_share_mean": _share_mean(train_rows, "tested_for_viral_load", "alive_on_art"),
        "lost_gap_share": _lost_gap_share(train_rows),
    }


def build_state_rows_with_params(
    observation_rows: list[dict[str, Any]],
    *,
    state_params: dict[str, float],
) -> list[dict[str, Any]]:
    base_rows = [
        _state_row(
            row,
            coverage_mean=float(state_params["coverage_mean"]),
            suppression_share_mean=float(state_params["suppression_share_mean"]),
            vl_testing_share_mean=float(state_params["vl_testing_share_mean"]),
            lost_gap_share=float(state_params["lost_gap_share"]),
        )
        for row in observation_rows
    ]
    return _assign_reengaged_art_state(base_rows)


def _assign_reengaged_art_state(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted([dict(row) for row in state_rows], key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    for index, row in enumerate(rows):
        state = canonical_state_values(dict(row.get("state_values") or {}))
        if index == 0:
            row["state_values"] = state
            continue
        previous = canonical_state_values(dict(rows[index - 1].get("state_values") or {}))
        previous_alive = state_sum(previous, ART_STATE_NAMES)
        current_alive = state_sum(state, ART_STATE_NAMES)
        art_growth = max(current_alive - previous_alive, 0.0)
        reengaged = min(max(previous["L"] - state["L"], 0.0), art_growth, state["A"])
        if reengaged > 0.0:
            state["A"] = max(state["A"] - reengaged, 0.0)
            state["R"] = float(reengaged)
            provenance = dict(row.get("state_provenance") or {})
            provenance["R"] = "rule_based_extrapolated"
            row["state_provenance"] = provenance
            row["state_row_tier"] = _worst_tier(list(provenance.values()))
        row["state_values"] = state
    return rows


def _allocate_state_total(total: float, state_basis: dict[str, float]) -> dict[str, float]:
    target = max(float(total), 0.0)
    if target <= 0.0:
        return {name: 0.0 for name in STATE_NAMES}
    positive_basis = {name: max(float(state_basis.get(name) or 0.0), 0.0) for name in STATE_NAMES}
    basis_total = float(sum(positive_basis.values()))
    if basis_total <= 0.0:
        return {name: 0.0 for name in STATE_NAMES}
    return {
        name: float(target) * float(positive_basis[name]) / basis_total
        for name in STATE_NAMES
    }


def _death_support_value(current: dict[str, Any]) -> float | None:
    reported_deaths = current.get("deaths_reported_period")
    if reported_deaths is not None:
        return max(float(reported_deaths), 0.0)
    annual_deaths = current.get("annual_aids_deaths")
    if annual_deaths is not None:
        return max(float(annual_deaths), 0.0) / 4.0
    return None


def _exit_channel_state_outflows(
    *,
    state_attrition_outflows: dict[str, float],
    current: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any]]:
    total_attrition = float(sum(max(float(value), 0.0) for value in state_attrition_outflows.values()))
    death_support = _death_support_value(current)
    mortality_total = 0.0 if death_support is None else min(float(death_support), total_attrition)
    mortality_by_state = _allocate_state_total(mortality_total, state_attrition_outflows)
    residual_by_state = {
        name: max(float(state_attrition_outflows.get(name) or 0.0) - float(mortality_by_state.get(name) or 0.0), 0.0)
        for name in STATE_NAMES
    }
    treatment_non_initiation_by_state = {name: 0.0 for name in STATE_NAMES}
    treatment_non_initiation_by_state["D"] = float(residual_by_state.get("D") or 0.0)
    unresolved_by_state = {
        name: 0.0 if name == "D" else float(residual_by_state.get(name) or 0.0)
        for name in STATE_NAMES
    }
    channel_state = {
        "mortality_removal": mortality_by_state,
        "treatment_non_initiation": treatment_non_initiation_by_state,
        "unresolved_external_removal": unresolved_by_state,
    }
    channel_totals = {
        channel: float(sum(float(values.get(name) or 0.0) for name in STATE_NAMES))
        for channel, values in channel_state.items()
    }
    support = {
        "mortality_removal": "death_observation_bound" if death_support is not None else "not_observed_in_split",
        "treatment_non_initiation": "diagnosed_not_art_residual_proxy",
        "unresolved_external_removal": "residual_not_yet_identified",
        "death_support_value": death_support,
    }
    return channel_state, channel_totals, support


def summarize_observation_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_tier_counts: dict[str, dict[str, int]] = {}
    missing_metric_counts: dict[str, int] = {}
    row_tier_counts = _empty_tier_counts()
    for row in rows:
        metric_provenance = dict(row.get("metric_provenance") or {})
        for metric_name in OBSERVATION_METRICS:
            metric_tier_counts.setdefault(metric_name, _empty_tier_counts())
            missing_metric_counts.setdefault(metric_name, 0)
            provenance = metric_provenance.get(metric_name)
            if provenance is None or row.get(metric_name) is None:
                missing_metric_counts[metric_name] += 1
                continue
            _increment_tier(metric_tier_counts[metric_name], str(provenance.get("tier") or "rejected_or_quarantined"))
        _increment_tier(row_tier_counts, str(row.get("row_provenance_tier") or "rejected_or_quarantined"))
    return {
        "row_count": len(rows),
        "row_tier_counts": row_tier_counts,
        "metric_tier_counts": metric_tier_counts,
        "missing_metric_counts": missing_metric_counts,
    }


def summarize_state_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_tier_counts = {state_name: _empty_tier_counts() for state_name in STATE_NAMES}
    row_tier_counts = _empty_tier_counts()
    for row in rows:
        provenance = dict(row.get("state_provenance") or {})
        for state_name in STATE_NAMES:
            _increment_tier(state_tier_counts[state_name], str(provenance.get(state_name) or "rejected_or_quarantined"))
        _increment_tier(row_tier_counts, str(row.get("state_row_tier") or "rejected_or_quarantined"))
    return {
        "row_count": len(rows),
        "row_tier_counts": row_tier_counts,
        "state_tier_counts": state_tier_counts,
    }


def _flow_from_row(current: dict[str, Any], previous_state: dict[str, float], next_state: dict[str, float], eps: float) -> float:
    flow = current.get("new_diagnosed_cases_period")
    if flow is not None:
        return max(float(flow), 0.0)
    diagnosed_delta = max(
        state_sum(next_state, DIAGNOSED_STATE_NAMES)
        - state_sum(previous_state, DIAGNOSED_STATE_NAMES),
        0.0,
    )
    return max(diagnosed_delta, 0.0 * eps)


def _allocate_over_names(total: float, state_basis: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    target = max(float(total), 0.0)
    basis = {name: max(float(state_basis.get(name) or 0.0), 0.0) for name in names}
    basis_total = float(sum(basis.values()))
    if target <= 0.0 or basis_total <= 0.0:
        return {name: 0.0 for name in names}
    return {name: target * basis[name] / basis_total for name in names}


def transition_rows(state_rows: list[dict[str, Any]], *, eps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous, current in zip(state_rows[:-1], state_rows[1:]):
        previous_state = canonical_state_values(dict(previous["state_values"]))
        current_state = canonical_state_values(dict(current["state_values"]))
        previous_total = state_sum(previous_state, STATE_NAMES)
        current_total = state_sum(current_state, STATE_NAMES)
        previous_population = previous.get("population_total")
        current_population = current.get("population_total")
        previous_population_value = None if previous_population is None else max(float(previous_population), 0.0)
        current_population_value = None if current_population is None else max(float(current_population), 0.0)
        susceptible_previous = None if previous_population_value is None else max(float(previous_population_value) - previous_total, 0.0)
        susceptible_current = None if current_population_value is None else max(float(current_population_value) - current_total, 0.0)
        u_to_d = _flow_from_row(current, previous_state, current_state, eps)
        u_mass_balance = float(current_state["U"]) - float(previous_state["U"]) + float(u_to_d)
        total_delta = float(current_total - previous_total)
        incidence_inflow = max(u_mass_balance, 0.0)
        l_to_r = min(float(current_state["R"]), float(previous_state["L"]))
        r_to_a = float(previous_state["R"])
        art_interruption_total = max(float(current_state["L"]) - float(previous_state["L"]) + l_to_r, 0.0)
        interruption_alloc = _allocate_over_names(art_interruption_total, previous_state, ART_INTERRUPTION_SOURCE_STATE_NAMES)
        a_to_l = float(interruption_alloc["A"])
        t_to_l = float(interruption_alloc["T"])
        v_to_l = float(interruption_alloc["V"])
        t_to_v = max(float(current_state["V"]) - float(previous_state["V"]) + v_to_l, 0.0)
        a_to_t = max(float(current_state["T"]) - float(previous_state["T"]) + t_to_v + t_to_l, 0.0)
        d_to_a = max(float(current_state["A"]) - float(previous_state["A"]) - r_to_a + a_to_t + a_to_l, 0.0)
        flows = {
            "U_to_D": u_to_d,
            "D_to_A": d_to_a,
            "A_to_T": a_to_t,
            "T_to_V": t_to_v,
            "A_to_L": a_to_l,
            "T_to_L": t_to_l,
            "V_to_L": v_to_l,
            "L_to_R": l_to_r,
            "R_to_A": r_to_a,
        }
        pre_exit_state = {
            "U": max(float(previous_state["U"]) - u_to_d + incidence_inflow, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a + r_to_a - a_to_t - a_to_l, 0.0),
            "T": max(float(previous_state["T"]) + a_to_t - t_to_v - t_to_l, 0.0),
            "V": max(float(previous_state["V"]) + t_to_v - v_to_l, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l + t_to_l + v_to_l - l_to_r, 0.0),
            "R": max(float(previous_state["R"]) + l_to_r - r_to_a, 0.0),
        }
        state_attrition_outflows = {
            name: max(float(pre_exit_state[name]) - float(current_state[name]), 0.0)
            for name in STATE_NAMES
        }
        attrition_outflow = float(sum(state_attrition_outflows.values()))
        exit_channel_state_outflows, exit_channel_totals, exit_channel_support = _exit_channel_state_outflows(
            state_attrition_outflows=state_attrition_outflows,
            current=current,
        )
        tested_for_viral_load = current.get("tested_for_viral_load")
        vl_testing_loss = current_state["A"] if tested_for_viral_load is not None else None
        exit_channel_totals["art_ltfu"] = float(a_to_l + t_to_l + v_to_l)
        exit_channel_totals["reengagement"] = float(l_to_r)
        exit_channel_totals["vl_testing_loss"] = None if vl_testing_loss is None else float(vl_testing_loss)
        exit_channel_support["art_ltfu"] = "A/T/V_to_L_state_transitions"
        exit_channel_support["reengagement"] = "L_to_R_state_transition"
        exit_channel_support["vl_testing_loss"] = "active_art_without_recent_vl_state" if vl_testing_loss is not None else "not_observed_in_split"
        denominators = {
            "U_to_D": max(float(previous_state["U"]), eps),
            "D_to_A": max(float(previous_state["D"]), eps),
            "A_to_T": max(float(previous_state["A"]), eps),
            "T_to_V": max(float(previous_state["T"]), eps),
            "A_to_L": max(float(previous_state["A"]), eps),
            "T_to_L": max(float(previous_state["T"]), eps),
            "V_to_L": max(float(previous_state["V"]), eps),
            "L_to_R": max(float(previous_state["L"]), eps),
            "R_to_A": max(float(previous_state["R"]), eps),
        }
        rows.append(
            {
                "quarter": str(current["quarter"]),
                "previous_quarter": str(previous["quarter"]),
                "flows": {name: float(value) for name, value in flows.items()},
                "hazards": {name: float(flows[name]) / denominators[name] for name in TRANSITION_NAMES},
                "stock_balance": {
                    "previous_total": float(previous_total),
                    "current_total": float(current_total),
                    "total_delta": float(total_delta),
                    "u_mass_balance": float(u_mass_balance),
                    "incidence_inflow": float(incidence_inflow),
                    "population_denominator_previous": previous_population_value,
                    "population_denominator_current": current_population_value,
                    "susceptible_effective_previous": susceptible_previous,
                    "susceptible_effective_current": susceptible_current,
                    "incidence_hazard_per_s_eff": None if susceptible_previous is None or susceptible_previous <= eps else float(incidence_inflow) / max(float(susceptible_previous), eps),
                    "attrition_outflow": float(attrition_outflow),
                    "state_attrition_outflows": state_attrition_outflows,
                    "exit_channel_state_outflows": exit_channel_state_outflows,
                    "exit_channel_outflows": exit_channel_totals,
                    "exit_channel_support": exit_channel_support,
                },
                "state_values_previous": {name: float(previous_state[name]) for name in STATE_NAMES},
                "state_values_current": {name: float(current_state[name]) for name in STATE_NAMES},
            }
        )
    return rows


def rolling_origin_splits(
    observation_rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> list[dict[str, Any]]:
    available_years = sorted({quarter_year(str(row.get("quarter") or "")) for row in observation_rows})
    bounded_years = [year for year in available_years if int(start_year) <= int(year) <= int(end_year)]
    splits: list[dict[str, Any]] = []
    for train_end_year in bounded_years:
        train_years = [year for year in bounded_years if year <= train_end_year]
        holdout_years = [year for year in bounded_years if train_end_year < year <= train_end_year + int(horizon_years)]
        if len(train_years) < int(min_train_years) or not holdout_years:
            continue
        splits.append(
            {
                "train_end_year": int(train_end_year),
                "train_years": list(train_years),
                "holdout_years": list(holdout_years),
            }
        )
    return splits


def build_blocked_time_dataset(observation_rows: list[dict[str, Any]], holdout_years: list[int]) -> BlockedTimeDataset:
    train_rows = [
        dict(row)
        for row in observation_rows
        if row.get("diagnosed_plhiv") is not None
        and row.get("alive_on_art") is not None
        and quarter_year(str(row.get("quarter") or "")) < min(holdout_years)
    ]
    holdout_rows = [
        dict(row)
        for row in observation_rows
        if row.get("diagnosed_plhiv") is not None
        and row.get("alive_on_art") is not None
        and quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
    ]
    state_params = _state_row_params(train_rows)
    train_state_rows = build_state_rows_with_params(train_rows, state_params=state_params)
    holdout_state_rows = build_state_rows_with_params(holdout_rows, state_params=state_params)
    eps = float(np.finfo(np.float32).eps)
    train_transition_rows = [
        dict(row)
        for row in transition_rows(train_state_rows, eps=eps)
        if quarter_year(str(row.get("quarter") or "")) < min(holdout_years)
    ]
    metric_scales = {}
    for metric_name in PRIMARY_METRICS:
        observed = [abs(float(row.get(metric_name) or 0.0)) for row in train_rows if row.get(metric_name) is not None]
        metric_scales[metric_name] = max(observed) if observed else eps
    provenance_summary = {
        "ladder": list(MISSING_DATA_LADDER),
        "observation_rows": summarize_observation_provenance(train_rows + holdout_rows),
        "train_observation_rows": summarize_observation_provenance(train_rows),
        "holdout_observation_rows": summarize_observation_provenance(holdout_rows),
        "train_state_rows": summarize_state_provenance(train_state_rows),
        "holdout_state_rows": summarize_state_provenance(holdout_state_rows),
        "state_parameter_contract": "train_rows_only",
        "metric_scale_contract": "train_rows_only",
    }
    return BlockedTimeDataset(
        holdout_years=list(holdout_years),
        observation_rows=list(train_rows + holdout_rows),
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        train_state_rows=train_state_rows,
        holdout_state_rows=holdout_state_rows,
        train_transition_rows=train_transition_rows,
        metric_scales=metric_scales,
        eps=eps,
        provenance_summary=provenance_summary,
    )


def quarter_positions(rows: list[dict[str, Any]]) -> dict[str, int]:
    quarters = sorted({str(row.get("quarter") or "") for row in rows if str(row.get("quarter") or "")}, key=quarter_sort_key)
    return {quarter: idx for idx, quarter in enumerate(quarters)}

EARLY_PARTIAL_METRICS: tuple[str, ...] = (
    'annual_aids_deaths',
    'annual_new_infections',
    'art_median_age',
    'estimated_plhiv',
    'youth_cases_15_24_period',
)


def _year_from_time_label(value: str) -> int:
    return int(str(value)[:4])


def build_early_partial_target_rows(epigraph_root: Path, source_run_id: str = 'smoke-latent-blocks', *, start_year: int = 2010, end_year: int = 2016) -> list[dict[str, Any]]:
    archive_path = epigraph_root / 'artifacts' / 'runs' / source_run_id / 'harp_archive' / 'historical_metric_rows.json'
    metric_rows = list(read_json(archive_path, default=[]) or [])
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for row in metric_rows:
        if str(row.get('region') or '').lower() != 'national':
            continue
        metric_name = str(row.get('metric_name') or '')
        if metric_name not in EARLY_PARTIAL_METRICS:
            continue
        time_label = str(row.get('time') or row.get('period_end') or '')
        if not time_label:
            continue
        year = _year_from_time_label(time_label)
        if year < int(start_year) or year > int(end_year):
            continue
        key = (metric_name, year)
        candidate = {'metric_name': metric_name, 'year': int(year), 'value': float(row.get('value') or 0.0), 'source_quality_tier': str(row.get('source_quality_tier') or row.get('measurement_class') or 'other'), 'evidence_confidence': float(row.get('evidence_confidence') or 0.0), 'source_id': str(row.get('source_id') or ''), 'missing_data_tier': _base_missing_data_tier(row)}
        previous = chosen.get(key)
        if previous is None or _priority_key(candidate) < _priority_key(previous):
            chosen[key] = candidate
    return sorted(chosen.values(), key=lambda row: (str(row['metric_name']), int(row['year'])))


def summarize_early_partial_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts = _empty_tier_counts()
    metric_tier_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        metric_name = str(row.get("metric_name") or "")
        metric_tier_counts.setdefault(metric_name, _empty_tier_counts())
        tier = str(row.get("missing_data_tier") or "rejected_or_quarantined")
        _increment_tier(tier_counts, tier)
        _increment_tier(metric_tier_counts[metric_name], tier)
    return {
        "row_count": len(rows),
        "row_tier_counts": tier_counts,
        "metric_tier_counts": metric_tier_counts,
    }


def partial_observation_splits(target_rows: list[dict[str, Any]], *, start_year: int, end_year: int, min_train_years: int, horizon_years: int) -> list[dict[str, Any]]:
    available_years = sorted({int(row['year']) for row in target_rows if int(start_year) <= int(row['year']) <= int(end_year)})
    splits: list[dict[str, Any]] = []
    for train_end_year in available_years:
        train_years = [year for year in available_years if year <= train_end_year]
        holdout_years = [year for year in available_years if train_end_year < year <= train_end_year + int(horizon_years)]
        if len(train_years) < int(min_train_years) or not holdout_years:
            continue
        for holdout_year in holdout_years:
            splits.append({'train_end_year': int(train_end_year), 'train_years': list(train_years), 'holdout_year': int(holdout_year)})
    return splits
