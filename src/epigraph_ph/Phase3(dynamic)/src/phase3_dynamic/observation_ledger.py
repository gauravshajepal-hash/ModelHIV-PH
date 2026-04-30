from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import read_json
from .art_retention_evidence import ART_RETENTION_CUMULATIVE_METRICS, ART_RETENTION_PROCESS_METRICS


OBSERVATION_ROLE_LEDGER_SCHEMA_VERSION = "phase3_observation_role_ledger.v1"
OBSERVATION_ROLES: tuple[str, ...] = (
    "direct_target",
    "auxiliary_likelihood",
    "validation_only",
    "prior_context",
    "quarantined",
)
TRAINING_USES_BY_ROLE: dict[str, str] = {
    "direct_target": "primary_training_and_scoring_target",
    "auxiliary_likelihood": "auxiliary_state_or_likelihood_evidence",
    "validation_only": "held_out_validation_or_diagnostic_only",
    "prior_context": "context_or_prior_only",
    "quarantined": "physically_unscorable_without_diagnostic_override",
}
MEASUREMENT_SEMANTICS: tuple[str, ...] = (
    "stock_anchor",
    "flow_count",
    "modeled_estimate",
    "proportion",
    "denominator",
    "reporting_process_covariate",
    "determinant_covariate",
)

DIRECT_TARGET_METRICS: frozenset[str] = frozenset(
    {
        "diagnosed_plhiv",
        "alive_on_art",
        "new_diagnosed_cases_period",
        "tested_for_viral_load",
        "virally_suppressed",
        "deaths_reported_period",
    }
)
AUXILIARY_LIKELIHOOD_METRICS: frozenset[str] = frozenset(
    {
        "estimated_plhiv",
        "median_cd4_at_enrollment",
        "on_art_not_suppressed",
        "deaths_reported_cumulative",
        "prep_newly_enrolled_period",
        *ART_RETENTION_PROCESS_METRICS,
    }
)
VALIDATION_ONLY_METRICS: frozenset[str] = frozenset(
    {
        "annual_new_infections",
        "annual_aids_deaths",
        "prep_people_receiving",
    }
)

_PREFERRED_ACTIVE_SOURCE_PREFIXES: tuple[str, ...] = (
    "tr-v3-current-champion-expanded-harp-compatibility-",
    "tr-v3-phase2-testing-prevention-rebuild-",
    "harp-archive-hiv-data-coverage-",
)
_PREFERRED_BASELINE_SOURCE_PREFIXES: tuple[str, ...] = (
    "harp-archive-wdi-standard-",
    "harp-archive-",
)


def historical_metric_rows_path(epigraph_root: Path, source_run_id: str) -> Path:
    return (
        Path(epigraph_root)
        / "artifacts"
        / "runs"
        / str(source_run_id)
        / "harp_archive"
        / "historical_metric_rows.json"
    )


def resolve_active_source_run_id(epigraph_root: Path, preferred: str | None = None) -> str:
    if preferred:
        preferred_path = historical_metric_rows_path(epigraph_root, preferred)
        if preferred_path.exists():
            return str(preferred)
    runs_dir = Path(epigraph_root) / "artifacts" / "runs"
    candidates = sorted(
        [
            path.parent.parent.name
            for path in runs_dir.glob("*/harp_archive/historical_metric_rows.json")
        ]
    )
    for prefix in _PREFERRED_ACTIVE_SOURCE_PREFIXES:
        preferred_candidates = [name for name in candidates if str(name).startswith(prefix)]
        if preferred_candidates:
            return str(preferred_candidates[-1])
    if not candidates:
        raise FileNotFoundError("No historical_metric_rows.json artifact was found under artifacts/runs.")
    return str(candidates[-1])


def resolve_baseline_source_run_id(
    epigraph_root: Path,
    *,
    source_run_id: str,
    preferred: str | None = None,
) -> str:
    if preferred:
        preferred_path = historical_metric_rows_path(epigraph_root, preferred)
        if preferred_path.exists():
            return str(preferred)
    runs_dir = Path(epigraph_root) / "artifacts" / "runs"
    candidates = sorted(
        [
            path.parent.parent.name
            for path in runs_dir.glob("*/harp_archive/historical_metric_rows.json")
        ]
    )
    for prefix in _PREFERRED_BASELINE_SOURCE_PREFIXES:
        preferred_candidates = [name for name in candidates if str(name).startswith(prefix)]
        if preferred_candidates:
            return str(preferred_candidates[-1])
    return str(source_run_id)


def build_source_row_path(archive_path: Path, row_index: int) -> str:
    return f"{archive_path.resolve().as_posix()}#row:{int(row_index)}"


def _time_granularity(row: dict[str, Any]) -> str:
    series_kind = str(row.get("series_kind") or "").lower()
    if "monthly" in series_kind:
        return "monthly"
    if "quarterly" in series_kind:
        return "quarterly"
    if "annual" in series_kind:
        return "annual"
    time_value = str(row.get("time") or row.get("period_end") or "")
    if len(time_value) >= 7:
        month_value = str(time_value[5:7])
        if month_value in {"03", "06", "09", "12"}:
            return "quarterly"
        return "monthly"
    if len(time_value) >= 4:
        return "annual"
    return "unknown"


def _time_start(row: dict[str, Any]) -> str:
    return str(row.get("period_start") or row.get("time") or row.get("period_end") or "")


def _time_end(row: dict[str, Any]) -> str:
    return str(row.get("period_end") or row.get("time") or row.get("period_start") or "")


def _geography(row: dict[str, Any]) -> str:
    province = str(row.get("province") or "").strip()
    region = str(row.get("region") or row.get("geo") or "").strip()
    if province and province.lower() != "philippines":
        return province
    if region:
        return region
    return "national"


def _population(row: dict[str, Any]) -> str:
    for key in ("population", "population_label", "population_group"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "all"


def _source_tier(row: dict[str, Any]) -> str:
    return str(
        row.get("source_tier")
        or row.get("source_quality_tier")
        or row.get("measurement_class")
        or "unspecified"
    ).strip()


def _is_external_reference(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(row.get("measurement_class") or ""),
            str(row.get("source_tier") or ""),
            str(row.get("source_quality_tier") or ""),
            str(row.get("integration_role") or ""),
            str(row.get("source_dataset") or ""),
            str(row.get("source_organization") or ""),
        ]
    ).lower()
    return "external" in text or "unaids" in text or "wdi" in text


def _metric_id(row: dict[str, Any]) -> str:
    return str(row.get("metric_name") or "").strip()


def infer_measurement_semantics(row: dict[str, Any]) -> str:
    metric_id = _metric_id(row)
    unit = str(row.get("unit") or "").lower()
    measurement_class = str(row.get("measurement_class") or "").lower()
    series_kind = str(row.get("series_kind") or "").lower()
    if metric_id in {
        "diagnosed_plhiv",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
        "deaths_reported_cumulative",
        "prep_people_receiving",
        *ART_RETENTION_CUMULATIVE_METRICS,
    }:
        return "stock_anchor"
    if metric_id in {
        "new_diagnosed_cases_period",
        "deaths_reported_period",
        "prep_newly_enrolled_period",
        *(metric for metric in ART_RETENTION_PROCESS_METRICS if metric not in ART_RETENTION_CUMULATIVE_METRICS),
    }:
        return "flow_count"
    if metric_id in {"estimated_plhiv", "annual_new_infections", "annual_aids_deaths"}:
        return "modeled_estimate"
    if "percent" in unit or unit.endswith("_percent"):
        return "proportion"
    if metric_id == "population_total" or "per_1000" in unit or "per_100k" in unit or metric_id.endswith("_rate"):
        return "denominator"
    if "cd4" in metric_id or "median_age" in metric_id or "positivity" in metric_id:
        return "reporting_process_covariate"
    if "model_estimate" in measurement_class:
        return "modeled_estimate"
    if "snapshot" in series_kind:
        return "stock_anchor"
    if "series" in series_kind or metric_id.startswith("annual_") or metric_id.endswith("_period"):
        return "flow_count"
    return "determinant_covariate"


def infer_observation_role(row: dict[str, Any]) -> tuple[str, str | None]:
    metric_id = _metric_id(row)
    time_end = _time_end(row)
    if not metric_id:
        return "quarantined", "missing_metric_id"
    if not time_end:
        return "quarantined", "missing_time_support"
    if row.get("value") is None:
        return "quarantined", "missing_value"
    if metric_id in DIRECT_TARGET_METRICS and not _is_external_reference(row):
        return "direct_target", None
    if metric_id in AUXILIARY_LIKELIHOOD_METRICS:
        return "auxiliary_likelihood", None
    if metric_id in VALIDATION_ONLY_METRICS:
        return "validation_only", None
    if _is_external_reference(row):
        semantics = infer_measurement_semantics(row)
        if semantics in {"modeled_estimate", "stock_anchor", "flow_count"}:
            return "validation_only", None
        return "prior_context", None
    return "prior_context", None


def build_contract_row_hash(
    *,
    row: dict[str, Any],
    source_path: str,
) -> str:
    payload = {
        "source_id": str(row.get("source_id") or ""),
        "source_path": str(source_path),
        "metric_id": _metric_id(row),
        "time_start": _time_start(row),
        "time_end": _time_end(row),
        "time_granularity": _time_granularity(row),
        "geography": _geography(row),
        "population": _population(row),
        "value": row.get("value"),
        "unit": str(row.get("unit") or ""),
        "extraction_method": str(row.get("extraction_method") or ""),
        "source_tier": _source_tier(row),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (_metric_id(row), _time_end(row), _geography(row), _population(row))


def _baseline_support_keys(epigraph_root: Path, baseline_source_run_id: str) -> set[tuple[str, str, str, str]]:
    baseline_path = historical_metric_rows_path(epigraph_root, baseline_source_run_id)
    rows = list(read_json(baseline_path, default=[]) or [])
    return {_support_key(row) for row in rows}


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observation_role_counts = Counter(str(row["observation_role"]) for row in rows)
    training_use_counts = Counter(str(row["training_use"]) for row in rows)
    measurement_semantics_counts = Counter(str(row["measurement_semantics"]) for row in rows)
    support_partition_counts = Counter(str(row["support_partition"]) for row in rows)
    leakage_status_counts = Counter(str(row["leakage_status"]) for row in rows)
    metric_counts: dict[str, dict[str, Any]] = {}
    for metric_id in sorted({str(row["metric_id"]) for row in rows}):
        metric_rows = [row for row in rows if str(row["metric_id"]) == metric_id]
        metric_counts[metric_id] = {
            "row_count": len(metric_rows),
            "observation_role_counts": dict(Counter(str(row["observation_role"]) for row in metric_rows)),
            "training_use_counts": dict(Counter(str(row["training_use"]) for row in metric_rows)),
            "support_partition_counts": dict(Counter(str(row["support_partition"]) for row in metric_rows)),
        }
    return {
        "row_count": len(rows),
        "observation_role_counts": dict(observation_role_counts),
        "training_use_counts": dict(training_use_counts),
        "measurement_semantics_counts": dict(measurement_semantics_counts),
        "support_partition_counts": dict(support_partition_counts),
        "leakage_status_counts": dict(leakage_status_counts),
        "metrics": metric_counts,
        "strict_contract": {
            "primary_training_roles": ["direct_target"],
            "auxiliary_training_roles": ["auxiliary_likelihood"],
            "validation_roles": ["validation_only"],
            "blocked_roles": ["quarantined"],
            "validation_only_training_allowed": False,
            "quarantined_training_allowed": False,
        },
    }


def build_observation_role_ledger(
    epigraph_root: Path,
    *,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root)
    archive_path = historical_metric_rows_path(epigraph_root, source_run_id)
    metric_rows = list(read_json(archive_path, default=[]) or [])
    if baseline_source_run_id is None:
        baseline_source_run_id = resolve_baseline_source_run_id(
            epigraph_root,
            source_run_id=source_run_id,
        )
    baseline_keys = _baseline_support_keys(epigraph_root, baseline_source_run_id)
    ledger_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(metric_rows):
        source_path = build_source_row_path(archive_path, row_index)
        observation_role, quarantine_reason = infer_observation_role(row)
        measurement_semantics = infer_measurement_semantics(row)
        support_partition = (
            "common_support"
            if _support_key(row) in baseline_keys
            else "expanded_support"
        )
        training_use = TRAINING_USES_BY_ROLE.get(
            observation_role,
            "context_or_prior_only",
        )
        ledger_rows.append(
            {
                "source_id": str(row.get("source_id") or ""),
                "source_path": source_path,
                "metric_id": _metric_id(row),
                "time_start": _time_start(row),
                "time_end": _time_end(row),
                "time_granularity": _time_granularity(row),
                "geography": _geography(row),
                "population": _population(row),
                "value": row.get("value"),
                "unit": str(row.get("unit") or ""),
                "extraction_method": str(row.get("extraction_method") or "unknown"),
                "source_tier": _source_tier(row),
                "source_quality_tier": str(row.get("source_quality_tier") or ""),
                "measurement_class": str(row.get("measurement_class") or ""),
                "observation_role": observation_role,
                "allowed_use": observation_role,
                "training_use": training_use,
                "support_partition": support_partition,
                "leakage_status": "split_unassigned",
                "measurement_semantics": measurement_semantics,
                "evidence_confidence": float(row.get("evidence_confidence") or 0.0),
                "row_hash": build_contract_row_hash(row=row, source_path=source_path),
                "quarantine_reason": quarantine_reason,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": OBSERVATION_ROLE_LEDGER_SCHEMA_VERSION,
        "source_run_id": str(source_run_id),
        "baseline_source_run_id": str(baseline_source_run_id),
        "archive_path": archive_path.resolve().as_posix(),
        "rows": ledger_rows,
        "summary": _build_summary(ledger_rows),
    }


def build_observation_contract_lookup(
    epigraph_root: Path,
    *,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ledger = build_observation_role_ledger(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    lookup = {str(row["row_hash"]): row for row in list(ledger["rows"])}
    return lookup, ledger
