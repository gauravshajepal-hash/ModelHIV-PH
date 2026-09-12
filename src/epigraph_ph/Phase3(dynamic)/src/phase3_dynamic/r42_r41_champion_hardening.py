from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .r11_sparse_state_space import (
    R10_COMPARABLE_METRICS,
    _build_r12_official_annual_challenge_gate_report,
    _candidate_predictions,
    _finite_float,
    _generated_at,
    _sha256,
    _write_r12_official_annual_challenge_dashboard,
)
from .r28_r10_contract_lineage_audit import _collect_r10_records
from .r29_strict_ledger_matched_r10_gate import (
    R29_RUN_ID,
    _is_program_lineage,
    _reference_by_scope_horizon,
    _strict_reference_rows,
    run_r29_strict_ledger_matched_r10_gate,
)
from .runtime import ensure_dir, read_json, write_json


R42_SCHEMA_VERSION = "phase3_dynamic.r42_r41_champion_hardening.v1"
R42_RUN_ID = "p3d-r42-r41-champion-hardening-20260502-s00"
R42_FAMILY = "r41_monotone_growth_component_process"
R42_REQUIRED_ROUTES: tuple[tuple[str, int], ...] = (
    ("all_mapped", 1),
    ("all_mapped", 3),
    ("all_mapped", 5),
    ("program_mapped", 3),
    ("program_mapped", 5),
)


def _source_rows_by_quarter(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("quarter") or ""): dict(row)
        for row in rows
        if row.get("quarter")
    }


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


def _record_scope(record: dict[str, Any], scope: str) -> bool:
    if bool(record.get("unknown_provenance")):
        return False
    if scope == "all_mapped":
        return True
    if scope == "program_mapped":
        return _is_program_lineage(record)
    raise ValueError(f"Unknown R42 route scope: {scope}")


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record.get("unknown_provenance")):
            continue
        grouped[(int(record.get("train_end_year") or 0), int(record.get("horizon_years") or 0))].append(dict(record))
    return grouped


def _source_family_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    metric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric_name in R10_COMPARABLE_METRICS:
            provenance = _metric_provenance(row, metric_name)
            if not provenance:
                continue
            family = _source_family_from_provenance(provenance)
            counts[family] += 1
            metric_counts[family][metric_name] += 1
    output: list[dict[str, Any]] = []
    for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        output.append(
            {
                "source_family": family,
                "entry_count": int(count),
                "metric_counts": dict(sorted(metric_counts[family].items())),
            }
        )
    return output


def _ablate_training_rows(train_rows: list[dict[str, Any]], source_family: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_counts: Counter[str] = Counter()
    affected_quarters: set[str] = set()
    for source_row in train_rows:
        row = dict(source_row)
        provenance = dict(row.get("metric_provenance") or {})
        for metric_name in R10_COMPARABLE_METRICS:
            metric_provenance = provenance.get(metric_name)
            if not isinstance(metric_provenance, dict):
                continue
            if _source_family_from_provenance(metric_provenance) != source_family:
                continue
            row[metric_name] = None
            updated = dict(metric_provenance)
            updated["ablation_status"] = "training_metric_removed"
            updated["ablated_source_family"] = source_family
            provenance[metric_name] = updated
            metric_counts[metric_name] += 1
            if row.get("quarter"):
                affected_quarters.add(str(row.get("quarter") or ""))
        row["metric_provenance"] = provenance
        rows.append(row)
    return rows, {
        "excluded_source_family": source_family,
        "removed_metric_counts": dict(sorted(metric_counts.items())),
        "affected_quarter_count": len(affected_quarters),
        "affected_quarters": sorted(affected_quarters, key=quarter_sort_key),
    }


def _prediction_rows(
    *,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    excluded_source_family: str | None = None,
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    output: dict[tuple[int, int, str], dict[str, Any]] = {}
    origin_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for (train_end_year, horizon), grouped_records in sorted(_group_records(records).items()):
        train_rows = [
            dict(row)
            for row in source_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        ablation_summary: dict[str, Any] | None = None
        if excluded_source_family is not None:
            train_rows, ablation_summary = _ablate_training_rows(train_rows, excluded_source_family)
        holdout_quarters = sorted(
            {str(record.get("quarter") or "") for record in grouped_records if record.get("quarter")},
            key=quarter_sort_key,
        )
        holdout_rows = [
            dict(source_by_quarter[quarter])
            for quarter in holdout_quarters
            if quarter in source_by_quarter
        ]
        if not train_rows or not holdout_rows:
            failures.append(
                {
                    "train_end_year": int(train_end_year),
                    "horizon_years": int(horizon),
                    "failure": "missing_train_or_holdout_rows",
                    "excluded_source_family": excluded_source_family,
                }
            )
            continue
        try:
            predictions, summary = _candidate_predictions(train_rows, holdout_rows, family=R42_FAMILY)
        except Exception as exc:  # pragma: no cover - persisted as diagnostic, not swallowed silently.
            failures.append(
                {
                    "train_end_year": int(train_end_year),
                    "horizon_years": int(horizon),
                    "failure": type(exc).__name__,
                    "message": str(exc),
                    "excluded_source_family": excluded_source_family,
                }
            )
            continue
        origin_rows.append(
            {
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon),
                "holdout_quarter_count": len(holdout_rows),
                "train_quarter_count": len(train_rows),
                "excluded_source_family": excluded_source_family,
                "ablation_summary": ablation_summary,
                "model_summary": summary,
            }
        )
        for row in predictions:
            quarter = str(row.get("quarter") or "")
            if quarter:
                output[(int(train_end_year), int(horizon), quarter)] = dict(row)
    return output, origin_rows, failures


def _score_records(
    *,
    records: list[dict[str, Any]],
    prediction_rows: dict[tuple[int, int, str], dict[str, Any]],
    source_family: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metric_name = str(record.get("metric_name") or "")
        if metric_name not in R10_COMPARABLE_METRICS:
            continue
        key = (
            int(record.get("train_end_year") or 0),
            int(record.get("horizon_years") or 0),
            str(record.get("quarter") or ""),
        )
        prediction = prediction_rows.get(key, {})
        value = _finite_float(prediction.get(metric_name))
        target = _finite_float(record.get("target_value"))
        scale = _finite_float(record.get("scale"))
        if target is None or scale is None:
            continue
        scale = max(float(scale), float(np.finfo(np.float32).eps))
        candidate_error = None if value is None else abs(float(value) - float(target)) / scale
        output = {
            key_name: record.get(key_name)
            for key_name in (
                "horizon_years",
                "train_end_year",
                "target_year",
                "quarter",
                "metric_name",
                "source_lineage",
                "source_id",
                "source_quality_tier",
                "measurement_class",
                "series_kind",
                "support_partition",
                "unknown_provenance",
                "target_value",
                "scale",
                "r10_norm_error",
                "carry_forward_norm_error",
            )
        }
        output.update(
            {
                "candidate_family": R42_FAMILY,
                "excluded_source_family": source_family,
                "candidate_value": None if value is None else float(value),
                "candidate_norm_error": None if candidate_error is None else float(candidate_error),
                "candidate_minus_r10_norm_error": None
                if candidate_error is None
                else float(candidate_error - float(record["r10_norm_error"])),
                "candidate_minus_carry_forward_norm_error": None
                if candidate_error is None
                else float(candidate_error - float(record["carry_forward_norm_error"])),
                "prediction_status": "missing_prediction" if candidate_error is None else "scored",
            }
        )
        rows.append(output)
    return rows


def _mean_error(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if _finite_float(row.get(key)) is not None
    ]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _gate_rows(
    *,
    scored_rows: list[dict[str, Any]],
    strict_reference_rows: list[dict[str, Any]],
    source_family: str | None = None,
) -> list[dict[str, Any]]:
    reference = _reference_by_scope_horizon(strict_reference_rows)
    rows: list[dict[str, Any]] = []
    for scope, horizon in R42_REQUIRED_ROUTES:
        scoped = [
            row
            for row in scored_rows
            if int(row.get("horizon_years") or 0) == int(horizon)
            and _record_scope(row, scope)
        ]
        scored = [row for row in scoped if _finite_float(row.get("candidate_norm_error")) is not None]
        ref = reference.get((scope, int(horizon)), {})
        candidate_mean = _mean_error(scored, "candidate_norm_error")
        matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
        carry = _mean_error(scored, "carry_forward_norm_error")
        missing_count = len(scoped) - len(scored)
        blockers: list[str] = []
        if not scoped:
            blockers.append("no_target_rows")
        if missing_count:
            blockers.append("candidate_predictions_missing_for_route")
        if candidate_mean is None or matched_r10 is None:
            blockers.append("candidate_or_strict_r10_missing")
        elif candidate_mean >= matched_r10:
            blockers.append("candidate_not_better_than_strict_mapped_r10")
        if candidate_mean is None or carry is None:
            blockers.append("candidate_or_carry_missing")
        elif candidate_mean >= carry:
            blockers.append("candidate_not_better_than_carry_forward")
        rows.append(
            {
                "candidate_family": R42_FAMILY,
                "excluded_source_family": source_family,
                "scope": scope,
                "horizon_years": int(horizon),
                "target_entry_count": len(scoped),
                "scored_entry_count": len(scored),
                "missing_prediction_count": missing_count,
                "candidate_mean_mae": candidate_mean,
                "strict_matched_r10_mean_mae": matched_r10,
                "candidate_minus_strict_r10": None if candidate_mean is None or matched_r10 is None else float(candidate_mean - matched_r10),
                "carry_forward_mean_mae": carry,
                "candidate_minus_carry_forward": None if candidate_mean is None or carry is None else float(candidate_mean - carry),
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
    return rows


def _metric_anatomy_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in ("all_mapped", "program_mapped"):
        for horizon in (1, 3, 5):
            for metric_name in R10_COMPARABLE_METRICS:
                scoped = [
                    row
                    for row in scored_rows
                    if int(row.get("horizon_years") or 0) == int(horizon)
                    and str(row.get("metric_name") or "") == metric_name
                    and _record_scope(row, scope)
                ]
                if not scoped:
                    continue
                scored = [row for row in scoped if _finite_float(row.get("candidate_norm_error")) is not None]
                candidate_mean = _mean_error(scored, "candidate_norm_error")
                r10_mean = _mean_error(scoped, "r10_norm_error")
                carry_mean = _mean_error(scored, "carry_forward_norm_error")
                output.append(
                    {
                        "candidate_family": R42_FAMILY,
                        "excluded_source_family": scoped[0].get("excluded_source_family"),
                        "scope": scope,
                        "horizon_years": int(horizon),
                        "metric_name": metric_name,
                        "target_entry_count": len(scoped),
                        "scored_entry_count": len(scored),
                        "missing_prediction_count": len(scoped) - len(scored),
                        "candidate_mean_mae": candidate_mean,
                        "strict_matched_r10_mean_mae": r10_mean,
                        "carry_forward_mean_mae": carry_mean,
                        "candidate_minus_strict_r10": None if candidate_mean is None or r10_mean is None else float(candidate_mean - r10_mean),
                    }
                )
    return output


def _source_family_ablation_report(
    *,
    source_rows: list[dict[str, Any]],
    mapped_records: list[dict[str, Any]],
    strict_reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    families = [str(row["source_family"]) for row in _source_family_counts(source_rows)]
    ablation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    baseline_predictions, baseline_origin_rows, baseline_failures = _prediction_rows(
        source_rows=source_rows,
        records=mapped_records,
        excluded_source_family=None,
    )
    baseline_scored = _score_records(
        records=mapped_records,
        prediction_rows=baseline_predictions,
        source_family=None,
    )
    baseline_gate_rows = _gate_rows(
        scored_rows=baseline_scored,
        strict_reference_rows=strict_reference_rows,
        source_family=None,
    )
    baseline_metric_rows = _metric_anatomy_rows(baseline_scored)
    origin_rows.extend(baseline_origin_rows)
    failures.extend(baseline_failures)
    for family in families:
        predictions, family_origin_rows, family_failures = _prediction_rows(
            source_rows=source_rows,
            records=mapped_records,
            excluded_source_family=family,
        )
        scored = _score_records(
            records=mapped_records,
            prediction_rows=predictions,
            source_family=family,
        )
        family_gate = _gate_rows(
            scored_rows=scored,
            strict_reference_rows=strict_reference_rows,
            source_family=family,
        )
        ablation_rows.extend(family_gate)
        metric_rows.extend(_metric_anatomy_rows(scored))
        origin_rows.extend(family_origin_rows)
        failures.extend(family_failures)
    by_family: list[dict[str, Any]] = []
    baseline_lookup = {
        (str(row.get("scope") or ""), int(row.get("horizon_years") or 0)): row
        for row in baseline_gate_rows
    }
    for family in families:
        family_rows = [row for row in ablation_rows if str(row.get("excluded_source_family") or "") == family]
        deltas: list[float] = []
        blockers: list[str] = []
        for row in family_rows:
            key = (str(row.get("scope") or ""), int(row.get("horizon_years") or 0))
            baseline = baseline_lookup.get(key, {})
            candidate = _finite_float(row.get("candidate_mean_mae"))
            baseline_value = _finite_float(baseline.get("candidate_mean_mae"))
            if candidate is not None and baseline_value is not None:
                deltas.append(float(candidate - baseline_value))
            blockers.extend(str(item) for item in list(row.get("blockers") or []))
        passed = bool(family_rows) and all(str(row.get("status") or "") == "pass" for row in family_rows)
        by_family.append(
            {
                "excluded_source_family": family,
                "status": "stable_pass" if passed else "source_dependent_or_failed",
                "route_count": len(family_rows),
                "passing_route_count": sum(1 for row in family_rows if str(row.get("status") or "") == "pass"),
                "mean_candidate_minus_baseline": None if not deltas else float(np.mean(np.asarray(deltas, dtype=np.float64))),
                "max_candidate_minus_baseline": None if not deltas else float(np.max(np.asarray(deltas, dtype=np.float64))),
                "unique_blockers": sorted(set(blockers)),
            }
        )
    stable_count = sum(1 for row in by_family if str(row.get("status") or "") == "stable_pass")
    return {
        "schema_version": "phase3_dynamic.r42_source_family_ablation.v1",
        "generated_at": _generated_at(),
        "candidate_family": R42_FAMILY,
        "contract": (
            "Each source-family ablation removes that family's R10-comparable metric values from each blocked "
            "training window only. The original strict mapped target rows remain fixed, so a pass means R41 is "
            "not dependent on that source family for training under the same R10/carry-forward routes."
        ),
        "source_family_counts": _source_family_counts(source_rows),
        "source_family_count": len(families),
        "stable_source_family_count": int(stable_count),
        "status": "pass" if families and stable_count == len(families) else "diagnostic_source_dependence_present",
        "baseline_gate_rows": baseline_gate_rows,
        "baseline_metric_anatomy_rows": baseline_metric_rows,
        "source_family_rows": by_family,
        "ablation_gate_rows": ablation_rows,
        "ablation_metric_anatomy_rows": metric_rows,
        "origin_rows": origin_rows,
        "origin_failures": failures,
    }


def _geography_claim_report(ledger: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in list(ledger.get("rows") or []) if isinstance(row, dict)]
    geography_counts = Counter(str(row.get("geography") or "unknown") for row in rows)
    direct_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    metric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        geography = str(row.get("geography") or "unknown")
        role = str(row.get("observation_role") or "unknown")
        metric = str(row.get("metric_id") or "unknown")
        role_counts[geography][role] += 1
        metric_counts[geography][metric] += 1
        if role == "direct_target":
            direct_counts[geography][metric] += 1
    nonnational = [geo for geo in geography_counts if geo.lower() not in {"national", "philippines"}]
    rows_out = []
    for geography, count in sorted(geography_counts.items(), key=lambda item: (-item[1], item[0])):
        rows_out.append(
            {
                "geography": geography,
                "row_count": int(count),
                "role_counts": dict(sorted(role_counts[geography].items())),
                "top_metric_counts": dict(metric_counts[geography].most_common(12)),
                "direct_target_metric_counts": dict(sorted(direct_counts[geography].items())),
            }
        )
    status = "national_only_validation" if not nonnational else "subnational_auxiliary_available"
    return {
        "schema_version": "phase3_dynamic.r42_geography_claim_boundary.v1",
        "generated_at": _generated_at(),
        "status": status,
        "n_geographies": len(geography_counts),
        "nonnational_geographies": sorted(nonnational),
        "geography_rows": rows_out,
        "claim_boundary": (
            "The active ObservationRoleLedger contains only national rows, so R41 can be frozen as a national "
            "research champion only. Subnational modeling is allowed next as an evidence-extraction and "
            "partial-pooling design task, not as a validated province/region performance claim."
            if not nonnational
            else "Subnational rows exist, but each geography still needs direct-target metric support and blocked gates before any province or region performance claim."
        ),
    }


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in {"blockers", "unique_blockers", "metric_counts", "role_counts", "top_metric_counts", "direct_target_metric_counts"}})
    for optional in ("blockers", "unique_blockers", "metric_counts", "role_counts", "top_metric_counts", "direct_target_metric_counts"):
        if any(optional in row for row in rows):
            fieldnames.append(optional)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in ("blockers", "unique_blockers"):
                if key in output:
                    output[key] = ";".join(str(item) for item in list(output.get(key) or []))
            for key in ("metric_counts", "role_counts", "top_metric_counts", "direct_target_metric_counts"):
                if key in output:
                    output[key] = str(dict(output.get(key) or {}))
            writer.writerow(output)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    ablation = dict(report.get("source_family_ablation") or {})
    geography = dict(report.get("geography_claim_boundary") or {})
    annual = dict(report.get("external_annual_gate") or {})
    lines = [
        "# Phase 3 R42 R41 Champion Hardening",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Champion Lock",
        "",
        f"- Candidate family: `{report.get('candidate_family')}`",
        f"- Strict gate status: `{report.get('strict_gate_status')}`",
        f"- Annual external gate status: `{annual.get('status')}`",
        f"- Source-family ablation status: `{ablation.get('status')}`",
        f"- Geography status: `{geography.get('status')}`",
        "",
        "## Strict R41 Gate",
        "",
        "| Scope | Horizon | Entries | Candidate | Strict R10 | Delta R10 | Carry | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(report.get("strict_gate_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | {row.get('target_entry_count')} | "
            f"{_format_float(row.get('candidate_mean_mae'))} | {_format_float(row.get('strict_matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('candidate_minus_strict_r10'))} | {_format_float(row.get('carry_forward_mean_mae'))} | "
            f"`{row.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Source-Family Ablation",
            "",
            "| Excluded source family | Status | Passing routes | Max delta vs baseline | Blockers |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in list(ablation.get("source_family_rows") or []):
        lines.append(
            f"| `{row.get('excluded_source_family')}` | `{row.get('status')}` | "
            f"{row.get('passing_route_count')}/{row.get('route_count')} | "
            f"{_format_float(row.get('max_candidate_minus_baseline'))} | "
            f"`{';'.join(str(item) for item in list(row.get('unique_blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Geography Boundary",
            "",
            str(geography.get("claim_boundary") or ""),
            "",
            "## Contract",
            "",
            "- R42 is not another model search. It freezes R41 and tests robustness.",
            "- Source-family ablations remove evidence from training windows only; target rows stay fixed.",
            "- Annual incidence, AIDS deaths, and estimated PLHIV remain validation-only weak-measurement heads.",
            "- Subnational claims remain blocked unless the ledger supplies geography-specific direct targets and blocked gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    strict_rows = list(report.get("strict_gate_rows") or [])
    ablation_rows = list(dict(report.get("source_family_ablation") or {}).get("source_family_rows") or [])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("R42 R41 champion hardening", fontsize=15, fontweight="bold")
    strict_labels = [f"{row.get('scope')} h{row.get('horizon_years')}" for row in strict_rows]
    strict_deltas = np.asarray([float(row.get("candidate_minus_strict_r10") or np.nan) for row in strict_rows], dtype=np.float64)
    x = np.arange(len(strict_rows), dtype=np.float64)
    axes[0].bar(x, strict_deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in strict_deltas])
    axes[0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strict_labels, rotation=25, ha="right")
    axes[0].set_ylabel("candidate minus strict R10")
    axes[0].set_title("Frozen R41 strict mapped gate")
    labels = [str(row.get("excluded_source_family") or "") for row in ablation_rows]
    values = np.asarray(
        [
            np.nan if _finite_float(row.get("max_candidate_minus_baseline")) is None else float(row["max_candidate_minus_baseline"])
            for row in ablation_rows
        ],
        dtype=np.float64,
    )
    y = np.arange(len(labels), dtype=np.float64)
    axes[1].barh(y, values, color=["#8a6f2a" if np.isfinite(value) and value <= 0.0 else "#b56a3a" for value in values])
    axes[1].axvline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("max ablated candidate minus baseline R41")
    axes[1].set_title("Train-data source-family ablation sensitivity")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r42_r41_champion_hardening(
    *,
    run_id: str = R42_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    source_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    ledger = build_observation_role_ledger(
        root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    r29_path = phase3_root / "artifacts" / "runs" / R29_RUN_ID / "analysis" / "r29_strict_ledger_matched_r10_gate.json"
    if r29_path.exists():
        r29_report = read_json(r29_path, default={})
    else:
        r29_report = run_r29_strict_ledger_matched_r10_gate(epigraph_root=root)
    strict_rows = list(dict(r29_report).get("strict_reference_rows") or [])
    records, reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    mapped_records = [record for record in records if not bool(record.get("unknown_provenance"))]
    if not strict_rows:
        strict_rows = _strict_reference_rows(mapped_records, horizons)
    predictions, origin_rows, origin_failures = _prediction_rows(
        source_rows=source_rows,
        records=mapped_records,
        excluded_source_family=None,
    )
    scored_rows = _score_records(records=mapped_records, prediction_rows=predictions)
    strict_gate_rows = _gate_rows(scored_rows=scored_rows, strict_reference_rows=strict_rows)
    strict_gate_status = "pass" if strict_gate_rows and all(str(row.get("status") or "") == "pass" for row in strict_gate_rows) else "fail"
    metric_rows = _metric_anatomy_rows(scored_rows)
    source_ablation = _source_family_ablation_report(
        source_rows=source_rows,
        mapped_records=mapped_records,
        strict_reference_rows=strict_rows,
    )
    annual_gate = _build_r12_official_annual_challenge_gate_report(
        rows=validation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        candidate_families=(R42_FAMILY,),
        horizons=horizons,
    )
    geography_report = _geography_claim_report(ledger)
    blockers: list[str] = []
    if strict_gate_status != "pass":
        blockers.append("strict_mapped_r10_gate_failed")
    if str(annual_gate.get("status") or "") != "pass":
        blockers.append("external_annual_gate_failed_or_blocked")
    promotion_claim = (
        "freeze_as_national_research_champion"
        if not blockers
        else "do_not_freeze_until_blockers_resolved"
    )
    verdict = (
        "R42 freezes R41 as the current national research champion. Source-family ablation and geography reports define the remaining publication-hardening work."
        if not blockers
        else "R42 does not freeze R41 because at least one required national hardening gate failed."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    annual_path = analysis_dir / "r42_external_annual_challenge_gate.json"
    annual_dashboard = analysis_dir / "r42_external_annual_challenge_dashboard.png"
    write_json(annual_path, annual_gate)
    _write_r12_official_annual_challenge_dashboard(annual_dashboard, annual_gate)
    report = {
        "schema_version": R42_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "candidate_family": R42_FAMILY,
        "promotion_claim": promotion_claim,
        "strict_gate_status": strict_gate_status,
        "blockers": blockers,
        "verdict": verdict,
        "contract": (
            "R42 is the champion hardening layer: it freezes R41, reruns strict mapped R10/carry-forward routes, "
            "runs training-only source-family ablations, scores external annual validation-only heads, and separates national "
            "from subnational claim eligibility."
        ),
        "reference_rows": reference_rows,
        "strict_reference_rows": strict_rows,
        "strict_gate_rows": strict_gate_rows,
        "metric_anatomy_rows": metric_rows,
        "strict_origin_rows": origin_rows,
        "strict_origin_failures": origin_failures,
        "source_family_ablation": source_ablation,
        "external_annual_gate": annual_gate,
        "geography_claim_boundary": geography_report,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r42_r41_champion_hardening_report.json"
    md_path = analysis_dir / "r42_r41_champion_hardening_report.md"
    strict_csv = analysis_dir / "r42_strict_gate_rows.csv"
    ablation_csv = analysis_dir / "r42_source_family_rows.csv"
    ablation_gate_csv = analysis_dir / "r42_source_family_ablation_gate_rows.csv"
    geography_csv = analysis_dir / "r42_geography_claim_rows.csv"
    dashboard_path = analysis_dir / "r42_r41_champion_hardening_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "strict_gate_csv": strict_csv.as_posix(),
        "source_family_csv": ablation_csv.as_posix(),
        "source_family_ablation_gate_csv": ablation_gate_csv.as_posix(),
        "geography_claim_csv": geography_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
        "external_annual_gate_json": annual_path.as_posix(),
        "external_annual_dashboard_png": annual_dashboard.as_posix(),
    }
    prior_artifacts = {
        "r41_strict_gate": phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r41-monotone-growth-strict-gate-20260502-s00"
        / "analysis"
        / "r36_strict_component_policy_gate.json",
        "r41_annual_gate": phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r41-monotone-growth-annual-gate-20260502-s00"
        / "analysis"
        / "r41_monotone_growth_annual_gate.json",
    }
    report["prior_champion_artifacts"] = {
        name: {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "available": path.exists(),
        }
        for name, path in prior_artifacts.items()
    }
    write_json(json_path, report)
    _write_csv(strict_csv, strict_gate_rows)
    _write_csv(ablation_csv, list(source_ablation.get("source_family_rows") or []))
    _write_csv(ablation_gate_csv, list(source_ablation.get("ablation_gate_rows") or []))
    _write_csv(geography_csv, list(geography_report.get("geography_rows") or []))
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R42 R41 champion hardening.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R42_RUN_ID)
    args = parser.parse_args()
    run_r42_r41_champion_hardening(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
