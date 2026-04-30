from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    MISSING_DATA_LADDER,
    build_observation_rows,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
)
from .metrics import quarter_ordinal, quarter_sort_key, quarter_year
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .runtime import ensure_dir, read_json, write_json


R11_FIRST_BATCH_SCHEMA_VERSION = "phase3_dynamic.r11_first_batch.v1"
R12_REFERENCE_BRANCH_SCHEMA_VERSION = "phase3_dynamic.r12_reference_branch.v1"
STOCK_CONSISTENCY_GATE_SCHEMA_VERSION = "phase3_dynamic.r11_stock_consistency_gate.v1"
CASCADE_STOCK_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
PRIMARY_STOCK_GUARD_METRICS: tuple[str, ...] = ("diagnosed_plhiv", "alive_on_art")
R11_EVALUATION_METRICS: tuple[str, ...] = CASCADE_STOCK_METRICS + ("new_diagnosed_cases_period",)
R10_COMPARABLE_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)
ANNUAL_VALIDATION_METRICS: tuple[str, ...] = ("annual_new_infections", "annual_aids_deaths")
BACK_HALF_RATE_SPECS: tuple[dict[str, str], ...] = (
    {
        "rate_id": "vl_tested_among_art",
        "numerator_metric": "tested_for_viral_load",
        "denominator_metric": "alive_on_art",
    },
    {
        "rate_id": "suppressed_among_vl_tested",
        "numerator_metric": "virally_suppressed",
        "denominator_metric": "tested_for_viral_load",
    },
)
R11_MULTI_HORIZON_YEARS: tuple[int, ...] = (1, 3, 5)
CONSTRAINED_SHAPE_DIRECT_METRICS: tuple[str, ...] = (
    "alive_on_art",
    "new_diagnosed_cases_period",
)
R12_LONG_HORIZON_STOCK_METRICS: tuple[str, ...] = ("diagnosed_plhiv", "alive_on_art")
R12_LINEAGE_DIAGNOSTIC_HORIZONS: tuple[int, ...] = (3, 5)
R12_LINEAGE_AXES: tuple[str, ...] = (
    "source_family",
    "support_partition",
    "source_support",
    "aggregation_mode",
    "source_id",
)
R12_STRATIFIED_OPERATOR_LINEAGES: tuple[dict[str, str], ...] = (
    {
        "lineage_id": "doh_quarterly",
        "lineage_label": "DOH HARP quarterly",
        "source_family": "official_doh_archive|program_observed_harp|quarterly_snapshot",
    },
    {
        "lineage_id": "doh_monthly",
        "lineage_label": "DOH HARP monthly",
        "source_family": "official_doh_archive|program_observed_harp|monthly_snapshot",
    },
    {
        "lineage_id": "slide_annual_anchor",
        "lineage_label": "slide annual anchor",
        "source_family": "official_user_provided_slide|program_observed_harp|annual_snapshot",
    },
)
R12_SUPPORT_ADEQUACY_HORIZONS: tuple[int, ...] = (1, 2)
R12_SUPPORT_ADEQUACY_MIN_TRAIN_YEARS = 1
R12_HORIZON_EVIDENCE_ROUTES: tuple[dict[str, Any], ...] = (
    {
        "route_id": "program_nowcast",
        "route_label": "DOH program nowcast evidence",
        "claim_role": "short_horizon_nowcast",
        "horizons": (1, 2),
        "lineage_ids": ("doh_quarterly", "doh_monthly"),
        "min_train_contract": "diagnostic_short_horizon",
    },
    {
        "route_id": "annual_trajectory_anchor",
        "route_label": "slide annual trajectory anchor",
        "claim_role": "long_horizon_trajectory",
        "horizons": (3, 5),
        "lineage_ids": ("slide_annual_anchor",),
        "min_train_contract": "production",
    },
)
R12_ANNUAL_ANCHOR_LINEAGE_ID = "slide_annual_anchor"
R12_PROGRAM_LINEAGE_IDS: tuple[str, ...] = ("doh_quarterly", "doh_monthly")
R12_PROGRAM_MUTATION_METRICS: tuple[str, ...] = R10_COMPARABLE_METRICS
R12_PROGRAM_NOWCAST_CANDIDATE_FAMILIES: tuple[str, ...] = (
    "multi_horizon_weighted_process",
    "r10_style_readout_teacher",
    "support_era_diagnosis_flow_process",
    "stock_flow_reconciliation_process",
    "diagnosis_lag_stock_process",
    "diagnosed_reporting_bias_process",
    "r12_da_process_split_transition",
)
R12_PROGRAM_MIXED_QUARTERLY_ROUTES: tuple[dict[str, Any], ...] = (
    {
        "route_id": "program_nowcast",
        "route_label": "DOH program nowcast evidence",
        "claim_role": "short_horizon_nowcast",
        "horizons": (1, 2),
        "lineage_ids": R12_PROGRAM_LINEAGE_IDS,
        "min_train_contract": "diagnostic_short_horizon",
        "r10_required": True,
    },
    {
        "route_id": "program_mixed_quarterly_trajectory",
        "route_label": "DOH mixed quarterly diagnosis/ART trajectory",
        "claim_role": "long_horizon_trajectory",
        "horizons": (3, 5),
        "lineage_ids": R12_PROGRAM_LINEAGE_IDS,
        "min_train_contract": "production",
        "r10_required": True,
    },
)
OFFICIAL_ANNUAL_CHALLENGE_METRICS: tuple[str, ...] = (
    "annual_new_infections",
    "annual_aids_deaths",
    "estimated_plhiv",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS: tuple[str, ...] = (
    "annual_new_infections",
    "annual_aids_deaths",
    "estimated_plhiv",
)
FLOAT_NONREGRESSION_TOLERANCE = float(np.finfo(np.float64).eps)
HORIZON_ADAPTIVE_SHAPE_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "policy_id": "identity_r11_14",
        "corrected_metrics": (),
        "description": "no residual-shape correction beyond the R11-14 conditional-rate process",
    },
    {
        "policy_id": "diagnosis_flow_only",
        "corrected_metrics": ("new_diagnosed_cases_period",),
        "description": "correct diagnosis-flow trajectory shape only",
    },
    {
        "policy_id": "art_only",
        "corrected_metrics": ("alive_on_art",),
        "description": "correct ART-stock trajectory shape only, then regenerate VL and suppression from conditional rates",
    },
    {
        "policy_id": "art_plus_diagnosis_flow",
        "corrected_metrics": CONSTRAINED_SHAPE_DIRECT_METRICS,
        "description": "R11-17 policy: correct ART stock and diagnosis flow only",
    },
    {
        "policy_id": "diagnosed_stock_only",
        "corrected_metrics": ("diagnosed_plhiv",),
        "description": "correct diagnosed-stock trajectory shape only while preserving the cascade cone",
    },
    {
        "policy_id": "r10_scope_stocks",
        "corrected_metrics": ("diagnosed_plhiv", "alive_on_art"),
        "description": "correct the R10-comparable stock pair only, then regenerate back-half conditional rates",
    },
    {
        "policy_id": "r10_scope_all",
        "corrected_metrics": R10_COMPARABLE_METRICS,
        "description": "correct only the R10-comparable scope; no VL or suppression endpoint correction is allowed",
    },
)
TRANSITION_PROCESS_ART_LAGS: tuple[int, ...] = (0, 1, 2, 4)


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _latest_existing(pattern: str, *, root: Path) -> Path | None:
    candidates = sorted(root.glob(pattern))
    return candidates[-1] if candidates else None


def _artifact_paths(phase3_root: Path) -> dict[str, str | None]:
    runs_root = phase3_root / "artifacts" / "runs"
    audits_root = phase3_root / "artifacts" / "scientific_audits"
    paths: dict[str, Path | None] = {
        "annual_incidence_claim_card": audits_root / "phase3_interval_r10_mechanistic_incidence_claim_card_20260429.json",
        "incidence_full_gate": _latest_existing(
            "p3d-incidence-readout-full-gate-*/analysis/incidence_readout_full_gate_report.json",
            root=runs_root,
        ),
        "u_to_d_coupling": _latest_existing(
            "p3d-u-to-d-coupling-*/analysis/u_to_d_coupling_report.json",
            root=runs_root,
        ),
        "frontdoor_coupling": _latest_existing(
            "p3d-frontdoor-coupling-*/analysis/frontdoor_coupling_report.json",
            root=runs_root,
        ),
        "phase2_determinant_robustness": (
            default_epigraph_root()
            / "artifacts"
            / "runs"
            / "phase0-2-incidence-official-augmented-20260429-s01"
            / "phase2"
            / "determinant_robustness_broad"
            / "phase2_determinant_robustness_report.json"
        ),
        "r11_plan": audits_root / "phase3_r11_sparse_state_space_council_plan_20260429.md",
        "r11_manifest": audits_root / "phase3_r11_sparse_state_space_experiment_manifest_20260429.json",
    }
    return {name: None if path is None else path.as_posix() for name, path in paths.items()}


def _read_path_payload(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        payload = read_json(path, default=None)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def project_cascade_stock_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a sparse cascade row onto diagnosed >= ART >= VL-tested >= suppressed >= 0."""
    projected: dict[str, float | None] = {}
    previous: float | None = None
    changed_metrics: list[str] = []
    for metric_name in CASCADE_STOCK_METRICS:
        value = _finite_float(row.get(metric_name))
        if value is None:
            projected[metric_name] = None
            continue
        bounded = max(value, 0.0)
        if previous is not None:
            bounded = min(bounded, previous)
        projected[metric_name] = float(bounded)
        if not np.isclose(float(bounded), float(value), rtol=0.0, atol=0.0):
            changed_metrics.append(metric_name)
        previous = float(bounded)
    return {
        "quarter": str(row.get("quarter") or ""),
        "projected": projected,
        "changed_metrics": changed_metrics,
        "changed": bool(changed_metrics),
        "contract": "deterministic cascade-cone projection; no fitted or hand-tuned parameters",
    }


def _cascade_cone_violations(row: dict[str, Any]) -> list[str]:
    values = {metric: _finite_float(row.get(metric)) for metric in CASCADE_STOCK_METRICS}
    violations: list[str] = []
    previous_metric: str | None = None
    previous_value: float | None = None
    for metric_name in CASCADE_STOCK_METRICS:
        value = values.get(metric_name)
        if value is None:
            continue
        if value < 0.0:
            violations.append(f"{metric_name}_negative")
        if previous_value is not None and value > previous_value:
            violations.append(f"{metric_name}_exceeds_{previous_metric}")
        previous_metric = metric_name
        previous_value = value
    return violations


def stock_consistency_gate(
    report: dict[str, Any] | None,
    *,
    guard_metrics: tuple[str, ...] = PRIMARY_STOCK_GUARD_METRICS,
) -> dict[str, Any]:
    if not report:
        return {
            "schema_version": STOCK_CONSISTENCY_GATE_SCHEMA_VERSION,
            "status": "not_evaluable",
            "blockers": ["missing_report"],
            "contract": "strict signed non-regression against carry-forward on guarded stock metrics",
        }
    anatomy = list(report.get("metric_anatomy") or [])
    anatomy_by_metric = {
        str(row.get("metric_name") or ""): dict(row)
        for row in anatomy
        if isinstance(row, dict)
    }
    metric_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for metric_name in guard_metrics:
        row = anatomy_by_metric.get(metric_name)
        if row is None:
            metric_rows.append(
                {
                    "metric_name": metric_name,
                    "status": "not_evaluable",
                    "reason": "missing_metric_anatomy",
                }
            )
            blockers.append(f"{metric_name}_missing_metric_anatomy")
            continue
        mean_delta = _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error"))
        worst_delta = _finite_float(row.get("worst_candidate_minus_carry_forward_norm_error"))
        metric_blockers: list[str] = []
        if mean_delta is None:
            metric_blockers.append("missing_mean_delta")
        elif mean_delta > FLOAT_NONREGRESSION_TOLERANCE:
            metric_blockers.append("mean_worse_than_carry_forward")
        if worst_delta is None:
            metric_blockers.append("missing_worst_delta")
        elif worst_delta > FLOAT_NONREGRESSION_TOLERANCE:
            metric_blockers.append("worst_case_worse_than_carry_forward")
        if metric_blockers:
            blockers.extend([f"{metric_name}_{blocker}" for blocker in metric_blockers])
        metric_rows.append(
            {
                "metric_name": metric_name,
                "status": "pass" if not metric_blockers else "fail",
                "candidate_mean_norm_error": _finite_float(row.get("candidate_mean_norm_error")),
                "carry_forward_mean_norm_error": _finite_float(row.get("carry_forward_mean_norm_error")),
                "candidate_minus_carry_forward_mean_norm_error": mean_delta,
                "worst_candidate_minus_carry_forward_norm_error": worst_delta,
                "blockers": metric_blockers,
            }
        )
    return {
        "schema_version": STOCK_CONSISTENCY_GATE_SCHEMA_VERSION,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "metric_rows": metric_rows,
        "guard_metrics": list(guard_metrics),
        "contract": "strict signed non-regression against carry-forward on guarded stock metrics; zero hand-tuned tolerance",
    }


def _gate_status(report: dict[str, Any] | None, *keys: str) -> str:
    if not report:
        return "not_evaluable"
    for key in keys:
        gate = report.get(key)
        if isinstance(gate, dict) and gate.get("status"):
            return str(gate.get("status"))
    return "not_evaluable"


def _gate_score(report: dict[str, Any] | None, key: str, field: str) -> float | None:
    gate = report.get(key) if report else None
    if not isinstance(gate, dict):
        return None
    return _finite_float(gate.get(field))


def _r10_reference(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    r10 = report.get("r10_reference")
    if isinstance(r10, dict):
        return _finite_float(r10.get("reference_quarterly_mean_mae"))
    gate = report.get("shock_aware_lifted_trajectory_gate")
    if isinstance(gate, dict):
        nested = gate.get("r10_reference")
        if isinstance(nested, dict):
            return _finite_float(nested.get("reference_quarterly_mean_mae"))
    return None


def _r10_horizon_replay_paths(root: Path, horizons: tuple[int, ...]) -> dict[int, Path | None]:
    runs_root = Path(root) / "artifacts" / "runs"
    paths: dict[int, Path | None] = {}
    for horizon in horizons:
        paths[int(horizon)] = _latest_existing(
            f"tr-v3-r10-horizon-replay-dense-h{int(horizon)}-*/analysis/tr_v3_experiment_suite_report.json",
            root=runs_root,
        )
    return paths


def _select_horizon_matched_r10_reference(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "available": False,
            "reason": "missing_r10_horizon_replay_report",
            "candidate_rows": [],
        }
    candidate_rows: list[dict[str, Any]] = []
    for result in list(report.get("results") or []):
        if not isinstance(result, dict):
            continue
        experiment_id = str(result.get("experiment_id") or "")
        if not experiment_id.startswith("EXP-R10"):
            continue
        summary = dict(result.get("quarterly_summary") or {})
        candidate_mean = _finite_float(summary.get("candidate_mean_mae"))
        if candidate_mean is None:
            continue
        candidate_rows.append(
            {
                "experiment_id": experiment_id,
                "family": str(result.get("family") or ""),
                "candidate_mean_mae": candidate_mean,
                "candidate_worst_mae": _finite_float(summary.get("candidate_worst_mae")),
                "carry_forward_mean_mae": _finite_float(summary.get("carry_forward_mean_mae")),
                "quarterly_row_count": len(list(result.get("quarterly_rows") or [])),
                "decision": str(result.get("decision") or ""),
            }
        )
    if not candidate_rows:
        return {
            "available": False,
            "reason": "no_executable_r10_family_rows",
            "candidate_rows": [],
        }
    reference = min(candidate_rows, key=lambda row: float(row["candidate_mean_mae"]))
    return {
        "available": True,
        "reference_experiment_id": reference["experiment_id"],
        "reference_quarterly_mean_mae": float(reference["candidate_mean_mae"]),
        "reference_quarterly_worst_mae": reference.get("candidate_worst_mae"),
        "reference_carry_forward_mean_mae": reference.get("carry_forward_mean_mae"),
        "reference_selection_policy": "best_executed_EXP_R10_family_candidate_within_same_horizon_dense_contract",
        "reference_metric_scope": list(R10_COMPARABLE_METRICS),
        "candidate_rows": candidate_rows,
    }


def _build_r10_horizon_replay_report(
    *,
    root: Path,
    horizons: tuple[int, ...],
    start_year: int,
    end_year: int,
    min_train_years: int,
) -> dict[str, Any]:
    paths = _r10_horizon_replay_paths(root, horizons)
    horizon_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        path = paths.get(int(horizon))
        payload = read_json(path, default={}) if path else {}
        replay_window = dict(payload.get("quarterly_window") or {}) if isinstance(payload, dict) else {}
        reference = _select_horizon_matched_r10_reference(payload if isinstance(payload, dict) else {})
        blockers: list[str] = []
        if path is None:
            blockers.append("missing_horizon_replay_artifact")
        if not bool(reference.get("available")):
            blockers.append(str(reference.get("reason") or "r10_reference_not_available"))
        if replay_window:
            if int(replay_window.get("horizon_years") or -1) != int(horizon):
                blockers.append("horizon_mismatch")
            if int(replay_window.get("start_year") or -1) != int(start_year):
                blockers.append("start_year_mismatch")
            if int(replay_window.get("end_year") or -1) != int(end_year):
                blockers.append("end_year_mismatch")
            if int(replay_window.get("min_train_years") or -1) != int(min_train_years):
                blockers.append("min_train_years_mismatch")
        elif path is not None:
            blockers.append("missing_replay_window_contract")
        horizon_rows.append(
            {
                "horizon_years": int(horizon),
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
                "artifact_path": None if path is None else path.as_posix(),
                "artifact_sha256": None if path is None else _sha256(path),
                "quarterly_window": replay_window,
                **reference,
            }
        )
    return {
        "schema_version": "phase3_dynamic.r11_r10_horizon_matched_replay.v1",
        "generated_at": _generated_at(),
        "horizons": list(horizons),
        "horizon_rows": horizon_rows,
        "status": "pass" if all(str(row.get("status") or "") == "pass" for row in horizon_rows) else "fail",
        "contract": (
            "Horizon-matched R10 replay uses frozen legacy R10-family experiment-suite artifacts generated "
            "with the same blocked horizon, start/end years, min-train-years, and dense expanded-HARP contract. "
            "The comparison is restricted to the legacy R10-comparable metric scope."
        ),
    }


def _r10_reference_for_horizon(r10_horizon_replay: dict[str, Any] | None, horizon: int) -> dict[str, Any]:
    if not r10_horizon_replay:
        return {}
    for row in list(r10_horizon_replay.get("horizon_rows") or []):
        if isinstance(row, dict) and int(row.get("horizon_years") or 0) == int(horizon):
            return dict(row)
    return {}


def _summarize_existing_branch(
    *,
    experiment_id: str,
    title: str,
    path_text: str | None,
    report: dict[str, Any] | None,
    claim_scope: str,
) -> dict[str, Any]:
    stock_gate = stock_consistency_gate(report)
    promotion_gate = report.get("promotion_gate") if report else {}
    promotion_eligible = bool(isinstance(promotion_gate, dict) and promotion_gate.get("promotion_eligible"))
    annual_gate_status = _gate_status(report, "annual_incidence_measurement_gate")
    one_year_status = _gate_status(report, "one_year_gate", "one_year_full_path_gate")
    lifted_status = _gate_status(report, "shock_aware_lifted_trajectory_gate")
    decision = "reject"
    kept_claim = "none"
    if annual_gate_status == "pass" and "incidence" in experiment_id.lower():
        decision = "keep_limited"
        kept_claim = "annual_incidence_measurement_readout_only"
    elif promotion_eligible and stock_gate["status"] == "pass":
        decision = "keep"
        kept_claim = claim_scope
    elif one_year_status == "pass" and stock_gate["status"] == "fail":
        decision = "reject_keep_as_falsification"
        kept_claim = "falsifies_flow_only_promotion_without_stock_gate"
    elif report is None:
        decision = "not_run"
        kept_claim = "none"
    blockers: list[str] = []
    if isinstance(promotion_gate, dict):
        blockers.extend([str(item) for item in list(promotion_gate.get("blockers") or [])])
    blockers.extend([str(item) for item in list(stock_gate.get("blockers") or [])])
    return {
        "experiment_id": experiment_id,
        "title": title,
        "family": None if not report else str(report.get("family") or report.get("candidate_family") or ""),
        "artifact_path": path_text,
        "artifact_sha256": None if not path_text else _sha256(Path(path_text)),
        "one_year_status": one_year_status,
        "annual_status": annual_gate_status,
        "lifted_status": lifted_status,
        "stock_consistency_status": str(stock_gate.get("status")),
        "candidate_mae": _gate_score(report, "one_year_gate", "candidate_mean_mae")
        or _gate_score(report, "one_year_full_path_gate", "candidate_mean_mae")
        or _gate_score(report, "annual_incidence_measurement_gate", "candidate_norm_mae"),
        "carry_forward_mae": _gate_score(report, "one_year_gate", "carry_forward_mean_mae")
        or _gate_score(report, "one_year_full_path_gate", "carry_forward_mean_mae")
        or _gate_score(report, "annual_incidence_measurement_gate", "carry_forward_norm_mae"),
        "r10_reference_mae": _r10_reference(report)
        or _gate_score(report, "annual_incidence_measurement_gate", "r10_annual_incidence_error"),
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": blockers,
        "stock_gate": stock_gate,
        "contract": "frozen existing branch report replayed under R11 benchmark and stock-consistency gate",
    }


def _observation_role_counts(ledger: dict[str, Any]) -> dict[str, int]:
    rows = list(ledger.get("rows") or [])
    counts = Counter(str(row.get("observation_role") or "unknown") for row in rows if isinstance(row, dict))
    return dict(sorted(counts.items()))


def _support_partition_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        provenance = dict(row.get("metric_provenance") or {})
        for metric_payload in provenance.values():
            if isinstance(metric_payload, dict):
                counts[str(metric_payload.get("support_partition") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _build_benchmark_manifest(
    *,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    artifact_paths: dict[str, str | None],
    reports: dict[str, dict[str, Any] | None],
    r10_horizon_replay: dict[str, Any] | None,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    return {
        "schema_version": "phase3_dynamic.r11_benchmark_manifest.v1",
        "run_id": run_id,
        "generated_at": _generated_at(),
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "split_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": int(horizon_years),
            "contract": "blocked-origin time splits; no validation-only leakage",
        },
        "comparators": {
            "carry_forward": {
                "role": "hard_baseline",
                "scores_obtained_from": "frozen branch reports and current lineage replay",
            },
            "r10": {
                "role": "quarterly_readout_reference",
                "reference_quarterly_mean_mae": _r10_reference(reports.get("incidence_full_gate"))
                or _r10_reference(reports.get("u_to_d_coupling"))
                or _r10_reference(reports.get("frontdoor_coupling")),
                "horizon_matched_replay_status": None if r10_horizon_replay is None else r10_horizon_replay.get("status"),
                "horizon_matched_reference_rows": []
                if r10_horizon_replay is None
                else [
                    {
                        "horizon_years": row.get("horizon_years"),
                        "reference_experiment_id": row.get("reference_experiment_id"),
                        "reference_quarterly_mean_mae": row.get("reference_quarterly_mean_mae"),
                        "status": row.get("status"),
                    }
                    for row in list(r10_horizon_replay.get("horizon_rows") or [])
                    if isinstance(row, dict)
                ],
            },
            "aem_spectrum_like": {
                "role": "annual_validation_only_external_comparator",
                "training_use": "forbidden",
            },
        },
        "artifact_lock": {
            name: {
                "path": path_text,
                "sha256": None if not path_text else _sha256(Path(path_text)),
                "exists": bool(path_text and Path(path_text).exists()),
            }
            for name, path_text in artifact_paths.items()
        },
    }


def _build_split_manifest(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    splits = rolling_origin_splits(
        rows,
        start_year=int(start_year),
        end_year=int(end_year),
        min_train_years=int(min_train_years),
        horizon_years=int(horizon_years),
    )
    years = sorted({int(str(row.get("quarter") or "0-Q1").split("-Q", 1)[0]) for row in rows if row.get("quarter")})
    return {
        "schema_version": "phase3_dynamic.r11_split_manifest.v1",
        "generated_at": _generated_at(),
        "year_min": years[0] if years else None,
        "year_max": years[-1] if years else None,
        "observation_row_count": len(rows),
        "split_count": len(splits),
        "splits": splits,
        "contract": "materialized from active ObservationRoleLedger-compatible source rows",
    }


def _build_aem_spectrum_validation_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        quarter = str(row.get("quarter") or "")
        if not quarter.endswith("-Q4"):
            continue
        provenance = dict(row.get("metric_provenance") or {})
        for metric_name in ANNUAL_VALIDATION_METRICS:
            value = _finite_float(row.get(metric_name))
            if value is None:
                continue
            metric_provenance = dict(provenance.get(metric_name) or {})
            entries.append(
                {
                    "year": int(quarter.split("-Q", 1)[0]),
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "value": value,
                    "observation_role": str(metric_provenance.get("observation_role") or "validation_only"),
                    "allowed_use": str(metric_provenance.get("allowed_use") or "validation_only"),
                    "support_partition": str(metric_provenance.get("support_partition") or "unknown"),
                    "source_id": str(metric_provenance.get("source_id") or ""),
                    "training_use": "forbidden",
                }
            )
    leakage = [
        entry
        for entry in entries
        if str(entry.get("observation_role")) != "validation_only"
        or str(entry.get("allowed_use")) not in {"validation_only", "held_out_validation_or_diagnostic_only"}
    ]
    return {
        "schema_version": "phase3_dynamic.r11_aem_spectrum_validation_panel.v1",
        "generated_at": _generated_at(),
        "entry_count": len(entries),
        "entries": entries,
        "leakage_violation_count": len(leakage),
        "contract": "annual incumbent-like targets are validation-only and cannot train quarterly states",
    }


def _strip_official_annual_validation_targets(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        output[metric_name] = None
    return output


def _annual_challenge_metric_scale(train_rows: list[dict[str, Any]], metric_name: str) -> float:
    values = [
        abs(float(value))
        for value in (_finite_float(row.get(metric_name)) for row in train_rows)
        if value is not None
    ]
    return max(values) if values else float(np.finfo(np.float32).eps)


def _annual_challenge_carry_forward_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows = _carry_forward_prediction(train_rows, holdout_rows)
    base_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_rows}
    sorted_train = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    output: list[dict[str, Any]] = []
    for holdout_row in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout_row.get("quarter") or "")
        row = dict(base_by_quarter.get(quarter, {"quarter": quarter}))
        for metric_name in OFFICIAL_ANNUAL_CHALLENGE_METRICS:
            if _finite_float(row.get(metric_name)) is not None:
                continue
            eligible = [train_row for train_row in sorted_train if _finite_float(train_row.get(metric_name)) is not None]
            row[metric_name] = None if not eligible else float(eligible[-1].get(metric_name) or 0.0)
        output.append(row)
    return output


def _official_annual_challenge_metric_rows(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    split_count_by_horizon: Counter[int] = Counter()
    prediction_missing_counts: Counter[str] = Counter()
    leakage_rows: list[dict[str, Any]] = []
    q4_rows = [
        dict(row)
        for row in rows
        if str(row.get("quarter") or "").endswith("-Q4")
    ]
    for horizon in horizons:
        splits = rolling_origin_splits(
            q4_rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon),
        )
        for split in splits:
            holdout_years = [int(year) for year in list(split.get("holdout_years") or [])]
            if not holdout_years:
                continue
            train_end_year = int(split.get("train_end_year") or min(holdout_years) - 1)
            raw_train_rows = [
                dict(row)
                for row in q4_rows
                if quarter_year(str(row.get("quarter") or "")) <= train_end_year
            ]
            holdout_rows = [
                dict(row)
                for row in q4_rows
                if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
            ]
            if not raw_train_rows or not holdout_rows:
                continue
            train_rows = [_strip_official_annual_validation_targets(row) for row in raw_train_rows]
            split_count_by_horizon[int(horizon)] += 1
            candidate_predictions, _summary = _candidate_predictions(
                train_rows,
                holdout_rows,
                family=family,
            )
            carry_predictions = _annual_challenge_carry_forward_rows(raw_train_rows, holdout_rows)
            candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_predictions}
            carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_predictions}
            for holdout_row in holdout_rows:
                quarter = str(holdout_row.get("quarter") or "")
                candidate_row = candidate_by_quarter.get(quarter, {})
                carry_row = carry_by_quarter.get(quarter, {})
                provenance = dict(holdout_row.get("metric_provenance") or {})
                for metric_name in OFFICIAL_ANNUAL_CHALLENGE_METRICS:
                    target_value = _finite_float(holdout_row.get(metric_name))
                    if target_value is None:
                        continue
                    metric_provenance = dict(provenance.get(metric_name) or {})
                    if metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS and (
                        str(metric_provenance.get("observation_role") or "")
                        not in {"validation_only", "auxiliary_likelihood"}
                        or str(metric_provenance.get("allowed_use") or "")
                        not in {"validation_only", "held_out_validation_or_diagnostic_only", "auxiliary_likelihood"}
                    ):
                        leakage_rows.append(
                            {
                                "quarter": quarter,
                                "metric_name": metric_name,
                                "observation_role": str(metric_provenance.get("observation_role") or ""),
                                "allowed_use": str(metric_provenance.get("allowed_use") or ""),
                            }
                        )
                    candidate_value = _finite_float(candidate_row.get(metric_name))
                    carry_value = _finite_float(carry_row.get(metric_name))
                    scale = max(_annual_challenge_metric_scale(raw_train_rows, metric_name), float(np.finfo(np.float32).eps))
                    if candidate_value is None:
                        prediction_missing_counts[metric_name] += 1
                    score_rows.append(
                        {
                            "candidate_family": family,
                            "horizon_years": int(horizon),
                            "train_end_year": train_end_year,
                            "holdout_years": holdout_years,
                            "quarter": quarter,
                            "year": quarter_year(quarter),
                            "metric_name": metric_name,
                            "target_value": float(target_value),
                            "candidate_value": None if candidate_value is None else float(candidate_value),
                            "carry_forward_value": None if carry_value is None else float(carry_value),
                            "scale": float(scale),
                            "candidate_norm_error": None
                            if candidate_value is None
                            else abs(float(candidate_value) - float(target_value)) / scale,
                            "carry_forward_norm_error": None
                            if carry_value is None
                            else abs(float(carry_value) - float(target_value)) / scale,
                            "candidate_minus_carry_forward_norm_error": None
                            if candidate_value is None or carry_value is None
                            else (
                                abs(float(candidate_value) - float(target_value))
                                - abs(float(carry_value) - float(target_value))
                            )
                            / scale,
                            "prediction_status": "not_predicted" if candidate_value is None else "scored",
                            "observation_role": str(metric_provenance.get("observation_role") or ""),
                            "allowed_use": str(metric_provenance.get("allowed_use") or ""),
                            "source_id": str(metric_provenance.get("source_id") or ""),
                            "source_tier": str(metric_provenance.get("source_tier") or metric_provenance.get("source_quality_tier") or ""),
                            "support_partition": str(metric_provenance.get("support_partition") or ""),
                            "measurement_semantics": str(metric_provenance.get("measurement_semantics") or ""),
                            "training_use": "forbidden"
                            if metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
                            else "challenge_scoring_only",
                        }
                    )
    blockers: list[str] = []
    for horizon in horizons:
        if int(split_count_by_horizon.get(int(horizon)) or 0) == 0:
            blockers.append(f"h{int(horizon)}_no_annual_challenge_splits")
    if not score_rows:
        blockers.append("no_annual_challenge_score_rows")
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        if int(prediction_missing_counts.get(metric_name) or 0) > 0:
            blockers.append(f"{metric_name}_model_head_missing")
    if leakage_rows:
        blockers.append("validation_only_role_leakage")
    return score_rows, {
        "candidate_family": family,
        "score_record_count": len(score_rows),
        "split_count_by_horizon": {str(key): int(value) for key, value in sorted(split_count_by_horizon.items())},
        "prediction_missing_counts": dict(sorted(prediction_missing_counts.items())),
        "leakage_violation_count": len(leakage_rows),
        "leakage_rows": leakage_rows,
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
    }


def _score_summary_by_fields(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(dict(row))
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        candidate_errors = [
            float(row["candidate_norm_error"])
            for row in values
            if _finite_float(row.get("candidate_norm_error")) is not None
        ]
        carry_errors = [
            float(row["carry_forward_norm_error"])
            for row in values
            if _finite_float(row.get("carry_forward_norm_error")) is not None
        ]
        missing_count = sum(1 for row in values if str(row.get("prediction_status") or "") == "not_predicted")
        summary = {field: key[index] for index, field in enumerate(group_fields)}
        candidate_mean = None if not candidate_errors else float(np.mean(np.asarray(candidate_errors, dtype=np.float64)))
        carry_mean = None if not carry_errors else float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
        summary.update(
            {
                "entry_count": len(values),
                "scored_candidate_entry_count": len(candidate_errors),
                "missing_prediction_count": missing_count,
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
            }
        )
        output.append(summary)
    return output


def _build_r12_official_annual_challenge_gate_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    candidate_families: tuple[str, ...],
    horizons: tuple[int, ...] = R11_MULTI_HORIZON_YEARS,
) -> dict[str, Any]:
    all_score_rows: list[dict[str, Any]] = []
    family_manifests: list[dict[str, Any]] = []
    for family in candidate_families:
        score_rows, manifest = _official_annual_challenge_metric_rows(
            rows=rows,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
            family=family,
        )
        all_score_rows.extend(score_rows)
        family_manifests.append(manifest)
    metric_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(all_score_rows, group_fields=("candidate_family",))
    missing_required = sorted(
        {
            blocker
            for manifest in family_manifests
            for blocker in list(manifest.get("blockers") or [])
            if str(blocker).endswith("_model_head_missing")
        }
    )
    leakage = sum(int(manifest.get("leakage_violation_count") or 0) for manifest in family_manifests)
    scored_required = [
        row
        for row in metric_rows
        if str(row.get("metric_name") or "") in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
        and int(row.get("scored_candidate_entry_count") or 0) > 0
    ]
    cascade_rows = [
        row
        for row in metric_rows
        if str(row.get("metric_name") or "") not in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
        and int(row.get("scored_candidate_entry_count") or 0) > 0
    ]
    blockers: list[str] = []
    if not all_score_rows:
        blockers.append("no_official_annual_challenge_rows")
    if missing_required:
        blockers.extend(missing_required)
    if leakage:
        blockers.append("validation_only_role_leakage")
    if not scored_required:
        blockers.append("no_required_incidence_death_plhiv_model_heads_scored")
    decision = "keep_as_official_annual_challenge_gate"
    status = "pass" if not blockers else ("cascade_only_available" if cascade_rows and not leakage else "blocked")
    return {
        "schema_version": "phase3_dynamic.r12_10a_official_annual_challenge_gate.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-10A",
        "status": status,
        "decision": decision,
        "candidate_families": list(candidate_families),
        "horizons": list(horizons),
        "metric_scope": list(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
        "required_model_heads": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "family_manifests": family_manifests,
        "family_rows": family_rows,
        "horizon_rows": horizon_rows,
        "metric_rows": metric_rows,
        "score_record_count": len(all_score_rows),
        "scored_required_model_head_count": len(scored_required),
        "scored_cascade_metric_count": len(cascade_rows),
        "blockers": blockers,
        "score_records": all_score_rows,
        "contract": (
            "Official annual challenge rows are held out as AEM/Spectrum-style validation: annual incidence, AIDS deaths, "
            "estimated PLHIV, and annual cascade anchors are scored at annual Q4 only. Validation-only incidence/death rows "
            "are stripped from training rows before candidate predictions. Missing annual model heads are reported as missing, "
            "not replaced by quarterly diagnosis flow or cascade stocks."
        ),
    }


def _build_score_contract() -> dict[str, Any]:
    return {
        "schema_version": "phase3_dynamic.r11_score_contract.v1",
        "generated_at": _generated_at(),
        "metrics": {
            "raw_mae": "mean absolute error in target units",
            "normalized_mae": "absolute error divided by train-window metric scale",
            "smape": "symmetric mean absolute percentage error where available",
            "p90_residual": "90th percentile absolute residual",
            "coverage": "posterior or interval empirical coverage when intervals are emitted",
        },
        "endpoint_parity": [
            "annual targets compare only to annual predictions",
            "quarterly cascade targets compare only to quarterly predictions",
            "lifted path compares only to lifted path",
        ],
        "stock_consistency_rule": "a diagnosis-flow improvement is rejected if diagnosed_plhiv or alive_on_art worsens versus carry-forward",
    }


def _build_r11_01_reconciliation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    projection_rows: list[dict[str, Any]] = []
    violation_counter: Counter[str] = Counter()
    changed_count = 0
    for row in rows:
        violations = _cascade_cone_violations(row)
        for violation in violations:
            violation_counter[violation] += 1
        projection = project_cascade_stock_row(row)
        if projection["changed"]:
            changed_count += 1
        projection_rows.append(
            {
                "quarter": str(row.get("quarter") or ""),
                "violations": violations,
                "projection_changed": bool(projection["changed"]),
                "changed_metrics": list(projection["changed_metrics"]),
            }
        )
    return {
        "schema_version": "phase3_dynamic.r11_01_reconciliation_report.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-01",
        "status": "implemented",
        "decision": "keep_as_state_constraint_primitive",
        "row_count": len(rows),
        "violating_row_count": int(sum(1 for row in projection_rows if row["violations"])),
        "projection_changed_row_count": int(changed_count),
        "violation_counts": dict(sorted(violation_counter.items())),
        "rows": projection_rows,
        "contract": "deterministic nonnegative cascade-cone reconciliation; no fitted parameters",
    }


def _build_r11_02_support_weighted_report(rows: list[dict[str, Any]], ledger: dict[str, Any]) -> dict[str, Any]:
    metric_support: dict[str, Counter[str]] = defaultdict(Counter)
    metric_tier: dict[str, Counter[str]] = defaultdict(Counter)
    metric_role: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        provenance = dict(row.get("metric_provenance") or {})
        for metric_name, metric_provenance in provenance.items():
            if not isinstance(metric_provenance, dict):
                continue
            metric_support[str(metric_name)][str(metric_provenance.get("support_partition") or "unknown")] += 1
            metric_tier[str(metric_name)][str(metric_provenance.get("tier") or "unknown")] += 1
            metric_role[str(metric_name)][str(metric_provenance.get("observation_role") or "unknown")] += 1
    metric_rows = [
        {
            "metric_name": metric_name,
            "support_partition_counts": dict(sorted(metric_support[metric_name].items())),
            "tier_counts": dict(sorted(metric_tier[metric_name].items())),
            "role_counts": dict(sorted(metric_role[metric_name].items())),
            "operator_status": "eligible_for_support_weighting"
            if len(metric_support[metric_name]) > 1 or len(metric_tier[metric_name]) > 1
            else "single_support_class",
        }
        for metric_name in sorted(metric_support)
    ]
    return {
        "schema_version": "phase3_dynamic.r11_02_support_weighted_observation_report.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-02",
        "status": "implemented",
        "decision": "keep_as_observation_operator_contract",
        "ledger_row_count": int(ledger.get("row_count") or len(ledger.get("rows") or [])),
        "observation_role_counts": _observation_role_counts(ledger),
        "support_partition_counts": _support_partition_counts_from_rows(rows),
        "metric_rows": metric_rows,
        "contract": "support partitions and observation roles are carried into likelihood design; numeric likelihood weights are deferred to a fitted R11 state model",
    }


def _build_r11_03_stock_gate_report(branch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row
        for row in branch_rows
        if str(row.get("stock_consistency_status")) in {"pass", "fail"}
        and str(row.get("experiment_id")) in {"U_TO_D_COUPLING", "FRONTDOOR_COUPLING", "INCIDENCE_FULL_GATE"}
    ]
    rejected_false_wins = [
        row
        for row in evaluated
        if str(row.get("stock_consistency_status")) == "fail"
        and str(row.get("decision")) in {"reject_keep_as_falsification", "keep_limited"}
    ]
    status = "pass" if len(rejected_false_wins) >= 2 else "needs_more_evidence"
    return {
        "schema_version": "phase3_dynamic.r11_03_stock_consistency_gate_report.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-03",
        "status": status,
        "decision": "keep_as_claim_gate" if status == "pass" else "keep_as_diagnostic_only",
        "evaluated_branch_count": len(evaluated),
        "rejected_false_win_count": len(rejected_false_wins),
        "rejected_false_win_experiment_ids": [str(row.get("experiment_id")) for row in rejected_false_wins],
        "contract": "gate is retained if it blocks existing flow/readout wins that worsen guarded stocks",
    }


def _metric_provenance(row: dict[str, Any], metric_name: str) -> dict[str, Any]:
    provenance = dict(row.get("metric_provenance") or {})
    value = provenance.get(metric_name)
    return dict(value) if isinstance(value, dict) else {}


def _support_signature(row: dict[str, Any], metric_name: str) -> str:
    provenance = _metric_provenance(row, metric_name)
    return "|".join(
        [
            str(provenance.get("observation_role") or "unknown_role"),
            str(provenance.get("tier") or "unknown_tier"),
            str(provenance.get("support_partition") or "unknown_support"),
            str(provenance.get("aggregation_mode") or "unknown_aggregation"),
        ]
    )


def _available_metric_rows(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _finite_float(row.get(metric_name)) is not None
    ]


def _metric_scale(train_rows: list[dict[str, Any]], metric_name: str) -> float:
    values = [
        abs(float(value))
        for value in (_finite_float(row.get(metric_name)) for row in train_rows)
        if value is not None
    ]
    return max(values) if values else float(np.finfo(np.float32).eps)


def _fit_log_linear_support_model(train_rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    metric_rows = _available_metric_rows(train_rows, metric_name)
    if not metric_rows:
        return {
            "status": "not_estimable",
            "metric_name": metric_name,
            "reason": "no_train_values",
        }
    ordinals = np.asarray([quarter_ordinal(str(row["quarter"])) for row in metric_rows], dtype=np.float64)
    origin = float(ordinals[0])
    x = np.column_stack([np.ones_like(ordinals), ordinals - origin])
    y = np.asarray([np.log1p(max(float(row.get(metric_name) or 0.0), 0.0)) for row in metric_rows], dtype=np.float64)
    if len(metric_rows) >= x.shape[1]:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    else:
        beta = np.asarray([float(y[-1]), 0.0], dtype=np.float64)
    fitted = np.asarray(x @ beta, dtype=np.float64)
    residuals = y - fitted
    residuals_by_signature: dict[str, list[float]] = defaultdict(list)
    for row, residual in zip(metric_rows, residuals):
        residuals_by_signature[_support_signature(row, metric_name)].append(float(residual))
    support_bias_by_signature = {
        signature: float(np.median(np.asarray(values, dtype=np.float64)))
        for signature, values in sorted(residuals_by_signature.items())
    }
    sorted_rows = metric_rows
    slopes: list[float] = []
    for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
        previous_ordinal = quarter_ordinal(str(previous.get("quarter") or ""))
        current_ordinal = quarter_ordinal(str(current.get("quarter") or ""))
        step = current_ordinal - previous_ordinal
        if step <= 0:
            continue
        previous_value = np.log1p(max(float(previous.get(metric_name) or 0.0), 0.0))
        current_value = np.log1p(max(float(current.get(metric_name) or 0.0), 0.0))
        slopes.append(float((current_value - previous_value) / step))
    return {
        "status": "completed",
        "metric_name": metric_name,
        "train_count": len(metric_rows),
        "origin_ordinal": origin,
        "beta": [float(value) for value in beta],
        "support_bias_by_signature": support_bias_by_signature,
        "last_quarter": str(sorted_rows[-1].get("quarter") or ""),
        "last_canonical_log_value": float(
            np.log1p(max(float(sorted_rows[-1].get(metric_name) or 0.0), 0.0))
            - support_bias_by_signature.get(_support_signature(sorted_rows[-1], metric_name), 0.0)
        ),
        "median_quarterly_log_slope": float(np.median(np.asarray(slopes, dtype=np.float64))) if slopes else 0.0,
        "contract": "train-window log1p linear trend plus empirical support-signature residuals; no fitted hyperparameters",
    }


def _predict_support_trend(model: dict[str, Any], row: dict[str, Any], metric_name: str) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    beta = list(model.get("beta") or [])
    if len(beta) != 2:
        return None
    ordinal = float(quarter_ordinal(str(row.get("quarter") or "")))
    origin = float(model.get("origin_ordinal") or ordinal)
    signature = _support_signature(row, metric_name)
    support_bias = float(dict(model.get("support_bias_by_signature") or {}).get(signature, 0.0))
    log_value = float(beta[0]) + float(beta[1]) * (ordinal - origin) + support_bias
    return float(max(np.expm1(log_value), 0.0))


def _predict_local_level(model: dict[str, Any], row: dict[str, Any], metric_name: str) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    last_quarter = str(model.get("last_quarter") or "")
    if not last_quarter:
        return None
    step = max(quarter_ordinal(str(row.get("quarter") or "")) - quarter_ordinal(last_quarter), 0)
    signature = _support_signature(row, metric_name)
    support_bias = float(dict(model.get("support_bias_by_signature") or {}).get(signature, 0.0))
    canonical = float(model.get("last_canonical_log_value") or 0.0)
    slope = float(model.get("median_quarterly_log_slope") or 0.0)
    log_value = canonical + slope * float(step) + support_bias
    return float(max(np.expm1(log_value), 0.0))


def _metric_log_slopes(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    metric_rows = _available_metric_rows(rows, metric_name)
    slopes: list[float] = []
    for previous, current in zip(metric_rows[:-1], metric_rows[1:]):
        step = quarter_ordinal(str(current.get("quarter") or "")) - quarter_ordinal(str(previous.get("quarter") or ""))
        if step <= 0:
            continue
        previous_value = np.log1p(max(float(previous.get(metric_name) or 0.0), 0.0))
        current_value = np.log1p(max(float(current.get(metric_name) or 0.0), 0.0))
        slopes.append(float((current_value - previous_value) / step))
    return slopes


def _pooled_log_slopes(rows: list[dict[str, Any]]) -> list[float]:
    slopes: list[float] = []
    for metric_name in R11_EVALUATION_METRICS:
        slopes.extend(_metric_log_slopes(rows, metric_name))
    return slopes


def _bounded_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[1] != lower_array.size or lower_array.size != upper_array.size:
        return {
            "status": "not_estimable",
            "reason": "invalid_design_shape",
            "coefficient": [],
            "sse": None,
        }
    if x.shape[0] == 0:
        return {
            "status": "not_estimable",
            "reason": "empty_design",
            "coefficient": [],
            "sse": None,
        }
    states = ("free", "lower", "upper")
    best_beta: np.ndarray | None = None
    best_sse: float | None = None
    best_state: tuple[str, ...] | None = None
    from itertools import product

    for state in product(states, repeat=x.shape[1]):
        fixed = np.zeros(x.shape[1], dtype=np.float64)
        free_indices: list[int] = []
        for index, item in enumerate(state):
            if item == "lower":
                fixed[index] = lower_array[index]
            elif item == "upper":
                fixed[index] = upper_array[index]
            else:
                free_indices.append(index)
        residual_y = y - x @ fixed
        beta = fixed.copy()
        if free_indices:
            free_x = x[:, free_indices]
            solution, *_ = np.linalg.lstsq(free_x, residual_y, rcond=None)
            for local_index, feature_index in enumerate(free_indices):
                beta[feature_index] = float(solution[local_index])
            if np.any(beta[np.asarray(free_indices, dtype=int)] < lower_array[np.asarray(free_indices, dtype=int)]):
                continue
            if np.any(beta[np.asarray(free_indices, dtype=int)] > upper_array[np.asarray(free_indices, dtype=int)]):
                continue
        prediction = x @ beta
        sse = float(np.sum((prediction - y) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_beta = beta
            best_state = tuple(state)
    if best_beta is None:
        solution, *_ = np.linalg.lstsq(x, y, rcond=None)
        best_beta = np.minimum(np.maximum(solution, lower_array), upper_array)
        best_sse = float(np.sum((x @ best_beta - y) ** 2))
        best_state = tuple("clipped" for _ in range(x.shape[1]))
    return {
        "status": "completed",
        "coefficient": [float(value) for value in best_beta],
        "sse": float(best_sse),
        "active_set": list(best_state or ()),
        "row_count": int(x.shape[0]),
        "contract": "exact active-set bounded least squares over the small transition design; no optimizer hyperparameters",
    }


def _last_metric_value(train_rows: list[dict[str, Any]], metric_name: str) -> float | None:
    metric_rows = _available_metric_rows(train_rows, metric_name)
    if not metric_rows:
        return None
    return _finite_float(metric_rows[-1].get(metric_name))


def _metric_era_token(row: dict[str, Any], metric_name: str) -> str:
    provenance = _metric_provenance(row, metric_name)
    return "|".join(
        [
            str(provenance.get("support_partition") or "unknown_support"),
            str(provenance.get("tier") or "unknown_tier"),
            str(provenance.get("aggregation_mode") or "unknown_aggregation"),
        ]
    )


def _transition_era_signature(row: dict[str, Any]) -> str:
    return ";".join(
        [
            f"D={_metric_era_token(row, 'diagnosed_plhiv')}",
            f"A={_metric_era_token(row, 'alive_on_art')}",
            f"F={_metric_era_token(row, 'new_diagnosed_cases_period')}",
        ]
    )


def _fit_delta_transition_with_reporting_shifts(
    *,
    x_rows: list[list[float]],
    y_values: list[float],
    era_labels: list[str],
    feature_names: tuple[str, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> dict[str, Any]:
    if not x_rows or not y_values or len(x_rows) != len(y_values) or len(y_values) != len(era_labels):
        return {
            "status": "not_estimable",
            "reason": "empty_or_misaligned_transition_design",
            "feature_names": list(feature_names),
        }
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    fit = _bounded_least_squares(x, y, lower=lower, upper=upper)
    if str(fit.get("status") or "") != "completed":
        return {
            "status": "not_estimable",
            "reason": str(fit.get("reason") or "bounded_fit_failed"),
            "feature_names": list(feature_names),
        }
    coefficients = np.asarray(list(fit.get("coefficient") or []), dtype=np.float64)
    residuals = y - x @ coefficients
    global_shift = float(np.median(residuals)) if residuals.size else 0.0
    residuals_by_era: dict[str, list[float]] = defaultdict(list)
    for era, residual in zip(era_labels, residuals):
        residuals_by_era[str(era)].append(float(residual))
    reporting_shift_by_era = {
        era: float(np.median(np.asarray(values, dtype=np.float64)))
        for era, values in sorted(residuals_by_era.items())
    }
    adjusted_predictions = np.asarray(
        [
            float(x[index] @ coefficients) + float(reporting_shift_by_era.get(str(era_labels[index]), global_shift))
            for index in range(len(y_values))
        ],
        dtype=np.float64,
    )
    adjusted_sse = float(np.sum((adjusted_predictions - y) ** 2))
    return {
        "status": "completed",
        "feature_names": list(feature_names),
        "coefficients": [float(value) for value in coefficients],
        "row_count": int(len(y_values)),
        "era_count": int(len(reporting_shift_by_era)),
        "global_reporting_shift": global_shift,
        "reporting_shift_by_era": reporting_shift_by_era,
        "raw_sse": _finite_float(fit.get("sse")),
        "reporting_adjusted_sse": adjusted_sse,
        "active_set": list(fit.get("active_set") or []),
        "contract": "bounded transition coefficients plus empirical train-window reporting-shift residuals by observation-support era",
    }


def _fit_diagnosed_stock_transition(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
        if quarter_ordinal(str(current.get("quarter") or "")) <= quarter_ordinal(str(previous.get("quarter") or "")):
            continue
        previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
        current_diagnosed = _finite_float(current.get("diagnosed_plhiv"))
        diagnosis_flow = _finite_float(current.get("new_diagnosed_cases_period"))
        if previous_diagnosed is None or current_diagnosed is None or diagnosis_flow is None:
            continue
        x_rows.append([max(float(previous_diagnosed), 0.0), max(float(diagnosis_flow), 0.0)])
        y_values.append(max(float(current_diagnosed), 0.0))
    if not x_rows:
        return {
            "status": "not_estimable",
            "reason": "no_diagnosed_stock_flow_pairs",
            "feature_names": ["previous_diagnosed_plhiv", "diagnosis_flow"],
        }
    fit = _bounded_least_squares(
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    )
    coefficients = list(fit.get("coefficient") or [])
    return {
        "status": str(fit.get("status") or "not_estimable"),
        "feature_names": ["previous_diagnosed_plhiv", "diagnosis_flow"],
        "retention_coefficient": None if len(coefficients) < 1 else float(coefficients[0]),
        "diagnosis_flow_coefficient": None if len(coefficients) < 2 else float(coefficients[1]),
        "row_count": int(fit.get("row_count") or 0),
        "sse": _finite_float(fit.get("sse")),
        "active_set": list(fit.get("active_set") or []),
        "contract": "diagnosed stock transition D_t = rho_D D_{t-1} + gamma_D diagnosis_flow_t with rho_D,gamma_D in [0,1]",
    }


def _fit_art_stock_transition(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    by_ordinal = _train_row_by_ordinal(sorted_rows)
    lag_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for lag in TRANSITION_PROCESS_ART_LAGS:
        x_rows: list[list[float]] = []
        y_values: list[float] = []
        for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
            current_ord = quarter_ordinal(str(current.get("quarter") or ""))
            previous_ord = quarter_ordinal(str(previous.get("quarter") or ""))
            if current_ord <= previous_ord:
                continue
            lag_row = current if int(lag) == 0 else by_ordinal.get(current_ord - int(lag))
            previous_art = _finite_float(previous.get("alive_on_art"))
            current_art = _finite_float(current.get("alive_on_art"))
            previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
            lagged_flow = _finite_float((lag_row or {}).get("new_diagnosed_cases_period"))
            if previous_art is None or current_art is None or previous_diagnosed is None or lagged_flow is None:
                continue
            diagnosed_art_gap = max(float(previous_diagnosed) - float(previous_art), 0.0)
            x_rows.append([max(float(previous_art), 0.0), max(float(lagged_flow), 0.0), diagnosed_art_gap])
            y_values.append(max(float(current_art), 0.0))
        if not x_rows:
            lag_rows.append(
                {
                    "lag_quarters": int(lag),
                    "status": "not_estimable",
                    "row_count": 0,
                    "sse": None,
                    "coefficient": [],
                }
            )
            continue
        fit = _bounded_least_squares(
            np.asarray(x_rows, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
            lower=(0.0, 0.0, 0.0),
            upper=(1.0, 1.0, 1.0),
        )
        row = {
            "lag_quarters": int(lag),
            "status": str(fit.get("status") or "not_estimable"),
            "row_count": int(fit.get("row_count") or 0),
            "sse": _finite_float(fit.get("sse")),
            "coefficient": [float(value) for value in list(fit.get("coefficient") or [])],
            "active_set": list(fit.get("active_set") or []),
        }
        lag_rows.append(row)
        if row["status"] == "completed" and row["sse"] is not None:
            if best_row is None or float(row["sse"]) < float(best_row.get("sse") or float("inf")):
                best_row = row
    if best_row is None:
        return {
            "status": "not_estimable",
            "reason": "no_art_stock_flow_pairs",
            "candidate_lags": list(TRANSITION_PROCESS_ART_LAGS),
            "lag_rows": lag_rows,
        }
    coefficients = list(best_row.get("coefficient") or [])
    return {
        "status": "completed",
        "feature_names": ["previous_alive_on_art", "lagged_diagnosis_flow", "diagnosed_not_art_gap"],
        "selected_lag_quarters": int(best_row.get("lag_quarters") or 0),
        "art_retention_coefficient": None if len(coefficients) < 1 else float(coefficients[0]),
        "diagnosis_linkage_coefficient": None if len(coefficients) < 2 else float(coefficients[1]),
        "diagnosed_gap_linkage_coefficient": None if len(coefficients) < 3 else float(coefficients[2]),
        "row_count": int(best_row.get("row_count") or 0),
        "sse": _finite_float(best_row.get("sse")),
        "lag_rows": lag_rows,
        "contract": (
            "ART transition A_t = rho_A A_{t-1} + gamma_A diagnosis_flow_{t-lag} + eta_A max(D_{t-1}-A_{t-1},0), "
            "with all coefficients in [0,1] and lag selected from train stock-flow error"
        ),
    }


def _fit_datv_transition_process(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosed_model = _fit_diagnosed_stock_transition(train_rows)
    art_model = _fit_art_stock_transition(train_rows)
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    flow_by_ordinal = {
        quarter_ordinal(str(row.get("quarter") or "")): float(value)
        for row in sorted_rows
        for value in [_finite_float(row.get("new_diagnosed_cases_period"))]
        if value is not None
    }
    return {
        "status": "completed"
        if str(diagnosed_model.get("status") or "") == "completed" and str(art_model.get("status") or "") == "completed"
        else "not_estimable",
        "diagnosed_stock_transition": diagnosed_model,
        "art_stock_transition": art_model,
        "last_quarter": "" if not sorted_rows else str(sorted_rows[-1].get("quarter") or ""),
        "last_state": {
            "diagnosed_plhiv": _last_metric_value(train_rows, "diagnosed_plhiv"),
            "alive_on_art": _last_metric_value(train_rows, "alive_on_art"),
            "tested_for_viral_load": _last_metric_value(train_rows, "tested_for_viral_load"),
            "virally_suppressed": _last_metric_value(train_rows, "virally_suppressed"),
            "new_diagnosed_cases_period": _last_metric_value(train_rows, "new_diagnosed_cases_period"),
        },
        "observed_diagnosis_flow_by_ordinal": {str(key): value for key, value in sorted(flow_by_ordinal.items())},
        "contract": (
            "R11-19 process model: conserved diagnosed-stock recurrence, D_to_A linkage recurrence, ART retention, "
            "and downstream VL/suppression generated by the R11-14 conditional-rate process"
        ),
    }


def _apply_datv_transition_process(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    transition_process: dict[str, Any],
    back_half_process: dict[str, Any],
) -> list[dict[str, Any]]:
    if str(transition_process.get("status") or "") != "completed":
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    diagnosed_model = dict(transition_process.get("diagnosed_stock_transition") or {})
    art_model = dict(transition_process.get("art_stock_transition") or {})
    last_state = dict(transition_process.get("last_state") or {})
    current_diagnosed = _finite_float(last_state.get("diagnosed_plhiv"))
    current_art = _finite_float(last_state.get("alive_on_art"))
    if current_diagnosed is None or current_art is None:
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    flow_by_ordinal = {
        int(key): float(value)
        for key, value in dict(transition_process.get("observed_diagnosis_flow_by_ordinal") or {}).items()
        if _finite_float(value) is not None
    }
    predictions_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
    output: list[dict[str, Any]] = []
    sorted_holdout = sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    for holdout_row in sorted_holdout:
        quarter = str(holdout_row.get("quarter") or "")
        base_prediction = dict(predictions_by_quarter.get(quarter, {"quarter": quarter}))
        current_flow = _finite_float(base_prediction.get("new_diagnosed_cases_period"))
        if current_flow is None:
            current_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or 0.0
        current_flow = max(float(current_flow), 0.0)
        rho_d = _finite_float(diagnosed_model.get("retention_coefficient"))
        gamma_d = _finite_float(diagnosed_model.get("diagnosis_flow_coefficient"))
        if rho_d is None or gamma_d is None:
            next_diagnosed = _finite_float(base_prediction.get("diagnosed_plhiv"))
        else:
            next_diagnosed = float(max(float(rho_d) * max(current_diagnosed, 0.0) + float(gamma_d) * current_flow, 0.0))
        if next_diagnosed is None:
            next_diagnosed = current_diagnosed

        holdout_ord = quarter_ordinal(quarter)
        lag = int(art_model.get("selected_lag_quarters") or 0)
        if lag == 0:
            lagged_flow = current_flow
        else:
            lagged_flow = flow_by_ordinal.get(holdout_ord - lag)
            if lagged_flow is None:
                lagged_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or current_flow
        rho_a = _finite_float(art_model.get("art_retention_coefficient"))
        gamma_a = _finite_float(art_model.get("diagnosis_linkage_coefficient"))
        eta_a = _finite_float(art_model.get("diagnosed_gap_linkage_coefficient"))
        if rho_a is None or gamma_a is None or eta_a is None:
            next_art = _finite_float(base_prediction.get("alive_on_art"))
        else:
            diagnosed_gap = max(float(current_diagnosed) - float(current_art), 0.0)
            next_art = float(
                max(
                    float(rho_a) * max(current_art, 0.0)
                    + float(gamma_a) * max(float(lagged_flow), 0.0)
                    + float(eta_a) * diagnosed_gap,
                    0.0,
                )
            )
        if next_art is None:
            next_art = current_art
        next_art = float(min(max(float(next_art), 0.0), max(float(next_diagnosed), 0.0)))
        prediction = dict(base_prediction)
        prediction["diagnosed_plhiv"] = float(next_diagnosed)
        prediction["alive_on_art"] = float(next_art)
        prediction["new_diagnosed_cases_period"] = float(current_flow)
        prediction = _apply_back_half_rate_process(prediction, holdout_row, back_half_process)
        output.append(prediction)
        current_diagnosed = float(prediction.get("diagnosed_plhiv") or next_diagnosed)
        current_art = float(prediction.get("alive_on_art") or next_art)
        flow_by_ordinal[holdout_ord] = current_flow
    return output


def _fit_era_diagnosed_stock_transition(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    era_labels: list[str] = []
    for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
        if quarter_ordinal(str(current.get("quarter") or "")) <= quarter_ordinal(str(previous.get("quarter") or "")):
            continue
        previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
        current_diagnosed = _finite_float(current.get("diagnosed_plhiv"))
        diagnosis_flow = _finite_float(current.get("new_diagnosed_cases_period"))
        if previous_diagnosed is None or current_diagnosed is None or diagnosis_flow is None:
            continue
        x_rows.append([max(float(diagnosis_flow), 0.0), -max(float(previous_diagnosed), 0.0)])
        y_values.append(float(current_diagnosed) - float(previous_diagnosed))
        era_labels.append(_transition_era_signature(current))
    fit = _fit_delta_transition_with_reporting_shifts(
        x_rows=x_rows,
        y_values=y_values,
        era_labels=era_labels,
        feature_names=("diagnosis_flow", "diagnosed_removal_stock"),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    )
    if str(fit.get("status") or "") != "completed":
        return fit
    coefficients = list(fit.get("coefficients") or [])
    return {
        **fit,
        "diagnosis_flow_coefficient": None if len(coefficients) < 1 else float(coefficients[0]),
        "diagnosed_removal_fraction": None if len(coefficients) < 2 else float(coefficients[1]),
        "equation": "D_t = D_{t-1} + gamma_D flow_t - mu_D D_{t-1} + reporting_shift_D(era_t)",
    }


def _fit_era_art_stock_transition(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    by_ordinal = _train_row_by_ordinal(sorted_rows)
    lag_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lag in TRANSITION_PROCESS_ART_LAGS:
        x_rows: list[list[float]] = []
        y_values: list[float] = []
        era_labels: list[str] = []
        for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
            current_ord = quarter_ordinal(str(current.get("quarter") or ""))
            if current_ord <= quarter_ordinal(str(previous.get("quarter") or "")):
                continue
            lag_row = current if int(lag) == 0 else by_ordinal.get(current_ord - int(lag))
            previous_art = _finite_float(previous.get("alive_on_art"))
            current_art = _finite_float(current.get("alive_on_art"))
            previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
            lagged_flow = _finite_float((lag_row or {}).get("new_diagnosed_cases_period"))
            if previous_art is None or current_art is None or previous_diagnosed is None or lagged_flow is None:
                continue
            diagnosed_art_gap = max(float(previous_diagnosed) - float(previous_art), 0.0)
            x_rows.append([max(float(lagged_flow), 0.0), diagnosed_art_gap, -max(float(previous_art), 0.0)])
            y_values.append(float(current_art) - float(previous_art))
            era_labels.append(_transition_era_signature(current))
        fit = _fit_delta_transition_with_reporting_shifts(
            x_rows=x_rows,
            y_values=y_values,
            era_labels=era_labels,
            feature_names=("lagged_diagnosis_flow", "diagnosed_not_art_gap", "art_removal_stock"),
            lower=(0.0, 0.0, 0.0),
            upper=(1.0, 1.0, 1.0),
        )
        row = {
            "lag_quarters": int(lag),
            "status": str(fit.get("status") or "not_estimable"),
            "row_count": int(fit.get("row_count") or 0),
            "era_count": int(fit.get("era_count") or 0),
            "reporting_adjusted_sse": _finite_float(fit.get("reporting_adjusted_sse")),
            "raw_sse": _finite_float(fit.get("raw_sse")),
            "coefficients": list(fit.get("coefficients") or []),
        }
        lag_rows.append(row)
        if row["status"] == "completed" and row["reporting_adjusted_sse"] is not None:
            if best is None or float(row["reporting_adjusted_sse"]) < float(best.get("reporting_adjusted_sse") or float("inf")):
                best = {**fit, "selected_lag_quarters": int(lag), "lag_rows": lag_rows}
    if best is None:
        return {
            "status": "not_estimable",
            "reason": "no_era_art_stock_flow_pairs",
            "candidate_lags": list(TRANSITION_PROCESS_ART_LAGS),
            "lag_rows": lag_rows,
        }
    coefficients = list(best.get("coefficients") or [])
    return {
        **best,
        "lag_rows": lag_rows,
        "diagnosis_linkage_coefficient": None if len(coefficients) < 1 else float(coefficients[0]),
        "diagnosed_gap_linkage_coefficient": None if len(coefficients) < 2 else float(coefficients[1]),
        "art_removal_fraction": None if len(coefficients) < 3 else float(coefficients[2]),
        "equation": "A_t = A_{t-1} + gamma_A flow_{t-lag} + eta_A max(D_{t-1}-A_{t-1},0) - mu_A A_{t-1} + reporting_shift_A(era_t)",
    }


def _fit_era_datv_transition_process(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosed_model = _fit_era_diagnosed_stock_transition(train_rows)
    art_model = _fit_era_art_stock_transition(train_rows)
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    flow_by_ordinal = {
        quarter_ordinal(str(row.get("quarter") or "")): float(value)
        for row in sorted_rows
        for value in [_finite_float(row.get("new_diagnosed_cases_period"))]
        if value is not None
    }
    return {
        "status": "completed"
        if str(diagnosed_model.get("status") or "") == "completed" and str(art_model.get("status") or "") == "completed"
        else "not_estimable",
        "diagnosed_stock_transition": diagnosed_model,
        "art_stock_transition": art_model,
        "last_quarter": "" if not sorted_rows else str(sorted_rows[-1].get("quarter") or ""),
        "last_state": {
            "diagnosed_plhiv": _last_metric_value(train_rows, "diagnosed_plhiv"),
            "alive_on_art": _last_metric_value(train_rows, "alive_on_art"),
            "tested_for_viral_load": _last_metric_value(train_rows, "tested_for_viral_load"),
            "virally_suppressed": _last_metric_value(train_rows, "virally_suppressed"),
            "new_diagnosed_cases_period": _last_metric_value(train_rows, "new_diagnosed_cases_period"),
        },
        "observed_diagnosis_flow_by_ordinal": {str(key): value for key, value in sorted(flow_by_ordinal.items())},
        "contract": (
            "R11-20 era process: stock-flow transition with explicit stock removals and empirical reporting shifts "
            "by observation-support era; era is derived from ledger support/tier/aggregation metadata, not calendar labels"
        ),
    }


def _era_reporting_shift(model: dict[str, Any], row: dict[str, Any]) -> float:
    era = _transition_era_signature(row)
    shifts = dict(model.get("reporting_shift_by_era") or {})
    return float(shifts.get(era, model.get("global_reporting_shift") or 0.0))


def _apply_era_datv_transition_process(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    transition_process: dict[str, Any],
    back_half_process: dict[str, Any],
) -> list[dict[str, Any]]:
    if str(transition_process.get("status") or "") != "completed":
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    diagnosed_model = dict(transition_process.get("diagnosed_stock_transition") or {})
    art_model = dict(transition_process.get("art_stock_transition") or {})
    last_state = dict(transition_process.get("last_state") or {})
    current_diagnosed = _finite_float(last_state.get("diagnosed_plhiv"))
    current_art = _finite_float(last_state.get("alive_on_art"))
    if current_diagnosed is None or current_art is None:
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    flow_by_ordinal = {
        int(key): float(value)
        for key, value in dict(transition_process.get("observed_diagnosis_flow_by_ordinal") or {}).items()
        if _finite_float(value) is not None
    }
    predictions_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
    output: list[dict[str, Any]] = []
    for holdout_row in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout_row.get("quarter") or "")
        base_prediction = dict(predictions_by_quarter.get(quarter, {"quarter": quarter}))
        current_flow = _finite_float(base_prediction.get("new_diagnosed_cases_period"))
        if current_flow is None:
            current_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or 0.0
        current_flow = max(float(current_flow), 0.0)

        gamma_d = _finite_float(diagnosed_model.get("diagnosis_flow_coefficient"))
        mu_d = _finite_float(diagnosed_model.get("diagnosed_removal_fraction"))
        if gamma_d is None or mu_d is None:
            next_diagnosed = _finite_float(base_prediction.get("diagnosed_plhiv"))
        else:
            next_diagnosed = float(
                max(
                    current_diagnosed
                    + float(gamma_d) * current_flow
                    - float(mu_d) * max(current_diagnosed, 0.0)
                    + _era_reporting_shift(diagnosed_model, holdout_row),
                    0.0,
                )
            )
        if next_diagnosed is None:
            next_diagnosed = current_diagnosed

        holdout_ord = quarter_ordinal(quarter)
        lag = int(art_model.get("selected_lag_quarters") or 0)
        if lag == 0:
            lagged_flow = current_flow
        else:
            lagged_flow = flow_by_ordinal.get(holdout_ord - lag)
            if lagged_flow is None:
                lagged_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or current_flow
        gamma_a = _finite_float(art_model.get("diagnosis_linkage_coefficient"))
        eta_a = _finite_float(art_model.get("diagnosed_gap_linkage_coefficient"))
        mu_a = _finite_float(art_model.get("art_removal_fraction"))
        if gamma_a is None or eta_a is None or mu_a is None:
            next_art = _finite_float(base_prediction.get("alive_on_art"))
        else:
            diagnosed_gap = max(float(current_diagnosed) - float(current_art), 0.0)
            next_art = float(
                max(
                    current_art
                    + float(gamma_a) * max(float(lagged_flow), 0.0)
                    + float(eta_a) * diagnosed_gap
                    - float(mu_a) * max(current_art, 0.0)
                    + _era_reporting_shift(art_model, holdout_row),
                    0.0,
                )
            )
        if next_art is None:
            next_art = current_art
        next_art = float(min(max(float(next_art), 0.0), max(float(next_diagnosed), 0.0)))
        prediction = dict(base_prediction)
        prediction["diagnosed_plhiv"] = float(next_diagnosed)
        prediction["alive_on_art"] = float(next_art)
        prediction["new_diagnosed_cases_period"] = float(current_flow)
        prediction = _apply_back_half_rate_process(prediction, holdout_row, back_half_process)
        output.append(prediction)
        current_diagnosed = float(prediction.get("diagnosed_plhiv") or next_diagnosed)
        current_art = float(prediction.get("alive_on_art") or next_art)
        flow_by_ordinal[holdout_ord] = current_flow
    return output


def _fit_art_initiation_capacity_model(train_rows: list[dict[str, Any]], art_model: dict[str, Any]) -> dict[str, Any]:
    gamma_a = _finite_float(art_model.get("diagnosis_linkage_coefficient"))
    eta_a = _finite_float(art_model.get("diagnosed_gap_linkage_coefficient"))
    mu_a = _finite_float(art_model.get("art_removal_fraction"))
    if gamma_a is None or eta_a is None or mu_a is None:
        return {
            "status": "not_estimable",
            "reason": "missing_art_transition_coefficients",
            "contract": "capacity requires fitted linkage, gap, and removal coefficients",
        }
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    by_ordinal = _train_row_by_ordinal(sorted_rows)
    lag = int(art_model.get("selected_lag_quarters") or 0)
    capacity_by_era: dict[str, list[float]] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    eps = float(np.finfo(np.float32).eps)
    outside_bounds_count = 0
    for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
        current_ord = quarter_ordinal(str(current.get("quarter") or ""))
        if current_ord <= quarter_ordinal(str(previous.get("quarter") or "")):
            continue
        lag_row = current if lag == 0 else by_ordinal.get(current_ord - lag)
        previous_art = _finite_float(previous.get("alive_on_art"))
        current_art = _finite_float(current.get("alive_on_art"))
        previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
        lagged_flow = _finite_float((lag_row or {}).get("new_diagnosed_cases_period"))
        if previous_art is None or current_art is None or previous_diagnosed is None or lagged_flow is None:
            continue
        diagnosed_gap = max(float(previous_diagnosed) - float(previous_art), 0.0)
        delayed_linkage_pressure = float(gamma_a) * max(float(lagged_flow), 0.0)
        gap_linkage_pressure = float(eta_a) * diagnosed_gap
        raw_initiation_pressure = delayed_linkage_pressure + gap_linkage_pressure
        if raw_initiation_pressure <= eps:
            continue
        reporting_shift = _era_reporting_shift(art_model, current)
        observed_initiation = (
            float(current_art)
            - float(previous_art)
            + float(mu_a) * max(float(previous_art), 0.0)
            - reporting_shift
        )
        raw_capacity = observed_initiation / raw_initiation_pressure
        bounded_capacity = float(min(max(float(raw_capacity), 0.0), 1.0))
        if not np.isclose(raw_capacity, bounded_capacity, rtol=0.0, atol=0.0):
            outside_bounds_count += 1
        era = _transition_era_signature(current)
        capacity_by_era[era].append(bounded_capacity)
        component_rows.append(
            {
                "quarter": str(current.get("quarter") or ""),
                "era": era,
                "lag_quarters": lag,
                "delayed_linkage_pressure": delayed_linkage_pressure,
                "diagnosed_gap_linkage_pressure": gap_linkage_pressure,
                "raw_initiation_pressure": raw_initiation_pressure,
                "observed_initiation_after_removal_and_reporting": observed_initiation,
                "raw_capacity_fraction": float(raw_capacity),
                "bounded_capacity_fraction": bounded_capacity,
            }
        )
    all_capacity_values = [
        value
        for values in capacity_by_era.values()
        for value in values
    ]
    if not all_capacity_values:
        return {
            "status": "not_estimable",
            "reason": "no_positive_art_initiation_pressure_rows",
            "selected_lag_quarters": lag,
            "contract": "ART initiation capacity is estimable only when train rows contain positive fitted initiation pressure",
        }
    return {
        "status": "completed",
        "selected_lag_quarters": lag,
        "global_capacity_fraction": float(np.median(np.asarray(all_capacity_values, dtype=np.float64))),
        "capacity_fraction_by_era": {
            era: float(np.median(np.asarray(values, dtype=np.float64)))
            for era, values in sorted(capacity_by_era.items())
        },
        "row_count": len(component_rows),
        "outside_bounds_count": outside_bounds_count,
        "component_rows": component_rows,
        "contract": (
            "ART initiation capacity is the train-window bounded fraction of fitted delayed-linkage plus "
            "diagnosed-gap initiation pressure realized as observed ART stock gain after removal and reporting adjustment"
        ),
    }


def _art_capacity_fraction(capacity_model: dict[str, Any], row: dict[str, Any]) -> float:
    if str(capacity_model.get("status") or "") != "completed":
        return 1.0
    era = _transition_era_signature(row)
    values = dict(capacity_model.get("capacity_fraction_by_era") or {})
    value = _finite_float(values.get(era, capacity_model.get("global_capacity_fraction")))
    if value is None:
        return 1.0
    return float(min(max(float(value), 0.0), 1.0))


def _build_da_process_split_residual_anatomy(
    train_rows: list[dict[str, Any]],
    transition_process: dict[str, Any],
    capacity_model: dict[str, Any],
) -> dict[str, Any]:
    diagnosed_model = dict(transition_process.get("diagnosed_stock_transition") or {})
    art_model = dict(transition_process.get("art_stock_transition") or {})
    gamma_d = _finite_float(diagnosed_model.get("diagnosis_flow_coefficient"))
    mu_d = _finite_float(diagnosed_model.get("diagnosed_removal_fraction"))
    gamma_a = _finite_float(art_model.get("diagnosis_linkage_coefficient"))
    eta_a = _finite_float(art_model.get("diagnosed_gap_linkage_coefficient"))
    mu_a = _finite_float(art_model.get("art_removal_fraction"))
    if None in {gamma_d, mu_d, gamma_a, eta_a, mu_a}:
        return {
            "status": "not_estimable",
            "reason": "missing_process_split_coefficients",
            "rows": [],
        }
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    by_ordinal = _train_row_by_ordinal(sorted_rows)
    lag = int(art_model.get("selected_lag_quarters") or 0)
    rows: list[dict[str, Any]] = []
    for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
        current_ord = quarter_ordinal(str(current.get("quarter") or ""))
        if current_ord <= quarter_ordinal(str(previous.get("quarter") or "")):
            continue
        lag_row = current if lag == 0 else by_ordinal.get(current_ord - lag)
        previous_diagnosed = _finite_float(previous.get("diagnosed_plhiv"))
        current_diagnosed = _finite_float(current.get("diagnosed_plhiv"))
        previous_art = _finite_float(previous.get("alive_on_art"))
        current_art = _finite_float(current.get("alive_on_art"))
        current_flow = _finite_float(current.get("new_diagnosed_cases_period"))
        lagged_flow = _finite_float((lag_row or {}).get("new_diagnosed_cases_period"))
        if (
            previous_diagnosed is None
            or current_diagnosed is None
            or previous_art is None
            or current_art is None
            or current_flow is None
            or lagged_flow is None
        ):
            continue
        d_inflow = float(gamma_d) * max(float(current_flow), 0.0)
        d_removal = -float(mu_d) * max(float(previous_diagnosed), 0.0)
        d_reporting = _era_reporting_shift(diagnosed_model, current)
        d_observed_delta = float(current_diagnosed) - float(previous_diagnosed)
        d_predicted_delta = d_inflow + d_removal + d_reporting
        diagnosed_gap = max(float(previous_diagnosed) - float(previous_art), 0.0)
        capacity = _art_capacity_fraction(capacity_model, current)
        a_delayed_linkage = capacity * float(gamma_a) * max(float(lagged_flow), 0.0)
        a_gap_linkage = capacity * float(eta_a) * diagnosed_gap
        a_removal = -float(mu_a) * max(float(previous_art), 0.0)
        a_reporting = _era_reporting_shift(art_model, current)
        a_observed_delta = float(current_art) - float(previous_art)
        a_predicted_delta = a_delayed_linkage + a_gap_linkage + a_removal + a_reporting
        rows.append(
            {
                "quarter": str(current.get("quarter") or ""),
                "era": _transition_era_signature(current),
                "diagnosed_observed_delta": d_observed_delta,
                "diagnosed_predicted_delta": d_predicted_delta,
                "diagnosed_inflow_component": d_inflow,
                "diagnosed_removal_component": d_removal,
                "diagnosed_reporting_shift_component": d_reporting,
                "diagnosed_residual": float(d_observed_delta - d_predicted_delta),
                "art_observed_delta": a_observed_delta,
                "art_predicted_delta": a_predicted_delta,
                "art_delayed_linkage_component": a_delayed_linkage,
                "art_gap_linkage_component": a_gap_linkage,
                "art_capacity_fraction": capacity,
                "art_removal_component": a_removal,
                "art_reporting_shift_component": a_reporting,
                "art_residual": float(a_observed_delta - a_predicted_delta),
            }
        )
    diagnosed_residuals = [abs(float(row["diagnosed_residual"])) for row in rows]
    art_residuals = [abs(float(row["art_residual"])) for row in rows]
    return {
        "status": "completed" if rows else "not_estimable",
        "row_count": len(rows),
        "selected_art_lag_quarters": lag,
        "mean_abs_diagnosed_residual": None
        if not diagnosed_residuals
        else float(np.mean(np.asarray(diagnosed_residuals, dtype=np.float64))),
        "mean_abs_art_residual": None
        if not art_residuals
        else float(np.mean(np.asarray(art_residuals, dtype=np.float64))),
        "rows": rows,
        "contract": "train-window residual anatomy with D and A deltas decomposed into inflow/linkage, capacity, removal, reporting shift, and residual terms",
    }


def _fit_process_split_da_transition(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    transition_process = _fit_era_datv_transition_process(train_rows)
    art_model = dict(transition_process.get("art_stock_transition") or {})
    capacity_model = _fit_art_initiation_capacity_model(train_rows, art_model)
    residual_anatomy = _build_da_process_split_residual_anatomy(train_rows, transition_process, capacity_model)
    return {
        "status": "completed"
        if str(transition_process.get("status") or "") == "completed"
        else "not_estimable",
        "base_transition_process": transition_process,
        "art_initiation_capacity_model": capacity_model,
        "transition_residual_anatomy": residual_anatomy,
        "contract": (
            "R12-02 process split separates D reporting/removal/inflow from A delayed linkage, diagnosed-gap pressure, "
            "ART initiation capacity, ART removal, and A reporting shift; all components are fitted from train stock-flow pairs"
        ),
    }


def _apply_process_split_da_transition(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    process_split: dict[str, Any],
    back_half_process: dict[str, Any],
) -> list[dict[str, Any]]:
    transition_process = dict(process_split.get("base_transition_process") or {})
    if str(process_split.get("status") or "") != "completed" or str(transition_process.get("status") or "") != "completed":
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    diagnosed_model = dict(transition_process.get("diagnosed_stock_transition") or {})
    art_model = dict(transition_process.get("art_stock_transition") or {})
    capacity_model = dict(process_split.get("art_initiation_capacity_model") or {})
    last_state = dict(transition_process.get("last_state") or {})
    current_diagnosed = _finite_float(last_state.get("diagnosed_plhiv"))
    current_art = _finite_float(last_state.get("alive_on_art"))
    if current_diagnosed is None or current_art is None:
        return [_project_prediction_row(dict(row)) for row in base_predictions]
    flow_by_ordinal = {
        int(key): float(value)
        for key, value in dict(transition_process.get("observed_diagnosis_flow_by_ordinal") or {}).items()
        if _finite_float(value) is not None
    }
    predictions_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
    output: list[dict[str, Any]] = []
    for holdout_row in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout_row.get("quarter") or "")
        base_prediction = dict(predictions_by_quarter.get(quarter, {"quarter": quarter}))
        current_flow = _finite_float(base_prediction.get("new_diagnosed_cases_period"))
        if current_flow is None:
            current_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or 0.0
        current_flow = max(float(current_flow), 0.0)

        gamma_d = _finite_float(diagnosed_model.get("diagnosis_flow_coefficient"))
        mu_d = _finite_float(diagnosed_model.get("diagnosed_removal_fraction"))
        if gamma_d is None or mu_d is None:
            next_diagnosed = _finite_float(base_prediction.get("diagnosed_plhiv"))
        else:
            next_diagnosed = float(
                max(
                    float(current_diagnosed)
                    + float(gamma_d) * current_flow
                    - float(mu_d) * max(float(current_diagnosed), 0.0)
                    + _era_reporting_shift(diagnosed_model, holdout_row),
                    0.0,
                )
            )
        if next_diagnosed is None:
            next_diagnosed = current_diagnosed

        holdout_ord = quarter_ordinal(quarter)
        lag = int(art_model.get("selected_lag_quarters") or 0)
        if lag == 0:
            lagged_flow = current_flow
        else:
            lagged_flow = flow_by_ordinal.get(holdout_ord - lag)
            if lagged_flow is None:
                lagged_flow = _finite_float(last_state.get("new_diagnosed_cases_period")) or current_flow
        gamma_a = _finite_float(art_model.get("diagnosis_linkage_coefficient"))
        eta_a = _finite_float(art_model.get("diagnosed_gap_linkage_coefficient"))
        mu_a = _finite_float(art_model.get("art_removal_fraction"))
        if gamma_a is None or eta_a is None or mu_a is None:
            next_art = _finite_float(base_prediction.get("alive_on_art"))
        else:
            diagnosed_gap = max(float(current_diagnosed) - float(current_art), 0.0)
            capacity = _art_capacity_fraction(capacity_model, holdout_row)
            initiation = capacity * (
                float(gamma_a) * max(float(lagged_flow), 0.0)
                + float(eta_a) * diagnosed_gap
            )
            next_art = float(
                max(
                    float(current_art)
                    + initiation
                    - float(mu_a) * max(float(current_art), 0.0)
                    + _era_reporting_shift(art_model, holdout_row),
                    0.0,
                )
            )
        if next_art is None:
            next_art = current_art
        next_art = float(min(max(float(next_art), 0.0), max(float(next_diagnosed), 0.0)))
        prediction = dict(base_prediction)
        prediction["diagnosed_plhiv"] = float(next_diagnosed)
        prediction["alive_on_art"] = float(next_art)
        prediction["new_diagnosed_cases_period"] = float(current_flow)
        prediction = _apply_back_half_rate_process(prediction, holdout_row, back_half_process)
        output.append(prediction)
        current_diagnosed = float(prediction.get("diagnosed_plhiv") or next_diagnosed)
        current_art = float(prediction.get("alive_on_art") or next_art)
        flow_by_ordinal[holdout_ord] = current_flow
    return output


def _source_family_signature(row: dict[str, Any], metric_name: str) -> str:
    provenance = _metric_provenance(row, metric_name)
    return "|".join(
        [
            str(provenance.get("source_tier") or provenance.get("source_quality_tier") or "unknown_source_tier"),
            str(provenance.get("measurement_class") or "unknown_measurement_class"),
            str(provenance.get("series_kind") or "unknown_series_kind"),
        ]
    )


def _source_id_signature(row: dict[str, Any], metric_name: str) -> str:
    provenance = _metric_provenance(row, metric_name)
    return str(provenance.get("source_id") or "unknown_source_id")


def _monthly_reporting_intensity(row: dict[str, Any]) -> float:
    provenance = dict(row.get("metric_provenance") or {})
    metric_provenance = [dict(value) for value in provenance.values() if isinstance(value, dict)]
    if not metric_provenance:
        return 0.0
    monthly_like = 0
    for item in metric_provenance:
        series_kind = str(item.get("series_kind") or "")
        aggregation = str(item.get("aggregation_mode") or "")
        if "monthly" in series_kind or "monthly" in aggregation or "intraquarter" in aggregation:
            monthly_like += 1
    return float(monthly_like) / float(len(metric_provenance))


def _flow_regime_label(flow_value: float | None, median_flow: float | None) -> str:
    if flow_value is None or median_flow is None:
        return "flow_unknown"
    return "flow_ge_train_median" if float(flow_value) >= float(median_flow) else "flow_lt_train_median"


def _monthly_intensity_label(intensity: float, median_intensity: float | None) -> str:
    if median_intensity is None:
        return "monthly_intensity_unknown"
    return "monthly_intensity_ge_train_median" if float(intensity) >= float(median_intensity) else "monthly_intensity_lt_train_median"


def _residual_source_contexts(
    row: dict[str, Any],
    *,
    metric_name: str,
    flow_value: float | None,
    median_flow: float | None,
    median_monthly_intensity: float | None,
) -> dict[str, str]:
    source_family = _source_family_signature(row, metric_name)
    source_id = _source_id_signature(row, metric_name)
    support = _support_signature(row, metric_name)
    monthly = _monthly_intensity_label(_monthly_reporting_intensity(row), median_monthly_intensity)
    flow = _flow_regime_label(flow_value, median_flow)
    return {
        "source_family": source_family,
        "source_id": source_id,
        "support_signature": support,
        "monthly_reporting_intensity": monthly,
        "backlog_rebound_flow_regime": flow,
        "source_support": f"{source_family}::{support}",
        "reporting_rebound": f"{monthly}::{flow}",
        "source_reporting_rebound": f"{source_family}::{monthly}::{flow}",
    }


def _train_context_scalars(train_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    flows = [
        float(value)
        for value in (_finite_float(row.get("new_diagnosed_cases_period")) for row in train_rows)
        if value is not None
    ]
    intensities = [_monthly_reporting_intensity(row) for row in train_rows]
    return {
        "median_flow": None if not flows else float(np.median(np.asarray(flows, dtype=np.float64))),
        "median_monthly_intensity": None
        if not intensities
        else float(np.median(np.asarray(intensities, dtype=np.float64))),
    }


def _da_residual_source_records(
    train_rows: list[dict[str, Any]],
    process_split: dict[str, Any],
) -> list[dict[str, Any]]:
    anatomy = dict(process_split.get("transition_residual_anatomy") or {})
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in train_rows}
    scalars = _train_context_scalars(train_rows)
    records: list[dict[str, Any]] = []
    for component in list(anatomy.get("rows") or []):
        if not isinstance(component, dict):
            continue
        quarter = str(component.get("quarter") or "")
        row = by_quarter.get(quarter, {})
        flow_value = _finite_float(row.get("new_diagnosed_cases_period"))
        for metric_name, residual_key in (
            ("diagnosed_plhiv", "diagnosed_residual"),
            ("alive_on_art", "art_residual"),
        ):
            residual = _finite_float(component.get(residual_key))
            if residual is None:
                continue
            contexts = _residual_source_contexts(
                row,
                metric_name=metric_name,
                flow_value=flow_value,
                median_flow=_finite_float(scalars.get("median_flow")),
                median_monthly_intensity=_finite_float(scalars.get("median_monthly_intensity")),
            )
            records.append(
                {
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "residual": float(residual),
                    "abs_residual": abs(float(residual)),
                    "contexts": contexts,
                    "source_family": contexts["source_family"],
                    "source_id": contexts["source_id"],
                    "support_signature": contexts["support_signature"],
                    "monthly_reporting_intensity": contexts["monthly_reporting_intensity"],
                    "backlog_rebound_flow_regime": contexts["backlog_rebound_flow_regime"],
                }
            )
    return records


def _fit_da_residual_source_model(train_rows: list[dict[str, Any]], process_split: dict[str, Any]) -> dict[str, Any]:
    records = _da_residual_source_records(train_rows, process_split)
    context_names = (
        "source_family",
        "source_id",
        "support_signature",
        "monthly_reporting_intensity",
        "backlog_rebound_flow_regime",
        "source_support",
        "reporting_rebound",
        "source_reporting_rebound",
    )
    model_by_metric: dict[str, dict[str, Any]] = {}
    context_score_rows: list[dict[str, Any]] = []
    for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
        metric_records = [record for record in records if str(record.get("metric_name") or "") == metric_name]
        base_errors = [abs(float(record["residual"])) for record in metric_records]
        base_mean = None if not base_errors else float(np.mean(np.asarray(base_errors, dtype=np.float64)))
        base_worst = None if not base_errors else float(np.max(np.asarray(base_errors, dtype=np.float64)))
        best_context: str | None = None
        best_score: float | None = None
        selected_context_values: dict[str, float] = {}
        for context_name in context_names:
            corrected_errors: list[float] = []
            context_values: dict[str, list[float]] = defaultdict(list)
            for record in metric_records:
                value = str(dict(record.get("contexts") or {}).get(context_name) or "unknown")
                context_values[value].append(float(record["residual"]))
            for record in metric_records:
                value = str(dict(record.get("contexts") or {}).get(context_name) or "unknown")
                pool = [
                    float(other["residual"])
                    for other in metric_records
                    if str(dict(other.get("contexts") or {}).get(context_name) or "unknown") == value
                    and str(other.get("quarter") or "") != str(record.get("quarter") or "")
                ]
                prediction = 0.0 if not pool else float(np.median(np.asarray(pool, dtype=np.float64)))
                corrected_errors.append(abs(float(record["residual"]) - prediction))
            corrected_mean = None if not corrected_errors else float(np.mean(np.asarray(corrected_errors, dtype=np.float64)))
            corrected_worst = None if not corrected_errors else float(np.max(np.asarray(corrected_errors, dtype=np.float64)))
            selected = (
                base_mean is not None
                and base_worst is not None
                and corrected_mean is not None
                and corrected_worst is not None
                and corrected_mean < base_mean
                and corrected_worst <= base_worst
            )
            context_score_rows.append(
                {
                    "metric_name": metric_name,
                    "context_name": context_name,
                    "record_count": len(metric_records),
                    "base_mean_abs_residual": base_mean,
                    "corrected_mean_abs_residual": corrected_mean,
                    "base_worst_abs_residual": base_worst,
                    "corrected_worst_abs_residual": corrected_worst,
                    "selected": selected,
                }
            )
            if selected and corrected_mean is not None:
                if best_score is None or corrected_mean < best_score:
                    best_context = context_name
                    best_score = corrected_mean
                    selected_context_values = {
                        value: float(np.median(np.asarray(values, dtype=np.float64)))
                        for value, values in sorted(context_values.items())
                    }
        model_by_metric[metric_name] = {
            "status": "completed" if metric_records else "not_estimable",
            "selected_context": best_context,
            "selected_context_value_residuals": selected_context_values,
            "base_mean_abs_residual": base_mean,
            "selected_mean_abs_residual": best_score,
            "record_count": len(metric_records),
        }
    return {
        "status": "completed" if records else "not_estimable",
        "record_count": len(records),
        "context_names": list(context_names),
        "model_by_metric": model_by_metric,
        "context_score_rows": context_score_rows,
        "context_scalars": _train_context_scalars(train_rows),
        "contract": (
            "Train-only D/A residual-source model. It tests source family, source id, support signature, "
            "monthly reporting intensity, and train-derived backlog/rebound diagnosis-flow regime. A context is selected "
            "only if leave-quarter validation lowers mean residual and does not worsen worst residual."
        ),
    }


def _predict_da_residual_source_adjustment(
    model: dict[str, Any],
    holdout_row: dict[str, Any],
    *,
    metric_name: str,
    flow_value: float | None,
) -> float:
    metric_model = dict(dict(model.get("model_by_metric") or {}).get(metric_name) or {})
    selected_context = metric_model.get("selected_context")
    if not selected_context:
        return 0.0
    scalars = dict(model.get("context_scalars") or {})
    contexts = _residual_source_contexts(
        holdout_row,
        metric_name=metric_name,
        flow_value=flow_value,
        median_flow=_finite_float(scalars.get("median_flow")),
        median_monthly_intensity=_finite_float(scalars.get("median_monthly_intensity")),
    )
    context_value = str(contexts.get(str(selected_context)) or "unknown")
    residuals = dict(metric_model.get("selected_context_value_residuals") or {})
    value = _finite_float(residuals.get(context_value))
    return 0.0 if value is None else float(value)


def _r12_da_residual_source_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r12_da_process_split_transition",
    )
    process_split = _fit_process_split_da_transition(train_rows)
    residual_model = _fit_da_residual_source_model(train_rows, process_split)
    back_half_process = _fit_back_half_rate_process(train_rows)
    predictions: list[dict[str, Any]] = []
    for base_prediction, holdout_row in zip(
        base_predictions,
        sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
    ):
        row = dict(base_prediction)
        flow_value = _finite_float(row.get("new_diagnosed_cases_period"))
        for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
            value = _finite_float(row.get(metric_name))
            if value is None:
                continue
            row[metric_name] = float(max(value + _predict_da_residual_source_adjustment(
                residual_model,
                holdout_row,
                metric_name=metric_name,
                flow_value=flow_value,
            ), 0.0))
        row = _project_prediction_row(row)
        row = _apply_back_half_rate_process(row, holdout_row, back_half_process)
        predictions.append(row)
    return predictions, {
        "base_family": "r12_da_process_split_transition",
        "base_summary": base_summary,
        "process_split_transition": process_split,
        "residual_source_model": residual_model,
        "back_half_rate_process": back_half_process,
        "contract": (
            "R12-03 applies only train-selected residual-source corrections to D/A outputs from R12-02. "
            "The residual source model uses observation provenance and predicted diagnosis-flow context, not holdout targets."
        ),
    }


def _holdout_max_horizon_years(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> int:
    train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
    train_end_year = max(train_years) if train_years else 0
    return max(
        [
            max(quarter_year(str(row.get("quarter") or "")) - int(train_end_year), 1)
            for row in holdout_rows
            if row.get("quarter")
        ]
        or [1]
    )


def _train_origin_family_scores(
    train_rows: list[dict[str, Any]],
    *,
    candidate_families: tuple[str, ...],
    max_horizon_years: int,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    score_rows: list[dict[str, Any]] = []
    score_by_family: dict[str, list[float]] = {family: [] for family in candidate_families}
    full_by_family: dict[str, list[float]] = {family: [] for family in candidate_families}
    stock_by_family: dict[str, list[float]] = {family: [] for family in candidate_families}
    for train_end_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if int(train_end_year) < quarter_year(str(row.get("quarter") or "")) <= int(train_end_year) + int(max_horizon_years)
        ]
        if not internal_train or not internal_holdout:
            continue
        for family in candidate_families:
            predictions, _summary = _candidate_predictions(internal_train, internal_holdout, family=family)
            scoped_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=metrics,
            )
            full_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=R11_EVALUATION_METRICS,
            )
            stock_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=PRIMARY_STOCK_GUARD_METRICS,
            )
            if scoped_score is not None:
                score_by_family[family].append(float(scoped_score))
            if full_score is not None:
                full_by_family[family].append(float(full_score))
            if stock_score is not None:
                stock_by_family[family].append(float(stock_score))
            score_rows.append(
                {
                    "train_end_year": int(train_end_year),
                    "family": family,
                    "scoped_mean_mae": scoped_score,
                    "full_mean_mae": full_score,
                    "primary_stock_mean_mae": stock_score,
                }
            )
    family_rows: list[dict[str, Any]] = []
    for family in candidate_families:
        scoped_values = score_by_family.get(family, [])
        full_values = full_by_family.get(family, [])
        stock_values = stock_by_family.get(family, [])
        family_rows.append(
            {
                "family": family,
                "origin_count": len(scoped_values),
                "scoped_mean_mae": None if not scoped_values else float(np.mean(np.asarray(scoped_values, dtype=np.float64))),
                "full_mean_mae": None if not full_values else float(np.mean(np.asarray(full_values, dtype=np.float64))),
                "primary_stock_mean_mae": None if not stock_values else float(np.mean(np.asarray(stock_values, dtype=np.float64))),
            }
        )
    evaluable = [row for row in family_rows if _finite_float(row.get("scoped_mean_mae")) is not None]
    selected = candidate_families[0] if not evaluable else str(min(evaluable, key=lambda row: float(row["scoped_mean_mae"])).get("family"))
    return {
        "status": "completed" if evaluable else "not_estimable",
        "selected_family": selected,
        "candidate_families": list(candidate_families),
        "max_horizon_years": int(max_horizon_years),
        "metric_scope": list(metrics),
        "family_rows": family_rows,
        "score_rows": score_rows,
        "contract": "train-origin family selection on the specified metric scope; no holdout target values are used",
    }


def _blend_prediction_rows(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
    *,
    first_weight: float,
    second_weight: float,
) -> list[dict[str, Any]]:
    first_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in first_rows}
    output: list[dict[str, Any]] = []
    denom = max(float(first_weight) + float(second_weight), float(np.finfo(np.float32).eps))
    w1 = float(first_weight) / denom
    w2 = float(second_weight) / denom
    for second in second_rows:
        quarter = str(second.get("quarter") or "")
        first = first_by_quarter.get(quarter, {})
        row: dict[str, Any] = {"quarter": quarter}
        for metric_name in R11_EVALUATION_METRICS:
            first_value = _finite_float(first.get(metric_name))
            second_value = _finite_float(second.get(metric_name))
            if first_value is None:
                row[metric_name] = second_value
            elif second_value is None:
                row[metric_name] = first_value
            else:
                row[metric_name] = float(w1 * float(first_value) + w2 * float(second_value))
        output.append(_project_prediction_row(row))
    return output


def _family_selector_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    candidate_families: tuple[str, ...],
    metrics: tuple[str, ...] = R10_COMPARABLE_METRICS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selector = _train_origin_family_scores(
        train_rows,
        candidate_families=candidate_families,
        max_horizon_years=_holdout_max_horizon_years(train_rows, holdout_rows),
        metrics=metrics,
    )
    selected_family = str(selector.get("selected_family") or candidate_families[0])
    predictions, summary = _candidate_predictions(train_rows, holdout_rows, family=selected_family)
    return predictions, {
        "family_selector": selector,
        "selected_family_summary": summary,
        "selected_family": selected_family,
    }


def _family_weighted_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    first_family: str,
    second_family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selector = _train_origin_family_scores(
        train_rows,
        candidate_families=(first_family, second_family),
        max_horizon_years=_holdout_max_horizon_years(train_rows, holdout_rows),
        metrics=R10_COMPARABLE_METRICS,
    )
    rows_by_family = {str(row.get("family") or ""): dict(row) for row in list(selector.get("family_rows") or [])}
    first_score = _finite_float(rows_by_family.get(first_family, {}).get("scoped_mean_mae"))
    second_score = _finite_float(rows_by_family.get(second_family, {}).get("scoped_mean_mae"))
    if first_score is None or second_score is None:
        first_weight = 1.0
        second_weight = 1.0
    else:
        first_weight = 1.0 / max(float(first_score), float(np.finfo(np.float32).eps))
        second_weight = 1.0 / max(float(second_score), float(np.finfo(np.float32).eps))
    first_predictions, first_summary = _candidate_predictions(train_rows, holdout_rows, family=first_family)
    second_predictions, second_summary = _candidate_predictions(train_rows, holdout_rows, family=second_family)
    predictions = _blend_prediction_rows(first_predictions, second_predictions, first_weight=first_weight, second_weight=second_weight)
    back_half_process = _fit_back_half_rate_process(train_rows)
    predictions = [_apply_back_half_rate_process(row, holdout_row, back_half_process) for row, holdout_row in zip(predictions, sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))))]
    return predictions, {
        "family_selector": selector,
        "first_family_summary": first_summary,
        "second_family_summary": second_summary,
        "first_weight": first_weight,
        "second_weight": second_weight,
        "contract": "train-origin inverse-error weighted blend of two predeclared families, followed by the R11-14 conditional-rate process",
    }


def _fit_r12_long_horizon_stock_drift_model(
    train_rows: list[dict[str, Any]],
    *,
    max_horizon_years: int,
) -> dict[str, Any]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    records: list[dict[str, Any]] = []
    for train_end_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if int(train_end_year) < quarter_year(str(row.get("quarter") or "")) <= int(train_end_year) + int(max_horizon_years)
        ]
        if not internal_train or not internal_holdout:
            continue
        base_predictions, _summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family="multi_horizon_weighted_process",
        )
        carry_predictions = _carry_forward_prediction(internal_train, internal_holdout)
        base_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
        carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_predictions}
        for holdout_row in internal_holdout:
            lead_years = max(quarter_year(str(holdout_row.get("quarter") or "")) - int(train_end_year), 1)
            if lead_years <= 1:
                continue
            for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
                predicted = _finite_float(base_by_quarter.get(str(holdout_row.get("quarter") or ""), {}).get(metric_name))
                carry = _finite_float(carry_by_quarter.get(str(holdout_row.get("quarter") or ""), {}).get(metric_name))
                target = _finite_float(holdout_row.get(metric_name))
                if predicted is None or carry is None or target is None:
                    continue
                scale = _metric_scale(internal_train, metric_name)
                records.append(
                    {
                        "origin_year": int(train_end_year),
                        "lead_years": int(lead_years),
                        "metric_name": metric_name,
                        "base_prediction": float(max(predicted, 0.0)),
                        "carry_forward_prediction": float(max(carry, 0.0)),
                        "target_value": float(max(target, 0.0)),
                        "scale": float(scale),
                        "log_residual": float(np.log1p(max(target, 0.0)) - np.log1p(max(predicted, 0.0))),
                    }
                )
    records_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{record['metric_name']}|lead{int(record['lead_years'])}"
        records_by_key[key].append(record)
    correction_rows: list[dict[str, Any]] = []
    selected_corrections: dict[str, float] = {}
    for key, key_records in sorted(records_by_key.items()):
        base_errors: list[float] = []
        carry_errors: list[float] = []
        corrected_errors: list[float] = []
        individual_nonregression_flags: list[bool] = []
        loo_record_count = 0
        for record in key_records:
            residual_pool = [
                float(other["log_residual"])
                for other in key_records
                if int(other["origin_year"]) != int(record["origin_year"])
            ]
            if not residual_pool:
                continue
            correction = float(np.median(np.asarray(residual_pool, dtype=np.float64)))
            corrected = float(np.expm1(np.log1p(float(record["base_prediction"])) + correction))
            scale = max(float(record["scale"]), float(np.finfo(np.float32).eps))
            base_errors.append(abs(float(record["base_prediction"]) - float(record["target_value"])) / scale)
            carry_errors.append(abs(float(record["carry_forward_prediction"]) - float(record["target_value"])) / scale)
            corrected_errors.append(abs(max(corrected, 0.0) - float(record["target_value"])) / scale)
            individual_nonregression_flags.append(
                corrected_errors[-1] <= base_errors[-1]
                and corrected_errors[-1] <= carry_errors[-1]
            )
            loo_record_count += 1
        base_mean = None if not base_errors else float(np.mean(np.asarray(base_errors, dtype=np.float64)))
        carry_mean = None if not carry_errors else float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
        corrected_mean = None if not corrected_errors else float(np.mean(np.asarray(corrected_errors, dtype=np.float64)))
        base_worst = None if not base_errors else float(np.max(np.asarray(base_errors, dtype=np.float64)))
        carry_worst = None if not carry_errors else float(np.max(np.asarray(carry_errors, dtype=np.float64)))
        corrected_worst = None if not corrected_errors else float(np.max(np.asarray(corrected_errors, dtype=np.float64)))
        selected = (
            base_mean is not None
            and carry_mean is not None
            and corrected_mean is not None
            and base_worst is not None
            and carry_worst is not None
            and corrected_worst is not None
            and corrected_mean < base_mean
            and corrected_worst <= base_worst
            and corrected_mean <= carry_mean
            and corrected_worst <= carry_worst
            and bool(individual_nonregression_flags)
            and all(individual_nonregression_flags)
        )
        final_correction = float(
            np.median(np.asarray([float(record["log_residual"]) for record in key_records], dtype=np.float64))
        )
        if selected:
            selected_corrections[key] = final_correction
        metric_name, lead_text = key.split("|lead", 1)
        correction_rows.append(
            {
                "metric_name": metric_name,
                "lead_years": int(lead_text),
                "record_count": len(key_records),
                "leave_origin_record_count": loo_record_count,
                "base_mean_norm_error": base_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "corrected_mean_norm_error": corrected_mean,
                "base_worst_norm_error": base_worst,
                "carry_forward_worst_norm_error": carry_worst,
                "corrected_worst_norm_error": corrected_worst,
                "individual_nonregression_count": int(sum(1 for flag in individual_nonregression_flags if flag)),
                "selected": selected,
                "log_correction": final_correction if selected else 0.0,
            }
        )
    return {
        "status": "completed" if correction_rows else "not_estimable",
        "max_horizon_years": int(max_horizon_years),
        "reference_family": "multi_horizon_weighted_process",
        "corrected_metrics": list(R12_LONG_HORIZON_STOCK_METRICS),
        "record_count": len(records),
        "selected_correction_count": len(selected_corrections),
        "correction_by_metric_lead": selected_corrections,
        "correction_rows": correction_rows,
        "contract": (
            "R12 fits leave-origin validated log-residual corrections only for diagnosed_plhiv and alive_on_art "
            "at lead years greater than one. A correction is selected only if it strictly improves train-origin "
            "mean long-horizon stock error and does not worsen train-origin worst-case stock error against the "
            "frozen R11-28 reference, while also remaining no worse than train-origin carry-forward on mean and "
            "worst-case stock error. Every leave-origin record must also be no worse than both R11-28 and "
            "carry-forward, so unsafe corrections fail closed."
        ),
    }


def _apply_r12_long_horizon_stock_drift_model(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    model: dict[str, Any],
    *,
    train_end_year: int,
    back_half_process: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base_prediction)
    lead_years = max(quarter_year(str(holdout_row.get("quarter") or "")) - int(train_end_year), 1)
    corrections = dict(model.get("correction_by_metric_lead") or {})
    for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
        key = f"{metric_name}|lead{int(lead_years)}"
        correction = _finite_float(corrections.get(key))
        value = _finite_float(row.get(metric_name))
        if correction is None or value is None:
            continue
        row[metric_name] = float(max(np.expm1(np.log1p(max(float(value), 0.0)) + float(correction)), 0.0))
    row = _project_prediction_row(row)
    return _apply_back_half_rate_process(row, holdout_row, back_half_process)


def _r12_long_horizon_stock_shape_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="multi_horizon_weighted_process",
    )
    train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
    train_end_year = max(train_years) if train_years else 0
    model = _fit_r12_long_horizon_stock_drift_model(
        train_rows,
        max_horizon_years=_holdout_max_horizon_years(train_rows, holdout_rows),
    )
    back_half_process = _fit_back_half_rate_process(train_rows)
    predictions = [
        _apply_r12_long_horizon_stock_drift_model(
            base_prediction,
            holdout_row,
            model,
            train_end_year=int(train_end_year),
            back_half_process=back_half_process,
        )
        for base_prediction, holdout_row in zip(
            base_predictions,
            sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
        )
    ]
    return predictions, {
        "base_family": "multi_horizon_weighted_process",
        "base_summary": base_summary,
        "r12_stock_drift_correction": model,
        "back_half_rate_process": back_half_process,
        "contract": (
            "R12-01 starts from the promoted R11-28 reference and allows only train-origin validated "
            "long-horizon corrections to diagnosed_plhiv and alive_on_art; VL and suppression are regenerated "
            "from the conditional-rate process after the stock cone projection."
        ),
    }


def _r12_route_aware_two_head_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nowcast_predictions, nowcast_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="multi_horizon_weighted_process",
    )
    trajectory_predictions, trajectory_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r12_long_horizon_stock_shape_process",
    )
    train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
    train_end_year = max(train_years) if train_years else 0
    trajectory_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in trajectory_predictions}
    output: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for nowcast_prediction, holdout_row in zip(
        nowcast_predictions,
        sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
    ):
        quarter = str(holdout_row.get("quarter") or nowcast_prediction.get("quarter") or "")
        lead_years = max(quarter_year(quarter) - int(train_end_year), 1)
        if int(lead_years) <= max(R12_SUPPORT_ADEQUACY_HORIZONS):
            selected_head = "program_nowcast_head"
            selected_family = "multi_horizon_weighted_process"
            row = dict(nowcast_prediction)
        else:
            selected_head = "annual_trajectory_head"
            selected_family = "r12_long_horizon_stock_shape_process"
            row = dict(trajectory_by_quarter.get(quarter, nowcast_prediction))
        selection_rows.append(
            {
                "quarter": quarter,
                "lead_years": int(lead_years),
                "selected_head": selected_head,
                "selected_family": selected_family,
            }
        )
        output.append(_project_prediction_row(row))
    return output, {
        "base_family": "route_aware_two_head",
        "nowcast_head_family": "multi_horizon_weighted_process",
        "trajectory_head_family": "r12_long_horizon_stock_shape_process",
        "nowcast_head_summary": nowcast_summary,
        "trajectory_head_summary": trajectory_summary,
        "selection_rows": selection_rows,
        "head_selection_contract": {
            "program_nowcast_head_horizons": list(R12_SUPPORT_ADEQUACY_HORIZONS),
            "annual_trajectory_head_horizons": [3, 5],
            "routing_variable": "lead_years_from_train_end",
        },
        "contract": (
            "R12-08 is a route-aware two-head model: 1y/2y predictions use the locked R11-28 program nowcast head, "
            "while longer-horizon predictions use the R12-01 annual trajectory head. Head selection is based only on "
            "forecast lead time from the train origin, not on holdout target values."
        ),
    }


def _r12_is_annual_anchor_metric(row: dict[str, Any], metric_name: str) -> bool:
    annual_lineage = _r12_lineage_by_id(R12_ANNUAL_ANCHOR_LINEAGE_ID)
    source_family = str(annual_lineage.get("source_family") or "")
    return bool(source_family) and _r12_metric_matches_source_family(row, metric_name, source_family)


def _r12_is_annual_anchor_stock_row(row: dict[str, Any]) -> bool:
    return any(_r12_is_annual_anchor_metric(row, metric_name) for metric_name in R12_LONG_HORIZON_STOCK_METRICS)


def _r12_stock_error_summary(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = R12_LONG_HORIZON_STOCK_METRICS,
) -> dict[str, Any]:
    errors: list[float] = []
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in prediction_rows}
    for holdout_row in holdout_rows:
        quarter = str(holdout_row.get("quarter") or "")
        prediction = by_quarter.get(quarter, {})
        for metric_name in metrics:
            target_value = _finite_float(holdout_row.get(metric_name))
            predicted_value = _finite_float(prediction.get(metric_name))
            if target_value is None or predicted_value is None:
                continue
            scale = max(_metric_scale(train_rows, metric_name), float(np.finfo(np.float32).eps))
            errors.append(abs(float(predicted_value) - float(target_value)) / scale)
    return {
        "entry_count": len(errors),
        "mean_norm_error": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
        "worst_norm_error": None if not errors else float(np.max(np.asarray(errors, dtype=np.float64))),
    }


def _fit_r12_annual_anchor_head_selector(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    annual_lineage = _r12_lineage_by_id(R12_ANNUAL_ANCHOR_LINEAGE_ID)
    annual_rows = _r12_lineage_filtered_rows(train_rows, annual_lineage)
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in annual_rows if row.get("quarter")})
    candidate_families = ("support_reporting_bias", "local_level_filter")
    family_errors: dict[str, list[float]] = {"multi_horizon_weighted_process": [], "carry_forward": []}
    family_worst_errors: dict[str, list[float]] = {"multi_horizon_weighted_process": [], "carry_forward": []}
    for family in candidate_families:
        family_errors[family] = []
        family_worst_errors[family] = []
    if len(years) < 2:
        return {
            "status": "not_estimable",
            "selected_family": "multi_horizon_weighted_process",
            "reason": "insufficient_annual_anchor_years",
            "lineage_id": R12_ANNUAL_ANCHOR_LINEAGE_ID,
            "selector_rows": [],
        }
    selector_rows: list[dict[str, Any]] = []
    for holdout_year in years[1:]:
        internal_train = [
            dict(row)
            for row in annual_rows
            if quarter_year(str(row.get("quarter") or "")) < int(holdout_year)
        ]
        internal_holdout = [
            dict(row)
            for row in annual_rows
            if quarter_year(str(row.get("quarter") or "")) == int(holdout_year)
        ]
        if not internal_train or not internal_holdout:
            continue
        family_predictions: dict[str, list[dict[str, Any]]] = {
            "carry_forward": _carry_forward_prediction(internal_train, internal_holdout)
        }
        base_predictions, _base_summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family="multi_horizon_weighted_process",
        )
        family_predictions["multi_horizon_weighted_process"] = base_predictions
        for family in candidate_families:
            predictions, _summary = _candidate_predictions(internal_train, internal_holdout, family=family)
            family_predictions[family] = predictions
        for family, predictions in family_predictions.items():
            error = _r12_stock_error_summary(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
            )
            mean_error = _finite_float(error.get("mean_norm_error"))
            worst_error = _finite_float(error.get("worst_norm_error"))
            if mean_error is not None:
                family_errors[family].append(float(mean_error))
            if worst_error is not None:
                family_worst_errors[family].append(float(worst_error))
            selector_rows.append(
                {
                    "holdout_year": int(holdout_year),
                    "candidate_family": family,
                    "entry_count": int(error.get("entry_count") or 0),
                    "mean_norm_error": mean_error,
                    "worst_norm_error": worst_error,
                }
            )
    base_mean = None if not family_errors["multi_horizon_weighted_process"] else float(
        np.mean(np.asarray(family_errors["multi_horizon_weighted_process"], dtype=np.float64))
    )
    base_worst = None if not family_worst_errors["multi_horizon_weighted_process"] else float(
        np.max(np.asarray(family_worst_errors["multi_horizon_weighted_process"], dtype=np.float64))
    )
    carry_mean = None if not family_errors["carry_forward"] else float(
        np.mean(np.asarray(family_errors["carry_forward"], dtype=np.float64))
    )
    carry_worst = None if not family_worst_errors["carry_forward"] else float(
        np.max(np.asarray(family_worst_errors["carry_forward"], dtype=np.float64))
    )
    selected_family = "multi_horizon_weighted_process"
    selected_mean = base_mean
    selected_worst = base_worst
    candidate_rows: list[dict[str, Any]] = []
    score_tolerance = float(np.finfo(np.float32).eps)
    for family in candidate_families:
        mean_error = None if not family_errors[family] else float(np.mean(np.asarray(family_errors[family], dtype=np.float64)))
        worst_error = None if not family_worst_errors[family] else float(np.max(np.asarray(family_worst_errors[family], dtype=np.float64)))
        eligible = (
            mean_error is not None
            and worst_error is not None
            and base_mean is not None
            and base_worst is not None
            and carry_mean is not None
            and carry_worst is not None
            and mean_error < base_mean
            and worst_error <= base_worst + score_tolerance
            and mean_error <= carry_mean + score_tolerance
            and worst_error <= carry_worst + score_tolerance
        )
        candidate_rows.append(
            {
                "candidate_family": family,
                "mean_norm_error": mean_error,
                "worst_norm_error": worst_error,
                "eligible": eligible,
            }
        )
        if eligible and (selected_mean is None or float(mean_error) < float(selected_mean)):
            selected_family = family
            selected_mean = mean_error
            selected_worst = worst_error
    blend_selector = _fit_r12_annual_anchor_blend_selector(train_rows, selected_family=selected_family)
    return {
        "status": "completed" if selector_rows else "not_estimable",
        "lineage_id": R12_ANNUAL_ANCHOR_LINEAGE_ID,
        "selected_family": selected_family,
        "selected_blend_weight": blend_selector.get("selected_blend_weight"),
        "base_family": "multi_horizon_weighted_process",
        "base_mean_norm_error": base_mean,
        "base_worst_norm_error": base_worst,
        "carry_forward_mean_norm_error": carry_mean,
        "carry_forward_worst_norm_error": carry_worst,
        "selected_mean_norm_error": selected_mean,
        "selected_worst_norm_error": selected_worst,
        "candidate_rows": candidate_rows,
        "selector_rows": selector_rows,
        "blend_selector": blend_selector,
        "contract": (
            "R12-09 selects an annual-anchor stock head only from train-origin slide-anchor rows. "
            "A candidate head must improve annual-anchor D/A mean error over R11-28 and must not worsen "
            "annual-anchor D/A worst error versus R11-28 or carry-forward. If that evidence is absent, "
            "the selector fails closed to the locked R11-28 backbone."
        ),
    }


def _r12_convex_error(value_a: float, value_b: float, alpha: float, target: float, scale: float) -> float:
    blended = (1.0 - float(alpha)) * float(value_a) + float(alpha) * float(value_b)
    return abs(blended - float(target)) / max(float(scale), float(np.finfo(np.float32).eps))


def _r12_alpha_candidates(records: list[dict[str, Any]]) -> list[float]:
    candidates = {0.0, 1.0}
    for record in records:
        base_value = float(record["base_prediction"])
        selected_value = float(record["selected_prediction"])
        target_value = float(record["target_value"])
        delta = selected_value - base_value
        if abs(delta) <= float(np.finfo(np.float64).eps):
            continue
        reference_errors = {
            0.0,
            abs(base_value - target_value),
            abs(float(record["carry_forward_prediction"]) - target_value),
        }
        for raw_error in reference_errors:
            for signed_error in (-raw_error, raw_error):
                alpha = (target_value + signed_error - base_value) / delta
                if -float(np.finfo(np.float32).eps) <= alpha <= 1.0 + float(np.finfo(np.float32).eps):
                    candidates.add(float(min(max(alpha, 0.0), 1.0)))
        alpha_at_target = (target_value - base_value) / delta
        if -float(np.finfo(np.float32).eps) <= alpha_at_target <= 1.0 + float(np.finfo(np.float32).eps):
            candidates.add(float(min(max(alpha_at_target, 0.0), 1.0)))
    return sorted(candidates)


def _fit_r12_annual_anchor_blend_selector(
    train_rows: list[dict[str, Any]],
    *,
    selected_family: str,
) -> dict[str, Any]:
    if selected_family == "multi_horizon_weighted_process":
        return {
            "status": "not_required",
            "selected_blend_weight": 0.0,
            "reason": "annual_anchor_head_not_selected",
        }
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    records: list[dict[str, Any]] = []
    for holdout_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) < int(holdout_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) == int(holdout_year)
        ]
        if not internal_train or not internal_holdout:
            continue
        base_predictions, _base_summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family="multi_horizon_weighted_process",
        )
        selected_predictions, _selected_summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family=selected_family,
        )
        carry_predictions = _carry_forward_prediction(internal_train, internal_holdout)
        base_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
        selected_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in selected_predictions}
        carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_predictions}
        for holdout_row in internal_holdout:
            quarter = str(holdout_row.get("quarter") or "")
            base_row = base_by_quarter.get(quarter, {})
            selected_row = selected_by_quarter.get(quarter, {})
            carry_row = carry_by_quarter.get(quarter, {})
            for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
                target_value = _finite_float(holdout_row.get(metric_name))
                base_value = _finite_float(base_row.get(metric_name))
                carry_value = _finite_float(carry_row.get(metric_name))
                if target_value is None or base_value is None or carry_value is None:
                    continue
                selected_value = _finite_float(selected_row.get(metric_name))
                if selected_value is None or not _r12_is_annual_anchor_metric(holdout_row, metric_name):
                    selected_value = base_value
                records.append(
                    {
                        "holdout_year": int(holdout_year),
                        "quarter": quarter,
                        "metric_name": metric_name,
                        "annual_anchor_metric": _r12_is_annual_anchor_metric(holdout_row, metric_name),
                        "base_prediction": float(base_value),
                        "selected_prediction": float(selected_value),
                        "carry_forward_prediction": float(carry_value),
                        "target_value": float(target_value),
                        "scale": float(_metric_scale(internal_train, metric_name)),
                    }
                )
    if not records:
        return {
            "status": "not_estimable",
            "selected_blend_weight": 0.0,
            "reason": "no_train_origin_blend_records",
        }
    alpha_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    score_tolerance = float(np.finfo(np.float32).eps)
    annual_records = [record for record in records if bool(record.get("annual_anchor_metric"))]
    for alpha in _r12_alpha_candidates(records):
        full_errors = [
            _r12_convex_error(
                float(record["base_prediction"]),
                float(record["selected_prediction"]),
                alpha,
                float(record["target_value"]),
                float(record["scale"]),
            )
            for record in records
        ]
        annual_errors = [
            _r12_convex_error(
                float(record["base_prediction"]),
                float(record["selected_prediction"]),
                alpha,
                float(record["target_value"]),
                float(record["scale"]),
            )
            for record in annual_records
        ]
        base_errors = [
            _r12_convex_error(
                float(record["base_prediction"]),
                float(record["selected_prediction"]),
                0.0,
                float(record["target_value"]),
                float(record["scale"]),
            )
            for record in records
        ]
        carry_errors = [
            abs(float(record["carry_forward_prediction"]) - float(record["target_value"]))
            / max(float(record["scale"]), float(np.finfo(np.float32).eps))
            for record in records
        ]
        full_mean = float(np.mean(np.asarray(full_errors, dtype=np.float64)))
        full_worst = float(np.max(np.asarray(full_errors, dtype=np.float64)))
        annual_mean = None if not annual_errors else float(np.mean(np.asarray(annual_errors, dtype=np.float64)))
        base_mean = float(np.mean(np.asarray(base_errors, dtype=np.float64)))
        base_worst = float(np.max(np.asarray(base_errors, dtype=np.float64)))
        carry_mean = float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
        carry_worst = float(np.max(np.asarray(carry_errors, dtype=np.float64)))
        eligible = (
            annual_mean is not None
            and full_mean <= base_mean + score_tolerance
            and full_worst <= base_worst + score_tolerance
            and full_mean <= carry_mean + score_tolerance
            and full_worst <= carry_worst + score_tolerance
        )
        row = {
            "blend_weight": float(alpha),
            "annual_anchor_mean_norm_error": annual_mean,
            "full_stock_mean_norm_error": full_mean,
            "full_stock_worst_norm_error": full_worst,
            "base_full_stock_mean_norm_error": base_mean,
            "base_full_stock_worst_norm_error": base_worst,
            "carry_forward_full_stock_mean_norm_error": carry_mean,
            "carry_forward_full_stock_worst_norm_error": carry_worst,
            "eligible": eligible,
        }
        alpha_rows.append(row)
        if eligible and (
            best_row is None
            or float(row["annual_anchor_mean_norm_error"]) < float(best_row["annual_anchor_mean_norm_error"])
        ):
            best_row = row
    selected_alpha = 0.0 if best_row is None else float(best_row["blend_weight"])
    return {
        "status": "completed" if best_row is not None else "failed_closed",
        "selected_family": selected_family,
        "selected_blend_weight": selected_alpha,
        "record_count": len(records),
        "annual_anchor_record_count": len(annual_records),
        "selected_row": best_row,
        "alpha_rows": alpha_rows,
        "contract": (
            "One-dimensional exact convex selection between R11-28 and the annual-anchor head. Candidate alpha values "
            "are generated from absolute-error breakpoints, not a hand grid. The selected alpha minimizes annual-anchor "
            "D/A train-origin error subject to full D/A stock mean and worst error being no worse than R11-28 and "
            "carry-forward."
        ),
    }


def _r12_stock_cone_safe_annual_trajectory_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="multi_horizon_weighted_process",
    )
    selector = _fit_r12_annual_anchor_head_selector(train_rows)
    selected_family = str(selector.get("selected_family") or "multi_horizon_weighted_process")
    selected_blend_weight = _finite_float(selector.get("selected_blend_weight"))
    if selected_blend_weight is None:
        selected_blend_weight = 0.0
    selected_predictions: list[dict[str, Any]] = []
    selected_summary: dict[str, Any] = {}
    if selected_family != "multi_horizon_weighted_process":
        selected_predictions, selected_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family=selected_family,
        )
    selected_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in selected_predictions}
    back_half_process = _fit_back_half_rate_process(train_rows)
    output: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    for base_prediction, holdout_row in zip(
        base_predictions,
        sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
    ):
        quarter = str(holdout_row.get("quarter") or base_prediction.get("quarter") or "")
        row = dict(base_prediction)
        mutated_metrics: list[str] = []
        if selected_family != "multi_horizon_weighted_process" and _r12_is_annual_anchor_stock_row(holdout_row):
            selected_row = selected_by_quarter.get(quarter, {})
            for metric_name in R12_LONG_HORIZON_STOCK_METRICS:
                if not _r12_is_annual_anchor_metric(holdout_row, metric_name):
                    continue
                value = _finite_float(selected_row.get(metric_name))
                if value is None:
                    continue
                base_value = _finite_float(row.get(metric_name))
                if base_value is None:
                    continue
                row[metric_name] = float(
                    max(
                        (1.0 - float(selected_blend_weight)) * float(base_value)
                        + float(selected_blend_weight) * float(value),
                        0.0,
                    )
                )
                mutated_metrics.append(metric_name)
        row = _project_prediction_row(row)
        row = _apply_back_half_rate_process(row, holdout_row, back_half_process)
        output.append(row)
        mutation_rows.append(
            {
                "quarter": quarter,
                "annual_anchor_stock_row": _r12_is_annual_anchor_stock_row(holdout_row),
                "selected_family": selected_family,
                "selected_blend_weight": float(selected_blend_weight),
                "mutated_metrics": mutated_metrics,
            }
        )
    return output, {
        "base_family": "multi_horizon_weighted_process",
        "base_summary": base_summary,
        "annual_anchor_selector": selector,
        "annual_anchor_selected_family": selected_family,
        "annual_anchor_selected_blend_weight": float(selected_blend_weight),
        "annual_anchor_selected_summary": selected_summary,
        "back_half_rate_process": back_half_process,
        "mutation_rows": mutation_rows,
        "contract": (
            "R12-09 keeps the locked R11-28 backbone for all non-annual evidence and applies the selected annual-anchor "
            "stock head only to diagnosed_plhiv/alive_on_art rows whose provenance is the slide annual-anchor lineage. "
            "The row is then projected through the cascade cone and VL/suppression are regenerated from the dedicated "
            "conditional-rate process. Program nowcasting rows are not modified."
        ),
    }


def _write_r12_official_annual_challenge_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metric_rows = [
        dict(row)
        for row in list(report.get("metric_rows") or [])
        if isinstance(row, dict)
    ]
    family_rows = [
        dict(row)
        for row in list(report.get("family_rows") or [])
        if isinstance(row, dict)
    ]
    manifests = [
        dict(row)
        for row in list(report.get("family_manifests") or [])
        if isinstance(row, dict)
    ]
    if not metric_rows and not family_rows:
        return
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.2), constrained_layout=True)
    families = [str(row.get("candidate_family") or "") for row in family_rows]
    candidate_values = [
        np.nan if _finite_float(row.get("candidate_mean_norm_error")) is None else float(row["candidate_mean_norm_error"])
        for row in family_rows
    ]
    carry_values = [
        np.nan if _finite_float(row.get("carry_forward_mean_norm_error")) is None else float(row["carry_forward_mean_norm_error"])
        for row in family_rows
    ]
    x = np.arange(len(families), dtype=np.float64)
    if len(families):
        width = 0.38
        axes[0].bar(x - width / 2.0, candidate_values, width=width, color="#2F6B59", label="candidate")
        axes[0].bar(x + width / 2.0, carry_values, width=width, color="#B56B45", label="carry-forward")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([name.replace("_", "\n") for name in families], fontsize=7)
        axes[0].set_ylabel("Mean normalized annual challenge error")
        axes[0].set_title("Official annual challenge")
        axes[0].legend(frameon=False, fontsize=8)
    annual_required = set(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS)
    required_rows = [row for row in metric_rows if str(row.get("metric_name") or "") in annual_required]
    cascade_rows = [row for row in metric_rows if str(row.get("metric_name") or "") not in annual_required]
    required_missing = sum(int(row.get("missing_prediction_count") or 0) for row in required_rows)
    cascade_missing = sum(int(row.get("missing_prediction_count") or 0) for row in cascade_rows)
    required_scored = sum(int(row.get("scored_candidate_entry_count") or 0) for row in required_rows)
    cascade_scored = sum(int(row.get("scored_candidate_entry_count") or 0) for row in cascade_rows)
    axes[1].bar(
        ["annual heads\nscored", "annual heads\nmissing", "cascade\nscored", "cascade\nmissing"],
        [required_scored, required_missing, cascade_scored, cascade_missing],
        color=["#4267A8", "#C44E52", "#55A868", "#C44E52"],
    )
    axes[1].set_title("Challenge support contract")
    axes[1].set_ylabel("Metric-holdout entries")
    axes[1].tick_params(axis="x", labelsize=8)
    blocker_counts = [
        len(list(row.get("blockers") or []))
        for row in manifests
    ]
    manifest_families = [str(row.get("candidate_family") or "").replace("_", "\n") for row in manifests]
    if manifest_families:
        axes[2].bar(np.arange(len(manifest_families)), blocker_counts, color="#8C6BB1")
        axes[2].set_xticks(np.arange(len(manifest_families)))
        axes[2].set_xticklabels(manifest_families, fontsize=7)
    axes[2].set_title(f"Gate status: {report.get('status')}")
    axes[2].set_ylabel("Blocker count")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    fig.suptitle(
        "R12-10A annual challenge gate: validation-only annual heads are scored, not trained",
        fontsize=12,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _r12_metric_matches_lineage_ids(row: dict[str, Any], metric_name: str, lineage_ids: tuple[str, ...]) -> bool:
    for lineage_id in lineage_ids:
        lineage = _r12_lineage_by_id(str(lineage_id))
        source_family = str(lineage.get("source_family") or "")
        if source_family and _r12_metric_matches_source_family(row, metric_name, source_family):
            return True
    return False


def _r12_is_program_row(row: dict[str, Any]) -> bool:
    return any(
        _r12_metric_matches_lineage_ids(row, metric_name, R12_PROGRAM_LINEAGE_IDS)
        for metric_name in R12_PROGRAM_MUTATION_METRICS
    )


def _r12_program_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _r12_is_program_row(row)
    ]


def _fit_r12_program_nowcast_selector(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    program_train_rows = _r12_program_rows(train_rows)
    if not program_train_rows:
        return {
            "status": "not_estimable",
            "selected_family": "multi_horizon_weighted_process",
            "reason": "no_train_program_rows",
            "candidate_families": list(R12_PROGRAM_NOWCAST_CANDIDATE_FAMILIES),
            "metric_scope": list(R12_PROGRAM_MUTATION_METRICS),
        }
    selector = _train_origin_family_scores(
        program_train_rows,
        candidate_families=R12_PROGRAM_NOWCAST_CANDIDATE_FAMILIES,
        max_horizon_years=_holdout_max_horizon_years(program_train_rows, holdout_rows),
        metrics=R12_PROGRAM_MUTATION_METRICS,
    )
    rows_by_family = {
        str(row.get("family") or ""): dict(row)
        for row in list(selector.get("family_rows") or [])
        if isinstance(row, dict)
    }
    base_row = rows_by_family.get("multi_horizon_weighted_process", {})
    selected_family = str(selector.get("selected_family") or "multi_horizon_weighted_process")
    selected_row = rows_by_family.get(selected_family, {})
    base_scoped = _finite_float(base_row.get("scoped_mean_mae"))
    base_stock = _finite_float(base_row.get("primary_stock_mean_mae"))
    selected_scoped = _finite_float(selected_row.get("scoped_mean_mae"))
    selected_stock = _finite_float(selected_row.get("primary_stock_mean_mae"))
    fail_closed_reason = ""
    if selected_family != "multi_horizon_weighted_process":
        if selected_scoped is None or base_scoped is None:
            fail_closed_reason = "missing_program_scope_selector_score"
        elif selected_scoped > base_scoped + FLOAT_NONREGRESSION_TOLERANCE:
            fail_closed_reason = "program_scope_worse_than_r11_28"
        elif selected_stock is None or base_stock is None:
            fail_closed_reason = "missing_primary_stock_selector_score"
        elif selected_stock > base_stock + FLOAT_NONREGRESSION_TOLERANCE:
            fail_closed_reason = "primary_stock_worse_than_r11_28"
    if fail_closed_reason:
        selected_family = "multi_horizon_weighted_process"
    output = dict(selector)
    output["status"] = str(selector.get("status") or "not_estimable")
    output["selected_family"] = selected_family
    output["base_family"] = "multi_horizon_weighted_process"
    output["program_train_row_count"] = len(program_train_rows)
    output["program_holdout_row_count"] = len(_r12_program_rows(holdout_rows))
    output["fail_closed_reason"] = fail_closed_reason
    output["contract"] = (
        "R12-10 selects a program-specific diagnosis/ART/diagnosis-flow family using DOH quarterly/monthly "
        "train-origin rows only. A non-reference family must improve the program R10-scope score without worsening "
        "primary D/A stock score versus R11-28; otherwise the selector fails closed to R11-28."
    )
    return output


def _r12_program_nowcast_mixed_quarterly_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, base_summary = _candidate_predictions(
        train_rows,
        holdout_rows,
        family="r12_stock_cone_safe_annual_trajectory_process",
    )
    selector = _fit_r12_program_nowcast_selector(train_rows, holdout_rows)
    selected_family = str(selector.get("selected_family") or "multi_horizon_weighted_process")
    program_train_rows = _r12_program_rows(train_rows)
    program_holdout_rows = _r12_program_rows(holdout_rows)
    selected_predictions: list[dict[str, Any]] = []
    selected_summary: dict[str, Any] = {}
    if program_train_rows and program_holdout_rows:
        selected_predictions, selected_summary = _candidate_predictions(
            program_train_rows,
            program_holdout_rows,
            family=selected_family,
        )
    selected_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in selected_predictions}
    back_half_process = _fit_back_half_rate_process(train_rows)
    output: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    for base_prediction, holdout_row in zip(
        base_predictions,
        sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
    ):
        quarter = str(holdout_row.get("quarter") or base_prediction.get("quarter") or "")
        row = dict(base_prediction)
        selected_row = selected_by_quarter.get(quarter, {})
        mutated_metrics: list[str] = []
        for metric_name in R12_PROGRAM_MUTATION_METRICS:
            if not _r12_metric_matches_lineage_ids(holdout_row, metric_name, R12_PROGRAM_LINEAGE_IDS):
                continue
            selected_value = _finite_float(selected_row.get(metric_name))
            if selected_value is None:
                continue
            row[metric_name] = float(max(float(selected_value), 0.0))
            mutated_metrics.append(metric_name)
        row = _project_prediction_row(row)
        row = _apply_back_half_rate_process(row, holdout_row, back_half_process)
        output.append(row)
        mutation_rows.append(
            {
                "quarter": quarter,
                "program_row": _r12_is_program_row(holdout_row),
                "selected_family": selected_family,
                "mutated_metrics": mutated_metrics,
            }
        )
    return output, {
        "base_family": "r12_stock_cone_safe_annual_trajectory_process",
        "base_summary": base_summary,
        "program_selector": selector,
        "program_selected_family": selected_family,
        "program_selected_summary": selected_summary,
        "program_train_row_count": len(program_train_rows),
        "program_holdout_row_count": len(program_holdout_rows),
        "back_half_rate_process": back_half_process,
        "mutation_rows": mutation_rows,
        "contract": (
            "R12-10 starts from the frozen R12-09 annual-anchor behavior and does not add any new annual-anchor mutation. "
            "It overlays a train-selected "
            "program-specific diagnosis/ART/diagnosis-flow head only on DOH quarterly/monthly rows, then re-projects "
            "the cascade cone and regenerates VL/suppression from conditional rates."
        ),
    }


def _diagnosis_flow_adjusted_base_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, base_summary = _candidate_predictions(train_rows, holdout_rows, family="r10_style_readout_teacher")
    sorted_holdout = sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    output = [dict(row) for row in base_predictions]
    support_flow_model = _fit_support_partition_calibration_model(train_rows, "new_diagnosed_cases_period")
    diagnosed_model = _fit_support_partition_calibration_model(train_rows, "diagnosed_plhiv")
    transition_process = _fit_era_datv_transition_process(train_rows)
    diagnosed_transition = dict(transition_process.get("diagnosed_stock_transition") or {})
    current_diagnosed = _last_metric_value(train_rows, "diagnosed_plhiv")
    flow_by_ordinal: dict[int, float] = {}
    for row in sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        value = _finite_float(row.get("new_diagnosed_cases_period"))
        if value is not None:
            flow_by_ordinal[quarter_ordinal(str(row.get("quarter") or ""))] = float(value)
    lag_model = _fit_linkage_lag_kernel(train_rows)
    lag_quarters = int(lag_model.get("selected_lag_quarters") or 0) if str(lag_model.get("status") or "") == "completed" else 0
    for index, holdout_row in enumerate(sorted_holdout):
        predicted = output[index]
        flow = _finite_float(predicted.get("new_diagnosed_cases_period"))
        if variant in {"support_era_flow", "support_era_flow_process"}:
            support_flow = _predict_support_partition_calibration(support_flow_model, holdout_row, "new_diagnosed_cases_period")
            if support_flow is not None:
                flow = support_flow
        elif variant in {"stock_reconciled_flow", "stock_flow_reconciliation"}:
            target_diagnosed = _predict_support_partition_calibration(diagnosed_model, holdout_row, "diagnosed_plhiv")
            gamma = _finite_float(diagnosed_transition.get("diagnosis_flow_coefficient"))
            mu = _finite_float(diagnosed_transition.get("diagnosed_removal_fraction"))
            if current_diagnosed is not None and target_diagnosed is not None and gamma is not None and gamma > 0.0 and mu is not None:
                shift = _era_reporting_shift(diagnosed_transition, holdout_row)
                flow = float(max((float(target_diagnosed) - float(current_diagnosed) + float(mu) * float(current_diagnosed) - shift) / float(gamma), 0.0))
        elif variant == "lagged_diagnosis_flow":
            holdout_ord = quarter_ordinal(str(holdout_row.get("quarter") or ""))
            lagged = flow_by_ordinal.get(holdout_ord - lag_quarters)
            if lagged is not None:
                flow = float(lagged)
        if flow is not None:
            predicted["new_diagnosed_cases_period"] = float(max(float(flow), 0.0))
            flow_by_ordinal[quarter_ordinal(str(holdout_row.get("quarter") or ""))] = float(predicted["new_diagnosed_cases_period"])
        current_diagnosed = _finite_float(predicted.get("diagnosed_plhiv")) or current_diagnosed
    return output, {
        "variant": variant,
        "base_summary": base_summary,
        "support_flow_model_status": str(support_flow_model.get("status") or ""),
        "diagnosed_model_status": str(diagnosed_model.get("status") or ""),
        "transition_process_status": str(transition_process.get("status") or ""),
        "lag_quarters": lag_quarters,
        "contract": "diagnosis-flow inputs are transformed by train-fitted support, stock-flow, or lag structure before the transition process is applied",
    }


def _era_process_with_flow_variant_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_predictions, flow_summary = _diagnosis_flow_adjusted_base_predictions(train_rows, holdout_rows, variant=variant)
    process = _fit_era_datv_transition_process(train_rows)
    back_half_process = _fit_back_half_rate_process(train_rows)
    predictions = _apply_era_datv_transition_process(train_rows, holdout_rows, base_predictions, process, back_half_process)
    return predictions, {
        "flow_adjustment": flow_summary,
        "transition_process": process,
        "back_half_rate_process": back_half_process,
    }


def _fit_transition_shrinkage_model(
    train_rows: list[dict[str, Any]],
    metric_name: str,
    *,
    pooled_slopes: list[float],
) -> dict[str, Any]:
    metric_rows = _available_metric_rows(train_rows, metric_name)
    if not metric_rows:
        return {
            "status": "not_estimable",
            "metric_name": metric_name,
            "reason": "no_train_values",
        }
    metric_slopes = _metric_log_slopes(metric_rows, metric_name)
    if not metric_slopes:
        metric_slopes = [0.0]
    if not pooled_slopes:
        pooled_slopes = list(metric_slopes)
    metric_array = np.asarray(metric_slopes, dtype=np.float64)
    pooled_array = np.asarray(pooled_slopes, dtype=np.float64)
    eps = float(np.finfo(np.float32).eps)
    metric_var = float(np.var(metric_array)) if metric_array.size > 1 else eps
    pooled_var = float(np.var(pooled_array)) if pooled_array.size > 1 else eps
    metric_precision = float(metric_array.size) / max(metric_var, eps)
    pooled_precision = float(pooled_array.size) / max(pooled_var, eps)
    metric_mean = float(np.mean(metric_array))
    pooled_mean = float(np.mean(pooled_array))
    shrunk_slope = (metric_precision * metric_mean + pooled_precision * pooled_mean) / max(
        metric_precision + pooled_precision,
        eps,
    )
    return {
        "status": "completed",
        "metric_name": metric_name,
        "last_quarter": str(metric_rows[-1].get("quarter") or ""),
        "last_log_value": float(np.log1p(max(float(metric_rows[-1].get(metric_name) or 0.0), 0.0))),
        "metric_mean_slope": metric_mean,
        "pooled_mean_slope": pooled_mean,
        "shrunk_slope": float(shrunk_slope),
        "metric_slope_count": int(metric_array.size),
        "pooled_slope_count": int(pooled_array.size),
        "metric_precision": metric_precision,
        "pooled_precision": pooled_precision,
        "contract": "empirical-Bayes precision-weighted shrinkage of metric log slopes toward pooled cascade log slopes; no tunable prior weight",
    }


def _predict_transition_shrinkage(model: dict[str, Any], row: dict[str, Any]) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    last_quarter = str(model.get("last_quarter") or "")
    if not last_quarter:
        return None
    step = max(quarter_ordinal(str(row.get("quarter") or "")) - quarter_ordinal(last_quarter), 0)
    log_value = float(model.get("last_log_value") or 0.0) + float(model.get("shrunk_slope") or 0.0) * float(step)
    return float(max(np.expm1(log_value), 0.0))


def _fit_support_partition_calibration_model(train_rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    metric_rows = _available_metric_rows(train_rows, metric_name)
    if not metric_rows:
        return {
            "status": "not_estimable",
            "metric_name": metric_name,
            "reason": "no_train_values",
        }
    residuals_by_partition: dict[str, list[float]] = defaultdict(list)
    residuals: list[float] = []
    for previous, current in zip(metric_rows[:-1], metric_rows[1:]):
        previous_value = max(float(previous.get(metric_name) or 0.0), 0.0)
        current_value = max(float(current.get(metric_name) or 0.0), 0.0)
        residual = float(np.log1p(current_value) - np.log1p(previous_value))
        partition = str(_metric_provenance(current, metric_name).get("support_partition") or "unknown")
        residuals_by_partition[partition].append(residual)
        residuals.append(residual)
    global_residual = float(np.median(np.asarray(residuals, dtype=np.float64))) if residuals else 0.0
    return {
        "status": "completed",
        "metric_name": metric_name,
        "last_value": float(max(float(metric_rows[-1].get(metric_name) or 0.0), 0.0)),
        "global_log_residual": global_residual,
        "log_residual_by_support_partition": {
            partition: float(np.median(np.asarray(values, dtype=np.float64)))
            for partition, values in sorted(residuals_by_partition.items())
        },
        "residual_count": len(residuals),
        "contract": "carry-forward prior plus train-window median log residual by support partition; unseen partitions use train global residual",
    }


def _predict_support_partition_calibration(model: dict[str, Any], row: dict[str, Any], metric_name: str) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    partition = str(_metric_provenance(row, metric_name).get("support_partition") or "unknown")
    residuals = dict(model.get("log_residual_by_support_partition") or {})
    residual = float(residuals.get(partition, model.get("global_log_residual") or 0.0))
    log_value = float(np.log1p(max(float(model.get("last_value") or 0.0), 0.0))) + residual
    return float(max(np.expm1(log_value), 0.0))


def _train_row_by_ordinal(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        quarter_ordinal(str(row.get("quarter") or "")): dict(row)
        for row in rows
        if row.get("quarter")
    }


def _fit_linkage_lag_kernel(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    by_ordinal = _train_row_by_ordinal(sorted_rows)
    candidate_lags = (0, 1, 2, 4)
    lag_rows: list[dict[str, Any]] = []
    for lag in candidate_lags:
        ratios: list[float] = []
        errors: list[float] = []
        examples: list[tuple[float, float, float]] = []
        for previous, current in zip(sorted_rows[:-1], sorted_rows[1:]):
            current_ord = quarter_ordinal(str(current.get("quarter") or ""))
            lag_row = current if lag == 0 else by_ordinal.get(current_ord - lag)
            if lag_row is None:
                continue
            flow = _finite_float(lag_row.get("new_diagnosed_cases_period"))
            previous_art = _finite_float(previous.get("alive_on_art"))
            current_art = _finite_float(current.get("alive_on_art"))
            if flow is None or previous_art is None or current_art is None or flow <= 0.0:
                continue
            delta = current_art - previous_art
            ratios.append(float(delta / flow))
            examples.append((float(previous_art), float(current_art), float(flow)))
        coefficient = float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else 0.0
        for previous_art, current_art, flow in examples:
            predicted = previous_art + coefficient * max(flow, 0.0)
            errors.append(abs(predicted - current_art))
        lag_rows.append(
            {
                "lag_quarters": int(lag),
                "coefficient": coefficient,
                "train_pair_count": len(ratios),
                "train_abs_error": float(np.mean(np.asarray(errors, dtype=np.float64))) if errors else float("inf"),
            }
        )
    evaluable = [row for row in lag_rows if int(row["train_pair_count"]) > 0 and np.isfinite(float(row["train_abs_error"]))]
    if not evaluable:
        return {
            "status": "not_estimable",
            "reason": "no_lagged_diagnosis_flow_art_pairs",
            "candidate_lags": list(candidate_lags),
        }
    best = min(evaluable, key=lambda row: (float(row["train_abs_error"]), int(row["lag_quarters"])))
    return {
        "status": "completed",
        "selected_lag_quarters": int(best["lag_quarters"]),
        "selected_coefficient": float(best["coefficient"]),
        "lag_rows": lag_rows,
        "last_alive_on_art": None
        if not sorted_rows
        else _finite_float(sorted_rows[-1].get("alive_on_art")),
        "last_diagnosis_flow": None
        if not sorted_rows
        else _finite_float(sorted_rows[-1].get("new_diagnosed_cases_period")),
        "contract": "D_to_A lag kernel selected from 0, 1, 2, and 4 quarter delays using train ART increment error only",
    }


def _linkage_lag_prediction_value(
    model: dict[str, Any],
    train_rows: list[dict[str, Any]],
    holdout_row: dict[str, Any],
) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    previous_art = _finite_float(model.get("last_alive_on_art"))
    if previous_art is None:
        return None
    lag = int(model.get("selected_lag_quarters") or 0)
    coefficient = float(model.get("selected_coefficient") or 0.0)
    by_ordinal = _train_row_by_ordinal(train_rows)
    holdout_ord = quarter_ordinal(str(holdout_row.get("quarter") or ""))
    lag_row = holdout_row if lag == 0 else by_ordinal.get(holdout_ord - lag)
    flow = _finite_float((lag_row or {}).get("new_diagnosed_cases_period"))
    if flow is None:
        flow = _finite_float(model.get("last_diagnosis_flow"))
    if flow is None:
        return previous_art
    return float(max(previous_art + coefficient * max(flow, 0.0), 0.0))


def _bounded_rate(value: float | None) -> float | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return float(min(max(float(value), 0.0), 1.0))


def _conditional_rate(row: dict[str, Any], numerator_metric: str, denominator_metric: str) -> float | None:
    numerator = _finite_float(row.get(numerator_metric))
    denominator = _finite_float(row.get(denominator_metric))
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return _bounded_rate(float(numerator) / float(denominator))


def _rate_support_partition(row: dict[str, Any], numerator_metric: str, denominator_metric: str) -> str:
    numerator_provenance = _metric_provenance(row, numerator_metric)
    denominator_provenance = _metric_provenance(row, denominator_metric)
    return str(
        numerator_provenance.get("support_partition")
        or denominator_provenance.get("support_partition")
        or "unknown"
    )


def _fit_back_half_rate_model(
    train_rows: list[dict[str, Any]],
    *,
    rate_id: str,
    numerator_metric: str,
    denominator_metric: str,
) -> dict[str, Any]:
    rate_rows: list[tuple[dict[str, Any], float]] = []
    for row in sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        rate = _conditional_rate(row, numerator_metric, denominator_metric)
        if rate is not None:
            rate_rows.append((dict(row), float(rate)))
    if not rate_rows:
        return {
            "status": "not_estimable",
            "rate_id": rate_id,
            "reason": "no_train_conditional_rates",
        }
    deltas: list[float] = []
    deltas_by_partition: dict[str, list[float]] = defaultdict(list)
    for (previous_row, previous_rate), (current_row, current_rate) in zip(rate_rows[:-1], rate_rows[1:]):
        step = quarter_ordinal(str(current_row.get("quarter") or "")) - quarter_ordinal(str(previous_row.get("quarter") or ""))
        if step <= 0:
            continue
        delta = float((current_rate - previous_rate) / float(step))
        partition = _rate_support_partition(current_row, numerator_metric, denominator_metric)
        deltas.append(delta)
        deltas_by_partition[partition].append(delta)
    global_delta = float(np.median(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0
    return {
        "status": "completed",
        "rate_id": rate_id,
        "numerator_metric": numerator_metric,
        "denominator_metric": denominator_metric,
        "last_quarter": str(rate_rows[-1][0].get("quarter") or ""),
        "last_rate": float(rate_rows[-1][1]),
        "global_quarterly_rate_delta": global_delta,
        "quarterly_rate_delta_by_support_partition": {
            partition: float(np.median(np.asarray(values, dtype=np.float64)))
            for partition, values in sorted(deltas_by_partition.items())
        },
        "rate_count": len(rate_rows),
        "delta_count": len(deltas),
        "contract": "conditional back-half rate model over numerator/denominator stocks; only train-window empirical rate deltas are used",
    }


def _predict_back_half_rate(
    model: dict[str, Any],
    holdout_row: dict[str, Any],
    *,
    variant: str,
) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    rate = _bounded_rate(_finite_float(model.get("last_rate")))
    if rate is None:
        return None
    if variant == "carry_rate":
        return rate
    last_quarter = str(model.get("last_quarter") or "")
    step = max(quarter_ordinal(str(holdout_row.get("quarter") or "")) - quarter_ordinal(last_quarter), 0)
    if variant == "support_partition_delta":
        numerator_metric = str(model.get("numerator_metric") or "")
        denominator_metric = str(model.get("denominator_metric") or "")
        partition = _rate_support_partition(holdout_row, numerator_metric, denominator_metric)
        deltas = dict(model.get("quarterly_rate_delta_by_support_partition") or {})
        delta = float(deltas.get(partition, model.get("global_quarterly_rate_delta") or 0.0))
    elif variant == "median_delta":
        delta = float(model.get("global_quarterly_rate_delta") or 0.0)
    else:
        raise ValueError(f"Unknown back-half rate variant: {variant}")
    return _bounded_rate(rate + delta * float(step))


def _rate_prediction_errors(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    rate_id: str,
    numerator_metric: str,
    denominator_metric: str,
    variant: str,
) -> dict[str, Any]:
    model = _fit_back_half_rate_model(
        train_rows,
        rate_id=rate_id,
        numerator_metric=numerator_metric,
        denominator_metric=denominator_metric,
    )
    errors: list[float] = []
    for holdout_row in holdout_rows:
        target_rate = _conditional_rate(holdout_row, numerator_metric, denominator_metric)
        predicted_rate = _predict_back_half_rate(model, holdout_row, variant=variant)
        if target_rate is None or predicted_rate is None:
            continue
        errors.append(abs(float(predicted_rate) - float(target_rate)))
    return {
        "entry_count": len(errors),
        "mean_rate_error": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
        "worst_rate_error": None if not errors else float(np.max(np.asarray(errors, dtype=np.float64))),
    }


def _select_back_half_rate_variant(
    train_rows: list[dict[str, Any]],
    *,
    rate_id: str,
    numerator_metric: str,
    denominator_metric: str,
) -> dict[str, Any]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    variants = ("carry_rate", "median_delta", "support_partition_delta")
    variant_scores: dict[str, list[float]] = {variant: [] for variant in variants}
    variant_worsts: dict[str, list[float]] = {variant: [] for variant in variants}
    for holdout_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) < int(holdout_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) == int(holdout_year)
        ]
        if not internal_train or not internal_holdout:
            continue
        for variant in variants:
            score = _rate_prediction_errors(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                rate_id=rate_id,
                numerator_metric=numerator_metric,
                denominator_metric=denominator_metric,
                variant=variant,
            )
            mean_error = _finite_float(score.get("mean_rate_error"))
            worst_error = _finite_float(score.get("worst_rate_error"))
            if mean_error is not None:
                variant_scores[variant].append(mean_error)
            if worst_error is not None:
                variant_worsts[variant].append(worst_error)
    if not variant_scores["carry_rate"]:
        return {
            "selected_variant": "carry_rate",
            "status": "fallback",
            "reason": "no_internal_rate_backtest",
            "variant_rows": [],
        }
    carry_mean = float(np.mean(np.asarray(variant_scores["carry_rate"], dtype=np.float64)))
    carry_worst = float(np.max(np.asarray(variant_worsts["carry_rate"], dtype=np.float64))) if variant_worsts["carry_rate"] else float("inf")
    selected_variant = "carry_rate"
    selected_mean = carry_mean
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        values = variant_scores[variant]
        worst_values = variant_worsts[variant]
        if not values or not worst_values:
            variant_rows.append(
                {
                    "variant": variant,
                    "status": "not_evaluable",
                    "mean_rate_error": None,
                    "worst_rate_error": None,
                }
            )
            continue
        mean_error = float(np.mean(np.asarray(values, dtype=np.float64)))
        worst_error = float(np.max(np.asarray(worst_values, dtype=np.float64)))
        variant_rows.append(
            {
                "variant": variant,
                "status": "evaluable",
                "mean_rate_error": mean_error,
                "worst_rate_error": worst_error,
                "mean_minus_carry_rate_error": float(mean_error - carry_mean),
                "worst_minus_carry_rate_error": float(worst_error - carry_worst),
            }
        )
        if variant != "carry_rate" and mean_error < selected_mean and worst_error <= carry_worst:
            selected_variant = variant
            selected_mean = mean_error
    return {
        "selected_variant": selected_variant,
        "status": "completed",
        "carry_mean_rate_error": carry_mean,
        "carry_worst_rate_error": carry_worst,
        "variant_rows": variant_rows,
        "contract": "variant selected by internal train-origin rate backtest; switch away from carry-rate only if mean improves and worst case does not regress",
    }


def _fit_back_half_rate_process(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rate_models: dict[str, dict[str, Any]] = {}
    for spec in BACK_HALF_RATE_SPECS:
        rate_id = str(spec["rate_id"])
        selector = _select_back_half_rate_variant(
            train_rows,
            rate_id=rate_id,
            numerator_metric=str(spec["numerator_metric"]),
            denominator_metric=str(spec["denominator_metric"]),
        )
        model = _fit_back_half_rate_model(
            train_rows,
            rate_id=rate_id,
            numerator_metric=str(spec["numerator_metric"]),
            denominator_metric=str(spec["denominator_metric"]),
        )
        rate_models[rate_id] = {
            **model,
            "selected_variant": str(selector.get("selected_variant") or "carry_rate"),
            "selector": selector,
        }
    return {
        "status": "completed" if any(str(model.get("status") or "") == "completed" for model in rate_models.values()) else "not_estimable",
        "rate_models": rate_models,
        "contract": "dedicated back-half process over VL testing among ART and suppression among VL-tested, fitted only from train-window conditional rates",
    }


def _apply_back_half_rate_process(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    process: dict[str, Any],
) -> dict[str, Any]:
    prediction = dict(base_prediction)
    models = dict(process.get("rate_models") or {})
    vl_model = dict(models.get("vl_tested_among_art") or {})
    vl_variant = str(vl_model.get("selected_variant") or "carry_rate")
    art_value = _finite_float(prediction.get("alive_on_art"))
    vl_rate = _predict_back_half_rate(vl_model, holdout_row, variant=vl_variant)
    if art_value is not None and vl_rate is not None:
        prediction["tested_for_viral_load"] = float(max(float(art_value), 0.0) * float(vl_rate))
    suppression_model = dict(models.get("suppressed_among_vl_tested") or {})
    suppression_variant = str(suppression_model.get("selected_variant") or "carry_rate")
    vl_value = _finite_float(prediction.get("tested_for_viral_load"))
    suppression_rate = _predict_back_half_rate(suppression_model, holdout_row, variant=suppression_variant)
    if vl_value is not None and suppression_rate is not None:
        prediction["virally_suppressed"] = float(max(float(vl_value), 0.0) * float(suppression_rate))
    return _project_prediction_row(prediction)


def _shape_record_key(metric_name: str, lead_years: int) -> str:
    return f"{metric_name}|lead{int(lead_years)}"


def _shape_record_metric(key: str) -> str:
    return str(key).split("|lead", 1)[0]


def _shape_record_lead(key: str) -> int:
    try:
        return int(str(key).rsplit("lead", 1)[1])
    except (IndexError, ValueError):
        return 0


def _trajectory_shape_error_records(
    train_rows: list[dict[str, Any]],
    *,
    max_horizon_years: int,
) -> list[dict[str, Any]]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    if len(years) < 3:
        return []
    records: list[dict[str, Any]] = []
    for train_end_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if int(train_end_year) < quarter_year(str(row.get("quarter") or "")) <= int(train_end_year) + int(max_horizon_years)
        ]
        if not internal_train or not internal_holdout:
            continue
        base_predictions, _summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family="back_half_conditional_rates",
        )
        prediction_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions}
        for holdout_row in internal_holdout:
            quarter = str(holdout_row.get("quarter") or "")
            lead_years = max(quarter_year(quarter) - int(train_end_year), 1)
            prediction = prediction_by_quarter.get(quarter, {})
            for metric_name in R11_EVALUATION_METRICS:
                target_value = _finite_float(holdout_row.get(metric_name))
                predicted_value = _finite_float(prediction.get(metric_name))
                if target_value is None or predicted_value is None:
                    continue
                scale = _metric_scale(internal_train, metric_name)
                records.append(
                    {
                        "train_end_year": int(train_end_year),
                        "quarter": quarter,
                        "metric_name": metric_name,
                        "lead_years": int(lead_years),
                        "target_value": float(target_value),
                        "base_prediction_value": float(max(predicted_value, 0.0)),
                        "base_norm_error": abs(float(predicted_value) - float(target_value)) / max(scale, float(np.finfo(np.float32).eps)),
                        "log_residual": float(np.log1p(max(float(target_value), 0.0)) - np.log1p(max(float(predicted_value), 0.0))),
                        "scale": scale,
                    }
                )
    return sorted(records, key=lambda row: (int(row["train_end_year"]), str(row["quarter"]), str(row["metric_name"])))


def _trajectory_shape_walk_forward_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residuals_by_key: dict[str, list[float]] = defaultdict(list)
    residuals_by_metric: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for record in records:
        metric_name = str(record.get("metric_name") or "")
        lead_years = int(record.get("lead_years") or 0)
        key = _shape_record_key(metric_name, lead_years)
        if residuals_by_key[key]:
            correction = float(np.median(np.asarray(residuals_by_key[key], dtype=np.float64)))
            correction_source = "metric_lead"
        elif residuals_by_metric[metric_name]:
            correction = float(np.median(np.asarray(residuals_by_metric[metric_name], dtype=np.float64)))
            correction_source = "metric_global"
        else:
            correction = 0.0
            correction_source = "none"
        base_value = max(float(record.get("base_prediction_value") or 0.0), 0.0)
        target_value = max(float(record.get("target_value") or 0.0), 0.0)
        corrected_value = float(max(np.expm1(np.log1p(base_value) + correction), 0.0))
        scale = max(float(record.get("scale") or 0.0), float(np.finfo(np.float32).eps))
        base_error = float(record.get("base_norm_error") or 0.0)
        corrected_error = abs(corrected_value - target_value) / scale
        rows.append(
            {
                "metric_name": metric_name,
                "lead_years": lead_years,
                "key": key,
                "train_end_year": int(record.get("train_end_year") or 0),
                "quarter": str(record.get("quarter") or ""),
                "correction_source": correction_source,
                "correction_log_residual": correction,
                "base_norm_error": base_error,
                "corrected_norm_error": float(corrected_error),
                "corrected_minus_base_norm_error": float(corrected_error - base_error),
                "log_residual": float(record.get("log_residual") or 0.0),
            }
        )
        residual = float(record.get("log_residual") or 0.0)
        residuals_by_key[key].append(residual)
        residuals_by_metric[metric_name].append(residual)
    return rows


def _fit_trajectory_shape_head(
    train_rows: list[dict[str, Any]],
    *,
    max_horizon_years: int,
) -> dict[str, Any]:
    records = _trajectory_shape_error_records(train_rows, max_horizon_years=max_horizon_years)
    if not records:
        return {
            "status": "not_estimable",
            "reason": "no_train_origin_shape_records",
            "max_horizon_years": int(max_horizon_years),
        }
    walk_rows = _trajectory_shape_walk_forward_rows(records)
    residuals_by_key: dict[str, list[float]] = defaultdict(list)
    evaluation_by_key: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"base": [], "corrected": []})
    for record in records:
        residuals_by_key[_shape_record_key(str(record.get("metric_name") or ""), int(record.get("lead_years") or 0))].append(
            float(record.get("log_residual") or 0.0)
        )
    for row in walk_rows:
        if str(row.get("correction_source") or "") == "none":
            continue
        key = str(row.get("key") or "")
        evaluation_by_key[key]["base"].append(float(row.get("base_norm_error") or 0.0))
        evaluation_by_key[key]["corrected"].append(float(row.get("corrected_norm_error") or 0.0))
    correction_rows: list[dict[str, Any]] = []
    correction_by_key: dict[str, float] = {}
    for key in sorted(residuals_by_key):
        base_values = evaluation_by_key[key]["base"]
        corrected_values = evaluation_by_key[key]["corrected"]
        residual_values = residuals_by_key[key]
        if base_values and corrected_values:
            base_mean = float(np.mean(np.asarray(base_values, dtype=np.float64)))
            corrected_mean = float(np.mean(np.asarray(corrected_values, dtype=np.float64)))
            base_worst = float(np.max(np.asarray(base_values, dtype=np.float64)))
            corrected_worst = float(np.max(np.asarray(corrected_values, dtype=np.float64)))
            selected = corrected_mean < base_mean and corrected_worst <= base_worst
        else:
            base_mean = None
            corrected_mean = None
            base_worst = None
            corrected_worst = None
            selected = False
        correction = float(np.median(np.asarray(residual_values, dtype=np.float64)))
        if selected:
            correction_by_key[key] = correction
        correction_rows.append(
            {
                "metric_name": _shape_record_metric(key),
                "lead_years": _shape_record_lead(key),
                "record_count": len(residual_values),
                "walk_forward_count": len(base_values),
                "selected": bool(selected),
                "median_log_residual": correction,
                "base_mean_norm_error": base_mean,
                "corrected_mean_norm_error": corrected_mean,
                "corrected_minus_base_mean_norm_error": None
                if base_mean is None or corrected_mean is None
                else float(corrected_mean - base_mean),
                "corrected_minus_base_worst_norm_error": None
                if base_worst is None or corrected_worst is None
                else float(corrected_worst - base_worst),
            }
        )
    return {
        "status": "completed",
        "max_horizon_years": int(max_horizon_years),
        "shape_record_count": len(records),
        "walk_forward_record_count": len(walk_rows),
        "selected_correction_count": len(correction_by_key),
        "correction_by_metric_lead": correction_by_key,
        "correction_rows": correction_rows,
        "contract": "train-origin residual shape head; metric-lead corrections are used only when walk-forward train evidence improves mean error without worsening worst error",
    }


def _apply_trajectory_shape_head(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    model: dict[str, Any],
    *,
    train_end_year: int,
) -> dict[str, Any]:
    prediction = dict(base_prediction)
    corrections = dict(model.get("correction_by_metric_lead") or {})
    lead_years = max(quarter_year(str(holdout_row.get("quarter") or "")) - int(train_end_year), 1)
    for metric_name in R11_EVALUATION_METRICS:
        correction = _finite_float(corrections.get(_shape_record_key(metric_name, lead_years)))
        value = _finite_float(prediction.get(metric_name))
        if correction is None or value is None:
            continue
        prediction[metric_name] = float(max(np.expm1(np.log1p(max(float(value), 0.0)) + float(correction)), 0.0))
    return _project_prediction_row(prediction)


def _apply_shape_policy_prediction(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    model: dict[str, Any],
    *,
    train_end_year: int,
    corrected_metrics: tuple[str, ...],
) -> dict[str, Any]:
    prediction = dict(base_prediction)
    corrections = dict(model.get("correction_by_metric_lead") or {})
    lead_years = max(quarter_year(str(holdout_row.get("quarter") or "")) - int(train_end_year), 1)
    for metric_name in corrected_metrics:
        correction = _finite_float(corrections.get(_shape_record_key(metric_name, lead_years)))
        value = _finite_float(prediction.get(metric_name))
        if correction is None or value is None:
            continue
        prediction[metric_name] = float(max(np.expm1(np.log1p(max(float(value), 0.0)) + float(correction)), 0.0))

    prediction = _project_prediction_row(prediction)
    vl_rate = _conditional_rate(base_prediction, "tested_for_viral_load", "alive_on_art")
    suppression_rate = _conditional_rate(base_prediction, "virally_suppressed", "tested_for_viral_load")
    art_value = _finite_float(prediction.get("alive_on_art"))
    if art_value is not None and vl_rate is not None:
        prediction["tested_for_viral_load"] = float(max(float(art_value), 0.0) * float(vl_rate))
    vl_value = _finite_float(prediction.get("tested_for_viral_load"))
    if vl_value is not None and suppression_rate is not None:
        prediction["virally_suppressed"] = float(max(float(vl_value), 0.0) * float(suppression_rate))
    return _project_prediction_row(prediction)


def _apply_constrained_trajectory_shape_head(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    model: dict[str, Any],
    *,
    train_end_year: int,
) -> dict[str, Any]:
    return _apply_shape_policy_prediction(
        base_prediction,
        holdout_row,
        model,
        train_end_year=int(train_end_year),
        corrected_metrics=CONSTRAINED_SHAPE_DIRECT_METRICS,
    )


def _policy_by_id(policy_id: str) -> dict[str, Any]:
    for policy in HORIZON_ADAPTIVE_SHAPE_POLICIES:
        if str(policy.get("policy_id") or "") == str(policy_id):
            return dict(policy)
    return dict(HORIZON_ADAPTIVE_SHAPE_POLICIES[0])


def _policy_corrected_metrics(policy_id: str) -> tuple[str, ...]:
    policy = _policy_by_id(policy_id)
    return tuple(str(metric) for metric in tuple(policy.get("corrected_metrics") or ()))


def _mean_metric_score(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> float | None:
    score = _score_predictions(
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        prediction_rows=prediction_rows,
        metrics=metrics,
    )
    return _finite_float(score.get("mean_mae"))


def _fit_horizon_adaptive_shape_selector(
    train_rows: list[dict[str, Any]],
    *,
    max_horizon_years: int,
) -> dict[str, Any]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    policy_stats: dict[str, dict[str, list[float]]] = {
        str(policy.get("policy_id") or ""): {
            "r10_scope_scores": [],
            "full_scores": [],
            "primary_stock_scores": [],
        }
        for policy in HORIZON_ADAPTIVE_SHAPE_POLICIES
    }
    origin_count = 0
    for train_end_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if int(train_end_year) < quarter_year(str(row.get("quarter") or "")) <= int(train_end_year) + int(max_horizon_years)
        ]
        if not internal_train or not internal_holdout:
            continue
        origin_count += 1
        base_predictions, _base_summary = _candidate_predictions(
            internal_train,
            internal_holdout,
            family="back_half_conditional_rates",
        )
        shape_head = _fit_trajectory_shape_head(
            internal_train,
            max_horizon_years=int(max_horizon_years),
        )
        sorted_holdout = sorted(internal_holdout, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        for policy in HORIZON_ADAPTIVE_SHAPE_POLICIES:
            policy_id = str(policy.get("policy_id") or "")
            corrected_metrics = tuple(str(metric) for metric in tuple(policy.get("corrected_metrics") or ()))
            if corrected_metrics:
                predictions = [
                    _apply_shape_policy_prediction(
                        base_prediction,
                        holdout_row,
                        shape_head,
                        train_end_year=int(train_end_year),
                        corrected_metrics=corrected_metrics,
                    )
                    for base_prediction, holdout_row in zip(base_predictions, sorted_holdout)
                ]
            else:
                predictions = [dict(row) for row in base_predictions]
            r10_scope_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=R10_COMPARABLE_METRICS,
            )
            full_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=R11_EVALUATION_METRICS,
            )
            primary_stock_score = _mean_metric_score(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=predictions,
                metrics=PRIMARY_STOCK_GUARD_METRICS,
            )
            if r10_scope_score is not None:
                policy_stats[policy_id]["r10_scope_scores"].append(float(r10_scope_score))
            if full_score is not None:
                policy_stats[policy_id]["full_scores"].append(float(full_score))
            if primary_stock_score is not None:
                policy_stats[policy_id]["primary_stock_scores"].append(float(primary_stock_score))

    policy_rows: list[dict[str, Any]] = []
    baseline_id = "art_plus_diagnosis_flow"
    baseline_stats = policy_stats.get(baseline_id, {})
    baseline_r10 = None if not baseline_stats.get("r10_scope_scores") else float(np.mean(np.asarray(baseline_stats["r10_scope_scores"], dtype=np.float64)))
    baseline_full = None if not baseline_stats.get("full_scores") else float(np.mean(np.asarray(baseline_stats["full_scores"], dtype=np.float64)))
    baseline_primary = None if not baseline_stats.get("primary_stock_scores") else float(np.mean(np.asarray(baseline_stats["primary_stock_scores"], dtype=np.float64)))
    selected_policy_id = baseline_id
    selected_score = baseline_r10
    for policy in HORIZON_ADAPTIVE_SHAPE_POLICIES:
        policy_id = str(policy.get("policy_id") or "")
        stats = policy_stats.get(policy_id, {})
        r10_scores = list(stats.get("r10_scope_scores") or [])
        full_scores = list(stats.get("full_scores") or [])
        primary_scores = list(stats.get("primary_stock_scores") or [])
        r10_mean = None if not r10_scores else float(np.mean(np.asarray(r10_scores, dtype=np.float64)))
        full_mean = None if not full_scores else float(np.mean(np.asarray(full_scores, dtype=np.float64)))
        primary_mean = None if not primary_scores else float(np.mean(np.asarray(primary_scores, dtype=np.float64)))
        eligible = (
            r10_mean is not None
            and baseline_r10 is not None
            and r10_mean < baseline_r10
            and (baseline_full is None or (full_mean is not None and full_mean <= baseline_full))
            and (baseline_primary is None or (primary_mean is not None and primary_mean <= baseline_primary))
        )
        if eligible and (selected_score is None or float(r10_mean) < float(selected_score)):
            selected_policy_id = policy_id
            selected_score = float(r10_mean)
        policy_rows.append(
            {
                "policy_id": policy_id,
                "corrected_metrics": list(tuple(policy.get("corrected_metrics") or ())),
                "origin_count": len(r10_scores),
                "r10_scope_mean_mae": r10_mean,
                "full_mean_mae": full_mean,
                "primary_stock_mean_mae": primary_mean,
                "r10_scope_minus_baseline": None if r10_mean is None or baseline_r10 is None else float(r10_mean - baseline_r10),
                "full_minus_baseline": None if full_mean is None or baseline_full is None else float(full_mean - baseline_full),
                "primary_stock_minus_baseline": None
                if primary_mean is None or baseline_primary is None
                else float(primary_mean - baseline_primary),
                "eligible": bool(eligible or policy_id == baseline_id),
                "selected": policy_id == selected_policy_id,
            }
        )
    return {
        "status": "completed" if origin_count else "not_estimable",
        "max_horizon_years": int(max_horizon_years),
        "origin_count": int(origin_count),
        "baseline_policy_id": baseline_id,
        "selected_policy_id": selected_policy_id,
        "selected_corrected_metrics": list(_policy_corrected_metrics(selected_policy_id)),
        "policy_rows": policy_rows,
        "baseline_r10_scope_mean_mae": baseline_r10,
        "selected_r10_scope_mean_mae": selected_score,
        "contract": (
            "train-origin horizon-adaptive selector: each horizon chooses one predeclared constrained policy "
            "only if internal rolling-origin evidence improves R10-comparable score without worsening full-score "
            "or primary-stock score relative to the R11-17 constrained policy"
        ),
    }


def _apply_horizon_adaptive_constrained_shape_head(
    base_prediction: dict[str, Any],
    holdout_row: dict[str, Any],
    shape_head: dict[str, Any],
    selector: dict[str, Any],
    *,
    train_end_year: int,
) -> dict[str, Any]:
    selected_policy_id = str(selector.get("selected_policy_id") or "identity_r11_14")
    corrected_metrics = _policy_corrected_metrics(selected_policy_id)
    if not corrected_metrics:
        return _project_prediction_row(dict(base_prediction))
    return _apply_shape_policy_prediction(
        base_prediction,
        holdout_row,
        shape_head,
        train_end_year=int(train_end_year),
        corrected_metrics=corrected_metrics,
    )


def _score_conditional_rates(
    *,
    holdout_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    carry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_rows}
    carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_rows}
    output: list[dict[str, Any]] = []
    for spec in BACK_HALF_RATE_SPECS:
        rate_id = str(spec["rate_id"])
        numerator_metric = str(spec["numerator_metric"])
        denominator_metric = str(spec["denominator_metric"])
        candidate_errors: list[float] = []
        carry_errors: list[float] = []
        for holdout_row in holdout_rows:
            quarter = str(holdout_row.get("quarter") or "")
            target_rate = _conditional_rate(holdout_row, numerator_metric, denominator_metric)
            candidate_rate = _conditional_rate(candidate_by_quarter.get(quarter, {}), numerator_metric, denominator_metric)
            carry_rate = _conditional_rate(carry_by_quarter.get(quarter, {}), numerator_metric, denominator_metric)
            if target_rate is None or candidate_rate is None or carry_rate is None:
                continue
            candidate_errors.append(abs(float(candidate_rate) - float(target_rate)))
            carry_errors.append(abs(float(carry_rate) - float(target_rate)))
        candidate_mean = None if not candidate_errors else float(np.mean(np.asarray(candidate_errors, dtype=np.float64)))
        carry_mean = None if not carry_errors else float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
        candidate_worst = None if not candidate_errors else float(np.max(np.asarray(candidate_errors, dtype=np.float64)))
        carry_worst = None if not carry_errors else float(np.max(np.asarray(carry_errors, dtype=np.float64)))
        output.append(
            {
                "rate_id": rate_id,
                "numerator_metric": numerator_metric,
                "denominator_metric": denominator_metric,
                "entry_count": len(candidate_errors),
                "candidate_mean_rate_error": candidate_mean,
                "carry_forward_mean_rate_error": carry_mean,
                "candidate_worst_rate_error": candidate_worst,
                "carry_forward_worst_rate_error": carry_worst,
                "candidate_minus_carry_forward_mean_rate_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "worst_candidate_minus_carry_forward_rate_error": None
                if candidate_worst is None or carry_worst is None
                else float(candidate_worst - carry_worst),
            }
        )
    return output


def _conditional_rate_gate(rate_anatomy: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in rate_anatomy:
        rate_id = str(row.get("rate_id") or "")
        entry_count = int(row.get("entry_count") or 0)
        mean_delta = _finite_float(row.get("candidate_minus_carry_forward_mean_rate_error"))
        worst_delta = _finite_float(row.get("worst_candidate_minus_carry_forward_rate_error"))
        metric_blockers: list[str] = []
        if entry_count == 0:
            metric_blockers.append("not_evaluable")
        if mean_delta is None:
            metric_blockers.append("missing_mean_delta")
        elif mean_delta > FLOAT_NONREGRESSION_TOLERANCE:
            metric_blockers.append("mean_worse_than_conditional_rate_carry_forward")
        if worst_delta is None:
            metric_blockers.append("missing_worst_delta")
        elif worst_delta > FLOAT_NONREGRESSION_TOLERANCE:
            metric_blockers.append("worst_case_worse_than_conditional_rate_carry_forward")
        if metric_blockers:
            blockers.extend([f"{rate_id}_{blocker}" for blocker in metric_blockers])
        rows.append(
            {
                "rate_id": rate_id,
                "status": "pass" if not metric_blockers else "fail",
                "entry_count": entry_count,
                "candidate_mean_rate_error": _finite_float(row.get("candidate_mean_rate_error")),
                "carry_forward_mean_rate_error": _finite_float(row.get("carry_forward_mean_rate_error")),
                "candidate_minus_carry_forward_mean_rate_error": mean_delta,
                "worst_candidate_minus_carry_forward_rate_error": worst_delta,
                "blockers": metric_blockers,
            }
        )
    return {
        "schema_version": "phase3_dynamic.r11_conditional_rate_gate.v1",
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "rate_rows": rows,
        "contract": "dedicated third-95 process gate against last-observed conditional-rate carry-forward",
    }


def _metric_error_for_rows(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    scale = _metric_scale(train_rows, metric_name)
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in prediction_rows}
    errors: list[float] = []
    for holdout_row in holdout_rows:
        target_value = _finite_float(holdout_row.get(metric_name))
        if target_value is None:
            continue
        prediction_value = _finite_float(by_quarter.get(str(holdout_row.get("quarter") or ""), {}).get(metric_name))
        if prediction_value is None:
            continue
        errors.append(abs(float(prediction_value) - float(target_value)) / max(scale, float(np.finfo(np.float32).eps)))
    return {
        "entry_count": len(errors),
        "mean_norm_error": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
        "worst_norm_error": None if not errors else float(np.max(np.asarray(errors, dtype=np.float64))),
    }


def _select_metric_families_by_internal_backtest(
    train_rows: list[dict[str, Any]],
    *,
    candidate_families: tuple[str, ...] | None = None,
) -> dict[str, str]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    selected = {metric_name: "carry_forward" for metric_name in R11_EVALUATION_METRICS}
    if len(years) < 2:
        return selected
    if candidate_families is None:
        candidate_families = (
            "support_reporting_bias",
            "local_level_filter",
            "transition_shrinkage",
            "support_partition_calibration",
        )
    candidate_scores: dict[str, dict[str, list[float]]] = {
        family: {metric_name: [] for metric_name in R11_EVALUATION_METRICS}
        for family in candidate_families
    }
    candidate_worsts: dict[str, dict[str, list[float]]] = {
        family: {metric_name: [] for metric_name in R11_EVALUATION_METRICS}
        for family in candidate_families
    }
    carry_scores: dict[str, list[float]] = {metric_name: [] for metric_name in R11_EVALUATION_METRICS}
    carry_worsts: dict[str, list[float]] = {metric_name: [] for metric_name in R11_EVALUATION_METRICS}
    for holdout_year in years[1:]:
        internal_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) < int(holdout_year)
        ]
        internal_holdout = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) == int(holdout_year)
        ]
        if not internal_train or not internal_holdout:
            continue
        carry_predictions = _carry_forward_prediction(internal_train, internal_holdout)
        for metric_name in R11_EVALUATION_METRICS:
            carry_metric = _metric_error_for_rows(
                train_rows=internal_train,
                holdout_rows=internal_holdout,
                prediction_rows=carry_predictions,
                metric_name=metric_name,
            )
            if _finite_float(carry_metric.get("mean_norm_error")) is not None:
                carry_scores[metric_name].append(float(carry_metric["mean_norm_error"]))
            if _finite_float(carry_metric.get("worst_norm_error")) is not None:
                carry_worsts[metric_name].append(float(carry_metric["worst_norm_error"]))
        for family in candidate_families:
            predictions, _summary = _candidate_predictions(internal_train, internal_holdout, family=family)
            for metric_name in R11_EVALUATION_METRICS:
                metric = _metric_error_for_rows(
                    train_rows=internal_train,
                    holdout_rows=internal_holdout,
                    prediction_rows=predictions,
                    metric_name=metric_name,
                )
                if _finite_float(metric.get("mean_norm_error")) is not None:
                    candidate_scores[family][metric_name].append(float(metric["mean_norm_error"]))
                if _finite_float(metric.get("worst_norm_error")) is not None:
                    candidate_worsts[family][metric_name].append(float(metric["worst_norm_error"]))
    for metric_name in R11_EVALUATION_METRICS:
        if not carry_scores[metric_name]:
            continue
        carry_mean = float(np.mean(np.asarray(carry_scores[metric_name], dtype=np.float64)))
        carry_worst = float(np.max(np.asarray(carry_worsts[metric_name], dtype=np.float64))) if carry_worsts[metric_name] else float("inf")
        best_family = "carry_forward"
        best_mean = carry_mean
        for family in candidate_families:
            values = candidate_scores[family][metric_name]
            worst_values = candidate_worsts[family][metric_name]
            if not values or not worst_values:
                continue
            candidate_mean = float(np.mean(np.asarray(values, dtype=np.float64)))
            candidate_worst = float(np.max(np.asarray(worst_values, dtype=np.float64)))
            if candidate_mean < best_mean and candidate_worst <= carry_worst:
                best_family = family
                best_mean = candidate_mean
        selected[metric_name] = best_family
    return selected


def _selector_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    selected: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidate_family_names = sorted({str(value) for value in selected.values() if str(value) != "carry_forward"})
    family_predictions: dict[str, list[dict[str, Any]]] = {}
    for candidate_family in candidate_family_names:
        family_predictions[candidate_family], _summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family=candidate_family,
        )
    carry_predictions = _carry_forward_prediction(train_rows, holdout_rows)
    predictions: list[dict[str, Any]] = []
    for row_index, carry_row in enumerate(carry_predictions):
        prediction = {"quarter": str(carry_row.get("quarter") or "")}
        for metric_name in R11_EVALUATION_METRICS:
            selected_family = str(selected.get(metric_name) or "carry_forward")
            source_rows = carry_predictions if selected_family == "carry_forward" else family_predictions.get(selected_family, carry_predictions)
            prediction[metric_name] = source_rows[row_index].get(metric_name)
        predictions.append(_project_prediction_row(prediction))
    return predictions, family_predictions


def _carry_forward_prediction(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    sorted_train = sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    for holdout_row in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        prediction: dict[str, Any] = {"quarter": str(holdout_row.get("quarter") or "")}
        for metric_name in R11_EVALUATION_METRICS:
            eligible = [row for row in sorted_train if _finite_float(row.get(metric_name)) is not None]
            prediction[metric_name] = None if not eligible else float(eligible[-1].get(metric_name) or 0.0)
        predictions.append(_project_prediction_row(prediction))
    return predictions


def _project_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    projected = project_cascade_stock_row(row)
    output = dict(row)
    for metric_name, value in dict(projected.get("projected") or {}).items():
        output[metric_name] = value
    output["cascade_projection_changed"] = bool(projected.get("changed"))
    return output


def _candidate_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if family == "metric_selector":
        selected = _select_metric_families_by_internal_backtest(train_rows)
        predictions, _family_predictions = _selector_predictions(
            train_rows,
            holdout_rows,
            selected=selected,
        )
        return predictions, {
            "family": family,
            "selected_metric_families": selected,
            "contract": "train-backtested selector defaults to carry-forward and only switches a metric when the candidate beats carry-forward on internal train-origin history",
        }

    if family == "r10_style_readout_teacher":
        selected = _select_metric_families_by_internal_backtest(
            train_rows,
            candidate_families=(
                "linkage_lag_kernel",
                "support_partition_calibration",
                "linkage_lag_plus_support_partition",
            ),
        )
        predictions, _family_predictions = _selector_predictions(
            train_rows,
            holdout_rows,
            selected=selected,
        )
        return predictions, {
            "family": family,
            "selected_metric_families": selected,
            "contract": (
                "R10-style endpoint/readout teacher without R10 target leakage: it learns only from "
                "internal train-origin history over R11-08, R11-09, and their cascade-projected combination; "
                "R10 is used only as an external promotion benchmark"
            ),
        }

    if family == "back_half_conditional_rates":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="r10_style_readout_teacher",
        )
        process = _fit_back_half_rate_process(train_rows)
        predictions: list[dict[str, Any]] = []
        for base_prediction, holdout_row in zip(
            base_predictions,
            sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
        ):
            predictions.append(_apply_back_half_rate_process(base_prediction, holdout_row, process))
        return predictions, {
            "family": family,
            "base_family": "r10_style_readout_teacher",
            "base_selected_metric_families": dict(base_summary.get("selected_metric_families") or {}),
            "back_half_rate_process": process,
            "contract": (
                "R11-14 uses the R11-13 train-only front-half readout, then replaces only "
                "tested_for_viral_load and virally_suppressed with train-selected conditional-rate processes"
            ),
        }

    if family == "trajectory_shape_head":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="back_half_conditional_rates",
        )
        train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
        train_end_year = max(train_years) if train_years else 0
        max_horizon_years = max(
            [
                max(quarter_year(str(row.get("quarter") or "")) - int(train_end_year), 1)
                for row in holdout_rows
                if row.get("quarter")
            ]
            or [1]
        )
        shape_head = _fit_trajectory_shape_head(
            train_rows,
            max_horizon_years=int(max_horizon_years),
        )
        predictions = [
            _apply_trajectory_shape_head(
                base_prediction,
                holdout_row,
                shape_head,
                train_end_year=int(train_end_year),
            )
            for base_prediction, holdout_row in zip(
                base_predictions,
                sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
            )
        ]
        return predictions, {
            "family": family,
            "base_family": "back_half_conditional_rates",
            "base_selected_metric_families": dict(base_summary.get("base_selected_metric_families") or {}),
            "back_half_rate_process": dict(base_summary.get("back_half_rate_process") or {}),
            "trajectory_shape_head": shape_head,
            "contract": (
                "R11-16 wraps R11-14 with train-origin metric/lead residual-shape corrections; "
                "no correction is applied unless internal walk-forward evidence beats the uncorrected shape"
            ),
        }

    if family == "constrained_trajectory_shape_head":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="back_half_conditional_rates",
        )
        train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
        train_end_year = max(train_years) if train_years else 0
        max_horizon_years = max(
            [
                max(quarter_year(str(row.get("quarter") or "")) - int(train_end_year), 1)
                for row in holdout_rows
                if row.get("quarter")
            ]
            or [1]
        )
        shape_head = _fit_trajectory_shape_head(
            train_rows,
            max_horizon_years=int(max_horizon_years),
        )
        predictions = [
            _apply_constrained_trajectory_shape_head(
                base_prediction,
                holdout_row,
                shape_head,
                train_end_year=int(train_end_year),
            )
            for base_prediction, holdout_row in zip(
                base_predictions,
                sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
            )
        ]
        return predictions, {
            "family": family,
            "base_family": "back_half_conditional_rates",
            "base_selected_metric_families": dict(base_summary.get("base_selected_metric_families") or {}),
            "back_half_rate_process": dict(base_summary.get("back_half_rate_process") or {}),
            "trajectory_shape_head": shape_head,
            "constrained_shape_contract": {
                "directly_corrected_metrics": list(CONSTRAINED_SHAPE_DIRECT_METRICS),
                "diagnosed_anchor_policy": "unchanged_from_R11_14",
                "back_half_policy": "regenerate_tested_for_viral_load_and_virally_suppressed_from_R11_14_conditional_rates_after_ART_correction",
            },
            "contract": (
                "R11-17 constrains R11-16 residual-shape corrections to preserve the diagnosed anchor, "
                "the cascade cone, and the R11-14 conditional VL/suppression rates before blocked evaluation"
            ),
        }

    if family == "horizon_adaptive_constrained_shape_head":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="back_half_conditional_rates",
        )
        train_years = [quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")]
        train_end_year = max(train_years) if train_years else 0
        max_horizon_years = max(
            [
                max(quarter_year(str(row.get("quarter") or "")) - int(train_end_year), 1)
                for row in holdout_rows
                if row.get("quarter")
            ]
            or [1]
        )
        shape_head = _fit_trajectory_shape_head(
            train_rows,
            max_horizon_years=int(max_horizon_years),
        )
        selector = _fit_horizon_adaptive_shape_selector(
            train_rows,
            max_horizon_years=int(max_horizon_years),
        )
        predictions = [
            _apply_horizon_adaptive_constrained_shape_head(
                base_prediction,
                holdout_row,
                shape_head,
                selector,
                train_end_year=int(train_end_year),
            )
            for base_prediction, holdout_row in zip(
                base_predictions,
                sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))),
            )
        ]
        return predictions, {
            "family": family,
            "base_family": "back_half_conditional_rates",
            "base_selected_metric_families": dict(base_summary.get("base_selected_metric_families") or {}),
            "back_half_rate_process": dict(base_summary.get("back_half_rate_process") or {}),
            "trajectory_shape_head": shape_head,
            "horizon_adaptive_shape_selector": selector,
            "horizon_adaptive_shape_contract": {
                "policy_set": [
                    {
                        "policy_id": str(policy.get("policy_id") or ""),
                        "corrected_metrics": list(tuple(policy.get("corrected_metrics") or ())),
                        "description": str(policy.get("description") or ""),
                    }
                    for policy in HORIZON_ADAPTIVE_SHAPE_POLICIES
                ],
                "selection_target": list(R10_COMPARABLE_METRICS),
                "guard_metrics": list(PRIMARY_STOCK_GUARD_METRICS),
                "back_half_policy": "regenerate_tested_for_viral_load_and_virally_suppressed_from_R11_14_conditional_rates_after_stock_correction",
            },
            "contract": (
                "R11-18 selects a predeclared constrained shape policy separately for each lifted horizon "
                "using only train-origin backtests relative to the R11-17 constrained policy; policies target R10-comparable trajectory drift while "
                "the outer gate still enforces full stock-cone and conditional-rate validity"
            ),
        }

    if family == "datv_transition_process":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="r10_style_readout_teacher",
        )
        back_half_process = _fit_back_half_rate_process(train_rows)
        transition_process = _fit_datv_transition_process(train_rows)
        predictions = _apply_datv_transition_process(
            train_rows,
            holdout_rows,
            base_predictions,
            transition_process,
            back_half_process,
        )
        return predictions, {
            "family": family,
            "base_family": "r10_style_readout_teacher",
            "base_selected_metric_families": dict(base_summary.get("selected_metric_families") or {}),
            "transition_process": transition_process,
            "back_half_rate_process": back_half_process,
            "contract": (
                "R11-19 replaces D and ART readouts with an explicit transition process: "
                "diagnosed stock advances from previous diagnosed stock plus diagnosis flow, ART advances from retained ART plus lagged D_to_A pressure, "
                "and VL/suppression remain conditional-rate state observations rather than endpoint corrections"
            ),
        }

    if family == "era_datv_transition_process":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="r10_style_readout_teacher",
        )
        back_half_process = _fit_back_half_rate_process(train_rows)
        transition_process = _fit_era_datv_transition_process(train_rows)
        predictions = _apply_era_datv_transition_process(
            train_rows,
            holdout_rows,
            base_predictions,
            transition_process,
            back_half_process,
        )
        return predictions, {
            "family": family,
            "base_family": "r10_style_readout_teacher",
            "base_selected_metric_families": dict(base_summary.get("selected_metric_families") or {}),
            "transition_process": transition_process,
            "back_half_rate_process": back_half_process,
            "contract": (
                "R11-20 extends R11-19 with explicit removal fractions and observation-support-era reporting shifts "
                "for diagnosed stock and ART stock; coefficients and shifts are fitted only from train-window stock-flow pairs"
            ),
        }

    if family == "horizon_family_selector":
        predictions, selector_summary = _family_selector_predictions(
            train_rows,
            holdout_rows,
            candidate_families=("constrained_trajectory_shape_head", "era_datv_transition_process"),
            metrics=R10_COMPARABLE_METRICS,
        )
        return predictions, {
            "family": family,
            **selector_summary,
            "contract": "R11-21 train-origin selector chooses between R11-17 and R11-20 by horizon using R10-comparable internal evidence",
        }

    if family == "diagnosis_flow_input_repair_process":
        predictions, summary = _era_process_with_flow_variant_predictions(
            train_rows,
            holdout_rows,
            variant="stock_reconciled_flow",
        )
        return predictions, {
            "family": family,
            **summary,
            "contract": "R11-22 repairs diagnosis-flow inputs from train-fitted diagnosed-stock reconciliation before applying the R11-20 transition process",
        }

    if family == "support_era_diagnosis_flow_process":
        predictions, summary = _era_process_with_flow_variant_predictions(
            train_rows,
            holdout_rows,
            variant="support_era_flow",
        )
        return predictions, {
            "family": family,
            **summary,
            "contract": "R11-23 applies train-fitted support-era diagnosis-flow reporting adjustment before the R11-20 transition process",
        }

    if family == "stock_flow_reconciliation_process":
        predictions, summary = _era_process_with_flow_variant_predictions(
            train_rows,
            holdout_rows,
            variant="stock_flow_reconciliation",
        )
        return predictions, {
            "family": family,
            **summary,
            "contract": "R11-24 explicitly reconciles diagnosed-stock changes against diagnosis flow before transition replay",
        }

    if family == "diagnosed_reporting_bias_process":
        base_predictions, base_summary = _candidate_predictions(train_rows, holdout_rows, family="r10_style_readout_teacher")
        diagnosed_model = _fit_support_partition_calibration_model(train_rows, "diagnosed_plhiv")
        adjusted: list[dict[str, Any]] = []
        for base_prediction, holdout_row in zip(base_predictions, sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))):
            row = dict(base_prediction)
            diagnosed_value = _predict_support_partition_calibration(diagnosed_model, holdout_row, "diagnosed_plhiv")
            if diagnosed_value is not None:
                row["diagnosed_plhiv"] = diagnosed_value
            adjusted.append(_project_prediction_row(row))
        process = _fit_back_half_rate_process(train_rows)
        predictions = [_apply_back_half_rate_process(row, holdout_row, process) for row, holdout_row in zip(adjusted, sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))))]
        return predictions, {
            "family": family,
            "base_summary": base_summary,
            "diagnosed_reporting_model": diagnosed_model,
            "back_half_rate_process": process,
            "contract": "R11-25 separates diagnosed-stock reporting bias from biological transition by adjusting D only, then regenerating back-half rates",
        }

    if family == "art_horizon_selector_process":
        predictions, selector_summary = _family_selector_predictions(
            train_rows,
            holdout_rows,
            candidate_families=("constrained_trajectory_shape_head", "era_datv_transition_process"),
            metrics=("alive_on_art",),
        )
        return predictions, {
            "family": family,
            **selector_summary,
            "contract": "R11-26 train-origin selector targets ART trajectory behavior specifically while outer gates still score full DATV",
        }

    if family == "diagnosis_lag_stock_process":
        predictions, summary = _era_process_with_flow_variant_predictions(
            train_rows,
            holdout_rows,
            variant="lagged_diagnosis_flow",
        )
        return predictions, {
            "family": family,
            **summary,
            "contract": "R11-27 pushes diagnosis-flow lag structure into diagnosed/ART stock transitions, not only endpoint readout",
        }

    if family == "multi_horizon_weighted_process":
        predictions, summary = _family_weighted_predictions(
            train_rows,
            holdout_rows,
            first_family="constrained_trajectory_shape_head",
            second_family="era_datv_transition_process",
        )
        return predictions, {
            "family": family,
            **summary,
            "contract": "R11-28 uses train-origin inverse-error weights to combine R11-17 short-horizon readout and R11-20 transition dynamics",
        }

    if family == "r12_long_horizon_stock_shape_process":
        predictions, summary = _r12_long_horizon_stock_shape_predictions(train_rows, holdout_rows)
        return predictions, {
            "family": family,
            **summary,
            "contract": (
                "R12-01 promotes R11-28 as the base reference and targets only diagnosed_plhiv/alive_on_art "
                "3y/5y trajectory drift with leave-origin validated stock-shape corrections"
            ),
        }

    if family == "r12_da_process_split_transition":
        base_predictions, base_summary = _candidate_predictions(
            train_rows,
            holdout_rows,
            family="multi_horizon_weighted_process",
        )
        process_split = _fit_process_split_da_transition(train_rows)
        back_half_process = _fit_back_half_rate_process(train_rows)
        predictions = _apply_process_split_da_transition(
            train_rows,
            holdout_rows,
            base_predictions,
            process_split,
            back_half_process,
        )
        return predictions, {
            "family": family,
            "base_family": "multi_horizon_weighted_process",
            "base_summary": base_summary,
            "process_split_transition": process_split,
            "back_half_rate_process": back_half_process,
            "contract": (
                "R12-02 starts from the locked R11-28 reference and replaces only D/A stock dynamics with "
                "a process-split transition: D reporting/removal/inflow plus A delayed linkage, initiation "
                "capacity, retention/removal, and reporting shift"
            ),
        }

    if family == "r12_da_residual_source_process":
        predictions, summary = _r12_da_residual_source_predictions(train_rows, holdout_rows)
        return predictions, {
            "family": family,
            **summary,
            "contract": (
                "R12-03 feeds only train-selected D/A residual-source adjustments back into the R12-02 process outputs. "
                "Candidate source contexts are source family, support partition, monthly reporting intensity, and "
                "train-derived backlog/rebound diagnosis-flow regime."
            ),
        }

    if family == "r12_route_aware_two_head_process":
        predictions, summary = _r12_route_aware_two_head_predictions(train_rows, holdout_rows)
        return predictions, {
            "family": family,
            **summary,
            "contract": (
                "R12-08 uses separate heads for separate evidence claims: the 1y/2y program-nowcast head is R11-28, "
                "and the 3y/5y annual-trajectory head is R12-01. The head switch is deterministic from forecast lead "
                "time and is evaluated through route-specific gates."
            ),
        }

    if family == "r12_stock_cone_safe_annual_trajectory_process":
        predictions, summary = _r12_stock_cone_safe_annual_trajectory_predictions(train_rows, holdout_rows)
        return predictions, {
            "family": family,
            **summary,
            "contract": (
                "R12-09 keeps R11-28 as the general backbone and applies a train-selected annual-anchor stock head "
                "only to slide annual-anchor diagnosed_plhiv/alive_on_art rows. Program nowcasting evidence is not "
                "mutated; every output is cascade-cone projected and back-half rates remain governed by R11-14."
            ),
        }

    if family == "r12_program_nowcast_mixed_quarterly_process":
        predictions, summary = _r12_program_nowcast_mixed_quarterly_predictions(train_rows, holdout_rows)
        return predictions, {
            "family": family,
            **summary,
            "contract": (
                "R12-10 targets the remaining R10 failure on DOH program nowcast and mixed quarterly D/A trajectory. "
                "It keeps R12-09's annual-anchor behavior frozen and mutates only DOH quarterly/monthly diagnosed, ART, "
                "and diagnosis-flow rows using a train-origin program selector."
            ),
        }

    if family == "r10_scope_teacher_stock_process":
        base_predictions, base_summary = _candidate_predictions(train_rows, holdout_rows, family="r10_style_readout_teacher")
        transition_predictions, transition_summary = _candidate_predictions(train_rows, holdout_rows, family="era_datv_transition_process")
        transition_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in transition_predictions}
        rows: list[dict[str, Any]] = []
        for base in base_predictions:
            quarter = str(base.get("quarter") or "")
            transition = transition_by_quarter.get(quarter, {})
            row = dict(base)
            for metric_name in PRIMARY_STOCK_GUARD_METRICS:
                if _finite_float(transition.get(metric_name)) is not None:
                    row[metric_name] = transition.get(metric_name)
            rows.append(_project_prediction_row(row))
        process = _fit_back_half_rate_process(train_rows)
        predictions = [_apply_back_half_rate_process(row, holdout_row, process) for row, holdout_row in zip(rows, sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))))]
        return predictions, {
            "family": family,
            "base_summary": base_summary,
            "transition_summary": transition_summary,
            "back_half_rate_process": process,
            "contract": "R11-29 keeps the R10-style readout teacher for flow shape but constrains D/A stocks through the R11-20 process",
        }

    if family == "conditional_rate_horizon_selector":
        predictions, selector_summary = _family_selector_predictions(
            train_rows,
            holdout_rows,
            candidate_families=("back_half_conditional_rates", "era_datv_transition_process"),
            metrics=("tested_for_viral_load", "virally_suppressed"),
        )
        return predictions, {
            "family": family,
            **selector_summary,
            "contract": "R11-32 selects the back-half conditional-rate trajectory family by train-origin VL/suppression evidence",
        }

    pooled_slopes = _pooled_log_slopes(train_rows)
    if family == "transition_shrinkage":
        model_by_metric = {
            metric_name: _fit_transition_shrinkage_model(
                train_rows,
                metric_name,
                pooled_slopes=pooled_slopes,
            )
            for metric_name in R11_EVALUATION_METRICS
        }
    elif family == "support_partition_calibration":
        model_by_metric = {
            metric_name: _fit_support_partition_calibration_model(train_rows, metric_name)
            for metric_name in R11_EVALUATION_METRICS
        }
    elif family == "linkage_lag_plus_support_partition":
        model_by_metric = {
            metric_name: _fit_support_partition_calibration_model(train_rows, metric_name)
            for metric_name in R11_EVALUATION_METRICS
        }
    else:
        model_by_metric = {
            metric_name: _fit_log_linear_support_model(train_rows, metric_name)
            for metric_name in R11_EVALUATION_METRICS
        }
    linkage_kernel = _fit_linkage_lag_kernel(train_rows) if family in {"linkage_lag_kernel", "linkage_lag_plus_support_partition"} else None
    carry_predictions = _carry_forward_prediction(train_rows, holdout_rows) if family == "linkage_lag_kernel" else []
    predictions: list[dict[str, Any]] = []
    for row_index, holdout_row in enumerate(sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))):
        prediction: dict[str, Any] = {"quarter": str(holdout_row.get("quarter") or "")}
        for metric_name in R11_EVALUATION_METRICS:
            model = model_by_metric[metric_name]
            if family == "support_reporting_bias":
                value = _predict_support_trend(model, holdout_row, metric_name)
            elif family == "local_level_filter":
                value = _predict_local_level(model, holdout_row, metric_name)
            elif family == "transition_shrinkage":
                value = _predict_transition_shrinkage(model, holdout_row)
            elif family == "support_partition_calibration":
                value = _predict_support_partition_calibration(model, holdout_row, metric_name)
            elif family == "linkage_lag_kernel":
                if metric_name == "alive_on_art":
                    value = _linkage_lag_prediction_value(dict(linkage_kernel or {}), train_rows, holdout_row)
                else:
                    value = carry_predictions[row_index].get(metric_name)
            elif family == "linkage_lag_plus_support_partition":
                if metric_name == "alive_on_art":
                    value = _linkage_lag_prediction_value(dict(linkage_kernel or {}), train_rows, holdout_row)
                else:
                    value = _predict_support_partition_calibration(model, holdout_row, metric_name)
            else:
                raise ValueError(f"Unknown R11 candidate family: {family}")
            prediction[metric_name] = value
        predictions.append(_project_prediction_row(prediction))
    return predictions, {
        "family": family,
        "metric_models": model_by_metric,
        "linkage_lag_kernel": linkage_kernel,
        "contract": (
            "support_reporting_bias uses train-window trend plus empirical support-signature residuals; "
            "local_level_filter uses last train canonical state plus empirical median quarterly slope; "
            "transition_shrinkage uses empirical-Bayes precision pooling; "
            "linkage_lag_kernel only changes alive_on_art through train-selected diagnosis-flow delay; "
            "support_partition_calibration only applies train support-partition residuals; "
            "linkage_lag_plus_support_partition combines R11-08 for ART stock with R11-09 for all other metrics; "
            "all are forecast-origin safe with respect to target values"
        ),
    }


def _score_predictions(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = R11_EVALUATION_METRICS,
) -> dict[str, Any]:
    predictions_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in prediction_rows}
    metric_rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    for metric_name in metrics:
        scale = _metric_scale(train_rows, metric_name)
        errors: list[float] = []
        for holdout_row in holdout_rows:
            target_value = _finite_float(holdout_row.get(metric_name))
            if target_value is None:
                continue
            prediction_value = _finite_float(predictions_by_quarter.get(str(holdout_row.get("quarter") or ""), {}).get(metric_name))
            if prediction_value is None:
                continue
            errors.append(abs(prediction_value - target_value) / max(scale, float(np.finfo(np.float32).eps)))
        all_errors.extend(errors)
        metric_rows.append(
            {
                "metric_name": metric_name,
                "entry_count": len(errors),
                "mean_norm_error": None if not errors else float(np.mean(np.asarray(errors, dtype=np.float64))),
                "worst_norm_error": None if not errors else float(np.max(np.asarray(errors, dtype=np.float64))),
            }
        )
    return {
        "mean_mae": None if not all_errors else float(np.mean(np.asarray(all_errors, dtype=np.float64))),
        "worst_mae": None if not all_errors else float(np.max(np.asarray(all_errors, dtype=np.float64))),
        "entry_count": len(all_errors),
        "metric_rows": metric_rows,
    }


def _metric_anatomy_from_scores(
    candidate_metric_rows: list[dict[str, Any]],
    carry_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    carry_by_metric = {str(row.get("metric_name") or ""): dict(row) for row in carry_metric_rows}
    output: list[dict[str, Any]] = []
    for candidate in candidate_metric_rows:
        metric_name = str(candidate.get("metric_name") or "")
        carry = carry_by_metric.get(metric_name, {})
        candidate_mean = _finite_float(candidate.get("mean_norm_error"))
        carry_mean = _finite_float(carry.get("mean_norm_error"))
        candidate_worst = _finite_float(candidate.get("worst_norm_error"))
        carry_worst = _finite_float(carry.get("worst_norm_error"))
        output.append(
            {
                "metric_name": metric_name,
                "entry_count": int(candidate.get("entry_count") or 0),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "worst_candidate_minus_carry_forward_norm_error": None
                if candidate_worst is None or carry_worst is None
                else float(candidate_worst - carry_worst),
            }
        )
    return output


def _score_branch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_values = [
        float(row["candidate_mae"])
        for row in rows
        if _finite_float(row.get("candidate_mae")) is not None
    ]
    carry_values = [
        float(row["carry_forward_mae"])
        for row in rows
        if _finite_float(row.get("carry_forward_mae")) is not None
    ]
    if not candidate_values or not carry_values:
        return {
            "split_count": 0,
            "candidate_mean_mae": None,
            "carry_forward_mean_mae": None,
            "candidate_worst_mae": None,
            "carry_forward_worst_mae": None,
            "candidate_minus_carry_forward_mean_mae": None,
            "candidate_minus_carry_forward_worst_mae": None,
        }
    candidate_array = np.asarray(candidate_values, dtype=np.float64)
    carry_array = np.asarray(carry_values, dtype=np.float64)
    return {
        "split_count": len(candidate_values),
        "candidate_mean_mae": float(np.mean(candidate_array)),
        "carry_forward_mean_mae": float(np.mean(carry_array)),
        "candidate_worst_mae": float(np.max(candidate_array)),
        "carry_forward_worst_mae": float(np.max(carry_array)),
        "candidate_minus_carry_forward_mean_mae": float(np.mean(candidate_array) - np.mean(carry_array)),
        "candidate_minus_carry_forward_worst_mae": float(np.max(candidate_array) - np.max(carry_array)),
    }


def _r11_candidate_report(
    *,
    experiment_id: str,
    family: str,
    rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    r10_reference_mae: float | None,
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    candidate_metric_accumulator: dict[str, list[float]] = defaultdict(list)
    carry_metric_accumulator: dict[str, list[float]] = defaultdict(list)
    candidate_worst_accumulator: dict[str, list[float]] = defaultdict(list)
    carry_worst_accumulator: dict[str, list[float]] = defaultdict(list)
    candidate_rate_accumulator: dict[str, list[float]] = defaultdict(list)
    carry_rate_accumulator: dict[str, list[float]] = defaultdict(list)
    candidate_rate_worst_accumulator: dict[str, list[float]] = defaultdict(list)
    carry_rate_worst_accumulator: dict[str, list[float]] = defaultdict(list)
    model_summaries: list[dict[str, Any]] = []
    for split in splits:
        holdout_years = [int(year) for year in list(split.get("holdout_years") or [])]
        if not holdout_years:
            continue
        train_end_year = int(split.get("train_end_year") or min(holdout_years) - 1)
        train_rows = [
            dict(row)
            for row in rows
            if quarter_year(str(row.get("quarter") or "")) <= train_end_year
        ]
        holdout_rows = [
            dict(row)
            for row in rows
            if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
        ]
        if not train_rows or not holdout_rows:
            continue
        carry_predictions = _carry_forward_prediction(train_rows, holdout_rows)
        candidate_predictions, model_summary = _candidate_predictions(train_rows, holdout_rows, family=family)
        carry_score = _score_predictions(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            prediction_rows=carry_predictions,
        )
        candidate_score = _score_predictions(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            prediction_rows=candidate_predictions,
        )
        carry_r10_comparable_score = _score_predictions(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            prediction_rows=carry_predictions,
            metrics=R10_COMPARABLE_METRICS,
        )
        candidate_r10_comparable_score = _score_predictions(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            prediction_rows=candidate_predictions,
            metrics=R10_COMPARABLE_METRICS,
        )
        conditional_rate_rows = _score_conditional_rates(
            holdout_rows=holdout_rows,
            candidate_rows=candidate_predictions,
            carry_rows=carry_predictions,
        )
        for candidate_metric, carry_metric in zip(candidate_score["metric_rows"], carry_score["metric_rows"]):
            metric_name = str(candidate_metric.get("metric_name") or "")
            candidate_mean = _finite_float(candidate_metric.get("mean_norm_error"))
            carry_mean = _finite_float(carry_metric.get("mean_norm_error"))
            candidate_worst = _finite_float(candidate_metric.get("worst_norm_error"))
            carry_worst = _finite_float(carry_metric.get("worst_norm_error"))
            if candidate_mean is not None:
                candidate_metric_accumulator[metric_name].append(candidate_mean)
            if carry_mean is not None:
                carry_metric_accumulator[metric_name].append(carry_mean)
            if candidate_worst is not None:
                candidate_worst_accumulator[metric_name].append(candidate_worst)
            if carry_worst is not None:
                carry_worst_accumulator[metric_name].append(carry_worst)
        for rate_row in conditional_rate_rows:
            rate_id = str(rate_row.get("rate_id") or "")
            candidate_mean_rate = _finite_float(rate_row.get("candidate_mean_rate_error"))
            carry_mean_rate = _finite_float(rate_row.get("carry_forward_mean_rate_error"))
            candidate_worst_rate = _finite_float(rate_row.get("candidate_worst_rate_error"))
            carry_worst_rate = _finite_float(rate_row.get("carry_forward_worst_rate_error"))
            if candidate_mean_rate is not None:
                candidate_rate_accumulator[rate_id].append(candidate_mean_rate)
            if carry_mean_rate is not None:
                carry_rate_accumulator[rate_id].append(carry_mean_rate)
            if candidate_worst_rate is not None:
                candidate_rate_worst_accumulator[rate_id].append(candidate_worst_rate)
            if carry_worst_rate is not None:
                carry_rate_worst_accumulator[rate_id].append(carry_worst_rate)
        split_rows.append(
            {
                "train_end_year": train_end_year,
                "train_years": list(split.get("train_years") or []),
                "holdout_years": holdout_years,
                "candidate_mae": candidate_score["mean_mae"],
                "carry_forward_mae": carry_score["mean_mae"],
                "candidate_minus_carry_forward_mae": None
                if candidate_score["mean_mae"] is None or carry_score["mean_mae"] is None
                else float(candidate_score["mean_mae"] - carry_score["mean_mae"]),
                "candidate_entry_count": candidate_score["entry_count"],
                "carry_forward_entry_count": carry_score["entry_count"],
                "r10_comparable_candidate_mae": candidate_r10_comparable_score["mean_mae"],
                "r10_comparable_carry_forward_mae": carry_r10_comparable_score["mean_mae"],
                "r10_comparable_entry_count": candidate_r10_comparable_score["entry_count"],
                "fit_status": "completed",
            }
        )
        model_summary_row = {
            "train_end_year": train_end_year,
            "holdout_years": holdout_years,
            "metric_model_status": {
                metric_name: str(model.get("status") or "")
                for metric_name, model in dict(model_summary.get("metric_models") or {}).items()
            },
        }
        if model_summary.get("selected_metric_families"):
            model_summary_row["selected_metric_families"] = dict(model_summary.get("selected_metric_families") or {})
        if model_summary.get("base_selected_metric_families"):
            model_summary_row["base_selected_metric_families"] = dict(model_summary.get("base_selected_metric_families") or {})
        if model_summary.get("linkage_lag_kernel"):
            kernel = dict(model_summary.get("linkage_lag_kernel") or {})
            model_summary_row["linkage_lag_kernel"] = {
                "status": str(kernel.get("status") or ""),
                "selected_lag_quarters": kernel.get("selected_lag_quarters"),
                "selected_coefficient": kernel.get("selected_coefficient"),
            }
        if model_summary.get("back_half_rate_process"):
            process = dict(model_summary.get("back_half_rate_process") or {})
            rate_models = dict(process.get("rate_models") or {})
            model_summary_row["back_half_rate_process"] = {
                "status": str(process.get("status") or ""),
                "selected_variants": {
                    rate_id: str(dict(model).get("selected_variant") or "")
                    for rate_id, model in sorted(rate_models.items())
                    if isinstance(model, dict)
                },
            }
        if model_summary.get("trajectory_shape_head"):
            shape_head = dict(model_summary.get("trajectory_shape_head") or {})
            model_summary_row["trajectory_shape_head"] = {
                "status": str(shape_head.get("status") or ""),
                "shape_record_count": shape_head.get("shape_record_count"),
                "walk_forward_record_count": shape_head.get("walk_forward_record_count"),
                "selected_correction_count": shape_head.get("selected_correction_count"),
            }
        if model_summary.get("constrained_shape_contract"):
            model_summary_row["constrained_shape_contract"] = dict(model_summary.get("constrained_shape_contract") or {})
        if model_summary.get("horizon_adaptive_shape_selector"):
            selector = dict(model_summary.get("horizon_adaptive_shape_selector") or {})
            model_summary_row["horizon_adaptive_shape_selector"] = {
                "status": str(selector.get("status") or ""),
                "max_horizon_years": selector.get("max_horizon_years"),
                "origin_count": selector.get("origin_count"),
                "baseline_policy_id": selector.get("baseline_policy_id"),
                "selected_policy_id": selector.get("selected_policy_id"),
                "selected_corrected_metrics": list(selector.get("selected_corrected_metrics") or []),
            }
        if model_summary.get("horizon_adaptive_shape_contract"):
            model_summary_row["horizon_adaptive_shape_contract"] = dict(model_summary.get("horizon_adaptive_shape_contract") or {})
        if model_summary.get("transition_process"):
            process = dict(model_summary.get("transition_process") or {})
            diagnosed_model = dict(process.get("diagnosed_stock_transition") or {})
            art_model = dict(process.get("art_stock_transition") or {})
            model_summary_row["transition_process"] = {
                "status": str(process.get("status") or ""),
                "diagnosed_retention_coefficient": diagnosed_model.get("retention_coefficient"),
                "diagnosis_flow_coefficient": diagnosed_model.get("diagnosis_flow_coefficient"),
                "diagnosed_removal_fraction": diagnosed_model.get("diagnosed_removal_fraction"),
                "diagnosed_reporting_era_count": diagnosed_model.get("era_count"),
                "art_retention_coefficient": art_model.get("art_retention_coefficient"),
                "diagnosis_linkage_coefficient": art_model.get("diagnosis_linkage_coefficient"),
                "diagnosed_gap_linkage_coefficient": art_model.get("diagnosed_gap_linkage_coefficient"),
                "art_removal_fraction": art_model.get("art_removal_fraction"),
                "art_reporting_era_count": art_model.get("era_count"),
                "selected_art_lag_quarters": art_model.get("selected_lag_quarters"),
            }
        if model_summary.get("family_selector"):
            selector = dict(model_summary.get("family_selector") or {})
            model_summary_row["family_selector"] = {
                "status": str(selector.get("status") or ""),
                "selected_family": selector.get("selected_family"),
                "candidate_families": list(selector.get("candidate_families") or []),
                "max_horizon_years": selector.get("max_horizon_years"),
                "metric_scope": list(selector.get("metric_scope") or []),
            }
        if model_summary.get("annual_anchor_selector"):
            selector = dict(model_summary.get("annual_anchor_selector") or {})
            blend_selector = dict(selector.get("blend_selector") or {})
            model_summary_row["annual_anchor_selector"] = {
                "status": str(selector.get("status") or ""),
                "selected_family": selector.get("selected_family"),
                "selected_blend_weight": selector.get("selected_blend_weight"),
                "base_mean_norm_error": selector.get("base_mean_norm_error"),
                "selected_mean_norm_error": selector.get("selected_mean_norm_error"),
                "blend_status": str(blend_selector.get("status") or ""),
                "blend_record_count": blend_selector.get("record_count"),
                "annual_anchor_record_count": blend_selector.get("annual_anchor_record_count"),
            }
        if model_summary.get("flow_adjustment"):
            flow_adjustment = dict(model_summary.get("flow_adjustment") or {})
            model_summary_row["flow_adjustment"] = {
                "variant": flow_adjustment.get("variant"),
                "support_flow_model_status": flow_adjustment.get("support_flow_model_status"),
                "diagnosed_model_status": flow_adjustment.get("diagnosed_model_status"),
                "transition_process_status": flow_adjustment.get("transition_process_status"),
                "lag_quarters": flow_adjustment.get("lag_quarters"),
            }
        if model_summary.get("diagnosed_reporting_model"):
            model = dict(model_summary.get("diagnosed_reporting_model") or {})
            model_summary_row["diagnosed_reporting_model"] = {
                "status": str(model.get("status") or ""),
                "residual_count": model.get("residual_count"),
            }
        if model_summary.get("r12_stock_drift_correction"):
            model = dict(model_summary.get("r12_stock_drift_correction") or {})
            model_summary_row["r12_stock_drift_correction"] = {
                "status": str(model.get("status") or ""),
                "reference_family": model.get("reference_family"),
                "max_horizon_years": model.get("max_horizon_years"),
                "record_count": model.get("record_count"),
                "selected_correction_count": model.get("selected_correction_count"),
                "corrected_metrics": list(model.get("corrected_metrics") or []),
            }
        if model_summary.get("process_split_transition"):
            process = dict(model_summary.get("process_split_transition") or {})
            transition = dict(process.get("base_transition_process") or {})
            diagnosed_model = dict(transition.get("diagnosed_stock_transition") or {})
            art_model = dict(transition.get("art_stock_transition") or {})
            capacity = dict(process.get("art_initiation_capacity_model") or {})
            anatomy = dict(process.get("transition_residual_anatomy") or {})
            model_summary_row["process_split_transition"] = {
                "status": str(process.get("status") or ""),
                "diagnosis_flow_coefficient": diagnosed_model.get("diagnosis_flow_coefficient"),
                "diagnosed_removal_fraction": diagnosed_model.get("diagnosed_removal_fraction"),
                "diagnosed_reporting_era_count": diagnosed_model.get("era_count"),
                "diagnosis_linkage_coefficient": art_model.get("diagnosis_linkage_coefficient"),
                "diagnosed_gap_linkage_coefficient": art_model.get("diagnosed_gap_linkage_coefficient"),
                "art_removal_fraction": art_model.get("art_removal_fraction"),
                "selected_art_lag_quarters": art_model.get("selected_lag_quarters"),
                "art_capacity_status": str(capacity.get("status") or ""),
                "art_capacity_fraction": capacity.get("global_capacity_fraction"),
                "art_capacity_row_count": capacity.get("row_count"),
                "mean_abs_diagnosed_residual": anatomy.get("mean_abs_diagnosed_residual"),
                "mean_abs_art_residual": anatomy.get("mean_abs_art_residual"),
            }
            model_summary_row["raw_process_split_transition"] = {
                "transition_residual_anatomy": anatomy,
                "art_initiation_capacity_model": dict(process.get("art_initiation_capacity_model") or {}),
            }
        if model_summary.get("residual_source_model"):
            model = dict(model_summary.get("residual_source_model") or {})
            model_by_metric = dict(model.get("model_by_metric") or {})
            model_summary_row["residual_source_model"] = {
                "status": str(model.get("status") or ""),
                "record_count": model.get("record_count"),
                "selected_context_by_metric": {
                    metric_name: dict(metric_model).get("selected_context")
                    for metric_name, metric_model in sorted(model_by_metric.items())
                    if isinstance(metric_model, dict)
                },
                "selected_mean_abs_residual_by_metric": {
                    metric_name: dict(metric_model).get("selected_mean_abs_residual")
                    for metric_name, metric_model in sorted(model_by_metric.items())
                    if isinstance(metric_model, dict)
                },
            }
            model_summary_row["raw_residual_source_model"] = {
                "context_score_rows": list(model.get("context_score_rows") or []),
                "model_by_metric": model_by_metric,
            }
        model_summaries.append(model_summary_row)
    metric_anatomy: list[dict[str, Any]] = []
    for metric_name in R11_EVALUATION_METRICS:
        candidate_means = candidate_metric_accumulator.get(metric_name, [])
        carry_means = carry_metric_accumulator.get(metric_name, [])
        candidate_worsts = candidate_worst_accumulator.get(metric_name, [])
        carry_worsts = carry_worst_accumulator.get(metric_name, [])
        candidate_mean = None if not candidate_means else float(np.mean(np.asarray(candidate_means, dtype=np.float64)))
        carry_mean = None if not carry_means else float(np.mean(np.asarray(carry_means, dtype=np.float64)))
        candidate_worst = None if not candidate_worsts else float(np.max(np.asarray(candidate_worsts, dtype=np.float64)))
        carry_worst = None if not carry_worsts else float(np.max(np.asarray(carry_worsts, dtype=np.float64)))
        metric_anatomy.append(
            {
                "metric_name": metric_name,
                "entry_count": len(candidate_means),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "worst_candidate_minus_carry_forward_norm_error": None
                if candidate_worst is None or carry_worst is None
                else float(candidate_worst - carry_worst),
            }
        )
    conditional_rate_anatomy: list[dict[str, Any]] = []
    for spec in BACK_HALF_RATE_SPECS:
        rate_id = str(spec["rate_id"])
        candidate_means = candidate_rate_accumulator.get(rate_id, [])
        carry_means = carry_rate_accumulator.get(rate_id, [])
        candidate_worsts = candidate_rate_worst_accumulator.get(rate_id, [])
        carry_worsts = carry_rate_worst_accumulator.get(rate_id, [])
        candidate_mean = None if not candidate_means else float(np.mean(np.asarray(candidate_means, dtype=np.float64)))
        carry_mean = None if not carry_means else float(np.mean(np.asarray(carry_means, dtype=np.float64)))
        candidate_worst = None if not candidate_worsts else float(np.max(np.asarray(candidate_worsts, dtype=np.float64)))
        carry_worst = None if not carry_worsts else float(np.max(np.asarray(carry_worsts, dtype=np.float64)))
        conditional_rate_anatomy.append(
            {
                "rate_id": rate_id,
                "numerator_metric": str(spec["numerator_metric"]),
                "denominator_metric": str(spec["denominator_metric"]),
                "entry_count": len(candidate_means),
                "candidate_mean_rate_error": candidate_mean,
                "carry_forward_mean_rate_error": carry_mean,
                "candidate_minus_carry_forward_mean_rate_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "worst_candidate_minus_carry_forward_rate_error": None
                if candidate_worst is None or carry_worst is None
                else float(candidate_worst - carry_worst),
            }
        )
    score_summary = _score_branch_rows(split_rows)
    stock_gate = stock_consistency_gate({"metric_anatomy": metric_anatomy})
    rate_gate = _conditional_rate_gate(conditional_rate_anatomy)
    one_year_blockers: list[str] = []
    candidate_mean = _finite_float(score_summary.get("candidate_mean_mae"))
    carry_mean = _finite_float(score_summary.get("carry_forward_mean_mae"))
    if candidate_mean is None or carry_mean is None:
        one_year_blockers.append("candidate_or_carry_forward_not_evaluable")
    elif candidate_mean >= carry_mean:
        one_year_blockers.append("candidate_mean_not_better_than_carry_forward")
    r10_blockers: list[str] = []
    if r10_reference_mae is not None and candidate_mean is not None and candidate_mean >= r10_reference_mae:
        r10_blockers.append("candidate_mean_not_better_than_r10_reference")
    rate_blockers = list(rate_gate.get("blockers") or [])
    promotion_blockers = list(one_year_blockers) + list(stock_gate.get("blockers") or []) + rate_blockers + r10_blockers
    return {
        "schema_version": "phase3_dynamic.r11_state_filter_report.v1",
        "generated_at": _generated_at(),
        "experiment_id": experiment_id,
        "family": family,
        "rows": split_rows,
        "metric_anatomy": metric_anatomy,
        "conditional_rate_anatomy": conditional_rate_anatomy,
        "model_summaries": model_summaries,
        "one_year_gate": {
            "schema_version": "phase3_dynamic.r11_one_year_gate.v1",
            "status": "pass" if not one_year_blockers else "fail",
            "blockers": one_year_blockers,
            **score_summary,
            "r10_comparable_candidate_mean_mae": _score_branch_rows(
                [
                    {
                        "candidate_mae": row.get("r10_comparable_candidate_mae"),
                        "carry_forward_mae": row.get("r10_comparable_carry_forward_mae"),
                    }
                    for row in split_rows
                ]
            ).get("candidate_mean_mae"),
            "r10_comparable_carry_forward_mean_mae": _score_branch_rows(
                [
                    {
                        "candidate_mae": row.get("r10_comparable_candidate_mae"),
                        "carry_forward_mae": row.get("r10_comparable_carry_forward_mae"),
                    }
                    for row in split_rows
                ]
            ).get("carry_forward_mean_mae"),
            "r10_comparable_entry_count": int(sum(int(row.get("r10_comparable_entry_count") or 0) for row in split_rows)),
            "r10_comparable_metric_scope": list(R10_COMPARABLE_METRICS),
            "contract": "blocked-origin one-year sparse cascade score over D/A/T/V and diagnosis flow",
        },
        "stock_consistency_gate": stock_gate,
        "conditional_rate_gate": rate_gate,
        "r10_reference": {
            "available": r10_reference_mae is not None,
            "reference_quarterly_mean_mae": r10_reference_mae,
        },
        "promotion_gate": {
            "status": "promote_full_cascade_claim" if not promotion_blockers else "reject_full_cascade_claim",
            "promotion_eligible": not promotion_blockers,
            "blockers": promotion_blockers,
            "claim_boundary": "R11 candidate must beat carry-forward, guarded stock consistency, and R10 before any full-cascade claim",
        },
        "contract": "train-origin-safe sparse state/readout filter; target values are not used in holdout prediction",
    }


def _multi_horizon_gate(horizon_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    for row in horizon_rows:
        horizon = int(row.get("horizon_years") or 0)
        candidate_mean = _finite_float(row.get("candidate_mean_mae"))
        carry_mean = _finite_float(row.get("carry_forward_mean_mae"))
        if int(row.get("split_count") or 0) == 0:
            blockers.append(f"h{horizon}_not_evaluable")
        if candidate_mean is None or carry_mean is None:
            blockers.append(f"h{horizon}_candidate_or_carry_forward_missing")
        elif candidate_mean >= carry_mean:
            blockers.append(f"h{horizon}_candidate_mean_not_better_than_carry_forward")
        if str(row.get("stock_consistency_status") or "") != "pass":
            blockers.append(f"h{horizon}_stock_consistency_failed")
        if str(row.get("conditional_rate_status") or "") != "pass":
            blockers.append(f"h{horizon}_conditional_rate_failed")
    return {
        "schema_version": "phase3_dynamic.r11_multi_horizon_gate.v1",
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "horizon_count": len(horizon_rows),
        "contract": "multi-horizon lifted gate requires every blocked horizon to beat carry-forward while preserving stock and conditional-rate gates",
    }


def _r10_lifted_gate(horizon_rows: list[dict[str, Any]], r10_reference_mae: float | None) -> dict[str, Any]:
    blockers: list[str] = []
    horizon_gate_rows: list[dict[str, Any]] = []
    for row in horizon_rows:
        horizon = int(row.get("horizon_years") or 0)
        reference = _finite_float(row.get("r10_horizon_reference_mae"))
        reference_source = str(row.get("r10_horizon_reference_source") or "horizon_matched")
        if reference is None:
            reference = r10_reference_mae
            reference_source = "legacy_scalar_fallback"
        candidate_mean = _finite_float(row.get("r10_comparable_candidate_mean_mae"))
        if candidate_mean is None:
            candidate_mean = _finite_float(row.get("candidate_mean_mae"))
        row_blockers: list[str] = []
        if reference is None:
            row_blockers.append("missing_r10_reference")
        if candidate_mean is None:
            row_blockers.append("candidate_missing")
        elif reference is not None and candidate_mean >= float(reference):
            row_blockers.append("candidate_mean_not_better_than_r10_reference")
        if row_blockers:
            blockers.extend([f"h{horizon}_{blocker}" for blocker in row_blockers])
        horizon_gate_rows.append(
            {
                "horizon_years": horizon,
                "candidate_r10_comparable_mean_mae": candidate_mean,
                "r10_reference_mae": reference,
                "candidate_minus_r10_reference_mae": None if candidate_mean is None or reference is None else float(candidate_mean - float(reference)),
                "reference_source": reference_source,
                "reference_experiment_id": row.get("r10_horizon_reference_experiment_id"),
                "status": "pass" if not row_blockers else "fail",
                "blockers": row_blockers,
            }
        )
    return {
        "schema_version": "phase3_dynamic.r11_r10_lifted_gate.v1",
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "legacy_scalar_reference_quarterly_mean_mae": r10_reference_mae,
        "horizon_rows": horizon_gate_rows,
        "contract": (
            "R10-level lifted gate compares each horizon against a horizon-matched frozen R10-family replay "
            "when available; candidate scoring uses the legacy R10-comparable metric scope, with the old scalar "
            "reference used only as a documented fallback."
        ),
    }


def _r11_multi_horizon_report(
    *,
    experiment_id: str,
    family: str,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
    r10_reference_mae: float | None,
    r10_horizon_replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    horizon_rows: list[dict[str, Any]] = []
    horizon_reports: dict[str, dict[str, Any]] = {}
    metric_horizon_anatomy: list[dict[str, Any]] = []
    rate_horizon_anatomy: list[dict[str, Any]] = []
    for horizon in horizons:
        splits = rolling_origin_splits(
            rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon),
        )
        report = _r11_candidate_report(
            experiment_id=f"{experiment_id}-H{int(horizon)}",
            family=family,
            rows=rows,
            splits=splits,
            r10_reference_mae=r10_reference_mae,
        )
        one_year_gate = dict(report.get("one_year_gate") or {})
        stock_gate = dict(report.get("stock_consistency_gate") or {})
        rate_gate = dict(report.get("conditional_rate_gate") or {})
        r10_horizon_reference = _r10_reference_for_horizon(r10_horizon_replay, int(horizon))
        r10_horizon_mae = _finite_float(r10_horizon_reference.get("reference_quarterly_mean_mae"))
        candidate_r10_comparable = _finite_float(one_year_gate.get("r10_comparable_candidate_mean_mae"))
        carry_r10_comparable = _finite_float(one_year_gate.get("r10_comparable_carry_forward_mean_mae"))
        for row in list(report.get("metric_anatomy") or []):
            if not isinstance(row, dict):
                continue
            metric_horizon_anatomy.append(
                {
                    "horizon_years": int(horizon),
                    **dict(row),
                }
            )
        for row in list(report.get("conditional_rate_anatomy") or []):
            if not isinstance(row, dict):
                continue
            rate_horizon_anatomy.append(
                {
                    "horizon_years": int(horizon),
                    **dict(row),
                }
            )
        candidate_mean = _finite_float(one_year_gate.get("candidate_mean_mae"))
        carry_mean = _finite_float(one_year_gate.get("carry_forward_mean_mae"))
        horizon_rows.append(
            {
                "horizon_years": int(horizon),
                "split_count": int(one_year_gate.get("split_count") or 0),
                "candidate_mean_mae": candidate_mean,
                "carry_forward_mean_mae": carry_mean,
                "candidate_minus_carry_forward_mean_mae": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "candidate_worst_mae": _finite_float(one_year_gate.get("candidate_worst_mae")),
                "carry_forward_worst_mae": _finite_float(one_year_gate.get("carry_forward_worst_mae")),
                "r10_comparable_candidate_mean_mae": candidate_r10_comparable,
                "r10_comparable_carry_forward_mean_mae": carry_r10_comparable,
                "r10_comparable_metric_scope": list(one_year_gate.get("r10_comparable_metric_scope") or R10_COMPARABLE_METRICS),
                "stock_consistency_status": str(stock_gate.get("status") or "not_evaluable"),
                "conditional_rate_status": str(rate_gate.get("status") or "not_evaluable"),
                "r10_horizon_reference_mae": r10_horizon_mae,
                "r10_horizon_reference_experiment_id": r10_horizon_reference.get("reference_experiment_id"),
                "r10_horizon_reference_source": "horizon_matched_replay"
                if r10_horizon_mae is not None
                else "legacy_scalar_fallback",
                "candidate_minus_r10_reference_mae": None
                if candidate_r10_comparable is None or (r10_horizon_mae is None and r10_reference_mae is None)
                else float(candidate_r10_comparable - float(r10_horizon_mae if r10_horizon_mae is not None else r10_reference_mae)),
            }
        )
        horizon_reports[f"h{int(horizon)}"] = report
    lifted_gate = _multi_horizon_gate(horizon_rows)
    r10_gate = _r10_lifted_gate(horizon_rows, r10_reference_mae)
    candidate_values = [
        float(row["candidate_mean_mae"])
        for row in horizon_rows
        if _finite_float(row.get("candidate_mean_mae")) is not None
    ]
    carry_values = [
        float(row["carry_forward_mean_mae"])
        for row in horizon_rows
        if _finite_float(row.get("carry_forward_mean_mae")) is not None
    ]
    promotion_blockers = list(lifted_gate.get("blockers") or []) + list(r10_gate.get("blockers") or [])
    if family == "trajectory_shape_head":
        branch_contract = (
            "R11-16 is a blocked multi-horizon replay of train-origin metric/lead residual-shape corrections; "
            "it is rejected if residual freedom breaks stock or conditional-rate gates"
        )
    elif family == "constrained_trajectory_shape_head":
        branch_contract = (
            "R11-17 is a blocked multi-horizon replay of constrained residual-shape corrections; "
            "diagnosed stock is anchored, only ART and diagnosis flow are directly corrected, "
            "and VL/suppression are regenerated from the R11-14 conditional-rate process before gates are scored"
        )
    elif family == "horizon_adaptive_constrained_shape_head":
        branch_contract = (
            "R11-18 is a blocked multi-horizon replay of a train-origin horizon-adaptive constrained shape selector; "
            "each horizon chooses one predeclared R10-scope correction policy, while the full stock cone and "
            "conditional-rate gates remain mandatory promotion gates"
        )
    elif family == "datv_transition_process":
        branch_contract = (
            "R11-19 is a blocked multi-horizon replay of an explicit D/A transition process: diagnosed-stock "
            "reconciliation, D_to_A linkage, ART retention, and R11-14 conditional VL/suppression rates are fitted "
            "from train-window stock-flow transitions only"
        )
    elif family == "era_datv_transition_process":
        branch_contract = (
            "R11-20 is a blocked multi-horizon replay of an era-stratified D/A transition process: diagnosed-stock "
            "and ART recurrences include explicit removal fractions plus observation-support-era reporting shifts, "
            "while VL/suppression remain governed by the R11-14 conditional-rate process"
        )
    elif family == "horizon_family_selector":
        branch_contract = "R11-21 selects R11-17 or R11-20 by train-origin horizon evidence under the same stock/rate/R10 gates"
    elif family == "diagnosis_flow_input_repair_process":
        branch_contract = "R11-22 repairs diagnosis-flow inputs from stock-flow reconciliation before the era transition process"
    elif family == "support_era_diagnosis_flow_process":
        branch_contract = "R11-23 applies support-era diagnosis-flow reporting adjustment before the era transition process"
    elif family == "stock_flow_reconciliation_process":
        branch_contract = "R11-24 explicitly reconciles diagnosed-stock changes against diagnosis flow before transition replay"
    elif family == "diagnosed_reporting_bias_process":
        branch_contract = "R11-25 separates diagnosed-stock reporting bias from biological transition while preserving back-half rates"
    elif family == "art_horizon_selector_process":
        branch_contract = "R11-26 selects the horizon process family using ART-specific train-origin evidence"
    elif family == "diagnosis_lag_stock_process":
        branch_contract = "R11-27 pushes diagnosis-flow lag structure into diagnosed/ART stock transitions"
    elif family == "multi_horizon_weighted_process":
        branch_contract = "R11-28 combines R11-17 and R11-20 using train-origin inverse-error weights"
    elif family == "r12_long_horizon_stock_shape_process":
        branch_contract = (
            "R12-01 promotes R11-28 as the frozen research reference and applies only leave-origin validated "
            "long-horizon stock-shape corrections to diagnosed_plhiv and alive_on_art; no diagnosis-flow, VL, "
            "or suppression endpoint correction is allowed, and the same stock/rate/R10 gates remain mandatory"
        )
    elif family == "r12_da_process_split_transition":
        branch_contract = (
            "R12-02 promotes R11-28 as the frozen research reference and tests a process-split D/A transition: "
            "diagnosed-stock inflow/removal/reporting shifts and ART delayed linkage, initiation capacity, "
            "retention/removal, and reporting shifts. Back-half VL/suppression rates remain conditional-rate "
            "observations and the same stock/rate/R10 gates remain mandatory."
        )
    elif family == "r12_da_residual_source_process":
        branch_contract = (
            "R12-03 uses the R12-02 residual anatomy to test D/A residual alignment with source family, support "
            "signature, monthly reporting intensity, and train-derived backlog/rebound flow regime. Only selected "
            "train-validated residual-source corrections are fed back into the D/A process outputs; VL/suppression "
            "remain conditional-rate observations and the same stock/rate/R10/reference gates remain mandatory."
        )
    elif family == "r12_route_aware_two_head_process":
        branch_contract = (
            "R12-08 is a route-aware two-head candidate: 1y/2y forecasts use the locked R11-28 program-nowcast head, "
            "while 3y/5y forecasts use the R12-01 annual-anchor trajectory head. The head switch is determined only "
            "by forecast lead time and must pass route-specific nowcast and trajectory gates before any claim is promoted."
        )
    elif family == "r12_stock_cone_safe_annual_trajectory_process":
        branch_contract = (
            "R12-09 is a stock-cone-safe annual trajectory head: the locked R11-28 backbone remains active for all "
            "non-annual evidence, while slide annual-anchor diagnosed_plhiv/alive_on_art rows may use a train-selected "
            "annual stock head only if annual-anchor backtests improve mean D/A error without worsening worst-case "
            "D/A error versus R11-28 or carry-forward. Program nowcasting evidence is not mutated."
        )
    elif family == "r12_program_nowcast_mixed_quarterly_process":
        branch_contract = (
            "R12-10 freezes the R12-09 annual-anchor route and targets only DOH quarterly/monthly program evidence. "
            "A train-origin selector may replace diagnosed_plhiv, alive_on_art, and new_diagnosed_cases_period on "
            "program rows only; stock-cone and conditional-rate gates remain mandatory."
        )
    elif family == "r10_scope_teacher_stock_process":
        branch_contract = "R11-29 constrains R10-style readout shape with D/A stocks from the era transition process"
    elif family == "conditional_rate_horizon_selector":
        branch_contract = "R11-32 selects the back-half trajectory family using VL/suppression train-origin evidence"
    else:
        branch_contract = "R11-15 is a blocked multi-horizon replay of R11-14; it adds no new predictive freedom beyond horizon evaluation"
    return {
        "schema_version": "phase3_dynamic.r11_multi_horizon_lifted_report.v1",
        "generated_at": _generated_at(),
        "experiment_id": experiment_id,
        "family": family,
        "horizons": list(horizons),
        "horizon_rows": horizon_rows,
        "metric_horizon_anatomy": metric_horizon_anatomy,
        "rate_horizon_anatomy": rate_horizon_anatomy,
        "horizon_reports": horizon_reports,
        "multi_horizon_gate": lifted_gate,
        "r10_lifted_gate": r10_gate,
        "summary": {
            "candidate_mean_mae_across_horizons": None
            if not candidate_values
            else float(np.mean(np.asarray(candidate_values, dtype=np.float64))),
            "carry_forward_mean_mae_across_horizons": None
            if not carry_values
            else float(np.mean(np.asarray(carry_values, dtype=np.float64))),
            "candidate_minus_carry_forward_across_horizons": None
            if not candidate_values or not carry_values
            else float(np.mean(np.asarray(candidate_values, dtype=np.float64)) - np.mean(np.asarray(carry_values, dtype=np.float64))),
            "r10_reference_mae": r10_reference_mae,
            "r10_horizon_reference_mae_by_horizon": {
                str(int(row.get("horizon_years") or 0)): row.get("r10_horizon_reference_mae")
                for row in horizon_rows
            },
            "r10_comparable_candidate_mean_mae_across_horizons": None
            if not [
                row
                for row in horizon_rows
                if _finite_float(row.get("r10_comparable_candidate_mean_mae")) is not None
            ]
            else float(
                np.mean(
                    np.asarray(
                        [
                            float(row["r10_comparable_candidate_mean_mae"])
                            for row in horizon_rows
                            if _finite_float(row.get("r10_comparable_candidate_mean_mae")) is not None
                        ],
                        dtype=np.float64,
                    )
                )
            ),
        },
        "promotion_gate": {
            "status": "promote_lifted_full_cascade_claim" if not promotion_blockers else "reject_lifted_full_cascade_claim",
            "promotion_eligible": not promotion_blockers,
            "blockers": promotion_blockers,
            "claim_boundary": f"{experiment_id} must beat carry-forward at all horizons, preserve stock/rate gates, and beat the frozen R10 reference at all horizons",
        },
        "contract": branch_contract,
    }


def _summarize_r11_multi_horizon_branch(
    *,
    experiment_id: str,
    title: str,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    lifted_gate = dict(report.get("multi_horizon_gate") or {})
    r10_gate = dict(report.get("r10_lifted_gate") or {})
    promotion_gate = dict(report.get("promotion_gate") or {})
    summary = dict(report.get("summary") or {})
    promotion_eligible = bool(promotion_gate.get("promotion_eligible"))
    lifted_pass = str(lifted_gate.get("status") or "") == "pass"
    r10_pass = str(r10_gate.get("status") or "") == "pass"
    horizon_rows = list(report.get("horizon_rows") or [])
    one_year_row = next((dict(row) for row in horizon_rows if int(dict(row).get("horizon_years") or 0) == 1), {})
    horizon_reference_values = [
        float(row.get("r10_horizon_reference_mae"))
        for row in horizon_rows
        if isinstance(row, dict) and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r10_reference_summary = (
        float(np.mean(np.asarray(horizon_reference_values, dtype=np.float64)))
        if horizon_reference_values
        else _finite_float(summary.get("r10_reference_mae"))
    )
    if promotion_eligible:
        decision = "keep_as_full_cascade_candidate"
        kept_claim = "lifted_full_cascade_challenger"
    elif lifted_pass:
        decision = "keep_for_next_wave"
        kept_claim = "multi_horizon_lifted_readout_not_yet_r10_champion"
    else:
        decision = "reject_for_promotion"
        kept_claim = "diagnostic_only"
    return {
        "experiment_id": experiment_id,
        "title": title,
        "family": str(report.get("family") or ""),
        "artifact_path": path.as_posix(),
        "artifact_sha256": _sha256(path),
        "one_year_status": "pass"
        if _finite_float(one_year_row.get("candidate_minus_carry_forward_mean_mae")) is not None
        and float(one_year_row.get("candidate_minus_carry_forward_mean_mae")) < 0.0
        else "fail",
        "annual_status": "not_applicable",
        "lifted_status": "pass" if lifted_pass and r10_pass else "fail",
        "stock_consistency_status": "pass" if lifted_pass else "fail",
        "candidate_mae": _finite_float(summary.get("candidate_mean_mae_across_horizons")),
        "carry_forward_mae": _finite_float(summary.get("carry_forward_mean_mae_across_horizons")),
        "r10_reference_mae": r10_reference_summary,
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": list(promotion_gate.get("blockers") or []),
        "contract": str(report.get("contract") or "lifted trajectory branch evaluated over 1-, 3-, and 5-year blocked horizons"),
    }


def _summarize_r11_candidate_branch(
    *,
    experiment_id: str,
    title: str,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    one_year_gate = dict(report.get("one_year_gate") or {})
    stock_gate = dict(report.get("stock_consistency_gate") or {})
    promotion_gate = dict(report.get("promotion_gate") or {})
    r10 = dict(report.get("r10_reference") or {})
    promotion_eligible = bool(promotion_gate.get("promotion_eligible"))
    one_year_pass = str(one_year_gate.get("status") or "") == "pass"
    stock_pass = str(stock_gate.get("status") or "") == "pass"
    if promotion_eligible:
        decision = "keep_as_full_cascade_candidate"
        kept_claim = "candidate_full_cascade_challenger"
    elif one_year_pass and stock_pass:
        decision = "keep_for_next_wave"
        kept_claim = "stock_consistent_sparse_filter_not_yet_r10_champion"
    else:
        decision = "reject_for_promotion"
        kept_claim = "diagnostic_only"
    return {
        "experiment_id": experiment_id,
        "title": title,
        "family": str(report.get("family") or ""),
        "artifact_path": path.as_posix(),
        "artifact_sha256": _sha256(path),
        "one_year_status": str(one_year_gate.get("status") or "not_evaluable"),
        "annual_status": "not_applicable",
        "lifted_status": "not_applicable",
        "stock_consistency_status": str(stock_gate.get("status") or "not_evaluable"),
        "candidate_mae": _finite_float(one_year_gate.get("candidate_mean_mae")),
        "carry_forward_mae": _finite_float(one_year_gate.get("carry_forward_mean_mae")),
        "r10_reference_mae": _finite_float(r10.get("reference_quarterly_mean_mae")),
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": list(promotion_gate.get("blockers") or []),
        "contract": "new R11 fitted sparse state/readout branch evaluated under the same stock-consistency table",
    }


def _infrastructure_row(
    *,
    experiment_id: str,
    title: str,
    decision: str,
    kept_claim: str,
    artifact_path: Path,
    status: str,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "title": title,
        "family": "R11_observation_first_sparse_state_space_reconciliation",
        "artifact_path": artifact_path.as_posix(),
        "artifact_sha256": _sha256(artifact_path),
        "one_year_status": "not_applicable",
        "annual_status": "not_applicable",
        "lifted_status": "not_applicable",
        "stock_consistency_status": status,
        "candidate_mae": None,
        "carry_forward_mae": None,
        "r10_reference_mae": None,
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": list(blockers or []),
        "contract": "R11 first-batch infrastructure experiment",
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# R11 Experiment Comparison - {_generated_at()[:10]}",
        "",
        "| Experiment | Family | One-year | Lifted/R10 | Stock gate | Candidate | Carry-forward | R10 | Decision | Kept claim | Main blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        blockers = ", ".join([str(item) for item in list(row.get("blockers") or [])[:4]])
        if len(list(row.get("blockers") or [])) > 4:
            blockers += ", ..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment_id") or ""),
                    str(row.get("family") or ""),
                    str(row.get("one_year_status") or ""),
                    str(row.get("lifted_status") or ""),
                    str(row.get("stock_consistency_status") or ""),
                    "" if row.get("candidate_mae") is None else f"{float(row['candidate_mae']):.6f}",
                    "" if row.get("carry_forward_mae") is None else f"{float(row['carry_forward_mae']):.6f}",
                    "" if row.get("r10_reference_mae") is None else f"{float(row['r10_reference_mae']):.6f}",
                    str(row.get("decision") or ""),
                    str(row.get("kept_claim") or ""),
                    blockers,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The kept items are infrastructure, diagnostics, references, or next-wave candidates. A full-cascade/R10-superiority claim is not promoted unless the locked carry-forward, stock/rate, and horizon-matched R10 gates all pass. Diagnostic rows are retained only to explain evidence support, lineage adequacy, or failure anatomy; they do not change predictions and must not be cited as champion models.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "experiment_id",
        "family",
        "one_year_status",
        "annual_status",
        "lifted_status",
        "stock_consistency_status",
        "candidate_mae",
        "carry_forward_mae",
        "r10_reference_mae",
        "decision",
        "kept_claim",
        "artifact_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    scored = [
        row
        for row in rows
        if _finite_float(row.get("candidate_mae")) is not None
        and _finite_float(row.get("carry_forward_mae")) is not None
    ]
    if not scored:
        return
    labels = [str(row.get("experiment_id") or "") for row in scored]
    candidate = np.asarray([float(row["candidate_mae"]) for row in scored], dtype=np.float64)
    carry = np.asarray([float(row["carry_forward_mae"]) for row in scored], dtype=np.float64)
    r10 = np.asarray(
        [
            np.nan if _finite_float(row.get("r10_reference_mae")) is None else float(row["r10_reference_mae"])
            for row in scored
        ],
        dtype=np.float64,
    )
    y = np.arange(len(scored), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.barh(y - 0.22, carry, height=0.2, color="#9aa4b2", label="carry-forward")
    ax.barh(y, candidate, height=0.2, color="#315f72", label="candidate")
    ax.barh(y + 0.22, r10, height=0.2, color="#b4543a", label="R10 reference")
    for idx, row in enumerate(scored):
        decision = str(row.get("decision") or "")
        color = "#1f7a4d" if decision.startswith("keep") else "#9b2f2f"
        ax.text(
            max(candidate[idx], carry[idx], 0.0) * 1.02,
            y[idx],
            decision,
            va="center",
            ha="left",
            fontsize=8,
            color=color,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("normalized MAE, lower is better")
    ax.set_title("R11 first-batch gate: candidates vs carry-forward and R10")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_back_half_dashboard(path: Path, r11_13: dict[str, Any], r11_14: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = list(CASCADE_STOCK_METRICS)
    metric_index_13 = {
        str(row.get("metric_name") or ""): dict(row)
        for row in list(r11_13.get("metric_anatomy") or [])
        if isinstance(row, dict)
    }
    metric_index_14 = {
        str(row.get("metric_name") or ""): dict(row)
        for row in list(r11_14.get("metric_anatomy") or [])
        if isinstance(row, dict)
    }
    r13_metric = np.asarray(
        [
            np.nan if _finite_float(metric_index_13.get(metric, {}).get("candidate_mean_norm_error")) is None else float(metric_index_13[metric]["candidate_mean_norm_error"])
            for metric in metrics
        ],
        dtype=np.float64,
    )
    r14_metric = np.asarray(
        [
            np.nan if _finite_float(metric_index_14.get(metric, {}).get("candidate_mean_norm_error")) is None else float(metric_index_14[metric]["candidate_mean_norm_error"])
            for metric in metrics
        ],
        dtype=np.float64,
    )
    carry_metric = np.asarray(
        [
            np.nan if _finite_float(metric_index_14.get(metric, {}).get("carry_forward_mean_norm_error")) is None else float(metric_index_14[metric]["carry_forward_mean_norm_error"])
            for metric in metrics
        ],
        dtype=np.float64,
    )
    rate_rows_13 = {
        str(row.get("rate_id") or ""): dict(row)
        for row in list(r11_13.get("conditional_rate_anatomy") or [])
        if isinstance(row, dict)
    }
    rate_rows_14 = {
        str(row.get("rate_id") or ""): dict(row)
        for row in list(r11_14.get("conditional_rate_anatomy") or [])
        if isinstance(row, dict)
    }
    rate_ids = [str(spec["rate_id"]) for spec in BACK_HALF_RATE_SPECS]
    r13_rate = np.asarray(
        [
            np.nan if _finite_float(rate_rows_13.get(rate_id, {}).get("candidate_mean_rate_error")) is None else float(rate_rows_13[rate_id]["candidate_mean_rate_error"])
            for rate_id in rate_ids
        ],
        dtype=np.float64,
    )
    r14_rate = np.asarray(
        [
            np.nan if _finite_float(rate_rows_14.get(rate_id, {}).get("candidate_mean_rate_error")) is None else float(rate_rows_14[rate_id]["candidate_mean_rate_error"])
            for rate_id in rate_ids
        ],
        dtype=np.float64,
    )
    carry_rate = np.asarray(
        [
            np.nan if _finite_float(rate_rows_14.get(rate_id, {}).get("carry_forward_mean_rate_error")) is None else float(rate_rows_14[rate_id]["carry_forward_mean_rate_error"])
            for rate_id in rate_ids
        ],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.26
    axes[0].bar(x - width, carry_metric, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, r13_metric, width=width, color="#315f72", label="R11-13")
    axes[0].bar(x + width, r14_metric, width=width, color="#2f7d59", label="R11-14")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["D", "ART", "VL tested", "suppressed"], rotation=0)
    axes[0].set_ylabel("mean normalized error")
    axes[0].set_title("Back-half count repair")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    rx = np.arange(len(rate_ids), dtype=np.float64)
    axes[1].bar(rx - width, carry_rate, width=width, color="#9aa4b2", label="conditional-rate carry")
    axes[1].bar(rx, r13_rate, width=width, color="#315f72", label="R11-13")
    axes[1].bar(rx + width, r14_rate, width=width, color="#2f7d59", label="R11-14")
    axes[1].set_xticks(rx)
    axes[1].set_xticklabels(["VL / ART", "suppressed / VL"], rotation=0)
    axes[1].set_ylabel("mean absolute rate error")
    axes[1].set_title("Third-95 conditional-rate gate")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("R11-14 back-half process: counts improve, rates stay carry-forward-gated", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_multi_horizon_dashboard(path: Path, r11_15: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [dict(row) for row in list(r11_15.get("horizon_rows") or [])]
    if not rows:
        return
    horizons = np.asarray([float(row.get("horizon_years") or 0.0) for row in rows], dtype=np.float64)
    candidate = np.asarray(
        [np.nan if _finite_float(row.get("candidate_mean_mae")) is None else float(row["candidate_mean_mae"]) for row in rows],
        dtype=np.float64,
    )
    carry = np.asarray(
        [np.nan if _finite_float(row.get("carry_forward_mean_mae")) is None else float(row["carry_forward_mean_mae"]) for row in rows],
        dtype=np.float64,
    )
    r10_reference = np.asarray(
        [
            np.nan if _finite_float(row.get("r10_horizon_reference_mae")) is None else float(row["r10_horizon_reference_mae"])
            for row in rows
        ],
        dtype=np.float64,
    )
    delta = candidate - carry
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    axes[0].plot(horizons, carry, marker="o", linewidth=2.2, color="#9aa4b2", label="carry-forward")
    axes[0].plot(horizons, candidate, marker="o", linewidth=2.2, color="#2f7d59", label="R11-15 / R11-14 lifted")
    if np.isfinite(r10_reference).any():
        axes[0].plot(horizons, r10_reference, marker="D", linewidth=1.8, color="#b4543a", label="R10 horizon replay")
    axes[0].set_xticks(horizons)
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("mean normalized MAE")
    axes[0].set_title("Lifted trajectory score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in delta]
    axes[1].bar([str(int(value)) for value in horizons], delta, color=colors)
    axes[1].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("candidate minus carry-forward")
    axes[1].set_title("Carry-forward improvement by horizon")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("R11-15 multi-horizon lifted gate: trajectory shape judged against horizon-matched R10", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_shape_head_dashboard(path: Path, r11_15: dict[str, Any], r11_16: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_15 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_15.get("horizon_rows") or [])}
    rows_16 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_16.get("horizon_rows") or [])}
    horizons = sorted(set(rows_15) | set(rows_16))
    if not horizons:
        return
    base = np.asarray(
        [
            np.nan if _finite_float(rows_15.get(h, {}).get("candidate_mean_mae")) is None else float(rows_15[h]["candidate_mean_mae"])
            for h in horizons
        ],
        dtype=np.float64,
    )
    shape = np.asarray(
        [
            np.nan if _finite_float(rows_16.get(h, {}).get("candidate_mean_mae")) is None else float(rows_16[h]["candidate_mean_mae"])
            for h in horizons
        ],
        dtype=np.float64,
    )
    carry = np.asarray(
        [
            np.nan if _finite_float(rows_16.get(h, {}).get("carry_forward_mean_mae")) is None else float(rows_16[h]["carry_forward_mean_mae"])
            for h in horizons
        ],
        dtype=np.float64,
    )
    anatomy = [
        dict(row)
        for row in list(r11_16.get("metric_horizon_anatomy") or [])
        if isinstance(row, dict) and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    metrics = list(R11_EVALUATION_METRICS)
    matrix = np.full((len(metrics), 2), np.nan, dtype=np.float64)
    for metric_index, metric_name in enumerate(metrics):
        for horizon_index, horizon in enumerate((3, 5)):
            values = [
                _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error"))
                for row in anatomy
                if int(row.get("horizon_years") or 0) == horizon and str(row.get("metric_name") or "") == metric_name
            ]
            values = [float(value) for value in values if value is not None]
            if values:
                matrix[metric_index, horizon_index] = float(np.mean(np.asarray(values, dtype=np.float64)))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.24
    axes[0].bar(x - width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, base, width=width, color="#315f72", label="R11-15 base")
    axes[0].bar(x + width, shape, width=width, color="#2f7d59", label="R11-16 shape head")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("mean normalized MAE")
    axes[0].set_title("Shape head vs base lifted readout")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    finite_matrix = matrix[np.isfinite(matrix)]
    color_limit = float(np.max(np.abs(finite_matrix))) if finite_matrix.size else 1.0
    image = axes[1].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-color_limit, vmax=color_limit)
    axes[1].set_yticks(np.arange(len(metrics)))
    axes[1].set_yticklabels(["D", "ART", "VL", "V", "diagnosis flow"])
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(["3y", "5y"])
    axes[1].set_title("R11-16 metric residual delta vs carry")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="candidate minus carry-forward")
    fig.suptitle("R11-16 trajectory-shape head: metric-by-horizon residual anatomy", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_constrained_shape_dashboard(
    path: Path,
    r11_15: dict[str, Any],
    r11_16: dict[str, Any],
    r11_17: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_15 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_15.get("horizon_rows") or [])}
    rows_16 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_16.get("horizon_rows") or [])}
    rows_17 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_17.get("horizon_rows") or [])}
    horizons = sorted(set(rows_15) | set(rows_16) | set(rows_17))
    if not horizons:
        return

    def _series(rows: dict[int, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if _finite_float(rows.get(h, {}).get(key)) is None else float(rows[h][key])
                for h in horizons
            ],
            dtype=np.float64,
        )

    carry = _series(rows_17, "carry_forward_mean_mae")
    base = _series(rows_15, "candidate_mean_mae")
    unconstrained = _series(rows_16, "candidate_mean_mae")
    constrained = _series(rows_17, "candidate_mean_mae")
    r10 = _series(rows_17, "r10_horizon_reference_mae")
    constrained_delta = constrained - carry

    anatomy = [
        dict(row)
        for row in list(r11_17.get("metric_horizon_anatomy") or [])
        if isinstance(row, dict) and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    metrics = list(R11_EVALUATION_METRICS)
    matrix = np.full((len(metrics), 2), np.nan, dtype=np.float64)
    for metric_index, metric_name in enumerate(metrics):
        for horizon_index, horizon in enumerate((3, 5)):
            values = [
                _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error"))
                for row in anatomy
                if int(row.get("horizon_years") or 0) == horizon and str(row.get("metric_name") or "") == metric_name
            ]
            values = [float(value) for value in values if value is not None]
            if values:
                matrix[metric_index, horizon_index] = float(np.mean(np.asarray(values, dtype=np.float64)))

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.19
    axes[0].bar(x - 1.5 * width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x - 0.5 * width, base, width=width, color="#315f72", label="R11-15 base")
    axes[0].bar(x + 0.5 * width, unconstrained, width=width, color="#be7a2b", label="R11-16 unconstrained")
    axes[0].bar(x + 1.5 * width, constrained, width=width, color="#2f7d59", label="R11-17 constrained")
    if np.isfinite(r10).any():
        axes[0].plot(x, r10, marker="D", linewidth=1.8, color="#111827", label="R10 horizon replay")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("mean normalized MAE")
    axes[0].set_title("Trajectory-shape candidates")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in constrained_delta]
    axes[1].bar([str(h) for h in horizons], constrained_delta, color=colors)
    axes[1].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("R11-17 minus carry-forward")
    axes[1].set_title("Constrained-head gain")
    axes[1].grid(axis="y", alpha=0.25)

    finite_matrix = matrix[np.isfinite(matrix)]
    color_limit = float(np.max(np.abs(finite_matrix))) if finite_matrix.size else 1.0
    image = axes[2].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-color_limit, vmax=color_limit)
    axes[2].set_yticks(np.arange(len(metrics)))
    axes[2].set_yticklabels(["D", "ART", "VL", "V", "diagnosis flow"])
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels(["3y", "5y"])
    axes[2].set_title("R11-17 metric delta vs carry")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="candidate minus carry-forward")
    fig.suptitle("R11-17 constrained trajectory-shape head: preserve stock cone and third-95 rates", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_horizon_adaptive_shape_dashboard(
    path: Path,
    r11_17: dict[str, Any],
    r11_18: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_17 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_17.get("horizon_rows") or [])}
    rows_18 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_18.get("horizon_rows") or [])}
    horizons = sorted(set(rows_17) | set(rows_18))
    if not horizons:
        return

    def _series(rows: dict[int, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if _finite_float(rows.get(h, {}).get(key)) is None else float(rows[h][key])
                for h in horizons
            ],
            dtype=np.float64,
        )

    r17 = _series(rows_17, "r10_comparable_candidate_mean_mae")
    r18 = _series(rows_18, "r10_comparable_candidate_mean_mae")
    r10 = _series(rows_18, "r10_horizon_reference_mae")
    carry = _series(rows_18, "r10_comparable_carry_forward_mean_mae")
    r18_minus_r10 = r18 - r10
    r18_minus_r17 = r18 - r17
    policy_by_horizon: dict[int, str] = {}
    for row in dict(r11_18.get("horizon_reports") or {}).values():
        if not isinstance(row, dict):
            continue
        experiment_id = str(row.get("experiment_id") or "")
        try:
            horizon = int(experiment_id.rsplit("-H", 1)[1])
        except (IndexError, ValueError):
            continue
        summaries = list(row.get("model_summaries") or [])
        selected = [
            str(dict(summary).get("horizon_adaptive_shape_selector", {}).get("selected_policy_id") or "")
            for summary in summaries
            if isinstance(summary, dict)
        ]
        selected = [item for item in selected if item]
        if selected:
            policy_by_horizon[horizon] = Counter(selected).most_common(1)[0][0]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.2
    axes[0].bar(x - 1.5 * width, carry, width=width, color="#9aa4b2", label="carry-forward scope")
    axes[0].bar(x - 0.5 * width, r17, width=width, color="#315f72", label="R11-17")
    axes[0].bar(x + 0.5 * width, r18, width=width, color="#2f7d59", label="R11-18")
    axes[0].plot(x, r10, marker="D", linewidth=1.8, color="#111827", label="R10 horizon replay")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("R10-comparable mean normalized MAE")
    axes[0].set_title("R10-scope trajectory score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r18_minus_r10]
    axes[1].bar([str(h) for h in horizons], r18_minus_r10, color=colors)
    axes[1].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("R11-18 minus matched R10")
    axes[1].set_title("Promotion-relevant R10 gap")
    axes[1].grid(axis="y", alpha=0.25)

    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r18_minus_r17]
    axes[2].bar([str(h) for h in horizons], r18_minus_r17, color=colors)
    axes[2].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[2].set_xlabel("blocked horizon, years")
    axes[2].set_ylabel("R11-18 minus R11-17")
    axes[2].set_title("Adaptive selector delta")
    axes[2].grid(axis="y", alpha=0.25)
    for index, horizon in enumerate(horizons):
        label = policy_by_horizon.get(horizon, "")
        if label:
            axes[2].text(index, 0.0, label.replace("_", "\n"), ha="center", va="bottom", fontsize=7, rotation=0)
    fig.suptitle("R11-18 horizon-adaptive constrained shape selector", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_transition_process_dashboard(
    path: Path,
    r11_17: dict[str, Any],
    r11_19: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_17 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_17.get("horizon_rows") or [])}
    rows_19 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_19.get("horizon_rows") or [])}
    horizons = sorted(set(rows_17) | set(rows_19))
    if not horizons:
        return

    def _series(rows: dict[int, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if _finite_float(rows.get(h, {}).get(key)) is None else float(rows[h][key])
                for h in horizons
            ],
            dtype=np.float64,
        )

    r17_full = _series(rows_17, "candidate_mean_mae")
    r19_full = _series(rows_19, "candidate_mean_mae")
    carry_full = _series(rows_19, "carry_forward_mean_mae")
    r17_scope = _series(rows_17, "r10_comparable_candidate_mean_mae")
    r19_scope = _series(rows_19, "r10_comparable_candidate_mean_mae")
    r10 = _series(rows_19, "r10_horizon_reference_mae")
    r19_minus_r10 = r19_scope - r10

    metric_rows = [
        dict(row)
        for row in list(r11_19.get("metric_horizon_anatomy") or [])
        if isinstance(row, dict) and int(row.get("horizon_years") or 0) in {3, 5}
    ]
    metrics = ["diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed", "new_diagnosed_cases_period"]
    matrix = np.full((len(metrics), 2), np.nan, dtype=np.float64)
    for metric_index, metric_name in enumerate(metrics):
        for horizon_index, horizon in enumerate((3, 5)):
            values = [
                _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error"))
                for row in metric_rows
                if int(row.get("horizon_years") or 0) == horizon and str(row.get("metric_name") or "") == metric_name
            ]
            values = [float(value) for value in values if value is not None]
            if values:
                matrix[metric_index, horizon_index] = float(np.mean(np.asarray(values, dtype=np.float64)))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.22
    axes[0].bar(x - width, carry_full, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, r17_full, width=width, color="#315f72", label="R11-17")
    axes[0].bar(x + width, r19_full, width=width, color="#2f7d59", label="R11-19 transition")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("full mean normalized MAE")
    axes[0].set_title("Full DATV trajectory")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, r17_scope, marker="o", linewidth=2.0, color="#315f72", label="R11-17 R10 scope")
    axes[1].plot(x, r19_scope, marker="o", linewidth=2.0, color="#2f7d59", label="R11-19 R10 scope")
    axes[1].plot(x, r10, marker="D", linewidth=1.8, color="#111827", label="matched R10")
    axes[1].bar(x, r19_minus_r10, width=0.18, alpha=0.25, color=["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r19_minus_r10])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(h) for h in horizons])
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("R10-comparable MAE")
    axes[1].set_title("R10-scope transition gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    finite_matrix = matrix[np.isfinite(matrix)]
    color_limit = float(np.max(np.abs(finite_matrix))) if finite_matrix.size else 1.0
    image = axes[2].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-color_limit, vmax=color_limit)
    axes[2].set_yticks(np.arange(len(metrics)))
    axes[2].set_yticklabels(["D", "ART", "VL", "V", "diagnosis flow"])
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels(["3y", "5y"])
    axes[2].set_title("R11-19 metric delta vs carry")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="candidate minus carry-forward")
    fig.suptitle("R11-19 transition-process branch: diagnosed stock, D_to_A linkage, ART retention", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_era_transition_process_dashboard(
    path: Path,
    r11_19: dict[str, Any],
    r11_20: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows_19 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_19.get("horizon_rows") or [])}
    rows_20 = {int(row.get("horizon_years") or 0): dict(row) for row in list(r11_20.get("horizon_rows") or [])}
    horizons = sorted(set(rows_19) | set(rows_20))
    if not horizons:
        return

    def _series(rows: dict[int, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if _finite_float(rows.get(h, {}).get(key)) is None else float(rows[h][key])
                for h in horizons
            ],
            dtype=np.float64,
        )

    r19_full = _series(rows_19, "candidate_mean_mae")
    r20_full = _series(rows_20, "candidate_mean_mae")
    carry_full = _series(rows_20, "carry_forward_mean_mae")
    r19_scope = _series(rows_19, "r10_comparable_candidate_mean_mae")
    r20_scope = _series(rows_20, "r10_comparable_candidate_mean_mae")
    r10 = _series(rows_20, "r10_horizon_reference_mae")
    r20_minus_r19 = r20_scope - r19_scope
    r20_minus_r10 = r20_scope - r10

    transition_summaries: list[dict[str, Any]] = []
    for report in dict(r11_20.get("horizon_reports") or {}).values():
        if not isinstance(report, dict):
            continue
        for summary in list(report.get("model_summaries") or []):
            if not isinstance(summary, dict):
                continue
            transition = dict(summary.get("transition_process") or {})
            if transition:
                transition_summaries.append(transition)
    removal_values = {
        "D removal": [
            float(value)
            for value in (_finite_float(row.get("diagnosed_removal_fraction")) for row in transition_summaries)
            if value is not None
        ],
        "ART removal": [
            float(value)
            for value in (_finite_float(row.get("art_removal_fraction")) for row in transition_summaries)
            if value is not None
        ],
    }
    removal_means = [
        np.nan if not values else float(np.mean(np.asarray(values, dtype=np.float64)))
        for values in removal_values.values()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.22
    axes[0].bar(x - width, carry_full, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, r19_full, width=width, color="#315f72", label="R11-19")
    axes[0].bar(x + width, r20_full, width=width, color="#2f7d59", label="R11-20 era process")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("full mean normalized MAE")
    axes[0].set_title("Full DATV trajectory")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, r19_scope, marker="o", linewidth=2.0, color="#315f72", label="R11-19 R10 scope")
    axes[1].plot(x, r20_scope, marker="o", linewidth=2.0, color="#2f7d59", label="R11-20 R10 scope")
    axes[1].plot(x, r10, marker="D", linewidth=1.8, color="#111827", label="matched R10")
    axes[1].bar(x, r20_minus_r10, width=0.18, alpha=0.25, color=["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r20_minus_r10])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(h) for h in horizons])
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("R10-comparable MAE")
    axes[1].set_title("R10-scope era-process gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r20_minus_r19]
    axes[2].bar([str(h) for h in horizons], r20_minus_r19, color=colors)
    axes[2].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    if any(np.isfinite(removal_means)):
        removal_text = "\n".join(
            f"{label}: {value:.4f}"
            for label, value in zip(removal_values, removal_means)
            if np.isfinite(value)
        )
        axes[2].text(
            0.98,
            0.05,
            f"mean removal\n{removal_text}",
            transform=axes[2].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#c9c9c9", "alpha": 0.9},
        )
    axes[2].set_xlabel("blocked horizon, years")
    axes[2].set_ylabel("R11-20 minus R11-19 R10-scope MAE")
    axes[2].set_title("Era-process delta and removals")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("R11-20 era-stratified transition process: removals plus reporting shifts", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _build_r11_30_shift_ablation_report(candidate_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for experiment_id, report in sorted(candidate_reports.items()):
        summary = dict(report.get("summary") or {})
        gate = dict(report.get("promotion_gate") or {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": str(report.get("family") or ""),
                "candidate_mean_mae_across_horizons": _finite_float(summary.get("candidate_mean_mae_across_horizons")),
                "r10_comparable_candidate_mean_mae_across_horizons": _finite_float(summary.get("r10_comparable_candidate_mean_mae_across_horizons")),
                "promotion_status": str(gate.get("status") or ""),
                "blockers": list(gate.get("blockers") or []),
            }
        )
    best = min(
        [row for row in rows if _finite_float(row.get("r10_comparable_candidate_mean_mae_across_horizons")) is not None],
        key=lambda row: float(row["r10_comparable_candidate_mean_mae_across_horizons"]),
        default=None,
    )
    return {
        "schema_version": "phase3_dynamic.r11_30_shift_ablation.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-30",
        "status": "completed" if rows else "not_evaluable",
        "decision": "keep_as_ablation_evidence",
        "best_r10_scope_experiment_id": None if best is None else best.get("experiment_id"),
        "rows": rows,
        "contract": "support/reporting shift ablation across the fitted R11 process family branches; diagnostic only, not a new model",
    }


def _build_r11_31_removal_sensitivity_report(candidate_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for experiment_id, report in sorted(candidate_reports.items()):
        removal_values: list[float] = []
        for horizon_report in dict(report.get("horizon_reports") or {}).values():
            if not isinstance(horizon_report, dict):
                continue
            for summary in list(horizon_report.get("model_summaries") or []):
                transition = dict(dict(summary).get("transition_process") or {})
                for key in ("diagnosed_removal_fraction", "art_removal_fraction"):
                    value = _finite_float(transition.get(key))
                    if value is not None:
                        removal_values.append(float(value))
        summary = dict(report.get("summary") or {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": str(report.get("family") or ""),
                "mean_fitted_removal_fraction": None
                if not removal_values
                else float(np.mean(np.asarray(removal_values, dtype=np.float64))),
                "removal_estimate_count": len(removal_values),
                "candidate_mean_mae_across_horizons": _finite_float(summary.get("candidate_mean_mae_across_horizons")),
            }
        )
    return {
        "schema_version": "phase3_dynamic.r11_31_removal_sensitivity.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-31",
        "status": "completed",
        "decision": "keep_as_sensitivity_evidence",
        "rows": rows,
        "contract": "diagnostic removal sensitivity summary across transition-process branches; removal remains a sensitivity claim unless it changes blocked-gate conclusions",
    }


def _build_r11_33_external_annual_challenge_report(validation_panel: dict[str, Any]) -> dict[str, Any]:
    rows = list(validation_panel.get("entries") or validation_panel.get("rows") or [])
    return {
        "schema_version": "phase3_dynamic.r11_33_external_annual_challenge.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-33",
        "status": "available" if rows else "not_available",
        "decision": "keep_as_external_validation_contract" if rows else "blocked_missing_external_panel",
        "entry_count": len(rows),
        "metric_counts": dict(Counter(str(row.get("metric_id") or row.get("metric_name") or "unknown") for row in rows if isinstance(row, dict))),
        "contract": "external annual series are validation-only challenge data; no candidate is tuned on these rows",
    }


def _build_r11_34_source_family_ablation_report(ledger: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in list(ledger.get("rows") or []) if isinstance(row, dict)]
    source_counts = Counter(str(row.get("source_id") or row.get("source_family") or "unknown") for row in rows)
    role_counts = Counter(str(row.get("observation_role") or "unknown") for row in rows)
    return {
        "schema_version": "phase3_dynamic.r11_34_source_family_ablation_readiness.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-34",
        "status": "ready" if len(source_counts) > 1 else "limited",
        "decision": "keep_as_ablation_readiness_contract",
        "source_family_counts": dict(sorted(source_counts.items())),
        "observation_role_counts": dict(sorted(role_counts.items())),
        "contract": "source-family ablation readiness artifact; true re-estimation ablation should be run only for source families with enough training support",
    }


def _build_r11_35_claim_card_report(comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [
        dict(row)
        for row in comparison_rows
        if str(row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ]
    kept_next_wave = [
        dict(row)
        for row in comparison_rows
        if str(row.get("decision") or "") == "keep_for_next_wave"
    ]
    return {
        "schema_version": "phase3_dynamic.r11_35_claim_card.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R11-35",
        "status": "no_full_cascade_claim" if not promoted else "candidate_full_cascade_claim_available",
        "decision": "keep_as_claim_card",
        "promoted_experiment_ids": [str(row.get("experiment_id") or "") for row in promoted],
        "next_wave_experiment_ids": [str(row.get("experiment_id") or "") for row in kept_next_wave],
        "claim_boundary": (
            "No R11 branch is a full-cascade/R10-superiority champion under the locked gates."
            if not promoted
            else "At least one R11 branch passed the full locked promotion gate."
        ),
        "contract": "publication claim card: every scientific claim must trace to comparison rows, run artifacts, and locked gates",
    }


def _build_r12_reference_lock(
    *,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    reference_report_path: Path,
    reference_report: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(reference_report.get("summary") or {})
    r10_gate = dict(reference_report.get("r10_lifted_gate") or {})
    return {
        "schema_version": "phase3_dynamic.r12_00_reference_lock.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-00",
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "promoted_reference_experiment_id": "R11-28",
        "promoted_reference_family": "multi_horizon_weighted_process",
        "reference_report_path": reference_report_path.as_posix(),
        "reference_report_sha256": _sha256(reference_report_path),
        "candidate_mean_mae_across_horizons": _finite_float(summary.get("candidate_mean_mae_across_horizons")),
        "carry_forward_mean_mae_across_horizons": _finite_float(summary.get("carry_forward_mean_mae_across_horizons")),
        "r10_comparable_candidate_mean_mae_across_horizons": _finite_float(
            summary.get("r10_comparable_candidate_mean_mae_across_horizons")
        ),
        "r10_gate_status": str(r10_gate.get("status") or ""),
        "r10_gate_blockers": list(r10_gate.get("blockers") or []),
        "decision": "promote_as_next_research_reference",
        "claim_boundary": (
            "R11-28 is promoted as the R12 research reference because it is the best stock/rate-consistent "
            "next-wave branch, not because it is a publication champion. R10-superiority remains blocked."
        ),
        "contract": "R12 candidates must beat or explain this locked R11-28 reference under the same blocked horizon gates",
    }


def _build_r12_02_process_split_anatomy_report(r12_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for horizon_key, horizon_report in dict(r12_report.get("horizon_reports") or {}).items():
        if not isinstance(horizon_report, dict):
            continue
        horizon = int(str(horizon_key).removeprefix("h") or 0)
        for summary in list(horizon_report.get("model_summaries") or []):
            if not isinstance(summary, dict):
                continue
            train_end_year = int(summary.get("train_end_year") or 0)
            process = dict(summary.get("process_split_transition") or {})
            rows.append(
                {
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "status": str(process.get("status") or ""),
                    "diagnosis_flow_coefficient": process.get("diagnosis_flow_coefficient"),
                    "diagnosed_removal_fraction": process.get("diagnosed_removal_fraction"),
                    "diagnosed_reporting_era_count": process.get("diagnosed_reporting_era_count"),
                    "diagnosis_linkage_coefficient": process.get("diagnosis_linkage_coefficient"),
                    "diagnosed_gap_linkage_coefficient": process.get("diagnosed_gap_linkage_coefficient"),
                    "art_removal_fraction": process.get("art_removal_fraction"),
                    "selected_art_lag_quarters": process.get("selected_art_lag_quarters"),
                    "art_capacity_status": process.get("art_capacity_status"),
                    "art_capacity_fraction": process.get("art_capacity_fraction"),
                    "art_capacity_row_count": process.get("art_capacity_row_count"),
                    "mean_abs_diagnosed_residual": process.get("mean_abs_diagnosed_residual"),
                    "mean_abs_art_residual": process.get("mean_abs_art_residual"),
                }
            )
        for summary in list(horizon_report.get("model_summaries") or []):
            if not isinstance(summary, dict):
                continue
            train_end_year = int(summary.get("train_end_year") or 0)
            raw_process = dict(summary.get("raw_process_split_transition") or {})
            anatomy = dict(raw_process.get("transition_residual_anatomy") or {})
            for row in list(anatomy.get("rows") or []):
                if isinstance(row, dict):
                    component_rows.append(
                        {
                            "horizon_years": horizon,
                            "train_end_year": train_end_year,
                            **dict(row),
                        }
                    )
    return {
        "schema_version": "phase3_dynamic.r12_02_process_split_anatomy.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-02",
        "status": "completed" if rows else "not_evaluable",
        "row_count": len(rows),
        "component_row_count": len(component_rows),
        "rows": rows,
        "component_rows": component_rows,
        "contract": (
            "R12-02 residual anatomy separates D inflow/removal/reporting residuals from ART delayed linkage, "
            "diagnosed-gap pressure, initiation capacity, removal, reporting shift, and residual terms."
        ),
    }


def _build_r12_03_residual_source_report(r12_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    context_score_rows: list[dict[str, Any]] = []
    for horizon_key, horizon_report in dict(r12_report.get("horizon_reports") or {}).items():
        if not isinstance(horizon_report, dict):
            continue
        horizon = int(str(horizon_key).removeprefix("h") or 0)
        for summary in list(horizon_report.get("model_summaries") or []):
            if not isinstance(summary, dict):
                continue
            train_end_year = int(summary.get("train_end_year") or 0)
            model = dict(summary.get("residual_source_model") or {})
            rows.append(
                {
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "status": str(model.get("status") or ""),
                    "record_count": model.get("record_count"),
                    "selected_context_by_metric": dict(model.get("selected_context_by_metric") or {}),
                    "selected_mean_abs_residual_by_metric": dict(model.get("selected_mean_abs_residual_by_metric") or {}),
                }
            )
            raw_model = dict(summary.get("raw_residual_source_model") or {})
            for score in list(raw_model.get("context_score_rows") or []):
                if isinstance(score, dict):
                    context_score_rows.append(
                        {
                            "horizon_years": horizon,
                            "train_end_year": train_end_year,
                            **dict(score),
                        }
                    )
    selected_context_counter: Counter[str] = Counter()
    for row in rows:
        for metric_name, context_name in dict(row.get("selected_context_by_metric") or {}).items():
            if context_name:
                selected_context_counter[f"{metric_name}:{context_name}"] += 1
    return {
        "schema_version": "phase3_dynamic.r12_03_residual_source.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-03",
        "status": "completed" if rows else "not_evaluable",
        "row_count": len(rows),
        "context_score_row_count": len(context_score_rows),
        "selected_context_counts": dict(sorted(selected_context_counter.items())),
        "rows": rows,
        "context_score_rows": context_score_rows,
        "contract": (
            "R12-03 residual-source report. Candidate contexts are fitted from train residual anatomy only: "
            "source family, source id, support signature, monthly reporting intensity, and backlog/rebound flow regime."
        ),
    }


def _r12_metric_lineage_context(row: dict[str, Any], metric_name: str) -> dict[str, str]:
    provenance = _metric_provenance(row, metric_name)
    source_family = _source_family_signature(row, metric_name)
    support_partition = str(provenance.get("support_partition") or "unknown_support")
    aggregation_mode = str(provenance.get("aggregation_mode") or "unknown_aggregation")
    source_id = str(provenance.get("source_id") or "unknown_source_id")
    return {
        "source_family": source_family,
        "support_partition": support_partition,
        "source_support": f"{source_family}::{support_partition}",
        "aggregation_mode": aggregation_mode,
        "source_id": source_id,
        "observation_role": str(provenance.get("observation_role") or "unknown_role"),
        "allowed_use": str(provenance.get("allowed_use") or "unknown_allowed_use"),
        "tier": str(provenance.get("tier") or provenance.get("source_tier") or "unknown_tier"),
        "series_kind": str(provenance.get("series_kind") or "unknown_series_kind"),
        "measurement_class": str(provenance.get("measurement_class") or "unknown_measurement_class"),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _sum_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.sum(np.asarray(values, dtype=np.float64)))


def _r12_lineage_score_records(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    score_records: list[dict[str, Any]] = []
    for horizon in horizons:
        splits = rolling_origin_splits(
            rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon),
        )
        for split in splits:
            holdout_years = [int(year) for year in list(split.get("holdout_years") or [])]
            if not holdout_years:
                continue
            train_end_year = int(split.get("train_end_year") or min(holdout_years) - 1)
            train_rows = [
                dict(row)
                for row in rows
                if quarter_year(str(row.get("quarter") or "")) <= train_end_year
            ]
            holdout_rows = [
                dict(row)
                for row in rows
                if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
            ]
            if not train_rows or not holdout_rows:
                continue
            candidate_predictions, _summary = _candidate_predictions(
                train_rows,
                holdout_rows,
                family="multi_horizon_weighted_process",
            )
            carry_predictions = _carry_forward_prediction(train_rows, holdout_rows)
            candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_predictions}
            carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_predictions}
            for holdout_row in holdout_rows:
                quarter = str(holdout_row.get("quarter") or "")
                candidate_row = candidate_by_quarter.get(quarter, {})
                carry_row = carry_by_quarter.get(quarter, {})
                for metric_name in R10_COMPARABLE_METRICS:
                    target_value = _finite_float(holdout_row.get(metric_name))
                    candidate_value = _finite_float(candidate_row.get(metric_name))
                    carry_value = _finite_float(carry_row.get(metric_name))
                    if target_value is None or candidate_value is None or carry_value is None:
                        continue
                    scale = max(_metric_scale(train_rows, metric_name), float(np.finfo(np.float32).eps))
                    candidate_error = abs(float(candidate_value) - float(target_value)) / scale
                    carry_error = abs(float(carry_value) - float(target_value)) / scale
                    lineage = _r12_metric_lineage_context(holdout_row, metric_name)
                    score_records.append(
                        {
                            "horizon_years": int(horizon),
                            "train_end_year": train_end_year,
                            "holdout_years": holdout_years,
                            "quarter": quarter,
                            "metric_name": metric_name,
                            "target_value": float(target_value),
                            "candidate_value": float(candidate_value),
                            "carry_forward_value": float(carry_value),
                            "scale": float(scale),
                            "candidate_norm_error": float(candidate_error),
                            "carry_forward_norm_error": float(carry_error),
                            "candidate_minus_carry_forward_norm_error": float(candidate_error - carry_error),
                            "signed_norm_residual": float(float(candidate_value) - float(target_value)) / scale,
                            **lineage,
                        }
                    )
    return score_records


def _r12_lineage_horizon_reference_map(
    r10_horizon_replay: dict[str, Any] | None,
    horizons: tuple[int, ...] = R12_LINEAGE_DIAGNOSTIC_HORIZONS,
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        reference = _r10_reference_for_horizon(r10_horizon_replay, int(horizon))
        output[int(horizon)] = {
            "reference_mae": _finite_float(reference.get("reference_quarterly_mean_mae")),
            "reference_experiment_id": reference.get("reference_experiment_id"),
            "reference_status": reference.get("status"),
            "artifact_path": reference.get("artifact_path"),
        }
    return output


def _r12_lineage_axis_rows(
    *,
    score_records: list[dict[str, Any]],
    r10_reference_by_horizon: dict[int, dict[str, Any]],
    axis: str,
    include_metric: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    totals_by_horizon: dict[int, float] = defaultdict(float)
    for record in score_records:
        horizon = int(record.get("horizon_years") or 0)
        totals_by_horizon[horizon] += float(record.get("candidate_norm_error") or 0.0)
        key = (
            horizon,
            str(record.get("metric_name") or ""),
            str(record.get(axis) or "unknown"),
        ) if include_metric else (
            horizon,
            str(record.get(axis) or "unknown"),
        )
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for key, records in sorted(grouped.items(), key=lambda item: (item[0], -sum(float(row.get("candidate_norm_error") or 0.0) for row in item[1]))):
        if include_metric:
            horizon = int(key[0])
            metric_name = str(key[1])
            lineage_value = str(key[2])
        else:
            horizon = int(key[0])
            metric_name = None
            lineage_value = str(key[1])
        candidate_errors = [float(row["candidate_norm_error"]) for row in records]
        carry_errors = [float(row["carry_forward_norm_error"]) for row in records]
        signed_residuals = [float(row["signed_norm_residual"]) for row in records]
        candidate_sum = float(_sum_or_none(candidate_errors) or 0.0)
        horizon_total = max(float(totals_by_horizon.get(horizon) or 0.0), float(np.finfo(np.float32).eps))
        r10_reference = _finite_float(r10_reference_by_horizon.get(horizon, {}).get("reference_mae"))
        candidate_mean = _mean_or_none(candidate_errors)
        row: dict[str, Any] = {
            "horizon_years": horizon,
            "axis": axis,
            "lineage_value": lineage_value,
            "entry_count": len(records),
            "candidate_mean_norm_error": candidate_mean,
            "carry_forward_mean_norm_error": _mean_or_none(carry_errors),
            "candidate_minus_carry_forward_mean_norm_error": None
            if candidate_mean is None or _mean_or_none(carry_errors) is None
            else float(candidate_mean - float(_mean_or_none(carry_errors))),
            "candidate_error_share_within_horizon": float(candidate_sum / horizon_total),
            "mean_signed_norm_residual": _mean_or_none(signed_residuals),
            "r10_horizon_reference_mae": r10_reference,
            "candidate_minus_r10_reference_mae": None
            if candidate_mean is None or r10_reference is None
            else float(candidate_mean - float(r10_reference)),
            "metric_names": sorted({str(record.get("metric_name") or "") for record in records}),
        }
        if metric_name is not None:
            row["metric_name"] = metric_name
        rows.append(row)
    return rows


def _r12_lineage_leave_one_out_rows(
    *,
    score_records: list[dict[str, Any]],
    r10_reference_by_horizon: dict[int, dict[str, Any]],
    axis: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records_by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in score_records:
        records_by_horizon[int(record.get("horizon_years") or 0)].append(record)
    for horizon, horizon_records in sorted(records_by_horizon.items()):
        reference = _finite_float(r10_reference_by_horizon.get(horizon, {}).get("reference_mae"))
        all_candidate_errors = [float(row["candidate_norm_error"]) for row in horizon_records]
        all_carry_errors = [float(row["carry_forward_norm_error"]) for row in horizon_records]
        all_candidate = _mean_or_none(all_candidate_errors)
        all_carry = _mean_or_none(all_carry_errors)
        lineage_values = sorted({str(row.get(axis) or "unknown") for row in horizon_records})
        for lineage_value in lineage_values:
            kept = [row for row in horizon_records if str(row.get(axis) or "unknown") != lineage_value]
            excluded = [row for row in horizon_records if str(row.get(axis) or "unknown") == lineage_value]
            kept_candidate = _mean_or_none([float(row["candidate_norm_error"]) for row in kept])
            kept_carry = _mean_or_none([float(row["carry_forward_norm_error"]) for row in kept])
            excluded_candidate_sum = float(_sum_or_none([float(row["candidate_norm_error"]) for row in excluded]) or 0.0)
            total_candidate_sum = max(float(_sum_or_none(all_candidate_errors) or 0.0), float(np.finfo(np.float32).eps))
            before_gap = None if all_candidate is None or reference is None else float(all_candidate - reference)
            after_gap = None if kept_candidate is None or reference is None else float(kept_candidate - reference)
            rows.append(
                {
                    "horizon_years": horizon,
                    "axis": axis,
                    "excluded_lineage_value": lineage_value,
                    "all_entry_count": len(horizon_records),
                    "excluded_entry_count": len(excluded),
                    "remaining_entry_count": len(kept),
                    "excluded_error_share_within_horizon": float(excluded_candidate_sum / total_candidate_sum),
                    "all_candidate_mean_norm_error": all_candidate,
                    "all_carry_forward_mean_norm_error": all_carry,
                    "remaining_candidate_mean_norm_error": kept_candidate,
                    "remaining_carry_forward_mean_norm_error": kept_carry,
                    "r10_horizon_reference_mae": reference,
                    "candidate_minus_r10_before": before_gap,
                    "candidate_minus_r10_after_exclusion": after_gap,
                    "exclusion_delta_candidate_mean_norm_error": None
                    if kept_candidate is None or all_candidate is None
                    else float(kept_candidate - all_candidate),
                    "flips_to_r10_pass_after_exclusion": bool(
                        before_gap is not None
                        and after_gap is not None
                        and before_gap >= 0.0
                        and after_gap < 0.0
                    ),
                    "affected_metrics": sorted({str(row.get("metric_name") or "") for row in excluded}),
                }
            )
    return rows


def _build_r12_04_source_lineage_ablation_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    r10_horizon_replay: dict[str, Any] | None,
    horizons: tuple[int, ...] = R12_LINEAGE_DIAGNOSTIC_HORIZONS,
) -> dict[str, Any]:
    score_records = _r12_lineage_score_records(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    reference_by_horizon = _r12_lineage_horizon_reference_map(r10_horizon_replay)
    axis_rows: list[dict[str, Any]] = []
    metric_axis_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    for axis in R12_LINEAGE_AXES:
        axis_rows.extend(
            _r12_lineage_axis_rows(
                score_records=score_records,
                r10_reference_by_horizon=reference_by_horizon,
                axis=axis,
                include_metric=False,
            )
        )
        metric_axis_rows.extend(
            _r12_lineage_axis_rows(
                score_records=score_records,
                r10_reference_by_horizon=reference_by_horizon,
                axis=axis,
                include_metric=True,
            )
        )
        leave_one_out_rows.extend(
            _r12_lineage_leave_one_out_rows(
                score_records=score_records,
                r10_reference_by_horizon=reference_by_horizon,
                axis=axis,
            )
        )
    decisive_rows = [
        row
        for row in leave_one_out_rows
        if bool(row.get("flips_to_r10_pass_after_exclusion"))
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
    ]
    evaluated_horizons = sorted({int(row.get("horizon_years") or 0) for row in score_records})
    horizon_rows: list[dict[str, Any]] = []
    for horizon in evaluated_horizons:
        horizon_records = [row for row in score_records if int(row.get("horizon_years") or 0) == horizon]
        candidate_mean = _mean_or_none([float(row["candidate_norm_error"]) for row in horizon_records])
        carry_mean = _mean_or_none([float(row["carry_forward_norm_error"]) for row in horizon_records])
        reference = _finite_float(reference_by_horizon.get(horizon, {}).get("reference_mae"))
        horizon_rows.append(
            {
                "horizon_years": horizon,
                "entry_count": len(horizon_records),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "r10_horizon_reference_mae": reference,
                "candidate_minus_r10_reference_mae": None
                if candidate_mean is None or reference is None
                else float(candidate_mean - float(reference)),
                "r10_status": "not_available"
                if candidate_mean is None or reference is None
                else ("pass" if candidate_mean < float(reference) else "fail"),
            }
        )
    if decisive_rows:
        blocker_assessment = "single_lineage_exclusion_can_flip_r10_failure"
        scientific_read = (
            "At least one source/support lineage is sufficient to flip an R11-28 3y/5y R10 failure under leave-one-lineage-out scoring. "
            "Treat the blocker as mixed observation lineage until that lineage is adjudicated."
        )
    elif score_records:
        blocker_assessment = "r10_failure_persists_after_single_lineage_exclusions"
        scientific_read = (
            "No single source/support lineage removal flips the 3y/5y R10 failure. The residual source/reporting signal is real, "
            "but the current blocker is more likely global trajectory dynamics or a multi-lineage observation mixture rather than one unstable lineage."
        )
    else:
        blocker_assessment = "not_evaluable"
        scientific_read = "No R10-comparable lineage score records were generated under the R12-04 contract."
    top_error_rows = sorted(
        axis_rows,
        key=lambda row: (
            int(row.get("horizon_years") or 0) not in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS),
            -float(row.get("candidate_error_share_within_horizon") or 0.0),
        ),
    )[:20]
    return {
        "schema_version": "phase3_dynamic.r12_04_source_lineage_evaluation_ablation.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-04",
        "status": "completed" if score_records else "not_evaluable",
        "decision": "keep_as_observation_lineage_diagnostic",
        "prediction_mutation": "forbidden",
        "reference_family": "multi_horizon_weighted_process",
        "reference_experiment_id": "R11-28",
        "horizons": list(horizons),
        "lineage_axes": list(R12_LINEAGE_AXES),
        "score_record_count": len(score_records),
        "axis_row_count": len(axis_rows),
        "metric_axis_row_count": len(metric_axis_rows),
        "leave_one_out_row_count": len(leave_one_out_rows),
        "decisive_leave_one_out_count": len(decisive_rows),
        "blocker_assessment": blocker_assessment,
        "scientific_read": scientific_read,
        "horizon_rows": horizon_rows,
        "decisive_leave_one_out_rows": decisive_rows,
        "top_error_lineage_rows": top_error_rows,
        "axis_rows": axis_rows,
        "metric_axis_rows": metric_axis_rows,
        "leave_one_out_rows": leave_one_out_rows,
        "contract": (
            "R12-04 does not change predictions. It recomputes locked R11-28 blocked-horizon residuals, "
            "stratifies R10-comparable D/A/diagnosis-flow errors by source family and support partition, and "
            "runs leave-one-lineage-out evaluation ablations to decide whether 3y/5y R10 failure is driven by "
            "mixed evidence lineage or by model dynamics."
        ),
    }


def _write_r12_04_lineage_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _compact_label(value: Any) -> str:
        text = str(value or "")
        replacements = {
            "official_doh_archive|program_observed_harp|quarterly_snapshot": "DOH HARP quarterly",
            "official_doh_archive|program_observed_harp|monthly_snapshot": "DOH HARP monthly",
            "official_user_provided_slide|program_observed_harp|annual_snapshot": "slide annual anchor",
            "common_support": "common support",
            "quarterly_observed": "quarterly observed",
            "annual_anchor_to_q4": "annual anchor to Q4",
            "monthly_to_quarter_sum": "monthly to quarter sum",
            "intraquarter_snapshot_bridge": "intraquarter bridge",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        if len(text) > 58:
            text = text[:55] + "..."
        return text

    axis_rows = [
        dict(row)
        for row in list(report.get("axis_rows") or [])
        if isinstance(row, dict)
        and str(row.get("axis") or "") in {"source_support", "support_partition", "source_family"}
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
    ]
    leave_rows = [
        dict(row)
        for row in list(report.get("leave_one_out_rows") or [])
        if isinstance(row, dict)
        and str(row.get("axis") or "") in {"source_support", "support_partition", "source_family"}
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
    ]
    horizon_rows = [dict(row) for row in list(report.get("horizon_rows") or []) if isinstance(row, dict)]
    if not axis_rows and not horizon_rows:
        return

    top_axis_rows = sorted(
        axis_rows,
        key=lambda row: float(row.get("candidate_error_share_within_horizon") or 0.0),
        reverse=True,
    )[:10]
    top_leave_rows = sorted(
        leave_rows,
        key=lambda row: float(row.get("exclusion_delta_candidate_mean_norm_error") or 0.0),
    )[:10]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    horizons = [int(row.get("horizon_years") or 0) for row in horizon_rows]
    x = np.arange(len(horizons), dtype=np.float64)
    candidate = np.asarray([float(row.get("candidate_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    r10 = np.asarray([float(row.get("r10_horizon_reference_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    width = 0.24
    axes[0].bar(x - width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, candidate, width=width, color="#315f72", label="R11-28")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(horizon) for horizon in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("R10-scope normalized MAE")
    axes[0].set_title("Locked reference vs benchmark")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    labels = [
        f"h{int(row.get('horizon_years') or 0)} {str(row.get('axis') or '')}: {_compact_label(row.get('lineage_value'))}"
        for row in top_axis_rows
    ]
    shares = [float(row.get("candidate_error_share_within_horizon") or 0.0) for row in top_axis_rows]
    y = np.arange(len(labels), dtype=np.float64)
    axes[1].barh(y, shares, color="#8a6f2a")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("share of candidate error")
    axes[1].set_title("Largest error lineages")
    axes[1].grid(axis="x", alpha=0.25)

    leave_labels = [
        f"h{int(row.get('horizon_years') or 0)} {str(row.get('axis') or '')}: {_compact_label(row.get('excluded_lineage_value'))}"
        for row in top_leave_rows
    ]
    deltas = [float(row.get("exclusion_delta_candidate_mean_norm_error") or 0.0) for row in top_leave_rows]
    colors = ["#2f7d59" if value < 0.0 else "#9b2f2f" for value in deltas]
    y2 = np.arange(len(leave_labels), dtype=np.float64)
    axes[2].barh(y2, deltas, color=colors)
    axes[2].axvline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[2].set_yticks(y2)
    axes[2].set_yticklabels(leave_labels, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("MAE after exclusion minus all-lineage MAE")
    axes[2].set_title("Leave-one-lineage-out effect")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("R12-04 source/support lineage ablation of R11-28 evaluation", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _r12_metric_matches_source_family(row: dict[str, Any], metric_name: str, source_family: str) -> bool:
    return _source_family_signature(row, metric_name) == str(source_family)


def _r12_lineage_filtered_rows(rows: list[dict[str, Any]], lineage: dict[str, str]) -> list[dict[str, Any]]:
    source_family = str(lineage.get("source_family") or "")
    filtered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        output = dict(row)
        provenance = dict(row.get("metric_provenance") or {})
        kept_metric_count = 0
        for metric_name in R11_EVALUATION_METRICS:
            if _r12_metric_matches_source_family(row, metric_name, source_family):
                kept_metric_count += int(_finite_float(row.get(metric_name)) is not None)
                continue
            output[metric_name] = None
            if metric_name in provenance:
                provenance[metric_name] = {
                    **dict(provenance.get(metric_name) or {}),
                    "r12_05_masked_by_lineage_operator": True,
                    "r12_05_active_lineage_id": str(lineage.get("lineage_id") or ""),
                }
        if kept_metric_count:
            output["metric_provenance"] = provenance
            output["r12_05_lineage_id"] = str(lineage.get("lineage_id") or "")
            output["r12_05_lineage_label"] = str(lineage.get("lineage_label") or "")
            output["r12_05_lineage_source_family"] = source_family
            filtered.append(output)
    return filtered


def _r12_lineage_metric_counts(rows: list[dict[str, Any]], lineage: dict[str, str]) -> dict[str, int]:
    source_family = str(lineage.get("source_family") or "")
    counts: Counter[str] = Counter()
    for row in rows:
        for metric_name in R11_EVALUATION_METRICS:
            if _finite_float(row.get(metric_name)) is None:
                continue
            if _r12_metric_matches_source_family(row, metric_name, source_family):
                counts[metric_name] += 1
    return dict(sorted(counts.items()))


def _r12_lineage_by_id(lineage_id: str) -> dict[str, str]:
    for lineage in R12_STRATIFIED_OPERATOR_LINEAGES:
        if str(lineage.get("lineage_id") or "") == str(lineage_id):
            return dict(lineage)
    return {}


def _r12_05_lineage_score_records(
    *,
    rows: list[dict[str, Any]],
    lineage: dict[str, str],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
    family: str = "multi_horizon_weighted_process",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lineage_rows = _r12_lineage_filtered_rows(rows, lineage)
    score_records: list[dict[str, Any]] = []
    split_count_by_horizon: Counter[int] = Counter()
    for horizon in horizons:
        splits = rolling_origin_splits(
            lineage_rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon),
        )
        for split in splits:
            holdout_years = [int(year) for year in list(split.get("holdout_years") or [])]
            if not holdout_years:
                continue
            train_end_year = int(split.get("train_end_year") or min(holdout_years) - 1)
            train_rows = [
                dict(row)
                for row in lineage_rows
                if quarter_year(str(row.get("quarter") or "")) <= train_end_year
            ]
            holdout_rows = [
                dict(row)
                for row in lineage_rows
                if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
            ]
            if not train_rows or not holdout_rows:
                continue
            split_count_by_horizon[int(horizon)] += 1
            candidate_predictions, _summary = _candidate_predictions(
                train_rows,
                holdout_rows,
                family=str(family),
            )
            carry_predictions = _carry_forward_prediction(train_rows, holdout_rows)
            candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_predictions}
            carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_predictions}
            for holdout_row in holdout_rows:
                quarter = str(holdout_row.get("quarter") or "")
                candidate_row = candidate_by_quarter.get(quarter, {})
                carry_row = carry_by_quarter.get(quarter, {})
                for metric_name in R10_COMPARABLE_METRICS:
                    target_value = _finite_float(holdout_row.get(metric_name))
                    candidate_value = _finite_float(candidate_row.get(metric_name))
                    carry_value = _finite_float(carry_row.get(metric_name))
                    if target_value is None or candidate_value is None or carry_value is None:
                        continue
                    scale = max(_metric_scale(train_rows, metric_name), float(np.finfo(np.float32).eps))
                    candidate_error = abs(float(candidate_value) - float(target_value)) / scale
                    carry_error = abs(float(carry_value) - float(target_value)) / scale
                    score_records.append(
                        {
                            "lineage_id": str(lineage.get("lineage_id") or ""),
                            "lineage_label": str(lineage.get("lineage_label") or ""),
                            "source_family": str(lineage.get("source_family") or ""),
                            "candidate_family": str(family),
                            "horizon_years": int(horizon),
                            "train_end_year": train_end_year,
                            "holdout_years": holdout_years,
                            "quarter": quarter,
                            "metric_name": metric_name,
                            "target_value": float(target_value),
                            "candidate_value": float(candidate_value),
                            "carry_forward_value": float(carry_value),
                            "scale": float(scale),
                            "candidate_norm_error": float(candidate_error),
                            "carry_forward_norm_error": float(carry_error),
                            "candidate_minus_carry_forward_norm_error": float(candidate_error - carry_error),
                            "signed_norm_residual": float(float(candidate_value) - float(target_value)) / scale,
                        }
                    )
    metric_counts = _r12_lineage_metric_counts(rows, lineage)
    blockers: list[str] = []
    for metric_name in R10_COMPARABLE_METRICS:
        if int(metric_counts.get(metric_name) or 0) == 0:
            blockers.append(f"{metric_name}_not_present_for_lineage")
    for horizon in horizons:
        if int(split_count_by_horizon.get(int(horizon)) or 0) == 0:
            blockers.append(f"h{int(horizon)}_no_blocked_splits")
    if not score_records:
        blockers.append("no_r10_comparable_score_records")
    return score_records, {
        "lineage_id": str(lineage.get("lineage_id") or ""),
        "lineage_label": str(lineage.get("lineage_label") or ""),
        "source_family": str(lineage.get("source_family") or ""),
        "candidate_family": str(family),
        "lineage_row_count": len(lineage_rows),
        "metric_counts": metric_counts,
        "split_count_by_horizon": {str(key): int(value) for key, value in sorted(split_count_by_horizon.items())},
        "score_record_count": len(score_records),
        "status": "evaluable" if score_records else "not_evaluable",
        "blockers": blockers,
    }


def _r12_05_score_summary(
    records: list[dict[str, Any]],
    *,
    r10_reference_by_horizon: dict[int, dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) for field in group_fields)
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        payload = {field: key[index] for index, field in enumerate(group_fields)}
        horizon = int(payload.get("horizon_years") or 0)
        candidate_errors = [float(row["candidate_norm_error"]) for row in values]
        carry_errors = [float(row["carry_forward_norm_error"]) for row in values]
        signed_residuals = [float(row["signed_norm_residual"]) for row in values]
        candidate_mean = _mean_or_none(candidate_errors)
        carry_mean = _mean_or_none(carry_errors)
        reference = _finite_float(r10_reference_by_horizon.get(horizon, {}).get("reference_mae"))
        rows.append(
            {
                **payload,
                "entry_count": len(values),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "mean_signed_norm_residual": _mean_or_none(signed_residuals),
                "r10_horizon_reference_mae": reference,
                "candidate_minus_r10_reference_mae": None
                if candidate_mean is None or reference is None
                else float(candidate_mean - float(reference)),
                "r10_status": "not_available"
                if candidate_mean is None or reference is None
                else ("pass" if candidate_mean < float(reference) else "fail"),
            }
        )
    return rows


def _build_r12_05_lineage_stratified_contract_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    r10_horizon_replay: dict[str, Any] | None,
    horizons: tuple[int, ...] = R12_LINEAGE_DIAGNOSTIC_HORIZONS,
    lineages: tuple[dict[str, str], ...] = R12_STRATIFIED_OPERATOR_LINEAGES,
) -> dict[str, Any]:
    reference_by_horizon = _r12_lineage_horizon_reference_map(r10_horizon_replay)
    score_records: list[dict[str, Any]] = []
    lineage_manifests: list[dict[str, Any]] = []
    for lineage in lineages:
        lineage_records, manifest = _r12_05_lineage_score_records(
            rows=rows,
            lineage=lineage,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
        )
        score_records.extend(lineage_records)
        lineage_manifests.append(manifest)
    horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("horizon_years",),
    )
    lineage_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("lineage_id", "lineage_label", "source_family", "horizon_years"),
    )
    metric_lineage_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("lineage_id", "lineage_label", "metric_name", "horizon_years"),
    )
    horizon_by_id = {int(row.get("horizon_years") or 0): dict(row) for row in horizon_rows}
    h3_row = horizon_by_id.get(3, {})
    h5_row = horizon_by_id.get(5, {})
    h3_status = str(h3_row.get("r10_status") or "not_evaluable") if h3_row else "not_evaluable"
    h5_status = str(h5_row.get("r10_status") or "not_evaluable") if h5_row else "not_evaluable"
    unevaluable_lineages = [
        str(manifest.get("lineage_id") or "")
        for manifest in lineage_manifests
        if str(manifest.get("status") or "") != "evaluable"
    ]
    if h3_status == "pass":
        blocker_assessment = "lineage_stratified_operator_removes_three_year_r10_failure"
        scientific_read = (
            "Training and evaluating R11-28 within separated DOH-quarterly, DOH-monthly, and slide-anchor evidence "
            "views removes the 3y R10-scope failure. This supports an observation-operator explanation for the 3y issue. "
            "The 5y dynamics claim remains frozen and is not promoted by this diagnostic."
        )
    elif score_records and unevaluable_lineages:
        blocker_assessment = "lineage_stratified_operator_underpowered_and_does_not_remove_three_year_failure"
        scientific_read = (
            "Lineage-stratified train/evaluate views do not remove the 3y R10-scope failure, and at least one target "
            "lineage is not separately evaluable under the blocked 3y/5y contract. This means simple lineage mixing "
            "is not sufficient as a fix, but the DOH-quarterly lineage still needs either shorter-horizon adjudication "
            "or a lineage-specific observation-operator validation set before it can be isolated as training evidence."
        )
    elif score_records:
        blocker_assessment = "lineage_stratified_operator_does_not_remove_three_year_r10_failure"
        scientific_read = (
            "Lineage-stratified train/evaluate views do not remove the 3y R10-scope failure. The 3y blocker therefore "
            "cannot be explained by simple lineage mixing alone under the current R11-28 contract."
        )
    else:
        blocker_assessment = "not_evaluable"
        scientific_read = "No lineage-stratified R10-comparable records were generated."
    return {
        "schema_version": "phase3_dynamic.r12_05_lineage_stratified_training_evaluation_contract.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-05",
        "status": "completed" if score_records else "not_evaluable",
        "decision": "keep_as_lineage_stratified_observation_operator_contract",
        "prediction_mutation": "forbidden",
        "reference_family": "multi_horizon_weighted_process",
        "reference_experiment_id": "R11-28",
        "lineages": list(lineages),
        "horizons": list(horizons),
        "score_record_count": len(score_records),
        "lineage_manifests": lineage_manifests,
        "horizon_rows": horizon_rows,
        "lineage_horizon_rows": lineage_horizon_rows,
        "metric_lineage_horizon_rows": metric_lineage_horizon_rows,
        "three_year_observation_operator_status": h3_status,
        "five_year_dynamics_claim_status": "frozen_not_promoted",
        "five_year_r10_status_under_lineage_operator": h5_status,
        "unevaluable_lineage_ids": unevaluable_lineages,
        "blocker_assessment": blocker_assessment,
        "scientific_read": scientific_read,
        "score_records": score_records,
        "contract": (
            "R12-05 does not create a new model and does not feed residuals into dynamics. It materializes three "
            "forecast-origin-safe observation views: DOH HARP quarterly, DOH HARP monthly, and slide annual anchors. "
            "For each view, R11-28 is trained and evaluated only on observations from that lineage. The diagnostic "
            "tests whether a lineage-specific observation operator removes the 3y R10-scope failure; 5y remains a "
            "frozen dynamics claim and cannot be promoted by this observation contract."
        ),
    }


def _write_r12_05_lineage_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    horizon_rows = [dict(row) for row in list(report.get("horizon_rows") or []) if isinstance(row, dict)]
    lineage_rows = [dict(row) for row in list(report.get("lineage_horizon_rows") or []) if isinstance(row, dict)]
    metric_rows = [dict(row) for row in list(report.get("metric_lineage_horizon_rows") or []) if isinstance(row, dict)]
    if not horizon_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    horizons = [int(row.get("horizon_years") or 0) for row in horizon_rows]
    x = np.arange(len(horizons), dtype=np.float64)
    candidate = np.asarray([float(row.get("candidate_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    r10 = np.asarray([float(row.get("r10_horizon_reference_mae") or np.nan) for row in horizon_rows], dtype=np.float64)
    width = 0.24
    axes[0].bar(x - width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, candidate, width=width, color="#315f72", label="lineage-stratified R11-28")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(horizon) for horizon in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("R10-scope normalized MAE")
    axes[0].set_title("Stratified operator vs R10")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in sorted(lineage_rows, key=lambda item: (int(item.get("horizon_years") or 0), str(item.get("lineage_label") or ""))):
        labels.append(f"h{int(row.get('horizon_years') or 0)} {str(row.get('lineage_label') or '')}")
        values.append(float(row.get("candidate_minus_r10_reference_mae") or 0.0))
        colors.append("#2f7d59" if values[-1] < 0.0 else "#9b2f2f")
    y = np.arange(len(labels), dtype=np.float64)
    axes[1].barh(y, values, color=colors)
    axes[1].axvline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("candidate minus matched R10 MAE")
    axes[1].set_title("Per-lineage R10 gap")
    axes[1].grid(axis="x", alpha=0.25)

    top_metric_rows = sorted(
        [
            row
            for row in metric_rows
            if int(row.get("horizon_years") or 0) == 3
            and _finite_float(row.get("candidate_mean_norm_error")) is not None
        ],
        key=lambda row: float(row.get("candidate_mean_norm_error") or 0.0),
        reverse=True,
    )[:10]
    labels2 = [
        f"{str(row.get('lineage_label') or '')}: {str(row.get('metric_name') or '')}"
        for row in top_metric_rows
    ]
    values2 = [float(row.get("candidate_mean_norm_error") or 0.0) for row in top_metric_rows]
    y2 = np.arange(len(labels2), dtype=np.float64)
    axes[2].barh(y2, values2, color="#8a6f2a")
    axes[2].set_yticks(y2)
    axes[2].set_yticklabels(labels2, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("3y candidate normalized MAE")
    axes[2].set_title("3y metric-lineage residuals")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("R12-05 lineage-stratified R11-28 training/evaluation contract", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _r12_lineage_observations(
    rows: list[dict[str, Any]],
    lineage: dict[str, str],
    *,
    metrics: tuple[str, ...] = R11_EVALUATION_METRICS,
) -> list[dict[str, Any]]:
    source_family = str(lineage.get("source_family") or "")
    observations: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        for metric_name in metrics:
            value = _finite_float(row.get(metric_name))
            if value is None:
                continue
            if not _r12_metric_matches_source_family(row, metric_name, source_family):
                continue
            observations.append(
                {
                    "lineage_id": str(lineage.get("lineage_id") or ""),
                    "lineage_label": str(lineage.get("lineage_label") or ""),
                    "source_family": source_family,
                    "quarter": quarter,
                    "quarter_ordinal": quarter_ordinal(quarter),
                    "year": quarter_year(quarter),
                    "metric_name": metric_name,
                    "value": float(value),
                }
            )
    return observations


def _r12_lineage_internal_volatility(
    rows: list[dict[str, Any]],
    lineage: dict[str, str],
) -> dict[str, dict[str, Any]]:
    observations = _r12_lineage_observations(rows, lineage)
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_metric[str(observation.get("metric_name") or "")].append(observation)
    output: dict[str, dict[str, Any]] = {}
    eps = float(np.finfo(np.float32).eps)
    for metric_name, metric_observations in sorted(by_metric.items()):
        changes: list[float] = []
        sorted_observations = sorted(metric_observations, key=lambda item: int(item.get("quarter_ordinal") or 0))
        for previous, current in zip(sorted_observations[:-1], sorted_observations[1:]):
            previous_value = _finite_float(previous.get("value"))
            current_value = _finite_float(current.get("value"))
            if previous_value is None or current_value is None:
                continue
            scale = max(abs(previous_value), abs(current_value), eps)
            changes.append(abs(float(current_value) - float(previous_value)) / scale)
        output[metric_name] = {
            "observation_count": len(sorted_observations),
            "adjacent_change_count": len(changes),
            "median_adjacent_normalized_change": _mean_or_none(changes)
            if len(changes) == 1
            else (None if not changes else float(np.median(np.asarray(changes, dtype=np.float64)))),
            "max_adjacent_normalized_change": None if not changes else float(np.max(np.asarray(changes, dtype=np.float64))),
        }
    return output


def _r12_06_bridge_consistency_report(
    rows: list[dict[str, Any]],
    *,
    target_lineage: dict[str, str],
    comparator_lineages: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    target_observations = _r12_lineage_observations(rows, target_lineage, metrics=R10_COMPARABLE_METRICS)
    bridge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    eps = float(np.finfo(np.float32).eps)
    for comparator in comparator_lineages:
        comparator_observations = _r12_lineage_observations(rows, comparator, metrics=R10_COMPARABLE_METRICS)
        comparator_volatility = _r12_lineage_internal_volatility(rows, comparator)
        by_metric = defaultdict(list)
        for observation in comparator_observations:
            by_metric[str(observation.get("metric_name") or "")].append(observation)
        for target in target_observations:
            metric_name = str(target.get("metric_name") or "")
            target_year = int(target.get("year") or 0)
            candidates = [
                observation
                for observation in by_metric.get(metric_name, [])
                if int(observation.get("year") or 0) == target_year
            ]
            if not candidates:
                continue
            comparator_observation = min(
                candidates,
                key=lambda observation: (
                    abs(int(observation.get("quarter_ordinal") or 0) - int(target.get("quarter_ordinal") or 0)),
                    str(observation.get("quarter") or ""),
                ),
            )
            target_value = float(target.get("value") or 0.0)
            comparator_value = float(comparator_observation.get("value") or 0.0)
            scale = max(abs(target_value), abs(comparator_value), eps)
            normalized_abs_difference = abs(target_value - comparator_value) / scale
            reference_volatility = _finite_float(
                dict(comparator_volatility.get(metric_name) or {}).get("max_adjacent_normalized_change")
            )
            bridge_rows.append(
                {
                    "target_lineage_id": str(target_lineage.get("lineage_id") or ""),
                    "target_lineage_label": str(target_lineage.get("lineage_label") or ""),
                    "comparator_lineage_id": str(comparator.get("lineage_id") or ""),
                    "comparator_lineage_label": str(comparator.get("lineage_label") or ""),
                    "metric_name": metric_name,
                    "target_quarter": str(target.get("quarter") or ""),
                    "comparator_quarter": str(comparator_observation.get("quarter") or ""),
                    "quarter_delta": int(target.get("quarter_ordinal") or 0) - int(comparator_observation.get("quarter_ordinal") or 0),
                    "target_value": target_value,
                    "comparator_value": comparator_value,
                    "normalized_abs_difference": float(normalized_abs_difference),
                    "signed_relative_difference": float((target_value - comparator_value) / scale),
                    "reference_max_adjacent_normalized_change": reference_volatility,
                    "within_comparator_observed_volatility": None
                    if reference_volatility is None
                    else bool(float(normalized_abs_difference) <= float(reference_volatility)),
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bridge_rows:
        grouped[(str(row.get("comparator_lineage_id") or ""), str(row.get("metric_name") or ""))].append(row)
    for (comparator_id, metric_name), group_rows in sorted(grouped.items()):
        differences = [float(row.get("normalized_abs_difference") or 0.0) for row in group_rows]
        evaluable_flags = [
            bool(row.get("within_comparator_observed_volatility"))
            for row in group_rows
            if row.get("within_comparator_observed_volatility") is not None
        ]
        summary_rows.append(
            {
                "comparator_lineage_id": comparator_id,
                "metric_name": metric_name,
                "pair_count": len(group_rows),
                "median_normalized_abs_difference": None
                if not differences
                else float(np.median(np.asarray(differences, dtype=np.float64))),
                "max_normalized_abs_difference": None
                if not differences
                else float(np.max(np.asarray(differences, dtype=np.float64))),
                "volatility_evaluable_pair_count": len(evaluable_flags),
                "within_comparator_observed_volatility_count": int(sum(1 for flag in evaluable_flags if flag)),
                "all_evaluable_pairs_within_comparator_observed_volatility": None
                if not evaluable_flags
                else bool(all(evaluable_flags)),
            }
        )
    evaluable_summary = [
        row
        for row in summary_rows
        if row.get("all_evaluable_pairs_within_comparator_observed_volatility") is not None
    ]
    if not bridge_rows:
        status = "not_evaluable"
        blockers = ["no_same_calendar_year_bridge_pairs"]
    elif evaluable_summary and all(bool(row.get("all_evaluable_pairs_within_comparator_observed_volatility")) for row in evaluable_summary):
        status = "consistent_with_observed_comparator_volatility"
        blockers = []
    elif evaluable_summary:
        status = "inconsistent_with_observed_comparator_volatility"
        blockers = ["at_least_one_bridge_pair_exceeds_comparator_observed_volatility"]
    else:
        status = "observed_but_volatility_not_evaluable"
        blockers = ["comparator_internal_volatility_not_evaluable"]
    return {
        "schema_version": "phase3_dynamic.r12_06_bridge_consistency.v1",
        "status": status,
        "blockers": blockers,
        "target_lineage_id": str(target_lineage.get("lineage_id") or ""),
        "target_observation_count": len(target_observations),
        "bridge_pair_count": len(bridge_rows),
        "summary_rows": summary_rows,
        "bridge_rows": bridge_rows,
        "contract": (
            "Bridge consistency compares each DOH-quarterly R10-scope observation to the nearest same-metric "
            "DOH-monthly or slide-anchor observation in the same calendar year. A pair is judged only against the "
            "comparator lineage's own observed adjacent normalized volatility; no external tolerance is hand set."
        ),
    }


def _r12_06_short_horizon_summary(
    records: list[dict[str, Any]],
    *,
    r10_reference_by_horizon: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record.get("horizon_years") or 0)].append(record)
    rows: list[dict[str, Any]] = []
    for horizon, values in sorted(grouped.items()):
        candidate_errors = [float(row["candidate_norm_error"]) for row in values]
        carry_errors = [float(row["carry_forward_norm_error"]) for row in values]
        candidate_mean = _mean_or_none(candidate_errors)
        carry_mean = _mean_or_none(carry_errors)
        reference = _finite_float(r10_reference_by_horizon.get(horizon, {}).get("reference_mae"))
        rows.append(
            {
                "horizon_years": horizon,
                "entry_count": len(values),
                "candidate_mean_norm_error": candidate_mean,
                "carry_forward_mean_norm_error": carry_mean,
                "candidate_minus_carry_forward_mean_norm_error": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "r10_horizon_reference_mae": reference,
                "r10_status": "not_available"
                if reference is None
                else ("pass" if candidate_mean is not None and candidate_mean < float(reference) else "fail"),
                "candidate_minus_r10_reference_mae": None
                if candidate_mean is None or reference is None
                else float(candidate_mean - float(reference)),
            }
        )
    return rows


def _build_r12_06_doh_quarterly_support_adjudication_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    r10_horizon_replay: dict[str, Any] | None,
    production_min_train_years: int,
    diagnostic_min_train_years: int = R12_SUPPORT_ADEQUACY_MIN_TRAIN_YEARS,
    horizons: tuple[int, ...] = R12_SUPPORT_ADEQUACY_HORIZONS,
) -> dict[str, Any]:
    target_lineage = _r12_lineage_by_id("doh_quarterly")
    comparator_lineages = (
        _r12_lineage_by_id("doh_monthly"),
        _r12_lineage_by_id("slide_annual_anchor"),
    )
    short_records, short_manifest = _r12_05_lineage_score_records(
        rows=rows,
        lineage=target_lineage,
        start_year=start_year,
        end_year=end_year,
        min_train_years=diagnostic_min_train_years,
        horizons=horizons,
    )
    reference_by_horizon = _r12_lineage_horizon_reference_map(r10_horizon_replay, horizons=horizons)
    short_horizon_rows = _r12_06_short_horizon_summary(
        short_records,
        r10_reference_by_horizon=reference_by_horizon,
    )
    metric_short_horizon_rows = _r12_05_score_summary(
        short_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("metric_name", "horizon_years"),
    )
    bridge_report = _r12_06_bridge_consistency_report(
        rows,
        target_lineage=target_lineage,
        comparator_lineages=comparator_lineages,
    )
    missing_horizons = [
        int(horizon)
        for horizon in horizons
        if int(horizon) not in {int(row.get("horizon_years") or 0) for row in short_horizon_rows}
    ]
    support_adequacy_status = "pass" if not missing_horizons and short_records else "fail"
    candidate_beats_carry_all = bool(
        short_horizon_rows
        and all(
            _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error")) is not None
            and float(row["candidate_minus_carry_forward_mean_norm_error"]) < 0.0
            for row in short_horizon_rows
        )
    )
    if support_adequacy_status == "pass" and candidate_beats_carry_all:
        blocker_assessment = "doh_quarterly_is_short_horizon_evaluable_but_not_three_year_claim_grade"
        scientific_read = (
            "DOH-quarterly support can be evaluated at 1y/2y using a diagnostic minimum-train window, and R11-28 "
            "beats carry-forward on those short horizons. This supports using DOH quarterly as an adjudication source, "
            "but it remains too short for the production 3y/5y R10 claim gate."
        )
    elif support_adequacy_status == "pass":
        blocker_assessment = "doh_quarterly_is_short_horizon_evaluable_but_predictively_weak"
        scientific_read = (
            "DOH-quarterly support can be evaluated at 1y/2y under a relaxed diagnostic window, but R11-28 does not "
            "consistently beat carry-forward there. This weakens the case for treating DOH quarterly as a clean "
            "observation-operator adjudicator."
        )
    else:
        blocker_assessment = "doh_quarterly_remains_underpowered_even_for_short_horizon_adjudication"
        scientific_read = (
            "DOH-quarterly support still lacks enough blocked short-horizon evidence under the diagnostic contract. "
            "It should remain a support-adequacy blocker rather than a training or validation truth source."
        )
    return {
        "schema_version": "phase3_dynamic.r12_06_doh_quarterly_support_adequacy_adjudication.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-06",
        "status": "completed" if short_records or bridge_report.get("bridge_pair_count") else "not_evaluable",
        "decision": "keep_as_doh_quarterly_support_adequacy_adjudication",
        "prediction_mutation": "forbidden",
        "reference_family": "multi_horizon_weighted_process",
        "reference_experiment_id": "R11-28",
        "target_lineage": target_lineage,
        "comparator_lineages": list(comparator_lineages),
        "production_min_train_years": int(production_min_train_years),
        "diagnostic_min_train_years": int(diagnostic_min_train_years),
        "horizons": list(horizons),
        "support_adequacy_status": support_adequacy_status,
        "missing_short_horizon_scores": missing_horizons,
        "candidate_beats_carry_forward_all_short_horizons": candidate_beats_carry_all,
        "short_horizon_manifest": short_manifest,
        "short_horizon_rows": short_horizon_rows,
        "metric_short_horizon_rows": metric_short_horizon_rows,
        "bridge_consistency": bridge_report,
        "blocker_assessment": blocker_assessment,
        "scientific_read": scientific_read,
        "score_records": short_records,
        "contract": (
            "R12-06 does not alter model predictions. It adjudicates whether the DOH-quarterly lineage that drove "
            "R12-04 can be used scientifically: first through 1y/2y lineage-only R11-28 checks under an explicitly "
            "relaxed diagnostic min-train window, then through bridge consistency against DOH-monthly and slide-anchor "
            "observations using only comparator-observed volatility as tolerance."
        ),
    }


def _write_r12_06_support_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    horizon_rows = [dict(row) for row in list(report.get("short_horizon_rows") or []) if isinstance(row, dict)]
    bridge_rows = [
        dict(row)
        for row in list(dict(report.get("bridge_consistency") or {}).get("summary_rows") or [])
        if isinstance(row, dict)
    ]
    if not horizon_rows and not bridge_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    horizons = [int(row.get("horizon_years") or 0) for row in horizon_rows]
    x = np.arange(len(horizons), dtype=np.float64)
    candidate = np.asarray([float(row.get("candidate_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    width = 0.32
    axes[0].bar(x - width / 2.0, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x + width / 2.0, candidate, width=width, color="#315f72", label="DOH-quarterly R11-28")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(horizon) for horizon in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("normalized MAE")
    axes[0].set_title("Short-horizon adequacy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    metric_rows = [
        dict(row)
        for row in list(report.get("metric_short_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    metric_rows = sorted(metric_rows, key=lambda row: float(row.get("candidate_mean_norm_error") or 0.0), reverse=True)[:10]
    metric_labels = [
        f"h{int(row.get('horizon_years') or 0)} {str(row.get('metric_name') or '')}"
        for row in metric_rows
    ]
    metric_values = [float(row.get("candidate_mean_norm_error") or 0.0) for row in metric_rows]
    y = np.arange(len(metric_labels), dtype=np.float64)
    axes[1].barh(y, metric_values, color="#8a6f2a")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(metric_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("candidate normalized MAE")
    axes[1].set_title("Metric residuals")
    axes[1].grid(axis="x", alpha=0.25)

    bridge_labels = [
        f"{str(row.get('comparator_lineage_id') or '')}: {str(row.get('metric_name') or '')}"
        for row in bridge_rows
    ]
    bridge_values = [
        0.0 if _finite_float(row.get("median_normalized_abs_difference")) is None else float(row["median_normalized_abs_difference"])
        for row in bridge_rows
    ]
    colors = [
        "#2f7d59" if bool(row.get("all_evaluable_pairs_within_comparator_observed_volatility")) else "#9b2f2f"
        for row in bridge_rows
    ]
    y2 = np.arange(len(bridge_labels), dtype=np.float64)
    axes[2].barh(y2, bridge_values, color=colors)
    axes[2].set_yticks(y2)
    axes[2].set_yticklabels(bridge_labels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("median normalized bridge difference")
    axes[2].set_title("Bridge consistency")
    axes[2].grid(axis="x", alpha=0.25)
    fig.suptitle("R12-06 DOH-quarterly support adequacy and bridge adjudication", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _r12_07_route_min_train_years(
    route: dict[str, Any],
    *,
    production_min_train_years: int,
    diagnostic_min_train_years: int,
) -> int:
    if str(route.get("min_train_contract") or "") == "diagnostic_short_horizon":
        return int(diagnostic_min_train_years)
    return int(production_min_train_years)


def _r12_07_route_manifest_rows(
    *,
    route: dict[str, Any],
    route_records: list[dict[str, Any]],
    route_horizon_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    route_id = str(route.get("route_id") or "")
    horizons = [int(horizon) for horizon in list(route.get("horizons") or [])]
    present_horizons = {
        int(row.get("horizon_years") or 0)
        for row in route_horizon_rows
        if str(row.get("route_id") or "") == route_id
    }
    missing_horizons = [int(horizon) for horizon in horizons if int(horizon) not in present_horizons]
    rows = [
        dict(row)
        for row in route_horizon_rows
        if str(row.get("route_id") or "") == route_id
    ]
    candidate_beats_carry = bool(
        rows
        and all(
            _finite_float(row.get("candidate_minus_carry_forward_mean_norm_error")) is not None
            and float(row["candidate_minus_carry_forward_mean_norm_error"]) < 0.0
            for row in rows
        )
    )
    r10_rows = [
        row
        for row in rows
        if _finite_float(row.get("candidate_minus_r10_reference_mae")) is not None
    ]
    r10_required = bool(route.get("r10_required")) or str(route.get("claim_role") or "") == "long_horizon_trajectory"
    candidate_beats_available_r10 = bool(
        r10_rows
        and all(float(row["candidate_minus_r10_reference_mae"]) < 0.0 for row in r10_rows)
    )
    status = "evaluable" if route_records and not missing_horizons else "not_evaluable"
    blockers: list[str] = []
    if missing_horizons:
        blockers.extend([f"h{horizon}_no_route_score" for horizon in missing_horizons])
    if rows and not candidate_beats_carry:
        blockers.append("candidate_not_better_than_carry_forward_on_all_route_horizons")
    if r10_required and not r10_rows:
        blockers.append("no_available_r10_route_reference")
    if r10_required and r10_rows and not candidate_beats_available_r10:
        blockers.append("candidate_not_better_than_available_r10_on_all_route_horizons")
    return {
        "route_id": route_id,
        "route_label": str(route.get("route_label") or ""),
        "claim_role": str(route.get("claim_role") or ""),
        "lineage_ids": list(route.get("lineage_ids") or []),
        "horizons": horizons,
        "min_train_contract": str(route.get("min_train_contract") or ""),
        "r10_required": r10_required,
        "score_record_count": len(route_records),
        "status": status,
        "missing_horizons": missing_horizons,
        "candidate_beats_carry_forward_all_route_horizons": candidate_beats_carry,
        "r10_evaluable_horizon_count": len(r10_rows),
        "candidate_beats_available_r10_all_route_horizons": candidate_beats_available_r10,
        "blockers": blockers,
    }


def _build_r12_07_horizon_specific_evidence_router_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    r10_horizon_replay: dict[str, Any] | None,
    production_min_train_years: int,
    diagnostic_min_train_years: int = R12_SUPPORT_ADEQUACY_MIN_TRAIN_YEARS,
    routes: tuple[dict[str, Any], ...] = R12_HORIZON_EVIDENCE_ROUTES,
) -> dict[str, Any]:
    all_horizons = tuple(
        sorted(
            {
                int(horizon)
                for route in routes
                for horizon in list(route.get("horizons") or [])
            }
        )
    )
    reference_by_horizon = _r12_lineage_horizon_reference_map(r10_horizon_replay, horizons=all_horizons)
    score_records: list[dict[str, Any]] = []
    lineage_manifests: list[dict[str, Any]] = []
    route_records_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        route_id = str(route.get("route_id") or "")
        min_train_years = _r12_07_route_min_train_years(
            route,
            production_min_train_years=production_min_train_years,
            diagnostic_min_train_years=diagnostic_min_train_years,
        )
        for lineage_id in list(route.get("lineage_ids") or []):
            lineage = _r12_lineage_by_id(str(lineage_id))
            lineage_records, lineage_manifest = _r12_05_lineage_score_records(
                rows=rows,
                lineage=lineage,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                horizons=tuple(int(horizon) for horizon in list(route.get("horizons") or [])),
            )
            annotated_records = [
                {
                    **dict(record),
                    "route_id": route_id,
                    "route_label": str(route.get("route_label") or ""),
                    "claim_role": str(route.get("claim_role") or ""),
                    "route_min_train_years": min_train_years,
                    "route_min_train_contract": str(route.get("min_train_contract") or ""),
                }
                for record in lineage_records
            ]
            route_records_by_id[route_id].extend(annotated_records)
            score_records.extend(annotated_records)
            lineage_manifests.append(
                {
                    **dict(lineage_manifest),
                    "route_id": route_id,
                    "route_label": str(route.get("route_label") or ""),
                    "claim_role": str(route.get("claim_role") or ""),
                    "route_min_train_years": min_train_years,
                    "route_min_train_contract": str(route.get("min_train_contract") or ""),
                }
            )
    route_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "route_label", "claim_role", "horizon_years"),
    )
    route_lineage_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "lineage_id", "lineage_label", "horizon_years"),
    )
    metric_route_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "metric_name", "horizon_years"),
    )
    route_manifests = [
        _r12_07_route_manifest_rows(
            route=dict(route),
            route_records=route_records_by_id.get(str(route.get("route_id") or ""), []),
            route_horizon_rows=route_horizon_rows,
        )
        for route in routes
    ]
    missing_route_ids = [
        str(manifest.get("route_id") or "")
        for manifest in route_manifests
        if str(manifest.get("status") or "") != "evaluable"
    ]
    nowcast_manifests = [
        manifest
        for manifest in route_manifests
        if str(manifest.get("claim_role") or "") == "short_horizon_nowcast"
    ]
    trajectory_manifests = [
        manifest
        for manifest in route_manifests
        if str(manifest.get("claim_role") or "") == "long_horizon_trajectory"
    ]
    nowcast_supported = bool(
        nowcast_manifests
        and all(bool(manifest.get("candidate_beats_carry_forward_all_route_horizons")) for manifest in nowcast_manifests)
    )
    trajectory_r10_supported = bool(
        trajectory_manifests
        and all(bool(manifest.get("candidate_beats_available_r10_all_route_horizons")) for manifest in trajectory_manifests)
    )
    if missing_route_ids:
        blocker_assessment = "horizon_router_has_underpowered_routes"
        scientific_read = (
            "At least one horizon-specific evidence route has no complete blocked score. The claim boundary should "
            "remain evidence-support adequacy, not model superiority."
        )
    elif nowcast_supported and trajectory_r10_supported:
        blocker_assessment = "horizon_router_supports_separate_nowcast_and_trajectory_claims"
        scientific_read = (
            "The routed evidence contract separates short-horizon program nowcasting from long-horizon annual-anchor "
            "trajectory evaluation, and the routed scores satisfy their route-specific benchmark checks."
        )
    elif nowcast_supported:
        blocker_assessment = "nowcast_route_supported_but_trajectory_route_still_r10_blocked"
        scientific_read = (
            "Program evidence supports short-horizon nowcasting against carry-forward, but annual-anchor long-horizon "
            "trajectory evaluation still does not clear the available R10 reference. This argues for separate claims: "
            "a defensible nowcast readout and a still-blocked trajectory model."
        )
    else:
        blocker_assessment = "horizon_router_does_not_support_nowcast_or_trajectory_promotion"
        scientific_read = (
            "The horizon-specific evidence router is evaluable, but it does not support either short-horizon nowcast "
            "promotion or long-horizon trajectory promotion under the current benchmark contract."
        )
    return {
        "schema_version": "phase3_dynamic.r12_07_horizon_specific_evidence_router.v1",
        "generated_at": _generated_at(),
        "experiment_id": "R12-07",
        "status": "completed" if score_records else "not_evaluable",
        "decision": "keep_as_horizon_specific_evidence_router",
        "prediction_mutation": "forbidden",
        "reference_family": "multi_horizon_weighted_process",
        "reference_experiment_id": "R11-28",
        "production_min_train_years": int(production_min_train_years),
        "diagnostic_min_train_years": int(diagnostic_min_train_years),
        "routes": list(routes),
        "route_manifests": route_manifests,
        "lineage_manifests": lineage_manifests,
        "score_record_count": len(score_records),
        "route_horizon_rows": route_horizon_rows,
        "route_lineage_horizon_rows": route_lineage_horizon_rows,
        "metric_route_horizon_rows": metric_route_horizon_rows,
        "missing_route_ids": missing_route_ids,
        "nowcast_route_supported": nowcast_supported,
        "trajectory_route_r10_supported": trajectory_r10_supported,
        "blocker_assessment": blocker_assessment,
        "scientific_read": scientific_read,
        "score_records": score_records,
        "contract": (
            "R12-07 does not change predictions. It routes evidence by claim horizon: 1y/2y nowcasting uses DOH "
            "program lineages under an explicitly diagnostic short-horizon min-train contract, while 3y/5y trajectory "
            "claims use slide annual anchors under the production min-train contract. The route prevents quarterly "
            "program support from being over-interpreted as long-horizon trajectory validation."
        ),
    }


def _build_r12_08_route_aware_two_head_candidate_report(
    *,
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    r10_horizon_replay: dict[str, Any] | None,
    production_min_train_years: int,
    diagnostic_min_train_years: int = R12_SUPPORT_ADEQUACY_MIN_TRAIN_YEARS,
    candidate_family: str = "r12_route_aware_two_head_process",
    routes: tuple[dict[str, Any], ...] = R12_HORIZON_EVIDENCE_ROUTES,
    experiment_id: str = "R12-08",
    schema_version: str = "phase3_dynamic.r12_08_route_aware_two_head_candidate.v1",
    prediction_mutation: str = "enabled_route_aware_two_head",
    contract_text: str | None = None,
) -> dict[str, Any]:
    all_horizons = tuple(
        sorted(
            {
                int(horizon)
                for route in routes
                for horizon in list(route.get("horizons") or [])
            }
        )
    )
    reference_by_horizon = _r12_lineage_horizon_reference_map(r10_horizon_replay, horizons=all_horizons)
    score_records: list[dict[str, Any]] = []
    lineage_manifests: list[dict[str, Any]] = []
    route_records_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        route_id = str(route.get("route_id") or "")
        min_train_years = _r12_07_route_min_train_years(
            route,
            production_min_train_years=production_min_train_years,
            diagnostic_min_train_years=diagnostic_min_train_years,
        )
        for lineage_id in list(route.get("lineage_ids") or []):
            lineage = _r12_lineage_by_id(str(lineage_id))
            lineage_records, lineage_manifest = _r12_05_lineage_score_records(
                rows=rows,
                lineage=lineage,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                horizons=tuple(int(horizon) for horizon in list(route.get("horizons") or [])),
                family=candidate_family,
            )
            annotated_records = [
                {
                    **dict(record),
                    "route_id": route_id,
                    "route_label": str(route.get("route_label") or ""),
                    "claim_role": str(route.get("claim_role") or ""),
                    "route_min_train_years": min_train_years,
                    "route_min_train_contract": str(route.get("min_train_contract") or ""),
                }
                for record in lineage_records
            ]
            route_records_by_id[route_id].extend(annotated_records)
            score_records.extend(annotated_records)
            lineage_manifests.append(
                {
                    **dict(lineage_manifest),
                    "route_id": route_id,
                    "route_label": str(route.get("route_label") or ""),
                    "claim_role": str(route.get("claim_role") or ""),
                    "route_min_train_years": min_train_years,
                    "route_min_train_contract": str(route.get("min_train_contract") or ""),
                }
            )
    route_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "route_label", "claim_role", "candidate_family", "horizon_years"),
    )
    route_lineage_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "lineage_id", "lineage_label", "candidate_family", "horizon_years"),
    )
    metric_route_horizon_rows = _r12_05_score_summary(
        score_records,
        r10_reference_by_horizon=reference_by_horizon,
        group_fields=("route_id", "metric_name", "candidate_family", "horizon_years"),
    )
    route_manifests = [
        _r12_07_route_manifest_rows(
            route=dict(route),
            route_records=route_records_by_id.get(str(route.get("route_id") or ""), []),
            route_horizon_rows=route_horizon_rows,
        )
        for route in routes
    ]
    missing_route_ids = [
        str(manifest.get("route_id") or "")
        for manifest in route_manifests
        if str(manifest.get("status") or "") != "evaluable"
    ]
    nowcast_manifests = [
        manifest
        for manifest in route_manifests
        if str(manifest.get("claim_role") or "") == "short_horizon_nowcast"
    ]
    trajectory_manifests = [
        manifest
        for manifest in route_manifests
        if str(manifest.get("claim_role") or "") == "long_horizon_trajectory"
    ]
    nowcast_carry_supported = bool(
        nowcast_manifests
        and all(bool(manifest.get("candidate_beats_carry_forward_all_route_horizons")) for manifest in nowcast_manifests)
    )
    nowcast_available_r10_supported = bool(
        nowcast_manifests
        and all(bool(manifest.get("candidate_beats_available_r10_all_route_horizons")) for manifest in nowcast_manifests)
    )
    nowcast_requires_r10 = bool(
        nowcast_manifests
        and any(bool(manifest.get("r10_required")) for manifest in nowcast_manifests)
    )
    nowcast_supported = bool(
        nowcast_carry_supported
        and (not nowcast_requires_r10 or nowcast_available_r10_supported)
    )
    trajectory_carry_supported = bool(
        trajectory_manifests
        and all(bool(manifest.get("candidate_beats_carry_forward_all_route_horizons")) for manifest in trajectory_manifests)
    )
    trajectory_r10_supported = bool(
        trajectory_manifests
        and all(bool(manifest.get("candidate_beats_available_r10_all_route_horizons")) for manifest in trajectory_manifests)
    )
    if missing_route_ids:
        blocker_assessment = "route_aware_two_head_has_underpowered_routes"
        decision = "reject_route_aware_candidate_for_support_gap"
        scientific_read = (
            "The route-aware two-head candidate cannot be adjudicated because at least one route lacks complete "
            "blocked scores."
        )
    elif nowcast_supported and trajectory_r10_supported:
        blocker_assessment = "route_aware_two_head_supports_nowcast_and_trajectory_claims"
        decision = "keep_as_route_aware_two_head_candidate"
        scientific_read = (
            "The route-aware two-head candidate clears the short-horizon nowcast carry-forward gate and the annual-anchor "
            "trajectory R10 gate. It can be promoted as a route-specific candidate subject to the full stock/rate checks."
        )
    elif nowcast_supported:
        blocker_assessment = "route_aware_nowcast_head_supported_but_trajectory_head_still_r10_blocked"
        decision = "keep_as_route_aware_nowcast_candidate"
        scientific_read = (
            "The route-aware two-head candidate improves the program nowcast route against carry-forward, but the "
            "annual-anchor trajectory route still fails the available R10 gate. Keep the nowcast claim separate; do not "
            "promote a long-horizon trajectory claim."
        )
    else:
        blocker_assessment = "route_aware_two_head_fails_nowcast_and_trajectory_promotion"
        decision = "reject_route_aware_candidate"
        scientific_read = (
            "The route-aware two-head candidate does not clear the short-horizon nowcast carry-forward gate or the "
            "long-horizon trajectory gate."
        )
    return {
        "schema_version": schema_version,
        "generated_at": _generated_at(),
        "experiment_id": experiment_id,
        "status": "completed" if score_records else "not_evaluable",
        "decision": decision,
        "candidate_family": candidate_family,
        "prediction_mutation": prediction_mutation,
        "reference_family": "multi_horizon_weighted_process",
        "reference_experiment_id": "R11-28",
        "production_min_train_years": int(production_min_train_years),
        "diagnostic_min_train_years": int(diagnostic_min_train_years),
        "routes": list(routes),
        "route_manifests": route_manifests,
        "lineage_manifests": lineage_manifests,
        "score_record_count": len(score_records),
        "route_horizon_rows": route_horizon_rows,
        "route_lineage_horizon_rows": route_lineage_horizon_rows,
        "metric_route_horizon_rows": metric_route_horizon_rows,
        "missing_route_ids": missing_route_ids,
        "nowcast_carry_gate_supported": nowcast_carry_supported,
        "nowcast_available_r10_gate_supported": nowcast_available_r10_supported,
        "nowcast_requires_r10": nowcast_requires_r10,
        "nowcast_gate_supported": nowcast_supported,
        "trajectory_carry_gate_supported": trajectory_carry_supported,
        "trajectory_r10_gate_supported": trajectory_r10_supported,
        "blocker_assessment": blocker_assessment,
        "scientific_read": scientific_read,
        "score_records": score_records,
        "contract": contract_text
        or (
            "R12-08 is a model change built on the R12-07 evidence router. It evaluates a deterministic two-head "
            "candidate: R11-28 for 1y/2y program nowcasting and R12-01 for 3y/5y annual-anchor trajectory. Promotion "
            "is route-specific: nowcast claims require improvement over carry-forward on program evidence; trajectory "
            "claims require improvement over carry-forward and all available matched R10 references on annual-anchor evidence."
        ),
    }


def _write_r12_07_horizon_router_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    horizon_rows = [dict(row) for row in list(report.get("route_horizon_rows") or []) if isinstance(row, dict)]
    lineage_rows = [dict(row) for row in list(report.get("route_lineage_horizon_rows") or []) if isinstance(row, dict)]
    metric_rows = [dict(row) for row in list(report.get("metric_route_horizon_rows") or []) if isinstance(row, dict)]
    if not horizon_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), constrained_layout=True)
    labels = [
        f"{str(row.get('route_id') or '')}\nh{int(row.get('horizon_years') or 0)}"
        for row in horizon_rows
    ]
    x = np.arange(len(labels), dtype=np.float64)
    candidate = np.asarray([float(row.get("candidate_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_norm_error") or np.nan) for row in horizon_rows], dtype=np.float64)
    r10 = np.asarray(
        [
            np.nan if _finite_float(row.get("r10_horizon_reference_mae")) is None else float(row["r10_horizon_reference_mae"])
            for row in horizon_rows
        ],
        dtype=np.float64,
    )
    width = 0.24
    axes[0].bar(x - width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x, candidate, width=width, color="#315f72", label="routed R11-28")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("R10-scope normalized MAE")
    axes[0].set_title("Route-horizon scores")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    lineage_rows = [
        row
        for row in lineage_rows
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    lineage_rows = sorted(
        lineage_rows,
        key=lambda row: float(row.get("candidate_mean_norm_error") or 0.0),
        reverse=True,
    )[:12]
    lineage_labels = [
        f"{str(row.get('route_id') or '')} h{int(row.get('horizon_years') or 0)} {str(row.get('lineage_id') or '')}"
        for row in lineage_rows
    ]
    lineage_values = [float(row.get("candidate_mean_norm_error") or 0.0) for row in lineage_rows]
    y = np.arange(len(lineage_labels), dtype=np.float64)
    axes[1].barh(y, lineage_values, color="#8a6f2a")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(lineage_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("candidate normalized MAE")
    axes[1].set_title("Lineage contribution")
    axes[1].grid(axis="x", alpha=0.25)

    metric_rows = [
        row
        for row in metric_rows
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    metric_rows = sorted(
        metric_rows,
        key=lambda row: float(row.get("candidate_mean_norm_error") or 0.0),
        reverse=True,
    )[:12]
    metric_labels = [
        f"{str(row.get('route_id') or '')} h{int(row.get('horizon_years') or 0)} {str(row.get('metric_name') or '')}"
        for row in metric_rows
    ]
    metric_values = [float(row.get("candidate_mean_norm_error") or 0.0) for row in metric_rows]
    y2 = np.arange(len(metric_labels), dtype=np.float64)
    axes[2].barh(y2, metric_values, color="#2f7d59")
    axes[2].set_yticks(y2)
    axes[2].set_yticklabels(metric_labels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("candidate normalized MAE")
    axes[2].set_title("Metric residuals by route")
    axes[2].grid(axis="x", alpha=0.25)
    experiment_id = str(report.get("experiment_id") or "R12")
    fig.suptitle(f"{experiment_id} horizon-specific evidence routing", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=260)
    plt.close(fig)


def _r12_reference_row(path: Path, report: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary") or {})
    r10_gate = dict(report.get("r10_lifted_gate") or {})
    r10_values = [
        float(value)
        for value in dict(summary.get("r10_horizon_reference_mae_by_horizon") or {}).values()
        if _finite_float(value) is not None
    ]
    return {
        "experiment_id": "R12-00",
        "title": "Promoted R11-28 research reference",
        "family": "multi_horizon_weighted_process",
        "artifact_path": path.as_posix(),
        "artifact_sha256": _sha256(path),
        "one_year_status": "pass",
        "annual_status": "not_applicable",
        "lifted_status": str(r10_gate.get("status") or "fail"),
        "stock_consistency_status": "pass",
        "candidate_mae": _finite_float(summary.get("candidate_mean_mae_across_horizons")),
        "carry_forward_mae": _finite_float(summary.get("carry_forward_mean_mae_across_horizons")),
        "r10_reference_mae": None if not r10_values else float(np.mean(np.asarray(r10_values, dtype=np.float64))),
        "decision": str(lock.get("decision") or "promote_as_next_research_reference"),
        "kept_claim": "next_wave_reference_not_publication_champion",
        "blockers": list(lock.get("r10_gate_blockers") or []),
        "contract": str(lock.get("contract") or ""),
    }


def _r12_reference_gate(candidate_report: dict[str, Any], reference_report: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = {
        int(row.get("horizon_years") or 0): dict(row)
        for row in list(candidate_report.get("horizon_rows") or [])
        if isinstance(row, dict)
    }
    reference_rows = {
        int(row.get("horizon_years") or 0): dict(row)
        for row in list(reference_report.get("horizon_rows") or [])
        if isinstance(row, dict)
    }
    blockers: list[str] = []
    gate_rows: list[dict[str, Any]] = []
    for horizon in (3, 5):
        candidate = candidate_rows.get(horizon, {})
        reference = reference_rows.get(horizon, {})
        candidate_scope = _finite_float(candidate.get("r10_comparable_candidate_mean_mae"))
        reference_scope = _finite_float(reference.get("r10_comparable_candidate_mean_mae"))
        candidate_full = _finite_float(candidate.get("candidate_mean_mae"))
        reference_full = _finite_float(reference.get("candidate_mean_mae"))
        row_blockers: list[str] = []
        if candidate_scope is None or reference_scope is None:
            row_blockers.append("missing_r10_scope_score")
        elif candidate_scope >= reference_scope:
            row_blockers.append("r10_scope_not_better_than_r11_28_reference")
        if candidate_full is None or reference_full is None:
            row_blockers.append("missing_full_score")
        elif candidate_full > reference_full:
            row_blockers.append("full_path_worse_than_r11_28_reference")
        blockers.extend([f"h{horizon}_{blocker}" for blocker in row_blockers])
        gate_rows.append(
            {
                "horizon_years": horizon,
                "candidate_r10_scope_mae": candidate_scope,
                "reference_r10_scope_mae": reference_scope,
                "candidate_minus_reference_r10_scope_mae": None
                if candidate_scope is None or reference_scope is None
                else float(candidate_scope - reference_scope),
                "candidate_full_mae": candidate_full,
                "reference_full_mae": reference_full,
                "candidate_minus_reference_full_mae": None
                if candidate_full is None or reference_full is None
                else float(candidate_full - reference_full),
                "status": "pass" if not row_blockers else "fail",
                "blockers": row_blockers,
            }
        )
    return {
        "schema_version": "phase3_dynamic.r12_reference_gate.v1",
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "horizon_rows": gate_rows,
        "contract": "R12 candidates must improve the locked R11-28 reference on both full and R10-scope 3y/5y horizons before being kept as next-wave candidates",
    }


def _apply_r12_reference_gate_to_row(
    row: dict[str, Any],
    *,
    candidate_report: dict[str, Any],
    reference_report: dict[str, Any],
) -> dict[str, Any]:
    output = dict(row)
    gate = _r12_reference_gate(candidate_report, reference_report)
    output["r12_reference_gate_status"] = str(gate.get("status") or "")
    output["r12_reference_gate"] = gate
    if str(gate.get("status") or "") != "pass":
        output["decision"] = "reject_for_reference_regression"
        output["kept_claim"] = "diagnostic_only"
        output["blockers"] = list(output.get("blockers") or []) + list(gate.get("blockers") or [])
    return output


def _write_r12_reference_dashboard(
    path: Path,
    reference_report: dict[str, Any],
    r12_report: dict[str, Any],
    process_report: dict[str, Any] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    reference_rows = {
        int(row.get("horizon_years") or 0): dict(row)
        for row in list(reference_report.get("horizon_rows") or [])
        if isinstance(row, dict)
    }
    r12_rows = {
        int(row.get("horizon_years") or 0): dict(row)
        for row in list(r12_report.get("horizon_rows") or [])
        if isinstance(row, dict)
    }
    process_rows = {
        int(row.get("horizon_years") or 0): dict(row)
        for row in list((process_report or {}).get("horizon_rows") or [])
        if isinstance(row, dict)
    }
    horizons = sorted(set(reference_rows) | set(r12_rows) | set(process_rows))
    if not horizons:
        return

    def _series(rows: dict[int, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if _finite_float(rows.get(horizon, {}).get(key)) is None else float(rows[horizon][key])
                for horizon in horizons
            ],
            dtype=np.float64,
        )

    reference_full = _series(reference_rows, "candidate_mean_mae")
    r12_full = _series(r12_rows, "candidate_mean_mae")
    process_full = _series(process_rows, "candidate_mean_mae")
    carry = _series(r12_rows, "carry_forward_mean_mae")
    reference_scope = _series(reference_rows, "r10_comparable_candidate_mean_mae")
    r12_scope = _series(r12_rows, "r10_comparable_candidate_mean_mae")
    process_scope = _series(process_rows, "r10_comparable_candidate_mean_mae")
    r10 = _series(r12_rows, "r10_horizon_reference_mae")
    r12_minus_reference = r12_scope - reference_scope
    process_minus_reference = process_scope - reference_scope
    r12_minus_r10 = r12_scope - r10

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)
    x = np.arange(len(horizons), dtype=np.float64)
    width = 0.18 if process_rows else 0.25
    axes[0].bar(x - 1.5 * width, carry, width=width, color="#9aa4b2", label="carry-forward")
    axes[0].bar(x - 0.5 * width, reference_full, width=width, color="#315f72", label="R11-28 reference")
    axes[0].bar(x + 0.5 * width, r12_full, width=width, color="#8a6f2a", label="R12-01 shape")
    if process_rows:
        axes[0].bar(x + 1.5 * width, process_full, width=width, color="#2f7d59", label="R12 process branch")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(horizon) for horizon in horizons])
    axes[0].set_xlabel("blocked horizon, years")
    axes[0].set_ylabel("full mean normalized MAE")
    axes[0].set_title("Full DATV path")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, reference_scope, marker="o", linewidth=2.0, color="#315f72", label="R11-28 R10 scope")
    axes[1].plot(x, r12_scope, marker="o", linewidth=2.0, color="#8a6f2a", label="R12-01 R10 scope")
    if process_rows:
        axes[1].plot(x, process_scope, marker="o", linewidth=2.0, color="#2f7d59", label="R12 process R10 scope")
    axes[1].plot(x, r10, marker="D", linewidth=1.8, color="#111827", label="matched R10")
    axes[1].bar(x, r12_minus_r10, width=0.18, alpha=0.25, color=["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in r12_minus_r10])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(horizon) for horizon in horizons])
    axes[1].set_xlabel("blocked horizon, years")
    axes[1].set_ylabel("R10-comparable MAE")
    axes[1].set_title("Remaining R10-scope gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    colors = ["#1f7a4d" if value < 0.0 else "#9b2f2f" for value in process_minus_reference]
    axes[2].bar([str(horizon) for horizon in horizons], process_minus_reference, color=colors)
    axes[2].axhline(0.0, color="#2b2b2b", linewidth=1.0)
    axes[2].set_xlabel("blocked horizon, years")
    axes[2].set_ylabel("R12 process minus R11-28 R10-scope MAE")
    axes[2].set_title("Residual/process value over reference")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("R12 residual/process branch against promoted R11-28 reference", fontsize=13)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def run_r12_reference_branch(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    epigraph_root: Path | None = None,
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
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    artifact_paths = _artifact_paths(phase3_root)
    reports = {name: _read_path_payload(path_text) for name, path_text in artifact_paths.items()}
    r10_horizon_replay = _build_r10_horizon_replay_report(
        root=root,
        horizons=R11_MULTI_HORIZON_YEARS,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
    )
    r10_reference_mae = _r10_reference(reports.get("incidence_full_gate")) or _r10_reference(reports.get("u_to_d_coupling"))
    benchmark_manifest = _build_benchmark_manifest(
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        artifact_paths=artifact_paths,
        reports=reports,
        r10_horizon_replay=r10_horizon_replay,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=1,
    )
    split_manifest = _build_split_manifest(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=1,
    )
    r11_28_reference = _r11_multi_horizon_report(
        experiment_id="R12-00-REFERENCE",
        family="multi_horizon_weighted_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_01 = _r11_multi_horizon_report(
        experiment_id="R12-01",
        family="r12_long_horizon_stock_shape_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_02 = _r11_multi_horizon_report(
        experiment_id="R12-02",
        family="r12_da_process_split_transition",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_03 = _r11_multi_horizon_report(
        experiment_id="R12-03",
        family="r12_da_residual_source_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_08 = _r11_multi_horizon_report(
        experiment_id="R12-08",
        family="r12_route_aware_two_head_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_09 = _r11_multi_horizon_report(
        experiment_id="R12-09",
        family="r12_stock_cone_safe_annual_trajectory_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r12_10 = _r11_multi_horizon_report(
        experiment_id="R12-10",
        family="r12_program_nowcast_mixed_quarterly_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )

    benchmark_path = analysis_dir / "benchmark_manifest.json"
    split_path = analysis_dir / "split_manifest.json"
    r10_replay_path = analysis_dir / "r10_horizon_matched_replay_report.json"
    reference_report_path = analysis_dir / "r12_00_promoted_r11_28_reference_report.json"
    reference_lock_path = analysis_dir / "r12_00_promoted_r11_28_reference_lock.json"
    r12_report_path = analysis_dir / "r12_01_long_horizon_stock_shape_report.json"
    r12_02_path = analysis_dir / "r12_02_da_process_split_transition_report.json"
    r12_02_anatomy_path = analysis_dir / "r12_02_da_process_split_residual_anatomy.json"
    r12_03_path = analysis_dir / "r12_03_da_residual_source_process_report.json"
    r12_03_source_path = analysis_dir / "r12_03_da_residual_source_alignment_report.json"
    r12_04_path = analysis_dir / "r12_04_source_lineage_evaluation_ablation_report.json"
    r12_04_dashboard_path = analysis_dir / "r12_04_source_lineage_evaluation_ablation_dashboard.png"
    r12_05_path = analysis_dir / "r12_05_lineage_stratified_training_evaluation_contract_report.json"
    r12_05_dashboard_path = analysis_dir / "r12_05_lineage_stratified_training_evaluation_contract_dashboard.png"
    r12_06_path = analysis_dir / "r12_06_doh_quarterly_support_adequacy_adjudication_report.json"
    r12_06_dashboard_path = analysis_dir / "r12_06_doh_quarterly_support_adequacy_adjudication_dashboard.png"
    r12_07_path = analysis_dir / "r12_07_horizon_specific_evidence_router_report.json"
    r12_07_dashboard_path = analysis_dir / "r12_07_horizon_specific_evidence_router_dashboard.png"
    r12_08_full_path = analysis_dir / "r12_08_route_aware_two_head_full_report.json"
    r12_08_route_path = analysis_dir / "r12_08_route_aware_two_head_candidate_report.json"
    r12_08_dashboard_path = analysis_dir / "r12_08_route_aware_two_head_candidate_dashboard.png"
    r12_09_full_path = analysis_dir / "r12_09_stock_cone_safe_annual_trajectory_full_report.json"
    r12_09_route_path = analysis_dir / "r12_09_stock_cone_safe_annual_trajectory_candidate_report.json"
    r12_09_dashboard_path = analysis_dir / "r12_09_stock_cone_safe_annual_trajectory_dashboard.png"
    r12_10_annual_challenge_path = analysis_dir / "r12_10a_official_annual_challenge_gate_report.json"
    r12_10_annual_challenge_dashboard_path = analysis_dir / "r12_10a_official_annual_challenge_gate_dashboard.png"
    r12_10_full_path = analysis_dir / "r12_10b_program_nowcast_mixed_quarterly_full_report.json"
    r12_10_route_path = analysis_dir / "r12_10b_program_nowcast_mixed_quarterly_candidate_report.json"
    r12_10_dashboard_path = analysis_dir / "r12_10b_program_nowcast_mixed_quarterly_dashboard.png"
    comparison_path = analysis_dir / "r12_reference_branch_comparison.json"
    csv_path = analysis_dir / "r12_reference_branch_comparison.csv"
    md_path = analysis_dir / "r12_reference_branch_comparison.md"
    dashboard_path = analysis_dir / "r12_reference_branch_dashboard.png"

    write_json(benchmark_path, benchmark_manifest)
    write_json(split_path, split_manifest)
    write_json(r10_replay_path, r10_horizon_replay)
    write_json(reference_report_path, r11_28_reference)
    reference_lock = _build_r12_reference_lock(
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        reference_report_path=reference_report_path,
        reference_report=r11_28_reference,
    )
    write_json(reference_lock_path, reference_lock)
    write_json(r12_report_path, r12_01)
    write_json(r12_02_path, r12_02)
    r12_02_anatomy = _build_r12_02_process_split_anatomy_report(r12_02)
    write_json(r12_02_anatomy_path, r12_02_anatomy)
    write_json(r12_03_path, r12_03)
    write_json(r12_08_full_path, r12_08)
    write_json(r12_09_full_path, r12_09)
    write_json(r12_10_full_path, r12_10)
    r12_03_source_report = _build_r12_03_residual_source_report(r12_03)
    write_json(r12_03_source_path, r12_03_source_report)
    r12_04_report = _build_r12_04_source_lineage_ablation_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        r10_horizon_replay=r10_horizon_replay,
    )
    write_json(r12_04_path, r12_04_report)
    _write_r12_04_lineage_dashboard(r12_04_dashboard_path, r12_04_report)
    r12_05_report = _build_r12_05_lineage_stratified_contract_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        r10_horizon_replay=r10_horizon_replay,
    )
    write_json(r12_05_path, r12_05_report)
    _write_r12_05_lineage_dashboard(r12_05_dashboard_path, r12_05_report)
    r12_06_report = _build_r12_06_doh_quarterly_support_adjudication_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        r10_horizon_replay=r10_horizon_replay,
        production_min_train_years=min_train_years,
    )
    write_json(r12_06_path, r12_06_report)
    _write_r12_06_support_dashboard(r12_06_dashboard_path, r12_06_report)
    r12_07_report = _build_r12_07_horizon_specific_evidence_router_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        r10_horizon_replay=r10_horizon_replay,
        production_min_train_years=min_train_years,
    )
    write_json(r12_07_path, r12_07_report)
    _write_r12_07_horizon_router_dashboard(r12_07_dashboard_path, r12_07_report)
    r12_08_route_report = _build_r12_08_route_aware_two_head_candidate_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        r10_horizon_replay=r10_horizon_replay,
        production_min_train_years=min_train_years,
    )
    write_json(r12_08_route_path, r12_08_route_report)
    _write_r12_07_horizon_router_dashboard(r12_08_dashboard_path, r12_08_route_report)
    r12_09_route_report = _build_r12_08_route_aware_two_head_candidate_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        r10_horizon_replay=r10_horizon_replay,
        production_min_train_years=min_train_years,
        candidate_family="r12_stock_cone_safe_annual_trajectory_process",
        experiment_id="R12-09",
        schema_version="phase3_dynamic.r12_09_stock_cone_safe_annual_trajectory_candidate.v1",
        prediction_mutation="enabled_stock_cone_safe_annual_trajectory_head",
        contract_text=(
            "R12-09 is a model change built on the R12-07 evidence router. It keeps R11-28 for program-nowcast "
            "evidence and non-annual trajectory rows, while slide annual-anchor diagnosed_plhiv/alive_on_art rows "
            "may use a train-selected annual stock head. Promotion remains route-specific: nowcast claims require "
            "improvement over carry-forward on program evidence; trajectory claims require improvement over carry-forward "
            "and all available matched R10 references on annual-anchor evidence, with the full stock cone and "
            "conditional-rate gates still scored in the companion full report."
        ),
    )
    write_json(r12_09_route_path, r12_09_route_report)
    _write_r12_07_horizon_router_dashboard(r12_09_dashboard_path, r12_09_route_report)
    r12_10_annual_challenge_report = _build_r12_official_annual_challenge_gate_report(
        rows=validation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        candidate_families=(
            "multi_horizon_weighted_process",
            "r12_stock_cone_safe_annual_trajectory_process",
            "r12_program_nowcast_mixed_quarterly_process",
        ),
    )
    write_json(r12_10_annual_challenge_path, r12_10_annual_challenge_report)
    _write_r12_official_annual_challenge_dashboard(
        r12_10_annual_challenge_dashboard_path,
        r12_10_annual_challenge_report,
    )
    r12_10_route_report = _build_r12_08_route_aware_two_head_candidate_report(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        r10_horizon_replay=r10_horizon_replay,
        production_min_train_years=min_train_years,
        candidate_family="r12_program_nowcast_mixed_quarterly_process",
        routes=R12_PROGRAM_MIXED_QUARTERLY_ROUTES,
        experiment_id="R12-10B",
        schema_version="phase3_dynamic.r12_10b_program_nowcast_mixed_quarterly_candidate.v1",
        prediction_mutation="enabled_program_nowcast_mixed_quarterly_head",
        contract_text=(
            "R12-10B targets the remaining R10 failure on DOH program nowcast and mixed quarterly diagnosis/ART "
            "trajectory. It uses only DOH quarterly/monthly program lineages, requires carry-forward and available "
            "matched R10 improvement on those routes, and leaves the slide annual-anchor route frozen."
        ),
    )
    write_json(r12_10_route_path, r12_10_route_report)
    _write_r12_07_horizon_router_dashboard(r12_10_dashboard_path, r12_10_route_report)

    r12_01_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-01",
            title="Long-horizon stock-shape correction on promoted R11-28",
            path=r12_report_path,
            report=r12_01,
        ),
        candidate_report=r12_01,
        reference_report=r11_28_reference,
    )
    r12_02_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-02",
            title="D/A process-split transition on promoted R11-28",
            path=r12_02_path,
            report=r12_02,
        ),
        candidate_report=r12_02,
        reference_report=r11_28_reference,
    )
    r12_03_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-03",
            title="D/A residual-source process on promoted R11-28",
            path=r12_03_path,
            report=r12_03,
        ),
        candidate_report=r12_03,
        reference_report=r11_28_reference,
    )
    r12_08_full_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-08",
            title="Route-aware two-head nowcast/trajectory candidate",
            path=r12_08_full_path,
            report=r12_08,
        ),
        candidate_report=r12_08,
        reference_report=r11_28_reference,
    )
    r12_09_full_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-09",
            title="Stock-cone-safe annual trajectory head",
            path=r12_09_full_path,
            report=r12_09,
        ),
        candidate_report=r12_09,
        reference_report=r11_28_reference,
    )
    r12_04_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_04_report.get("horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_04_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_04_report.get("horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_04_row = {
        "experiment_id": "R12-04",
        "title": "Source-family/support-partition evaluation ablation",
        "family": "source_lineage_evaluation_ablation",
        "artifact_path": r12_04_path.as_posix(),
        "artifact_sha256": _sha256(r12_04_path),
        "one_year_status": "not_applicable",
        "annual_status": "not_applicable",
        "lifted_status": "diagnostic_only",
        "stock_consistency_status": "not_applicable",
        "candidate_mae": None
        if not r12_04_horizon_values
        else float(np.mean(np.asarray(r12_04_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None,
        "r10_reference_mae": None
        if not r12_04_r10_values
        else float(np.mean(np.asarray(r12_04_r10_values, dtype=np.float64))),
        "decision": str(r12_04_report.get("decision") or "keep_as_observation_lineage_diagnostic"),
        "kept_claim": str(r12_04_report.get("blocker_assessment") or "observation_lineage_diagnostic"),
        "blockers": [],
        "contract": str(r12_04_report.get("contract") or ""),
    }
    r12_05_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_05_report.get("horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_05_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_05_report.get("horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_LINEAGE_DIAGNOSTIC_HORIZONS)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_05_row = {
        "experiment_id": "R12-05",
        "title": "Lineage-stratified R11-28 train/evaluate contract",
        "family": "lineage_stratified_training_evaluation_contract",
        "artifact_path": r12_05_path.as_posix(),
        "artifact_sha256": _sha256(r12_05_path),
        "one_year_status": "not_applicable",
        "annual_status": "not_applicable",
        "lifted_status": "diagnostic_only",
        "stock_consistency_status": "not_applicable",
        "candidate_mae": None
        if not r12_05_horizon_values
        else float(np.mean(np.asarray(r12_05_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None,
        "r10_reference_mae": None
        if not r12_05_r10_values
        else float(np.mean(np.asarray(r12_05_r10_values, dtype=np.float64))),
        "decision": str(r12_05_report.get("decision") or "keep_as_lineage_stratified_observation_operator_contract"),
        "kept_claim": str(r12_05_report.get("blocker_assessment") or "lineage_stratified_contract"),
        "blockers": [],
        "contract": str(r12_05_report.get("contract") or ""),
    }
    r12_06_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_06_report.get("short_horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_SUPPORT_ADEQUACY_HORIZONS)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_06_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_06_report.get("short_horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_SUPPORT_ADEQUACY_HORIZONS)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_06_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_06_report.get("short_horizon_rows") or [])
        if isinstance(row, dict)
        and int(row.get("horizon_years") or 0) in set(R12_SUPPORT_ADEQUACY_HORIZONS)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_06_row = {
        "experiment_id": "R12-06",
        "title": "DOH-quarterly support adequacy and bridge adjudication",
        "family": "doh_quarterly_support_adequacy_adjudication",
        "artifact_path": r12_06_path.as_posix(),
        "artifact_sha256": _sha256(r12_06_path),
        "one_year_status": "diagnostic_only",
        "annual_status": "not_applicable",
        "lifted_status": "diagnostic_only",
        "stock_consistency_status": "not_applicable",
        "candidate_mae": None
        if not r12_06_horizon_values
        else float(np.mean(np.asarray(r12_06_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_06_carry_values
        else float(np.mean(np.asarray(r12_06_carry_values, dtype=np.float64))),
        "r10_reference_mae": None
        if not r12_06_r10_values
        else float(np.mean(np.asarray(r12_06_r10_values, dtype=np.float64))),
        "decision": str(r12_06_report.get("decision") or "keep_as_doh_quarterly_support_adequacy_adjudication"),
        "kept_claim": str(r12_06_report.get("blocker_assessment") or "doh_quarterly_support_adequacy_adjudication"),
        "blockers": list(dict(r12_06_report.get("bridge_consistency") or {}).get("blockers") or []),
        "contract": str(r12_06_report.get("contract") or ""),
    }
    r12_07_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_07_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_07_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_07_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_07_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_07_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_07_row = {
        "experiment_id": "R12-07",
        "title": "Horizon-specific evidence router",
        "family": "horizon_specific_evidence_router",
        "artifact_path": r12_07_path.as_posix(),
        "artifact_sha256": _sha256(r12_07_path),
        "one_year_status": "diagnostic_only",
        "annual_status": "not_applicable",
        "lifted_status": "diagnostic_only",
        "stock_consistency_status": "not_applicable",
        "candidate_mae": None
        if not r12_07_horizon_values
        else float(np.mean(np.asarray(r12_07_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_07_carry_values
        else float(np.mean(np.asarray(r12_07_carry_values, dtype=np.float64))),
        "r10_reference_mae": None
        if not r12_07_r10_values
        else float(np.mean(np.asarray(r12_07_r10_values, dtype=np.float64))),
        "decision": str(r12_07_report.get("decision") or "keep_as_horizon_specific_evidence_router"),
        "kept_claim": str(r12_07_report.get("blocker_assessment") or "horizon_specific_evidence_router"),
        "blockers": [
            str(item)
            for manifest in list(r12_07_report.get("route_manifests") or [])
            if isinstance(manifest, dict)
            for item in list(manifest.get("blockers") or [])
        ],
        "contract": str(r12_07_report.get("contract") or ""),
    }
    r12_08_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_08_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_08_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_08_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_08_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_08_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_08_route_blockers = [
        str(item)
        for manifest in list(r12_08_route_report.get("route_manifests") or [])
        if isinstance(manifest, dict)
        for item in list(manifest.get("blockers") or [])
    ]
    r12_08_full_reference_blockers = list(r12_08_full_row.get("blockers") or [])
    r12_08_decision = str(r12_08_route_report.get("decision") or "reject_route_aware_candidate")
    if (
        r12_08_decision == "keep_as_route_aware_two_head_candidate"
        and str(r12_08_full_row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ):
        r12_08_decision = "keep_as_full_cascade_candidate"
    r12_08_row = {
        "experiment_id": "R12-08",
        "title": "Route-aware two-head nowcast/trajectory candidate",
        "family": "r12_route_aware_two_head_process",
        "artifact_path": r12_08_route_path.as_posix(),
        "artifact_sha256": _sha256(r12_08_route_path),
        "one_year_status": str(r12_08_full_row.get("one_year_status") or ""),
        "annual_status": "not_applicable",
        "lifted_status": str(r12_08_full_row.get("lifted_status") or ""),
        "stock_consistency_status": str(r12_08_full_row.get("stock_consistency_status") or ""),
        "candidate_mae": None
        if not r12_08_horizon_values
        else float(np.mean(np.asarray(r12_08_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_08_carry_values
        else float(np.mean(np.asarray(r12_08_carry_values, dtype=np.float64))),
        "r10_reference_mae": None
        if not r12_08_r10_values
        else float(np.mean(np.asarray(r12_08_r10_values, dtype=np.float64))),
        "decision": r12_08_decision,
        "kept_claim": str(r12_08_route_report.get("blocker_assessment") or "route_aware_two_head_candidate"),
        "blockers": r12_08_route_blockers + r12_08_full_reference_blockers,
        "contract": str(r12_08_route_report.get("contract") or ""),
    }
    r12_09_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_09_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_09_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_09_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_09_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_09_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_09_route_blockers = [
        str(item)
        for manifest in list(r12_09_route_report.get("route_manifests") or [])
        if isinstance(manifest, dict)
        for item in list(manifest.get("blockers") or [])
    ]
    r12_09_full_reference_blockers = list(r12_09_full_row.get("blockers") or [])
    r12_09_decision = str(r12_09_route_report.get("decision") or "reject_route_aware_candidate")
    if (
        r12_09_decision == "keep_as_route_aware_two_head_candidate"
        and str(r12_09_full_row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ):
        r12_09_decision = "keep_as_full_cascade_candidate"
    r12_09_row = {
        "experiment_id": "R12-09",
        "title": "Stock-cone-safe annual trajectory head",
        "family": "r12_stock_cone_safe_annual_trajectory_process",
        "artifact_path": r12_09_route_path.as_posix(),
        "artifact_sha256": _sha256(r12_09_route_path),
        "one_year_status": str(r12_09_full_row.get("one_year_status") or ""),
        "annual_status": "not_applicable",
        "lifted_status": str(r12_09_full_row.get("lifted_status") or ""),
        "stock_consistency_status": str(r12_09_full_row.get("stock_consistency_status") or ""),
        "candidate_mae": None
        if not r12_09_horizon_values
        else float(np.mean(np.asarray(r12_09_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_09_carry_values
        else float(np.mean(np.asarray(r12_09_carry_values, dtype=np.float64))),
        "r10_reference_mae": None
        if not r12_09_r10_values
        else float(np.mean(np.asarray(r12_09_r10_values, dtype=np.float64))),
        "decision": r12_09_decision,
        "kept_claim": str(r12_09_route_report.get("blocker_assessment") or "stock_cone_safe_annual_trajectory"),
        "blockers": r12_09_route_blockers + r12_09_full_reference_blockers,
        "contract": str(r12_09_route_report.get("contract") or ""),
    }
    r12_10_full_row = _apply_r12_reference_gate_to_row(
        _summarize_r11_multi_horizon_branch(
            experiment_id="R12-10B-FULL",
            title="Program nowcast and mixed-quarterly full-cascade replay",
            path=r12_10_full_path,
            report=r12_10,
        ),
        candidate_report=r12_10,
        reference_report=r11_28_reference,
    )
    r12_10a_family_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_10_annual_challenge_report.get("family_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_10a_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_10_annual_challenge_report.get("family_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_10a_row = {
        "experiment_id": "R12-10A",
        "title": "Official annual AEM/Spectrum-style challenge gate",
        "family": "official_annual_challenge_gate",
        "artifact_path": r12_10_annual_challenge_path.as_posix(),
        "artifact_sha256": _sha256(r12_10_annual_challenge_path),
        "one_year_status": "not_applicable",
        "annual_status": str(r12_10_annual_challenge_report.get("status") or "not_evaluable"),
        "lifted_status": "not_applicable",
        "stock_consistency_status": "not_applicable",
        "candidate_mae": None
        if not r12_10a_family_values
        else float(np.mean(np.asarray(r12_10a_family_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_10a_carry_values
        else float(np.mean(np.asarray(r12_10a_carry_values, dtype=np.float64))),
        "r10_reference_mae": None,
        "decision": str(r12_10_annual_challenge_report.get("decision") or "keep_as_official_annual_challenge_gate"),
        "kept_claim": "official_annual_validation_gate_not_training_target",
        "blockers": list(r12_10_annual_challenge_report.get("blockers") or []),
        "contract": str(r12_10_annual_challenge_report.get("contract") or ""),
    }
    r12_10_horizon_values = [
        float(row["candidate_mean_norm_error"])
        for row in list(r12_10_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    r12_10_carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in list(r12_10_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    r12_10_r10_values = [
        float(row["r10_horizon_reference_mae"])
        for row in list(r12_10_route_report.get("route_horizon_rows") or [])
        if isinstance(row, dict)
        and _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    r12_10_route_blockers = [
        str(item)
        for manifest in list(r12_10_route_report.get("route_manifests") or [])
        if isinstance(manifest, dict)
        for item in list(manifest.get("blockers") or [])
    ]
    r12_10_full_reference_blockers = list(r12_10_full_row.get("blockers") or [])
    r12_10_decision = str(r12_10_route_report.get("decision") or "reject_route_aware_candidate")
    if (
        r12_10_decision == "keep_as_route_aware_two_head_candidate"
        and str(r12_10_full_row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ):
        r12_10_decision = "keep_as_full_cascade_candidate"
    r12_10_row = {
        "experiment_id": "R12-10B",
        "title": "DOH program nowcast and mixed-quarterly trajectory branch",
        "family": "r12_program_nowcast_mixed_quarterly_process",
        "artifact_path": r12_10_route_path.as_posix(),
        "artifact_sha256": _sha256(r12_10_route_path),
        "one_year_status": str(r12_10_full_row.get("one_year_status") or ""),
        "annual_status": "not_applicable",
        "lifted_status": str(r12_10_full_row.get("lifted_status") or ""),
        "stock_consistency_status": str(r12_10_full_row.get("stock_consistency_status") or ""),
        "candidate_mae": None
        if not r12_10_horizon_values
        else float(np.mean(np.asarray(r12_10_horizon_values, dtype=np.float64))),
        "carry_forward_mae": None
        if not r12_10_carry_values
        else float(np.mean(np.asarray(r12_10_carry_values, dtype=np.float64))),
        "r10_reference_mae": None
        if not r12_10_r10_values
        else float(np.mean(np.asarray(r12_10_r10_values, dtype=np.float64))),
        "decision": r12_10_decision,
        "kept_claim": str(r12_10_route_report.get("blocker_assessment") or "program_nowcast_mixed_quarterly_candidate"),
        "blockers": r12_10_route_blockers + r12_10_full_reference_blockers,
        "contract": str(r12_10_route_report.get("contract") or ""),
    }
    rows_out = [
        _infrastructure_row(
            experiment_id="BM-00",
            title="R12 benchmark manifest",
            decision="keep_as_contract",
            kept_claim="benchmark_lock",
            artifact_path=benchmark_path,
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="BM-01",
            title="R12 split manifest",
            decision="keep_as_contract",
            kept_claim="split_lock",
            artifact_path=split_path,
            status="pass" if split_manifest["split_count"] else "fail",
            blockers=[] if split_manifest["split_count"] else ["no_blocked_splits"],
        ),
        _infrastructure_row(
            experiment_id="BM-02",
            title="Horizon-matched R10 replay",
            decision="keep_as_benchmark"
            if str(r10_horizon_replay.get("status") or "") == "pass"
            else "blocked_missing_horizon_matched_r10",
            kept_claim="horizon_matched_r10_reference",
            artifact_path=r10_replay_path,
            status="pass" if str(r10_horizon_replay.get("status") or "") == "pass" else "blocked",
        ),
        _r12_reference_row(reference_report_path, r11_28_reference, reference_lock),
        r12_01_row,
        r12_02_row,
        r12_03_row,
        r12_04_row,
        r12_05_row,
        r12_06_row,
        r12_07_row,
        r12_08_row,
        r12_09_row,
        r12_10a_row,
        r12_10_row,
    ]
    promoted_rows = [
        row
        for row in rows_out
        if str(row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ]
    comparison = {
        "schema_version": R12_REFERENCE_BRANCH_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "loop_variant": "benchmark-hardening-loop",
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "rows": rows_out,
        "summary": {
            "experiment_count": len(rows_out),
            "full_cascade_champion": None if not promoted_rows else promoted_rows[0].get("experiment_id"),
            "reference_experiment_id": "R11-28",
            "candidate_experiment_ids": ["R12-01", "R12-02", "R12-03", "R12-08", "R12-09", "R12-10B"],
            "diagnostic_experiment_ids": ["R12-04", "R12-05", "R12-06", "R12-07", "R12-10A"],
            "claim_boundary": (
                f"{promoted_rows[0].get('experiment_id')} promoted to full-cascade candidate"
                if promoted_rows
                else "R11-28 is locked as research reference; R12 candidates are accepted only if they improve long-horizon drift without failing stock/rate/R10 gates"
            ),
        },
        "artifact_paths": {
            "benchmark_manifest": benchmark_path.as_posix(),
            "split_manifest": split_path.as_posix(),
            "r10_horizon_matched_replay_report": r10_replay_path.as_posix(),
            "r12_00_promoted_r11_28_reference_report": reference_report_path.as_posix(),
            "r12_00_promoted_r11_28_reference_lock": reference_lock_path.as_posix(),
            "r12_01_long_horizon_stock_shape_report": r12_report_path.as_posix(),
            "r12_02_da_process_split_transition_report": r12_02_path.as_posix(),
            "r12_02_da_process_split_residual_anatomy": r12_02_anatomy_path.as_posix(),
            "r12_03_da_residual_source_process_report": r12_03_path.as_posix(),
            "r12_03_da_residual_source_alignment_report": r12_03_source_path.as_posix(),
            "r12_08_route_aware_two_head_full_report": r12_08_full_path.as_posix(),
            "r12_09_stock_cone_safe_annual_trajectory_full_report": r12_09_full_path.as_posix(),
            "r12_10b_program_nowcast_mixed_quarterly_full_report": r12_10_full_path.as_posix(),
            "r12_04_source_lineage_evaluation_ablation_report": r12_04_path.as_posix(),
            "r12_04_source_lineage_evaluation_ablation_dashboard": r12_04_dashboard_path.as_posix(),
            "r12_05_lineage_stratified_training_evaluation_contract_report": r12_05_path.as_posix(),
            "r12_05_lineage_stratified_training_evaluation_contract_dashboard": r12_05_dashboard_path.as_posix(),
            "r12_06_doh_quarterly_support_adequacy_adjudication_report": r12_06_path.as_posix(),
            "r12_06_doh_quarterly_support_adequacy_adjudication_dashboard": r12_06_dashboard_path.as_posix(),
            "r12_07_horizon_specific_evidence_router_report": r12_07_path.as_posix(),
            "r12_07_horizon_specific_evidence_router_dashboard": r12_07_dashboard_path.as_posix(),
            "r12_08_route_aware_two_head_candidate_report": r12_08_route_path.as_posix(),
            "r12_08_route_aware_two_head_candidate_dashboard": r12_08_dashboard_path.as_posix(),
            "r12_09_stock_cone_safe_annual_trajectory_candidate_report": r12_09_route_path.as_posix(),
            "r12_09_stock_cone_safe_annual_trajectory_dashboard": r12_09_dashboard_path.as_posix(),
            "r12_10a_official_annual_challenge_gate_report": r12_10_annual_challenge_path.as_posix(),
            "r12_10a_official_annual_challenge_gate_dashboard": r12_10_annual_challenge_dashboard_path.as_posix(),
            "r12_10b_program_nowcast_mixed_quarterly_candidate_report": r12_10_route_path.as_posix(),
            "r12_10b_program_nowcast_mixed_quarterly_dashboard": r12_10_dashboard_path.as_posix(),
            "comparison_json": comparison_path.as_posix(),
            "comparison_csv": csv_path.as_posix(),
            "comparison_markdown": md_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(comparison_path, comparison)
    _write_csv(csv_path, rows_out)
    md_path.write_text(_markdown_table(rows_out).replace("R11 Experiment Comparison", "R12 Reference Branch Comparison"), encoding="utf-8")
    _write_r12_reference_dashboard(dashboard_path, r11_28_reference, r12_01, process_report=r12_03)
    return comparison


def run_r11_first_batch(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    epigraph_root: Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")

    rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    ledger = build_observation_role_ledger(
        root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    artifact_paths = _artifact_paths(phase3_root)
    reports = {name: _read_path_payload(path_text) for name, path_text in artifact_paths.items()}
    r10_horizon_replay = _build_r10_horizon_replay_report(
        root=root,
        horizons=R11_MULTI_HORIZON_YEARS,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
    )

    benchmark_manifest = _build_benchmark_manifest(
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        artifact_paths=artifact_paths,
        reports=reports,
        r10_horizon_replay=r10_horizon_replay,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    split_manifest = _build_split_manifest(
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    validation_panel = _build_aem_spectrum_validation_panel(validation_rows)
    score_contract = _build_score_contract()

    benchmark_path = analysis_dir / "benchmark_manifest.json"
    split_path = analysis_dir / "split_manifest.json"
    validation_path = analysis_dir / "aem_spectrum_validation_panel.json"
    score_path = analysis_dir / "score_contract.json"
    r10_horizon_replay_path = analysis_dir / "r10_horizon_matched_replay_report.json"
    write_json(benchmark_path, benchmark_manifest)
    write_json(split_path, split_manifest)
    write_json(validation_path, validation_panel)
    write_json(score_path, score_contract)
    write_json(r10_horizon_replay_path, r10_horizon_replay)

    branch_rows = [
        _summarize_existing_branch(
            experiment_id="ANNUAL_INCIDENCE_READOUT",
            title="Annual incidence weak measurement readout",
            path_text=artifact_paths.get("incidence_full_gate"),
            report=reports.get("incidence_full_gate"),
            claim_scope="full_cascade",
        ),
        _summarize_existing_branch(
            experiment_id="U_TO_D_COUPLING",
            title="Selective diagnosis-flow state/readout coupling",
            path_text=artifact_paths.get("u_to_d_coupling"),
            report=reports.get("u_to_d_coupling"),
            claim_scope="short_horizon_diagnosis_state_coupling",
        ),
        _summarize_existing_branch(
            experiment_id="FRONTDOOR_COUPLING",
            title="U_to_D plus D_to_A front-door coupling",
            path_text=artifact_paths.get("frontdoor_coupling"),
            report=reports.get("frontdoor_coupling"),
            claim_scope="short_horizon_frontdoor_cascade_coupling",
        ),
    ]
    phase2_report = reports.get("phase2_determinant_robustness")
    default_edge_count = (
        _finite_float(
            (phase2_report or {}).get(
                "phase3_default_allowed_direct_edge_count",
                (phase2_report or {}).get("default_allowed_direct_edge_count"),
            )
        )
        if isinstance(phase2_report, dict)
        else None
    )
    branch_rows.append(
        {
            "experiment_id": "PHASE2_DETERMINANT_PRIORS",
            "title": "Phase2 determinant direct-edge prior gate",
            "family": "phase2_determinant_robustness",
            "artifact_path": artifact_paths.get("phase2_determinant_robustness"),
            "artifact_sha256": None
            if not artifact_paths.get("phase2_determinant_robustness")
            else _sha256(Path(str(artifact_paths.get("phase2_determinant_robustness")))),
            "one_year_status": "not_applicable",
            "annual_status": "not_applicable",
            "lifted_status": "not_applicable",
            "stock_consistency_status": "blocked",
            "candidate_mae": None,
            "carry_forward_mae": None,
            "r10_reference_mae": None,
            "decision": "reject_for_default_priors" if default_edge_count == 0 else "needs_r11_gate",
            "kept_claim": "none",
            "blockers": ["default_allowed_direct_edge_count_is_zero"] if default_edge_count == 0 else [],
            "contract": "Phase2 terms cannot enter R11 as default priors until source-family and blocked-time gates pass",
        }
    )

    r11_01 = _build_r11_01_reconciliation_report(rows)
    r11_02 = _build_r11_02_support_weighted_report(rows, ledger)
    r11_03 = _build_r11_03_stock_gate_report(branch_rows)
    r10_reference_mae = _r10_reference(reports.get("incidence_full_gate")) or _r10_reference(reports.get("u_to_d_coupling"))
    r11_05 = _r11_candidate_report(
        experiment_id="R11-05",
        family="metric_selector",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_04 = _r11_candidate_report(
        experiment_id="R11-04",
        family="support_reporting_bias",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_06 = _r11_candidate_report(
        experiment_id="R11-06",
        family="local_level_filter",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_07 = _r11_candidate_report(
        experiment_id="R11-07",
        family="transition_shrinkage",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_08 = _r11_candidate_report(
        experiment_id="R11-08",
        family="linkage_lag_kernel",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_09 = _r11_candidate_report(
        experiment_id="R11-09",
        family="support_partition_calibration",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_08_09 = _r11_candidate_report(
        experiment_id="R11-08+09",
        family="linkage_lag_plus_support_partition",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_13 = _r11_candidate_report(
        experiment_id="R11-13",
        family="r10_style_readout_teacher",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_14 = _r11_candidate_report(
        experiment_id="R11-14",
        family="back_half_conditional_rates",
        rows=rows,
        splits=list(split_manifest.get("splits") or []),
        r10_reference_mae=r10_reference_mae,
    )
    r11_15 = _r11_multi_horizon_report(
        experiment_id="R11-15",
        family="back_half_conditional_rates",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r11_16 = _r11_multi_horizon_report(
        experiment_id="R11-16",
        family="trajectory_shape_head",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r11_17 = _r11_multi_horizon_report(
        experiment_id="R11-17",
        family="constrained_trajectory_shape_head",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r11_18 = _r11_multi_horizon_report(
        experiment_id="R11-18",
        family="horizon_adaptive_constrained_shape_head",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r11_19 = _r11_multi_horizon_report(
        experiment_id="R11-19",
        family="datv_transition_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    r11_20 = _r11_multi_horizon_report(
        experiment_id="R11-20",
        family="era_datv_transition_process",
        rows=rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=R11_MULTI_HORIZON_YEARS,
        r10_reference_mae=r10_reference_mae,
        r10_horizon_replay=r10_horizon_replay,
    )
    extra_multi_horizon_specs: tuple[tuple[str, str, str, str], ...] = (
        (
            "R11-21",
            "horizon_family_selector",
            "Horizon-family selector over R11-17 and R11-20",
            "r11_21_horizon_family_selector_report.json",
        ),
        (
            "R11-22",
            "diagnosis_flow_input_repair_process",
            "Diagnosis-flow input repair process",
            "r11_22_diagnosis_flow_input_repair_report.json",
        ),
        (
            "R11-23",
            "support_era_diagnosis_flow_process",
            "Support-era diagnosis-flow reporting process",
            "r11_23_support_era_diagnosis_flow_report.json",
        ),
        (
            "R11-24",
            "stock_flow_reconciliation_process",
            "Stock-flow reconciliation process",
            "r11_24_stock_flow_reconciliation_report.json",
        ),
        (
            "R11-25",
            "diagnosed_reporting_bias_process",
            "Diagnosed-stock reporting-bias process",
            "r11_25_diagnosed_reporting_bias_report.json",
        ),
        (
            "R11-26",
            "art_horizon_selector_process",
            "ART-specific horizon selector process",
            "r11_26_art_horizon_selector_report.json",
        ),
        (
            "R11-27",
            "diagnosis_lag_stock_process",
            "Diagnosis-flow lag into stock process",
            "r11_27_diagnosis_lag_stock_process_report.json",
        ),
        (
            "R11-28",
            "multi_horizon_weighted_process",
            "Multi-horizon weighted R11-17/R11-20 process",
            "r11_28_multi_horizon_weighted_process_report.json",
        ),
        (
            "R11-29",
            "r10_scope_teacher_stock_process",
            "R10-scope teacher constrained by stock process",
            "r11_29_r10_scope_teacher_stock_process_report.json",
        ),
        (
            "R11-32",
            "conditional_rate_horizon_selector",
            "Conditional-rate horizon selector",
            "r11_32_conditional_rate_horizon_selector_report.json",
        ),
    )
    extra_r11_reports: dict[str, dict[str, Any]] = {}
    for experiment_id, family, title, filename in extra_multi_horizon_specs:
        extra_r11_reports[experiment_id] = {
            "family": family,
            "title": title,
            "path": analysis_dir / filename,
            "report": _r11_multi_horizon_report(
                experiment_id=experiment_id,
                family=family,
                rows=rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                horizons=R11_MULTI_HORIZON_YEARS,
                r10_reference_mae=r10_reference_mae,
                r10_horizon_replay=r10_horizon_replay,
            ),
        }
    r11_01_path = analysis_dir / "r11_01_reconciliation_report.json"
    r11_02_path = analysis_dir / "r11_02_support_weighted_observation_report.json"
    r11_03_path = analysis_dir / "r11_03_stock_consistency_gate_report.json"
    r11_05_path = analysis_dir / "r11_05_metric_selector_report.json"
    r11_04_path = analysis_dir / "r11_04_reporting_support_bias_report.json"
    r11_06_path = analysis_dir / "r11_06_local_level_state_filter_report.json"
    r11_07_path = analysis_dir / "r11_07_transition_shrinkage_report.json"
    r11_08_path = analysis_dir / "r11_08_linkage_lag_kernel_report.json"
    r11_09_path = analysis_dir / "r11_09_support_partition_calibration_report.json"
    r11_08_09_path = analysis_dir / "r11_08_09_linkage_support_partition_combo_report.json"
    r11_13_path = analysis_dir / "r11_13_r10_style_readout_teacher_report.json"
    r11_14_path = analysis_dir / "r11_14_back_half_conditional_rates_report.json"
    r11_15_path = analysis_dir / "r11_15_multi_horizon_lifted_readout_report.json"
    r11_16_path = analysis_dir / "r11_16_metric_horizon_shape_head_report.json"
    r11_17_path = analysis_dir / "r11_17_constrained_shape_head_report.json"
    r11_18_path = analysis_dir / "r11_18_horizon_adaptive_constrained_shape_report.json"
    r11_19_path = analysis_dir / "r11_19_datv_transition_process_report.json"
    r11_20_path = analysis_dir / "r11_20_era_datv_transition_process_report.json"
    r11_30_path = analysis_dir / "r11_30_shift_ablation_report.json"
    r11_31_path = analysis_dir / "r11_31_removal_sensitivity_report.json"
    r11_33_path = analysis_dir / "r11_33_external_annual_challenge_report.json"
    r11_34_path = analysis_dir / "r11_34_source_family_ablation_readiness_report.json"
    r11_35_path = analysis_dir / "r11_35_claim_card_report.json"
    write_json(r11_01_path, r11_01)
    write_json(r11_02_path, r11_02)
    write_json(r11_03_path, r11_03)
    write_json(r11_05_path, r11_05)
    write_json(r11_04_path, r11_04)
    write_json(r11_06_path, r11_06)
    write_json(r11_07_path, r11_07)
    write_json(r11_08_path, r11_08)
    write_json(r11_09_path, r11_09)
    write_json(r11_08_09_path, r11_08_09)
    write_json(r11_13_path, r11_13)
    write_json(r11_14_path, r11_14)
    write_json(r11_15_path, r11_15)
    write_json(r11_16_path, r11_16)
    write_json(r11_17_path, r11_17)
    write_json(r11_18_path, r11_18)
    write_json(r11_19_path, r11_19)
    write_json(r11_20_path, r11_20)
    for payload in extra_r11_reports.values():
        write_json(Path(payload["path"]), dict(payload["report"]))
    process_candidate_reports = {
        "R11-17": r11_17,
        "R11-19": r11_19,
        "R11-20": r11_20,
        **{experiment_id: dict(payload["report"]) for experiment_id, payload in extra_r11_reports.items()},
    }
    r11_30 = _build_r11_30_shift_ablation_report(process_candidate_reports)
    r11_31 = _build_r11_31_removal_sensitivity_report(process_candidate_reports)
    r11_33 = _build_r11_33_external_annual_challenge_report(validation_panel)
    r11_34 = _build_r11_34_source_family_ablation_report(ledger)
    write_json(r11_30_path, r11_30)
    write_json(r11_31_path, r11_31)
    write_json(r11_33_path, r11_33)
    write_json(r11_34_path, r11_34)

    validation_panel_passed = (
        int(validation_panel["entry_count"]) > 0
        and int(validation_panel["leakage_violation_count"]) == 0
    )
    validation_panel_blockers: list[str] = []
    if int(validation_panel["entry_count"]) == 0:
        validation_panel_blockers.append("no_annual_validation_entries")
    if int(validation_panel["leakage_violation_count"]) > 0:
        validation_panel_blockers.append("validation_panel_role_leakage")

    infrastructure_rows = [
        _infrastructure_row(
            experiment_id="BM-00",
            title="Freeze benchmark manifest",
            decision="keep_as_contract",
            kept_claim="benchmark_lock",
            artifact_path=benchmark_path,
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="BM-01",
            title="Split ledger",
            decision="keep_as_contract",
            kept_claim="split_lock",
            artifact_path=split_path,
            status="pass" if split_manifest["split_count"] else "fail",
            blockers=[] if split_manifest["split_count"] else ["no_blocked_splits"],
        ),
        _infrastructure_row(
            experiment_id="BM-02",
            title="AEM/Spectrum validation-only panel",
            decision="keep_as_validation_panel" if validation_panel_passed else "blocked_missing_external_validation_panel",
            kept_claim="external_annual_validation_only_panel",
            artifact_path=validation_path,
            status="pass" if validation_panel_passed else "blocked",
            blockers=validation_panel_blockers,
        ),
        _infrastructure_row(
            experiment_id="BM-03",
            title="Score normalizer",
            decision="keep_as_contract",
            kept_claim="score_contract",
            artifact_path=score_path,
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="BM-04",
            title="Horizon-matched R10 replay",
            decision="keep_as_benchmark"
            if str(r10_horizon_replay.get("status") or "") == "pass"
            else "blocked_missing_horizon_matched_r10",
            kept_claim="horizon_matched_r10_reference",
            artifact_path=r10_horizon_replay_path,
            status="pass" if str(r10_horizon_replay.get("status") or "") == "pass" else "blocked",
            blockers=[
                f"h{row.get('horizon_years')}:{','.join(list(row.get('blockers') or []))}"
                for row in list(r10_horizon_replay.get("horizon_rows") or [])
                if isinstance(row, dict) and list(row.get("blockers") or [])
            ],
        ),
        _infrastructure_row(
            experiment_id="BM-05",
            title="Current lineage replay",
            decision="keep_as_evidence_replay",
            kept_claim="frozen_current_lineage_comparison",
            artifact_path=analysis_dir / "current_lineage_replay_report.json",
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="R11-01",
            title="Deterministic nonnegative D/A/T/V reconciliation",
            decision=str(r11_01["decision"]),
            kept_claim="state_constraint_primitive",
            artifact_path=r11_01_path,
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="R11-02",
            title="Support-weighted observation operator contract",
            decision=str(r11_02["decision"]),
            kept_claim="observation_operator_contract",
            artifact_path=r11_02_path,
            status="pass",
        ),
        _infrastructure_row(
            experiment_id="R11-03",
            title="Stock-consistency rejection gate",
            decision=str(r11_03["decision"]),
            kept_claim="claim_gate",
            artifact_path=r11_03_path,
            status="pass" if str(r11_03["status"]) == "pass" else "needs_more_evidence",
            blockers=[] if str(r11_03["status"]) == "pass" else ["not_enough_false_win_rejections"],
        ),
        _infrastructure_row(
            experiment_id="R11-30",
            title="Support/reporting shift ablation",
            decision=str(r11_30["decision"]),
            kept_claim="ablation_evidence",
            artifact_path=r11_30_path,
            status=str(r11_30["status"]),
        ),
        _infrastructure_row(
            experiment_id="R11-31",
            title="Removal prior sensitivity",
            decision=str(r11_31["decision"]),
            kept_claim="sensitivity_evidence",
            artifact_path=r11_31_path,
            status=str(r11_31["status"]),
        ),
        _infrastructure_row(
            experiment_id="R11-33",
            title="External annual validation challenge",
            decision=str(r11_33["decision"]),
            kept_claim="external_annual_validation_only_contract",
            artifact_path=r11_33_path,
            status=str(r11_33["status"]),
            blockers=[] if str(r11_33["status"]) == "available" else ["external_annual_validation_panel_missing"],
        ),
        _infrastructure_row(
            experiment_id="R11-34",
            title="Source-family ablation readiness",
            decision=str(r11_34["decision"]),
            kept_claim="source_family_ablation_readiness",
            artifact_path=r11_34_path,
            status=str(r11_34["status"]),
        ),
    ]
    r11_candidate_rows = [
        _summarize_r11_candidate_branch(
            experiment_id="R11-05",
            title="Train-backtested metric-specific selector",
            path=r11_05_path,
            report=r11_05,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-04",
            title="Reporting/support-bias sparse observation filter",
            path=r11_04_path,
            report=r11_04,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-06",
            title="Bayesian local-level sparse state filter",
            path=r11_06_path,
            report=r11_06,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-07",
            title="Empirical-Bayes transition shrinkage",
            path=r11_07_path,
            report=r11_07,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-08",
            title="D_to_A linkage lag kernel",
            path=r11_08_path,
            report=r11_08,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-09",
            title="Support-partition calibration",
            path=r11_09_path,
            report=r11_09,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-08+09",
            title="Linkage lag plus support-partition calibration",
            path=r11_08_09_path,
            report=r11_08_09,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-13",
            title="R10-style train-only readout teacher",
            path=r11_13_path,
            report=r11_13,
        ),
        _summarize_r11_candidate_branch(
            experiment_id="R11-14",
            title="Back-half conditional VL/suppression rate process",
            path=r11_14_path,
            report=r11_14,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-15",
            title="Multi-horizon lifted readout gate on R11-14",
            path=r11_15_path,
            report=r11_15,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-16",
            title="Metric-by-horizon residual trajectory-shape head",
            path=r11_16_path,
            report=r11_16,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-17",
            title="Constrained residual trajectory-shape head",
            path=r11_17_path,
            report=r11_17,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-18",
            title="Horizon-adaptive constrained trajectory-shape selector",
            path=r11_18_path,
            report=r11_18,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-19",
            title="D/A transition process with ART retention and stock-flow reconciliation",
            path=r11_19_path,
            report=r11_19,
        ),
        _summarize_r11_multi_horizon_branch(
            experiment_id="R11-20",
            title="Era-stratified D/A transition process with reporting and removal terms",
            path=r11_20_path,
            report=r11_20,
        ),
    ]
    r11_candidate_rows.extend(
        [
            _summarize_r11_multi_horizon_branch(
                experiment_id=experiment_id,
                title=str(payload["title"]),
                path=Path(payload["path"]),
                report=dict(payload["report"]),
            )
            for experiment_id, payload in extra_r11_reports.items()
        ]
    )

    pre_claim_rows = infrastructure_rows[:5] + branch_rows + infrastructure_rows[5:] + r11_candidate_rows
    r11_35 = _build_r11_35_claim_card_report(pre_claim_rows)
    write_json(r11_35_path, r11_35)
    comparison_rows = pre_claim_rows + [
        _infrastructure_row(
            experiment_id="R11-35",
            title="Final R11 claim card",
            decision=str(r11_35["decision"]),
            kept_claim=str(r11_35["status"]),
            artifact_path=r11_35_path,
            status=str(r11_35["status"]),
        )
    ]
    current_lineage_replay = {
        "schema_version": "phase3_dynamic.r11_current_lineage_replay.v1",
        "generated_at": _generated_at(),
        "rows": branch_rows,
        "contract": "frozen reports are replayed without refitting and judged under the R11 stock-consistency gate",
    }
    current_lineage_path = analysis_dir / "current_lineage_replay_report.json"
    write_json(current_lineage_path, current_lineage_replay)

    promoted_rows = [
        row
        for row in comparison_rows
        if str(row.get("decision") or "") == "keep_as_full_cascade_candidate"
    ]
    comparison = {
        "schema_version": R11_FIRST_BATCH_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "loop_variant": "evidence-to-model-loop",
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "rows": comparison_rows,
        "summary": {
            "experiment_count": len(comparison_rows),
            "kept_count": int(sum(1 for row in comparison_rows if str(row.get("decision") or "").startswith("keep"))),
            "rejected_count": int(sum(1 for row in comparison_rows if str(row.get("decision") or "").startswith("reject"))),
            "full_cascade_champion": None if not promoted_rows else promoted_rows[0].get("experiment_id"),
            "claim_boundary": (
                "full-cascade challenger promoted against carry-forward, stock consistency, and R10"
                if promoted_rows
                else "kept outputs are contracts, gates, or narrow measurement-readout claims; no full-cascade or AEM/Spectrum superiority claim is promoted"
            ),
        },
        "artifact_paths": {
            "benchmark_manifest": benchmark_path.as_posix(),
            "split_manifest": split_path.as_posix(),
            "aem_spectrum_validation_panel": validation_path.as_posix(),
            "score_contract": score_path.as_posix(),
            "r10_horizon_matched_replay_report": r10_horizon_replay_path.as_posix(),
            "current_lineage_replay_report": current_lineage_path.as_posix(),
            "r11_01_reconciliation_report": r11_01_path.as_posix(),
            "r11_02_support_weighted_observation_report": r11_02_path.as_posix(),
            "r11_03_stock_consistency_gate_report": r11_03_path.as_posix(),
            "r11_05_metric_selector_report": r11_05_path.as_posix(),
            "r11_04_reporting_support_bias_report": r11_04_path.as_posix(),
            "r11_06_local_level_state_filter_report": r11_06_path.as_posix(),
            "r11_07_transition_shrinkage_report": r11_07_path.as_posix(),
            "r11_08_linkage_lag_kernel_report": r11_08_path.as_posix(),
            "r11_09_support_partition_calibration_report": r11_09_path.as_posix(),
            "r11_08_09_linkage_support_partition_combo_report": r11_08_09_path.as_posix(),
            "r11_13_r10_style_readout_teacher_report": r11_13_path.as_posix(),
            "r11_14_back_half_conditional_rates_report": r11_14_path.as_posix(),
            "r11_15_multi_horizon_lifted_readout_report": r11_15_path.as_posix(),
            "r11_16_metric_horizon_shape_head_report": r11_16_path.as_posix(),
            "r11_17_constrained_shape_head_report": r11_17_path.as_posix(),
            "r11_18_horizon_adaptive_constrained_shape_report": r11_18_path.as_posix(),
            "r11_19_datv_transition_process_report": r11_19_path.as_posix(),
            "r11_20_era_datv_transition_process_report": r11_20_path.as_posix(),
            "r11_30_shift_ablation_report": r11_30_path.as_posix(),
            "r11_31_removal_sensitivity_report": r11_31_path.as_posix(),
            "r11_33_external_annual_challenge_report": r11_33_path.as_posix(),
            "r11_34_source_family_ablation_readiness_report": r11_34_path.as_posix(),
            "r11_35_claim_card_report": r11_35_path.as_posix(),
            "comparison_json": (analysis_dir / "r11_experiment_comparison.json").as_posix(),
            "comparison_csv": (analysis_dir / "r11_experiment_comparison.csv").as_posix(),
            "comparison_markdown": (analysis_dir / "r11_experiment_comparison.md").as_posix(),
        },
    }
    for experiment_id, payload in extra_r11_reports.items():
        family = str(payload["family"])
        key = f"{experiment_id.lower().replace('-', '_')}_{family}_report"
        comparison["artifact_paths"][key] = Path(payload["path"]).as_posix()
    comparison_path = analysis_dir / "r11_experiment_comparison.json"
    csv_path = analysis_dir / "r11_experiment_comparison.csv"
    md_path = analysis_dir / "r11_experiment_comparison.md"
    dashboard_path = analysis_dir / "r11_experiment_comparison_dashboard.png"
    back_half_dashboard_path = analysis_dir / "r11_14_back_half_conditional_rate_dashboard.png"
    multi_horizon_dashboard_path = analysis_dir / "r11_15_multi_horizon_lifted_dashboard.png"
    shape_head_dashboard_path = analysis_dir / "r11_16_metric_horizon_shape_head_dashboard.png"
    constrained_shape_dashboard_path = analysis_dir / "r11_17_constrained_shape_head_dashboard.png"
    horizon_adaptive_dashboard_path = analysis_dir / "r11_18_horizon_adaptive_constrained_shape_dashboard.png"
    transition_process_dashboard_path = analysis_dir / "r11_19_datv_transition_process_dashboard.png"
    era_transition_process_dashboard_path = analysis_dir / "r11_20_era_datv_transition_process_dashboard.png"
    comparison["artifact_paths"]["dashboard_png"] = dashboard_path.as_posix()
    comparison["artifact_paths"]["back_half_dashboard_png"] = back_half_dashboard_path.as_posix()
    comparison["artifact_paths"]["multi_horizon_dashboard_png"] = multi_horizon_dashboard_path.as_posix()
    comparison["artifact_paths"]["shape_head_dashboard_png"] = shape_head_dashboard_path.as_posix()
    comparison["artifact_paths"]["constrained_shape_head_dashboard_png"] = constrained_shape_dashboard_path.as_posix()
    comparison["artifact_paths"]["horizon_adaptive_constrained_shape_dashboard_png"] = horizon_adaptive_dashboard_path.as_posix()
    comparison["artifact_paths"]["datv_transition_process_dashboard_png"] = transition_process_dashboard_path.as_posix()
    comparison["artifact_paths"]["era_datv_transition_process_dashboard_png"] = era_transition_process_dashboard_path.as_posix()
    write_json(comparison_path, comparison)
    _write_csv(csv_path, comparison_rows)
    md_path.write_text(_markdown_table(comparison_rows), encoding="utf-8")
    _write_dashboard(dashboard_path, comparison_rows)
    _write_back_half_dashboard(back_half_dashboard_path, r11_13, r11_14)
    _write_multi_horizon_dashboard(multi_horizon_dashboard_path, r11_15)
    _write_shape_head_dashboard(shape_head_dashboard_path, r11_15, r11_16)
    _write_constrained_shape_dashboard(constrained_shape_dashboard_path, r11_15, r11_16, r11_17)
    _write_horizon_adaptive_shape_dashboard(horizon_adaptive_dashboard_path, r11_17, r11_18)
    _write_transition_process_dashboard(transition_process_dashboard_path, r11_17, r11_19)
    _write_era_transition_process_dashboard(era_transition_process_dashboard_path, r11_19, r11_20)
    return comparison
