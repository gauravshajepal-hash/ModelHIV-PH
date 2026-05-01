from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, rolling_origin_splits, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_CHALLENGE_METRICS,
    PRIMARY_STOCK_GUARD_METRICS,
    R10_COMPARABLE_METRICS,
    R11_EVALUATION_METRICS,
    R11_MULTI_HORIZON_YEARS,
    R12_ANNUAL_ANCHOR_LINEAGE_ID,
    R12_PROGRAM_LINEAGE_IDS,
    _artifact_paths,
    _build_r10_horizon_replay_report,
    _build_r12_official_annual_challenge_gate_report,
    _candidate_predictions,
    _carry_forward_prediction,
    _finite_float,
    _generated_at,
    _metric_provenance,
    _project_prediction_row,
    _r10_reference,
    _r10_reference_for_horizon,
    _r12_is_program_row,
    _r12_metric_matches_lineage_ids,
    _read_path_payload,
    _score_predictions,
)
from .runtime import ensure_dir, write_json


R13_PRIORITY_EXPERIMENT_SCHEMA_VERSION = "phase3_dynamic.r13_priority_experiment_queue.v1"
R13_EXPERIMENT_COUNT = 50


def _r13_priority_experiment_specs() -> list[dict[str, Any]]:
    all_metrics = tuple(R11_EVALUATION_METRICS)
    r10_metrics = tuple(R10_COMPARABLE_METRICS)
    da_metrics = ("diagnosed_plhiv", "alive_on_art")
    back_half_metrics = ("alive_on_art", "tested_for_viral_load", "virally_suppressed")
    vl_metrics = ("tested_for_viral_load", "virally_suppressed")
    flow_metric = ("new_diagnosed_cases_period",)
    specs: list[dict[str, Any]] = [
        {
            "experiment_id": "R13-001",
            "priority": 1,
            "layer": "annual_official_measurement",
            "title": "Conserved annual incidence/death/PLHIV gate on R12-09",
            "family": "official_annual_challenge_gate",
            "candidate_family": "r12_stock_cone_safe_annual_trajectory_process",
            "row_scope": "official_annual_q4",
            "metrics": tuple(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "The annual official-style head is publishable only if incidence, deaths, and PLHIV score as a conserved mass-balance triplet.",
        },
        {
            "experiment_id": "R13-002",
            "priority": 2,
            "layer": "monthly_program_state",
            "title": "Two-factor monthly program nowcast on DOH program evidence",
            "family": "r14_two_factor_program_process",
            "row_scope": "program",
            "metrics": r10_metrics,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "If separating support availability from true program-volume shock is useful, it should improve 1y/2y diagnosis, ART, and diagnosis-flow nowcasts on DOH program rows.",
        },
        {
            "experiment_id": "R13-003",
            "priority": 3,
            "layer": "reference_lock",
            "title": "Locked R11-28 full-support research reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "R11-28 remains the minimum viable research reference unless a later branch beats it under carry-forward and matched R10 gates.",
        },
        {
            "experiment_id": "R13-004",
            "priority": 4,
            "layer": "annual_anchor_route",
            "title": "Stock-cone-safe annual-anchor trajectory replay",
            "family": "r12_stock_cone_safe_annual_trajectory_process",
            "row_scope": "annual_anchor",
            "metrics": da_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "The annual-anchor route should remain the cleanest trajectory claim because it avoids mixed monthly/quarterly support conflict.",
        },
        {
            "experiment_id": "R13-005",
            "priority": 5,
            "layer": "support_adequacy",
            "title": "DOH quarterly short-horizon adequacy under R11-28",
            "family": "multi_horizon_weighted_process",
            "row_scope": "doh_quarterly",
            "metrics": r10_metrics,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "DOH quarterly rows should be claim-grade only if the locked reference beats carry-forward and does not fall far behind R10 at short horizons.",
        },
        {
            "experiment_id": "R13-006",
            "priority": 6,
            "layer": "program_route",
            "title": "R17 ART-flow teacher program route stress test",
            "family": "r17_art_flow_teacher_process",
            "row_scope": "program",
            "metrics": da_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "If the remaining R10 gap is ART trajectory plus diagnosis-flow shape rather than diagnosed-stock dynamics, a train-origin frozen-R10 ART/flow teacher on top of R16 should improve 3y/5y program trajectory without breaking the stock cone.",
        },
        {
            "experiment_id": "R13-007",
            "priority": 7,
            "layer": "transition_process",
            "title": "D/A process-split transition on full support",
            "family": "r12_da_process_split_transition",
            "row_scope": "all",
            "metrics": da_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Separating diagnosed reporting/removal from ART linkage/retention should reduce diagnosed-stock and ART-stock drift if dynamics, not lineage, is the blocker.",
        },
        {
            "experiment_id": "R13-008",
            "priority": 8,
            "layer": "lineage_residual",
            "title": "D/A residual-source process replay",
            "family": "r12_da_residual_source_process",
            "row_scope": "all",
            "metrics": da_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Source-family residual structure should not be promoted unless it generalizes without target leakage.",
        },
        {
            "experiment_id": "R13-009",
            "priority": 9,
            "layer": "transition_process",
            "title": "Era-stratified D/A transition baseline",
            "family": "era_datv_transition_process",
            "row_scope": "all",
            "metrics": da_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Era-stratified removals and reporting shifts are the strongest existing mechanistic D/A baseline.",
        },
        {
            "experiment_id": "R13-010",
            "priority": 10,
            "layer": "transition_process",
            "title": "Non-era D/A transition baseline",
            "family": "datv_transition_process",
            "row_scope": "all",
            "metrics": da_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "If non-era dynamics match era-stratified dynamics, the extra reporting-shift structure is not scientifically carrying the result.",
        },
        {
            "experiment_id": "R13-011",
            "priority": 11,
            "layer": "diagnosis_flow",
            "title": "Diagnosis-flow lagged stock process",
            "family": "diagnosis_lag_stock_process",
            "row_scope": "all",
            "metrics": r10_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Lagged diagnosis flow should improve D/A trajectory shape if same-quarter linkage is causing stock mismatch.",
        },
        {
            "experiment_id": "R13-012",
            "priority": 12,
            "layer": "diagnosis_flow",
            "title": "Stock-flow reconciliation process",
            "family": "stock_flow_reconciliation_process",
            "row_scope": "all",
            "metrics": r10_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Diagnosis-flow repair is useful only if it reconciles diagnosed stock without worsening ART stock.",
        },
        {
            "experiment_id": "R13-013",
            "priority": 13,
            "layer": "diagnosis_flow",
            "title": "Support-era diagnosis-flow reporting process",
            "family": "support_era_diagnosis_flow_process",
            "row_scope": "all",
            "metrics": r10_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Support-era diagnosis-flow shifts are allowed only if they improve blocked flow/D/A alignment beyond carry-forward and R10.",
        },
        {
            "experiment_id": "R13-014",
            "priority": 14,
            "layer": "diagnosed_observation",
            "title": "Diagnosed-stock reporting-bias process",
            "family": "diagnosed_reporting_bias_process",
            "row_scope": "all",
            "metrics": da_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "If diagnosed-stock residuals are observation-lineage artifacts, a D-only reporting operator should help without changing biological transitions.",
        },
        {
            "experiment_id": "R13-015",
            "priority": 15,
            "layer": "art_process",
            "title": "ART-specific horizon selector",
            "family": "art_horizon_selector_process",
            "row_scope": "all",
            "metrics": ("alive_on_art",),
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "The ART trajectory may need a different process family from diagnosed stock or diagnosis flow.",
        },
        {
            "experiment_id": "R13-016",
            "priority": 16,
            "layer": "back_half_rates",
            "title": "Conditional VL/suppression horizon selector",
            "family": "conditional_rate_horizon_selector",
            "row_scope": "all",
            "metrics": vl_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": False,
            "hypothesis": "Third-95 claims should be separated from count-level D/A failures and scored as conditional VL/suppression behavior.",
        },
        {
            "experiment_id": "R13-017",
            "priority": 17,
            "layer": "back_half_rates",
            "title": "Back-half conditional-rate replay",
            "family": "back_half_conditional_rates",
            "row_scope": "all",
            "metrics": back_half_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": False,
            "hypothesis": "VL and suppression should be modeled through conditional rates, not endpoint count corrections.",
        },
        {
            "experiment_id": "R13-018",
            "priority": 18,
            "layer": "readout_teacher",
            "title": "R10-style train-origin readout teacher",
            "family": "r10_style_readout_teacher",
            "row_scope": "all",
            "metrics": r10_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "The R10-like readout teacher is a benchmark, not a mechanistic champion; it must be measured against true matched R10.",
        },
        {
            "experiment_id": "R13-019",
            "priority": 19,
            "layer": "observation_filter",
            "title": "Local-level state filter baseline",
            "family": "local_level_filter",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "A local-level filter must beat carry-forward internally before any smoother state-space variant is worth adding.",
        },
        {
            "experiment_id": "R13-020",
            "priority": 20,
            "layer": "observation_filter",
            "title": "Support reporting-bias filter baseline",
            "family": "support_reporting_bias",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "Support/reporting bias should be an observation operator, not a biological hazard.",
        },
        {
            "experiment_id": "R13-021",
            "priority": 21,
            "layer": "observation_filter",
            "title": "Support-partition calibration baseline",
            "family": "support_partition_calibration",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "Support partition alone should not be trusted if it cannot beat carry-forward on held-out years.",
        },
        {
            "experiment_id": "R13-022",
            "priority": 22,
            "layer": "transition_shrinkage",
            "title": "Empirical-Bayes transition shrinkage baseline",
            "family": "transition_shrinkage",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "Shrinkage should stabilize sparse transitions but must not become a smoothed carry-forward clone.",
        },
        {
            "experiment_id": "R13-023",
            "priority": 23,
            "layer": "linkage_delay",
            "title": "ART linkage lag kernel baseline",
            "family": "linkage_lag_kernel",
            "row_scope": "all",
            "metrics": ("alive_on_art",),
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "A pure linkage lag should improve ART stock if ART initiation delay is the dominant failure mode.",
        },
        {
            "experiment_id": "R13-024",
            "priority": 24,
            "layer": "linkage_delay",
            "title": "Linkage lag plus support partition",
            "family": "linkage_lag_plus_support_partition",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": (1, 2),
            "r10_required": False,
            "hypothesis": "Combining delay and support calibration should improve short horizons without damaging the stock cone.",
        },
        {
            "experiment_id": "R13-025",
            "priority": 25,
            "layer": "annual_anchor_route",
            "title": "R12 long-horizon stock shape on annual anchors",
            "family": "r12_long_horizon_stock_shape_process",
            "row_scope": "annual_anchor",
            "metrics": da_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "Annual-anchor stock shape should be compared against R12-09 to see whether the stock-cone-safe head adds value.",
        },
        {
            "experiment_id": "R13-026",
            "priority": 26,
            "layer": "program_route",
            "title": "Program D/A nowcast with two-factor monthly state",
            "family": "r14_two_factor_program_process",
            "row_scope": "program",
            "metrics": da_metrics,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "The monthly latent state should first win where monthly support exists: D/A program rows at short horizons.",
        },
        {
            "experiment_id": "R13-027",
            "priority": 27,
            "layer": "doh_quarterly",
            "title": "DOH quarterly D/A short-horizon reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "doh_quarterly",
            "metrics": da_metrics,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "DOH quarterly D/A rows are the most direct test of program state nowcasting.",
        },
        {
            "experiment_id": "R13-028",
            "priority": 28,
            "layer": "doh_monthly",
            "title": "DOH monthly D/A short-horizon reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "doh_monthly",
            "metrics": da_metrics,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "Monthly support should help nowcasts only if the monthly rows are stable enough to generalize.",
        },
        {
            "experiment_id": "R13-029",
            "priority": 29,
            "layer": "quarterly_anchors",
            "title": "Q4-only long-horizon reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "q4_only",
            "metrics": da_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "Q4-only scoring separates annual reporting cadence from mixed quarterly artifacts.",
        },
        {
            "experiment_id": "R13-030",
            "priority": 30,
            "layer": "shock_period",
            "title": "Post-2021 R14 program nowcast stress test",
            "family": "r14_two_factor_program_process",
            "row_scope": "post_2021_program",
            "metrics": r10_metrics,
            "horizons": (1,),
            "r10_required": True,
            "hypothesis": "The reporting process must handle the 2022-2025 rebound era without using COVID-specific hand labels.",
        },
        {
            "experiment_id": "R13-031",
            "priority": 31,
            "layer": "shock_period",
            "title": "Post-2021 full-support reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "post_2021",
            "metrics": r10_metrics,
            "horizons": (1, 3),
            "r10_required": True,
            "hypothesis": "If failures concentrate post-2021, the model is missing reporting/rebound dynamics rather than early-cascade structure.",
        },
        {
            "experiment_id": "R13-032",
            "priority": 32,
            "layer": "shock_period",
            "title": "Pre-2020 stability reference",
            "family": "multi_horizon_weighted_process",
            "row_scope": "pre_2020",
            "metrics": r10_metrics,
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "Pre-2020 performance establishes whether the model fails only during reporting disruption/rebound periods.",
        },
        {
            "experiment_id": "R13-033",
            "priority": 33,
            "layer": "support_partition",
            "title": "Common-support only R11-28",
            "family": "multi_horizon_weighted_process",
            "row_scope": "common_support",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "If common-support-only scoring improves sharply, mixed evidence lineage is a key blocker.",
        },
        {
            "experiment_id": "R13-034",
            "priority": 34,
            "layer": "support_partition",
            "title": "Exact-observed only R11-28",
            "family": "multi_horizon_weighted_process",
            "row_scope": "exact_observed",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Exact-observed rows test whether model-estimated support is contaminating trajectory claims.",
        },
        {
            "experiment_id": "R13-035",
            "priority": 35,
            "layer": "non_program",
            "title": "Non-program annual trajectory route",
            "family": "r12_stock_cone_safe_annual_trajectory_process",
            "row_scope": "non_program",
            "metrics": da_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "If non-program rows pass while program rows fail, the blocker is monthly/quarterly program observation dynamics.",
        },
        {
            "experiment_id": "R13-036",
            "priority": 36,
            "layer": "route_selector",
            "title": "Route-aware two-head full replay",
            "family": "r12_route_aware_two_head_process",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Head separation should beat a single family only if evidence routes are genuinely different processes.",
        },
        {
            "experiment_id": "R13-037",
            "priority": 37,
            "layer": "shape_head",
            "title": "Constrained trajectory-shape replay",
            "family": "constrained_trajectory_shape_head",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Constrained readout correction should be blocked if it wins by breaking stock or rate semantics.",
        },
        {
            "experiment_id": "R13-038",
            "priority": 38,
            "layer": "shape_head",
            "title": "Horizon-adaptive constrained shape replay",
            "family": "horizon_adaptive_constrained_shape_head",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Horizon-specific correction policies must be judged against exact horizon-matched R10, not a scalar R10 reference.",
        },
        {
            "experiment_id": "R13-039",
            "priority": 39,
            "layer": "shape_head",
            "title": "Unconstrained trajectory-shape placebo",
            "family": "trajectory_shape_head",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "Unconstrained shape gains are diagnostic only if they expose what constraints are blocking, not if they become a champion.",
        },
        {
            "experiment_id": "R13-040",
            "priority": 40,
            "layer": "art_process",
            "title": "ART-only horizon selector",
            "family": "art_horizon_selector_process",
            "row_scope": "program",
            "metrics": ("alive_on_art",),
            "horizons": (1, 2, 3),
            "r10_required": True,
            "hypothesis": "ART process repair should be judged on ART rows first before being allowed to move the whole cascade.",
        },
        {
            "experiment_id": "R13-041",
            "priority": 41,
            "layer": "diagnosis_flow",
            "title": "Flow-only support-era repair",
            "family": "support_era_diagnosis_flow_process",
            "row_scope": "program",
            "metrics": flow_metric,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "A flow-only repair is allowed only if it improves diagnosis flow without hiding stock consequences.",
        },
        {
            "experiment_id": "R13-042",
            "priority": 42,
            "layer": "diagnosis_flow",
            "title": "Flow-only stock-flow reconciliation",
            "family": "stock_flow_reconciliation_process",
            "row_scope": "program",
            "metrics": flow_metric,
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "Stock-flow reconciliation should be tested on flow alone and then separately against D/A stock guards.",
        },
        {
            "experiment_id": "R13-043",
            "priority": 43,
            "layer": "diagnosed_observation",
            "title": "Diagnosed-stock only reporting-bias repair",
            "family": "diagnosed_reporting_bias_process",
            "row_scope": "program",
            "metrics": ("diagnosed_plhiv",),
            "horizons": (1, 2, 3),
            "r10_required": True,
            "hypothesis": "D-only reporting correction is safer than pushing source residuals into the full transition process.",
        },
        {
            "experiment_id": "R13-044",
            "priority": 44,
            "layer": "linkage_delay",
            "title": "ART-only linkage lag repair",
            "family": "linkage_lag_kernel",
            "row_scope": "program",
            "metrics": ("alive_on_art",),
            "horizons": (1, 2),
            "r10_required": True,
            "hypothesis": "If ART lag alone fails, the ART gap is not just delayed linkage.",
        },
        {
            "experiment_id": "R13-045",
            "priority": 45,
            "layer": "back_half_rates",
            "title": "VL-tested count conditional-rate replay",
            "family": "back_half_conditional_rates",
            "row_scope": "all",
            "metrics": ("tested_for_viral_load",),
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": False,
            "hypothesis": "VL testing should be interpreted as service coverage conditional on ART, not a free count endpoint.",
        },
        {
            "experiment_id": "R13-046",
            "priority": 46,
            "layer": "back_half_rates",
            "title": "Suppression count conditional-rate replay",
            "family": "back_half_conditional_rates",
            "row_scope": "all",
            "metrics": ("virally_suppressed",),
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": False,
            "hypothesis": "Suppression claims must remain conditional on VL testing evidence until third-95 support is stronger.",
        },
        {
            "experiment_id": "R13-047",
            "priority": 47,
            "layer": "annual_official_measurement",
            "title": "Conserved annual gate on R14 program branch",
            "family": "official_annual_challenge_gate",
            "candidate_family": "r14_two_factor_program_process",
            "row_scope": "official_annual_q4",
            "metrics": tuple(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "The monthly program branch must not degrade official annual incidence/death/PLHIV validation.",
        },
        {
            "experiment_id": "R13-048",
            "priority": 48,
            "layer": "annual_official_measurement",
            "title": "Conserved annual gate on R11-28 reference",
            "family": "official_annual_challenge_gate",
            "candidate_family": "multi_horizon_weighted_process",
            "row_scope": "official_annual_q4",
            "metrics": tuple(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "The annual official-style validation head should be benchmarked against the locked R11-28 reference.",
        },
        {
            "experiment_id": "R13-049",
            "priority": 49,
            "layer": "phase2_lockbox",
            "title": "Phase2 determinant lockbox diagnostic",
            "family": "multi_horizon_weighted_process",
            "row_scope": "all",
            "metrics": r10_metrics,
            "horizons": (1,),
            "r10_required": True,
            "phase2_status": "locked_until_source_stable",
            "hypothesis": "No Phase2 determinant bundle should enter hazards until source-family and time-window falsification pass.",
        },
        {
            "experiment_id": "R13-050",
            "priority": 50,
            "layer": "publication_gate",
            "title": "Full publication gate sentinel",
            "family": "r17_art_flow_teacher_process",
            "row_scope": "all",
            "metrics": all_metrics,
            "horizons": R11_MULTI_HORIZON_YEARS,
            "r10_required": True,
            "hypothesis": "The current publication blocker is removed only when the R17 hybrid ART/flow teacher cascade beats carry-forward and matched R10 without weakening stock or rate gates.",
        },
    ]
    return sorted(specs, key=lambda item: int(item["priority"]))


def _r13_metric_has_scope(row: dict[str, Any], metric_name: str, scope: str) -> bool:
    provenance = _metric_provenance(row, metric_name)
    if scope == "common_support":
        return str(provenance.get("support_partition") or "") == "common_support"
    if scope == "exact_observed":
        semantics = str(provenance.get("measurement_semantics") or "")
        role = str(provenance.get("observation_role") or "")
        tier = str(provenance.get("source_tier") or provenance.get("source_quality_tier") or "")
        return semantics in {"stock_anchor", "flow_count"} or role == "direct_target" or "official" in tier
    if scope == "doh_quarterly":
        return _r12_metric_matches_lineage_ids(row, metric_name, ("doh_quarterly",))
    if scope == "doh_monthly":
        return _r12_metric_matches_lineage_ids(row, metric_name, ("doh_monthly",))
    if scope == "annual_anchor":
        return _r12_metric_matches_lineage_ids(row, metric_name, (R12_ANNUAL_ANCHOR_LINEAGE_ID,))
    return False


def _r13_filter_rows(rows: list[dict[str, Any]], *, scope: str, metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        year = quarter_year(quarter)
        if scope == "all":
            include = True
        elif scope == "program":
            include = _r12_is_program_row(row)
        elif scope == "non_program":
            include = not _r12_is_program_row(row)
        elif scope == "q4_only" or scope == "official_annual_q4":
            include = quarter.endswith("-Q4")
        elif scope == "post_2021":
            include = int(year) >= 2022
        elif scope == "pre_2020":
            include = int(year) <= 2019
        elif scope == "post_2021_program":
            include = int(year) >= 2022 and _r12_is_program_row(row)
        elif scope in {"common_support", "exact_observed", "doh_quarterly", "doh_monthly", "annual_anchor"}:
            include = any(_r13_metric_has_scope(row, metric_name, scope) for metric_name in metrics)
        else:
            include = False
        if include:
            output.append(dict(row))
    return output


def _r13_score_cached_predictions(
    *,
    rows: list[dict[str, Any]],
    family: str,
    split: dict[str, Any],
    prediction_cache: dict[tuple[str, str, int, tuple[int, ...]], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    scope_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_end_year = int(split.get("train_end_year") or 0)
    holdout_years = tuple(int(year) for year in list(split.get("holdout_years") or []))
    key = (family, scope_key, train_end_year, holdout_years)
    if key in prediction_cache:
        return prediction_cache[key]
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
        prediction_cache[key] = (train_rows, holdout_rows, [], [])
        return prediction_cache[key]
    candidate_rows, _summary = _candidate_predictions(train_rows, holdout_rows, family=family)
    carry_rows = _carry_forward_prediction(train_rows, holdout_rows)
    prediction_cache[key] = (train_rows, holdout_rows, candidate_rows, carry_rows)
    return prediction_cache[key]


def _r13_stock_cone_violation_count(prediction_rows: list[dict[str, Any]]) -> int:
    violations = 0
    for row in prediction_rows:
        projected = _project_prediction_row(dict(row))
        for metric_name in ("diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"):
            original = _finite_float(row.get(metric_name))
            repaired = _finite_float(projected.get(metric_name))
            if original is not None and repaired is not None and abs(float(original) - float(repaired)) > 1e-9:
                violations += 1
                break
    return violations


def _r13_model_spec_report(
    *,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    r10_horizon_replay: dict[str, Any],
    prediction_cache: dict[tuple[str, str, int, tuple[int, ...]], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, Any]:
    metrics = tuple(str(item) for item in tuple(spec.get("metrics") or R11_EVALUATION_METRICS))
    horizons = tuple(int(item) for item in tuple(spec.get("horizons") or (1,)))
    scope = str(spec.get("row_scope") or "all")
    family = str(spec.get("family") or "")
    scoped_rows = _r13_filter_rows(rows, scope=scope, metrics=metrics)
    horizon_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    stock_cone_violations = 0
    if not scoped_rows:
        blockers.append("no_rows_for_scope")
    for horizon in horizons:
        splits = rolling_origin_splits(
            scoped_rows,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizon_years=int(horizon),
        )
        candidate_values: list[float] = []
        carry_values: list[float] = []
        r10_candidate_values: list[float] = []
        split_count = 0
        for split in splits:
            train_rows, holdout_rows, candidate_rows, carry_rows = _r13_score_cached_predictions(
                rows=scoped_rows,
                family=family,
                split=split,
                prediction_cache=prediction_cache,
                scope_key=scope,
            )
            if not candidate_rows or not carry_rows:
                continue
            split_count += 1
            stock_cone_violations += _r13_stock_cone_violation_count(candidate_rows)
            candidate_score = _score_predictions(
                train_rows=train_rows,
                holdout_rows=holdout_rows,
                prediction_rows=candidate_rows,
                metrics=metrics,
            )
            carry_score = _score_predictions(
                train_rows=train_rows,
                holdout_rows=holdout_rows,
                prediction_rows=carry_rows,
                metrics=metrics,
            )
            candidate_mean = _finite_float(candidate_score.get("mean_mae"))
            carry_mean = _finite_float(carry_score.get("mean_mae"))
            if candidate_mean is not None:
                candidate_values.append(float(candidate_mean))
            if carry_mean is not None:
                carry_values.append(float(carry_mean))
            for candidate_metric, carry_metric in zip(
                list(candidate_score.get("metric_rows") or []),
                list(carry_score.get("metric_rows") or []),
            ):
                if not isinstance(candidate_metric, dict) or not isinstance(carry_metric, dict):
                    continue
                metric_name = str(candidate_metric.get("metric_name") or "")
                metric_rows.append(
                    {
                        "horizon_years": int(horizon),
                        "train_end_year": int(split.get("train_end_year") or 0),
                        "metric_name": metric_name,
                        "candidate_mean_norm_error": candidate_metric.get("mean_norm_error"),
                        "carry_forward_mean_norm_error": carry_metric.get("mean_norm_error"),
                        "entry_count": candidate_metric.get("entry_count"),
                    }
                )
            r10_metrics = tuple(metric_name for metric_name in metrics if metric_name in set(R10_COMPARABLE_METRICS))
            if r10_metrics:
                r10_score = _score_predictions(
                    train_rows=train_rows,
                    holdout_rows=holdout_rows,
                    prediction_rows=candidate_rows,
                    metrics=r10_metrics,
                )
                r10_candidate_mean = _finite_float(r10_score.get("mean_mae"))
                if r10_candidate_mean is not None:
                    r10_candidate_values.append(float(r10_candidate_mean))
        r10_reference = _r10_reference_for_horizon(r10_horizon_replay, int(horizon))
        r10_reference_mae = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))
        candidate_mean = None if not candidate_values else float(np.mean(np.asarray(candidate_values, dtype=np.float64)))
        carry_mean = None if not carry_values else float(np.mean(np.asarray(carry_values, dtype=np.float64)))
        r10_candidate_mean = None if not r10_candidate_values else float(np.mean(np.asarray(r10_candidate_values, dtype=np.float64)))
        horizon_rows.append(
            {
                "horizon_years": int(horizon),
                "split_count": split_count,
                "candidate_mean_mae": candidate_mean,
                "carry_forward_mean_mae": carry_mean,
                "candidate_minus_carry_forward_mean_mae": None
                if candidate_mean is None or carry_mean is None
                else float(candidate_mean - carry_mean),
                "r10_comparable_candidate_mean_mae": r10_candidate_mean,
                "r10_horizon_reference_mae": r10_reference_mae,
                "candidate_minus_r10_reference_mae": None
                if r10_candidate_mean is None or r10_reference_mae is None
                else float(r10_candidate_mean - r10_reference_mae),
            }
        )
        if split_count == 0:
            blockers.append(f"h{int(horizon)}_no_blocked_splits")
    candidate_summary = [
        float(row["candidate_mean_mae"])
        for row in horizon_rows
        if _finite_float(row.get("candidate_mean_mae")) is not None
    ]
    carry_summary = [
        float(row["carry_forward_mean_mae"])
        for row in horizon_rows
        if _finite_float(row.get("carry_forward_mean_mae")) is not None
    ]
    r10_candidate_summary = [
        float(row["r10_comparable_candidate_mean_mae"])
        for row in horizon_rows
        if _finite_float(row.get("r10_comparable_candidate_mean_mae")) is not None
    ]
    r10_reference_summary = [
        float(row["r10_horizon_reference_mae"])
        for row in horizon_rows
        if _finite_float(row.get("r10_horizon_reference_mae")) is not None
    ]
    carry_gate = bool(candidate_summary and carry_summary and all(
        _finite_float(row.get("candidate_minus_carry_forward_mean_mae")) is not None
        and float(row["candidate_minus_carry_forward_mean_mae"]) < 0.0
        for row in horizon_rows
        if _finite_float(row.get("candidate_mean_mae")) is not None
    ))
    r10_required = bool(spec.get("r10_required"))
    r10_gate = True
    if r10_required:
        r10_gate = bool(r10_candidate_summary and r10_reference_summary and all(
            _finite_float(row.get("candidate_minus_r10_reference_mae")) is not None
            and float(row["candidate_minus_r10_reference_mae"]) < 0.0
            for row in horizon_rows
            if _finite_float(row.get("r10_comparable_candidate_mean_mae")) is not None
        ))
        if not r10_gate:
            blockers.append("matched_r10_gate_failed")
    if stock_cone_violations:
        blockers.append("stock_cone_projection_would_change_predictions")
    if str(spec.get("phase2_status") or "") == "locked_until_source_stable":
        blockers.append("phase2_locked_until_source_family_and_time_window_falsification_pass")
    if carry_gate and r10_gate and not blockers:
        decision = "promote_for_next_wave"
        kept_claim = "candidate_passes_priority_gate"
    elif str(spec.get("phase2_status") or "") == "locked_until_source_stable":
        decision = "keep_as_diagnostic"
        kept_claim = "phase2_determinant_lockbox_only"
    elif carry_gate:
        decision = "keep_as_diagnostic"
        kept_claim = "beats_carry_forward_but_not_all_promotion_gates"
    elif candidate_summary:
        decision = "reject"
        kept_claim = "does_not_beat_carry_forward"
    else:
        decision = "blocked"
        kept_claim = "not_evaluable"
    return {
        **{key: spec.get(key) for key in ("experiment_id", "priority", "layer", "title", "family", "row_scope", "hypothesis")},
        "candidate_family": spec.get("candidate_family"),
        "metrics": list(metrics),
        "horizons": list(horizons),
        "row_count": len(scoped_rows),
        "horizon_rows": horizon_rows,
        "metric_rows": metric_rows,
        "candidate_mean_mae": None if not candidate_summary else float(np.mean(np.asarray(candidate_summary, dtype=np.float64))),
        "carry_forward_mean_mae": None if not carry_summary else float(np.mean(np.asarray(carry_summary, dtype=np.float64))),
        "r10_comparable_candidate_mean_mae": None
        if not r10_candidate_summary
        else float(np.mean(np.asarray(r10_candidate_summary, dtype=np.float64))),
        "r10_reference_mae": None if not r10_reference_summary else float(np.mean(np.asarray(r10_reference_summary, dtype=np.float64))),
        "carry_gate": "pass" if carry_gate else "fail",
        "r10_gate": "pass" if r10_gate else "fail",
        "stock_cone_violation_count": int(stock_cone_violations),
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": sorted(set(blockers)),
    }


def _r13_annual_spec_report(
    *,
    spec: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    annual_cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    candidate_family = str(spec.get("candidate_family") or "multi_horizon_weighted_process")
    key = (candidate_family,)
    if key not in annual_cache:
        annual_cache[key] = _build_r12_official_annual_challenge_gate_report(
            rows=validation_rows,
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            candidate_families=(candidate_family,),
        )
    report = dict(annual_cache[key])
    family_rows = [dict(row) for row in list(report.get("family_rows") or []) if str(row.get("candidate_family") or "") == candidate_family]
    candidate_values = [
        float(row["candidate_mean_norm_error"])
        for row in family_rows
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    carry_values = [
        float(row["carry_forward_mean_norm_error"])
        for row in family_rows
        if _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    head_rows = [
        dict(row)
        for row in list(report.get("annual_measurement_head_rows") or [])
        if str(row.get("candidate_family") or "") == candidate_family
    ]
    conservation_residuals = [
        abs(float(row["conservation_residual"]))
        for row in head_rows
        if str(row.get("metric_name") or "") == "estimated_plhiv"
        and _finite_float(row.get("conservation_residual")) is not None
    ]
    blockers = list(report.get("blockers") or [])
    if not candidate_values:
        blockers.append("no_annual_family_score")
    if conservation_residuals and max(conservation_residuals) > 1e-6:
        blockers.append("annual_mass_balance_residual_nonzero")
    carry_gate = bool(candidate_values and carry_values and float(np.mean(candidate_values)) < float(np.mean(carry_values)))
    if carry_gate and not blockers:
        decision = "promote_for_next_wave"
        kept_claim = "conserved_annual_gate_passes"
    elif carry_gate:
        decision = "keep_as_diagnostic"
        kept_claim = "annual_gate_beats_carry_but_has_blockers"
    else:
        decision = "reject"
        kept_claim = "annual_gate_does_not_beat_carry"
    return {
        **{key_name: spec.get(key_name) for key_name in ("experiment_id", "priority", "layer", "title", "family", "row_scope", "hypothesis")},
        "candidate_family": candidate_family,
        "metrics": list(tuple(spec.get("metrics") or OFFICIAL_ANNUAL_CHALLENGE_METRICS)),
        "horizons": list(tuple(spec.get("horizons") or (1,))),
        "row_count": int(report.get("score_record_count") or 0),
        "horizon_rows": [],
        "metric_rows": [],
        "candidate_mean_mae": None if not candidate_values else float(np.mean(np.asarray(candidate_values, dtype=np.float64))),
        "carry_forward_mean_mae": None if not carry_values else float(np.mean(np.asarray(carry_values, dtype=np.float64))),
        "r10_comparable_candidate_mean_mae": None,
        "r10_reference_mae": None,
        "carry_gate": "pass" if carry_gate else "fail",
        "r10_gate": "not_applicable",
        "stock_cone_violation_count": 0,
        "annual_status": str(report.get("status") or "not_evaluable"),
        "annual_max_conservation_residual": None if not conservation_residuals else float(max(conservation_residuals)),
        "decision": decision,
        "kept_claim": kept_claim,
        "blockers": sorted(set(str(item) for item in blockers)),
    }


def _r13_flat_result_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = _finite_float(row.get("candidate_mean_mae"))
    carry = _finite_float(row.get("carry_forward_mean_mae"))
    r10_candidate = _finite_float(row.get("r10_comparable_candidate_mean_mae"))
    r10_reference = _finite_float(row.get("r10_reference_mae"))
    return {
        "priority": int(row.get("priority") or 0),
        "experiment_id": str(row.get("experiment_id") or ""),
        "title": str(row.get("title") or ""),
        "layer": str(row.get("layer") or ""),
        "family": str(row.get("family") or ""),
        "candidate_family": "" if row.get("candidate_family") is None else str(row.get("candidate_family")),
        "row_scope": str(row.get("row_scope") or ""),
        "candidate_mean_mae": candidate,
        "carry_forward_mean_mae": carry,
        "candidate_minus_carry_forward": None if candidate is None or carry is None else float(candidate - carry),
        "r10_comparable_candidate_mean_mae": r10_candidate,
        "r10_reference_mae": r10_reference,
        "candidate_minus_r10": None if r10_candidate is None or r10_reference is None else float(r10_candidate - r10_reference),
        "carry_gate": str(row.get("carry_gate") or ""),
        "r10_gate": str(row.get("r10_gate") or ""),
        "decision": str(row.get("decision") or ""),
        "kept_claim": str(row.get("kept_claim") or ""),
        "blocker_count": len(list(row.get("blockers") or [])),
    }


def _write_r13_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_r13_flat_result_row(row) for row in rows]
    if not flat:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def _write_r13_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_r13_flat_result_row(row) for row in rows]
    columns = [
        "priority",
        "experiment_id",
        "layer",
        "row_scope",
        "family",
        "candidate_minus_carry_forward",
        "candidate_minus_r10",
        "decision",
    ]
    lines = ["# R13 Priority Experiment Queue", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in flat:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_r13_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    flat = [_r13_flat_result_row(row) for row in rows]
    scored = [row for row in flat if _finite_float(row.get("candidate_minus_carry_forward")) is not None]
    top = scored[: min(20, len(scored))]
    labels = [str(row["experiment_id"]) for row in top]
    carry_delta = [float(row["candidate_minus_carry_forward"]) for row in top]
    r10_delta = [
        np.nan if _finite_float(row.get("candidate_minus_r10")) is None else float(row["candidate_minus_r10"])
        for row in top
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.4), constrained_layout=True)
    x = np.arange(len(labels))
    colors = ["#2f6b4f" if value < 0 else "#9a3f3f" for value in carry_delta]
    axes[0].bar(x, carry_delta, color=colors)
    axes[0].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[0].set_title("Top-priority experiments: candidate minus carry-forward")
    axes[0].set_ylabel("Mean normalized error delta")
    r10_colors = ["#2f6b4f" if np.isfinite(value) and value < 0 else "#9a3f3f" for value in r10_delta]
    axes[1].bar(x, r10_delta, color=r10_colors)
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_title("Top-priority experiments: candidate minus matched R10")
    axes[1].set_ylabel("Mean normalized error delta")
    fig.suptitle("R13 prioritized HIV model experiment queue", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _r13_research_council_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in results if str(row.get("decision") or "") == "promote_for_next_wave"]
    diagnostic = [row for row in results if str(row.get("decision") or "") == "keep_as_diagnostic"]
    rejected = [row for row in results if str(row.get("decision") or "") == "reject"]
    blocked = [row for row in results if str(row.get("decision") or "") == "blocked"]
    blockers = defaultdict(int)
    for row in results:
        for blocker in list(row.get("blockers") or []):
            blockers[str(blocker)] += 1
    return {
        "Evidence Agent": (
            "The active evidence universe is still sufficient for annual conserved-head validation and route-specific "
            "program/annual-anchor diagnostics, but not for unconstrained province or determinant-causal claims."
        ),
        "Validity Skeptic": (
            "Promotion must stay blocked whenever a branch beats carry-forward only by failing matched R10, stock-cone, "
            "or validation-only evidence rules."
        ),
        "Representation / Modeling Agent": (
            "The experiment queue separates observation operators, D/A transition dynamics, annual mass balance, "
            "monthly reporting states, back-half conditional rates, and Phase2 determinant lockbox tests."
        ),
        "Evaluation / Failure Agent": (
            "Each executable experiment is scored by blocked time, support scope, metric scope, carry-forward delta, "
            "and matched R10 delta where available."
        ),
        "Execution Planner": (
            "Run experiments in priority order; promote only next-wave candidates, keep diagnostics for failure anatomy, "
            "and do not add Phase2 determinants until the lockbox tests pass."
        ),
        "experiment_handoff": {
            "promoted_count": len(promoted),
            "diagnostic_count": len(diagnostic),
            "rejected_count": len(rejected),
            "blocked_count": len(blocked),
            "top_promoted_experiment_ids": [str(row.get("experiment_id") or "") for row in promoted[:10]],
            "top_diagnostic_experiment_ids": [str(row.get("experiment_id") or "") for row in diagnostic[:10]],
            "common_blockers": [
                {"blocker": blocker, "count": int(count)}
                for blocker, count in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
    }


def run_r13_priority_experiment_queue(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    epigraph_root: Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    max_experiments: int | None = None,
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
    specs = _r13_priority_experiment_specs()
    if max_experiments is not None:
        specs = specs[: max(int(max_experiments), 0)]
    prediction_cache: dict[tuple[str, str, int, tuple[int, ...]], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    annual_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for spec in specs:
        if str(spec.get("family") or "") == "official_annual_challenge_gate":
            result = _r13_annual_spec_report(
                spec=spec,
                validation_rows=validation_rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                annual_cache=annual_cache,
            )
        else:
            result = _r13_model_spec_report(
                spec=spec,
                rows=rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                r10_horizon_replay=r10_horizon_replay,
                prediction_cache=prediction_cache,
            )
        result["execution_order"] = len(results) + 1
        result["r10_scalar_reference_mae"] = r10_reference_mae
        results.append(result)
    manifest = {
        "schema_version": R13_PRIORITY_EXPERIMENT_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "min_train_years": int(min_train_years),
        "experiment_count": len(specs),
        "full_catalog_count": R13_EXPERIMENT_COUNT,
        "auto_deep_researcher_contract": (
            "bounded priority queue: experiments are predeclared, run sequentially by priority, and promoted only "
            "by blocked-time carry-forward/R10/stock semantics rather than open-ended optimizer budget"
        ),
        "specs": specs,
    }
    council = _r13_research_council_summary(results)
    report = {
        "schema_version": R13_PRIORITY_EXPERIMENT_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "experiment_count": len(results),
        "results": results,
        "council_summary": council,
        "artifact_paths": {},
    }
    manifest_path = analysis_dir / "r13_priority_experiment_manifest.json"
    report_path = analysis_dir / "r13_priority_experiment_results.json"
    csv_path = analysis_dir / "r13_priority_experiment_results.csv"
    md_path = analysis_dir / "r13_priority_experiment_results.md"
    dashboard_path = analysis_dir / "r13_priority_experiment_dashboard.png"
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    _write_r13_csv(csv_path, results)
    _write_r13_markdown(md_path, results)
    _write_r13_dashboard(dashboard_path, results)
    report["artifact_paths"] = {
        "manifest": manifest_path.as_posix(),
        "results_json": report_path.as_posix(),
        "results_csv": csv_path.as_posix(),
        "results_markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(report_path, report)
    return report
