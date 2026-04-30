from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3.dense_quarterly_panel import build_dense_quarterly_panel_rows
from epigraph_ph.phase3.tr_v3_05_autoresearch import (
    ANNUAL_METRICS,
    PRIMARY_METRICS,
    STATE_NAMES,
    TRANSITION_NAMES,
    AnnualIncidenceConfig,
    DynamicControlConfig,
    ObservationConfig,
    _annual_metric_error,
    _fit_annual_incidence_model,
    _fit_scalar_ar_trend,
    _provisional_quarterly_inflow,
    _quarter_positions,
    build_annual_anchor_rows,
    build_control_channels,
    build_quarterly_dataset,
    build_quarterly_observation_rows,
    carry_forward_hazards,
    default_frozen_04d_report,
    fit_observation_model,
    normalized_mae,
    quarter_gap,
    quarter_ordinal,
    quarter_sort_key,
    quarter_year,
    repo_root,
    rolling_origin_year_splits,
    simulate_closed_flow,
    simulate_open_inflow,
    smape,
)
from epigraph_ph.runtime import ensure_dir, write_json


@dataclass(slots=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    description: str
    transition_controls: dict[str, tuple[str, ...]]
    transition_model: str = "logit_ar"
    control_lag: int = 0
    lag_candidates: tuple[int, ...] = ()
    inflow_mode: str = "none"
    inflow_controls: tuple[str, ...] = ()
    use_infectious_pool: bool = True
    use_leakage: bool = True
    transition_ridge_multipliers: dict[str, float] = field(default_factory=dict)
    repair_params: dict[str, Any] = field(default_factory=dict)
    emit_hazard_curves: bool = False
    diagnostic_kind: str | None = None
    status: str = "executed"
    skip_reason: str | None = None
    benchmark_role: str | None = None


CANONICAL_14_EXPERIMENT_IDS: tuple[str, ...] = (
    "EXP-B0",
    "EXP-B1",
    "EXP-B2",
    "EXP-05a-01",
    "EXP-05a-02",
    "EXP-05a-04",
    "EXP-05a-05",
    "EXP-05a-06",
    "EXP-N1",
    "EXP-N2",
    "EXP-N3",
    "EXP-L1",
    "EXP-M1",
    "EXP-05b-full-open-leaky-model",
)

MECHANISTIC_RESEARCH_ANCHOR_ID = "EXP-R1"


def _latest_standard_archive_run() -> str:
    artifacts_dir = repo_root() / "artifacts" / "runs"
    candidates = sorted(
        [
            path.name
            for path in artifacts_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("harp-archive-wdi-standard-")
            and (path / "harp_archive" / "historical_metric_rows.json").exists()
        ]
    )
    if not candidates:
        raise FileNotFoundError("No harp-archive-wdi-standard-* archive run was found.")
    return str(candidates[-1])


def _control_map_a_only() -> dict[str, tuple[str, ...]]:
    return {transition: (("A",) if transition == "U_to_D" else tuple()) for transition in TRANSITION_NAMES}


def _control_map_c_only() -> dict[str, tuple[str, ...]]:
    return {
        "U_to_D": tuple(),
        "D_to_A": ("C",),
        "A_to_V": ("C",),
        "A_to_L": ("C",),
        "L_to_A": ("C",),
    }


def _control_map_r_only() -> dict[str, tuple[str, ...]]:
    return {transition: ("R",) for transition in TRANSITION_NAMES}


def _control_map_ac() -> dict[str, tuple[str, ...]]:
    return {
        "U_to_D": ("A",),
        "D_to_A": ("C",),
        "A_to_V": ("C",),
        "A_to_L": ("C",),
        "L_to_A": ("C",),
    }


def _control_map_acr() -> dict[str, tuple[str, ...]]:
    return {
        "U_to_D": ("A", "R"),
        "D_to_A": ("C", "R"),
        "A_to_V": ("C", "R"),
        "A_to_L": ("C", "R"),
        "L_to_A": ("C", "R"),
    }


def _control_map_acr_with_c_only_leakage() -> dict[str, tuple[str, ...]]:
    return {
        "U_to_D": ("A", "R"),
        "D_to_A": ("C", "R"),
        "A_to_V": ("C", "R"),
        "A_to_L": ("C",),
        "L_to_A": ("C",),
    }


def _repair_param_int(spec: ExperimentSpec, name: str, default: int) -> int:
    return int(spec.repair_params.get(name, default))


def _repair_param_float(spec: ExperimentSpec, name: str, default: float) -> float:
    return float(spec.repair_params.get(name, default))


def _repair_param_str(spec: ExperimentSpec, name: str, default: str | None = None) -> str | None:
    value = spec.repair_params.get(name, default)
    if value is None:
        return None
    return str(value)


def _repair_param_bool(spec: ExperimentSpec, name: str, default: bool) -> bool:
    return bool(spec.repair_params.get(name, default))


def build_experiment_suite_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec("EXP-B0", "diagnostic", "Benchmark-contract ablation across exact and dense training contracts.", {}, diagnostic_kind="contract_ablation"),
        ExperimentSpec("EXP-B1", "diagnostic", "Quarterly and annual metric availability map from 2010 onward.", {}, diagnostic_kind="missingness_map"),
        ExperimentSpec("EXP-B2", "diagnostic", "Imputation-contract ablation with observed-only holdout scoring.", {}, diagnostic_kind="dense_contract_summary"),
        ExperimentSpec("EXP-V1", "diagnostic", "Purged dense-contract audit with split-local dense reconstruction and targeted leaderboard comparison.", {}, diagnostic_kind="purged_dense_contract"),
        ExperimentSpec("EXP-V2", "diagnostic", "Endpoint and provenance-tier audit for the promoted winners and mechanistic anchor.", {}, diagnostic_kind="endpoint_tier_audit"),
        ExperimentSpec("EXP-S1-A1", "diagnostic", "Annual susceptible sidecar exploration using denominator, PLHIV, and incidence anchors only.", {}, diagnostic_kind="susceptible_sidecar"),
        ExperimentSpec("EXP-L2-A1", "diagnostic", "Late-era suppression-linked leakage sensitivity restricted to the partial-support window and kept diagnostic-only.", {}, diagnostic_kind="late_leakage_sensitivity"),
        ExperimentSpec("EXP-05a-01", "05a", "Ascertainment-only hazard control ablation.", _control_map_a_only(), control_lag=0),
        ExperimentSpec("EXP-05a-02", "05a", "Care-only hazard control ablation.", _control_map_c_only(), control_lag=0),
        ExperimentSpec("EXP-05a-04", "05a", "Hazard controls with A on diagnosis and C on care transitions.", _control_map_ac(), control_lag=0),
        ExperimentSpec("EXP-05a-05", "05a", "Full A/C/R hazard controls.", _control_map_acr(), control_lag=0),
        ExperimentSpec("EXP-05a-06", "05a", "Lag ablation for full A/C/R hazard controls.", _control_map_acr(), lag_candidates=(0, 1, 2)),
        ExperimentSpec(
            "EXP-R1",
            "repair",
            "Strict-support bounded-drift hazard repair with A plus C controls and explicit hazard diagnostics.",
            _control_map_ac(),
            transition_model="strict_support_drift",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R2",
            "repair",
            "Care-share repair: strict-support diagnosis/leakage hazards plus directly forecast ART and suppression shares for D_to_A and A_to_V.",
            _control_map_ac(),
            transition_model="care_share_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R3",
            "repair",
            "ART-stock reconciliation repair: strict-support diagnosis/leakage hazards plus direct alive_on_art stock forecasting and non-free suppression carry.",
            _control_map_ac(),
            transition_model="art_stock_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R4",
            "repair",
            "Hybrid care repair: use strict-support care hazards when identified and fall back to ART-stock reconciliation or suppression carry only when support is sparse.",
            _control_map_ac(),
            transition_model="hybrid_care_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R5",
            "repair",
            "D_to_A-only hybrid: use stock reconciliation for care initiation when needed, but keep A_to_V frozen on suppression carry unless direct suppression support is genuinely strong.",
            _control_map_ac(),
            transition_model="hybrid_d_to_a_only",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R6",
            "repair",
            "Pure D_to_A repair: switch only D_to_A between strict-support and ART-stock reconciliation, while A_to_V stays frozen on suppression carry for all splits.",
            _control_map_ac(),
            transition_model="d_to_a_only_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R7",
            "repair",
            "Regime-aware D_to_A repair: blend strict-support care initiation with ART-stock reconciliation when support is partial, while A_to_V stays frozen on suppression carry.",
            _control_map_ac(),
            transition_model="regime_aware_d_to_a_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R8",
            "repair",
            "Two-regime D_to_A repair: use reconciliation in the sparse early regime and strict-support only on a recent contiguous supported care block, while A_to_V stays frozen on suppression carry.",
            _control_map_ac(),
            transition_model="two_regime_d_to_a_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R9",
            "repair",
            "Net ART-delta D_to_A repair: identify care initiation from alive_on_art change rather than absolute stock level, with suppression still carried.",
            _control_map_ac(),
            transition_model="art_delta_repair",
            control_lag=0,
            emit_hazard_curves=True,
        ),
        ExperimentSpec(
            "EXP-R10",
            "repair",
            "Observation-first stock-flow repair: forecast diagnosed stock, ART stock, and diagnosis flow directly, then derive care quantities from those scored targets.",
            _control_map_ac(),
            transition_model="direct_observation_repair",
            control_lag=0,
            emit_hazard_curves=False,
        ),
        ExperimentSpec(
            "EXP-R10-M1",
            "repair",
            "Joint-consistency R10 refinement: forecast diagnosed stock, ART stock, and diagnosis flow jointly with empirical stock-flow reconciliation bounds and suppression as a sidecar only.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
            },
        ),
        ExperimentSpec(
            "EXP-R10-M1-B1",
            "repair",
            "Exact-lane joint-consistency refinement with train-only per-metric residual bias correction layered on top of M1.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency_bias_corrected",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-M1-F1",
            "repair",
            "Exact-lane flow correction on top of M1-B1: blend the diagnosis-flow head toward train-only stock-flow consistency implied by the diagnosed-stock path.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency_bias_corrected",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
                "flow_consistency_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-M1-F1-C1",
            "repair",
            "Exact-lane diagnosed calibration on top of M1-F1: add train-only inner rolling diagnosed calibration while preserving joint consistency, flow correction, and exact-lane bias correction.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency_crossfit_calibrated",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
                "use_bias_correction": True,
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
                "flow_consistency_weight": 1.0,
                "diagnosed_crossfit_weight": 1.0,
                "diagnosed_crossfit_min_points": 6,
                "diagnosed_crossfit_recent_pool": 12,
                "diagnosed_crossfit_min_train_years": 3,
            },
        ),
        ExperimentSpec(
            "EXP-R10-M2",
            "repair",
            "Train-only bias-corrected R10 refinement with per-metric residual correction learned from supported training residuals only.",
            _control_map_ac(),
            transition_model="direct_observation_bias_corrected",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-CHAMPION",
            "repair",
            "Legacy promoted quarterly benchmark candidate from SEARCH-R10-s100-f50.",
            _control_map_ac(),
            transition_model="direct_observation_repair",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.5,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-EXACT-CHAMPION",
            "repair",
            "Default exact-only predictive benchmark candidate promoted from SEARCH-R10C-a100-f25-delta-sup75.",
            _control_map_ac(),
            transition_model="direct_observation_repair",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
                "suppression_carry_weight": 0.75,
            },
            benchmark_role="predictive_exact_candidate",
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-CHAMPION",
            "repair",
            "Default dense-contract predictive benchmark candidate promoted from SEARCH-R10C-artdelta-ab100-f100.",
            _control_map_ac(),
            transition_model="direct_observation_repair",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
            },
            benchmark_role="predictive_dense_candidate",
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-H1",
            "repair",
            "Dense-champion suppression-honesty repair: keep the dense observation-first heads but stop emitting unsupported suppression level carries.",
            _control_map_ac(),
            transition_model="direct_observation_repair",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "suppression_fallback_mode": "unclaimed",
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M1",
            "repair",
            "Dense-champion plus joint-consistency refinement with supported stock-flow reconciliation bounds.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M1-H1",
            "repair",
            "Dense-champion plus joint-consistency refinement with unsupported suppression explicitly left unclaimed.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "suppression_fallback_mode": "unclaimed",
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M1-C1-H1",
            "repair",
            "Dense-champion plus joint-consistency and train-only inner rolling diagnosed calibration, with unsupported suppression explicitly left unclaimed.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency_crossfit_calibrated",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "suppression_fallback_mode": "unclaimed",
                "diagnosed_crossfit_weight": 1.0,
                "diagnosed_crossfit_min_points": 6,
                "diagnosed_crossfit_recent_pool": 16,
                "diagnosed_crossfit_min_train_years": 3,
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M1-B1-H1",
            "repair",
            "Dense-champion plus joint-consistency, train-only residual bias correction, and unsupported suppression left unclaimed.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency_bias_corrected",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "suppression_fallback_mode": "unclaimed",
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M1-F1-H1",
            "repair",
            "Dense-champion plus joint-consistency, stock-flow anchored diagnosis-flow correction, and unsupported suppression left unclaimed.",
            _control_map_ac(),
            transition_model="direct_observation_joint_consistency",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "suppression_fallback_mode": "unclaimed",
                "flow_consistency_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R10-DENSE-M2",
            "repair",
            "Dense-champion plus train-only per-metric bias correction.",
            _control_map_ac(),
            transition_model="direct_observation_bias_corrected",
            control_lag=0,
            emit_hazard_curves=False,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 1.0,
                "diagnosed_series_model": "level",
                "art_series_model": "delta",
                "flow_series_model": "level",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 1.0,
                "suppression_carry_weight": 1.0,
                "diagnosed_bias_weight": 1.0,
                "art_bias_weight": 1.0,
                "flow_bias_weight": 1.0,
            },
        ),
        ExperimentSpec(
            "EXP-R11",
            "repair",
            "Minimal mechanistic overlay on R10 observation heads: constrained diagnosis/infection overlay with annual incidence anchoring, no richer leakage, no quarterly mortality block, and no suppression claims.",
            _control_map_ac(),
            transition_model="direct_observation_mechanistic_overlay",
            control_lag=0,
            emit_hazard_curves=True,
            repair_params={
                "diagnosed_weight": 1.0,
                "art_weight": 1.0,
                "flow_weight": 0.25,
                "diagnosed_series_model": "level",
                "art_series_model": "level",
                "flow_series_model": "delta",
                "diagnosed_recent_blend_weight": 1.0,
                "art_recent_blend_weight": 1.0,
                "flow_recent_blend_weight": 0.75,
            },
        ),
        ExperimentSpec("EXP-N1", "05b", "Open inflow model with annual total forecast and equal quarterly inflow shares.", _control_map_acr(), control_lag=0, inflow_mode="equal_share", inflow_controls=(), use_infectious_pool=False, use_leakage=True),
        ExperimentSpec("EXP-N2", "05b", "Open inflow model with infectious-pool quarterly share scores only.", _control_map_acr(), control_lag=0, inflow_mode="score_model", inflow_controls=(), use_infectious_pool=True, use_leakage=True),
        ExperimentSpec("EXP-N3", "05b", "Open inflow model with infectious-pool plus A/C/R quarterly share scores.", _control_map_acr(), control_lag=0, inflow_mode="score_model", inflow_controls=("A", "C", "R"), use_infectious_pool=True, use_leakage=True),
        ExperimentSpec(
            "EXP-L1",
            "05a",
            "Honest net ART leakage and re-entry build with stronger leakage shrinkage and no reporting control on leakage transitions.",
            _control_map_acr_with_c_only_leakage(),
            control_lag=0,
            use_leakage=True,
            transition_ridge_multipliers={"A_to_L": 8.0, "L_to_A": 8.0},
        ),
        ExperimentSpec("EXP-M1", "deferred", "Shared mortality baseline block.", {}, status="skipped", skip_reason="Quarterly/state-specific mortality targets are still insufficient for an honest calibration loop."),
        ExperimentSpec("EXP-05b-full-open-leaky-model", "deferred", "Full open leaky incidence-flow model with mortality.", {}, status="skipped", skip_reason="Depends on the deferred richer leakage and mortality blocks."),
    ]


def canonical_design_experiment_ids() -> tuple[str, ...]:
    return CANONICAL_14_EXPERIMENT_IDS


def default_quarterly_benchmark_candidate_id() -> str:
    return default_predictive_candidate_id("exact_only")


def default_predictive_candidate_id(contract_name: str) -> str:
    role = "predictive_exact_candidate" if contract_name == "exact_only" else "predictive_dense_candidate"
    for spec in build_experiment_suite_specs():
        if spec.benchmark_role == role:
            return str(spec.experiment_id)
    raise LookupError(f"No predictive benchmark candidate is registered for contract {contract_name}.")


def mechanistic_research_anchor_id() -> str:
    return str(MECHANISTIC_RESEARCH_ANCHOR_ID)


def _suite_dynamic_grid() -> list[DynamicControlConfig]:
    return [
        DynamicControlConfig(ridge_penalty=ridge_penalty, rho_clip=rho_clip, trend_scale=1.0)
        for ridge_penalty in (0.01, 0.1, 1.0)
        for rho_clip in (0.8, 0.95)
    ]


def _suite_observation_grid() -> list[ObservationConfig]:
    return [
        ObservationConfig(calibration_ridge=obs_ridge, share_ridge_penalty=0.1, share_rho_clip=0.9, share_trend_scale=1.0)
        for obs_ridge in (0.01, 0.1)
    ]


def _suite_annual_grid() -> list[AnnualIncidenceConfig]:
    return [
        AnnualIncidenceConfig(ridge_penalty=annual_ridge, inflow_scale_clip=4.0)
        for annual_ridge in (0.01, 0.1)
    ]


def _lagged_controls(rows: list[dict[str, Any]], lag: int) -> dict[str, list[float]]:
    controls = build_control_channels(rows)
    if lag <= 0:
        return controls
    lagged: dict[str, list[float]] = {}
    for name, values in controls.items():
        shifted = [0.0] * len(values)
        for idx in range(lag, len(values)):
            shifted[idx] = float(values[idx - lag])
        lagged[name] = shifted
    return lagged


def _controls_for_transition_rows(
    train_rows: list[dict[str, Any]],
    train_transition_rows: list[dict[str, Any]],
    controls_train_full: dict[str, list[float]],
) -> dict[str, list[float]]:
    quarter_to_index = {str(row["quarter"]): idx for idx, row in enumerate(train_rows)}
    aligned: dict[str, list[float]] = {}
    for control_name, values in controls_train_full.items():
        aligned_values: list[float] = []
        for row in train_transition_rows:
            idx = quarter_to_index.get(str(row["quarter"]))
            aligned_values.append(float(values[idx]) if idx is not None and idx < len(values) else 0.0)
        aligned[control_name] = aligned_values
    return aligned


def _fit_logit_transition_model_configurable(
    train_transition_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    transition_name: str,
    controls_train: dict[str, list[float]],
    control_names: tuple[str, ...],
    *,
    cfg: DynamicControlConfig,
    eps: float,
    ridge_multiplier: float = 1.0,
) -> dict[str, float]:
    from epigraph_ph.phase3.tr_v3_05_autoresearch import inv_logit, logit

    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    train_hazards = [float(row["hazards"][transition_name]) for row in train_transition_rows]
    if len(train_hazards) < 2:
        constant = float(train_hazards[-1]) if train_hazards else 0.0
        return {quarter: constant for quarter in forecast_quarters}
    all_quarters = sorted(set(train_quarters + forecast_quarters), key=quarter_sort_key)
    base_position = quarter_ordinal(all_quarters[0])
    quarter_positions = {quarter: float(quarter_ordinal(quarter) - base_position) for quarter in all_quarters}
    etas = [logit(value, eps=eps) for value in train_hazards]
    y = np.asarray(etas[1:], dtype=np.float64)
    x_rows = []
    for idx in range(1, len(train_quarters)):
        quarter = train_quarters[idx]
        row = [1.0, quarter_positions[quarter] * cfg.trend_scale, etas[idx - 1]]
        for control_name in control_names:
            row.append(float(controls_train[control_name][idx]))
        x_rows.append(row)
    x = np.asarray(x_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty) * float(ridge_multiplier)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    beta[2] = float(np.clip(beta[2], -cfg.rho_clip, cfg.rho_clip))
    control_forecasts = {
        control_name: _fit_scalar_ar_trend(
            train_quarters,
            controls_train[control_name],
            forecast_quarters,
            ridge_penalty=cfg.ridge_penalty,
            rho_clip=cfg.rho_clip,
            trend_scale=cfg.trend_scale,
        )
        for control_name in control_names
    }
    last_eta = float(etas[-1])
    hazards: dict[str, float] = {}
    for quarter in forecast_quarters:
        row = [1.0, quarter_positions[quarter] * cfg.trend_scale, last_eta]
        row.extend(float(control_forecasts[control_name][quarter]) for control_name in control_names)
        eta = float(np.dot(np.asarray(row, dtype=np.float64), beta))
        hazards[quarter] = float(inv_logit(eta))
        last_eta = eta
    return hazards


def _control_lookup_for_transition_rows(
    train_transition_rows: list[dict[str, Any]],
    controls_train: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(train_transition_rows):
        quarter = str(row["quarter"])
        lookup[quarter] = {
            control_name: float(values[idx]) if idx < len(values) else 0.0
            for control_name, values in controls_train.items()
        }
    return lookup


def _clip_hazard(value: float, *, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _fit_strict_support_transition_model(
    train_transition_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    transition_name: str,
    controls_train: dict[str, list[float]],
    control_names: tuple[str, ...],
    *,
    cfg: DynamicControlConfig,
    ridge_multiplier: float = 1.0,
) -> dict[str, Any]:
    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    train_hazards = [float(row["hazards"][transition_name]) for row in train_transition_rows]
    control_lookup = _control_lookup_for_transition_rows(train_transition_rows, controls_train)
    support_mask = [bool(dict(row.get("support_flags") or {}).get(transition_name)) for row in train_transition_rows]
    support_sources = [str(dict(row.get("support_sources") or {}).get(transition_name) or "") for row in train_transition_rows]
    supported_rows = [(idx, row) for idx, row in enumerate(train_transition_rows) if support_mask[idx]]
    support_count = len(supported_rows)
    total_count = len(train_transition_rows)
    support_fraction = float(support_count) / float(total_count) if total_count else 0.0
    baseline_hazard = float(train_hazards[-1]) if train_hazards else 0.0
    if supported_rows:
        baseline_hazard = float(supported_rows[-1][1]["hazards"][transition_name])
    supported_hazards = [float(row["hazards"][transition_name]) for _, row in supported_rows]
    hazard_diffs = [abs(curr - prev) for prev, curr in zip(supported_hazards[:-1], supported_hazards[1:])]
    margin = float(max(hazard_diffs)) if hazard_diffs else 0.0
    lower = max(0.0, float(min(supported_hazards) if supported_hazards else baseline_hazard) - margin)
    upper = min(1.0, float(max(supported_hazards) if supported_hazards else baseline_hazard) + margin)
    if upper < lower:
        upper = lower
    if support_count < 2:
        fitted_map = {quarter: float(baseline_hazard) for quarter in train_quarters}
        forecast_map = {quarter: float(baseline_hazard) for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "transition_name": transition_name,
                "train_quarters": train_quarters,
                "train_observed": train_hazards,
                "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
                "train_supported": list(support_mask),
                "train_support_sources": list(support_sources),
                "forecast_quarters": list(forecast_quarters),
                "forecast_hazards": [float(forecast_map[quarter]) for quarter in forecast_quarters],
                "support_count": int(support_count),
                "support_fraction": float(support_fraction),
                "baseline_hazard": float(baseline_hazard),
                "forecast_lower": float(lower),
                "forecast_upper": float(upper),
                "model_kind": "strict_support_constant",
            },
        }

    y_rows: list[float] = []
    x_rows: list[list[float]] = []
    previous_supported_idx, previous_supported_row = supported_rows[0]
    previous_supported_quarter = str(previous_supported_row["quarter"])
    previous_delta = 0.0
    for idx, row in supported_rows[1:]:
        quarter = str(row["quarter"])
        dt = max(quarter_gap(previous_supported_quarter, quarter), 1)
        anchor_hazard = float(previous_supported_row["hazards"][transition_name])
        target_delta = float(row["hazards"][transition_name]) - anchor_hazard
        feature_row = [1.0, float(dt), float(previous_delta)]
        feature_row.extend(float(control_lookup[quarter].get(control_name, 0.0)) for control_name in control_names)
        x_rows.append(feature_row)
        y_rows.append(float(target_delta))
        previous_delta = float(target_delta)
        previous_supported_idx = idx
        previous_supported_row = row
        previous_supported_quarter = quarter
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty) * float(ridge_multiplier)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    delta_cap = float(max(abs(value) for value in y_rows)) if y_rows else 0.0
    control_forecasts = {
        control_name: _fit_scalar_ar_trend(
            train_quarters,
            controls_train[control_name],
            forecast_quarters,
            ridge_penalty=cfg.ridge_penalty,
            rho_clip=cfg.rho_clip,
            trend_scale=cfg.trend_scale,
        )
        for control_name in control_names
    }

    def predict_delta(*, quarter: str, dt: int, prev_delta_value: float, control_values: dict[str, float]) -> float:
        row = [1.0, float(dt), float(prev_delta_value)]
        row.extend(float(control_values.get(control_name, 0.0)) for control_name in control_names)
        raw_delta = float(np.dot(np.asarray(row, dtype=np.float64), beta))
        if delta_cap <= 0.0:
            return 0.0
        return float(np.clip(raw_delta * max(support_fraction, 0.0), -delta_cap, delta_cap))

    fitted_map: dict[str, float] = {}
    anchor_hazard = float(supported_rows[0][1]["hazards"][transition_name])
    anchor_quarter = str(supported_rows[0][1]["quarter"])
    previous_pred_delta = 0.0
    first_supported_idx = int(supported_rows[0][0])
    for idx, row in enumerate(train_transition_rows):
        quarter = str(row["quarter"])
        if idx < first_supported_idx:
            fitted_map[quarter] = float(anchor_hazard)
            continue
        if support_mask[idx]:
            if idx == first_supported_idx:
                fitted_map[quarter] = float(anchor_hazard)
            else:
                dt = max(quarter_gap(anchor_quarter, quarter), 1)
                predicted_delta = predict_delta(
                    quarter=quarter,
                    dt=dt,
                    prev_delta_value=previous_pred_delta,
                    control_values=control_lookup.get(quarter, {}),
                )
                fitted_hazard = _clip_hazard(anchor_hazard + predicted_delta, lower=lower, upper=upper)
                fitted_map[quarter] = float(fitted_hazard)
                previous_pred_delta = float(predicted_delta)
            anchor_hazard = float(row["hazards"][transition_name])
            anchor_quarter = quarter
        else:
            fitted_map[quarter] = float(anchor_hazard)

    forecast_map: dict[str, float] = {}
    forecast_anchor_hazard = float(anchor_hazard)
    forecast_anchor_quarter = str(anchor_quarter)
    forecast_prev_delta = float(previous_pred_delta)
    for quarter in forecast_quarters:
        dt = max(quarter_gap(forecast_anchor_quarter, quarter), 1)
        control_values = {control_name: float(control_forecasts[control_name].get(quarter, 0.0)) for control_name in control_names}
        predicted_delta = predict_delta(
            quarter=quarter,
            dt=dt,
            prev_delta_value=forecast_prev_delta,
            control_values=control_values,
        )
        predicted_hazard = _clip_hazard(forecast_anchor_hazard + predicted_delta, lower=lower, upper=upper)
        forecast_map[quarter] = float(predicted_hazard)
        forecast_anchor_hazard = float(predicted_hazard)
        forecast_anchor_quarter = quarter
        forecast_prev_delta = float(predicted_delta)

    return {
        "train_fitted_map": fitted_map,
        "forecast_map": forecast_map,
        "diagnostics": {
            "transition_name": transition_name,
            "train_quarters": train_quarters,
            "train_observed": train_hazards,
            "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
            "train_supported": list(support_mask),
            "train_support_sources": list(support_sources),
            "forecast_quarters": list(forecast_quarters),
            "forecast_hazards": [float(forecast_map[quarter]) for quarter in forecast_quarters],
            "support_count": int(support_count),
            "support_fraction": float(support_fraction),
            "baseline_hazard": float(baseline_hazard),
            "forecast_lower": float(lower),
            "forecast_upper": float(upper),
            "coefficients": [float(value) for value in beta.tolist()],
            "delta_cap": float(delta_cap),
            "model_kind": "strict_support_drift",
        },
    }


def _fit_supported_share_series(
    train_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    *,
    numerator_metric: str,
    denominator_metric: str,
) -> dict[str, Any]:
    supported = []
    for row in train_rows:
        numerator = row.get(numerator_metric)
        denominator = row.get(denominator_metric)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value <= 1e-9:
            continue
        if _metric_tier(row, numerator_metric) not in {"exact_observed", "bridge_observed"}:
            continue
        if _metric_tier(row, denominator_metric) not in {"exact_observed", "bridge_observed"}:
            continue
        supported.append((str(row["quarter"]), float(numerator) / denominator_value))
    train_quarters = [str(row["quarter"]) for row in train_rows]
    if not supported:
        fitted_map = {quarter: 0.0 for quarter in train_quarters}
        forecast_map = {quarter: 0.0 for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": numerator_metric,
                "denominator_metric": denominator_metric,
                "train_quarters": train_quarters,
                "train_observed": [None for _ in train_quarters],
                "train_fitted": [0.0 for _ in train_quarters],
                "train_supported": [False for _ in train_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [0.0 for _ in forecast_quarters],
                "model_kind": "share_constant_zero",
                "support_count": 0,
            },
        }
    supported_quarters = [quarter for quarter, _ in supported]
    supported_values = [float(value) for _, value in supported]
    lower = max(0.0, min(supported_values) - max([abs(curr - prev) for prev, curr in zip(supported_values[:-1], supported_values[1:])] or [0.0]))
    upper = min(1.0, max(supported_values) + max([abs(curr - prev) for prev, curr in zip(supported_values[:-1], supported_values[1:])] or [0.0]))
    if len(supported_values) < 2:
        constant = float(supported_values[-1])
        fitted_map = {quarter: constant for quarter in train_quarters}
        forecast_map = {quarter: constant for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": numerator_metric,
                "denominator_metric": denominator_metric,
                "train_quarters": train_quarters,
                "train_observed": [dict(supported).get(quarter) for quarter in train_quarters],
                "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
                "train_supported": [quarter in dict(supported) for quarter in train_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
                "model_kind": "share_constant_last",
                "support_count": 1,
                "forecast_lower": float(lower),
                "forecast_upper": float(upper),
            },
        }
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    previous_quarter = supported_quarters[0]
    previous_value = supported_values[0]
    previous_delta = 0.0
    for quarter, value in zip(supported_quarters[1:], supported_values[1:]):
        dt = max(quarter_gap(previous_quarter, quarter), 1)
        delta = float(value - previous_value)
        x_rows.append([1.0, float(dt), float(previous_delta)])
        y_rows.append(delta)
        previous_quarter = quarter
        previous_value = value
        previous_delta = delta
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * 0.1
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    delta_cap = float(max(abs(value) for value in y_rows)) if y_rows else 0.0

    def predict_delta(dt: int, prev_delta_value: float) -> float:
        raw_delta = float(np.dot(np.asarray([1.0, float(dt), float(prev_delta_value)], dtype=np.float64), beta))
        if delta_cap <= 0.0:
            return 0.0
        return float(np.clip(raw_delta, -delta_cap, delta_cap))

    supported_map = dict(supported)
    fitted_map: dict[str, float] = {}
    anchor_quarter = supported_quarters[0]
    anchor_value = supported_values[0]
    previous_pred_delta = 0.0
    for quarter in train_quarters:
        if quarter in supported_map:
            if quarter == supported_quarters[0]:
                fitted_map[quarter] = float(anchor_value)
            else:
                dt = max(quarter_gap(anchor_quarter, quarter), 1)
                predicted_delta = predict_delta(dt, previous_pred_delta)
                fitted_map[quarter] = float(np.clip(anchor_value + predicted_delta, lower, upper))
                previous_pred_delta = float(predicted_delta)
            anchor_quarter = quarter
            anchor_value = float(supported_map[quarter])
        else:
            fitted_map[quarter] = float(anchor_value)

    forecast_map: dict[str, float] = {}
    forecast_anchor_quarter = supported_quarters[-1]
    forecast_anchor_value = supported_values[-1]
    forecast_prev_delta = previous_pred_delta
    for quarter in forecast_quarters:
        dt = max(quarter_gap(forecast_anchor_quarter, quarter), 1)
        predicted_delta = predict_delta(dt, forecast_prev_delta)
        predicted_value = float(np.clip(forecast_anchor_value + predicted_delta, lower, upper))
        forecast_map[quarter] = predicted_value
        forecast_anchor_quarter = quarter
        forecast_anchor_value = predicted_value
        forecast_prev_delta = predicted_delta

    return {
        "train_fitted_map": fitted_map,
        "forecast_map": forecast_map,
        "diagnostics": {
            "metric": numerator_metric,
            "denominator_metric": denominator_metric,
            "train_quarters": train_quarters,
            "train_observed": [supported_map.get(quarter) for quarter in train_quarters],
            "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
            "train_supported": [quarter in supported_map for quarter in train_quarters],
            "forecast_quarters": list(forecast_quarters),
            "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
            "model_kind": "share_bounded_drift",
            "support_count": len(supported_values),
            "forecast_lower": float(lower),
            "forecast_upper": float(upper),
            "coefficients": [float(value) for value in beta.tolist()],
        },
    }


def _fit_supported_level_series(
    train_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    *,
    metric_name: str,
) -> dict[str, Any]:
    supported = []
    for row in train_rows:
        value = row.get(metric_name)
        if value is None:
            continue
        if _metric_tier(row, metric_name) not in {"exact_observed", "bridge_observed"}:
            continue
        supported.append((str(row["quarter"]), float(value)))
    train_quarters = [str(row["quarter"]) for row in train_rows]
    if not supported:
        fitted_map = {quarter: 0.0 for quarter in train_quarters}
        forecast_map = {quarter: 0.0 for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": metric_name,
                "train_quarters": train_quarters,
                "train_observed": [None for _ in train_quarters],
                "train_fitted": [0.0 for _ in train_quarters],
                "train_supported": [False for _ in train_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [0.0 for _ in forecast_quarters],
                "model_kind": "level_constant_zero",
                "support_count": 0,
            },
        }
    supported_quarters = [quarter for quarter, _ in supported]
    supported_values = [float(value) for _, value in supported]
    step_diffs = [abs(curr - prev) for prev, curr in zip(supported_values[:-1], supported_values[1:])]
    margin = float(max(step_diffs)) if step_diffs else 0.0
    lower = 0.0
    upper = float(max(supported_values) + margin)
    if len(supported_values) < 2:
        constant = float(supported_values[-1])
        fitted_map = {quarter: constant for quarter in train_quarters}
        forecast_map = {quarter: constant for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": metric_name,
                "train_quarters": train_quarters,
                "train_observed": [dict(supported).get(quarter) for quarter in train_quarters],
                "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
                "train_supported": [quarter in dict(supported) for quarter in train_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
                "model_kind": "level_constant_last",
                "support_count": 1,
                "forecast_lower": float(lower),
                "forecast_upper": float(upper),
            },
        }
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    previous_quarter = supported_quarters[0]
    previous_value = supported_values[0]
    previous_delta = 0.0
    for quarter, value in zip(supported_quarters[1:], supported_values[1:]):
        dt = max(quarter_gap(previous_quarter, quarter), 1)
        delta = float(value - previous_value)
        x_rows.append([1.0, float(dt), float(previous_delta)])
        y_rows.append(delta)
        previous_quarter = quarter
        previous_value = value
        previous_delta = delta
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * 0.1
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    delta_cap = float(max(abs(value) for value in y_rows)) if y_rows else 0.0

    def predict_delta(dt: int, prev_delta_value: float) -> float:
        raw_delta = float(np.dot(np.asarray([1.0, float(dt), float(prev_delta_value)], dtype=np.float64), beta))
        if delta_cap <= 0.0:
            return 0.0
        return float(np.clip(raw_delta, -delta_cap, delta_cap))

    supported_map = dict(supported)
    fitted_map: dict[str, float] = {}
    anchor_quarter = supported_quarters[0]
    anchor_value = supported_values[0]
    previous_pred_delta = 0.0
    for quarter in train_quarters:
        if quarter in supported_map:
            if quarter == supported_quarters[0]:
                fitted_map[quarter] = float(anchor_value)
            else:
                dt = max(quarter_gap(anchor_quarter, quarter), 1)
                predicted_delta = predict_delta(dt, previous_pred_delta)
                fitted_map[quarter] = float(np.clip(anchor_value + predicted_delta, lower, upper))
                previous_pred_delta = float(predicted_delta)
            anchor_quarter = quarter
            anchor_value = float(supported_map[quarter])
        else:
            fitted_map[quarter] = float(anchor_value)

    forecast_map: dict[str, float] = {}
    forecast_anchor_quarter = supported_quarters[-1]
    forecast_anchor_value = supported_values[-1]
    forecast_prev_delta = previous_pred_delta
    for quarter in forecast_quarters:
        dt = max(quarter_gap(forecast_anchor_quarter, quarter), 1)
        predicted_delta = predict_delta(dt, forecast_prev_delta)
        predicted_value = float(np.clip(forecast_anchor_value + predicted_delta, lower, upper))
        forecast_map[quarter] = predicted_value
        forecast_anchor_quarter = quarter
        forecast_anchor_value = predicted_value
        forecast_prev_delta = predicted_delta

    return {
        "train_fitted_map": fitted_map,
        "forecast_map": forecast_map,
        "diagnostics": {
            "metric": metric_name,
            "train_quarters": train_quarters,
            "train_observed": [supported_map.get(quarter) for quarter in train_quarters],
            "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
            "train_supported": [quarter in supported_map for quarter in train_quarters],
            "forecast_quarters": list(forecast_quarters),
            "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
            "model_kind": "level_bounded_drift",
            "support_count": len(supported_values),
            "forecast_lower": float(lower),
            "forecast_upper": float(upper),
            "coefficients": [float(value) for value in beta.tolist()],
        },
    }


def _fit_supported_delta_series(
    train_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    *,
    metric_name: str,
    recent_blend_weight: float = 1.0,
) -> dict[str, Any]:
    supported = []
    for row in train_rows:
        value = row.get(metric_name)
        if value is None:
            continue
        if _metric_tier(row, metric_name) not in {"exact_observed", "bridge_observed"}:
            continue
        supported.append((str(row["quarter"]), float(value)))
    train_target_quarters = [str(row["quarter"]) for row in train_rows[1:]]
    if len(supported) < 2:
        fitted_map = {quarter: 0.0 for quarter in train_target_quarters}
        forecast_map = {quarter: 0.0 for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": metric_name,
                "train_quarters": list(train_target_quarters),
                "train_observed": [None for _ in train_target_quarters],
                "train_fitted": [0.0 for _ in train_target_quarters],
                "train_supported": [False for _ in train_target_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [0.0 for _ in forecast_quarters],
                "model_kind": "delta_constant_zero",
                "support_count": max(len(supported) - 1, 0),
                "recent_blend_weight": float(recent_blend_weight),
            },
        }

    delta_quarters: list[str] = []
    delta_values: list[float] = []
    previous_quarter, previous_value = supported[0]
    for quarter, value in supported[1:]:
        dt = max(quarter_gap(previous_quarter, quarter), 1)
        delta_quarters.append(str(quarter))
        delta_values.append(float(value - previous_value) / float(dt))
        previous_quarter, previous_value = quarter, value

    supported_map = {quarter: float(value) for quarter, value in zip(delta_quarters, delta_values, strict=False)}
    margin = float(max(abs(curr - prev) for prev, curr in zip(delta_values[:-1], delta_values[1:], strict=False))) if len(delta_values) > 1 else abs(float(delta_values[-1]))
    lower = float(min(delta_values) - margin)
    upper = float(max(delta_values) + margin)
    carry_delta = float(delta_values[-1])

    if len(delta_values) < 2:
        fitted_map = {
            quarter: float(supported_map.get(quarter, carry_delta))
            for quarter in train_target_quarters
        }
        forecast_map = {quarter: float(carry_delta) for quarter in forecast_quarters}
        return {
            "train_fitted_map": fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": metric_name,
                "train_quarters": list(train_target_quarters),
                "train_observed": [supported_map.get(quarter) for quarter in train_target_quarters],
                "train_fitted": [float(fitted_map[quarter]) for quarter in train_target_quarters],
                "train_supported": [quarter in supported_map for quarter in train_target_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
                "model_kind": "delta_constant_last",
                "support_count": len(delta_values),
                "forecast_lower": float(lower),
                "forecast_upper": float(upper),
                "recent_blend_weight": float(recent_blend_weight),
                "carry_delta": float(carry_delta),
            },
        }

    forecast_raw = _fit_scalar_ar_trend(
        delta_quarters,
        delta_values,
        forecast_quarters,
        ridge_penalty=0.1,
        rho_clip=0.9,
        trend_scale=1.0,
    )
    blend = float(np.clip(float(recent_blend_weight), 0.0, 1.0))
    forecast_map = {
        quarter: float(np.clip((blend * float(forecast_raw[quarter])) + ((1.0 - blend) * carry_delta), lower, upper))
        for quarter in forecast_quarters
    }
    fitted_map: dict[str, float] = {}
    anchor_delta = float(delta_values[0])
    for quarter in train_target_quarters:
        if quarter in supported_map:
            anchor_delta = float(supported_map[quarter])
            fitted_map[quarter] = anchor_delta
        else:
            fitted_map[quarter] = float(anchor_delta)
    return {
        "train_fitted_map": fitted_map,
        "forecast_map": forecast_map,
        "diagnostics": {
            "metric": metric_name,
            "train_quarters": list(train_target_quarters),
            "train_observed": [supported_map.get(quarter) for quarter in train_target_quarters],
            "train_fitted": [float(fitted_map[quarter]) for quarter in train_target_quarters],
            "train_supported": [quarter in supported_map for quarter in train_target_quarters],
            "forecast_quarters": list(forecast_quarters),
            "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
            "model_kind": "delta_bounded_drift",
            "support_count": len(delta_values),
            "forecast_lower": float(lower),
            "forecast_upper": float(upper),
            "recent_blend_weight": float(blend),
            "carry_delta": float(carry_delta),
        },
    }


def _fit_supported_piecewise_series(
    train_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    *,
    metric_name: str,
    max_changepoints: int = 2,
    min_segment_points: int = 3,
) -> dict[str, Any]:
    supported = []
    for row in train_rows:
        value = row.get(metric_name)
        if value is None:
            continue
        if _metric_tier(row, metric_name) not in {"exact_observed", "bridge_observed"}:
            continue
        supported.append((str(row["quarter"]), float(value)))
    train_quarters = [str(row["quarter"]) for row in train_rows]
    if len(supported) < max(min_segment_points + 1, 4):
        fallback = _fit_supported_level_series(train_rows, forecast_quarters, metric_name=metric_name)
        diagnostics = dict(fallback["diagnostics"])
        diagnostics["model_kind"] = "piecewise_fallback_level"
        diagnostics["changepoint_quarters"] = []
        fallback["diagnostics"] = diagnostics
        return fallback

    supported_quarters = [quarter for quarter, _ in supported]
    supported_values = np.asarray([float(value) for _, value in supported], dtype=np.float64)
    base_ordinal = quarter_ordinal(supported_quarters[0])
    supported_positions = np.asarray(
        [float(quarter_ordinal(quarter) - base_ordinal) for quarter in supported_quarters],
        dtype=np.float64,
    )

    def fit_segment(start_idx: int, end_idx: int) -> tuple[np.ndarray, float, float]:
        x = supported_positions[start_idx : end_idx + 1]
        y = supported_values[start_idx : end_idx + 1]
        x_center = float(np.mean(x))
        x_design = np.asarray([[1.0, float(value - x_center)] for value in x], dtype=np.float64)
        ridge = np.eye(2, dtype=np.float64) * 1e-6
        ridge[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + ridge, x_design.T @ y)
        fitted = (x_design @ beta).astype(np.float64)
        sse = float(np.sum((y - fitted) ** 2))
        return fitted, float(sse), float(x_center), beta

    def segment_partitions(n_points: int) -> list[tuple[tuple[int, int], ...]]:
        partitions: list[tuple[tuple[int, int], ...]] = [((0, n_points - 1),)]
        max_cp = max(0, min(int(max_changepoints), n_points - 2))
        candidate_breaks = list(range(1, n_points - 1))
        for cp_count in range(1, max_cp + 1):
            for breaks in combinations(candidate_breaks, cp_count):
                edges = (0,) + tuple(int(value) for value in breaks) + (n_points,)
                segments: list[tuple[int, int]] = []
                valid = True
                for start, end in zip(edges[:-1], edges[1:], strict=False):
                    if int(end) - int(start) < int(min_segment_points):
                        valid = False
                        break
                    segments.append((int(start), int(end) - 1))
                if valid:
                    partitions.append(tuple(segments))
        return partitions

    best_partition: tuple[tuple[int, int], ...] | None = None
    best_partition_fitted: np.ndarray | None = None
    best_partition_betas: list[list[float]] = []
    best_partition_sse = float("inf")
    best_partition_bic = float("inf")
    for partition in segment_partitions(len(supported_quarters)):
        fitted_values = np.zeros_like(supported_values)
        partition_sse = 0.0
        partition_betas: list[list[float]] = []
        for start_idx, end_idx in partition:
            segment_fitted, segment_sse, x_center, beta = fit_segment(start_idx, end_idx)
            fitted_values[start_idx : end_idx + 1] = segment_fitted
            partition_sse += float(segment_sse)
            partition_betas.append([float(beta[0]), float(beta[1]), float(x_center)])
        k = 2 * len(partition)
        n = len(supported_values)
        partition_bic = (float(n) * math.log(max(partition_sse / float(max(n, 1)), 1e-9))) + (float(k) * math.log(float(max(n, 1))))
        if partition_bic < best_partition_bic:
            best_partition = partition
            best_partition_fitted = fitted_values
            best_partition_sse = float(partition_sse)
            best_partition_bic = float(partition_bic)
            best_partition_betas = list(partition_betas)
    assert best_partition is not None
    assert best_partition_fitted is not None

    segment_meta: list[dict[str, Any]] = []
    for (start_idx, end_idx), beta_row in zip(best_partition, best_partition_betas, strict=False):
        intercept, slope, x_center = beta_row
        segment_meta.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_quarter": str(supported_quarters[start_idx]),
                "end_quarter": str(supported_quarters[end_idx]),
                "intercept": float(intercept),
                "slope": float(slope),
                "x_center": float(x_center),
            }
        )

    def evaluate_segment(segment: dict[str, Any], quarter: str) -> float:
        x_value = float(quarter_ordinal(quarter) - base_ordinal)
        return float(segment["intercept"] + (segment["slope"] * (x_value - float(segment["x_center"]))))

    def segment_for_quarter(quarter: str) -> dict[str, Any]:
        q_ord = quarter_ordinal(quarter)
        for segment in segment_meta:
            if q_ord <= quarter_ordinal(str(segment["end_quarter"])):
                return segment
        return segment_meta[-1]

    level_lower = 0.0
    diffs = [abs(curr - prev) for prev, curr in zip(supported_values[:-1], supported_values[1:], strict=False)]
    level_upper = float(np.max(supported_values) + (max(diffs) if diffs else 0.0))
    fitted_supported_map = {
        str(quarter): float(np.clip(value, level_lower, level_upper))
        for quarter, value in zip(supported_quarters, best_partition_fitted.tolist(), strict=False)
    }
    fitted_map: dict[str, float] = {}
    carry_value = float(fitted_supported_map[supported_quarters[0]])
    for quarter in train_quarters:
        if quarter in fitted_supported_map:
            carry_value = float(fitted_supported_map[quarter])
        else:
            segment = segment_for_quarter(quarter)
            carry_value = float(np.clip(evaluate_segment(segment, quarter), level_lower, level_upper))
        fitted_map[quarter] = float(carry_value)

    forecast_map: dict[str, float] = {}
    last_segment = segment_meta[-1]
    for quarter in forecast_quarters:
        forecast_map[quarter] = float(
            np.clip(evaluate_segment(last_segment, quarter), level_lower, level_upper)
        )

    supported_map = dict(supported)
    changepoint_quarters = [
        str(supported_quarters[end_idx])
        for _, end_idx in best_partition[:-1]
    ]
    return {
        "train_fitted_map": fitted_map,
        "forecast_map": forecast_map,
        "diagnostics": {
            "metric": metric_name,
            "train_quarters": train_quarters,
            "train_observed": [supported_map.get(quarter) for quarter in train_quarters],
            "train_fitted": [float(fitted_map[quarter]) for quarter in train_quarters],
            "train_supported": [quarter in supported_map for quarter in train_quarters],
            "forecast_quarters": list(forecast_quarters),
            "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
            "model_kind": "piecewise_linear_changepoint",
            "support_count": len(supported_values),
            "forecast_lower": float(level_lower),
            "forecast_upper": float(level_upper),
            "segment_count": len(segment_meta),
            "changepoint_quarters": list(changepoint_quarters),
            "segments": segment_meta,
            "bic": float(best_partition_bic),
            "sse": float(best_partition_sse),
        },
    }


def _fit_supported_series(
    train_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    *,
    metric_name: str,
    model_kind: str = "level",
    recent_blend_weight: float = 1.0,
) -> dict[str, Any]:
    mode = str(model_kind)
    if mode == "level":
        return _fit_supported_level_series(train_rows, forecast_quarters, metric_name=metric_name)
    if mode == "delta":
        delta_result = _fit_supported_delta_series(
            train_rows,
            forecast_quarters,
            metric_name=metric_name,
            recent_blend_weight=recent_blend_weight,
        )
        supported_map: dict[str, float] = {}
        train_quarters = [str(row["quarter"]) for row in train_rows]
        for row in train_rows:
            value = row.get(metric_name)
            if value is None:
                continue
            if _metric_tier(row, metric_name) not in {"exact_observed", "bridge_observed"}:
                continue
            supported_map[str(row["quarter"])] = float(value)
        last_supported_value, support_count = _last_supported_level(train_rows, metric_name=metric_name)
        train_fitted_map: dict[str, float] = {}
        carried_value = 0.0
        for quarter in train_quarters:
            if quarter in supported_map:
                carried_value = float(supported_map[quarter])
            train_fitted_map[quarter] = float(carried_value)
        forecast_map: dict[str, float] = {}
        previous_quarter = train_quarters[-1] if train_quarters else None
        previous_value = float(last_supported_value)
        for quarter in forecast_quarters:
            dt = max(quarter_gap(previous_quarter, quarter), 1) if previous_quarter is not None else 1
            predicted_delta = float(delta_result["forecast_map"].get(quarter, 0.0))
            next_value = float(max(previous_value + (predicted_delta * float(dt)), 0.0))
            forecast_map[quarter] = next_value
            previous_quarter = str(quarter)
            previous_value = next_value
        return {
            "train_fitted_map": train_fitted_map,
            "forecast_map": forecast_map,
            "diagnostics": {
                "metric": metric_name,
                "train_quarters": train_quarters,
                "train_observed": [supported_map.get(quarter) for quarter in train_quarters],
                "train_fitted": [float(train_fitted_map[quarter]) for quarter in train_quarters],
                "train_supported": [quarter in supported_map for quarter in train_quarters],
                "forecast_quarters": list(forecast_quarters),
                "forecast_values": [float(forecast_map[quarter]) for quarter in forecast_quarters],
                "model_kind": "level_from_delta_bounded_drift",
                "support_count": int(support_count),
                "recent_blend_weight": float(recent_blend_weight),
                "delta_diagnostics": dict(delta_result["diagnostics"]),
            },
        }
    if mode == "piecewise":
        return _fit_supported_piecewise_series(
            train_rows,
            forecast_quarters,
            metric_name=metric_name,
        )
    raise ValueError(f"Unsupported supported-series model: {model_kind}")


def _last_supported_share(
    train_rows: list[dict[str, Any]],
    *,
    numerator_metric: str,
    denominator_metric: str,
) -> tuple[float, int]:
    supported_values: list[float] = []
    for row in train_rows:
        numerator = row.get(numerator_metric)
        denominator = row.get(denominator_metric)
        if numerator is None or denominator is None:
            continue
        if _metric_tier(row, numerator_metric) not in {"exact_observed", "bridge_observed"}:
            continue
        if _metric_tier(row, denominator_metric) not in {"exact_observed", "bridge_observed"}:
            continue
        denominator_value = float(denominator)
        if denominator_value <= 1e-9:
            continue
        supported_values.append(float(numerator) / denominator_value)
    if not supported_values:
        return 0.0, 0
    return float(np.clip(supported_values[-1], 0.0, 1.0)), len(supported_values)


def _last_supported_level(
    train_rows: list[dict[str, Any]],
    *,
    metric_name: str,
) -> tuple[float, int]:
    supported_values: list[float] = []
    for row in train_rows:
        value = row.get(metric_name)
        if value is None:
            continue
        if _metric_tier(row, metric_name) not in {"exact_observed", "bridge_observed"}:
            continue
        supported_values.append(float(value))
    if not supported_values:
        return 0.0, 0
    return float(supported_values[-1]), len(supported_values)


def _last_available_level(
    train_rows: list[dict[str, Any]],
    *,
    metric_name: str,
) -> tuple[float, int]:
    values = [float(row[metric_name]) for row in train_rows if row.get(metric_name) is not None]
    if not values:
        return 0.0, 0
    return float(values[-1]), len(values)


def _mean_supported_residual_from_diagnostics(diagnostics: dict[str, Any]) -> float:
    observed = list(diagnostics.get("train_observed") or [])
    fitted = list(diagnostics.get("train_fitted") or [])
    supported = list(diagnostics.get("train_supported") or [])
    residuals = [
        float(obs) - float(fit)
        for obs, fit, is_supported in zip(observed, fitted, supported, strict=False)
        if is_supported and obs is not None and fit is not None
    ]
    if not residuals:
        return 0.0
    return float(np.mean(np.asarray(residuals, dtype=np.float64)))


def _crossfit_supported_metric_correction(
    train_rows: list[dict[str, Any]],
    spec: ExperimentSpec,
    *,
    metric_name: str,
    diagnosed_weight: float,
    art_weight: float,
    flow_weight: float,
    diagnosed_series_model: str,
    art_series_model: str,
    flow_series_model: str,
    diagnosed_recent_blend_weight: float,
    art_recent_blend_weight: float,
    flow_recent_blend_weight: float,
    suppression_carry_weight: float,
    suppression_fallback_mode: str,
    use_joint_consistency: bool,
    use_bias_correction: bool,
    diagnosed_bias_weight: float,
    art_bias_weight: float,
    flow_bias_weight: float,
    flow_consistency_weight: float,
    min_train_years: int,
    min_points: int,
    recent_pool: int,
) -> tuple[float, dict[str, Any]]:
    eligible_years = sorted(
        {
            quarter_year(str(row["quarter"]))
            for row in train_rows
            if row.get(metric_name) is not None and _metric_tier(row, metric_name) in {"exact_observed", "bridge_observed"}
        }
    )
    if len(eligible_years) < max(int(min_train_years), 1) + 1:
        return 0.0, {
            "status": "insufficient_train_history",
            "metric_name": str(metric_name),
            "eligible_years": list(eligible_years),
            "residual_count": 0,
        }

    residual_points: list[dict[str, Any]] = []
    for holdout_year in eligible_years[max(int(min_train_years), 1):]:
        try:
            mini_dataset = build_quarterly_dataset(train_rows, [int(holdout_year)])
        except ValueError:
            continue
        if not mini_dataset.train_rows or not mini_dataset.holdout_rows:
            continue
        mini_candidate = _fit_direct_observation_repair_candidate(
            mini_dataset,
            DynamicControlConfig(),
            ObservationConfig(),
            spec,
            diagnosed_weight=diagnosed_weight,
            art_weight=art_weight,
            flow_weight=flow_weight,
            diagnosed_series_model=diagnosed_series_model,
            art_series_model=art_series_model,
            flow_series_model=flow_series_model,
            diagnosed_recent_blend_weight=diagnosed_recent_blend_weight,
            art_recent_blend_weight=art_recent_blend_weight,
            flow_recent_blend_weight=flow_recent_blend_weight,
            suppression_carry_weight=suppression_carry_weight,
            suppression_fallback_mode=suppression_fallback_mode,
            use_joint_consistency=use_joint_consistency,
            use_bias_correction=use_bias_correction,
            diagnosed_bias_weight=diagnosed_bias_weight,
            art_bias_weight=art_bias_weight,
            flow_bias_weight=flow_bias_weight,
            flow_consistency_weight=flow_consistency_weight,
            use_diagnosed_crossfit_calibration=False,
        )
        for target_row, prediction_row in zip(mini_dataset.holdout_rows, mini_candidate["prediction_rows"], strict=False):
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            tier_name = _metric_tier(target_row, metric_name)
            if tier_name not in {"exact_observed", "bridge_observed"}:
                continue
            residual_points.append(
                {
                    "quarter": str(target_row["quarter"]),
                    "year": int(quarter_year(str(target_row["quarter"]))),
                    "tier": str(tier_name),
                    "residual": float(prediction_value) - float(target_value),
                }
            )

    if len(residual_points) < max(int(min_points), 1):
        return 0.0, {
            "status": "insufficient_crossfit_points",
            "metric_name": str(metric_name),
            "eligible_years": list(eligible_years),
            "residual_count": int(len(residual_points)),
            "residual_points": residual_points,
        }

    residual_values = [float(point["residual"]) for point in residual_points]
    if int(recent_pool) > 0:
        residual_values = residual_values[-int(recent_pool):]
    correction = float(np.median(np.asarray(residual_values, dtype=np.float64))) if residual_values else 0.0
    return correction, {
        "status": "crossfit_correction_ready",
        "metric_name": str(metric_name),
        "eligible_years": list(eligible_years),
        "residual_count": int(len(residual_points)),
        "recent_pool": int(min(int(recent_pool), len(residual_points))) if int(recent_pool) > 0 else int(len(residual_points)),
        "correction": float(correction),
        "residual_points": residual_points,
    }


def _supported_stock_flow_residual_bounds(train_rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    ordered = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row["quarter"])))
    residuals: list[float] = []
    previous_row: dict[str, Any] | None = None
    for row in ordered:
        if previous_row is None:
            previous_row = row
            continue
        quarter = str(row["quarter"])
        previous_quarter = str(previous_row["quarter"])
        if quarter_gap(previous_quarter, quarter) != 1:
            previous_row = row
            continue
        diagnosed_previous = previous_row.get("diagnosed_plhiv")
        diagnosed_current = row.get("diagnosed_plhiv")
        flow_current = row.get("new_diagnosed_cases_period")
        if diagnosed_previous is None or diagnosed_current is None or flow_current is None:
            previous_row = row
            continue
        if _metric_tier(previous_row, "diagnosed_plhiv") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        if _metric_tier(row, "diagnosed_plhiv") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        if _metric_tier(row, "new_diagnosed_cases_period") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        residual = (float(diagnosed_current) - float(diagnosed_previous)) - float(flow_current)
        residuals.append(float(residual))
        previous_row = row
    if not residuals:
        return 0.0, 0.0, 0
    return float(min(residuals)), float(max(residuals)), len(residuals)


def _supported_stock_flow_residual_center(train_rows: list[dict[str, Any]]) -> tuple[float, int]:
    ordered = sorted(list(train_rows), key=lambda row: quarter_sort_key(str(row["quarter"])))
    residuals: list[float] = []
    previous_row: dict[str, Any] | None = None
    for row in ordered:
        if previous_row is None:
            previous_row = row
            continue
        quarter = str(row["quarter"])
        previous_quarter = str(previous_row["quarter"])
        if quarter_gap(previous_quarter, quarter) != 1:
            previous_row = row
            continue
        diagnosed_previous = previous_row.get("diagnosed_plhiv")
        diagnosed_current = row.get("diagnosed_plhiv")
        flow_current = row.get("new_diagnosed_cases_period")
        if diagnosed_previous is None or diagnosed_current is None or flow_current is None:
            previous_row = row
            continue
        if _metric_tier(previous_row, "diagnosed_plhiv") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        if _metric_tier(row, "diagnosed_plhiv") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        if _metric_tier(row, "new_diagnosed_cases_period") not in {"exact_observed", "bridge_observed"}:
            previous_row = row
            continue
        residuals.append((float(diagnosed_current) - float(diagnosed_previous)) - float(flow_current))
        previous_row = row
    if not residuals:
        return 0.0, 0
    return float(np.median(np.asarray(residuals, dtype=np.float64))), len(residuals)


def _supported_metric_level_bounds(train_rows: list[dict[str, Any]], *, metric_name: str) -> tuple[float, float]:
    supported_values = [
        float(row[metric_name])
        for row in train_rows
        if row.get(metric_name) is not None and _metric_tier(row, metric_name) in {"exact_observed", "bridge_observed"}
    ]
    if not supported_values:
        return 0.0, float("inf")
    step_diffs = [abs(curr - prev) for prev, curr in zip(supported_values[:-1], supported_values[1:], strict=False)]
    margin = float(max(step_diffs)) if step_diffs else abs(float(supported_values[-1]))
    return 0.0, float(max(supported_values) + margin)


def _care_support_mode(
    *,
    support_count: int,
    total_count: int,
    min_count: int = 4,
    min_fraction: float = 0.35,
) -> str:
    support_fraction = float(support_count) / float(total_count) if total_count > 0 else 0.0
    if int(support_count) >= int(min_count) and support_fraction >= float(min_fraction):
        return "strict_support"
    return "repair"


def _d_to_a_blend_weight(
    diagnostics: dict[str, Any],
    *,
    total_count: int,
    min_count: int,
    min_fraction: float,
) -> float:
    support_count = int(diagnostics.get("support_count") or 0)
    support_fraction = float(support_count) / float(total_count) if total_count > 0 else 0.0
    if support_count <= 0:
        return 0.0
    if support_count >= int(min_count) and support_fraction >= float(min_fraction):
        return 1.0
    train_quarters = [str(value) for value in diagnostics.get("train_quarters") or []]
    train_supported = [bool(value) for value in diagnostics.get("train_supported") or []]
    supported_quarters = [quarter for quarter, supported in zip(train_quarters, train_supported, strict=False) if supported]
    if not supported_quarters or not train_quarters:
        return 0.0
    latest_train_quarter = train_quarters[-1]
    latest_supported_quarter = supported_quarters[-1]
    recent_gap = max(int(quarter_gap(latest_supported_quarter, latest_train_quarter)), 0)
    recency_factor = 1.0 / float(1 + recent_gap)
    count_factor = min(float(support_count) / float(max(min_count, 1)), 1.0)
    weight = (0.5 * support_fraction) + (0.5 * count_factor * recency_factor)
    return float(np.clip(weight, 0.0, 0.95))


def _recent_supported_block_indices(
    train_transition_rows: list[dict[str, Any]],
    transition_name: str,
    *,
    min_supported: int = 3,
    max_inner_gap: int = 2,
    max_tail_gap: int = 1,
) -> list[int]:
    supported_indices = [
        idx
        for idx, row in enumerate(train_transition_rows)
        if bool(dict(row.get("support_flags") or {}).get(transition_name))
    ]
    if len(supported_indices) < int(min_supported):
        return []
    last_index = supported_indices[-1]
    end_index = len(train_transition_rows) - 1
    last_quarter = str(train_transition_rows[last_index]["quarter"])
    end_quarter = str(train_transition_rows[end_index]["quarter"])
    if quarter_gap(last_quarter, end_quarter) > int(max_tail_gap):
        return []
    block = [last_index]
    for idx in reversed(supported_indices[:-1]):
        current_quarter = str(train_transition_rows[idx]["quarter"])
        next_quarter = str(train_transition_rows[block[0]]["quarter"])
        if (block[0] - idx) == 1 and quarter_gap(current_quarter, next_quarter) <= int(max_inner_gap):
            block.insert(0, idx)
        else:
            break
    if len(block) < int(min_supported):
        return []
    return list(block)


def _quarterly_inflow_scores_configurable(
    trajectory_rows: list[dict[str, Any]],
    quarter_controls: dict[str, dict[str, float]],
    coeffs: dict[str, float],
    *,
    inflow_controls: tuple[str, ...],
    use_infectious_pool: bool,
) -> dict[str, float]:
    from epigraph_ph.phase3.tr_v3_05_autoresearch import softplus

    scores: dict[str, float] = {}
    for row in trajectory_rows:
        quarter = str(row["quarter"])
        state_values = dict(row["state_values"])
        value = float(coeffs.get("intercept", 0.0))
        if use_infectious_pool:
            infectious_pool = float(
                state_values["U"] + state_values["D"] + state_values["A"] + state_values["L"] + 0.02 * state_values["V"]
            )
            value += float(coeffs.get("log_infectious", 0.0)) * math.log1p(max(infectious_pool, 0.0))
        controls = quarter_controls.get(quarter, {})
        for control_name in inflow_controls:
            value += float(coeffs.get(control_name, 0.0)) * float(controls.get(control_name, 0.0))
        scores[quarter] = softplus(value)
    return scores


def _fit_quarterly_inflow_share_model_configurable(
    train_transition_rows: list[dict[str, Any]],
    train_controls: dict[str, list[float]],
    annual_rows: list[dict[str, Any]],
    cfg: AnnualIncidenceConfig,
    *,
    inflow_controls: tuple[str, ...],
    use_infectious_pool: bool,
) -> dict[str, float]:
    provisional = _provisional_quarterly_inflow(train_transition_rows, annual_rows)
    train_quarters = [str(row["quarter"]) for row in train_transition_rows if str(row["quarter"]) in provisional]
    if len(train_quarters) < 2:
        return {}
    quarter_to_idx = {str(row["quarter"]): idx for idx, row in enumerate(train_transition_rows)}
    x_rows = []
    y_rows = []
    for quarter in train_quarters:
        idx = quarter_to_idx[quarter]
        row_features = [1.0]
        if use_infectious_pool:
            state_previous = dict(train_transition_rows[idx]["state_values_previous"])
            infectious_pool = float(
                state_previous["U"] + state_previous["D"] + state_previous["A"] + state_previous["L"] + 0.02 * state_previous["V"]
            )
            row_features.append(math.log1p(max(infectious_pool, 0.0)))
        for control_name in inflow_controls:
            row_features.append(float(train_controls[control_name][idx]))
        x_rows.append(row_features)
        y_rows.append(math.log1p(max(float(provisional[quarter]), 0.0)))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    names = ["intercept"] + (["log_infectious"] if use_infectious_pool else []) + list(inflow_controls)
    return {name: float(value) for name, value in zip(names, beta)}


def _apply_leakage_mode(hazard_map: dict[str, dict[str, float]], use_leakage: bool) -> dict[str, dict[str, float]]:
    adjusted = {quarter: dict(values) for quarter, values in hazard_map.items()}
    if use_leakage:
        return adjusted
    for values in adjusted.values():
        values["A_to_L"] = 0.0
        values["L_to_A"] = 0.0
    return adjusted


def _fit_05a_experiment_candidate(
    dataset: Any,
    annual_rows: list[dict[str, Any]],
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    if spec.transition_model == "care_share_repair":
        return _fit_care_share_repair_candidate(dataset, dynamic_cfg, observation_cfg, spec)
    if spec.transition_model == "art_stock_repair":
        return _fit_art_stock_repair_candidate(dataset, dynamic_cfg, observation_cfg, spec)
    if spec.transition_model == "art_delta_repair":
        return _fit_art_delta_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            d_to_a_min_count=_repair_param_int(spec, "d_to_a_min_count", 4),
            d_to_a_min_fraction=_repair_param_float(spec, "d_to_a_min_fraction", 0.35),
            d_to_a_use_blend=_repair_param_bool(spec, "d_to_a_use_blend", True),
            art_delta_model_weight=_repair_param_float(spec, "art_delta_model_weight", 1.0),
        )
    if spec.transition_model == "direct_observation_repair":
        return _fit_direct_observation_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
        )
    if spec.transition_model == "direct_observation_joint_consistency":
        return _fit_direct_observation_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            use_joint_consistency=True,
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
        )
    if spec.transition_model == "direct_observation_bias_corrected":
        return _fit_direct_observation_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            use_bias_correction=True,
            diagnosed_bias_weight=_repair_param_float(spec, "diagnosed_bias_weight", 1.0),
            art_bias_weight=_repair_param_float(spec, "art_bias_weight", 1.0),
            flow_bias_weight=_repair_param_float(spec, "flow_bias_weight", 1.0),
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
        )
    if spec.transition_model == "direct_observation_joint_consistency_bias_corrected":
        return _fit_direct_observation_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            use_joint_consistency=True,
            use_bias_correction=True,
            diagnosed_bias_weight=_repair_param_float(spec, "diagnosed_bias_weight", 1.0),
            art_bias_weight=_repair_param_float(spec, "art_bias_weight", 1.0),
            flow_bias_weight=_repair_param_float(spec, "flow_bias_weight", 1.0),
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
        )
    if spec.transition_model == "direct_observation_joint_consistency_crossfit_calibrated":
        return _fit_direct_observation_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            use_joint_consistency=True,
            use_bias_correction=_repair_param_bool(spec, "use_bias_correction", False),
            diagnosed_bias_weight=_repair_param_float(spec, "diagnosed_bias_weight", 1.0),
            art_bias_weight=_repair_param_float(spec, "art_bias_weight", 1.0),
            flow_bias_weight=_repair_param_float(spec, "flow_bias_weight", 1.0),
            use_diagnosed_crossfit_calibration=True,
            diagnosed_crossfit_weight=_repair_param_float(spec, "diagnosed_crossfit_weight", 1.0),
            diagnosed_crossfit_min_points=_repair_param_int(spec, "diagnosed_crossfit_min_points", 5),
            diagnosed_crossfit_recent_pool=_repair_param_int(spec, "diagnosed_crossfit_recent_pool", 12),
            diagnosed_crossfit_min_train_years=_repair_param_int(spec, "diagnosed_crossfit_min_train_years", 3),
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
        )
    if spec.transition_model == "direct_observation_mechanistic_overlay":
        return _fit_direct_observation_mechanistic_overlay_candidate(
            dataset,
            annual_rows,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
        )
    if spec.transition_model == "direct_observation_shared_shock":
        return _fit_direct_observation_shared_shock_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            diagnosed_weight=_repair_param_float(spec, "diagnosed_weight", 1.0),
            art_weight=_repair_param_float(spec, "art_weight", 1.0),
            flow_weight=_repair_param_float(spec, "flow_weight", 1.0),
            diagnosed_series_model=_repair_param_str(spec, "diagnosed_series_model", "level") or "level",
            art_series_model=_repair_param_str(spec, "art_series_model", "level") or "level",
            flow_series_model=_repair_param_str(spec, "flow_series_model", "level") or "level",
            diagnosed_recent_blend_weight=_repair_param_float(spec, "diagnosed_recent_blend_weight", 1.0),
            art_recent_blend_weight=_repair_param_float(spec, "art_recent_blend_weight", 1.0),
            flow_recent_blend_weight=_repair_param_float(spec, "flow_recent_blend_weight", 1.0),
            suppression_carry_weight=_repair_param_float(spec, "suppression_carry_weight", 1.0),
            suppression_fallback_mode=_repair_param_str(spec, "suppression_fallback_mode", "level_carry") or "level_carry",
            use_joint_consistency=_repair_param_bool(spec, "use_joint_consistency", False),
            use_bias_correction=_repair_param_bool(spec, "use_bias_correction", False),
            diagnosed_bias_weight=_repair_param_float(spec, "diagnosed_bias_weight", 1.0),
            art_bias_weight=_repair_param_float(spec, "art_bias_weight", 1.0),
            flow_bias_weight=_repair_param_float(spec, "flow_bias_weight", 1.0),
            flow_consistency_weight=_repair_param_float(spec, "flow_consistency_weight", 0.0),
            shock_min_shared_metrics=_repair_param_int(spec, "shock_min_shared_metrics", 2),
            shock_threshold_quantile=_repair_param_float(spec, "shock_threshold_quantile", 0.7),
            shock_threshold_scale=_repair_param_float(spec, "shock_threshold_scale", 1.0),
            shock_forecast_blend_weight=_repair_param_float(spec, "shock_forecast_blend_weight", 0.75),
            shock_scale=_repair_param_float(spec, "shock_scale", 1.0),
        )
    if spec.transition_model == "hybrid_care_repair":
        return _fit_hybrid_care_repair_candidate(dataset, dynamic_cfg, observation_cfg, spec)
    if spec.transition_model == "hybrid_d_to_a_only":
        return _fit_hybrid_care_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            d_to_a_min_count=_repair_param_int(spec, "d_to_a_min_count", 4),
            d_to_a_min_fraction=_repair_param_float(spec, "d_to_a_min_fraction", 0.35),
            a_to_v_min_count=_repair_param_int(spec, "a_to_v_min_count", 6),
            a_to_v_min_fraction=_repair_param_float(spec, "a_to_v_min_fraction", 0.5),
        )
    if spec.transition_model == "d_to_a_only_repair":
        return _fit_hybrid_care_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            d_to_a_min_count=_repair_param_int(spec, "d_to_a_min_count", 4),
            d_to_a_min_fraction=_repair_param_float(spec, "d_to_a_min_fraction", 0.35),
            a_to_v_force_mode=_repair_param_str(spec, "a_to_v_force_mode", "repair"),
        )
    if spec.transition_model == "regime_aware_d_to_a_repair":
        return _fit_hybrid_care_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            d_to_a_min_count=_repair_param_int(spec, "d_to_a_min_count", 4),
            d_to_a_min_fraction=_repair_param_float(spec, "d_to_a_min_fraction", 0.35),
            a_to_v_force_mode=_repair_param_str(spec, "a_to_v_force_mode", "repair"),
            d_to_a_use_blend=_repair_param_bool(spec, "d_to_a_use_blend", True),
        )
    if spec.transition_model == "two_regime_d_to_a_repair":
        return _fit_two_regime_d_to_a_repair_candidate(
            dataset,
            dynamic_cfg,
            observation_cfg,
            spec,
            min_supported=_repair_param_int(spec, "recent_min_supported", 3),
            max_inner_gap=_repair_param_int(spec, "recent_max_inner_gap", 2),
            max_tail_gap=_repair_param_int(spec, "recent_max_tail_gap", 1),
        )
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_transition_rows}
    transition_diagnostics: dict[str, Any] = {}
    for transition_name in TRANSITION_NAMES:
        if spec.transition_model == "strict_support_drift":
            fit_result = _fit_strict_support_transition_model(
                dataset.train_transition_rows,
                holdout_quarters,
                transition_name,
                controls_transition,
                spec.transition_controls.get(transition_name, tuple()),
                cfg=dynamic_cfg,
                ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
            )
            forecast = dict(fit_result["forecast_map"])
            for quarter, value in dict(fit_result["train_fitted_map"]).items():
                train_hazard_map[quarter][transition_name] = float(value)
            transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        else:
            forecast = _fit_logit_transition_model_configurable(
                dataset.train_transition_rows,
                holdout_quarters,
                transition_name,
                controls_transition,
                spec.transition_controls.get(transition_name, tuple()),
                cfg=dynamic_cfg,
                eps=dataset.eps,
                ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
            )
            for row in dataset.train_transition_rows:
                train_hazard_map[str(row["quarter"])][transition_name] = float(row["hazards"][transition_name])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = value
    adjusted_hazard_map = _apply_leakage_mode(hazard_map, spec.use_leakage)
    raw = simulate_closed_flow(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        adjusted_hazard_map,
    )
    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        raw["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": raw["trajectory_rows"],
        "hazard_map": adjusted_hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _simulate_care_share_repair(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    base_hazard_map: dict[str, dict[str, float]],
    art_share_targets: dict[str, float],
    suppression_share_targets: dict[str, float],
    *,
    eps: float,
) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(base_hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)

        diagnosed_total_post_diag = max(float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"] + u_to_d), 0.0)
        art_total_after_leakage = max(float(previous_state["A"] + previous_state["V"] - a_to_l + l_to_a), 0.0)
        art_share = float(np.clip(float(art_share_targets.get(quarter, 0.0)), 0.0, 1.0))
        target_art_total = float(art_share * diagnosed_total_post_diag)
        eligible_diagnosed = max(float(previous_state["D"] + u_to_d), 0.0)
        d_to_a_flow = float(np.clip(target_art_total - art_total_after_leakage, 0.0, eligible_diagnosed))
        unsuppressed_after_initiation = max(float(previous_state["A"] - a_to_l + l_to_a + d_to_a_flow), 0.0)

        suppression_share = float(np.clip(float(suppression_share_targets.get(quarter, 0.0)), 0.0, 1.0))
        target_v_total = float(suppression_share * max(art_total_after_leakage + d_to_a_flow, 0.0))
        a_to_v_flow = float(np.clip(target_v_total - float(previous_state["V"]), 0.0, unsuppressed_after_initiation))

        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a_flow, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a_flow + l_to_a - a_to_l - a_to_v_flow, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v_flow, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        hazards["D_to_A"] = float(d_to_a_flow / max(eligible_diagnosed, eps))
        hazards["A_to_V"] = float(a_to_v_flow / max(unsuppressed_after_initiation, eps))
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
                "alive_on_art": float(current_state["A"] + current_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": None,
                "virally_suppressed": float(current_state["V"]),
            }
        )
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _simulate_art_stock_repair(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    base_hazard_map: dict[str, dict[str, float]],
    art_stock_targets: dict[str, float],
    *,
    suppression_share_carry: float,
    eps: float,
) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    share_carry = float(np.clip(suppression_share_carry, 0.0, 1.0))
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(base_hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)

        diagnosed_total_post_diag = max(float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"] + u_to_d), 0.0)
        art_total_after_leakage = max(float(previous_state["A"] + previous_state["V"] - a_to_l + l_to_a), 0.0)
        target_art_total = float(np.clip(float(art_stock_targets.get(quarter, art_total_after_leakage)), 0.0, diagnosed_total_post_diag))
        eligible_diagnosed = max(float(previous_state["D"] + u_to_d), 0.0)
        d_to_a_flow = float(np.clip(target_art_total - art_total_after_leakage, 0.0, eligible_diagnosed))

        art_total_current = max(art_total_after_leakage + d_to_a_flow, 0.0)
        target_v_total = float(np.clip(share_carry * art_total_current, 0.0, art_total_current))
        available_unsuppressed = max(float(previous_state["A"] - a_to_l + l_to_a + d_to_a_flow), 0.0)
        a_to_v_flow = float(np.clip(target_v_total - float(previous_state["V"]), 0.0, available_unsuppressed))

        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a_flow, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a_flow + l_to_a - a_to_l - a_to_v_flow, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v_flow, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        hazards["D_to_A"] = float(d_to_a_flow / max(eligible_diagnosed, eps))
        hazards["A_to_V"] = float(a_to_v_flow / max(available_unsuppressed, eps))
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
                "alive_on_art": float(current_state["A"] + current_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": None,
                "virally_suppressed": float(current_state["V"]),
            }
        )
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _simulate_art_delta_repair(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    base_hazard_map: dict[str, dict[str, float]],
    art_delta_targets: dict[str, float],
    *,
    d_to_a_mode: str,
    d_to_a_blend_weight: float = 0.0,
    suppression_share_carry: float,
    eps: float,
) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    share_carry = float(np.clip(suppression_share_carry, 0.0, 1.0))
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(base_hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)

        art_total_previous = max(float(previous_state["A"] + previous_state["V"]), 0.0)
        art_total_after_leakage = max(art_total_previous - a_to_l + l_to_a, 0.0)
        diagnosed_total_post_diag = max(float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"] + u_to_d), 0.0)
        eligible_diagnosed = max(float(previous_state["D"] + u_to_d), 0.0)

        target_delta = float(art_delta_targets.get(quarter, 0.0))
        target_art_total = float(np.clip(art_total_previous + target_delta, 0.0, diagnosed_total_post_diag))
        repair_d_to_a_flow = float(np.clip(target_art_total - art_total_after_leakage, 0.0, eligible_diagnosed))
        strict_d_to_a_hazard = float(np.clip(float(hazards.get("D_to_A") or 0.0), 0.0, 1.0))
        strict_d_to_a_flow = float(np.clip(strict_d_to_a_hazard * eligible_diagnosed, 0.0, eligible_diagnosed))

        if d_to_a_mode == "strict_support":
            d_to_a_flow = float(strict_d_to_a_flow)
            d_to_a_hazard = float(strict_d_to_a_hazard)
        elif d_to_a_mode == "blend":
            blend = float(np.clip(d_to_a_blend_weight, 0.0, 1.0))
            d_to_a_flow = float(np.clip((blend * strict_d_to_a_flow) + ((1.0 - blend) * repair_d_to_a_flow), 0.0, eligible_diagnosed))
            d_to_a_hazard = float(d_to_a_flow / max(eligible_diagnosed, eps))
        else:
            d_to_a_flow = float(repair_d_to_a_flow)
            d_to_a_hazard = float(d_to_a_flow / max(eligible_diagnosed, eps))

        art_total_current = max(art_total_after_leakage + d_to_a_flow, 0.0)
        available_unsuppressed = max(float(previous_state["A"] - a_to_l + l_to_a + d_to_a_flow), 0.0)
        target_v_total = float(np.clip(share_carry * art_total_current, 0.0, art_total_current))
        a_to_v_flow = float(np.clip(target_v_total - float(previous_state["V"]), 0.0, available_unsuppressed))
        a_to_v_hazard = float(a_to_v_flow / max(available_unsuppressed, eps))

        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a_flow, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a_flow + l_to_a - a_to_l - a_to_v_flow, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v_flow, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        hazards["D_to_A"] = float(d_to_a_hazard)
        hazards["A_to_V"] = float(a_to_v_hazard)
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
                "alive_on_art": float(current_state["A"] + current_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": None,
                "virally_suppressed": float(current_state["V"]),
            }
        )
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _simulate_hybrid_care_repair(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    base_hazard_map: dict[str, dict[str, float]],
    art_stock_targets: dict[str, float],
    *,
    d_to_a_mode: str,
    a_to_v_mode: str,
    d_to_a_blend_weight: float = 0.0,
    suppression_share_carry: float,
    eps: float,
) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    share_carry = float(np.clip(suppression_share_carry, 0.0, 1.0))
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(base_hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)

        diagnosed_total_post_diag = max(float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"] + u_to_d), 0.0)
        art_total_after_leakage = max(float(previous_state["A"] + previous_state["V"] - a_to_l + l_to_a), 0.0)
        eligible_diagnosed = max(float(previous_state["D"] + u_to_d), 0.0)
        target_art_total = float(np.clip(float(art_stock_targets.get(quarter, art_total_after_leakage)), 0.0, diagnosed_total_post_diag))
        repair_d_to_a_flow = float(np.clip(target_art_total - art_total_after_leakage, 0.0, eligible_diagnosed))
        strict_d_to_a_hazard = float(np.clip(float(hazards.get("D_to_A") or 0.0), 0.0, 1.0))
        strict_d_to_a_flow = float(np.clip(strict_d_to_a_hazard * eligible_diagnosed, 0.0, eligible_diagnosed))

        if d_to_a_mode == "strict_support":
            d_to_a_flow = float(strict_d_to_a_flow)
            d_to_a_hazard = float(strict_d_to_a_hazard)
        elif d_to_a_mode == "blend":
            blend = float(np.clip(d_to_a_blend_weight, 0.0, 1.0))
            d_to_a_flow = float(np.clip((blend * strict_d_to_a_flow) + ((1.0 - blend) * repair_d_to_a_flow), 0.0, eligible_diagnosed))
            d_to_a_hazard = float(d_to_a_flow / max(eligible_diagnosed, eps))
        else:
            d_to_a_flow = float(repair_d_to_a_flow)
            d_to_a_hazard = float(d_to_a_flow / max(eligible_diagnosed, eps))

        available_unsuppressed = max(float(previous_state["A"] - a_to_l + l_to_a + d_to_a_flow), 0.0)
        if a_to_v_mode == "strict_support":
            a_to_v_hazard = float(np.clip(float(hazards.get("A_to_V") or 0.0), 0.0, 1.0))
            a_to_v_flow = float(np.clip(a_to_v_hazard * available_unsuppressed, 0.0, available_unsuppressed))
        else:
            art_total_current = max(art_total_after_leakage + d_to_a_flow, 0.0)
            target_v_total = float(np.clip(share_carry * art_total_current, 0.0, art_total_current))
            a_to_v_flow = float(np.clip(target_v_total - float(previous_state["V"]), 0.0, available_unsuppressed))
            a_to_v_hazard = float(a_to_v_flow / max(available_unsuppressed, eps))

        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a_flow, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a_flow + l_to_a - a_to_v_flow - a_to_l, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v_flow, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        hazards["D_to_A"] = float(d_to_a_hazard)
        hazards["A_to_V"] = float(a_to_v_hazard)
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
                "alive_on_art": float(current_state["A"] + current_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": None,
                "virally_suppressed": float(current_state["V"]),
            }
        )
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _simulate_two_regime_d_to_a_repair(
    initial_state: dict[str, float],
    target_rows: list[dict[str, Any]],
    base_hazard_map: dict[str, dict[str, float]],
    art_stock_targets: dict[str, float],
    *,
    d_to_a_mode_by_quarter: dict[str, str],
    suppression_share_carry: float,
    eps: float,
) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    share_carry = float(np.clip(suppression_share_carry, 0.0, 1.0))
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(base_hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)

        diagnosed_total_post_diag = max(float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"] + u_to_d), 0.0)
        art_total_after_leakage = max(float(previous_state["A"] + previous_state["V"] - a_to_l + l_to_a), 0.0)
        eligible_diagnosed = max(float(previous_state["D"] + u_to_d), 0.0)
        strict_d_to_a_hazard = float(np.clip(float(hazards.get("D_to_A") or 0.0), 0.0, 1.0))
        strict_d_to_a_flow = float(np.clip(strict_d_to_a_hazard * eligible_diagnosed, 0.0, eligible_diagnosed))
        target_art_total = float(np.clip(float(art_stock_targets.get(quarter, art_total_after_leakage)), 0.0, diagnosed_total_post_diag))
        repair_d_to_a_flow = float(np.clip(target_art_total - art_total_after_leakage, 0.0, eligible_diagnosed))
        d_to_a_mode = str(d_to_a_mode_by_quarter.get(quarter, "repair"))
        if d_to_a_mode == "strict_support":
            d_to_a_flow = float(strict_d_to_a_flow)
            d_to_a_hazard = float(strict_d_to_a_hazard)
        else:
            d_to_a_flow = float(repair_d_to_a_flow)
            d_to_a_hazard = float(d_to_a_flow / max(eligible_diagnosed, eps))

        available_unsuppressed = max(float(previous_state["A"] - a_to_l + l_to_a + d_to_a_flow), 0.0)
        art_total_current = max(art_total_after_leakage + d_to_a_flow, 0.0)
        target_v_total = float(np.clip(share_carry * art_total_current, 0.0, art_total_current))
        a_to_v_flow = float(np.clip(target_v_total - float(previous_state["V"]), 0.0, available_unsuppressed))
        a_to_v_hazard = float(a_to_v_flow / max(available_unsuppressed, eps))

        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a_flow, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a_flow + l_to_a - a_to_v_flow - a_to_l, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v_flow, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        hazards["D_to_A"] = float(d_to_a_hazard)
        hazards["A_to_V"] = float(a_to_v_hazard)
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
                "alive_on_art": float(current_state["A"] + current_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": None,
                "virally_suppressed": float(current_state["V"]),
            }
        )
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _fit_care_share_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_transition_rows}
    transition_diagnostics: dict[str, Any] = {}

    for transition_name in ("U_to_D", "A_to_L", "L_to_A"):
        fit_result = _fit_strict_support_transition_model(
            dataset.train_transition_rows,
            holdout_quarters,
            transition_name,
            controls_transition,
            spec.transition_controls.get(transition_name, tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
        )
        forecast = dict(fit_result["forecast_map"])
        for quarter, value in dict(fit_result["train_fitted_map"]).items():
            train_hazard_map[quarter][transition_name] = float(value)
        transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = float(value)

    art_share_result = _fit_supported_share_series(
        dataset.train_rows,
        holdout_quarters,
        numerator_metric="alive_on_art",
        denominator_metric="diagnosed_plhiv",
    )
    suppression_share_result = _fit_supported_share_series(
        dataset.train_rows,
        holdout_quarters,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )
    care_sim = _simulate_care_share_repair(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        hazard_map,
        dict(art_share_result["forecast_map"]),
        dict(suppression_share_result["forecast_map"]),
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(holdout_quarters, care_sim["trajectory_rows"]):
        hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])
    transition_diagnostics["D_to_A"] = {
        "transition_name": "D_to_A",
        "train_quarters": list(art_share_result["diagnostics"]["train_quarters"]),
        "train_observed": list(art_share_result["diagnostics"]["train_observed"]),
        "train_fitted": list(art_share_result["diagnostics"]["train_fitted"]),
        "train_supported": list(art_share_result["diagnostics"]["train_supported"]),
        "forecast_quarters": list(art_share_result["diagnostics"]["forecast_quarters"]),
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "forecast_lower": 0.0,
        "forecast_upper": 1.0,
        "support_count": int(art_share_result["diagnostics"]["support_count"]),
        "support_fraction": float(art_share_result["diagnostics"]["support_count"]) / max(len(dataset.train_rows), 1),
        "model_kind": "care_share_repair_art_share",
        "share_diagnostics": dict(art_share_result["diagnostics"]),
    }
    transition_diagnostics["A_to_V"] = {
        "transition_name": "A_to_V",
        "train_quarters": list(suppression_share_result["diagnostics"]["train_quarters"]),
        "train_observed": list(suppression_share_result["diagnostics"]["train_observed"]),
        "train_fitted": list(suppression_share_result["diagnostics"]["train_fitted"]),
        "train_supported": list(suppression_share_result["diagnostics"]["train_supported"]),
        "forecast_quarters": list(suppression_share_result["diagnostics"]["forecast_quarters"]),
        "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
        "forecast_lower": 0.0,
        "forecast_upper": 1.0,
        "support_count": int(suppression_share_result["diagnostics"]["support_count"]),
        "support_fraction": float(suppression_share_result["diagnostics"]["support_count"]) / max(len(dataset.train_rows), 1),
        "model_kind": "care_share_repair_suppression_share",
        "share_diagnostics": dict(suppression_share_result["diagnostics"]),
    }

    for row in dataset.train_transition_rows:
        quarter = str(row["quarter"])
        train_hazard_map[quarter]["D_to_A"] = float(row["hazards"]["D_to_A"])
        train_hazard_map[quarter]["A_to_V"] = float(row["hazards"]["A_to_V"])
    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        care_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": care_sim["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_art_stock_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_rows[1:]}
    transition_diagnostics: dict[str, Any] = {}

    for transition_name in ("U_to_D", "A_to_L", "L_to_A"):
        fit_result = _fit_strict_support_transition_model(
            dataset.train_transition_rows,
            holdout_quarters,
            transition_name,
            controls_transition,
            spec.transition_controls.get(transition_name, tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
        )
        forecast = dict(fit_result["forecast_map"])
        for quarter, value in dict(fit_result["train_fitted_map"]).items():
            train_hazard_map[quarter][transition_name] = float(value)
        transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = float(value)

    art_stock_result = _fit_supported_level_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
    )
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )

    train_target_rows = list(dataset.train_rows[1:])
    train_target_quarters = [str(row["quarter"]) for row in train_target_rows]
    train_art_targets = {
        quarter: float(art_stock_result["train_fitted_map"].get(quarter, 0.0))
        for quarter in train_target_quarters
    }
    train_sim = _simulate_art_stock_repair(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_target_rows,
        train_hazard_map,
        train_art_targets,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(train_target_quarters, train_sim["trajectory_rows"]):
        train_hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        train_hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    holdout_sim = _simulate_art_stock_repair(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        hazard_map,
        dict(art_stock_result["forecast_map"]),
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(holdout_quarters, holdout_sim["trajectory_rows"]):
        hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    art_stock_diag = dict(art_stock_result["diagnostics"])
    art_stock_support_map = {
        str(quarter): bool(supported)
        for quarter, supported in zip(
            art_stock_diag.get("train_quarters") or [],
            art_stock_diag.get("train_supported") or [],
            strict=False,
        )
    }
    suppression_support_map = {
        str(row["quarter"]): bool(
            _metric_tier(row, "virally_suppressed") in {"exact_observed", "bridge_observed"}
            and _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"}
            and row.get("virally_suppressed") is not None
            and row.get("alive_on_art") is not None
        )
        for row in dataset.train_rows
    }
    train_transition_quarters = [str(row["quarter"]) for row in dataset.train_transition_rows]
    observed_d_to_a = [float(row["hazards"]["D_to_A"]) for row in dataset.train_transition_rows]
    fitted_d_to_a = [float(train_hazard_map[quarter]["D_to_A"]) for quarter in train_transition_quarters]
    observed_a_to_v = [float(row["hazards"]["A_to_V"]) for row in dataset.train_transition_rows]
    fitted_a_to_v = [float(train_hazard_map[quarter]["A_to_V"]) for quarter in train_transition_quarters]
    transition_diagnostics["D_to_A"] = {
        "transition_name": "D_to_A",
        "train_quarters": list(train_transition_quarters),
        "train_observed": list(observed_d_to_a),
        "train_fitted": list(fitted_d_to_a),
        "train_supported": [bool(art_stock_support_map.get(quarter, False)) for quarter in train_transition_quarters],
        "forecast_quarters": list(holdout_quarters),
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "forecast_lower": 0.0,
        "forecast_upper": 1.0,
        "support_count": int(art_stock_diag.get("support_count") or 0),
        "support_fraction": float(art_stock_diag.get("support_count") or 0.0) / max(len(dataset.train_rows), 1),
        "model_kind": "art_stock_repair_alive_on_art",
        "level_diagnostics": art_stock_diag,
    }
    transition_diagnostics["A_to_V"] = {
        "transition_name": "A_to_V",
        "train_quarters": list(train_transition_quarters),
        "train_observed": list(observed_a_to_v),
        "train_fitted": list(fitted_a_to_v),
        "train_supported": [bool(suppression_support_map.get(quarter, False)) for quarter in train_transition_quarters],
        "forecast_quarters": list(holdout_quarters),
        "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
        "forecast_lower": 0.0,
        "forecast_upper": 1.0,
        "support_count": int(suppression_support_count),
        "support_fraction": float(suppression_support_count) / max(len(dataset.train_rows), 1),
        "model_kind": "art_stock_repair_suppression_carry",
        "suppression_share_carry": float(suppression_share_carry),
    }

    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        holdout_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": holdout_sim["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_art_delta_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    d_to_a_min_count: int = 4,
    d_to_a_min_fraction: float = 0.35,
    d_to_a_use_blend: bool = True,
    art_delta_model_weight: float = 1.0,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_rows[1:]}
    transition_diagnostics: dict[str, Any] = {}
    strict_fit_results: dict[str, dict[str, Any]] = {}

    for transition_name in TRANSITION_NAMES:
        fit_result = _fit_strict_support_transition_model(
            dataset.train_transition_rows,
            holdout_quarters,
            transition_name,
            controls_transition,
            spec.transition_controls.get(transition_name, tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
        )
        strict_fit_results[transition_name] = fit_result
        forecast = dict(fit_result["forecast_map"])
        for quarter, value in dict(fit_result["train_fitted_map"]).items():
            train_hazard_map[quarter][transition_name] = float(value)
        transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = float(value)

    d_to_a_diag = dict(strict_fit_results["D_to_A"]["diagnostics"])
    d_to_a_mode = _care_support_mode(
        support_count=int(d_to_a_diag.get("support_count") or 0),
        total_count=len(dataset.train_transition_rows),
        min_count=d_to_a_min_count,
        min_fraction=d_to_a_min_fraction,
    )
    d_to_a_blend_weight = 0.0
    if d_to_a_use_blend and d_to_a_mode != "strict_support":
        d_to_a_blend_weight = _d_to_a_blend_weight(
            d_to_a_diag,
            total_count=len(dataset.train_transition_rows),
            min_count=d_to_a_min_count,
            min_fraction=d_to_a_min_fraction,
        )
        if d_to_a_blend_weight > 0.0:
            d_to_a_mode = "blend"

    art_delta_result = _fit_supported_delta_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
        recent_blend_weight=art_delta_model_weight,
    )
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )

    train_target_rows = list(dataset.train_rows[1:])
    train_target_quarters = [str(row["quarter"]) for row in train_target_rows]
    train_delta_targets = {
        quarter: float(art_delta_result["train_fitted_map"].get(quarter, 0.0))
        for quarter in train_target_quarters
    }
    train_sim = _simulate_art_delta_repair(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_target_rows,
        train_hazard_map,
        train_delta_targets,
        d_to_a_mode=d_to_a_mode,
        d_to_a_blend_weight=d_to_a_blend_weight,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(train_target_quarters, train_sim["trajectory_rows"]):
        train_hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        train_hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    holdout_sim = _simulate_art_delta_repair(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        hazard_map,
        dict(art_delta_result["forecast_map"]),
        d_to_a_mode=d_to_a_mode,
        d_to_a_blend_weight=d_to_a_blend_weight,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(holdout_quarters, holdout_sim["trajectory_rows"]):
        hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    art_delta_diag = dict(art_delta_result["diagnostics"])
    transition_diagnostics["D_to_A"] = {
        **d_to_a_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["D_to_A"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "model_kind": "art_delta_repair",
        "selection_mode": str(d_to_a_mode),
        "selection_thresholds": {"min_count": int(d_to_a_min_count), "min_fraction": float(d_to_a_min_fraction)},
        "blend_weight": float(d_to_a_blend_weight),
        "delta_diagnostics": art_delta_diag,
    }
    base_a_to_v_diag = dict(transition_diagnostics["A_to_V"])
    transition_diagnostics["A_to_V"] = {
        **base_a_to_v_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["A_to_V"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
        "model_kind": "art_delta_repair_suppression_carry",
        "selection_mode": "repair",
        "forced_mode": "repair",
        "suppression_share_carry": float(suppression_share_carry),
        "carry_support_count": int(suppression_support_count),
    }

    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        holdout_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": holdout_sim["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_direct_observation_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    diagnosed_weight: float = 1.0,
    art_weight: float = 1.0,
    flow_weight: float = 1.0,
    diagnosed_series_model: str = "level",
    art_series_model: str = "level",
    flow_series_model: str = "level",
    diagnosed_recent_blend_weight: float = 1.0,
    art_recent_blend_weight: float = 1.0,
    flow_recent_blend_weight: float = 1.0,
    suppression_carry_weight: float = 1.0,
    suppression_fallback_mode: str = "level_carry",
    use_joint_consistency: bool = False,
    use_bias_correction: bool = False,
    diagnosed_bias_weight: float = 1.0,
    art_bias_weight: float = 1.0,
    flow_bias_weight: float = 1.0,
    flow_consistency_weight: float = 0.0,
    use_diagnosed_crossfit_calibration: bool = False,
    diagnosed_crossfit_weight: float = 1.0,
    diagnosed_crossfit_min_points: int = 5,
    diagnosed_crossfit_recent_pool: int = 12,
    diagnosed_crossfit_min_train_years: int = 3,
) -> dict[str, Any]:
    del dynamic_cfg
    del observation_cfg
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    diagnosed_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="diagnosed_plhiv",
        model_kind=diagnosed_series_model,
        recent_blend_weight=diagnosed_recent_blend_weight,
    )
    art_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
        model_kind=art_series_model,
        recent_blend_weight=art_recent_blend_weight,
    )
    flow_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="new_diagnosed_cases_period",
        model_kind=flow_series_model,
        recent_blend_weight=flow_recent_blend_weight,
    )
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )
    previous_diagnosed, _ = _last_supported_level(dataset.train_rows, metric_name="diagnosed_plhiv")
    previous_art, _ = _last_supported_level(dataset.train_rows, metric_name="alive_on_art")
    previous_flow, _ = _last_supported_level(dataset.train_rows, metric_name="new_diagnosed_cases_period")
    previous_suppressed, _ = _last_supported_level(dataset.train_rows, metric_name="virally_suppressed")
    previous_suppressed_any, suppression_level_count = _last_available_level(dataset.train_rows, metric_name="virally_suppressed")
    diagnosed_bias = float(np.clip(_mean_supported_residual_from_diagnostics(dict(diagnosed_result["diagnostics"])) * float(diagnosed_bias_weight), -float("inf"), float("inf")))
    art_bias = float(np.clip(_mean_supported_residual_from_diagnostics(dict(art_result["diagnostics"])) * float(art_bias_weight), -float("inf"), float("inf")))
    flow_bias = float(np.clip(_mean_supported_residual_from_diagnostics(dict(flow_result["diagnostics"])) * float(flow_bias_weight), -float("inf"), float("inf")))
    diagnosed_crossfit_correction = 0.0
    diagnosed_crossfit_diagnostics: dict[str, Any] = {
        "status": "disabled",
        "metric_name": "diagnosed_plhiv",
        "residual_count": 0,
        "correction": 0.0,
    }
    if use_diagnosed_crossfit_calibration:
        diagnosed_crossfit_correction, diagnosed_crossfit_diagnostics = _crossfit_supported_metric_correction(
            dataset.train_rows,
            spec,
            metric_name="diagnosed_plhiv",
            diagnosed_weight=diagnosed_weight,
            art_weight=art_weight,
            flow_weight=flow_weight,
            diagnosed_series_model=diagnosed_series_model,
            art_series_model=art_series_model,
            flow_series_model=flow_series_model,
            diagnosed_recent_blend_weight=diagnosed_recent_blend_weight,
            art_recent_blend_weight=art_recent_blend_weight,
            flow_recent_blend_weight=flow_recent_blend_weight,
            suppression_carry_weight=suppression_carry_weight,
            suppression_fallback_mode=suppression_fallback_mode,
            use_joint_consistency=use_joint_consistency,
            use_bias_correction=use_bias_correction,
            diagnosed_bias_weight=diagnosed_bias_weight,
            art_bias_weight=art_bias_weight,
            flow_bias_weight=flow_bias_weight,
            flow_consistency_weight=flow_consistency_weight,
            min_train_years=diagnosed_crossfit_min_train_years,
            min_points=diagnosed_crossfit_min_points,
            recent_pool=diagnosed_crossfit_recent_pool,
        )
    reporting_lower, reporting_upper, reporting_support_count = _supported_stock_flow_residual_bounds(dataset.train_rows)
    reporting_center, reporting_center_count = _supported_stock_flow_residual_center(dataset.train_rows)
    _, flow_level_upper = _supported_metric_level_bounds(dataset.train_rows, metric_name="new_diagnosed_cases_period")
    if suppression_support_count <= 0 and previous_suppressed <= 0.0 and previous_suppressed_any > 0.0:
        previous_suppressed = float(previous_suppressed_any)
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    hazard_map: dict[str, dict[str, float]] = {}
    train_transition_quarters = [str(row["quarter"]) for row in dataset.train_transition_rows]
    strict_d_to_a = [float(row["hazards"]["D_to_A"]) for row in dataset.train_transition_rows]
    strict_a_to_v = [float(row["hazards"]["A_to_V"]) for row in dataset.train_transition_rows]
    for quarter in holdout_quarters:
        diagnosed_target = float(diagnosed_result["forecast_map"].get(quarter, previous_diagnosed))
        art_target = float(art_result["forecast_map"].get(quarter, previous_art))
        flow_target = float(flow_result["forecast_map"].get(quarter, previous_flow))
        if use_bias_correction:
            diagnosed_target = float(max(diagnosed_target + diagnosed_bias, 0.0))
            art_target = float(max(art_target + art_bias, 0.0))
            flow_target = float(max(flow_target + flow_bias, 0.0))
        if use_diagnosed_crossfit_calibration:
            diagnosed_target = float(max(diagnosed_target - (float(diagnosed_crossfit_weight) * float(diagnosed_crossfit_correction)), 0.0))
        diagnosed_value = float(np.clip((float(diagnosed_weight) * diagnosed_target) + ((1.0 - float(diagnosed_weight)) * previous_diagnosed), 0.0, float("inf")))
        art_value = float(np.clip((float(art_weight) * art_target) + ((1.0 - float(art_weight)) * previous_art), 0.0, diagnosed_value))
        flow_value = float(np.clip((float(flow_weight) * flow_target) + ((1.0 - float(flow_weight)) * previous_flow), 0.0, float("inf")))
        if use_joint_consistency and reporting_support_count > 0:
            implied_reporting = float((diagnosed_value - previous_diagnosed) - flow_value)
            bounded_reporting = float(np.clip(implied_reporting, reporting_lower, reporting_upper))
            diagnosed_value = float(max(previous_diagnosed + flow_value + bounded_reporting, 0.0))
            art_value = float(min(max(art_value, 0.0), diagnosed_value))
        if float(flow_consistency_weight) > 0.0 and reporting_center_count > 0:
            implied_flow = float(max((diagnosed_value - previous_diagnosed) - reporting_center, 0.0))
            flow_value = float(
                np.clip(
                    (float(flow_consistency_weight) * implied_flow) + ((1.0 - float(flow_consistency_weight)) * flow_value),
                    0.0,
                    flow_level_upper,
                )
            )
            if use_joint_consistency and reporting_support_count > 0:
                implied_reporting = float((diagnosed_value - previous_diagnosed) - flow_value)
                bounded_reporting = float(np.clip(implied_reporting, reporting_lower, reporting_upper))
                diagnosed_value = float(max(previous_diagnosed + flow_value + bounded_reporting, 0.0))
                art_value = float(min(max(art_value, 0.0), diagnosed_value))
        suppression_weight = float(np.clip(float(suppression_carry_weight), 0.0, 1.5))
        if suppression_support_count > 0:
            carried_suppressed = float(np.clip(float(suppression_share_carry) * art_value, 0.0, art_value))
            suppressed_value = float(
                np.clip(
                    (suppression_weight * carried_suppressed) + ((1.0 - suppression_weight) * previous_suppressed),
                    0.0,
                    art_value,
                )
            )
        else:
            if str(suppression_fallback_mode) == "unclaimed":
                carried_suppressed = 0.0
                suppressed_value = None
            else:
                carried_suppressed = float(np.clip(previous_suppressed, 0.0, art_value))
                suppressed_value = float(carried_suppressed)
        art_delta = float(art_value - previous_art)
        eligible_diagnosed = max(previous_diagnosed, dataset.eps)
        available_unsuppressed = max(previous_art - (float(suppression_share_carry) * previous_art), dataset.eps)
        d_to_a_hazard = float(np.clip(max(art_delta, 0.0) / eligible_diagnosed, 0.0, 1.0))
        if suppression_support_count > 0:
            a_to_v_hazard = float(np.clip(max(suppressed_value - (float(suppression_share_carry) * previous_art), 0.0) / available_unsuppressed, 0.0, 1.0))
        else:
            a_to_v_hazard = 0.0
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed_value,
                "alive_on_art": art_value,
                "new_diagnosed_cases_period": flow_value,
                "tested_for_viral_load": None,
                "virally_suppressed": float(suppressed_value) if suppressed_value is not None else None,
            }
        )
        trajectory_rows.append(
            {
                "quarter": quarter,
                "state_values": {
                    "U": 0.0,
                    "D": 0.0,
                    "A": max(art_value - float(suppressed_value or 0.0), 0.0),
                    "V": float(suppressed_value or 0.0),
                    "L": 0.0,
                },
                "hazards": {
                    "U_to_D": 0.0,
                    "D_to_A": d_to_a_hazard,
                    "A_to_V": a_to_v_hazard,
                    "A_to_L": 0.0,
                    "L_to_A": 0.0,
                },
            }
        )
        hazard_map[quarter] = dict(trajectory_rows[-1]["hazards"])
        previous_diagnosed = diagnosed_value
        previous_art = art_value
        previous_flow = flow_value
        previous_suppressed = float(suppressed_value or 0.0)

    transition_diagnostics = {
        "D_to_A": {
            "transition_name": "D_to_A",
            "train_quarters": list(train_transition_quarters),
            "train_observed": list(strict_d_to_a),
            "train_fitted": list(strict_d_to_a),
            "train_supported": [bool(dict(row.get("support_flags") or {}).get("D_to_A")) for row in dataset.train_transition_rows],
            "forecast_quarters": list(holdout_quarters),
            "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
            "support_count": int(sum(bool(dict(row.get("support_flags") or {}).get("D_to_A")) for row in dataset.train_transition_rows)),
            "support_fraction": float(sum(bool(dict(row.get("support_flags") or {}).get("D_to_A")) for row in dataset.train_transition_rows)) / max(len(dataset.train_transition_rows), 1),
            "model_kind": "direct_observation_repair_diagnosed_art_levels",
            "diagnosed_level_diagnostics": dict(diagnosed_result["diagnostics"]),
            "art_level_diagnostics": dict(art_result["diagnostics"]),
            "weights": {
                "diagnosed_weight": float(diagnosed_weight),
                "art_weight": float(art_weight),
                "diagnosed_series_model": str(diagnosed_series_model),
                "art_series_model": str(art_series_model),
                "diagnosed_recent_blend_weight": float(diagnosed_recent_blend_weight),
                "art_recent_blend_weight": float(art_recent_blend_weight),
                "use_joint_consistency": bool(use_joint_consistency),
                "use_bias_correction": bool(use_bias_correction),
                "diagnosed_bias": float(diagnosed_bias),
                "art_bias": float(art_bias),
                "flow_bias": float(flow_bias),
                "use_diagnosed_crossfit_calibration": bool(use_diagnosed_crossfit_calibration),
                "diagnosed_crossfit_correction": float(diagnosed_crossfit_correction),
                "diagnosed_crossfit_weight": float(diagnosed_crossfit_weight),
                "diagnosed_crossfit_diagnostics": dict(diagnosed_crossfit_diagnostics),
                "flow_consistency_weight": float(flow_consistency_weight),
                "reporting_lower": float(reporting_lower),
                "reporting_center": float(reporting_center),
                "reporting_upper": float(reporting_upper),
                "reporting_support_count": int(reporting_support_count),
                "reporting_center_count": int(reporting_center_count),
            },
        },
        "A_to_V": {
            "transition_name": "A_to_V",
            "train_quarters": list(train_transition_quarters),
            "train_observed": list(strict_a_to_v),
            "train_fitted": list(strict_a_to_v),
            "train_supported": [bool(dict(row.get("support_flags") or {}).get("A_to_V")) for row in dataset.train_transition_rows],
            "forecast_quarters": list(holdout_quarters),
            "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
            "support_count": int(suppression_support_count),
            "support_fraction": float(suppression_support_count) / max(len(dataset.train_transition_rows), 1),
            "model_kind": (
                "direct_observation_repair_suppression_carry"
                if suppression_support_count > 0
                else (
                    "direct_observation_repair_suppression_unclaimed"
                    if str(suppression_fallback_mode) == "unclaimed"
                    else "direct_observation_repair_suppression_level_carry"
                )
            ),
            "suppression_share_carry": float(suppression_share_carry),
            "suppression_carry_weight": float(suppression_carry_weight),
            "suppression_fallback_mode": str(suppression_fallback_mode),
            "suppression_level_count": int(suppression_level_count),
            "suppression_level_fallback": float(previous_suppressed_any),
            "flow_level_diagnostics": dict(flow_result["diagnostics"]),
            "flow_weight": float(flow_weight),
            "flow_series_model": str(flow_series_model),
            "flow_recent_blend_weight": float(flow_recent_blend_weight),
            "flow_bias": float(flow_bias),
            "flow_consistency_weight": float(flow_consistency_weight),
            "reporting_center": float(reporting_center),
        },
    }
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": trajectory_rows,
        "hazard_map": hazard_map,
        "mae": normalized_mae(prediction_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(prediction_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_direct_observation_shared_shock_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    diagnosed_weight: float = 1.0,
    art_weight: float = 1.0,
    flow_weight: float = 1.0,
    diagnosed_series_model: str = "level",
    art_series_model: str = "level",
    flow_series_model: str = "level",
    diagnosed_recent_blend_weight: float = 1.0,
    art_recent_blend_weight: float = 1.0,
    flow_recent_blend_weight: float = 1.0,
    suppression_carry_weight: float = 1.0,
    suppression_fallback_mode: str = "level_carry",
    use_joint_consistency: bool = False,
    use_bias_correction: bool = False,
    diagnosed_bias_weight: float = 1.0,
    art_bias_weight: float = 1.0,
    flow_bias_weight: float = 1.0,
    flow_consistency_weight: float = 0.0,
    shock_min_shared_metrics: int = 2,
    shock_threshold_quantile: float = 0.7,
    shock_threshold_scale: float = 1.0,
    shock_forecast_blend_weight: float = 0.75,
    shock_scale: float = 1.0,
) -> dict[str, Any]:
    del dynamic_cfg
    del observation_cfg
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    train_quarters = [str(row["quarter"]) for row in dataset.train_rows]
    diagnosed_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="diagnosed_plhiv",
        model_kind=diagnosed_series_model,
        recent_blend_weight=diagnosed_recent_blend_weight,
    )
    art_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
        model_kind=art_series_model,
        recent_blend_weight=art_recent_blend_weight,
    )
    flow_result = _fit_supported_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="new_diagnosed_cases_period",
        model_kind=flow_series_model,
        recent_blend_weight=flow_recent_blend_weight,
    )
    base = _fit_direct_observation_repair_candidate(
        dataset,
        DynamicControlConfig(ridge_penalty=0.0, rho_clip=0.0, trend_scale=1.0),
        ObservationConfig(calibration_ridge=0.0, share_ridge_penalty=0.0, share_rho_clip=0.0, share_trend_scale=1.0),
        spec,
        diagnosed_weight=diagnosed_weight,
        art_weight=art_weight,
        flow_weight=flow_weight,
        diagnosed_series_model=diagnosed_series_model,
        art_series_model=art_series_model,
        flow_series_model=flow_series_model,
        diagnosed_recent_blend_weight=diagnosed_recent_blend_weight,
        art_recent_blend_weight=art_recent_blend_weight,
        flow_recent_blend_weight=flow_recent_blend_weight,
        suppression_carry_weight=suppression_carry_weight,
        suppression_fallback_mode=suppression_fallback_mode,
        use_joint_consistency=use_joint_consistency,
        use_bias_correction=use_bias_correction,
        diagnosed_bias_weight=diagnosed_bias_weight,
        art_bias_weight=art_bias_weight,
        flow_bias_weight=flow_bias_weight,
        flow_consistency_weight=flow_consistency_weight,
    )

    metric_results = {
        "diagnosed_plhiv": diagnosed_result,
        "alive_on_art": art_result,
        "new_diagnosed_cases_period": flow_result,
    }
    residuals_by_metric: dict[str, dict[str, float]] = {}
    standardized_by_metric: dict[str, dict[str, float]] = {}
    for metric_name, result in metric_results.items():
        diagnostics = dict(result["diagnostics"])
        quarter_residuals: dict[str, float] = {}
        for quarter, observed, fitted, supported in zip(
            list(diagnostics.get("train_quarters") or []),
            list(diagnostics.get("train_observed") or []),
            list(diagnostics.get("train_fitted") or []),
            list(diagnostics.get("train_supported") or []),
            strict=False,
        ):
            if not supported or observed is None or fitted is None:
                continue
            quarter_residuals[str(quarter)] = float(observed) - float(fitted)
        residuals_by_metric[metric_name] = dict(quarter_residuals)
        if not quarter_residuals:
            standardized_by_metric[metric_name] = {}
            continue
        residual_values = np.asarray(list(quarter_residuals.values()), dtype=np.float64)
        mad = float(np.median(np.abs(residual_values - np.median(residual_values))))
        scale = mad * 1.4826
        if scale <= 1e-9:
            scale = float(np.std(residual_values))
        if scale <= 1e-9:
            scale = 1.0
        standardized_by_metric[metric_name] = {
            quarter: float(value) / float(scale) for quarter, value in quarter_residuals.items()
        }

    shared_train_raw: dict[str, float] = {}
    shared_train_counts: dict[str, int] = {}
    for quarter in train_quarters:
        values = [
            float(metric_map[quarter])
            for metric_map in standardized_by_metric.values()
            if quarter in metric_map
        ]
        shared_train_counts[quarter] = int(len(values))
        if len(values) >= int(max(shock_min_shared_metrics, 1)):
            shared_train_raw[quarter] = float(np.mean(np.asarray(values, dtype=np.float64)))
        else:
            shared_train_raw[quarter] = 0.0
    nonzero_train = [abs(value) for value in shared_train_raw.values() if abs(value) > 1e-9]
    if nonzero_train:
        threshold = float(np.quantile(np.asarray(nonzero_train, dtype=np.float64), np.clip(float(shock_threshold_quantile), 0.0, 1.0)))
        threshold *= float(max(shock_threshold_scale, 0.0))
    else:
        threshold = 0.0

    def sparse_shock(value: float) -> float:
        magnitude = abs(float(value))
        if magnitude <= float(threshold):
            return 0.0
        return float(math.copysign(magnitude - float(threshold), float(value)))

    shared_train_sparse = {quarter: sparse_shock(value) for quarter, value in shared_train_raw.items()}
    train_shock_values = [float(shared_train_sparse[quarter]) for quarter in train_quarters]
    shared_forecast_raw = _fit_scalar_ar_trend(
        train_quarters,
        train_shock_values,
        holdout_quarters,
        ridge_penalty=0.1,
        rho_clip=0.9,
        trend_scale=1.0,
    )
    blend = float(np.clip(float(shock_forecast_blend_weight), 0.0, 1.0))
    last_train_shock = float(train_shock_values[-1]) if train_shock_values else 0.0
    shared_forecast = {
        quarter: sparse_shock((blend * float(shared_forecast_raw[quarter])) + ((1.0 - blend) * last_train_shock))
        for quarter in holdout_quarters
    }

    loadings: dict[str, float] = {}
    for metric_name, quarter_residuals in residuals_by_metric.items():
        pairs = [
            (float(shared_train_sparse[quarter]), float(residual))
            for quarter, residual in quarter_residuals.items()
            if abs(float(shared_train_sparse.get(quarter, 0.0))) > 1e-9
        ]
        if len(pairs) < 2:
            loadings[metric_name] = 0.0
            continue
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        denom = float(np.dot(x, x))
        if denom <= 1e-9:
            loadings[metric_name] = 0.0
            continue
        raw_loading = float(np.dot(x, y) / denom)
        cap = float(np.quantile(np.abs(y), 0.9)) if y.size else 0.0
        if cap <= 1e-9:
            cap = float(np.max(np.abs(y))) if y.size else 0.0
        if cap <= 1e-9:
            cap = abs(raw_loading)
        loadings[metric_name] = float(np.clip(raw_loading, -cap, cap))

    base_predictions = {
        str(row["quarter"]): dict(row) for row in list(base.get("prediction_rows") or [])
    }
    previous_diagnosed, _ = _last_supported_level(dataset.train_rows, metric_name="diagnosed_plhiv")
    previous_art, _ = _last_supported_level(dataset.train_rows, metric_name="alive_on_art")
    previous_suppressed, _ = _last_supported_level(dataset.train_rows, metric_name="virally_suppressed")
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )
    prediction_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    hazard_map: dict[str, dict[str, float]] = {}
    for quarter in holdout_quarters:
        base_row = dict(base_predictions.get(quarter) or {})
        shared_shock = float(shared_forecast.get(quarter, 0.0)) * float(shock_scale)
        diagnosed_value = float(max(float(base_row.get("diagnosed_plhiv") or 0.0) + (float(loadings.get("diagnosed_plhiv", 0.0)) * shared_shock), 0.0))
        art_value = float(np.clip(float(base_row.get("alive_on_art") or 0.0) + (float(loadings.get("alive_on_art", 0.0)) * shared_shock), 0.0, diagnosed_value))
        flow_value = float(max(float(base_row.get("new_diagnosed_cases_period") or 0.0) + (float(loadings.get("new_diagnosed_cases_period", 0.0)) * shared_shock), 0.0))
        if suppression_support_count > 0:
            suppressed_value = float(np.clip(float(suppression_share_carry) * art_value, 0.0, art_value))
        else:
            suppressed_value = None if str(suppression_fallback_mode) == "unclaimed" else float(np.clip(previous_suppressed, 0.0, art_value))
        art_delta = float(art_value - previous_art)
        eligible_diagnosed = max(previous_diagnosed, dataset.eps)
        available_unsuppressed = max(previous_art - (float(suppression_share_carry) * previous_art), dataset.eps)
        d_to_a_hazard = float(np.clip(max(art_delta, 0.0) / eligible_diagnosed, 0.0, 1.0))
        if suppression_support_count > 0 and suppressed_value is not None:
            a_to_v_hazard = float(np.clip(max(suppressed_value - (float(suppression_share_carry) * previous_art), 0.0) / available_unsuppressed, 0.0, 1.0))
        else:
            a_to_v_hazard = 0.0
        prediction_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": diagnosed_value,
                "alive_on_art": art_value,
                "new_diagnosed_cases_period": flow_value,
                "tested_for_viral_load": None,
                "virally_suppressed": float(suppressed_value) if suppressed_value is not None else None,
            }
        )
        trajectory_rows.append(
            {
                "quarter": quarter,
                "state_values": {
                    "U": 0.0,
                    "D": 0.0,
                    "A": max(art_value - float(suppressed_value or 0.0), 0.0),
                    "V": float(suppressed_value or 0.0),
                    "L": 0.0,
                },
                "hazards": {
                    "U_to_D": 0.0,
                    "D_to_A": d_to_a_hazard,
                    "A_to_V": a_to_v_hazard,
                    "A_to_L": 0.0,
                    "L_to_A": 0.0,
                },
            }
        )
        hazard_map[quarter] = dict(trajectory_rows[-1]["hazards"])
        previous_diagnosed = diagnosed_value
        previous_art = art_value
        previous_suppressed = float(suppressed_value or 0.0)

    transition_diagnostics = dict(base.get("transition_diagnostics") or {})
    transition_diagnostics["shared_shock"] = {
        "train_quarters": list(train_quarters),
        "train_shock_raw": [float(shared_train_raw[quarter]) for quarter in train_quarters],
        "train_shock_sparse": [float(shared_train_sparse[quarter]) for quarter in train_quarters],
        "train_shared_metric_count": [int(shared_train_counts[quarter]) for quarter in train_quarters],
        "forecast_quarters": list(holdout_quarters),
        "forecast_shock_values": [float(shared_forecast[quarter]) for quarter in holdout_quarters],
        "threshold": float(threshold),
        "shock_scale": float(shock_scale),
        "loadings": {metric_name: float(value) for metric_name, value in loadings.items()},
        "model_kind": "direct_observation_shared_shock_overlay",
        "base_transition_model": str(spec.transition_model),
    }
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": trajectory_rows,
        "hazard_map": hazard_map,
        "mae": normalized_mae(prediction_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(prediction_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_direct_observation_mechanistic_overlay_candidate(
    dataset: Any,
    annual_rows: list[dict[str, Any]],
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    diagnosed_weight: float = 1.0,
    art_weight: float = 1.0,
    flow_weight: float = 1.0,
    diagnosed_series_model: str = "level",
    art_series_model: str = "level",
    flow_series_model: str = "level",
    diagnosed_recent_blend_weight: float = 1.0,
    art_recent_blend_weight: float = 1.0,
    flow_recent_blend_weight: float = 1.0,
) -> dict[str, Any]:
    base = _fit_direct_observation_repair_candidate(
        dataset,
        dynamic_cfg,
        observation_cfg,
        spec,
        diagnosed_weight=diagnosed_weight,
        art_weight=art_weight,
        flow_weight=flow_weight,
        diagnosed_series_model=diagnosed_series_model,
        art_series_model=art_series_model,
        flow_series_model=flow_series_model,
        diagnosed_recent_blend_weight=diagnosed_recent_blend_weight,
        art_recent_blend_weight=art_recent_blend_weight,
        flow_recent_blend_weight=flow_recent_blend_weight,
        suppression_carry_weight=0.0,
    )
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    holdout_years = sorted({quarter_year(quarter) for quarter in holdout_quarters})
    yearly_rows = _yearly_feature_rows_filtered(dataset.train_rows + dataset.holdout_rows, annual_rows)
    annual_cfg = AnnualIncidenceConfig(ridge_penalty=0.1, inflow_scale_clip=4.0)
    annual_train_rows = [row for row in yearly_rows if int(row["year"]) <= max(sorted({quarter_year(str(row["quarter"])) for row in dataset.train_rows}), default=0)]
    annual_forecast = _fit_annual_incidence_model(annual_train_rows, holdout_years, annual_cfg) if annual_train_rows else {}
    holdout_predictions = {str(row["quarter"]): dict(row) for row in base["prediction_rows"]}
    by_year: dict[int, list[str]] = {}
    for quarter in holdout_quarters:
        by_year.setdefault(quarter_year(quarter), []).append(quarter)
    quarter_inflows: dict[str, float] = {}
    for year, quarters in by_year.items():
        annual_total = float(annual_forecast.get(year, 0.0))
        weights = np.asarray(
            [max(float(holdout_predictions.get(quarter, {}).get("new_diagnosed_cases_period") or 0.0), 0.0) for quarter in quarters],
            dtype=np.float64,
        )
        if float(weights.sum()) <= 1e-9:
            weights = np.full((len(quarters),), 1.0 / max(len(quarters), 1), dtype=np.float64)
        else:
            weights = weights / float(weights.sum())
        for quarter, weight in zip(quarters, weights, strict=False):
            quarter_inflows[quarter] = float(annual_total * float(weight))
    train_last_state = dict(dataset.train_state_rows[-1]["state_values"]) if dataset.train_state_rows else {name: 0.0 for name in STATE_NAMES}
    previous_diagnosed, _ = _last_supported_level(dataset.train_rows, metric_name="diagnosed_plhiv")
    previous_art, _ = _last_supported_level(dataset.train_rows, metric_name="alive_on_art")
    flow_sequence = [max(float(holdout_predictions.get(quarter, {}).get("new_diagnosed_cases_period") or 0.0), 0.0) for quarter in holdout_quarters]
    inflow_sequence = [max(float(quarter_inflows.get(quarter, 0.0)), 0.0) for quarter in holdout_quarters]
    deficit = 0.0
    running = 0.0
    for flow_value, inflow_value in zip(flow_sequence, inflow_sequence, strict=False):
        running += float(flow_value) - float(inflow_value)
        deficit = max(deficit, running)
    previous_u = float(max(float(train_last_state.get("U", 0.0)), deficit))
    trajectory_rows: list[dict[str, Any]] = []
    hazard_map: dict[str, dict[str, float]] = {}
    transition_diagnostics = dict(base.get("transition_diagnostics") or {})
    for quarter in holdout_quarters:
        prediction = dict(holdout_predictions.get(quarter) or {})
        diagnosed_total = float(max(prediction.get("diagnosed_plhiv") or 0.0, 0.0))
        art_total = float(np.clip(float(prediction.get("alive_on_art") or 0.0), 0.0, diagnosed_total))
        diagnosis_flow = float(max(prediction.get("new_diagnosed_cases_period") or 0.0, 0.0))
        inflow = float(max(quarter_inflows.get(quarter, 0.0), 0.0))
        current_u = float(max(previous_u + inflow - diagnosis_flow, 0.0))
        untreated_diagnosed = float(max(diagnosed_total - art_total, 0.0))
        eligible_diagnosed = max(previous_diagnosed - previous_art, dataset.eps)
        d_to_a_flow = float(max(art_total - previous_art, 0.0))
        u_to_d_hazard = float(np.clip(diagnosis_flow / max(previous_u + inflow, dataset.eps), 0.0, 1.0))
        d_to_a_hazard = float(np.clip(d_to_a_flow / eligible_diagnosed, 0.0, 1.0))
        hazard_map[quarter] = {
            "U_to_D": u_to_d_hazard,
            "D_to_A": d_to_a_hazard,
            "A_to_V": 0.0,
            "A_to_L": 0.0,
            "L_to_A": 0.0,
        }
        trajectory_rows.append(
            {
                "quarter": quarter,
                "state_values": {"U": current_u, "D": untreated_diagnosed, "A": art_total, "V": 0.0, "L": 0.0},
                "hazards": dict(hazard_map[quarter]),
                "inflow": inflow,
            }
        )
        previous_u = current_u
        previous_diagnosed = diagnosed_total
        previous_art = art_total
    transition_diagnostics["U_to_D_overlay"] = {
        "transition_name": "U_to_D",
        "forecast_quarters": list(holdout_quarters),
        "forecast_hazards": [float(hazard_map[quarter]["U_to_D"]) for quarter in holdout_quarters],
        "model_kind": "minimal_mechanistic_overlay",
        "annual_infections": dict(annual_forecast),
        "quarter_inflows": dict(quarter_inflows),
    }
    transition_diagnostics["D_to_A_overlay"] = {
        "transition_name": "D_to_A",
        "forecast_quarters": list(holdout_quarters),
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "model_kind": "minimal_mechanistic_overlay",
    }
    prediction_rows = [
        {
            **dict(row),
            "virally_suppressed": None,
        }
        for row in base["prediction_rows"]
    ]
    return {
        "prediction_rows": prediction_rows,
        "trajectory_rows": trajectory_rows,
        "hazard_map": hazard_map,
        "annual_infections": annual_forecast,
        "mae": normalized_mae(prediction_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(prediction_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_hybrid_care_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    d_to_a_min_count: int = 4,
    d_to_a_min_fraction: float = 0.35,
    a_to_v_min_count: int = 4,
    a_to_v_min_fraction: float = 0.35,
    a_to_v_force_mode: str | None = None,
    d_to_a_use_blend: bool = False,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_rows[1:]}
    transition_diagnostics: dict[str, Any] = {}
    strict_fit_results: dict[str, dict[str, Any]] = {}

    for transition_name in TRANSITION_NAMES:
        fit_result = _fit_strict_support_transition_model(
            dataset.train_transition_rows,
            holdout_quarters,
            transition_name,
            controls_transition,
            spec.transition_controls.get(transition_name, tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
        )
        strict_fit_results[transition_name] = fit_result
        forecast = dict(fit_result["forecast_map"])
        for quarter, value in dict(fit_result["train_fitted_map"]).items():
            train_hazard_map[quarter][transition_name] = float(value)
        transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = float(value)

    d_to_a_diag = dict(strict_fit_results["D_to_A"]["diagnostics"])
    a_to_v_diag = dict(strict_fit_results["A_to_V"]["diagnostics"])
    d_to_a_mode = _care_support_mode(
        support_count=int(d_to_a_diag.get("support_count") or 0),
        total_count=len(dataset.train_transition_rows),
        min_count=d_to_a_min_count,
        min_fraction=d_to_a_min_fraction,
    )
    d_to_a_blend_weight = 0.0
    if d_to_a_use_blend and d_to_a_mode != "strict_support":
        d_to_a_blend_weight = _d_to_a_blend_weight(
            d_to_a_diag,
            total_count=len(dataset.train_transition_rows),
            min_count=d_to_a_min_count,
            min_fraction=d_to_a_min_fraction,
        )
        if d_to_a_blend_weight > 0.0:
            d_to_a_mode = "blend"
    a_to_v_mode = _care_support_mode(
        support_count=int(a_to_v_diag.get("support_count") or 0),
        total_count=len(dataset.train_transition_rows),
        min_count=a_to_v_min_count,
        min_fraction=a_to_v_min_fraction,
    )
    if a_to_v_force_mode is not None:
        a_to_v_mode = str(a_to_v_force_mode)

    art_stock_result = _fit_supported_level_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
    )
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )

    train_target_rows = list(dataset.train_rows[1:])
    train_target_quarters = [str(row["quarter"]) for row in train_target_rows]
    train_art_targets = {
        quarter: float(art_stock_result["train_fitted_map"].get(quarter, 0.0))
        for quarter in train_target_quarters
    }
    train_sim = _simulate_hybrid_care_repair(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_target_rows,
        train_hazard_map,
        train_art_targets,
        d_to_a_mode=d_to_a_mode,
        a_to_v_mode=a_to_v_mode,
        d_to_a_blend_weight=d_to_a_blend_weight,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(train_target_quarters, train_sim["trajectory_rows"]):
        train_hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        train_hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    holdout_sim = _simulate_hybrid_care_repair(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        hazard_map,
        dict(art_stock_result["forecast_map"]),
        d_to_a_mode=d_to_a_mode,
        a_to_v_mode=a_to_v_mode,
        d_to_a_blend_weight=d_to_a_blend_weight,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(holdout_quarters, holdout_sim["trajectory_rows"]):
        hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    art_stock_diag = dict(art_stock_result["diagnostics"])
    transition_diagnostics["D_to_A"] = {
        **d_to_a_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["D_to_A"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "model_kind": "hybrid_care_repair_strict" if d_to_a_mode == "strict_support" else "hybrid_care_repair_alive_on_art",
        "selection_mode": str(d_to_a_mode),
        "selection_thresholds": {"min_count": int(d_to_a_min_count), "min_fraction": float(d_to_a_min_fraction)},
        "blend_weight": float(d_to_a_blend_weight),
        "level_diagnostics": art_stock_diag,
    }
    transition_diagnostics["A_to_V"] = {
        **a_to_v_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["A_to_V"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
        "model_kind": "hybrid_care_repair_strict" if a_to_v_mode == "strict_support" else "hybrid_care_repair_suppression_carry",
        "selection_mode": str(a_to_v_mode),
        "selection_thresholds": {"min_count": int(a_to_v_min_count), "min_fraction": float(a_to_v_min_fraction)},
        "forced_mode": str(a_to_v_force_mode) if a_to_v_force_mode is not None else None,
        "suppression_share_carry": float(suppression_share_carry),
        "carry_support_count": int(suppression_support_count),
    }

    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        holdout_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": holdout_sim["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _fit_two_regime_d_to_a_repair_candidate(
    dataset: Any,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    spec: ExperimentSpec,
    *,
    min_supported: int = 3,
    max_inner_gap: int = 2,
    max_tail_gap: int = 1,
) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
    controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_rows[1:]}
    transition_diagnostics: dict[str, Any] = {}

    for transition_name in ("U_to_D", "D_to_A", "A_to_L", "L_to_A", "A_to_V"):
        fit_result = _fit_strict_support_transition_model(
            dataset.train_transition_rows,
            holdout_quarters,
            transition_name,
            controls_transition,
            spec.transition_controls.get(transition_name, tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get(transition_name, 1.0)),
        )
        forecast = dict(fit_result["forecast_map"])
        for quarter, value in dict(fit_result["train_fitted_map"]).items():
            train_hazard_map[quarter][transition_name] = float(value)
        transition_diagnostics[transition_name] = dict(fit_result["diagnostics"])
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = float(value)

    recent_block_indices = _recent_supported_block_indices(
        dataset.train_transition_rows,
        "D_to_A",
        min_supported=min_supported,
        max_inner_gap=max_inner_gap,
        max_tail_gap=max_tail_gap,
    )
    d_to_a_forecast_mode = "repair"
    d_to_a_fit_result: dict[str, Any] | None = None
    strict_regime_start_quarter: str | None = None
    if recent_block_indices:
        regime_rows = list(dataset.train_transition_rows[recent_block_indices[0] :])
        regime_controls = _controls_for_transition_rows(dataset.train_rows, regime_rows, controls_train_full)
        d_to_a_fit_result = _fit_strict_support_transition_model(
            regime_rows,
            holdout_quarters,
            "D_to_A",
            regime_controls,
            spec.transition_controls.get("D_to_A", tuple()),
            cfg=dynamic_cfg,
            ridge_multiplier=float(spec.transition_ridge_multipliers.get("D_to_A", 1.0)),
        )
        strict_regime_start_quarter = str(regime_rows[0]["quarter"])
        d_to_a_forecast_mode = "strict_support"

    art_stock_result = _fit_supported_level_series(
        dataset.train_rows,
        holdout_quarters,
        metric_name="alive_on_art",
    )
    suppression_share_carry, suppression_support_count = _last_supported_share(
        dataset.train_rows,
        numerator_metric="virally_suppressed",
        denominator_metric="alive_on_art",
    )

    train_target_rows = list(dataset.train_rows[1:])
    train_target_quarters = [str(row["quarter"]) for row in train_target_rows]
    train_art_targets = {
        quarter: float(art_stock_result["train_fitted_map"].get(quarter, 0.0))
        for quarter in train_target_quarters
    }
    train_d_to_a_mode_by_quarter: dict[str, str] = {}
    for quarter in train_target_quarters:
        if strict_regime_start_quarter and quarter_sort_key(quarter) >= quarter_sort_key(strict_regime_start_quarter):
            train_d_to_a_mode_by_quarter[quarter] = "strict_support"
            if d_to_a_fit_result is not None and quarter in d_to_a_fit_result["train_fitted_map"]:
                train_hazard_map[quarter]["D_to_A"] = float(d_to_a_fit_result["train_fitted_map"][quarter])
        else:
            train_d_to_a_mode_by_quarter[quarter] = "repair"

    train_sim = _simulate_two_regime_d_to_a_repair(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_target_rows,
        train_hazard_map,
        train_art_targets,
        d_to_a_mode_by_quarter=train_d_to_a_mode_by_quarter,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(train_target_quarters, train_sim["trajectory_rows"]):
        train_hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        train_hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    holdout_d_to_a_mode_by_quarter = {quarter: d_to_a_forecast_mode for quarter in holdout_quarters}
    if d_to_a_fit_result is not None:
        for quarter in holdout_quarters:
            hazard_map[quarter]["D_to_A"] = float(d_to_a_fit_result["forecast_map"].get(quarter, hazard_map[quarter]["D_to_A"]))
    holdout_sim = _simulate_two_regime_d_to_a_repair(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        hazard_map,
        dict(art_stock_result["forecast_map"]),
        d_to_a_mode_by_quarter=holdout_d_to_a_mode_by_quarter,
        suppression_share_carry=suppression_share_carry,
        eps=dataset.eps,
    )
    for quarter, trajectory in zip(holdout_quarters, holdout_sim["trajectory_rows"]):
        hazard_map[quarter]["D_to_A"] = float(trajectory["hazards"]["D_to_A"])
        hazard_map[quarter]["A_to_V"] = float(trajectory["hazards"]["A_to_V"])

    art_stock_diag = dict(art_stock_result["diagnostics"])
    base_d_to_a_diag = dict(transition_diagnostics["D_to_A"])
    transition_diagnostics["D_to_A"] = {
        **base_d_to_a_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["D_to_A"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["D_to_A"]) for quarter in holdout_quarters],
        "model_kind": "two_regime_d_to_a_repair",
        "selection_mode": str(d_to_a_forecast_mode),
        "strict_regime_start_quarter": strict_regime_start_quarter,
        "recent_supported_block_indices": list(recent_block_indices),
        "recent_block_params": {
            "min_supported": int(min_supported),
            "max_inner_gap": int(max_inner_gap),
            "max_tail_gap": int(max_tail_gap),
        },
        "level_diagnostics": art_stock_diag,
        "strict_regime_diagnostics": dict(d_to_a_fit_result["diagnostics"]) if d_to_a_fit_result is not None else None,
    }
    base_a_to_v_diag = dict(transition_diagnostics["A_to_V"])
    transition_diagnostics["A_to_V"] = {
        **base_a_to_v_diag,
        "train_fitted": [float(train_hazard_map[str(row["quarter"])]["A_to_V"]) for row in dataset.train_transition_rows],
        "forecast_hazards": [float(hazard_map[quarter]["A_to_V"]) for quarter in holdout_quarters],
        "model_kind": "two_regime_d_to_a_repair_suppression_carry",
        "selection_mode": "repair",
        "forced_mode": "repair",
        "suppression_share_carry": float(suppression_share_carry),
        "carry_support_count": int(suppression_support_count),
    }

    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    calibrated_rows = apply_observation_model(
        holdout_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": holdout_sim["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
        "transition_diagnostics": transition_diagnostics,
    }


def _yearly_feature_rows_filtered(
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    from epigraph_ph.phase3.tr_v3_05_autoresearch import _yearly_feature_rows

    return _yearly_feature_rows(observation_rows, annual_rows)


def _fit_05b_experiment_candidate(
    dataset: Any,
    annual_rows: list[dict[str, Any]],
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    annual_cfg: AnnualIncidenceConfig,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    from epigraph_ph.phase3.tr_v3_05_autoresearch import apply_observation_model

    base = _fit_05a_experiment_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, spec)
    if spec.inflow_mode == "none":
        return base
    full_rows = dataset.train_rows + dataset.holdout_rows
    yearly_rows = _yearly_feature_rows_filtered(full_rows, annual_rows)
    train_years = sorted({quarter_year(str(row["quarter"])) for row in dataset.train_rows})
    holdout_years = sorted({quarter_year(str(row["quarter"])) for row in dataset.holdout_rows})
    annual_train_rows = [row for row in yearly_rows if int(row["year"]) in set(train_years)]
    annual_forecast = _fit_annual_incidence_model(annual_train_rows, holdout_years, annual_cfg)
    forecast_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    by_year: dict[int, list[str]] = {}
    for quarter in forecast_quarters:
        by_year.setdefault(quarter_year(quarter), []).append(quarter)
    if spec.inflow_mode == "equal_share":
        quarter_inflows = {
            quarter: float(annual_forecast.get(year, 0.0)) / max(len(quarters), 1)
            for year, quarters in by_year.items()
            for quarter in quarters
        }
    else:
        controls_train_full = _lagged_controls(dataset.train_rows, spec.control_lag)
        controls_transition = _controls_for_transition_rows(dataset.train_rows, dataset.train_transition_rows, controls_train_full)
        inflow_coeffs = _fit_quarterly_inflow_share_model_configurable(
            dataset.train_transition_rows,
            controls_transition,
            annual_rows,
            annual_cfg,
            inflow_controls=spec.inflow_controls,
            use_infectious_pool=spec.use_infectious_pool,
        )
        forecast_controls = {
            control_name: _fit_scalar_ar_trend(
                [str(row["quarter"]) for row in dataset.train_rows],
                controls_train_full[control_name],
                forecast_quarters,
                ridge_penalty=dynamic_cfg.ridge_penalty,
                rho_clip=dynamic_cfg.rho_clip,
                trend_scale=dynamic_cfg.trend_scale,
            )
            for control_name in ("A", "C", "R")
        }
        interim_hazard_map = _apply_leakage_mode(base["hazard_map"], spec.use_leakage)
        interim = simulate_closed_flow(
            dict(dataset.train_state_rows[-1]["state_values"]),
            dataset.holdout_rows,
            interim_hazard_map,
        )
        quarter_controls = {
            quarter: {
                control_name: float(forecast_controls[control_name].get(quarter, 0.0))
                for control_name in ("A", "C", "R")
            }
            for quarter in forecast_quarters
        }
        score_map = (
            _quarterly_inflow_scores_configurable(
                interim["trajectory_rows"],
                quarter_controls,
                inflow_coeffs,
                inflow_controls=spec.inflow_controls,
                use_infectious_pool=spec.use_infectious_pool,
            )
            if inflow_coeffs
            else {quarter: 1.0 for quarter in forecast_quarters}
        )
        quarter_inflows = {}
        for year, quarters in by_year.items():
            total = float(annual_forecast.get(year, 0.0))
            weights = np.asarray([float(score_map.get(quarter, 1.0)) for quarter in quarters], dtype=np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-9:
                weights = np.full((len(quarters),), 1.0 / max(len(quarters), 1), dtype=np.float64)
            else:
                weights = weights / weight_sum
            for quarter, weight in zip(quarters, weights):
                quarter_inflows[quarter] = float(total * float(weight))
    open_sim = simulate_open_inflow(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        _apply_leakage_mode(base["hazard_map"], spec.use_leakage),
        quarter_inflows,
    )
    observation_model = fit_observation_model(
        dataset,
        {
            str(row["quarter"]): {transition: float(row["hazards"][transition]) for transition in TRANSITION_NAMES}
            for row in dataset.train_transition_rows
        },
        observation_cfg,
    )
    calibrated_rows = apply_observation_model(
        open_sim["prediction_rows"],
        observation_model,
        positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows),
    )
    predicted_annual_infections = {
        year: float(sum(quarter_inflows[quarter] for quarter in quarters))
        for year, quarters in by_year.items()
    }
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": open_sim["trajectory_rows"],
        "annual_infections": predicted_annual_infections,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
    }


def _run_quarterly_experiment(
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    spec: ExperimentSpec,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    annual_cfg: AnnualIncidenceConfig | None,
    scoring_tiers: set[str],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
    archive_run_id: str | None = None,
    split_local_dense: bool = False,
    full_dense_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    years = sorted({quarter_year(str(row["quarter"])) for row in observation_rows})
    splits = rolling_origin_year_splits(
        years,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rows: list[dict[str, Any]] = []
    candidate_maes: list[float] = []
    carry_forward_maes: list[float] = []
    candidate_overlay_annual_errors: list[float] = []
    carry_forward_overlay_annual_errors: list[float] = []
    for split in splits:
        split_observation_rows = observation_rows
        if split_local_dense:
            if archive_run_id is None or full_dense_rows is None:
                raise ValueError("split_local_dense requires archive_run_id and full_dense_rows.")
            split_observation_rows = _build_purged_dense_split_rows(
                archive_run_id,
                full_dense_rows=full_dense_rows,
                train_end_year=int(split["train_end_year"]),
                holdout_years=list(split["holdout_years"]),
            )
        dataset = build_quarterly_dataset(split_observation_rows, list(split["holdout_years"]))
        carry_forward = simulate_closed_flow(
            dict(dataset.train_state_rows[-1]["state_values"]),
            dataset.holdout_rows,
            carry_forward_hazards(dataset),
        )
        carry_forward_mae = _filtered_normalized_mae(
            carry_forward["prediction_rows"],
            dataset.holdout_rows,
            dataset.metric_scales,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        )
        carry_forward_smape = _filtered_smape(
            carry_forward["prediction_rows"],
            dataset.holdout_rows,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        )
        if spec.family in {"05a", "repair"}:
            candidate = _fit_05a_experiment_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, spec)
        else:
            assert annual_cfg is not None
            candidate = _fit_05b_experiment_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, annual_cfg, spec)
        candidate_mae = _filtered_normalized_mae(
            candidate["prediction_rows"],
            dataset.holdout_rows,
            dataset.metric_scales,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        )
        candidate_smape = _filtered_smape(
            candidate["prediction_rows"],
            dataset.holdout_rows,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        )
        endpoint_audit = {
            "candidate": _raw_endpoint_audit(
                candidate["prediction_rows"],
                dataset.holdout_rows,
                dataset.metric_scales,
                allowed_tiers=scoring_tiers,
                eps=dataset.eps,
            ),
            "carry_forward": _raw_endpoint_audit(
                carry_forward["prediction_rows"],
                dataset.holdout_rows,
                dataset.metric_scales,
                allowed_tiers=scoring_tiers,
                eps=dataset.eps,
            ),
            "train_support_counts": {
                metric_name: _count_metric_support(dataset.train_rows, metric_name, allowed_tiers=scoring_tiers)
                for metric_name in AUDIT_METRICS
            },
            "holdout_support_counts": {
                metric_name: _count_metric_support(dataset.holdout_rows, metric_name, allowed_tiers=scoring_tiers)
                for metric_name in AUDIT_METRICS
            },
            "suppression_honesty_flag": _suppression_honesty_flag(
                dataset,
                dict(candidate.get("transition_diagnostics") or {}),
                allowed_tiers=scoring_tiers,
            ),
        }
        yearly_rows_for_split = _yearly_feature_rows_filtered(dataset.train_rows + dataset.holdout_rows, annual_rows)
        holdout_years = list(split["holdout_years"])
        candidate_overlay_annual_error = (
            _annual_metric_error(yearly_rows_for_split, holdout_years, dict(candidate.get("annual_infections") or {}), "annual_new_infections")
            if candidate.get("annual_infections")
            else None
        )
        carry_forward_annual_infections = _annual_diagnosis_flow_totals(carry_forward["prediction_rows"])
        carry_forward_overlay_annual_error = (
            _annual_metric_error(yearly_rows_for_split, holdout_years, carry_forward_annual_infections, "annual_new_infections")
            if carry_forward_annual_infections
            else None
        )
        rows.append(
            {
                "train_end_year": int(split["train_end_year"]),
                "holdout_years": list(split["holdout_years"]),
                "carry_forward": {"mae": float(carry_forward_mae), "smape": float(carry_forward_smape)},
                "candidate": {"mae": float(candidate_mae), "smape": float(candidate_smape)},
                "candidate_overlay_annual_error": candidate_overlay_annual_error,
                "carry_forward_overlay_annual_error": carry_forward_overlay_annual_error,
                "holdout_target_rows": [
                    {
                        "quarter": str(row["quarter"]),
                        **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
                        "metric_tiers": {metric_name: _metric_tier(row, metric_name) for metric_name in PRIMARY_METRICS},
                    }
                    for row in dataset.holdout_rows
                ],
                "candidate_prediction_rows": [
                    {
                        "quarter": str(row["quarter"]),
                        **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
                    }
                    for row in candidate["prediction_rows"]
                ],
                "carry_forward_prediction_rows": [
                    {
                        "quarter": str(row["quarter"]),
                        **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
                    }
                    for row in carry_forward["prediction_rows"]
                ],
                "candidate_annual_infections": dict(candidate.get("annual_infections") or {}),
                "transition_diagnostics": dict(candidate.get("transition_diagnostics") or {}),
                "endpoint_audit": endpoint_audit,
                "scored_metric_count": int(
                    sum(
                        1
                        for row in dataset.holdout_rows
                        for metric_name in PRIMARY_METRICS
                        if _metric_tier(row, metric_name) in scoring_tiers and row.get(metric_name) is not None
                    )
                ),
            }
        )
        candidate_maes.append(float(candidate_mae))
        carry_forward_maes.append(float(carry_forward_mae))
        if candidate_overlay_annual_error is not None and math.isfinite(float(candidate_overlay_annual_error)):
            candidate_overlay_annual_errors.append(float(candidate_overlay_annual_error))
        if carry_forward_overlay_annual_error is not None and math.isfinite(float(carry_forward_overlay_annual_error)):
            carry_forward_overlay_annual_errors.append(float(carry_forward_overlay_annual_error))
    summary = {
        "candidate_mean_mae": float(np.mean(candidate_maes)) if candidate_maes else float("inf"),
        "carry_forward_mean_mae": float(np.mean(carry_forward_maes)) if carry_forward_maes else float("inf"),
        "candidate_worst_mae": float(max(candidate_maes)) if candidate_maes else float("inf"),
        "carry_forward_worst_mae": float(max(carry_forward_maes)) if carry_forward_maes else float("inf"),
        "candidate_overlay_mean_annual_error": float(np.mean(candidate_overlay_annual_errors)) if candidate_overlay_annual_errors else None,
        "carry_forward_overlay_mean_annual_error": float(np.mean(carry_forward_overlay_annual_errors)) if carry_forward_overlay_annual_errors else None,
        "endpoint_audit_summary": _aggregate_endpoint_audit(rows),
    }
    return rows, summary


def _run_annual_diagnostics(
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    annual_cfg: AnnualIncidenceConfig,
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    yearly_rows = _yearly_feature_rows_filtered(observation_rows, annual_rows)
    years = [int(row["year"]) for row in yearly_rows]
    splits = rolling_origin_year_splits(
        years,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    incidence_map = {int(row["year"]): float(row["annual_new_infections"]) for row in yearly_rows}
    plhiv_map = {int(row["year"]): float(row["estimated_plhiv"]) for row in yearly_rows}
    for split in splits:
        train = [row for row in yearly_rows if int(row["year"]) <= int(split["train_end_year"])]
        candidate_pred = _fit_annual_incidence_model(train, list(split["holdout_years"]), annual_cfg)
        baseline_pred = {year: float(train[-1]["annual_new_infections"]) for year in split["holdout_years"]}
        candidate_error = _annual_metric_error(yearly_rows, list(split["holdout_years"]), candidate_pred, "annual_new_infections")
        baseline_error = _annual_metric_error(yearly_rows, list(split["holdout_years"]), baseline_pred, "annual_new_infections")
        plhiv_baseline = {year: float(plhiv_map.get(year - 1, train[-1]["estimated_plhiv"])) for year in split["holdout_years"]}
        plhiv_candidate = {
            year: float(plhiv_map.get(year - 1, train[-1]["estimated_plhiv"]) + candidate_pred.get(year, 0.0) - baseline_pred.get(year, 0.0))
            for year in split["holdout_years"]
        }
        rows.append(
            {
                "train_end_year": int(split["train_end_year"]),
                "holdout_years": list(split["holdout_years"]),
                "candidate_incidence_error": float(candidate_error),
                "baseline_incidence_error": float(baseline_error),
                "candidate_plhiv_error": float(_annual_metric_error(yearly_rows, list(split["holdout_years"]), plhiv_candidate, "estimated_plhiv")),
                "baseline_plhiv_error": float(_annual_metric_error(yearly_rows, list(split["holdout_years"]), plhiv_baseline, "estimated_plhiv")),
            }
        )
        candidate_errors.append(float(candidate_error))
        baseline_errors.append(float(baseline_error))
    summary = {
        "candidate_mean_incidence_error": float(np.mean(candidate_errors)) if candidate_errors else float("inf"),
        "baseline_mean_incidence_error": float(np.mean(baseline_errors)) if baseline_errors else float("inf"),
    }
    return rows, summary


def _score_experiment_result(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(result["quarterly_summary"]["candidate_mean_mae"]),
        float(result["annual_summary"]["candidate_mean_incidence_error"]),
        float(result["quarterly_summary"]["candidate_worst_mae"]),
    )


def _select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return min(results, key=_score_experiment_result)


MISSING_DATA_LADDER: tuple[str, ...] = (
    "exact_observed",
    "bridge_observed",
    "rule_based_extrapolated",
    "latent_imputed",
    "rejected_or_quarantined",
)

_TIER_CODES: dict[str, int] = {tier: idx for idx, tier in enumerate(MISSING_DATA_LADDER)}
_MISSING_CODE = len(MISSING_DATA_LADDER)


def _tier_from_snapshot_source_kind(source_kind: str) -> str:
    kind = str(source_kind or "").lower()
    if kind in {"exact_snapshot", "quarterly_exact"}:
        return "exact_observed"
    if kind in {"bridge_cumulative_minus_deaths", "monthly_bridge", "monthly_aggregate"}:
        return "bridge_observed"
    if kind == "missing":
        return "rejected_or_quarantined"
    return "rejected_or_quarantined"


def _metric_tier(row: dict[str, Any], metric_name: str) -> str:
    value = row.get(f"{metric_name}_tier")
    if value:
        return str(value)
    return "exact_observed"


def _filtered_normalized_mae(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, Any]],
    metric_scales: dict[str, float],
    *,
    allowed_tiers: set[str],
    eps: float,
) -> float:
    errors: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            if _metric_tier(target, metric_name) not in allowed_tiers:
                continue
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            errors.append(abs(float(prediction_value) - float(target_value)) / max(float(metric_scales.get(metric_name) or eps), eps))
    return float(np.mean(errors)) if errors else float("inf")


def _filtered_smape(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
    eps: float,
) -> float:
    scores: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            if _metric_tier(target, metric_name) not in allowed_tiers:
                continue
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            denom = abs(float(prediction_value)) + abs(float(target_value))
            if denom > eps:
                scores.append((2.0 * abs(float(prediction_value) - float(target_value))) / denom)
    return float(np.mean(scores)) if scores else 0.0


AUDIT_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)

AUDIT_TIERS: tuple[str, ...] = ("exact_observed", "bridge_observed")


def _count_metric_support(rows: list[dict[str, Any]], metric_name: str, *, allowed_tiers: set[str]) -> dict[str, int]:
    counts = {tier: 0 for tier in AUDIT_TIERS}
    counts["scored"] = 0
    counts["any_nonnull"] = 0
    for row in rows:
        if row.get(metric_name) is None:
            continue
        counts["any_nonnull"] += 1
        tier = _metric_tier(row, metric_name)
        if tier in counts:
            counts[tier] += 1
        if tier in allowed_tiers:
            counts["scored"] += 1
    return counts


def _raw_endpoint_audit(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, Any]],
    metric_scales: dict[str, float],
    *,
    allowed_tiers: set[str],
    eps: float,
) -> dict[str, Any]:
    by_metric: dict[str, dict[str, float | None]] = {}
    by_metric_tier: dict[str, dict[str, dict[str, float | None]]] = {}
    for metric_name in AUDIT_METRICS:
        overall_raw: list[float] = []
        overall_norm: list[float] = []
        tier_rows: dict[str, dict[str, list[float]]] = {
            tier: {"raw": [], "normalized": []}
            for tier in AUDIT_TIERS
        }
        for prediction, target in zip(prediction_rows, target_rows, strict=False):
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            target_tier = _metric_tier(target, metric_name)
            if prediction_value is None or target_value is None or target_tier not in allowed_tiers:
                continue
            raw_error = abs(float(prediction_value) - float(target_value))
            normalized_error = raw_error / max(float(metric_scales.get(metric_name) or eps), eps)
            overall_raw.append(raw_error)
            overall_norm.append(normalized_error)
            if target_tier in tier_rows:
                tier_rows[target_tier]["raw"].append(raw_error)
                tier_rows[target_tier]["normalized"].append(normalized_error)
        by_metric[metric_name] = {
            "raw_mae": float(np.mean(overall_raw)) if overall_raw else None,
            "normalized_mae": float(np.mean(overall_norm)) if overall_norm else None,
            "count": len(overall_raw),
        }
        by_metric_tier[metric_name] = {
            tier: {
                "raw_mae": float(np.mean(values["raw"])) if values["raw"] else None,
                "normalized_mae": float(np.mean(values["normalized"])) if values["normalized"] else None,
                "count": len(values["raw"]),
            }
            for tier, values in tier_rows.items()
        }
    return {"by_metric": by_metric, "by_metric_tier": by_metric_tier}


def _suppression_honesty_flag(
    dataset: Any,
    transition_diagnostics: dict[str, Any],
    *,
    allowed_tiers: set[str],
) -> str:
    holdout_supported = sum(
        1
        for row in dataset.holdout_rows
        if row.get("virally_suppressed") is not None and _metric_tier(row, "virally_suppressed") in allowed_tiers
    )
    a_to_v = dict(transition_diagnostics.get("A_to_V") or {})
    support_count = int(a_to_v.get("support_count") or 0)
    model_kind = str(a_to_v.get("model_kind") or "")
    if holdout_supported > 0 and support_count > 0:
        return "scored_direct_support"
    if support_count > 0:
        return "share_carry_from_supported_train"
    if "level_carry" in model_kind:
        return "unsupported_level_carry"
    if "carry" in model_kind:
        return "unsupported_share_carry"
    return "unsupported_or_unclaimed"


def _aggregate_endpoint_audit(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "candidate": {"by_metric": {}, "by_metric_tier": {}},
        "carry_forward": {"by_metric": {}, "by_metric_tier": {}},
        "train_support_counts": {},
        "holdout_support_counts": {},
        "suppression_honesty_flags": {},
    }
    for model_name in ("candidate", "carry_forward"):
        for metric_name in AUDIT_METRICS:
            raw_values: list[float] = []
            norm_values: list[float] = []
            counts = 0
            tier_payload: dict[str, Any] = {}
            for tier in AUDIT_TIERS:
                tier_raw: list[float] = []
                tier_norm: list[float] = []
                tier_count = 0
                for split in split_rows:
                    endpoint_audit = dict(split.get("endpoint_audit") or {})
                    model_audit = dict(endpoint_audit.get(model_name) or {})
                    metric_tier_row = dict(dict(model_audit.get("by_metric_tier") or {}).get(metric_name) or {}).get(tier) or {}
                    if metric_tier_row.get("raw_mae") is not None:
                        tier_raw.append(float(metric_tier_row["raw_mae"]))
                    if metric_tier_row.get("normalized_mae") is not None:
                        tier_norm.append(float(metric_tier_row["normalized_mae"]))
                    tier_count += int(metric_tier_row.get("count") or 0)
                tier_payload[tier] = {
                    "raw_mae": float(np.mean(tier_raw)) if tier_raw else None,
                    "normalized_mae": float(np.mean(tier_norm)) if tier_norm else None,
                    "count": tier_count,
                }
            for split in split_rows:
                endpoint_audit = dict(split.get("endpoint_audit") or {})
                model_audit = dict(endpoint_audit.get(model_name) or {})
                metric_row = dict(dict(model_audit.get("by_metric") or {}).get(metric_name) or {})
                if metric_row.get("raw_mae") is not None:
                    raw_values.append(float(metric_row["raw_mae"]))
                if metric_row.get("normalized_mae") is not None:
                    norm_values.append(float(metric_row["normalized_mae"]))
                counts += int(metric_row.get("count") or 0)
            summary[model_name]["by_metric"][metric_name] = {
                "raw_mae": float(np.mean(raw_values)) if raw_values else None,
                "normalized_mae": float(np.mean(norm_values)) if norm_values else None,
                "count": counts,
            }
            summary[model_name]["by_metric_tier"][metric_name] = tier_payload
    for support_key in ("train_support_counts", "holdout_support_counts"):
        for metric_name in AUDIT_METRICS:
            counts = {tier: 0 for tier in (*AUDIT_TIERS, "scored", "any_nonnull")}
            for split in split_rows:
                endpoint_audit = dict(split.get("endpoint_audit") or {})
                metric_counts = dict(dict(endpoint_audit.get(support_key) or {}).get(metric_name) or {})
                for tier in counts:
                    counts[tier] += int(metric_counts.get(tier) or 0)
            summary[support_key][metric_name] = counts
    for split in split_rows:
        endpoint_audit = dict(split.get("endpoint_audit") or {})
        flag = str(endpoint_audit.get("suppression_honesty_flag") or "unsupported_or_unclaimed")
        summary["suppression_honesty_flags"][flag] = int(summary["suppression_honesty_flags"].get(flag, 0)) + 1
    return summary


def _annual_diagnosis_flow_totals(rows: list[dict[str, Any]]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for row in rows:
        value = row.get("new_diagnosed_cases_period")
        if value is None:
            continue
        totals.setdefault(quarter_year(str(row["quarter"])), 0.0)
        totals[quarter_year(str(row["quarter"]))] += float(value)
    return totals


def _annual_row_tier(row: dict[str, Any]) -> str:
    quality = str(row.get("source_quality_tier") or row.get("measurement_class") or "other").lower()
    if quality.startswith("official"):
        return "exact_observed"
    if quality == "model_estimate":
        return "rule_based_extrapolated"
    if quality == "derived":
        return "bridge_observed"
    return "rejected_or_quarantined"


def _build_availability_payload(archive_run_id: str) -> dict[str, Any]:
    from epigraph_ph.phase3.bridge_quarterly_panel import build_bridge_quarterly_panel_rows
    from epigraph_ph.phase3.tr_v3_05_autoresearch import load_archive_rows

    bridge_panel = build_bridge_quarterly_panel_rows(archive_run_id)
    bridge_rows = list(bridge_panel["rows"])
    annual_rows = build_annual_anchor_rows(archive_run_id)
    exact_rows = build_quarterly_observation_rows(archive_run_id)
    archive_rows = load_archive_rows(archive_run_id)
    quarterly_metrics = [
        ("diagnosed_plhiv", "diagnosed_plhiv_source_kind"),
        ("alive_on_art", "alive_on_art_source_kind"),
        ("new_diagnosed_cases_period", "new_diagnosed_cases_source_kind"),
        ("tested_for_viral_load", "tested_for_viral_load_source_kind"),
        ("virally_suppressed", "virally_suppressed_source_kind"),
        ("estimated_plhiv", "estimated_plhiv_source_kind"),
        ("deaths_reported_period", "deaths_reported_source_kind"),
    ]
    quarters = [str(row["quarter"]) for row in bridge_rows]
    quarterly_matrix: list[list[int]] = []
    quarterly_counts: dict[str, dict[str, int]] = {}
    for metric_name, source_field in quarterly_metrics:
        row_codes: list[int] = []
        tier_counts = {tier: 0 for tier in MISSING_DATA_LADDER}
        tier_counts["missing"] = 0
        for row in bridge_rows:
            value = row.get(metric_name)
            if value is None:
                row_codes.append(_MISSING_CODE)
                tier_counts["missing"] += 1
                continue
            tier = _tier_from_snapshot_source_kind(str(row.get(source_field) or "missing"))
            row_codes.append(_TIER_CODES[tier])
            tier_counts[tier] += 1
        quarterly_matrix.append(row_codes)
        quarterly_counts[metric_name] = tier_counts
    annual_metric_names = list(ANNUAL_METRICS)
    annual_years = sorted({int(row["year"]) for row in annual_rows})
    annual_row_map = {(str(row["metric_name"]), int(row["year"])): row for row in annual_rows}
    annual_matrix: list[list[int]] = []
    annual_counts: dict[str, dict[str, int]] = {}
    for metric_name in annual_metric_names:
        row_codes = []
        tier_counts = {tier: 0 for tier in MISSING_DATA_LADDER}
        tier_counts["missing"] = 0
        for year in annual_years:
            row = annual_row_map.get((metric_name, year))
            if row is None:
                row_codes.append(_MISSING_CODE)
                tier_counts["missing"] += 1
                continue
            tier = _annual_row_tier(row)
            row_codes.append(_TIER_CODES[tier])
            tier_counts[tier] += 1
        annual_matrix.append(row_codes)
        annual_counts[metric_name] = tier_counts
    exact_years = sorted({quarter_year(str(row["quarter"])) for row in exact_rows})
    bridge_complete_years = sorted({quarter_year(str(row["quarter"])) for row in bridge_rows if bool(row.get("stock_anchor_complete"))})
    flow_metric_count = sum(1 for row in archive_rows if str(row.get("metric_name") or "") == "new_diagnosed_cases_period")
    return {
        "archive_run_id": archive_run_id,
        "missing_data_ladder": list(MISSING_DATA_LADDER),
        "quarterly": {
            "metrics": [metric_name for metric_name, _ in quarterly_metrics],
            "quarters": quarters,
            "matrix": quarterly_matrix,
            "tier_counts": quarterly_counts,
            "exact_observation_row_count": len(exact_rows),
            "exact_observation_years": exact_years,
            "bridge_complete_years": bridge_complete_years,
            "bridge_summary": dict(bridge_panel["summary"]),
            "flow_metric_row_count": flow_metric_count,
        },
        "annual": {
            "metrics": annual_metric_names,
            "years": annual_years,
            "matrix": annual_matrix,
            "tier_counts": annual_counts,
        },
    }


def _build_dense_contract_payload(archive_run_id: str, *, max_quarter: str | None = None) -> dict[str, Any]:
    dense_panel = build_dense_quarterly_panel_rows(archive_run_id, max_quarter=max_quarter)
    rows = list(dense_panel["rows"])
    return {
        "archive_run_id": archive_run_id,
        "rows": rows,
        "summary": dict(dense_panel["summary"]),
        "score_eligible_years": sorted({quarter_year(str(row["quarter"])) for row in rows if bool(row.get("score_eligible"))}),
    }


def _build_purged_dense_split_rows(
    archive_run_id: str,
    *,
    full_dense_rows: list[dict[str, Any]],
    train_end_year: int,
    holdout_years: list[int],
) -> list[dict[str, Any]]:
    max_train_quarter = f"{int(train_end_year):04d}-Q4"
    purged_dense_contract = _build_dense_contract_payload(archive_run_id, max_quarter=max_train_quarter)
    train_rows = [
        dict(row)
        for row in list(purged_dense_contract["rows"])
        if quarter_year(str(row["quarter"])) <= int(train_end_year)
    ]
    holdout_year_set = {int(year) for year in holdout_years}
    holdout_rows = [
        dict(row)
        for row in full_dense_rows
        if quarter_year(str(row["quarter"])) in holdout_year_set
    ]
    return sorted(train_rows + holdout_rows, key=lambda row: quarter_sort_key(str(row["quarter"])))


def _contract_support_summary(
    rows: list[dict[str, Any]],
    *,
    scoring_tiers: set[str],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    years = sorted({quarter_year(str(row["quarter"])) for row in rows})
    split_rows = []
    for split in rolling_origin_year_splits(
        years,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    ):
        holdout_rows = [dict(row) for row in rows if quarter_year(str(row["quarter"])) in set(split["holdout_years"])]
        scored_metric_count = int(
            sum(
                1
                for row in holdout_rows
                for metric_name in PRIMARY_METRICS
                if _metric_tier(row, metric_name) in scoring_tiers and row.get(metric_name) is not None
            )
        )
        split_rows.append(
            {
                "train_end_year": int(split["train_end_year"]),
                "holdout_years": list(split["holdout_years"]),
                "train_row_count": int(sum(1 for row in rows if quarter_year(str(row["quarter"])) <= int(split["train_end_year"]))),
                "holdout_row_count": int(len(holdout_rows)),
                "scored_metric_count": scored_metric_count,
            }
        )
    return {
        "row_count": len(rows),
        "years": years,
        "split_rows": split_rows,
    }


def _build_contract_comparison_payload(
    archive_run_id: str,
    *,
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    exact_rows = build_quarterly_observation_rows(archive_run_id)
    dense_contract = _build_dense_contract_payload(archive_run_id)
    dense_rows = list(dense_contract["rows"])
    return {
        "archive_run_id": archive_run_id,
        "contracts": [
            {
                "name": "exact_only",
                "training_row_count": len(exact_rows),
                "score_eligible_quarters": len(exact_rows),
                "score_eligible_years": sorted({quarter_year(str(row["quarter"])) for row in exact_rows}),
                "support": _contract_support_summary(
                    exact_rows,
                    scoring_tiers={"exact_observed"},
                    start_year=quarterly_start_year,
                    end_year=quarterly_end_year,
                    min_train_years=quarterly_min_train_years,
                    horizon_years=horizon_years,
                ),
            },
            {
                "name": "dense_train_observed_score",
                "training_row_count": len(dense_rows),
                "score_eligible_quarters": int(dense_contract["summary"]["score_eligible_quarters"]),
                "score_eligible_years": list(dense_contract["summary"]["score_eligible_years"]),
                "support": _contract_support_summary(
                    dense_rows,
                    scoring_tiers={"exact_observed", "bridge_observed"},
                    start_year=quarterly_start_year,
                    end_year=quarterly_end_year,
                    min_train_years=quarterly_min_train_years,
                    horizon_years=horizon_years,
                ),
            },
        ],
    }


def _build_imputation_contract_payload(archive_run_id: str) -> dict[str, Any]:
    dense_contract = _build_dense_contract_payload(archive_run_id)
    rows = list(dense_contract["rows"])
    earliest = rows[:16]
    return {
        "archive_run_id": archive_run_id,
        "summary": dict(dense_contract["summary"]),
        "early_rows": [
            {
                "quarter": str(row["quarter"]),
                "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
                "diagnosed_plhiv_tier": str(row["diagnosed_plhiv_tier"]),
                "diagnosed_plhiv_imputation_method": str(row.get("diagnosed_plhiv_imputation_method") or ""),
                "alive_on_art": float(row["alive_on_art"]),
                "alive_on_art_tier": str(row["alive_on_art_tier"]),
                "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
                "new_diagnosed_cases_period_tier": str(row["new_diagnosed_cases_period_tier"]),
                "score_eligible": bool(row["score_eligible"]),
            }
            for row in earliest
        ],
    }


def _annual_metric_year_map(annual_rows: list[dict[str, Any]], metric_name: str) -> dict[int, float]:
    return {
        int(row["year"]): float(row["value"])
        for row in annual_rows
        if str(row.get("metric_name") or "") == metric_name and row.get("value") is not None
    }


def _build_susceptible_identification_contract_payload(archive_run_id: str) -> dict[str, Any]:
    dense_contract = _build_dense_contract_payload(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    rows = list(dense_contract["rows"])
    population_map = _annual_metric_year_map(annual_rows, "population_total")
    plhiv_map = _annual_metric_year_map(annual_rows, "estimated_plhiv")
    incidence_map = _annual_metric_year_map(annual_rows, "annual_new_infections")
    annual_joint_years = sorted(set(population_map) & set(plhiv_map) & set(incidence_map))
    proxy_rows = []
    for year in annual_joint_years:
        population_value = float(population_map[year])
        plhiv_value = float(plhiv_map[year])
        susceptible_proxy = float(max(population_value - plhiv_value, 0.0))
        susceptible_fraction = float(susceptible_proxy / max(population_value, 1.0))
        proxy_rows.append(
            {
                "year": int(year),
                "population_total": population_value,
                "estimated_plhiv": plhiv_value,
                "annual_new_infections": float(incidence_map[year]),
                "susceptible_proxy": susceptible_proxy,
                "susceptible_fraction": susceptible_fraction,
            }
        )
    yearly_quarterly_support = []
    years = sorted({quarter_year(str(row["quarter"])) for row in rows})
    for year in years:
        year_rows = [row for row in rows if quarter_year(str(row["quarter"])) == year]
        yearly_quarterly_support.append(
            {
                "year": int(year),
                "diagnosed_stock_quarters": sum(
                    1 for row in year_rows if _metric_tier(row, "diagnosed_plhiv") in {"exact_observed", "bridge_observed"}
                ),
                "art_stock_quarters": sum(
                    1 for row in year_rows if _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"}
                ),
                "diagnosis_flow_quarters": sum(
                    1
                    for row in year_rows
                    if _metric_tier(row, "new_diagnosed_cases_period") in {"exact_observed", "bridge_observed"}
                ),
            }
        )
    annual_sidecar_allowed = len(annual_joint_years) >= 10
    return {
        "archive_run_id": archive_run_id,
        "annual_years": {
            "population_total": sorted(population_map),
            "estimated_plhiv": sorted(plhiv_map),
            "annual_new_infections": sorted(incidence_map),
            "joint_support_years": annual_joint_years,
        },
        "allowed_blocks": {
            "annual_s_proxy": "allowed_with_shrinkage" if annual_sidecar_allowed else "weak_support_only",
            "annual_force_of_infection_sanity": "allowed_with_shrinkage" if annual_sidecar_allowed else "weak_support_only",
            "quarterly_explicit_s": "not_identifiable_for_primary_loop",
            "free_joint_s_and_n_estimation": "not_identifiable",
        },
        "proxy_rows": proxy_rows,
        "yearly_quarterly_support": yearly_quarterly_support,
        "decision": {
            "status": "annual_sidecar_only" if annual_sidecar_allowed else "defer",
            "why": (
                "Annual denominator, PLHIV, and incidence anchors are strong enough for an annual susceptible sidecar or denominator sanity layer, "
                "but the archive does not support a free quarterly S(t) block jointly with infection pressure, reporting, and care dynamics."
            ),
        },
    }


def _build_susceptible_sidecar_payload(archive_run_id: str) -> dict[str, Any]:
    contract = _build_susceptible_identification_contract_payload(archive_run_id)
    pressure_rows = []
    for row in list(contract.get("proxy_rows") or []):
        susceptible_proxy = float(row["susceptible_proxy"])
        population_total = float(row["population_total"])
        annual_new_infections = float(row["annual_new_infections"])
        pressure_rows.append(
            {
                **dict(row),
                "plhiv_fraction": float(float(row["estimated_plhiv"]) / max(population_total, 1.0)),
                "annual_incidence_per_susceptible": float(annual_new_infections / max(susceptible_proxy, 1.0)),
                "annual_incidence_per_100k_susceptible": float((annual_new_infections / max(susceptible_proxy, 1.0)) * 100000.0),
            }
        )
    decision = dict(contract.get("decision") or {})
    return {
        "archive_run_id": archive_run_id,
        "contract": contract,
        "pressure_rows": pressure_rows,
        "decision": {
            "status": str(decision.get("status") or "defer"),
            "why": (
                "Annual susceptible exploration is defensible as a denominator/coherence sidecar only. "
                "It can inform annual incidence pressure sanity checks, but it still does not justify a free quarterly S(t) state."
            ),
        },
    }


def _build_leakage_identification_contract_payload(archive_run_id: str) -> dict[str, Any]:
    dense_contract = _build_dense_contract_payload(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    rows = list(dense_contract["rows"])
    yearly_support = []
    years = sorted({quarter_year(str(row["quarter"])) for row in rows})
    for year in years:
        year_rows = [row for row in rows if quarter_year(str(row["quarter"])) == year]
        diagnosed_support = sum(1 for row in year_rows if _metric_tier(row, "diagnosed_plhiv") in {"exact_observed", "bridge_observed"})
        alive_support = sum(1 for row in year_rows if _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"})
        flow_support = sum(1 for row in year_rows if _metric_tier(row, "new_diagnosed_cases_period") in {"exact_observed", "bridge_observed"})
        yearly_support.append(
            {
                "year": int(year),
                "diagnosed_stock_quarters": diagnosed_support,
                "art_stock_quarters": alive_support,
                "diagnosis_flow_quarters": flow_support,
                "can_score_art_leakage_net": bool(alive_support >= 2 and flow_support >= 2),
                "can_score_diagnosed_leakage_separately": False,
            }
        )
    annual_deaths_years = sorted({int(row["year"]) for row in annual_rows if str(row.get("metric_name") or "") == "annual_aids_deaths"})
    return {
        "archive_run_id": archive_run_id,
        "allowed_blocks": {
            "net_art_leakage": "allowed_with_strong_shrinkage",
            "reentry_to_art": "allowed_with_strong_shrinkage",
            "diagnosed_to_lost": "not_separately_identifiable",
            "virally_suppressed_to_lost": "not_identifiable",
            "state_specific_mortality": "not_identifiable",
        },
        "primary_targets": [
            "diagnosed_plhiv",
            "alive_on_art",
            "new_diagnosed_cases_period",
        ],
        "annual_sanity_targets": ["annual_aids_deaths"],
        "annual_aids_deaths_years": annual_deaths_years,
        "yearly_support": yearly_support,
        "decision": {
            "exp_l1_status": "contract_defined_build_not_free",
            "why": "The current archive supports net ART leakage and re-entry only under strong shrinkage. Separate D_to_L, V_to_L, and mortality blocks remain under-identified.",
        },
    }


def _build_richer_leakage_identification_contract_payload(archive_run_id: str) -> dict[str, Any]:
    dense_contract = _build_dense_contract_payload(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    rows = list(dense_contract["rows"])
    annual_deaths_years = sorted(
        {
            int(row["year"])
            for row in annual_rows
            if str(row.get("metric_name") or "") == "annual_aids_deaths" and row.get("value") is not None
        }
    )
    yearly_support = []
    years = sorted({quarter_year(str(row["quarter"])) for row in rows})
    for year in years:
        year_rows = [row for row in rows if quarter_year(str(row["quarter"])) == year]
        diagnosed_support = sum(
            1 for row in year_rows if _metric_tier(row, "diagnosed_plhiv") in {"exact_observed", "bridge_observed"}
        )
        alive_support = sum(
            1 for row in year_rows if _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"}
        )
        suppression_support = sum(
            1
            for row in year_rows
            if _metric_tier(row, "virally_suppressed") in {"exact_observed", "bridge_observed"}
            and _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"}
        )
        flow_support = sum(
            1
            for row in year_rows
            if _metric_tier(row, "new_diagnosed_cases_period") in {"exact_observed", "bridge_observed"}
        )
        deaths_flow_support = sum(
            1
            for row in year_rows
            if _metric_tier(row, "deaths_reported_period") in {"exact_observed", "bridge_observed"}
        )
        yearly_support.append(
            {
                "year": int(year),
                "diagnosed_stock_quarters": diagnosed_support,
                "art_stock_quarters": alive_support,
                "suppression_quarters": suppression_support,
                "diagnosis_flow_quarters": flow_support,
                "deaths_flow_quarters": deaths_flow_support,
                "can_score_art_leakage_net": bool(alive_support >= 2 and flow_support >= 2),
                "can_score_suppressed_leakage_separately": bool(suppression_support >= 2 and deaths_flow_support >= 2),
                "can_score_quarterly_mortality_coupled_leakage": False,
            }
        )
    return {
        "archive_run_id": archive_run_id,
        "allowed_blocks": {
            "net_art_leakage": "allowed_with_strong_shrinkage",
            "reentry_to_art": "allowed_with_strong_shrinkage",
            "diagnosed_to_lost": "not_separately_identifiable",
            "virally_suppressed_to_lost": "not_identifiable",
            "quarterly_mortality_coupled_leakage": "not_identifiable",
            "annual_mortality_sanity": "allowed_as_sidecar_only" if annual_deaths_years else "not_available",
            "full_d_a_v_l_network": "not_identifiable",
        },
        "annual_aids_deaths_years": annual_deaths_years,
        "yearly_support": yearly_support,
        "decision": {
            "status": "defer_richer_leakage",
            "why": (
                "The archive supports net ART leakage and re-entry only. Separate diagnosed leakage, suppression-specific leakage, "
                "and mortality-coupled quarterly leakage remain under-identified even after deaths repair."
            ),
        },
    }


def _build_late_leakage_sensitivity_payload(archive_run_id: str) -> dict[str, Any]:
    dense_contract = _build_dense_contract_payload(archive_run_id)
    rows = sorted(list(dense_contract["rows"]), key=lambda row: quarter_sort_key(str(row["quarter"])))
    late_rows = [row for row in rows if quarter_year(str(row["quarter"])) >= 2023]
    proxy_rows: list[dict[str, Any]] = []
    previous_row: dict[str, Any] | None = None
    for row in late_rows:
        if previous_row is None:
            previous_row = row
            continue
        previous_quarter = str(previous_row["quarter"])
        quarter = str(row["quarter"])
        if quarter_gap(previous_quarter, quarter) != 1:
            previous_row = row
            continue
        art_supported = (
            previous_row.get("alive_on_art") is not None
            and row.get("alive_on_art") is not None
            and _metric_tier(previous_row, "alive_on_art") in {"exact_observed", "bridge_observed"}
            and _metric_tier(row, "alive_on_art") in {"exact_observed", "bridge_observed"}
        )
        suppressed_supported = (
            previous_row.get("virally_suppressed") is not None
            and row.get("virally_suppressed") is not None
            and _metric_tier(previous_row, "virally_suppressed") in {"exact_observed", "bridge_observed"}
            and _metric_tier(row, "virally_suppressed") in {"exact_observed", "bridge_observed"}
        )
        deaths_supported = (
            row.get("deaths_reported_period") is not None
            and _metric_tier(row, "deaths_reported_period") in {"exact_observed", "bridge_observed"}
        )
        if not art_supported or not suppressed_supported:
            previous_row = row
            continue
        previous_art = float(previous_row["alive_on_art"])
        current_art = float(row["alive_on_art"])
        previous_suppressed = float(previous_row["virally_suppressed"])
        current_suppressed = float(row["virally_suppressed"])
        previous_share = float(previous_suppressed / max(previous_art, 1.0))
        carry_suppressed = float(np.clip(previous_share * current_art, 0.0, current_art))
        shortfall_vs_share_carry = float(max(carry_suppressed - current_suppressed, 0.0))
        deaths_value = float(row["deaths_reported_period"]) if deaths_supported else None
        proxy_rows.append(
            {
                "quarter": quarter,
                "year": int(quarter_year(quarter)),
                "previous_share": previous_share,
                "current_share": float(current_suppressed / max(current_art, 1.0)),
                "carry_suppressed": carry_suppressed,
                "observed_suppressed": current_suppressed,
                "shortfall_vs_share_carry": shortfall_vs_share_carry,
                "shortfall_after_reported_deaths": (
                    float(max(shortfall_vs_share_carry - float(deaths_value), 0.0)) if deaths_value is not None else None
                ),
                "reported_deaths_period": deaths_value,
            }
        )
        previous_row = row
    yearly_summary: list[dict[str, Any]] = []
    for year in sorted({int(row["year"]) for row in proxy_rows}):
        year_rows = [row for row in proxy_rows if int(row["year"]) == year]
        yearly_summary.append(
            {
                "year": int(year),
                "quarter_count": len(year_rows),
                "mean_shortfall_vs_share_carry": float(np.mean([float(row["shortfall_vs_share_carry"]) for row in year_rows])),
                "total_shortfall_vs_share_carry": float(np.sum([float(row["shortfall_vs_share_carry"]) for row in year_rows])),
                "total_shortfall_after_reported_deaths": (
                    float(np.sum([float(row["shortfall_after_reported_deaths"]) for row in year_rows if row["shortfall_after_reported_deaths"] is not None]))
                    if any(row["shortfall_after_reported_deaths"] is not None for row in year_rows)
                    else None
                ),
            }
        )
    return {
        "archive_run_id": archive_run_id,
        "support_window": {"start_year": 2023, "end_year": 2025},
        "proxy_rows": proxy_rows,
        "yearly_summary": yearly_summary,
        "decision": {
            "status": "late_window_sensitivity_only" if proxy_rows else "defer",
            "why": (
                "Late-era suppression-linked leakage can only be explored as an upper-bound proxy based on suppression-share shortfall relative to share carry. "
                "That proxy is not an identifiable quarterly leakage flow and should stay outside the benchmark loop."
            ),
        },
    }


def _evaluate_targeted_contract_rows(
    archive_run_id: str,
    *,
    experiment_ids: list[str],
    quarterly_contract: str,
    availability: dict[str, Any],
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
    split_local_dense: bool,
) -> list[dict[str, Any]]:
    spec_map = {spec.experiment_id: spec for spec in build_experiment_suite_specs()}
    annual_rows = build_annual_anchor_rows(archive_run_id)
    if quarterly_contract == "exact_only":
        observation_rows = build_quarterly_observation_rows(archive_run_id)
        scoring_tiers = {"exact_observed"}
        full_dense_rows = None
    elif quarterly_contract == "dense_train_observed_score":
        dense_contract = _build_dense_contract_payload(archive_run_id)
        observation_rows = list(dense_contract["rows"])
        scoring_tiers = {"exact_observed", "bridge_observed"}
        full_dense_rows = list(dense_contract["rows"])
    else:
        raise ValueError(f"Unsupported quarterly_contract: {quarterly_contract}")
    results = []
    for experiment_id in experiment_ids:
        spec = spec_map[experiment_id]
        result = _evaluate_experiment_spec(
            spec,
            observation_rows=observation_rows,
            annual_rows=annual_rows,
            availability=availability,
            scoring_tiers=scoring_tiers,
            archive_run_id=archive_run_id,
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
            split_local_dense=split_local_dense,
            full_dense_rows=full_dense_rows,
        )
        results.append(
            {
                "experiment_id": str(result["experiment_id"]),
                "decision": str(result.get("decision") or ""),
                "quarterly_mean_mae": float(dict(result.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")),
                "quarterly_baseline_mae": float(dict(result.get("quarterly_summary") or {}).get("carry_forward_mean_mae") or float("inf")),
                "annual_mean_error": float(dict(result.get("annual_summary") or {}).get("candidate_mean_incidence_error") or float("inf")),
                "endpoint_audit_summary": dict(dict(result.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {}),
            }
        )
    return sorted(results, key=lambda row: float(row["quarterly_mean_mae"]))


def _build_purged_dense_contract_audit_payload(
    archive_run_id: str,
    *,
    availability: dict[str, Any],
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    experiment_ids = ["EXP-R10-EXACT-CHAMPION", "EXP-R10-DENSE-CHAMPION", "EXP-R1"]
    exact_reference = _evaluate_targeted_contract_rows(
        archive_run_id,
        experiment_ids=experiment_ids,
        quarterly_contract="exact_only",
        availability=availability,
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
        split_local_dense=False,
    )
    legacy_dense = _evaluate_targeted_contract_rows(
        archive_run_id,
        experiment_ids=experiment_ids,
        quarterly_contract="dense_train_observed_score",
        availability=availability,
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
        split_local_dense=False,
    )
    purged_dense = _evaluate_targeted_contract_rows(
        archive_run_id,
        experiment_ids=experiment_ids,
        quarterly_contract="dense_train_observed_score",
        availability=availability,
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
        split_local_dense=True,
    )
    legacy_winner = str(legacy_dense[0]["experiment_id"]) if legacy_dense else ""
    purged_winner = str(purged_dense[0]["experiment_id"]) if purged_dense else ""
    legacy_order = [str(row["experiment_id"]) for row in legacy_dense]
    purged_order = [str(row["experiment_id"]) for row in purged_dense]
    legacy_winner_mae = float(legacy_dense[0]["quarterly_mean_mae"]) if legacy_dense else float("inf")
    purged_winner_mae = float(purged_dense[0]["quarterly_mean_mae"]) if purged_dense else float("inf")
    winner_mae_delta = abs(purged_winner_mae - legacy_winner_mae)
    decision = "keep_dense_lane" if legacy_winner == purged_winner and winner_mae_delta <= 0.02 else "revert_dense_lane"
    return {
        "archive_run_id": archive_run_id,
        "experiment_ids": experiment_ids,
        "exact_reference": exact_reference,
        "legacy_dense": legacy_dense,
        "purged_dense": purged_dense,
        "legacy_winner": legacy_winner,
        "purged_winner": purged_winner,
        "legacy_order": legacy_order,
        "purged_order": purged_order,
        "winner_mae_delta": float(winner_mae_delta),
        "decision": decision,
    }


def _build_endpoint_tier_audit_payload(results: list[dict[str, Any]], *, quarterly_contract: str) -> dict[str, Any]:
    result_map = {str(row["experiment_id"]): row for row in results if row.get("status") == "executed"}
    tracked_ids = [
        default_predictive_candidate_id("exact_only"),
        default_predictive_candidate_id("dense_train_observed_score"),
        mechanistic_research_anchor_id(),
        "EXP-R10-M1",
        "EXP-R10-M1-B1",
        "EXP-R10-M1-F1",
        "EXP-R10-M2",
        "EXP-R10-DENSE-H1",
        "EXP-R10-DENSE-M1",
        "EXP-R10-DENSE-M1-H1",
        "EXP-R10-DENSE-M1-B1-H1",
        "EXP-R10-DENSE-M1-F1-H1",
        "EXP-R10-DENSE-M2",
        "EXP-R11",
    ]
    rows = []
    for experiment_id in tracked_ids:
        row = result_map.get(experiment_id)
        if not row:
            continue
        quarterly_summary = dict(row.get("quarterly_summary") or {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": str(row.get("family") or ""),
                "decision": str(row.get("decision") or ""),
                "quarterly_mean_mae": float(quarterly_summary.get("candidate_mean_mae") or float("inf")),
                "quarterly_baseline_mae": float(quarterly_summary.get("carry_forward_mean_mae") or float("inf")),
                "endpoint_audit_summary": dict(quarterly_summary.get("endpoint_audit_summary") or {}),
            }
        )
    return {
        "quarterly_contract": quarterly_contract,
        "tracked_results": rows,
    }


def _plot_placeholder(path: Path, *, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(0.02, 0.72, title, fontsize=14, fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.42, body, fontsize=11, ha="left", va="top", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_missingness_heatmap(availability: dict[str, Any], path: Path) -> None:
    quarterly = dict(availability["quarterly"])
    annual = dict(availability["annual"])
    cmap = matplotlib.colors.ListedColormap(["#2166ac", "#4daf4a", "#fdae61", "#d73027", "#762a83", "#f0f0f0"])
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), constrained_layout=True)
    q_matrix = np.asarray(quarterly["matrix"], dtype=np.float64)
    a_matrix = np.asarray(annual["matrix"], dtype=np.float64)
    q_ax = axes[0]
    a_ax = axes[1]
    q_ax.imshow(q_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=_MISSING_CODE)
    q_ax.set_title("Quarterly Availability and Provenance")
    q_ax.set_yticks(range(len(quarterly["metrics"])))
    q_ax.set_yticklabels(quarterly["metrics"])
    q_ax.set_xticks(range(len(quarterly["quarters"])))
    q_ax.set_xticklabels(quarterly["quarters"], rotation=90, fontsize=8)
    a_ax.imshow(a_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=_MISSING_CODE)
    a_ax.set_title("Annual Availability and Provenance")
    a_ax.set_yticks(range(len(annual["metrics"])))
    a_ax.set_yticklabels(annual["metrics"])
    annual_ticks = list(range(0, len(annual["years"]), max(1, len(annual["years"]) // 18 or 1)))
    if annual_ticks[-1] != len(annual["years"]) - 1:
        annual_ticks.append(len(annual["years"]) - 1)
    a_ax.set_xticks(annual_ticks)
    a_ax.set_xticklabels([annual["years"][idx] for idx in annual_ticks], rotation=0, fontsize=8)
    legend_labels = list(MISSING_DATA_LADDER) + ["missing"]
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(idx), markersize=10, label=label)
        for idx, label in enumerate(legend_labels)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_experiment_graph(result: dict[str, Any], path: Path) -> None:
    if result["status"] == "skipped":
        _plot_placeholder(path, title=result["experiment_id"], body=str(result.get("skip_reason") or "Skipped"))
        return
    if result["family"] == "diagnostic":
        kind = str(result.get("diagnostic_kind") or "")
        if kind == "missingness_map":
            _save_missingness_heatmap(dict(result["availability"]), path)
            return
        if kind == "contract_ablation":
            contracts = list(result.get("contracts") or [])
            labels = [str(row["name"]) for row in contracts]
            counts = [int(row["score_eligible_quarters"]) for row in contracts]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, counts)
            ax.set_title("Contract Support Comparison")
            ax.set_ylabel("Score-eligible quarters")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "dense_contract_summary":
            summary = dict(result.get("dense_summary") or {})
            counts = dict(summary.get("row_tier_counts") or {})
            labels = list(counts.keys())
            values = [int(counts[label]) for label in labels]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, values)
            ax.set_title("Dense Panel Row Tiers")
            ax.set_ylabel("Quarter count")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "leakage_identification_contract":
            support_rows = list(result.get("leakage_contract", {}).get("yearly_support") or [])
            years = [int(row["year"]) for row in support_rows]
            art_support = [int(row["art_stock_quarters"]) for row in support_rows]
            flow_support = [int(row["diagnosis_flow_quarters"]) for row in support_rows]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(years, art_support, marker="o", label="ART stock support")
            ax.plot(years, flow_support, marker="o", label="Diagnosis flow support")
            ax.set_title("Leakage Identification Support by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Observed/bridge quarters")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "purged_dense_contract":
            payload = dict(result.get("purged_dense_contract") or {})
            legacy_rows = list(payload.get("legacy_dense") or [])
            purged_rows = list(payload.get("purged_dense") or [])
            labels = [str(row["experiment_id"]) for row in legacy_rows]
            legacy_values = [float(row["quarterly_mean_mae"]) for row in legacy_rows]
            purged_map = {str(row["experiment_id"]): float(row["quarterly_mean_mae"]) for row in purged_rows}
            purged_values = [float(purged_map.get(label, float("nan"))) for label in labels]
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(labels))
            ax.bar(x - 0.18, legacy_values, width=0.35, label="Legacy dense")
            ax.bar(x + 0.18, purged_values, width=0.35, label="Purged dense")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title("Purged Dense Leaderboard Audit")
            ax.set_ylabel("Quarterly mean MAE")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "endpoint_tier_audit":
            payload = dict(result.get("endpoint_tier_audit") or {})
            tracked = list(payload.get("tracked_results") or [])
            labels = [str(row["experiment_id"]) for row in tracked]
            values = [float(row["quarterly_mean_mae"]) for row in tracked]
            baseline = [float(row["quarterly_baseline_mae"]) for row in tracked]
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(labels))
            ax.bar(x - 0.18, baseline, width=0.35, label="Baseline")
            ax.bar(x + 0.18, values, width=0.35, label="Candidate")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title("Endpoint/Tier Audit Overview")
            ax.set_ylabel("Quarterly mean MAE")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "susceptible_sidecar":
            payload = dict(result.get("susceptible_sidecar") or {})
            pressure_rows = list(payload.get("pressure_rows") or [])
            years = [int(row["year"]) for row in pressure_rows]
            susceptible_fraction = [float(row["susceptible_fraction"]) for row in pressure_rows]
            incidence_pressure = [float(row["annual_incidence_per_100k_susceptible"]) for row in pressure_rows]
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = ax1.twinx()
            ax1.plot(years, susceptible_fraction, marker="o", label="Susceptible fraction")
            ax2.plot(years, incidence_pressure, marker="o", color="#d95f02", label="Incidence per 100k susceptible")
            ax1.set_title("Annual Susceptible Sidecar")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Susceptible fraction")
            ax2.set_ylabel("Annual incidence / 100k susceptible")
            ax1.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        if kind == "late_leakage_sensitivity":
            payload = dict(result.get("late_leakage_sensitivity") or {})
            proxy_rows = list(payload.get("proxy_rows") or [])
            labels = [str(row["quarter"]) for row in proxy_rows]
            shortfall = [float(row["shortfall_vs_share_carry"]) for row in proxy_rows]
            after_deaths = [
                float(row["shortfall_after_reported_deaths"]) if row.get("shortfall_after_reported_deaths") is not None else np.nan
                for row in proxy_rows
            ]
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(labels))
            ax.plot(x, shortfall, marker="o", label="Shortfall vs share carry")
            if any(math.isfinite(value) for value in after_deaths):
                ax.plot(x, after_deaths, marker="o", label="Shortfall after reported deaths")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_title("Late-era Suppression-linked Leakage Sensitivity")
            ax.set_ylabel("Upper-bound shortfall proxy")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return
        _plot_placeholder(path, title=result["experiment_id"], body="Diagnostic result")
        return
    quarterly_rows = list(result.get("quarterly_rows") or [])
    annual_rows = list(result.get("annual_rows") or [])
    if not quarterly_rows:
        _plot_placeholder(path, title=result["experiment_id"], body="No quarterly rows were produced.")
        return
    quarterly_years = [int(row["holdout_years"][0]) for row in quarterly_rows if row.get("holdout_years")]
    quarterly_candidate = [float(row["candidate"]["mae"]) for row in quarterly_rows]
    quarterly_baseline = [float(row["carry_forward"]["mae"]) for row in quarterly_rows]
    annual_years = [int(row["holdout_years"][0]) for row in annual_rows if row.get("holdout_years")]
    annual_candidate = [float(row["candidate_incidence_error"]) for row in annual_rows]
    annual_baseline = [float(row["baseline_incidence_error"]) for row in annual_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    q_ax = axes[0]
    a_ax = axes[1]
    q_ax.plot(quarterly_years, quarterly_baseline, marker="o", label="Carry-forward")
    q_ax.plot(quarterly_years, quarterly_candidate, marker="o", label=result["experiment_id"])
    q_ax.set_title(f"{result['experiment_id']} Quarterly MAE")
    q_ax.set_xlabel("Holdout year")
    q_ax.set_ylabel("Normalized MAE")
    q_ax.grid(alpha=0.3)
    q_ax.legend()
    a_ax.plot(annual_years, annual_baseline, marker="o", label="Baseline annual incidence")
    a_ax.plot(annual_years, annual_candidate, marker="o", label=result["experiment_id"])
    a_ax.set_title(f"{result['experiment_id']} Annual Incidence Error")
    a_ax.set_xlabel("Holdout year")
    a_ax.set_ylabel("Normalized error")
    a_ax.grid(alpha=0.3)
    a_ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_hazard_curve_graph(result: dict[str, Any], path: Path) -> bool:
    quarterly_rows = list(result.get("quarterly_rows") or [])
    diagnostic_row = next((row for row in quarterly_rows if row.get("transition_diagnostics")), None)
    if not diagnostic_row:
        return False
    diagnostics = dict(diagnostic_row.get("transition_diagnostics") or {})
    if not diagnostics:
        return False
    transitions = [name for name in TRANSITION_NAMES if diagnostics.get(name)]
    if not transitions:
        return False
    fig, axes = plt.subplots(len(transitions), 1, figsize=(13, max(3.0, 2.6 * len(transitions))), constrained_layout=True)
    if len(transitions) == 1:
        axes = [axes]
    holdout_years = list(diagnostic_row.get("holdout_years") or [])
    split_label = ",".join(str(year) for year in holdout_years) if holdout_years else "n/a"
    for ax, transition_name in zip(axes, transitions):
        payload = dict(diagnostics[transition_name])
        train_quarters = list(payload.get("train_quarters") or [])
        train_observed = [float(value) if value is not None else float("nan") for value in payload.get("train_observed") or []]
        train_fitted = [float(value) for value in payload.get("train_fitted") or []]
        train_supported = [bool(value) for value in payload.get("train_supported") or []]
        forecast_quarters = list(payload.get("forecast_quarters") or [])
        forecast_hazards = [float(value) for value in payload.get("forecast_hazards") or []]
        x_train = np.arange(len(train_quarters))
        x_forecast = np.arange(len(train_quarters), len(train_quarters) + len(forecast_quarters))
        ax.plot(x_train, train_observed, color="#777777", linewidth=1.5, marker="o", label="Train observed")
        ax.plot(x_train, train_fitted, color="#1f78b4", linewidth=1.8, marker="o", label="Train fitted")
        supported_x = [x for x, supported in zip(x_train, train_supported) if supported]
        supported_y = [y for y, supported in zip(train_observed, train_supported) if supported and not np.isnan(y)]
        if supported_x:
            ax.scatter(supported_x, supported_y, color="#33a02c", s=30, label="Supported fit points", zorder=5)
        if forecast_quarters:
            ax.plot(x_forecast, forecast_hazards, color="#d95f02", linewidth=1.8, marker="o", label="Forecast")
        tick_positions = list(x_train) + list(x_forecast)
        tick_labels = train_quarters + forecast_quarters
        if tick_positions:
            tick_step = max(1, len(tick_positions) // 12)
            tick_positions = tick_positions[::tick_step]
            tick_labels = tick_labels[::tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        observed_numeric = [value for value in train_observed if not np.isnan(value)]
        y_max = max(observed_numeric + train_fitted + forecast_hazards + [0.0]) if (observed_numeric or train_fitted or forecast_hazards) else 0.0
        ax.set_ylim(-0.02, max(0.05, y_max * 1.08))
        ax.set_ylabel("Hazard")
        ax.set_title(
            f"{transition_name} | support={int(payload.get('support_count', 0))}/{len(train_quarters)} | "
            f"bounds=[{float(payload.get('forecast_lower', 0.0)):.3f}, {float(payload.get('forecast_upper', 0.0)):.3f}]"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{result['experiment_id']} hazard curves | holdout={split_label}", fontsize=14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_observation_curve_graph(result: dict[str, Any], path: Path) -> bool:
    quarterly_rows = list(result.get("quarterly_rows") or [])
    if not quarterly_rows:
        return False
    target_points: dict[str, list[tuple[str, float]]] = {metric: [] for metric in PRIMARY_METRICS}
    candidate_points: dict[str, list[tuple[str, float]]] = {metric: [] for metric in PRIMARY_METRICS}
    baseline_points: dict[str, list[tuple[str, float]]] = {metric: [] for metric in PRIMARY_METRICS}
    for split in quarterly_rows:
        target_rows = list(split.get("holdout_target_rows") or [])
        candidate_rows = list(split.get("candidate_prediction_rows") or [])
        baseline_rows = list(split.get("carry_forward_prediction_rows") or [])
        target_map = {str(row["quarter"]): row for row in target_rows}
        candidate_map = {str(row["quarter"]): row for row in candidate_rows}
        baseline_map = {str(row["quarter"]): row for row in baseline_rows}
        ordered_quarters = sorted(target_map.keys(), key=quarter_sort_key)
        for quarter in ordered_quarters:
            target_row = target_map.get(quarter) or {}
            candidate_row = candidate_map.get(quarter) or {}
            baseline_row = baseline_map.get(quarter) or {}
            for metric_name in PRIMARY_METRICS:
                target_value = target_row.get(metric_name)
                candidate_value = candidate_row.get(metric_name)
                baseline_value = baseline_row.get(metric_name)
                if target_value is not None:
                    target_points[metric_name].append((quarter, float(target_value)))
                if candidate_value is not None:
                    candidate_points[metric_name].append((quarter, float(candidate_value)))
                if baseline_value is not None:
                    baseline_points[metric_name].append((quarter, float(baseline_value)))
    metrics = [metric for metric in PRIMARY_METRICS if target_points[metric]]
    if not metrics:
        return False
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, max(3.0, 2.8 * len(metrics))), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric_name in zip(axes, metrics, strict=True):
        ordered_quarters = sorted({quarter for quarter, _ in target_points[metric_name]}, key=quarter_sort_key)
        index_map = {quarter: idx for idx, quarter in enumerate(ordered_quarters)}
        def _xy(points: list[tuple[str, float]]) -> tuple[list[int], list[float]]:
            xs = [index_map[quarter] for quarter, _ in points if quarter in index_map]
            ys = [float(value) for quarter, value in points if quarter in index_map]
            return xs, ys
        tx, ty = _xy(target_points[metric_name])
        cx, cy = _xy(candidate_points[metric_name])
        bx, by = _xy(baseline_points[metric_name])
        if bx:
            ax.plot(bx, by, marker="o", linewidth=1.5, alpha=0.75, label="Carry-forward")
        if cx:
            ax.plot(cx, cy, marker="o", linewidth=1.8, label=result["experiment_id"])
        if tx:
            ax.plot(tx, ty, marker="o", linewidth=1.5, linestyle="--", color="black", label="Observed")
        ax.set_title(metric_name)
        tick_idx = list(range(0, len(ordered_quarters), max(1, len(ordered_quarters) // 10 or 1)))
        if tick_idx[-1] != len(ordered_quarters) - 1:
            tick_idx.append(len(ordered_quarters) - 1)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([ordered_quarters[idx] for idx in tick_idx], rotation=45, ha="right")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_suite_overview(results: list[dict[str, Any]], path: Path) -> None:
    executed = [row for row in results if row["status"] == "executed" and row["family"] in {"05a", "05b", "repair"}]
    if not executed:
        _plot_placeholder(path, title="Suite overview", body="No executed experiment rows were available.")
        return
    labels = [str(row["experiment_id"]) for row in executed]
    candidate = [float(row["quarterly_summary"]["candidate_mean_mae"]) for row in executed]
    baseline = [float(row["quarterly_summary"]["carry_forward_mean_mae"]) for row in executed]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.45)))
    ax.barh(y - 0.18, baseline, height=0.35, label="Carry-forward")
    ax.barh(y + 0.18, candidate, height=0.35, label="Candidate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean quarterly normalized MAE")
    ax.set_title("TR-V3 Experiment Suite Quarterly Overview")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_suite_split_heatmap(results: list[dict[str, Any]], path: Path) -> None:
    executed = [row for row in results if row["status"] == "executed" and row["family"] in {"05a", "05b", "repair"}]
    if not executed:
        _plot_placeholder(path, title="Split heatmap", body="No executed experiment rows were available.")
        return
    years = sorted({int(split["holdout_years"][0]) for row in executed for split in list(row.get("quarterly_rows") or []) if split.get("holdout_years")})
    matrix = np.full((len(executed), len(years)), np.nan, dtype=np.float64)
    for row_idx, result in enumerate(executed):
        split_map = {
            int(split["holdout_years"][0]): float(split["candidate"]["mae"]) - float(split["carry_forward"]["mae"])
            for split in list(result.get("quarterly_rows") or [])
            if split.get("holdout_years")
        }
        for col_idx, year in enumerate(years):
            if year in split_map:
                matrix[row_idx, col_idx] = split_map[year]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(executed) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdYlGn_r")
    ax.set_yticks(range(len(executed)))
    ax.set_yticklabels([str(row["experiment_id"]) for row in executed])
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_title("Quarterly Holdout Delta MAE (candidate - carry-forward)")
    ax.set_xlabel("Holdout year")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_experiment_spec(
    spec: ExperimentSpec,
    *,
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    availability: dict[str, Any],
    scoring_tiers: set[str],
    archive_run_id: str,
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
    split_local_dense: bool = False,
    full_dense_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if spec.status != "executed":
        return {
            "experiment_id": spec.experiment_id,
            "family": spec.family,
            "description": spec.description,
            "status": spec.status,
            "skip_reason": spec.skip_reason,
        }
    if spec.family == "diagnostic":
        if spec.diagnostic_kind == "contract_ablation":
            comparison = _build_contract_comparison_payload(
                availability["archive_run_id"],
                quarterly_start_year=quarterly_start_year,
                quarterly_end_year=quarterly_end_year,
                quarterly_min_train_years=quarterly_min_train_years,
                horizon_years=horizon_years,
            )
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "diagnostic_only",
                "diagnostic_kind": spec.diagnostic_kind,
                "contracts": list(comparison["contracts"]),
                "contract_comparison": comparison,
            }
        if spec.diagnostic_kind == "purged_dense_contract":
            purged_dense_audit = _build_purged_dense_contract_audit_payload(
                archive_run_id,
                availability=availability,
                quarterly_start_year=quarterly_start_year,
                quarterly_end_year=quarterly_end_year,
                quarterly_min_train_years=quarterly_min_train_years,
                annual_start_year=annual_start_year,
                annual_end_year=annual_end_year,
                annual_min_train_years=annual_min_train_years,
                horizon_years=horizon_years,
            )
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": str(purged_dense_audit["decision"]),
                "diagnostic_kind": spec.diagnostic_kind,
                "purged_dense_contract": purged_dense_audit,
            }
        if spec.diagnostic_kind == "dense_contract_summary":
            dense_contract = _build_imputation_contract_payload(availability["archive_run_id"])
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "diagnostic_only",
                "diagnostic_kind": spec.diagnostic_kind,
                "dense_summary": dense_contract["summary"],
                "imputation_contract": dense_contract,
            }
        if spec.diagnostic_kind == "leakage_identification_contract":
            leakage_contract = _build_leakage_identification_contract_payload(availability["archive_run_id"])
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "diagnostic_only",
                "diagnostic_kind": spec.diagnostic_kind,
                "leakage_contract": leakage_contract,
            }
        if spec.diagnostic_kind == "endpoint_tier_audit":
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "posthoc_pending",
                "diagnostic_kind": spec.diagnostic_kind,
            }
        if spec.diagnostic_kind == "susceptible_sidecar":
            payload = _build_susceptible_sidecar_payload(availability["archive_run_id"])
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "diagnostic_only",
                "diagnostic_kind": spec.diagnostic_kind,
                "susceptible_sidecar": payload,
            }
        if spec.diagnostic_kind == "late_leakage_sensitivity":
            payload = _build_late_leakage_sensitivity_payload(availability["archive_run_id"])
            return {
                "experiment_id": spec.experiment_id,
                "family": spec.family,
                "description": spec.description,
                "status": "executed",
                "decision": "diagnostic_only",
                "diagnostic_kind": spec.diagnostic_kind,
                "late_leakage_sensitivity": payload,
            }
        return {
            "experiment_id": spec.experiment_id,
            "family": spec.family,
            "description": spec.description,
            "status": "executed",
            "decision": "diagnostic_only",
            "diagnostic_kind": spec.diagnostic_kind,
            "availability": availability,
        }
    annual_diag_cfg = AnnualIncidenceConfig(ridge_penalty=0.1, inflow_scale_clip=4.0)
    candidates: list[dict[str, Any]] = []
    lag_values = list(spec.lag_candidates) if spec.lag_candidates else [spec.control_lag]
    if spec.family in {"05a", "repair"}:
        dynamic_grid = _suite_dynamic_grid()
        observation_grid = _suite_observation_grid()
        # Direct-observation repairs ignore the dynamic and observation grids internally,
        # so evaluating the full grid only repeats the same candidate multiple times.
        if str(spec.transition_model).startswith("direct_observation"):
            dynamic_grid = dynamic_grid[:1]
            observation_grid = observation_grid[:1]
        for lag_value in lag_values:
            lagged_spec = ExperimentSpec(**{**asdict(spec), "control_lag": int(lag_value), "lag_candidates": tuple()})
            for dynamic_cfg in dynamic_grid:
                for observation_cfg in observation_grid:
                    quarterly_rows, quarterly_summary = _run_quarterly_experiment(
                        observation_rows,
                        annual_rows,
                        lagged_spec,
                        dynamic_cfg,
                        observation_cfg,
                        None,
                        scoring_tiers,
                        start_year=quarterly_start_year,
                        end_year=quarterly_end_year,
                        min_train_years=quarterly_min_train_years,
                        horizon_years=horizon_years,
                        archive_run_id=archive_run_id,
                        split_local_dense=split_local_dense,
                        full_dense_rows=full_dense_rows,
                    )
                    annual_rows_diag, annual_summary = _run_annual_diagnostics(
                        observation_rows,
                        annual_rows,
                        annual_diag_cfg,
                        start_year=annual_start_year,
                        end_year=annual_end_year,
                        min_train_years=annual_min_train_years,
                        horizon_years=horizon_years,
                    )
                    candidates.append(
                        {
                            "config": {
                                "dynamic_cfg": asdict(dynamic_cfg),
                                "observation_cfg": asdict(observation_cfg),
                                "annual_cfg": asdict(annual_diag_cfg),
                                "control_lag": int(lag_value),
                            },
                            "quarterly_rows": quarterly_rows,
                            "quarterly_summary": quarterly_summary,
                            "annual_rows": annual_rows_diag,
                            "annual_summary": annual_summary,
                        }
                    )
    else:
        for dynamic_cfg in _suite_dynamic_grid():
            for observation_cfg in _suite_observation_grid():
                for annual_cfg in _suite_annual_grid():
                    quarterly_rows, quarterly_summary = _run_quarterly_experiment(
                        observation_rows,
                        annual_rows,
                        spec,
                        dynamic_cfg,
                        observation_cfg,
                        annual_cfg,
                        scoring_tiers,
                        start_year=quarterly_start_year,
                        end_year=quarterly_end_year,
                        min_train_years=quarterly_min_train_years,
                        horizon_years=horizon_years,
                        archive_run_id=archive_run_id,
                        split_local_dense=split_local_dense,
                        full_dense_rows=full_dense_rows,
                    )
                    annual_rows_diag, annual_summary = _run_annual_diagnostics(
                        observation_rows,
                        annual_rows,
                        annual_cfg,
                        start_year=annual_start_year,
                        end_year=annual_end_year,
                        min_train_years=annual_min_train_years,
                        horizon_years=horizon_years,
                    )
                    candidates.append(
                        {
                            "config": {
                                "dynamic_cfg": asdict(dynamic_cfg),
                                "observation_cfg": asdict(observation_cfg),
                                "annual_cfg": asdict(annual_cfg),
                            },
                            "quarterly_rows": quarterly_rows,
                            "quarterly_summary": quarterly_summary,
                            "annual_rows": annual_rows_diag,
                            "annual_summary": annual_summary,
                        }
                    )
    best = _select_best_result(candidates)
    quarterly_summary = dict(best["quarterly_summary"])
    annual_summary = dict(best["annual_summary"])
    keep = (
        float(quarterly_summary["candidate_mean_mae"]) < float(quarterly_summary["carry_forward_mean_mae"])
        and float(quarterly_summary["candidate_worst_mae"]) <= float(quarterly_summary["carry_forward_worst_mae"])
        and float(annual_summary["candidate_mean_incidence_error"]) <= float(annual_summary["baseline_mean_incidence_error"])
    )
    return {
        "experiment_id": spec.experiment_id,
        "family": spec.family,
        "description": spec.description,
        "status": "executed",
        "decision": "keep" if keep else "revert",
        "decision_reason": (
            "Quarterly mean/worst MAE and annual incidence error all beat or match the baseline."
            if keep
            else "The candidate did not beat the baseline simultaneously on quarterly blocked-time error and annual incidence error."
        ),
        "best_candidate": best["config"],
        "quarterly_rows": best["quarterly_rows"],
        "quarterly_summary": quarterly_summary,
        "annual_rows": best["annual_rows"],
        "annual_summary": annual_summary,
        "candidate_count": len(candidates),
        "leakage_contract": (
            _build_leakage_identification_contract_payload(availability["archive_run_id"])
            if spec.experiment_id == "EXP-L1"
            else None
        ),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    availability = dict(payload["availability"])
    quarterly_availability = dict(availability["quarterly"])
    annual_availability = dict(availability["annual"])
    dense_contract = dict(payload.get("dense_contract") or {})
    benchmark_candidate_id = payload.get("benchmark_candidate_experiment_id")
    track_registry = dict(payload.get("track_registry") or {})
    diagnostics = [row for row in payload["results"] if row["status"] == "executed" and row["family"] == "diagnostic"]
    executed = [row for row in payload["results"] if row["status"] == "executed" and row["family"] in {"05a", "05b", "repair"}]
    skipped = [row for row in payload["results"] if row["status"] == "skipped"]
    lines = [
        "# TR-V3 Experiment Suite Report",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Quarterly execution contract: `{payload['quarterly_execution_contract']}`",
        f"- Quarterly years requested: `{payload['quarterly_window']['start_year']}` -> `{payload['quarterly_window']['end_year']}`",
        f"- Annual years requested: `{payload['annual_window']['start_year']}` -> `{payload['annual_window']['end_year']}`",
        f"- Default quarterly benchmark candidate: `{benchmark_candidate_id or 'n/a'}`",
        f"- Predictive exact candidate: `{track_registry.get('predictive_exact_candidate_id', 'n/a')}`",
        f"- Predictive dense candidate: `{track_registry.get('predictive_dense_candidate_id', 'n/a')}`",
        f"- Mechanistic research anchor: `{track_registry.get('mechanistic_research_anchor_id', 'n/a')}`",
        "",
        "## Data Contract",
        "",
        f"- Exact quarterly observation rows used for executable loops: `{quarterly_availability['exact_observation_row_count']}`",
        f"- Exact quarterly observation years: `{', '.join(str(year) for year in quarterly_availability['exact_observation_years'])}`",
        f"- Bridge-complete quarterly years available for diagnostics: `{', '.join(str(year) for year in quarterly_availability['bridge_complete_years'])}`",
        f"- Quarterly diagnosis-flow archive rows: `{quarterly_availability['flow_metric_row_count']}`",
        f"- Missing-data ladder: `{', '.join(payload['availability']['missing_data_ladder'])}`",
        "",
    ]
    if dense_contract:
        summary = dict(dense_contract.get("summary") or {})
        quarter_range = list(summary.get("quarter_range") or ["n/a", "n/a"])
        lines.extend(
            [
                "## Dense Contract",
                "",
                f"- Dense quarter count: `{summary.get('quarter_count', 0)}`",
                f"- Dense quarter range: `{quarter_range[0]}` -> `{quarter_range[1]}`",
                f"- Score-eligible quarters under dense contract: `{summary.get('score_eligible_quarters', 0)}`",
                f"- Score-eligible years under dense contract: `{', '.join(str(year) for year in summary.get('score_eligible_years', []))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Diagnostics",
            "",
            "| Experiment | Decision | Graph | Notes |",
            "|---|---|---|---|",
        ]
    )
    for row in diagnostics:
        note = ""
        if row.get("diagnostic_kind") == "contract_ablation":
            contracts = list(row.get("contracts") or [])
            note = "; ".join(f"{item['name']}={item['score_eligible_quarters']}" for item in contracts)
        elif row.get("diagnostic_kind") == "dense_contract_summary":
            summary = dict(row.get('dense_summary') or {})
            note = f"score_eligible_quarters={summary.get('score_eligible_quarters', 0)}; diagnosed_methods={dict(summary.get('diagnosed_imputation_method_counts') or {})}"
        elif row.get("diagnostic_kind") == "leakage_identification_contract":
            note = dict(row.get("leakage_contract") or {}).get("decision", {}).get("why", "")
        elif row.get("diagnostic_kind") == "missingness_map":
            note = "quarterly and annual availability heatmap"
        elif row.get("diagnostic_kind") == "purged_dense_contract":
            payload_v1 = dict(row.get("purged_dense_contract") or {})
            note = (
                f"legacy_winner={payload_v1.get('legacy_winner', '')}; "
                f"purged_winner={payload_v1.get('purged_winner', '')}; "
                f"winner_mae_delta={float(payload_v1.get('winner_mae_delta') or 0.0):.6f}"
            )
        elif row.get("diagnostic_kind") == "endpoint_tier_audit":
            tracked = list(dict(row.get("endpoint_tier_audit") or {}).get("tracked_results") or [])
            note = f"tracked_results={len(tracked)}"
        elif row.get("diagnostic_kind") == "susceptible_sidecar":
            payload_s1 = dict(row.get("susceptible_sidecar") or {})
            note = dict(payload_s1.get("decision") or {}).get("why", "")
        elif row.get("diagnostic_kind") == "late_leakage_sensitivity":
            payload_l2 = dict(row.get("late_leakage_sensitivity") or {})
            note = dict(payload_l2.get("decision") or {}).get("why", "")
        lines.append(f"| {row['experiment_id']} | {row['decision']} | `{row['graph_file']}` | {note} |")
    lines.extend(
        [
            "",
            "## Executed Experiments",
            "",
            "| Experiment | Family | Role | Decision | Quarterly mean MAE | Carry-forward mean MAE | Annual incidence error | Baseline annual incidence error | Candidates | Graph | Hazard curves | Observation curves |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in executed:
        role = ""
        if row["experiment_id"] == track_registry.get("predictive_exact_candidate_id"):
            role = "predictive_exact_candidate"
        elif row["experiment_id"] == track_registry.get("predictive_dense_candidate_id"):
            role = "predictive_dense_candidate"
        elif row["experiment_id"] == track_registry.get("mechanistic_research_anchor_id"):
            role = "mechanistic_research_anchor"
        lines.append(
            f"| {row['experiment_id']} | {row['family']} | {role} | {row['decision']} | "
            f"{float(row['quarterly_summary']['candidate_mean_mae']):.6f} | "
            f"{float(row['quarterly_summary']['carry_forward_mean_mae']):.6f} | "
            f"{float(row['annual_summary']['candidate_mean_incidence_error']):.6f} | "
            f"{float(row['annual_summary']['baseline_mean_incidence_error']):.6f} | "
            f"{int(row['candidate_count'])} | `{row['graph_file']}` | `{row.get('hazard_graph_file') or ''}` | `{row.get('observation_curve_graph_file') or ''}` |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Availability graph: `{payload['availability_graph_file']}`",
            f"- Suite overview graph: `{payload['overview_graph_file']}`",
            f"- Split heatmap graph: `{payload['heatmap_graph_file']}`",
            f"- Canonical 14 vs benchmark artifact: `canonical_14_vs_benchmark.md`",
            f"- Track registry artifact: `phase3_track_registry.md`",
            "",
            "## Skipped / Deferred Experiments",
            "",
            "| Experiment | Status | Reason |",
            "|---|---|---|",
        ]
    )
    for row in skipped:
        lines.append(f"| {row['experiment_id']} | {row['status']} | {row.get('skip_reason') or ''} |")
    lines.extend(
        [
            "",
            "## Annual Availability Counts",
            "",
            "| Metric | Exact | Bridge | Rule-based | Latent | Rejected | Missing |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric_name in annual_availability["metrics"]:
        counts = dict(annual_availability["tier_counts"][metric_name])
        lines.append(
            f"| {metric_name} | {counts.get('exact_observed', 0)} | {counts.get('bridge_observed', 0)} | "
            f"{counts.get('rule_based_extrapolated', 0)} | {counts.get('latent_imputed', 0)} | "
            f"{counts.get('rejected_or_quarantined', 0)} | {counts.get('missing', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_contract_comparison(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-B0 Contract Comparison",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        "",
        "| Contract | Training rows | Score-eligible quarters | Score-eligible years |",
        "|---|---:|---:|---|",
    ]
    for row in payload["contracts"]:
        lines.append(
            f"| {row['name']} | {int(row['training_row_count'])} | {int(row['score_eligible_quarters'])} | {', '.join(str(year) for year in row['score_eligible_years'])} |"
        )
    lines.extend(
        [
            "",
            "## Split Support",
            "",
        ]
    )
    for row in payload["contracts"]:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                "| Train end | Holdout | Train rows | Holdout rows | Scored metrics |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for split in row["support"]["split_rows"]:
            lines.append(
                f"| {split['train_end_year']} | {', '.join(str(year) for year in split['holdout_years'])} | {split['train_row_count']} | {split['holdout_row_count']} | {split['scored_metric_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _markdown_imputation_contract(payload: dict[str, Any]) -> str:
    summary = dict(payload["summary"])
    lines = [
        "# EXP-B2 Imputation Contract",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Quarter range: `{summary['quarter_range'][0]}` -> `{summary['quarter_range'][1]}`",
        f"- Score-eligible quarters: `{summary['score_eligible_quarters']}`",
        "",
        "## Diagnosed Imputation Methods",
        "",
        "| Method | Count |",
        "|---|---:|",
    ]
    for method, count in dict(summary.get("diagnosed_imputation_method_counts") or {}).items():
        lines.append(f"| {method} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Early Rows",
            "",
            "| Quarter | Diagnosed | Diagnosed tier | Diagnosed method | Alive on ART | ART tier | Diagnosis flow | Flow tier | Score eligible |",
            "|---|---:|---|---|---:|---|---:|---|---|",
        ]
    )
    for row in payload["early_rows"]:
        lines.append(
            f"| {row['quarter']} | {row['diagnosed_plhiv']:.0f} | {row['diagnosed_plhiv_tier']} | {row['diagnosed_plhiv_imputation_method']} | {row['alive_on_art']:.0f} | {row['alive_on_art_tier']} | {row['new_diagnosed_cases_period']:.0f} | {row['new_diagnosed_cases_period_tier']} | {row['score_eligible']} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_leakage_contract(payload: dict[str, Any]) -> str:
    decision = dict(payload["decision"])
    lines = [
        "# EXP-L1 Leakage Identification Contract",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Status: `{decision['exp_l1_status']}`",
        f"- Why: {decision['why']}",
        "",
        "## Allowed Blocks",
        "",
        "| Block | Status |",
        "|---|---|",
    ]
    for name, status in dict(payload["allowed_blocks"]).items():
        lines.append(f"| {name} | {status} |")
    lines.extend(
        [
            "",
            "## Yearly Support",
            "",
            "| Year | Diagnosed stock quarters | ART stock quarters | Diagnosis-flow quarters | Can score net ART leakage | Can score D->L separately |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["yearly_support"]:
        lines.append(
            f"| {row['year']} | {row['diagnosed_stock_quarters']} | {row['art_stock_quarters']} | {row['diagnosis_flow_quarters']} | {row['can_score_art_leakage_net']} | {row['can_score_diagnosed_leakage_separately']} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_susceptible_sidecar(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    contract = dict(payload.get("contract") or {})
    lines = [
        "# EXP-S1-A1 Annual Susceptible Sidecar",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Status: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        f"- Joint annual support years: `{len(dict(contract.get('annual_years') or {}).get('joint_support_years') or [])}`",
        "",
        "| Year | Susceptible proxy | Susceptible fraction | PLHIV fraction | Annual new infections | Annual incidence / susceptible | Annual incidence / 100k susceptible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("pressure_rows") or []):
        lines.append(
            f"| {row['year']} | {float(row['susceptible_proxy']):.1f} | {float(row['susceptible_fraction']):.6f} | "
            f"{float(row['plhiv_fraction']):.6f} | {float(row['annual_new_infections']):.1f} | "
            f"{float(row['annual_incidence_per_susceptible']):.8f} | {float(row['annual_incidence_per_100k_susceptible']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_late_leakage_sensitivity(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    lines = [
        "# EXP-L2-A1 Late Leakage Sensitivity",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Status: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "| Quarter | Previous share | Current share | Carry suppressed | Observed suppressed | Shortfall vs share carry | Shortfall after reported deaths |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("proxy_rows") or []):
        after_deaths = row.get("shortfall_after_reported_deaths")
        after_deaths_str = "" if after_deaths is None else f"{float(after_deaths):.3f}"
        lines.append(
            f"| {row['quarter']} | {float(row['previous_share']):.6f} | {float(row['current_share']):.6f} | "
            f"{float(row['carry_suppressed']):.3f} | {float(row['observed_suppressed']):.3f} | "
            f"{float(row['shortfall_vs_share_carry']):.3f} | {after_deaths_str} |"
        )
    if payload.get("yearly_summary"):
        lines.extend(
            [
                "",
                "| Year | Quarter count | Mean shortfall | Total shortfall | Total shortfall after reported deaths |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in list(payload.get("yearly_summary") or []):
            after_deaths = row.get("total_shortfall_after_reported_deaths")
            after_deaths_str = "" if after_deaths is None else f"{float(after_deaths):.3f}"
            lines.append(
                f"| {row['year']} | {int(row['quarter_count'])} | {float(row['mean_shortfall_vs_share_carry']):.3f} | "
                f"{float(row['total_shortfall_vs_share_carry']):.3f} | {after_deaths_str} |"
            )
    return "\n".join(lines) + "\n"


def _markdown_purged_dense_contract(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-V1 Purged Dense Contract Audit",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Legacy dense winner: `{payload['legacy_winner']}`",
        f"- Purged dense winner: `{payload['purged_winner']}`",
        f"- Winner MAE delta: `{float(payload['winner_mae_delta']):.6f}`",
        "",
    ]
    for section_name, rows in (
        ("Exact Reference", payload["exact_reference"]),
        ("Legacy Dense", payload["legacy_dense"]),
        ("Purged Dense", payload["purged_dense"]),
    ):
        lines.extend(
            [
                f"## {section_name}",
                "",
                "| Experiment | Decision | Quarterly mean MAE | Baseline | Annual mean error |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['experiment_id']} | {row['decision']} | {float(row['quarterly_mean_mae']):.6f} | "
                f"{float(row['quarterly_baseline_mae']):.6f} | {float(row['annual_mean_error']):.6f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _markdown_endpoint_tier_audit(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-V2 Endpoint/Tier Audit",
        "",
        f"- Quarterly contract: `{payload['quarterly_contract']}`",
        "",
    ]
    for result in payload["tracked_results"]:
        audit = dict(result.get("endpoint_audit_summary") or {})
        candidate = dict(audit.get("candidate") or {})
        train_support = dict(audit.get("train_support_counts") or {})
        holdout_support = dict(audit.get("holdout_support_counts") or {})
        lines.extend(
            [
                f"## {result['experiment_id']}",
                "",
                f"- Family: `{result['family']}`",
                f"- Decision: `{result['decision']}`",
                f"- Quarterly mean MAE: `{float(result['quarterly_mean_mae']):.6f}`",
                f"- Quarterly baseline MAE: `{float(result['quarterly_baseline_mae']):.6f}`",
                f"- Suppression honesty flags: `{dict(audit.get('suppression_honesty_flags') or {})}`",
                "",
                "| Metric | Raw MAE | Normalized MAE | Train exact | Train bridge | Holdout exact | Holdout bridge |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric_name in AUDIT_METRICS:
            metric_row = dict(dict(candidate.get("by_metric") or {}).get(metric_name) or {})
            train_row = dict(train_support.get(metric_name) or {})
            holdout_row = dict(holdout_support.get(metric_name) or {})
            raw_mae = "" if metric_row.get("raw_mae") is None else f"{float(metric_row['raw_mae']):.3f}"
            normalized_mae = "" if metric_row.get("normalized_mae") is None else f"{float(metric_row['normalized_mae']):.6f}"
            lines.append(
                f"| {metric_name} | {raw_mae} | {normalized_mae} | "
                f"{int(train_row.get('exact_observed') or 0)} | {int(train_row.get('bridge_observed') or 0)} | "
                f"{int(holdout_row.get('exact_observed') or 0)} | {int(holdout_row.get('bridge_observed') or 0)} |"
            )
        lines.append("")
        lines.extend(
            [
                "| Metric | Tier | Raw MAE | Normalized MAE | Count |",
                "|---|---|---:|---:|---:|",
            ]
        )
        metric_tier_rows = dict(candidate.get("by_metric_tier") or {})
        for metric_name in AUDIT_METRICS:
            for tier in AUDIT_TIERS:
                tier_row = dict(dict(metric_tier_rows.get(metric_name) or {}).get(tier) or {})
                raw_mae = "" if tier_row.get("raw_mae") is None else f"{float(tier_row['raw_mae']):.3f}"
                normalized_mae = "" if tier_row.get("normalized_mae") is None else f"{float(tier_row['normalized_mae']):.6f}"
                lines.append(
                    f"| {metric_name} | {tier} | {raw_mae} | {normalized_mae} | {int(tier_row.get('count') or 0)} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_canonical_benchmark_comparison(
    results: list[dict[str, Any]],
    *,
    benchmark_candidate_id: str | None,
) -> dict[str, Any] | None:
    if benchmark_candidate_id is None:
        return None
    result_map = {str(row["experiment_id"]): row for row in results}
    benchmark_row = result_map.get(benchmark_candidate_id)
    if not benchmark_row or benchmark_row.get("status") != "executed":
        return None
    benchmark_quarterly = dict(benchmark_row["quarterly_summary"])
    benchmark_annual = dict(benchmark_row["annual_summary"])
    rows: list[dict[str, Any]] = []
    ordered_ids = [benchmark_candidate_id, *CANONICAL_14_EXPERIMENT_IDS]
    seen: set[str] = set()
    for experiment_id in ordered_ids:
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        row = result_map.get(experiment_id)
        if row is None:
            continue
        entry = {
            "experiment_id": str(row["experiment_id"]),
            "family": str(row["family"]),
            "status": str(row["status"]),
            "decision": str(row.get("decision") or ""),
        }
        if row.get("status") == "executed" and row.get("family") in {"05a", "05b", "repair"}:
            quarterly = dict(row["quarterly_summary"])
            annual = dict(row["annual_summary"])
            entry.update(
                {
                    "quarterly_mean_mae": float(quarterly["candidate_mean_mae"]),
                    "annual_mean_incidence_error": float(annual["candidate_mean_incidence_error"]),
                    "delta_vs_benchmark_quarterly_mean_mae": float(quarterly["candidate_mean_mae"]) - float(benchmark_quarterly["candidate_mean_mae"]),
                    "delta_vs_benchmark_annual_error": float(annual["candidate_mean_incidence_error"]) - float(benchmark_annual["candidate_mean_incidence_error"]),
                }
            )
        else:
            entry.update(
                {
                    "quarterly_mean_mae": None,
                    "annual_mean_incidence_error": None,
                    "delta_vs_benchmark_quarterly_mean_mae": None,
                    "delta_vs_benchmark_annual_error": None,
                }
            )
        rows.append(entry)
    return {
        "benchmark_candidate_id": str(benchmark_candidate_id),
        "benchmark_quarterly_mean_mae": float(benchmark_quarterly["candidate_mean_mae"]),
        "benchmark_annual_mean_incidence_error": float(benchmark_annual["candidate_mean_incidence_error"]),
        "rows": rows,
    }


def _markdown_canonical_benchmark_comparison(payload: dict[str, Any]) -> str:
    lines = [
        "# Canonical 14 vs Benchmark Candidate",
        "",
        f"- Benchmark candidate: `{payload['benchmark_candidate_id']}`",
        f"- Benchmark quarterly mean MAE: `{float(payload['benchmark_quarterly_mean_mae']):.6f}`",
        f"- Benchmark annual incidence error: `{float(payload['benchmark_annual_mean_incidence_error']):.6f}`",
        "",
        "| Experiment | Family | Status | Decision | Quarterly mean MAE | Delta vs benchmark | Annual incidence error | Annual delta vs benchmark |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        quarterly = "" if row["quarterly_mean_mae"] is None else f"{float(row['quarterly_mean_mae']):.6f}"
        quarterly_delta = "" if row["delta_vs_benchmark_quarterly_mean_mae"] is None else f"{float(row['delta_vs_benchmark_quarterly_mean_mae']):+.6f}"
        annual = "" if row["annual_mean_incidence_error"] is None else f"{float(row['annual_mean_incidence_error']):.6f}"
        annual_delta = "" if row["delta_vs_benchmark_annual_error"] is None else f"{float(row['delta_vs_benchmark_annual_error']):+.6f}"
        lines.append(
            f"| {row['experiment_id']} | {row['family']} | {row['status']} | {row['decision']} | "
            f"{quarterly} | {quarterly_delta} | {annual} | {annual_delta} |"
        )
    return "\n".join(lines) + "\n"


def _build_track_registry(results: list[dict[str, Any]], *, quarterly_contract: str) -> dict[str, Any]:
    result_map = {str(row["experiment_id"]): row for row in results}
    exact_id = default_predictive_candidate_id("exact_only")
    dense_id = default_predictive_candidate_id("dense_train_observed_score")
    mechanistic_id = mechanistic_research_anchor_id()
    active_predictive_id = exact_id if quarterly_contract == "exact_only" else dense_id
    rows = []
    for role, experiment_id in (
        ("predictive_exact_candidate", exact_id),
        ("predictive_dense_candidate", dense_id),
        ("mechanistic_research_anchor", mechanistic_id),
    ):
        row = result_map.get(experiment_id)
        if row is None:
            continue
        entry = {
            "role": role,
            "experiment_id": experiment_id,
            "status": str(row.get("status") or ""),
            "decision": str(row.get("decision") or ""),
        }
        if row.get("status") == "executed":
            quarterly = dict(row["quarterly_summary"])
            annual = dict(row["annual_summary"])
            entry.update(
                {
                    "quarterly_mean_mae": float(quarterly["candidate_mean_mae"]),
                    "quarterly_baseline_mean_mae": float(quarterly["carry_forward_mean_mae"]),
                    "annual_mean_incidence_error": float(annual["candidate_mean_incidence_error"]),
                    "annual_baseline_incidence_error": float(annual["baseline_mean_incidence_error"]),
                }
            )
        rows.append(entry)
    return {
        "quarterly_contract": str(quarterly_contract),
        "active_predictive_candidate_id": active_predictive_id,
        "predictive_exact_candidate_id": exact_id,
        "predictive_dense_candidate_id": dense_id,
        "mechanistic_research_anchor_id": mechanistic_id,
        "rows": rows,
    }


def _markdown_track_registry(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Track Registry",
        "",
        f"- Quarterly contract: `{payload['quarterly_contract']}`",
        f"- Active predictive candidate: `{payload['active_predictive_candidate_id']}`",
        f"- Predictive exact candidate: `{payload['predictive_exact_candidate_id']}`",
        f"- Predictive dense candidate: `{payload['predictive_dense_candidate_id']}`",
        f"- Mechanistic research anchor: `{payload['mechanistic_research_anchor_id']}`",
        "",
        "| Role | Experiment | Status | Decision | Quarterly mean MAE | Baseline quarterly MAE | Annual incidence error | Baseline annual error |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        q = "" if row.get("quarterly_mean_mae") is None else f"{float(row['quarterly_mean_mae']):.6f}"
        qb = "" if row.get("quarterly_baseline_mean_mae") is None else f"{float(row['quarterly_baseline_mean_mae']):.6f}"
        a = "" if row.get("annual_mean_incidence_error") is None else f"{float(row['annual_mean_incidence_error']):.6f}"
        ab = "" if row.get("annual_baseline_incidence_error") is None else f"{float(row['annual_baseline_incidence_error']):.6f}"
        lines.append(
            f"| {row['role']} | {row['experiment_id']} | {row['status']} | {row['decision']} | {q} | {qb} | {a} | {ab} |"
        )
    return "\n".join(lines) + "\n"


def run_tr_v3_experiment_suite(
    *,
    run_id: str,
    archive_run_id: str,
    quarterly_contract: str = "exact_only",
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    if quarterly_contract == "exact_only":
        observation_rows = build_quarterly_observation_rows(archive_run_id)
        scoring_tiers = {"exact_observed"}
        quarterly_execution_contract = "exact_quarterly_only_for_model_loop; bridge_and_annual_history_for_availability_and_diagnostics"
        dense_contract = None
        split_local_dense = False
        full_dense_rows = None
    elif quarterly_contract == "dense_train_observed_score":
        dense_contract = _build_dense_contract_payload(archive_run_id)
        observation_rows = list(dense_contract["rows"])
        scoring_tiers = {"exact_observed", "bridge_observed"}
        quarterly_execution_contract = "dense_quarterly_train_on_exact_bridge_rule_based_with_split_local_purged_rebuild; score_holdout_only_on_exact_and_bridge"
        split_local_dense = True
        full_dense_rows = list(dense_contract["rows"])
    else:
        raise ValueError(f"Unsupported quarterly_contract: {quarterly_contract}")
    annual_rows = build_annual_anchor_rows(archive_run_id)
    availability = _build_availability_payload(archive_run_id)
    spec_list = build_experiment_suite_specs()
    results = [
        _evaluate_experiment_spec(
            spec,
            observation_rows=observation_rows,
            annual_rows=annual_rows,
            availability=availability,
            scoring_tiers=scoring_tiers,
            archive_run_id=archive_run_id,
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
            split_local_dense=split_local_dense,
            full_dense_rows=full_dense_rows,
        )
        for spec in spec_list
    ]
    for idx, result in enumerate(results):
        if str(result.get("diagnostic_kind") or "") != "endpoint_tier_audit":
            continue
        results[idx] = {
            **result,
            "decision": "diagnostic_only",
            "endpoint_tier_audit": _build_endpoint_tier_audit_payload(results, quarterly_contract=quarterly_contract),
        }
    benchmark_candidate_id = default_predictive_candidate_id(quarterly_contract)
    canonical_benchmark_comparison = _build_canonical_benchmark_comparison(
        results,
        benchmark_candidate_id=benchmark_candidate_id,
    )
    track_registry = _build_track_registry(results, quarterly_contract=quarterly_contract)
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    availability_graph = analysis_dir / "B1_missingness_map.png"
    _save_missingness_heatmap(availability, availability_graph)
    for result in results:
        graph_path = analysis_dir / f"{result['experiment_id']}.png"
        _save_experiment_graph(result, graph_path)
        result["graph_file"] = graph_path.name
        if result.get("status") == "executed" and result.get("family") == "repair":
            hazard_graph_path = analysis_dir / f"{result['experiment_id']}_hazard_curves.png"
            if _save_hazard_curve_graph(result, hazard_graph_path):
                result["hazard_graph_file"] = hazard_graph_path.name
            if result["experiment_id"] in {
                track_registry["predictive_exact_candidate_id"],
                track_registry["predictive_dense_candidate_id"],
                track_registry["mechanistic_research_anchor_id"],
            }:
                curve_graph_path = analysis_dir / f"{result['experiment_id']}_observation_curves.png"
                if _save_observation_curve_graph(result, curve_graph_path):
                    result["observation_curve_graph_file"] = curve_graph_path.name
        if result["experiment_id"] == "EXP-B0" and result.get("contract_comparison"):
            write_json(analysis_dir / "exp_b0_contract_comparison.json", result["contract_comparison"])
            (analysis_dir / "exp_b0_contract_comparison.md").write_text(
                _markdown_contract_comparison(result["contract_comparison"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-B2" and result.get("imputation_contract"):
            write_json(analysis_dir / "exp_b2_imputation_contract.json", result["imputation_contract"])
            (analysis_dir / "exp_b2_imputation_contract.md").write_text(
                _markdown_imputation_contract(result["imputation_contract"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-V1" and result.get("purged_dense_contract"):
            write_json(analysis_dir / "exp_v1_purged_dense_contract.json", result["purged_dense_contract"])
            (analysis_dir / "exp_v1_purged_dense_contract.md").write_text(
                _markdown_purged_dense_contract(result["purged_dense_contract"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-V2" and result.get("endpoint_tier_audit"):
            write_json(analysis_dir / "exp_v2_endpoint_tier_audit.json", result["endpoint_tier_audit"])
            (analysis_dir / "exp_v2_endpoint_tier_audit.md").write_text(
                _markdown_endpoint_tier_audit(result["endpoint_tier_audit"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-L1" and result.get("leakage_contract"):
            write_json(analysis_dir / "exp_l1_leakage_identification_contract.json", result["leakage_contract"])
            (analysis_dir / "exp_l1_leakage_identification_contract.md").write_text(
                _markdown_leakage_contract(result["leakage_contract"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-S1-A1" and result.get("susceptible_sidecar"):
            write_json(analysis_dir / "exp_s1_a1_susceptible_sidecar.json", result["susceptible_sidecar"])
            (analysis_dir / "exp_s1_a1_susceptible_sidecar.md").write_text(
                _markdown_susceptible_sidecar(result["susceptible_sidecar"]),
                encoding="utf-8",
            )
        if result["experiment_id"] == "EXP-L2-A1" and result.get("late_leakage_sensitivity"):
            write_json(analysis_dir / "exp_l2_a1_late_leakage_sensitivity.json", result["late_leakage_sensitivity"])
            (analysis_dir / "exp_l2_a1_late_leakage_sensitivity.md").write_text(
                _markdown_late_leakage_sensitivity(result["late_leakage_sensitivity"]),
                encoding="utf-8",
            )
    overview_graph = analysis_dir / "suite_overview_quarterly_mae.png"
    _save_suite_overview(results, overview_graph)
    heatmap_graph = analysis_dir / "suite_split_delta_heatmap.png"
    _save_suite_split_heatmap(results, heatmap_graph)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "archive_run_id": archive_run_id,
        "benchmark_candidate_experiment_id": benchmark_candidate_id,
        "quarterly_contract": quarterly_contract,
        "quarterly_execution_contract": quarterly_execution_contract,
        "quarterly_window": {
            "start_year": quarterly_start_year,
            "end_year": quarterly_end_year,
            "min_train_years": quarterly_min_train_years,
            "horizon_years": horizon_years,
        },
        "annual_window": {
            "start_year": annual_start_year,
            "end_year": annual_end_year,
            "min_train_years": annual_min_train_years,
            "horizon_years": horizon_years,
        },
        "availability": availability,
        "dense_contract": dense_contract,
        "results": results,
        "canonical_benchmark_comparison": canonical_benchmark_comparison,
        "track_registry": track_registry,
        "availability_graph_file": availability_graph.name,
        "overview_graph_file": overview_graph.name,
        "heatmap_graph_file": heatmap_graph.name,
    }
    write_json(analysis_dir / "tr_v3_experiment_suite_report.json", payload)
    (analysis_dir / "tr_v3_experiment_suite_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    if canonical_benchmark_comparison is not None:
        write_json(analysis_dir / "canonical_14_vs_benchmark.json", canonical_benchmark_comparison)
        (analysis_dir / "canonical_14_vs_benchmark.md").write_text(
            _markdown_canonical_benchmark_comparison(canonical_benchmark_comparison),
            encoding="utf-8",
        )
    write_json(analysis_dir / "phase3_track_registry.json", track_registry)
    (analysis_dir / "phase3_track_registry.md").write_text(
        _markdown_track_registry(track_registry),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr-v3-experiment-suite")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=_latest_standard_archive_run())
    parser.add_argument("--quarterly-contract", choices=("exact_only", "dense_train_observed_score"), default="exact_only")
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_experiment_suite(
        run_id=args.run_id,
        archive_run_id=args.archive_run_id,
        quarterly_contract=args.quarterly_contract,
        quarterly_start_year=args.quarterly_start_year,
        quarterly_end_year=args.quarterly_end_year,
        quarterly_min_train_years=args.quarterly_min_train_years,
        annual_start_year=args.annual_start_year,
        annual_end_year=args.annual_end_year,
        annual_min_train_years=args.annual_min_train_years,
        horizon_years=args.horizon_years,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
