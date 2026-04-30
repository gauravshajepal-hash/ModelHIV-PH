from __future__ import annotations

from typing import Any

from .aggregation_diagnostic import run_an03a
from .age_research import run_age_01a, run_age_01b, run_age_01c
from .analytics import run_an01a, run_an01b, run_an01c
from .artifacts import build_transition_research_context
from .diag_bundle_search import run_diag_02b
from .incidence_family_search import run_diag_02c
from .decomposition import run_decomp_01a, run_decomp_01b, run_decomp_01c, run_decomp_01d, run_decomp_01e, run_decomp_01f
from .denoising_diagnostic import run_an03b
from .hierarchical_autoresearch import run_hmba_00, run_hmba_01, run_hmba_02, run_hmba_03, run_hmba_03a
from .integrated_autoresearch import run_phase3_v2_int
from .strict_diagnosis_kernel_research import run_diag_01a, run_diag_01b, run_diag_02a
from .strict_spec_gap_audit import run_an03c
from .analytics import run_an02a, run_an02b, run_an02c
from .peak_windows import run_peak_01a, run_peak_01b, run_peak_01c, run_peak_01d, run_peak_01e, run_peak_01f
from .registry import get_experiment_id_for_cli
from .tr_v2 import (
    run_tr_v2_00,
    run_tr_v2_01,
    run_tr_v2_02,
    run_tr_v2_03,
    write_tr_v2_benchmark_dashboard_report,
    write_tr_v2_early_history_partial_report,
    write_tr_v2_rolling_origin_report,
)
from .transition_engine import run_kp_01a, run_mech_01a, run_mech_01b, run_mech_01c, run_mech_01d, run_mech_01e

EXPERIMENT_DISPATCH = {
    "AN-01A-national-yearly-factor-evolution": run_an01a,
    "AN-01B-regional-yearly-factor-evolution": run_an01b,
    "AN-01C-factor-importance-drift-report": run_an01c,
    "AN-02A-factor-to-transition-map": run_an02a,
    "AN-02B-factor-to-kp-map": run_an02b,
    "AN-02C-kp-transition-relevance-drift": run_an02c,
    "AN-03A-phase2-aggregation-loss-diagnostic": run_an03a,
    "AN-03B-phase2-partial-denoising-diagnostic": run_an03b,
    "AN-03C-phase3-strict-spec-gap-audit": run_an03c,
    "DIAG-01A-strict-diagnosis-kernel-blocked-time": run_diag_01a,
    "DIAG-01B-diagnosis-budget-sweep": run_diag_01b,
    "DIAG-02A-integrated-champion-strict-promotion": run_diag_02a,
    "DIAG-02B-blocked-time-incidence-ud-bundle-search": run_diag_02b,
    "DIAG-02C-blocked-time-incidence-family-search": run_diag_02c,
    "HMBA-00-hierarchical-contract-freeze": run_hmba_00,
    "HMBA-01-module-local-direct-bundle-promotion": run_hmba_01,
    "HMBA-02-hierarchical-geography-layer": run_hmba_02,
    "HMBA-03-joint-hierarchical-autoresearch-loop": run_hmba_03,
    "HMBA-03A-interpretation-dashboard": run_hmba_03a,
    "AGE-01A-age-evidence-audit": run_age_01a,
    "AGE-01B-youth-diagnosis-modifier": run_age_01b,
    "AGE-01C-youth-downstream-modifier": run_age_01c,
    "MECH-01A-national-udavl-baseline": run_mech_01a,
    "MECH-01B-mesoscopic-transition-helpers": run_mech_01b,
    "MECH-01C-residual-transition-helpers": run_mech_01c,
    "MECH-01D-diagnosis-locked-residual-helpers": run_mech_01d,
    "MECH-01E-anchored-downstream-residual-helpers": run_mech_01e,
    "KP-01A-national-kp-lite-overlay": run_kp_01a,
    "DECOMP-01A-transition-channel-decomposition": run_decomp_01a,
    "DECOMP-01B-channel-driver-coupling": run_decomp_01b,
    "DECOMP-01C-fused-mechanistic-forecast": run_decomp_01c,
    "DECOMP-01D-skill-gated-fused-forecast": run_decomp_01d,
    "DECOMP-01E-loo-gated-fused-forecast": run_decomp_01e,
    "DECOMP-01F-reverse-grasp-peak-clusters": run_decomp_01f,
    "PEAK-01A-regional-kp-window-detector": run_peak_01a,
    "PEAK-01B-detector-gated-fused-forecast": run_peak_01b,
    "PEAK-01C-region-only-window-detector": run_peak_01c,
    "PEAK-01D-region-only-gated-fused-forecast": run_peak_01d,
    "PEAK-01E-region-plus-kp-modifier-detector": run_peak_01e,
    "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast": run_peak_01f,
    "TR-V2-00-age01b-baseline-lock": run_tr_v2_00,
    "TR-V2-01-phase2-direct-hazard-priors": run_tr_v2_01,
    "TR-V2-02-phase2-hidden-shock-hazards": run_tr_v2_02,
    "TR-V2-03-phase2-ablation-suite": run_tr_v2_03,
    "PHASE3-V2-INT-explicit-incidence-autoresearch": run_phase3_v2_int,
}

REPORT_DISPATCH = {
    "rolling-origin": write_tr_v2_rolling_origin_report,
    "early-history-partial": write_tr_v2_early_history_partial_report,
    "benchmark-dashboard": write_tr_v2_benchmark_dashboard_report,
}


def run_phase3_transition_research(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    cli_experiment_name: str,
    phase3_result_dir_name: str | None = None,
) -> dict[str, Any]:
    experiment_id = get_experiment_id_for_cli(cli_experiment_name)
    if experiment_id not in EXPERIMENT_DISPATCH:
        raise KeyError(f"transition research experiment is registered but not implemented: {experiment_id}")
    ctx = build_transition_research_context(
        run_id=run_id,
        plugin_id=plugin_id,
        experiment_id=experiment_id,
        source_run_id=source_run_id,
        phase3_result_dir_name=phase3_result_dir_name,
    )
    return EXPERIMENT_DISPATCH[experiment_id](ctx)


def run_phase3_transition_report(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    cli_report_name: str,
    start_year: int | None = None,
    end_year: int | None = None,
    min_train_years: int | None = None,
    horizon_years: int | None = None,
    rolling_origin_run_id: str | None = None,
    early_history_run_id: str | None = None,
) -> dict[str, Any]:
    if cli_report_name not in REPORT_DISPATCH:
        raise KeyError(f"transition report is not implemented: {cli_report_name}")
    kwargs: dict[str, Any] = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "source_run_id": source_run_id,
    }
    if cli_report_name in {"rolling-origin", "early-history-partial"}:
        if start_year is not None:
            kwargs["start_year"] = int(start_year)
        if end_year is not None:
            kwargs["end_year"] = int(end_year)
        if min_train_years is not None:
            kwargs["min_train_years"] = int(min_train_years)
        if horizon_years is not None:
            kwargs["horizon_years"] = int(horizon_years)
    if cli_report_name == "benchmark-dashboard":
        if rolling_origin_run_id:
            kwargs["rolling_origin_run_id"] = str(rolling_origin_run_id)
        if early_history_run_id:
            kwargs["early_history_run_id"] = str(early_history_run_id)
    return REPORT_DISPATCH[cli_report_name](**kwargs)
