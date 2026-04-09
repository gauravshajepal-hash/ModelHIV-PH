from __future__ import annotations

from typing import Any

from .age_research import run_age_01a, run_age_01b, run_age_01c
from .analytics import run_an01a, run_an01b, run_an01c
from .artifacts import build_transition_research_context
from .decomposition import run_decomp_01a, run_decomp_01b, run_decomp_01c, run_decomp_01d, run_decomp_01e, run_decomp_01f
from .analytics import run_an02a, run_an02b, run_an02c
from .peak_windows import run_peak_01a, run_peak_01b, run_peak_01c, run_peak_01d, run_peak_01e, run_peak_01f
from .registry import get_experiment_id_for_cli
from .tr_v2 import run_tr_v2_00, run_tr_v2_01, run_tr_v2_02, run_tr_v2_03
from .transition_engine import run_kp_01a, run_mech_01a, run_mech_01b, run_mech_01c, run_mech_01d, run_mech_01e

EXPERIMENT_DISPATCH = {
    "AN-01A-national-yearly-factor-evolution": run_an01a,
    "AN-01B-regional-yearly-factor-evolution": run_an01b,
    "AN-01C-factor-importance-drift-report": run_an01c,
    "AN-02A-factor-to-transition-map": run_an02a,
    "AN-02B-factor-to-kp-map": run_an02b,
    "AN-02C-kp-transition-relevance-drift": run_an02c,
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
