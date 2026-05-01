from __future__ import annotations

from typing import Any

from .artifacts import build_incidence_research_context
from .audits import run_inc_00a, run_inc_00b, run_inc_00c
from .modeling import run_inc_01b, run_inc_01d, run_inc_v2_01
from .registry import get_experiment_id_for_cli
from epigraph_ph.phase3._lineage.shocks import run_shock_00a

EXPERIMENT_DISPATCH = {
    "INC-00A-incidence-evidence-audit": run_inc_00a,
    "INC-00B-population-denominator-audit": run_inc_00b,
    "INC-00C-incidence-identifiability-audit": run_inc_00c,
    "INC-01B-backlog-vs-incidence-swap-stress-test": run_inc_01b,
    "INC-01D-diagnosis-locked-incidence-branch": run_inc_01d,
    "INC-V2-01-observed-denominator-explicit-incidence": run_inc_v2_01,
    "SHOCK-00A-covid-shock-subparameter-audit": run_shock_00a,
}


def run_phase3_incidence_research(
    *,
    run_id: str,
    plugin_id: str,
    source_run_id: str,
    cli_experiment_name: str,
) -> dict[str, Any]:
    experiment_id = get_experiment_id_for_cli(cli_experiment_name)
    if experiment_id not in EXPERIMENT_DISPATCH:
        raise KeyError(f"incidence research experiment is registered but not implemented: {experiment_id}")
    ctx = build_incidence_research_context(
        run_id=run_id,
        plugin_id=plugin_id,
        experiment_id=experiment_id,
        source_run_id=source_run_id,
    )
    return EXPERIMENT_DISPATCH[experiment_id](ctx)
