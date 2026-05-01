from __future__ import annotations

from dataclasses import dataclass
from datetime import date


INCIDENCE_EXPERIMENT_IDS: tuple[str, ...] = (
    "INC-00A-incidence-evidence-audit",
    "INC-00B-population-denominator-audit",
    "INC-00C-incidence-identifiability-audit",
    "INC-01B-backlog-vs-incidence-swap-stress-test",
    "INC-01D-diagnosis-locked-incidence-branch",
    "INC-V2-01-observed-denominator-explicit-incidence",
    "SHOCK-00A-covid-shock-subparameter-audit",
)


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    experiment_id: str
    cli_name: str
    description: str
    expected_outputs: tuple[str, ...]


EXPERIMENT_REGISTRY: dict[str, ExperimentDefinition] = {
    "INC-00A-incidence-evidence-audit": ExperimentDefinition(
        experiment_id="INC-00A-incidence-evidence-audit",
        cli_name="inc-00a",
        description="Audit direct inflow, proxy, and validation-only evidence for a pre-U incidence layer.",
        expected_outputs=(
            "incidence_evidence_audit.json",
            "incidence_signal_coverage_summary.json",
            "incidence_support_heatmap.png",
        ),
    ),
    "INC-00B-population-denominator-audit": ExperimentDefinition(
        experiment_id="INC-00B-population-denominator-audit",
        cli_name="inc-00b",
        description="Audit observed population denominators that could sit before U without becoming a latent population state.",
        expected_outputs=(
            "population_denominator_audit.json",
            "population_alignment_summary.json",
        ),
    ),
    "INC-00C-incidence-identifiability-audit": ExperimentDefinition(
        experiment_id="INC-00C-incidence-identifiability-audit",
        cli_name="inc-00c",
        description="Stress-test whether incidence pressure can be separated from diagnosis backlog with current evidence.",
        expected_outputs=(
            "incidence_identifiability_audit.json",
            "identifiability_stress_table.json",
        ),
    ),
    "INC-01B-backlog-vs-incidence-swap-stress-test": ExperimentDefinition(
        experiment_id="INC-01B-backlog-vs-incidence-swap-stress-test",
        cli_name="inc-01b",
        description="Quantify how much of the diagnosis-locked incidence branch can be swapped between inflow and backlog-compatible residuals.",
        expected_outputs=(
            "swap_stress_summary.json",
            "identifiability_margin.json",
        ),
    ),
    "INC-01D-diagnosis-locked-incidence-branch": ExperimentDefinition(
        experiment_id="INC-01D-diagnosis-locked-incidence-branch",
        cli_name="inc-01d",
        description="Make a pre-U incidence layer explicit while inheriting the kept diagnosis path from the current transition-research winner.",
        expected_outputs=(
            "incidence_flow_summary.json",
            "diagnosis_locked_incidence_summary.json",
            "fit_artifact.json",
            "evaluation.json",
            "forecast_vs_locked_baseline.png",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "INC-V2-01-observed-denominator-explicit-incidence": ExperimentDefinition(
        experiment_id="INC-V2-01-observed-denominator-explicit-incidence",
        cli_name="inc-v2-01",
        description="Make incidence explicit as I_t = N_t * lambda_t using an observed official denominator while keeping the diagnosis path locked to the current transition winner.",
        expected_outputs=(
            "population_denominator_series.json",
            "explicit_incidence_hazard_summary.json",
            "incidence_flow_summary.json",
            "fit_artifact.json",
            "evaluation.json",
            "forecast_vs_locked_baseline.png",
            "state_estimates.npz",
            "forecast_states.npz",
        ),
    ),
    "SHOCK-00A-covid-shock-subparameter-audit": ExperimentDefinition(
        experiment_id="SHOCK-00A-covid-shock-subparameter-audit",
        cli_name="shock-00a",
        description="Audit typed Phase 1 normalized subparameters for diagnosis-service, backlog-release, incidence-side, and downstream-care shock support.",
        expected_outputs=(
            "shock_subparameter_audit.json",
            "shock_signal_coverage_summary.json",
            "shock_factor_family_heatmap.png",
        ),
    ),
}

CLI_TO_EXPERIMENT_ID: dict[str, str] = {
    definition.cli_name: experiment_id for experiment_id, definition in EXPERIMENT_REGISTRY.items()
}


def get_experiment_definition(experiment_id: str) -> ExperimentDefinition:
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise KeyError(f"unknown incidence research experiment: {experiment_id}")
    return EXPERIMENT_REGISTRY[experiment_id]


def get_experiment_id_for_cli(cli_name: str) -> str:
    if cli_name not in CLI_TO_EXPERIMENT_ID:
        raise KeyError(f"unknown incidence research CLI experiment: {cli_name}")
    return CLI_TO_EXPERIMENT_ID[cli_name]


def list_cli_names() -> list[str]:
    return sorted(CLI_TO_EXPERIMENT_ID)


def make_incidence_run_id(*, experiment_id: str, seed: int = 0, today: date | None = None) -> str:
    stamp = (today or date.today()).strftime("%Y%m%d")
    return f"inc-{stamp}-s{seed:02d}-{experiment_id}"
