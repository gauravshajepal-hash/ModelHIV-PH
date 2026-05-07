from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r66_scientific_source_base import R66_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R67_SCHEMA_VERSION = "phase3_dynamic.r67_transmission_model_family_queue.v1"
R67_RUN_ID = "p3d-r67-transmission-model-family-queue-20260506-s00"
R66_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R66_RUN_ID
    / "analysis"
    / "r66_scientific_source_base_report.json"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_index(r66_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("module_target") or ""): dict(row) for row in r66_report.get("module_coverage_rows") or []}


def _coverage_count(coverage: dict[str, dict[str, Any]], modules: list[str]) -> int:
    return int(sum(int((coverage.get(module) or {}).get("usable_supported_count") or 0) for module in modules))


def _determinant_count(coverage: dict[str, dict[str, Any]], modules: list[str]) -> int:
    return int(sum(int((coverage.get(module) or {}).get("determinant_context_count") or 0) for module in modules))


def _readiness_status(
    coverage: dict[str, dict[str, Any]],
    *,
    required_modules: list[str],
    determinant_modules: list[str] | None = None,
    needs_strict_phase2_prior: bool = False,
    requires_service_support: bool = False,
) -> str:
    usable = _coverage_count(coverage, required_modules)
    determinants = _determinant_count(coverage, determinant_modules or [])
    if needs_strict_phase2_prior:
        return "sensitivity_only_until_R46_source_stable"
    if requires_service_support and usable == 0:
        return "blocked_missing_service_support"
    if usable > 0 and determinants > 0:
        return "ready_for_bounded_branch_after_row_extraction"
    if usable > 0:
        return "ready_for_bounded_branch"
    if determinants > 0:
        return "context_only_until_direct_support"
    return "blocked_missing_source_support"


def _model_family_rows(coverage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "priority": 1,
            "family_id": "R67-M01_hidden_service_intensity_state_space",
            "source_domain": "astronomy survey-bias/source-separation",
            "target_mapping": "observed program counts are biased detections of latent epidemic/care states",
            "equation": "Y_{m,g,t} ~ Normal(q_{m,g,t} X_{m,g,t}, sigma_m^2); logit(q_{m,g,t}) = a_m + b_m^T s_{g,t}; X evolves by conserved cascade hazards",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service",
            "required_sources": "HASP monthly/quarterly diagnosis, ART, VL and reporting-support rows",
            "overfit_guard": "q_{m,g,t} estimated from train-only reporting/support covariates; no holdout residuals enter q",
            "promotion_gate": "R60/R63 regional gate improves regional/mass/share NAE and keeps stock cone and conditional-rate gates",
            "leakage_policy": "no direct leakage; same-holdout oracle may only define a ceiling",
            "failure_mode": "can absorb true incidence changes into reporting intensity if determinant/incidence validation is absent",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "art_retention", "vl_suppression_service"],
                requires_service_support=True,
            ),
        },
        {
            "priority": 2,
            "family_id": "R67-M02_competing_risk_semi_markov_cascade",
            "source_domain": "survival analysis and biophysics state-transition kinetics",
            "target_mapping": "each cascade stage has sojourn-time-dependent exits, not one aggregate leakage term",
            "equation": "h_{j->k,g,t}(a)=1-exp(-exp(alpha_{jk}+f_{jk}(a)+beta_{jk}^T z_{g,t}+u_{g,t})); X_{j,g,t+1}=sum_a X_{j,g,t,a}(1-sum_k h_{j->k,g,t}(a))",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|mortality_reporting",
            "required_sources": "diagnosis flow, ART active stock/refill/LTFU definitions, VL-tested/suppressed, reported deaths",
            "overfit_guard": "age-in-state kernels selected by blocked-time likelihood and rejected if stock cone breaks",
            "promotion_gate": "beats R41/R60 reference and carry-forward on blocked 1y/3y/5y without worsening annual challenge",
            "leakage_policy": "leakage exits are fitted transition channels only where evidence-backed; otherwise sensitivity-only",
            "failure_mode": "unidentified re-engagement/interruption channels if only endpoint stocks are present",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "art_retention", "vl_suppression_service", "mortality_reporting"],
                requires_service_support=True,
            ),
        },
        {
            "priority": 3,
            "family_id": "R67-M03_cd4_ahd_backcalculation_incidence",
            "source_domain": "epidemiological inverse problems and deconvolution",
            "target_mapping": "infections are latent events; diagnoses and late-diagnosis markers are delayed emissions",
            "equation": "D_{g,t}=sum_{ell>=0} I_{g,t-ell} K_{g,t}(ell)+B_{g,t}; C_{g,t} ~ Multinomial(D_{g,t}, pi_CD4(U_early,U_late)); annual I_y ~ Normal(sum_{t in y} I_t, tau_y^2)",
            "module_targets": "incidence_validation|diagnosis_reporting",
            "required_sources": "HASP diagnosis flow plus AHD/CD4/late diagnosis emissions; annual incidence validation only",
            "overfit_guard": "annual incidence remains weak measurement; diagnosis-flow-derived incidence readout is disallowed",
            "promotion_gate": "improves diagnosis-flow and annual incidence validation while not worsening diagnosed/ART stock",
            "leakage_policy": "no same-period incidence validation in transition fitting",
            "failure_mode": "delay kernel underidentified without late-diagnosis evidence",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "incidence_validation"],
            ),
        },
        {
            "priority": 4,
            "family_id": "R67-M04_kp_metapopulation_transmission_patch",
            "source_domain": "metapopulation ecology, percolation, and network epidemics",
            "target_mapping": "regions and key populations are coupled exposure patches with sparse observed KP denominators",
            "equation": "I_{k,g,t}=S_{k,g,t}(1-exp(-Delta_t * sum_{k',g'} C_{kk'} W_{gg'} rho_{k'} v_{k',g',t} / N_{k',g',t})); lambda includes PrEP_active and suppression-mediated infectivity reduction",
            "module_targets": "incidence_pressure|kp_overlay|regional_shrinkage|prep_persistence",
            "required_sources": "KP size/prevalence/testing/condom/PrEP, population denominators, regional diagnosis/VL support",
            "overfit_guard": "borrow strength hierarchically; forbid independent region-specific transmission coefficients with sparse support",
            "promotion_gate": "only after Phase2 determinant bundle survives source-family re-estimation and blocked-time gate",
            "leakage_policy": "annual incidence is validation/weak measurement, not quarterly target",
            "failure_mode": "false precision from sparse KP denominators or unobserved sexual-network mixing",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["prep_persistence"],
                determinant_modules=["incidence_pressure", "kp_overlay", "regional_shrinkage"],
                needs_strict_phase2_prior=True,
            ),
        },
        {
            "priority": 5,
            "family_id": "R67-M05_renormalized_hierarchical_regional_pooling",
            "source_domain": "statistical physics coarse-graining and hierarchical Bayes",
            "target_mapping": "national dynamics are coarse variables; regions are partially pooled deviations constrained by national totals",
            "equation": "theta_{g,m}=theta_{0,m}+eta_{region(g),m}+xi_{g,m}; eta,xi ~ GMRF(0,Q(W,tau)); sum_g X_{g,t}=X_{national,t}",
            "module_targets": "regional_shrinkage|diagnosis_reporting|art_retention|vl_suppression_service",
            "required_sources": "regional cascade rows plus denominator/geography similarity features",
            "overfit_guard": "region effects shrink to national unless blocked-time regional evidence beats pooled baseline",
            "promotion_gate": "regional split-stability improves; no national-total incoherence",
            "leakage_policy": "no region-specific coefficient learned from the same holdout residual",
            "failure_mode": "over-smoothing true local epidemics if similarity graph is wrong",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "art_retention", "vl_suppression_service"],
                determinant_modules=["regional_shrinkage"],
                requires_service_support=True,
            ),
        },
        {
            "priority": 6,
            "family_id": "R67-M06_service_capacity_queue_control",
            "source_domain": "control theory, queueing systems, and constrained flow networks",
            "target_mapping": "testing, linkage, ART refill, and VL labs are service-capacity processes with bottlenecks",
            "equation": "flow_{j->k,g,t}=min(X_{j,g,t}, cap_{k,g,t}) * p_{j->k,g,t}; cap_{k,g,t+1}=cap_{k,g,t}+r(u_{g,t}-cap_{k,g,t})+epsilon",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|prep_persistence",
            "required_sources": "facility/reporting counts, ART active stock, VL testing numerator, PrEP refill/new enrollment",
            "overfit_guard": "capacity state driven by observed support variables and smoothed train-only latent state",
            "promotion_gate": "reduces 3y/5y ART and VL drift without breaking 1y R10-equivalent gate",
            "leakage_policy": "no residual correction unless conditional-rate gates remain valid",
            "failure_mode": "capacity can become a hidden free correction if support variables are weak",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "art_retention", "vl_suppression_service", "prep_persistence"],
                requires_service_support=True,
            ),
        },
        {
            "priority": 7,
            "family_id": "R67-M07_low_rank_hidden_driver_with_sparse_direct_terms",
            "source_domain": "astronomy source separation and matrix factor models",
            "target_mapping": "hidden modes capture shared shocks; direct Phase2 surfaces remain sparse module covariates",
            "equation": "logit(h_{r,g,t})=alpha_r+beta_r^T z^{direct}_{r,g,t}+l_r^T f_{g,t}; f_{g,t}=Phi f_{g,t-1}+epsilon_{g,t}",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|incidence_pressure",
            "required_sources": "Phase2 structural payload plus R46 source-family-stable determinant bundles",
            "overfit_guard": "direct terms enter only after source-family ablation; hidden modes are shocks, not intervention targets",
            "promotion_gate": "beats baseline mechanism under blocked time and passes placebo edge separation",
            "leakage_policy": "hidden factors cannot be fitted from holdout endpoints",
            "failure_mode": "factor rotations can look mechanistic while being non-identifiable",
            "readiness_status": "sensitivity_only_until_R46_source_stable",
        },
        {
            "priority": 8,
            "family_id": "R67-M08_online_expert_leakage_teacher_student",
            "source_domain": "computer-science online learning with expert advice",
            "target_mapping": "leakage oracle identifies upper bound; train-only student learns safe family routing",
            "equation": "w_{i,t+1} proportional w_{i,t} exp(-eta L_{i,t}); yhat_{g,m,t}=sum_i w_{i,g,m,t} yhat_{i,g,m,t}",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|regional_shrinkage",
            "required_sources": "candidate family errors on train splits only plus R64 support gap targets",
            "overfit_guard": "same-holdout oracle is diagnostic only; student must beat R60 without holdout residual labels",
            "promotion_gate": "blocked-time student improves R60 and keeps split-stability; otherwise diagnostic only",
            "leakage_policy": "explicitly non-promotable if same-holdout labels are used",
            "failure_mode": "teacher leakage can create impossible performance ceilings if mistaken as model performance",
            "readiness_status": "diagnostic_only_until_student_passes_R60_contract",
        },
        {
            "priority": 9,
            "family_id": "R67-M09_reaction_diffusion_access_pressure",
            "source_domain": "reaction-diffusion systems and spatial ecology",
            "target_mapping": "exposure pressure and service capacity diffuse over mobility/access graphs while local reactions follow HIV cascade hazards",
            "equation": "x_{g,t+1}=x_{g,t}+F(x_{g,t},z_{g,t})+kappa sum_{g'} W_{g,g'}(x_{g',t}-x_{g,t})",
            "module_targets": "incidence_pressure|regional_shrinkage|diagnosis_reporting|art_retention",
            "required_sources": "mobility/access graph, regional denominator, regional cascade support",
            "overfit_guard": "W fixed from external geography/mobility evidence before fitting; kappa globally shrunk",
            "promotion_gate": "regional blocked-time improvement after W source-family ablation",
            "leakage_policy": "no graph edge learned from target residual co-movement alone",
            "failure_mode": "false analogy if movement proxy reflects reporting access rather than transmission contact",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["diagnosis_reporting", "art_retention"],
                determinant_modules=["regional_shrinkage", "incidence_pressure"],
                needs_strict_phase2_prior=True,
            ),
        },
        {
            "priority": 10,
            "family_id": "R67-M10_prEP_persistence_susceptibility_process",
            "source_domain": "pharmacologic protection and survival/persistence models",
            "target_mapping": "new PrEP enrollment is not protection; active refill persistence is the protection state",
            "equation": "P_active_{g,t+1}=P_active_{g,t}(1-lapse_{g,t})+new_refill_{g,t}; lambda_{k,g,t}=lambda^0_{k,g,t}(1-epsilon_PrEP * P_active_{k,g,t}/S_{k,g,t})",
            "module_targets": "prep_persistence|incidence_pressure|kp_overlay",
            "required_sources": "PrEP new enrollment, active refill/return, KP denominators",
            "overfit_guard": "PrEP effect bounded by literature prior and identifiable only through validation, not same-period incidence",
            "promotion_gate": "improves annual incidence validation and scenario plausibility without worsening diagnosis/cascade gates",
            "leakage_policy": "PrEP cannot directly correct diagnosis-flow residuals",
            "failure_mode": "PrEP uptake may proxy health-seeking behavior rather than causal protection",
            "readiness_status": _readiness_status(
                coverage,
                required_modules=["prep_persistence"],
                determinant_modules=["incidence_pressure", "kp_overlay"],
                needs_strict_phase2_prior=True,
            ),
        },
        {
            "priority": 11,
            "family_id": "R67-M11_posterior_stacking_model_averaging",
            "source_domain": "Bayesian forecast combination and astronomical catalog ensemble calibration",
            "target_mapping": "when mechanisms are partially identified, calibrated model averaging can beat any single brittle branch",
            "equation": "p(y_{t+h}|D)=sum_i w_{i,h,m} p_i(y_{t+h}|D); w=argmin train CRPS subject to stock-cone and rate constraints",
            "module_targets": "annual_challenge|diagnosis_reporting|art_retention|vl_suppression_service",
            "required_sources": "locked candidate predictions with train-only interval scores",
            "overfit_guard": "weights selected on rolling-origin train history; include carry-forward/R10 as competitors",
            "promotion_gate": "posterior coverage and mean score beat R41/R60/R10-scope reference across horizons",
            "leakage_policy": "holdout target is unavailable during weight fitting",
            "failure_mode": "model average can hide mechanistic failures if not reported module-wise",
            "readiness_status": "ready_for_bounded_branch",
        },
        {
            "priority": 12,
            "family_id": "R67-M12_identifiability_first_null_model",
            "source_domain": "mathematical identifiability and error-correcting type systems",
            "target_mapping": "some parameters should be declared non-identifiable rather than forced into a fitted transmission story",
            "equation": "promote(theta_j) only if rank(J_train,j)>threshold and source-ablation sign(theta_j) is stable; else theta_j in sensitivity set",
            "module_targets": "incidence_pressure|kp_overlay|art_retention|vl_suppression_service|regional_shrinkage",
            "required_sources": "all candidate source families plus source-ablation and placebo-edge runs",
            "overfit_guard": "non-identifiable parameters are locked, not optimized",
            "promotion_gate": "claim registry moves from sensitivity-only to strict only when identifiability and validation gates pass",
            "leakage_policy": "validation-only rows never contribute to Jacobian rank for training claims",
            "failure_mode": "can be too conservative, blocking useful scenario exploration",
            "readiness_status": "ready_as_guardrail_for_all_R67_models",
        },
    ]
    rows.sort(key=lambda row: int(row.get("priority") or 0))
    return rows


def _next_experiment_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in model_rows:
        status = str(row.get("readiness_status") or "")
        if status in {"ready_for_bounded_branch", "ready_for_bounded_branch_after_row_extraction"}:
            next_action = "implement_bounded_branch"
        elif "source_stable" in status:
            next_action = "run_phase2_source_family_falsification_before_model_fit"
        elif "diagnostic_only" in status:
            next_action = "keep_as_failure_anatomy_or_teacher_only"
        elif "missing" in status:
            next_action = "acquire_or_extract_required_support"
        else:
            next_action = "use_as_guardrail"
        rows.append(
            {
                "priority": row.get("priority"),
                "family_id": row.get("family_id"),
                "readiness_status": status,
                "next_action": next_action,
                "promotion_gate": row.get("promotion_gate"),
            }
        )
    return rows


def _gate(model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in model_rows if str(row.get("readiness_status") or "").startswith("ready_for_bounded")]
    strict_det = [row for row in model_rows if str(row.get("readiness_status") or "") == "sensitivity_only_until_R46_source_stable"]
    return {
        "status": "model_family_queue_ready",
        "ready_bounded_family_count": len(ready),
        "determinant_locked_family_count": len(strict_det),
        "top_ready_families": "|".join(str(row.get("family_id") or "") for row in ready[:5]),
        "contract": (
            "R67 is a model-family queue. It does not promote a model. Families are eligible only if they keep validation-only "
            "evidence out of training, respect stock-cone and conditional-rate gates, and beat locked references under blocked time."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("model_family_gate") or {})
    lines = [
        "# Phase 3 R67 Transmission Model Family Queue",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Ready bounded families: `{gate.get('ready_bounded_family_count')}`",
        f"- Determinant-locked families: `{gate.get('determinant_locked_family_count')}`",
        "",
        "## Ranked Families",
        "",
        "| Priority | Family | Readiness | Transfer |",
        "|---:|---|---|---|",
    ]
    for row in report.get("model_family_rows") or []:
        lines.append(
            f"| {int(row.get('priority') or 0)} | `{row.get('family_id')}` | `{row.get('readiness_status')}` | {row.get('source_domain')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r67_transmission_model_family_queue(
    *,
    run_id: str = R67_RUN_ID,
    r66_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r66_path = R66_DEFAULT_REPORT if r66_report_path is None else Path(r66_report_path)
    r66 = dict(read_json(r66_path, default={}) or {}) if r66_path.exists() else {}
    coverage = _coverage_index(r66)
    model_rows = _model_family_rows(coverage)
    experiment_rows = _next_experiment_rows(model_rows)
    gate = _gate(model_rows)
    report_path = analysis_dir / "r67_transmission_model_family_queue_report.json"
    markdown_path = analysis_dir / "r67_transmission_model_family_queue_report.md"
    model_csv = analysis_dir / "r67_model_family_rows.csv"
    experiment_csv = analysis_dir / "r67_next_experiment_rows.csv"
    report = {
        "schema_version": R67_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "model_family_gate": gate,
        "model_family_rows": model_rows,
        "next_experiment_rows": experiment_rows,
        "source_artifacts": {
            "r66": {"path": r66_path.as_posix(), "sha256": _sha256(r66_path) if r66_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "model_family_csv": model_csv.as_posix(),
            "next_experiment_csv": experiment_csv.as_posix(),
        },
    }
    _write_csv(model_csv, model_rows)
    _write_csv(experiment_csv, experiment_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build R67 transmission model family queue.")
    parser.add_argument("--run-id", default=R67_RUN_ID)
    parser.add_argument("--r66-report-path", default=None)
    args = parser.parse_args()
    run_r67_transmission_model_family_queue(
        run_id=str(args.run_id),
        r66_report_path=None if args.r66_report_path is None else Path(args.r66_report_path),
    )


if __name__ == "__main__":
    _main()
