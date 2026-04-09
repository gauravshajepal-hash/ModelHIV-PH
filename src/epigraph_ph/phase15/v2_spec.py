from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.latent_blocks import latent_block_specs
from epigraph_ph.phase15.latent_measurements import (
    month_slots_for_row,
    normalize_region_label,
    province_lookup_tokens,
    province_region_codes,
    province_region_members,
    target_province_indices,
)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "y"}


def _canonical_observation_row_id(index: int, row: Mapping[str, Any]) -> str:
    source_id = str(row.get("source_id") or row.get("candidate_id") or row.get("evidence_indicator_id") or "").strip()
    canonical_name = str(row.get("canonical_name") or "unknown").strip() or "unknown"
    if source_id:
        return f"{canonical_name}:{source_id}:{index:06d}"
    return f"{canonical_name}:row:{index:06d}"


def _time_support_mode(row: Mapping[str, Any], month_axis: list[str]) -> tuple[list[int], str]:
    month_slots = month_slots_for_row(row, month_axis)
    if not month_slots:
        return [], "unsupported"
    time_value = str(row.get("time") or row.get("effective_month") or "").strip()
    temporal_precision = str(row.get("temporal_precision") or "").strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}", time_value):
        return month_slots, "month_snapshot"
    if re.fullmatch(r"\d{4}", time_value):
        if "snapshot" in temporal_precision or _safe_bool(row.get("is_anchor_eligible")):
            return [month_slots[-1]], "year_end_snapshot"
        return month_slots, "year_average"
    return month_slots, "unsupported"


def _support_operator_kind(row: Mapping[str, Any], support_scope: str, time_support_mode: str) -> str:
    declared = str(row.get("observation_operator") or "").strip()
    if declared:
        return declared
    if support_scope == "province" and time_support_mode == "month_snapshot":
        return "province_month_snapshot"
    if support_scope == "province" and time_support_mode == "year_end_snapshot":
        return "province_year_end_snapshot"
    if support_scope == "province":
        return "province_year_average"
    if support_scope == "region" and time_support_mode == "month_snapshot":
        return "region_month_average"
    if support_scope == "region" and time_support_mode == "year_end_snapshot":
        return "region_year_end_snapshot"
    if support_scope == "region":
        return "region_year_average"
    if support_scope == "national" and time_support_mode == "month_snapshot":
        return "national_month_average"
    if support_scope == "national" and time_support_mode == "year_end_snapshot":
        return "national_year_end_snapshot"
    return "national_year_average"


def build_phase15_v2_observation_support(
    *,
    normalized_rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    region_labels: list[str] | None = None,
    include_national_rows: bool = True,
) -> dict[str, Any]:
    province_lookup = province_lookup_tokens(province_axis)
    region_codes = province_region_codes(province_axis, region_labels)
    region_members = province_region_members(region_codes)
    rows: list[dict[str, Any]] = []
    operator_counts: dict[str, int] = {}
    measurement_role_counts: dict[str, int] = {}
    for row_idx, row in enumerate(normalized_rows):
        measurement_role = str(row.get("measurement_role") or "").strip()
        if measurement_role == "context_only":
            continue
        province_indices, support_scope = target_province_indices(
            row,
            province_lookup=province_lookup,
            region_members=region_members,
            include_national_rows=include_national_rows,
        )
        if not province_indices:
            continue
        month_indices, time_support_mode = _time_support_mode(row, month_axis)
        if not month_indices:
            continue
        operator_kind = _support_operator_kind(row, support_scope, time_support_mode)
        cell_count = max(len(province_indices) * len(month_indices), 1)
        cell_weight = 1.0 / float(cell_count)
        operator_counts[operator_kind] = operator_counts.get(operator_kind, 0) + 1
        measurement_role_counts[measurement_role] = measurement_role_counts.get(measurement_role, 0) + 1
        rows.append(
            {
                "source_row_index": int(row_idx),
                "observation_row_id": _canonical_observation_row_id(row_idx, row),
                "canonical_name": str(row.get("canonical_name") or ""),
                "candidate_block": str(row.get("candidate_block") or ""),
                "measurement_role": measurement_role,
                "expected_sign": str(row.get("expected_sign") or ""),
                "source_bank": str(row.get("source_bank") or ""),
                "geo_resolution": str(row.get("geo_resolution") or ""),
                "time_resolution": str(row.get("time_resolution") or ""),
                "support_scope": support_scope,
                "time_support_mode": time_support_mode,
                "operator_kind": operator_kind,
                "province_indices": [int(idx) for idx in province_indices],
                "province_labels": [province_axis[int(idx)] for idx in province_indices],
                "region_labels": sorted({normalize_region_label(region_codes[int(idx)]) for idx in province_indices}),
                "month_indices": [int(idx) for idx in month_indices],
                "month_labels": [month_axis[int(idx)] for idx in month_indices],
                "cell_count": int(cell_count),
                "normalized_cell_weight": round(float(cell_weight), 8),
                "is_anchor_eligible": _safe_bool(row.get("is_anchor_eligible")),
                "is_direct_measurement": _safe_bool(row.get("is_direct_measurement")),
            }
        )
    return {
        "method": "phase15_v2_observation_support_v1",
        "province_axis": list(province_axis),
        "month_axis": list(month_axis),
        "row_count": len(rows),
        "summary": {
            "operator_counts": dict(sorted(operator_counts.items())),
            "measurement_role_counts": dict(sorted(measurement_role_counts.items())),
        },
        "rows": rows,
    }


def _phase15_v2_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase15_cfg = dict((plugin.constraint_settings or {}).get("phase15", {}) or {})
    return dict(phase15_cfg.get("latent_blocks_v2", {}) or {})


def build_phase15_v2_spec(
    *,
    plugin_id: str,
    normalized_rows: list[dict[str, Any]],
    observability_audit: Mapping[str, Any],
    measurement_spec: Mapping[str, Any],
    province_axis: list[str],
    month_axis: list[str],
    observation_support: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _phase15_v2_cfg(plugin_id)
    audit_rows = list(dict(observability_audit).get("rows") or [])
    retained_blocks = list(dict(measurement_spec).get("retained_blocks") or [])
    direct_rows = [row for row in normalized_rows if str(row.get("measurement_role") or "") == "direct_indicator"]
    proxy_rows = [row for row in normalized_rows if str(row.get("measurement_role") or "") == "proxy_indicator"]
    context_rows = [row for row in normalized_rows if str(row.get("measurement_role") or "") == "context_only"]
    sign_priors: list[dict[str, Any]] = []
    for block in latent_block_specs(plugin_id):
        block_id = str(block.get("block_id") or "")
        for canonical_name, indicator in dict(block.get("indicators") or {}).items():
            sign_priors.append(
                {
                    "block_id": block_id,
                    "canonical_name": canonical_name,
                    "expected_sign": str(indicator.get("expected_sign") or "neutral"),
                    "literature_basis": list(indicator.get("literature_basis") or []),
                }
            )

    direct_evidence = [
        {
            "claim": "Mixed-frequency observations should be represented by explicit support operators over province-month latent states rather than annual fan-out.",
            "repo_surface": "Replace build_sparse_indicator_cube annual spreading with sparse observation operators H_i.",
            "sources": [
                {
                    "title": "Dynamic Mortality Forecasting via Mixed-Frequency State-Space Models",
                    "date": "2026-01-09",
                    "url": "https://arxiv.org/abs/2601.05702",
                },
                {
                    "title": "Mixed Frequency FAVAR for Regional Output Developments in Ukraine",
                    "date": "2026-01-01",
                    "url": "https://www.oru.se/globalassets/oru-sv/institutioner/hh/workingpapers/workingpapers2026/wp-1-2026.pdf",
                },
            ],
        },
        {
            "claim": "Sparse irregular observations should be assimilated through learned latent observation maps and temporal smoothing, not same-month shrinkage only.",
            "repo_surface": "Replace province_factor_graph closed-form shrinkage with latent smoothing over all months.",
            "sources": [
                {
                    "title": "LEVDA: Latent 4DEnVar Data Assimilation",
                    "date": "2026-02-23",
                    "url": "https://arxiv.org/abs/2602.19406",
                },
                {
                    "title": "Latent Autoencoder Ensemble Kalman Filter",
                    "date": "2026-03-06",
                    "url": "https://arxiv.org/abs/2603.06752",
                },
                {
                    "title": "Physically Consistent Global Atmospheric Data Assimilation with Machine Learning in Latent Space",
                    "date": "2025-02-05",
                    "url": "https://arxiv.org/abs/2502.02884",
                },
            ],
        },
        {
            "claim": "National and regional series should be coherence constraints derived from lower-level latents, not top-down priors imposed on them.",
            "repo_surface": "Replace national-first scaffold with bottom-up province aggregation constraints.",
            "sources": [
                {
                    "title": "Hierarchical Forecast Reconciliation on Networks",
                    "date": "2025-05-06",
                    "url": "https://arxiv.org/abs/2505.03955",
                },
                {
                    "title": "Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy",
                    "date": "2025-02-17",
                    "url": "https://arxiv.org/abs/2502.12156",
                },
            ],
        },
        {
            "claim": "Loading signs should be constrained by prior knowledge, but loading magnitudes must be estimated under shrinkage.",
            "repo_surface": "Replace signed weighted averages with sign-constrained learned loadings.",
            "sources": [
                {
                    "title": "Are the Signs of Factor Loadings Arbitrary in Confirmatory Factor Analysis? Problems and Solutions",
                    "date": "2025-01-23",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11875518/",
                },
                {
                    "title": "Exact Exploratory Bi-factor Analysis: A Constraint-Based Optimization Approach",
                    "date": "2025-05-16",
                    "url": "https://www.cambridge.org/core/journals/psychometrika/article/exact-exploratory-bifactor-analysis-a-constraintbased-optimization-approach/B904E3DB981CD0F623D0B1E0152F36B6",
                },
                {
                    "title": "Bayesian (non-)unique sparse factor modeling",
                    "date": "2025-04-23",
                    "url": "https://www.sciencedirect.com/science/article/pii/S2452306225000231",
                },
            ],
        },
        {
            "claim": "Observation and shrinkage precisions should be estimated from data rather than fixed by hand.",
            "repo_surface": "Replace national_precision, region_precision, and fixed support weights with estimated precision models.",
            "sources": [
                {
                    "title": "Advancing snow data assimilation with a dynamic observation uncertainty",
                    "date": "2026-02-06",
                    "url": "https://tc.copernicus.org/articles/20/609/2026/tc-20-609-2026.html",
                },
                {
                    "title": "Observation error estimation in climate proxies with data assimilation and innovation statistics",
                    "date": "2025-09-19",
                    "url": "https://cp.copernicus.org/articles/21/1801/2025/cp-21-1801-2025.html",
                },
            ],
        },
    ]
    contextual_evidence = [
        {
            "claim": "Dynamic factors with long memory are preferable to static yearly spreading when persistence matters.",
            "repo_surface": "Use stable latent dynamics with smoothing rather than same-month-only borrowing.",
            "sources": [
                {
                    "title": "Dynamic Factor Models and Fractional Integration",
                    "date": "2024-11-01",
                    "url": "https://doi.org/10.3390/econometrics12040039",
                }
            ],
        },
        {
            "claim": "Model uncertainty should be carried explicitly in posterior inference.",
            "repo_surface": "Keep source disagreement and observation-quality uncertainty inside the model, not just in external audit tables.",
            "sources": [
                {
                    "title": "Incorporation of model accuracy in gravitational wave Bayesian inference",
                    "date": "2025-07-15",
                    "url": "https://www.nature.com/articles/s41550-025-02579-7",
                }
            ],
        },
    ]

    equations = [
        {
            "equation_id": "E1",
            "name": "Province latent state",
            "math": "x_{p,t,b} = mu_{p,b} + phi_b x_{p,t-1,b} + gamma_b r_{r(p),t-1,b} + beta_b^T u_{p,t,b} + eta_{p,t,b}",
            "english": "The latent block state in province p, month t, block b is driven by a province baseline, its own lag, the previous-month regional aggregate, optional exogenous block covariates, and process noise.",
        },
        {
            "equation_id": "E2",
            "name": "Region aggregation",
            "math": "r_{r,t,b} = sum_{p in r} omega_{p|r,b} x_{p,t,b}",
            "english": "Each regional block state is the weighted aggregation of its member province states at the same month.",
        },
        {
            "equation_id": "E3",
            "name": "National aggregation",
            "math": "n_{t,b} = sum_p omega_{p,b} x_{p,t,b}",
            "english": "The national block state is not primary. It is a weighted aggregation of province states, using population or burden weights.",
        },
        {
            "equation_id": "E4",
            "name": "Sign-constrained loading",
            "math": "lambda_{j,b} = s_{j,b} * softplus(theta_{j,b})",
            "english": "Each indicator loading has a fixed prior sign s from literature or ontology, but its magnitude is learned by optimizing theta.",
        },
        {
            "equation_id": "E5",
            "name": "Observation model",
            "math": "g_j(y_i) ~ Normal(alpha_j + sum_b lambda_{j,b} * sum_{p,t} H_i[p,t] x_{p,t,b}, sigma_i^2)",
            "english": "Each observed row y_i is linked to the latent province-month field through a sparse observation operator H_i and an indicator-specific transform g_j such as identity or logit.",
        },
        {
            "equation_id": "E6",
            "name": "Observation support normalization",
            "math": "sum_{p,t} H_i[p,t] = 1",
            "english": "Every observation operator is normalized so a province-month snapshot, a region-year mean, and a national-year mean are all represented coherently on the same latent grid.",
        },
        {
            "equation_id": "E7",
            "name": "Observation precision model",
            "math": "log tau_i = xi_0 + xi_role[r_i] + xi_source[s_i] + xi_geo[g_i] + xi_time[q_i] + xi_anchor a_i + xi_quality q_i_star",
            "english": "Observation precision is estimated from row role, source family, geography level, time granularity, anchor status, and quality metadata instead of being fixed by hand.",
        },
        {
            "equation_id": "E8",
            "name": "Province hierarchy prior",
            "math": "mu_{p,b} ~ Normal(mu_{r(p),b}, tau_{prov,b}^{-1})",
            "english": "Province baselines are partially pooled toward their region-specific baselines with an estimated province-level shrinkage precision.",
        },
        {
            "equation_id": "E9",
            "name": "Region hierarchy prior",
            "math": "mu_{r,b} ~ Normal(mu_{nat,b}, tau_{reg,b}^{-1})",
            "english": "Regional baselines are partially pooled toward a national baseline, again with an estimated precision rather than a fixed weight like 4.0 or 2.0.",
        },
        {
            "equation_id": "E10",
            "name": "Context-only prior rule",
            "math": "p(theta, s, tau | context) updates priors only; context rows are excluded from p(y | x, theta)",
            "english": "Literature-only rows never enter the observation likelihood. They only modify prior sign and shrinkage structure.",
        },
        {
            "equation_id": "E11",
            "name": "Temporal smoothing posterior",
            "math": "p(x_{1:T} | y_{1:N}) propto p(x_1) * product_t p(x_t | x_{t-1}) * product_i p(y_i | x_{1:T})",
            "english": "Missing months are inferred by smoothing across the full time path, not by copying annual values into every month.",
        },
        {
            "equation_id": "E12",
            "name": "Inference backbone",
            "math": "theta_hat = argmax_theta ELBO(theta) or argmax_theta p(theta, x_{1:T} | y)",
            "english": "The first serious implementation should use alternating optimization or variational EM on this structured model rather than hand-picked support constants.",
        },
    ]

    council = {
        "mode": "manual_same_model_council",
        "roles": {
            "evidence_agent": {
                "strongest_point": "Recent 2025-2026 work consistently replaces ad hoc spreading and top-down latent borrowing with explicit observation operators, aggregation constraints, and smoothing.",
                "direct_evidence_count": len(direct_evidence),
            },
            "validity_skeptic": {
                "strongest_point": "The current scaffold leaks NCR-heavy national signal into weak provinces and treats contextual literature as too close to measurement; v2 must separate priors from likelihood and enforce bottom-up coherence.",
                "non_negotiable_constraints": [
                    "No annual fan-out as if it were monthly truth.",
                    "No fixed national-first shrinkage weights.",
                    "No context-only rows in the measurement likelihood.",
                    "No national latent that is not an aggregation of province latents.",
                ],
            },
            "representation_modeling_agent": {
                "strongest_point": "The best feasible backbone for this repo is a mixed-frequency hierarchical dynamic factor model on transformed indicator space, with sign-constrained loadings and sparse observation operators.",
            },
            "evaluation_failure_agent": {
                "strongest_point": "A v2 rollout is a false win if it improves smoothness but breaks coherence, calibration, holdout behavior, or transportability outside NCR-heavy provinces.",
                "gates": [
                    "Cross-scale coherence holds exactly.",
                    "Posterior uncertainty widens when support is weak.",
                    "Leave-year-out holdout improves over scaffold.",
                    "Province drift no longer tracks NCR by default.",
                ],
            },
        },
        "chairman_synthesis": {
            "selected_autoresearch_variant": "evidence-to-model-loop",
            "recommended_backbone": "mixed_frequency_hierarchical_dynamic_factor_model_v2",
            "why": "It is the strongest mathematically coherent design that still matches the repo's sparse-data and evidence-mined structure.",
        },
    }

    evaluation = {
        "false_wins": [
            "Smoother province curves that are driven only by national shrinkage.",
            "Higher retained block counts obtained by letting context-only rows behave like observations.",
            "Apparent fit gains created by copying annual data into every month.",
            "Province calibration that collapses under leave-year-out or leave-region-out checks.",
        ],
        "required_checks": [
            "Exact region and national aggregation coherence from province states.",
            "Posterior interval width increases when local support is sparse.",
            "Held-out annual national anchors and held-out province/month rows are both scored.",
            "NCR-dominance audit comparing province trajectories with and without NCR-heavy national observations.",
            "Sign-stability audit on learned loadings against prior sign directions.",
        ],
        "keep_or_revert_rule": "Keep v2 only if coherence is exact, uncertainty is calibrated, and holdout or long-horizon diagnostics improve without increasing transportability bias.",
    }

    return {
        "model_id": "phase15_v2_mixed_frequency_hierarchical_dynamic_factor_model",
        "plugin_id": plugin_id,
        "selected_autoresearch_variant": "evidence-to-model-loop",
        "current_repo_state": {
            "normalized_row_count": len(normalized_rows),
            "direct_indicator_row_count": len(direct_rows),
            "proxy_indicator_row_count": len(proxy_rows),
            "context_only_row_count": len(context_rows),
            "observability_row_count": len(audit_rows),
            "retained_block_count": len(retained_blocks),
            "province_count": len(province_axis),
            "month_count": len(month_axis),
        },
        "implementation_scope": {
            "estimated_in_v2": [
                "province latent states",
                "region and national aggregation states",
                "loading magnitudes under sign constraints",
                "observation precisions",
                "province and region shrinkage precisions",
                "temporal persistence coefficients",
            ],
            "fixed_or_prior_constrained_in_v2": [
                "latent block taxonomy",
                "indicator-to-block candidate map",
                "expected loading sign priors",
                "operator family definitions",
            ],
        },
        "phase15_v2_config": cfg,
        "sign_priors": sign_priors,
        "observation_support": observation_support,
        "direct_evidence": direct_evidence,
        "contextual_evidence": contextual_evidence,
        "council": council,
        "equations": equations,
        "evaluation": evaluation,
    }


def render_phase15_v2_markdown(spec: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 15 v2 Mathematical Specification")
    lines.append("")
    lines.append(f"Model id: `{spec.get('model_id', '')}`")
    lines.append("")
    lines.append("## Chairman Synthesis")
    chairman = dict(dict(spec.get("council") or {}).get("chairman_synthesis") or {})
    lines.append(f"- Selected autoresearch variant: `{chairman.get('selected_autoresearch_variant', '')}`")
    lines.append(f"- Recommended backbone: `{chairman.get('recommended_backbone', '')}`")
    lines.append(f"- Why: {chairman.get('why', '')}")
    lines.append("")
    lines.append("## Current Repo State")
    repo_state = dict(spec.get("current_repo_state") or {})
    for key in (
        "normalized_row_count",
        "direct_indicator_row_count",
        "proxy_indicator_row_count",
        "context_only_row_count",
        "observability_row_count",
        "retained_block_count",
        "province_count",
        "month_count",
    ):
        lines.append(f"- `{key}`: {repo_state.get(key)}")
    lines.append("")
    lines.append("## Mathematical Backbone")
    for row in list(spec.get("equations") or []):
        lines.append(f"### {row.get('equation_id', '')}: {row.get('name', '')}")
        lines.append("")
        lines.append(f"`{row.get('math', '')}`")
        lines.append("")
        lines.append(str(row.get("english") or ""))
        lines.append("")
    lines.append("## Observation Support Contract")
    support = dict(spec.get("observation_support") or {})
    lines.append(f"- method: `{support.get('method', '')}`")
    summary = dict(support.get("summary") or {})
    lines.append(f"- operator counts: {summary.get('operator_counts', {})}")
    lines.append(f"- measurement role counts: {summary.get('measurement_role_counts', {})}")
    lines.append("")
    lines.append("## Direct Evidence")
    for row in list(spec.get("direct_evidence") or []):
        lines.append(f"- {row.get('claim', '')}")
        lines.append(f"  Repo mapping: {row.get('repo_surface', '')}")
        for source in list(row.get("sources") or []):
            lines.append(f"  Source: [{source.get('title', '')}]({source.get('url', '')}) ({source.get('date', '')})")
    lines.append("")
    lines.append("## Contextual Evidence")
    for row in list(spec.get("contextual_evidence") or []):
        lines.append(f"- {row.get('claim', '')}")
        lines.append(f"  Repo mapping: {row.get('repo_surface', '')}")
        for source in list(row.get("sources") or []):
            lines.append(f"  Source: [{source.get('title', '')}]({source.get('url', '')}) ({source.get('date', '')})")
    lines.append("")
    lines.append("## Evaluation Gates")
    evaluation = dict(spec.get("evaluation") or {})
    lines.append("### False Wins")
    for row in list(evaluation.get("false_wins") or []):
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### Required Checks")
    for row in list(evaluation.get("required_checks") or []):
        lines.append(f"- {row}")
    lines.append("")
    lines.append("### Keep Or Revert")
    lines.append(str(evaluation.get("keep_or_revert_rule") or ""))
    lines.append("")
    return "\n".join(lines)


def write_phase15_v2_artifacts(
    *,
    output_dir: Path,
    plugin_id: str,
    normalized_rows: list[dict[str, Any]],
    observability_audit: Mapping[str, Any],
    measurement_spec: Mapping[str, Any],
    province_axis: list[str],
    month_axis: list[str],
    region_labels: list[str] | None = None,
) -> dict[str, Any]:
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )
    spec = build_phase15_v2_spec(
        plugin_id=plugin_id,
        normalized_rows=normalized_rows,
        observability_audit=observability_audit,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        observation_support=observation_support,
    )
    markdown = render_phase15_v2_markdown(spec)
    support_path = output_dir / "phase15_v2_observation_support.json"
    spec_json_path = output_dir / "phase15_v2_mathematical_spec.json"
    spec_md_path = output_dir / "phase15_v2_mathematical_spec.md"
    support_path.write_text(json.dumps(observation_support, indent=2), encoding="utf-8")
    spec_json_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec_md_path.write_text(markdown, encoding="utf-8")
    return {
        "observation_support": observation_support,
        "spec": spec,
        "paths": {
            "phase15_v2_observation_support": str(support_path),
            "phase15_v2_mathematical_spec_json": str(spec_json_path),
            "phase15_v2_mathematical_spec_md": str(spec_md_path),
        },
    }


def build_phase15_v2_model_spec(
    *,
    normalized_rows: list[dict[str, Any]],
    observability_audit: Mapping[str, Any],
    axis_catalogs: Mapping[str, list[str]],
    plugin_id: str,
) -> dict[str, Any]:
    province_axis = list(axis_catalogs.get("province") or [])
    month_axis = list(axis_catalogs.get("month") or [])
    region_labels = list(axis_catalogs.get("region") or [])
    measurement_spec = {
        "retained_blocks": [
            {
                "block_id": str(block.get("block_id") or ""),
                "indicator_rows": [
                    {
                        "canonical_name": str(canonical_name),
                        "expected_sign": str(indicator.get("expected_sign") or "neutral"),
                    }
                    for canonical_name, indicator in dict(block.get("indicators") or {}).items()
                ],
            }
            for block in latent_block_specs(plugin_id)
        ]
    }
    observation_support = build_phase15_v2_observation_support(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        region_labels=region_labels,
        include_national_rows=True,
    )
    return build_phase15_v2_spec(
        plugin_id=plugin_id,
        normalized_rows=normalized_rows,
        observability_audit=observability_audit,
        measurement_spec=measurement_spec,
        province_axis=province_axis,
        month_axis=month_axis,
        observation_support=observation_support,
    )
