from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_phase2_substrate_equivalence_batch as substrate
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_BASELINE_RUN_ID = substrate.DEFAULT_BASELINE_RUN_ID
DEFAULT_CANDIDATE_RUN_ID = substrate.DEFAULT_CANDIDATE_RUN_ID
TESTING_SCENARIOS: tuple[str, ...] = ("testing_pulse", "testing_plateau")
TESTING_BLOCK_NAMES: tuple[str, ...] = ("testing_prevention_reach", "testing_engagement")


def _current_winner_configs() -> dict[str, dict[str, Any]]:
    return {
        "exact_only": {
            "winner_id": str(suite.default_predictive_candidate_id("exact_only")),
            "suite_contract": "exact_only",
            "forecast_contract": "exact_only",
            "allowed_tiers": {"exact_observed"},
            "title": "Exact champion",
        },
        "purged_dense": {
            "winner_id": str(suite.default_predictive_candidate_id("dense_train_observed_score")),
            "suite_contract": "purged_dense",
            "forecast_contract": "dense_train_observed_score",
            "allowed_tiers": {"exact_observed", "bridge_observed"},
            "title": "Dense champion",
        },
    }


def _seeded_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"missing seeded champion report: {path}")
    return payload


def _ensure_seeded_report(run_id: str, monthly_phase2_run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.json"
    if not path.exists():
        seeded.run_tr_v3_phase2_seeded_champion_batch(
            run_id=run_id,
            archive_run_id=monthly_phase2_run_id,
            readout_source_archive_run_id=monthly_phase2_run_id,
            monthly_phase2_run_id=monthly_phase2_run_id,
            winner_configs=_current_winner_configs(),
        )
    return _seeded_report(run_id)


def _testing_feature_name(feature_names: list[str], *, delta: bool) -> str:
    suffix = "_delta" if delta else ""
    for block_name in TESTING_BLOCK_NAMES:
        candidate = f"{block_name}{suffix}"
        if candidate in set(feature_names):
            return candidate
    return ""


def _scaled_feature_coeff(readout: dict[str, Any], metric_name: str, feature_name: str) -> float:
    if not feature_name:
        return 0.0
    feature_names = list(readout.get("feature_names") or [])
    if feature_name not in set(feature_names):
        return 0.0
    metric_payload = dict(dict(readout.get("metrics") or {}).get(metric_name) or {})
    beta = np.asarray(metric_payload.get("beta") or [], dtype=np.float64)
    if beta.size != len(feature_names):
        return 0.0
    scale = float(metric_payload.get("scale") or 0.0)
    return float(scale * beta[feature_names.index(feature_name)])


def _testing_readout_rows(baseline_report: dict[str, Any], candidate_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contract_names = sorted(set(dict(baseline_report.get("contracts") or {})) | set(dict(candidate_report.get("contracts") or {})))
    for contract_name in contract_names:
        baseline_contract = dict(dict(baseline_report.get("contracts") or {}).get(contract_name) or {})
        candidate_contract = dict(dict(candidate_report.get("contracts") or {}).get(contract_name) or {})
        baseline_readout = dict(baseline_contract.get("readout") or {})
        candidate_readout = dict(candidate_contract.get("readout") or {})
        baseline_feature_names = list(baseline_readout.get("feature_names") or [])
        candidate_feature_names = list(candidate_readout.get("feature_names") or [])
        for metric_name in seeded.METRIC_PLOT_ORDER:
            for feature_kind, is_delta in (("level", False), ("delta", True)):
                baseline_feature_name = _testing_feature_name(baseline_feature_names, delta=is_delta)
                candidate_feature_name = _testing_feature_name(candidate_feature_names, delta=is_delta)
                baseline_coeff = _scaled_feature_coeff(baseline_readout, metric_name, baseline_feature_name)
                candidate_coeff = _scaled_feature_coeff(candidate_readout, metric_name, candidate_feature_name)
                rows.append(
                    {
                        "contract": str(contract_name),
                        "metric": str(metric_name),
                        "feature_kind": str(feature_kind),
                        "baseline_feature_name": str(baseline_feature_name or "none"),
                        "candidate_feature_name": str(candidate_feature_name or "none"),
                        "baseline_scaled_coeff": float(baseline_coeff),
                        "candidate_scaled_coeff": float(candidate_coeff),
                        "baseline_abs_coeff": float(abs(baseline_coeff)),
                        "candidate_abs_coeff": float(abs(candidate_coeff)),
                    }
                )
    return rows


def _terminal_delta_lookup(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in list(payload.get("terminal_delta_rows") or []):
        contract = str(row.get("contract") or "")
        scenario = str(row.get("scenario") or "")
        for metric_name in seeded.METRIC_PLOT_ORDER:
            lookup[(contract, scenario, metric_name)] = float(row.get(f"{metric_name}_delta") or 0.0)
    return lookup


def _testing_scenario_rows(baseline_report: dict[str, Any], candidate_report: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_lookup = _terminal_delta_lookup(baseline_report)
    candidate_lookup = _terminal_delta_lookup(candidate_report)
    keys = sorted(set(baseline_lookup) | set(candidate_lookup))
    rows: list[dict[str, Any]] = []
    for contract_name, scenario_name, metric_name in keys:
        if scenario_name not in set(TESTING_SCENARIOS):
            continue
        baseline_delta = float(baseline_lookup.get((contract_name, scenario_name, metric_name), 0.0))
        candidate_delta = float(candidate_lookup.get((contract_name, scenario_name, metric_name), 0.0))
        rows.append(
            {
                "contract": str(contract_name),
                "scenario": str(scenario_name),
                "metric": str(metric_name),
                "baseline_delta": float(baseline_delta),
                "candidate_delta": float(candidate_delta),
                "baseline_abs_delta": float(abs(baseline_delta)),
                "candidate_abs_delta": float(abs(candidate_delta)),
            }
        )
    return rows


def _plot_testing_readout_compare(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No testing-feature readout rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [f"{row['contract']}:{row['metric']}:{row['feature_kind']}" for row in rows]
    baseline = [float(row["baseline_abs_coeff"]) for row in rows]
    candidate = [float(row["candidate_abs_coeff"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.42)))
    ax.barh(y - width / 2.0, baseline, height=width, label="baseline")
    ax.barh(y + width / 2.0, candidate, height=width, label="merged")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute scaled readout coefficient per 1 SD feature shift")
    ax.set_title("Current-champion testing-feature readout comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_testing_terminal_delta_compare(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No testing-scenario terminal deltas", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [f"{row['contract']}:{row['scenario']}:{row['metric']}" for row in rows]
    baseline = [float(row["baseline_abs_delta"]) for row in rows]
    candidate = [float(row["candidate_abs_delta"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(labels) * 0.42)))
    ax.barh(y - width / 2.0, baseline, height=width, label="baseline")
    ax.barh(y + width / 2.0, candidate, height=width, label="merged")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute terminal delta vs base champion")
    ax.set_title("Current-champion testing-scenario terminal deltas")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Champion-Conditioned Equivalence Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline monthly run: `{payload['baseline_monthly_run_id']}`",
        f"- Candidate monthly run: `{payload['candidate_monthly_run_id']}`",
        f"- Exact champion: `{payload['current_champions']['exact_only']}`",
        f"- Dense champion: `{payload['current_champions']['purged_dense']}`",
        "",
        "## Testing-family substrate comparison",
        "",
        "| Canonical | Baseline abs loading | Candidate abs loading | Baseline direct count | Candidate direct count | Baseline time mix | Candidate time mix |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in list(payload.get("testing_equivalence_rows") or []):
        lines.append(
            f"| `{row['canonical_name']}` | `{float(row['baseline_abs_loading']):.3f}` | `{float(row['candidate_abs_loading']):.3f}` | "
            f"`{int(row['baseline_direct_count'])}` | `{int(row['candidate_direct_count'])}` | "
            f"`{row['baseline_time_mix'] or 'none'}` | `{row['candidate_time_mix'] or 'none'}` |"
        )
    lines.extend(
        [
            "",
        "## Current-champion testing readout",
        "",
        "| Contract | Metric | Feature kind | Baseline feature | Candidate feature | Baseline coeff per 1 SD | Candidate coeff per 1 SD |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in list(payload.get("testing_readout_rows") or []):
        lines.append(
            f"| `{row['contract']}` | `{row['metric']}` | `{row['feature_kind']}` | "
            f"`{row['baseline_feature_name']}` | `{row['candidate_feature_name']}` | "
            f"`{float(row['baseline_scaled_coeff']):.4f}` | `{float(row['candidate_scaled_coeff']):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Current-champion testing scenario deltas",
            "",
            "| Contract | Scenario | Metric | Baseline delta | Candidate delta |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in list(payload.get("testing_scenario_rows") or []):
        lines.append(
            f"| `{row['contract']}` | `{row['scenario']}` | `{row['metric']}` | "
            f"`{float(row['baseline_delta']):.3f}` | `{float(row['candidate_delta']):.3f}` |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def run_tr_v3_phase2_champion_equivalence_batch(
    *,
    run_id: str,
    baseline_monthly_run_id: str = DEFAULT_BASELINE_RUN_ID,
    candidate_monthly_run_id: str = DEFAULT_CANDIDATE_RUN_ID,
) -> dict[str, Any]:
    baseline_loading_run = f"{run_id}-baseline-loading"
    candidate_loading_run = f"{run_id}-candidate-loading"
    baseline_seeded_run = f"{run_id}-baseline-seeded"
    candidate_seeded_run = f"{run_id}-candidate-seeded"

    baseline_loading = substrate._ensure_loading_report(baseline_loading_run, baseline_monthly_run_id)
    candidate_loading = substrate._ensure_loading_report(candidate_loading_run, candidate_monthly_run_id)
    baseline_seeded = _ensure_seeded_report(baseline_seeded_run, baseline_monthly_run_id)
    candidate_seeded = _ensure_seeded_report(candidate_seeded_run, candidate_monthly_run_id)

    baseline_audit_rows = [dict(row) for row in list(baseline_loading.get("audit_rows") or [])]
    candidate_audit_rows = [dict(row) for row in list(candidate_loading.get("audit_rows") or [])]
    baseline_summary = substrate._block_summary_by_family([dict(row) for row in list(baseline_loading.get("block_summary_rows") or [])])
    candidate_summary = substrate._block_summary_by_family([dict(row) for row in list(candidate_loading.get("block_summary_rows") or [])])
    families = sorted(set(baseline_summary) | set(candidate_summary))
    family_overlap_rows = [
        substrate._family_overlap_row(
            family=family,
            baseline_audit_rows=baseline_audit_rows,
            candidate_audit_rows=candidate_audit_rows,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
        )
        for family in families
    ]
    testing_equivalence_rows = substrate._testing_equivalence_rows(baseline_audit_rows, candidate_audit_rows)
    testing_readout_rows = _testing_readout_rows(baseline_seeded, candidate_seeded)
    testing_scenario_rows = _testing_scenario_rows(baseline_seeded, candidate_seeded)

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    family_jaccard_plot = analysis_dir / "substrate_family_jaccard.png"
    testing_loading_plot = analysis_dir / "substrate_testing_loading_compare.png"
    family_mass_plot = analysis_dir / "substrate_family_mass_compare.png"
    testing_readout_plot = analysis_dir / "champion_testing_readout_compare.png"
    testing_terminal_plot = analysis_dir / "champion_testing_terminal_delta_compare.png"
    substrate._plot_family_jaccard(family_overlap_rows, family_jaccard_plot)
    substrate._plot_testing_loading_compare(testing_equivalence_rows, testing_loading_plot)
    substrate._plot_family_mass_compare(family_overlap_rows, family_mass_plot)
    _plot_testing_readout_compare(testing_readout_rows, testing_readout_plot)
    _plot_testing_terminal_delta_compare(testing_scenario_rows, testing_terminal_plot)

    current_champions = {
        contract_name: str(config["winner_id"])
        for contract_name, config in _current_winner_configs().items()
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_monthly_run_id": str(baseline_monthly_run_id),
        "candidate_monthly_run_id": str(candidate_monthly_run_id),
        "baseline_loading_run_id": str(baseline_loading_run),
        "candidate_loading_run_id": str(candidate_loading_run),
        "baseline_seeded_run_id": str(baseline_seeded_run),
        "candidate_seeded_run_id": str(candidate_seeded_run),
        "current_champions": current_champions,
        "family_overlap_rows": family_overlap_rows,
        "testing_equivalence_rows": testing_equivalence_rows,
        "testing_readout_rows": testing_readout_rows,
        "testing_scenario_rows": testing_scenario_rows,
        "artifacts": {
            "family_jaccard_plot": family_jaccard_plot.name,
            "testing_loading_plot": testing_loading_plot.name,
            "family_mass_plot": family_mass_plot.name,
            "testing_readout_plot": testing_readout_plot.name,
            "testing_terminal_plot": testing_terminal_plot.name,
            "baseline_loading_report": str(Path(baseline_loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "candidate_loading_report": str(Path(candidate_loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "baseline_seeded_report": str(Path(baseline_seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
            "candidate_seeded_report": str(Path(candidate_seeded_run) / "analysis" / "tr_v3_phase2_seeded_champion_batch_report.md"),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_champion_equivalence_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_champion_equivalence_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and merged monthly Phase 2 testing substrates under the current predictive champions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-monthly-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--candidate-monthly-run-id", default=DEFAULT_CANDIDATE_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_champion_equivalence_batch(
        run_id=str(args.run_id),
        baseline_monthly_run_id=str(args.baseline_monthly_run_id),
        candidate_monthly_run_id=str(args.candidate_monthly_run_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
