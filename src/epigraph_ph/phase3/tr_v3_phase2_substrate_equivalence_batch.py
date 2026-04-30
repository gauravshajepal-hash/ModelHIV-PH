from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_monthly_loading_sanity_batch as loading_sanity
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_BASELINE_RUN_ID = "tr-v3-monthly-phase2-lane-20260418-s02"
DEFAULT_CANDIDATE_RUN_ID = "tr-v3-monthly-phase2-lane-20260419-s03-coverage-flag"
BLOCK_FAMILY_MAP = {
    "testing_engagement": "testing_family",
    "testing_prevention_reach": "testing_family",
    "care_access_continuity": "care_access_continuity",
    "suppression_capacity": "suppression_capacity",
    "mobility_exposure_pressure": "mobility_exposure_pressure",
}


def _family_name(block_id: str) -> str:
    return str(BLOCK_FAMILY_MAP.get(str(block_id), str(block_id)))


def _loading_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"missing loading report: {path}")
    return payload


def _ensure_loading_report(run_id: str, monthly_phase2_run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.json"
    if not path.exists():
        loading_sanity.run_tr_v3_monthly_loading_sanity_batch(
            run_id=run_id,
            monthly_phase2_run_id=monthly_phase2_run_id,
        )
    return _loading_report(run_id)


def _block_summary_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[_family_name(str(row.get("block_id") or ""))].append(dict(row))
    summary: dict[str, dict[str, Any]] = {}
    for family, items in family_rows.items():
        total_mass = float(sum(float(item.get("total_loading_mass") or 0.0) for item in items))
        indicator_count = int(sum(int(item.get("indicator_count") or 0) for item in items))
        if total_mass <= 0.0:
            total_mass = 1e-9
        weighted = lambda key: float(
            sum(float(item.get(key) or 0.0) * float(item.get("total_loading_mass") or 0.0) for item in items) / total_mass
        )
        summary[family] = {
            "block_family": family,
            "indicator_count": indicator_count,
            "total_loading_mass": float(sum(float(item.get("total_loading_mass") or 0.0) for item in items)),
            "champion_loading_share": weighted("champion_loading_share"),
            "cascade_loading_share": weighted("cascade_loading_share"),
            "singleton_loading_share": weighted("singleton_loading_share"),
            "annual_only_loading_share": weighted("annual_only_loading_share"),
            "mean_ppc_corr": weighted("mean_ppc_corr"),
            "max_risk_score": int(max(int(item.get("max_risk_score") or 0) for item in items)) if items else 0,
            "source_block_ids": sorted(str(item.get("block_id") or "") for item in items),
        }
    return summary


def _canonical_rows_for_family(audit_rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in audit_rows if _family_name(str(row.get("block_id") or "")) == str(family)]
    rows.sort(key=lambda row: str(row.get("canonical_name") or ""))
    return rows


def _testing_equivalence_rows(
    baseline_audit_rows: list[dict[str, Any]],
    candidate_audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_rows = {str(row.get("canonical_name") or ""): dict(row) for row in _canonical_rows_for_family(baseline_audit_rows, "testing_family")}
    candidate_rows = {str(row.get("canonical_name") or ""): dict(row) for row in _canonical_rows_for_family(candidate_audit_rows, "testing_family")}
    canonicals = sorted(set(baseline_rows) | set(candidate_rows))
    out: list[dict[str, Any]] = []
    for canonical in canonicals:
        baseline = dict(baseline_rows.get(canonical) or {})
        candidate = dict(candidate_rows.get(canonical) or {})
        out.append(
            {
                "canonical_name": canonical,
                "baseline_loading": float(baseline.get("loading") or 0.0),
                "candidate_loading": float(candidate.get("loading") or 0.0),
                "baseline_abs_loading": float(abs(float(baseline.get("loading") or 0.0))),
                "candidate_abs_loading": float(abs(float(candidate.get("loading") or 0.0))),
                "baseline_direct_count": int(baseline.get("direct_indicator_count") or 0),
                "candidate_direct_count": int(candidate.get("direct_indicator_count") or 0),
                "baseline_time_mix": str(baseline.get("measurement_time_mix") or ""),
                "candidate_time_mix": str(candidate.get("measurement_time_mix") or ""),
                "loading_delta": float(abs(float(candidate.get("loading") or 0.0)) - abs(float(baseline.get("loading") or 0.0))),
            }
        )
    return out


def _family_overlap_row(
    *,
    family: str,
    baseline_audit_rows: list[dict[str, Any]],
    candidate_audit_rows: list[dict[str, Any]],
    baseline_summary: dict[str, dict[str, Any]],
    candidate_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = _canonical_rows_for_family(baseline_audit_rows, family)
    candidate_rows = _canonical_rows_for_family(candidate_audit_rows, family)
    baseline_set = {str(row.get("canonical_name") or "") for row in baseline_rows}
    candidate_set = {str(row.get("canonical_name") or "") for row in candidate_rows}
    intersection = baseline_set & candidate_set
    union = baseline_set | candidate_set
    jaccard = float(len(intersection) / max(len(union), 1))
    return {
        "block_family": family,
        "baseline_indicator_count": int(dict(baseline_summary.get(family) or {}).get("indicator_count") or 0),
        "candidate_indicator_count": int(dict(candidate_summary.get(family) or {}).get("indicator_count") or 0),
        "shared_indicator_count": int(len(intersection)),
        "union_indicator_count": int(len(union)),
        "indicator_jaccard": jaccard,
        "baseline_mass": float(dict(baseline_summary.get(family) or {}).get("total_loading_mass") or 0.0),
        "candidate_mass": float(dict(candidate_summary.get(family) or {}).get("total_loading_mass") or 0.0),
        "baseline_champion_share": float(dict(baseline_summary.get(family) or {}).get("champion_loading_share") or 0.0),
        "candidate_champion_share": float(dict(candidate_summary.get(family) or {}).get("champion_loading_share") or 0.0),
        "baseline_annual_only_share": float(dict(baseline_summary.get(family) or {}).get("annual_only_loading_share") or 0.0),
        "candidate_annual_only_share": float(dict(candidate_summary.get(family) or {}).get("annual_only_loading_share") or 0.0),
    }


def _plot_family_jaccard(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["block_family"]) for row in rows]
    values = [float(row["indicator_jaccard"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Indicator Jaccard overlap")
    ax.set_title("Block-family substrate overlap")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_testing_loading_compare(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["canonical_name"]) for row in rows]
    baseline = [float(row["baseline_abs_loading"]) for row in rows]
    candidate = [float(row["candidate_abs_loading"]) for row in rows]
    y = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.45)))
    ax.barh(y - width / 2.0, baseline, height=width, label="baseline")
    ax.barh(y + width / 2.0, candidate, height=width, label="merged")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute loading")
    ax.set_title("Testing-family canonical loadings")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_family_mass_compare(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["block_family"]) for row in rows]
    baseline = [float(row["baseline_mass"]) for row in rows]
    candidate = [float(row["candidate_mass"]) for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2.0, baseline, width=width, label="baseline")
    ax.bar(x + width / 2.0, candidate, width=width, label="merged")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Total loading mass")
    ax.set_title("Block-family loading mass comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Substrate Equivalence Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline monthly run: `{payload['baseline_monthly_run_id']}`",
        f"- Candidate monthly run: `{payload['candidate_monthly_run_id']}`",
        "",
        "## Family overlap",
        "",
        "| Family | Baseline indicators | Candidate indicators | Shared | Jaccard | Baseline mass | Candidate mass | Baseline annual-only | Candidate annual-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("family_overlap_rows") or []):
        lines.append(
            f"| `{row['block_family']}` | `{int(row['baseline_indicator_count'])}` | `{int(row['candidate_indicator_count'])}` | "
            f"`{int(row['shared_indicator_count'])}` | `{float(row['indicator_jaccard']):.3f}` | "
            f"`{float(row['baseline_mass']):.3f}` | `{float(row['candidate_mass']):.3f}` | "
            f"`{float(row['baseline_annual_only_share']):.3f}` | `{float(row['candidate_annual_only_share']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Testing-family canonical comparison",
            "",
            "| Canonical | Baseline abs loading | Candidate abs loading | Baseline direct count | Candidate direct count | Baseline time mix | Candidate time mix |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in list(payload.get("testing_equivalence_rows") or []):
        lines.append(
            f"| `{row['canonical_name']}` | `{float(row['baseline_abs_loading']):.3f}` | `{float(row['candidate_abs_loading']):.3f}` | "
            f"`{int(row['baseline_direct_count'])}` | `{int(row['candidate_direct_count'])}` | "
            f"`{row['baseline_time_mix'] or 'none'}` | `{row['candidate_time_mix'] or 'none'}` |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def run_tr_v3_phase2_substrate_equivalence_batch(
    *,
    run_id: str,
    baseline_monthly_run_id: str = DEFAULT_BASELINE_RUN_ID,
    candidate_monthly_run_id: str = DEFAULT_CANDIDATE_RUN_ID,
) -> dict[str, Any]:
    baseline_loading_run = f"{run_id}-baseline-loading"
    candidate_loading_run = f"{run_id}-candidate-loading"
    baseline_loading = _ensure_loading_report(baseline_loading_run, baseline_monthly_run_id)
    candidate_loading = _ensure_loading_report(candidate_loading_run, candidate_monthly_run_id)

    baseline_audit_rows = [dict(row) for row in list(baseline_loading.get("audit_rows") or [])]
    candidate_audit_rows = [dict(row) for row in list(candidate_loading.get("audit_rows") or [])]
    baseline_summary = _block_summary_by_family([dict(row) for row in list(baseline_loading.get("block_summary_rows") or [])])
    candidate_summary = _block_summary_by_family([dict(row) for row in list(candidate_loading.get("block_summary_rows") or [])])

    families = sorted(set(baseline_summary) | set(candidate_summary))
    family_overlap_rows = [
        _family_overlap_row(
            family=family,
            baseline_audit_rows=baseline_audit_rows,
            candidate_audit_rows=candidate_audit_rows,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
        )
        for family in families
    ]
    testing_equivalence_rows = _testing_equivalence_rows(baseline_audit_rows, candidate_audit_rows)

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    family_jaccard_plot = analysis_dir / "substrate_family_jaccard.png"
    testing_loading_plot = analysis_dir / "substrate_testing_loading_compare.png"
    family_mass_plot = analysis_dir / "substrate_family_mass_compare.png"
    _plot_family_jaccard(family_overlap_rows, family_jaccard_plot)
    _plot_testing_loading_compare(testing_equivalence_rows, testing_loading_plot)
    _plot_family_mass_compare(family_overlap_rows, family_mass_plot)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_monthly_run_id": str(baseline_monthly_run_id),
        "candidate_monthly_run_id": str(candidate_monthly_run_id),
        "baseline_loading_run_id": str(baseline_loading_run),
        "candidate_loading_run_id": str(candidate_loading_run),
        "family_overlap_rows": family_overlap_rows,
        "testing_equivalence_rows": testing_equivalence_rows,
        "artifacts": {
            "family_jaccard_plot": family_jaccard_plot.name,
            "testing_loading_plot": testing_loading_plot.name,
            "family_mass_plot": family_mass_plot.name,
            "baseline_loading_report": str(Path(baseline_loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
            "candidate_loading_report": str(Path(candidate_loading_run) / "analysis" / "tr_v3_monthly_loading_sanity_batch_report.md"),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_substrate_equivalence_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_substrate_equivalence_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and merged monthly Phase 2 substrates at the block-measurement level.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-monthly-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--candidate-monthly-run-id", default=DEFAULT_CANDIDATE_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_substrate_equivalence_batch(
        run_id=str(args.run_id),
        baseline_monthly_run_id=str(args.baseline_monthly_run_id),
        candidate_monthly_run_id=str(args.candidate_monthly_run_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
