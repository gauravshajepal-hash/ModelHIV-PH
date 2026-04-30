from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


CHAMPION_CANONICALS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)
CASCade_SIDECAR_CANONICALS: tuple[str, ...] = (
    "art_uptake_rate",
    "newly_enrolled_to_treatment",
    "tested_for_viral_load",
    "virally_suppressed",
    "viral_suppression_rate",
    "suppression_outcomes",
)


def _latest_monthly_phase2_run() -> str:
    return seeded._latest_monthly_phase2_run()


def _load_rows(path: Path, key: str = "rows") -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    rows = list((payload or {}).get(key) or [])
    return [dict(row) for row in rows]


def _category_for_canonical(canonical_name: str) -> str:
    canonical = str(canonical_name)
    if canonical in CHAMPION_CANONICALS:
        return "champion_overlap"
    if canonical in CASCade_SIDECAR_CANONICALS:
        return "cascade_sidecar"
    return "context_or_proxy"


def _time_mix_label(*, monthly_support_count: int, annual_support_count: int) -> str:
    if monthly_support_count > 0 and annual_support_count > 0:
        return "mixed"
    if monthly_support_count > 0:
        return "monthly_only"
    if annual_support_count > 0:
        return "annual_only"
    return "unknown"


def _support_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["canonical_name"]): dict(row) for row in rows if row.get("canonical_name")}


def _parameter_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["block_id"]), str(row["canonical_name"])): dict(row) for row in rows}


def _measurement_time_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"monthly_rows": 0, "annual_rows": 0, "sources": set()})
    for row in rows:
        key = (str(row.get("block_id") or ""), str(row.get("canonical_name") or ""))
        resolution = str(row.get("time_resolution") or "")
        if resolution == "monthly":
            grouped[key]["monthly_rows"] += 1
        elif resolution == "annual":
            grouped[key]["annual_rows"] += 1
        source_bank = str(row.get("source_bank") or "")
        if source_bank:
            grouped[key]["sources"].add(source_bank)
    for key, item in grouped.items():
        monthly_rows = int(item["monthly_rows"])
        annual_rows = int(item["annual_rows"])
        lookup[key] = {
            "monthly_rows": monthly_rows,
            "annual_rows": annual_rows,
            "source_banks": sorted(item["sources"]),
            "measurement_time_mix": _time_mix_label(monthly_support_count=monthly_rows, annual_support_count=annual_rows),
        }
    return lookup


def _build_audit_rows(
    *,
    loading_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    ppc_rows: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    support_lookup = _support_lookup(support_rows)
    parameter_lookup = _parameter_lookup(parameter_rows)
    ppc_lookup = _parameter_lookup(ppc_rows)
    time_lookup = _measurement_time_lookup(measurement_rows)
    audit_rows: list[dict[str, Any]] = []
    for row in loading_rows:
        block_id = str(row["block_id"])
        canonical_name = str(row["canonical_name"])
        support_row = dict(support_lookup.get(canonical_name) or {})
        param_row = dict(parameter_lookup.get((block_id, canonical_name)) or {})
        ppc_row = dict(ppc_lookup.get((block_id, canonical_name)) or {})
        time_row = dict(time_lookup.get((block_id, canonical_name)) or {})
        category = _category_for_canonical(canonical_name)
        direct_count = int(row.get("direct_indicator_count") or 0)
        monthly_support_count = int(support_row.get("monthly_support_count") or 0)
        annual_support_count = int(support_row.get("annual_support_count") or 0)
        sign_conflict = float(param_row.get("sign_conflict") or 0.0)
        weighted_corr = float(param_row.get("weighted_corr") or 0.0)
        ppc_corr = float(ppc_row.get("correlation_with_state") or 0.0)
        measurement_time_mix = str(time_row.get("measurement_time_mix") or _time_mix_label(monthly_support_count=monthly_support_count, annual_support_count=annual_support_count))
        singleton_support = bool(direct_count <= 1)
        annual_only = bool(measurement_time_mix == "annual_only")
        low_fit = bool(ppc_corr < 0.5 and direct_count >= 8)
        risk_score = 0
        risk_score += 4 if category == "champion_overlap" else 0
        risk_score += 2 if category == "cascade_sidecar" else 0
        risk_score += 2 if singleton_support else 0
        risk_score += 1 if annual_only else 0
        risk_score += 1 if low_fit else 0
        risk_score += 3 if sign_conflict > 0.0 else 0
        audit_rows.append(
            {
                "block_id": block_id,
                "canonical_name": canonical_name,
                "category": category,
                "loading": float(row.get("loading") or 0.0),
                "abs_loading": float(abs(float(row.get("loading") or 0.0))),
                "direct_indicator_count": direct_count,
                "monthly_support_count": monthly_support_count,
                "annual_support_count": annual_support_count,
                "source_bank_count": int(support_row.get("source_bank_count") or 0),
                "measurement_time_mix": measurement_time_mix,
                "weighted_corr": weighted_corr,
                "ppc_corr": ppc_corr,
                "ppc_mae": float(ppc_row.get("mean_absolute_error") or 0.0),
                "sign_conflict": sign_conflict,
                "singleton_support": singleton_support,
                "annual_only": annual_only,
                "low_fit": low_fit,
                "risk_score": int(risk_score),
            }
        )
    audit_rows.sort(key=lambda item: (-int(item["risk_score"]), -float(item["abs_loading"]), str(item["block_id"]), str(item["canonical_name"])))
    return audit_rows


def _build_block_summary_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped[str(row["block_id"])].append(dict(row))
    summary_rows: list[dict[str, Any]] = []
    for block_id, rows in sorted(grouped.items()):
        total_mass = float(sum(float(row["abs_loading"]) for row in rows))
        def _share(predicate: str) -> float:
            return float(sum(float(row["abs_loading"]) for row in rows if bool(row[predicate])) / max(total_mass, 1e-9))
        champion_share = float(sum(float(row["abs_loading"]) for row in rows if str(row["category"]) == "champion_overlap") / max(total_mass, 1e-9))
        sidecar_share = float(sum(float(row["abs_loading"]) for row in rows if str(row["category"]) == "cascade_sidecar") / max(total_mass, 1e-9))
        summary_rows.append(
            {
                "block_id": block_id,
                "indicator_count": int(len(rows)),
                "total_loading_mass": total_mass,
                "champion_loading_share": champion_share,
                "cascade_loading_share": sidecar_share,
                "singleton_loading_share": _share("singleton_support"),
                "annual_only_loading_share": _share("annual_only"),
                "mean_ppc_corr": float(np.mean([float(row["ppc_corr"]) for row in rows])) if rows else 0.0,
                "max_risk_score": int(max(int(row["risk_score"]) for row in rows)) if rows else 0,
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_loading_heatmap(audit_rows: list[dict[str, Any]], path: Path) -> None:
    blocks = sorted({str(row["block_id"]) for row in audit_rows})
    canonicals = [str(row["canonical_name"]) for row in audit_rows]
    matrix = np.zeros((len(canonicals), len(blocks)), dtype=np.float64)
    for row_idx, row in enumerate(audit_rows):
        matrix[row_idx, blocks.index(str(row["block_id"]))] = float(row["loading"])
    fig, ax = plt.subplots(figsize=(8, max(5.0, len(canonicals) * 0.33)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks, rotation=25, ha="right")
    ax.set_yticks(range(len(canonicals)))
    ax.set_yticklabels(canonicals)
    ax.set_title("Monthly retained block loadings")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_block_composition(summary_rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["block_id"]) for row in summary_rows]
    champion = [float(row["champion_loading_share"]) for row in summary_rows]
    sidecar = [float(row["cascade_loading_share"]) for row in summary_rows]
    singleton = [float(row["singleton_loading_share"]) for row in summary_rows]
    annual_only = [float(row["annual_only_loading_share"]) for row in summary_rows]
    y = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.65)))
    left = np.zeros(len(labels), dtype=np.float64)
    for values, label, color in [
        (champion, "Champion overlap", "#c44e52"),
        (sidecar, "Cascade sidecar", "#dd8452"),
        (singleton, "Singleton", "#8172b3"),
        (annual_only, "Annual only", "#55a868"),
    ]:
        ax.barh(y, values, left=left, label=label, color=color)
        left += np.asarray(values, dtype=np.float64)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of absolute loading mass")
    ax.set_title("Block composition by risk class")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_loading_support_scatter(audit_rows: list[dict[str, Any]], path: Path) -> None:
    colors = {
        "champion_overlap": "#c44e52",
        "cascade_sidecar": "#dd8452",
        "context_or_proxy": "#4c72b0",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for category in ["champion_overlap", "cascade_sidecar", "context_or_proxy"]:
        rows = [row for row in audit_rows if str(row["category"]) == category]
        if not rows:
            continue
        ax.scatter(
            [float(row["direct_indicator_count"]) for row in rows],
            [float(row["abs_loading"]) for row in rows],
            s=[40 + (25 * float(row["risk_score"])) for row in rows],
            alpha=0.8,
            label=category,
            color=colors[category],
        )
    for row in audit_rows:
        if int(row["risk_score"]) >= 4:
            ax.annotate(
                str(row["canonical_name"]),
                (float(row["direct_indicator_count"]), float(row["abs_loading"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Direct indicator count")
    ax.set_ylabel("Absolute loading")
    ax.set_title("Support vs loading by indicator class")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_bars(audit_rows: list[dict[str, Any]], path: Path, top_n: int = 12) -> None:
    rows = list(audit_rows[:top_n])
    labels = [f"{row['block_id']}:{row['canonical_name']}" for row in rows]
    scores = [int(row["risk_score"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(4.0, len(rows) * 0.45)))
    ax.barh(np.arange(len(rows)), scores, color="#4c72b0")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Risk score")
    ax.set_title("Top loading sanity flags")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Monthly Loading Sanity Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Monthly Phase 2 run: `{payload['monthly_phase2_run_id']}`",
        "",
        "## Why this exists",
        "",
        "- This audit checks whether retained monthly Phase 15 blocks are dominated by direct outcome heads, weak singleton indicators, or annual-only anchors that make the seeded Phase 2 scenarios hard to interpret.",
        "",
        "## Block summary",
        "",
        "| Block | Indicators | Champion share | Cascade sidecar share | Singleton share | Annual-only share | Mean PPC corr | Max risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("block_summary_rows") or []):
        lines.append(
            f"| `{row['block_id']}` | `{int(row['indicator_count'])}` | `{float(row['champion_loading_share']):.3f}` | "
            f"`{float(row['cascade_loading_share']):.3f}` | `{float(row['singleton_loading_share']):.3f}` | "
            f"`{float(row['annual_only_loading_share']):.3f}` | `{float(row['mean_ppc_corr']):.3f}` | `{int(row['max_risk_score'])}` |"
        )
    lines.extend(
        [
            "",
            "## Highest-risk retained indicators",
            "",
            "| Block | Canonical | Category | Loading | Direct count | Time mix | PPC corr | Risk |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in list(payload.get("audit_rows") or [])[:12]:
        lines.append(
            f"| `{row['block_id']}` | `{row['canonical_name']}` | `{row['category']}` | `{float(row['loading']):.3f}` | "
            f"`{int(row['direct_indicator_count'])}` | `{row['measurement_time_mix']}` | `{float(row['ppc_corr']):.3f}` | `{int(row['risk_score'])}` |"
        )
    lines.extend(["", "## Graphs", ""])
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def run_tr_v3_monthly_loading_sanity_batch(
    *,
    run_id: str,
    monthly_phase2_run_id: str | None = None,
) -> dict[str, Any]:
    monthly_run = str(monthly_phase2_run_id or _latest_monthly_phase2_run())
    run_dir = ROOT_DIR / "artifacts" / "runs" / monthly_run / "phase15"
    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")

    loading_rows = _load_rows(run_dir / "national_block_loadings.json")
    support_rows = _load_rows(run_dir / "augmented_latent_observability_audit.json")
    parameter_rows = _load_rows(run_dir / "phase15_v2_indicator_parameters.json")
    ppc_rows = _load_rows(run_dir / "national_block_ppc.json")
    measurement_rows = _load_rows(run_dir / "phase15_v2_measurement_rows.json")

    audit_rows = _build_audit_rows(
        loading_rows=loading_rows,
        support_rows=support_rows,
        parameter_rows=parameter_rows,
        ppc_rows=ppc_rows,
        measurement_rows=measurement_rows,
    )
    block_summary_rows = _build_block_summary_rows(audit_rows)

    csv_path = analysis_dir / "monthly_loading_sanity_table.csv"
    _write_csv(csv_path, audit_rows)

    heatmap_path = analysis_dir / "monthly_loading_heatmap.png"
    _plot_loading_heatmap(audit_rows, heatmap_path)
    composition_path = analysis_dir / "monthly_loading_block_composition.png"
    _plot_block_composition(block_summary_rows, composition_path)
    scatter_path = analysis_dir / "monthly_loading_support_scatter.png"
    _plot_loading_support_scatter(audit_rows, scatter_path)
    risk_path = analysis_dir / "monthly_loading_risk_bars.png"
    _plot_risk_bars(audit_rows, risk_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "monthly_phase2_run_id": monthly_run,
        "audit_rows": audit_rows,
        "block_summary_rows": block_summary_rows,
        "artifacts": {
            "audit_csv": csv_path.name,
            "loading_heatmap": heatmap_path.name,
            "block_composition": composition_path.name,
            "support_scatter": scatter_path.name,
            "risk_bars": risk_path.name,
        },
    }
    write_json(analysis_dir / "tr_v3_monthly_loading_sanity_batch_report.json", payload)
    (analysis_dir / "tr_v3_monthly_loading_sanity_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit retained monthly Phase 15 loadings for circularity and support risks.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monthly-phase2-run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_monthly_loading_sanity_batch(
        run_id=str(args.run_id),
        monthly_phase2_run_id=args.monthly_phase2_run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
