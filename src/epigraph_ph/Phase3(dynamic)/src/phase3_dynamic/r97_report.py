"""Build a portable, hash-linked R97 report and scientific figure archive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from .r96_monthly_diagnosis_state import FAMILIES, Q2_PDF, _digest, _write


LABELS = {"quarter_carry": "Quarter carry-forward", "last_month": "Last month repeated",
          "historical_mean": "Historical mean", "local_level": "Local level",
          "local_level_drift": "Local level + drift"}
PREFIX = "phase3_r97"


def figures(report: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={"wspace": .38, "hspace": .62})
    blue, orange = "#23768A", "#B9622D"
    accepted = [r for r in report["catalog"] if r["status"] == "accepted"]
    ax = axes[0, 0]
    for i, row in enumerate(accepted):
        end = datetime.fromisoformat(row["end_month"] + "-01")
        # Use the exact quarter-end day, not the first of its final month.
        end = end.replace(month=end.month + 1) if end.month < 12 else end.replace(year=end.year + 1, month=1)
        end -= timedelta(days=1)
        posted = datetime.fromisoformat(row["mirror_posted_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        ax.plot([end, posted], [i, i], color="#C3CBCE", lw=2)
        ax.scatter(end, i, color=blue, s=32, label="Report period ends" if i == 0 else None)
        ax.scatter(posted, i, marker="s", color=orange, s=32, label="Declared mirror post" if i == 0 else None)
    ax.set_yticks(range(len(accepted)), [r["period"] for r in accepted])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("A  Event time is not information availability", loc="left", weight="bold")
    ax.set_xlabel("Calendar date; mirror posting is NOT verified first release")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=.15)

    ax = axes[0, 1]
    pairs = list(dict.fromkeys((r["earlier_period"], r["later_period"]) for r in report["revisions"]))
    months = sorted({r["month"] for r in report["revisions"]})
    matrix = np.full((len(pairs), len(months)), np.nan)
    for row in report["revisions"]:
        matrix[pairs.index((row["earlier_period"], row["later_period"])), months.index(row["month"])] = row["revision"]
    vmax = max(1, float(np.nanmax(np.abs(matrix))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E4E7E8")
    heat = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(pairs)), [f"{a} to {b}" for a, b in pairs], fontsize=8)
    ticks = list(range(0, len(months), 6))
    ax.set_xticks(ticks, [months[t] for t in ticks], rotation=35, ha="right")
    biggest = max(report["revisions"], key=lambda r: abs(r["revision"]))
    ax.text(months.index(biggest["month"]), pairs.index((biggest["earlier_period"], biggest["later_period"])),
            str(biggest["revision"]), color="white", ha="center", va="center", fontsize=7, weight="bold")
    fig.colorbar(heat, ax=ax, fraction=.035, pad=.03).set_label("Later minus earlier count (people)")
    ax.set_title("B  Historical monthly counts are revised downward", loc="left", weight="bold")
    ax.set_xlabel("Diagnosis month; grey = no comparable pair; cells are not independent")

    ax = axes[1, 0]
    x = np.arange(len(FAMILIES))
    for i, period in enumerate(("2026-Q1", "2026-Q2")):
        rows = {r["family"]: r for r in report["availability_replays"] if r["target_period"] == period}
        rects = ax.bar(x + (i - .5) * .36, [rows[f]["absolute_error"] for f in FAMILIES], width=.36,
                      color=(blue, orange)[i], label=period)
        ax.bar_label(rects, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Quarter\ncarry", "Last\nmonth", "Historical\nmean", "Local\nlevel", "Level\n+ drift"])
    ax.set_ylabel("Quarterly absolute error (diagnoses)")
    ax.set_title("C  Mirror-availability sensitivity replay", loc="left", weight="bold", pad=23)
    ax.text(0, 1.01, "Only 2 quarters; selector defaults to carry at both historical origins", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(r["absolute_error"] for r in report["availability_replays"]) * 1.2)

    ax = axes[1, 1]
    lock = report["prospective_Q4"]
    rows = lock["monthly_candidates"]
    points = [r["prediction"] for r in rows] + [lock["R41_reference_unchanged"]["new_diagnosed_cases_period"]]
    names = [LABELS[r["family"]] + (" [selected]" if r["selected"] else "") for r in rows] + ["Frozen R41 readout"]
    colors = [orange if r["selected"] else blue for r in rows] + ["#777777"]
    bars = ax.barh(names, points, color=colors, height=.6)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(points) * 1.15)
    ax.set_xlabel("Predicted reported diagnoses in Oct-Dec 2026 (people)")
    ax.set_title("D  Q4-2026 forecast lock: UNOBSERVED / UNSCORED", loc="left", weight="bold", pad=23)
    ax.text(0, 1.01, "42 training months through June; forecast through 3-month missing gap", transform=ax.transAxes, fontsize=8)
    fig.suptitle("R97 | Report revisions and honest forecast timing", x=.07, ha="left", fontsize=20, weight="bold")
    fig.text(.07, .92, f"{len(accepted)} accepted HASP reports | {report['monthly_vintage_rows']} vintage rows, not independent months | no model promotion", fontsize=12)
    fig.subplots_adjust(top=.86, bottom=.13, left=.08, right=.95)
    fig.text(.07, .025, "Sources: DOH HASP monthly tables, SHIP posting metadata. R97 s01, 11 September 2026.\n"
             "Point forecasts only: intervals are not calibrated. Original release dates and historical file bytes are unverified.\n"
             "Q4 candidates retain R41 stocks but do not establish stock-flow coupling or national/regional/AEM superiority.", fontsize=9, color="#555555")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{PREFIX}_dashboard.{extension}", dpi=260, facecolor="white")
    plt.close(fig)


def markdown(report: dict) -> str:
    revisions = report["revisions"]
    negative = sum(r["revision"] < 0 for r in revisions)
    positive = sum(r["revision"] > 0 for r in revisions)
    lines = ["# R97: Report vintages, availability sensitivity, and a prospective forecast lock", "",
             f"Run: `{report['run_id']}`. **Diagnostic-only. R41 is unchanged; no new model champion.**", "",
             "![R97 diagnosis reporting dashboard](figures/phase3_r97_dashboard.png)", "",
             "## Scientific finding", "",
             f"Eight quarterly reports yield {report['monthly_vintage_rows']} report-month rows. Two reports remain quarantined because OCR does not agree. "
             f"Among {len(revisions)} adjacent-vintage same-month comparisons, {negative} change downward and {positive} upward. "
             "These are repeated measurements of overlapping months, not independent samples. September 2025 is revised from 1,799 to 1,708 diagnoses (-91). "
             "This falsifies an additions-only representation of these observed revisions. It does not identify their cause as deduplication, reporting failure, or a biological change.", "",
             "The SHIP mirror lists Q4-2025 on 26 February 2026 and both Q1/Q2-2026 on 26 August 2026. "
             "Under that declared-availability assumption, January 2026 can use Q3-2025, and April can use Q4-2025. "
             "Neither replay can use the preceding quarter as if it had already been published. "
             "[Source: SHIP registry](https://www.ship.ph/category/hiv-aids-art-registry/). "
             "These are **not verified original DOH release dates**, and currently mirrored PDF bytes are not proof of what a forecaster could download historically. "
             "The replay is therefore a sensitivity analysis, not repaired prospective evidence.", "",
             "## What was implemented", "",
             "- Preserve each report's own history and SHA-256; never overwrite an earlier history with later revisions.",
             "- Extract numeric PDF rows or require unique agreement between Tesseract segmentation modes 3 and 6. "
             "Use a whole-page fallback only when the right-column extraction fails. Agreement is a consistency check, not independent proof of OCR accuracy.",
             "- Check rounded printed averages where available. Mask zeros printed after a report's coverage end; they are not observed future zeros.",
             "- Filter training reports by both period end and posting timestamp. Propagate every intervening unobserved month.",
             "- Select among the five frozen R96 families only using prior target blocks whose reports were posted by the origin. "
             "No complete selection blocks are available at the two historical origins, so the selector uses carry-forward.",
             "- Keep confirming-laboratory counts as context only, not monthly reporting-completeness denominators or incidence truth.", "",
             "## Mathematics and meaning", "",
             r"Let $y_t^{(v)}$ be the reported diagnosis count for month $t$ in report vintage $v$, $e_v$ its coverage end, and $p_v$ its mirror posting time. At issue time $o$, choose", "",
             r"$$v(o)=\arg\max_{v:p_v\le o,\ e_v<o} e_v.$$", "",
             "In English: use the latest report that the declared source had actually posted, not the report whose calendar label looks most recent.", "",
             r"For a target quarter starting in month $s$, let $g=s-e_{v(o)}-1$ be the unobserved gap in months. Then", "",
             r"$$\widehat Y_{s:s+2}^{(f,o)}=\sum_{h=g+1}^{g+3}\widehat y_{e_{v(o)}+h}^{(f)}.$$", "",
             "In English: forecast through missing intermediate months before summing the three target months. "
             "A quarterly carry-forward repeats the last observed three months, a last-month baseline repeats one count, and a historical mean averages only the selected report's available prefix.", "",
             r"The frozen state families use $z_t=\log(1+y_t)$, $z_t=\ell_t+\epsilon_t$, and $\ell_t=\ell_{t-1}+d+\eta_t$. "
             r"Here $\epsilon_t\sim N(0,R)$ is observation noise, $\eta_t\sim N(0,Q)$ is state noise, and $d=0$ for local level. "
             "The drift and variances are fitted using only the training history, with exact diffuse initialization. "
             "They represent reported-count intensity, not a separately identified biological incidence hazard. "
             "Inverse-log point forecasts are not asserted to be calibrated predictive means or intervals; "
             "[R96 documents the fitting and limitations](phase3_r96_monthly_diagnosis_state_20260911.md).", "",
             r"The revision diagnostic is $\Delta_t^{v,w}=y_t^{(w)}-y_t^{(v)}$ for $w>v$. Negative values require a signed revision process if modeled. "
             "Overlapping vintage differences must not be treated as independent observations of a delay distribution.", "",
             "## Comparison under the same availability rule", "",
             "The outcome is the sum of that target report's three monthly table cells, not annual incidence or a substituted headline count. "
             "Q1 truth is 4,633; Q2 truth is 2,994. Original R96/R41 retrospective scores use a different availability contract and cannot be ranked against this table as if all inputs matched.", "",
             "| Family | Q1 prediction | Q1 absolute error | Q2 prediction | Q2 absolute error | Mean error |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        rows = [r for r in report["availability_replays"] if r["family"] == family]
        a, b = rows
        lines.append(f"| {LABELS[family]} | {a['prediction']:,.1f} | {a['absolute_error']:,.1f} | {b['prediction']:,.1f} | {b['absolute_error']:,.1f} | {np.mean([r['absolute_error'] for r in rows]):,.1f} |")
    lines += ["", "The historical mean has the lowest pooled error, but loses to carry-forward on Q2. "
              "The historical selector cannot retroactively pick it for either origin. Two previously inspected quarters do not support a superiority claim.", "",
              "## Prospective Q4-2026 lock", "",
              f"Generated `{report['prospective_Q4']['generated_at']}`, before 1 October 2026. Training ends June 2026; July-September are an unobserved gap. "
              "The previously specified selector now has both published challenge outcomes and chooses the historical mean. "
              "This family selection is development on known outcomes; only Q4 is future validation.", "",
              "| Candidate | Oct-Dec 2026 reported diagnoses | Status |", "|---|---:|---|"]
    for row in report["prospective_Q4"]["monthly_candidates"]:
        lines.append(f"| {LABELS[row['family']]} | {row['prediction']:,.1f} | {'Selected' if row['selected'] else 'Frozen comparator'}; unscored |")
    lines += [f"| Frozen R41 readout | {report['prospective_Q4']['R41_reference_unchanged']['new_diagnosed_cases_period']:,.1f} | Unscored |", "",
              "Use the first complete Q4 HASP monthly table for the primary outcome; score later revisions separately. "
              "Missing outcomes stay null. One future quarter cannot promote a full-cascade model. "
              "These are point forecasts, not fan charts with calibrated uncertainty. The R41 stock vector remains unchanged and satisfies its stock cone; "
              "that does **not** prove the diagnosis-flow replacements are mass-balanced with those stocks.", "",
              "[Immutable prospective lock](phase3_r97_prospective_Q4_lock_20260911.json) | "
              "[Full results](phase3_r97_results_20260911.json) | [Observation ledger](phase3_r97_observation_ledger_20260911.json) | "
              "[Reproducibility manifest](phase3_r97_bundle_manifest_20260911.json)", "",
              "## Cross-domain transfers and next experiments", "",
              "1. Multi-epoch survey calibration: compare the same time cell across report vintages, analogous to repeated measurements under revised calibration. "
              "This is valid for detecting measurement revision, not for assuming its cause. Test a signed revision observation model only after more vintage pairs and first-release evidence are available.",
              "2. Delayed-measurement filtering: carry the latent state through unobserved months instead of pretending a late report arrived at its event date. "
              "The delay mapping is explicit in the equation above; the uncertainty grows with the gap. Test it against the frozen Q4 predictions and subsequently calibrated interval coverage.",
              "3. Highest-value mechanism check: represent unknown CD4/AHD status explicitly before using late-diagnosis fractions to identify backlog/incidence. "
              "The Q2 footnote states that 1,732 of 2,166 nominally non-advanced cases lack immunologic/clinical criteria. "
              "Do not turn missingness into evidence of early disease or impute a causal re-engagement channel. "
              "[DOH Q2-2026 report, page 1](https://www.ship.ph/wp-content/uploads/2026/08/2026_Q2-HIV-AIDS-Surveillance-of-the-Philippines-2.pdf).", "",
              "## Reproduction", "", "```bash",
              "export PYTHONPATH='src/epigraph_ph/Phase3(dynamic)/src'",
              "# numpy, scipy; pdftotext/pdftoppm; Tesseract with English data for image tables",
              "python3 -m phase3_dynamic.r97_hasp_report_vintages --output-dir /path/to/new-run --tesseract /path/to/tesseract",
              "# matplotlib additionally required for the report bundle",
              "python3 -m phase3_dynamic.r97_report --report /path/to/new-run/report.json --project-root /path/to/ModelHIV-PH",
              "```", "",
              "The freeze command refuses to run once Q4 has begun. For later reproduction of the historical results, use the tracked report, "
              "OCR/metadata snapshots and their checksums rather than creating a backdated prospective lock. "
              "Heavy rendered pages are deleted after extraction; snapshots retain only small text/metadata files. "
              "The s00 run remains archived as the pre-fallback diagnostic, not silently overwritten.", ""]
    return "\n".join(lines)


def build_bundle(report_path: Path, project: Path) -> dict:
    import matplotlib
    import scipy
    report = json.loads(report_path.read_text())
    docs = project / "docs"
    figure_dir = docs / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source = report_path.parent
    outputs = []
    names = {"report.json": "results", "observation_ledger.json": "observation_ledger",
             "claim_card.json": "claim_card", "contract.json": "contract", "manifest.json": "run_manifest",
             "prospective_Q4_lock.json": "prospective_Q4_lock"}
    for name, label in names.items():
        path = docs / f"{PREFIX}_{label}_20260911.json"
        shutil.copyfile(source / name, path)
        outputs.append(path)
    cache = docs / "r97_inputs"
    cache.mkdir(exist_ok=True)
    for p in sorted((source / "source_snapshots").iterdir()):
        if p.suffix in (".json", ".txt"):
            target = cache / p.name
            shutil.copyfile(p, target)
            outputs.append(target)
    for label, rows in (("revisions", report["revisions"]), ("availability_replays", report["availability_replays"])):
        target = docs / f"{PREFIX}_{label}_20260911.csv"
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        outputs.append(target)
    target = docs / f"{PREFIX}_report_vintages_20260911.md"
    target.write_text(markdown(report), encoding="utf-8")
    outputs.append(target)
    figures(report, figure_dir)
    outputs.extend(figure_dir / f"{PREFIX}_dashboard.{ext}" for ext in ("png", "svg", "pdf"))
    manifest = {"run_id": report["run_id"], "generated_at": report["generated_at"],
                "source_report_sha256": _digest(report_path), "report_builder_sha256": _digest(Path(__file__)),
                "bundle_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                       "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "files": {str(p.relative_to(project)): _digest(p) for p in outputs},
                "source_pdfs": {str(Path(r["source_path"]).relative_to(project)): r["source_sha256"] for r in report["catalog"]},
                "freeze_status": "future_outcome_unscored; public Git commit provides additional timestamp evidence"}
    _write(docs / f"{PREFIX}_bundle_manifest_20260911.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Q2_PDF.parents[5])
    args = parser.parse_args()
    manifest = build_bundle(args.report, args.project_root)
    print(json.dumps({"run_id": manifest["run_id"], "exported_files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
