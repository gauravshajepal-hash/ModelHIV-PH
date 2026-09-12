"""Paper figure and portable evidence bundle for the R98 missing-status test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from .r96_monthly_diagnosis_state import Q2_PDF, _digest, _write
from .r98_ahd_missingness import FAMILIES
from .ahd_observation import AhdStatus, mar_deviance, naive_binary_deviance


NAMES = {"last_status_mix": "Last status mix", "pooled_status_mix": "Pooled status mix",
         "recent_coverage_pooled_known_mix": "Recent coverage + pooled known mix"}


def plot(report: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    rows = report["accepted_partitions"]
    x = np.arange(len(rows))
    blue, rust, grey = "#27758B", "#B96835", "#CED4D8"
    ax = axes[0, 0]
    bottom = np.zeros(len(rows))
    for key, label, color in (("advanced", "Known AHD", rust), ("known_nonadvanced", "Known non-AHD", blue), ("unknown", "Unknown status", grey)):
        heights = np.array([r[key] / r["total_diagnoses"] * 100 for r in rows])
        ax.bar(x, heights, bottom=bottom, label=label, color=color, width=.7)
        if key == "unknown":
            for i, height in enumerate(heights):
                ax.text(i, bottom[i]+height/2, f"{height:.0f}%", ha="center", va="center", fontsize=8)
        bottom += heights
    ax.set_xticks(x, [r["period"] for r in rows], rotation=25)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of reported diagnoses (%)")
    ax.set_title("A  Missing status varies sharply between reports", loc="left", weight="bold", pad=28)
    ax.legend(ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)

    ax = axes[0, 1]
    for i, row in enumerate(rows):
        lo, hi = np.array(row["bounds"]) * 100
        ax.plot([lo, hi], [i, i], color=grey, lw=9, solid_capstyle="butt")
        ax.scatter(lo, i, color=rust, s=30, label="Reported AHD / all diagnoses" if i == 0 else None, zorder=3)
        ax.scatter(row["complete_case_fraction"] * 100, i, color=blue, marker="x", s=45,
                   label="AHD / classified only (MAR assumption)" if i == 0 else None, zorder=3)
    ax.set_yticks(x, [r["period"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("AHD fraction among reported diagnoses (%)")
    ax.set_title("B  Known counts imply bounds, not a true fraction", loc="left", weight="bold", pad=28)
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1), frameon=False)
    ax.text(0, -.23, "Grey bars: finite-cohort identification bounds, NOT confidence intervals", transform=ax.transAxes, fontsize=8)

    ax = axes[1, 0]
    statuses = [AhdStatus.from_ledger(row) for row in rows]
    naive_p = sum(s.advanced for s in statuses) / sum(s.total for s in statuses)
    mar_p = sum(s.advanced for s in statuses) / sum(s.advanced+s.known_nonadvanced for s in statuses)
    offsets = {"naive_binary": sum(naive_binary_deviance(s, naive_p) for s in statuses),
               "mar": sum(mar_deviance(s, mar_p) for s in statuses), "profile": 0}
    for key, label, color in (("naive_binary", "Unknown treated as non-AHD (invalid control)", rust),
                              ("mar", "Equal classification probability (assumption)", blue),
                              ("profile", "Separate classification probabilities (profile)", "#343D42")):
        ax.plot([r["p"] * 100 for r in report["profile_rows"]],
                [None if r[key] is None else r[key] - offsets[key] for r in report["profile_rows"]], color=color, label=label)
    lo, hi = np.array(report["common_p_profile_identified_set"]) * 100
    ax.axvspan(lo, hi, color="#E3EBBD", alpha=.8)
    ax.annotate(f"Flat profile: {lo:.1f}-{hi:.1f}%", xy=((lo+hi)/2, 0), xytext=(40, 1200),
                arrowprops={"arrowstyle": "->", "color": "#343D42"}, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 5000)
    ax.set_xlabel("Hypothesized constant AHD fraction across six quarters (%)")
    ax.set_ylabel("Deviance increase from each curve's own optimum")
    ax.set_title("C  Changing ascertainment can mimic disease shifts", loc="left", weight="bold", pad=24)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.text(0, 1.01, "All-six-quarter identifiability diagnostic, NOT a blocked forecast fit", transform=ax.transAxes, fontsize=8)

    ax = axes[1, 1]
    for i, (key, label, color) in enumerate((("event_time_summary", "Event-time blocks (n=5)", blue),
                                           ("mirror_time_summary", "Mirror-availability blocks (n=2)", rust))):
        result = {r["family"]: r for r in report[key]}
        bars = ax.bar(np.arange(3) + (i-.5)*.36, [result[f]["mean_per_case_deviance"] for f in FAMILIES],
                      width=.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(range(3), ["Last mix", "Pooled mix", "Recent coverage\n+ pooled known mix"])
    ax.set_ylim(0, .8)
    ax.set_ylabel("Mean per-case classification deviance (lower is better)")
    ax.set_title("D  The apparent winner depends on evaluation", loc="left", weight="bold", pad=24)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("R98 | Unknown clinical status is not early HIV disease", x=.08, ha="left", fontsize=20, weight="bold")
    fig.text(.08, .92, "6 reconciled HASP quarters | 4 older partitions quarantined | no epidemic-model promotion", fontsize=12)
    fig.subplots_adjust(top=.82, bottom=.14, left=.08, right=.95, hspace=.67, wspace=.35)
    fig.text(.08, .025, "Source: first-page DOH HASP counts and missing-status footnotes; R97 SHIP metadata. R98, 11 September 2026.\n"
             "Known labels assumed correct. AHD definitions remain as reported; AHD does not identify infection duration.\n"
             "Panel D predicts classification probabilities conditional on diagnoses, not incidence or total diagnoses. No calibrated interval or causal claim.", fontsize=9, color="#555555")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"phase3_r98_dashboard.{ext}", dpi=260, facecolor="white")
    plt.close(fig)


def markdown(report: dict) -> str:
    rows = report["accepted_partitions"]
    lines = ["# R98: AHD ascertainment and diagnosis-delay evidence", "",
             f"Run `{report['run_id']}`. **Observation-layer correction implemented and tested; no new epidemic champion.**", "",
             "![R98 scientific dashboard](figures/phase3_r98_dashboard.png)", "",
             "## What the audit found", "",
             "The historical backlog implementation equates a reported AHD count with the latent late-diagnosis count, even without a measured classification denominator. "
             "It also ranks median CD4 values and averages those ranks with reported AHD proportions. A CD4 rank is not a disease probability. "
             "The affected historical code is `diagnosis_incidence_repair.py`, `_late_diagnosis_emission_observations` and `_backlog_emission_losses`. "
             "Its diagnosis-share/count targets are also correlated, and its equal-component loss is not a joint count likelihood.", "",
             "R98 does not silently rewrite frozen experiments or replace the R41 reference. It provides a separate typed observation module, an exact profile likelihood, "
             "and a tested adapter to the conserved monthly backlog simulator. The old proxy-rank path is **not** accepted as evidence of identified late-diagnosis mechanisms. "
             "The new adapter is candidate-only, not yet a promoted full-model fit. No monthly incidence truth or biochemical CD4 trajectory was invented.", "",
             "## Extracted evidence", "",
             "The six 2025-Q1 to 2026-Q2 reports explicitly say that some nominally non-advanced cases have missing criteria. "
             "Known non-AHD is therefore nominal non-advanced minus that missing subgroup. "
             "Exact reconciliation with the same report's diagnosis denominator is required. "
             "Four 2024 footnotes use ambiguous containment or inconsistent totals and stay quarantined; no count is repaired by guesswork.", "",
             "| Report | Diagnoses | Known AHD | Known non-AHD | Unknown | As-reported AHD fraction | Classified-only fraction | Identification bounds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['period']} | {r['total_diagnoses']:,} | {r['advanced']:,} | {r['known_nonadvanced']:,} | {r['unknown']:,} | {r['bounds'][0]:.1%} | {r['complete_case_fraction']:.1%} | {r['bounds'][0]:.1%}-{r['bounds'][1]:.1%} |")
    lines += ["", "For Q2-2026, 27.7% is the fraction *documented* advanced; 65.6% is the fraction advanced *among classified cases*. "
              "Neither is automatically the AHD fraction among all reported diagnoses. The sharp finite-cohort range is 27.7%-85.5% if known labels are correct and missing labels unrestricted. "
              "This is not a confidence interval, not a bound on national HIV prevalence, and not a measured undiagnosed backlog. "
              "[DOH Q2-2026, page 1](https://www.ship.ph/wp-content/uploads/2026/08/2026_Q2-HIV-AIDS-Surveillance-of-the-Philippines-2.pdf).", "",
              "HASP definitions are preserved as reported, not retrospectively changed to newer thresholds. "
              "The 2025 WHO guideline also has age-specific conditions and a CD4 threshold differing from some report text; harmonization needs individual/stratified evidence. "
              "[WHO definition](https://www.ncbi.nlm.nih.gov/books/NBK620073/).", "",
              "## Mathematics in plain English", "",
              r"Write $A$ for known AHD cases, $E$ for known non-AHD, $M$ for missing status, and $N=A+E+M$. "
              r"Let $p$ be the AHD fraction among reported diagnoses and $q_A,q_E$ the probabilities that AHD and non-AHD status are classified. Then", "",
              r"$$ (A,E,M)\mid N \sim \operatorname{Multinomial}\left[N;\ p q_A,\ (1-p)q_E,\ 1-pq_A-(1-p)q_E\right]. $$", "",
              "In English: observed advanced cases depend on both disease and classification. Missing cases may come from either disease group. "
              "The two classification probabilities are nuisance parameters, not biological hazards or intervention effects.", "",
              r"$$ \frac{A}{N}\le p\le\frac{A+M}{N}. $$", "",
              "The lower end assigns no missing cases to AHD; the upper end assigns all missing cases to AHD. "
              "For any fraction inside these limits, some pair of classification probabilities exactly reproduces the observed status proportions. "
              "The likelihood therefore has a flat region, not a unique optimum. "
              "Assuming equal classification probabilities yields the complete-case estimate A/(A+E), but the aggregate data do not establish that assumption. "
              "Partial identification under missing-not-at-random data is an established alternative to unsupported point estimates: "
              "[Jiang and Ding, HIV missingness study](https://arxiv.org/abs/1610.01198).", "",
              r"We profile $q_A,q_E$ separately for each quarter. With $a=A/N,e=E/N,m=M/N$, fitted observed probabilities at a fixed $p$ are", "",
              r"$$\widehat\pi(p)=\begin{cases}(p,(1-p)e/(e+m),(1-p)m/(e+m)),&p<a,\\(a,e,m),&a\le p\le1-e,\\(pa/(a+m),1-p,pm/(a+m)),&p>1-e.\end{cases}$$", "",
              r"The score is $2\sum_j n_j\log[(n_j/N)/\widehat\pi_j]$, with zero-count terms equal to zero. "
              "Impossible positive counts under zero forecast probability remain impossible; they are not hidden by arbitrary pseudocounts. JSON represents that score as null with an explicit impossible-outcome status.", "",
              "Under a descriptive constant-p hypothesis across all six quarters, the flat profile spans "
              f"{report['common_p_profile_identified_set'][0]:.1%}-{report['common_p_profile_identified_set'][1]:.1%}. "
              "This shows that changes in status-specific ascertainment can reproduce these aggregate observations without uniquely identifying a change in disease severity. "
              "It does not prove that severity is constant. The all-quarter profile is an identifiability diagnostic, not a training fit used for held-out predictions.", "",
              "Panel C centers each curve at its own analytic optimum. The invalid binary control and three-category models do not score the same observations; "
              "this panel compares identifiability shapes, not relative model quality. Only Panel D uses a common predictive scoring target.", "",
              "The backlog adapter aggregates the three simulated monthly late-diagnosis counts and divides by total simulated diagnoses before applying this likelihood once per quarter. "
              "It does not duplicate quarterly AHD rows across months or score AHD counts and their derived fraction twice. "
              "Equating the late-state emission with AHD still needs a validated progression/observation model. The adapter alone does not establish that mapping.", "",
              "## Blocked conditional forecast results", "",
              "Three families estimate the probability of each recorded status, not total diagnoses: last observed status mix; pooled count proportions; "
              "and the latest classified fraction combined with the training-pooled mix among classified cases. "
              "All probabilities use only eligible training rows. Event-time testing has five target quarters; declared-mirror-availability testing has two. "
              "Original release dates remain unverified, so neither view is promoted as prospective validation.", "",
              "| Family | Event-time mean deviance (5 blocks) | Mirror-availability mean deviance (2 blocks) | Decision |",
              "|---|---:|---:|---|"]
    for f in FAMILIES:
        e = next(r for r in report["event_time_summary"] if r["family"] == f)
        m = next(r for r in report["mirror_time_summary"] if r["family"] == f)
        lines.append(f"| {NAMES[f]} | {e['mean_per_case_deviance']:.6f} | {m['mean_per_case_deviance']:.6f} | Diagnostic only |")
    lines += ["", "Smaller is better. The pooled mix improves on last-mix carry under the two mirror-availability blocks, "
              "but regresses under event-time testing. This is not evidence that the epidemic model beats carry-forward, R10, or AEM. "
              "The outcome denominator N is conditioned on for this classification score; it is not used as a training input or claimed as a forecast.", "",
              "## Keep, reject, and next step", "",
              "- Keep the reconciled three-category evidence ledger, exact profiled likelihood, and strict quarantine/type gates.",
              "- Reject converting unknown status to early disease, interpreting CD4 ranks as probabilities, and promoting a unique backlog from these counts alone.",
              "- Keep R41 and the dated R97 Q4 forecast lock unchanged. No full historical backbone refit or diagnosis-flow improvement is claimed by R98.",
              "- Next: test a separately observable status-classification process, then carry its uncertainty into the diagnosis-delay branch. "
              "Collect testing/completion denominators or individual/stratified CD4 evidence before interpreting that branch as incidence identification.", "",
              "Cross-domain transfer 1: inverse-problem response matrices. The two unknown classification probabilities play the role of unknown detection efficiencies; "
              "a flat profile exposes a non-unique inverse. This mapping fails if the known labels themselves are wrong, requiring an additional misclassification model. "
              "Cross-domain transfer 2: set-membership state estimation. The measurement constrains a feasible set rather than forcing a point state. "
              "Test whether future independently classified observations shrink the feasible set while retaining count conservation.", "",
              "## Reproduction and artifacts", "", "```bash",
              "export PYTHONPATH='src/epigraph_ph/Phase3(dynamic)/src'",
              "python3 -m phase3_dynamic.r98_ahd_missingness --output-dir /path/to/new-run",
              "python3 -m phase3_dynamic.r98_report --report /path/to/new-run/report.json",
              "```", "",
              "[Ledger](phase3_r98_observation_ledger_20260911.json) | [Full results](phase3_r98_results_20260911.json) | "
              "[Bundle checksums](phase3_r98_bundle_manifest_20260911.json)", ""]
    return "\n".join(lines)


def build(report_path: Path) -> None:
    import matplotlib
    import scipy
    project = Q2_PDF.parents[5]
    docs = project / "docs"
    report = json.loads(report_path.read_text())
    paths = []
    for name, label in (("report", "results"), ("observation_ledger", "observation_ledger"),
                        ("claim_card", "claim_card"), ("manifest", "run_manifest"), ("contract", "contract")):
        p = docs / f"phase3_r98_{label}_20260911.json"
        shutil.copyfile(report_path.with_name(name + ".json"), p)
        paths.append(p)
    table = docs / "phase3_r98_scores_20260911.csv"
    rows = report["event_time_scores"] + report["mirror_time_scores"]
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    paths.append(table)
    p = docs / "phase3_r98_ahd_missingness_20260911.md"
    p.write_text(markdown(report), encoding="utf-8")
    paths.append(p)
    plot(report, docs / "figures")
    paths.extend(docs / "figures" / f"phase3_r98_dashboard.{ext}" for ext in ("png", "svg", "pdf"))
    _write(docs / "phase3_r98_bundle_manifest_20260911.json", {
        "run_id": report["run_id"], "files": {str(p.relative_to(project)): _digest(p) for p in paths},
        "builder_sha256": _digest(Path(__file__)), "source_report_sha256": _digest(report_path),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "frozen_R97_sha256": report["frozen_R97_sha256_after"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    build(args.report)


if __name__ == "__main__":
    main()
