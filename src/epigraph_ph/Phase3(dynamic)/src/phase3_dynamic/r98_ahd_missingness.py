"""R98: audit AHD ascertainment, test conditional forecasts, and expose nonidentification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata

import numpy as np

from .ahd_observation import AhdStatus, classification_deviance, mar_deviance, naive_binary_deviance, profile_deviance
from .data import sandbox_repo_root
from .r96_monthly_diagnosis_state import Q2_PDF, _digest, _write
from .r97_hasp_report_vintages import _timestamp


RUN_ID = "p3d-r98-ahd-missingness-20260911-s00"
FAMILIES = ("last_status_mix", "pooled_status_mix", "recent_coverage_pooled_known_mix")
CONTRACT = {
    "schema": "r98.ahd_missingness.v1", "scope": "national reported AHD classification; not incidence or diagnosis-delay identification",
    "source": "R97 locked report catalog; first-page quarterly HASP footnotes",
    "partitions": ["known_advanced", "known_nonadvanced", "unknown_immunologic_clinical_status"],
    "unknown_policy": "Unknown is not nonadvanced. Accept only explicit Of these containment and exact count reconciliation.",
    "definition_policy": "Preserve each HASP report's AHD definition; do not relabel to current WHO thresholds or pediatric definitions.",
    "families": FAMILIES,
    "controls": "naive binary likelihood explicitly collapses unknown into nonadvanced, diagnostic only",
    "likelihood": "(A,E,M)|N ~ Multinomial(N, (p*q_A,(1-p)*q_E,1-p*q_A-(1-p)*q_E))",
    "bounds": "[A/N,(A+M)/N]; sharp finite-cohort bounds assuming known labels correct, not confidence intervals",
    "profile": "profile separate q_A,q_E for each evidence quarter; no pseudo-counts or handmade hazard weights",
    "evaluation": "expanding event-time blocks AND declared-mirror-availability blocks; same three-category targets for every family",
    "score": "mean per-case multinomial deviance, equally weighted quarters; conditional on target N, not an unconditional count forecast",
    "selection": "no champion selection from this small diagnostic experiment",
    "role_gate": "only accepted auxiliary_likelihood classification rows; annual incidence excluded",
    "freeze": "R41 and R97 Q4 forecasts unchanged; no prior run overwritten",
    "blockers": ["arbitrary status-dependent missingness prevents point identification", "only six recent reconciled quarterly status partitions",
                 "first-release dates/historical PDF bytes unverified", "AHD is not infection duration without a validated CD4/progression observation model"],
}


def _capture(text: str, pattern: str) -> tuple[int, str]:
    matches = list(re.finditer(pattern, text, re.I))
    if len(matches) != 1:
        raise ValueError(f"Expected one unambiguous evidence span, found {len(matches)}")
    match = matches[0]
    return int(match.group(1).replace(",", "")), match.group(0)


def extract_status(text: str) -> dict:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    n, n_span = _capture(text, r"there were\s+([\d,]+)\s+confirmed HIV-positive individuals")
    a, a_span = _capture(text, r"([\d,]+)\s+\(\d+%\)\s+had an advanced HIV (?:infection|disease)")
    contained = re.search(r"([\d,]+)\s+cases had non-advanced HIV infection\.\s+Of these,\s+([\d,]+)\s+\(\d+%\)\s+had no data on immunologic/clinical criteria", text, re.I)
    if not contained:
        raise ValueError("Unknown containment ambiguous or missing; do not infer it from a nominal nonadvanced label")
    nominal, missing = (int(contained.group(i).replace(",", "")) for i in (1, 2))
    if a + nominal != n or missing > nominal:
        raise ValueError("AHD, nominal nonadvanced and missing-status counts do not reconcile")
    status = AhdStatus(a, nominal - missing, missing)
    definition = re.search(r"AHD definition is based on.{0,170}?cells/mm(?:3|\^3)", text, re.I)
    return {"total_diagnoses": n, "advanced": a, "known_nonadvanced": nominal - missing,
            "unknown": missing, "nominal_nonadvanced": nominal,
            "classified_fraction": status.classified_fraction, "complete_case_fraction": status.complete_case_fraction,
            "bounds": status.bounds, "source_spans": [n_span, a_span, contained.group(0)],
            "definition_span": definition.group(0) if definition else None,
            "definition_scope": "as_reported_HASP; not automatically harmonized to WHO 2025"}


def fit_status_forecast(rows: list[dict], family: str) -> list[float]:
    if family not in FAMILIES or not rows:
        raise ValueError("Unknown family or empty training evidence")
    ordered = sorted(rows, key=lambda r: r["period"])
    statuses = [AhdStatus.from_ledger(r) for r in ordered]
    counts = np.asarray([s.counts for s in statuses], dtype=float)
    if family == "last_status_mix":
        result = counts[-1] / counts[-1].sum()
    elif family == "pooled_status_mix":
        result = counts.sum(axis=0) / counts.sum()
    else:
        a, e, _ = counts.sum(axis=0)
        if a + e == 0:
            raise ValueError("No classified cases; cannot estimate a known-status mix")
        p = a / (a + e)
        q = statuses[-1].classified_fraction
        result = np.array([q * p, q * (1-p), 1-q])
    return result.tolist()


def eligible_training(rows: list[dict], target: dict, *, release_aware: bool) -> list[dict]:
    origin = f"{target['period'][:4]}-{3 * (int(target['period'][-1]) - 1) + 1:02d}-01"
    eligible = []
    for row in rows:
        if row["period"] >= target["period"]:
            continue
        if release_aware and (not row.get("mirror_posted_at") or _timestamp(row["mirror_posted_at"]) > _timestamp(origin)):
            continue
        AhdStatus.from_ledger(row)
        eligible.append(row)
    return sorted(eligible, key=lambda r: r["period"])


def score_blocks(rows: list[dict], *, release_aware: bool) -> tuple[list, list]:
    scores, blocked = [], []
    for target in sorted(rows, key=lambda r: r["period"]):
        train = eligible_training(rows, target, release_aware=release_aware)
        if not train:
            blocked.append({"period": target["period"], "reason": "no eligible earlier classification partition"})
            continue
        observed = AhdStatus.from_ledger(target)
        for family in FAMILIES:
            probs = fit_status_forecast(train, family)
            loss = classification_deviance(observed, probs) / observed.total
            scores.append({"period": target["period"], "family": family, "training_periods": [r["period"] for r in train],
                           "evaluation": "mirror_availability" if release_aware else "event_time_retrospective",
                           "predicted_probabilities": probs, "observed_counts": observed.counts,
                           "per_case_deviance": loss if math.isfinite(loss) else None,
                           "score_status": "finite" if math.isfinite(loss) else "impossible_positive_count_under_zero_probability",
                           "target_row_hash": target["row_hash"], "train_row_hashes": [r["row_hash"] for r in train]})
    return scores, blocked


def summarize(scores: list[dict]) -> list[dict]:
    result = []
    for family in FAMILIES:
        group = [r for r in scores if r["family"] == family]
        valid = [r["per_case_deviance"] for r in group if r["score_status"] == "finite"]
        result.append({"family": family, "blocked_quarters": len(group), "finite_scores": len(valid),
                       "mean_per_case_deviance": float(np.mean(valid)) if valid and len(valid) == len(group) else None})
    return result


def run(output: Path, catalog_path: Path) -> dict:
    if output.exists():
        raise FileExistsError("R98 output must be a fresh directory")
    output.mkdir(parents=True)
    _write(output / "contract.json", CONTRACT)
    project = Q2_PDF.parents[5]
    frozen = project / "docs/phase3_r97_prospective_Q4_lock_20260911.json"
    frozen_hash = _digest(frozen)
    input_report = json.loads(catalog_path.read_text())
    rows, quarantine = [], []
    for source in input_report["catalog"]:
        parts = Path(source["source_path"]).parts
        relative = Path(*parts[parts.index("HIV_Data") + 1:])
        pdf = Q2_PDF.parent.parent / relative
        if _digest(pdf) != source["source_sha256"]:
            raise ValueError("Locked source PDF digest changed")
        text = subprocess.check_output(["pdftotext", "-f", "1", "-l", "1", "-layout", str(pdf), "-"], text=True)
        common = {"period": source["period"], "source_path": str(pdf.relative_to(project)),
                  "source_id": source["source_sha256"], "source_sha256": source["source_sha256"], "source_page": 1,
                  "mirror_posted_at": source.get("mirror_posted_at"), "first_release_verified": False}
        try:
            row = {**common, **extract_status(text), "status": "accepted", "observation_role": "auxiliary_likelihood",
                   "allowed_use": ["classification_likelihood", "partial_identification", "conditional_forecast_evaluation"],
                   "measurement_semantics": "joint_classification_count", "geography": "national",
                   "population": "newly_reported_diagnoses_all_ages", "unit": "people", "source_tier": "official_surveillance_report",
                   "extraction_method": "first_page_numeric_spans_containment_and_sum_checked",
                   "time_granularity": "quarter", "support_partition": "explicit_unknown_status_partition",
                   "leakage_status": "event_cutoff_and_optional_mirror_release_filter_required",
                   "metric_id": "ahd_classification_partition", "time_start": source["period"], "time_end": source["period"]}
            row["value"] = {key: row[key] for key in ("advanced", "known_nonadvanced", "unknown")}
            row["row_hash"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            rows.append(row)
        except ValueError as error:
            quarantine.append({**common, "status": "quarantined", "observation_role": "quarantined", "allowed_use": [], "reason": str(error)})
    if not rows:
        raise ValueError("No reconciled AHD partitions")
    event, event_skips = score_blocks(rows, release_aware=False)
    mirror, mirror_skips = score_blocks(rows, release_aware=True)
    statuses = [AhdStatus.from_ledger(r) for r in rows]
    intersection = [max(s.bounds[0] for s in statuses), min(s.bounds[1] for s in statuses)]
    # Exact bounds determine the identified set; this mesh is for plotting only.
    profile_rows = []
    for p in np.linspace(0, 1, 401):
        values = {name: sum(fun(s, float(p)) for s in statuses) for name, fun in
                  (("profile", profile_deviance), ("mar", mar_deviance), ("naive_binary", naive_binary_deviance))}
        profile_rows.append({"p": float(p), **{k: v if math.isfinite(v) else None for k, v in values.items()}})
    # Explicit input lock: this pass cannot change an existing prospective forecast.
    if _digest(frozen) != frozen_hash:
        raise RuntimeError("R97 prospective lock changed during the experiment")
    report = {"run_id": output.name, "generated_at": datetime.now(timezone.utc).isoformat(), "contract": CONTRACT,
              "accepted_partitions": rows, "quarantined_partitions": quarantine,
              "event_time_scores": event, "mirror_time_scores": mirror,
              "event_time_skips": event_skips, "mirror_time_skips": mirror_skips,
              "event_time_summary": summarize(event), "mirror_time_summary": summarize(mirror),
              "common_p_profile_identified_set": intersection if intersection[0] <= intersection[1] else None,
              "profile_rows": profile_rows,
              "frozen_R97_sha256_before": frozen_hash, "frozen_R97_sha256_after": _digest(frozen),
              "gate": {"status": "observation_semantics_corrected_mechanism_identification_blocked", "champion": None,
                       "R41_changed": False, "R97_Q4_changed": False, "incidence_training_rows": 0,
                       "backlog_point_estimate": None, "backlog_point_estimate_reason": "not identifiable from classification alone",
                       "blocked_time_diagnosis_forecast_improvement": None,
                       "claim_limit": "Classification likelihood diagnostic only; no epidemic, intervention, regional or AEM superiority claim"}}
    _write(output / "report.json", report)
    _write(output / "observation_ledger.json", rows + quarantine)
    _write(output / "claim_card.json", report["gate"])
    _write(output / "manifest.json", {"source_pdf_hashes": {r["source_path"]: r["source_sha256"] for r in rows + quarantine},
                                      "input_catalog_sha256": _digest(catalog_path), "R97_lock_sha256": frozen_hash,
                                      "code_hashes": {p.name: _digest(p) for p in (Path(__file__), Path(__file__).with_name("ahd_observation.py"))},
                                      "outputs": {p.name: _digest(p) for p in output.iterdir() if p.is_file()}})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=sandbox_repo_root() / "artifacts/runs" / RUN_ID)
    parser.add_argument("--catalog", type=Path, default=Q2_PDF.parents[5] / "docs/phase3_r97_results_20260911.json")
    args = parser.parse_args()
    report = run(args.output_dir, args.catalog)
    print(json.dumps({k: report[k] for k in ("gate", "event_time_summary", "mirror_time_summary", "common_p_profile_identified_set")}, indent=2))


if __name__ == "__main__":
    main()
