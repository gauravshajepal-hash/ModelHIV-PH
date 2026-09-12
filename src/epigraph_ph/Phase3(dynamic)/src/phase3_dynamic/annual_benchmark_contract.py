"""Fail-closed training roles and explicit interpretation of annual benchmarks.

Legacy R75/R78/R93 artifacts remain immutable retrospective readouts. This
contract prevents their labels from becoming official-forecast superiority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .r75_bulk_unaids_annual_challenge import (
    _bulk_unaids_target_rows, _merge_external_targets_into_observations,
    _r69_annual_path, _rolling_annual_splits,
)
from .r78_public_annual_family_expansion import R69_DEFAULT_REPORT, _metric_train_arrays


def require_training_role(row: dict, metric: str) -> None:
    provenance = row.get("metric_provenance", {}).get(metric, row)
    role = provenance.get("observation_role")
    uses = provenance.get("allowed_use", [])
    uses = [uses] if isinstance(uses, str) else uses
    if not isinstance(uses, (list, tuple, set)):
        raise ValueError("Ambiguous allowed-use contract")
    if role not in {"direct_target", "auxiliary_likelihood"}:
        raise ValueError(f"{metric}: role {role!r} cannot enter a training likelihood")
    if not set(uses).intersection({"training_target", "auxiliary_likelihood", "train_origin_weak_measurement"}):
        raise ValueError(f"{metric}: explicit training use is absent")


def annual_claim_limit(claim: dict) -> dict:
    """Preserve legacy metrics/status but narrow the effective publication claim."""
    updated = dict(claim)
    updated["historical_claim_status"] = claim.get("claim_status")
    updated["claim_status"] = "diagnostic_only"
    updated["allowed_claim"] = "Retrospective agreement with model-estimated annual series, subject to training-role audit."
    updated["claim_limit"] = "Not a prospective forecast, calibrated predictive interval, or official AEM/Spectrum superiority result."
    updated["blockers"] = list(claim.get("blockers") or []) + [
        "annual_training_role_contract_requires_repair",
        "single_revised_estimate_vintage_is_not_a_historical_information_set",
        "target_uncertainty_containment_is_not_predictive_interval_coverage",
    ]
    return updated


def audit(annual_csv: Path) -> dict:
    targets = _bulk_unaids_target_rows(annual_csv, external_start_year=2010)
    panel = _merge_external_targets_into_observations([], targets)
    splits = _rolling_annual_splits(panel, start_year=2019, end_year=2024, min_train_years=5, horizons=(1,3,5))
    if not targets or not splits:
        raise ValueError("Missing locked annual benchmark support")
    metrics = sorted({r["metric_name"] for r in targets})
    rows = []
    for split in splits:
        train = [r for r in panel if int(r["quarter"][:4]) <= split["train_end_year"]]
        for metric in metrics:
            violations = []
            for row in train:
                if metric not in row:
                    continue
                try:
                    require_training_role(row, metric)
                except ValueError as error:
                    violations.append({"quarter": row["quarter"], "reason": str(error)})
            # Read the legacy design, but do not fit a new model using these rows.
            legacy_years, _ = _metric_train_arrays(train, metric)
            rows.append({**split, "metric": metric, "legacy_design_rows": len(legacy_years),
                         "strictly_inadmissible_rows": len(violations), "violations": violations,
                         "new_model_fit_performed": False})
    return {"schema": "annual_benchmark_integrity.v1", "audited_at": datetime.now(timezone.utc).isoformat(),
            "source_csv_sha256": hashlib.sha256(annual_csv.read_bytes()).hexdigest(),
            "source_filename": annual_csv.name, "source_vintages": sorted({r["metric_provenance"]["source_id"] for r in targets}),
            "target_row_count": len(targets), "split_count": len(splits), "metric_split_audits": rows,
            "violating_metric_split_count": sum(bool(r["violations"]) for r in rows),
            "fresh_fit_performed": False, "historical_models_changed": False,
            "effective_claim": "retrospective_estimate_agreement_only",
            "official_superiority": False, "predictive_interval_calibration_established": False,
            "required_repairs": ["explicit origin-scoped training/evaluation role separation without relabeling validation rows",
                                 "report-vintage and publication-availability lock for predictive claims",
                                 "predictive distributions evaluated against independent outcomes, not target interval containment",
                                 "actual official forecasts on a common outcome; locally fitted proxies remain internal baselines"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-csv", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a benchmark audit")
    csv = args.annual_csv or _r69_annual_path(json.loads(R69_DEFAULT_REPORT.read_text()))
    if csv is None:
        raise ValueError("Missing annual source")
    result = audit(csv)
    here = Path(__file__).parent
    result["code_hashes"] = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                             ("annual_benchmark_contract.py", "r75_bulk_unaids_annual_challenge.py", "r78_public_annual_family_expansion.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("effective_claim", "split_count", "violating_metric_split_count")}))


if __name__ == "__main__":
    main()
