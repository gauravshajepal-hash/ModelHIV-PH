# Phase 0/1/15/2 Hardening Plan

Date: 2026-04-03  
Repo: `D:\EpiGraph_PH`  
Selected autoresearch variant: `evidence-to-model-loop`

## Executive Position

The memos converge on one judgment:

- `Phase 0` and `Phase 1` are operationally strongest.
- `Phase 15` and `Phase 2` are mathematically much better than before, but still scientifically provisional.
- The next work should prioritize trust, identifiability, and falsification rather than more latent or graph complexity.

The practical translation is:

1. repair stage summaries so manifests cannot disagree with emitted artifacts,
2. expose the difference between `artifact complete` and `scientifically trusted`,
3. tighten subnational informativeness reporting so thin national indicators are not confused with province-informative evidence,
4. turn existing benchmark and audit machinery into an actual run-level gate,
5. then use those gates to drive the next model mutations.

## Direct Evidence From The Memos

1. `Phase 0` structured extraction parity is already strong, but human-labeled semantic truth is still missing for free-text and OCR-heavy extraction.
2. `Phase 1` observability is useful, but support counts are still easier to obtain than real subnational informativeness.
3. `Phase 15` is the mathematical center of the stack, but partial identification and pooling saturation remain real.
4. `Phase 2` should be treated as a temporal hypothesis layer, not a mechanistic truth layer.
5. Evaluation is currently strongest on contracts and determinism, and weaker on semantic truth, calibration, and perturbation sensitivity.

## Concrete Objectives

### Objective 1
Make manifests auditable against real emitted artifacts.

Why:
- the current memos correctly identify stage-complete metadata drift as a trust failure.

Success condition:
- each phase manifest reports counts recomputed from actual artifacts, not stale intermediary payloads.

### Objective 2
Add explicit cross-phase trust reporting.

Why:
- users and downstream phases need to know whether a stage is merely complete or scientifically provisional.

Success condition:
- every phase manifest contains `artifact_status`, `trust_status`, and a compact `trust_summary`.

### Objective 3
Separate broad province-graph eligibility from stricter subnational informativeness.

Why:
- some indicators are nationally useful but not province-informative.

Success condition:
- Phase 1 emits `eligible_for_subnational_inference` and quantitative support scores.

### Objective 4
Turn existing semantic benchmark work into a live audit component.

Why:
- the repo already has a semantic benchmark module, but it is not part of the trust gate.

Success condition:
- the benchmark is generated or refreshed during trust audit and contributes to Phase 0 status.

### Objective 5
Keep the repo honest about Phase 15 and Phase 2 interpretation.

Why:
- current outputs can look more structural than the evidence supports.

Success condition:
- the trust audit explicitly labels Phase 15 as latent summary under partial identification and Phase 2 as temporal hypothesis only unless stronger gates exist.

## Mutation Units

1. `phase0_manifest_repair`
2. `phase1_subnational_informativeness`
3. `cross_phase_trust_audit`
4. `semantic_benchmark_gate`
5. `manifest_status_propagation`

## Evaluation Harness

### Hard gates

1. Structured extraction parity remains passing.
2. HARP adjudication remains passing.
3. Manifest counts match actual emitted artifacts.
4. Phase 1 observability audit remains present and finite.
5. Phase 15 and Phase 2 trust audit artifacts are emitted.

### Scientific posture gates

1. If no human-labeled semantic benchmark exists, Phase 0 remains scientifically provisional.
2. If regional pooling saturates or loading floor/ceiling behavior dominates, Phase 15 remains partially identified.
3. Phase 2 remains `temporal_hypothesis_only` unless uncertainty propagation and stronger falsification gates are added.

## Keep-Or-Revert Rule

Keep a hardening change only if:

- artifact counts become more faithful,
- no existing extraction or contract audit regresses,
- trust reporting becomes more explicit rather than more flattering,
- and no model stage silently upgrades its scientific claim level.

## Implemented In This Slice

1. Artifact-derived Phase 0 manifest counts.
2. Phase 1 `eligible_for_subnational_inference`, `subnational_support_score`, and `aggregate_support_score`.
3. A new cross-phase trust audit that updates Phase 0, 1, 15, and 2 manifests with:
   - `artifact_status`
   - `trust_status`
   - `trust_summary`
4. Semantic benchmark integration into the trust audit.
5. Phase 2 emission of run-level trust audit artifacts.

## Next Iterations

1. Add a real human-labeled Phase 0 semantic benchmark file and scoring harness.
2. Add Phase 15 uncertainty calibration and weaker-pooling sensitivity fits.
3. Add Phase 2 perturbation and falsification tests using Phase 15 state perturbations.
4. Introduce trust-region labels for province-month latent outputs.
5. Move from heuristic noise weights toward learned observation-noise calibration in Phase 1.
