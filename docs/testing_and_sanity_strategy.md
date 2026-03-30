# Testing And Sanity Strategy

This document defines the research-grade testing strategy for the `EpiGraph_PH` pipeline. It exists for two reasons:

1. to make the pipeline auditable enough for publication
2. to make failure visible before it contaminates scientific conclusions

The goal is not just "more tests." The goal is a pipeline that is explicitly defended against hallucination, silent fallback, overload, and mechanistic nonsense.

## Principles

The test suite is organized around four non-negotiable rules.

### 1. No hallucinated truth

The pipeline is not allowed to fabricate authority. Every phase must expose where its outputs came from and what kind of truth they rely on.

The truth ladder is:

- `anchor_truth`: official program and surveillance anchors
- `benchmark_truth`: validated country or international reference estimates
- `proxy_truth`: operational proxies with stated limitations
- `prior_truth`: mechanistic or mathematical constraints
- `synthetic_truth`: controlled fixtures used to test numerics
- `null_test`: permutation, source-dropout, and invariance checks

Tests therefore verify:

- provenance fields exist on phase outputs
- registries preserve source identity and tiering
- reference-check artifacts are present where required
- exploratory objects cannot silently become mechanistic-active without promotion

### 2. No silent fallback

Fallbacks are acceptable only when they are visible.

Tests therefore verify:

- backend manifests record which runtime was available and which one was used
- parser choice is surfaced in parsed-document artifacts
- blocked phases emit explicit blocked artifacts instead of silently skipping work
- rescue profiles do not silently degrade into legacy behavior
- JAX-based rescue runs declare requested vs resolved inference family

If a component falls back from a preferred runtime, parser, or inference path, the artifact contract must expose it.

### 3. No overload by accident

The pipeline must be able to run on this machine without unbounded memory growth or runaway intermediate expansion.

Tests therefore verify:

- smoke runs stay bounded in document counts and working-set size
- heavy parse selection is capped
- factor counts are bounded in Phase 1.5
- promotion budgets are bounded in Phase 2
- rescue profiles keep Phase 4 blocked until earlier gates pass

The test suite uses session-scoped smoke fixtures so it does not rebuild full runs for every assertion. That is both a speed optimization and an overload-control mechanism.

### 4. No mechanistic nonsense

A model that compiles is not automatically scientifically acceptable.

Tests therefore verify:

- state tensors are finite and nonnegative
- CD4 overlays satisfy simplex constraints
- hierarchy reconciliation artifacts exist
- observation ladder classes are explicit
- transition names and hook masks stay within the allowed topology
- blocked readiness states remain blocked when core gates fail

## Phase-by-phase test responsibilities

### Registry

Registry tests verify:

- source registries preserve source tier, provenance notes, and extraction status
- subparameter registries preserve source-bank membership and verifiable locators
- registry outputs are consistent with upstream phase artifacts

### Phase 0

Phase 0 tests focus on anti-hallucination and overload control.

They verify:

- harvest manifests exist and record backend availability
- parsed documents expose parser choice
- canonical candidate rows preserve source identity, geo, and time
- aligned tensors and quality weights are finite
- embedding indices are finite and dimensioned correctly
- smoke-run document volume stays bounded

### Phase 1

Phase 1 tests focus on alignment truth.

They verify:

- normalized rows include bias and promotion fields
- tensor shapes match axis catalogs
- denominator and standardized tensors are finite
- subnational geography is not collapsed into a single national label
- helper functions for reliability and unit handling behave deterministically

### Phase 1.5

Phase 1.5 tests focus on mesoscopic factor legitimacy.

They verify:

- factor catalogs and tensors align
- source reliability summaries are internally consistent
- network feature families are present
- stability scores stay in valid ranges
- factor counts remain bounded to avoid overload
- network-derived feature families are emitted without requiring a full PDE engine

### Phase 2

Phase 2 tests focus on promotion discipline.

They verify:

- legacy masks and adjacency outputs are finite
- tournament outputs obey promotion budgets
- only allowed transition hooks are used
- exploratory factors are not promoted directly to main predictive status
- stability reports and diagnostics are emitted

### Phase 3

Phase 3 tests are the strongest in the suite because this is the scientific core.

They verify:

- fit and validation artifacts exist for all supported profiles
- rescue observation ladders are explicit
- official reference checks are present
- state tensors are finite and mass-bounded
- CD4 overlays remain simplex-valid
- subgroup/metapopulation operators are surfaced for `hiv_rescue_v2`
- no silent fallback occurs between requested and resolved inference family

### Phase 4

Phase 4 tests are intentionally conservative.

They verify:

- legacy outputs remain contract-valid
- rescue profiles stay blocked unless Phase 3 gates clear
- blocked artifacts explain why optimization is disabled

## What the suite does not claim

The suite proves contract integrity and numerical sanity. It does not, by itself, prove scientific correctness.

Scientific correctness additionally requires:

- external reference comparison
- assumption audits
- definition audits
- repeated runs under perturbation
- explicit benchmark ladders against incumbent approaches

That is why the test suite is paired with the ground-truth contract and the official reference-check artifacts.

## Research publication use

For publication, the intended claim is:

"Every module emits auditable artifacts with explicit truth classes, visible fallback behavior, overload guards, and mechanistic sanity checks."

That claim is materially stronger than:

"we ran some unit tests."

## Recommended next documentation

The next two documents that should accompany this file in a research package are:

- a benchmark protocol describing exact comparison targets and acceptance thresholds
- an assumptions register documenting every operational definition that can move the cascade materially
