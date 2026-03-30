# ModelHIV-PH

ModelHIV-PH is a probabilistic HIV evidence and semi-Markov cascade modeling platform for the Philippines.

The repository centers on a staged pipeline in `src/epigraph_ph`:

- `phase0`: uncertain evidence retrieval, parsing, literature alignment, and source-quality handling
- `phase1`: probabilistic measurement normalization and tensor preparation
- `phase15`: mesoscopic factor construction and network-derived feature systems
- `phase2`: constrained causal structure discovery and promotion admission
- `phase3`: hierarchical semi-Markov HIV cascade inference with HARP-aware calibration
- `phase4`: stochastic policy/control experiments with a node-graph runtime assurance layer

## Design Principles

- Probabilistic, calibrated, and interpretable modules
- Semi-Markov structure reserved for latent cascade dynamics where dwell time matters
- Explicit distinction between priors, numerical stabilizers, and policy or constraint settings
- Ground-truth and audit artifacts emitted across phases

## Repository Layout

- `src/epigraph_ph`: package source
- `tests`: pytest coverage and contract tests
- `docs`: design notes, audits, and standards
- `scripts`: helper entry points

## Quick Start

```powershell
python -m pip install -e .[core]
python -m pytest tests -q
python -m epigraph_ph.cli.main --help
```

## Current Scope

The current implementation focuses on the Philippines HIV modeling path with:

- HARP archive ingestion and frozen-history backtesting
- plugin-based coefficient contracts
- region-aware node-graph constraints
- optional OCR sidecar wiring for complex PDF extraction

The codebase keeps the package import path `epigraph_ph` for continuity while the repo-facing project identity is `ModelHIV-PH`.
