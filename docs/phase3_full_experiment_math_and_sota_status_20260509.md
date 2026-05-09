# Phase 3 Full Experiment, Mathematics, And SOTA Status

Generated: 2026-05-09

This document summarizes the project state after the R90-R92 claim-grade and process-repair gates. It is intentionally conservative: it separates readout wins from mechanistic claims, and it separates internal benchmarks from official HIV-estimation systems.

## Executive Verdict

The project has produced a credible evidence-to-model pipeline and several real wins:

- National quarterly readout: `R41` is promoted under strict mapped carry-forward/R10 gates.
- Subnational readout: regional adapter/proxy models improve mean error, but split-stability is limited.
- Annual validation: `R86` and `R88` are scoped annual/readout wins versus carry-forward.
- Claim safety: `R90` clears R86/R88 as claim-grade annual/readout claims and blocks raw incidence/death mechanism claims.
- Mechanism expansion: `R91` confirms the mechanism block; diagnosis-flow and reported-death proxy bridges are not enough.
- Process repair: `R92` finds a stronger train-origin proxy signal, but still blocks mechanism claims because direct incidence support is absent and mortality source-family stability fails.

The project has not yet beaten official/SOTA HIV-estimation systems in the broad sense. It has not yet reproduced or defeated Spectrum/EPP/AEM/Naomi-like outputs under matched official input files. It also does not yet identify raw incidence or AIDS-death mechanisms from direct process evidence.

## Relationship To State-Of-The-Art HIV Models

Current official and near-official HIV estimation practice is dominated by:

| Model family | What it does | Where our project stands |
| --- | --- | --- |
| UNAIDS Spectrum/EPP | National HIV estimation using surveillance, surveys, ART programme data, demographic/natural-history assumptions, and uncertainty ranges. UNAIDS describes annual country-team use of Spectrum/EPP and direct use of case reporting/mortality data where appropriate: <https://www.unaids.org/en/dataanalysis/knowyourresponse/HIVdata_estimates> | We have not imported official Philippines Spectrum files, so we cannot claim head-to-head superiority. We can compare against public annual targets and internal baselines. |
| AIDS Epidemic Model (AEM) in Spectrum | Behaviour-driven incidence trend generation, especially relevant for concentrated epidemics/key populations. The Spectrum manual describes AEM as estimating incidence trends from sexual and needle-sharing behaviours: <https://avenirhealth.org/Download/Spectrum/Manuals/SpectrumManualE.pdf> | Our determinant/Phase 2 graph is conceptually aligned with this direction, but R81 says determinants are directional sensitivity only, not fitted quantitative effects. |
| CSAVR / case surveillance and vital registration | Uses case reporting and AIDS-related mortality where surveillance/survey data are weak but programme/vital data are stronger. | This is exactly the R89-R91 blocker. Reported deaths are present, but the bridge to annual AIDS-death estimates is not better/stable enough yet. |
| Naomi small-area estimation | Bayesian subnational model jointly estimating PLHIV, ART coverage, incidence, and new infections from multiple subnational sources. UNAIDS 2024 methods describe Naomi as combining multiple outcomes and data sources in a Bayesian small-area model: <https://www.unaids.org/sites/default/files/media_asset/2024-unaids-global-aids-update-annex2-methods_en.pdf> | Our subnational layer has mean-level regional readout wins, but not strict split-stable process claims. Province-level validation remains insufficient. |

Position: the repository is now publication-grade for a narrower claim: strict evidence-ledger modeling falsified the old R10 endpoint-only champion under expanded support, rebuilt national/regional readout champions, and produced scoped annual validation wins while explicitly blocking unsupported mechanism claims.

It is not yet publication-grade for: “we beat Spectrum/AEM overall,” “we identified incidence mechanisms,” “we can use Phase 2 knobs as causal interventions,” or “subnational province forecasts are validated.”

## Mathematical Core In Plain English

The Phase3(dynamic) model is a state-space cascade model. It tracks where people are in the HIV care pathway and forecasts transitions between states.

### State Vector

The model state at quarter `t` is:

```text
x_t = (U_t, D_t, A_t, T_t, V_t, L_t, R_t)
```

| Symbol | Plain-English meaning |
| --- | --- |
| `U_t` | people living with HIV but undiagnosed |
| `D_t` | diagnosed, not on ART |
| `A_t` | active ART without recent viral-load evidence |
| `T_t` | viral-load tested but not suppressed |
| `V_t` | virally suppressed |
| `L_t` | interrupted/lost from active ART pathway |
| `R_t` | recently re-engaged/restarted ART |

The effective susceptible denominator is:

```text
S_eff,t = max(N_t - sum_j x_{j,t}, 0)
```

where `N_t` is the observed/available population denominator.

### Transition Hazards

For each transition `r`, the model fits a bounded hazard:

```text
h_{r,t} = sigmoid(eta_{r,t})
eta_{r,t} = alpha_r + beta_r tau_t + rho_r eta_{r,t-1} + shock_{r,t}
```

Plain English: each transition rate has a baseline level, a time trend, memory from the previous quarter, and optional shock correction. The coefficients are fit only from training history at each blocked forecast origin.

The main transitions are:

```text
U -> D
D -> A
A -> T
T -> V
A -> L
T -> L
V -> L
L -> R
R -> A
```

### Flow Equations

If `f_{r,t}` is the number of people moving along transition `r`:

```text
f_{U->D,t} = h_{U->D,t} U_{t-1}
f_{D->A,t} = h_{D->A,t} D_{t-1}
f_{A->T,t} = h_{A->T,t} A_{t-1}
f_{T->V,t} = h_{T->V,t} T_{t-1}
f_{A->L,t} = h_{A->L,t} A_{t-1}
f_{T->L,t} = h_{T->L,t} T_{t-1}
f_{V->L,t} = h_{V->L,t} V_{t-1}
f_{L->R,t} = h_{L->R,t} L_{t-1}
f_{R->A,t} = h_{R->A,t} R_{t-1}
```

Plain English: each quarter, a fraction of people in each origin state moves to the next state.

### Incidence And Conservation

Incidence enters the undiagnosed state:

```text
I_t = min(cap_t, lambda_t S_eff,t)
log(1 + lambda_t) or log(1 + I_t) follows an AR-trend process
```

The cascade update is:

```text
U_t = U_{t-1} - f_{U->D,t} + I_t - exits_U,t
D_t = D_{t-1} + f_{U->D,t} - f_{D->A,t} - exits_D,t
A_t = A_{t-1} + f_{D->A,t} + f_{R->A,t} - f_{A->T,t} - f_{A->L,t} - exits_A,t
T_t = T_{t-1} + f_{A->T,t} - f_{T->V,t} - f_{T->L,t} - exits_T,t
V_t = V_{t-1} + f_{T->V,t} - f_{V->L,t} - exits_V,t
L_t = L_{t-1} + f_{A->L,t} + f_{T->L,t} + f_{V->L,t} - f_{L->R,t} - exits_L,t
R_t = R_{t-1} + f_{L->R,t} - f_{R->A,t} - exits_R,t
```

Plain English: people are conserved across states except for new infections entering `U` and removals/leakage leaving state compartments.

The exit channels are:

```text
mortality_removal
treatment_non_initiation
unresolved_external_removal
art_ltfu
reengagement
vl_testing_loss
```

### Observation Model

The model does not assume every observed metric is the same kind of truth. Rows are typed as:

```text
direct_target
auxiliary_likelihood
validation_only
prior_context
quarantined
```

Readouts are:

```text
diagnosed_plhiv_t      = D_t + A_t + T_t + V_t + L_t + R_t
alive_on_art_t         = A_t + T_t + V_t + R_t
new_diagnosed_t        = f_{U->D,t}
tested_for_viral_load  = T_t + V_t
virally_suppressed_t   = V_t
estimated_plhiv_t      = sum_j x_{j,t}
```

Plain English: the state model generates latent state counts; the observation model maps those states to reported programme metrics.

### Annual Validation

For annual public targets, normalized error is:

```text
e_{m,t} = | prediction_{m,t} - target_{m,t} | / scale_m
```

where `scale_m` is derived from train-origin annual target scale. Annual targets are `validation_only`; they are not allowed to become quarterly training truth.

R86 adds train-origin annual weak-measurement heads. R88 uses a guarded selector:

```text
if raw_process beats carry-forward inside train:
    use raw process
else:
    use carry-forward prior
```

Plain English: R88 refuses to overfit weak incidence/death channels. It only trusts a raw channel if the training history proves it beats carry-forward.

## Experiment Timeline And Verdicts

| Stage | Question | Verdict |
| --- | --- | --- |
| Phase 0 | Can raw HARP/HASP/HIV_Data, official reports, literature, WDI/WHO/PSA/FIES/PhilHealth-like support be extracted into structured evidence? | Yes. Evidence extraction and structured adapters exist. |
| Phase 1 | Can heterogeneous rows become typed tensors with roles, units, and quality metadata? | Yes. Normalization and observation-role carrying are implemented. |
| Phase 15 v2 | Can mixed-frequency national/region/province latent states be built? | Yes. This is the bridge into Phase 2. |
| Phase 2 | Can determinant surfaces and hidden modes be extracted? | Yes as structural payload; no as causal priors. R81 keeps knobs directional/sensitivity-only. |
| TR-V3/R10 | Can older endpoint families survive expanded HARP/HASP support? | No as mechanistic champions. R10 remains a frozen benchmark/readout family. |
| R11 | Can stock-consistency gates and multi-horizon readouts stabilize national forecasts? | Partly. Led to R11-28 and then R41. |
| R12 | Can lineage-specific annual anchors and DOH programme routes explain failures? | Yes diagnostically. It clarified mixed-lineage failures. |
| R13-R41 | Can a national champion beat carry-forward/R10 under strict mapped gates? | Yes. R41 promoted. |
| R42 | Does R41 survive hardening? | Yes for main routes, but source dependence exists when monthly DOH HARP support is ablated. |
| R43-R64 | Can regional/subnational models work with sparse HASP support? | Mean-level regional readout wins exist; strict split-stability remains limited. |
| R65-R70 | Is the full transmission model and external evidence base ready? | Diagnostic/queue-ready, not final transmission-model ready. |
| R71-R74 | Do external service signals improve forecasts? | Diagnostic-only. Not promoted. |
| R75 | Does annual public validation pass against carry-forward? | Yes: best model mean normalized error `0.2435` vs carry-forward `0.5925` in R42 annual gate; bulk annual challenge also passes. |
| R76-R79 | Can public annual proxy baselines be beaten? | Mixed/negative. R78 public proxy v2 is strong (`0.1638`) and blocks broad annual superiority. |
| R80 | Can public annual projections be emitted? | Yes as public annual projection head, not proof of official superiority. |
| R81 | Can Phase 2 determinants be used as scenario knobs? | Directional sensitivity only. No quantitative intervention effects. |
| R82-R85 | Can quarterly predictions emit complete annual incidence/death/PLHIV ledgers? | R82/R83 blocked; R84/R85 diagnostic; coverage repaired but process quality insufficient. |
| R86 | Does train-origin annual calibration make the complete ledger beat carry-forward? | Yes. Mean normalized error `0.2419` vs `0.3900`; interval coverage `0.7500` vs `0.4881`. |
| R87 | Can free raw-emission ratio/trend calibration fix incidence/deaths? | No. Diagnostic only. |
| R88 | Can guarded raw-or-carry annual selector improve without overfitting? | Yes. Mean normalized error `0.2976` vs `0.3900`; interval coverage `0.6429` vs `0.4881`. |
| R89 | Are raw incidence/death mechanisms directly supported? | No. Direct incidence support count `0`; death bridge worse than carry-forward. |
| R90 | Are R86/R88 safe publication claims? | Yes for annual/readout; mechanisms blocked. |
| R91 | Can proxy bridges rescue mechanism support? | No. Diagnosis-flow incidence proxy worse than carry-forward; mortality bridge ties and is source-family unstable. |
| R92 | Can process-repair family selection rescue mechanism support? | Signal only. Incidence/death proxy bridges beat carry-forward, but direct incidence support is absent and mortality source-family ablation is unstable. |

## Key Quantitative Status

| Claim | Candidate | Benchmark | Status |
| --- | ---: | ---: | --- |
| R41 national 1y all-mapped MAE | 0.0953 | carry-forward 0.1207; R10 0.1007 | promoted |
| R41 national 3y all-mapped MAE | 0.0917 | carry-forward 0.2049; R10 0.1171 | promoted |
| R41 national 5y all-mapped MAE | 0.0976 | carry-forward 0.2551; R10 0.1268 | promoted |
| R54 regional mean error | 0.0529 | carry-forward 0.0574 | mean promoted |
| R55 regional split stability | limited | strict nonregression required | not strict |
| R78 public annual proxy v2 | 0.1638 | R76 proxy 0.1956 | promoted public proxy |
| R86 annual ledger | 0.2419 | carry-forward 0.3900 | scoped annual win |
| R88 guarded annual ledger | 0.2976 | carry-forward 0.3900 | scoped conservative win |
| R89 death bridge | 0.5529 | carry-forward 0.4764 | blocked |
| R91 diagnosis-flow incidence proxy | 0.4859 | carry-forward 0.3116 | blocked |
| R91 reported-death AIDS-death bridge | 0.4764 | carry-forward 0.4764 | tie, source-unstable |
| R92 diagnosis-flow incidence process repair | 0.0497 | carry-forward 0.3116 | signal diagnostic, mechanism blocked |
| R92 reported-death mortality process repair | 0.1636 | carry-forward 0.4764 | signal diagnostic, source-unstable |

## What Is Defensible Now

Allowed:

- The project has a rigorous observation-ledger contract and blocked-time evaluation framework.
- R41 is the current national quarterly readout research champion under strict mapped gates.
- R52/R54/R58/R59 show regional readout/adapter signal, but most regional claims are mean-level or split-limited.
- R86 and R88 are claim-grade scoped annual/readout wins versus carry-forward.
- Phase 2 determinants can be used for scenario labels and sensitivity analyses only.
- R89-R92 correctly block raw incidence/death mechanism claims while showing that R92 contains a useful proxy process signal.

Not allowed:

- Broad claim that this beats Spectrum/EPP/AEM/Naomi overall.
- Mechanistic claim that raw quarterly incidence is identified.
- Mechanistic claim that AIDS deaths are identified from reported-death data.
- Claim that R92 proxy process repair is direct incidence or mortality identification.
- Causal claim that Phase 2 determinant knobs quantify intervention effects.
- Province-level process validation.
- Third-95 process claims beyond the observed/support-limited back-half readout evidence.

## Main Scientific Gap

The largest gap is not another optimizer. It is evidence and identifiability:

```text
direct incidence process support = 0
reported-death bridge source-family stability fails
subnational truth sparse
Phase 2 determinants source-stable only as sensitivity labels
official Spectrum/AEM output files absent
```

The next defensible research step is therefore R93:

1. Build an open AEM/Spectrum-style annual comparator from public data if official country files remain unavailable.
2. Keep R92 as a process-signal diagnostic until direct incidence support and mortality source-family stability improve.
3. Search/acquire stronger direct or incidence-adjacent public evidence for the Philippines.
4. Re-run R90-style claim gates after the mechanism-support evidence improves.
