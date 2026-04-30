# Phase 3 Evidence Agent Memo

Scope: inspect current Phase 3 code, then review primary literature on how latent temporal graph outputs such as sparse lagged edges, low-rank hidden-driver structure, target blankets, and uncertainty are used downstream in dynamic state-space or mechanistic inference models.

## Current Phase 3 code reality

The current Phase 3 code is already a mixed-frequency, hierarchical, mechanistic transition model, but it is not yet wired to the newer latent temporal graph outputs from Phase 2.

What it does now:
- The legacy Phase 3 path still loads `markov_blanket.json` and `core_feature_tensor.npz` from Phase 2 and builds a legacy intervention tensor from blanket nodes in `D:\EpiGraph_PH\src\epigraph_ph\phase3\pipeline.py`.
- The main rescue path builds modifier covariates and transition hook masks from candidate profiles, Phase 2 promoted/supporting factor sets, and Phase 15 mesoscopic factor sets in `D:\EpiGraph_PH\src\epigraph_ph\phase3\rescue_core.py`.
- Mixed-frequency observation bundles are already present in `D:\EpiGraph_PH\src\epigraph_ph\phase3\mixed_frequency.py` and fed into `rescue_core`.

What it does not yet do:
- It does not directly load `latent_temporal_graph_bundle.json` or `latent_temporal_phase3_target_blankets.json`.
- It does not yet treat sparse lagged edges as structural priors on transition equations.
- It does not yet represent low-rank hidden-driver structure as explicit shared latent innovations or regime processes inside the Phase 3 transition model.

That gap matters. The literature below supports using the new Phase 2 outputs as structured priors, latent innovations, reconciliation constraints, and uncertainty-weighted model-selection signals, not merely as another list of covariates.

## 1. Direct Evidence Table

| Source | Date | Field | Method | Exact relevance to Phase 3 |
|---|---|---:|---|---|
| [You & Yu, *Sparse plus low-rank identification for dynamical latent-variable graphical AR models*](https://doi.org/10.1016/j.automatica.2023.111405) | 2024 | mathematics / control / statistics | Sparse AR graph plus low-rank latent dynamical component, estimated jointly | Closest direct analogue to Phase 2 output. Supports treating sparse lagged edges as direct transition couplings and low-rank Phase 2 rows as separate hidden dynamic drivers in Phase 3, rather than collapsing both into the same modifier channel. |
| [Li, Zhou, Pitt, *Dynamic Mortality Forecasting via Mixed-Frequency State-Space Models*](https://arxiv.org/abs/2601.05702) | 2026-01-09 | statistics / econometrics | Mixed-frequency state-space model with annual observations linked to monthly latent factors by explicit aggregation | Directly supports replacing ad hoc year-end anchor use with a proper aggregation operator inside Phase 3. Very close to the repo’s need to combine monthly province latent states with annual national HIV anchors. |
| [Si & Chen, *LEVDA: Latent Ensemble Variational Data Assimilation via Differentiable Dynamics*](https://arxiv.org/abs/2602.19406) | 2026-02-23 | physics / data assimilation / ML | Latent 4DEnVar smoother with irregular spatiotemporal observation support and joint state/parameter assimilation | Directly relevant to Phase 3 because current observations and supports are irregular across geography and time. Supports using Phase 2 graph outputs as state/parameter priors that are assimilated jointly with observations rather than inserted as fixed regressors. |
| [Tong, Wang, Yan, *Latent Autoencoder Ensemble Kalman Filter for Data assimilation*](https://arxiv.org/abs/2603.06752) | 2026-03-06 | mathematics / statistics / physics / CS | Learned latent linear dynamics plus consistent latent observation map, with stable filtering in latent space | Supports making low-rank hidden drivers explicit latent processes in Phase 3 and keeping the observation mapping structurally consistent. Relevant to how Phase 3 should absorb hidden-driver structure from Phase 2 without losing interpretability. |
| [Alippi & Zambon, *Graph Kalman Filters*](https://arxiv.org/abs/2303.12021) | 2023-03-21 | computer science / graph ML | State-space filtering where inputs, latent states, and outputs live on evolving graphs | Direct support for letting Phase 2 sparse lagged edges define graph-structured coupling in the Phase 3 state equation or transition readout, instead of using only blanket membership plus hand-built hook masks. |
| [Friston et al., *Dynamic causal modelling of COVID-19 and its mitigations*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9298167/) | 2022 | biology / epidemiology | Multifactor Bayesian mechanistic model combining infection, behavior, testing, and policy factors with model evidence | Strong evidence that latent graph-derived candidates should enter Phase 3 as competing mechanistic hypotheses on transition channels, then be judged by model evidence and uncertainty, not asserted as fixed truths. |
| [Prashad, *State-space modelling for infectious disease surveillance data: Dynamic regression and covariance analysis*](https://www.sciencedirect.com/science/article/pii/S2468042724001313) | 2025 (DOI page says 2024/accepted into 2025 issue) | epidemiology / statistics | Component linear Gaussian state-space model with dynamic regression and time-varying covariance analysis | Directly relevant to feeding Phase 2 blanket features and graph-supported modifiers into Phase 3 as dynamic regressors with covariance-aware uncertainty instead of simple deterministic hooks. |
| [Mancarella & Gerosa, *Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy*](https://arxiv.org/abs/2502.12156) | 2025-02-17 | astronomy / Bayesian computation | Joint hierarchical posterior over local-level parameters and population hyperparameters using HMC | Supports a Phase 3 design where province states, regional effects, and shared hyperparameters are inferred jointly, instead of treating Phase 2/15 outputs as fixed preprocessors and only fitting Phase 3 conditionally. |

## 2. Contextual Evidence Table

| Source | Date | Field | Method | Why it matters contextually for Phase 3 |
|---|---|---:|---|---|
| [d’Antonio et al., *State Space Modelling for detecting and characterising gravitational waves afterglows*](https://doi.org/10.1016/j.ascom.2024.100860) | 2024 | astronomy | State-space modelling for sparse irregular transient time series | Useful analogy for Phase 3’s sparse subnational signals: it reinforces the value of latent temporal state models under irregular support, but it is less directly about mechanistic transition equations than the sources above. |
| [Sharma, Aguerri, Guimarans, *Hierarchical Forecast Reconciliation on Networks*](https://arxiv.org/abs/2505.03955) | 2025-05-06 | computer science / optimization | Network-flow reconciliation for coherent multilevel forecasts | Contextually strong for province-region-national coherence. Suggests Phase 3 should reconcile forecasts and latent summaries across geography using explicit network structure, not only by averaging or penalties. |
| [Vélez-Cruz & Laubichler, *A Generalized Framework for Multiscale State-Space Modeling with Nested Nonlinear Dynamics*](https://arxiv.org/abs/2410.19074) | 2024-10-24 | statistics / complex systems | Multiscale state-space model with nested fast/slow dynamics and switching regimes | Contextual support for putting Phase 2 sparse direct edges, Phase 2 low-rank hidden structure, and Phase 3 temporal scaffold into one multiscale regime-aware system rather than separate ad hoc components. |
| [Singh et al., *KODA: A Data-Driven Recursive Model for Time Series Forecasting and Data Assimilation using Koopman Operators*](https://arxiv.org/abs/2409.19518) | 2024-09-29 | computer science / dynamical systems | Recursive decomposition into stable global dynamics plus local residual dynamics | Contextually supports a clean split in Phase 3 between persistent mechanistic transition structure and local hidden-driver residual dynamics inferred from Phase 2 low-rank structure. |
| [Friston et al., *Dynamic causal modelling of COVID-19*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7431977/) | 2020 | epidemiology / computational biology | Technical DCM framework for latent-state epidemic modelling and Bayesian model comparison | Contextual because it provides the deeper modelling logic behind the later mitigations paper: hidden factors and interventions belong inside the generative model, with explicit uncertainty and model comparison. |
| [Srinivasan et al., *Simulation-based population inference of LISA's Galactic binaries: Bypassing the global fit*](https://arxiv.org/abs/2506.22543) | 2025-06-27 | astronomy / simulation-based inference | Infer population parameters directly from compressed time/frequency data, bypassing exhaustive object-level estimation | Contextual support for not requiring a brittle one-by-one deterministic interpretation of every Phase 2 edge before using them downstream. Useful for Phase 3 if model comparison is done at the summary/operator level rather than edge-by-edge. |

## 3. Concise Recommendations for Phase 3 Integration

1. **Wire the new Phase 2 outputs in directly, not indirectly.**
   Load `latent_temporal_graph_bundle.json` and `latent_temporal_phase3_target_blankets.json` as first-class Phase 3 inputs, alongside the current mixed-frequency bundle and determinant modifier metadata.

2. **Map sparse lagged edges to structured priors on transition channels, not raw covariates.**
   For each retained Phase 2 direct edge `source -> target`, create a prior on the corresponding Phase 3 transition coefficient block:
   - edge sign initializes coefficient sign prior
   - edge weight/stability initializes shrinkage scale
   - edge support across province/region/national scales determines whether the prior is national, regional, or province-level
   This should replace part of the current blanket-to-hook heuristic in `rescue_core`, not merely append another modifier column.

3. **Map low-rank hidden-driver structure to explicit latent innovation processes.**
   The low-rank Phase 2 rows should become one or more shared latent innovation factors in the Phase 3 transition logits:
   - national shock factor
   - region-shared residual factor
   - regime-switch indicator if hidden rank is concentrated and unstable across time
   This is the cleanest use of the low-rank object. It should not be interpreted as another direct determinant.

4. **Use target blankets as candidate-screening gates, not as causal truth.**
   The merged blanket should define which Phase 15 blocks / Phase 1 indicators are eligible to affect each transition family. Then use model evidence, posterior shrinkage, and predictive checks to decide what survives. This is exactly where the current DCM and state-space literature is strongest.

5. **Push uncertainty all the way into the transition model.**
   Phase 2 uncertainty should affect:
   - whether an edge is active at all
   - how strong its prior is
   - whether it is global or local
   - whether a hidden-driver factor is included
   Edge stability, hidden-rank uncertainty, and support counts should become hyperpriors or spike-and-slab inclusion probabilities, not just report metadata.

6. **Unify mixed-frequency observations with latent graph-informed dynamics.**
   Phase 3 already has a mixed-frequency bundle. The literature strongly supports pairing that with latent-state assimilation rather than static transition fitting. The natural next step is:
   - Phase 2 supplies structured transition priors and latent hidden-driver channels
   - Phase 3 assimilates annual/monthly observations through explicit aggregation operators
   - posterior uncertainty is evaluated jointly over states and transition parameters

7. **Keep direct edges and hidden drivers separate in downstream outputs.**
   Direct sparse edges should feed transition priors and mechanistic interpretations.
   Low-rank hidden-driver terms should feed shared latent innovation channels, stress tests, and regime diagnostics.
   They should not be merged into one “importance” list.

8. **Evaluation should test structural recovery, not only fit.**
   Minimum evaluation additions:
   - synthetic recovery where the true generative model has both sparse transition couplings and low-rank hidden shocks
   - ablation comparing: no Phase 2 graph, sparse-only, sparse+low-rank
   - mixed-frequency stress tests where annual anchors are available but monthly province data are sparse
   - forecast reconciliation checks across province, region, and national levels

## Bottom line

The literature is not pointing toward “more feature engineering.” It is pointing toward a cleaner architecture:

- **Sparse lagged edges** from Phase 2 should become **structured priors on Phase 3 transition equations**.
- **Low-rank hidden-driver structure** from Phase 2 should become **explicit shared latent innovation or regime processes** inside Phase 3.
- **Blankets** should be **screening and shrinkage structures**, not final causal claims.
- **Uncertainty** should control inclusion, pooling, and reconciliation, not just be logged after fitting.

That is the most evidence-supported path for making Phase 3 scientifically stronger without discarding the current mechanistic rescue core.
