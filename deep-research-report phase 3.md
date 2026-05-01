# Executive Summary  
Phase 3 builds a **Hierarchical Semi-Markov State-Space Model** for the HIV cascade. We define explicit states (Undiagnosed, Diagnosed, On ART, Suppressed, Lost), include **time-in-state** (duration Δt) to capture non-exponential dwell, and use province/region-level priors. Transition probabilities are logit/softmax functions of ARD‑filtered features \(C_{i,t}\), Δt, interventions \(I_{iu,t}\), and hierarchical baselines \(\theta_i\). We outline a NumPyro/JAX implementation: vectorized plates for provinces and time, NUTS sampling with 8GB GPU limits, and parallelized rollout for counterfactuals under Model Predictive Control. Detailed validation (PPCs, held-out metrics) and pitfalls (identifiability, data sparsity) are discussed. 

**Assumptions:** Exact province list unspecified. We have monthly cascade counts by province. All tensors are float32.  


# 1. State-Space Definition  
- **States \(S\):** Susceptible (implicitly), then U (Undiagnosed HIV+), D (Diagnosed, not on ART), A (On ART), V (Virally suppressed), L (Lost). Transitions allowed: U→D→A→V, with L absorbing from D/A/V (loss to care).  
- **Time-in-state \(\Delta t\):** Integer months spent in current state. We track \(\Delta t_{i,t}^a\) for province \(i\) in state \(a\).  
- **Dwell distributions:** Implicitly modeled by transition hazards depending on Δt (semi-Markov). For example, probability of U→D grows with Δt, not constant. We **don’t pre-specify** a parametric dwell; instead we include Δt as a feature in the transition model.  

**Tensor shapes:**  
- State tensor \(S_{i,t,a}\): shape \([N_{\text{prov}}, T, A]\) (A = number of states).  
- Time-in-state \(\Delta_{i,t}\): \([N,T]\).  
- Features \(C_{i,t,k}\): \([N,T,K]\) after ARD filtering.  
- Interventions \(I_{i,t,u}\): \([N,T,U]\).  

All as float32 on GPU (or float64 if needed for stability).  

# 2. Hierarchical Priors & ARD  
- **Hierarchical θ:** For each transition \(a\to b\) we have baseline log-odds \(\beta_{ab}^0\). These can vary by province: \(\beta_{i,ab}^0 \sim N(\beta_{r[i],ab}^0, \sigma^2_{r})\), region means \(\beta_{r,ab}^0 \sim N(\beta_{\text{nat},ab}^0, \sigma^2_{\text{nat}})\). This encodes province/region effects.  
- **ARD/Horseshoe:** For each feature \(k\) and transition \(ab\), weight \(w_{ab,k} \sim N(0,\tau^2\lambda_k^2)\) with \(\lambda_k\sim C^+(0,1)\), \(\tau\sim C^+(0,1)\). Horseshoe prior shrinks irrelevant \(w\)→0. (In NumPyro, use `sample(..., HalfCauchy)`).  
- **Latent confounders:** Unobserved confounders can be partially handled by random province intercepts \(\beta_{i,ab}^0\). For more, one could add GP latent terms in JAX/NumPyro, but that’s complex.  

# 3. Transition Model Formulation  
For province \(i\) at time \(t\), let state \(S_t=a\). The logit of transitioning \(a\to b\) is:  
\[
\eta_{i,t}^{a\to b} = \beta_{i,ab}^0 + \sum_k w_{ab,k} C_{i,t,k} + \gamma_{ab}\,\Delta_{i,t}^a + \sum_u \psi_{ab,u} I_{i,t,u}.
\]  
Then probabilities (including "stay" option \(a\to a\)) via softmax:  
\[
P_{i,t}(a\to b) = \frac{\exp(\eta_{i,t}^{a\to b})}{\sum_{c} \exp(\eta_{i,t}^{a\to c})}.
\]  
Regularization: we fix an identifiability by omitting one transition's intercept or constrain \(\sum_c P(a\to c)=1\). We penalize large weights with priors (above) and L2 terms if needed.  

# 4. Bayesian Inference (NumPyro/JAX)  
- **Likelihood:** Use multinomial counts or categorical for transitions. If we have state counts \(N_{i,t,a}\), we can sample transitions \(N_{i,t}(a\to \cdot)\) using `numpyro.sample("N", Multinomial(...))`. Alternatively, model state fractions with Dirichlet noise.  
- **Priors:** As above. Use `plate("prov", N)` and `plate("time", T)` for vectorization. JIT-compile the model (JAX).  
- **Sampling:** NUTS (HMC) via `numpyro.infer.MCMC`. With 8GB GPU, target single-precision or small chains. We can also use `numpyro.infer.VI` (ADVI or SVI) for speed. Always set `rng_key` for reproducibility.  
- **Vectorized Sampling:** Model code uses `plate`s so NumPyro automatically parallelizes provinces. For rollout/simulation, we’ll use JAX `vmap` over posterior samples.  
- **Compute considerations:** Each MCMC iteration does forward/backprop on the whole time series. Limit T or subsample if needed. JAX XLA fuses operations, but memory stays ~O(N×T). Monitor GPU usage.  

# 5. Simulation & Counterfactuals  
- **Trajectory simulation:** Given posterior draws of parameters, simulate \(S_{i,t}\) forward. At each \(t\), sample \(S_{i,t+1}\sim \text{Categorical}(S_{i,t}P_{i,t})\). In practice, use `jax.lax.scan` for time, `vmap` over i and samples.  
- **Uncertainty:** Propagate posterior uncertainty by simulating multiple trajectories per posterior draw. Compute credible bands for cascade outputs.  
- **Causal gradients:** For an intervention \(do(X=x)\) on feature \(u\), recompute \(P_{i,t}\) with \(I_{u}=x\) and resimulate. Finite differences on aggregate outcomes (e.g. suppressed fraction) gives a do-calculus gradient.  
- **Pareto/MPC:** Sample many random intervention policies \(\vec a_t\) (e.g. via Dirichlet). Roll out 15-year horizon, score metrics (suppression vs equity). Discard dominated policies to get Pareto frontier. For MPC, apply first 1–2 steps and re-fit model at each decision epoch.  

# 6. Validation & Diagnostics  
- **Posterior Predictive:** Generate simulated cascade data from the model and compare to observed (e.g. new diagnoses per month, ART numbers). Check that posterior predictive means/intervals cover true data.  
- **Calibration & Coverage:** Use rank histogram or PICP (Prediction Interval Coverage) to ensure credible intervals are well-calibrated.  
- **ARD Convergence:** Check if many \(w_{ab,k}\) have posterior concentrated near 0 (good) or not. If not, consider increasing shrinkage.  
- **Sensitivity Analyses:** Vary priors and hyperparams, re-run, check output stability.  
- **Model comparison:**  
  - Compare predicted cascade rates against WHO/DOH reported (HARP) values【7†L139-L146】.  
  - Compute RMSE/AUC for held-out months/provinces.  
  - Compare to simple baselines (see §7).  

# 7. Baselines & Metrics  
- **Baselines:** ARIMA per province on cascade metrics; simple compartmental model calibrated to first-year data; persistence (repeat last known state).  
- **Metrics:**  
  - *RMSE* on cascade targets (diagnosed%, suppressed%).  
  - *ROC AUC* for events (e.g. whether new diagnosis occurs).  
  - *Log-likelihood* of held-out data.  
  - *Gain over baseline* (e.g. % improvement).  
- **Validation schemes:**  
  - *Time holdout:* train on initial months, test on later months.  
  - *Spatial holdout:* train on most provinces, test on withheld ones.  

# 8. Tensors & Shapes  

| Tensor               | Shape               | Dtype  | Description                  |
|----------------------|---------------------|--------|------------------------------|
| `C` (features)       | [N, T, K]           | float32| ARD-filtered features        |
| `I` (interventions)  | [N, T, U]           | float32| Intervention intensities     |
| `S` (state counts)   | [N, T, A]           | int64  | Observed # in each state     |
| `Δ` (time-in-state)  | [N, T]              | int32  | Months in current state      |
| `W` (adj matrix)     | [J, J]              | float32| Phase2 adjacency mask        |

*J, K, U, A unspecified.*

# 9. Implementation Notes  
- **Batched Inference:** Use `plate('prov', N)` for provinces, `plate('time', T)` for time in NumPyro model.  
- **JIT & PRNG:** Compile model with `@jit`; manage JAX PRNG keys for reproducibility.  
- **Mixed PyTorch/JAX:** Preprocess (Phase1/2) in PyTorch, then use DLPack to move `Z` into JAX for Phase3 if needed. Keep transitions in JAX once in inference.  
- **Memory:** For N≈100, T≈60, A≈5, J≈50, K≈20, GPU memory ~1-2GB for state & param tensors. NUTS may double this. Offload to CPU if VRAM is tight.  
- **Checkpointing:** Save intermediate traces using NumPyro’s `MCMC.run` checkpointer.  

# 10. Example Pseudocode (NumPyro/JAX)  
```python
def model(C, I, Δ, S=None):
    # Global priors
    β_nat = numpyro.sample('β_nat', Normal(0,5).expand([A,A]))
    σ_reg = numpyro.sample('σ_reg', HalfCauchy(2.5))
    τ = numpyro.sample('τ', HalfCauchy(1.))
    λ = numpyro.sample('λ', HalfCauchy(1.), sample_shape=[K])

    # Region-level
    with numpyro.plate('regions', R):
        β_reg = numpyro.sample('β_reg', Normal(β_nat, σ_reg))
    # Province-level
    with numpyro.plate('provs', N):
        β_i = numpyro.sample('β_i', Normal(β_reg[region_idx], 0.5))

    # Feature weights (shared across provinces)
    w = numpyro.sample('w', Normal(0, τ*λ).expand([A,A,K]))

    # Likelihood
    for i in range(N):
      for t in range(T-1):
        # compute transition logits
        η = β_i[i]   # shape [A,A]
        for k in range(K):
            η += w[:,:,k] * C[i,t,k]
        # add Δ and interventions similarly ...
        # softmax to get P
        P = jax.nn.softmax(η, axis=1)
        # sample next-state or transitions
        if S is not None:
          numpyro.sample(f"S_{i}_{t+1}", 
                          Categorical(probs=P[S[i,t]]), obs=S[i,t+1])
```
  
```python
# Vectorized rollout in JAX
def rollout_step(S, t, params):
    η = compute_eta(params, C[:,t], I[:,t], Δ[:,t])  # [N,A,A]
    P = jax.nn.softmax(η, axis=2)
    S_next = jnp.einsum('ia,iaj->ij', S, P)
    return S_next, S_next

@jax.jit
def simulate(params, S0):
    S_traj = []
    S = S0
    for t in range(T):
        S, S = rollout_step(S, t, params)
        S_traj.append(S)
    return jnp.stack(S_traj)
```

# 11. Hyperparameters & Tuning  
- NUTS chains=4, warmup=1000, samples=1000.  
- Horseshoe scale τ=0.5.  
- Softmax logits scale (implicit).  
- Learning rate 1e-3 for ADVI/Adam.  
- Step for CV: vary τ, λ for ARD shrinkage; β weight.  

# 12. Pitfalls & Remedies  
- **Non-identifiability:** Similar transitions and features cause collinearity. Fix by strong priors or merging states.  
- **Sparse data:** Few transitions in small provinces. Mitigate by pooling (hierarchy) and using Dirichlet smoothing.  
- **Model misspecification:** Cascade may not be truly Markov. Consider including hidden variables or time trends.  
- **Convergence:** NUTS may struggle; use diagnostic R-hat<1.1. Try ADVI if needed.  
- **GPU limits:** NUTS on 8GB might OOM; consider minibatching or chaining smaller models.  

# 13. Output Comparison  
We compare model forecasts to:  
- **HARP/DOH metrics:** Diagnosed% (target 95), ART%, suppression% by province and time【7†L139-L146】.  
- **Held-out data:** Next-year actual values.  
- **Baselines:** AR(1) predictions, SEIR-like differential eqns.  

# 14. Diagrams & Checklist  

```mermaid
flowchart TD
    A[Standardized Features C_{i,t}] --> B[Hierarchical Semi-Markov Model]
    B --> C[Simulate S_{i,t} over T]
    C --> D[Policy Gradients and MPC Rollouts]
    D --> E[Optimal Intervention \vec{a}]
```

**Deployment Checklist:** (0) Preprocessed \(\mathbf{Z}\) ready, (1) NumPyro model code implemented, (2) Posterior sampling tested on synthetic data, (3) Simulation/Vmap validated, (4) Model outputs match observed cascade on holdout.  

