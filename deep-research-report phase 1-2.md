# Executive Summary  
We define a **Province×Month×Feature** tensor \(\mathbf{X}_{i,t,j}\) (dtype float32) and transform it through Phase 1 into a standardized \(\mathbf{Z}_{i,t,j}\). Phase 2 then learns a sparse causal DAG on \(\mathbf{Z}\). We provide precise shapes/dtypes, PyTorch-based pseudocode for all transforms and optimization, and rigorous notes on pitfalls and evaluation. All recommendations are implementable on 8GB GPUs or multi-core CPUs.

**Assumptions:** Province list is not specified; let \(N\) provinces, \(T\) months, and \(J\) raw features. Denominators (population, subgroup sizes) are available. All code is tensorized; unspecified values (e.g. exact hyperparameters) are noted.

# 1. Tensor Schema & I/O  
- **Raw tensor**: \(\mathbf{X}\in\mathbb{R}^{N\times T\times J}\), float32. Each entry \(X_{i,t,j}\) is a raw input (count/rate).  
- **Denominator tensor**: \(\mathbf{D}\in\mathbb{R}^{N\times T\times K}\) for any required denominators (K ≤ J). E.g. population, PLHIV count.  
- **Outputs**: After Phase 1, \(\mathbf{Z}\in\mathbb{R}^{N\times T\times J}\) (float32) standardized. Phase 2 yields adjacency \(W\in\mathbb{R}^{J\times J}\).  

All data is held as GPU tensors (`torch.Tensor`). Missing values are indicated by a mask \(\mathbf{M}\in\{0,1\}^{N\times T\times J}\).

# 2. Phase 1: Preprocessing  
- **Density Conversion:** For each count feature \(X_{i,t,j}\), divide by appropriate denominator \(D_{i,t}\). In code:  
  ```python
  X[:,:,j] = X[:,:,j] / D[:,:,k]    # broadcast as needed
  ```  
- **Log/Box–Cox:** Apply a log1p or Box-Cox: e.g. `Y = torch.log1p(X)`. This stabilizes variance.  
- **Winsorization:** Compute low/high quantiles (e.g. 1% and 99%) using `Y.quantile()` on GPU:  
  ```python
  q1, q99 = Y.quantile(0.01, dim=(0,1)), Y.quantile(0.99, dim=(0,1))
  Y = torch.clip(Y, q1, q99)
  ```  
- **Robust Scaling:** Compute per-feature median and IQR (using 25th and 75th percentiles).  
  ```python
  med = Y.median(dim=(0,1)).values  # shape [J]
  IQR = Y.quantile(0.75, dim=(0,1)) - Y.quantile(0.25, dim=(0,1))
  Z = (Y - med) / (IQR + 1e-6)      # scaled tensor
  ```  
  Alternatively, apply a Huber scaler: \(Z_{ij} = \text{clip}( (Y_{ij}-\tilde\mu_j)/\tilde\sigma_j, -c, c)\) with \(c=1.5\).  
- **Missing-Value Handling:** Before scaling, fill NaNs with feature medians. After scaling, you can leave them as 0. Keep a mask \(\mathbf{M}\).  
- **Data-Quality Weighting:** Compute a province weight \(w_i = 1/(1+\beta m_i + \gamma v_i)\) where \(m_i\) = missing-rate in province \(i\), \(v_i\) = variance instability (e.g. year-to-year variance). Also apply temporal decay \(d_t = \exp(-\rho (T-t))\). Finally:  
  ```python
  Z *= w_i[:,None,None] * d_t[None,:,None]
  ```  
- **Batching & GPU:** Compute per-feature stats (median, IQR) once on GPU. If \(J\) is large, batch features (e.g. 50 at a time). Use PyTorch for all ops. CPU fallback: use cuDF (NVIDIA) or Polars (multi-threaded) to perform equivalent steps, then convert to `torch.Tensor`.  

# 3. Phase 2: Causal Discovery  
- **Pre-filter (MI/CMI):** Optionally drop features with near-zero mutual information with cascade outcomes to reduce noise. Compute MI on GPU via kNN or binning (skipped in pseudocode).  
- **PC Skeleton:** Compute pairwise (conditional) independence via Fisher’s Z-test. On GPU, calculate correlation matrix of \(\mathbf{Z}\) (reshape to [N*T,J]). Use significance threshold (e.g. p<0.01). Only test conditioning on 1 or 2 variables to limit complexity. Remove edges \(j\text{--}k\) if independent.  
- **Tier & Lag Masks:** Define epidemiological tiers for features; create mask \(M_{jk}=0\) if tier(j)>tier(k). Also enforce temporal lags: e.g. a “behavioral” feature cannot cause a “past” outcome. Combine into final mask \(M\in\{0,1\}^{J\times J}\).  
- **NOTEARS Optimization:** Parameterize adjacency \(W_{jk}\) (PyTorch `nn.Parameter[J×J]`). Loss:  
  \[
  \mathcal{L} = \|Z - ZW\|_F^2 + \lambda_1\|W\|_1 + \lambda_2\|W\|_F^2 + \mu\,h(W)^2,
  \]  
  where \(h(W)=\mathrm{tr}\exp(W\circ W)-J\) enforces acyclicity【2†L24-L30】. Use `torch.linalg.matrix_exp` for \(\exp\). Apply mask: `W.data *= M` after each update to zero forbidden edges. Use LBFGS or Adam with an augmented Lagrangian (penalty \(\mu\)) to enforce \(h=0\).  
- **Regularization:** Set \(\lambda_1\) (e.g. 1e-2) for sparsity, \(\lambda_2\) (1e-3) for ridge. Tune via cross-validation.  
- **Optimization Loop (pseudocode):**  
  ```python
  W = torch.zeros(J,J, requires_grad=True, device=device)
  optimizer = torch.optim.LBFGS([W], lr=0.1)
  def closure():
      optimizer.zero_grad()
      recon = Z @ W
      loss = ((Z - recon)**2).mean() + lam1*W.abs().sum() + lam2*(W**2).sum()
      h = torch.trace(torch.linalg.matrix_exp(W*W)) - J
      loss = loss + 100.0*h*h
      loss.backward()
      return loss
  optimizer.step(closure)
  W.data *= M  # apply mask
  ```  
- **Markov Blanket:** After W converges, extract parents/children of target cascade nodes. The Markov blanket = parents ∪ children ∪ co-parents in the graph. This yields the core feature set.  

# 4. Device Placement & Interop  
Keep data on GPU throughout. Use PyTorch (`torch.Tensor.to('cuda')`) for all Phase 1 transforms and gradient steps. For any JAX use (Phase 3), transfer via DLPack:  
```python
import torch; import numpy as np; 
torch_tensor = Z.to('cuda')
dlpack = torch.utils.dlpack.to_dlpack(torch_tensor)
jax_array = jax.dlpack.from_dlpack(dlpack)
```  
Minimize transfers: do all pre-scaling, MI, and NOTEARS in PyTorch; only move the final DAG or small matrices to JAX if needed. Memory: A 8GB GPU can hold ~10M floats (~40MB), so N*T*J up to ~2e7. W (J×J) for J=500 is 1e6 entries (~4MB). LBFGS stores a few copies of W (×5).  

# 5. Validation, Diagnostics & Pitfalls  
- **Small-N:** With few provinces, statistical noise is high. Mitigate by MI filter and strong priors.  
- **Collinearity:** Drop or combine highly correlated features (via PCA) before PC.  
- **Nonstationarity:** If relationships change, the learned DAG may not generalize. Use time-stratified CV.  
- **False Edges:** Tier mask prevents implausible links; permutation tests on edges help.  
- **Thresholds:** Independence test p-value and NOTEARS \(\lambda\) critically affect sparsity. Sensitivity sweep recommended.  
- **Missing Data:** Heavy missingness can bias MI/PC. Use data-quality weights and multiple imputation if needed.  

# 6. Testing & Evaluation  
- **Baselines:** Compare to (a) AR(1) model per province, (b) simple compartmental (e.g. SEIR-like) fit, (c) “no change” (last-month state carries forward).  
- **Metrics:**  
  - *Transition AUC:* For binary events (e.g. new diagnosis occurs) across province-months.  
  - *RMSE:* On predicted state counts (diagnosed, ART) vs held-out.  
  - *Calibration:* % of actual values falling within credible intervals.  
- **Cross-Validation:** Time-series CV (train on initial T-12 months, test next 6), and province-holdout (train on N-1 provinces, test on one).  
- **Targets:** Compare predicted cascade (95-95-95 metrics) to DOH/HARP reported values (quarterly totals)【7†L139-L146】. Evaluate error reduction over baselines.  

# 7. Pseudocode (Phase 1 & 2)  

```python
# Phase 1 Example (PyTorch)
X = raw.to(device)             # [N,T,J] float32
D = denom.to(device)           # denominators [N,T]
X = X / D[...,None]            # density
Y = torch.log1p(X)             # log-transform
q1,q99 = Y.quantile(0.01,dim=(0,1)), Y.quantile(0.99,dim=(0,1))
Y = torch.clamp(Y, q1, q99)    # winsorize
med = Y.median(dim=(0,1)).values; IQR = Y.quantile(0.75,dim=(0,1))-Y.quantile(0.25,dim=(0,1))
Z = (Y - med) / (IQR + 1e-6)    # robust scale
# handle missing
mask = torch.isnan(X)
Z[mask] = 0

# Phase 2 Example
# Compute correlation (for PC) on GPU:
Z_flat = Z.reshape(-1,J)             # [N*T, J]
corr = torch.corrcoef(Z_flat.T)      # [J,J]
# Tier/lag mask M prepared separately:
W = torch.zeros(J,J,requires_grad=True,device=device)
opt = torch.optim.LBFGS([W], lr=0.05)
for _ in range(50):
    def closure():
        opt.zero_grad()
        recon = Z_flat @ W
        loss = ((Z_flat - recon)**2).mean() + lam1*W.abs().sum()
        h = torch.trace(torch.linalg.matrix_exp(W*W)) - J
        loss = loss + 50*h*h
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        W.mul_(M)  # apply mask each iter
```

# 8. Hyperparameters & Tuning  
- PC p-value ~0.01; cond. max=2.  
- NOTEARS \(\lambda_1=0.01,\lambda_2=0.01\).  
- LBFGS max iter ~100, lr=0.1; or Adam lr=1e-3 with 1000 steps.  
- Winsor trim=(1%,99%); Huber δ≈1.  
- Missing decay ρ=0.1, weight β=γ=1.  
Tune via validation set.

# 9. Pipeline Flowchart & Checklist  

```mermaid
flowchart TB
    A[Raw X,Denom] --> B[Density & Log Conv]
    B --> C[Winsorize & Scale]
    C --> D[Standardized Z tensor]
    D --> E[MI Prefilter]
    E --> F[PC Skeleton (FisherZ)]
    F --> G[Tier/Lag Mask]
    G --> H[NOTEARS Optimization]
    H --> I[Adjacency W & Blanket]
```

**Deployment Checklist:**  
- [ ] Define raw tensor shape \((N,T,J)\), dtypes (`float32`).  
- [ ] Collect denominators/populations.  
- [ ] Implement Phase1 transforms and verify no NaNs.  
- [ ] Test PC independence code on toy data.  
- [ ] Tune NOTEARS regularization for sparsity.  
- [ ] Validate learned structure on holdout data.  

