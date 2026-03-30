from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.harp_archive import run_harp_archive_build
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.phase3.rescue_core import (
    OFFICIAL_REFERENCE_POINTS,
    RESCUE_INFERENCE_FAMILY,
    RESCUE_PROFILE_ID,
    RESCUE_V2_PROFILE_ID,
    TRANSITION_PRIOR,
    run_phase3_rescue_core,
)
from epigraph_ph.runtime import (
    RunContext,
    choose_jax_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    numpy_to_jax_handoff,
    read_json,
    save_tensor_artifact,
    set_global_seed,
    to_numpy,
    utc_now_iso,
    write_ground_truth_package,
    write_json,
)

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam
except Exception:  # pragma: no cover
    numpyro = None
    dist = None
    Predictive = None
    SVI = None
    Trace_ELBO = None
    AutoNormal = None
    Adam = None


STATE_NAMES = ["U", "D", "A", "V", "L"]
TRANSITION_NAMES = ["U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A"]
_HIV_PLUGIN = get_disease_plugin("hiv")
_PHASE3_PRIOR_CFG = dict((_HIV_PLUGIN.prior_hyperparameters or {}).get("phase3", {}) or {})
_FROZEN_BACKTEST_CFG = dict(_PHASE3_PRIOR_CFG.get("frozen_backtest", {}) or {})


def _phase3_required_prior(key: str) -> Any:
    if key not in _PHASE3_PRIOR_CFG:
        raise KeyError(f"Missing HIV phase3 prior hyperparameter: {key}")
    return _PHASE3_PRIOR_CFG[key]


def _phase3_required_prior_section(key: str) -> dict[str, Any]:
    value = _phase3_required_prior(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase3 prior hyperparameter '{key}' must be a mapping")
    return dict(value)


def _phase3_required_frozen(key: str) -> Any:
    if key not in _FROZEN_BACKTEST_CFG:
        raise KeyError(f"Missing HIV phase3 frozen-backtest prior: {key}")
    return _FROZEN_BACKTEST_CFG[key]


def _phase3_required_frozen_section(key: str) -> dict[str, Any]:
    value = _phase3_required_frozen(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase3 frozen-backtest prior '{key}' must be a mapping")
    return dict(value)


DEFAULT_INITIAL_STATE = np.asarray(_phase3_required_frozen("default_initial_state"), dtype=np.float32)


def _infer_region_from_name(name: str) -> str:
    lowered = (name or "").strip().lower()
    if not lowered:
        return "unknown"
    if lowered in {"philippines", "national"}:
        return "national"
    if "ncr" in lowered or "metro manila" in lowered:
        return "ncr"
    if "car" in lowered or "cordillera" in lowered:
        return "car"
    if "calabarzon" in lowered or "region iv-a" in lowered or "iv-a" in lowered:
        return "region_iv_a"
    if "central visayas" in lowered or "region vii" in lowered or "vii" in lowered:
        return "region_vii"
    if "davao" in lowered or "region xi" in lowered or "xi" in lowered:
        return "region_xi"
    return "region_unknown"


def _region_assignments(province_axis: list[str], normalized_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    province_to_region: dict[str, Counter[str]] = defaultdict(Counter)
    for row in normalized_rows:
        geo = str(row.get("geo") or row.get("province") or "").strip()
        region = str(row.get("region") or "").strip().lower()
        if geo:
            province_to_region[geo][region or _infer_region_from_name(geo)] += 1
    regions = []
    for province in province_axis:
        counter = province_to_region.get(province, Counter())
        if counter:
            regions.append(counter.most_common(1)[0][0])
        else:
            regions.append(_infer_region_from_name(province))
    region_axis = sorted(set(regions)) or ["national"]
    region_index = np.asarray([region_axis.index(region) for region in regions], dtype=np.int32)
    return region_axis, region_index


def _month_ordinal(month_label: str) -> int | None:
    value = str(month_label or "")
    if len(value) >= 7 and value[:4].isdigit() and value[5:7].isdigit():
        return int(value[:4]) * 12 + int(value[5:7]) - 1
    return None


def _month_year(month_label: str) -> int | None:
    ordinal = _month_ordinal(month_label)
    if ordinal is None:
        return None
    return ordinal // 12


def _month_label_from_ordinal(ordinal: int) -> str:
    year = ordinal // 12
    month = (ordinal % 12) + 1
    return f"{year:04d}-{month:02d}"


def _month_indices_through_year(month_axis: list[str], max_year: int) -> list[int]:
    indices = [idx for idx, month in enumerate(month_axis) if (_month_year(month) is not None and int(_month_year(month) or 0) <= max_year)]
    return indices


def _filter_rows_through_year(rows: list[dict[str, Any]], max_year: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        year = _month_year(str(row.get("time") or ""))
        if year is not None and year > max_year:
            continue
        filtered.append(dict(row))
    return filtered


def _latest_month_for_year(month_axis: list[str], year: int) -> str:
    candidates = [month for month in month_axis if _month_year(month) == year]
    if not candidates:
        return f"{year:04d}-01"
    return max(candidates, key=lambda item: _month_ordinal(item) or -1)


def _archive_point_month_lookup(metric_rows: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, tuple[float, str]] = {}
    for row in metric_rows:
        year = int(row.get("year") or 0)
        month = str(row.get("time") or "")
        if not month:
            continue
        score = float(row.get("evidence_confidence") or 0.0)
        previous = lookup.get(year)
        if previous is None or score >= previous[0]:
            lookup[year] = (score, month)
    return {year: month for year, (_, month) in lookup.items()}


def _archive_points_from_panel_rows(
    panel_rows: list[dict[str, Any]],
    years: list[int],
    month_axis: list[str],
    *,
    point_month_lookup: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    year_set = {int(year) for year in years}
    points: list[dict[str, Any]] = []
    for row in panel_rows:
        year = int(row.get("year") or 0)
        if year not in year_set:
            continue
        estimated_plhiv = row.get("estimated_plhiv")
        diagnosed = row.get("diagnosed_plhiv")
        on_art = row.get("alive_on_art")
        tested = row.get("tested_for_viral_load")
        suppressed = row.get("virally_suppressed")
        if any(value in {None, ""} for value in [estimated_plhiv, diagnosed, on_art, tested, suppressed]):
            continue
        month = str((point_month_lookup or {}).get(year) or _latest_month_for_year(month_axis, year))
        points.append(
            {
                "label": f"Historical HARP annual snapshot {year}",
                "month": month,
                "estimated_plhiv": float(estimated_plhiv),
                "diagnosed": float(diagnosed),
                "on_art": float(on_art),
                "viral_load_tested": float(tested),
                "suppressed": float(suppressed),
                "source_label": str(row.get("source_label") or "Historical HARP archive"),
                "year": year,
            }
        )
    points.sort(key=lambda item: (int(item.get("year") or 0), str(item.get("month") or "")))
    return points


def _empirical_transition_targets(core_tensor: np.ndarray) -> np.ndarray:
    cfg = _phase3_required_frozen_section("empirical_transition_targets")
    if core_tensor.size == 0:
        return np.zeros((0, 0, len(TRANSITION_NAMES)), dtype=np.float32)
    positive_mass = np.maximum(core_tensor, 0.0).mean(axis=-1)
    negative_mass = np.maximum(-core_tensor, 0.0).mean(axis=-1)
    feature_mean = core_tensor.mean(axis=-1)
    feature_std = core_tensor.std(axis=-1)
    positive_weights = np.asarray(cfg["positive_mass_weight"], dtype=np.float32)
    negative_weights = np.asarray(cfg["negative_mass_weight"], dtype=np.float32)
    feature_mean_weights = np.asarray(cfg["feature_mean_weight"], dtype=np.float32)
    feature_std_weights = np.asarray(cfg["feature_std_weight"], dtype=np.float32)
    intercept = np.asarray(cfg["intercept"], dtype=np.float32)
    targets = np.stack(
        [
            1.0 / (1.0 + np.exp(-(intercept[0] + positive_weights[0] * positive_mass + negative_weights[0] * negative_mass + feature_mean_weights[0] * feature_mean + feature_std_weights[0] * feature_std))),
            1.0 / (1.0 + np.exp(-(intercept[1] + positive_weights[1] * positive_mass + negative_weights[1] * negative_mass + feature_mean_weights[1] * feature_mean + feature_std_weights[1] * feature_std))),
            1.0 / (1.0 + np.exp(-(intercept[2] + positive_weights[2] * positive_mass + negative_weights[2] * negative_mass + feature_mean_weights[2] * feature_mean + feature_std_weights[2] * feature_std))),
            1.0 / (1.0 + np.exp(-(intercept[3] + positive_weights[3] * positive_mass + negative_weights[3] * negative_mass + feature_mean_weights[3] * feature_mean + feature_std_weights[3] * feature_std))),
            1.0 / (1.0 + np.exp(-(intercept[4] + positive_weights[4] * positive_mass + negative_weights[4] * negative_mass + feature_mean_weights[4] * feature_mean + feature_std_weights[4] * feature_std))),
        ],
        axis=-1,
    )
    return np.clip(targets.astype(np.float32), float(cfg["clip_floor"]), float(cfg["clip_ceiling"]))


def _semi_markov_model(features: Any, region_index: Any, empirical_targets: Any, region_count: int) -> None:
    if numpyro is None or jnp is None:
        raise RuntimeError("NumPyro/JAX backend is unavailable")
    cfg = _phase3_required_frozen_section("semi_markov_hyperpriors")
    province_count, month_count, feature_count = features.shape
    transition_count = empirical_targets.shape[-1]
    base_rate = jnp.clip(jnp.mean(empirical_targets, axis=(0, 1)), float(cfg["base_rate_floor"]), float(cfg["base_rate_ceiling"]))
    national_logit = numpyro.sample(
        "national_logit",
        dist.Normal(jnp.log(base_rate / (1.0 - base_rate)), float(cfg["national_logit_sigma"])).to_event(1),
    )
    sigma_region = numpyro.sample("sigma_region", dist.HalfNormal(jnp.ones((transition_count,)) * float(cfg["sigma_region"])).to_event(1))
    sigma_province = numpyro.sample("sigma_province", dist.HalfNormal(jnp.ones((transition_count,)) * float(cfg["sigma_province"])).to_event(1))
    sigma_feature = numpyro.sample("sigma_feature", dist.HalfNormal(jnp.ones((transition_count,)) * float(cfg["sigma_feature"])).to_event(1))
    region_offset = numpyro.sample(
        "region_offset",
        dist.Normal(jnp.zeros((region_count, transition_count)), sigma_region).to_event(2),
    )
    province_offset = numpyro.sample(
        "province_offset",
        dist.Normal(jnp.zeros((province_count, transition_count)), sigma_province).to_event(2),
    )
    feature_weights = numpyro.sample(
        "feature_weights",
        dist.Normal(jnp.zeros((feature_count, transition_count)), sigma_feature).to_event(2),
    )
    duration_weight = numpyro.sample(
        "duration_weight",
        dist.Normal(jnp.asarray(cfg["duration_prior_mean"]), float(cfg["duration_prior_sigma"])).to_event(1),
    )
    logits = (
        national_logit.reshape(1, 1, transition_count)
        + region_offset[region_index][:, None, :]
        + province_offset[:, None, :]
        + jnp.einsum("ptf,fk->ptk", features, feature_weights)
    )
    probs = jax.nn.sigmoid(logits)
    numpyro.deterministic("transition_probs", probs)
    concentration = float(cfg["beta_concentration"])
    alpha = jnp.clip(probs * concentration, float(cfg["beta_floor"]), None)
    beta = jnp.clip((1.0 - probs) * concentration, float(cfg["beta_floor"]), None)
    numpyro.sample("transition_obs", dist.Beta(alpha, beta).to_event(3), obs=empirical_targets)


def _posterior_draws(
    features_np: np.ndarray,
    region_index_np: np.ndarray,
    empirical_targets_np: np.ndarray,
    *,
    seed: int,
    svi_steps: int | None = None,
    posterior_samples: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if jax is None or numpyro is None or SVI is None or AutoNormal is None or Predictive is None or Trace_ELBO is None or Adam is None:
        raise RuntimeError("JAX/NumPyro backend is unavailable")
    infer_cfg = _phase3_required_frozen_section("posterior_inference")
    svi_steps = int(infer_cfg["svi_steps"]) if svi_steps is None else int(svi_steps)
    posterior_samples = int(infer_cfg["posterior_samples"]) if posterior_samples is None else int(posterior_samples)
    set_global_seed(seed)
    features = numpy_to_jax_handoff(features_np).astype(jnp.float32)
    region_index = numpy_to_jax_handoff(region_index_np).astype(jnp.int32)
    empirical_targets = numpy_to_jax_handoff(empirical_targets_np).astype(jnp.float32)
    guide = AutoNormal(_semi_markov_model)
    svi = SVI(_semi_markov_model, guide, Adam(float(infer_cfg["optimizer_lr"])), Trace_ELBO())
    rng_key = jax.random.PRNGKey(seed)
    region_count = int(region_index_np.max()) + 1 if region_index_np.size else 1
    svi_result = svi.run(
        rng_key,
        svi_steps,
        features=features,
        region_index=region_index,
        empirical_targets=empirical_targets,
        region_count=region_count,
        progress_bar=False,
    )
    predictive = Predictive(
        _semi_markov_model,
        guide=guide,
        params=svi_result.params,
        num_samples=posterior_samples,
        return_sites=["national_logit", "region_offset", "province_offset", "feature_weights", "duration_weight", "transition_probs"],
    )
    samples = predictive(
        jax.random.PRNGKey(seed + 1),
        features=features,
        region_index=region_index,
        empirical_targets=empirical_targets,
        region_count=region_count,
    )
    np_samples = {key: to_numpy(value) for key, value in samples.items()}
    diagnostics = {
        "svi_steps": svi_steps,
        "posterior_samples": posterior_samples,
        "final_loss": float(svi_result.losses[-1]) if len(svi_result.losses) else None,
        "loss_trace_tail": [float(value) for value in svi_result.losses[-10:]],
        "jax_device": choose_jax_device(prefer_gpu=True),
    }
    return np_samples, diagnostics


def _fallback_posterior_draws(
    features_np: np.ndarray,
    region_index_np: np.ndarray,
    empirical_targets_np: np.ndarray,
    *,
    seed: int,
    posterior_samples: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cfg = _phase3_required_frozen_section("fallback_posterior")
    posterior_samples = int(cfg["posterior_samples"]) if posterior_samples is None else int(posterior_samples)
    rng = np.random.default_rng(seed)
    province_count, _, feature_count = features_np.shape
    region_count = int(region_index_np.max()) + 1 if province_count else 1
    transition_count = empirical_targets_np.shape[-1]
    mean_probs = empirical_targets_np.mean(axis=1, keepdims=True)
    mean_floor = float(cfg["mean_floor"])
    mean_ceiling = float(cfg["mean_ceiling"])
    national_logit = np.log(np.clip(empirical_targets_np.mean(axis=(0, 1)), mean_floor, mean_ceiling) / np.clip(1.0 - empirical_targets_np.mean(axis=(0, 1)), mean_floor, mean_ceiling))
    transition_probs = np.clip(
        rng.normal(loc=mean_probs, scale=float(cfg["transition_noise_scale"]), size=(posterior_samples, *empirical_targets_np.shape)),
        float(cfg["clip_floor"]),
        float(cfg["clip_ceiling"]),
    ).astype(np.float32)
    samples = {
        "national_logit": rng.normal(loc=national_logit, scale=float(cfg["national_logit_scale"]), size=(posterior_samples, transition_count)).astype(np.float32),
        "region_offset": rng.normal(loc=0.0, scale=float(cfg["region_offset_scale"]), size=(posterior_samples, region_count, transition_count)).astype(np.float32),
        "province_offset": rng.normal(loc=0.0, scale=float(cfg["province_offset_scale"]), size=(posterior_samples, province_count, transition_count)).astype(np.float32),
        "feature_weights": rng.normal(loc=0.0, scale=float(cfg["feature_weight_scale"]), size=(posterior_samples, feature_count, transition_count)).astype(np.float32),
        "duration_weight": rng.normal(loc=np.asarray(cfg["duration_mean"]), scale=float(cfg["duration_scale"]), size=(posterior_samples, transition_count)).astype(np.float32),
        "transition_probs": transition_probs,
    }
    diagnostics = {
        "svi_steps": 0,
        "posterior_samples": posterior_samples,
        "final_loss": None,
        "loss_trace_tail": [],
        "jax_device": "unavailable",
        "fallback": True,
    }
    return samples, diagnostics


def _semi_markov_evolve(
    transition_probs: np.ndarray,
    duration_weight: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    forecast_horizon: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if jax is None or jnp is None:
        raise RuntimeError("JAX backend is unavailable")
    cfg = _phase3_required_frozen_section("evolution")
    probs = numpy_to_jax_handoff(transition_probs).astype(jnp.float32)
    duration_weight_jax = numpy_to_jax_handoff(duration_weight).astype(jnp.float32)
    province_count, month_count, _ = probs.shape
    init = numpy_to_jax_handoff(initial_state if initial_state is not None else DEFAULT_INITIAL_STATE).astype(jnp.float32)
    init = jnp.broadcast_to(init, (province_count, len(STATE_NAMES)))

    def _step(carry, prob_t):
        state, duration = carry
        duration_effect = jnp.tanh(duration / float(cfg["duration_effect_scale"]))
        adjusted = jnp.clip(
            prob_t + duration_effect[:, [0, 1, 2, 2, 4]] * duration_weight_jax.reshape(1, -1),
            float(cfg["transition_floor"]),
            float(cfg["transition_ceiling"]),
        )
        p_ud, p_da, p_av, p_al, p_la = [adjusted[:, idx] for idx in range(adjusted.shape[-1])]
        U, D, A, V, L = [state[:, idx] for idx in range(state.shape[-1])]
        flow_ud = U * p_ud
        flow_da = D * p_da
        flow_av = A * p_av
        flow_al = A * p_al
        flow_la = L * p_la
        next_state = jnp.stack(
            [
                U - flow_ud,
                D + flow_ud - flow_da,
                A + flow_da + flow_la - flow_av - flow_al,
                V + flow_av,
                L + flow_al - flow_la,
            ],
            axis=-1,
        )
        next_state = jnp.clip(next_state, 0.0, None)
        next_state = next_state / jnp.clip(jnp.sum(next_state, axis=-1, keepdims=True), float(cfg["state_mass_eps"]), None)
        stay_state = jnp.stack(
            [
                U - flow_ud,
                D - flow_da,
                A - flow_av - flow_al,
                V,
                L - flow_la,
            ],
            axis=-1,
        )
        next_duration = ((duration + 1.0) * jnp.clip(stay_state, 0.0, None)) / jnp.clip(next_state, float(cfg["state_mass_eps"]), None)
        next_duration = jnp.clip(next_duration, float(cfg["duration_floor"]), float(cfg["duration_ceiling"]))
        return (next_state, next_duration), next_state

    (_, terminal_duration), state_path = jax.lax.scan(
        _step,
        (init, jnp.zeros_like(init)),
        jnp.swapaxes(probs, 0, 1),
    )
    state_path = jnp.swapaxes(state_path, 0, 1)
    last_prob = probs[:, -1:, :]
    forecast_probs = jnp.repeat(last_prob, repeats=max(1, forecast_horizon), axis=1)
    (_, _), forecast_state_path = jax.lax.scan(
        _step,
        (state_path[:, -1, :], terminal_duration),
        jnp.swapaxes(forecast_probs, 0, 1),
    )
    forecast_state_path = jnp.swapaxes(forecast_state_path, 0, 1)
    return to_numpy(state_path).astype(np.float32), to_numpy(forecast_state_path).astype(np.float32), to_numpy(terminal_duration).astype(np.float32)


def _state_rows(state_tensor: np.ndarray, province_axis: list[str], month_axis: list[str]) -> list[dict[str, Any]]:
    rows = []
    for province_idx, province in enumerate(province_axis):
        for month_idx, month in enumerate(month_axis):
            for state_idx, state_name in enumerate(STATE_NAMES):
                rows.append(
                    {
                        "province": province,
                        "time": month,
                        "state": state_name,
                        "value": round(float(state_tensor[province_idx, month_idx, state_idx]), 6),
                    }
                )
    return rows


def _fit_rows(state_tensor: np.ndarray, transition_probs: np.ndarray, province_axis: list[str], month_axis: list[str]) -> list[dict[str, Any]]:
    rows = []
    for province_idx, province in enumerate(province_axis):
        for month_idx, month in enumerate(month_axis):
            row = {
                "province": province,
                "time": month,
                "state_U": round(float(state_tensor[province_idx, month_idx, 0]), 6),
                "state_D": round(float(state_tensor[province_idx, month_idx, 1]), 6),
                "state_A": round(float(state_tensor[province_idx, month_idx, 2]), 6),
                "state_V": round(float(state_tensor[province_idx, month_idx, 3]), 6),
                "state_L": round(float(state_tensor[province_idx, month_idx, 4]), 6),
                "p_U_to_D": round(float(transition_probs[province_idx, month_idx, 0]), 6),
                "p_D_to_A": round(float(transition_probs[province_idx, month_idx, 1]), 6),
                "p_A_to_V": round(float(transition_probs[province_idx, month_idx, 2]), 6),
                "p_A_to_L": round(float(transition_probs[province_idx, month_idx, 3]), 6),
                "p_L_to_A": round(float(transition_probs[province_idx, month_idx, 4]), 6),
            }
            rows.append(row)
    return rows


def _run_phase3_build_legacy(*, run_id: str, plugin_id: str, top_k_per_block: int | None = None) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase3_dir = ensure_dir(ctx.run_dir / "phase3")
    legacy_cfg = _phase3_required_frozen_section("legacy_selection")
    inference_cfg = _phase3_required_frozen_section("posterior_inference")
    top_k_per_block = int(legacy_cfg["top_k_per_block"]) if top_k_per_block is None else int(top_k_per_block)
    set_global_seed(int(inference_cfg["seed"]))

    axis_catalogs = read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    province_axis = list(axis_catalogs.get("province", [])) or ["national"]
    month_axis = list(axis_catalogs.get("month", [])) or ["unknown"]
    normalized_rows = read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    candidate_profiles = read_json(ctx.run_dir / "phase2" / "candidate_profiles.json", default=[])
    curated_candidate_blocks = read_json(ctx.run_dir / "phase2" / "curated_candidate_blocks.json", default=[])
    markov_blanket = read_json(ctx.run_dir / "phase2" / "markov_blanket.json", default={})
    core_tensor = load_tensor_artifact(ctx.run_dir / "phase2" / "core_feature_tensor.npz")

    region_axis, region_index = _region_assignments(province_axis, normalized_rows)
    empirical_targets = _empirical_transition_targets(core_tensor)

    backend_map = detect_backends()
    use_jax = backend_map["jax"].available and numpyro is not None and core_tensor.size > 0
    if use_jax:
        posterior_samples, posterior_diag = _posterior_draws(core_tensor, region_index, empirical_targets, seed=int(inference_cfg["seed"]))
    else:
        posterior_samples, posterior_diag = _fallback_posterior_draws(core_tensor, region_index, empirical_targets, seed=int(inference_cfg["seed"]))

    transition_probs = posterior_samples["transition_probs"].mean(axis=0).astype(np.float32)
    duration_weight = posterior_samples["duration_weight"].mean(axis=0).astype(np.float32)
    state_estimates, forecast_states, terminal_duration = _semi_markov_evolve(
        transition_probs,
        duration_weight,
        initial_state=DEFAULT_INITIAL_STATE,
        forecast_horizon=min(6, max(2, len(month_axis))),
    )

    state_rows = _state_rows(state_estimates, province_axis, month_axis)
    fit_rows = _fit_rows(state_estimates, transition_probs, province_axis, month_axis)
    forecast_month_labels = [f"forecast_h{idx + 1}" for idx in range(forecast_states.shape[1])]
    forecast_rows = _state_rows(forecast_states, province_axis, forecast_month_labels)

    direct_signal_names = [
        profile["canonical_name"]
        for profile in candidate_profiles
        if profile.get("curation_status") in {"promoted_candidate", "research_candidate"} and (profile.get("numeric_support", 0) > 0 or profile.get("anchor_support", 0) > 0)
    ]
    structural_prior_names = [
        profile["canonical_name"]
        for profile in candidate_profiles
        if profile.get("curation_status") in {"promoted_candidate", "research_candidate", "review"} and profile["canonical_name"] not in direct_signal_names
    ]
    inference_ready_candidates = []
    grouped_profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in candidate_profiles:
        grouped_profiles[profile.get("primary_block") or "mixed"].append(profile)
    for block_name, members in grouped_profiles.items():
        for profile in sorted(members, key=lambda item: (item.get("curation_score", 0.0), item.get("dag_score", 0.0)), reverse=True)[:top_k_per_block]:
            role = "direct_signal" if profile["canonical_name"] in direct_signal_names else "structural_prior"
            inference_ready_candidates.append(
                {
                    "canonical_name": profile["canonical_name"],
                    "primary_block": block_name,
                    "role": role,
                    "curation_status": profile.get("curation_status"),
                    "curation_score": profile.get("curation_score"),
                    "dag_score": profile.get("dag_score"),
                    "linkage_targets": profile.get("linkage_targets", []),
                    "blanket_member": profile.get("blanket_member", False),
                }
            )

    transition_parameters = {
        "transition_names": TRANSITION_NAMES,
        "national_logit_mean": posterior_samples["national_logit"].mean(axis=0).round(6).tolist(),
        "national_logit_sd": posterior_samples["national_logit"].std(axis=0).round(6).tolist(),
        "duration_weight_mean": posterior_samples["duration_weight"].mean(axis=0).round(6).tolist(),
        "duration_weight_sd": posterior_samples["duration_weight"].std(axis=0).round(6).tolist(),
        "region_axis": region_axis,
        "province_axis": province_axis,
    }
    posterior_summaries = {
        "transition_prob_mean": posterior_samples["transition_probs"].mean(axis=(0, 1)).round(6).tolist(),
        "transition_prob_sd": posterior_samples["transition_probs"].std(axis=(0, 1)).round(6).tolist(),
        "feature_weight_mean": posterior_samples["feature_weights"].mean(axis=0).round(6).tolist(),
        "feature_weight_sd": posterior_samples["feature_weights"].std(axis=0).round(6).tolist(),
        "posterior_diagnostics": posterior_diag,
    }

    mass_error = float(np.max(np.abs(state_estimates.sum(axis=-1) - 1.0))) if state_estimates.size else 0.0
    transition_floor = float(transition_probs.min()) if transition_probs.size else 0.0
    transition_ceiling = float(transition_probs.max()) if transition_probs.size else 0.0
    mass_error_tolerance = float(legacy_cfg["mass_error_tolerance"])
    validation_gates = [
        {"gate": "state_mass_conservation", "passed": mass_error < mass_error_tolerance, "value": round(mass_error, 8)},
        {"gate": "transition_probability_validity", "passed": transition_floor >= 0.0 and transition_ceiling <= 1.0, "value": {"min": round(transition_floor, 6), "max": round(transition_ceiling, 6)}},
        {"gate": "markov_blanket_nonempty", "passed": len(markov_blanket.get("blanket_nodes", [])) > 0, "value": len(markov_blanket.get("blanket_nodes", []))},
        {"gate": "core_feature_tensor_nonempty", "passed": core_tensor.size > 0, "value": list(core_tensor.shape)},
        {"gate": "posterior_draws_available", "passed": posterior_samples["transition_probs"].shape[0] > 0, "value": int(posterior_samples["transition_probs"].shape[0])},
        {"gate": "direct_signal_floor", "passed": len(direct_signal_names) > 0, "value": len(direct_signal_names)},
    ]
    claim_eligible = all(gate["passed"] for gate in validation_gates)

    model_artifact = {
        "model_family": "hierarchical_causal_semi_markov",
        "backend": "jax_numpyro" if use_jax else "numpy_fallback",
        "state_names": STATE_NAMES,
        "transition_names": TRANSITION_NAMES,
        "province_axis": province_axis,
        "region_axis": region_axis,
        "month_axis": month_axis,
        "core_feature_shape": list(core_tensor.shape),
        "markov_blanket": markov_blanket,
        "notes": [
            "phase3_state_space:U_D_A_V_L",
            "phase3_runtime_split:explicit_numpy_to_jax_handoff",
            "phase3_transition_kernel:feature_conditioned_duration_sensitive",
        ],
    }
    fit_artifact = {
        "fit_rows": fit_rows,
        "transition_parameters": transition_parameters,
        "posterior_summaries": posterior_summaries,
        "region_assignments": {province_axis[idx]: region_axis[region_index[idx]] for idx in range(len(province_axis))},
    }
    forecast_bundle = {
        "forecast_horizon": forecast_states.shape[1],
        "forecast_month_labels": forecast_month_labels,
        "forecast_rows": forecast_rows,
        "latest_state_by_province": {
            province_axis[idx]: {STATE_NAMES[state_idx]: round(float(state_estimates[idx, -1, state_idx]), 6) for state_idx in range(len(STATE_NAMES))}
            for idx in range(len(province_axis))
        },
    }
    validation_artifact = {
        "validation_gates": validation_gates,
        "claim_eligible": claim_eligible,
        "state_mass_error_max": round(mass_error, 8),
        "direct_signal_count": len(direct_signal_names),
        "structural_prior_count": len(structural_prior_names),
    }

    state_estimates_artifact = save_tensor_artifact(
        array=state_estimates,
        axis_names=["province", "month", "state"],
        artifact_dir=phase3_dir,
        stem="state_estimates",
        backend="jax" if use_jax else "numpy",
        device=choose_jax_device(prefer_gpu=True) if use_jax else "cpu",
        notes=["phase3_state_estimates"],
        save_pt=False,
    )
    forecast_states_artifact = save_tensor_artifact(
        array=forecast_states,
        axis_names=["province", "forecast_step", "state"],
        artifact_dir=phase3_dir,
        stem="forecast_states",
        backend="jax" if use_jax else "numpy",
        device=choose_jax_device(prefer_gpu=True) if use_jax else "cpu",
        notes=["phase3_forecast_states"],
        save_pt=False,
    )

    write_json(phase3_dir / "inference_ready_candidates.json", inference_ready_candidates)
    write_json(phase3_dir / "state_rows.json", state_rows)
    write_json(phase3_dir / "transition_parameters.json", transition_parameters)
    write_json(phase3_dir / "posterior_summaries.json", posterior_summaries)
    write_json(phase3_dir / "model_artifact.json", model_artifact)
    write_json(phase3_dir / "fit_artifact.json", fit_artifact)
    write_json(phase3_dir / "forecast_bundle.json", forecast_bundle)
    write_json(phase3_dir / "validation_artifact.json", validation_artifact)

    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase2"),
        extracted_dir=str(phase3_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase3": "completed"},
        artifact_paths={
            "state_estimates": state_estimates_artifact["value_path"],
            "forecast_states": forecast_states_artifact["value_path"],
            "inference_ready_candidates": str(phase3_dir / "inference_ready_candidates.json"),
            "transition_parameters": str(phase3_dir / "transition_parameters.json"),
            "posterior_summaries": str(phase3_dir / "posterior_summaries.json"),
            "model_artifact": str(phase3_dir / "model_artifact.json"),
            "fit_artifact": str(phase3_dir / "fit_artifact.json"),
            "forecast_bundle": str(phase3_dir / "forecast_bundle.json"),
            "validation_artifact": str(phase3_dir / "validation_artifact.json"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, False, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, use_jax, notes=backend_map["jax"].device),
        },
        source_count=len(candidate_profiles),
        canonical_candidate_count=len(inference_ready_candidates),
        numeric_observation_count=len(state_rows),
        notes=["phase3_hierarchical_semi_markov:jitted_runtime_boundary"],
    ).to_dict()
    truth_paths = write_ground_truth_package(
        phase_dir=phase3_dir,
        phase_name="phase3",
        profile_id="legacy",
        checks=[
            {"name": "state_estimates_finite", "passed": bool(np.isfinite(state_estimates).all())},
            {"name": "forecast_states_finite", "passed": bool(np.isfinite(forecast_states).all())},
            {"name": "validation_gates_present", "passed": bool(validation_gates)},
            {"name": "state_mass_conserved", "passed": mass_error < mass_error_tolerance},
        ],
        truth_sources=["benchmark_truth", "synthetic_truth", "prior_truth"],
        stage_manifest_path=str(phase3_dir / "phase3_manifest.json"),
        summary={
            "fit_row_count": len(fit_rows),
            "state_row_count": len(state_rows),
            "direct_signal_count": len(direct_signal_names),
            "structural_prior_count": len(structural_prior_names),
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase3_dir / "phase3_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase3_build",
        [
            phase3_dir / "state_estimates.npz",
            phase3_dir / "forecast_states.npz",
            phase3_dir / "inference_ready_candidates.json",
            phase3_dir / "state_rows.json",
            phase3_dir / "transition_parameters.json",
            phase3_dir / "posterior_summaries.json",
            phase3_dir / "model_artifact.json",
            phase3_dir / "fit_artifact.json",
            phase3_dir / "forecast_bundle.json",
            phase3_dir / "validation_artifact.json",
            phase3_dir / "phase3_manifest.json",
        ],
    )
    return manifest


def run_phase3_build(
    *,
    run_id: str,
    plugin_id: str,
    top_k_per_block: int | None = None,
    profile: str = "legacy",
    inference_family: str = RESCUE_INFERENCE_FAMILY,
) -> dict[str, Any]:
    if profile in {RESCUE_PROFILE_ID, RESCUE_V2_PROFILE_ID}:
        return run_phase3_rescue_core(
            run_id=run_id,
            plugin_id=plugin_id,
            profile_id=profile,
            requested_inference_family=inference_family,
        )
    return _run_phase3_build_legacy(run_id=run_id, plugin_id=plugin_id, top_k_per_block=top_k_per_block)


def run_phase3_frozen_backtest(
    *,
    run_id: str,
    plugin_id: str,
    profile: str = RESCUE_V2_PROFILE_ID,
    inference_family: str = RESCUE_INFERENCE_FAMILY,
    train_years: list[int] | None = None,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    if profile not in {RESCUE_PROFILE_ID, RESCUE_V2_PROFILE_ID}:
        raise ValueError("phase3 frozen backtest is only implemented for hiv rescue profiles")
    run_harp_archive_build(run_id=run_id, plugin_id=plugin_id)
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    archive_dir = ctx.run_dir / "harp_archive"
    archive_spec = read_json(archive_dir / "frozen_backtest_spec.json", default={})
    archive_panel = read_json(archive_dir / "historical_harp_panel.json", default={})
    observed_program_panel = read_json(archive_dir / "observed_program_panel.json", default={})
    if not archive_spec.get("ready_for_model_backtest"):
        raise RuntimeError("historical HARP archive is not ready for a frozen Phase 3 backtest")

    train_years = [int(year) for year in (train_years or archive_spec.get("train_years") or [])]
    holdout_years = [int(year) for year in (holdout_years or archive_spec.get("holdout_years") or [])]
    if not train_years or not holdout_years:
        raise RuntimeError("frozen Phase 3 backtest requires non-empty train and holdout years")

    axis_catalogs = read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    normalized_rows = read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    parameter_catalog = read_json(ctx.run_dir / "phase1" / "parameter_catalog.json", default=[])
    standardized_tensor = load_tensor_artifact(ctx.run_dir / "phase1" / "standardized_tensor.npz")

    full_month_axis = list(axis_catalogs.get("month", []))
    if not full_month_axis:
        raise RuntimeError("phase1 month axis is missing; cannot run frozen Phase 3 backtest")
    month_indices = _month_indices_through_year(full_month_axis, max(train_years))
    if not month_indices:
        raise RuntimeError("no phase1 months were retained after applying the frozen-history training cutoff")
    filtered_month_axis = [full_month_axis[idx] for idx in month_indices]
    filtered_axis_catalogs = dict(axis_catalogs)
    filtered_axis_catalogs["month"] = filtered_month_axis
    filtered_rows = _filter_rows_through_year(normalized_rows, max(train_years))
    filtered_tensor = np.asarray(standardized_tensor[:, month_indices, :], dtype=np.float32)

    panel_rows = list(archive_panel.get("rows") or [])
    point_month_lookup = _archive_point_month_lookup(list(observed_program_panel.get("rows") or []))
    train_harp_points = _archive_points_from_panel_rows(panel_rows, train_years, full_month_axis, point_month_lookup=point_month_lookup)
    holdout_harp_points = _archive_points_from_panel_rows(panel_rows, holdout_years, full_month_axis, point_month_lookup=point_month_lookup)
    if not train_harp_points or not holdout_harp_points:
        raise RuntimeError("historical HARP panel does not contain complete train/holdout program snapshots")
    official_points = [
        dict(point)
        for point in OFFICIAL_REFERENCE_POINTS
        if (_month_year(str(point.get("month") or "")) or 0) <= max(train_years)
    ]
    train_last_model_month = filtered_month_axis[-1]
    holdout_eval_months = [str(point.get("month") or _latest_month_for_year(full_month_axis, int(point.get("year") or 0))) for point in holdout_harp_points]
    last_train_ordinal = _month_ordinal(train_last_model_month)
    holdout_ordinals = [_month_ordinal(month) for month in holdout_eval_months if _month_ordinal(month) is not None]
    forecast_horizon = 1
    if last_train_ordinal is not None and holdout_ordinals:
        forecast_horizon = max(max(holdout_ordinals) - last_train_ordinal, 1)
    backtest_config = {
        "mode": "frozen_history",
        "train_years": train_years,
        "holdout_years": holdout_years,
        "train_last_model_month": train_last_model_month,
        "holdout_eval_months": holdout_eval_months,
        "forecast_horizon": forecast_horizon,
        "train_harp_points": train_harp_points,
        "holdout_harp_points": holdout_harp_points,
    }
    return run_phase3_rescue_core(
        run_id=run_id,
        plugin_id=plugin_id,
        profile_id=profile,
        requested_inference_family=inference_family,
        phase_dir_name="phase3_frozen_backtest",
        axis_catalogs_override=filtered_axis_catalogs,
        normalized_rows_override=filtered_rows,
        parameter_catalog_override=parameter_catalog,
        standardized_tensor_override=filtered_tensor,
        reference_overrides={"official_points": official_points, "harp_points": train_harp_points},
        backtest_config=backtest_config,
    )


def _frozen_tuning_candidates() -> list[dict[str, Any]]:
    rows = list(_phase3_required_frozen_section("tuning")["initial_candidates"] or [])
    return [dict(row) for row in rows]


def _trial_calibration_overrides(candidate: dict[str, Any]) -> dict[str, Any]:
    transition_prior = np.asarray(TRANSITION_PRIOR, dtype=np.float32).copy()
    transition_prior[0] = float(candidate.get("u_to_d_prior", transition_prior[0]))
    transition_prior[1] = float(candidate.get("d_to_a_prior", transition_prior[1]))
    torch_cfg = _phase3_required_prior_section("torch_map_loss_scales")
    tuning_cfg = dict(_phase3_required_frozen_section("tuning")["adaptive_rules"])
    return {
        "transition_prior_override": transition_prior.tolist(),
        "diagnosed_penalty_scale": float(candidate.get("diagnosed_penalty_scale", torch_cfg["diagnosed_penalty_scale"])),
        "official_reference_penalty_scale": float(candidate.get("official_reference_penalty_scale", torch_cfg["official_reference_penalty_scale"])),
        "national_anchor_penalty_scale": float(candidate.get("national_anchor_penalty_scale", torch_cfg["national_anchor_penalty_scale"])),
        "harp_program_penalty_scale": float(candidate.get("harp_program_penalty_scale", torch_cfg["harp_program_penalty_scale"])),
        "linkage_penalty_scale": float(candidate.get("linkage_penalty_scale", torch_cfg["linkage_penalty_scale"])),
        "suppression_penalty_scale": float(candidate.get("suppression_penalty_scale", torch_cfg["suppression_penalty_scale"])),
        "fit_steps": int(candidate.get("fit_steps", tuning_cfg["adaptive_fit_steps"])),
    }


def _adaptive_frozen_tuning_candidates(best_trial: dict[str, Any]) -> list[dict[str, Any]]:
    rules = dict(_phase3_required_frozen_section("tuning")["adaptive_rules"])
    torch_cfg = _phase3_required_prior_section("torch_map_loss_scales")
    base = dict(best_trial.get("calibration_overrides") or {})
    transition_prior = np.asarray(base.get("transition_prior_override") or TRANSITION_PRIOR, dtype=np.float32)
    u_to_d = float(transition_prior[0])
    d_to_a = float(transition_prior[1])
    diagnosed_penalty = float(base.get("diagnosed_penalty_scale", torch_cfg["diagnosed_penalty_scale"]))
    official_penalty = float(base.get("official_reference_penalty_scale", torch_cfg["official_reference_penalty_scale"]))
    national_penalty = float(base.get("national_anchor_penalty_scale", torch_cfg["national_anchor_penalty_scale"]))
    harp_penalty = float(base.get("harp_program_penalty_scale", torch_cfg["harp_program_penalty_scale"]))
    linkage_penalty = float(base.get("linkage_penalty_scale", torch_cfg["linkage_penalty_scale"]))
    return [
        {
            "label": "adaptive_dx_floor_link_push",
            "u_to_d_prior": max(float(rules["u_to_d_floor_soft"]), round(u_to_d * float(rules["u_to_d_scale_soft"]), 4)),
            "d_to_a_prior": min(float(rules["d_to_a_ceiling_soft"]), round(d_to_a + float(rules["d_to_a_increment_soft"]), 4)),
            "diagnosed_penalty_scale": diagnosed_penalty + float(rules["diagnosed_penalty_soft"]),
            "official_reference_penalty_scale": max(float(rules["official_penalty_floor_soft"]), official_penalty - float(rules["official_penalty_reduction_soft"])),
            "national_anchor_penalty_scale": max(float(rules["national_penalty_floor_soft"]), national_penalty - float(rules["national_penalty_reduction_soft"])),
            "harp_program_penalty_scale": harp_penalty + float(rules["harp_penalty_add_soft"]),
            "linkage_penalty_scale": linkage_penalty + float(rules["linkage_penalty_add_soft"]),
            "fit_steps": int(rules["adaptive_fit_steps"]),
        },
        {
            "label": "adaptive_dx_floor_link_hard",
            "u_to_d_prior": max(float(rules["u_to_d_floor_hard"]), round(u_to_d * float(rules["u_to_d_scale_hard"]), 4)),
            "d_to_a_prior": min(float(rules["d_to_a_ceiling_hard"]), round(d_to_a + float(rules["d_to_a_increment_hard"]), 4)),
            "diagnosed_penalty_scale": diagnosed_penalty + float(rules["diagnosed_penalty_hard"]),
            "official_reference_penalty_scale": max(float(rules["official_penalty_floor_hard"]), official_penalty - float(rules["official_penalty_reduction_hard"])),
            "national_anchor_penalty_scale": max(float(rules["national_penalty_floor_hard"]), national_penalty - float(rules["national_penalty_reduction_hard"])),
            "harp_program_penalty_scale": harp_penalty + float(rules["harp_penalty_add_hard"]),
            "linkage_penalty_scale": linkage_penalty + float(rules["linkage_penalty_add_hard"]),
            "fit_steps": int(rules["adaptive_fit_steps"]),
        },
        {
            "label": "adaptive_dx_floor_link_hard_supp",
            "u_to_d_prior": max(float(rules["u_to_d_floor_hard"]), round(u_to_d * float(rules["u_to_d_scale_hard"]), 4)),
            "d_to_a_prior": min(float(rules["d_to_a_ceiling_supp"]), round(d_to_a + float(rules["d_to_a_increment_supp"]), 4)),
            "diagnosed_penalty_scale": diagnosed_penalty + float(rules["diagnosed_penalty_hard"]),
            "official_reference_penalty_scale": max(float(rules["official_penalty_floor_hard"]), official_penalty - float(rules["official_penalty_reduction_hard"])),
            "national_anchor_penalty_scale": max(float(rules["national_penalty_floor_hard"]), national_penalty - float(rules["national_penalty_reduction_hard"])),
            "harp_program_penalty_scale": harp_penalty + float(rules["harp_penalty_add_supp"]),
            "linkage_penalty_scale": linkage_penalty + float(rules["linkage_penalty_add_supp"]),
            "suppression_penalty_scale": float(rules["suppression_penalty_trial"]),
            "fit_steps": int(rules["adaptive_fit_steps"]),
        },
        {
            "label": "adaptive_dx_micro_link_balanced",
            "u_to_d_prior": max(float(rules["u_to_d_floor_balanced"]), round(u_to_d * float(rules["u_to_d_scale_soft"]), 4)),
            "d_to_a_prior": min(float(rules["d_to_a_ceiling_balanced"]), round(d_to_a + float(rules["d_to_a_increment_balanced"]), 4)),
            "diagnosed_penalty_scale": diagnosed_penalty + float(rules["diagnosed_penalty_balanced"]),
            "official_reference_penalty_scale": max(float(rules["official_penalty_floor_soft"]), official_penalty - float(rules["official_penalty_reduction_balanced"])),
            "national_anchor_penalty_scale": max(float(rules["national_penalty_floor_soft"]), national_penalty - float(rules["national_penalty_reduction_balanced"])),
            "harp_program_penalty_scale": harp_penalty + float(rules["harp_penalty_add_balanced"]),
            "linkage_penalty_scale": linkage_penalty + float(rules["linkage_penalty_add_balanced"]),
            "fit_steps": int(rules["adaptive_fit_steps"]),
        },
    ]


def run_phase3_frozen_backtest_tuning(
    *,
    run_id: str,
    plugin_id: str,
    profile: str = RESCUE_V2_PROFILE_ID,
    inference_family: str = RESCUE_INFERENCE_FAMILY,
) -> dict[str, Any]:
    if inference_family != "torch_map":
        raise ValueError("phase3 frozen backtest tuning is only implemented for torch_map")
    if profile not in {RESCUE_PROFILE_ID, RESCUE_V2_PROFILE_ID}:
        raise ValueError("phase3 frozen backtest tuning is only implemented for hiv rescue profiles")

    run_harp_archive_build(run_id=run_id, plugin_id=plugin_id)
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    archive_dir = ctx.run_dir / "harp_archive"
    archive_spec = read_json(archive_dir / "frozen_backtest_spec.json", default={})
    archive_panel = read_json(archive_dir / "historical_harp_panel.json", default={})
    observed_program_panel = read_json(archive_dir / "observed_program_panel.json", default={})
    if not archive_spec.get("ready_for_model_backtest"):
        raise RuntimeError("historical HARP archive is not ready for frozen-history tuning")

    train_years = [int(year) for year in (archive_spec.get("train_years") or [])]
    holdout_years = [int(year) for year in (archive_spec.get("holdout_years") or [])]
    axis_catalogs = read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    normalized_rows = read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    parameter_catalog = read_json(ctx.run_dir / "phase1" / "parameter_catalog.json", default=[])
    standardized_tensor = load_tensor_artifact(ctx.run_dir / "phase1" / "standardized_tensor.npz")
    full_month_axis = list(axis_catalogs.get("month", []))
    month_indices = _month_indices_through_year(full_month_axis, max(train_years))
    filtered_month_axis = [full_month_axis[idx] for idx in month_indices]
    filtered_axis_catalogs = dict(axis_catalogs)
    filtered_axis_catalogs["month"] = filtered_month_axis
    filtered_rows = _filter_rows_through_year(normalized_rows, max(train_years))
    filtered_tensor = np.asarray(standardized_tensor[:, month_indices, :], dtype=np.float32)
    panel_rows = list(archive_panel.get("rows") or [])
    point_month_lookup = _archive_point_month_lookup(list(observed_program_panel.get("rows") or []))
    train_harp_points = _archive_points_from_panel_rows(panel_rows, train_years, full_month_axis, point_month_lookup=point_month_lookup)
    holdout_harp_points = _archive_points_from_panel_rows(panel_rows, holdout_years, full_month_axis, point_month_lookup=point_month_lookup)
    official_points = [
        dict(point)
        for point in OFFICIAL_REFERENCE_POINTS
        if (_month_year(str(point.get("month") or "")) or 0) <= max(train_years)
    ]
    train_last_model_month = filtered_month_axis[-1]
    holdout_eval_months = [str(point.get("month") or "") for point in holdout_harp_points]
    last_train_ordinal = _month_ordinal(train_last_model_month)
    holdout_ordinals = [_month_ordinal(month) for month in holdout_eval_months if _month_ordinal(month) is not None]
    forecast_horizon = max(max(holdout_ordinals) - last_train_ordinal, 1) if (holdout_ordinals and last_train_ordinal is not None) else 12
    backtest_config = {
        "mode": "frozen_history",
        "train_years": train_years,
        "holdout_years": holdout_years,
        "train_last_model_month": train_last_model_month,
        "holdout_eval_months": holdout_eval_months,
        "forecast_horizon": forecast_horizon,
        "train_harp_points": train_harp_points,
        "holdout_harp_points": holdout_harp_points,
    }

    tuning_root = ensure_dir(ctx.run_dir / "phase3_tuning")
    trial_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    carry_forward_target = None
    evaluated_labels: set[str] = set()
    next_trial_index = 1

    def _run_candidates(candidates: list[dict[str, Any]]) -> bool:
        nonlocal best_trial, carry_forward_target, next_trial_index
        for candidate in candidates:
            label = str(candidate["label"])
            if label in evaluated_labels:
                continue
            evaluated_labels.add(label)
            phase_dir_name = f"phase3_tuning/trial_{next_trial_index:02d}_{label}"
            next_trial_index += 1
            calibration_overrides = _trial_calibration_overrides(candidate)
            manifest = run_phase3_rescue_core(
                run_id=run_id,
                plugin_id=plugin_id,
                profile_id=profile,
                requested_inference_family=inference_family,
                phase_dir_name=phase_dir_name,
                axis_catalogs_override=filtered_axis_catalogs,
                normalized_rows_override=filtered_rows,
                parameter_catalog_override=parameter_catalog,
                standardized_tensor_override=filtered_tensor,
                reference_overrides={"official_points": official_points, "harp_points": train_harp_points},
                backtest_config=backtest_config,
                calibration_overrides=calibration_overrides,
            )
            fit_artifact = read_json(Path(manifest["artifact_paths"]["fit_artifact"]), default={})
            frozen_summary = fit_artifact.get("frozen_history_backtest", {})
            model_mae = float(frozen_summary.get("model_mean_absolute_error") or 0.0)
            carry_forward_mae = float(frozen_summary.get("carry_forward_mean_absolute_error") or 0.0)
            carry_forward_target = carry_forward_mae
            trial_row = {
                "trial_label": label,
                "phase_dir": str(ctx.run_dir / phase_dir_name),
                "model_mean_absolute_error": round(model_mae, 6),
                "carry_forward_mean_absolute_error": round(carry_forward_mae, 6),
                "beats_carry_forward": bool(model_mae < carry_forward_mae),
                "calibration_overrides": calibration_overrides,
            }
            trial_rows.append(trial_row)
            if best_trial is None or model_mae < float(best_trial["model_mean_absolute_error"]):
                best_trial = trial_row
            if trial_row["beats_carry_forward"]:
                return True
        return False

    beat_baseline = _run_candidates(_frozen_tuning_candidates())
    if not beat_baseline and best_trial is not None:
        _run_candidates(_adaptive_frozen_tuning_candidates(best_trial))

    if best_trial is None:
        raise RuntimeError("phase3 frozen backtest tuning produced no trial results")

    selected_overrides = dict(best_trial["calibration_overrides"])
    selected_overrides["fit_steps"] = int(dict(_phase3_required_frozen_section("tuning")["adaptive_rules"])["final_fit_steps"])
    final_manifest = run_phase3_rescue_core(
        run_id=run_id,
        plugin_id=plugin_id,
        profile_id=profile,
        requested_inference_family=inference_family,
        phase_dir_name="phase3_tuned_backtest",
        axis_catalogs_override=filtered_axis_catalogs,
        normalized_rows_override=filtered_rows,
        parameter_catalog_override=parameter_catalog,
        standardized_tensor_override=filtered_tensor,
        reference_overrides={"official_points": official_points, "harp_points": train_harp_points},
        backtest_config=backtest_config,
        calibration_overrides=selected_overrides,
    )
    final_fit = read_json(Path(final_manifest["artifact_paths"]["fit_artifact"]), default={})
    final_summary = final_fit.get("frozen_history_backtest", {})
    tuning_summary = {
        "train_years": train_years,
        "holdout_years": holdout_years,
        "trial_count": len(trial_rows),
        "carry_forward_target_mae": round(float(carry_forward_target or 0.0), 6),
        "best_trial": best_trial,
        "final_model_mean_absolute_error": round(float(final_summary.get("model_mean_absolute_error") or 0.0), 6),
        "final_beats_carry_forward": bool(float(final_summary.get("model_mean_absolute_error") or 0.0) < float(final_summary.get("carry_forward_mean_absolute_error") or 0.0)),
        "selected_calibration_overrides": selected_overrides,
        "trial_rows": trial_rows,
    }
    write_json(tuning_root / "frozen_history_tuning_summary.json", tuning_summary)
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "tuning_summary_path": str(tuning_root / "frozen_history_tuning_summary.json"),
        "selected_phase_dir": str(ctx.run_dir / "phase3_tuned_backtest"),
        "summary": tuning_summary,
    }
