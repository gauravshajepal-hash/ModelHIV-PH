from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None
    Matern = None
    WhiteKernel = None


_TARGET_NAMES = [
    "diagnosed_stock",
    "art_stock",
    "documented_suppression",
    "testing_coverage",
]


def _representation_type(row: dict[str, Any], factor_row: dict[str, Any]) -> str:
    factor_class = str(factor_row.get("factor_class") or row.get("factor_class") or "")
    member_count = int(factor_row.get("member_count") or row.get("member_count") or 0)
    if factor_class == "network_feature":
        return "network"
    if member_count <= 1:
        return "unclumped"
    return "clumped"


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    return float(np.mean(np.abs(left - right)))


def _smape(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    denom = np.abs(left) + np.abs(right) + 1e-6
    return float(np.mean(2.0 * np.abs(left - right) / denom))


def _time_holdout_columns(month_count: int, *, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if month_count <= 1:
        return np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32)
    holdout_fraction = float(cfg["holdout_fraction"])
    min_holdout = max(1, int(cfg["holdout_min_months"]))
    holdout_months = max(min_holdout, int(np.ceil(month_count * holdout_fraction)))
    holdout_months = min(max(1, month_count - 1), holdout_months)
    split = month_count - holdout_months
    train_idx = np.arange(0, split, dtype=np.int32)
    holdout_idx = np.arange(split, month_count, dtype=np.int32)
    return train_idx, holdout_idx


def _multitarget_holdout_objective(
    *,
    factor_tensor: np.ndarray,
    factor_catalog: list[dict[str, Any]],
    selected_factor_ids: list[str],
    observation_targets: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, float]:
    if factor_tensor.ndim != 3 or not selected_factor_ids:
        return {
            "objective": -1.0,
            "mean_mae_improvement": -1.0,
            "mean_smape_improvement": -1.0,
            "calibration_score": 0.0,
            "selection_penalty": 1.0,
        }
    factor_index = {str(row.get("factor_id") or ""): idx for idx, row in enumerate(factor_catalog)}
    selected_indices = [factor_index[factor_id] for factor_id in selected_factor_ids if factor_id in factor_index]
    if not selected_indices:
        return {
            "objective": -1.0,
            "mean_mae_improvement": -1.0,
            "mean_smape_improvement": -1.0,
            "calibration_score": 0.0,
            "selection_penalty": 1.0,
        }
    month_count = int(factor_tensor.shape[1])
    train_idx, holdout_idx = _time_holdout_columns(month_count, cfg=cfg)
    if len(train_idx) == 0 or len(holdout_idx) == 0:
        return {
            "objective": -1.0,
            "mean_mae_improvement": -1.0,
            "mean_smape_improvement": -1.0,
            "calibration_score": 0.0,
            "selection_penalty": 1.0,
        }
    x_train = factor_tensor[:, train_idx][:, :, selected_indices].reshape(-1, len(selected_indices)).astype(np.float64)
    x_holdout = factor_tensor[:, holdout_idx][:, :, selected_indices].reshape(-1, len(selected_indices)).astype(np.float64)
    target_stack = np.stack([observation_targets[name] for name in _TARGET_NAMES], axis=-1).astype(np.float64)
    y_train = target_stack[:, train_idx, :].reshape(-1, len(_TARGET_NAMES))
    y_holdout = target_stack[:, holdout_idx, :].reshape(-1, len(_TARGET_NAMES))
    design_train = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1)
    coeffs, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    design_holdout = np.concatenate([np.ones((x_holdout.shape[0], 1), dtype=np.float64), x_holdout], axis=1)
    pred_holdout = design_holdout @ coeffs
    baseline = np.repeat(target_stack[:, [train_idx[-1]], :], len(holdout_idx), axis=1).reshape(-1, len(_TARGET_NAMES))
    train_pred = design_train @ coeffs
    residual_scale = np.maximum(np.std(y_train - train_pred, axis=0), 1e-6)
    target_coverage = float(cfg["target_coverage"])
    coverage = np.mean(np.abs(y_holdout - pred_holdout) <= residual_scale.reshape(1, -1), axis=0)
    calibration = float(np.mean(np.clip(1.0 - np.abs(coverage - target_coverage) / max(target_coverage, 1e-6), 0.0, 1.0)))
    mae_improvements = []
    smape_improvements = []
    for idx in range(len(_TARGET_NAMES)):
        model_mae = _mae(y_holdout[:, idx], pred_holdout[:, idx])
        base_mae = _mae(y_holdout[:, idx], baseline[:, idx])
        model_smape = _smape(y_holdout[:, idx], pred_holdout[:, idx])
        base_smape = _smape(y_holdout[:, idx], baseline[:, idx])
        mae_improvements.append((base_mae - model_mae) / max(base_mae, 1e-6))
        smape_improvements.append((base_smape - model_smape) / max(base_smape, 1e-6))
    mean_mae_improvement = float(np.mean(mae_improvements))
    mean_smape_improvement = float(np.mean(smape_improvements))
    selection_penalty = float(min(1.0, len(selected_indices) / max(1.0, float(cfg.get("selection_penalty_denominator", 12.0)))))
    objective = float(
        0.45 * mean_mae_improvement
        + 0.35 * mean_smape_improvement
        + 0.20 * calibration
        - float(cfg.get("selection_penalty_weight", 0.05)) * selection_penalty
    )
    return {
        "objective": round(objective, 6),
        "mean_mae_improvement": round(mean_mae_improvement, 6),
        "mean_smape_improvement": round(mean_smape_improvement, 6),
        "calibration_score": round(calibration, 6),
        "selection_penalty": round(selection_penalty, 6),
    }


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - best) / sigma_safe
    pdf = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    ei = (mu - best) * cdf + sigma_safe * pdf
    ei[sigma <= 1e-9] = 0.0
    return ei


def _score_factor_rows(
    rows: list[dict[str, Any]],
    *,
    score_params: dict[str, float],
    factor_catalog: list[dict[str, Any]],
    primary_per_block: int,
    secondary_per_block: int,
    representation_mix: dict[str, float] | None = None,
    mix_bonus_scale: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    factor_catalog_by_id = {str(row.get("factor_id") or ""): row for row in factor_catalog}
    scored_rows: list[dict[str, Any]] = []
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_mix = dict(representation_mix or {"unclumped": 1.0 / 3.0, "clumped": 1.0 / 3.0, "network": 1.0 / 3.0})
    for row in rows:
        factor_id = str(row.get("factor_id") or "")
        factor_row = factor_catalog_by_id.get(factor_id, {})
        representation_type = _representation_type(row, factor_row)
        mix_weight = float(normalized_mix.get(representation_type, 1.0 / 3.0))
        representation_bonus = float(mix_bonus_scale * (mix_weight - (1.0 / 3.0)))
        score = float(
            score_params["mae_improvement"] * max(0.0, float(row.get("holdout_mae_improvement", 0.0)))
            + score_params["smape_improvement"] * max(0.0, float(row.get("holdout_smape_improvement", 0.0)))
            + score_params["calibration"] * float(row.get("calibration_score", 0.0))
            + score_params["predictive_gain"] * float(row.get("predictive_gain", 0.0))
            + score_params["subnational_gain"] * float(row.get("subnational_anomaly_gain", 0.0))
            + score_params["stability"] * float(row.get("stability_score", 0.0))
            - score_params["sparsity_penalty"] * float(row.get("sparsity_penalty", 0.0))
            - score_params["resampling_penalty"] * float(row.get("resampling_stability_penalty", 0.0))
            + representation_bonus
        )
        scored = dict(row)
        scored["survival_score"] = round(float(np.clip(score, 0.0, 1.0)), 6)
        scored["representation_type"] = representation_type
        scored["representation_bonus"] = round(representation_bonus, 6)
        scored_rows.append(scored)
        by_block[str(row.get("block_name") or "mixed")].append(scored)
    promotion_pool: list[dict[str, Any]] = []
    selected_factor_ids: list[str] = []
    for block_rows in by_block.values():
        block_rows.sort(
            key=lambda item: (
                bool(item.get("hard_checks_passed")),
                float(item.get("survival_score", 0.0)),
                float(item.get("holdout_mae_improvement", 0.0)),
                float(item.get("holdout_smape_improvement", 0.0)),
                -float(item.get("sparsity_penalty", 0.0)),
            ),
            reverse=True,
        )
        for rank, row in enumerate(block_rows):
            if not bool(row.get("hard_checks_passed")):
                survival_class = "discarded"
            elif rank < primary_per_block:
                survival_class = "survivor_primary"
            elif rank < primary_per_block + secondary_per_block:
                survival_class = "survivor_secondary"
            else:
                survival_class = "reserve"
            row["survival_rank_in_block"] = int(rank + 1)
            row["survival_class"] = survival_class
            factor_id = str(row.get("factor_id") or "")
            if survival_class in {"survivor_primary", "survivor_secondary"}:
                selected_factor_ids.append(factor_id)
            promotion_pool.append(
                row
                | {
                    "promotion_class": survival_class,
                    "factor_class": factor_catalog_by_id.get(factor_id, {}).get("factor_class", "mesoscopic_factor"),
                    "transition_hooks": factor_catalog_by_id.get(factor_id, {}).get("transition_hooks", []),
                    "representation_type": row.get("representation_type", ""),
                    "representation_bonus": row.get("representation_bonus", 0.0),
                }
            )
    scored_rows.sort(key=lambda item: (str(item.get("block_name") or ""), int(item.get("survival_rank_in_block", 9999))))
    promotion_pool.sort(key=lambda item: (str(item.get("block_name") or ""), int(item.get("survival_rank_in_block", 9999))))
    return scored_rows, promotion_pool, selected_factor_ids


def optimize_survival_tournament(
    *,
    stability_rows: list[dict[str, Any]],
    factor_catalog: list[dict[str, Any]],
    factor_tensor: np.ndarray,
    observation_targets: dict[str, np.ndarray],
    stability_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not stability_rows:
        return stability_rows, [], {"enabled": False, "reason": "no_stability_rows"}
    default_score_params = {
        "mae_improvement": float(stability_cfg["tournament_weights"]["mae_improvement"]),
        "smape_improvement": float(stability_cfg["tournament_weights"]["smape_improvement"]),
        "calibration": float(stability_cfg["tournament_weights"]["calibration"]),
        "predictive_gain": float(stability_cfg["tournament_weights"]["predictive_gain"]),
        "subnational_gain": float(stability_cfg["tournament_weights"]["subnational_gain"]),
        "stability": float(stability_cfg["tournament_weights"]["stability"]),
        "sparsity_penalty": float(stability_cfg["penalty_weights"]["sparsity"]),
        "resampling_penalty": float(stability_cfg["penalty_weights"]["resampling_instability"]),
    }
    bayes_cfg = dict(stability_cfg.get("bayesian_optimization", {}) or {})
    selection_penalty_denominator = float(
        bayes_cfg.get("selection_penalty_denominator", stability_cfg.get("selection_penalty_denominator", 12.0))
    )
    selection_penalty_weight = float(
        bayes_cfg.get("selection_penalty_weight", stability_cfg.get("selection_penalty_weight", 0.05))
    )
    objective_cfg = dict(stability_cfg)
    objective_cfg["selection_penalty_denominator"] = selection_penalty_denominator
    objective_cfg["selection_penalty_weight"] = selection_penalty_weight

    default_rows, default_pool, default_selected = _score_factor_rows(
        stability_rows,
        score_params=default_score_params,
        factor_catalog=factor_catalog,
        primary_per_block=max(1, int(stability_cfg["primary_survivors_per_block"])),
        secondary_per_block=max(0, int(stability_cfg["secondary_survivors_per_block"])),
        representation_mix={"unclumped": 1.0 / 3.0, "clumped": 1.0 / 3.0, "network": 1.0 / 3.0},
        mix_bonus_scale=0.0,
    )
    default_objective_payload = _multitarget_holdout_objective(
        factor_tensor=factor_tensor,
        factor_catalog=factor_catalog,
        selected_factor_ids=default_selected,
        observation_targets=observation_targets,
        cfg=objective_cfg,
    )
    if GaussianProcessRegressor is None or Matern is None or WhiteKernel is None:
        return default_rows, default_pool, {
            "enabled": False,
            "reason": "sklearn_unavailable",
            "selected_factor_count": len(default_selected),
            "default_objective": dict(default_objective_payload),
            "best_objective": dict(default_objective_payload),
            "best_score_params": dict(default_score_params),
            "improved_over_default": False,
        }

    if not bool(bayes_cfg.get("enabled", True)):
        return default_rows, default_pool, {
            "enabled": False,
            "reason": "disabled",
            "selected_factor_count": len(default_selected),
            "default_objective": dict(default_objective_payload),
            "best_objective": dict(default_objective_payload),
            "best_score_params": dict(default_score_params),
            "improved_over_default": False,
        }
    rng = np.random.default_rng(int(bayes_cfg.get("random_seed", 17)))
    seed_trials = max(2, int(bayes_cfg.get("seed_trials", 6)))
    total_trials = max(seed_trials, int(bayes_cfg.get("trials", 12)))
    candidate_pool = max(8, int(bayes_cfg.get("candidate_pool", 64)))
    default_primary_per_block = max(1, int(stability_cfg["primary_survivors_per_block"]))
    default_secondary_per_block = max(0, int(stability_cfg["secondary_survivors_per_block"]))
    primary_range = tuple(bayes_cfg.get("primary_survivors_range", [default_primary_per_block, max(default_primary_per_block, default_primary_per_block + 2)]))
    secondary_range = tuple(bayes_cfg.get("secondary_survivors_range", [default_secondary_per_block, max(default_secondary_per_block, default_secondary_per_block + 2)]))
    primary_min = max(1, int(primary_range[0]))
    primary_max = max(primary_min, int(primary_range[1]))
    secondary_min = max(0, int(secondary_range[0]))
    secondary_max = max(secondary_min, int(secondary_range[1]))
    mix_bonus_scale = float(bayes_cfg.get("representation_mix_bonus_scale", 0.12))

    default_vector = np.asarray(
        [
            float(stability_cfg["tournament_weights"]["mae_improvement"]),
            float(stability_cfg["tournament_weights"]["smape_improvement"]),
            float(stability_cfg["tournament_weights"]["calibration"]),
            float(stability_cfg["tournament_weights"]["predictive_gain"]),
            float(stability_cfg["tournament_weights"]["subnational_gain"]),
            float(stability_cfg["tournament_weights"]["stability"]),
            float(stability_cfg["penalty_weights"]["sparsity"]),
            float(stability_cfg["penalty_weights"]["resampling_instability"]),
            float(default_primary_per_block),
            float(default_secondary_per_block),
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
        dtype=np.float64,
    )

    def _vector_to_params(vector: np.ndarray) -> dict[str, Any]:
        reward = np.clip(np.asarray(vector[:6], dtype=np.float64), 1e-6, None)
        reward = reward / max(float(reward.sum()), 1e-6)
        penalties = np.clip(np.asarray(vector[6:8], dtype=np.float64), 0.0, 0.25)
        primary_raw = float(vector[8])
        secondary_raw = float(vector[9])
        primary_per_block = int(np.clip(np.rint(primary_raw), primary_min, primary_max))
        secondary_per_block = int(np.clip(np.rint(secondary_raw), secondary_min, secondary_max))
        mix = np.clip(np.asarray(vector[10:13], dtype=np.float64), 1e-6, None)
        mix = mix / max(float(mix.sum()), 1e-6)
        return {
            "mae_improvement": float(reward[0]),
            "smape_improvement": float(reward[1]),
            "calibration": float(reward[2]),
            "predictive_gain": float(reward[3]),
            "subnational_gain": float(reward[4]),
            "stability": float(reward[5]),
            "sparsity_penalty": float(penalties[0]),
            "resampling_penalty": float(penalties[1]),
            "primary_per_block": primary_per_block,
            "secondary_per_block": secondary_per_block,
            "representation_mix": {
                "unclumped": float(mix[0]),
                "clumped": float(mix[1]),
                "network": float(mix[2]),
            },
        }

    def _evaluate(vector: np.ndarray) -> tuple[float, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        params = _vector_to_params(vector)
        scored_rows, promotion_pool, selected_factor_ids = _score_factor_rows(
            stability_rows,
            score_params=params,
            factor_catalog=factor_catalog,
            primary_per_block=int(params["primary_per_block"]),
            secondary_per_block=int(params["secondary_per_block"]),
            representation_mix=dict(params["representation_mix"]),
            mix_bonus_scale=mix_bonus_scale,
        )
        objective = _multitarget_holdout_objective(
            factor_tensor=factor_tensor,
            factor_catalog=factor_catalog,
            selected_factor_ids=selected_factor_ids,
            observation_targets=observation_targets,
            cfg=objective_cfg,
        )
        objective_value = float(objective["objective"])
        details = {
            "params": params,
            "primary_per_block": int(params["primary_per_block"]),
            "secondary_per_block": int(params["secondary_per_block"]),
            "representation_mix": dict(params["representation_mix"]),
            "selected_factor_count": len(selected_factor_ids),
            "selected_factor_ids": selected_factor_ids,
            "objective": objective,
        }
        return objective_value, details, scored_rows, promotion_pool

    observed_x: list[np.ndarray] = []
    observed_y: list[float] = []
    trial_rows: list[dict[str, Any]] = []
    best_payload: tuple[float, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]] | None = None

    for trial_idx in range(total_trials):
        if trial_idx == 0:
            vector = default_vector
        elif trial_idx < seed_trials or len(observed_x) < 3:
            reward = rng.random(6)
            penalties = rng.uniform(0.0, 0.25, size=2)
            budgets = np.asarray(
                [
                    rng.integers(primary_min, primary_max + 1),
                    rng.integers(secondary_min, secondary_max + 1),
                ],
                dtype=np.float64,
            )
            mix = rng.random(3)
            vector = np.concatenate([reward, penalties, budgets, mix], axis=0).astype(np.float64)
        else:
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=int(bayes_cfg.get("random_seed", 17)))
            x_train = np.asarray(observed_x, dtype=np.float64)
            y_train = np.asarray(observed_y, dtype=np.float64)
            gp.fit(x_train, y_train)
            reward = rng.random((candidate_pool, 6))
            penalties = rng.uniform(0.0, 0.25, size=(candidate_pool, 2))
            budgets = np.column_stack(
                [
                    rng.integers(primary_min, primary_max + 1, size=candidate_pool),
                    rng.integers(secondary_min, secondary_max + 1, size=candidate_pool),
                ]
            ).astype(np.float64)
            mix = rng.random((candidate_pool, 3))
            candidates = np.concatenate([reward, penalties, budgets, mix], axis=1).astype(np.float64)
            mu, sigma = gp.predict(candidates, return_std=True)
            ei = _expected_improvement(mu, sigma, float(np.max(y_train)))
            vector = candidates[int(np.argmax(ei))]
        objective_value, details, scored_rows, promotion_pool = _evaluate(vector)
        observed_x.append(np.asarray(vector, dtype=np.float64))
        observed_y.append(objective_value)
        trial_rows.append(
            {
                "trial_index": trial_idx,
                "objective": round(objective_value, 6),
                "selected_factor_count": int(details["selected_factor_count"]),
                "selected_factor_ids": list(details["selected_factor_ids"]),
                "score_params": dict(details["params"]),
                "primary_per_block": int(details["primary_per_block"]),
                "secondary_per_block": int(details["secondary_per_block"]),
                "representation_mix": dict(details["representation_mix"]),
                "holdout_objective": dict(details["objective"]),
            }
        )
        if best_payload is None or objective_value > best_payload[0]:
            best_payload = (objective_value, details, scored_rows, promotion_pool)

    assert best_payload is not None
    best_objective, best_details, best_rows, best_pool = best_payload
    default_objective_value = float(default_objective_payload["objective"])
    report = {
        "enabled": True,
        "trial_count": len(trial_rows),
        "default_objective": round(default_objective_value, 6),
        "default_holdout_objective": dict(default_objective_payload),
        "best_objective": round(float(best_objective), 6),
        "improved_over_default": bool(best_objective > default_objective_value + 1e-6),
        "best_score_params": dict(best_details["params"]),
        "best_primary_per_block": int(best_details["primary_per_block"]),
        "best_secondary_per_block": int(best_details["secondary_per_block"]),
        "best_representation_mix": dict(best_details["representation_mix"]),
        "selected_factor_count": int(best_details["selected_factor_count"]),
        "selected_factor_ids": list(best_details["selected_factor_ids"]),
        "best_holdout_objective": dict(best_details["objective"]),
        "trial_rows": trial_rows,
        "selection_penalty_denominator": round(selection_penalty_denominator, 6),
        "selection_penalty_weight": round(selection_penalty_weight, 6),
        "representation_mix_bonus_scale": round(mix_bonus_scale, 6),
        "search_space": {
            "primary_survivors_range": [primary_min, primary_max],
            "secondary_survivors_range": [secondary_min, secondary_max],
            "representation_mix_labels": ["unclumped", "clumped", "network"],
        },
    }
    return best_rows, best_pool, report
