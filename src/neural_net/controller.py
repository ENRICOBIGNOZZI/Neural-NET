from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from .feature_value import constant_speed_feature_value_proxy
from .spectral import (
    accessibility_from_features,
    effective_rank,
    grassmann_distance,
    target_greedy_neuron_order,
)


@dataclass
class ContractionConfig:
    min_step: int = 80
    checkpoint_interval: int = 20
    min_rank: int = 8
    rank_stride: int = 4
    max_accessibility_loss: float = 0.01
    max_probe_risk_increase: float = 0.01
    max_subspace_speed: float = 0.004
    speed_rank: int = 8
    persistence: int = 2
    ridge: float = 1e-3
    future_horizon_steps: int = 40


@dataclass
class ContractionDecision:
    step: int
    should_contract: bool
    proposed_rank: int
    keep_indices: np.ndarray | None
    full_accessibility: float
    proposed_accessibility: float
    full_probe_mse: float
    proposed_probe_mse: float
    subspace_speed: float
    future_accessibility_gain_proxy: float
    effective_rank: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class FeatureSpanContractionController:
    """Conservative last-hidden-layer contraction controller.

    Training data construct a target-aware neuron ordering; the independent probe only
    validates current target accessibility/risk and measures subspace motion. Expensive
    subset searches are skipped while the representation is still moving quickly.

    ``future_accessibility_gain_proxy`` is logged prospectively from the feature-value
    theorem using a constant-speed extrapolation.  It is intentionally *not* yet a gate:
    recent subspace speed is not a certified upper bound on future speed.  This lets the
    experiments test whether the theory-derived quantity forecasts safe onset before it is
    promoted into the controller.
    """

    def __init__(self, config: ContractionConfig):
        self.config = config
        self._previous_probe_features: np.ndarray | None = None
        self._stable_count = 0

    @staticmethod
    def _ridge_probe_mse(train_h, train_y, probe_h, probe_y, alpha: float) -> float:
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(train_h, train_y)
        pred = reg.predict(probe_h)
        return float(np.mean((probe_y - pred) ** 2))

    def _candidate_ranks(self, width: int) -> list[int]:
        cfg = self.config
        ranks = list(range(max(1, cfg.min_rank), width, max(1, cfg.rank_stride)))
        ranks.append(width)
        return sorted(set(ranks))

    def evaluate(self, step: int, train_h, train_y, probe_h, probe_y) -> ContractionDecision:
        cfg = self.config
        train_h = np.asarray(train_h, dtype=float)
        probe_h = np.asarray(probe_h, dtype=float)
        train_y = np.asarray(train_y, dtype=float).reshape(-1)
        probe_y = np.asarray(probe_y, dtype=float).reshape(-1)
        width = probe_h.shape[1]

        full_q = accessibility_from_features(probe_h, probe_y)
        full_mse = self._ridge_probe_mse(train_h, train_y, probe_h, probe_y, cfg.ridge)
        reff = effective_rank(probe_h)

        speed = float("inf")
        if self._previous_probe_features is not None and self._previous_probe_features.shape[1] == width:
            angle = grassmann_distance(
                self._previous_probe_features,
                probe_h,
                rank=min(cfg.speed_rank, width),
            )
            speed = angle / cfg.checkpoint_interval
        self._previous_probe_features = probe_h.copy()

        future_proxy = constant_speed_feature_value_proxy(
            full_q,
            speed,
            cfg.future_horizon_steps,
        )

        reasons = []
        ready_to_search = step >= cfg.min_step and np.isfinite(speed) and speed <= cfg.max_subspace_speed
        if step < cfg.min_step:
            reasons.append("discovery warm-up not finished")
        if not np.isfinite(speed) or speed > cfg.max_subspace_speed:
            reasons.append("resolved representation subspace is still moving")

        chosen_rank = width
        chosen_q = full_q
        chosen_mse = full_mse
        chosen_idx = np.arange(width)

        if ready_to_search:
            order = target_greedy_neuron_order(train_h, train_y)
            for r in self._candidate_ranks(width):
                idx = order[:r]
                q_r = accessibility_from_features(probe_h[:, idx], probe_y)
                mse_r = self._ridge_probe_mse(
                    train_h[:, idx], train_y, probe_h[:, idx], probe_y, cfg.ridge
                )
                access_loss = max(0.0, full_q - q_r)
                risk_increase = max(0.0, mse_r - full_mse)
                if access_loss <= cfg.max_accessibility_loss and risk_increase <= cfg.max_probe_risk_increase:
                    chosen_rank, chosen_q, chosen_mse, chosen_idx = r, q_r, mse_r, idx.copy()
                    break

        current_preserved = chosen_rank < width
        if ready_to_search and not current_preserved:
            reasons.append("no smaller neuron subset preserves current probe geometry")

        if ready_to_search and current_preserved:
            self._stable_count += 1
        else:
            self._stable_count = 0
        if self._stable_count < cfg.persistence:
            reasons.append("persistence requirement not met")

        should = current_preserved and self._stable_count >= cfg.persistence
        return ContractionDecision(
            step=step,
            should_contract=should,
            proposed_rank=int(chosen_rank),
            keep_indices=chosen_idx if should else None,
            full_accessibility=float(full_q),
            proposed_accessibility=float(chosen_q),
            full_probe_mse=float(full_mse),
            proposed_probe_mse=float(chosen_mse),
            subspace_speed=float(speed),
            future_accessibility_gain_proxy=float(future_proxy),
            effective_rank=float(reff),
            reasons=tuple(reasons),
        )
