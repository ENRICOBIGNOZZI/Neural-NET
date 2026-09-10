from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge

from .feature_value import constant_speed_feature_value_proxy
from .spectral import (
    accessibility_from_features,
    effective_rank,
    target_angle_from_accessibility,
    target_greedy_neuron_order,
)


@dataclass
class TargetValueConfig:
    """Proxy configuration for the second controller generation.

    This is deliberately not called a certificate.  The finite-horizon target-angle
    extrapolation uses observed recent progress, which need not upper-bound future
    acceleration.  The point of the pilot is to test whether the value object is a
    better onset statistic than total fixed-rank Grassmann speed.
    """

    min_step: int = 60
    checkpoint_interval: int = 20
    resolved_rank: int = 5
    future_horizon_steps: int = 40
    max_future_accessibility_gain: float = 0.005
    min_rank: int = 8
    rank_stride: int = 4
    max_accessibility_loss: float = 0.01
    max_probe_risk_increase: float = 0.01
    max_fraction_removed: float = 0.25
    persistence: int = 2
    ridge: float = 1e-3


@dataclass
class TargetValueDecision:
    step: int
    should_contract: bool
    proposed_rank: int
    keep_indices: np.ndarray | None
    resolved_accessibility: float
    proposed_resolved_accessibility: float
    full_probe_mse: float
    proposed_probe_mse: float
    target_angle_speed: float
    future_accessibility_gain_proxy: float
    saturation_ceiling: float
    effective_rank: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class TargetValueContractionController:
    """One-shot, target-value-oriented final-hidden-layer controller.

    Compared with :class:`FeatureSpanContractionController`, this version removes the
    hard gate on total principal-angle motion.  It monitors target accessibility in a
    predeclared leading singular subspace and the *target angle* itself.  Structural
    candidates are still constructed on training data and validated on the probe.
    """

    def __init__(self, config: TargetValueConfig):
        self.config = config
        self._previous_target_angle: float | None = None
        self._stable_count = 0

    @staticmethod
    def _ridge_probe_mse(train_h, train_y, probe_h, probe_y, alpha: float) -> float:
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(train_h, train_y)
        pred = reg.predict(probe_h)
        return float(np.mean((probe_y - pred) ** 2))

    def _candidate_ranks(self, width: int) -> list[int]:
        cfg = self.config
        # A single event cannot remove more than ``max_fraction_removed``.  This is a
        # conservative stand-in for the still-open nonlinear structural-transfer bound.
        event_floor = int(np.ceil(width * (1.0 - cfg.max_fraction_removed)))
        floor = max(1, cfg.min_rank, event_floor)
        ranks = list(range(floor, width, max(1, cfg.rank_stride)))
        ranks.append(width)
        return sorted(set(ranks))

    def evaluate(self, step: int, train_h, train_y, probe_h, probe_y) -> TargetValueDecision:
        cfg = self.config
        train_h = np.asarray(train_h, dtype=float)
        probe_h = np.asarray(probe_h, dtype=float)
        train_y = np.asarray(train_y, dtype=float).reshape(-1)
        probe_y = np.asarray(probe_y, dtype=float).reshape(-1)
        width = probe_h.shape[1]

        resolved_rank = min(cfg.resolved_rank, width)
        q = accessibility_from_features(probe_h, probe_y, rank=resolved_rank)
        psi = target_angle_from_accessibility(q)
        reff = effective_rank(probe_h)
        full_mse = self._ridge_probe_mse(train_h, train_y, probe_h, probe_y, cfg.ridge)

        target_speed = float("inf")
        if self._previous_target_angle is not None:
            target_speed = abs(psi - self._previous_target_angle) / cfg.checkpoint_interval
        self._previous_target_angle = psi

        future_proxy = constant_speed_feature_value_proxy(
            q,
            target_speed,
            cfg.future_horizon_steps,
        )
        saturation = max(0.0, 1.0 - q)

        reasons: list[str] = []
        ready = step >= cfg.min_step and np.isfinite(future_proxy)
        if step < cfg.min_step:
            reasons.append("discovery warm-up not finished")
        if not np.isfinite(future_proxy):
            reasons.append("target-value velocity unavailable")
        elif future_proxy > cfg.max_future_accessibility_gain:
            reasons.append("prospective target accessibility still has material value")

        value_gate = ready and future_proxy <= cfg.max_future_accessibility_gain

        chosen_rank = width
        chosen_q = q
        chosen_mse = full_mse
        chosen_idx = np.arange(width)

        if value_gate:
            order = target_greedy_neuron_order(train_h, train_y)
            for r in self._candidate_ranks(width):
                idx = order[:r]
                q_r = accessibility_from_features(
                    probe_h[:, idx], probe_y, rank=min(resolved_rank, r)
                )
                mse_r = self._ridge_probe_mse(
                    train_h[:, idx], train_y, probe_h[:, idx], probe_y, cfg.ridge
                )
                access_loss = max(0.0, q - q_r)
                risk_increase = max(0.0, mse_r - full_mse)
                if (
                    access_loss <= cfg.max_accessibility_loss
                    and risk_increase <= cfg.max_probe_risk_increase
                ):
                    chosen_rank, chosen_q, chosen_mse, chosen_idx = r, q_r, mse_r, idx.copy()
                    break

        structurally_valid = chosen_rank < width
        if value_gate and not structurally_valid:
            reasons.append("no smaller neuron subset preserves current probe value")

        if value_gate and structurally_valid:
            self._stable_count += 1
        else:
            self._stable_count = 0
        if self._stable_count < cfg.persistence:
            reasons.append("persistence requirement not met")

        should = structurally_valid and self._stable_count >= cfg.persistence
        return TargetValueDecision(
            step=step,
            should_contract=should,
            proposed_rank=int(chosen_rank),
            keep_indices=chosen_idx if should else None,
            resolved_accessibility=float(q),
            proposed_resolved_accessibility=float(chosen_q),
            full_probe_mse=float(full_mse),
            proposed_probe_mse=float(chosen_mse),
            target_angle_speed=float(target_speed),
            future_accessibility_gain_proxy=float(future_proxy),
            saturation_ceiling=float(saturation),
            effective_rank=float(reff),
            reasons=tuple(reasons),
        )
