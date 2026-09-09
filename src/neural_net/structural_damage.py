from __future__ import annotations

import math

import numpy as np


def paired_damage_statistics(
    reduced_losses,
    dense_losses,
) -> tuple[float, float, int]:
    """Mean and unbiased variance of paired structural loss differences."""
    reduced = np.asarray(reduced_losses, dtype=float).reshape(-1)
    dense = np.asarray(dense_losses, dtype=float).reshape(-1)
    if reduced.shape != dense.shape:
        raise ValueError("reduced_losses and dense_losses must have the same shape")
    if len(reduced) < 2:
        raise ValueError("at least two paired examples are required")
    z = reduced - dense
    return float(np.mean(z)), float(np.var(z, ddof=1)), int(len(z))


def paired_empirical_bernstein_ucb(
    reduced_losses,
    dense_losses,
    *,
    loss_bound: float,
    delta: float = 0.05,
    num_candidates: int = 1,
) -> float:
    """Uniform upper confidence bound on population structural damage.

    Both individual losses must lie in ``[0, loss_bound]``.  The paired difference
    therefore lies in ``[-loss_bound, loss_bound]``.  Applying the empirical Bernstein
    inequality after shifting/scaling to ``[0, 1]`` gives

        D <= D_hat + sqrt(2 v_hat log(2M/delta)/n)
             + 14 B log(2M/delta)/(3(n-1)).

    The union-bound factor ``M`` permits simultaneous evaluation of a finite candidate
    family constructed independently of the probe.
    """
    b = float(loss_bound)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    m = int(num_candidates)
    if m < 1:
        raise ValueError("num_candidates must be positive")

    reduced = np.asarray(reduced_losses, dtype=float).reshape(-1)
    dense = np.asarray(dense_losses, dtype=float).reshape(-1)
    if np.any(reduced < 0) or np.any(reduced > b) or np.any(dense < 0) or np.any(dense > b):
        raise ValueError("all losses must lie in [0, loss_bound]")

    mean, var, n = paired_damage_statistics(reduced, dense)
    logterm = math.log(2.0 * m / delta)
    radius = math.sqrt(2.0 * var * logterm / n)
    radius += 14.0 * b * logterm / (3.0 * (n - 1))
    return float(mean + radius)


def clipped_squared_losses(prediction, target, *, loss_bound: float) -> np.ndarray:
    """Per-example squared losses clipped to a predeclared bound."""
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError("prediction and target must have the same shape")
    b = float(loss_bound)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    return np.minimum((pred - y) ** 2, b)


def damage_is_certified(
    reduced_losses,
    dense_losses,
    *,
    risk_budget: float,
    loss_bound: float,
    delta: float = 0.05,
    num_candidates: int = 1,
) -> tuple[bool, float]:
    """Return whether the paired structural-damage UCB is inside a risk budget."""
    ucb = paired_empirical_bernstein_ucb(
        reduced_losses,
        dense_losses,
        loss_bound=loss_bound,
        delta=delta,
        num_candidates=num_candidates,
    )
    return bool(ucb <= float(risk_budget)), float(ucb)
