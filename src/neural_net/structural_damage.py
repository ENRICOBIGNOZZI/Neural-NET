from __future__ import annotations

import math

import numpy as np


def _paired_arrays(reduced_losses, dense_losses) -> tuple[np.ndarray, np.ndarray]:
    reduced = np.asarray(reduced_losses, dtype=float).reshape(-1)
    dense = np.asarray(dense_losses, dtype=float).reshape(-1)
    if reduced.shape != dense.shape:
        raise ValueError("reduced_losses and dense_losses must have the same shape")
    if len(reduced) < 2:
        raise ValueError("at least two paired examples are required")
    return reduced, dense


def paired_damage_statistics(
    reduced_losses,
    dense_losses,
) -> tuple[float, float, int]:
    """Mean and unbiased variance of paired structural loss differences."""
    reduced, dense = _paired_arrays(reduced_losses, dense_losses)
    z = reduced - dense
    return float(np.mean(z)), float(np.var(z, ddof=1)), int(len(z))


def _bounded_empirical_bernstein_ucb(
    values,
    *,
    absolute_bound: float,
    delta: float,
    num_candidates: int,
) -> float:
    """Empirical-Bernstein UCB for variables known to lie in [-R, R]."""
    z = np.asarray(values, dtype=float).reshape(-1)
    if len(z) < 2:
        raise ValueError("at least two observations are required")
    r = float(absolute_bound)
    if r <= 0:
        raise ValueError("absolute_bound must be positive")
    if np.any(np.abs(z) > r + 1e-12):
        raise ValueError("observed values exceed the declared absolute bound")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    m = int(num_candidates)
    if m < 1:
        raise ValueError("num_candidates must be positive")

    mean = float(np.mean(z))
    var = float(np.var(z, ddof=1))
    n = int(len(z))
    logterm = math.log(2.0 * m / delta)
    radius = math.sqrt(2.0 * var * logterm / n)
    radius += 14.0 * r * logterm / (3.0 * (n - 1))
    return float(mean + radius)


def paired_empirical_bernstein_ucb(
    reduced_losses,
    dense_losses,
    *,
    loss_bound: float,
    delta: float = 0.05,
    num_candidates: int = 1,
    difference_bound: float | None = None,
) -> float:
    """Uniform UCB on population structural damage.

    Individual losses must lie in ``[0, loss_bound]``. Hence their paired difference is
    always in ``[-loss_bound, loss_bound]``. A smaller ``difference_bound`` may be supplied
    only when it is certified independently of the probe; observing a small sample range is
    not enough to justify it.
    """
    b = float(loss_bound)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    reduced, dense = _paired_arrays(reduced_losses, dense_losses)
    if np.any(reduced < 0) or np.any(reduced > b) or np.any(dense < 0) or np.any(dense > b):
        raise ValueError("all losses must lie in [0, loss_bound]")

    r = b if difference_bound is None else float(difference_bound)
    if r <= 0 or r > b:
        raise ValueError("difference_bound must lie in (0, loss_bound]")
    return _bounded_empirical_bernstein_ucb(
        reduced - dense,
        absolute_bound=r,
        delta=delta,
        num_candidates=num_candidates,
    )


def distribution_free_zero_path_barrier(
    *,
    range_bound: float,
    n: int,
    delta: float = 0.05,
) -> float:
    """Rare-event lower barrier for any distribution-free one-sided mean UCB.

    If a valid UCB must cover the mean of every distribution supported on ``[0, R]``, then
    after observing ``n`` zeros its value cannot be below

        R * (1 - delta**(1/n)).

    The same obstruction applies to paired differences supported on ``[-R, R]`` because
    the adversarial two-point distribution can be supported on ``{0, R}``.
    """
    r = float(range_bound)
    nn = int(n)
    if r <= 0:
        raise ValueError("range_bound must be positive")
    if nn < 1:
        raise ValueError("n must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    return float(r * (1.0 - delta ** (1.0 / nn)))


def clipped_square_difference_second_moment_bound(
    *,
    loss_bound: float,
    prediction_l2_bound: float,
) -> float:
    """Convert an L2 predictor-gap certificate into a paired-loss second-moment bound.

    For clipped squared loss ``ell_B``, the prediction map is ``2 sqrt(B)``-Lipschitz.
    Thus if ``||f_reduced - f_dense||_{L2(P_X)} <= d``, then

        E[(ell_B(f_reduced)-ell_B(f_dense))^2] <= 4 B d^2.

    The trivial bound ``B^2`` is used whenever it is tighter.
    """
    b = float(loss_bound)
    d = float(prediction_l2_bound)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    if d < 0:
        raise ValueError("prediction_l2_bound must be nonnegative")
    return float(min(b * b, 4.0 * b * d * d))


def optimal_truncation_level(
    *,
    loss_bound: float,
    second_moment_bound: float,
    n: int,
    delta: float = 0.05,
    num_candidates: int = 1,
) -> float:
    """Probe-independent truncation level minimizing the deterministic EB remainder.

    It minimizes ``a*c + M2/(4*c)`` with
    ``a = 14 log(2M/delta)/(3(n-1))`` and is capped at ``loss_bound``. If the cap binds,
    no truncation is needed and the ordinary paired empirical-Bernstein bound is recovered.
    """
    b = float(loss_bound)
    m2 = float(second_moment_bound)
    nn = int(n)
    m = int(num_candidates)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    if m2 < 0:
        raise ValueError("second_moment_bound must be nonnegative")
    if nn < 2:
        raise ValueError("n must be at least two")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    if m < 1:
        raise ValueError("num_candidates must be positive")
    if m2 == 0:
        return 0.0

    logterm = math.log(2.0 * m / delta)
    a = 14.0 * logterm / (3.0 * (nn - 1))
    c = math.sqrt(min(m2, b * b) / (4.0 * a))
    return float(min(b, c))


def paired_truncated_second_moment_ucb(
    reduced_losses,
    dense_losses,
    *,
    loss_bound: float,
    second_moment_bound: float,
    delta: float = 0.05,
    num_candidates: int = 1,
    truncation_level: float | None = None,
) -> float:
    """Localized UCB using a probe-independent second-moment certificate.

    Let ``Z`` be the paired loss difference and suppose a deterministic, probe-independent
    bound ``E[Z^2] <= M2`` is available. For ``W_c = clip(Z, -c, c)``, the pointwise
    inequality ``Z <= W_c + Z^2/(4c)`` gives

        E[Z] <= E[W_c] + M2/(4c).

    Empirical Bernstein is then applied to ``W_c in [-c,c]``. This replaces the global
    ``loss_bound / n`` rare-event term by a local truncation scale plus an explicit moment
    remainder. The supplied moment bound must come from theory or data independent of the
    probe used in this call.
    """
    b = float(loss_bound)
    m2 = float(second_moment_bound)
    if b <= 0:
        raise ValueError("loss_bound must be positive")
    if m2 < 0:
        raise ValueError("second_moment_bound must be nonnegative")

    reduced, dense = _paired_arrays(reduced_losses, dense_losses)
    if np.any(reduced < 0) or np.any(reduced > b) or np.any(dense < 0) or np.any(dense > b):
        raise ValueError("all losses must lie in [0, loss_bound]")
    z = reduced - dense
    n = len(z)

    m2 = min(m2, b * b)
    if m2 == 0:
        if np.any(np.abs(z) > 1e-12):
            raise ValueError("a zero second-moment bound is incompatible with observed differences")
        return 0.0

    if truncation_level is None:
        c = optimal_truncation_level(
            loss_bound=b,
            second_moment_bound=m2,
            n=n,
            delta=delta,
            num_candidates=num_candidates,
        )
    else:
        c = float(truncation_level)
        if c <= 0 or c > b:
            raise ValueError("truncation_level must lie in (0, loss_bound]")

    if c >= b:
        return paired_empirical_bernstein_ucb(
            reduced,
            dense,
            loss_bound=b,
            delta=delta,
            num_candidates=num_candidates,
        )

    clipped = np.clip(z, -c, c)
    ucb = _bounded_empirical_bernstein_ucb(
        clipped,
        absolute_bound=c,
        delta=delta,
        num_candidates=num_candidates,
    )
    return float(ucb + m2 / (4.0 * c))


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
    """Return whether the generic paired structural-damage UCB is inside a risk budget."""
    ucb = paired_empirical_bernstein_ucb(
        reduced_losses,
        dense_losses,
        loss_bound=loss_bound,
        delta=delta,
        num_candidates=num_candidates,
    )
    return bool(ucb <= float(risk_budget)), float(ucb)
