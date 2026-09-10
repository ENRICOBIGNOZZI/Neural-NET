from __future__ import annotations

import math

from scipy.optimize import brentq


def rank_one_risk(q: float, omega: float, theta: float) -> float:
    """Population risk ``0.5 ||w w' - theta u u'||_F^2``.

    ``q = ||w||^2`` and ``omega`` is the squared teacher/active-vector overlap.
    """
    q = float(q)
    omega = float(omega)
    theta = float(theta)
    if q < 0.0 or not 0.0 <= omega <= 1.0 or theta <= 0.0:
        raise ValueError("q >= 0, omega in [0,1], and theta > 0 are required")
    return 0.5 * (q * q + theta * theta - 2.0 * theta * q * omega)


def rank_one_state_after(
    q0: float,
    omega0: float,
    theta: float,
    horizon: float,
) -> tuple[float, float]:
    """Exact rank-one quadratic-flow state after ``horizon``.

    Under ``dw/dt = -2 (w w' - theta u u') w`` the overlap is logistic and
    the radial coordinate ``q=||w||^2`` is obtained by a scalar integrating
    factor.  This is the closed form proved in the theory manuscript.
    """
    q0 = float(q0)
    omega0 = float(omega0)
    theta = float(theta)
    horizon = float(horizon)
    if q0 <= 0.0 or not 0.0 <= omega0 <= 1.0 or theta <= 0.0 or horizon < 0.0:
        raise ValueError("q0 > 0, omega0 in [0,1], theta > 0, horizon >= 0 required")

    growth = math.exp(4.0 * theta * horizon)
    numerator = 1.0 - omega0 + omega0 * growth
    omega = omega0 * growth / numerator
    denominator = (
        1.0 / q0
        + 4.0 * (1.0 - omega0) * horizon
        + (omega0 / theta) * (growth - 1.0)
    )
    q = numerator / denominator
    return float(q), float(omega)


def rank_one_risk_after(
    q0: float,
    omega0: float,
    theta: float,
    horizon: float,
) -> float:
    q, omega = rank_one_state_after(q0, omega0, theta, horizon)
    return rank_one_risk(q, omega, theta)


def rank_one_hitting_time(
    q0: float,
    omega0: float,
    theta: float,
    target_risk: float,
) -> float:
    """Earliest exact-flow time at which rank-one risk is <= ``target_risk``."""
    target_risk = float(target_risk)
    if target_risk < 0.0:
        raise ValueError("target_risk must be non-negative")
    initial = rank_one_risk(q0, omega0, theta)
    if initial <= target_risk:
        return 0.0

    # With exactly zero teacher overlap the target direction is invariantly
    # inaccessible and the risk converges to theta^2 / 2 as q -> 0.
    if omega0 == 0.0 and target_risk < 0.5 * theta * theta:
        return math.inf

    def residual(t: float) -> float:
        return rank_one_risk_after(q0, omega0, theta, t) - target_risk

    hi = max(1e-8, 0.25 / theta)
    while residual(hi) > 0.0:
        hi *= 2.0
        if 4.0 * theta * hi > 650.0:
            return math.inf
    return float(brentq(residual, 0.0, hi))


def critical_wide_cost_ratio(
    compact_q0: float,
    compact_omega0: float,
    wide_contracted_q: float,
    wide_contracted_omega: float,
    theta: float,
    wide_time: float,
    compact_horizon: float,
) -> float:
    """Maximum wide/compact time-cost ratio repaid by temporary discovery.

    A wide model trains for ``wide_time``, is contracted to the supplied
    rank-one state, and then trains rank one for ``compact_horizon``.  The
    matched compact control starts from ``(compact_q0, compact_omega0)``.
    If a wide unit of flow costs rho times a compact unit, the compact control
    receives ``compact_horizon + rho * wide_time`` flow time at equal compute.
    """
    wide_time = float(wide_time)
    compact_horizon = float(compact_horizon)
    if wide_time <= 0.0 or compact_horizon < 0.0:
        raise ValueError("wide_time must be positive and compact_horizon non-negative")

    achieved = rank_one_risk_after(
        wide_contracted_q,
        wide_contracted_omega,
        theta,
        compact_horizon,
    )
    compact_time = rank_one_hitting_time(
        compact_q0,
        compact_omega0,
        theta,
        achieved,
    )
    if math.isinf(compact_time):
        return math.inf
    return float((compact_time - compact_horizon) / wide_time)
