from __future__ import annotations

import math


def feature_value_envelope(accessibility: float, path_length: float) -> float:
    """Maximum normalized span-value gain under a Grassmann path-length budget.

    If q is current target accessibility and L is an upper bound on future
    Grassmannian path length, the architecture-free theorem gives

        q_future <= sin^2(min(pi/2, asin(sqrt(q)) + L)).

    This function returns the resulting upper bound on q_future - q.
    When ``path_length`` is only a forecast rather than a certified upper bound,
    the output is a proxy and must not be called a certificate.
    """
    q = float(accessibility)
    length = float(path_length)
    if not 0.0 <= q <= 1.0:
        raise ValueError("accessibility must lie in [0, 1]")
    if length < 0:
        raise ValueError("path_length must be non-negative")
    angle = min(math.pi / 2.0, math.asin(math.sqrt(q)) + length)
    return float(max(0.0, math.sin(angle) ** 2 - q))


def constant_speed_feature_value_proxy(
    accessibility: float,
    subspace_speed: float,
    horizon_steps: int,
) -> float:
    """Constant-speed finite-horizon proxy for future representation value.

    This is intentionally named a *proxy*: recent speed need not upper-bound
    future speed.  It is useful as a prospective diagnostic before a calibrated
    or architecture-specific motion model is available.
    """
    speed = float(subspace_speed)
    horizon = int(horizon_steps)
    if horizon < 0:
        raise ValueError("horizon_steps must be non-negative")
    if not math.isfinite(speed):
        return math.inf
    if speed < 0:
        raise ValueError("subspace_speed must be non-negative")
    return feature_value_envelope(accessibility, speed * horizon)
