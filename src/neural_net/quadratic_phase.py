from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


def mp_edges(gamma: float) -> tuple[float, float]:
    """Marchenko--Pastur edges for d / m -> gamma in (0, 1)."""
    gamma = float(gamma)
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    root = math.sqrt(gamma)
    return (1.0 - root) ** 2, (1.0 + root) ** 2


def null_bulk_edges(t: float, gamma: float) -> tuple[float, float]:
    """Exact transported MP covariance-bulk edges b_-(t), b_+(t)."""
    if t < 0:
        raise ValueError("t must be non-negative")
    lam_minus, lam_plus = mp_edges(gamma)
    g = lambda x: x / (1.0 + 4.0 * t * x)
    return g(lam_minus), g(lam_plus)


def edge_strength(theta: float, gamma: float, t: float) -> float:
    """Exact dynamic-BBP edge-strength chi_{theta,gamma}(t).

    The formula is equation (5.4.2) of the endogenous spectral-geometry
    quadratic teacher--student theorem.  Its derivative is strictly positive:

        chi'(t) = 4 theta exp(4 theta t) (1 + 4 t lambda_+) / sqrt(gamma).
    """
    theta = float(theta)
    gamma = float(gamma)
    t = float(t)
    if theta <= 0 or t < 0:
        raise ValueError("theta must be positive and t non-negative")
    _, lam_plus = mp_edges(gamma)
    r = math.exp(4.0 * theta * t)
    numerator = (r - 1.0) * (1.0 + 4.0 * t * lam_plus - lam_plus / theta)
    numerator += 4.0 * t * lam_plus
    return numerator / math.sqrt(gamma)


def edge_strength_derivative(theta: float, gamma: float, t: float) -> float:
    """Closed derivative of :func:`edge_strength`."""
    theta = float(theta)
    t = float(t)
    _, lam_plus = mp_edges(gamma)
    return (
        4.0
        * theta
        * math.exp(4.0 * theta * t)
        * (1.0 + 4.0 * t * lam_plus)
        / math.sqrt(gamma)
    )


def bbp_crossing_time(theta: float, gamma: float) -> float:
    """Unique tau_{theta,gamma} solving chi_{theta,gamma}(tau)=1."""
    theta = float(theta)
    if theta <= 0:
        raise ValueError("theta must be positive")
    hi = max(1e-6, 0.25 / theta)
    while edge_strength(theta, gamma, hi) < 1.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("failed to bracket BBP crossing time")
    return float(brentq(lambda s: edge_strength(theta, gamma, s) - 1.0, 0.0, hi))


def explicit_bulk_small_time(theta_min: float, gamma: float, epsilon: float) -> float:
    """Earliest t for which b_+(t) <= epsilon * theta_min.

    This is an exact algebraic consequence of
        b_+(t) = lambda_+ / (1 + 4 t lambda_+).
    It is a bulk-contraction time, not by itself a target-resolution certificate.
    """
    if theta_min <= 0 or epsilon <= 0:
        raise ValueError("theta_min and epsilon must be positive")
    _, lam_plus = mp_edges(gamma)
    rhs = (lam_plus / (epsilon * theta_min) - 1.0) / (4.0 * lam_plus)
    return float(max(0.0, rhs))


def exact_covariance_flow(a0: np.ndarray, a_star: np.ndarray, t: float) -> np.ndarray:
    """Exact solution A_t of dA/dt = 2(A* A + A A*) - 4 A^2.

    Implements
        A_t = E_t (A_0^{-1} + F_t)^{-1} E_t,
        E_t = exp(2 t A*),
        F_t = 4 int_0^t exp(4 s A*) ds.

    ``a0`` must be symmetric positive definite and ``a_star`` symmetric
    positive semidefinite.
    """
    a0 = np.asarray(a0, dtype=float)
    a_star = np.asarray(a_star, dtype=float)
    if a0.shape != a_star.shape or a0.ndim != 2 or a0.shape[0] != a0.shape[1]:
        raise ValueError("a0 and a_star must be square matrices of the same shape")
    if t < 0:
        raise ValueError("t must be non-negative")
    if not np.allclose(a0, a0.T, atol=1e-10) or not np.allclose(a_star, a_star.T, atol=1e-10):
        raise ValueError("a0 and a_star must be symmetric")
    eig0 = np.linalg.eigvalsh(a0)
    if eig0.min() <= 0:
        raise ValueError("a0 must be positive definite")

    evals, evecs = np.linalg.eigh(a_star)
    if evals.min() < -1e-10:
        raise ValueError("a_star must be positive semidefinite")
    evals = np.maximum(evals, 0.0)
    f_diag = np.empty_like(evals)
    positive = evals > 1e-12
    f_diag[positive] = np.expm1(4.0 * evals[positive] * t) / evals[positive]
    f_diag[~positive] = 4.0 * t
    f_t = (evecs * f_diag) @ evecs.T
    e_t = expm(2.0 * t * a_star)

    middle = np.linalg.inv(a0) + f_t
    # Solve instead of forming the inverse of ``middle`` explicitly.
    at = e_t @ np.linalg.solve(middle, e_t)
    return 0.5 * (at + at.T)


def riccati_velocity(a: np.ndarray, a_star: np.ndarray) -> np.ndarray:
    """Right-hand side of the exact covariance Riccati equation."""
    a = np.asarray(a, dtype=float)
    a_star = np.asarray(a_star, dtype=float)
    return 2.0 * (a_star @ a + a @ a_star) - 4.0 * (a @ a)


def covariance_risk(a: np.ndarray, a_star: np.ndarray) -> float:
    """Population excess risk 0.5 ||A-A*||_F^2 in the quadratic model."""
    delta = np.asarray(a, dtype=float) - np.asarray(a_star, dtype=float)
    return 0.5 * float(np.sum(delta * delta))


def top_teacher_overlap(a: np.ndarray, teacher_direction: np.ndarray) -> tuple[float, float]:
    """Largest-eigenvalue and its squared overlap with a teacher direction."""
    a = np.asarray(a, dtype=float)
    u = np.asarray(teacher_direction, dtype=float).reshape(-1)
    u = u / np.linalg.norm(u)
    vals, vecs = np.linalg.eigh(a)
    v = vecs[:, -1]
    return float(vals[-1]), float((u @ v) ** 2)


def rank_one_overlap_after(omega0: float, theta: float, horizon: float) -> float:
    """Exact target overlap after rank-one contraction.

    If omega is the squared teacher/active-vector overlap, then
        (1-omega_h)/omega_h = ((1-omega_0)/omega_0) exp(-4 theta h).
    """
    omega0 = float(omega0)
    theta = float(theta)
    horizon = float(horizon)
    if not 0.0 <= omega0 <= 1.0 or theta <= 0 or horizon < 0:
        raise ValueError("invalid omega0, theta, or horizon")
    if omega0 == 0.0:
        return 0.0
    if omega0 == 1.0:
        return 1.0
    odds = ((1.0 - omega0) / omega0) * math.exp(-4.0 * theta * horizon)
    return float(1.0 / (1.0 + odds))


def rank_one_alignment_delay(omega0: float, theta: float, epsilon: float) -> float:
    """Exact extra time needed to reach squared overlap at least 1-epsilon."""
    omega0 = float(omega0)
    theta = float(theta)
    epsilon = float(epsilon)
    if not 0.0 < omega0 <= 1.0:
        if omega0 == 0.0:
            return math.inf
        raise ValueError("omega0 must lie in [0,1]")
    if theta <= 0 or not 0.0 < epsilon < 1.0:
        raise ValueError("theta must be positive and epsilon in (0,1)")
    raw = math.log(((1.0 - omega0) * (1.0 - epsilon)) / (omega0 * epsilon))
    return float(max(0.0, raw / (4.0 * theta)))
