from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


def fixed_geometry_mode_value(mu, a, horizon: float):
    """Exact square-loss value of keeping fixed-geometry modes for ``horizon``.

    For one mode this is 0.5 * a^2 * (1 - exp(-2 * mu * H)).
    Inputs may be scalars or arrays.
    """
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(a, dtype=float)
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    return 0.5 * a**2 * (-np.expm1(-2.0 * mu * horizon))


def fixed_geometry_subset_cost(mu, a, horizon: float, drop_mask):
    """Exact terminal-risk increase from freezing the selected fixed modes."""
    values = fixed_geometry_mode_value(mu, a, horizon)
    mask = np.asarray(drop_mask, dtype=bool)
    return float(np.sum(values[mask]))


def omitted_gradient_energy(mu, a, drop_mask=None):
    """Squared norm of the gradient component in selected singular directions."""
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(a, dtype=float)
    energy = mu * a**2
    if drop_mask is None:
        return float(np.sum(energy))
    return float(np.sum(energy[np.asarray(drop_mask, dtype=bool)]))


def sequence_drop_delta(mu, a, sigma2_over_n: float, ridge: float):
    """Exact risk change (drop minus keep-with-ridge) in the Gaussian sequence model.

    Negative values mean that dropping the mode improves conditional risk.
    """
    mu = np.asarray(mu, dtype=float)
    a = np.asarray(a, dtype=float)
    lam = float(ridge)
    if lam < 0 or sigma2_over_n < 0:
        raise ValueError("ridge and sigma2_over_n must be non-negative")
    denom = (mu + lam) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(
            denom > 0,
            a**2 * mu * (mu + 2.0 * lam) / denom
            - sigma2_over_n * mu**2 / denom,
            0.0,
        )
    return delta


def _orthonormal_basis(x: np.ndarray, rtol: float = 1e-10, rank: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("matrix required")
    if x.size == 0:
        return np.zeros((x.shape[0], 0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if len(s) == 0:
        return np.zeros((x.shape[0], 0))
    numerical_rank = int(np.sum(s > rtol * max(s[0], 1.0)))
    if rank is None:
        use_rank = numerical_rank
    else:
        use_rank = min(max(0, int(rank)), numerical_rank)
    return u[:, :use_rank]


def accessibility_from_features(
    features: np.ndarray,
    target: np.ndarray,
    rtol: float = 1e-10,
    rank: int | None = None,
) -> float:
    """Squared target projection fraction onto an empirical feature subspace.

    ``rank=None`` uses the full numerical column span.  A finite ``rank`` uses only
    the leading left-singular subspace and is the appropriate diagnostic when the
    remaining empirical feature directions are treated as unresolved/noisy.
    """
    h = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if h.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same number of rows")
    h = h - h.mean(axis=0, keepdims=True)
    y = y - y.mean()
    denom = float(y @ y)
    if denom == 0:
        return 0.0
    q = _orthonormal_basis(h, rtol=rtol, rank=rank)
    if q.shape[1] == 0:
        return 0.0
    proj = q.T @ y
    return float((proj @ proj) / denom)


def target_angle_from_accessibility(accessibility: float) -> float:
    """Dimensionless target-accessibility angle asin(sqrt(q))."""
    q = float(accessibility)
    if not 0.0 <= q <= 1.0:
        raise ValueError("accessibility must lie in [0, 1]")
    return float(math.asin(math.sqrt(q)))


def projector_from_features(features: np.ndarray, rank: int | None = None) -> np.ndarray:
    h = np.asarray(features, dtype=float)
    h = h - h.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(h, full_matrices=False)
    if rank is None:
        rank = int(np.sum(s > 1e-10 * max(s[0] if len(s) else 1.0, 1.0)))
    rank = max(0, min(int(rank), u.shape[1]))
    q = u[:, :rank]
    return q @ q.T


def grassmann_distance(features_a: np.ndarray, features_b: np.ndarray, rank: int) -> float:
    """Largest principal angle between rank-r empirical column spans."""
    a = np.asarray(features_a, dtype=float)
    b = np.asarray(features_b, dtype=float)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    ua, _, _ = np.linalg.svd(a, full_matrices=False)
    ub, _, _ = np.linalg.svd(b, full_matrices=False)
    r = min(int(rank), ua.shape[1], ub.shape[1])
    if r <= 0:
        return 0.0
    s = np.linalg.svd(ua[:, :r].T @ ub[:, :r], compute_uv=False)
    cos_min = float(np.clip(s.min(), 0.0, 1.0))
    return float(math.acos(cos_min))


def effective_rank(features: np.ndarray) -> float:
    """Entropy effective rank of centered empirical features."""
    h = np.asarray(features, dtype=float)
    h = h - h.mean(axis=0, keepdims=True)
    s = np.linalg.svd(h, compute_uv=False)
    w = s**2
    total = float(w.sum())
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def target_greedy_neuron_order(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Leakage-safe OMP-style neuron ordering when called on training data.

    The ordering is target-aware: at each step it adds the neuron most correlated with
    the current least-squares residual after standardizing feature columns. The probe
    sample should be used only to validate a candidate prefix, not to construct it.
    """
    h = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    h = h - h.mean(axis=0, keepdims=True)
    y = y - y.mean()
    scale = np.linalg.norm(h, axis=0)
    x = h / np.where(scale > 1e-12, scale, 1.0)
    remaining = list(range(x.shape[1]))
    selected: list[int] = []
    residual = y.copy()
    while remaining:
        corr = np.abs(x[:, remaining].T @ residual)
        chosen_pos = int(np.argmax(corr))
        j = remaining.pop(chosen_pos)
        selected.append(j)
        xs = x[:, selected]
        coef, *_ = np.linalg.lstsq(xs, y, rcond=None)
        residual = y - xs @ coef
    return np.asarray(selected, dtype=int)


def pivoted_neuron_order(features: np.ndarray) -> np.ndarray:
    """Rank-revealing ordering of neuron columns using pivoted QR."""
    h = np.asarray(features, dtype=float)
    h = h - h.mean(axis=0, keepdims=True)
    _, _, piv = qr(h, mode="economic", pivoting=True)
    return np.asarray(piv, dtype=int)


@dataclass(frozen=True)
class TangentSpectrum:
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    eigenvectors: np.ndarray
    residual_risk: float


def tangent_spectrum_from_jacobian(jacobian: np.ndarray, residual: np.ndarray) -> TangentSpectrum:
    """Empirical prediction spectrum in the normalized sample Hilbert space.

    ``jacobian`` has shape (n, p); square loss is 0.5 * mean(residual**2).
    """
    j = np.asarray(jacobian, dtype=float)
    e = np.asarray(residual, dtype=float).reshape(-1)
    n = j.shape[0]
    if e.shape[0] != n:
        raise ValueError("jacobian and residual row counts differ")
    k = (j @ j.T) / n
    vals, vecs = np.linalg.eigh(k)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    a = vecs.T @ (e / np.sqrt(n))
    return TangentSpectrum(vals, a, vecs, 0.5 * float(np.mean(e**2)))
