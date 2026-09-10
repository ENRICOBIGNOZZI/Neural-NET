import math

import numpy as np

from neural_net.quadratic_phase import (
    bbp_crossing_time,
    covariance_risk,
    edge_strength,
    edge_strength_derivative,
    exact_covariance_flow,
    mp_edges,
    null_bulk_edges,
    rank_one_alignment_delay,
    rank_one_overlap_after,
    riccati_velocity,
)


def test_edge_strength_crossing_and_derivative():
    theta, gamma = 2.5, 0.6
    tau = bbp_crossing_time(theta, gamma)
    assert abs(edge_strength(theta, gamma, tau) - 1.0) < 1e-10

    h = 1e-6
    numeric = (
        edge_strength(theta, gamma, tau + h)
        - edge_strength(theta, gamma, tau - h)
    ) / (2 * h)
    exact = edge_strength_derivative(theta, gamma, tau)
    assert math.isclose(numeric, exact, rel_tol=2e-6, abs_tol=2e-6)


def test_transported_bulk_contracts_to_zero():
    gamma = 0.5
    _, lp = mp_edges(gamma)
    _, b0 = null_bulk_edges(0.0, gamma)
    _, b1 = null_bulk_edges(1.0, gamma)
    _, b10 = null_bulk_edges(10.0, gamma)
    assert math.isclose(b0, lp)
    assert b10 < b1 < b0
    assert math.isclose(4.0 * 1000.0 * null_bulk_edges(1000.0, gamma)[1], 1.0, rel_tol=1e-3)


def test_exact_covariance_flow_satisfies_riccati_equation():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    a0 = q @ np.diag([0.3, 0.6, 0.9, 1.2, 1.6]) @ q.T
    u = rng.normal(size=5)
    u /= np.linalg.norm(u)
    a_star = 2.0 * np.outer(u, u)

    t, h = 0.13, 1e-6
    at = exact_covariance_flow(a0, a_star, t)
    numeric = (
        exact_covariance_flow(a0, a_star, t + h)
        - exact_covariance_flow(a0, a_star, t - h)
    ) / (2 * h)
    exact = riccati_velocity(at, a_star)
    assert np.linalg.norm(numeric - exact) / np.linalg.norm(exact) < 2e-6
    assert covariance_risk(exact_covariance_flow(a0, a_star, t + 0.05), a_star) < covariance_risk(at, a_star)


def test_rank_one_alignment_delay_is_exact():
    omega0, theta, eps = 0.03, 1.7, 0.1
    delay = rank_one_alignment_delay(omega0, theta, eps)
    reached = rank_one_overlap_after(omega0, theta, delay)
    assert math.isclose(reached, 1.0 - eps, rel_tol=1e-12, abs_tol=1e-12)
    assert rank_one_overlap_after(omega0, theta, 0.5 * delay) < 1.0 - eps
