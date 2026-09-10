import math

from neural_net.rank_one_phase import (
    critical_wide_cost_ratio,
    rank_one_hitting_time,
    rank_one_risk,
    rank_one_risk_after,
    rank_one_state_after,
)


def test_closed_form_satisfies_rank_one_odes():
    q0, omega0, theta = 1.7, 0.08, 1.3
    h = 1e-6
    qh, wh = rank_one_state_after(q0, omega0, theta, h)
    qdot_numeric = (qh - q0) / h
    wdot_numeric = (wh - omega0) / h
    qdot_exact = -4.0 * q0 * (q0 - theta * omega0)
    wdot_exact = 4.0 * theta * omega0 * (1.0 - omega0)
    assert math.isclose(qdot_numeric, qdot_exact, rel_tol=2e-5, abs_tol=2e-5)
    assert math.isclose(wdot_numeric, wdot_exact, rel_tol=2e-5, abs_tol=2e-5)


def test_rank_one_risk_is_nonincreasing_along_exact_flow():
    q0, omega0, theta = 2.4, 0.02, 1.0
    times = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    risks = [rank_one_risk_after(q0, omega0, theta, t) for t in times]
    assert all(right <= left + 1e-13 for left, right in zip(risks, risks[1:]))
    assert risks[-1] < risks[0]


def test_hitting_time_inverts_exact_risk_curve():
    q0, omega0, theta = 2.0, 0.03, 1.1
    target_time = 0.73
    target = rank_one_risk_after(q0, omega0, theta, target_time)
    recovered = rank_one_hitting_time(q0, omega0, theta, target)
    assert math.isclose(recovered, target_time, rel_tol=1e-10, abs_tol=1e-10)


def test_zero_overlap_has_irreducible_teacher_floor():
    q0, theta = 2.0, 1.0
    assert math.isinf(rank_one_hitting_time(q0, 0.0, theta, 0.49))
    assert rank_one_hitting_time(q0, 0.0, theta, rank_one_risk(q0, 0.0, theta)) == 0.0


def test_critical_cost_ratio_is_exact_break_even_boundary():
    compact_q0, compact_omega0 = 2.5, 0.02
    wide_q, wide_omega = 1.05, 0.75
    theta, t, horizon = 1.0, 0.2, 0.2
    rho_star = critical_wide_cost_ratio(
        compact_q0,
        compact_omega0,
        wide_q,
        wide_omega,
        theta,
        t,
        horizon,
    )
    wide_risk = rank_one_risk_after(wide_q, wide_omega, theta, horizon)
    compact_at_boundary = rank_one_risk_after(
        compact_q0,
        compact_omega0,
        theta,
        horizon + rho_star * t,
    )
    assert math.isclose(wide_risk, compact_at_boundary, rel_tol=1e-10, abs_tol=1e-10)
    assert rank_one_risk_after(
        compact_q0,
        compact_omega0,
        theta,
        horizon + 0.9 * rho_star * t,
    ) > wide_risk
    assert rank_one_risk_after(
        compact_q0,
        compact_omega0,
        theta,
        horizon + 1.1 * rho_star * t,
    ) < wide_risk
