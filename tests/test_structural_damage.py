import math

import numpy as np
import pytest

from neural_net.structural_damage import (
    clipped_square_difference_second_moment_bound,
    clipped_squared_losses,
    distribution_free_zero_path_barrier,
    optimal_truncation_level,
    paired_damage_statistics,
    paired_empirical_bernstein_ucb,
    paired_truncated_second_moment_ucb,
)


def test_paired_damage_statistics_use_difference_variance():
    dense = np.array([0.1, 0.2, 0.3, 0.4])
    reduced = dense + 0.02
    mean, var, n = paired_damage_statistics(reduced, dense)
    assert n == 4
    assert abs(mean - 0.02) < 1e-12
    assert var < 1e-30


def test_empirical_bernstein_ucb_exceeds_empirical_damage():
    dense = np.linspace(0.05, 0.25, 100)
    reduced = dense + 0.01 * np.sin(np.arange(100))
    mean, _, _ = paired_damage_statistics(reduced, dense)
    ucb = paired_empirical_bernstein_ucb(
        reduced,
        dense,
        loss_bound=1.0,
        delta=0.05,
        num_candidates=4,
    )
    assert ucb >= mean


def test_tighter_certified_difference_range_tightens_generic_ucb():
    dense = np.full(200, 0.1)
    reduced = dense + 1e-3 * np.sin(np.arange(200))
    generic = paired_empirical_bernstein_ucb(
        reduced,
        dense,
        loss_bound=1.0,
        delta=0.05,
        num_candidates=2,
    )
    local = paired_empirical_bernstein_ucb(
        reduced,
        dense,
        loss_bound=1.0,
        difference_bound=0.01,
        delta=0.05,
        num_candidates=2,
    )
    assert local < generic


def test_declared_difference_range_must_cover_observations():
    dense = np.zeros(20)
    reduced = np.full(20, 0.02)
    with pytest.raises(ValueError):
        paired_empirical_bernstein_ucb(
            reduced,
            dense,
            loss_bound=1.0,
            difference_bound=0.01,
        )


def test_distribution_free_zero_path_barrier_has_correct_formula():
    r = 2.0
    n = 100
    delta = 0.05
    expected = r * (1.0 - delta ** (1.0 / n))
    assert math.isclose(
        distribution_free_zero_path_barrier(range_bound=r, n=n, delta=delta),
        expected,
    )


def test_prediction_l2_certificate_localizes_paired_loss_second_moment():
    bound = clipped_square_difference_second_moment_bound(
        loss_bound=4.0,
        prediction_l2_bound=0.01,
    )
    assert math.isclose(bound, 0.0016)


def test_second_moment_bound_is_never_worse_than_trivial_loss_bound():
    bound = clipped_square_difference_second_moment_bound(
        loss_bound=4.0,
        prediction_l2_bound=10.0,
    )
    assert bound == 16.0


def test_optimal_truncation_is_probe_independent_and_inside_loss_range():
    c = optimal_truncation_level(
        loss_bound=4.0,
        second_moment_bound=1e-4,
        n=256,
        delta=0.05,
        num_candidates=2,
    )
    assert 0.0 < c < 4.0


def test_truncated_moment_ucb_can_be_much_tighter_for_small_structural_effects():
    n = 512
    dense = np.full(n, 0.1)
    reduced = dense + 5e-4 * np.sin(np.arange(n))
    z = reduced - dense
    m2 = float(np.mean(z**2)) * 1.05

    generic = paired_empirical_bernstein_ucb(
        reduced,
        dense,
        loss_bound=4.0,
        delta=0.05,
        num_candidates=2,
    )
    localized = paired_truncated_second_moment_ucb(
        reduced,
        dense,
        loss_bound=4.0,
        second_moment_bound=m2,
        delta=0.05,
        num_candidates=2,
    )
    assert localized < 0.05 * generic


def test_zero_second_moment_requires_zero_observed_difference():
    dense = np.zeros(20)
    reduced = np.zeros(20)
    assert paired_truncated_second_moment_ucb(
        reduced,
        dense,
        loss_bound=1.0,
        second_moment_bound=0.0,
    ) == 0.0

    reduced[0] = 0.01
    with pytest.raises(ValueError):
        paired_truncated_second_moment_ucb(
            reduced,
            dense,
            loss_bound=1.0,
            second_moment_bound=0.0,
        )


def test_clipped_squared_losses_respect_bound():
    pred = np.array([0.0, 2.0, 10.0])
    y = np.zeros(3)
    loss = clipped_squared_losses(pred, y, loss_bound=4.0)
    assert np.allclose(loss, [0.0, 4.0, 4.0])
