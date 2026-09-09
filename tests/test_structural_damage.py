import numpy as np

from neural_net.structural_damage import (
    clipped_squared_losses,
    paired_damage_statistics,
    paired_empirical_bernstein_ucb,
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


def test_clipped_squared_losses_respect_bound():
    pred = np.array([0.0, 2.0, 10.0])
    y = np.zeros(3)
    loss = clipped_squared_losses(pred, y, loss_bound=4.0)
    assert np.allclose(loss, [0.0, 4.0, 4.0])
