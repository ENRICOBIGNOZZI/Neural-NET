import numpy as np

from neural_net.spectral import (
    fixed_geometry_mode_value,
    omitted_gradient_energy,
    sequence_drop_delta,
    tangent_spectrum_from_jacobian,
)


def test_fixed_geometry_terminal_value_matches_direct_solution():
    mu = np.array([4.0, 1.0, 0.1])
    a = np.array([1.2, -0.7, 0.4])
    h = 0.8
    full = 0.5 * np.sum(a**2 * np.exp(-2 * mu * h))
    drop_last = 0.5 * (
        np.sum(a[:2] ** 2 * np.exp(-2 * mu[:2] * h)) + a[2] ** 2
    )
    assert np.allclose(drop_last - full, fixed_geometry_mode_value(mu[2], a[2], h))


def test_gradient_energy_identity_from_jacobian():
    rng = np.random.default_rng(2)
    n, p = 7, 11
    j = rng.normal(size=(n, p))
    e = rng.normal(size=n)
    spec = tangent_spectrum_from_jacobian(j, e)
    spectral = omitted_gradient_energy(spec.eigenvalues, spec.coefficients)
    grad = (j.T @ e) / n
    assert np.allclose(spectral, grad @ grad, rtol=1e-10, atol=1e-10)


def test_sequence_drop_delta_sign_can_be_negative_for_nuisance_mode():
    # weak target coefficient, appreciable sampling variance
    delta = sequence_drop_delta(mu=1.0, a=0.01, sigma2_over_n=0.1, ridge=0.0)
    assert float(delta) < 0
