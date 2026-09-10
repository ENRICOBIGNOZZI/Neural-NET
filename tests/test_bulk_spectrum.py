import math

import numpy as np

from neural_net.bulk_spectrum import bulk_state_from_tangent_spectrum
from neural_net.spectral import TangentSpectrum, fixed_geometry_mode_value


def _spectrum():
    mu = np.array([5.0, 2.0, 0.4, 0.1])
    a = np.array([1.0, -2.0, 3.0, 4.0])
    return TangentSpectrum(
        eigenvalues=mu,
        coefficients=a,
        eigenvectors=np.eye(4),
        residual_risk=0.5 * float(np.sum(a * a)),
    )


def test_known_signal_rank_reports_exact_bulk_mass_and_energy():
    spec = _spectrum()
    state = bulk_state_from_tangent_spectrum(spec, signal_rank=2, horizon=3.0)

    assert state.signal_floor == 2.0
    assert state.bulk_edge == 0.4
    assert math.isclose(state.eigengap, 1.6)
    assert math.isclose(state.bulk_target_mass, 3.0**2 + 4.0**2)
    assert math.isclose(state.total_target_mass, 1.0 + 4.0 + 9.0 + 16.0)
    assert math.isclose(state.bulk_gradient_energy, 0.4 * 9.0 + 0.1 * 16.0)
    assert math.isclose(
        state.total_gradient_energy,
        5.0 * 1.0 + 2.0 * 4.0 + 0.4 * 9.0 + 0.1 * 16.0,
    )

    expected = np.sum(
        fixed_geometry_mode_value(
            np.array([0.4, 0.1]),
            np.array([3.0, 4.0]),
            3.0,
        )
    )
    assert math.isclose(state.bulk_fixed_geometry_value, float(expected))


def test_all_modes_resolved_gives_empty_bulk():
    spec = _spectrum()
    state = bulk_state_from_tangent_spectrum(spec, signal_rank=4, horizon=1.0)
    assert state.bulk_edge == 0.0
    assert state.bulk_target_mass == 0.0
    assert state.bulk_gradient_energy == 0.0
    assert state.bulk_fixed_geometry_value == 0.0


def test_zero_resolved_rank_treats_everything_as_bulk_without_inventing_gap():
    spec = _spectrum()
    state = bulk_state_from_tangent_spectrum(spec, signal_rank=0)
    assert state.bulk_edge == 5.0
    assert state.signal_floor == 0.0
    assert state.eigengap == 0.0
    assert math.isclose(state.bulk_target_mass_fraction, 1.0)
    assert math.isclose(state.bulk_gradient_energy_fraction, 1.0)


def test_zero_target_mass_has_well_defined_zero_fractions():
    spec = TangentSpectrum(
        eigenvalues=np.array([2.0, 1.0]),
        coefficients=np.zeros(2),
        eigenvectors=np.eye(2),
        residual_risk=0.0,
    )
    state = bulk_state_from_tangent_spectrum(spec, signal_rank=1)
    assert state.bulk_target_mass_fraction == 0.0
    assert state.bulk_gradient_energy_fraction == 0.0


def test_unsorted_spectrum_is_rejected():
    spec = TangentSpectrum(
        eigenvalues=np.array([1.0, 2.0]),
        coefficients=np.ones(2),
        eigenvectors=np.eye(2),
        residual_risk=1.0,
    )
    try:
        bulk_state_from_tangent_spectrum(spec, signal_rank=1)
    except ValueError as exc:
        assert "sorted" in str(exc)
    else:
        raise AssertionError("unsorted spectrum should be rejected")
