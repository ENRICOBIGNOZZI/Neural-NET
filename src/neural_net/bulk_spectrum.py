from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import TangentSpectrum, fixed_geometry_mode_value


@dataclass(frozen=True)
class BulkSpectralState:
    """Target-weighted state of a predeclared lower tangent-spectrum bulk.

    ``signal_rank`` is deliberately supplied by the caller.  This diagnostic does
    not search over eigengaps or choose a rank from the evaluation data.  In exact
    teacher--student audits it should be the known signal rank; in later empirical
    work it must come from a predeclared cluster/null rule.
    """

    signal_rank: int
    signal_floor: float
    bulk_edge: float
    eigengap: float
    total_target_mass: float
    bulk_target_mass: float
    bulk_target_mass_fraction: float
    total_gradient_energy: float
    bulk_gradient_energy: float
    bulk_gradient_energy_fraction: float
    bulk_fixed_geometry_value: float | None


def bulk_state_from_tangent_spectrum(
    spectrum: TangentSpectrum,
    *,
    signal_rank: int,
    horizon: float | None = None,
) -> BulkSpectralState:
    """Split a sorted empirical tangent spectrum into resolved outliers and bulk.

    Eigenvalues in :class:`TangentSpectrum` are descending.  The first
    ``signal_rank`` modes are treated as the resolved signal/outlier cluster and all
    remaining modes as the lower bulk.  The function reports the exact current
    target mass ``sum a_j^2`` and gradient energy ``sum mu_j a_j^2`` carried by the
    bulk, plus its exact frozen-geometry finite-horizon value when ``horizon`` is
    supplied.

    This function is a measurement primitive, not a pruning rule.
    """

    mu = np.asarray(spectrum.eigenvalues, dtype=float).reshape(-1)
    a = np.asarray(spectrum.coefficients, dtype=float).reshape(-1)
    if mu.shape != a.shape:
        raise ValueError("eigenvalues and coefficients must have the same shape")
    if np.any(mu < -1e-12):
        raise ValueError("prediction eigenvalues must be nonnegative")
    if len(mu) == 0:
        raise ValueError("spectrum must contain at least one mode")
    if np.any(mu[:-1] + 1e-12 < mu[1:]):
        raise ValueError("eigenvalues must be sorted in descending order")

    k = int(signal_rank)
    if k < 0 or k > len(mu):
        raise ValueError("signal_rank must lie between zero and the spectrum size")
    if horizon is not None and horizon < 0:
        raise ValueError("horizon must be nonnegative")

    total_mass = float(np.sum(a * a))
    total_energy = float(np.sum(mu * a * a))

    if k == len(mu):
        signal_floor = float(mu[-1])
        bulk_edge = 0.0
        gap = float(signal_floor)
        bulk_mass = 0.0
        bulk_energy = 0.0
        bulk_value = 0.0 if horizon is not None else None
    else:
        bulk_mu = mu[k:]
        bulk_a = a[k:]
        bulk_edge = float(bulk_mu[0])
        bulk_mass = float(np.sum(bulk_a * bulk_a))
        bulk_energy = float(np.sum(bulk_mu * bulk_a * bulk_a))
        if k == 0:
            signal_floor = 0.0
            gap = 0.0
        else:
            signal_floor = float(mu[k - 1])
            gap = float(max(0.0, signal_floor - bulk_edge))
        bulk_value = (
            float(np.sum(fixed_geometry_mode_value(bulk_mu, bulk_a, float(horizon))))
            if horizon is not None
            else None
        )

    mass_fraction = bulk_mass / total_mass if total_mass > 0.0 else 0.0
    energy_fraction = bulk_energy / total_energy if total_energy > 0.0 else 0.0

    return BulkSpectralState(
        signal_rank=k,
        signal_floor=float(signal_floor),
        bulk_edge=float(bulk_edge),
        eigengap=float(gap),
        total_target_mass=total_mass,
        bulk_target_mass=bulk_mass,
        bulk_target_mass_fraction=float(mass_fraction),
        total_gradient_energy=total_energy,
        bulk_gradient_energy=bulk_energy,
        bulk_gradient_energy_fraction=float(energy_fraction),
        bulk_fixed_geometry_value=bulk_value,
    )
