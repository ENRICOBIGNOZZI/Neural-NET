from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BulkReadinessCertificate:
    """Checkpoint certificate for deleting an isolated lower prediction bulk.

    The lower cluster is assumed to lie below ``bulk_edge`` and to be separated
    from the resolved upper cluster by ``prediction_gap``.  The certificate keeps
    target mass, eigenvalue scale, reachable spectral motion, and structural
    mismatch as distinct quantities.
    """

    horizon: float
    parameter_path_bound: float
    dense_jacobian_bound: float
    operator_path_bound: float
    prediction_gap_floor: float
    parameter_gap_floor: float
    bulk_mass_envelope: float
    bulk_energy_envelope_from_mass: float
    bulk_energy_envelope_from_energy: float
    bulk_energy_envelope: float
    bulk_dense_path_value_bound: float
    structural_mismatch_envelope: float
    omitted_gradient_path_bound: float
    parameter_discrepancy_bound: float
    prediction_discrepancy_bound: float
    gap_certified: bool


def _exp_integral(lipschitz: float, horizon: float) -> float:
    if lipschitz == 0.0:
        return float(horizon)
    x = lipschitz * horizon
    if x > 700.0:
        return float("inf")
    return float(math.expm1(x) / lipschitz)


def bulk_readiness_certificate(
    *,
    risk: float,
    gradient_norm: float,
    jacobian_op_norm: float,
    curvature_bound: float,
    vector_field_lipschitz: float,
    common_tube_jacobian_bound: float,
    bulk_edge: float,
    prediction_gap: float,
    parameter_gap: float,
    bulk_target_mass: float,
    bulk_gradient_energy: float,
    horizon: float,
    structural_mismatch: float = 0.0,
    initial_parameter_discrepancy: float = 0.0,
) -> BulkReadinessCertificate:
    """Compute the lower-bulk specialization of the checkpoint safety theorem.

    Conventions follow the theory draft: ``risk = 0.5 * ||e||^2`` and the dense
    flow is square-loss gradient flow.  ``bulk_target_mass`` is ``||P_B e||^2``
    and ``bulk_gradient_energy`` is ``<e, L P_B e>`` at the checkpoint.

    If reachable operator motion can close either the prediction-space or the
    paired parameter-space gap, ``gap_certified`` is false and all downstream
    discrepancy bounds are infinite.
    """

    values = {
        "risk": risk,
        "gradient_norm": gradient_norm,
        "jacobian_op_norm": jacobian_op_norm,
        "curvature_bound": curvature_bound,
        "vector_field_lipschitz": vector_field_lipschitz,
        "common_tube_jacobian_bound": common_tube_jacobian_bound,
        "bulk_edge": bulk_edge,
        "prediction_gap": prediction_gap,
        "parameter_gap": parameter_gap,
        "bulk_target_mass": bulk_target_mass,
        "bulk_gradient_energy": bulk_gradient_energy,
        "horizon": horizon,
        "structural_mismatch": structural_mismatch,
        "initial_parameter_discrepancy": initial_parameter_discrepancy,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if prediction_gap <= 0.0 or parameter_gap <= 0.0:
        raise ValueError("spectral gaps must be positive")

    r = float(risk)
    g0 = float(gradient_norm)
    j0 = float(jacobian_op_norm)
    bh = float(curvature_bound)
    lg = float(vector_field_lipschitz)
    bj_tube = float(common_tube_jacobian_bound)
    b = float(bulk_edge)
    gap_l0 = float(prediction_gap)
    gap_g0 = float(parameter_gap)
    m0 = float(bulk_target_mass)
    e0 = float(bulk_gradient_energy)
    h = float(horizon)
    delta0 = float(structural_mismatch)
    d0 = float(initial_parameter_discrepancy)

    # Dense parameter path: exact dissipation envelope and local Lipschitz envelope.
    a_loss = math.sqrt(h * r)
    a_local = g0 * _exp_integral(lg, h)
    a = min(a_loss, a_local)

    # Curvature converts parameter motion into a Jacobian and operator path budget.
    j_bar = j0 + bh * a
    v = 2.0 * bh * j_bar * a
    gap_l = gap_l0 - 2.0 * v
    gap_g = gap_g0 - 2.0 * v
    gap_ok = gap_l > 0.0 and gap_g > 0.0

    if not gap_ok:
        inf = float("inf")
        return BulkReadinessCertificate(
            horizon=h,
            parameter_path_bound=float(a),
            dense_jacobian_bound=float(j_bar),
            operator_path_bound=float(v),
            prediction_gap_floor=float(gap_l),
            parameter_gap_floor=float(gap_g),
            bulk_mass_envelope=inf,
            bulk_energy_envelope_from_mass=inf,
            bulk_energy_envelope_from_energy=inf,
            bulk_energy_envelope=inf,
            bulk_dense_path_value_bound=inf,
            structural_mismatch_envelope=inf,
            omitted_gradient_path_bound=inf,
            parameter_discrepancy_bound=inf,
            prediction_discrepancy_bound=inf,
            gap_certified=False,
        )

    residual_bound_sq = 2.0 * r
    l_op_bound = j_bar * j_bar

    # New lower-bulk target-mass envelope.  Projector total variation is bounded by
    # 2*V/gap_floor, and only projector motion can increase ||P_B e||^2 because
    # fitting inside a PSD spectral cluster decreases that mass.
    mass_bar = m0 + 2.0 * residual_bound_sq * v / gap_l
    energy_from_mass = (b + v) * mass_bar

    # Independent energy-persistence envelope from the general checkpoint theorem.
    energy_from_energy = e0 + residual_bound_sq * (
        1.0 + 2.0 * l_op_bound / gap_l
    ) * v

    # Both are valid, so their minimum is a valid sharper bulk envelope.
    e_bar = min(energy_from_mass, energy_from_energy)
    bulk_value = h * e_bar

    # Drift of the fixed structural realization relative to the moving kept cluster.
    delta_bar = delta0 + 2.0 * v / gap_g
    omitted_path = (
        h * math.sqrt(max(e_bar, 0.0))
        + delta0 * a
        + (2.0 * bh * j_bar / gap_g) * a * a
    )

    growth = math.exp(lg * h) if lg * h <= 700.0 else float("inf")
    parameter_bound = growth * (d0 + omitted_path)
    prediction_bound = bj_tube * parameter_bound

    return BulkReadinessCertificate(
        horizon=h,
        parameter_path_bound=float(a),
        dense_jacobian_bound=float(j_bar),
        operator_path_bound=float(v),
        prediction_gap_floor=float(gap_l),
        parameter_gap_floor=float(gap_g),
        bulk_mass_envelope=float(mass_bar),
        bulk_energy_envelope_from_mass=float(energy_from_mass),
        bulk_energy_envelope_from_energy=float(energy_from_energy),
        bulk_energy_envelope=float(e_bar),
        bulk_dense_path_value_bound=float(bulk_value),
        structural_mismatch_envelope=float(delta_bar),
        omitted_gradient_path_bound=float(omitted_path),
        parameter_discrepancy_bound=float(parameter_bound),
        prediction_discrepancy_bound=float(prediction_bound),
        gap_certified=True,
    )


def bulk_is_geometrically_ready(
    certificate: BulkReadinessCertificate,
    *,
    max_bulk_value: float,
    max_prediction_discrepancy: float,
) -> bool:
    """Population geometric gate before finite-sample and compute-value checks."""
    if max_bulk_value < 0.0 or max_prediction_discrepancy < 0.0:
        raise ValueError("readiness tolerances must be nonnegative")
    return bool(
        certificate.gap_certified
        and certificate.bulk_dense_path_value_bound <= float(max_bulk_value)
        and certificate.prediction_discrepancy_bound
        <= float(max_prediction_discrepancy)
    )
