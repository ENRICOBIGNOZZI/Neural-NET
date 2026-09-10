from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CheckpointPredictionGapCertificate:
    """A checkpoint-computable finite-horizon safety envelope.

    The certificate is derived for square-loss gradient flow with
    ``risk = 0.5 * ||residual||^2``.  It removes two oracle quantities from the
    earlier master theorem: the unknown future loss drop and the assumption that
    the relevant spectral gaps remain open.  Gap persistence is certified from
    the current gaps and a curvature-controlled path budget.

    ``common_tube_jacobian_bound`` and ``vector_field_lipschitz`` are still
    declared local/tube constants.  They must be valid for both the dense and the
    structurally reduced continuations; point estimates should not be presented as
    theorem-level certificates.
    """

    horizon: float
    loss_path_bound: float
    local_vector_field_path_bound: float
    path_length_bound: float
    dense_jacobian_bound: float
    operator_path_bound: float
    prediction_gap_floor: float
    parameter_gap_floor: float
    discarded_energy_envelope: float
    structural_mismatch_envelope: float
    omitted_gradient_path_bound: float
    parameter_discrepancy_bound: float
    prediction_discrepancy_bound: float
    gap_certified: bool


@dataclass(frozen=True)
class StructuralGroupPredictionGapCertificate:
    """Gap-free certificate for an actual fixed structural parameter projector.

    This version is useful when the physical deletion is already represented by a
    literal coordinate/group projector ``S``.  It uses the directly observable
    omitted group-gradient energy ``||(I-S)g_t||^2`` and a Lipschitz envelope for
    how much that group gradient can regrow over the horizon.  No spectral-cluster
    identification is required for the safety bound itself.
    """

    horizon: float
    loss_path_bound: float
    local_vector_field_path_bound: float
    path_length_bound: float
    integrated_loss_path_bound: float
    integrated_local_path_bound: float
    integrated_path_bound: float
    omitted_gradient_path_bound: float
    parameter_discrepancy_bound: float
    prediction_discrepancy_bound: float


def _exp_integral(lipschitz: float, horizon: float) -> float:
    """Integral of exp(lipschitz * s) over [0, horizon]."""
    if lipschitz == 0.0:
        return float(horizon)
    x = lipschitz * horizon
    if x > 700.0:
        return float("inf")
    return float(math.expm1(x) / lipschitz)


def _integrated_exp_integral(lipschitz: float, horizon: float) -> float:
    """Integral_0^H Integral_0^h exp(lipschitz*s) ds dh."""
    if lipschitz == 0.0:
        return 0.5 * float(horizon) ** 2
    x = lipschitz * horizon
    if x > 700.0:
        return float("inf")
    return float((math.expm1(x) - x) / (lipschitz * lipschitz))


def checkpoint_prediction_gap_certificate(
    *,
    risk: float,
    gradient_norm: float,
    jacobian_op_norm: float,
    curvature_bound: float,
    vector_field_lipschitz: float,
    common_tube_jacobian_bound: float,
    prediction_gap: float,
    parameter_gap: float,
    discarded_gradient_energy: float,
    horizon: float,
    structural_mismatch: float = 0.0,
    initial_parameter_discrepancy: float = 0.0,
) -> CheckpointPredictionGapCertificate:
    """Compute the current-state finite-horizon prediction-gap certificate.

    Parameters
    ----------
    risk:
        Current square-loss population/probe analogue, using the convention
        ``R = 0.5 ||e||^2``.
    gradient_norm:
        Current norm ``||J^* e||``.
    jacobian_op_norm:
        Current dense ``||J_t||_op``.
    curvature_bound:
        A valid bound on ``||D^2 F||`` along the dense horizon.
    vector_field_lipschitz:
        A valid Lipschitz constant for ``g(theta)=J_theta^* e_theta`` on a tube
        containing both dense and reduced trajectories.
    common_tube_jacobian_bound:
        A valid Jacobian/operator Lipschitz bound for the predictor on that common
        tube.  This is deliberately explicit rather than silently replaced by the
        dense-path Jacobian envelope.
    prediction_gap, parameter_gap:
        Current eigengaps isolating the proposed prediction-space discarded cluster
        and the paired parameter-space kept cluster.
    discarded_gradient_energy:
        Current target-weighted energy ``||Q_D J^* e||^2``.
    structural_mismatch:
        Current ``||(I-S) Q_K||_op`` for the realizable structural action.
    initial_parameter_discrepancy:
        Instantaneous parameter embedding/rebuild error at contraction.

    Returns
    -------
    A dataclass containing all intermediate gates.  If either current eigengap
    cannot be certified to remain open over the horizon, the terminal discrepancy
    bounds are ``inf`` and ``gap_certified`` is false.
    """

    values = {
        "risk": risk,
        "gradient_norm": gradient_norm,
        "jacobian_op_norm": jacobian_op_norm,
        "curvature_bound": curvature_bound,
        "vector_field_lipschitz": vector_field_lipschitz,
        "common_tube_jacobian_bound": common_tube_jacobian_bound,
        "prediction_gap": prediction_gap,
        "parameter_gap": parameter_gap,
        "discarded_gradient_energy": discarded_gradient_energy,
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
        raise ValueError("spectral gaps must be positive at the checkpoint")

    r = float(risk)
    g0 = float(gradient_norm)
    j0 = float(jacobian_op_norm)
    bh = float(curvature_bound)
    lg = float(vector_field_lipschitz)
    bj_tube = float(common_tube_jacobian_bound)
    gap_l0 = float(prediction_gap)
    gap_g0 = float(parameter_gap)
    e0 = float(discarded_gradient_energy)
    h = float(horizon)
    delta0 = float(structural_mismatch)
    d0 = float(initial_parameter_discrepancy)

    # Exact loss dissipation gives integral ||g||^2 <= R_t and hence this
    # parameter-path bound by Cauchy--Schwarz.
    a_loss = math.sqrt(h * r)

    # Lipschitz flow gives ||g_{t+s}|| <= exp(L_g s)||g_t||.
    a_local = g0 * _exp_integral(lg, h)
    a = min(a_loss, a_local)

    # Curvature turns parameter path length into a dense Jacobian envelope.
    bj_dense = j0 + bh * a

    # Both L=JJ* and G=J*J have operator path length at most this quantity.
    v = 2.0 * bh * bj_dense * a

    # Weyl: an inter-cluster gap can close by at most twice the operator motion.
    gap_l = gap_l0 - 2.0 * v
    gap_g = gap_g0 - 2.0 * v
    gap_ok = gap_l > 0.0 and gap_g > 0.0

    if not gap_ok:
        inf = float("inf")
        return CheckpointPredictionGapCertificate(
            horizon=h,
            loss_path_bound=float(a_loss),
            local_vector_field_path_bound=float(a_local),
            path_length_bound=float(a),
            dense_jacobian_bound=float(bj_dense),
            operator_path_bound=float(v),
            prediction_gap_floor=float(gap_l),
            parameter_gap_floor=float(gap_g),
            discarded_energy_envelope=inf,
            structural_mismatch_envelope=inf,
            omitted_gradient_path_bound=inf,
            parameter_discrepancy_bound=inf,
            prediction_discrepancy_bound=inf,
            gap_certified=False,
        )

    residual_bound = math.sqrt(2.0 * r)
    l_op_bound = bj_dense * bj_dense

    # Persistence of the target-weighted discarded gradient energy.
    e_bar = e0 + residual_bound**2 * (1.0 + 2.0 * l_op_bound / gap_l) * v

    # Fixed structural projector versus moving kept parameter cluster.
    delta_bar = delta0 + 2.0 * v / gap_g

    # A(s)=int_t^s ||g_r||dr.  Integrating
    # [delta0 + (4 B_H B_J / gap_g) A(s)] ||g_s||
    # yields delta0*A + (2 B_H B_J / gap_g) A^2 exactly at the envelope level.
    omitted_path = (
        h * math.sqrt(max(e_bar, 0.0))
        + delta0 * a
        + (2.0 * bh * bj_dense / gap_g) * a * a
    )

    growth = math.exp(lg * h) if lg * h <= 700.0 else float("inf")
    param_gap_bound = growth * (d0 + omitted_path)
    pred_gap_bound = bj_tube * param_gap_bound

    return CheckpointPredictionGapCertificate(
        horizon=h,
        loss_path_bound=float(a_loss),
        local_vector_field_path_bound=float(a_local),
        path_length_bound=float(a),
        dense_jacobian_bound=float(bj_dense),
        operator_path_bound=float(v),
        prediction_gap_floor=float(gap_l),
        parameter_gap_floor=float(gap_g),
        discarded_energy_envelope=float(e_bar),
        structural_mismatch_envelope=float(delta_bar),
        omitted_gradient_path_bound=float(omitted_path),
        parameter_discrepancy_bound=float(param_gap_bound),
        prediction_discrepancy_bound=float(pred_gap_bound),
        gap_certified=True,
    )


def structural_group_prediction_gap_certificate(
    *,
    risk: float,
    gradient_norm: float,
    omitted_group_gradient_energy: float,
    vector_field_lipschitz: float,
    common_tube_jacobian_bound: float,
    horizon: float,
    initial_parameter_discrepancy: float = 0.0,
) -> StructuralGroupPredictionGapCertificate:
    """Current-state certificate for a literal structural group deletion.

    If ``S`` is the fixed coordinate projector of the kept physical parameter
    groups, then ``omitted_group_gradient_energy`` is exactly
    ``||(I-S)g_t||^2``.  Lipschitzness of the vector field gives

        ||(I-S)g_s|| <= sqrt(E_D(t)) + L_g * ||theta_s-theta_t||.

    Integrating this inequality and using square-loss dissipation produces a
    gap-free finite-horizon bound.  It is particularly useful for rank components,
    channels, heads, or neurons whose deletion has an exact full-space embedding.
    """

    values = {
        "risk": risk,
        "gradient_norm": gradient_norm,
        "omitted_group_gradient_energy": omitted_group_gradient_energy,
        "vector_field_lipschitz": vector_field_lipschitz,
        "common_tube_jacobian_bound": common_tube_jacobian_bound,
        "horizon": horizon,
        "initial_parameter_discrepancy": initial_parameter_discrepancy,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")

    r = float(risk)
    g0 = float(gradient_norm)
    e_group = float(omitted_group_gradient_energy)
    lg = float(vector_field_lipschitz)
    bj_tube = float(common_tube_jacobian_bound)
    h = float(horizon)
    d0 = float(initial_parameter_discrepancy)

    a_loss = math.sqrt(h * r)
    a_local = g0 * _exp_integral(lg, h)
    a = min(a_loss, a_local)

    # Integrate the two pointwise path envelopes A(h).  The integral of the
    # pointwise minimum is no larger than the minimum of the two integrals.
    int_a_loss = (2.0 / 3.0) * math.sqrt(r) * h ** 1.5
    int_a_local = g0 * _integrated_exp_integral(lg, h)
    int_a = min(int_a_loss, int_a_local)

    omitted_path = h * math.sqrt(e_group) + lg * int_a
    growth = math.exp(lg * h) if lg * h <= 700.0 else float("inf")
    param_gap_bound = growth * (d0 + omitted_path)
    pred_gap_bound = bj_tube * param_gap_bound

    return StructuralGroupPredictionGapCertificate(
        horizon=h,
        loss_path_bound=float(a_loss),
        local_vector_field_path_bound=float(a_local),
        path_length_bound=float(a),
        integrated_loss_path_bound=float(int_a_loss),
        integrated_local_path_bound=float(int_a_local),
        integrated_path_bound=float(int_a),
        omitted_gradient_path_bound=float(omitted_path),
        parameter_discrepancy_bound=float(param_gap_bound),
        prediction_discrepancy_bound=float(pred_gap_bound),
    )
