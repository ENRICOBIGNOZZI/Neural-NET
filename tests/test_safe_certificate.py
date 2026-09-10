import math

from neural_net.safe_certificate import checkpoint_prediction_gap_certificate


def test_zero_curvature_zero_omitted_energy_is_exactly_safe():
    cert = checkpoint_prediction_gap_certificate(
        risk=1.0,
        gradient_norm=0.5,
        jacobian_op_norm=2.0,
        curvature_bound=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=2.0,
        prediction_gap=0.5,
        parameter_gap=0.5,
        discarded_gradient_energy=0.0,
        horizon=3.0,
        structural_mismatch=0.0,
        initial_parameter_discrepancy=0.0,
    )
    assert cert.gap_certified
    assert cert.operator_path_bound == 0.0
    assert cert.prediction_discrepancy_bound == 0.0


def test_fixed_geometry_bound_reduces_to_h_sqrt_energy():
    cert = checkpoint_prediction_gap_certificate(
        risk=10.0,
        gradient_norm=10.0,
        jacobian_op_norm=1.0,
        curvature_bound=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=1.0,
        prediction_gap=1.0,
        parameter_gap=1.0,
        discarded_gradient_energy=0.04,
        horizon=2.0,
    )
    assert cert.gap_certified
    assert math.isclose(cert.omitted_gradient_path_bound, 0.4, rel_tol=1e-12)
    assert math.isclose(cert.prediction_discrepancy_bound, 0.4, rel_tol=1e-12)


def test_current_gradient_can_tighten_path_budget_late_in_training():
    cert = checkpoint_prediction_gap_certificate(
        risk=10.0,
        gradient_norm=0.01,
        jacobian_op_norm=1.0,
        curvature_bound=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=1.0,
        prediction_gap=1.0,
        parameter_gap=1.0,
        discarded_gradient_energy=0.0,
        horizon=10.0,
    )
    assert math.isclose(cert.loss_path_bound, 10.0)
    assert math.isclose(cert.local_vector_field_path_bound, 0.1)
    assert math.isclose(cert.path_length_bound, 0.1)


def test_gap_gate_fails_before_using_projector_persistence_bound():
    cert = checkpoint_prediction_gap_certificate(
        risk=1.0,
        gradient_norm=1.0,
        jacobian_op_norm=1.0,
        curvature_bound=1.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=2.0,
        prediction_gap=0.1,
        parameter_gap=0.1,
        discarded_gradient_energy=0.0,
        horizon=1.0,
    )
    assert not cert.gap_certified
    assert cert.prediction_gap_floor <= 0.0
    assert math.isinf(cert.prediction_discrepancy_bound)


def test_structural_drift_term_is_quadratic_in_path_budget():
    base = checkpoint_prediction_gap_certificate(
        risk=1.0,
        gradient_norm=0.1,
        jacobian_op_norm=1.0,
        curvature_bound=0.1,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=1.0,
        prediction_gap=10.0,
        parameter_gap=10.0,
        discarded_gradient_energy=0.0,
        horizon=0.1,
        structural_mismatch=0.0,
    )
    assert base.gap_certified
    a = base.path_length_bound
    expected_drift_piece = (
        2.0 * 0.1 * base.dense_jacobian_bound / base.parameter_gap_floor
    ) * a * a
    energy_piece = 0.1 * math.sqrt(base.discarded_energy_envelope)
    assert math.isclose(
        base.omitted_gradient_path_bound,
        energy_piece + expected_drift_piece,
        rel_tol=1e-12,
    )
