import math

from neural_net.safe_certificate import structural_group_prediction_gap_certificate


def test_group_certificate_is_exactly_zero_for_zero_jump_zero_energy_zero_lipschitz():
    cert = structural_group_prediction_gap_certificate(
        risk=1.0,
        gradient_norm=0.5,
        omitted_group_gradient_energy=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=2.0,
        horizon=3.0,
        initial_parameter_discrepancy=0.0,
    )
    assert cert.omitted_gradient_path_bound == 0.0
    assert cert.prediction_discrepancy_bound == 0.0


def test_group_certificate_fixed_vector_field_reduces_to_h_sqrt_energy():
    cert = structural_group_prediction_gap_certificate(
        risk=10.0,
        gradient_norm=10.0,
        omitted_group_gradient_energy=0.04,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=1.0,
        horizon=2.0,
    )
    assert math.isclose(cert.omitted_gradient_path_bound, 0.4, rel_tol=1e-12)
    assert math.isclose(cert.prediction_discrepancy_bound, 0.4, rel_tol=1e-12)


def test_group_certificate_uses_integrated_current_gradient_envelope():
    cert = structural_group_prediction_gap_certificate(
        risk=100.0,
        gradient_norm=0.1,
        omitted_group_gradient_energy=0.0,
        vector_field_lipschitz=1.0,
        common_tube_jacobian_bound=1.0,
        horizon=0.2,
    )
    expected_local_path = 0.1 * math.expm1(0.2)
    expected_integrated = 0.1 * (math.expm1(0.2) - 0.2)
    assert math.isclose(cert.local_vector_field_path_bound, expected_local_path, rel_tol=1e-12)
    assert math.isclose(cert.integrated_local_path_bound, expected_integrated, rel_tol=1e-12)
    assert math.isclose(cert.integrated_path_bound, expected_integrated, rel_tol=1e-12)


def test_group_certificate_charges_initial_structural_jump():
    cert = structural_group_prediction_gap_certificate(
        risk=1.0,
        gradient_norm=0.0,
        omitted_group_gradient_energy=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=3.0,
        horizon=1.0,
        initial_parameter_discrepancy=0.2,
    )
    assert math.isclose(cert.parameter_discrepancy_bound, 0.2)
    assert math.isclose(cert.prediction_discrepancy_bound, 0.6)
