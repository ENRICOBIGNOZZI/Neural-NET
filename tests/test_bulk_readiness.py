import math

from neural_net.bulk_readiness import (
    bulk_is_geometrically_ready,
    bulk_readiness_certificate,
)
from neural_net.safe_certificate import checkpoint_prediction_gap_certificate


def _kwargs():
    return dict(
        risk=0.5,
        gradient_norm=0.1,
        jacobian_op_norm=1.0,
        curvature_bound=0.0,
        vector_field_lipschitz=0.0,
        common_tube_jacobian_bound=1.0,
        bulk_edge=0.2,
        prediction_gap=1.0,
        parameter_gap=1.0,
        bulk_target_mass=0.04,
        bulk_gradient_energy=0.01,
        horizon=2.0,
        structural_mismatch=0.0,
        initial_parameter_discrepancy=0.0,
    )


def test_fixed_geometry_bulk_mass_envelope_is_exact_upper_bound():
    cert = bulk_readiness_certificate(**_kwargs())
    assert cert.gap_certified
    assert cert.operator_path_bound == 0.0
    assert math.isclose(cert.bulk_mass_envelope, 0.04)
    assert math.isclose(cert.bulk_energy_envelope_from_mass, 0.2 * 0.04)
    # The mass-derived envelope is tighter than the separately supplied energy bound.
    assert math.isclose(cert.bulk_energy_envelope, 0.008)
    assert math.isclose(cert.bulk_dense_path_value_bound, 2.0 * 0.008)
    assert math.isclose(
        cert.prediction_discrepancy_bound,
        2.0 * math.sqrt(0.008),
    )


def test_zero_bulk_value_gives_zero_prediction_damage_in_fixed_geometry():
    kwargs = _kwargs()
    kwargs.update(bulk_target_mass=0.0, bulk_gradient_energy=0.0)
    cert = bulk_readiness_certificate(**kwargs)
    assert cert.gap_certified
    assert cert.bulk_energy_envelope == 0.0
    assert cert.prediction_discrepancy_bound == 0.0
    assert bulk_is_geometrically_ready(
        cert,
        max_bulk_value=0.0,
        max_prediction_discrepancy=0.0,
    )


def test_reachable_operator_motion_can_refuse_to_identify_the_bulk():
    kwargs = _kwargs()
    kwargs.update(
        curvature_bound=1.0,
        gradient_norm=1.0,
        prediction_gap=1.0,
        parameter_gap=1.0,
        horizon=1.0,
    )
    cert = bulk_readiness_certificate(**kwargs)
    assert not cert.gap_certified
    assert math.isinf(cert.bulk_energy_envelope)
    assert math.isinf(cert.prediction_discrepancy_bound)
    assert not bulk_is_geometrically_ready(
        cert,
        max_bulk_value=1.0,
        max_prediction_discrepancy=1.0,
    )


def test_bulk_mass_specialization_is_never_looser_than_general_energy_envelope():
    kwargs = _kwargs()
    kwargs.update(
        curvature_bound=0.01,
        vector_field_lipschitz=0.02,
        prediction_gap=2.0,
        parameter_gap=2.0,
    )
    bulk = bulk_readiness_certificate(**kwargs)
    general = checkpoint_prediction_gap_certificate(
        risk=kwargs["risk"],
        gradient_norm=kwargs["gradient_norm"],
        jacobian_op_norm=kwargs["jacobian_op_norm"],
        curvature_bound=kwargs["curvature_bound"],
        vector_field_lipschitz=kwargs["vector_field_lipschitz"],
        common_tube_jacobian_bound=kwargs["common_tube_jacobian_bound"],
        prediction_gap=kwargs["prediction_gap"],
        parameter_gap=kwargs["parameter_gap"],
        discarded_gradient_energy=kwargs["bulk_gradient_energy"],
        horizon=kwargs["horizon"],
        structural_mismatch=kwargs["structural_mismatch"],
        initial_parameter_discrepancy=kwargs["initial_parameter_discrepancy"],
    )
    assert bulk.gap_certified == general.gap_certified
    assert bulk.bulk_energy_envelope <= general.discarded_energy_envelope + 1e-15
    assert bulk.prediction_discrepancy_bound <= general.prediction_discrepancy_bound + 1e-15


def test_readiness_requires_both_value_and_prediction_budgets():
    cert = bulk_readiness_certificate(**_kwargs())
    assert not bulk_is_geometrically_ready(
        cert,
        max_bulk_value=cert.bulk_dense_path_value_bound / 2.0,
        max_prediction_discrepancy=10.0,
    )
    assert not bulk_is_geometrically_ready(
        cert,
        max_bulk_value=10.0,
        max_prediction_discrepancy=cert.prediction_discrepancy_bound / 2.0,
    )
    assert bulk_is_geometrically_ready(
        cert,
        max_bulk_value=cert.bulk_dense_path_value_bound,
        max_prediction_discrepancy=cert.prediction_discrepancy_bound,
    )
