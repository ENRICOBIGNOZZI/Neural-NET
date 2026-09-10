import math

from neural_net.feature_value import (
    constant_speed_feature_value_proxy,
    feature_value_envelope,
)


def test_feature_value_envelope_limits():
    assert feature_value_envelope(1.0, 10.0) == 0.0
    assert feature_value_envelope(0.0, 0.0) == 0.0
    assert math.isclose(feature_value_envelope(0.0, math.pi / 2), 1.0, rel_tol=1e-12)


def test_feature_value_envelope_monotone_in_path_length():
    q = 0.35
    small = feature_value_envelope(q, 0.05)
    large = feature_value_envelope(q, 0.20)
    assert 0.0 < small < large <= 1.0 - q


def test_constant_speed_proxy_matches_path_length_formula():
    q, speed, horizon = 0.6, 0.003, 40
    assert math.isclose(
        constant_speed_feature_value_proxy(q, speed, horizon),
        feature_value_envelope(q, speed * horizon),
        rel_tol=1e-12,
    )
