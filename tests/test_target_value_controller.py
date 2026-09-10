import numpy as np

from neural_net.spectral import accessibility_from_features
from neural_net.target_value_controller import TargetValueConfig, TargetValueContractionController


def test_rank_limited_accessibility_is_not_full_span_accessibility():
    rng = np.random.default_rng(2)
    h = rng.normal(size=(80, 12))
    y = h[:, 7] + 0.01 * rng.normal(size=80)
    q_full = accessibility_from_features(h, y)
    q_one = accessibility_from_features(h, y, rank=1)
    assert q_full > q_one
    assert 0 <= q_one <= 1


def test_target_value_controller_waits_for_target_angle_stability():
    rng = np.random.default_rng(3)
    n_train, n_probe, width = 120, 80, 16
    train_h = rng.normal(size=(n_train, width))
    probe_h = rng.normal(size=(n_probe, width))
    train_y = train_h[:, 0]
    probe_y = probe_h[:, 0]

    cfg = TargetValueConfig(
        min_step=0,
        checkpoint_interval=20,
        resolved_rank=5,
        future_horizon_steps=40,
        max_future_accessibility_gain=1e-6,
        min_rank=8,
        rank_stride=2,
        max_fraction_removed=0.25,
        persistence=1,
    )
    ctl = TargetValueContractionController(cfg)
    first = ctl.evaluate(20, train_h, train_y, probe_h, probe_y)
    assert not first.should_contract
    second = ctl.evaluate(40, train_h, train_y, probe_h, probe_y)
    # Identical geometry implies zero target-angle speed, so the value gate opens.
    assert second.future_accessibility_gain_proxy == 0.0
    assert second.proposed_rank >= int(np.ceil(width * 0.75))
