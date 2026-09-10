# Exact rank-factor bridge for online rank annealing

The general prediction-spectrum theory identifies capacity that has little target-weighted value, but a prediction eigenvector is usually a dense direction in parameter space. Deleting it does not automatically delete neurons, channels, or FLOPs. The rank-factor bridge makes one structural case exact.

A linear map is parameterized as

```text
W = sum_q s_q u_q v_q^T
```

with trainable rank-one components. Dropping a set `D` means physically rebuilding the layer with only the kept components. The same compact predictor has an exact embedding in the old parameter space: retain the dropped `u_q, v_q` as frozen ghosts and set only `s_q = 0`. Therefore the contraction jump in that embedding is exactly

```text
d0 = ||s_D||_2.
```

This is useful because the structural action is no longer an approximate mapping from a spectral direction to arbitrary weights. The physical model genuinely has lower rank, fewer trainable parameters, and fewer factorized matrix-vector operations.

The second exact quantity is the gradient energy of a candidate group:

```text
E_q = ||grad(u_q)||^2 + grad(s_q)^2 + ||grad(v_q)||^2.
```

For a dropped set, `sum_q E_q` is exactly `||(I-S) g||^2` for the corresponding coordinate projector. This gives a gap-free finite-horizon structural safety bound: a group can only be considered dormant when its present gradient energy is small *and* a valid Lipschitz path budget says it cannot regrow enough over the horizon to matter.

The intended online decision has two complementary screens:

1. **Prediction-space value screen.** Use target-weighted spectral geometry to establish that the capacity sector has little current value and little reachable feature-discovery value. This is where phase separation and the prediction eigenspectrum matter.
2. **Physical-group safety screen.** Use the exact structural group energy, contraction jump, and group prediction-gap certificate to ensure that the rank factors actually being deleted are dynamically inactive enough.

Neither `|s_q|` nor `E_q` alone is a pruning score. A tiny scale with a large scale-gradient is a component ready to regrow. A small group gradient early in training can also be misleading if prediction geometry is still unresolved. The proposed controller therefore follows the principle

```text
overparameterize -> resolve target geometry -> identify low-value sector
-> verify physical group inactivity -> certify damage -> contract rank.
```

## Implementation

`src/neural_net/rank_factor.py` provides:

- `RankFactorLinear`
- exact `contracted_copy(...)`
- exact `ghost_zeroed_copy(...)`
- `rank_factor_group_gradient_energy(...)`
- `deleted_scale_norm(...)`
- `RankContractibleMLP`

`structural_group_prediction_gap_certificate(...)` in `safe_certificate.py` converts current group-gradient energy into a horizon prediction-gap bound. That bound can then be converted into the localized paired-loss second-moment bound already used by the finite-sample observability layer.

The next falsification experiment should compare dynamic rank annealing against both the full-rank factorized model and the final compact rank trained from scratch under equal compute. If the wide-then-contract path does not beat the compact-from-start control, the discovery-phase argument has not produced practical value even if the safety certificate is correct.
