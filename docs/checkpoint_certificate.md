# Checkpoint-computable prediction-gap certificate

The original master theorem had the right decomposition but still contained two future/oracle objects: the unknown future loss decrease and the assumption that the relevant spectral clusters stayed isolated. The new checkpoint certificate removes both from the online decision rule.

For square-loss gradient flow, define the dense parameter-path budget

```text
A(H) = min(
    sqrt(H * current_risk),
    current_gradient_norm * integral_0^H exp(L_g s) ds,
)
```

where `L_g` is a valid local Lipschitz constant for the gradient vector field. Curvature then gives a dense Jacobian envelope

```text
J_bar = ||J_t|| + B_H * A(H)
```

and a common operator-motion budget for `L = J J*` and `G = J* J`,

```text
V(H) = 2 * B_H * J_bar * A(H).
```

A current prediction/parameter eigengap is usable only if it survives this path budget:

```text
prediction_gap_floor = prediction_gap_now - 2 * V(H)
parameter_gap_floor  = parameter_gap_now  - 2 * V(H)
```

If either floor is non-positive, the certificate refuses to contract. This turns spectral persistence from an assumption into a gate.

When the gaps survive, the target-weighted discarded gradient energy is propagated through the same curvature budget. If `E_D = ||Q_D J*e||^2` and `delta_0 = ||(I-S)Q_K||_op`, the integrated omitted-gradient envelope is

```text
H * sqrt(E_bar)
+ delta_0 * A(H)
+ 2 * B_H * J_bar / parameter_gap_floor * A(H)^2.
```

The three terms have separate meanings:

1. current plus reachable future target value of the discarded spectral cluster;
2. present structural mismatch between the ideal kept spectral subspace and the physical neuron/channel/rank action;
3. future drift of that kept subspace away from the fixed structural action.

The nonlinear flow-stability theorem converts this into a terminal prediction-gap certificate `d_chk`. For clipped square loss it immediately produces a pre-probe paired-loss second-moment bound

```text
M2_chk = min(B^2, 4 * B * d_chk^2).
```

That moment bound is exactly the input needed by the localized structural-damage UCB. The controller chain is therefore now

```text
current checkpoint
    -> path budget
    -> gap-persistence gate
    -> target-weighted discarded-energy envelope
    -> structural-transfer envelope
    -> prediction-gap certificate
    -> localized paired-loss UCB
    -> risk-vs-compute decision.
```

## Implementation

`src/neural_net/safe_certificate.py` implements the theorem as

```python
checkpoint_prediction_gap_certificate(...)
```

and returns every intermediate quantity rather than only a Boolean. In particular `gap_certified=False` and an infinite terminal discrepancy are returned when the spectral clusters cannot be guaranteed to remain isolated over the requested horizon.

The implementation deliberately requires `common_tube_jacobian_bound` and `vector_field_lipschitz` as declared inputs. Replacing them silently by point estimates would turn a theorem-level certificate into a heuristic. The next architecture-specific task is to make these constants tight enough to be useful in an actual MLP/channel or low-rank contraction controller.
