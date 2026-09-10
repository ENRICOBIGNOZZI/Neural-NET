# Dynamic-BBP discovery frontier: precommitted 50-seed result

## Result in one sentence

The exact quadratic phase model produces the mechanism that the earlier rank-factor MLP did not: temporary wide geometry creates a large same-time target-discovery advantage after spectral separation, and that advantage can repay a substantial but finite wide-to-compact compute ratio. The repayable ratio peaks shortly after the dynamic-BBP transition and then declines, so the theory predicts a **cost-dependent contraction window**, not a universal one-sided pruning time.

## Precommitted setup

The protocol was frozen before the first run in `experiments/PHASE_DISCOVERY_FRONTIER_PROTOCOL.md`. We used 50 independent Wishart seeds, dimension `d=64`, columns `m=128`, `gamma=1/2`, rank-one teacher strength `theta=1`, compact continuation horizon `H=0.2`, checkpoints `0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60`, and declared wide/compact cost ratios `rho in {1,2,4,8,16}`.

The structural rule is target-free: at a checkpoint it keeps only the largest covariance eigenpair. Teacher overlap is used only for evaluation. The matched compact control applies the same target-free top-eigenpair rule to the identical initial covariance and then follows the exact rank-one gradient flow. Thus the matched comparison removes initialization luck and isolates the value of allowing the wide covariance geometry to evolve before contraction.

The theoretical high-dimensional dynamic-BBP crossing is

`tau_BBP = 0.0927600431`.

## Discovery geometry

| time | phase | top teacher overlap, mean [95% CI] | top eigenvalue - theoretical bulk edge | discovery advantage D | median rho_star [10%, 90%] |
|---:|:---:|:---|:---|:---|:---|
| 0.05 | pre | 0.0369 [0.0217, 0.0521] | -0.0499 [-0.0652, -0.0346] | 0.0198 [0.0120, 0.0275] | 1.13 [1.01, 1.79] |
| 0.10 | post / transition | 0.1649 [0.1260, 0.2037] | -0.0067 [-0.0175, 0.0041] | 0.1283 [0.0989, 0.1577] | 2.77 [1.09, 9.00] |
| 0.15 | post | 0.5109 [0.4693, 0.5525] | 0.0624 [0.0491, 0.0756] | 0.3665 [0.3377, 0.3952] | **9.31 [4.80, 13.98]** |
| 0.20 | post | 0.7336 [0.7164, 0.7508] | 0.1553 [0.1420, 0.1686] | 0.4731 [0.4587, 0.4874] | **8.85 [6.12, 12.60]** |
| 0.30 | post | 0.8947 [0.8882, 0.9013] | 0.3177 [0.3070, 0.3284] | 0.5106 [0.4968, 0.5245] | 6.70 [5.04, 9.17] |
| 0.40 | post | 0.9513 [0.9480, 0.9545] | 0.4391 [0.4309, 0.4474] | 0.5061 [0.4899, 0.5223] | 5.45 [4.25, 7.29] |
| 0.60 | post | 0.9870 [0.9860, 0.9879] | 0.6007 [0.5961, 0.6053] | 0.4763 [0.4543, 0.4984] | 4.11 [3.36, 5.32] |

Here `D = R_compact(t+H) - R_wide_then_rank1(t,H)`, so positive values mean temporary wide training has produced a better compact state than the matched compact path reaches in the same optimization time. `rho_star` is the maximum wide-to-compact cost ratio that the observed discovery advantage can repay at equal compute.

The qualitative phase prediction is visible. Before the theoretical crossing, the target-free top eigenvector is almost unaligned with the teacher and lies below the asymptotic bulk edge; the repayable cost ratio is barely above one. Around and immediately after the transition, teacher overlap rises sharply, the selected eigenvalue separates from the bulk, and `rho_star` jumps. It peaks around `t=0.15-0.20`, after which overlap continues to improve but the repayable cost ratio declines because additional wide training creates less new discovery value while continuing to pay the wide cost.

## Equal-compute windows on the predeclared cost grid

The equal-compute margin is

`M_rho = D - O_rho = R_compact(H + rho t) - R_wide_then_rank1(t,H)`.

Positive values favor the wide-discovery-then-contract path.

| declared rho | earliest robust positive grid point | later behavior |
|---:|:---|:---|
| 1 | 0.05 | positive throughout the grid |
| 2 | 0.10 | strongly positive from 0.10 through 0.60 |
| 4 | 0.15 | strongly positive from 0.15 through 0.60 |
| 8 | **0.15** | robustly positive at 0.15 and 0.20; no longer statistically resolved by 0.30 |
| 16 | none | negative at the post-BBP checkpoints |

Representative margins show the finite window clearly. At `rho=8`, the mean margin is `+0.0609 [0.0173, 0.1045]` at `t=0.15` and `+0.0756 [0.0315, 0.1197]` at `t=0.20`, but only `+0.0013 [-0.0301, 0.0327]` at `t=0.30` and is thereafter unresolved or negative. At `rho=4`, the margin is `+0.2099 [0.1772, 0.2427]` at `t=0.15` and remains positive even at `t=0.60`, `+0.0444 [0.0130, 0.0758]`. At `rho=16`, no predeclared checkpoint provides a positive equal-compute result.

## What this establishes

This experiment supplies a clean positive mechanism result in the exact model. Temporary overparameterization can create something the compact path does not reach as quickly: a target-aligned spectral outlier. Once that outlier is sufficiently resolved, target-free structural contraction preserves the discovered direction and avoids the dimension-dependent discovery delay of a premature rank-one path. The same mechanism also has an economic limit: if wide computation is too expensive, compact-from-start can use the saved compute to erase the discovery advantage.

The result also separates three clocks that should not be collapsed into one heuristic threshold:

`resolution time != safe contraction time != compute-optimal contraction time`.

Spectral resolution is an identification statement, safety is a statistical-damage statement, and compute optimality compares the marginal discovery value of remaining wide with its incremental cost. In this experiment the best economic window occurs after the BBP crossing but before target overlap has fully saturated.

## What this does not establish

The Wishart column count controls the random-matrix geometry and is **not** a hardware FLOP ratio. The declared `rho` grid is a cost frontier, not a hardware claim. The result therefore validates the phase/discovery mechanism and its compute economics in the exact quadratic model, not a speedup for a modern neural network.

The next bridge should be a finite-sample **quadratic-feature neural network** `f_W(x)=x' W W' x`, because it is a literal trainable factorization of the exact phase model. It provides physical parameter/rank reduction, finite-sample SGD, and a forward cost that scales with retained factor count while preserving a direct theoretical bridge to the Riccati dynamics. Only after this bridge survives should the controller move to ordinary MLP/CNN architectures.