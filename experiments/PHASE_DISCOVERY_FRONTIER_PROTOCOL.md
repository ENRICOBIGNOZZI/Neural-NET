# Dynamic-BBP discovery frontier: precommitted protocol

## Question

The preceding rank-factor MLP falsification showed that a contraction can be safe while temporary overparameterization creates essentially no matched same-step state advantage. This experiment therefore does **not** tune the controller. It changes the controlled model to instantiate a mechanism already proved in the theory: a target direction is initially buried in an isotropic random covariance bulk, becomes identifiable only after a dynamic BBP transition, and can then be retained by a target-free top-outlier contraction.

The primary question is deliberately split in two:

1. Does temporary wide geometry create a genuine **discovery advantage** over a matched compact-from-start path at the same optimization time?
2. If yes, how expensive may the wide phase be before that advantage is erased by the extra optimization time a compact model could buy?

The second question is reported as a **critical cost frontier**, not by pretending that the number of random-matrix columns is a hardware FLOP ratio.

## Exact model

Teacher:

`A* = theta u u'`, with `theta = 1` and fixed unit vector `u=e_1`.

Wide initialization:

`A_0 = X X' / m`, with i.i.d. standard-normal `X`, dimension `d=64`, and `m=128`, so `gamma=d/m=1/2`.

Wide population dynamics are evaluated from the exact Riccati solution

`A_t = E_t (A_0^{-1}+F_t)^{-1} E_t`.

The theory gives a deterministic high-dimensional dynamic-BBP time `tau_{theta,gamma}`. The experiment evaluates checkpoints

`0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60`,

which include one point clearly before the theoretical crossing, one near it, and several after it.

## Target-free structural rule

At each checkpoint, diagonalize `A_t` and retain **only its largest eigenpair**. The retained eigenvector is selected by eigenvalue alone; the teacher direction is never used by the contraction rule. Its teacher overlap is recorded only for evaluation.

The retained eigenpair `(q_t, v_t)` defines the rank-one state `q_t v_t v_t'`. It then follows the exact rank-one gradient flow for a compact horizon `H=0.2`.

## Matched compact control

At time zero, apply exactly the same target-free top-eigenpair rule to `A_0`. This produces the matched rank-one initialization `(q_0^C, omega_0^C)`. The compact control then follows the same exact rank-one gradient flow.

This control is intentionally stronger and cleaner than an independently initialized rank-one model because it removes initialization luck: both branches start from the same random covariance, and the only difference is whether the wide covariance is allowed to evolve before structural contraction.

## Primary estimands

For wide checkpoint `t`, let `R_WC(t,H)` be the risk after wide training to `t`, top-eigenpair contraction, and rank-one continuation for `H`. Let `R_C(s)` be the matched compact rank-one risk after compact flow time `s`.

The **same-time discovery advantage** is

`D(t,H) = R_C(t+H) - R_WC(t,H)`.

Positive `D` means the wide phase has created a better compact state than the matched compact path reaches in the same optimization time. This is the first primary estimand.

For a declared wide-to-compact cost ratio `rho >= 1`, equal compute gives the compact control total flow time `H + rho t`. Define

`O_rho(t,H) = R_C(t+H) - R_C(H+rho t)`

and the equal-compute break-even margin

`M_rho(t,H) = D(t,H) - O_rho(t,H) = R_C(H+rho t) - R_WC(t,H)`.

Positive `M_rho` means temporary width wins at that declared cost ratio.

The **critical cost ratio**

`rho_star(t,H)`

is the unique break-even value derived in the theory: temporary width wins for `rho < rho_star` and loses for `rho > rho_star`, whenever the compact risk is strictly decreasing through the crossing.

## Spectral phase diagnostics

At every checkpoint record:

- theoretical dynamic-BBP time;
- theoretical transported bulk edge `b_+(t)`;
- largest and second-largest empirical covariance eigenvalues;
- top-minus-second eigengap;
- top-minus-theoretical-bulk-edge;
- squared overlap of the selected top eigenvector with the teacher direction.

The phase prediction is not merely that `D>0`. It is that target overlap and the repayable cost frontier should increase sharply as the top mode becomes separated from the nuisance bulk.

## Cost frontier

Before looking at results, declare `rho in {1,2,4,8,16}`. These are hypothetical structural cost ratios used to trace the frontier. They are **not** identified with `m=128`. The model's Wishart column count controls the random-matrix geometry; a real neural architecture must measure its own `C_W/C_Q` and place that measured ratio on the learned frontier.

## Repetitions and uncertainty

Use 50 independent Wishart seeds. Every checkpoint statistic is paired within seed. Report mean and 95% normal interval across the 50 independent seeds for discovery advantage, overlap, eigengap, and each declared-ratio break-even margin. For `rho_star`, report the median plus 10th/90th percentiles because the distribution can be skewed near the transition.

## Falsification criteria

The mechanism is falsified in this controlled model if post-BBP `D(t,H)` is not positive, because the entire purpose of the wide phase is to create target accessibility before rank-one contraction. A stronger compute claim at a declared ratio `rho` is falsified if `M_rho(t,H) <= 0`. We will not retune checkpoints or ratios after seeing the first run. If the finite-dimensional crossing is shifted relative to the asymptotic `tau`, that shift is reported rather than hidden.

A positive result here would establish only a **mechanism and cost frontier in the exact quadratic phase model**. It would not establish hardware speedup for a neural network. The next bridge would have to reproduce the same onset/discovery signatures in a structurally contractible neural architecture and then use measured training FLOPs or wall-clock.