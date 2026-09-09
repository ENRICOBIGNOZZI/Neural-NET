# Neural-NET

Research repository for **safe dynamic spectral reduction during neural-network training**.

The central question is:

> When is a neural network ready to become smaller without giving up target-relevant future learning?

The project starts from prediction-space spectral geometry rather than weight magnitude. The intended principle is:

> **Overparameterize to discover; contract only after useful geometry has separated.**

## Current theory

The canonical draft in `theory/` now contains exact or explicitly conditional results for the full chain from spectral value to structural deletion:

- modal gradient energy `mu_j a_j^2`;
- exact fixed-geometry finite-horizon deletion value `0.5 a_j^2(1-exp(-2 mu_j H))`;
- a value-per-compute active-set oracle;
- an exact **compute-to-accuracy theorem** above the deletion floor, separating per-step cost reduction from condition-number improvement;
- an exact Gaussian-sequence nuisance-mode deletion criterion;
- curvature-to-prediction-spectrum path-length control;
- nonlinear structural-flow stability with an initial contraction error;
- spectral-cluster persistence from operator velocity and eigengap;
- persistence of a fixed structural mask relative to a moving parameter spectral cluster;
- a **master finite-horizon structural-contraction bound** combining discarded target energy, geometry motion, structural mismatch, and the initial contraction jump;
- a compute-aware dominance corollary: contract only when a valid structural-risk upper bound is below the risk-equivalent measured compute saving;
- an exact counterexample showing that arbitrarily large total subspace motion can carry exactly zero target value;
- exact current final-hidden neuron-basis contraction;
- the exact dynamic-BBP onset function and crossing time in the solvable quadratic teacher--student model;
- **dynamic target compressibility:** after all target spikes separate, the target becomes asymptotically supported on a fixed `k`-dimensional resolved prediction subspace with geometry error `sum_q theta_q^2(1-omega_q^2) -> 0`;
- exact rank-one post-contraction target-angle dynamics showing a `log(d)` discovery delay after blind premature contraction;
- a **paired empirical-Bernstein structural-damage theorem**: on an independent probe, a finite family of candidate contractions receives simultaneous upper confidence bounds using paired loss differences against the matched dense continuation;
- a fresh-probe finite-grid oracle for repeated adaptive contraction decisions.

The theoretical controller therefore has **five gates**: current value, future discovery value, structural realizability, finite-sample observability, and compute value.

## Controller development

The first **Spectral Rank Annealing (SRA)** proxy used total fixed-rank principal-angle speed as a hard gate. It is retained as a falsification baseline: it tracks unresolved nuisance motion too aggressively and has negative compute economics.

The second **Target-Value SRA (TV-SRA)** development proxy replaces that gate with resolved target accessibility and target-angle progress. It is closer to the theory but is still a proxy: recent target-angle speed is not an architecture-free upper bound on future target-relevant geometry motion, and point-estimated probe damage is not a certificate.

The next controller generation is **risk-budgeted TV-SRA**. It will construct a small finite family of structurally realizable actions and contract only when a paired upper confidence bound on finite-horizon structural damage lies below the candidate's predeclared risk / risk-equivalent compute budget.

## Current audits

- Fixed-geometry theorem identity error: `7.33e-17`.
- Quadratic onset audit (`theta=1`, `gamma=.5`): exact BBP crossing `0.0927600431`. Across dimensions 32--256 and 12 seeds each, blind initialization directions have `O(1/d)` teacher overlap, while the top direction at `2 x tau_BBP` has mean squared teacher overlap about `0.69--0.72`. The exact post-contraction angle law implies roughly `1.5--1.9` units less additional population time to reach 90% target overlap after waiting for separation.
- First SRA, Digits: 4/4 contractions; `14.96%` final parameter reduction; `2.83%` cumulative active-parameter-step saving; mean test MSE `0.06860` versus `0.06665` for the matched dense readout-refit control; `1.72x` dense CPU wall-clock including diagnostics. Negative as an efficiency result.
- First SRA, Cancer: 0/4 contractions and `1.33x` wall-clock due monitoring overhead.
- TV-SRA development, Digits: 4/4 contractions, mean step 140, `11.02%` final parameter reduction, `3.15%` active-parameter-step saving; mean MSE `0.06684` versus dense `0.07033` and dense-readout-refit `0.07077`. Development evidence only.
- TV-SRA development, Cancer: 4/4 contractions, mean step 130, `14.88%` final parameter reduction, `5.10%` active-parameter-step saving; mean MSE `0.17558` versus dense `0.17047` and dense-readout-refit `0.18326`. Mixed across seeds.
- TV-SRA still costs `1.32--1.45x` dense CPU wall-clock on the small implementation. **No compute-matched speedup has been demonstrated.**

### Finite-horizon structural-value audit

A paired counterfactual audit now measures the exact empirical object that the master theorem bounds: from the same checkpoint, compare a structurally contracted continuation with a matched full-width ridge-refit continuation over the next 40 updates.

Across 32 development transitions, leave-one-trajectory-out prediction of **future probe structural damage** gives:

- time + contraction size: `R^2=-0.234`, Pearson `-0.455`;
- current structural damage + time + size: `R^2=0.160`, Pearson `0.416`;
- target-value state: **`R^2=0.362`, Pearson `0.618`**.

But the decisive finding is negative: realized future probe damage and realized future test damage have Pearson **`-0.581`** overall (`-0.599` Cancer, `-0.522` Digits). The small probe is not reliably resolving the structural effect. A better predictor of the noisy probe quantity is therefore not enough.

This result is why finite-sample observability is now a separate theorem and a separate controller gate. The next experiment is a coverage/tightness audit of the paired structural-damage UCB against an effectively population-sized independent evaluation sample before any more real-data threshold tuning.

Seeds 0--3 are permanently designated development runs. See `docs/development_protocol.md`, `docs/pilot_results.md`, and `docs/structural_value_audit.md`.

## Literature boundary

The closest prior work on **when** to prune is Shen et al., *When To Prune? A Policy Towards Early Structural Pruning* (CVPR 2022), whose Early Pruning Indicator detects stabilization of the dominant pruned architecture. Therefore the contribution cannot be “learn dense briefly, then prune.” The research target is the mathematical distinction between **architecture stability** and **target-compressibility of learned prediction geometry**, including finite-horizon deletion value, phase-theoretic onset, structural-transfer error, and finite-sample observability of the deletion effect.

See `docs/literature_map.md` for the current novelty audit.

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — exact diagnostics, confidence bounds, and conservative controller prototypes.
- `experiments/` — theorem audits, dense-vs-reduced pilots, and finite-horizon structural-value counterfactuals.
- `tests/` — unit tests for exact identities, readout refitting, confidence bounds, and structural contraction.
- `docs/` — research plan, literature map, claim ledger, empirical protocol, and current pilot results.

## Scientific standard

No claim of universal speedup is made before matched-compute experiments establish it. Controller overhead is charged, unresolved bulk eigenvectors are not treated as identifiable, readout improvements are controlled separately, development seeds are not reused as confirmation, and a lower final parameter count alone is not counted as a training-compute win.
