# Neural-NET

Research repository for **safe dynamic spectral reduction during neural-network training**.

The central question is:

> When is a neural network ready to become smaller without giving up target-relevant future learning?

The project starts from prediction-space spectral geometry rather than weight magnitude. The theory separates:

1. **current fitting value** of a mode;
2. **finite-horizon future value** of keeping it active;
3. **future feature-discovery value** under moving geometry;
4. **optimization conditioning and finite-sample variance** consequences;
5. **structural realizability** of an ideal spectral deletion as neurons/channels/rank/FLOPs.

The intended training principle is:

> **Overparameterize to discover; contract only after useful geometry has separated.**

## Current theory

The canonical draft in `theory/` now contains exact or explicitly conditional results for:

- modal gradient energy `mu_j a_j^2`;
- exact fixed-geometry finite-horizon deletion value;
- a value-per-compute active-set oracle;
- conditioning of the retained quadratic subproblem;
- an exact Gaussian-sequence nuisance-mode deletion criterion;
- curvature-to-prediction-spectrum path-length control;
- nonlinear structural-flow stability with an initial contraction error;
- a spectral persistence certificate from operator velocity and eigengap;
- persistence of a fixed structural mask relative to a moving parameter spectral cluster;
- a **master finite-horizon structural-contraction bound** combining discarded target energy, geometry motion, structural mismatch, and the initial contraction jump;
- a compute-aware dominance corollary: contract only when the certified risk price is below the measured compute saved;
- an exact counterexample showing that arbitrarily large total subspace motion can carry exactly zero target value;
- exact current final-hidden neuron-basis contraction;
- the exact dynamic-BBP onset function and crossing time in the solvable quadratic teacher--student model;
- **dynamic target compressibility:** after all target spikes separate, the target becomes asymptotically supported on a fixed `k`-dimensional resolved prediction subspace with geometry error `sum_q theta_q^2(1-omega_q^2) -> 0`;
- exact rank-one post-contraction target-angle dynamics showing a `log(d)` discovery delay after blind premature contraction;
- a fresh-probe finite-grid oracle for repeated adaptive contraction decisions.

## Controller development

The first **Spectral Rank Annealing (SRA)** proxy used total fixed-rank principal-angle speed as a hard gate. It is retained as a falsification baseline: it tracks unresolved nuisance motion too aggressively and has negative compute economics.

The second **Target-Value SRA (TV-SRA)** development proxy is closer to the theory:

- train/probe/test are disjoint;
- train data construct a target-aware structural candidate;
- the probe monitors target accessibility in a predeclared leading singular subspace;
- recent target-angle motion is converted into a finite-horizon accessibility-gain proxy;
- a single event removes at most 25% of the current last-hidden width;
- the same ridge-readout action is given to a full-width control so that contraction is isolated from readout solving;
- test labels are ex-post only.

TV-SRA is still a proxy rather than a certificate: recent target-angle speed is not an architecture-free upper bound on future target-relevant geometry motion.

## Current audits

- Fixed-geometry theorem identity error: `7.33e-17`.
- Quadratic onset audit (`theta=1`, `gamma=.5`): exact BBP crossing `0.0927600431`. Across dimensions 32--256 and 12 seeds each, blind initialization directions have `O(1/d)` teacher overlap, while the top direction at `2 x tau_BBP` has mean squared teacher overlap about `0.69--0.72`. The exact post-contraction angle law implies roughly `1.5--1.9` units less additional population time to reach 90% target overlap after waiting for separation.
- First SRA, Digits: 4/4 contractions; `14.96%` final parameter reduction; `2.83%` cumulative active-parameter-step saving; mean test MSE `0.06860` versus `0.06665` for the matched dense readout-refit control; `1.72x` dense CPU wall-clock including diagnostics. Negative as an efficiency result.
- First SRA, Cancer: 0/4 contractions and `1.33x` wall-clock due monitoring overhead.
- TV-SRA development, Digits: 4/4 contractions, mean step 140, `11.02%` final parameter reduction, `3.15%` active-parameter-step saving; mean MSE `0.06684` versus dense `0.07033` and dense-readout-refit `0.07077`. Promising only as development evidence.
- TV-SRA development, Cancer: 4/4 contractions, mean step 130, `14.88%` final parameter reduction, `5.10%` active-parameter-step saving; mean MSE `0.17558` versus dense `0.17047` and dense-readout-refit `0.18326`. Mixed across seeds.
- TV-SRA still costs `1.32--1.45x` dense CPU wall-clock on the small implementation. **No compute-matched speedup has been demonstrated.**

Seeds 0--3 are now permanently designated development runs. See `docs/development_protocol.md`.

A new paired counterfactual audit is being used to measure the exact empirical target of the master theorem: finite-horizon structural damage after contracting versus continuing full-width from the same checkpoint. The forecaster is fit only to future **probe** damage; test labels remain ex-post.

See `docs/pilot_results.md` for the exact seed-level results and interpretation.

## Literature boundary

The closest prior work on **when** to prune is Shen et al., *When To Prune? A Policy Towards Early Structural Pruning* (CVPR 2022), whose Early Pruning Indicator detects stabilization of the dominant pruned architecture. Therefore the contribution cannot be “learn dense briefly, then prune.” The research target is the mathematical distinction between **architecture stability** and **target-compressibility of learned prediction geometry**, including finite-horizon deletion value and a phase-theoretic reason not to contract before unresolved target modes separate.

See `docs/literature_map.md` for the current novelty audit.

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — exact diagnostics and conservative controller prototypes.
- `experiments/` — theorem audits, dense-vs-reduced pilots, and finite-horizon structural-value counterfactuals.
- `tests/` — unit tests for exact identities, readout refitting, and structural contraction.
- `docs/` — research plan, literature map, claim ledger, empirical protocol, and current pilot results.

## Scientific standard

No claim of universal speedup is made before matched-compute experiments establish it. Controller overhead is charged, unresolved bulk eigenvectors are not treated as identifiable, readout improvements are controlled separately, and a lower final parameter count alone is not counted as a training-compute win.
