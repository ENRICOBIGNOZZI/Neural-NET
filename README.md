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
- exact current final-hidden neuron-basis contraction;
- the exact dynamic-BBP onset function and crossing time in the solvable quadratic teacher--student model;
- **dynamic target compressibility:** after all target spikes separate, the target becomes asymptotically supported on a fixed `k`-dimensional resolved prediction subspace with geometry error `sum_q theta_q^2(1-omega_q^2) -> 0`;
- exact rank-one post-contraction target-angle dynamics showing a `log(d)` discovery delay after blind premature contraction;
- a fresh-probe finite-grid oracle for repeated adaptive contraction decisions.

## First algorithmic realization

The first **Spectral Rank Annealing (SRA)** proxy contracts only the last hidden layer:

- train/probe/test are disjoint;
- train data construct a target-aware neuron ordering;
- the probe checks target accessibility, frozen-readout risk, and subspace persistence;
- a prospective finite-horizon feature-value proxy is logged but is not yet used as a certificate;
- the test set is ex-post only;
- after contraction the final linear readout is refit by ridge;
- two full-width controls fork from the same checkpoint: optimizer reset only, and dense ridge-readout refit;
- the base optimizer remains AdamW.

This first controller is now treated as a **falsification baseline**, not as the final algorithm: its fixed-rank subspace-speed gate tracks unresolved nuisance directions too aggressively and is expensive.

## Current audits

- Fixed-geometry theorem identity error: `7.33e-17`.
- Quadratic onset audit (`theta=1`, `gamma=.5`): the exact BBP crossing is `0.0927600431`. Across dimensions 32--256 and 12 seeds each, blind initialization directions have `O(1/d)` teacher overlap, while the top direction at `2 x tau_BBP` has mean squared teacher overlap about `0.69--0.72`. The exact post-contraction angle law implies roughly `1.5--1.9` units less additional population time to reach 90% target overlap after waiting for separation.
- Controlled Digits pilot: 4/4 contractions, about `14.96%` final parameter reduction and only `2.83%` cumulative active-parameter-step saving. Adaptive mean test MSE is `0.06860`, versus `0.07033` for uninterrupted dense AdamW but `0.06665` for the **dense readout-refit control**. Thus the current experiment does **not** show a contraction accuracy advantage once the readout action is controlled.
- The same Digits controller costs `1.72x` dense CPU wall-clock including diagnostics. This is a negative compute result.
- Cancer: 0/4 contractions and `1.33x` wall-clock due monitoring overhead. The failure is informative: accessibility is nearly saturated while weak rank-8 directions keep rotating, showing that a total-subspace-speed gate is not the right target-value object.

See `docs/pilot_results.md` for the exact seed-level results and interpretation.

## Literature boundary

The closest prior work on **when** to prune is Shen et al., *When To Prune? A Policy Towards Early Structural Pruning* (CVPR 2022), whose Early Pruning Indicator detects stabilization of the dominant pruned architecture. Therefore the contribution cannot be “learn dense briefly, then prune.” The research target is the mathematical distinction between **architecture stability** and **target-compressibility of learned prediction geometry**, including finite-horizon deletion value and a phase-theoretic reason not to contract before unresolved target modes separate.

See `docs/literature_map.md` for the current novelty audit.

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — exact diagnostics and conservative controller prototypes.
- `experiments/` — theorem audits and dense-vs-reduced controlled pilots.
- `tests/` — unit tests for exact identities, readout refitting, and structural contraction.
- `docs/` — research plan, literature map, claim ledger, empirical protocol, and current pilot results.

## Scientific standard

No claim of universal speedup is made before matched-compute experiments establish it. Controller overhead is charged, unresolved bulk eigenvectors are not treated as identifiable, readout improvements are controlled separately, and a lower final parameter count alone is not counted as a training-compute win.
