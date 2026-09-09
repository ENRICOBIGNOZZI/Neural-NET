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
- nonlinear projected-flow stability via an omitted-gradient envelope;
- a spectral persistence certificate from operator velocity and eigengap;
- transfer from ideal spectral deletion to a structural parameter deletion;
- exact current final-hidden neuron-basis contraction;
- a nonzero contraction-onset time in the solvable dynamic-BBP teacher--student model.

## First algorithmic realization

The provisional controller is **Spectral Rank Annealing (SRA)**. The implementation is deliberately conservative and currently contracts only the last hidden layer:

- train/probe/test are disjoint;
- train data construct a target-aware neuron ordering;
- the probe checks target accessibility, frozen-readout risk, and subspace persistence;
- the test set is ex-post only;
- the base optimizer remains AdamW.

The present CPU pilot is a falsification exercise, not a speedup claim. Across four Digits seeds the controller contracts 4/4 runs, reduces final parameter count by about 15%, and has slightly lower mean test MSE, but only saves about 2.8% active-parameter updates because contraction occurs late. Once controller overhead is charged, wall-clock is about 1.32x dense training. On Cancer it contracts 0/4 runs because the measured subspace remains above the preset motion threshold. See `docs/pilot_results.md`.

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — exact diagnostics and the first conservative controller.
- `experiments/` — fixed-geometry audit and dense-vs-SRA pilots.
- `tests/` — unit tests for exact identities and structural contraction.
- `docs/` — research plan, literature map, claim ledger, empirical protocol, and current pilot results.

## Scientific standard

No claim of universal speedup is made before matched-compute experiments establish it. Controller overhead is charged, unresolved bulk eigenvectors are not treated as identifiable, and a lower final parameter count alone is not counted as a training-compute win.
