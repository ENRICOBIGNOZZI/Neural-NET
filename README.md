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
- a nonzero contraction-onset time in the solvable dynamic-BBP teacher--student model;
- exact rank-one post-contraction target-angle dynamics showing a `log(d)` discovery delay after blind premature contraction;
- a fresh-probe finite-grid oracle for repeated adaptive contraction decisions.

## First algorithmic realization

The provisional controller is **Spectral Rank Annealing (SRA)**. The implementation is deliberately conservative and currently contracts only the last hidden layer:

- train/probe/test are disjoint;
- train data construct a target-aware neuron ordering;
- the probe checks target accessibility, frozen-readout risk, and subspace persistence;
- the test set is ex-post only;
- after contraction the final linear readout is refit by ridge, matching the theorem;
- a dense readout-refit control isolates the value of the solve from the value of contraction;
- the base optimizer remains AdamW.

## Current audits

- Fixed-geometry theorem identity error: `7.33e-17`.
- Quadratic onset audit (`theta=1`, `gamma=.5`): the exact BBP crossing is `0.08328`; after `3 x tau_BBP`, the retained top direction has median squared teacher overlap around `0.83--0.86` over dimensions 32--256, versus `0.003--0.016` at isotropic initialization. The implied extra time to 90% teacher overlap is roughly `0.10--0.16` post-BBP versus `1.58--2.02` after blind initialization contraction.
- Real-data CPU pilot, four seeds per dataset: Digits contracts 4/4 runs, reduces final parameters by about 15%, and is essentially performance-neutral versus dense AdamW; Cancer contracts 0/4 runs because the measured subspace remains too mobile. Because onset is late and diagnostics are expensive, the current Digits implementation uses about `1.53x` dense wall-clock. **This is explicitly a negative compute result.**

See `docs/pilot_results.md` for the exact interpretation.

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — exact diagnostics and the first conservative controller.
- `experiments/` — theorem audits and dense-vs-SRA pilots.
- `tests/` — unit tests for exact identities, readout refitting, and structural contraction.
- `docs/` — research plan, literature map, claim ledger, empirical protocol, and current pilot results.

## Scientific standard

No claim of universal speedup is made before matched-compute experiments establish it. Controller overhead is charged, unresolved bulk eigenvectors are not treated as identifiable, readout improvements are controlled separately, and a lower final parameter count alone is not counted as a training-compute win.
