# Neural-NET

Research repository for **safe dynamic spectral reduction during neural-network training**.

The central question is:

> When is a neural network ready to become smaller without giving up target-relevant future learning?

The project starts from prediction-space spectral geometry rather than weight magnitude. The initial theory separates:

1. **current fitting value** of a mode;
2. **future feature-discovery value** of a mode/subspace;
3. **optimization conditioning** gained by removing weak directions;
4. **finite-sample variance** saved by reducing nuisance capacity;
5. **structural realizability** of a spectral reduction as an actual reduction in width/rank/FLOPs.

The intended training principle is:

> **Overparameterize to discover; contract only after useful geometry has separated.**

## Repository structure

- `theory/` — canonical mathematical draft and proofs.
- `src/neural_net/` — reference implementation of theory-derived diagnostics/controllers.
- `experiments/` — dense-vs-reduced controlled experiments.
- `tests/` — unit tests for exact identities and controller invariants.
- `docs/` — research plan, literature map, claim ledger, and empirical protocol.

## Status

The first milestone is a rigorous theory of **finite-horizon mode value** and **safe spectral reduction**. The first algorithm is deliberately conservative: it leaves the base optimizer unchanged, waits for a persistence/separation certificate, and then contracts only structurally realizable low-value capacity.

No claim of universal speedup is made before matched-compute experiments establish it.
