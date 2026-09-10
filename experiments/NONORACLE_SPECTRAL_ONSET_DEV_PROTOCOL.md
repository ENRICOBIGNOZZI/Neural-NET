# Non-oracle spectral onset development protocol

## Question

Can the quadratic factor network infer both the retained rank and the onset of safe contraction from its **current covariance spectrum only**, without access to the teacher rank or target subspace?

## Frozen development family

- ambient dimension: `d = 24`
- teacher rank: `12`, evaluation only
- teacher strength: `theta = 1`
- wide factors: `24, 30, 36, 42, 48`
- development seeds: `0--49`
- SGD learning rate: `0.005`
- batch size: `256`
- horizon: `40` updates
- checkpoints: `20, 40, 60, 80, 100, 120`
- observation noise: `0`

These seeds are development data. Any rule chosen after inspecting them must be frozen and tested on a disjoint confirmatory seed family before it supports a claim.

## Information barrier

The detector may use only the ordered eigenvalues of the current covariance `W W^T` and quantities computed from them. It may not use:

- teacher rank;
- teacher basis or overlap;
- future loss or future structural damage;
- target labels beyond those already used by ordinary SGD;
- a checkpoint chosen ex post from evaluation outcomes.

Teacher rank, teacher overlap, and future damage are recorded only to score the target-free decision.

## Two pre-specified rank detectors

### A. Largest exceptional eigengap

With descending eigenvalues `lambda_1 >= ... >= lambda_d`, choose

```text
k_gap = argmax_k (lambda_k - lambda_{k+1}).
```

Record the absolute gap, its dominance over the second-largest adjacent gap, the boundary eigenvalue ratio, and retained spectral-mass fraction.

### B. Two-cluster log-spectrum split

For every split `k`, partition the log eigenvalues into a top and bottom cluster and minimize total within-cluster squared dispersion. The minimizing split is `k_log`. Record its standardized cluster separation and boundary eigenvalue ratio.

Neither detector knows `k*=12`.

## Development objective

Use the development family to determine whether either detector has a stable phase in which:

1. inferred rank is persistent across successive checkpoints;
2. inferred rank agrees with the target-supporting rank often enough to avoid destructive under-ranking;
3. contraction at the inferred rank has non-positive or negligible finite-horizon structural damage;
4. a simple spectral threshold can be frozen without consulting future outcomes at deployment time.

The preferred rule is the simplest one that is conservative near the transition. A false-late decision costs compute; a false-early decision destroys learned target geometry and is therefore treated as the more serious error.

## Next step

After development, freeze one controller including all thresholds and persistence requirements. Then run a new confirmatory family on unused seeds and evaluate:

```text
rank recovery -> onset calibration -> structural safety -> equal-compute margin.
```

Only the confirmatory family may upgrade the non-oracle controller claim in `docs/claim_ledger.md`.
