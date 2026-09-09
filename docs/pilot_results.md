# Pilot results: dense AdamW vs proxy SRA

These experiments are an implementation audit, not evidence for a final algorithmic claim. The controller is intentionally conservative and the test label is never used in a contraction decision.

## Setup

- Two-hidden-layer tanh MLP.
- Initial last-hidden width: 64.
- 200 AdamW updates.
- Candidate contraction checkpoints every 20 updates.
- A fixed train/probe/test split is used.
- **Train data** determine the target-aware neuron ordering.
- The independent **probe** validates accessibility, frozen-readout MSE, and subspace persistence.
- The **test set** is used only after training for ex-post comparison.
- Four seeds per dataset.

The proxy controller requires two consecutive checkpoints with resolved-subspace speed below `0.006`, permits at most `0.01` accessibility loss and `0.01` probe-MSE increase, and then contracts the final hidden layer to a structurally realizable neuron subset.

## Fixed-geometry theorem audit

For a synthetic 40-mode fixed geometry, the numerical difference between the directly simulated deletion cost and

\[
\frac12\sum_{j\in D}a_j^2(1-e^{-2\mu_jH})
\]

was `7.33e-17`. This is a code-level check of the exact finite-horizon identity.

## Current pilot

| Dataset | Runs | Contracted | Dense test MSE | SRA test MSE | Final parameter reduction | Active-parameter-step saving | Wall-clock ratio incl. controller |
|---|---:|---:|---:|---:|---:|---:|---:|
| Digits | 4 | 4/4 | 0.07033 | 0.06967 | 14.96% | 2.83% | 1.32x |
| Cancer | 4 | 0/4 | 0.17047 | 0.17047 | 0% | 0% | 0.91x |

### Interpretation

1. **The decision rule is not trivially aggressive.** It contracts all four Digits runs but refuses to contract Cancer because the resolved feature subspace remains above the preset motion threshold. This is the intended behavior of a discovery-before-compression rule.
2. **The small Digits pilot does not show a statistically established accuracy gain.** Mean MSE is slightly lower after contraction, but individual seeds are mixed. The claim at this stage is feasibility, not superiority.
3. **The structural contraction is real.** The final Digits models use roughly 15% fewer parameters in this architecture. Because contraction happens late, cumulative active-parameter-update savings are only 2.8%; this is too small to claim useful training-compute savings.
4. **The controller is still too expensive on this CPU-scale implementation.** Including probe/SVD/readout work, total Digits wall-clock is about 1.32x dense training. This falsifies any present claim of compute efficiency. The next engineering target is therefore not looser pruning, but lower-overhead state estimation and earlier safe onset.
5. The Cancer wall-clock ratio below one is ordinary timing noise because no contraction occurred; it should not be interpreted as a speedup.

## What must improve before a serious empirical claim

The decisive quantity is not final parameter count. It is the total risk/accuracy versus **charged compute** frontier. The next experiments must therefore:

- amortize geometry estimation with a small fixed reservoir and randomized/truncated updates;
- separate diagnostic overhead from saved forward/backward FLOPs;
- test whether the onset certificate can safely move earlier;
- compare against dense AdamW, magnitude pruning, a structured dynamic-sparse baseline, and a rank/spectral baseline;
- run enough seeds for paired confidence intervals;
- include at least one convolutional/residual architecture after the MLP audit passes.

The present result is useful because it gives a clean falsification target: a final method must preserve the conservative behavior while moving the compute frontier left.
