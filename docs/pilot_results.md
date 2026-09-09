# Pilot results: dense AdamW vs proxy SRA

These experiments are implementation audits, not evidence for a final algorithmic claim. The controller is intentionally conservative and test labels are never used in a contraction decision.

## 1. Exact fixed-geometry theorem audit

For a synthetic 40-mode fixed geometry, the numerical difference between the directly simulated deletion cost and

\[
\frac12\sum_{j\in D}a_j^2(1-e^{-2\mu_jH})
\]

was `7.33e-17`. This is a machine-precision audit of the exact finite-horizon identity.

## 2. Exact quadratic onset audit

The rank-one quadratic teacher model gives a sharper test of **when** contraction should begin. With `theta=1`, `gamma=0.5`, the exact dynamic-BBP crossing time is `0.08328`. We compare a rank-one direction selected from the isotropic Wishart edge at initialization with the top covariance direction at `3 x tau_BBP = 0.24985`.

The theory proves that after rank-one contraction the target-overlap odds satisfy

\[
\frac{1-\omega_{t+h}}{\omega_{t+h}}
=
\frac{1-\omega_t}{\omega_t}e^{-4\theta h}.
\]

Thus a direction with only `O(1/d)` teacher overlap pays an additional `log(d)/(4 theta)` discovery delay, whereas a detached teacher outlier with order-one overlap does not.

Finite-dimensional audit, 12 seeds per dimension:

| d | median overlap at initialization | median overlap after `3 x tau_BBP` | extra time to 90% overlap: early | extra time: post-BBP |
|---:|---:|---:|---:|---:|
| 32 | 0.0161 | 0.8358 | 1.579 | 0.143 |
| 64 | 0.0031 | 0.8243 | 1.998 | 0.163 |
| 128 | 0.0033 | 0.8574 | 1.977 | 0.101 |
| 256 | 0.0027 | 0.8442 | 2.024 | 0.127 |

The point is not the exact finite-d slope. It is the large separation between a blind pre-discovery contraction and a contraction performed after the target direction is spectrally resolved.

## 3. Real-data CPU pilot

### Setup

- Two-hidden-layer tanh MLP.
- Initial last-hidden width: 64.
- 200 AdamW updates.
- Candidate contraction checkpoints every 20 updates.
- Fixed disjoint train/probe/test split.
- **Train data** construct the target-aware neuron ordering.
- The independent **probe** validates accessibility, frozen-readout MSE, and subspace persistence.
- The **test set** is ex-post only.
- Four seeds per dataset.
- After contraction the final linear readout is refit by ridge on training features, because this is the action covered by the final-hidden-span theorem.
- A dense readout-refit control is also run from the same checkpoint so that a Freeze-and-Solve benefit is not falsely attributed to contraction.

The proxy controller requires two consecutive checkpoints with resolved-subspace speed below `0.006`, permits at most `0.01` accessibility loss and `0.01` probe-MSE increase, then physically contracts the final hidden layer to a neuron subset.

### Current results

| Dataset | Runs | Contracted | Dense test MSE | SRA test MSE | Final parameter reduction | Active-parameter-step saving | Wall-clock ratio incl. controller/action |
|---|---:|---:|---:|---:|---:|---:|---:|
| Digits | 4 | 4/4 | 0.07033 | 0.07034 | 14.96% | 2.83% | 1.53x |
| Cancer | 4 | 0/4 | 0.17047 | 0.17047 | 0% | 0% | 0.82x |

### Interpretation

1. **The current contraction is performance-neutral on average, not better.** After aligning the implementation with the theory by refitting the readout, mean Digits test MSE is essentially identical to dense AdamW. Individual seeds remain heterogeneous. There is no accuracy-improvement claim.
2. **The decision rule is not trivially aggressive.** It contracts all four Digits runs but refuses to contract Cancer because the resolved feature subspace remains above the preset motion threshold.
3. **The structural contraction is real.** Final Digits parameter count falls by about 15%. Because onset is late (steps 140--180), cumulative active-parameter-update saving is only about 2.8%.
4. **The current controller loses on wall-clock.** Probe/SVD/ridge overhead plus late contraction produces about `1.53x` the dense CPU wall-clock on Digits. This falsifies any present claim of compute efficiency.
5. The Cancer wall-clock ratio below one is ordinary timing noise because no contraction occurred; it is not a speedup.

## What must improve before a serious empirical claim

The decisive quantity is the total risk/accuracy versus **charged compute** frontier, not final parameter count. The next implementation must therefore reduce state-estimation cost (small fixed reservoirs, randomized low-rank updates), derive an earlier but safe onset rule from the curvature/persistence theorem, and move to structural units whose removal creates large real FLOP savings. It must then beat dense AdamW and strong structured sparse/rank baselines on paired, compute-matched experiments.
