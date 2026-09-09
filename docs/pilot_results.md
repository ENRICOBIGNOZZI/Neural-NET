# Pilot results: dense AdamW vs theory-derived structural contraction

These experiments are implementation audits, not evidence for a final algorithmic claim. The controller is intentionally conservative and test labels are never used in a contraction decision.

## 1. Exact fixed-geometry theorem audit

For a synthetic 40-mode fixed geometry, the numerical difference between the directly simulated deletion cost and

\[
\frac12\sum_{j\in D}a_j^2(1-e^{-2\mu_jH})
\]

was `7.33e-17`. This is a machine-precision audit of the exact finite-horizon identity.

## 2. Exact quadratic onset audit

The rank-one quadratic teacher model gives the cleanest test of **when** contraction should begin. With `theta=1` and `gamma=0.5`, the exact edge-strength equation

\[
\chi_{\theta,\gamma}(\tau)=1
\]

gives

\[
\boxed{\tau_{\rm BBP}=0.0927600431.}
\]

The same model proves that after a rank-one structural contraction the target-overlap odds obey

\[
\frac{1-\omega_{t+h}}{\omega_{t+h}}
=
\frac{1-\omega_t}{\omega_t}e^{-4\theta h}.
\]

Hence a blind direction with `O(1/d)` teacher overlap pays a logarithmic discovery delay, while a post-separation teacher outlier has order-one overlap.

A fresh exact-flow audit used dimensions 32, 64, 128, 256 and 12 seeds per dimension. The post-BBP checkpoint was `2 * tau_BBP`.

| d | mean initial overlap | `d * mean initial overlap` | mean overlap at `2 tau_BBP` | mean extra time to 90% overlap: blind | post-BBP | delay reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 32  | 0.0281 | 0.898 | 0.7219 | 1.817 | 0.298 | 1.519 |
| 64  | 0.0187 | 1.194 | 0.6954 | 1.994 | 0.338 | 1.656 |
| 128 | 0.00557 | 0.713 | 0.6855 | 2.246 | 0.349 | 1.897 |
| 256 | 0.00451 | 1.156 | 0.7057 | 2.124 | 0.330 | 1.795 |

This audit now matches the exact Riccati flow implemented in `src/neural_net/quadratic_phase.py`. The important result is not a fitted slope: it is the order-one post-separation target overlap versus the `O(1/d)` blind overlap and the resulting large gap in target-discovery time.

## 3. Controlled real-data CPU pilot

### Setup

- Two-hidden-layer tanh MLP.
- Initial last-hidden width: 64.
- 200 AdamW updates.
- Candidate checkpoints every 20 updates.
- Fixed disjoint 60/20/20 train/probe/test split.
- **Train data** construct the target-aware neuron ordering.
- The independent **probe** validates current accessibility, frozen-readout MSE, and subspace persistence.
- The **test set** is ex-post only.
- Four seeds per dataset.
- The adaptive branch physically contracts the final hidden layer and then refits the linear readout by ridge.
- At the same checkpoint two full-width controls are forked from exactly the same pre-contraction state:
  1. **optimizer-reset control:** fresh AdamW state, no readout solve;
  2. **dense readout-refit control:** same ridge solve as the adaptive branch, but no structural contraction.

The second control is essential: it isolates contraction from the already-known value of Freeze-and-Solve / readout refitting.

### Results

| Dataset | Runs | Contracted | Dense MSE | Adaptive MSE | Dense readout-refit MSE | Adaptive minus dense-refit | Final parameter reduction | Active-param-step saving | Wall-clock ratio incl. controller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Digits | 4 | 4/4 | 0.07033 | 0.06860 | **0.06665** | **+0.00195** | 14.96% | 2.83% | **1.72x** |
| Cancer | 4 | 0/4 | 0.17047 | 0.17047 | n/a | n/a | 0% | 0% | **1.33x** |

Digits contraction checkpoints were 140, 140, 160, and 180; final last-hidden widths were 48, 56, 44, and 32 respectively.

Seed-level comparison on Digits:

| Seed | contraction step | width | Dense | Adaptive | Dense + readout refit | Adaptive - refit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 160 | 44 | 0.05535 | 0.04935 | 0.05059 | -0.00123 |
| 1 | 140 | 48 | 0.05683 | 0.05232 | 0.04823 | +0.00409 |
| 2 | 140 | 56 | 0.08040 | 0.06519 | 0.06641 | -0.00123 |
| 3 | 180 | 32 | 0.08872 | 0.10754 | 0.10138 | +0.00617 |

### What the controlled pilot says

1. **The earlier apparent benefit over dense AdamW was not a clean contraction effect.** Once the same ridge-readout action is given to a full-width control, dense readout-refit is better on average (`0.06665` vs `0.06860`). With four seeds this is only a diagnostic, not a statistical conclusion, but it removes any present claim that contraction improves accuracy.
2. **Contraction remains approximately competitive in three of four seeds, but one aggressive contraction is clearly harmful.** Seed 3 contracts to width 32 at step 180 and loses to both dense and dense-refit. The onset/size rule still needs work.
3. **The controller is too expensive.** It saves only `2.83%` of active parameter-steps because contraction happens late, while total CPU time including diagnostics is `1.72x` dense on Digits. This is a clean negative compute result.
4. **Cancer exposes a theoretical bug in the proxy rule.** Full-span empirical accessibility is already near one, yet the fixed rank-8 principal-angle gate remains active because weak/unresolved feature directions continue to rotate. The endogenous spectral theory says unresolved bulk eigenvectors should not be individually trusted. Measuring total rank-8 motion can therefore block contraction for the wrong reason.
5. **The finite-horizon feature-value proxy was informative but not yet a certificate.** At Digits contraction times the constant-speed accessibility-gain proxy remained around 0.029--0.041; on Cancer it was typically below 0.001 because accessibility was already saturated. This is another sign that the current hard speed gate and the target-value object are misaligned.

## 4. Design correction implied by the evidence

The next controller should not ask whether an arbitrary fixed-rank hidden subspace has stopped moving. It should ask whether **resolved target-relevant geometry still has material reachable value**.

Three changes follow directly from the theory:

1. **Resolved-subspace motion only.** Estimate motion only on modes that are empirically resolved above a null/bulk threshold; aggregate the unresolved bulk rather than tracking its individual eigenvectors.
2. **Target-value gate.** Combine current unresolved target mass with a finite-horizon motion envelope. A representation can rotate rapidly in nuisance directions while having almost no remaining target value.
3. **Contraction magnitude from value budget.** Choose the smallest structural model whose estimated finite-horizon deletion cost plus structural-transfer error is below a predeclared risk budget, rather than jumping to the first subset that passes current probe risk.

The present SRA implementation is therefore retained as a falsified first proxy, not promoted to the final algorithm.

## 5. What must happen before a serious empirical claim

The decisive quantity remains test performance versus **charged compute**. The next stage must use cheaper state estimation (small fixed reservoirs / randomized low-rank updates), compare explicitly against EPI because it is the closest prior work on pruning onset, and move to structured units whose removal creates substantial real FLOP savings. A larger experiment is justified only after the revised target-value controller beats this first proxy on the same small controlled suite.
