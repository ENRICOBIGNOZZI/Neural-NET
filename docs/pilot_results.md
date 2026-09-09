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

This audit matches the exact Riccati flow implemented in `src/neural_net/quadratic_phase.py`. The important result is not a fitted slope: it is the order-one post-separation target overlap versus the `O(1/d)` blind overlap and the resulting large gap in target-discovery time.

## 3. First controlled real-data CPU pilot: total-subspace-speed SRA

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

### What the first controlled pilot says

1. **The earlier apparent benefit over dense AdamW was not a clean contraction effect.** Once the same ridge-readout action is given to a full-width control, dense readout-refit is better on average (`0.06665` vs `0.06860`).
2. **Contraction remains approximately competitive in three of four seeds, but one aggressive contraction is clearly harmful.** Seed 3 contracts to width 32 at step 180 and loses to both dense and dense-refit.
3. **The controller is too expensive.** It saves only `2.83%` of active parameter-steps because contraction happens late, while total CPU time including diagnostics is `1.72x` dense on Digits.
4. **Cancer exposes a theoretical bug in the proxy rule.** Full-span empirical accessibility is already near one, yet the fixed rank-8 principal-angle gate remains active because weak/unresolved feature directions continue to rotate. Measuring total rank-8 motion can therefore block contraction for the wrong reason.
5. **The finite-horizon feature-value proxy and total-subspace speed are misaligned.** At Digits contraction times the constant-speed accessibility-gain proxy remained around `0.029--0.041`; on Cancer it was typically below `0.001` because accessibility was already saturated.

The first SRA implementation is therefore retained as a falsified proxy, not promoted to the final algorithm.

## 4. Second development pilot: target-value SRA (TV-SRA)

The theory was changed before the second pilot. An exact counterexample now shows that total subspace speed can be arbitrarily large while target accessibility is exactly constant. TV-SRA therefore removes the hard fixed-rank Grassmann-speed gate and monitors the motion of the **target-accessibility angle** in a predeclared leading singular subspace. It also limits a single contraction event to at most 25% of the current width. The observed target-angle speed is still only a prospective proxy, not a finite-horizon certificate.

The same four development seeds, data splits, AdamW settings, ridge action, optimizer-reset control, and dense readout-refit control were used. These seeds have now been inspected and are permanently designated **development** rather than confirmatory evidence.

### Aggregate results

| Dataset | Contracted | Mean step | Dense MSE | TV-SRA MSE | Dense readout-refit MSE | TV-SRA minus refit | Final parameter reduction | Active-param-step saving | Wall-clock ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Digits | 4/4 | 140 | 0.07033 | **0.06684** | 0.07077 | **-0.00394** | 11.02% | 3.15% | **1.45x** |
| Cancer | 4/4 | 130 | 0.17047 | 0.17558 | 0.18326 | **-0.00768** | 14.88% | 5.10% | **1.32x** |

TV-SRA fixes the pathological “never contract on Cancer” behavior and moves onset earlier. On Digits it is better on average than both uninterrupted dense training and the matched dense readout-refit control in these four development runs. On Cancer it is better than the readout-refit control but slightly worse than uninterrupted dense training on average. The sample is far too small to call either pattern a result.

### Seed-level results

| Dataset | Seed | Step | Final width | Dense | TV-SRA | Dense refit | TV-SRA - dense | TV-SRA - refit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Digits | 0 | 160 | 48 | 0.05535 | 0.04989 | 0.05059 | -0.00546 | -0.00070 |
| Digits | 1 | 120 | 48 | 0.05683 | 0.05535 | 0.06813 | -0.00148 | -0.01278 |
| Digits | 2 | 120 | 56 | 0.08040 | 0.06917 | 0.06991 | -0.01124 | -0.00075 |
| Digits | 3 | 160 | 48 | 0.08872 | 0.09295 | 0.09446 | +0.00423 | -0.00152 |
| Cancer | 0 | 80 | 48 | 0.22641 | 0.18901 | 0.19497 | -0.03740 | -0.00596 |
| Cancer | 1 | 160 | 48 | 0.10627 | 0.08033 | 0.09474 | -0.02594 | -0.01441 |
| Cancer | 2 | 160 | 48 | 0.18464 | 0.22679 | 0.20934 | +0.04215 | +0.01745 |
| Cancer | 3 | 120 | 56 | 0.16456 | 0.20619 | 0.23399 | +0.04163 | -0.02781 |

At the contraction checkpoints, resolved target accessibility is already high (`0.867--0.981`) and the 40-update constant-speed accessibility-gain proxy lies between roughly `0.0007` and `0.0050`. Candidate structural reductions lose at most about one percentage point of resolved accessibility by construction.

### What changed, and what did not

The change in the onset statistic matters. It allows contraction on Cancer and modestly increases active-parameter-step savings. More importantly, it is aligned with the target-value theorem rather than with arbitrary nuisance rotation. But the central engineering problem remains: **monitoring still costs more than the saved CPU compute**. The second controller is `1.32--1.45x` dense wall-clock on this tiny implementation. A paper cannot call this efficient.

The mixed Cancer seeds also show that high current accessibility plus small recent target-angle motion is not sufficient to guarantee benign future nonlinear training. This is exactly the gap between the empirical proxy and the master theorem: the latter also requires a structural mismatch term and a valid upper bound on future target-relevant geometry motion.

## 5. Design correction implied by both pilots

The next version should be a risk-budgeted controller rather than another hand-tuned threshold rule. It should:

1. identify only **resolved target-bearing** spectral clusters and aggregate the unresolved bulk;
2. estimate current structural damage and remaining future target value in the same risk units;
3. choose contraction magnitude from a predeclared risk budget rather than “first passing subset”;
4. reduce monitoring overhead using a small fixed probe reservoir and sparse checkpoints;
5. treat EPI as the mandatory onset baseline because it is the closest prior method on *when* to prune.

No further parameter tuning should use seeds 0--3 as confirmatory evidence. They are development runs. Once a risk-budgeted controller is frozen, new seeds/configurations must be specified before looking at test outcomes.

## 6. Current empirical status

The theory is substantially stronger than the current algorithm. The exact quadratic model now gives a genuine dense-discovery-to-compressible-phase theorem and a quantitative penalty for blind premature contraction. The general moving-geometry theory gives a modular finite-horizon structural risk bound. Empirically, target-value onset is more plausible than total-subspace-speed onset, but **no compute-matched speedup has been demonstrated** and the present structural realization is only the final hidden layer of a small MLP.
