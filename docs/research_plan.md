# Research plan

## Paper question

**When is a neural network ready to become smaller during training?**

The intended answer is not "when weights are small." It is: when a target-irrelevant prediction sector has low finite-horizon value, is statistically resolved as nuisance rather than hidden signal, has low future motion/discovery potential, and admits a low-distortion structural realization.

## Milestone A — theory (now)

- [x] exact modal gradient-energy identity;
- [x] exact fixed-geometry finite-horizon deletion value;
- [x] exact value-per-compute oracle;
- [x] active conditioning result;
- [x] finite-sample sequence-model drop criterion;
- [x] nonlinear projected-flow stability theorem;
- [x] eigengap/operator-velocity persistence theorem;
- [x] structural-transfer theorem;
- [x] exact current final-hidden neuron-basis theorem;
- [x] phase-aware onset corollary from dynamic BBP separation;
- [x] exact rank-one post-contraction target-angle dynamics and discovery-delay law;
- [x] derive a general curvature-to-`dot L` path-length bound;
- [ ] sharpen it with architecture-specific constants/state evolution for a standard MLP;
- [ ] derive a cluster-level empirical estimator that never identifies unstable eigenvectors inside the bulk;
- [x] finite-sample adaptive-probe theorem for repeated contraction decisions with fresh probes;
- [ ] reusable-probe/cross-fitting theorem with lower sample overhead.

## Milestone B — first algorithm

The first controller is intentionally conservative and only contracts the last hidden layer.

At checkpoint `t`:

1. compute final hidden features on an independent probe;
2. estimate target accessibility and effective rank;
3. estimate principal-angle speed relative to the previous checkpoint;
4. construct a target-aware neuron ordering on training data;
5. use the independent probe to choose the smallest prefix preserving current accessibility and frozen-readout probe risk;
6. require slow subspace motion for multiple checkpoints;
7. physically rebuild the layer at the smaller width;
8. refit the linear readout on training features, rebuild optimizer state, and continue training; report dense optimizer-reset and dense readout-refit controls separately.

This is a **proxy** controller, not yet the fully certified spectral controller.

## Milestone C — decisive experiments

1. synthetic fixed-geometry identity audit;
2. quadratic teacher--student early-vs-late contraction;
3. Digits and Cancer dense-vs-contracted MLP;
4. CIFAR ResNet channel contraction;
5. transformer width/rank contraction.

## Paper-killing tests

Stop or substantially revise the algorithmic claim if:

- onset is not predictable without test leakage;
- structural contraction consistently destroys future features despite low measured motion;
- controller overhead erases compute savings;
- static small models match the adaptive method, meaning the discovery phase is unnecessary;
- strong structured sparse/rank baselines dominate across matched compute.
