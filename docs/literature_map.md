# Literature map and novelty boundary

The project must not be sold as **dynamic pruning**, **adaptive rank**, or even merely **when to prune**. All three already have substantial prior art. The novelty target is the sharper chain

`endogenous target-weighted prediction geometry -> finite-horizon capacity value -> phase/discovery onset -> certified or forecast structural contraction`.

The strongest conceptual claim is not that a smaller network can train well. It is that a learned representation can pass from a **discovery phase**, in which aggressive contraction destroys or delays unresolved target directions, into a **compressible phase**, in which target signal is concentrated in a resolved low-dimensional prediction subspace. The solvable quadratic model makes this transition exact through dynamic BBP separation and target overlap.

## Closest method families

| Work / family | What it does | Why it is not the same claim |
|---|---|---|
| **When To Prune? / EPI** (CVPR 2022) | Explicitly asks when structural pruning should start. It monitors stability of the dominant pruned architecture across consecutive epochs and reports substantial training-cost reduction. | **Closest prior work on the onset question.** Its onset signal is architecture/mask stability. It does not derive a target-weighted prediction-space phase transition, finite-horizon deletion value, future feature-discovery bound, or a theorem identifying a low-dimensional target-sufficient phase. Any paper from this project must compare against EPI directly. |
| Early-Bird Tickets (ICLR 2020) | Detects early stabilization of channel-pruning masks and switches to a sparse subnetwork. | Same broad philosophy of initial dense discovery followed by sparse training; the criterion is mask stability rather than target-relevant learned geometry. |
| RigL / Dynamic Sparse Training | Drops small-magnitude connections and regrows high-gradient connections during sparse training. | Topology/sparsity dynamics, not a target-weighted prediction-spectrum value theorem. Regrowth also means the sparse support remains exploratory rather than an irreversible post-separation contraction certificate. |
| Structured RigL (ICLR 2024) | Hardware-friendly structured dynamic sparse training. | Essential structured-compute baseline; salience and sparsity constraints are not a finite-horizon spectral onset certificate. |
| EigenDamage (ICML 2019) | Structured pruning in a K-FAC/Hessian eigenbasis. | Curvature-aware checkpoint damage estimation. It is not a theory of endogenous feature discovery, target-weighted prediction modes, or when a mode loses reachable future value. |
| Spectral pruning of fully connected layers (Scientific Reports 2022) | Uses spectral parameterization/eigenvalues to rank removable nodes. | Close to the word “spectral pruning,” but uses internal spectral structure rather than the target-weighted prediction operator and dynamic phase/onset theory here. |
| NTK-SAP (ICLR 2023) | Foresight pruning designed to preserve the NTK spectrum/training dynamics, mainly before training. | It tries to preserve dense dynamics after pruning. Here the goal is opposite after discovery: deliberately discard capacity whose current **and future** target value has become negligible. It also highlights why a fixed initial NTK cannot represent later feature-learning geometry. |
| Dynamic Parameter Rank Pruning (2024/2025) | Trains SVD-factorized CNN layers and dynamically reduces parameter rank using regularization. | Directly relevant dynamic-rank baseline; its rank is a weight-matrix property, not a target-weighted prediction-space value or phase-separation certificate. |
| Dynamic Rank Adjustment (2025) | Interleaves low- and full-rank training because effective weight rank changes/collapses during training. | Strong adaptive-rank baseline, but the objective is to maintain useful weight rank and full-rank expressivity rather than identify a target-sufficient prediction subspace and irreversibly contract after discovery. |
| AdaLoRA (2023) | Adaptive budget/rank allocation for LoRA updates. | Adaptive low-rank PEFT; operates on update rank rather than full-network structural contraction. |
| FARSS (ACL Findings 2026) | Fisher-guided adaptive LoRA rank plus task-aware singular-vector selection from activation/gradient second-order statistics. | Important recent target/task-aware low-rank baseline. It remains PEFT around a pretrained model and allocates update rank; it does not supply the feature-learning onset/compressibility theorem sought here. |
| GaLore (ICML 2024) | Low-rank gradient projection to reduce optimizer memory while retaining full parameters. | Gradient-memory device; the deployed/trained architecture itself is not contracted. |
| SPARCS (npj AI 2025) | Spectral architecture search using spectra of inter-layer transfer matrices and continuous differentiable architecture manifolds. | A spectral route to architecture design/search, but not target-weighted finite-horizon deletion value or an endogenous dense-to-small training transition. Conceptually, it explores architecture space; this project asks when learned target geometry makes capacity disposable. |

## The closest challenge to our framing

**EPI is the paper that prevents us from claiming novelty for the sentence “learn densely for a while, then detect when it is safe to prune.”** That problem was explicitly posed and solved empirically in CVPR 2022 using dominant-subnetwork stability.

The defensible distinction must therefore be mathematical and target-specific:

1. **What is being stabilized?** EPI stabilizes a proposed pruned architecture. We track the prediction subspace and target mass.
2. **Why is pruning safe?** EPI uses empirical architecture stability. We seek an explicit bound on lost finite-horizon target value plus structural-transfer error.
3. **Why not prune earlier?** In the solvable model, a target mode is provably hidden in the bulk before dynamic BBP separation; blind rank-one contraction then incurs a `log(d)` discovery delay. This is a mechanism, not only an empirical timing observation.
4. **What becomes small?** After separation, the exact model becomes target-compressible: the unresolved geometry error of the `k` resolved spike directions is `sum_q theta_q^2 (1-omega_q(t)^2) -> 0`.
5. **What is the algorithmic object?** A future-value/onset estimator around an unchanged base optimizer, followed by physically realizable structured contraction.

If these five points do not survive larger models, the project collapses back into an incremental early-pruning method. This is the correct falsification standard.

## Sources checked

- Shen, Molchanov, Yin, Alvarez, *When To Prune? A Policy Towards Early Structural Pruning*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html
- Early-Bird Tickets, ICLR 2020: https://github.com/GATECH-EIC/Early-Bird-Tickets
- RigL official implementation: https://github.com/google-research/rigl
- Structured RigL / ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c5f30296296d2ae402ebbd09aaa9c12-Abstract-Conference.html
- EigenDamage, ICML 2019: https://proceedings.mlr.press/v97/wang19g.html
- Spectral pruning: https://www.nature.com/articles/s41598-022-14805-7
- NTK-SAP, ICLR 2023: https://arxiv.org/abs/2304.02840
- Dynamic Parameter Rank Pruning: https://arxiv.org/abs/2401.08014
- Dynamic Rank Adjustment: https://arxiv.org/abs/2508.08625
- AdaLoRA: https://arxiv.org/abs/2303.10512
- FARSS, ACL Findings 2026: https://aclanthology.org/2026.findings-acl.883/
- GaLore, ICML 2024: https://proceedings.mlr.press/v235/zhao24s.html
- SPARCS, npj Artificial Intelligence 2025: https://www.nature.com/articles/s44387-025-00039-1

## Strongest novelty formulation currently defensible

> **When does feature learning make overparameterization disposable?** Existing methods choose sparse/rank structure from weights, gradients, curvature, masks, Fisher information, or fixed/initial kernel information. We study when an endogenously learned prediction-space sector has so little current and reachable future target value that removing it is provably low-cost, and when a nonlinear training phase transition makes a low-dimensional target subspace identifiable in the first place.

The strongest empirical message, if it survives, is correspondingly narrower than “better pruning”:

> **Overparameterize while the network is discovering target directions; structurally contract only after the learned prediction geometry says those directions have separated and the remaining capacity has negligible future value.**
