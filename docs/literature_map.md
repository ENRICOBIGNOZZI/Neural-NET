# Literature map and novelty boundary

The project must not be sold as "dynamic pruning" or "adaptive rank". Those ideas already exist. The novelty target is the chain

`target-weighted prediction geometry -> finite-horizon capacity value -> onset certificate -> structural contraction`.

## Closest method families

| Work / family | What it does | Why it is not the same claim |
|---|---|---|
| RigL / Dynamic Sparse Training | Drops small-magnitude connections and regrows high-gradient connections during sparse training | topology/sparsity dynamics, not a target-weighted prediction-spectrum value theorem |
| Structured RigL (ICLR 2024) | Hardware-friendly structured DST plus neuron ablation | very important practical baseline; salience and sparsity constraints are not a finite-horizon spectral onset certificate |
| Early-Bird Tickets (ICLR 2020) | Detects early stabilization of pruning masks and starts sparse training early | directly relevant to **when** pruning can start, but uses mask stability rather than target-weighted prediction geometry |
| EigenDamage (ICML 2019) | Structured pruning in a K-FAC/Hessian eigenbasis | curvature-aware compression, mainly pruning damage at a checkpoint; not endogenous target-weighted feature-discovery value |
| Spectral pruning of fully connected layers (Scientific Reports 2022) | Uses spectral parameterization/eigenvalues to rank nodes | close to spectral pruning, but does not establish the prediction-space target-alignment and dynamic-onset theory pursued here |
| NTK-SAP (2023) | Prunes connections to preserve the NTK spectrum, mainly before training | preserves dense training dynamics; our goal is to deliberately discard low-value capacity after feature learning separates signal from nuisance |
| AdaLoRA (2023) | SVD-style adaptive budget allocation for LoRA updates | adaptive low-rank PEFT; operates on update rank/importance rather than full-network prediction-space finite-horizon value |
| GaLore (ICML 2024) | Low-rank gradient projection to reduce optimizer memory while retaining full-parameter learning | does not shrink the model itself; rank is a gradient-memory device rather than a target-value deletion decision |
| Dynamic Rank Adjustment (2025) | Interleaves low/full rank training because weight effective rank evolves | empirically adaptive weight rank; strongest recent rank baseline, but uses weight-rank behavior rather than target-weighted prediction geometry and a safe-onset theorem |

## Sources checked

- RigL official implementation: https://github.com/google-research/rigl
- Structured RigL / ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c5f30296296d2ae402ebbd09aaa9c12-Abstract-Conference.html
- Early-Bird Tickets: https://github.com/GATECH-EIC/Early-Bird-Tickets
- EigenDamage: https://proceedings.mlr.press/v97/wang19g.html
- Spectral pruning: https://www.nature.com/articles/s41598-022-14805-7
- NTK-SAP: https://arxiv.org/abs/2304.02840
- AdaLoRA: https://arxiv.org/abs/2303.10512
- GaLore: https://proceedings.mlr.press/v235/zhao24s.html
- Dynamic Rank Adjustment: https://arxiv.org/abs/2508.08625

## Strongest novelty formulation currently defensible

> Existing methods choose sparse/rank structure using weights, gradients, curvature, masks, or fixed NTK information. This project asks a different question: when does a prediction-space capacity sector have so little **current and reachable future target value** that physically removing it is guaranteed or empirically predicted not to matter?

This formulation stays narrow enough to survive comparison with prior work.
