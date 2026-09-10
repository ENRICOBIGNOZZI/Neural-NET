# SVD structural bridge protocol

The strict oracle sweep showed that choosing a subset of the existing rank-one factors is not a reliable realization of spectral contraction. The learned rank-8 matrix can distribute the same useful low-dimensional map across many non-orthogonal factors, so deleting six factor coordinates may damage the network even when the effective matrix itself is nearly low rank.

This experiment separates those issues. At every checkpoint the contractible hidden weight is formed explicitly and recompressed by truncated SVD. For target rank r this is the Eckart--Young optimal rank-r approximation in squared Frobenius norm. The result is then refactorized into a genuinely smaller trainable rank-r module and training continues for the same finite horizon.

For each seed, checkpoint, and target rank r in {2,3,4}, we compare four paths: dense rank 8, SVD recompression, the best held-out-target subset of r existing rank factors, and a deterministic random subset. A compact-from-start SVD control is obtained by SVD-recompressing the identical initial dense network, so at checkpoint zero its continuation is exactly the same starting compact model as the wide-then-SVD path.

The main structural diagnostic is the singular-value tail fraction

`sum_{j>r} sigma_j^2 / sum_j sigma_j^2`,

which equals the relative squared Frobenius error of the SVD recompression. We test whether it predicts future network-risk damage, whether SVD contraction is safer than deletion in the learned factor coordinates, and whether the target-weighted tangent bulk value adds information beyond weight low-rankness.

A contraction is marked safe when future test-risk damage relative to dense continuation is no larger than max(0.001, 5% of dense future risk). The persistent safe onset is computed separately for each target rank. This remains a structural diagnostic rather than a deployable method: the bulk split uses the known teacher rank, and the oracle component-subset baseline uses held-out targets. SVD itself uses no target labels at contraction time.
