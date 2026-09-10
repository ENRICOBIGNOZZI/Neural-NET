# Oracle SVD rank frontier protocol

A binary 8-to-2 pruning test can hide the actual phenomenon. The relevant object is the smallest rank that can be adopted at each training checkpoint without paying more than a prescribed finite-horizon risk tolerance.

At checkpoint t, the rank-8 student is copied and its contractible hidden matrix is recompressed by truncated SVD to every candidate rank r=1,...,7. Each candidate is then trained for the same horizon H as the dense continuation. Rank 8 is included as the zero-damage reference. For candidate r define

`D_r(t,H) = R_r(t+H) - R_dense(t+H)`.

A candidate is safe when

`D_r(t,H) <= max(0.001, 0.05 * R_dense(t+H))`.

The retrospective oracle minimal safe rank is

`r_star(t) = min { r : D_r(t,H) is safe }`.

Because a physical architecture should not need to grow again after permanent contraction, we also construct the irreversible oracle frontier

`r_irrev(t) = max_{s >= t} r_star(s)`.

This is the smallest rank that could have been adopted at t while remaining sufficient at every later measured checkpoint. Its first drop below rank 8 is the cleanest ground-truth definition of contraction onset in this experiment.

For every candidate rank we record the singular-value tail fraction of the current contractible weight, immediate and finite-horizon test damage, parameter savings, and same-step path value relative to a compact-from-start control obtained by SVD-recompressing the identical initial wide network. At the checkpoint level we also record target-weighted tangent bulk mass, gradient energy, finite-horizon value, and eigengap.

This experiment is deliberately oracle only in the timing/rank label: truncated SVD itself uses no targets, while `r_star(t)` is defined using future test outcomes. The resulting rank frontier is therefore ground truth for the next stage: learning a leakage-free observable rule that predicts when and how far the network may be contracted.
