# Equal-compute oracle falsification protocol

The strict onset experiment asks whether overparameterized training creates a better state before safe contraction. This experiment asks the harder question: is that benefit still present after the compact-from-start control is given at least the same cumulative rank-dependent matrix-multiplication compute?

For a rank-r factorized MLP, the analytic forward matmul MAC proxy per sample is

`input_dim * width1 + width1 * r + r * width2 + width2`.

For the default 5-12-12 network this is 264 MACs at rank 8 and 120 MACs at rank 2. At checkpoint t with a post-contraction horizon H, the wide path therefore receives compute proxy

`C_wide(t,H) = 264 * t + 120 * H`.

The compact control is initialized with the same held-out oracle rank-2 subset used by the strict protocol and is trained for

`ceil(C_wide(t,H) / 120)`

steps. The ceiling deliberately gives the compact control weakly more compute than the wide path. The primary statistic is

`R_wide_then_contract - R_compact_equal_compute`.

Negative values favor wide-then-contract; positive values favor compact-from-start. Checkpoint zero is an invariant check and should be equal up to numerical precision. This remains an oracle experiment because structural subset selection uses held-out labels. The MAC count is an analytic compute proxy, not measured hardware FLOPs; later experiments must repeat the comparison against cumulative measured wall time and accelerator FLOPs.
