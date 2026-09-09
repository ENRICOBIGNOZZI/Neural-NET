# Finite-horizon structural-value audit

This experiment measures the empirical object that the master theorem bounds.

At a dense checkpoint `t`, two branches are created from the same state and receive the same terminal readout action and optimizer reset:

- full-width continuation;
- structural last-hidden-layer contraction followed by continuation.

After horizon `H`, define probe structural damage

`D_probe(t,H,S) = R_probe,reduced(t+H) - R_probe,dense(t+H)`.

The controller should ultimately predict an upper confidence bound for this quantity from information available at `t`. Test labels are recorded only for ex-post transfer auditing and never enter the forecaster.

The first development audit uses Digits and Cancer, seeds 0--3, checkpoints 80/120/160, horizon 40, and candidate retained-width fractions 0.75 and 0.875. These are development configurations, not confirmatory evidence.

The main comparison is whether a target-value state (resolved accessibility, target-angle motion, current structural damage, effective rank, and candidate size) forecasts future probe damage better under leave-one-trajectory-out validation than time/size or current damage/time/size alone.
