# Paired structural-damage UCB coverage audit

The finite-sample theorem is only useful if its upper confidence bound has the advertised coverage at practical probe sizes.

This audit uses a synthetic nonlinear regression task. Structural candidates are constructed from training data only. Dense and reduced branches are forked from the same checkpoint, receive the same ridge-readout action, and train for the same future horizon. A large independent Monte Carlo sample approximates each candidate's population **clipped-loss structural damage**.

Fresh probes of sizes 64, 128, 256, and 512 are then drawn repeatedly. For the finite candidate family, we record:

- simultaneous coverage of the paired empirical-Bernstein UCB;
- per-candidate coverage;
- mean and median UCB radius;
- true population structural-damage scale.

The important quantity is not merely whether coverage exceeds 95%. A valid bound that is much wider than the risk/compute budget is operationally useless. The audit therefore measures **coverage and tightness together**.

This is a certificate audit, not a controller benchmark. No threshold is tuned from its outcome.