# Strict oracle onset protocol

This experiment isolates the timing question before attempting a deployable controller.

The dense student has rank 8 and the teacher has rank 2. At initialization, the compact-from-start control is given the same held-out oracle subset search that the later contraction receives. Therefore the checkpoint-zero wide-then-contract path and the compact-from-start path begin from the same rank-2 subnetwork. Any later advantage of wide-then-contract must come from training while overparameterized, not from choosing a luckier initial subset.

At each checkpoint we compare three continuations for the same horizon: dense rank 8, oracle-selected rank 2, and a deterministic random rank-2 contraction. We also track the tangent-spectrum bulk diagnostics, including eigengap, target mass, gradient energy, and fixed-geometry finite-horizon value.

A contraction is marked safe when its future test-risk increase over the dense continuation is no larger than max(0.001, 5% of the dense future test risk). The persistent safe onset is the earliest checkpoint for which this condition remains true at every later measured checkpoint.

The main quantities are future structural damage, width-path value relative to the strict compact-from-start control, oracle gain over random contraction, and parameter-step savings. This protocol remains oracle because subset selection uses held-out targets and the bulk split uses the known teacher rank; it is intended to establish whether a meaningful contraction window exists before replacing the oracle pieces with observable estimators.
