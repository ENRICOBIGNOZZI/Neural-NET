# Development versus confirmatory protocol

The empirical controller is now being changed in response to seeds 0--3 on Digits and Cancer. Those runs are therefore **development data forever**. Their test labels may diagnose failure modes, but they cannot support a final performance claim.

## Development set

- Datasets: scikit-learn Digits even/odd and Wisconsin breast cancer.
- Seeds: 0, 1, 2, 3.
- Width: 64; 200 AdamW updates.
- Controllers already inspected: total-subspace-speed SRA and target-value SRA.

All thresholds, structural selection rules, monitoring approximations, and compute optimizations may be changed while using these runs.

## Freeze condition

Before any confirmatory test, the following must be committed in one immutable protocol file:

1. controller source commit SHA;
2. all hyperparameters and risk budgets;
3. candidate structural units and maximum event size;
4. checkpoint/probe schedule;
5. dense, readout-refit, optimizer-reset, EPI, and structural-pruning baselines;
6. primary performance tolerance and compute metric;
7. confirmatory datasets/configurations/seeds;
8. statistical aggregation and confidence-interval procedure.

## Confirmatory rule

The first confirmatory suite must use seeds and/or configurations that have never been inspected during controller development. No controller threshold may be changed after viewing confirmatory test outcomes. If the controller is changed, a new confirmatory suite must be declared.

This is especially important here because the decision time itself is adaptive and small benchmark suites make post-hoc tuning deceptively easy.
