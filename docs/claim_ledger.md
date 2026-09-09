# Claim ledger

This file is the guardrail for the project. A claim moves from **open** to **empirical** or **theorem** only when its evidence exists in the repository.

| Claim | Status | Evidence needed |
|---|---|---|
| Prediction-mode gradient energy is `mu_j * a_j^2` | theorem | `theory/main.tex`, exact proof |
| Fixed-geometry deletion value is `0.5 a_j^2 (1-exp(-2 mu_j H))` | theorem | exact proof + numerical audit |
| Eigenvalues alone cannot determine safe pruning | theorem | two-target counterexample |
| Removing weak modes improves the active condition-number bound | theorem | fixed quadratic subproblem |
| Dropping a nuisance mode can reduce finite-sample risk | theorem in sequence model | exact bias/variance calculation |
| Small omitted-gradient energy implies small finite-horizon trajectory distortion | conditional theorem | Lipschitz/tube assumptions |
| Operator velocity + eigengap controls future discarded-cluster energy | conditional theorem | Riesz/Davis--Kahan bound |
| Final hidden rank deficiency allows exact current neuron-basis contraction | theorem | spanning-set argument |
| Safe contraction should not start before target spikes become resolvable | exact in quadratic model | dynamic BBP theorem/corollary |
| The empirical controller transfers across architectures | open | held-out architecture experiments |
| The controller saves total wall-clock at matched performance | open | compute-charged benchmarks |
| Internal nonlinear layers can always be contracted from prediction spectrum alone | false / not claimed | structural realization gap prevents this |

## Non-claims

- We do **not** claim that every small eigenvalue slows every other mode.
- We do **not** equate weight spectra with prediction spectra.
- We do **not** call a step-count reduction a speedup if diagnostics consume the saved compute.
- We do **not** use test labels to trigger contraction.
- We do **not** interpret an unresolved empirical bulk eigenvector as an identifiable direction.
