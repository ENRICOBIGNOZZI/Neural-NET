# Claim ledger

This file is the guardrail for the project. A claim moves from **open** to **empirical** or **theorem** only when its evidence exists in the repository.

| Claim | Status | Evidence needed |
|---|---|---|
| Prediction-mode gradient energy is `mu_j * a_j^2` | theorem | `theory/main.tex`, exact proof |
| Fixed-geometry deletion value is `0.5 a_j^2 (1-exp(-2 mu_j H))` | theorem | exact proof + numerical audit |
| Eigenvalues alone cannot determine safe pruning | theorem | two-target counterexample |
| Removing weak modes improves the active condition-number bound | theorem | fixed quadratic subproblem |
| Dropping a nuisance mode can reduce finite-sample risk | theorem in sequence model | exact bias/variance calculation |
| Small omitted-gradient energy implies small finite-horizon trajectory distortion | conditional theorem | Lipschitz/tube assumptions; initial structural jump allowed |
| Curvature bounds finite-horizon prediction-operator path length | conditional theorem | exact loss dissipation + bounded Jacobian/Hessian |
| Operator velocity + eigengap controls future discarded-cluster energy | conditional theorem | Riesz/Davis--Kahan bound |
| A fixed structural mask tracks a moving kept spectral cluster under a parameter-Gram gap | conditional theorem | curvature + Riesz-projector bound |
| Final hidden rank deficiency allows exact current neuron-basis contraction | theorem | spanning-set argument |
| Quadratic null bulk contracts exactly as `b_+(t)=lambda_+/(1+4 t lambda_+)` | exact in quadratic model | exact Riccati solution + MP pushforward |
| The dynamic-BBP onset is the unique solution of `chi_{theta,gamma}(t)=1` | exact in quadratic model | monotonicity of exact edge-strength function |
| Safe contraction should not start before target spikes become resolvable | exact in quadratic model | dynamic BBP theorem/corollary |
| After all target spikes separate, the target becomes asymptotically `k`-dimensional with geometry error `sum theta_q^2(1-omega_q^2)` | exact in quadratic model | covariance-to-prediction bridge + target-overlap theorem |
| Premature rank-one contraction can create a logarithmic-in-d target discovery delay | exact in rank-one quadratic model | exact overlap-odds dynamics + isotropic initialization |
| Repeated adaptive contraction can be probe-safe with fresh independent probes | theorem | finite candidate grids + clipped loss |
| The empirical controller transfers across architectures | open | held-out architecture experiments |
| The controller saves total wall-clock at matched performance | open | compute-charged benchmarks |
| Internal nonlinear layers can always be contracted from prediction spectrum alone | false / not claimed | structural realization gap prevents this |

## Non-claims

- We do **not** claim that every small eigenvalue slows every other mode.
- We do **not** equate weight spectra with prediction spectra.
- We do **not** call a step-count reduction a speedup if diagnostics consume the saved compute.
- We do **not** use test labels to trigger contraction.
- We do **not** interpret an unresolved empirical bulk eigenvector as an identifiable direction.
- We do **not** claim that a BBP threshold is a universal pruning threshold outside the solvable phase model.
