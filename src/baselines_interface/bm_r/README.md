# BM-r baseline (Chiu & Zhang, ICML 2023)

"Tight Certification of Adversarially Trained Neural Networks via Nonconvex Low-Rank
Semidefinite Relaxations" (arXiv:2211.17244). Full Python reimplementation of
`bm_linf.m`/`bm_l2.m` ([Hong-Ming/BM-r](https://github.com/Hong-Ming/BM-r)) -- no MATLAB
on this machine, and no Knitro license confirmed available, so MATLAB+Knitro was never
an option; see `baselines/README.md` for that decision. Runs in the dedicated
`baselines` conda env (`cyipopt`, installed via conda-forge) -- torch/MOSEK stay in
`certif`, same two-env split as `sdp_fo/` and for the same reason (archived/pinned
dependency incompatible with the rest of the project).

## What the paper does

BM-r solves the *same* SDP relaxation this project's `SDP-IP` config already computes
via MOSEK (Raghunathan et al. 2018, no chordal decomposition, all neurons kept as
variables) -- its contribution is a **different, more scalable way to solve it**, not a
different/tighter relaxation:

1. **Low-rank factorization** (Burer & Monteiro 2003): instead of optimizing the full
   `(n+1)x(n+1)` PSD matrix `P`, factor `P = UU'` with `U` of shape `(n+1, r)`, `r << n`.
   This turns the SDP into a **nonconvex but much smaller** NLP (`O(nr)` variables
   instead of `O(n^2)`), solved by a general-purpose NLP solver (Knitro in the
   original, with an **exact** Hessian supplied, not quasi-Newton).
2. **Riemannian staircase** (Boumal et al.): solve at rank `r`, then check whether the
   solver's own KKT multipliers `(y, z)` certify the point as the SDP relaxation's
   **global** optimum -- concretely, whether the dual slack matrix `S(y,z)` (built
   purely from those multipliers, matrix-free) has a non-negative smallest eigenvalue.
   If yes: done, with a mathematical proof, not just a converged NLP. If no: the
   eigenvector of the most negative eigenvalue gives a **provable escape direction**
   (Theorem 4.4) that strictly decreases the objective to second order; append it as a
   new column, increment `r`, repeat. Theory (Boumal et al. 2020) guarantees this
   closes within `r = O(sqrt(n))` rank increments.
3. Crucially, **any** feasible dual multipliers give a valid, if possibly loose, lower
   bound on the true QCQP (Proposition 3.1: `phi_lb = z0 + R^2 * min(0, lambda_min(S))`)
   -- so the method still returns *something* useful even before/without full
   certification, just more conservative the further `lambda_min` is from 0.

## Code map

- `problem.py` -- builds the flat QCQP data (`Wmat`, `Smat`, `c`/`c0`, bounds) from this
  project's `network.extract_weights()` + `bounds.py`, for either norm. Equivalent of
  `utils/get_arguments.m`/`check_inputs.m`.
- `bm_objective.py`, `bm_constraints.py` -- objective/constraints + their Jacobian, for
  the rank-`r` factorized NLP (`mycon`/`myobj`). Vectorized numpy/scipy.sparse, no
  per-neuron Python loops. Jacobian sparsity is built **analytically** (never inferred
  from evaluating the Jacobian at some point) -- see the comments in
  `jacobian_structure` for why that approach was tried first and broke.
- `bm_hessian.py` -- the **exact** Lagrangian Hessian (`myhess`/`myhess_mv`), required
  in practice (see "Known issue" below) -- same closed-form structure as
  `dual_certificate.S_mv`, replicated per rank-column.
- `ipopt_backend.py` -- the only file that imports `cyipopt`; wraps the above into its
  `Problem` callback interface, including dual warm-starting across ranks.
- `dual_certificate.py` -- `S(y,z)`'s matrix-free matvec, smallest-eigenvalue check
  (Lanczos via `scipy.sparse.linalg.eigsh`, dense fallback), and the valid-lower-bound
  formula (Prop. 3.1). Independently re-derived and finite-difference-verified against
  this package's own `constraints()` (not a line-by-line port of `bm_linf.m`'s "generic
  A,B" abstraction layer, whose own definition isn't fully shown in the source).
- `riemannian_staircase.py` -- the outer loop (Algorithm 1): solve, check certificate,
  escape + re-solve at `r+1`, up to a rank budget.
- `instance_io.py`, `export_instances.py` (certif env), `run_bm_r.py` (baselines env) --
  the yaml-driven CLI pipeline, split across the env boundary like `sdp_fo/`.

## Differences from the original (`bm_linf.m`/`bm_l2.m`)

| | Original | This port | Why |
|---|---|---|---|
| Language / NLP solver | MATLAB + Knitro (exact Hessian) | Python + IPOPT (exact Hessian) | No MATLAB or Knitro license on this machine |
| Bounds | Trivial `[0,1]x[0,inf)` by default, or an expensive per-neuron LP bound propagation (`utils/get_bound_*.m`) | This project's own precomputed tight bounds (`bounds.py`) | Same "our bounds, their relaxation/solver" convention as every other baseline here (`sdp_fo/README.md`) -- apples-to-apples comparison, and far cheaper than their LP-based bounds |
| Warm start | PGD attack result (`pgd_linf.m`/`pgd_l2.m`), forward-propagated | Perturbation ball's center, forward-propagated | Simpler; still exactly feasible (correctness doesn't depend on which point), but likely a weaker starting point than an adversarially-informed one -- porting PGD is a plausible follow-up (see below) |
| Hessian | Exact, from the start | **Tried L-BFGS first, switched to exact** | See "Known issue" -- L-BFGS was not precise enough for the certificate |

## Known issue: the certificate does not reliably close

**What is solidly verified:**
- `bm_objective.py`/`bm_constraints.py`'s gradient and Jacobian match finite differences
  to ~1e-10 (synthetic random networks, both norms).
- `bm_hessian.py` matches a finite-difference Hessian to ~1e-10.
- `dual_certificate.py`'s `s`/`S_mv` match a finite-difference check of the full
  Lagrangian's gradient to ~1e-10 (caught and fixed a missing factor of 2 along the
  way -- a first version passed a *weaker* check at a near-converged point with
  near-zero multipliers, where the bug was numerically invisible; a check with
  *random* multipliers exposed it immediately. Lesson: validate dual/Hessian code
  against random multipliers, not just a converged solve's -- near-zero multipliers
  make almost any formula's error negligible).
- **The primal value matches this project's own `SDP-IP` (MOSEK) solve exactly** on a
  real instance (`blob_1x2`, both tested targets): `-1.593524` vs `-1.5935239`
  (6 decimal places), `-1.314046` for the second target. This is strong evidence the
  NLP is solving the *right* problem to the *right* answer.

**What does not work:** the dual certificate (`lambda_min(S(y,z))`) does not reach ~0
as the Riemannian staircase increases rank, even though the theory guarantees it
should within a small number of rank increments for a problem this size (`nx=4`,
8 constraints -> theoretical bound ~4). On `blob_1x2`/target 0, `lambda_min` improves
from -0.0243 (rank 2) to -0.0171 (rank 3) and then **plateaus exactly** through rank 7
-- the escape/rank-increase mechanism visibly does *something* but stops improving far
short of 0.

**Diagnostics tried, each ruling out one hypothesis:**
1. Lanczos (`eigsh`) vs dense `eigh` on the same `S(y,z)`: **identical** eigenvalue --
   not an ARPACK convergence issue.
2. Escape step scale 0.01 through 2.0: **no effect** on the post-resolve eigenvalue.
3. Escape eigenvector sign (`+xi` vs `-xi`): **identical** result in both the raw
   constraint violation at the escape point (which scales correctly as `O(t^2)`,
   confirming the escape path's theoretical structure *is* being built correctly) and
   the post-resolve eigenvalue -- not a sign bug.
4. No dual warm-start (IPOPT recomputing its own initial multipliers from scratch every
   rank) vs warm-starting `mult_g`/`mult_x_L`/`mult_x_U` from the previous rank's
   solution (`warm_start_init_point=yes`): **no effect** by itself.
5. IPOPT's default initial barrier parameter vs a small `mu_init` (1e-8): **partial,
   real improvement** (-0.0243 -> -0.0171 between rank 2 and 3) that then **plateaus**
   rather than continuing to close with further rank increases.

**Current best hypothesis:** a remaining difference between IPOPT's specific
interior-point trajectory and Knitro's (which the original code drives with its own
`knitro.opt` options file, not available/ported here) causes IPOPT to converge back to
essentially the same local dual solution regardless of the primal/dual warm start --
i.e. the escape direction is being correctly constructed (point 3 above) but not
genuinely exploited by IPOPT's specific algorithm. Not confirmed; would need either a
from-scratch re-derivation of the exact escape/feasibility-restoration step bm_linf.m
performs (it explicitly "solves for feasibility" along the escape path as a step
distinct from the full re-optimization this port does instead), or a comparison against
a different NLP backend (e.g. `scipy.optimize.minimize(method="trust-constr")`) as a
diagnostic to check whether the issue is IPOPT-specific.

**Decision (explicit, from the user) given the above:** `solve_bm_r`/`run_bm_r.py`
report `optimal_value` as the **raw primal value** (`c . u + c0` at the last rank
tried), not the certified-but-currently-loose `Proposition 3.1` lower bound (still
computed and returned separately, as `certified_lower_bound`, for completeness). The
`status` column distinguishes `"certified"` (the paper's actual guarantee holds) from
`"uncertified_max_rank"` (primal value only, matching SDP-IP empirically in testing,
but **not accompanied by the global-optimality proof that is the entire point of the
BM-r method**) -- this is a real, explicit deviation from the paper, not a detail, and
should be treated accordingly in any comparison that uses these numbers: they can
inform relaxation-tightness comparisons, they cannot stand in for the paper's own
reported certification guarantees or runtimes.

## Verified so far

- `blob_1x2`, Linf, both targets, via the full CLI (`export_instances.py` ->
  `run_bm_r.py`): `optimal_value` matches `SDP-IP` (MOSEK) to 6 decimal places;
  `status: uncertified_max_rank` in both cases (see above).
- Synthetic random networks (both norms): gradient/Jacobian/Hessian/dual-certificate
  finite-difference checks, as described above.

## Not yet done

- `bm_l2` (L2 norm) has only been checked on synthetic random networks, not against a
  real network's `SDP-IP` reference (the `blob_1x2` config only exercises Linf).
- PGD-based warm start (`pgd_linf.m`/`pgd_l2.m`) not ported -- currently using the ball
  center instead (see table above).
- The certificate issue itself (see above) -- open, not resolved, explicitly deprioritized
  in favor of shipping the primal-value comparison per the user's decision.
- No test on a network larger than `blob_1x2`/`blob_4x10`-scale; `BM-r`'s own paper
  targets this exact regime (small per-instance NLPs, not `mnist-9x100`-scale), so this
  is less of a concern here than it was for `sdp_bab`.
