# SDP-BaB baseline (Lan, Brückner & Lomuscio, AAAI'23)

"A Semidefinite Relaxation Based Branch-and-Bound Method for Tight Neural Network
Verification". No public code -- reimplemented from the paper, reusing as much of
`solve/sdp_solve/` as possible. Runs in the `certif` env (plain MOSEK/numpy, no
jax/MATLAB) -- no separate env needed, unlike `sdp_fo`.

## What the paper does

Base relaxation: LayerSDP (Batten et al. 2021) -- same chordal decomposition `[k,k+1]`
as this project's `SDP-Layer` config, but with the **full quadratic** inter-matrix
consistency (their eq. 6e) instead of this project's default linear-only subset.

Tightening: for each ReLU neuron `(i,j)`, the complementarity constraint's quadratic
form `Q1_{i,j}` has **at most one negative eigenvalue** (Proposition 1). Deriving it
directly (it's an arrowhead matrix: zero block on `x_i`, border `-W_i[j,:]/2`, corner
`1`) gives a **closed form**, no generic eigensolver needed:

```
lambda_neg = (1 - sqrt(1 + ||W_i[j,:]||^2)) / 2
v = normalize([ -W_i[j,:]/2 ; lambda_neg ])        # support on x_i and x_{i+1,j} only
```

This only depends on `W_i` in the unpruned case. In practice `LayerSDPBaBNode` prunes
stable neurons exactly like Batten et al. 2021 does (`use_inactive_neurons=False`), and
a pruned (stable-inactive) neuron is not an SDP variable at all -- referencing it
crashes. Since such a neuron has `z === 0` identically (not an approximation, a
consequence of its bounds), `eigencuts.compute_eigencuts(W,
stable_inactives_neurons=...)` substitutes it exactly: zero the corresponding column of
`W_i[j,:]` before computing `z/lambda_neg/v` (every term it would have contributed is
`weight * 0 = 0` regardless of the weight). This makes the eigenpair depend on this
sample's bounds too (through its pruning), not purely on `W_i` -- so it's recomputed
once per **sample** (`branch_and_bound.solve_sdp_bab`, reused across every node of that
sample's tree, since `stable_inactives_neurons` only depends on `L, U`, fixed for the
whole tree), not once per network. Still cheap: a few ms even for `mnist-9x100`.

For each such `v = (v_xi, v_out)`, the SDP gets one extra **linear** cut per node,
parameterized by the node's current box `[l, u]` on `t_{i,j} = v . x`:

```
s_{i,j} <= (l + u) * t_{i,j} - l * u      (s_{i,j} = v^T P_i v)
```

B&B (Algorithm 1) branches on whichever `(i,j)` has the largest `|s* - t*^2|`
violation at the parent's relaxation optimum, splitting `[l, u]` at the midpoint,
best-first search (pop smallest `gamma` first), early stop once `gamma_bab > 0`.

## Code map

- `eigencuts.py` -- closed-form `(lambda_neg, v)` per ReLU neuron, exactly substituting
  this sample's pruned (stable-inactive) neurons. Pure numpy, no MOSEK.
- `constraints_eigenvector_secant.py` -- builds the cut (`add_eigenvector_secant_cut`)
  and evaluates `t*, s*` post-solve (`eval_eigencut_expression_at_solution`), both via
  `self.handler.Constraints.add_linear_variable`/`add_quad_variable` -- the same
  primitives every file in `solve/sdp_solve/SDPmodels/` uses.
- `sdp_bab_model.py` -- `LayerSDPBaBNode(TargetedSDP)`: one B&B node. Thin, additive
  subclass -- see "Touch points" below for the only two changes to shared files.
- `branch_and_bound.py` -- Algorithm 1's orchestration loop (priority queue,
  branching, 1h default wall-clock budget).
- `run_sdp_bab.py` -- yaml-driven CLI, mirrors `certification_problem.py`'s main loop.

## Touch points in `solve/sdp_solve/` (both small, opt-in, backward-compatible)

1. `sdp_generic_solver.py`: `self.full_quadratic_consistency = kwargs.get(..., False)`.
2. `Targeted_SDP.py`: `only_linear_constraints=not self.full_quadratic_consistency`
   (was hardcoded `True`), plus a `_add_extra_constraints_hook()` no-op called just
   before `end_constraints()`, overridden by `LayerSDPBaBNode` to add the secant cuts.
   The hook exists because `end_constraints()` is **not idempotent** -- it advances
   `Constraints.current_num_constraint` without appending a row, so anything added
   after a plain `super().add_constraints()` call would make the next
   `new_constraint()` call `IndexError`. Every other config in the project still gets
   `full_quadratic_consistency=False` (default) and never calls the hook -- identical
   behavior to before this change.

Neither `RLT`, `McCormick`, `Untargeted_SDP.py`, `Mzbar.py`, nor any handler file is
touched.

## The coefficient-doubling pitfall (read this before changing the cut math)

`handler/variable_elements.py`'s `add_dict_quad_to_elements`/`add_dict_linear_to_elements`
**divide the passed value by 2** for any off-diagonal (or non-constant-row, for linear
terms) entry before storing it (`dividing_non_diag=True`, the MOSEK triplet format
convention). `handler/mosek_classic/handler_classic.py`'s `is_feasible`/`value_solution`
**multiply by 2** when reading an off-diagonal entry back against a dense solution
matrix. These cancel exactly -- so `add_quad_variable(value=c, ...)` nets out to
"contributes exactly `c * P[a,b]`", for either argument order, with **no implicit
doubling anywhere**.

This project's own `CHORDAL_DECOMPOSITION_rec` only ever uses this API for direct
1:1 entry *equalities* (`P_left[a,b] == P_right[a,b]`), where the question never comes
up. Expanding an actual quadratic FORM `v^T P v = sum_a v_a^2 P[a,a] + sum_{a!=b}
v_a v_b P[a,b]` is a different use case: a diagonal term (`a==b`) needs
`value=v_a**2` (one call); an **unordered** off-diagonal pair (`a!=b`, which the
double sum counts as two equal terms) needs `value=2*v_a*v_b` in **one** call -- not
`v_a*v_b`, and not two separate calls of `v_a*v_b` (both of those undercount by half).

This was caught empirically, not by reading the code correctly the first time: on the
tiny `blob_1x2` network (`K=2`, `n=[2,2,3]`, a single `P` matrix), the solver's
optimum landed almost exactly at a vertex of the input box, making the relaxation
near rank-1 there -- so `s*` (computed from the *raw* `X` matrix by hand) and `t*^2`
had to match almost exactly, and a first ("doubling assumed automatic") version of
this file was off by a reproducible factor tied to exactly the off-diagonal terms.
`eval_eigencut_expression_at_solution` exists specifically so this can be re-checked
against a hand computation on `X` any time the cut math changes.

## Pruned-neuron substitution (not just dropping the cut)

A first version simply dropped (`filter_usable_eigencuts`) any cut referencing a
pruned neuron -- caught the crash, but threw away far more tightening than necessary:
on `mnist-9x100` (478/900 neurons pruned, spread densely across 9 layers of 100), a
cut at a hidden layer references *all 100* neurons of its input layer, so almost any
single pruned neuron among them killed the whole cut -- only 39/900 cuts survived, all
on the one layer (the 784-wide input) that's never prunable in this codebase. The
zero-substitution described above (exact, not an approximation) restores the other 383
cuts at the cost of `O(n_i_effective^2)` instead of `O(n_i^2)` each. Verified: on
`blob_4x10` (2 pruned neurons), 40 -> 38 usable cuts (only the 2 whose *output* neuron
is itself pruned are genuinely dropped -- unavoidable, their complementarity
constraint is trivially `0=0`); on `mnist-9x100`, 900 -> 422 (vs. 39 with the old
all-or-nothing filter).

Important nuance: this is a correctness/fidelity fix, not a performance fix -- total
estimated coefficient count on `mnist-9x100` actually went **up** slightly (12.0M ->
14.0M: 383 more, individually cheap, hidden-layer cuts on top of the same 39 expensive
input-layer ones) because coverage, not raw cost, was the goal. See "Known scaling
limit" below -- the input layer remains the bottleneck regardless.

## Verified

Tiny `blob_1x2` network (`K=2`, `n=[2,2,3]`, scratch test, not committed):
1. Base `LayerSDPBaBNode` (no secant cuts, `full_quadratic_consistency=True`) solves
   without error.
2. A secant cut with deliberately loose (non-binding) bounds leaves `optimal_value`
   unchanged (diff ~1e-7).
3. `eval_eigencut_expression_at_solution`'s `t*` and `s*` match an independent
   hand computation off the raw `X` primal matrix.
4. A tight/binding bound strictly **increases** `optimal_value` (+0.028 in the test)
   while never decreasing it below the vanilla LayerSDP bound -- the monotonicity
   invariant from CLAUDE.md ("ajouter des coupes ne peut qu'augmenter ou laisser
   égale la valeur optimale").
5. The full `branch_and_bound.solve_sdp_bab` loop runs end to end (11 nodes, `status:
   max_nodes`), bound never looser than the vanilla LayerSDP value.

`blob_4x10` (`K=5`, up to 10 neurons/layer, via the real `run_sdp_bab.py` CLI, 2
genuinely pruned neurons -- the case that first caught the stable-inactive crash, see
git history of this file/`constraints_eigenvector_secant.py`):
6. Full B&B run, 21 nodes / ~9-12s, root (vanilla LayerSDP) bound -17.49 tightened to
   ~-17.0/-17.15 depending on exact cut set (run-to-run differences expected: a
   best-first search on a different-sized cut set explores a different path through
   the tree for the same node budget -- only the monotonicity vs. the root bound is a
   hard invariant, not a specific node-by-node trajectory).
7. One child node returned a genuine MOSEK primal-infeasibility certificate
   (`prim_infeas_cer`) -- correctly pruned (not pushed back on the queue), no crash.

## Known scaling limit: wide input layers

On `mnist-9x100` (`n_0=784`), even the *cheapest possible* attempt (root node only,
`max_nodes=1`) was killed by a `timeout 700` after ~11.7 CPU-minutes, still inside
`Objective.add_to_task()` -- never reached `task.optimize()`. Root-caused precisely:
- The vanilla LayerSDP relaxation alone (no eigencuts, `full_quadratic_consistency`
  only) builds in 5s and solves in 55s -- tractable, and for this sample/target it
  already certifies robustness (`gamma=9.8>0`) without any B&B tightening at all.
- One single eigencut on the 784-wide input layer takes ~5.3s to build
  (`n_i=784 -> ~308k coefficients` via `_add_s_terms`'s dense block, `O(n_i^2)`); the
  39 input-layer cuts that survive pruning extrapolate to ~210s of pure Python-side
  construction, before MOSEK even ingests them.
- Input-layer pruning doesn't exist in this codebase (`bounds.check_stability_neurons`
  only loops `k in range(1, K)`) so the substitution fix above cannot shrink this
  specific layer's cuts, unlike every hidden layer.

Consistent with the paper's own reported runtimes (thousands of seconds/image) and
several of its own baselines failing outright at that network scale (their Table 1).
A pragmatic (paper-deviating) fix would be excluding layer-0 eigencuts entirely --
not done without explicit agreement, since it changes fidelity to the method.

## Short-circuit on the vanilla relaxation

`solve_sdp_bab` first solves the vanilla LayerSDP relaxation alone (`full_quadratic_
consistency=True`, no eigencuts -- the cheap ~55s-on-mnist-9x100 part). If that alone
already gives `gamma > 0`, it returns immediately without ever building the (expensive)
eigencuts -- by monotonicity, the eigenvector-tightened bound could only agree. This
is not a paper deviation: Algorithm 1 still runs in full whenever it's actually needed
(see below), this only skips redundant work for instances the base relaxation already
resolves.

This alone turned out to matter more than expected: on `mnist-9x100` sample
`data_index=7`, comparing against an existing SDP-Layer run
(`results/benchmark/9x100-0.026/2026_06_05_13h18_54s_test-jz-SDP-Layer-PP/results_all.csv`,
linear-consistency-only, this project's long-standing default), **all 9 targets**
resolved via this short-circuit alone (`n_nodes=1` each, ~55-116s) -- including two
that the linear-consistency SDP-Layer had left uncertified (`target=2`: -0.63 -> +0.75;
`target=6`: -0.18 -> +0.93). The full quadratic inter-matrix consistency (Batten et al.
2021's actual LayerSDP, vs. this project's historical linear-only relaxation of it) was
by itself enough to close both gaps -- the paper's eigenvector contribution was never
needed for this sample. A useful, if accidental, finding in its own right: this
project's default chordal consistency has been strictly looser than Batten et al.'s
original formulation (see CLAUDE.md "Contrainte de cohérence relaxée"), and that gap
is apparently large enough to flip real instances.

To actually exercise the eigenvector/B&B path at this scale, `data_index=10, target=0`
was used instead (SDP-Layer value -23.17, too negative to expect the quadratic
consistency alone to close it): the short-circuit correctly did *not* fire, and the
full root node (with all usable eigencuts) built and solved successfully --
`-23.17 -> -19.01` after 3 nodes / 1712s (~9.5 min/node average), `status: max_nodes`,
no crash. Confirms the whole pipeline (construction, MOSEK solve, branching, violation
re-evaluation) works correctly at `9x100` scale when it's actually invoked, it's just
slow -- consistent with "Known scaling limit" above.

## Not yet done

- `results.csv`-schema reconciliation with the main pipeline beyond the columns
  written today (`optimal_value, status, n_nodes, time`) -- no `Nb_constraints`,
  `iterations`, dual value, etc.
- No test on a network between `blob_4x10` and `mnist-9x100` in size, to find where
  the input-layer bottleneck actually starts to bite in practice.
