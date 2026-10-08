# OptBranch baseline (Anderson, Ma, Li & Sojoudi)

"Towards Optimal Branching for Neural Network Robustness Certification"
(arXiv:2101.09306, JMLR 2025 vol. 26 paper 21-0068). No name given to the
method by the authors; "OptBranch" chosen here from the paper's title.
Runs entirely in the `certif` conda env -- unlike `sdp_fo`/`bm_r`, there is
no third-party code to vendor or port: the paper's own SDP relaxation
(section 2.4, eq. (7)-(8)) is exactly this project's `SDP-IP` config
(Raghunathan et al. 2018 baseline, no chordal decomposition, no cuts), so
the only thing to implement is a branch-and-bound loop that repeatedly calls
`solve/sdp_solve` with shrinking `L`/`U`.

## What the paper does, for the SDP part

The paper's own SDP relaxation is not new -- its contribution is a
**branch-and-bound wrapper over the input uncertainty set** `X`, proving
which way of partitioning `X` minimizes the relaxation's worst-case error:

1. **Partitioning strictly helps** (Proposition 11): solving the relaxation
   on each part of any partition of `X` and taking the max can only tighten
   the bound compared to solving it on `X` directly.
2. **Which partition is best, in form** (section 4.2): the paper measures
   how far the SDP solution `P*` is from rank-1 (an exact relaxation) via a
   "rank-1 gap" `g(P*) >= 0`, bounds it above by a term depending only on the
   partition's interval widths (Lemma 14), and shows (Theorem 16) that a
   **uniform** split along any single coordinate minimizes that bound --
   i.e. always bisect, never split asymmetrically.
3. **Which coordinate to branch on** (section 4.3): under a row-normalization
   assumption on `W` (Assumption 17, WLOG via rescaling -- **not implemented
   here**, see "Differences" below), Theorem 19 bounds the relaxation error
   by a term scaling with `q(l,u) = max_k max{|l_k|, |u_k|}` -- the **largest
   magnitude** (not width) among the current bounds. Theorem 20 then shows
   branching on `i* = argmax_k max{|l_k|, |u_k|}` is worst-case optimal. The
   paper notes (end of section 4.3.2) that, because this score only depends
   on the bounds of the layer being branched, it extends as a **heuristic**
   to any neuron of any layer in a multi-layer network, not just the input
   -- this is the extension `branching_score.py` implements.

## Code map

- `branching_score.py` -- `select_branching_neuron(L, U)`: Theorem 20's
  score, computed over every unstable neuron of every layer (not just the
  input), per the multi-layer heuristic above.
- `node_solver.py` -- the "hook": `apply_override(L, U, k, i, l, u)` returns
  fresh bound lists with neuron `i` of layer `k` tightened; `solve_node(...)`
  builds a fresh `TargetedSDP` on those bounds and solves it directly (one
  config, no fork isolation -- see "Known limitations"). No change was
  needed in `solve/sdp_solve` or `generic_solver.py`: `Solver.__init__`
  already uses `L`/`U` verbatim (skipping bound computation) when passed in,
  including re-running `check_stability_neurons` on them, so a neuron that
  becomes stable after a branch is pruned automatically by the existing
  pipeline.
- `branch_and_bound.py` -- `solve_optbranch(...)`: the depth-first search.
  See "Divergence from Algorithm 2" below for why it explores both children
  of every branch instead of discarding one, as the paper's own pseudocode
  does.
- `run_optbranch.py` -- yaml-driven CLI, same `bounds_method: from_file`
  convention as `sdp_bab/run_sdp_bab.py`; writes `results/baselines/optbranch/<network>/<title>/results.csv`.

## Divergence from Algorithm 2

The paper's "Algorithm 2" pseudocode (section 5), read literally, computes
both children's relaxation values at each step but then keeps only **one**
of them (the one with the larger/easier value) for the next iteration,
discarding the other, unexplored. Taken at face value this cannot certify
the whole of `X`: the discarded half is never checked again, so a negative
relaxation value there would go unnoticed. The user explicitly asked for a
true guarantee over all of `X`, so `branch_and_bound.py` instead does a
**real DFS**: both children of every branch are explored to completion
(down to `max_depth`/`max_nodes`, or an already-certified subtree) before
backtracking, and `status == "certified"` only when every leaf of the tree
reached a non-negative value. This is a deliberate, documented departure
from the paper's pseudocode, not an attempt to reproduce it exactly.

## Infeasible branch cells

During testing on `blob_1x2`, some DFS leaves came back with MOSEK status
`prim_infeas_cer` (relaxation infeasible). The first hypothesis -- a
numerical artifact of an overly small bisected interval -- turned out to be
wrong on inspection: the intervals involved at the failing node were not
unusually small (width ~1, same order as everywhere else in the tree).

The actual explanation, confirmed with an independent check (solving a plain
LP: does any input in the original ball produce pre-activations in both
branched neurons' current sub-ranges simultaneously?) on the exact failing
cell, is structural, not numerical: that specific combination of per-neuron
sub-ranges -- each chosen by a different DFS branching step, independently
of the others -- corresponds to **no actual input at all**. The LP was
infeasible by construction (`scipy.optimize.linprog`, `HiGHS Status 8`),
matching MOSEK's own finding exactly.

This is expected, not a bug, and it is actually useful: the SDP relaxation
still carries the ReLU complementarity equality `z_i*(z_i - W_i x - b_i) = 0`
for every unstable neuron, which ties the lifted variables to `x` tightly
enough that MOSEK's relaxation-level infeasibility *proves* the cell's true
input region is empty (the relaxation's feasible set is a superset of "exists
`x` achieving this combination of neuron sub-ranges", so if even the
superset is empty, so is the true cell). `branch_and_bound.py` treats such a
leaf as **vacuously certified** (`node_solver.solve_node` returns
`is_infeasible=True`, `recurse` returns `(True, None)` -- no value, nothing
to add to the worst-case aggregate), not as an inconclusive failure.

This is more likely to happen the more neurons get branched independently
relative to the input dimension (`blob_1x2` has `n_x=2`, so three
independently-tightened neuron ranges from three different DFS levels
already over-determine the system) -- a direct consequence of branching on
*derived* (hidden-layer) coordinates without ever re-deriving them from a
shrunk input box (see "Input-region branching" in the differences table
below). It is still sound either way: an infeasible cell is always a cell
with nothing left to prove.

## Differences from the paper (full table)

| | Paper | This port | Why |
|---|---|---|---|
| Search strategy | Algorithm 2: single surviving branch per step (see above) | Full DFS, both children explored | Needed for a real certification guarantee on all of `X` — see above |
| Row normalization (Assumption 17) | `W` rescaled so every row has unit L1 norm, required for Theorem 19/20's exact worst-case-optimality guarantee | Not implemented — branching score computed on raw bounds | The score `max(|l|,|u|)` is still a reasonable heuristic without rescaling; without it, Theorem 20's literal optimality claim isn't guaranteed to hold, only the qualitative "loosest bound first" intuition |
| Input-region branching | Bisecting an input coordinate implicitly tightens every downstream layer's bounds too, via the paper's own bound propagation re-run on the smaller sub-box | Bisecting `L[0][i]`/`U[0][i]` only narrows that one input bound; hidden-layer `L[k]`/`U[k]` (`k>=1`) are left exactly as computed for the *original* full box unless a hidden neuron is *separately* chosen for branching | Re-running bound propagation (e.g. CROWN) at every DFS node was judged too expensive for a first version; branching directly on hidden-layer neurons (enabled by the multi-layer heuristic, and by `node_solver.py` accepting overrides at any layer) is the substitute used instead, but it is not equivalent to the paper's own scheme |
| Multi-layer networks | Theorem 19/20 proven only for a single hidden layer; multi-layer use is explicitly called a "heuristic" by the paper itself (end of 4.3.2) | Same — used as a heuristic, not reproducing a proof that doesn't exist for this case either in the paper | Matches the paper's own framing, not a divergence |
| Crash isolation | N/A (MATLAB/Python + MOSEK via CVXPY, no isolation mentioned) | None — `node_solver.solve_node` calls `run_optimization` directly in-process, unlike the main pipeline's fork-per-solve (`_run_optimization_isolated`) | A DFS tree does many fast small solves; forking each one would dominate runtime. Acceptable given the paper's own target scale (small, single-hidden-layer networks) — see "Known limitations" |

## Known limitations

- No crash isolation (see table above): a MOSEK crash on one node kills the
  whole DFS run for that (sample, target) pair, unlike the main pipeline.
- No bound-propagation re-run on input-coordinate branches (see table
  above): input-layer branches are weaker than the paper's own scheme in
  that specific sense.
- `optimal_value` is a lower bound on the true QCQP only at **certified**
  leaves; at budget-exhausted leaves it is simply the last relaxation value
  computed there, which by Proposition 11 is still a valid (if possibly
  loose, since the leaf's region wasn't fully refined) lower bound on that
  sub-region — so the reported `optimal_value` is always a valid global
  lower bound regardless of `status`, just not necessarily a tight one when
  `status != "certified"`.
- Infeasible (empty) branch cells are expected and handled soundly — see
  "Infeasible branch cells" above. Not a limitation, documented here only
  because the first diagnosis of this behavior (during testing) was wrong.
- Not yet tested beyond small networks (`blob_1x2`/`blob_4x10` scale) --
  see `baselines/README.md`.

## Not yet done

- Assumption 17's row-rescaling (see table above).
- Exposing the inner relaxation's config (cuts, chordal decomposition) via
  CLI flags on `run_optbranch.py` -- currently fixed to `node_solver.DEFAULT_INNER_MODEL_KWARGS`
  (the paper's own SDP-IP-equivalent params); editable by passing `model_kwargs`
  to `solve_optbranch` directly, or by editing the constant.
- Validation against a real `SDP-IP` (MOSEK) solve on an unbisected box, to
  confirm the root node reproduces the un-branched baseline exactly (the
  same kind of cross-check `bm_r`'s README reports for its own primal value).
