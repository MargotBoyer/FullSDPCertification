"""Depth-first branch-and-bound wrapping node_solver.solve_node with the
branching rule of branching_score.select_branching_neuron (Theorem 20 +
section 4.3.2's multi-layer extension).

Divergence from the paper's own Algorithm 2 (documented in detail in
README.md -- "Divergence from Algorithm 2"): Algorithm 2, read literally,
keeps only ONE of the two children at each step and discards the other,
unexplored. That cannot certify the whole region X -- only the single
surviving sub-path. This module instead explores BOTH children of every
branch to completion (down to max_depth, or an already-certified subtree)
before backtracking, so a "certified" result here is a genuine guarantee on
the whole of the original X, per the user's explicit request.
"""
import time

from .branching_score import select_branching_neuron
from .node_solver import apply_override, solve_node


def solve_optbranch(network, epsilon, norm, x, ytrue, ytarget, L, U, model_kwargs=None,
                     data_index=None, dataset_name=None, network_name=None, folder_name=None,
                     max_depth=20, max_nodes=None, max_time=3600.0, verbose=False):
    """Returns a dict: optimal_value, status, n_nodes, max_depth_reached, time.

    optimal_value is the worst (minimum) relaxation value found among the
    LEAVES of the search tree actually reported -- not among every node
    visited, since by Proposition 11 a child's relaxation value is always
    >= its parent's, so a parent that was subsequently refined is superseded
    by its children and must not count towards the aggregate (otherwise a
    fully-certified tree could incorrectly report a negative optimal_value
    inherited from an ancestor).

    status: "certified" (every leaf is either >= 0 or a proven-empty/
    infeasible cell -- see node_solver.solve_node's docstring and README
    "Infeasible branch cells"), "uncertified_budget_exhausted" (DFS hit
    max_depth/max_nodes on some leaf still < 0), "timeout", or "solve_failed"
    (some leaf did not return a usable value for a reason other than proven
    infeasibility -- e.g. a solver hiccup).
    """
    t_start = time.time()
    stats = {"n_nodes": 0, "max_depth_reached": 0, "timed_out": False, "budget_exhausted": False}
    model_kwargs = model_kwargs or {}

    def recurse(Lk, Uk, depth):
        if stats["timed_out"] or (time.time() - t_start > max_time):
            stats["timed_out"] = True
            return False, None

        stats["n_nodes"] += 1
        stats["max_depth_reached"] = max(stats["max_depth_reached"], depth)
        value, is_infeasible = solve_node(
            network, epsilon, norm, x, ytrue, ytarget, Lk, Uk,
            model_kwargs=model_kwargs, data_index=data_index, folder_name=folder_name,
            verbose=verbose,
        )
        if is_infeasible:
            # The relaxation (which already includes the ReLU complementarity
            # equality z_i*(z_i - W_i x - b_i) = 0) being infeasible on this
            # cell is a sound proof that no input in the original ball maps
            # here at all -- see node_solver.solve_node's docstring and
            # README "Infeasible branch cells". Nothing to certify: this leaf
            # is vacuously done, and contributes no value to the aggregate.
            return True, None

        if value is None:
            return False, None  # solve did not return a usable value -> inconclusive leaf

        if value >= 0:
            return True, value  # certified leaf: this sub-region is done

        if depth >= max_depth or (max_nodes is not None and stats["n_nodes"] >= max_nodes):
            stats["budget_exhausted"] = True
            return False, value  # budget-exhausted leaf: value is the binding (negative) bound

        branch = select_branching_neuron(Lk, Uk)
        if branch is None:
            # No unstable neuron left to refine, yet value < 0: nothing more
            # branching can do here. A negative RELAXED value is never proof
            # of non-robustness on its own (see README), so this is reported
            # as a binding, unresolved leaf rather than a certified failure.
            stats["budget_exhausted"] = True
            return False, value

        k, i, _ = branch
        l, u = float(Lk[k][i]), float(Uk[k][i])
        mid = (l + u) / 2.0
        L1, U1 = apply_override(Lk, Uk, k, i, l, mid)
        L2, U2 = apply_override(Lk, Uk, k, i, mid, u)

        left_ok, left_val = recurse(L1, U1, depth + 1)
        right_ok, right_val = recurse(L2, U2, depth + 1)

        child_vals = [v for v in (left_val, right_val) if v is not None]
        worst = min(child_vals) if child_vals else value
        return (left_ok and right_ok), worst

    certified, worst_value = recurse(L, U, 0)

    if stats["timed_out"]:
        status = "timeout"
    elif certified:
        # worst_value can legitimately be None here only if every leaf of the
        # tree turned out infeasible (empty cell) -- vacuously certified.
        status = "certified"
    elif worst_value is None:
        status = "solve_failed"
    else:
        status = "uncertified_budget_exhausted"

    return {
        "optimal_value": worst_value,
        "status": status,
        "n_nodes": stats["n_nodes"],
        "max_depth_reached": stats["max_depth_reached"],
        "time": time.time() - t_start,
    }
