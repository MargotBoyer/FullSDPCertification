"""The branch-and-bound "hook": solve the inner SDP relaxation on a single
node's (L, U). This is the only point of contact with solve/sdp_solve, and
no change was needed there to support it: Solver.__init__ already skips
compute_bounds_ and uses L/U verbatim when they're passed in explicitly
(generic_solver.py, around line 105), including re-running
check_stability_neurons on them -- so a neuron that becomes stable after a
branch's bisection is pruned by the existing pipeline automatically, with no
special-casing needed here.

Deliberately bypasses SDPSolver.solve()/._run_optimization_isolated(): those
implement the multi-(cuts, RLT_prop) sweep and fork-based crash isolation the
*main* certification pipeline needs across one slow solve per sample; a
branch-and-bound run instead needs many fast single-config solves in the same
process. Crash isolation is accordingly NOT provided here -- see README
"Known limitations".
"""
import copy

from solve.sdp_solve.SDPmodels.Targeted_SDP import TargetedSDP

# Matches CLAUDE.md's "SDP-IP" config exactly (Raghunathan et al. 2018, no
# chordal decomposition, no cuts, no pruning of stable neurons) -- i.e. the
# paper's own relaxation (2.4), eq. (7)-(8). Can be overridden per-call via
# solve_node's model_kwargs to instead wrap OptBranch around SDPT/SDPU with
# cuts -- the branching mechanism is orthogonal to which relaxation is solved
# at each node.
DEFAULT_INNER_MODEL_KWARGS = dict(
    CHORDAL_DECOMPOSITION=False,
    LAST_LAYER=False,
    use_active_neurons=True,
    use_inactive_neurons=True,
    keep_penultimate_actives=True,
    cuts=[],  # SDPSolver.__init__ does self.cuts = kwargs.get("cuts") -- must be [] not
              # None here (unlike the yaml path, pydantic's own default), since some
              # constraint-building code assumes self.cuts is iterable unconditionally.
    all_combinations_cuts=False,
    RLT_props=[1.0],  # unused (no "RLT" in cuts) but SDPSolver.solve()/run_optimization expect it set
)


def copy_bounds(L, U):
    return [list(Lk) for Lk in L], [list(Uk) for Uk in U]


def apply_override(L, U, k, i, l_new, u_new):
    """Returns fresh (L, U) copies with L[k][i], U[k][i] replaced -- the
    "hook" the user asked for: changing neuron i of layer k's bounds
    iteratively to drive the branch-and-bound."""
    L2, U2 = copy_bounds(L, U)
    L2[k][i] = l_new
    U2[k][i] = u_new
    return L2, U2


def solve_node(network, epsilon, norm, x, ytrue, ytarget, L, U, model_kwargs=None,
               data_index=0, folder_name=None, verbose=False):
    """Build a fresh TargetedSDP on (L, U) and solve it directly: a single
    cuts/RLT_prop configuration per call (no sweep), no fork isolation.

    Returns (optimal_value, is_infeasible):
    - optimal_value is None whenever MOSEK did not reach a usable status
      (solver hiccup, crash, time limit with no bound extracted, ...).
    - is_infeasible is True when MOSEK *specifically* reports the relaxation
      as infeasible on this node's (L, U). This is NOT a numerical artifact
      of a too-tight bisection -- it was checked directly (an independent LP
      feasibility query on a concrete branch-and-bound cell confirmed no
      input in the original ball maps there at all). Branching on a neuron's
      PRE-activation bound still carries the ReLU complementarity equality
      z_i*(z_i - W_i x - b_i) = 0, which ties the lifted variables tightly
      enough to x that a combination of per-neuron sub-ranges chosen
      independently (one per DFS level, each neuron branched without regard
      to what was already imposed on the others) can correspond to no actual
      input at all -- especially likely when the number of branched neurons
      exceeds the input dimension. When that happens, the SDP relaxation
      being infeasible is a *sound proof* that this branch cell is empty:
      there is nothing there to certify, so the caller should treat it as a
      vacuously certified leaf, not an inconclusive one."""
    kwargs = dict(DEFAULT_INNER_MODEL_KWARGS)
    kwargs.update(model_kwargs or {})
    model = TargetedSDP(
        network=network, epsilon=epsilon, norm=norm, x=x, ytrue=ytrue, ytarget=ytarget,
        L=copy.deepcopy(L), U=copy.deepcopy(U), data_index=data_index, folder_name=folder_name,
        **kwargs,
    )

    if model.is_trivially_solved:
        model.get_results_trivially_solved()
        return getattr(model.handler, "optimal_value", None), False

    cuts = model.cuts_to_test[0]
    model.RLT_prop = model.RLT_props[0] if model.RLT_props else None
    model.run_optimization(cuts, verbose=verbose)
    value = getattr(model.handler, "optimal_value", None)
    is_infeasible = False
    try:
        is_infeasible = bool(model.handler.is_status_infeasible())
    except Exception:
        pass
    return value, is_infeasible
