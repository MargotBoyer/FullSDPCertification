"""LayerSDPBaBNode: one B&B node's SDP relaxation (Lan, Bruckner & Lomuscio AAAI'23,
eq. 16 / Algorithm 1 "Problem k"). A thin, additive subclass of TargetedSDP -- no
existing file in solve/sdp_solve/SDPmodels is modified beyond the two small, opt-in,
backward-compatible hooks added to Targeted_SDP.py and sdp_generic_solver.py
(full_quadratic_consistency flag, _add_extra_constraints_hook).

Reuses, unchanged: TargetedSDP's objective, ReLU/triangularization/bound constraints,
chordal decomposition (with full_quadratic_consistency=True to match LayerSDP (Batten
et al. 2021) exactly, instead of this project's default linear-only consistency), and
the whole MOSEK solve pipeline (sdp_generic_solver.SDPSolver).

Config matching the paper's base LayerSDP relaxation (7): CHORDAL_DECOMPOSITION=True,
cuts=["triangularization"], no RLT.
"""
from fastsdp_tools.utils import add_functions_to_class

from solve.sdp_solve.SDPmodels.Targeted_SDP import TargetedSDP
from .constraints_eigenvector_secant import (
    add_eigenvector_secant_cut,
    add_all_eigenvector_secant_cuts,
    eval_eigencut_expression_at_solution,
)

# Matches this project's SDP-Layer config (CLAUDE.md), the paper's LayerSDP starting
# point. Shared with branch_and_bound.py's stable_inactives_neurons probe (must see the
# exact same pruning config this class will actually build with, so eigencuts.py's
# zero-substitution lines up with what verify_variable_z allows) -- single source of
# truth, not duplicated.
DEFAULT_KWARGS = {
    "CHORDAL_DECOMPOSITION": True,
    "cuts": ["triangularization"],
    "RLT_props": [0.0],
    "use_active_neurons": True,
    "use_inactive_neurons": False,
    "keep_penultimate_actives": True,
}


@add_functions_to_class(
    add_eigenvector_secant_cut,
    add_all_eigenvector_secant_cuts,
    eval_eigencut_expression_at_solution,
)
class LayerSDPBaBNode(TargetedSDP):
    def __init__(self, eigencuts, eigencuts_bounds, **kwargs):
        """eigencuts: list[eigencuts.EigenCut], already restricted to this sample's
        pruning (see eigencuts.compute_eigencuts's stable_inactives_neurons parameter
        and branch_and_bound.py, which computes it once per sample and reuses it for
        every node of that sample's B&B tree -- stable_inactives_neurons only depends
        on L, U, which don't change across nodes, only eigencuts_bounds does).
        eigencuts_bounds: dict[(layer, neuron) -> (l, u)], THIS node's branch -- the
        box the B&B loop has narrowed t_{i,j} to for each neuron it has touched so far.
        Neurons absent from this dict get no secant cut (a valid, just looser, relaxation).
        """
        kwargs["full_quadratic_consistency"] = True
        for key, value in DEFAULT_KWARGS.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)
        self.eigencuts = eigencuts
        self.eigencuts_bounds = eigencuts_bounds

    def _add_extra_constraints_hook(self):
        self.add_all_eigenvector_secant_cuts()

    def get_results(self, cuts, verbose: bool = False):
        """Overrides TargetedSDP's (mixed-in) get_results to also grab every P_i
        solution matrix while self.handler.task is still open -- it is closed by
        cleanup_mosek() in run_optimization()'s `finally`, right after get_results()
        returns, so this is the last point at which get_solution() is callable.
        Writes results.csv exactly as before (super() call first, unchanged)."""
        result = super().get_results(cuts, verbose)
        self.last_X = None
        if self.handler.is_status_optimal():
            self.last_X = {}
            for num_matrix in range(self.handler.indexes_matrices.nb_matrices):
                dim = self.handler.indexes_matrices.get_shape_matrix(num_matrix)
                self.last_X[num_matrix] = self.handler.get_solution(ind_solution=num_matrix, dim=dim)
        return result

    def solve_node(self, verbose: bool = False):
        """Solves this single node (one eigencuts_bounds box, one ytarget) directly --
        unlike SDPSolver.solve(), which loops over self.ytargets and self.RLT_props.
        Each B&B node needs exactly one (cuts, ytarget) combination, chosen once at
        construction time, so this bypasses that loop and calls run_optimization()
        (in-process, no fork -- crash isolation is unneeded here: a crash should
        surface immediately rather than be silently swallowed into a 'crashed' row,
        since it would indicate a bug in the new eigenvector-secant cut, not an
        expected MOSEK numerical failure on an otherwise-validated model).

        Returns (gamma, last_X): gamma is self.handler.optimal_value (None if the node
        did not solve to optimality), last_X the dict num_matrix -> dense ndarray of
        P_i solution matrices (None likewise).
        """
        if self.is_trivially_solved:
            self.get_results_trivially_solved()
            return getattr(self.handler, "optimal_value", None), None
        self.RLT_prop = self.RLT_props[0]
        cuts = self.cuts_to_test[0]
        self.run_optimization(cuts, verbose)
        if self.handler.is_status_optimal():
            return self.handler.optimal_value, self.last_X
        return None, None
