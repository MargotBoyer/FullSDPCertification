import numpy as np
import mosek
import yaml
import os
import sys

from fastsdp_tools.utils import infinity, add_functions_to_class
import logging
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ..mosek_generic_solver import SDPSolver
from networks import ReLUNN
from .certification_problem_objective import objective_Lan
from .certification_problem_constraints_bounds import (
    L2_ball_bounds,
    quad_bounds,
    McCormick_inter_layers,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
from .certification_problem_constraints_forward_pass import (
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    ReLU_triangularization,
    last_layer_linear_equality
)
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_division_by_layers import CHORDAL_DECOMPOSITION_rec
from .certification_problem_constraints_sdp import first_term_equal_zero
from .certification_problem_relaxed_relu_heuristic import (
    add_z_quad_active_neuron_heuristic, 
    quadratic_constraint_heuristic,
    quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS,
    quadratic_constraint_heuristic_RANDOM
)


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Lan,
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    quad_bounds,
    ReLU_triangularization,
    add_RLT_constraints,
    McCormick_inter_layers,
    CHORDAL_DECOMPOSITION_rec,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
    L2_ball_bounds,
    last_layer_linear_equality,
    add_z_quad_active_neuron_heuristic, 
    quadratic_constraint_heuristic,
    quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS,
    quadratic_constraint_heuristic_RANDOM
)
class TargetedSDP(SDPSolver):
    def __init__(self, **kwargs):
        # print("kwargs in TargetedSDP: ", kwargs)
        super().__init__(certification_model_type="TargetedSDP", **kwargs)

        logger_mosek.debug("beginning TargetedSDP init")
        self.BETAS = False
        self.BETAS_Z = False

        self.possible_targets = [
            target for target in self.ytargets if target != self.ytrue
        ]
        if "ytarget" in kwargs:
            self.ytarget = kwargs["ytarget"]
        elif not self.is_trivially_solved:
            self.ytarget = np.random.choice(self.possible_targets)

        print("Neurones stables actives: ", self.stable_actives_neurons)
        print("Neurones stables inactives: ", self.stable_inactives_neurons)

        logger_mosek.debug(f"Bounds for the network :  {self.L} and {self.U}")
        logger_mosek.debug("ending TargetedSDP init")

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Lan()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        self._cut_constraint_counts = {}
        self.handler.Constraints._skipped_count = 0

        def _snap():
            return len(self.handler.Constraints.list_cstr) + self.handler.Constraints._skipped_count

        _n = _snap()
        self.ReLU_constraint_Lan()
        self._cut_constraint_counts["baseline_relu"] = _snap() - _n

        _n = _snap()
        self.ReLU_triangularization()
        self._cut_constraint_counts["triangularization"] = _snap() - _n

        _n = _snap()
        self.quad_bounds()
        if self.norm == "L2" and self.INPUT_IN_VARIABLES and len(self.pruned_input_neurons) == 0:
            self.L2_ball_bounds()
        self._cut_constraint_counts["baseline_bounds"] = _snap() - _n

        _n = _snap()
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)
        self._cut_constraint_counts["RLT"] = _snap() - _n

        _n = _snap()
        if "allMC" in cuts:
            self.all_Mc_Cormick_all_layers()
        self._cut_constraint_counts["allMC"] = _snap() - _n

        _n = _snap()
        if self.CHORDAL_DECOMPOSITION:
            self.CHORDAL_DECOMPOSITION_rec(only_linear_constraints=True)
        if self.LAST_LAYER:
            self.last_layer_linear_equality()
        self.first_term_equal_zero()
        self._cut_constraint_counts["baseline_other"] = _snap() - _n

        self.handler.Constraints.end_constraints()
