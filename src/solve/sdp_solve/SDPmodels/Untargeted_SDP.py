import numpy as np
import mosek
import yaml
import os
import sys
import logging
from typing import List


from fastsdp_tools.utils import infinity, add_functions_to_class


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from ..mosek_generic_solver import SDPSolver
from networks import ReLUNN
from .certification_problem_objective import objective_Md
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
from .certification_problem_constraints_beta import (
    discrete_betas,
    sum_betas_equals_1,
    sum_beta_i_beta_j_equal_beta_i,
    McCormick_beta_z,
    McCormick_beta_z_all_valid_layers,
    betai_betaj,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
    z_j2_zj_big_m,
    sum_beta_j_z_i_equal_z_i,
    sum_beta_j_z_i_equal_z_i_layer
)
from .certification_problem_constraints_division_by_layers import (
    CHORDAL_DECOMPOSITION_rec,
)
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_sdp import first_term_equal_zero
from .certification_problem_relaxed_relu_heuristic import (
    add_z_quad_active_neuron_heuristic, 
    quadratic_constraint_heuristic,
    quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS,
    quadratic_constraint_heuristic_RANDOM
)


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Md,
    ReLU_constraint_Lan,
    ReLU_constraint_stable_active_relaxation,
    quad_bounds,
    discrete_betas,
    sum_betas_equals_1,
    sum_beta_i_beta_j_equal_beta_i,
    betai_betaj,
    ReLU_triangularization,
    CHORDAL_DECOMPOSITION_rec,
    add_RLT_constraints,
    McCormick_inter_layers,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
    McCormick_beta_z,
    McCormick_beta_z_all_valid_layers,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
    L2_ball_bounds,
    last_layer_linear_equality,
    z_j2_zj_big_m,
    sum_beta_j_z_i_equal_z_i,
    sum_beta_j_z_i_equal_z_i_layer,
    add_z_quad_active_neuron_heuristic, 
    quadratic_constraint_heuristic,
    quadratic_constraint_heuristic_4_DETERMINISTIC_CONSTRAINTS,
    quadratic_constraint_heuristic_RANDOM
)
class UntargetedSDP(SDPSolver):
    def __init__(self, **kwargs):

        super().__init__(
            certification_model_type="UntargetedSDP", BETAS=True, BETAS_Z=True, **kwargs
        )

        logger_mosek.debug(f"Bounds for the network :  {self.L} and {self.U}")
        print("ytargets in UntargetedSDP:", self.ytargets)

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Md()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        self._cut_constraint_counts = {}
        self.handler.Constraints._skipped_count = 0

        def _snap():
            return len(self.handler.Constraints.list_cstr) + self.handler.Constraints._skipped_count

        # sum_beta_logits_equal_logit
        _n = _snap()
        if "sum_beta_logits_equal_logit" in cuts:
            self.sum_beta_j_z_i_equal_z_i()
        self._cut_constraint_counts["sum_beta_logits_equal_logit"] = _snap() - _n

        # RELU + BOUNDS (baseline)
        logger_mosek.debug("Adding ReLU constraints...")
        _n = _snap()
        self.ReLU_constraint_Lan()
        logger_mosek.debug("ReLU constraints added.")
        self._cut_constraint_counts["baseline_relu"] = _snap() - _n

        _n = _snap()
        if "triangularization" in cuts:
            self.ReLU_triangularization()
        self._cut_constraint_counts["triangularization"] = _snap() - _n

        _n = _snap()
        self.quad_bounds()
        if self.norm == "L2" and self.INPUT_IN_VARIABLES and len(self.pruned_input_neurons) == 0:
            self.L2_ball_bounds()
        self._cut_constraint_counts["baseline_bounds"] = _snap() - _n

        # BETA (base)
        _n = _snap()
        self.discrete_betas()
        self.sum_betas_equals_1()
        self.betai_betaj()
        self._cut_constraint_counts["baseline_beta"] = _snap() - _n

        _n = _snap()
        if "McCormick_beta_z" in cuts:
            self.McCormick_beta_z_all_valid_layers(cuts=cuts)
        self._cut_constraint_counts["McCormick_beta_z"] = _snap() - _n

        _n = _snap()
        if "beta_logits_comparaison" in cuts:
            self.z_j2_beta_j2_greater_than_zj()
            self.z_j2_beta_j2_less_than_zj()
        self._cut_constraint_counts["beta_logits_comparaison"] = _snap() - _n

        _n = _snap()
        if "beta_logits_comparaison_big_M" in cuts:
            self.z_j2_zj_big_m()
        self._cut_constraint_counts["beta_logits_comparaison_big_M"] = _snap() - _n

        # RLT
        _n = _snap()
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)
        self._cut_constraint_counts["RLT"] = _snap() - _n

        # Finalisation (first_term, décomposition chordale, last_layer)
        _n = _snap()
        self.first_term_equal_zero()
        if self.CHORDAL_DECOMPOSITION:
            self.CHORDAL_DECOMPOSITION_rec(only_linear_constraints=True)
        if self.LAST_LAYER:
            self.last_layer_linear_equality()
        self._cut_constraint_counts["baseline_other"] = _snap() - _n

        self.handler.Constraints.end_constraints()
