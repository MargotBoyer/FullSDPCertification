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

from ..sdp_generic_solver import SDPSolver
from networks import ReLUNN
from .certification_problem_objective import objective_Mzbar
from .certification_problem_constraints_bounds import (
    quad_bounds,
    McCormick_inter_layers,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
from .certification_problem_constraints_forward_pass import (
    ReLU_constraint_Lan,
    ReLU_triangularization,
)
from .certification_problem_constraints_beta import (
    discrete_betas,
    sum_betas_equals_1,
    McCormick_beta_z,
    McCormick_beta_z,
    betai_betaj,
    zbar_sum_beta_z,
    zbar_max_z,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
)
from .certification_problem_constraints_division_by_layers import (
    CHORDAL_DECOMPOSITION_rec,
)
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_sdp import first_term_equal_zero


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Mzbar,
    ReLU_constraint_Lan,
    quad_bounds,
    discrete_betas,
    sum_betas_equals_1,
    zbar_sum_beta_z,
    betai_betaj,
    zbar_max_z,
    ReLU_triangularization,
    CHORDAL_DECOMPOSITION_rec,
    add_RLT_constraints,
    McCormick_inter_layers,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
    McCormick_beta_z,
    McCormick_beta_z,
    z_j2_beta_j2_greater_than_zj,
    z_j2_beta_j2_less_than_zj,
)
class MzbarSDP(SDPSolver):
    def __init__(self, **kwargs):
        super().__init__(
            certification_model_type="MzbarSDP",
            BETAS=True,
            BETAS_Z=True,
            ZBAR=True,
            **kwargs,
        )
        print("MZBAR MATRIX BY LAYERS", self.CHORDAL_DECOMPOSITION)

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Mzbar()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        # RELU
        logger_mosek.debug("Adding ReLU constraints...")
        self.ReLU_constraint_Lan()
        logger_mosek.debug("ReLU constraints added.")

        if "triangularization" in cuts:
            self.ReLU_triangularization()

        #ZBAR
        self.zbar_sum_beta_z()
        self.zbar_max_z()

        # BOUNDS
        self.quad_bounds()
        if self.norm == "L2" and self.INPUT_IN_VARIABLES and len(self.pruned_input_neurons) == 0:
            self.L2_ball_bounds()

        # BETA
        self.discrete_betas()
        self.sum_betas_equals_1()
        self.betai_betaj()

        if "McCormick_beta_z" in cuts:
            self.McCormick_beta_z_all_valid_layers()

        # # Some cuts comparing different logits
        if "beta_logits_comparaison" in cuts:
            self.z_j2_beta_j2_greater_than_zj()
            self.z_j2_beta_j2_less_than_zj()
        
        if "beta_logits_comparaison_big_M" in cuts : 
            self.z_j2_zj_big_m()

        # RLT
        if "RLT" in cuts:
            self.add_RLT_constraints(p=self.RLT_prop)

        self.first_term_equal_zero()

        # MATRIX BY LAYERS
        if self.CHORDAL_DECOMPOSITION:
            self.CHORDAL_DECOMPOSITION_rec(only_linear_constraints=True)

        if self.LAST_LAYER:
            self.last_layer_linear_equality()

        self.handler.Constraints.end_constraints()
