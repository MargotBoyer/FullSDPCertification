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
from .certification_problem_constraints_rlt import add_RLT_constraints
from .certification_problem_constraints_division_by_layers import CHORDAL_DECOMPOSITION_rec
from .certification_problem_constraints_sdp import first_term_equal_zero


logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    objective_Lan,
    ReLU_constraint_Lan,
    quad_bounds,
    ReLU_triangularization,
    add_RLT_constraints,
    McCormick_inter_layers,
    CHORDAL_DECOMPOSITION_rec,
    first_term_equal_zero,
    all_Mc_Cormick_all_layers,
    all_4_McCormick,
    is_front_of_matrix,
)
class SDP_attack(SDPSolver):
    def __init__(self, **kwargs):
        print("kwargs in SDP_attack: ", kwargs)
        super().__init__(certification_model_type="SDP_attack", **kwargs)
        self.BETAS = False
        self.BETAS_Z = False

        self.possible_targets = [
            target for target in self.ytargets if target != self.ytrue
        ]
        if "ytarget" in kwargs:
            self.ytarget = kwargs["ytarget"]
        else:
            self.ytarget = np.random.choice(self.possible_targets)

        logger_mosek.debug(f"Bounds for the network :  {self.L} and {self.U}")

    def add_objective(self):
        """
        Add the objective to the Objective class.
        """
        self.objective_Lan()

    def add_constraints(self, cuts: List = []):
        """
        Add constraints to the task.
        """
        # RELU
        self.ReLU_constraint_Lan()

        # BOUNDS
        self.quad_bounds()
        self.first_term_equal_zero()

        self.handler.Constraints.end_constraints()
