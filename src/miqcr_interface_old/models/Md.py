import os
import sys
from typing import List
from .certification_problem_ReLU_constraints import (
    ReLU_cstr,
    ReLU_cstr_greater_than_zero_part,
    ReLU_triangular_constraints,
)
from .certification_problem_bounds_constraints import (
    McCormick_same_neurons,
    McCormick_different_neurons,
    McCormick_all_layers_neurons,
    McCormick_diagonal,
)
from .certification_problem_beta_constraints import (
    beta_sum_equal_1,
)
from .certification_problem_objective import Md_objective

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


from conic_bundle import ConicBundleParser
from fastsdp_tools.utils import add_functions_to_class


@add_functions_to_class(
    ReLU_cstr,
    Md_objective,
    beta_sum_equal_1,
    McCormick_all_layers_neurons,
    McCormick_diagonal,
    McCormick_different_neurons,
    McCormick_same_neurons,
    ReLU_cstr_greater_than_zero_part,
)
class MdParser(ConicBundleParser):
    def __init__(self, filename, **kwargs):
        self.filename = filename + "_Md"
        super().__init__(LAST_LAYER=False, BETAS=True, BETAS_Z=True, **kwargs)

    def add_model(self):
        for j in self.ytargets:
            if j == self.ytrue:
                continue
            self.add_variable(binary=True)
            self.delta_bounds = 0.0

        for k in range(self.K):
            for j in range(self.n[k]):
                if (k, j) in self.stable_inactives_neurons:
                    continue
                self.add_variable(
                    lb=0, ub=max(self.U[k][j], 0) + self.delta_bounds, binary=False
                )
        self.Md_objective()
        self.ReLU_cstr()
        self.beta_sum_equal_1()

        if self.which_McCormick == "none":
            pass
        elif self.which_McCormick == "all":
            self.McCormick_all_layers_neurons()
        elif self.which_McCormick == "diagonal":
            self.McCormick_diagonal()
