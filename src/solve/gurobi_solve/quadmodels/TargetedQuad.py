import logging
import numpy as np

from ..gurobi_generic_solver import GurobiSolver
from ..objective import add_objective_Lan
from ..constraints import ReLU_constraint_Lan, quad_bounds, ball_constraint
from ..variables import add_variable_z, _add_variable_z
from fastsdp_tools.utils import add_functions_to_class
from ..callback import NonConvexQuadraticProgramCallback

logger_gurobi = logging.getLogger("Gurobi_logger")


@add_functions_to_class(
    add_objective_Lan, ReLU_constraint_Lan, quad_bounds, add_variable_z, _add_variable_z, ball_constraint
)
class TargetedQuad(GurobiSolver):
    """
    A solver that uses Gurobi to solve the optimization problem.
    """

    def __init__(
        self,
        **kwargs,
    ):
        print("kwargs in init TargetedQuad : ", kwargs)

        super().__init__(certification_model_type="TargetedQuad", **kwargs)
        self.possible_targets = [
            target for target in self.ytargets if target != self.ytrue
        ]
        if "ytarget" in kwargs:
            self.ytarget = kwargs["ytarget"]
        else:
            self.ytarget = np.random.choice(self.possible_targets)

    def add_objective(self):
        self.add_objective_Lan()

    def add_variables(self):
        """
        Add variables to the model.
        """
        self.add_variable_z()

    def add_constraints(self):
        """
        Add constraints to the model.
        """
        # RELUMIX
        self.ReLU_constraint_Lan()

        # BOUNDS
        self.quad_bounds()
        # self.ball_constraint()
