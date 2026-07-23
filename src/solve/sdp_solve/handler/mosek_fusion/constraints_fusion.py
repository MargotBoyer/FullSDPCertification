from typing import List
import mosek
import time
import numpy as np
from mosek.fusion import Expr, Domain, Matrix, Model, Variable
import mosek.fusion.pythonic

from ..indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from ..indexes_variables import Indexes_Variables_for_Mosek_Solver
import numba
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements

from ..constraints import CommonConstraints
import os
import sys
import logging
from fastsdp_tools import (
    infinity,
    sort_lists_by_first,
    deduplicate_and_sum,
    add_functions_to_class,
)


logger_mosek = logging.getLogger("Mosek_logger")

dict_type_bounds = {
    mosek.boundkey.up: "up",
    mosek.boundkey.lo: "lo",
    mosek.boundkey.fx: "fx",
}


class ConstraintsFusion(CommonConstraints):
    """
    Class to handle the current constraint.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        model: mosek.fusion.Model = None,
        **kwargs,
    ):
        """
        Initialize the CurrentConstraint class.

        Parameters
        ----------
        task: mosek.Task
            The MOSEK task.
        index: int
            The index of the constraint.
        """
        super().__init__(
            indexes_matrices=indexes_matrices,
            indexes_variables=indexes_variables,
            **kwargs,
        )
        self.model = model

    def add_var(
        self, dict1: numba.typed.Dict, value: float, dict2: numba.typed.Dict = None
    ):
        """
        Add a variable to the current constraint.
        ----------
        """
        if dict2 is None:
            add_dict_linear_to_elements(
                elements=self.list_cstr[self.current_num_constraint][
                    "elements"
                ].elements,
                dict=dict1,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=False,
            )
        else:
            add_dict_quad_to_elements(
                elements=self.list_cstr[self.current_num_constraint][
                    "elements"
                ].elements,
                dict1=dict1,
                dict2=dict2,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=False,
            )
            self.list_cstr[self.current_num_constraint]["is_quadratic"] = True

    def add_to_task(self):
        """
        Add the constraint to the task.
        """
        logger_mosek.info(f"Adding {self.list_cstr} constraints to the task...")
        if self.verbose : 
            print(f"CALLBACK : Number of constraints : {len(self.list_cstr)}")
        time_start = time.time()
        for ind_cstr in range(len(self.list_cstr)):
            
            # if ind_cstr % 10 == 0:
            #     print(f"CALLBACK : Adding constraint {ind_cstr}/{len(self.list_cstr)}")
            #     time_stop = time.time()
            #     print(
            #         f"Current time to add constraints : {time_stop - time_start:.2f}s"
            #     )
            res = sort_lists_by_first(
                self.list_cstr[ind_cstr]["num_matrix"],
                self.list_cstr[ind_cstr]["i"],
                self.list_cstr[ind_cstr]["j"],
                self.list_cstr[ind_cstr]["value"],
            )

            # Group elements by num_matrix: one dense matrix per PSD variable
            # (res[0] is already sorted by num_matrix after sort_lists_by_first)
            unique_matrices, group_starts = np.unique(res[0], return_index=True)
            group_ends = np.append(group_starts[1:], len(res[0]))

            terms = []
            for nm, s, e in zip(unique_matrices, group_starts, group_ends):
                dim = self.indexes_matrices.get_shape_matrix(nm)
                M = np.zeros((dim, dim))
                M[res[1][s:e], res[2][s:e]] = res[3][s:e]
                name = self.indexes_matrices.get_name_matrix(nm)
                var = self.model.getVariable(name)
                terms.append(Expr.sum(Expr.mulElm(Matrix.dense(M), var)))

            expression = terms[0]
            for t in terms[1:]:
                expression = Expr.add(expression, t)
            if self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.fx:
                lb = self.list_cstr[ind_cstr]["lb"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.equalsTo(lb),
                )
            elif self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.up:
                ub = self.list_cstr[ind_cstr]["ub"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.lessThan(ub),
                )
            elif self.list_cstr[ind_cstr]["bound_type"] == mosek.boundkey.lo:
                lb = self.list_cstr[ind_cstr]["lb"]
                constraint = self.model.constraint(
                    self.list_cstr[ind_cstr]["name"],
                    expression,
                    Domain.greaterThan(lb),
                )

    def add_model(self, model: mosek.fusion.Model):
        """
        Add model to the objective.

        Parameters
        ----------
        model: mosek.fusion.Model
            The MOSEK model.
        """
        self.model = model
