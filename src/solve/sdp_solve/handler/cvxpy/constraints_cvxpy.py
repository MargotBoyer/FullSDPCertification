import numpy as np
import logging
from collections import defaultdict
from typing import List
import cvxpy as cp
import mosek
import numba
from numba.typed import Dict

from ..indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from ..indexes_variables import Indexes_Variables_for_Mosek_Solver
from ..constraints import CommonConstraints
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements
from fastsdp_tools import infinity

logger_mosek = logging.getLogger("Mosek_logger")


class ConstraintsCvxpy(CommonConstraints):

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        super().__init__(indexes_matrices, indexes_variables, **kwargs)

    def add_var(self, dict1, value, dict2=None):
        if dict2 is None:
            add_dict_linear_to_elements(
                elements=self.list_cstr[self.current_num_constraint]["elements"].elements,
                dict=dict1,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=True,
            )
        else:
            add_dict_quad_to_elements(
                elements=self.list_cstr[self.current_num_constraint]["elements"].elements,
                dict1=dict1,
                dict2=dict2,
                value=value,
                nb_index=self.indexes_variables.max_index,
                dividing_non_diag=True,
            )
            self.list_cstr[self.current_num_constraint]["is_quadratic"] = True

    def build_cvxpy_constraints(self, cp_vars: List[cp.Variable]) -> List[cp.Expression]:
        """
        Traduit self.list_cstr en contraintes CVXPY scalaires.

        Appelé une seule fois depuis CvxpyHandler.optimize(), après que toutes
        les contraintes ont été accumulées (add_linear_variable / add_quad_variable).
        """
        cvxpy_constraints = []
        for cstr in self.list_cstr:
            expr = self._build_trace_expr(cstr, cp_vars)
            bound_type = cstr["bound_type"]
            lb = cstr["lb"]
            ub = cstr["ub"]

            if bound_type == mosek.boundkey.fx:
                cvxpy_constraints.append(expr == lb)
            elif bound_type == mosek.boundkey.up:
                cvxpy_constraints.append(expr <= ub)
            elif bound_type == mosek.boundkey.lo:
                cvxpy_constraints.append(expr >= lb)
            else:
                logger_mosek.warning(
                    "Contrainte '%s' : bound_type inconnu (%s), ignorée.",
                    cstr.get("name", "?"),
                    bound_type,
                )
        return cvxpy_constraints

    def _build_trace_expr(self, cstr: dict, cp_vars: List[cp.Variable]) -> cp.Expression:
        """
        Construit Σ_m trace(C_m @ P_m) pour une contrainte donnée.

        Regroupe les termes (i, j, value) par matrice m, construit la matrice
        coefficient C_m symétrique, et retourne la somme des traces.
        """
        grouped = defaultdict(list)
        for m, i, j, v in zip(cstr["num_matrix"], cstr["i"], cstr["j"], cstr["value"]):
            grouped[int(m)].append((int(i), int(j), float(v)))

        expr = 0.0
        for m, terms in grouped.items():
            P = cp_vars[m]
            dim = P.shape[0]
            C = np.zeros((dim, dim))
            for i, j, v in terms:
                C[i, j] += v
                if i != j:
                    C[j, i] += v
            expr = expr + cp.trace(C @ P)

        return expr

    def add_task(self, task):
        pass

    def add_to_task(self):
        pass
