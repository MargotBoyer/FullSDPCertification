from tabnanny import verbose
import numpy as np
from typing import List, Dict
import sys
import os
import mosek
import logging
import time

from ..indexes import Indexes_Mosek_Solver

from .objective_fusion import ObjectiveFusion
from .constraints_fusion import ConstraintsFusion
from .callback_fusion import makeUserCallback

from .results_fusion import (
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    add_all_infos_optimal_values_to_dic,
    save_beta_values_fusion,
)
from ..common_handler_functions import (
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
    initialize_variables,
    save_matrix_csv,
    save_matrix_png,
    Matrices_Solutions,
    get_matrices_variables,
    compute_solutions,
    save_beta_values,
    diagnose_infeasibility,
)
from solve.sdp_solve.run_benchmark import compute_cuts_str
from fastsdp_tools import get_project_path


from mosek.fusion import Model, Domain


from fastsdp_tools.utils import count_calls, add_functions_to_class

logger_mosek = logging.getLogger("Mosek_logger")


class LoggerWriter:
    def write(self, msg):
        msg = msg.rstrip("\n")
        if msg:
            logger_mosek.debug(msg)

    def flush(self):
        pass


@add_functions_to_class(
    initialize_variables,
    save_matrix_csv,
    save_matrix_png,
    get_matrices_variables,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    add_all_infos_optimal_values_to_dic,
    compute_solutions,
    save_beta_values,
    save_beta_values_fusion,
    diagnose_infeasibility,
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
)
class MosekFusionHandler:
    """
    Class to handle the constraints for the MOSEK solver.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ConstraintHandler class.

        Parameters
        ----------
        n: List[int]
            List of the number of neurons in each layer.
        K: int
            Number of layers.
        CHORDAL_DECOMPOSITION: bool
            Whether to use matrix by layers or not.
        last_layer: bool
            Whether the last layer is included in the matrix of the z variables or not.
        betas: bool
            Whether to include the beta variables or not.
        betas_z: bool
            Whether to include the beta variables in the matrixes for z variables.
        zbar: bool
            Whether to include the zbar variables or not.
        CHORDAL_DECOMPOSITION: bool
            Whether to divide matrix variables by layers or not divide them.
        LAST_LAYER: bool
            Whether the last layer is included in the matrix of the z variables or not.
        """
        print("Initializing MosekFusionHandler")
        self.CHORDAL_DECOMPOSITION = kwargs.get("CHORDAL_DECOMPOSITION", False)
        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.BETAS = kwargs.get("BETAS", False)
        self.BETAS_Z = kwargs.get("BETAS_Z", False)
        self.ZBAR = kwargs.get("ZBAR", False)

        self.n = kwargs.get("n", None)
        self.K = kwargs.get("K", None)

        self.folder_name = kwargs.pop("folder_name", None)
        self.name = kwargs.pop("name", None)

        self.ytrue = kwargs.get("ytrue", None)
        self.ytarget = kwargs.get("ytarget", None)
        print('TEST TARGET IN HANDLER : ', self.ytarget)

        self.epsilon = kwargs.pop("epsilon", None)

        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons", None)
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons", None)

        self.indexes_matrices = Indexes_Mosek_Solver(**kwargs)
        self.indexes_variables = self.indexes_matrices  # même objet fusionné

        # print(
        #     "\n \n \n INDEXES \n \n \n : ",
        # )
        # self.print_index_variables_matrices()

        self.final_number_constraints = None

        self.Objective = ObjectiveFusion(
            self.indexes_matrices, self.indexes_variables, **kwargs
        )
        self.Constraints = ConstraintsFusion(
            self.indexes_matrices,
            self.indexes_variables,
            **kwargs,
        )

        self.vector_variables = []

    def initiate_env(self, verbose : bool = False):
        """
        Initialize the model of MOSEK solver.
        """
        logger_mosek.info("Initializing MOSEK solver")
        self.verbose = verbose
        self.model = Model(self.name)
        self.model.__enter__()

        self.indexes_matrices.current_matrices_variables = []
        self.vector_variables = []
        self.Constraints.reinitialize(verbose)
        self.Constraints.add_model(self.model)
        self.Objective.reinitialize(verbose)
        self.Objective.add_model(self.model)
        return self

    def adjust_solver_parameters(self, **parameters):
        """
        Adjust the parameters of the MOSEK solver.
        Parameters
        ----------
        parameters: dict
            The parameters to adjust.
        """

        # ===== TOLÉRANCES STRICTES pour réduire le gap primal-dual =====
        # Réduire le gap relatif entre primal et dual (était 1e-3, c'est trop lâche !)
        self.model.setSolverParam("intpntTolRelGap", 1e-8)  # 1e-3 → 1e-8
        
        # Faisabilité primale et duale plus stricte
        self.model.setSolverParam("intpntTolPfeas", 1e-8)   # 1e-3 → 1e-8
        self.model.setSolverParam("intpntTolDfeas", 1e-8)   # 1e-3 → 1e-8

        # ===== TEMPS MAX =====
        # Limiter le temps de calcul à 3600 secondes (1h)
        self.model.setSolverParam("optimizerMaxTime", 3600.0)

        # ===== ITÉRATIONS MAX =====
        # Augmenter le nombre d'itérations pour convergence forcée
        self.model.setSolverParam("intpntMaxIterations", 300)

        # ===== THREADS =====
        # Nombre de threads (augmenter si vous avez CPU disponible)
        self.model.setSolverParam("numThreads", 4)

        # ===== AUTRES PARAMÈTRES =====
        # Write solutions of the optimization problem
        self.model.setSolverParam("ptfWriteSolutions", "on")
        
        # ===== OPTIONS OPTIONNELLES COMMENTÉES =====
        # Décommenter si vous avez des problèmes numériques :
        # self.model.setSolverParam("presolveUse", "off")
        # self.model.setSolverParam("optimizer", "dualSimplex")

    @count_calls(
        "init_variables"
    )  # Create an attribute init_variable to count the number of calls of this function
    def add_matrix_variable(self, name: str, dim: int):
        """
        Add a matrix variable of dimension dim.
        """
        logger_mosek.debug(f"Adding a variable matrix {name} of dimension %s", dim)
        if any(
            d["name"] == name for d in self.indexes_matrices.current_matrices_variables
        ):
            logger_mosek.warning(
                f"Variable matrix {name} already exists. Skipping addition."
            )
        else:
            if verbose :
                print(f"Adding a variable matrix {name} of dimension %s", dim)
            logger_mosek.debug(f"Variable matrix {name} added.")
            self.indexes_matrices.current_matrices_variables.append(
                {"name": name, "dim": dim, "value": Matrices_Solutions()}
            )
        var = self.model.variable(name, Domain.inPSDCone(dim))

    def add_vector_variable(self, name: str, dim: int):
        """
        Add a vector variable of dimension dim."""
        logger_mosek.info(f"Adding a vector variable {name} of dimension %s", dim)
        self.vector_variables.append(dim)
        x = self.model.variable(name, dim, Domain.unbounded())
        # x = self.model.variable(name, dim, Domain.inRange(lower_bound, upper_bound))
        # x = self.model.variable(name, n, Domain.binary())

    def initialize_constraints(self):
        """
        Initialize the number of constraints.

        Parameters
        ----------
        num_constraints: int
            The number of constraints to initialize.
        """
        logger_mosek.info(
            f"Initializing {self.Constraints.current_num_constraint} constraints"
        )
        # No need to initialize the number of constraints here with the fusion API
        self.final_number_constraints = self.Constraints.current_num_constraint

    def cleanup_mosek(self):
        """Close MOSEK environment en model."""
        logger_mosek.info("Cleaning up MOSEK environment and model \n \n \n")
        if hasattr(self, 'model') and self.model:
            self.model.__exit__(None, None, None)

    def add_constraints(self):
        """
        Add constraints to the model.
        """
        raise NotImplementedError(
            "The method add_constraints is not implemented in the base class."
        )

    def add_objective(self):
        """
        Add the objective function to the model.
        """
        raise NotImplementedError(
            "The method add_objective is not implemented in the base class."
        )

    def is_feasible(self, variables_matrices, precision: float = 1e-6) -> bool:
        """
        Check if the constraint is feasible.

        Parameters
        ----------
        variables_matrices: List[float]
            The value of the variable z.

        Returns
        -------
        bool
            True if the constraint is feasible, False otherwise.
        """
        for constraint in self.Constraints.list_cstr:
            #print("Adding constraint : ", constraint["name"])
            try:
                val = 0
                for index in range(len(constraint["num_matrix"])):

                    num_matrix = constraint["num_matrix"][index]
                    i = constraint["i"][index]
                    j = constraint["j"][index]
                    coeff = constraint["value"][index]
                    val_matrix = variables_matrices[num_matrix][i][j]

                    if i != j:
                        coeff *= 2
                    print(
                        f"num_matrix : {num_matrix}, i : {i}, j : {j}, coeff : {coeff}, val_matrix : {val_matrix}"
                    )

                    val += coeff * val_matrix

                lb = constraint["lb"]
                ub = constraint["ub"]

                if val < lb - precision:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is not feasible: {val} < {lb}"
                    )
                    print("Constraint  : ", constraint)
                    return False
                if val > ub + precision:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is not feasible: {val} > {ub}"
                    )
                    print("Constraint  : ", constraint)
                    return False
                else:
                    logger_mosek.debug(
                        f"Constraint {constraint['name']} is feasible: {val} in [{lb}, {ub}]"
                    )
            except Exception as e:
                logger_mosek.error(f"Error in constraint {constraint['name']}: {e}")
                print("Constraint  : ", constraint)
        return True

    def value_solution(self, variables_matrices):
        """
        Compute the value of the solution.

        Parameters
        ----------
        variables_matrices: List[float]
            The value of the variable z.

        Returns
        -------
        float
            The value of the solution.
        """

        try:
            val = self.Objective.constant
            for index in range(len(self.Objective.list_indexes_matrixes)):
                num_matrix = self.Objective.list_indexes_matrixes[index]
                i = self.Objective.list_indexes_variables_i[index]
                j = self.Objective.list_indexes_variables_j[index]

                coeff = self.Objective.list_values[index]
                val_matrix = variables_matrices[num_matrix][i][j]

                if i != j:
                    coeff *= 2
                val += coeff * val_matrix

            return val
        except Exception as e:
            logger_mosek.error(f"Error in computing the objective: {e}")
            return None

    def define_objective_sense(self):
        """
        Define the objective sense : necessary for the classic MOSEK api (definition of the sense separately from the objective).
        """
        pass

    def optimize(self):
        """
        Optimize the task.
        """
        self.callback = makeUserCallback(model=self.model, maxtime=20000)
        self.model.setDataCallbackHandler(self.callback)
        logger_mosek.info("Optimizing the model")
        self.model.solve()

    def write_model(self, cuts: List = [], RLT_prop : float = 0.0, data_index : int = None, ytarget : int = None):
        """
        Write the results of the optimization to a file.
        """
        logger_mosek.info("Writing results to file...")
        cuts_str = compute_cuts_str(cuts)

        print(
            "Writing ptf : ",
            f"{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_fusion.ptf",
        )

        self.model.writeTask(
            get_project_path(
                f"{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_fusion.ptf"
            )
        )
        logger_mosek.info(
            f"Results written to {get_project_path(f'{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_fusion.ptf')}"
        )

    def print_solver_info(self, verbose: bool = False):
        """
        Print the information of the solver.
        """

        def mosek_to_logger(msg):
            msg = msg.rstrip("\n")
            if msg:  # Évite les messages vides
                logger_mosek.debug(msg)

        if verbose:
            self.model.setLogHandler(LoggerWriter())

    def get_solution_status(self):
        """
        Get the status of the optimization.
        """
        self.problem_status = self.model.getProblemStatus()
        self.solution_status = self.model.getPrimalSolutionStatus()
        logger_mosek.debug("Solution status: %s", self)
        return self.solution_status

    def get_num_iterations(self):
        """
        Get the number of iterations of the optimization.
        """
        num_iterations = self.model.getSolverIntInfo("intpntIter")
        return num_iterations

    def get_solution(self, **kwargs):
        """
        Get the solution of the optimization.
        """
        name_solution = kwargs.get("name_solution", None)
        
        print("name solution : ", name_solution)
        psd_var = self.model.getVariable(name_solution)
        dim = kwargs.get("dim", None)
        return psd_var.level().reshape((dim, dim))

    def get_dual_variables(self):
        for ind, ctsr in enumerate(self.Constraints.list_cstr):
            constraint_name = ctsr["name"]
            try:
                dual_value = self.model.getConstraint(constraint_name).dual()
                if dual_value is None:
                    print(f"Variable duale None (solution duale non disponible) pour {constraint_name}")
                    continue
                self.Constraints.list_cstr[ind]["dual_value"] = round(dual_value[0], 6)
            except Exception as e:
                print(f"Impossible de récupérer la variable duale pour {constraint_name}: {e}")
