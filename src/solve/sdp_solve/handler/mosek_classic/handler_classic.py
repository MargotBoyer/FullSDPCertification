import numpy as np
from typing import List, Dict
import sys
import os
import mosek
import logging
import time

from ..indexes import Indexes_Mosek_Solver
from .objective_classic import ObjectiveClassic
from .constraints_classic import ConstraintsClassic
from .results_classic import (
    add_all_infos_optimal_values_to_dic,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    reconstruct_matrix,
)
from .callback_classic import makeUserCallback

from solve.sdp_solve.run_benchmark import compute_cuts_str

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

from fastsdp_tools.utils import count_calls, add_functions_to_class, get_project_path

logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    initialize_variables,
    reconstruct_matrix,
    save_matrix_csv,
    save_matrix_png,
    add_all_infos_optimal_values_to_dic,
    get_matrices_variables,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
    compute_solutions,
    save_beta_values,
    diagnose_infeasibility,
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
)
class MosekClassicHandler:
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
        print("Initializing MosekClassicHandler")
        self.CHORDAL_DECOMPOSITION = kwargs.get("CHORDAL_DECOMPOSITION", False)

        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.BETAS = kwargs.get("BETAS", False)
        self.BETAS_Z = kwargs.get("BETAS_Z", False)
        self.ZBAR = kwargs.get("ZBAR", False)
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons", None)
        self.stable_active_neurons = kwargs.get("stable_active_neurons", None)

        self.n = kwargs.get("n", None)
        self.K = kwargs.get("K", None)

        self.folder_name = kwargs.pop("folder_name", None)
        print("\n \n folder name dans handler: ", self.folder_name)
        self.name = kwargs.pop("name", None)

        self.epsilon = kwargs.pop("epsilon", None)
        self.solver_time_limit = kwargs.pop("solver_time_limit", None)
        self.rescode = None

        self.ytrue = kwargs.get("ytrue", None)
        self.ytarget = kwargs.get("ytarget", None)



        self.indexes_matrices = Indexes_Mosek_Solver(**kwargs)
        self.indexes_variables = self.indexes_matrices  # même objet fusionné

        # print(
        #     "\n \n \n INDEXES \n \n \n : ",
        # )
        # self.print_index_variables_matrices()

        self.vector_variables = []
        self.final_number_constraints = None

        self.Objective = ObjectiveClassic(
            self.indexes_matrices, self.indexes_variables, **kwargs
        )
        self.Constraints = ConstraintsClassic(
            self.indexes_matrices,
            self.indexes_variables,
            **kwargs,
        )

    def initiate_env(self, verbose: bool = False):
        """
        Initialize the task and env of MOSEK solver.
        Add log stream to the task."
        """
        logger_mosek.info("Initializing MOSEK solver")
        self.verbose = verbose
        if self.verbose:
            print("Initializing MOSEK solver")
        self.env = mosek.Env()
        self.task = self.env.Task(0, 0)
        self.env.__enter__()  # Équivalent à entrer dans le bloc "with"
        self.task.__enter__()  # Équivalent à entrer dans le bloc "with"
        if self.verbose:
            print("Adding callback to the task")

        usercallback = makeUserCallback(maxtime=20000, task=self.task)
        self.task.set_InfoCallback(usercallback)

        self.adjust_solver_parameters()

        self.indexes_matrices.current_matrices_variables = []
        self.vector_variables = []
        self.Objective.add_task(self.task)
        self.Objective.reinitialize(verbose)
        self.Constraints.add_task(self.task)
        self.Constraints.reinitialize(verbose)
        return self  # Pour permettre le chaînage des méthodes

    def adjust_solver_parameters(self, **parameters):
        """
        Adjust the parameters of the MOSEK solver.
        Parameters
        ----------
        parameters: dict
            The parameters to adjust.
        """
        print("Adjusting MOSEK solver parameters")
        
        # ===== TOLÉRANCES STRICTES pour réduire le gap primal-dual =====
        # Réduire le gap relatif entre primal et dual
        self.task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, 1e-3)  # 1e-6 → 1e-8 : plus strict
        
        # Faisabilité primale et duale plus stricte
        self.task.putdouparam(mosek.dparam.intpnt_tol_pfeas, 1e-3)    # Gap primal plus petit
        self.task.putdouparam(mosek.dparam.intpnt_tol_dfeas, 1e-3)    # Gap dual plus petit
        
        # ===== AUGMENTER LES ITÉRATIONS =====
        # Par défaut ~300, vous pouvez l'augmenter pour forcer la convergence
        self.task.putintparam(mosek.iparam.intpnt_max_iterations, 400)
        
        # ===== SCALING (pour les problèmes mal conditionnés) =====
        # Aide à réduire les problèmes numériques
        # self.task.putintparam(mosek.iparam.intpnt_scaling, mosek.scalingtype.free)
        
        # ===== THREADS =====
        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        self.task.putintparam(mosek.iparam.num_threads, num_threads)

        # ===== TIME LIMIT =====
        if self.solver_time_limit is not None:
            self.task.putdouparam(mosek.dparam.optimizer_max_time, float(self.solver_time_limit))
            print(f"MOSEK time limit set to {self.solver_time_limit}s")

        # ===== OPTIONS OPTIONNELLES COMMENTÉES =====
        # Décommenter si vous avez des problèmes numériques :      
        # self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        # self.task.putdouparam(mosek.dparam.optimizer_max_time, 3600)  # Limite temps à 3600s

    @count_calls(
        "init_variables"
    )  # Create an attribute init_variable to count the number of calls of this function
    def add_matrix_variable(self, name: str, dim: int):
        """
        Add a matrix variable of dimension dim to the task.
        """
        logger_mosek.debug(f"Adding a variable matrix {name} of dimension %s", dim)
        if self.verbose:
            print("Adding a variable matrix %s of dimension %s", name, dim)
        if any(
            d["name"] == name for d in self.indexes_matrices.current_matrices_variables
        ):
            logger_mosek.debug(
                f"Variable matrix {name} already exists. Skipping addition."
            )
            return
        else:
            if self.verbose:
                print(f"Adding a variable matrix {name} of dimension %s", dim)
            logger_mosek.debug(f"Variable matrix {name} added.")
            self.indexes_matrices.current_matrices_variables.append(
                {"name": name, "dim": dim, "value": Matrices_Solutions()}
            )
            self.task.appendbarvars([dim])

    def add_vector_variable(self, name: str, dim: int):
        """
        Add a vector variable of dimension dim to the task."""
        logger_mosek.info(f"Adding a variable vector {name} of dimension %s", dim)
        self.vector_variables.append(dim)
        self.task.appendvars(dim)

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
        self.task.appendcons(self.Constraints.current_num_constraint)
        self.final_number_constraints = self.Constraints.current_num_constraint

    def cleanup_mosek(self):
        """Ferme proprement l'environnement et la tâche MOSEK."""
        logger_mosek.info("Cleaning up MOSEK environment and task \n \n \n")
        if self.task:
            self.task.__exit__(None, None, None)  # Équivalent à sortir du bloc "with"
            self.task = None
        if self.env:
            self.env.__exit__(None, None, None)  # Équivalent à sortir du bloc "with"
            self.env = None

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
        Define the objective sense.
        """
        self.task.putobjsense(mosek.objsense.minimize)

    def optimize(self):
        """
        Optimize the task.
        """
        logger_mosek.info("Optimizing the task")
        self.rescode = self.task.optimize()

    def is_time_limit(self):
        return self.rescode == mosek.rescode.trm_max_time

    def write_model(
        self,
        cuts: List = [],
        RLT_prop: float = 0.0,
        data_index: int = None,
        ytarget: int = None,
    ):
        """
        Write the results of the optimization to a file.
        """
        logger_mosek.info("Writing results to file...")
        cuts_str = compute_cuts_str(cuts)
        print(
            "Writing results fo file : ",
            get_project_path(
                f"{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_classic.ptf"
            ),
        )
        self.task.writedata(
            get_project_path(
                f"{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_classic.ptf"
            )
        )
        # self.task.writedata(
        #     get_project_path(f"{self.folder_name}/{self.name}_{cuts_str}_classic.ptf")
        # )
        logger_mosek.info(
            f"Results written to {get_project_path(f'{self.folder_name}/{self.name}/{self.name}_{cuts_str}_ind={data_index}_ytarget={ytarget}_RLT={RLT_prop}_classic.ptf')}"
        )

    def print_solver_info(self, verbose: bool = False):
        def mosek_to_logger(msg):
            msg = msg.rstrip("\n")
            if msg:
                print(msg, flush=True)

        if verbose:
            self.task.set_Stream(mosek.streamtype.log, mosek_to_logger)

    def get_solution_status(self):
        """
        Get the status of the optimization.
        """
        self.status = self.task.getsolsta(mosek.soltype.itr)
        return self.status

    def get_num_iterations(self):
        """
        Get the number of iterations of the optimization.
        """
        num_iterations = self.task.getintinf(mosek.iinfitem.intpnt_iter)
        return num_iterations

    def get_solution(self, **kwargs):
        """
        Get the solution of the optimization.
        """
        ind_solution = kwargs.get("ind_solution", None)
        dim = kwargs.get("dim", None)
        mat = self.task.getbarxj(mosek.soltype.itr, ind_solution)
        return self.reconstruct_matrix(dim, mat)

    def get_dual_variables(self):
        """
        Get the dual variables of the optimization.
        """
        dual_variables = self.task.gety(mosek.soltype.itr)

        # Check solution status
        prosta = self.task.getprosta(mosek.soltype.itr)
        solsta = self.task.getsolsta(mosek.soltype.itr)

        print(f"Primal status: {prosta}")
        print(f"Solution status: {solsta}")

        y = [0.0] * self.final_number_constraints

        assert len(dual_variables) == len(self.Constraints.list_cstr), "Le nombre de variables duales ne correspond pas au nombre de contraintes."
        for i in range(len(dual_variables)):
            self.Constraints.list_cstr[i]["dual_value"] = dual_variables[i]

            name = self.Constraints.list_cstr[i]["name"]

            val1 = dual_variables[i]

            print(f"Constraint {name}: {val1}")

        return dual_variables
