import numpy as np
import logging
import os
from typing import List
try:
    import cvxpy as cp
except ImportError:
    cp = None

from ..indexes import Indexes_Mosek_Solver
from .objective_cvxpy import ObjectiveCvxpy
from .constraints_cvxpy import ConstraintsCvxpy
from .results_cvxpy import (
    add_all_infos_optimal_values_to_dic,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
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
from fastsdp_tools.utils import add_functions_to_class

logger_mosek = logging.getLogger("Mosek_logger")


@add_functions_to_class(
    initialize_variables,
    print_index_variables_matrices,
    num_matrices_variables,
    print_num_variables,
    save_matrix_csv,
    save_matrix_png,
    get_matrices_variables,
    compute_solutions,
    save_beta_values,
    diagnose_infeasibility,
    add_all_infos_optimal_values_to_dic,
    is_status_optimal,
    is_status_infeasible,
    is_status_unknown,
)
class CvxpyHandler:
    """
    Handler CVXPY : remplace MosekClassicHandler en utilisant CVXPY comme couche de modélisation.

    Interface identique à MosekClassicHandler du point de vue de SDPSolver.run_optimization() :
    mêmes méthodes (initiate_env, add_matrix_variable, initialize_constraints, optimize,
    cleanup_mosek, ...), mais sans aucun appel à l'API MOSEK bas-niveau.

    Backends supportés : MOSEK via CVXPY, SCS, CVXOPT, CLARABEL, ...
    Passer cp_solver="SCS" pour utiliser le solver SCS (libre, sans licence).

    Paramètre cp_solver_kwargs : dict passé directement à cp.Problem.solve(**cp_solver_kwargs).
    Exemple pour MOSEK :
        cp_solver_kwargs={"mosek_params": {"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6}}
    Exemple pour SCS :
        cp_solver_kwargs={"eps": 1e-4, "max_iters": 50000}
    """

    def __init__(self, **kwargs):
        print("Initializing CvxpyHandler")
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
        self.name = kwargs.pop("name", None)
        self.epsilon = kwargs.pop("epsilon", None)
        self.solver_time_limit = kwargs.pop("solver_time_limit", None)
        self.ytrue = kwargs.get("ytrue", None)
        self.ytarget = kwargs.get("ytarget", None)

        self.cp_solver = kwargs.pop("cp_solver", "MOSEK")
        self.cp_solver_kwargs = kwargs.pop("cp_solver_kwargs", {}) or {}

        self.indexes_matrices = Indexes_Mosek_Solver(**kwargs)
        self.indexes_variables = self.indexes_matrices

        self.cp_matrices = []
        self.psd_constraints = []
        self.problem = None
        self.status_cp = None
        self.primal_obj_value = None
        self.dual_obj_value = None
        self.optimal_value = None
        self.is_robust = False
        self.verbose = False
        self.time_pretreatment = 0.0
        self.time_solving = 0.0
        self.rescode = None

        self.Objective = ObjectiveCvxpy(
            self.indexes_matrices, self.indexes_variables, **kwargs
        )
        self.Constraints = ConstraintsCvxpy(
            self.indexes_matrices, self.indexes_variables, **kwargs
        )

        print("CVXPY Handler initialized with solver:", self.cp_solver)

    def initiate_env(self, verbose: bool = False):
        """Réinitialise l'état pour un nouveau run (équivalent à la création d'un task MOSEK)."""
        self.verbose = verbose
        self.is_robust = False
        self.cp_matrices = []
        self.psd_constraints = []
        self.problem = None
        self.status_cp = None
        self.primal_obj_value = None
        self.dual_obj_value = None
        self.optimal_value = None
        self.indexes_matrices.current_matrices_variables = []
        self.Objective.reinitialize(verbose)
        self.Constraints.reinitialize(verbose)
        return self

    def add_matrix_variable(self, name: str, dim: int):
        """Crée une variable matricielle SDP symétrique de taille dim×dim."""
        if any(d["name"] == name for d in self.indexes_matrices.current_matrices_variables):
            logger_mosek.debug("Variable matrix '%s' already exists. Skipping.", name)
            return
        P = cp.Variable((dim, dim), symmetric=True, name=name)
        self.cp_matrices.append({"name": name, "dim": dim, "var": P})
        self.psd_constraints.append(P >> 0)
        self.indexes_matrices.current_matrices_variables.append(
            {"name": name, "dim": dim, "value": Matrices_Solutions()}
        )
        logger_mosek.debug("Added CVXPY matrix variable '%s' of dim %d.", name, dim)

    def add_vector_variable(self, name: str, dim: int):
        logger_mosek.warning(
            "add_vector_variable('%s', %d) : not supported in CvxpyHandler, skipped.", name, dim
        )

    def initialize_constraints(self):
        pass

    def optimize(self):
        """
        Assemble et résout le problème CVXPY.

        Ordre :
          1. Récupère les variables CVXPY [P_0, P_1, ...]
          2. Construit l'objectif via ObjectiveCvxpy.to_cvxpy_expr()
          3. Traduit list_cstr en contraintes CVXPY via ConstraintsCvxpy.build_cvxpy_constraints()
          4. Ajoute les contraintes PSD P_m ⪰ 0
          5. Résout avec cp.Problem.solve(solver=self.cp_solver)
        """
        cp_vars = [entry["var"] for entry in self.cp_matrices]
        obj_expr = self.Objective.to_cvxpy_expr(cp_vars)
        cstr_list = self.Constraints.build_cvxpy_constraints(cp_vars)

        self.problem = cp.Problem(
            cp.Minimize(obj_expr),
            cstr_list + self.psd_constraints,
        )

        solver_kwargs = dict(self.cp_solver_kwargs)
        if self.solver_time_limit is not None:
            if self.cp_solver == "MOSEK":
                solver_kwargs.setdefault("mosek_params", {})
                solver_kwargs["mosek_params"].setdefault(
                    "MSK_DPAR_OPTIMIZER_MAX_TIME", float(self.solver_time_limit)
                )
            elif self.cp_solver == "SCS":
                solver_kwargs.setdefault("max_iters", 100000)

        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        if self.cp_solver == "MOSEK":
            solver_kwargs.setdefault("mosek_params", {})
            solver_kwargs["mosek_params"].setdefault("MSK_IPAR_NUM_THREADS", num_threads)
            solver_kwargs["mosek_params"].setdefault("MSK_DPAR_INTPNT_TOL_REL_GAP", 0.001)
            solver_kwargs["mosek_params"].setdefault("MSK_DPAR_INTPNT_TOL_PFEAS", 0.001)
            solver_kwargs["mosek_params"].setdefault("MSK_DPAR_INTPNT_TOL_DFEAS", 0.001)
            solver_kwargs["mosek_params"].setdefault("MSK_IPAR_INTPNT_MAX_ITERATIONS", 400)

        logger_mosek.info(
            "CVXPY solving with solver=%s, %d constraints, %d matrices.",
            self.cp_solver,
            len(cstr_list) + len(self.psd_constraints),
            len(cp_vars),
        )

        self.problem.solve(solver=self.cp_solver, verbose=self.verbose, **solver_kwargs)
        self.status_cp = self.problem.status
        self.primal_obj_value = self.problem.value
        logger_mosek.info("CVXPY solved : status=%s, value=%s", self.status_cp, self.primal_obj_value)

    def get_solution(self, ind_solution: int, dim: int, **kwargs) -> np.ndarray:
        """Retourne la matrice solution P_m (numpy array dim×dim)."""
        var = self.cp_matrices[ind_solution]["var"]
        if var.value is not None:
            return np.array(var.value)
        logger_mosek.warning(
            "get_solution : matrice %d non disponible (problème non résolu ou infaisable). Retourne zéros.",
            ind_solution,
        )
        return np.zeros((dim, dim))

    def get_solution_status(self):
        """Mémorise et retourne le statut CVXPY (string) dans self.status."""
        self.status = self.status_cp
        return self.status

    def get_num_iterations(self) -> int:
        if self.problem is None:
            return 0
        try:
            return int(self.problem.solver_stats.num_iters or 0)
        except Exception:
            return 0

    def get_dual_variables(self):
        logger_mosek.warning("get_dual_variables : not available with CvxpyHandler.")

    def cleanup_mosek(self):
        pass

    def define_objective_sense(self):
        pass

    def print_solver_info(self, verbose: bool = False):
        pass

    def is_time_limit(self) -> bool:
        return self.status_cp in frozenset({"user_limit"}) and self.solver_time_limit is not None

    def write_model(self, cuts=None, RLT_prop=0.0, data_index=None, ytarget=None):
        logger_mosek.info("write_model : not implemented for CvxpyHandler.")

    def adjust_solver_parameters(self):
        pass

    def add_task(self, task):
        pass

    def _dynamic_conic_bundle_not_implemented(self):
        raise NotImplementedError(
            "La dynamic conic bundle method (src/dynamic_conic_bundle/) n'est "
            "implémentée que pour solver='mosek_classic' pour l'instant."
        )

    def setup_dualization(self, dualizable_names):
        self._dynamic_conic_bundle_not_implemented()

    def get_dualized_constraints_data(self):
        self._dynamic_conic_bundle_not_implemented()

    def get_constraint_dualization_data(self, names):
        self._dynamic_conic_bundle_not_implemented()

    def resolve_dualized(self, theta):
        self._dynamic_conic_bundle_not_implemented()

    def teardown_dualization(self):
        self._dynamic_conic_bundle_not_implemented()


