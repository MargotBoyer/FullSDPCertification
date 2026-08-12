import numpy as np
from typing import List, Dict
import sys
import os
import mosek
import logging
import time

from ..indexes import Indexes_Mosek_Solver
from ..constraints import ConstraintRole
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

    # ------------------------------------------------------------------
    # Dynamic conic bundle support (src/dynamic_conic_bundle/).
    #
    # These four methods are the only place where the dynamic conic bundle
    # method touches MOSEK — everything else (subgradient computation, proximal
    # master problem, add/drop of dualized constraints) lives in pure Python/numpy
    # in src/dynamic_conic_bundle/. Not used by the classic solve()/run_optimization()
    # path, which never calls them.
    #
    # Phase 2 scope: single SDP matrix only (CHORDAL_DECOMPOSITION=False) — the
    # interaction with the chordal decomposition (multiple P_k blocks) is a TODO
    # for Phase 3 (see task-dynamic-conic-bundle.md).
    # ------------------------------------------------------------------

    def setup_dualization(self, dualizable_names: List[str]):
        """
        One-time setup before a dynamic conic bundle run on the model built by
        build_model(): cache the base objective triplets, and for each named
        DUALIZABLE constraint, cache its signed triplet/rhs and deactivate its
        bound in the task (mosek.boundkey.fr, i.e. removed from the hard model).

        Must be called once per build_model() call, before any resolve_dualized().
        """
        assert not self.CHORDAL_DECOMPOSITION, (
            "setup_dualization: seul le cas CHORDAL_DECOMPOSITION=False (une seule "
            "matrice SDP) est supporté en Phase 2 — l'interaction avec la "
            "décomposition chordale est un TODO Phase 3 (task-dynamic-conic-bundle.md)."
        )

        self.Objective.format_obj()
        self._cb_base_obj = (
            np.asarray(self.Objective.num_matrix, dtype=np.int32),
            np.asarray(self.Objective.i, dtype=np.int32),
            np.asarray(self.Objective.j, dtype=np.int32),
            np.asarray(self.Objective.value, dtype=np.float64),
        )
        self._cb_base_obj_constant = float(self.Objective.constant)

        self._cb_dualized = []
        self._cb_saved_bounds = {}

        for name in dualizable_names:
            entry, idx, bound_type, lb, ub = self._dualization_triplet_for(name)
            self._cb_dualized.append(entry)
            self._cb_saved_bounds[idx] = (bound_type, lb, ub)
            self.task.putconbound(idx, mosek.boundkey.fr, -1e30, 1e30)

        logger_mosek.info(
            "setup_dualization: %d contraintes dualisées, %d matrices distinctes référencées",
            len(self._cb_dualized),
            len({int(nm) for d in self._cb_dualized for nm in d["num_matrix"]}),
        )

    def _dualization_triplet_for(self, name: str):
        """
        Construit le triplet signé/normalisé en <= (cf. setup_dualization) pour la
        contrainte DUALIZABLE `name`, sans muter le task. Factorisation commune entre
        setup_dualization() (qui en plus désactive la borne dans le task) et
        get_constraint_dualization_data() (lecture pure, utilisée par le mode
        dynamique pour évaluer la violation de contraintes pas encore dualisées).
        """
        name_to_idx = {c["name"]: idx for idx, c in enumerate(self.Constraints.list_cstr)}
        assert name in name_to_idx, f"Contrainte dualisable inconnue : {name}"
        idx = name_to_idx[name]
        c = self.Constraints.list_cstr[idx]
        assert c["role"] == ConstraintRole.DUALIZABLE, (
            f"La contrainte '{name}' n'est pas taguée ConstraintRole.DUALIZABLE "
            "(mark_current_dualizable() n'a pas été appelée à sa création)."
        )
        bound_type = c["bound_type"]
        assert bound_type in (mosek.boundkey.up, mosek.boundkey.lo, mosek.boundkey.fx), (
            f"Type de borne non supporté pour la dualisation : {bound_type}"
        )
        # Normalise en <=, forme sur laquelle porte theta_r >= 0 (fx : signe libre).
        if bound_type == mosek.boundkey.lo:
            sign = -1.0
            rhs = -float(c["lb"])
        else:
            sign = 1.0
            rhs = float(c["ub"] if bound_type == mosek.boundkey.up else c["lb"])

        entry = {
            "name": name,
            "idx": idx,
            "free_sign": bound_type == mosek.boundkey.fx,
            "num_matrix": np.asarray(c["num_matrix"], dtype=np.int32),
            "i": np.asarray(c["i"], dtype=np.int32),
            "j": np.asarray(c["j"], dtype=np.int32),
            "value": sign * np.asarray(c["value"], dtype=np.float64),
            "rhs": rhs,
        }
        return entry, idx, bound_type, c["lb"], c["ub"]

    def get_constraint_dualization_data(self, names: List[str]) -> List[dict]:
        """
        Version en lecture seule de setup_dualization() : ne touche pas au task, ne
        désactive rien. Utilisée par le mode dynamique (DynamicConicBundleSolver)
        pour évaluer g_r(X*) sur des contraintes DUALIZABLE pas encore dualisées
        (candidates à l'ajout), à partir de la solution primale déjà obtenue par un
        resolve_dualized() courant — pas de nouvel appel MOSEK.
        """
        return [self._dualization_triplet_for(name)[0] for name in names]

    def get_dualized_constraints_data(self):
        """Retourne la liste des contraintes dualisées mises en cache par
        setup_dualization() — consommée par dynamic_conic_bundle.subgradient
        pour calculer g_r = <A_r, X*> - rhs_r sans revenir dans MOSEK."""
        return self._cb_dualized

    def resolve_dualized(self, theta: dict):
        """
        Résout le sous-problème SDP courant avec l'objectif pénalisé
        C + Σ_r theta_r * A_r (contraintes dualisées désactivées par
        setup_dualization). Ne modifie que l'objectif du task, pas ses contraintes.

        Parameters
        ----------
        theta : dict[str, float]
            Sous-ensemble courant de multiplicateurs {constraint_name: theta_r}
            (notation gardée identique à main.pdf, Algorithme 2 — ne pas confondre
            avec les alpha d'alpha-CROWN, concept sans rapport).
            Un nom absent de `theta` est traité comme theta_r = 0 (mode dynamique :
            seules les contraintes ajoutées au round courant contribuent).

        Returns
        -------
        dict avec "status" (mosek.solsta), "primal_obj" (valeur MOSEK de
        <C+Σtheta_r A_r, X*>), "lb_value" (valeur du dual lagrangien
        h(theta) = primal_obj + cte - Σ theta_r * rhs_r), et "X" (dict
        {num_matrix: np.ndarray} des matrices primales nécessaires au calcul des
        sous-gradients, cf. get_dualized_constraints_data()).
        """
        nm_base, i_base, j_base, v_base = self._cb_base_obj
        coeff: dict = {}
        for k in range(len(nm_base)):
            key = (int(nm_base[k]), int(i_base[k]), int(j_base[k]))
            coeff[key] = coeff.get(key, 0.0) + float(v_base[k])

        theta_sum_rhs = 0.0
        for d in self._cb_dualized:
            t = float(theta.get(d["name"], 0.0))
            if t == 0.0:
                continue
            theta_sum_rhs += t * d["rhs"]
            for k in range(len(d["i"])):
                key = (int(d["num_matrix"][k]), int(d["i"][k]), int(d["j"][k]))
                coeff[key] = coeff.get(key, 0.0) + t * float(d["value"][k])

        if coeff:
            nm_all, i_all, j_all, v_all = zip(*[(nm, i, j, v) for (nm, i, j), v in coeff.items()])
            self.task.putbarcblocktriplet(
                np.array(nm_all, dtype=np.int32),
                np.array(i_all, dtype=np.int32),
                np.array(j_all, dtype=np.int32),
                np.array(v_all, dtype=np.float64),
            )

        self.task.optimize()
        status = self.task.getsolsta(mosek.soltype.itr)

        result = {"status": status, "primal_obj": None, "lb_value": None, "X": {}}
        if status not in (mosek.solsta.optimal, mosek.solsta.prim_and_dual_feas):
            return result

        primal_obj = self.task.getprimalobj(mosek.soltype.itr)
        result["primal_obj"] = primal_obj
        result["lb_value"] = primal_obj + self._cb_base_obj_constant - theta_sum_rhs

        # Extrait X pour TOUTES les matrices du modèle (pas seulement celles des
        # contraintes actuellement dualisées) : le mode dynamique évalue la
        # violation de contraintes DUALIZABLE pas encore dualisées sur ce même X
        # (cf. DynamicConicBundleSolver._solve_dynamic / get_constraint_dualization_data),
        # donc toutes les matrices référencées par des candidates potentielles doivent
        # être disponibles, pas seulement le sous-ensemble déjà actif. Phase 2 scope
        # (CHORDAL_DECOMPOSITION=False) : une seule matrice, coût négligeable.
        for num_matrix in range(self.indexes_matrices.nb_matrices):
            dim = self.indexes_matrices.get_shape_matrix(num_matrix)
            result["X"][num_matrix] = self.get_solution(ind_solution=num_matrix, dim=dim)

        return result

    def teardown_dualization(self):
        """Restaure les contraintes dualisées (bornes d'origine) et l'objectif de
        base dans le task. À appeler une fois à la fin d'un run de conic bundle."""
        for idx, (bound_type, lb, ub) in self._cb_saved_bounds.items():
            self.task.putconbound(idx, bound_type, lb, ub)

        nm_base, i_base, j_base, v_base = self._cb_base_obj
        self.task.putbarcblocktriplet(nm_base, i_base, j_base, v_base)

        self._cb_dualized = []
        self._cb_saved_bounds = {}
