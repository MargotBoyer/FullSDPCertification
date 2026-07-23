import pandas as pd
from typing import List
import torch
from networks import ReLUNN
import mosek
import yaml
import time
import os
import logging
import sys
import itertools
from pydantic import ValidationError
from typing import Dict

import matplotlib.pyplot as plt
import json


from fastsdp_tools import (
    get_project_path,
    create_folder,
    FullCertificationConfig,
    add_functions_to_class,
    add_row_from_dict
)
from ..generic_solver import Solver
from .handler.mosek_fusion import MosekFusionHandler
from .handler.mosek_classic.handler_classic import MosekClassicHandler
try:
    from .handler.cvxpy import CvxpyHandler
except ImportError:
    CvxpyHandler = None
from .run_benchmark import (
    create_all_cuts_to_test,
    adapt_number_RLT,
    compute_number_RLT
)
from .get_variables import get_results, get_results_width_model


from fastsdp_tools import change_to_zero_negative_values


logger_mosek = logging.getLogger("Mosek_logger")


def _append_csv(path: str, row_df: pd.DataFrame) -> None:
    """Append row_df to path, creating the file if needed.

    Fast path (O(1)): when the file already exists and has the same column set,
    append the row without reading the existing data.
    Slow path (O(N)): when schemas differ (e.g. first presolve row vs. result row),
    read the full file, concat, and rewrite so all columns are preserved.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        row_df.to_csv(path, index=False)
        return
    try:
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        row_df.to_csv(path, index=False)
        return
    if set(existing_cols) == set(row_df.columns.tolist()):
        row_df[existing_cols].to_csv(path, mode='a', header=False, index=False)
    else:
        try:
            existing = pd.read_csv(path)
            merged = pd.concat([existing, row_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            merged = row_df
        merged.to_csv(path, index=False)


@add_functions_to_class(
    create_all_cuts_to_test, get_results, get_results_width_model, adapt_number_RLT, compute_number_RLT
)
class SDPSolver(Solver):
    """
    A solver that uses MOSEK to solve the optimization problem.
    """

    def __init__(
        self,
        CHORDAL_DECOMPOSITION = False,  # Union[bool, List[List[int]]]
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        use_fusion: bool = False,
        solver: str = "mosek_classic",  # "mosek_classic" | "mosek_fusion" | "cvxpy"
        cp_solver: str = "MOSEK",       # backend CVXPY : "MOSEK", "SCS", "CLARABEL", ...
        cp_solver_kwargs: dict = None,
        INPUT_IN_VARIABLES = True,  # Union[bool, float]: 0.0=no input vars, 1.0=all, 0<p<1=partial
        solver_time_limit: int = None,
        **kwargs,
    ):
        super().__init__(LAST_LAYER=LAST_LAYER, INPUT_IN_VARIABLES=INPUT_IN_VARIABLES, **kwargs)
        self.solver_time_limit = solver_time_limit

        self.CHORDAL_DECOMPOSITION = CHORDAL_DECOMPOSITION
        # INPUT_IN_VARIABLES normalized to bool by generic_solver.__init__; do not override here
        assert self.keep_penultimate_actives is not None

        self.cuts = kwargs.get("cuts")
        self.all_combinations_cuts = kwargs.get("all_combinations_cuts", False)
        self.create_all_cuts_to_test()
        self.RLT_props = kwargs.get("RLT_props")

        # Résolution use_fusion (ancien param) vs solver (nouveau param)
        if use_fusion and solver == "mosek_classic":
            solver = "mosek_fusion"
        self.solver = solver
        self.use_fusion = (solver == "mosek_fusion")
        self.cp_solver = cp_solver
        self.cp_solver_kwargs = cp_solver_kwargs or {}

        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR

        self.alpha_1 = kwargs.get("alpha_1")
        self.alpha_2 = kwargs.get("alpha_2")

        self.write_model_ptf = kwargs.get("write_model")

        logger_mosek.info(f"Model {self.__class__.__name__} initialized.")

        self.initiate_solver()
        self.only_width_model = False


    @staticmethod
    def parse_yaml_mosek(yaml_file):
        with open(yaml_file, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            validated_config = FullCertificationConfig(**raw_config)
        except ValidationError as e:
            print(f"Erreur de validation du fichier YAML :\n{e}")
            raise

        return dict(
            cuts=validated_config.certification_problem.cuts,
            all_combinations_cuts=validated_config.certification_problem.all_combinations_cuts,
            RLT_props=validated_config.certification_problem.RLT_props,
        )

    def return_solutions(self):
        """
        Return the solutions of the optimization problem.
        """
        return self.handler.indexes_matrices.current_matrices_variables

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = Solver.parse_yaml(yaml_file)
        params_sdp = cls.parse_yaml_mosek(yaml_file)
        return cls(**params, **params_sdp, **kwargs)

    def _common_handler_kwargs(self):
        return dict(
            dataset=self.dataset,
            epsilon=self.epsilon,
            ytrue=self.ytrue,
            ytarget=self.ytarget,
            ytargets=self.ytargets,
            K=self.network.K,
            n=self.network.n,
            W=self.network.W,
            b=self.network.b,
            L=self.L,
            U=self.U,
            CHORDAL_DECOMPOSITION=self.CHORDAL_DECOMPOSITION,
            keep_penultimate_actives=self.keep_penultimate_actives,
            LAST_LAYER=self.LAST_LAYER,
            BETAS=self.BETAS,
            BETAS_Z=self.BETAS_Z,
            ZBAR=self.ZBAR,
            stable_inactives_neurons=self.stable_inactives_neurons,
            stable_actives_neurons=self.stable_actives_neurons,
            folder_name=self.folder_name,
            name=self.name,
            INPUT_IN_VARIABLES=self.INPUT_IN_VARIABLES,
            kept_input_neurons=self.kept_input_neurons,
            pruned_input_neurons=self.pruned_input_neurons,
            solver_time_limit=self.solver_time_limit,
        )

    def initiate_solver(self):
        kw = self._common_handler_kwargs()
        if self.solver == "cvxpy":
            self.handler = CvxpyHandler(
                **kw,
                cp_solver=self.cp_solver,
                cp_solver_kwargs=self.cp_solver_kwargs,
            )
        elif self.solver == "mosek_fusion":
            self.handler = MosekFusionHandler(**kw)
        else:
            self.handler = MosekClassicHandler(**kw)

    def _write_presolve_row(self, cuts: Dict, nb_variables: int):
        """Write a pre-solve row to results.csv right after constraints are built, before MOSEK runs."""
        print("Cuts in _write_presolve_row: ", cuts, flush=True)
        from .run_benchmark import all_possible_cuts
        print("Number of constraints before optimization: ", len(self.handler.Constraints.list_cstr), flush=True)
        nb_constraints = len(self.handler.Constraints.list_cstr)
        print("Number of variables before optimization: ", nb_variables, flush=True)
        dic = {
            "network": self.network_name,
            "model": self.name,
            "dataset": self.dataset_name,
            "data_index": self.data_index,
            "label": self.ytrue,
            "label_predicted": self.label_predicted,
            "target": getattr(self, "ytarget", None),
            "epsilon": self.epsilon,
            "status": "pre-solve",
            "CHORDAL_DECOMPOSITION": str(self.CHORDAL_DECOMPOSITION),
            "LAST_LAYER": self.LAST_LAYER,
            "USE_STABLE_ACTIVES": self.use_active_neurons,
            "USE_STABLE_INACTIVES": self.use_inactive_neurons,
            "Nb_stable_inactives": len(self.stable_inactives_neurons),
            "Nb_stable_actives": len(self.stable_actives_neurons),
            "Nb_constraints": nb_constraints,
            "Nb_variables": nb_variables,
        }
        print("All possible cuts: ", all_possible_cuts, flush=True)
        dic.update({cut: (cut in cuts) for cut in all_possible_cuts})
        print("Cuts added to dic for _write_presolve_row: ", {cut: (cut in cuts) for cut in all_possible_cuts}, flush=True)
        if "RLT" in cuts:
            dic["RLT_prop"] = self.RLT_prop
        path = get_project_path(f"{self.folder_name}/results.csv")

        row_df = pd.DataFrame(dic, index=[0])
        print("Writing pre-solve row to results.csv with data_index:", self.data_index, "target:", getattr(self, 'ytarget', None), flush=True)
        _append_csv(path, row_df)
        print("Pre-solve row written to results.csv with data_index:", self.data_index, "target:", getattr(self, 'ytarget', None), flush=True)
        logger_mosek.debug(f"Pre-solve row written — Nb_constraints={nb_constraints}, Nb_variables={nb_variables}")
        self._write_model_size_csv(cuts, nb_variables)
        print(f"Pre-solve row written to results.csv with {nb_constraints} constraints and {nb_variables} variables.", flush=True)

    
    _TRACKED_CUTS = [
        "RLT",
        "triangularization",
        "McCormick_beta_z",
        "beta_logits_comparaison",
        "beta_logits_comparaison_big_M",
        "sum_beta_logits_equal_logit",
    ]

    def _write_model_size_csv(self, cuts: Dict, nb_variables: int):
        """Append one row to taille_modele.csv with per-cut constraint counts."""
        counts = getattr(self, "_cut_constraint_counts", {})
        baseline = (
            counts.get("baseline_relu", 0)
            + counts.get("baseline_bounds", 0)
            + counts.get("baseline_beta", 0)
            + counts.get("baseline_other", 0)
        )
        row = {
            "data_index": self.data_index,
            "Nb_constraints_total": len(self.handler.Constraints.list_cstr),
            "Nb_variables": nb_variables,
            "Nb_constraints_baseline": baseline,
        }
        for cut in self._TRACKED_CUTS:
            row[f"Nb_constraints_{cut}"] = counts.get(cut, 0)
            row[cut] = cut in cuts
        if "RLT" in cuts:
            row["RLT_prop"] = self.RLT_prop

        path = get_project_path(f"{self.folder_name}/taille_modele.csv")
        row_df = pd.DataFrame(row, index=[0])
        _append_csv(path, row_df)
        logger_mosek.debug(f"taille_modele.csv updated — data_index={self.data_index}")

    def run_optimization(self, cuts: Dict, verbose: bool = False):
        try:
            self.handler.is_robust = False
            if verbose : 
                logger_mosek.debug("RLT_prop in run_optimization: %s", self.RLT_prop)
                logger_mosek.debug("Beginnning of run_optimization with cuts: %s", cuts)
            # self.handler.renew_solver()
            start_pretreatment_time = time.time()
            if verbose :
                print("Initializing ENV...", flush=True)
            self.handler.initiate_env(verbose)
            if verbose :
                print("Intializing ENV : DONE.", flush=True)
            self.handler.print_solver_info(verbose)
            if verbose :
                logger_mosek.debug("Handler initialized.")
            self.add_objective()
            # print(
            #     "Objective indexes matrices: ",
            #     self.handler.Objective.list_indexes_matrixes,
            # )
            # print(
            #     "Objective indexes variables i: ",
            #     self.handler.Objective.list_indexes_variables_i,
            # )
            # print(
            #     "Objective indexes variables j: ",
            #     self.handler.Objective.list_indexes_variables_j,
            # )
            # print(
            #     "Objective indexes variables value: ",
            #     self.handler.Objective.list_values,
            # )
            if verbose : 
                logger_mosek.debug("; Objective created.")
            self.handler.initialize_variables()
            nb_variables = self.handler.print_num_variables()
            if verbose :
                logger_mosek.debug("Variables initialized.")
                print("Adding constraints to the task...", flush=True)
            time_1 = time.time()
            self.adapt_number_RLT()
            self.add_constraints(cuts)  # Constraints must be added after variables
            #self.handler.Constraints.get_histogram_of_coefficients_name_constraint("ReLU Relaxed")
            self._write_presolve_row(cuts, nb_variables)
            if verbose:
                logger_mosek.debug("Constraints added.")
            if self.only_width_model:
                logger_mosek.debug("Only width model, getting results without optimization...")
                self.get_results(cuts, verbose)
                return False
            logger_mosek.debug("not only width model, proceeding to optimization...")
            time_2 = time.time()

            # print(self.Constraints)
            
            self.handler.initialize_constraints()
            if verbose :
                logger_mosek.debug("Constraints initialized.")
                logger_mosek.debug("Number of constraints: %s", len(self.handler.Constraints.list_cstr))
            # # STATISTICS ON PARAMETER VALUES
            # (
            #     histogram_coeff,
            #     min_coeff,
            #     max_coeff,
            #     mean_coeff,
            #     close_to_zero_total_coeff,
            #     histogram_bound,
            #     min_bound,
            #     max_bound,
            #     mean_bound,
            #     close_to_zero_total_bound,
            #     comparaison_by_constraints,
            # ) = self.handler.Constraints.get_histogram_of_coefficients()

            # actives_str = "use_actives" if self.use_active_neurons else "no_actives"
            # width = round((max_coeff - min_coeff) / 100, 5)
            # plt.bar(
            #     histogram_coeff.keys(),
            #     histogram_coeff.values(),
            #     width,
            #     color="g",
            # )
            # plt.savefig(
            #     get_project_path(
            #         f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_coefficients_histogram.png"
            #     )
            # )
            # tab = pd.DataFrame(comparaison_by_constraints)
            # tab.to_csv(
            #     get_project_path(
            #         f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_comparaison_by_constraints.csv"
            #     ),
            #     index=False,
            # )
            # json.dump(
            #     {
            #         "coefficients_histogram": {
            #             "min": min_coeff,
            #             "max": max_coeff,
            #             "mean": mean_coeff,
            #             "close_to_zero": close_to_zero_total_coeff,
            #         },
            #         "bounds_histogram": {
            #             "min": min_bound,
            #             "max": max_bound,
            #             "mean": mean_bound,
            #             "close_to_zero": close_to_zero_total_bound,
            #         },
            #     },
            #     open(
            #         get_project_path(
            #             f"{self.folder_name}/HISTOGRAM/{self.network_name}/{self.network_name}_{actives_str}_coefficients_sdp_stats.json"
            #         ),
            #         "w",
            #     ),
            # )



            self.handler.Objective.add_to_task()
            if verbose : 
                logger_mosek.debug("Objective added to the task.")
            self.handler.Constraints.add_to_task()
            if verbose :
                logger_mosek.debug("Constraints added to the task.")
            if self.write_model_ptf : 
                self.handler.write_model(
                    cuts,
                    RLT_prop=self.RLT_prop,
                    data_index=self.data_index,
                    ytarget=self.ytarget,
                )
            self.handler.define_objective_sense()
            if verbose :
                logger_mosek.debug("Objective sense defined.")

            end_pretreatment_time = time.time()
            self.handler.time_pretreatment = (
                end_pretreatment_time - start_pretreatment_time
            )
            if verbose:
                print(
                    "STUDY : Pretreatment computing time: ",
                    self.handler.time_pretreatment,
                )
            start_time = time.time()
            self.handler.optimize()
            end_time = time.time()
            self.handler.time_solving = end_time - start_time
            # self.handler.write_model(cuts)
            logger_mosek.info(
                "Time taken to solve: %s seconds", self.handler.time_solving
            )
            # print("Tracker : ", self.handler.tracker.get_arrays())
            if verbose :
                print("CALLBACK : Getting results ...")
            time_results_start = time.time()

            results = self.get_results(cuts, verbose)
            time_results_end = time.time()
            logger_mosek.debug("Results obtained.")
            if verbose :
                print(
                    "Time taken to get results: %s seconds",
                    time_results_end - time_results_start,
                )
                print("results obtained: ", results)
                print("x : ", self.x)
                print(
                    "self.use_inactive_neurons: ",
                    self.use_inactive_neurons,
                    "self.use_active_neurons: ",
                    self.use_active_neurons,
                )
                print(
                    "Inactive neurons : ",
                    self.stable_inactives_neurons,
                    "  Active neurons : ",
                    self.stable_actives_neurons,
                )
                print("is robust in run_optimization: ", self.handler.is_robust)
            
        except Exception as e:
            if verbose :
                print("ERROR : An error occurred during optimization:", str(e))
            logger_mosek.error("An error occurred during optimization: %s", str(e))
            self.handler.is_robust = False
            try:
                self.get_results(cuts, verbose)
            except Exception as e2:
                logger_mosek.error("Could not retrieve results after exception: %s", str(e2))
        finally:
            self.handler.cleanup_mosek()

    def _run_optimization_isolated(self, cuts: Dict, verbose: bool = False):
        """
        Run run_optimization() in a forked child process.

        MOSEK C-level crashes (SIGSEGV, SIGBUS, …) kill only the child; the
        parent detects the abnormal exit, writes a 'crashed' row to results.csv,
        and continues with the next target / cut combination.

        is_robust is communicated back to the parent via a shared-memory byte
        (multiprocessing.Value) created before the fork.
        """
        import multiprocessing as mp

        # CUDA cannot be re-initialized in a forked subprocess — move network to
        # CPU before fork so the child never touches CUDA (MOSEK is CPU-only).
        network_device = next(self.network.parameters()).device
        if network_device.type == "cuda":
            self.network.cpu()

        self.handler.is_robust = False  # safe default in parent
        is_robust_shared = mp.Value("b", 0)

        # Flush before fork so the child starts with an empty buffer and cannot
        # accidentally re-emit buffered parent output (triangular-pattern bug).
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            # ---- CHILD PROCESS ----
            try:
                self.run_optimization(cuts, verbose)
                is_robust_shared.value = int(self.handler.is_robust)
            except Exception as e:
                logger_mosek.error("Child process exception: %s", e)
                is_robust_shared.value = 0
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                os._exit(0)

        # ---- PARENT PROCESS ----
        # Restore network to its original device before any GPU-dependent calls.
        if network_device.type == "cuda":
            self.network.to(network_device)

        _, wstatus = os.waitpid(pid, 0)

        if os.WIFEXITED(wstatus) and os.WEXITSTATUS(wstatus) == 0:
            self.handler.is_robust = bool(is_robust_shared.value)
        else:
            sig = os.WTERMSIG(wstatus) if os.WIFSIGNALED(wstatus) else None
            code = os.WEXITSTATUS(wstatus) if os.WIFEXITED(wstatus) else None
            print(
                f"CRASH: MOSEK child died — signal={sig}, code={code}, "
                f"data_index={self.data_index}, ytarget={getattr(self, 'ytarget', None)}"
            )
            logger_mosek.error(
                "MOSEK child crashed: signal=%s, code=%s, data_index=%s, ytarget=%s",
                sig,
                code,
                self.data_index,
                getattr(self, "ytarget", None),
            )
            self._write_crash_row(cuts)
            self.handler.is_robust = False

    def _write_crash_row(self, cuts: Dict):
        """Write a 'crashed' row to results.csv when the MOSEK child process crashes."""
        from .run_benchmark import all_possible_cuts

        dic = {
            "network": self.network_name,
            "model": self.name,
            "dataset": self.dataset_name,
            "data_index": self.data_index,
            "label": self.ytrue,
            "label_predicted": self.label_predicted,
            "target": getattr(self, "ytarget", None),
            "epsilon": self.epsilon,
            "status": "crashed",
            "optimal_value": None,
            "CHORDAL_DECOMPOSITION": str(self.CHORDAL_DECOMPOSITION),
            "LAST_LAYER": self.LAST_LAYER,
            "USE_STABLE_ACTIVES": self.use_active_neurons,
            "USE_STABLE_INACTIVES": self.use_inactive_neurons,
            "Nb_stable_inactives": len(self.stable_inactives_neurons),
            "Nb_stable_actives": len(self.stable_actives_neurons),
        }
        dic.update({cut: (cut in cuts) for cut in all_possible_cuts})
        if "RLT" in cuts:
            dic["RLT_prop"] = getattr(self, "RLT_prop", None)
        path = get_project_path(f"{self.folder_name}/results.csv")
        row_df = pd.DataFrame(dic, index=[0])
        _append_csv(path, row_df)
        logger_mosek.info(
            "Crash row written for data_index=%s, ytarget=%s",
            self.data_index,
            getattr(self, "ytarget", None),
        )

    def solve(self, verbose: bool = False, only_bounds: bool = False, skip_pairs: set = None):
        """
        Solve the optimization problem using MOSEK.
        """
        print("VERBOSE IN SOLVE : ", verbose)
        if self.is_trivially_solved or only_bounds:
            if verbose :
                logger_mosek.debug("Trivially solved problem, no need to run optimization.")
            self.get_results_trivially_solved()
            return True
        for cuts in self.cuts_to_test:
            if verbose :
                print("Testing cuts: ", cuts)

            if "Lan" in self.__class__.__name__:
                if verbose :
                    print("CALLBACK ytargets : ", self.ytargets)
                for ytarget in self.ytargets:
                    if skip_pairs and (self.data_index, ytarget) in skip_pairs:
                        print(f"Skipping target {ytarget} for data_index {self.data_index} (already processed).")
                        continue

                    for RLT_prop in self.RLT_props:

                        if verbose :
                            print(f"Testing RLT_prop for ytarget {ytarget} ! ", RLT_prop)
                        self.RLT_prop = RLT_prop
                        self.ytarget = ytarget
                        self._run_optimization_isolated(cuts, verbose)
                        if self.handler.is_robust:
                            if verbose :
                                print("Robust solution found for ytarget:", ytarget)
                            break
                        else:
                            print("No robust solution found for ytarget:", ytarget)

            else:
                for RLT_prop in self.RLT_props:
                    if verbose :
                        print(f"Testing RLT_prop ! ", RLT_prop)
                    self.RLT_prop = RLT_prop
                    self._run_optimization_isolated(cuts, verbose)
                    if self.handler.is_robust:
                        if verbose :
                            print("Robust solution found for RLT_prop:", RLT_prop)
                        break
                    else:
                        if verbose :
                            print("No robust solution found for RLT_prop:", RLT_prop)

    def __str__(self):
        """
        String representation of the solver.
        """
        line = f"SDPSolver(K={self.network.K}, n={self.network.n} \n"
        line += f"  cuts={self.cuts} \n"
        line += f"  all_combinations_cuts={self.all_combinations_cuts} \n"
        line += self.handler.print_index_variables_matrices()
        line += "\n \n                   Weights : \n"
        for k in range(1, self.K + 1):
            line += f"  Layer {k} : \n"
            for j in range(self.n[k]):
                line += f"      Neuron {j} : \n"
                line += f"              W : {self.W[k-1][j]} \n"
                line += f"              b : {self.b[k-1][j]} \n\n"

        return line

    def add_constraints(self):
        """
        Add constraints to the task.
        """
        raise NotImplementedError(
            "The method add_constraints is not implemented in the base class."
        )

    def add_objective(self):
        """
        Add the objective function to the task.
        """
        raise NotImplementedError(
            "The method add_objective is not implemented in the base class."
        )
