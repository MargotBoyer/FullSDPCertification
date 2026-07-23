import logging
import mosek
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List

from fastsdp_tools import get_project_path, create_folder, add_row_from_dict
from .run_benchmark import (
    compute_cuts_str,
    all_possible_cuts,
    print_solution_to_file_for_cb_solver,
    print_dual_variable_to_file_for_cb_solver,
)

logger_mosek = logging.getLogger("Mosek_logger")


def _append_row_to_results_csv(self, dic_benchmark: dict):
    """Write one result row to results.csv immediately (without waiting for the full sample loop)."""
    path = get_project_path(f"{self.folder_name}/results.csv")
    row_df = pd.DataFrame(dic_benchmark, index=[0])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing.get("status", pd.Series(dtype=str)) != "pre-solve"]
        merged = pd.concat([existing, row_df], ignore_index=True)
    else:
        merged = row_df
    merged.to_csv(path, index=False)


def get_results_width_model(self, cuts: List, verbose: bool = False):  
    logger_mosek.debug("Recuperation of optimization results for width model...")
    nb_constraints = len(self.handler.Constraints.list_cstr)
    nb_variables = self.handler.print_num_variables()
    dic_benchmark = {
        "network": self.network_name,
        "model": self.name,
        "dataset": self.dataset_name,
        "data_index": self.data_index,
        "label": self.ytrue,
        "label_predicted": self.label_predicted,
        "target": self.ytarget if "Lan" in self.__class__.__name__ else None,
        "epsilon": self.epsilon,
        "CHORDAL_DECOMPOSITION": str(self.CHORDAL_DECOMPOSITION),
        "LAST_LAYER": self.LAST_LAYER,
        "USE_STABLE_ACTIVES": self.use_active_neurons,
        "USE_STABLE_INACTIVES": self.use_inactive_neurons,
        "Nb_stable_inactives": len(self.stable_inactives_neurons),
        "Nb_stable_actives": len(self.stable_actives_neurons),
        "Nb_constraints": nb_constraints,
        "Nb_variables": nb_variables,
    }
    dic_benchmark.update({cut: (cut in cuts) for cut in all_possible_cuts})
    if "RLT" in cuts:
        dic_benchmark.update({"RLT_prop": self.RLT_prop})
    if self.benchmark_dataframe is None:
        self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
    else:
        self.benchmark_dataframe = add_row_from_dict(
            self.benchmark_dataframe, dic_benchmark
        )
    logger_mosek.debug("at the end of get_results_width_model: benchmark_dataframe   : ", self.benchmark_dataframe)


def get_results(self, cuts: List, verbose: bool = False):
    """
    Recuperation of optimization results
    """
    logger_mosek.info("Recuperation of optimization results...")
    logger_mosek.info("Verbose in get_results : %s", verbose)
    if self.only_width_model:
        logger_mosek.debug("Only width model, getting width model results...")
        self.get_results_width_model(cuts, verbose)
        return
    status = self.handler.get_solution_status()

    print("Status of the solution: ", status)
    num_iterations = self.handler.get_num_iterations()
    logger_mosek.info("Number of iterations: %s", num_iterations)

    dic_benchmark = {
        "network": self.network_name,
        "model": self.name,
        "dataset": self.dataset_name,
        "data_index": self.data_index,
        "label": self.ytrue,
        "label_predicted": self.label_predicted,
        "target": self.ytarget if "Lan" in self.__class__.__name__ else None,
        "epsilon": self.epsilon,
        "status": status,
        "iterations": num_iterations,
        "time": self.handler.time_solving,
        "pretreatment_time": self.handler.time_pretreatment,
        "bound_time": self.compute_bounds_time,
        "CHORDAL_DECOMPOSITION": str(self.CHORDAL_DECOMPOSITION),
        "LAST_LAYER": self.LAST_LAYER,
        "USE_STABLE_ACTIVES": self.use_active_neurons,
        "USE_STABLE_INACTIVES": self.use_inactive_neurons,
        "Nb_stable_inactives": len(self.stable_inactives_neurons),
        "Nb_stable_actives": len(self.stable_actives_neurons),
        "Nb_constraints" : len(self.handler.Constraints.list_cstr)
    }
    dic_benchmark.update({cut: (cut in cuts) for cut in all_possible_cuts})
    if "RLT" in cuts:
        dic_benchmark.update({"RLT_prop": self.RLT_prop})

    if self.handler.is_status_optimal():
        print("CALLBACK : optimal status")
        dic_info_optimal_values = self.handler.add_all_infos_optimal_values_to_dic(
            cuts
        )
        dic_benchmark.update(dic_info_optimal_values)
        print("CALLBACK : dic_info_optimal_values: ", dic_info_optimal_values)

    elif self.handler.is_status_infeasible():
        print ("CALLBACK : infeasible status")
        if verbose:
            logger_mosek.debug("Primal or dual infeasibility certificate found.\n")
        print("CALLBACK : Déclenchement du diagnostic (contraintes contradictoires)")
        try:
            self.handler.diagnose_infeasibility()
        except Exception as e:
            print(f"CALLBACK : Diagnostic échoué : {e}")
        self.handler.get_dual_variables()
        cuts_str = compute_cuts_str(cuts)
        dual_path = get_project_path(
            f"{self.folder_name}/dual_infeasible_ind={self.data_index}_{cuts_str}.txt"
        )
        os.makedirs(os.path.dirname(dual_path), exist_ok=True)
        with open(dual_path, "w") as file_cb:
            file_cb.write("Dual Solutions \n")
            print_dual_variable_to_file_for_cb_solver(
                list_cstr=self.handler.Constraints.list_cstr, file_cb=file_cb
            )
        print(f"CALLBACK : Variables duales sauvegardées dans {dual_path}")
    elif self.handler.is_status_unknown():
        if self.handler.is_time_limit():
            print("CALLBACK : time limit reached")
            logger_mosek.warning("MOSEK terminated: time limit reached")
            dic_benchmark["status"] = "time_limit"
        else:
            print("CALLBACK : unknown status")
            logger_mosek.debug("Unknown solution status")
        try:
            dic_info_optimal_values = self.handler.add_all_infos_optimal_values_to_dic(
                cuts,
            )
            print("CALLBACK : dic_info_optimal_values: ", dic_info_optimal_values)
            dic_benchmark.update(dic_info_optimal_values)
        except Exception as e:
            print("ERROR in get_results : ", e)
            logger_mosek.critical("ERROR IN GETTING SOLUTIONS: %s", e)
            pass
    else:
        print ("CALLBACK : other status: ")
        if verbose:
            logger_mosek.debug("Other solution status")

    print("dic benchmark keys : ", dic_benchmark)
    if self.benchmark_dataframe is None:
        logger_mosek.debug("self.benchmark is None ")
        self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
    else:
        logger_mosek.debug("self.benchmark is not None")
        self.benchmark_dataframe = add_row_from_dict(
            self.benchmark_dataframe, dic_benchmark
        )
    print("benchmark_dataframe   : ", self.benchmark_dataframe)
    _append_row_to_results_csv(self, dic_benchmark)
