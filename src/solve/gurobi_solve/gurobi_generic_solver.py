from typing import List
import torch
from networks import ReLUNN
import mosek
import yaml
import time
import os
import logging
import sys
import json
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import sys
from functools import partial


from ..generic_solver import Solver
from fastsdp_tools import get_project_path, add_row_from_dict
from .callback import SolutionCallback, CallbackData, mycallback
from .analyser import analyze_model


logger_gurobi = logging.getLogger("Gurobi_logger")


class GurobiSolver(Solver):
    """
    A solver that uses MOSEK to solve the optimization problem.
    """

    def __init__(
        self,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logger_gurobi.debug("Initializing GurobiSolver...")

        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS

        self.K = self.network.K
        self.n = self.network.n
        self.W = self.network.W
        self.b = self.network.b

        self.constant = 0
        logger_gurobi.debug("GurobiSolver initialized.")


        if self.__class__.__name__ != "LPBoundLayer":
            self.max_layer_z = self.K + 1 if self.LAST_LAYER else self.K
        else :
            self.max_layer_z = None


        for layer in range(len(self.n)):
            
            # self.L[layer + 1] = self.L[layer + 1] - 10
            # self.U[layer + 1] = self.U[layer + 1] + 10
            # max_abs_L = np.max(np.abs(self.L[layer + 1]))
            # max_abs_U = np.max(np.abs(self.U[layer + 1]))
            print(
                f"STUDY Layer {layer} L: {self.L[layer]}, U: {self.U[layer]}"
            )

        

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        params = Solver.parse_yaml(yaml_file)
        return cls(**params, **kwargs)

    def run_optimization(self, verbose: bool = False):
        assert self.max_layer_z is not None, "max_layer_z must be specified to run optimization"
        logger_gurobi.debug("Running optimization...")
        self.initiate_solver()
        logger_gurobi.debug("Solver initiated.")
        self.add_variables()
        logger_gurobi.debug("Variables added.")
        self.add_objective()
        logger_gurobi.debug("Objective added.")
        self.add_constraints()
        logger_gurobi.debug("Constraints added.")
        # Ensure the model is updated so NumConstrs and NumVars reflect added elements
        self.m.update()
        print(f"Nombre de contraints avant optimisation : {self.m.NumConstrs}")
        print(f"Nombre de variables avant optimisation : {self.m.NumVars}")
        self.write_model()
        self.print_solver_info(verbose)
        logger_gurobi.debug("Starting optimization...")
        callback_data = CallbackData(self.m.getVars())
        callback_func = partial(
            mycallback, cbdata=callback_data, logfile=f"gurobi_{self.name}.log"
        )
        logger_gurobi.debug("Callback function prepared.")
      
        self.m.optimize()  # self.callback

        results = self.get_results(verbose)
        logger_gurobi.info("STUDY :Time taken to solve: %s seconds", self.time_solving)
        
        return results

    def retrieve_z(self):
        z_values = {}

        for layer in range(self.max_layer_z):
            for neuron in range(self.n[layer]):
                if (layer, neuron) in self.stable_inactives_neurons:
                    continue
                z_values[(layer, neuron)] = self.z[(layer, neuron)].X
        return z_values

    def retrieve_beta(self):
        beta_values = {}
        for class_label in self.ytargets:
            if class_label != self.ytrue:
                beta_values[class_label] = self.beta[class_label].X
        return beta_values

    def get_adversarial_attack(self):
        z_values = self.retrieve_z()
        Sol = []
        for j in range(self.n[0]):
            Sol.append(z_values[(0, j)])
        return Sol

    def get_results(self, verbose: bool = False):
        """
        Get the results of the optimization.
        """
        self.time_solving = self.m.runtime
        self.nb_nodes = self.m.NodeCount
        logger_gurobi.info("Number of nodes explored: %s", self.nb_nodes)
        logger_gurobi.info("Time taken to solve: %s seconds", self.time_solving)
        logger_gurobi.info("Number of variables: %s", self.m.NumVars)
        logger_gurobi.info("Number of constraints: %s", self.m.NumConstrs)
        logger_gurobi.info("Number of non-zero coefficients: %s", self.m.NumNZs)
        logger_gurobi.debug("Status: %s", self.m.Status)
        self.opt = None
        if self.m.Status == GRB.OPTIMAL:
            opt = self.m.ObjVal
            logger_gurobi.debug("Optimal objective value: ", opt)
            print("Constant to add : ", self.constant)
            self.opt = opt + self.constant
            logger_gurobi.debug(
                f"Optimal objective value (with added constant) for model {self.name}: %s",
                self.opt,
            )
            logger_gurobi.debug("Optimal objective value (with added constant): ", self.opt)
            z_values = self.retrieve_z()
            dict_json = {layer : {neuron : float(z_values[(layer, neuron)]) for neuron in range(self.n[layer]) if (layer, neuron) not in self.stable_inactives_neurons} for layer in range(self.max_layer_z)}
            with open(f"{self.folder_name}/solutions.json", "w") as f :
                json.dump(dict_json, f)
            logger_gurobi.debug(f"z values: {z_values}")
            if self.BETAS:
                beta_values = self.retrieve_beta()
                logger_gurobi.debug(f"beta values: {beta_values} ")
            else:
                beta_values = None
        elif self.m.Status == GRB.INFEASIBLE:
            logger_gurobi.error("STUDY : Model is infeasible")
            # analyze_model(self.m)
        else:
            opt = self.m.ObjVal
            logger_gurobi.debug("UNKNOWN STATUS : Optimal objective value: ", opt)
        print("compute bound time in get result de : ", self.__class__.__name__)
        print("compute bounds : ", self.compute_bounds_time)
        dic_benchmark = {
            "network": self.network_name,
            "model": self.name,
            "dataset": self.dataset_name,
            "data_index": self.data_index,
            "label": self.ytrue,
            "label_predicted": self.label_predicted,
            "target": self.ytarget if "Targeted" in self.__class__.__name__ else None,
            "epsilon": self.epsilon,
            "status": self.m.Status,
            "optimal_value": getattr(self, "opt", None),
            "nb_nodes": self.nb_nodes,
            "time": self.time_solving,
            "bound_time": self.compute_bounds_time,
            "LAST_LAYER": self.LAST_LAYER,
            "USE_STABLE_ACTIVES": self.use_active_neurons,
            "USE_STABLE_INACTIVES": self.use_inactive_neurons,
            "Nb_stable_inactives": len(self.stable_inactives_neurons),
            "Nb_stable_actives": len(self.stable_actives_neurons),
        }
        if self.__class__.__name__ == "LPBoundLayer":
            dic_benchmark["layer_obj"] = self.layer_obj
            dic_benchmark["neuron_obj"] = self.neuron_obj
            dic_benchmark["bound_obj"] = self.bound_obj
        if self.benchmark_dataframe is None:
            self.benchmark_dataframe = pd.DataFrame(dic_benchmark, index=[0])
        else:
            self.benchmark_dataframe = add_row_from_dict(
                self.benchmark_dataframe, dic_benchmark
            )

    def write_model(self):
        """
        Write the model to a file.
        """
        if self.folder_name is not None:
            logger_gurobi.debug("Writing model to file...")
            self.m.write(
                get_project_path(f"{self.folder_name}/{self.name}/{self.name}.lp")
            )
            self.m.printStats()
            logger_gurobi.info(
                f"Model written to {self.folder_name}/{self.name}/{self.name}.lp"
            )
            self.m.setParam(
                "LogFile",
                get_project_path(f"{self.folder_name}/{self.name}/{self.name}.log"),
            )
        else :
            logger_gurobi.debug("Folder name not specified, model not written to file.")

    def initiate_solver(self, **parameters):
        """
        Initialize the solver with the given parameters.
        """
        self.env = gp.Env(empty=True)
        self.env.start()
        print("self name : ", self.name)
        self.m = gp.Model(self.name, env=self.env)

        self.m.Params.NonConvex = 2
        for key, value in parameters.items():
            if key in ["DualReductions"]:
                self.m.setParam(key, value)
            else:
                self.m.setParam(key, value)
        self.m.setParam("Threads", 1)

    def print_solver_info(self, verbose: bool = False):
        print("printing solver info : verbose = ", verbose)
        if verbose:
            self.env.setParam("OutputFlag", 1)
            self.m.setParam("LogToConsole", 1)
        else:
            self.env.setParam("OutputFlag", 0)

        if self.folder_name is not None:
            self.m.setParam(
                "LogFile",
                get_project_path(f"{self.folder_name}/{self.name}/Gurobi_logger.log"),
            )
        else:
            self.m.setParam("LogFile", get_project_path("results/Gurobi_logger.log"))

    def solve(self, verbose: bool = False, **kwargs):
        """
        Solve the optimization problem using MOSEK.
        """
        if self.is_trivially_solved :
            if verbose : 
                logger_gurobi.debug("Trivially solved problem, no need to run optimization.")
            self.get_results_trivially_solved()
            return True
        else :
            if self.__class__.__name__=="LanQuad" or self.__class__.__name__=="ClassicLP" :
                for ytarget in self.ytargets:
                    self.ytarget = ytarget
                    self.run_optimization(verbose=verbose)
            else : 
                self.run_optimization(verbose=verbose)

    def get_optimal_value(self):
        """
        Get the optimal value of the objective function.
        """
        if self.m.Status == GRB.OPTIMAL:
            return self.opt
        else:
            raise ValueError("Model is not optimal or has not been solved yet.")

    def add_variables(self):
        raise NotImplementedError(
            "add_variables method not implemented in GurobiSolver"
        )

    def add_objective(self):
        raise NotImplementedError(
            "add_objective method not implemented in GurobiSolver"
        )

    def add_constraints(self):
        raise NotImplementedError(
            "add_constraints method not implemented in GurobiSolver"
        )
