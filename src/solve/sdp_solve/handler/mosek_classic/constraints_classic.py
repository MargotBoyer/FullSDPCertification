from typing import List
import mosek
from ..indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from ..indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)

from ..constraints import CommonConstraints
from ..variable_elements import add_dict_linear_to_elements, add_dict_quad_to_elements
import os
import sys
import logging
import numpy as np
import numba


logger_mosek = logging.getLogger("Mosek_logger")

dict_type_bounds = {
    mosek.boundkey.up: "up",
    mosek.boundkey.lo: "lo",
    mosek.boundkey.fx: "fx",
}


class ConstraintsClassic(CommonConstraints):
    """
    Class to handle the current constraint.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        task: mosek.Task = None,
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

        self.task = task

        # A SUPPRIMER -- TEST
        # self.new_constraint(name="test")
        # self.add_linear_variable(
        #     var="z", layer=2, neuron=40, front_of_matrix=True, value=1
        # )
        # print(
        #     "Self stable actives neurons:",
        #     self.stable_actives_neurons + self.stable_inactives_neurons,
        # )
        # for neuron in range(self.n[self.K - 1]):
        #     if (self.K - 1, neuron) in self.stable_inactives_neurons or (
        #         self.K - 1,
        #         neuron,
        #     ) in self.stable_actives_neurons:
        #         print("Inactive  or active neuron:", neuron)
        #         continue

        #     self.add_quad_variable(
        #         var1="z",
        #         layer1=self.K - 1,
        #         neuron1=neuron,
        #         front_of_matrix1=False,
        #         var2="beta",
        #         class_label=2,
        #         value=1,
        #     )
        #     break
        # for neuron in range(self.n[self.K - 1]):
        #     if (self.K - 1, neuron) in self.stable_inactives_neurons or (
        #         self.K - 1,
        #         neuron,
        #     ) in self.stable_actives_neurons:
        #         print("Inactive or active  neuron:", neuron)
        #         continue

        #     self.add_quad_variable(
        #         var1="z",
        #         layer1=self.K - 1,
        #         neuron1=neuron,
        #         front_of_matrix1=False,
        #         var2="z",
        #         layer2=self.K - 1,
        #         neuron2=neuron,
        #         front_of_matrix2=False,
        #         value=1,
        #     )
        #     break


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
                dividing_non_diag=True,
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
                dividing_non_diag=True,
            )
            self.list_cstr[self.current_num_constraint]["is_quadratic"] = True

    def add_task(self, task: mosek.Task):
        """
        Add the task to the constraint.

        Parameters
        ----------
        task: mosek.Task
            The MOSEK task.
        """
        self.task = task

    def add_to_task(self):
        """
        Add the constraint to the task.

        Chargée en 2 appels MOSEK batch (putbarablocktriplet + putconboundlist) au lieu
        d'un appel par contrainte -- les deux API MOSEK acceptent nativement des tableaux
        couvrant plusieurs (ou toutes les) contraintes à la fois (subi n'est pas limité à
        une seule contrainte par appel, cf. doc mosek.Task.putbarablocktriplet), donc
        appeler 44507 fois avec une seule contrainte à chaque fois n'apportait rien
        d'utile -- juste de l'overhead d'appel (mesuré ~0.67s de temps propre sous
        cProfile sur mnist-9x100, cf. investigation temps de processing hors résolution SDP).
        """
        n = len(self.list_cstr)
        print(f"CALLBACK : Number of constraints : {n}")
        logger_mosek.info(f"Adding {n} constraints to the task...")

        for ind_cstr, c in enumerate(self.list_cstr):
            assert (
                len(c["num_matrix"]) == len(c["i"]) == len(c["j"]) == len(c["value"])
            ), (
                f"The length of num_matrix, i, j, and value must be the same "
                f"(constraint {ind_cstr}: {c['name']})."
            )
            if self.verbose:
                print(
                    f"Adding to task constraint {c['name']} with num matrix: "
                    f"{c['num_matrix'].size} , i= {c['i'].size}, j = {c['j'].size}, "
                    f"value = {c['value'].size}"
                )

        if n > 0:
            subi_all = np.concatenate([
                np.full(len(c["num_matrix"]), ind_cstr, dtype=np.int32)
                for ind_cstr, c in enumerate(self.list_cstr)
            ])
            subj_all = np.concatenate([c["num_matrix"] for c in self.list_cstr])
            subk_all = np.concatenate([c["i"] for c in self.list_cstr])
            subl_all = np.concatenate([c["j"] for c in self.list_cstr])
            val_all = np.concatenate([c["value"] for c in self.list_cstr])
            self.task.putbarablocktriplet(subi_all, subj_all, subk_all, subl_all, val_all)

        self.task.putconboundlist(
            np.arange(n, dtype=np.int32),
            [c["bound_type"] for c in self.list_cstr],
            [c["lb"] for c in self.list_cstr],
            [c["ub"] for c in self.list_cstr],
        )

        logger_mosek.debug("All constraints added to the task.")
