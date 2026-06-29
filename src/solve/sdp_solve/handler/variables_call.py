import numpy as np
import random

from .indexes_matrices import (
    Indexes_Matrixes_for_Mosek_Solver,
)
from .indexes_variables import (
    Indexes_Variables_for_Mosek_Solver,
)
from .variable_elements import (
    Equivalent_Neurons_Index,
    Equivalent_Betas_Index,
    _get_linear_indices_from_key,
    _get_key_from_layer_neuron_,
    _get_layer_neuron_from_key_,
    _get_key_linear_,
)
import logging
from typing import List
from collections import Counter
from numba import njit
import numba
from numba.typed import Dict

logger_mosek = logging.getLogger("Mosek_logger")

from fastsdp_tools import summing_values_two_dicts, change_to_zero_negative_values


def get_only_one_variable_kwargs(index: int = 1, **kwargs):
    assert index in [1, 2], "Index must be either 1 or 2."
    return {
        k.strip(f"{index}"): v for k, v in kwargs.items() if str(k).endswith(f"{index}")
    }


class LayersValues:
    """
    A class to handle the values of neurons (in particular stable active neurons) accros layers
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        W: list,
        b: list,
        stable_inactives_neurons: List[tuple] = [],
        stable_actives_neurons: List[tuple] = [],
        L: List[List[float]] = None,
        U: List[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize the LayersValues class.
        """
        self.n = n
        self.W = W
        self.b = b
        self.K = K
        self.stable_inactives_neurons = stable_inactives_neurons
        self.stable_actives_neurons = stable_actives_neurons
        self.LAST_LAYER = kwargs.get("LAST_LAYER", False)
        self.keep_actives_penultimate = kwargs.get("keep_penultimate_actives", None)
        assert (
            self.keep_actives_penultimate is not None
        ), "keep_penultimate_actives must be specified."

        self.equivalent_values_layers = {
            (layer, neuron): {"neurons_weight": {}, "constant": 0}
            for layer in range(K + 1)
            for neuron in range(n[layer])
        }
        for k in range(K + 1):
            for j in range(n[k]):
                self.add_equivalent_values(k, j)

    def add_equivalent_values(self, layer: int, neuron: int):
        # print(f"LAYER = {layer}, neuron = {neuron}")
        # print(f"Active ? ", (layer, neuron) in self.stable_actives_neurons)
        # print(f"Penultimate or before ? ", (not self.keep_actives_penultimate or layer < self.K - 1))
        if ( ((layer, neuron) in self.stable_actives_neurons
            and (not self.keep_actives_penultimate or layer < self.K - 1)) or (layer == self.K and not self.LAST_LAYER)):
            # print("Decomposing...")
            self.equivalent_values_layers[(layer, neuron)]["constant"] += self.b[
                layer - 1
            ][neuron]
            # print("Getting constant, ok")
            for i in range(self.n[layer - 1]):
                # print(f"decomposing into neuron {i} of layer {layer-1}")
                # print(f"    Decomposing into i = {i}, layer - 1 = {layer - 1}")
                # print(f"    Current dictionnary ", self.equivalent_values_layers[(layer, neuron)])
                # print("     Adding : ", {
                #             (layer2, neuron2): (value * self.W[layer - 1][neuron][i])
                #             for (
                #                 layer2,
                #                 neuron2,
                #             ), value in self.equivalent_values_layers[(layer - 1, i)][
                #                 "neurons_weight"
                #             ].items()
                #         })
                self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = (
                    summing_values_two_dicts(
                        self.equivalent_values_layers[(layer, neuron)][
                            "neurons_weight"
                        ],
                        {
                            (layer2, neuron2): (value * self.W[layer - 1][neuron][i])
                            for (
                                layer2,
                                neuron2,
                            ), value in self.equivalent_values_layers[(layer - 1, i)][
                                "neurons_weight"
                            ].items()
                        },
                    )
                )
                self.equivalent_values_layers[(layer, neuron)]["constant"] += (
                    self.equivalent_values_layers[(layer - 1, i)]["constant"]
                    * self.W[layer - 1][neuron][i]
                )
                # print("     Dictionnary after add : ", self.equivalent_values_layers[(layer, neuron)])
                # print()

           

            coordinates = [
                (layer, neuron)
                for (layer, neuron), value in self.equivalent_values_layers[
                    (layer, neuron)
                ]["neurons_weight"].items()
            ]
            counts = Counter(coordinates)

        elif (layer, neuron) in self.stable_inactives_neurons:
            #print("Stable inactive neurons, pass ...")
            pass
        elif layer == self.K and self.LAST_LAYER:
            #print("Last layer...")
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }

        else:
            #print("Others ...")
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"] = {
                (layer, neuron): 1
            }
        # print(f"layer = {layer}, neuron = {neuron}, weight = ",  self.equivalent_values_layers[(layer, neuron)]["neurons_weight"], "constant = ", self.equivalent_values_layers[(layer, neuron)]["constant"] )
        # print()
        # print()
        # print()

    def get_equivalent_values(self, layer: int, neuron: int):
        """
        Get the equivalent values for a given layer and neuron.
        """
        if layer < 0 or layer > self.K:
            raise ValueError(f"Layer {layer} is out of bounds (0 to {self.K}).")
        if neuron < 0 or neuron >= self.n[layer]:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is out of bounds (0 to {self.n[layer] - 1})."
            )
        return (
            self.equivalent_values_layers[(layer, neuron)]["neurons_weight"],
            self.equivalent_values_layers[(layer, neuron)]["constant"],
        )

    def is_unstable(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) not in (
            self.stable_actives_neurons + self.stable_inactives_neurons
        ) and (layer is not None and neuron is not None)

    def is_stable_active(self, layer: int, neuron: int) -> bool:
        """
        Check if the neuron is a stable active neuron.
        """
        return (layer, neuron) in self.stable_actives_neurons and (
            layer is not None and neuron is not None
        )

    def computing_bounds_based_on_stable_neurons(
        self,
        L: List[List[float]] = None,
        U: List[List[float]] = None,
    ):
        """
        Compute the bounds based on stable neurons.
        """
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                if (k, j) in self.stable_actives_neurons:
                    upper_bounds = (
                        sum(
                            value * U[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * L[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if upper_bounds < U[k][j]:
                        print(
                            "The computed upper bound is BETTER on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial upper bound.",
                        )
                        U[k][j] = upper_bounds
                    else:
                        print(
                            "The computed upper bound is not better on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial upper bound.",
                        )
                    # print(f"Upper bound for layer {k}, neuron {j}: {self.upper_bounds[(k, j)]} and U = {U[k][j]}")
                    lower_bounds = (
                        sum(
                            value * L[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value > 0
                        )
                        + sum(
                            value * U[layer][neuron]
                            for (layer, neuron), value in self.equivalent_values_layers[k, j][
                                "neurons_weight"
                            ].items()
                            if value < 0
                        )
                        + self.equivalent_values_layers[k, j]["constant"]
                    )
                    if lower_bounds > L[k][j]:
                        print(
                            "The computed lower bound is BETTER on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial lower bound.",
                        )
                        L[k][j] = lower_bounds
                    else:
                        print(
                            "The computed lower bound is not better on layer",
                            k,
                            "neuron",
                            j,
                            "than the initial lower bound.",
                        )
                    # print(f"Lower bound for layer {k}, neuron {j}: {self.lower_bounds[(k, j)]} and L = {L[k][j]}")
        return L, U


# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************
# ********************************************************************************************************************************


class VariablesCall:
    """
    A class to handle the variables call in the MOSEK solver.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        """
        Initialize the VariablesCall class.
        """
        self.indexes_matrices = indexes_matrices
        self.indexes_variables = indexes_variables

        self.stable_inactives_neurons = kwargs.pop("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.pop("stable_actives_neurons")
        self.ytargets = kwargs.get("ytargets")

        self.K = kwargs.pop("K")
        self.n = kwargs.pop("n")
        self.W = kwargs.pop("W")
        self.b = kwargs.pop("b")

        self.L = kwargs.get("L", None)
        self.U = kwargs.get("U", None)

        # print(
        #     "maxabs(U) : ", max([max([abs(U_i_j) for U_i_j in U_i])] for U_i in self.U)
        # )
        # print(
        #     "maxabs(L) : ", max([max([abs(L_i_j) for L_i_j in L_i])] for L_i in self.L)
        # )

        # print("Starting Layers Values initialization with K:", self.K)
        self.layers_values = LayersValues(
            K=self.K,
            n=self.n,
            W=self.W,
            b=self.b,
            stable_actives_neurons=self.stable_actives_neurons,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )

        self.L, self.U = self.layers_values.computing_bounds_based_on_stable_neurons(
            L=self.L, U=self.U
        )
        self.U_above_zero = change_to_zero_negative_values(
            self.U, dim=2
        )  # ATTENTION U N'EST PAS PRECIS : POUR EVITER CAS DES NEURONES STABLES INACTIFS
        self.L_above_zero = change_to_zero_negative_values(
            self.L, dim=2
        )  # ATTENTION : CECI POSERA UN PROBLEME POUR LES CONTRAINTES TRIANGULAIRES

        self.LAST_LAYER = kwargs.get("LAST_LAYER", None)
        self.MATRIX_BY_LAYERS = kwargs.get("MATRIX_BY_LAYERS", None)
        self.BETAS = kwargs.get("BETAS", None)
        self.INPUT_IN_VARIABLES = kwargs.get("INPUT_IN_VARIABLES", True)
        self.kept_input_neurons = kwargs.get("kept_input_neurons", set(range(int(self.n[0]))))
        self.pruned_input_neurons = kwargs.get("pruned_input_neurons", set())
        assert self.LAST_LAYER is not None, "LAST_LAYER must be specified."
        assert self.BETAS is not None, "BETAS must be specified."

        self.equivalent_neurons = Equivalent_Neurons_Index(K = self.K, LAST_LAYER=self.LAST_LAYER)
        self.equivalent_indexes_betas = Equivalent_Betas_Index(ytargets=self.ytargets)

        self.create_equivalent_indexes_matrices()

        #self._print_equivalent_indexes_()
        self.study_indexes_equivalent_neurons() # Check errors

       

    def create_equivalent_indexes_matrices(self):

        for layer in range(self.K + 1):
            if layer == 0 and not self.INPUT_IN_VARIABLES:
                continue  # z_0 entirely removed from SDP
            for neuron in range(self.n[layer]):
                if layer == 0 and neuron in self.pruned_input_neurons:
                    continue  # partial input pruning: z_0[neuron] not a SDP variable
                #print(f'Creating equivalent for layer = {layer}, neuron = {neuron}')
                equivalent_values_neurons, constant = (
                    self.layers_values.get_equivalent_values(layer, neuron)
                )

                is_boundary_layer = len(self.indexes_matrices._layer_to_groups.get(layer, [])) > 1
                decomposed_in_front_and_back_matrix = ((not ((layer,neuron) in self.stable_actives_neurons)) and (layer < self.K or self.LAST_LAYER)) and self.MATRIX_BY_LAYERS and is_boundary_layer
                
                self.equivalent_neurons.create_dict(layer=layer, neuron=neuron, K = self.K, 
                                                    LAST_LAYER=self.LAST_LAYER, 
                                                    decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix)

                if decomposed_in_front_and_back_matrix:
                    #print(f"Layer {layer} : decomposed is true")
                    for (k, j), val in equivalent_values_neurons.items():
                        #print(f"k = {k}, j = {j}, val = {val}, LAST_LAYER = {self.LAST_LAYER}")
                        k_groups = self.indexes_matrices._layer_to_groups.get(k, [])
                        k_can_be_front = any(
                            pos < len(self.indexes_matrices.layer_groups[g]) - 1
                            for g, pos in k_groups
                        )
                        k_can_be_back = any(
                            pos == len(self.indexes_matrices.layer_groups[g]) - 1
                            for g, pos in k_groups
                        )

                        if k_can_be_front:
                                #print(f"        Decomposing with layer = {layer}, neuron = {neuron}, val = {1} - FRONT OF MATRIX")
                                ind_i_front = self.indexes_variables._get_variable_index(
                                    "z", layer=k, neuron=j, front_of_matrix=True
                                )
                                ind_num_matrix_front = self.indexes_matrices._get_matrix_index(
                                    "z", layer=k, neuron=j, front_of_matrix=True
                                )
                                self.equivalent_neurons.add(
                                    layer=layer,
                                    neuron=neuron,
                                    i=ind_i_front,
                                    num_matrix=ind_num_matrix_front,
                                    value=1,
                                    front_of_matrix=True,
                                )
                        if k_can_be_back:
                            #print(f"        Decomposing with layer = {layer}, neuron = {neuron}, val = {1} - BACK OF MATRIX")
                            ind_i_back = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            ind_num_matrix_back = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_back,
                                num_matrix=ind_num_matrix_back,
                                value=1,
                                front_of_matrix=False,
                            )

                else : 
                    #print(f"Layer {layer} : decomposed is false")
                    for (k, j), val in equivalent_values_neurons.items():
                        #print(f"k = {k}, j = {j}, val = {val}, LAST_LAYER = {self.LAST_LAYER}")

                        if (k < self.K - 1 and not self.LAST_LAYER) or (
                            k < self.K and self.LAST_LAYER
                        ):
                            #print(f"        Decomposing with k = {k}, j = {j}, val = {val} - FRONT OF MATRIX")
                            ind_i_front = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=True
                            )
                            ind_num_matrix_front = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=True
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_front,
                                num_matrix=ind_num_matrix_front,
                                value=val,
                            )
                        else:
                            #print(f"        Decomposing with k = {k}, j = {j}, val = {val} - BACK OF MATRIX")
                            ind_i_back = self.indexes_variables._get_variable_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            ind_num_matrix_back = self.indexes_matrices._get_matrix_index(
                                "z", layer=k, neuron=j, front_of_matrix=False
                            )
                            self.equivalent_neurons.add(
                                layer=layer,
                                neuron=neuron,
                                i=ind_i_back,
                                num_matrix=ind_num_matrix_back,
                                value=val,
                            )



                self.equivalent_neurons.add_constant(
                    layer=layer,
                    neuron=neuron,
                   
                     value=constant,
                )
            #print('Equivalent neurons index result : ', self.equivalent_neurons.equivalent_neurons)

        if self.BETAS:

            for class_label in self.ytargets:

                i = self.indexes_variables._get_variable_index(
                    "beta", class_label=class_label
                )
                num_matrix = self.indexes_matrices._get_matrix_index(
                    "beta", class_label=class_label
                )
                self.equivalent_indexes_betas.add(
                    class_label=class_label,
                    i=i,
                    num_matrix=num_matrix,
                )
            #print(self.equivalent_indexes_betas.equivalent_indexes_betas)

    def study_indexes_equivalent_neurons(self):
        for k in range(self.K + 1):
            for j in range(self.n[k]):
                key = self.equivalent_neurons.get_index(layer=k, neuron=j)
                layer, neuron = _get_layer_neuron_from_key_(key=key)
                assert (
                    layer == k and neuron == j
                ), f"ERROR in STUDY1: layer = {layer}, neuron = {neuron}, k = {k}, j = {j}"
        for num_matrix in range(self.indexes_matrices.nb_matrices):
            logger_mosek.debug(f"in study_indexes_equivalent_neurons: num_matrix = {num_matrix}")
            try:
                for i in range(self.indexes_variables.max_index):

                    key = _get_key_linear_(i, num_matrix)

                    i2, num_matrix2 = _get_linear_indices_from_key(key)

                    assert (
                        i == i2 and num_matrix == num_matrix2
                    ), f"ERROR in STUDY2: i = {i}, num_matrix = {num_matrix}, i2 = {i2}, num_matrix2 = {num_matrix2}"

            except ValueError as e:
                print("Error : ", e)
                pass

    def _print_equivalent_indexes_(self):
        line = ""

        for layer in range(self.K + 1):
            line += f"Layer {layer}:\n"
            for neuron in range(self.n[layer]):
                
                decomposed_in_front_and_back_matrix = (not ((layer,neuron) in self.stable_actives_neurons)) and (layer < self.K or self.LAST_LAYER) and self.MATRIX_BY_LAYERS

                constant = self.equivalent_neurons.get_constant(
                    layer=layer, neuron=neuron
                )
                line += f"\n  Layer {layer} Neuron {neuron}: \n"

                if ((layer <= self.K - 1 and not self.LAST_LAYER) or (
                      self.LAST_LAYER
                    )) and decomposed_in_front_and_back_matrix:
                    front = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, front_of_matrix=True
                    )
                    line += f"    FRONT_OF_MATRIX \n"
                    for key, value in front.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"  {(num_matrix,i)} : {value};   "
                if layer > 0 and decomposed_in_front_and_back_matrix: 
                    line += f"\n    BACK_OF_MATRIX \n"
                    back = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, front_of_matrix=False
                    )
                    for key, value in back.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"{(num_matrix,i)} : {value};   "
                elif not decomposed_in_front_and_back_matrix:
                    eq = self.equivalent_neurons.get_equivalent(
                        layer=layer, neuron=neuron, 
                    )
                    for key, value in eq.items():
                        i, num_matrix = _get_linear_indices_from_key(key, 13)
                        line += f"{(num_matrix,i)} : {value};   "
                line += f"\n    constant : {constant}\n"
        if self.BETAS:
            for class_label in self.ytargets:
                line += f"Class {class_label}:"
                dict_beta = self.equivalent_indexes_betas.get_equivalent(class_label)
                for key, value in dict_beta.items():
                    i, num_matrix = _get_linear_indices_from_key(key, 13)
                    line += f"  i : {i} ; "
                    line += f"  num_matrix : {num_matrix}\n"
        print(line)

        ""

    def add_constant(self, value: float):
        """
        Add a constant to the constraint.
        """
        raise NotImplementedError("This method should be implemented in the subclass COnstraint or Objective.")

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass Constraint or Objective.")

  
    # def call_variable(self, var: str, **kwargs):
    #     pass


    def verify_variable_z(self, layer : int, neuron : int, front_of_matrix : bool= None):
        assert layer is not None, "Layer must be specified for z variable."
        assert neuron is not None, "Neuron must be specified for z variable."
        if layer == 0 and (not self.INPUT_IN_VARIABLES or neuron in self.pruned_input_neurons):
            raise ValueError(
                f"Cannot reference z_(0,{neuron}) as a SDP variable: "
                "this input neuron was removed from the SDP (INPUT_IN_VARIABLES=False or partial pruning). "
                "Check that no constraint explicitly uses this layer-0 variable."
            )
        assert (layer, neuron) not in self.stable_inactives_neurons, "Stable inactive neurons should not be added as variables."
        if (layer, neuron) in self.stable_actives_neurons :
            decomposed_in_front_and_back_matrix = False
        else : 
            if self.MATRIX_BY_LAYERS :
                if front_of_matrix is None:
                    if layer < self.K-1 :
                        front_of_matrix = True
                    elif layer == self.K-1 and self.LAST_LAYER :
                        front_of_matrix = True
                    else :
                        front_of_matrix = False
                is_boundary_layer = len(self.indexes_matrices._layer_to_groups.get(layer, [])) > 1
                decomposed_in_front_and_back_matrix = (layer < self.K or self.LAST_LAYER) and is_boundary_layer
            else : 
                decomposed_in_front_and_back_matrix = False
        return decomposed_in_front_and_back_matrix, front_of_matrix


              

    def add_linear_variable(self, var: str, value: float, **kwargs):
        """
        Add a linear variables to the constraint.
        Checks if the variable present corresponds to stable active neurons and divides it in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        if var == "z":
            layer = kwargs.get("layer", None)
            neuron = kwargs.get("neuron", None)
            front_of_matrix = kwargs.get("front_of_matrix", None)
            #print("STUDY COEFF layer : ", layer, "neuron : ", neuron, "front_of_matrix : ", front_of_matrix)

            decomposed_in_front_and_back_matrix, front_of_matrix = self.verify_variable_z(layer, neuron, front_of_matrix)
            dict1 = self.equivalent_neurons.get_equivalent(
                    layer=layer, neuron=neuron, front_of_matrix=front_of_matrix, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix
                )
            assert len(dict1) > 0, "Dictionnary used in add_linear_variable is empty"

            #print(f"STUDY COEFF In add_linear_variable : Layer = {layer}, neuron = {neuron}, front_of_matrix = {front_of_matrix}, dict1 = {dict1}, value = {value}")
            self.add_var(
                dict1=dict1,
                value=value,
            )

            constant = self.equivalent_neurons.get_constant(layer, neuron)
            self.add_constant(value * constant)
        else:
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
            self.add_var(
                dict1=self.equivalent_indexes_betas.get_equivalent(
                    class_label=class_label
                ),
                value=value,
            )



    def add_quad_variable(self, var1: str, var2: str, value: float, **kwargs):
        """
        Add a product of two variables to the constraint.
        Checks if the variables present corresponds to stable active neurons and divides them in this case to the precedent layer's z variables.
        """
        if value == 0:
            return
        assert var1 in ["z", "beta"], "var1 must be either 'z' or 'beta'."
        assert var2 in ["z", "beta"], "var2 must be either 'z' or 'beta'."
        if var1 == "z" and var2 == "beta":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", None)
            decomposed_in_front_and_back_matrix, front_of_matrix1 = self.verify_variable_z(layer1, neuron1, front_of_matrix1)
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."

            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict1 = self.equivalent_neurons.get_equivalent(
                layer1, neuron1, front_of_matrix1, decomposed_in_front_and_back_matrix
            )
            
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = 0

        elif var1 == "beta" and var2 == "z":
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", None)
            decomposed_in_front_and_back_matrix, front_of_matrix2 = self.verify_variable_z(layer2, neuron2, front_of_matrix2)
            class_label = kwargs.get("class_label", None)
            assert (
                class_label is not None
            ), "Class label must be specified for beta variable."
           
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer2, neuron2, front_of_matrix2, decomposed_in_front_and_back_matrix
            )

            constant1 = 0
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)

        elif var1 == "beta" and var2 == "beta":
            class_label1 = kwargs.get("class_label1", None)
            class_label2 = kwargs.get("class_label2", None)
            assert (
                class_label1 is not None
            ), "Class label1 must be specified for beta variable."
            assert (
                class_label2 is not None
            ), "Class label2 must be specified for beta variable."
            dict1 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label1
            )
            dict2 = self.equivalent_indexes_betas.get_equivalent(
                class_label=class_label2
            )
            constant1 = 0
            constant2 = 0

        elif var1 == "z" and var2 == "z":
            layer1 = kwargs.get("layer1", None)
            neuron1 = kwargs.get("neuron1", None)
            front_of_matrix1 = kwargs.get("front_of_matrix1", None)
            layer2 = kwargs.get("layer2", None)
            neuron2 = kwargs.get("neuron2", None)
            front_of_matrix2 = kwargs.get("front_of_matrix2", None)
            
            decomposed_in_front_and_back_matrix1, front_of_matrix1 = self.verify_variable_z(layer1, neuron1, front_of_matrix1)
            decomposed_in_front_and_back_matrix2, front_of_matrix2 = self.verify_variable_z(layer2, neuron2, front_of_matrix2)
            constant1 = self.equivalent_neurons.get_constant(layer1, neuron1)
            constant2 = self.equivalent_neurons.get_constant(layer2, neuron2)
            dict1 = self.equivalent_neurons.get_equivalent(
                layer=layer1, neuron=neuron1, front_of_matrix=front_of_matrix1, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix1
            )
            dict2 = self.equivalent_neurons.get_equivalent(
                layer=layer2, neuron=neuron2, front_of_matrix=front_of_matrix2, decomposed_in_front_and_back_matrix=decomposed_in_front_and_back_matrix2
            )

        assert len(dict1) > 0, "Dictionnary used in add_quad_variable is empty"
        assert len(dict2) > 0, "Dictionnary used in add_quad_variable is empty"
        self.add_var(
            dict1=dict1,
            value=value,
            dict2=dict2,
        )
        if constant1 != 0 and constant2 != 0:
            self.add_constant(value * constant1 * constant2)
        if constant1 != 0:
            self.add_var(dict1=dict2, value=value * constant1)
        if constant2 != 0:
            self.add_var(dict1=dict1, value=value * constant2)

    def add_z_quad_bound_composed(self, layer1, neuron1, front_of_matrix1,
                                   weight, coeff1, U_next, bound_sense):
        
        v = weight * coeff1
        L1 = self.L[layer1][neuron1]
        U1 = self.U[layer1][neuron1]

        if (bound_sense == "upper" and v>=0) or (bound_sense == "lower" and v<0):
            # Adding constraint z_next * z_{layer1} <= L_{layer1} * z_next + U_{next} * z_{layer1} - L_{layer1} * U_{next}
            coeff_next = -v * L1
            cst        = v * L1 * U_next
            z1_coeff   = -v * U_next
        else: 
                # Adding constraint z_next * z_{layer1} >= U_{layer1} * z_next + U_next * z_{layer1} - U_{layer1} * U_{layer1}
                coeff_next = -v * U1
                cst        = v * U1 * U_next
                z1_coeff   = -v * U_next
        
        self.add_linear_variable(
            var="z",
            layer=layer1,
            neuron=neuron1,
            value=z1_coeff,
            front_of_matrix=front_of_matrix1,
        )
        return coeff_next, cst
    
    
    def add_z_quad_bound_one_variable(self, layer1, neuron1, weight, coeff1, bound_sense):
        v = weight * coeff1
        if (bound_sense == "upper" and v>=0) or (bound_sense == "lower" and v<0):
            # Adding constraint z_next * z_{layer1} <= U_{layer1} * z_next
            coeff_next = -v * self.U[layer1][neuron1]
            
        else:  
            if layer1 == 0: # L_layer1 is zero on all layer outputs (every variable except the one representing the input)
                # Adding constraint z_next * z_{layer1} >= L_{layer1} * z_next
                coeff_next = -v * self.L[layer1][neuron1]
            else :
                coeff_next = 0

        return coeff_next, 0

        

    def add_z_quad_active_neuron(
        self,
        layer_prev: int,
        neuron_prev: int,
        layer_next: int,
        neuron_next: int,
        front_of_matrix_prev: bool,
        front_of_matrix_next: bool,
        weight: float,
        bound_sense: str = "upper",
        bound_type: str = "one_variable",
        bound_strategy: dict = None,
    ):
        assert mccormick_type in ["one_variable", "composed", "random"]
        assert bound_sense in ["lower", "upper"]

        assert (layer_prev, neuron_prev) in self.stable_actives_neurons
        equivalent_values_neurons, constant = self.layers_values.get_equivalent_values(
            layer_prev, neuron_prev
        )

        U_next = float(self.U[layer_next][neuron_next])
        coeff_next = 0.0  # accumulated coefficient for z_{layer_next}
        cst = 0.0         # accumulated constant

        for (layer1, neuron1), coeff1 in equivalent_values_neurons.items():
            # Vérification que le produit quadratique z_next * z_{layer1} est présent dans les matrices variables
            groups_layer1 = {g for g, _ in self.indexes_matrices._layer_to_groups.get(layer1, [])}
            groups_layer_next = {g for g, _ in self.indexes_matrices._layer_to_groups.get(layer_next, [])}

            if groups_layer1 & groups_layer_next: # Produit présent, pas d'encadrement à faire
                if layer_next - layer1 > 2:
                    logger_mosek.debug(f"RELU : ", f"Product of z_{layer_next} and z_{layer1} is PRESENT in the variable matrices, adding product without bounding.")
                decomposed1, front1 = self.verify_variable_z(layer1, neuron1, None)
                dict_layer1 = self.equivalent_neurons.get_equivalent(layer1, neuron1, front1, decomposed1)
                decomposed_next, front_next = self.verify_variable_z(layer_next, neuron_next, front_of_matrix_next)
                dict_layer_next = self.equivalent_neurons.get_equivalent(layer_next, neuron_next, front_next, decomposed_next)
                self.add_var(dict1=dict_layer1, dict2=dict_layer_next, value=-weight * coeff1)
            else:
                # if (layer_next - layer1 > 2) and (layer1>0):
                #     print(f"STUDY RELU : ", f"Product of z_{layer_next} and z_{layer1} is NOT present in the variable matrices, adding product with bounding.")
                # Produit non présent, utilisation d'encadrement avec bornes de mccormick
                if bound_strategy is not None:
                    # Clé unique pour ce produit croisé spécifique
                    strategy_key = (layer_prev, neuron_prev, layer1, neuron1)
                    # On suit la recette. Si la clé n'est pas dans la recette, on utilise 'composed' par défaut.
                    bound_type_ = bound_strategy.get(strategy_key, "composed")
                elif bound_type == "random":
                    # Comportement aléatoire original
                    bound_type_ = random.choice(["one_variable", "composed"])
                else:
                    mccormick_type = mccormick_type

                if mccormick_type == "composed":
                    coeff_next_, cst_ = self.add_z_quad_bound_composed(layer1, neuron1, front_of_matrix_prev,
                                                                       weight, coeff1, U_next, bound_sense)
                else:
                    coeff_next_, cst_ = self.add_z_quad_bound_one_variable(layer1, neuron1, weight, coeff1, bound_sense)

                coeff_next += coeff_next_
                cst        += cst_
        
            
        if coeff_next != 0:
            self.add_linear_variable(
                var="z",
                layer=layer_next,
                neuron=neuron_next,
                value=coeff_next,
                front_of_matrix=front_of_matrix_next,
            )
        if cst != 0:
            self.add_constant(cst)

        # Linear term from the constant part of the stable-active decomposition
        if weight * constant != 0:
            self.add_linear_variable(
                var="z",
                layer=layer_next,
                neuron=neuron_next,
                value=-weight * constant,
                front_of_matrix=front_of_matrix_next,
            )


   
    # def add_linear_variable(self, var: str, value: float, **kwargs):
    #     """
    #     Add a linear variables to the constraint.
    #     Checks if the variable present corresponds to stable active neurons and divides it in this case to the precedent layer's z variables.
    #     """
    #     print(
    #         "\n \n \n           Adding linear variable:",
    #         var,
    #         "with value:",
    #         value,
    #         "and kwargs:",
    #         kwargs,
    #     )
    #     if value == 0:
    #         print("Value is 0, skipping addition of variable.")
    #         return
    #     i, num_matrix, val, constant = self.call_variable(var, **kwargs)
    #     j = np.zeros(i.shape, dtype=int)  # Assuming j is always 0 for linear variables
    #     self.add_var(
    #         i=i,
    #         j=j,
    #         num_matrix=num_matrix,
    #         value=value * val,
    #     )
    #     self.add_constant(value * constant)

    # def add_quad_variable(self, var1: str, var2: str, value: float, **kwargs):
    #     """
    #     Add a product of two variables to the constraint.
    #     Checks if the variables present corresponds to stable active neurons and divides them in this case to the precedent layer's z variables.
    #     """
    #     print(
    #         "\n \n          Adding quadratic variable:",
    #         var1,
    #         "and",
    #         var2,
    #         "with value:",
    #         value,
    #         "and kwargs:",
    #         kwargs,
    #     )
    #     if value == 0:
    #         print("Value is 0, skipping addition of variable.")
    #         return
    #     if var1 == "z":
    #         layer = kwargs.get("layer1", None)
    #         neuron = kwargs.get("neuron1", None)
    #         front_of_matrix = kwargs.get("front_of_matrix1", True)
    #         assert layer is not None, "Layer must be specified for z variable."
    #         assert neuron is not None, "Neuron must be specified for z variable."
    #         assert (
    #             front_of_matrix is not None
    #         ), "Front of matrix must be specified for z variable."
    #         i1, num_matrix1, val1, constant1 = self.call_variable(
    #             var1, layer=layer, neuron=neuron, front_of_matrix=front_of_matrix
    #         )

    #     else:
    #         class_label = kwargs.get("class_label", None)
    #         assert (
    #             class_label is not None
    #         ), "Class label must be specified for beta variable."
    #         i1, num_matrix1, val1, constant1 = self.call_variable(
    #             var1, class_label=class_label
    #         )

    #     if var2 == "z":
    #         layer = kwargs.get("layer2", None)
    #         neuron = kwargs.get("neuron2", None)
    #         front_of_matrix = kwargs.get("front_of_matrix2", True)
    #         assert layer is not None, "Layer must be specified for z variable."
    #         assert neuron is not None, "Neuron must be specified for z variable."
    #         assert (
    #             front_of_matrix is not None
    #         ), "Front of matrix must be specified for z variable."
    #         i2, num_matrix2, val2, constant2 = self.call_variable(
    #             var2, layer=layer, neuron=neuron, front_of_matrix=front_of_matrix
    #         )

    #     else:
    #         class_label = kwargs.get("class_label", None)
    #         assert (
    #             class_label is not None
    #         ), "Class label must be specified for beta variable."
    #         i2, num_matrix2, val2, constant2 = self.call_variable(
    #             var2, class_label=class_label
    #         )

    #     print("num_matrix1 :", num_matrix1, "num_matrix2 :", num_matrix2)
    #     assert (
    #         set(num_matrix1) == set(num_matrix2) and len(set(num_matrix1)) == 1
    #     ), "The matrices for the two variables must be the same and should have only one value."

    #     i1_broadcast = i1[:, np.newaxis]
    #     i2_broadcast = i2[np.newaxis, :]
    #     val1_broadcast = val1[:, np.newaxis]
    #     val2_broadcast = val2[np.newaxis, :]

    #     val_quad = np.multiply(val1_broadcast, val2_broadcast).flatten()
    #     print("val :", val_quad)
    #     i_j = np.array(np.meshgrid(i2_broadcast, i1_broadcast)).flatten()
    #     i_j = np.array(i_j).flatten()
    #     print("val .shape :     ", val_quad.shape)
    #     i_quad = i_j[: val_quad.shape[0]]
    #     j_quad = i_j[val_quad.shape[0] :]
    #     print("i :", i_quad)
    #     print("j : ", j_quad)
    #     num_matrix = np.broadcast_to(
    #         num_matrix1[:, np.newaxis], (len(num_matrix1), len(num_matrix2))
    #     ).flatten()
    #     print("num_matrix :", num_matrix)

    #     self.add_var(i=i_quad, j=j_quad, num_matrix=num_matrix, value=val_quad * value)

    #     if constant1 != 0 or constant2 != 0:
    #         self.add_constant(value * (constant1 + constant2))
    #     if constant1 != 0:
    #         self.add_var(
    #             i=i2,
    #             j=np.zeros(i2.shape, dtype=int),
    #             num_matrix=num_matrix2,
    #             value=val2 * value * constant1,
    #         )
    #     if constant2 != 0:
    #         self.add_var(
    #             i=i1,
    #             j=np.zeros(i1.shape, dtype=int),
    #             num_matrix=num_matrix1,
    #             value=val1 * value * constant2,
    #         )