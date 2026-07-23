from typing import List


class Indexes_Variables_for_Conic_Bundle_Parser:
    """
    Class to handle the indexes of the variables and constraints in the MOSEK solver.
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        CHORDAL_DECOMPOSITION: bool = False,
        LAST_LAYER: bool = True,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        **kwargs,
    ):
        """
        Initialize the Indexes_Mosek_Solver class.

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
        """
        self.n = n
        self.K = K
        self.ytrue = kwargs.get("ytrue", None)
        self.CHORDAL_DECOMPOSITION = CHORDAL_DECOMPOSITION
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")

        print("stable_inactives_neurons : ", self.stable_inactives_neurons)

    def get_number_stable_neurons_before_layer(
        self, layer: int, neuron: int = None
    ) -> int:
        """
        Get the number of inactive neurons before a given layer and neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        neuron: int
            The neuron number.

        Returns
        -------
        int
            The number of inactive neurons.
        """
        if self.stable_inactives_neurons is None:
            return 0

        if neuron is None:
            neuron = self.n[layer]
        return len(
            [
                (k, j)
                for k, j in self.stable_inactives_neurons
                if (k < layer or (k == layer and j < neuron))
            ]
        )

    def index_variable_z(self, layer: int, neuron: int, matrix: bool = True) -> int:
        """
        Get the index of the z variable for a given layer and neuron.

        Parameters
        ----------
        layer: int
            The layer number.
        neuron: int
            The neuron number.
        matrix : bool
            Whether the variable is in a matrix or in a vector

        Returns
        -------
        int
            The index of the z variable.
        """

        if (layer == self.K and not self.LAST_LAYER) or layer < 0 or layer > self.K:
            raise ValueError(f"Layer index {layer} out of range.")

        if (layer, neuron) in self.stable_inactives_neurons:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is inactive and cannot be used."
            )
        if self.BETAS:
            print("self.BETAS dans index_variables")
            if matrix:
                return (
                    1
                    + self.n[self.K]
                    - 1
                    + sum(self.n[:layer])
                    + neuron
                    - self.get_number_stable_neurons_before_layer(layer, neuron)
                )
            else:
                return (
                    self.n[self.K]
                    - 1
                    + sum(self.n[:layer])
                    + neuron
                    - self.get_number_stable_neurons_before_layer(layer, neuron)
                )
        else:
            print("NOT self.BETAS dans index_variables")
            if matrix:
                return (
                    1
                    + sum(self.n[:layer])
                    + neuron
                    - self.get_number_stable_neurons_before_layer(layer, neuron)
                )
            else:
                return (
                    sum(self.n[:layer])
                    + neuron
                    - self.get_number_stable_neurons_before_layer(layer, neuron)
                )

    def index_variable_beta(self, class_label: int, matrix: bool = True) -> int:
        """
        Get the index of the beta variable for a given class label.

        Parameters
        ----------
        class_label: int
            The class label.

        Returns
        -------
        int
            The index of the beta variable.
        """
        assert self.BETAS
        print("class_label : ", class_label)
        print("matrix : ", matrix)
        if matrix:
            if class_label < self.ytrue:
                ind = 1 + class_label
            else:
                ind = class_label
        else:
            if class_label < self.ytrue:
                ind = class_label
            else:
                ind = class_label - 1
        print("Indice : ", ind)
        return ind

    def index_variable_zbar(self, matrix: bool = True) -> int:
        """
        Get the index of the zbar variable.

        Returns
        -------
        int
            The index of the zbar variable.
        """
        assert self.ZBAR
        assert self.BETAS_Z

        if self.LAST_LAYER:
            if matrix:
                return (
                    1
                    + self.n[self.K]
                    - 1
                    + sum(self.n)
                    - self.get_number_stable_neurons_before_layer(self.K + 1, 0)
                )
            else:
                return (
                    self.n[self.K]
                    - 1
                    + sum(self.n)
                    - self.get_number_stable_neurons_before_layer(self.K + 1, 0)
                )
        else:
            if matrix:
                return (
                    1
                    + self.n[self.K]
                    - 1
                    + sum(self.n[: self.K])
                    - self.get_number_stable_neurons_before_layer(self.K + 1, 0)
                )
            else:
                return (
                    self.n[self.K]
                    - 1
                    + sum(self.n[: self.K])
                    - self.get_number_stable_neurons_before_layer(self.K + 1, 0)
                )

    def _get_variable_index(self, var_type: str, is_first: bool, **kwargs):
        """
        Helper method to get variable index based on type and parameters for z, beta, and zbar variables.
        Parameters :
        var_type : str
            Type of the variable (z, beta, zbar)
        is_first : bool
            Whether the variable is the first or second in the pair (True for linear variables)
        """
        suffix = "1" if is_first else "2"
        matrix = kwargs.get("matrix", True)

        if var_type == "z":
            # For z variables
            layer = kwargs.get(
                f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            )
            neuron = kwargs.get(
                f"neuron{suffix}" if f"neuron{suffix}" in kwargs else "neuron"
            )

            if layer is None or neuron is None:
                raise ValueError(
                    f"Layer and neuron required for z variable ({'first' if is_first else 'second'})"
                )

            return self.index_variable_z(layer, neuron, matrix)

        elif var_type == "beta":
            # For beta variables
            class_label = kwargs.get(
                f"class_label{suffix}"
                if f"class_label{suffix}" in kwargs
                else "class_label"
            )

            if class_label is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )

            return self.index_variable_beta(class_label, matrix)

        elif var_type == "zbar":
            # For zbar variables
            zbar_ind = self.index_variable_zbar(matrix)
            print("zbar index : ", zbar_ind)
            return zbar_ind

        else:
            raise ValueError(f"Unknown variable type: {var_type}")
