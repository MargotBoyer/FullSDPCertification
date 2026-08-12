def McCormick_all_layers_neurons(self):
    """
    Add the McCormick constraints to the conic bundle.
    """
    for layer1 in range(self.K + 1 if self.LAST_LAYER else self.K):
        for layer2 in range(layer1 + 1):
            for neuron1 in range(self.n[layer1]):
                if (layer1, neuron1) in self.stable_inactives_neurons:
                    continue
                for neuron2 in range(self.n[layer2]):
                    if (layer2, neuron2) in self.stable_inactives_neurons:
                        continue
                    if layer1 == layer2 and neuron1 <= neuron2:
                        continue
                    self.McCormick_different_neurons(layer1, neuron1, layer2, neuron2)


def McCormick_diagonal(self):
    """
    Add the McCormick constraints for diagonal elements.
    """
    for layer in range(self.K + 1 if self.LAST_LAYER else self.K):
        for neuron in range(self.n[layer]):
            if (layer, neuron) in self.stable_inactives_neurons:
                continue
            self.McCormick_same_neurons(layer, neuron)


def McCormick_same_neurons(self, layer: int, neuron: int):
    """
    Add the McCormick constraints for the same neuron in a layer.
    This is a special case where the layer and neuron indices are the same.
    This wiull treat specificaaly the diagonal variables.
    """
    # z_{k j}² >= 2 * z_{k j} * U_{k j} - U_{k j}²
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        var2="z",
        layer2=layer,
        neuron2=neuron,
        value=-1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        value=2 * self.U_above_zero[layer][neuron],
    )

    self.Quad_inequality.add_bound(
        self.U_above_zero[layer][neuron] * self.U_above_zero[layer][neuron],
    )

    # z_{k j}² >= 2 * z_{k j} * L_{k j} - L_{k j}²
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        var2="z",
        layer2=layer,
        neuron2=neuron,
        value=-1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        value=2 * self.L[layer][neuron],
    )
    self.Quad_inequality.add_bound(
        self.L[layer][neuron] * self.L[layer][neuron],
    )

    # z_{k j}² <=  z_{k j} * (U_{k j} + L_{k j}) - L_{k j} * U_{k j}
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        var2="z",
        layer2=layer,
        neuron2=neuron,
        value=1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer,
        neuron1=neuron,
        value=-(self.U_above_zero[layer][neuron] + self.L[layer][neuron]),
    )

    self.Quad_inequality.add_bound(
        -(self.U_above_zero[layer][neuron] * self.L[layer][neuron]),
    )


def McCormick_different_neurons(
    self, layer1: int, neuron1: int, layer2: int, neuron2: int
):
    """
    Add the constraint that zbar is equal to the sum of beta * z.
    """
    assert layer1 != layer2 or neuron1 != neuron2, (
        f"Trying to add McCormick constraints for the same neuron in layer {layer1} and {layer2}."
        f" Neuron {neuron1} and {neuron2}."
    )
    # z_{k1 j1} * z_{k2 j2} >= z_{k1 j1} * U_{k2 j2} + z_{k2 j2} * U_{k1 j1} - U_{k1 j1} * U_{k2 j2}
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=-1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        value=self.U_above_zero[layer2][neuron2],
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer2,
        neuron1=neuron2,
        value=self.U_above_zero[layer1][neuron1],
    )
    self.Quad_inequality.add_bound(
        self.U_above_zero[layer1][neuron1] * self.U_above_zero[layer2][neuron2],
    )

    # z_{k1 j1} * z_{k2 j2} >= z_{k1 j1} * L_{k2 j2} + z_{k2 j2} * L_{k1 j1} - L_{k1 j1} * L_{k2 j2}
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=-1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        value=self.L[layer2][neuron2],
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer2,
        neuron1=neuron2,
        value=self.L[layer1][neuron1],
    )
    self.Quad_inequality.add_bound(
        self.L[layer1][neuron1] * self.L[layer2][neuron2],
    )

    # z_{k1 j1} * z_{k2 j2} <= z_{k1 j1} * U_{k2 j2} + z_{k2 j2} * U_{k1 j1} - L_{k1 j1} * U_{k2 j2}
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        value=-self.U_above_zero[layer2][neuron2],
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer2,
        neuron1=neuron2,
        value=-self.L[layer1][neuron1],
    )
    self.Quad_inequality.add_bound(
        -self.L[layer1][neuron1] * self.U_above_zero[layer2][neuron2],
    )

    # z_{k1 j1} * z_{k2 j2} <= z_{k1 j1} * L_{k2 j2} + z_{k2 j2} * U_{k1 j1} - U_{k1 j1} * L_{k2 j2}
    self.Quad_inequality.new_constraint()
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        var2="z",
        layer2=layer2,
        neuron2=neuron2,
        value=1.0,
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer1,
        neuron1=neuron1,
        value=-self.L[layer2][neuron2],
    )
    self.Quad_inequality.add_matrix_values(
        var1="z",
        layer1=layer2,
        neuron1=neuron2,
        value=-self.U_above_zero[layer1][neuron1],
    )
    self.Quad_inequality.add_bound(
        -self.U_above_zero[layer1][neuron1] * self.L[layer2][neuron2],
    )
