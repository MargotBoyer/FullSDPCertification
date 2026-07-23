import mosek


def CHORDAL_DECOMPOSITION_rec(self, only_linear_constraints: bool = False):
    """
    Contraintes de cohérence entre matrices adjacentes (décomposition chordale).

    Pour chaque couche frontière k partagée entre deux groupes consécutifs
    (layer_groups[i][-1] == layer_groups[i+1][0]), on impose :

      Linéaire  : P_{groupe gauche}[z_{k,j}]           == P_{groupe droit}[z_{k,j}]
      Quadratique: P_{groupe gauche}[z_{k,j} * z_{k,j2}] == P_{groupe droit}[z_{k,j} * z_{k,j2}]

    front_of_matrix=False → groupe gauche (k est le dernier élément du groupe)
    front_of_matrix=True  → groupe droit  (k est le premier élément du groupe)
    """
    print("Adding rec matrices constraint")

    layer_groups = self.handler.indexes_matrices.layer_groups

    # Calcul des couches qui sont présentes dans deux matrices consécutives 
    repetitive_layers = [
        layer_groups[i][-1]
        for i in range(len(layer_groups) - 1)
    ]

    sum_cstr = 0

    # --- Contraintes linéaires ---
    for k in repetitive_layers:
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons and not self.use_inactive_neurons:
                continue
            elif (k, j) in self.stable_actives_neurons and not self.use_active_neurons:
                continue
            if self.handler.Constraints.new_constraint(
                f"Rec: P_left[z_{k},{j}] == P_right[z_{k},{j}]",
                label="same_for_data",
            ):
                continue
            self.handler.Constraints.add_linear_variable(
                "z", value=1, layer=k, neuron=j, front_of_matrix=False,
            )
            self.handler.Constraints.add_linear_variable(
                "z", value=-1, layer=k, neuron=j, front_of_matrix=True,
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)
            sum_cstr += 1

    # --- Contraintes quadratiques ---
    if not only_linear_constraints:
        print("Adding rec matrices quadratic constraint")
        for k in repetitive_layers:
            for j in range(self.n[k]):
                if (k, j) in self.stable_inactives_neurons and not self.use_inactive_neurons:
                    continue
                elif (k, j) in self.stable_actives_neurons and not self.use_active_neurons:
                    continue
                for j2 in range(j + 1):
                    if (k, j2) in self.stable_inactives_neurons and not self.use_inactive_neurons:
                        continue
                    elif (k, j2) in self.stable_actives_neurons and not self.use_active_neurons:
                        continue
                    if self.handler.Constraints.new_constraint(
                        f"Rec: P_left[z_{k},{j} * z_{k},{j2}] == P_right[z_{k},{j} * z_{k},{j2}]",
                        label="same_for_data",
                    ):
                        continue
                    self.handler.Constraints.add_quad_variable(
                        var1="z", layer1=k, neuron1=j,
                        var2="z", layer2=k, neuron2=j2,
                        value=1,
                        front_of_matrix1=False, front_of_matrix2=False,
                    )
                    self.handler.Constraints.add_quad_variable(
                        var1="z", layer1=k, neuron1=j,
                        var2="z", layer2=k, neuron2=j2,
                        value=-1,
                        front_of_matrix1=True, front_of_matrix2=True,
                    )
                    self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)
                    sum_cstr += 1

    print(
        f"Number of constraints for the matrix by layers: {sum_cstr} "
        f"for {self.K} layers, {self.n} neurons, "
        f"boundary layers: {repetitive_layers}"
    )
