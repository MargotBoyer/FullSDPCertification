import mosek
from fastsdp_tools import infinity
import logging
import random

logger_mosek = logging.getLogger("Mosek_logger")


def ReLU_constraint_stable_active_relaxation(
    self, k, j,
    bound_sense: str = "upper",
    bound_type: str = "composed",
    name: str = "",
    bound_strategy: dict = None
):
    assert mccormick_type in ["one_variable", "composed", "random"]
    assert bound_sense in ["lower", "upper"]
    assert any(
        (k - 1, i) in self.stable_actives_neurons for i in range(self.n[k - 1])
    ), f"Neuron ({k}, {j}) has no previous stable active neuron."


    if self.handler.Constraints.new_constraint( # type: ignore
        f"ReLU Relaxed - Layer {k} - z_{k,j} * (z{k,j} - W_{k,j}' z_{k-1}' - b_{k,j}) - M_{k,j} * z_{k,j}'' <= 0 - {bound_type} - {bound_sense} - {name}", label = "same_for_data"
    ):
        return

    # print(f"    STUDY NEW SUBCONSTRAINT  RELU k = {k} j = {j}, bound_sense = {bound_sense}; mccormick_type = {mccormick_type}")
    self.handler.Constraints.add_quad_variable(
        var1="z",
        layer1=k,
        neuron1=j,
        var2="z",
        layer2=k,
        neuron2=j,
        value=1,
        front_of_matrix1=False,
        front_of_matrix2=False,
    )

    self.handler.Constraints.add_linear_variable(
        "z",
        value=-self.network.b[k - 1][j],
        layer=k,
        neuron=j,
        front_of_matrix=False,
    )

    for i in range(self.n[k - 1]):
        if (k - 1, i) in self.stable_inactives_neurons:
            continue
        elif (k - 1, i) in self.stable_actives_neurons :
            #print(f"STUDY RELU : layer = {k-1}, neuron = {i}, weight = {self.network.W[k - 1][j][i]}")

            self.add_z_quad_active_neuron_heuristic(
                layer_prev=k - 1,
                neuron_prev=i,
                layer_next=k,
                neuron_next=j,
                front_of_matrix_prev=True,
                front_of_matrix_next=False,
                weight=self.network.W[k - 1][j][i],
                bound_sense=bound_sense,
                bound_type=bound_type, # Gardé pour la compatibilité
                bound_strategy=bound_strategy # Nouvel argument
            )
            

        else:
            # print(f"Adding non-stable neuron ({k-1}, {i}) with weight {self.network.W[k - 1][j][i]}")
            
            self.handler.Constraints.add_quad_variable(
                var1="z",
                layer1=k,
                neuron1=j,
                var2="z",
                layer2=k - 1,
                neuron2=i,
                value=-self.network.W[k - 1][j][i],
                front_of_matrix1=False,
                front_of_matrix2=True,
            )
           

    if bound_sense == "upper":
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)
    else:
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)


def ReLU_constraint_Lan(
    self, relu_quadratic_random : bool = False
):
    # print("Adding quadratic RELU constraint")
    # print(f"STUDY NEW CONSTRAINT RELU")

    for k in range(1, self.K):
        print(f"Adding ReLU constraints for layer {k}")
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                # print(f"Skipping stable inactive neuron ({k}, {j})")
                continue
            if (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                # print(f"Skipping stable active neuron ({k}, {j})")
                continue
            # print(f"STUDY : Adding ReLU constraint for layer {k}, neuron {j} : z_{k,j}>=0")
            # zk >= 0
            if self.handler.Constraints.new_constraint(f"ReLU - z_{k,j}>=0", label = "same_for_data"):
                continue
            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=0,
            )

            # zk >= Wk zk-1 + bk
            if k == 1 and not self.INPUT_IN_VARIABLES:
                # z_0 entirely removed: sub-constraints 2 and 3 skipped
                continue
            # print(f"STUDY : Adding ReLU constraint for layer {k}, neuron {j} : z_{k,j}>= Wk zk-1 + bk")
            if self.handler.Constraints.new_constraint(
                f"ReLU - z_{k,j} >= W_{k,j} z_{k-1} + b{k,j}", label = "same_for_data"
            ):
                continue

            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )

            relu_bound_adj = 0.0
            for i in range(self.n[k - 1]):
                if (k - 1, i) in self.stable_inactives_neurons:
                    continue
                if k == 1 and i in self.pruned_input_neurons:
                    # Pruned z_0[i]: replace by M_0[i] = L_0[i] if W>=0 else U_0[i] (valid lower bound)
                    W_ji = float(self.network.W[k - 1][j][i])
                    L_i = self.handler.Constraints.L[0][i]
                    U_i = self.handler.Constraints.U[0][i]
                    M_i = L_i if W_ji >= 0 else U_i
                    relu_bound_adj += W_ji * M_i
                    continue
                self.handler.Constraints.add_linear_variable(
                    "z",
                    value=-self.network.W[k - 1][j][i],
                    layer=k - 1,
                    neuron=i,
                    front_of_matrix = True
                )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=self.b[k - 1][j] + relu_bound_adj,
            )

            # zk * (zk - Wk zk-1 - bk) = 0
            has_pruned_at_prev = (k == 1 and len(self.pruned_input_neurons) > 0)
            if has_pruned_at_prev:
                # z_0[pruned] not SDP variables: quadratic terms z_1[j]*z_0[i] unavailable.
                # Skip this sub-constraint (valid relaxation: feasible set enlarges).
                continue
            if self.MATRIX_BY_LAYERS and (k - 1 <= l_max) and (
                any((k - 1, i) in self.stable_actives_neurons for i in range(self.n[k - 1]))):
                # The constraint cannot be added as it links products of variables from different matrices : a relaxation is needed

                # Le type de relaxation est maintenant contrôlé par un paramètre central.
                # Par défaut, on utilise 'all' si le paramètre n'est pas défini.
                relaxation_type = getattr(self, 'relu_relaxation_type', 'all')
                
                if relaxation_type == 'custom':
                    # Utilise la stratégie personnalisée fournie via le fichier YAML
                    if not hasattr(self, 'bound_strategy') or self.bound_strategy is None:
                        raise ValueError("relu_relaxation_type='custom' requires a 'bound_strategy' dictionary.")
                    self.ReLU_constraint_stable_active_relaxation(
                        k, j, bound_sense="upper", name="strategy_upper", bound_strategy=self.bound_strategy
                    )
                    self.ReLU_constraint_stable_active_relaxation(
                        k, j, bound_sense="lower", name="strategy_lower", bound_strategy=self.bound_strategy
                    )
                elif relaxation_type == 'one_variable':
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="upper", bound_type="one_variable")
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="lower", bound_type="one_variable")

                elif relaxation_type == 'composed':
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="upper", bound_type="composed")
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="lower", bound_type="composed")

                elif relaxation_type == 'random':
                    # Note: self.relu_random_samples doit être défini dans la configuration du solveur
                    for i in range(getattr(self, 'relu_random_samples', 8)):
                        self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="upper", bound_type="random", name=f"random_{i}")
                        self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="lower", bound_type="random", name=f"random_{i}")
                
                else:  # 'all' est le comportement par défaut
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="upper", bound_type="one_variable")
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="lower", bound_type="one_variable")
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="upper", bound_type="composed")
                    self.ReLU_constraint_stable_active_relaxation(k, j, bound_sense="lower", bound_type="composed")

            else:
                # print("Adding normal ReLU constraint for layer", k, "neuron", j)
                if self.handler.Constraints.new_constraint(
                    f"ReLU - z_{k,j} * (z{k,j} - W_{k,j} z_{k-1} - b_{k,j}) = 0", label = "same_for_data"
                ):
                    continue

                self.handler.Constraints.add_quad_variable(
                    var1="z",
                    layer1=k,
                    neuron1=j,
                    var2="z",
                    layer2=k,
                    neuron2=j,
                    value=1,
                    front_of_matrix1=False,
                    front_of_matrix2=False,
                )

                self.handler.Constraints.add_linear_variable(
                    "z",
                    value=-self.network.b[k - 1][j],
                    layer=k,
                    neuron=j,
                    front_of_matrix=False,
                )
                # Constraint can be added as it links products of variables from the same matrix
                for i in range(self.n[k - 1]):
                    if (k - 1, i) in self.stable_inactives_neurons:
                        continue
                    self.handler.Constraints.add_quad_variable(
                        var1="z",
                        layer1=k,
                        neuron1=j,
                        var2="z",
                        layer2=k - 1,
                        neuron2=i,
                        value=-self.network.W[k - 1][j][i],
                        front_of_matrix1=False,
                        front_of_matrix2=True,
                    )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.fx, bound=0
                )


def ReLU_triangularization(self):
    for k in range(1, self.K):
        if k == 1 and not self.INPUT_IN_VARIABLES:
            # z_0 entirely removed: triangular constraint skipped
            continue
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            if (k, j) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            U_kj = self.handler.Constraints.U[k][j]
            L_kj = self.handler.Constraints.L[k][j]

            if U_kj < L_kj:
                raise ValueError(
                    f"Layer {k}, Neuron {j} : inverted bounds L={L_kj} > U={U_kj}."
                )
            if abs(U_kj - L_kj) <= 1e-6:
                logger_mosek.warning(
                    f"Layer {k}, Neuron {j} : L={L_kj} and U={U_kj} are equal, triangular ReLU constraint is not added."
                )
                continue

            rel_u = max(U_kj, 0)
            rel_l = max(L_kj, 0)
            k_cst = (rel_u - rel_l) / (U_kj - L_kj)
            # print(
            #     "k_cst:",
            #     k_cst,
            #     "rel_u:",
            #     rel_u,
            #     "rel_l:",
            #     rel_l,
            #     "L : ",
            #     self.handler.Constraints.L[k][j],
            #     "U :",
            #     self.handler.Constraints.U[k][j],
            # )

            # zk <= k * (Wk zk-1 + bk - Lk) + ReLU(Lk)
            if self.handler.Constraints.new_constraint(
                f"ReLU - z_{k,j} <= kcst * (W{k,j} z_{k-1} + b_{k,j} - L{k,j}) + ReLU(L_{k,j})", label = "same_for_data"
            ):
                continue
            self.handler.Constraints.add_linear_variable(
                "z",
                value=1,
                layer=k,
                neuron=j,
                front_of_matrix=False,
            )
            tri_bound_adj = 0.0
            for i in range(self.n[k - 1]):
                if (k - 1, i) in self.stable_inactives_neurons:
                    continue
                if k == 1 and i in self.pruned_input_neurons:
                    # Pruned z_0[i]: replace by M_0[i] = U_0[i] if W>=0 else L_0[i] (valid upper bound)
                    W_ji = float(self.network.W[k - 1][j][i])
                    L_i = self.handler.Constraints.L[0][i]
                    U_i = self.handler.Constraints.U[0][i]
                    M_i = U_i if W_ji >= 0 else L_i
                    tri_bound_adj += k_cst * W_ji * M_i
                    continue
                self.handler.Constraints.add_linear_variable(
                    "z",
                    layer=k - 1,
                    neuron=i,
                    value=-k_cst * self.network.W[k - 1][j][i],
                    front_of_matrix = True,
                )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.up,
                bound=rel_l
                + k_cst * (self.network.b[k - 1][j] - self.handler.Constraints.L[k][j])
                + tri_bound_adj,
            )

            # self.handler.Constraints.print_current_constraint()


def last_layer_linear_equality(self):
    """
    Add z_{K,j} = W_K z_{K-1} + b_K for each output neuron j.
    Mandatory when LAST_LAYER=True: the last layer is linear (no ReLU), so
    this equality is never added by ReLU_constraint_Lan and must be explicit.
    z_K  → front_of_matrix=False (last element of last chordal group)
    z_{K-1} → front_of_matrix=True  (first element of last chordal group)
    """
    assert self.LAST_LAYER
    for class_label in list(set([self.ytrue]).union(self.ytargets)):
        logger_mosek.debug("adding last layer linear equality for class ", class_label)
        if self.handler.Constraints.new_constraint(
            f"Last layer linear equality: z_{{{self.K},{class_label}}} = W_K z_{{K-1}} + b_K",
            label="same_for_data",
        ):
            continue
        self.handler.Constraints.add_linear_variable(
            "z",
            value=1,
            layer=self.K,
            neuron=class_label,
            front_of_matrix=False,
        )
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.handler.Constraints.add_linear_variable(
                "z",
                value=-self.network.W[self.K - 1][class_label][i],
                layer=self.K - 1,
                neuron=i,
                front_of_matrix=True,
            )
        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.fx,
            bound=self.network.b[self.K - 1][class_label],
        )
