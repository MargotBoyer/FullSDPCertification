import mosek
from fastsdp_tools import infinity


# *********************************************BETAS ***************************************************************
def discrete_betas(self):
    """
    Add the constraint beta_j = beta_j² for the beta variable (ensuring beta is in [0,1])
    """
    assert self.BETAS
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        # beta_j = beta_j²
        if self.handler.Constraints.new_constraint(
            f"beta_{j,j} = beta_{j}", label="same_for_data"
        ):
            continue
        self.handler.Constraints.add_quad_variable(
            var1="beta",
            class_label1=j,
            var2="beta",
            class_label2=j,
            value=1,
        )
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=-1,
        )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)


def sum_betas_equals_1(self):
    """
    Add the constraint sum(betaj)=1 for the beta variable (ensuring beta is in [0,1])
    """
    assert self.BETAS

    if self.handler.Constraints.new_constraint("sum(beta)=1", label="same_for_data"):
        return
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=j,
            value=1,
        )
    self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=1)



def sum_beta_i_beta_j_equal_beta_i(self):
    """
    Add the constraint sum(betai*beta_j for all j) = beta_i for all i 
    """
    assert self.BETAS

    for i in self.ytargets:
        if self.handler.Constraints.new_constraint(f"sum(beta_{i} * beta_j for all j)=beta_{i}", label="same_for_data"):
            continue
        self.handler.Constraints.add_linear_variable(
            var="beta",
            class_label=i,
            value=1,
        )
        for j in self.ytargets:
            if j == self.ytrue:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="beta",
                class_label2=i,
                value=-1,
            )
        self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)




def McCormick_beta_z(self, layer: int, cuts=None):
    if layer == self.K:
        assert self.LAST_LAYER
        list_z = self.ytargets
    else :
        list_z = [i for i in range(self.n[layer]) if (layer, i) not in self.stable_inactives_neurons
                  and (layer, i) not in self.stable_actives_neurons]
    
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        for i in list_z:
            front_of_matrix = (
                False
                if (
                    layer == self.K
                    and self.LAST_LAYER
                    or layer == self.K - 1
                    and not self.LAST_LAYER
                )
                else True
            )

            if layer == self.K or layer == 0  :
                L_layer_i = self.handler.Constraints.L[layer][i]
            else :
                L_layer_i = 0
            # *************************************************
            if self.handler.Constraints.new_constraint(
                f"T_{(layer, i),j}  <= U_{layer, i} beta_{j}", label="same_for_data"
            ):
                continue

            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j,
                value=-self.handler.Constraints.U_above_zero[layer][i],
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)
            self.handler.Constraints.mark_current_dualizable("McCormick_beta_z")

            # ****************************************************
            if cuts is None or "sum_beta_logits_equal_logit" not in cuts:
                # Constraint is redundant when we have the constraint sum_beta_j_z_i_equal_z_i, but it can help convergence when we don't have it
                name_cstr_2 = f"T_{(layer, i),j} <= z_{layer, i}"
                if layer == self.K and self.LAST_LAYER or layer ==0 :
                    name_cstr_2 = f"T_{(layer, i),j} + {L_layer_i} (1 - beta_{j}) <= z_{layer, i}"
                if self.handler.Constraints.new_constraint(
                    name_cstr_2, label="same_for_data"
                ):
                    continue

                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label=j,
                    var2="z",
                    layer2=layer,
                    neuron2=i,
                    value=1,
                    front_of_matrix2=front_of_matrix,
                )
                self.handler.Constraints.add_linear_variable(
                    var="z",
                    layer=layer,
                    neuron=i,
                    value=-1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j,
                    value=-L_layer_i,
                )
                self.handler.Constraints.add_constant(
                    value=L_layer_i
                )
                self.handler.Constraints.add_bound(bound_type=mosek.boundkey.up, bound=0)
                self.handler.Constraints.mark_current_dualizable("McCormick_beta_z")

            # ****************************************************
            if cuts is None or "sum_beta_logits_equal_logit" not in cuts:
                # Constraint is redundant when we have the constraint sum_beta_j_z_i_equal_z_i, but it can help convergence when we don't have it
                if self.handler.Constraints.new_constraint(
                    f"T_{(layer, i),j}  >= U_{layer, i} beta_{j} + z_{layer, i} - U_{layer, i}",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label=j,
                    var2="z",
                    layer2=layer,
                    neuron2=i,
                    value=1,
                    front_of_matrix2=front_of_matrix,
                )
                self.handler.Constraints.add_linear_variable(
                    var="z",
                    layer=layer,
                    neuron=i,
                    value=-1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j,
                    value=-self.handler.Constraints.U_above_zero[layer][i],
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.lo,
                    bound=-self.handler.Constraints.U_above_zero[layer][i],
                )
                self.handler.Constraints.mark_current_dualizable("McCormick_beta_z")

            # ****************************************************
            name_cstr_4 = f"T_{(layer, i),j}  >= 0"
            if layer == self.K and self.LAST_LAYER or layer ==0 :
                name_cstr_4 = f"T_{(layer, i),j} >= beta_{j} L_{layer, i}"
            if self.handler.Constraints.new_constraint(
                name_cstr_4, label="same_for_data"
            ):
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=1,
                front_of_matrix2=front_of_matrix,
            )
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j,
                value=-L_layer_i,
            )
            self.handler.Constraints.add_bound(bound_type=mosek.boundkey.lo, bound=0)
            self.handler.Constraints.mark_current_dualizable("McCormick_beta_z")


def McCormick_beta_z_all_valid_layers(self, cuts=None):
    """
    Call McCormick_beta_z for every layer that shares the last PSD matrix with beta.
    Beta lives in the last chordal group; the valid layers are exactly the layers
    listed in that group.
    """
    last_group = self.handler.indexes_matrices.layer_groups[-1]
    for layer in last_group:
        self.McCormick_beta_z(layer=layer, cuts=cuts)



def betai_betaj(self):
    """
    Add the constraint betai * betaj
    """
    assert self.BETAS
    for j1 in self.ytargets:
        if j1 == self.ytrue:
            continue
        for j2 in self.ytargets:
            if j2 == self.ytrue or j2 >= j1:
                continue

            if self.BETAS_Z:
                # beta_j1 * beta_j2 = 0
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} * beta_{j2} = 0", label="same_for_data"
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.fx, bound=0
                )
            else:
                # beta_j1 * beta_j2 >= 0
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} * beta_{j2} >= 0", label="same_for_data"
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )

                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.lo, bound=0
                )

                # beta_j1 * beta_j2 >= betaj1 + betaj2 - 1
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} >= beta_{j1} + beta_{j2} - 1",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j1,
                    value=-1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j2,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.lo, bound=-1
                )

                # beta_j1 * beta_j2 <= min(betaj1, betaj2)
                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} <= beta_{j1}",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j1,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.up, bound=0
                )

                if self.handler.Constraints.new_constraint(
                    f"betaibetaj - beta_{j1} beta_{j2} <= beta_{j2}",
                    label="same_for_data",
                ):
                    continue
                self.handler.Constraints.add_quad_variable(
                    var1="beta",
                    class_label1=j1,
                    var2="beta",
                    class_label2=j2,
                    value=1,
                )
                self.handler.Constraints.add_linear_variable(
                    var="beta",
                    class_label=j2,
                    value=-1,
                )
                self.handler.Constraints.add_bound(
                    bound_type=mosek.boundkey.up,
                    bound=0,
                )


def z_j2_zj_big_m(self):
    """
    Add the big M constraint z_j1 >= z_j2 + (1 - beta_j1) (L_j1 - U_j2)
    """
    assert self.BETAS

    for j2 in self.ytargets:
        if j2 == self.ytrue:
            continue
        for j1 in self.ytargets:
            if j1 == self.ytrue or j1 == j2:
                continue
            if self.handler.Constraints.new_constraint(
                f"z_{self.K, j1}  >= z_{self.K,j2} + (1 - beta_{j1}) (L_{j1} - U_{j2})", label="same_for_data"
            ):
                continue
            big_M = self.handler.Constraints.L[self.K][j1] - self.handler.Constraints.U_above_zero[self.K][j2]
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K,
                neuron=j1,
                value=1,
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K,
                neuron=j2,
                value=-1,
            )
            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j1,
                value=big_M,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=big_M,
            )


def z_j2_beta_j2_greater_than_zj(self):
    """
    Add the constraint z_2 beta_2 >= z_1 - (1 - beta_2) U_1    (11)
    """
    assert self.BETAS
    assert self.BETAS_Z

    for j2 in self.ytargets:
        if j2 == self.ytrue:
            continue
        for j1 in self.ytargets:
            if j1 == self.ytrue or j1 == j2:
                continue
            if self.handler.Constraints.new_constraint(
                f"z_{self.K, j2} beta_{j2} >= z_{self.K,j1} - (1 - beta_{j2}) U_{j1}",
                label="same_for_data",
            ):
                continue

            self.handler.Constraints.add_quad_variable(
                var1="z",
                layer1=self.K,
                neuron1=j2,
                var2="beta",
                class_label=j2,
                value=1,
                front_of_matrix1=False,
            )

            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K,
                neuron=j1,
                value=-1,
            )

            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=-self.handler.Constraints.U_above_zero[self.K][j1],
            )

            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.lo,
                bound=-self.handler.Constraints.U_above_zero[self.K][j1],
            )
            self.handler.Constraints.mark_current_dualizable("beta_logits_comparaison")


def z_j2_beta_j2_less_than_zj(self):
    """
    Add the constraint z_2 beta_2 <= (1 - beta_1) z_1 + beta_2 U_2 - (1 - beta_1) L_1 + beta_2 (L_1 - z_1)   (12)
    ie      z_2 beta_2 + beta_1 z_1 - z_1 + beta_2 z_1 <=  beta_2 U_2 + beta_1 L_1 - L_1 + beta_2 L_1
    """
    assert self.BETAS

    for j2 in self.ytargets:
        if j2 == self.ytrue:
            continue
        for j1 in self.ytargets:
            if j1 == self.ytrue or j1 == j2:
                continue
            if self.handler.Constraints.new_constraint(
                f"z_{j2} beta_{j2} <= (1 - beta_{j1}) z_{j1} + beta_{j2} U_{j2} - (1 - beta_{j1}) L_{j1} + beta_{j2} (L_{j1} - z_{j1})",
                label="same_for_data",
            ):
                continue

            self.handler.Constraints.add_quad_variable(
                var1="z",
                layer1=self.K,
                neuron1=j2,
                var2="beta",
                class_label=j2,
                value=1,
                front_of_matrix1=False,
            )
            self.handler.Constraints.add_linear_variable(
                var="z",
                layer=self.K,
                neuron=j1,
                value=-1,
                front_of_matrix=False,
            )
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j1,
                var2="z",
                layer2=self.K,
                neuron2=j1,
                
                value=1,
                front_of_matrix2=False,
            )

            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j2,
                value=-self.handler.Constraints.U_above_zero[self.K][j2] - self.handler.Constraints.L[self.K][j1],
            )

            self.handler.Constraints.add_linear_variable(
                var="beta",
                class_label=j1,
                value=-self.handler.Constraints.L[self.K][j1],
            )

            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=j2,
                var2="z",
                layer2=self.K,
                neuron2=j1,
                value=1,
                front_of_matrix2=False,
            )
            self.handler.Constraints.add_bound(
                bound_type=mosek.boundkey.up,
                bound=-self.handler.Constraints.L[self.K][j1],
            )
            self.handler.Constraints.mark_current_dualizable("beta_logits_comparaison")


def sum_beta_j_z_i_equal_z_i_layer(self, layer: int):

    if layer == self.K:
        assert self.LAST_LAYER
        list_z = self.ytargets
    else :
        list_z = [i for i in range(self.n[layer]) if (layer, i) not in self.stable_inactives_neurons
                  and (layer, i) not in self.stable_actives_neurons]
        
    # Constraint sum(beta_j z_i for j in ytargets) <= z_i for every neuron i in layer
    for i in list_z:
        if self.handler.Constraints.new_constraint(
                f"sum(z_{layer}^{i} beta_j for j in targets) == z_{layer}^{i}",
                label="same_for_data",
            ):
                continue
        self.handler.Constraints.add_linear_variable(
            var="z",
            layer = layer,
            neuron = i,
            value = 1
        )
        for class_label in self.ytargets:
            if class_label == self.ytrue:
                continue
            self.handler.Constraints.add_quad_variable(
                var1="beta",
                class_label=class_label,
                var2="z",
                layer2=layer,
                neuron2=i,
                value=-1,
            )
        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.fx,
            bound=0,
        )
        self.handler.Constraints.mark_current_dualizable("sum_beta_logits_equal_logit")





def sum_beta_j_z_i_equal_z_i(self):
    """
    Add the constraint sum(beta_j z_i for j in ytargets) = z_i
    """
    
    assert self.BETAS
    assert self.BETAS_Z

    last_group = self.handler.indexes_matrices.layer_groups[-1]
    for layer in last_group:
        self.sum_beta_j_z_i_equal_z_i_layer(layer=layer)

   


# *****************************************************************************************************************
# *********************************************ZBAR ***************************************************************
# *****************************************************************************************************************
def zbar_sum_beta_z(self):
    """
    Add the constraint zbar = sum(betaj * zj) for the beta variable (ensuring beta is in [0,1])
    """
    assert self.ZBAR
    assert self.BETAS

    name_cstr = "zbar = sum("
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        name_cstr += f"beta_{j} * z_{self.K-1,j}"
        if j < len(self.ytargets) - 1:
            name_cstr += " + "
    name_cstr += ")"

    if self.handler.Constraints.new_constraint(name_cstr, label = "same_for_data"):
        return
    self.handler.Constraints.add_linear_variable(
        var="zbar",
        value=1,
    )
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        self.handler.Constraints.add_quad_variable(
            var1="beta",
            class_label=j,
            var2="z",
            layer2=self.K,
            neuron2=j,
            value=-1,
        )

    self.handler.Constraints.add_bound(bound_type=mosek.boundkey.fx, bound=0)


def zbar_max_z(self):
    """
    Add the constraint zbar >= max(zj)
    """
    assert self.ZBAR
    assert self.BETAS

    for j in self.ytargets:
        if j == self.ytrue:
            continue
        # zbar >= zj
        if self.handler.Constraints.new_constraint(f"zbar >= z_{self.K,j}", label="same_for_data"):
            continue
        self.handler.Constraints.add_linear_variable(
            var="zbar",
            value=1,
        )
        self.handler.Constraints.add_linear_variable(
            var="z",
            layer=self.K,
            neuron=j,
            value=-1,
        )

        self.handler.Constraints.add_bound(
            bound_type=mosek.boundkey.lo, bound=self.b[self.K - 1][j]
        )
