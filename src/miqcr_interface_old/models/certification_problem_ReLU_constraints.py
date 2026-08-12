import logging

logger_cb = logging.getLogger("Conic_bundle_logger")

# ************************** ReLU Activation Constraints **************************


def ReLU_cstr(self):
    """
    Add ReLU activation functions to the conic bundle.
    """
    for k in range(1, self.K):
        for neuron in range(self.n[k]):
            if (k, neuron) in self.stable_inactives_neurons:
                continue
            # Add Linear Constraint   -zk + Wk zk-1    <= -bk
            self.Linear_inequality.new_constraint()
            self.Linear_inequality.add_vector_values("z", -1.0, layer=k, neuron=neuron)
            for j in range(self.n[k - 1]):
                if (k - 1, j) in self.stable_inactives_neurons:
                    continue

                self.Linear_inequality.add_vector_values(
                    "z", self.W[k - 1][neuron][j], layer=k - 1, neuron=j
                )

            self.Linear_inequality.add_bound(-self.b[k - 1][neuron])

            # Add Equality Quad Constraint zk * (zk - Wk zk-1 - bk) = 0
            self.Quad_equality.new_constraint()
            self.Quad_equality.add_matrix_values(
                var1="z",
                layer1=k,
                neuron1=neuron,
                var2="z",
                layer2=k,
                neuron2=neuron,
                value=1.0,
            )
            for j in range(self.n[k - 1]):
                if (k - 1, j) in self.stable_inactives_neurons:
                    continue

                self.Quad_equality.add_matrix_values(
                    var1="z",
                    layer1=k,
                    neuron1=neuron,
                    var2="z",
                    layer2=k - 1,
                    neuron2=j,
                    value=-self.W[k - 1][neuron][j],
                )
            self.Quad_equality.add_matrix_values(
                var1="z",
                layer1=k,
                neuron1=neuron,
                value=-self.b[k - 1][neuron],
            )
            self.Quad_equality.add_bound(0.0)


def ReLU_triangular_constraints(self):
    """
    Add the triangular constraints for ReLU activation functions.
    """
    for k in range(1, self.K + 1 if self.LAST_LAYER else self.K):
        for j in range(self.n[k]):
            if (k, j) in self.stable_inactives_neurons:
                continue
            if abs(self.U[k][j] - self.L[k][j]) <= 1e-6:
                logger_cb.warning(
                    f"Layer {k}, Neuron {j} : L={self.L[k][j]} and U={self.U[k][j]} are equal, triangular ReLU constraint is not added."
                )
                continue

            rel_u = max(self.U[k][j], 0)
            rel_l = max(self.L[k][j], 0)
            k_cst = (rel_u - rel_l) / (self.U[k][j] - self.L[k][j])

            # zk <= k * (Wk zk-1 + bk - Lk) + ReLU(Lk)
            self.Linear_inequality.new_constraint()
            self.Linear_inequality.add_vector_values("z", 1.0, layer=k, neuron=j)

            for i in range(self.n[k - 1]):
                if (k - 1, i) in self.stable_inactives_neurons:
                    continue
                self.Linear_inequality.add_vector_values(
                    "z", -k_cst * self.network.W[k - 1][j][i], layer=k - 1, neuron=i
                )

            self.Linear_inequality.add_bound(
                rel_l + k_cst * (self.b[k - 1][j] - self.L[k][j])
            )


def ReLU_cstr_greater_than_zero_part(self):
    for k in range(1, self.K):
        for neuron in range(self.n[k]):
            if (k, neuron) in self.stable_inactives_neurons:
                continue
            # Add Linear Constraint   -zk <= 0
            self.Linear_inequality.new_constraint()
            self.Linear_inequality.add_vector_values("z", -1.0, layer=k, neuron=neuron)
            self.Linear_inequality.add_bound(0.0)
