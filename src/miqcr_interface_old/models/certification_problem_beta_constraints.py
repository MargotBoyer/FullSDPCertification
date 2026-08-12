def beta_sum_equal_1(self):
    """
    Add the constraint that the sum of beta variables is equal to 1.
    """
    self.Linear_equality.new_constraint()
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        self.Linear_equality.add_vector_values(
            var="beta",
            class_label=j,
            value=1.0,
        )
    self.Linear_equality.add_bound(1.0)


def zbar_sum_betaz(self):
    """
    Add the constraint that zbar is equal to the sum of beta * z.
    """
    self.Quad_equality.new_constraint()
    self.Quad_equality.add_matrix_values(
        var1="zbar",
        value=1.0,
    )
    for j in self.ytargets:
        if j == self.ytrue:
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.Quad_equality.add_matrix_values(
                var1="beta",
                class_label=j,
                var2="z",
                layer=self.K - 1,
                neuron=i,
                value=-self.W[self.K - 1][j][i],
            )
        self.Quad_equality.add_matrix_values(
            var1="beta",
            class_label1=j,
            value=-self.b[self.K - 1][j],
        )

    self.Quad_equality.add_bound(-self.cst_zbar)
