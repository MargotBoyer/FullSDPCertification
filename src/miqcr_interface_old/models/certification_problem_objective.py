def Lan_objective(self):
    """
    Add the Lan objective function to the conic bundle.
    """
    print("ytarget in Lan_objective : ", self.ytarget)
    for j in range(self.n[self.K - 1]):
        if (self.K - 1, j) in self.stable_inactives_neurons:
            continue
        self.Objective.add_vector_values(
            "z",
            self.W[self.K - 1][self.ytrue][j] - self.W[self.K - 1][self.ytarget][j],
            layer=self.K - 1,
            neuron=j,
        )

    self.Objective.add_constant_value(
        value=self.b[self.K - 1][self.ytrue] - self.b[self.K - 1][self.ytarget]
    )


def Md_objective(self):
    """
    Add the Md objective function to the conic bundle.
    """
    for i in range(self.n[self.K - 1]):
        if (self.K - 1, i) in self.stable_inactives_neurons:
            continue
        self.Objective.add_vector_values(
            "z",
            value=self.W[self.K - 1][self.ytrue][i],
            layer=self.K - 1,
            neuron=i,
        )
    self.Objective.add_constant_value(value=self.b[self.K - 1][self.ytrue])

    for j in self.ytargets:
        if j == self.ytrue:
            continue
        for i in range(self.n[self.K - 1]):
            if (self.K - 1, i) in self.stable_inactives_neurons:
                continue
            self.Objective.add_matrix_values(
                var2="beta",
                class_label=j,
                var1="z",
                layer=self.K - 1,
                neuron=i,
                value=-self.W[self.K - 1][j][i],
            )
        self.Objective.add_vector_values(
            "beta",
            class_label=j,
            value=-self.b[self.K - 1][j],
        )


def Mzbar_objective(self):
    """
    Add the Mzbar objective function to the conic bundle.
    """
    for i in range(self.n[self.K - 1]):
        if (self.K - 1, i) in self.stable_inactives_neurons:
            continue
        self.Objective.add_vector_values(
            "z",
            value=self.W[self.K - 1][self.ytrue][i],
            layer=self.K - 1,
            neuron=i,
        )
    self.Objective.add_constant_value(
        value=self.b[self.K - 1][self.ytrue] - self.cst_zbar
    )

    self.Objective.add_vector_values("zbar", value=-1)
