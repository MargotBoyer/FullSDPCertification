from .indexes_variables import Indexes_Variables_for_Conic_Bundle_Parser


class Objective(Indexes_Variables_for_Conic_Bundle_Parser):
    def __init__(self, file, *args, **kwargs):
        """ "
        Initialize the Objective class.
        """
        super().__init__(*args, **kwargs)
        self.initialize(file)

        print("K dans objectif : ", self.K)
        print("n dans objectif : ", self.n)

    def add_vector_values(self, var1: str, value: float, **kwargs):
        """
        Add a value for an index of the vector to the current constraint.

        Parameters
        ----------
        var1: str
            The first variable.
        var2: str (optional)
            The second variable.
        **kwargs: List
            The keyword arguments for the variables.
        """

        j = self._get_variable_index(var1, is_first=True, matrix=False, **kwargs)

        if kwargs.get("var2", None) is not None:
            raise ValueError("The second variable is not supported for vector values.")

        dic = {"i": j, "value": value}
        self.print_vector.append(dic)

    def add_matrix_values(self, var1: str, value: float, **kwargs):
        """
        Add a value to an index of the matrix to the current constraint
        Parameters
        ----------
        var1: str
            The first variable.
        var2: str (optional)
            The second variable.
        **kwargs: List
            The keyword arguments for the variables.
        """

        i = self._get_variable_index(var1, is_first=True, matrix=True, **kwargs)

        if kwargs.get("var2", None) is not None:
            var2 = kwargs["var2"]
            j = self._get_variable_index(var2, is_first=False, matrix=True, **kwargs)

        if i < j:
            dic = {"i": i - 1, "j": j - 1, "value": value}
        else:
            dic = {"i": j - 1, "j": i - 1, "value": value}
        self.print_matrix.append(dic)

    def add_constant_value(self, value: float):
        """
        Add a constant value to the objective function.
        """
        self.print_constant = value

    def check_conformity(self):
        pass

    def initialize(self, file):
        """
        Initialize the objective function.
        """
        self.file = file
        self.print_vector = []
        self.print_matrix = []
        self.print_constant = 0

    def to_file(self):
        """
        Write the objective to the file.
        """
        self.file.write("Q\n")
        self.file.write(f"{len(self.print_matrix)}\n")
        for line in self.print_matrix:
            self.file.write(f"{line['i']} {line['j']} {line['value']}\n")

        self.file.write("C\n")
        self.file.write(f"{len(self.print_vector)}\n")
        for line in self.print_vector:
            self.file.write(f"{line['i']} {line['value']}\n")

        self.file.write("r\n")
        self.file.write(f"{self.print_constant}\n")
