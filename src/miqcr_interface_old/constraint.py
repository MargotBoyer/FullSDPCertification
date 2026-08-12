from .indexes_variables import Indexes_Variables_for_Conic_Bundle_Parser
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from fastsdp_tools.utils import check_condition_decorator


class Constraint(Indexes_Variables_for_Conic_Bundle_Parser):
    def __init__(self, file, **kwargs):
        """ "
        Initialize the Constraint class.
        """
        super().__init__(**kwargs)
        self.file = file
        self.empty = True

        self.total_number_constr = 0
        self.current_num_constraint = 0

        self.print_vector = []
        self.print_bound = []

    def add_vector_values(self, var: str, value: float, **kwargs):
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

        j = self._get_variable_index(var, is_first=True, matrix=False, **kwargs)

        if kwargs.get("var2", None) is not None:
            raise ValueError("The second variable is not supported for vector values.")

        dic = {"num_ctsr": self.current_num_constraint, "i": j, "value": value}
        self.print_vector.append(dic)

    def add_bound(self, bound):
        self.print_bound.append(
            {"num_ctsr": self.current_num_constraint, "bound": bound}
        )
        self.total_number_constr += 1

    def new_constraint(self):
        """
        Start a new constraint.
        """
        if not self.empty:
            self.current_num_constraint += 1
        else:
            self.empty = False

    def is_not_empty_and_conform(self):
        assert self.total_number_constr == len(self.print_bound)
        return len(self.print_bound) > 0

    def to_file(self):
        pass


class QuadConstraint(Constraint):
    def __init__(
        self,
        file,
        **kwargs,
    ):
        super().__init__(file, **kwargs)
        self.print_matrix = []

    def add_matrix_values(self, var1: str, value: float, **kwargs):

        j = self._get_variable_index(var1, is_first=True, matrix=True, **kwargs)

        if kwargs.get("var2", None) is not None:
            var2 = kwargs["var2"]
            i = self._get_variable_index(var2, is_first=False, matrix=True, **kwargs)
        else:
            i = 0

        if i < j:
            dic = {
                "num_ctsr": self.current_num_constraint,
                "i": i,
                "j": j,
                "value": value,
            }
        else:
            dic = {
                "num_ctsr": self.current_num_constraint,
                "i": j,
                "j": i,
                "value": value,
            }
        self.print_matrix.append(dic)


class LinearEquality(Constraint):
    def __init__(
        self,
        file,
        **kwargs,
    ):
        super().__init__(file, **kwargs)

    @check_condition_decorator("is_not_empty_and_conform")
    def to_file(self):
        """
        Write the linear equality constraint to the file.
        """
        self.file.write("A\n")
        self.file.write(f"{len(self.print_vector)}\n")
        for line in self.print_vector:
            self.file.write(f"{line['num_ctsr']} {line['i']} {line['value']}\n")
        self.file.write("b\n")
        self.file.write(f"{len(self.print_bound)}\n")
        for line in self.print_bound:
            self.file.write(f"{line['num_ctsr']} {line['bound']}\n")


class LinearInequality(Constraint):
    def __init__(
        self,
        file,
        **kwargs,
    ):
        super().__init__(file, **kwargs)

    @check_condition_decorator("is_not_empty_and_conform")
    def to_file(self):
        """
        Write the linear inequality constraint to the file.
        """
        self.file.write("D\n")
        self.file.write(f"{len(self.print_vector)}\n")
        for line in self.print_vector:
            self.file.write(f"{line['num_ctsr']} {line['i']} {line['value']}\n")
        self.file.write("e\n")
        self.file.write(f"{len(self.print_bound)}\n")
        for line in self.print_bound:
            self.file.write(f"{line['num_ctsr']} {line['bound']}\n")


class QuadEquality(QuadConstraint):
    def __init__(
        self,
        file,
        **kwargs,
    ):
        super().__init__(file, **kwargs)

    @check_condition_decorator("is_not_empty_and_conform")
    def to_file(self):
        """
        Write the quadratic equality constraint to the file.
        """
        self.file.write("Aq\n")
        self.file.write(f"{len(self.print_matrix)}\n")
        for line in self.print_matrix:
            self.file.write(
                f"{line['num_ctsr']} {line['i']} {line['j']} {line['value']}\n"
            )
        self.file.write("bq\n")
        self.file.write(f"{len(self.print_bound)}\n")
        for line in self.print_bound:
            self.file.write(f"{line['num_ctsr']} {line['bound']}\n")


class QuadInequality(QuadConstraint):
    def __init__(
        self,
        file,
        **kwargs,
    ):
        super().__init__(file, **kwargs)

    @check_condition_decorator("is_not_empty_and_conform")
    def to_file(self):
        """
        Write the quadratic equality constraint to the file.
        """
        self.file.write("Dq\n")
        self.file.write(f"{len(self.print_matrix)}\n")
        for line in self.print_matrix:
            self.file.write(
                f"{line['num_ctsr']} {line['i']} {line['j']} {line['value']}\n"
            )
        self.file.write("eq\n")
        self.file.write(f"{len(self.print_bound)}\n")
        for line in self.print_bound:
            self.file.write(f"{line['num_ctsr']} {line['bound']}\n")
