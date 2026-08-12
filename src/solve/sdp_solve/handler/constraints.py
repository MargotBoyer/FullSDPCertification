from ast import Set
from enum import Enum
from fastsdp_tools import exists_two_similar_pairs_in_three_lists, deduct_two_lists
import logging
import numpy as np
import matplotlib.pyplot as plt
import mosek
from numba.typed import Dict
import numba
import decimal
from fastsdp_tools import infinity, deduplicate_and_sum, get_project_path

from .indexes_matrices import Indexes_Matrixes_for_Mosek_Solver
from .indexes_variables import Indexes_Variables_for_Mosek_Solver
from .variables_call import VariablesCall
from .variable_elements import (
    ElementsinConstraintsObjectives,
    add_dict_linear_to_elements,
)
import matplotlib.pyplot as plt

logger_mosek = logging.getLogger("Mosek_logger")


class ConstraintRole(Enum):
    """Rôle d'une contrainte vis-à-vis de la dynamic conic bundle method (voir
    task-dynamic-conic-bundle.md, Phase 1). HARD = toujours poussée dans le task
    MOSEK ; DUALIZABLE = éligible à être retirée du task et repliée dans l'objectif
    lagrangien (Σ alpha_r · A_r) par DynamicConicBundleSolver. Par défaut HARD :
    un run SDP classique (sans conic bundle) n'est jamais affecté par ce champ.
    """

    HARD = "hard"
    DUALIZABLE = "dualizable"


class CommonConstraints(VariablesCall):
    """
    Common functions for handling constraints in MOSEK.
    """

    def __init__(
        self,
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver,
        indexes_variables: Indexes_Variables_for_Mosek_Solver,
        **kwargs,
    ):
        """
        Initialize the CommonConstraints class.

        Parameters
        ----------
        indexes_matrices: Indexes_Matrixes_for_Mosek_Solver
            The indexes of the matrices.
        indexes_variables: Indexes_Variables_for_Mosek_Solver
            The indexes of the variables.
        """
        super().__init__(
            indexes_matrices=indexes_matrices,
            indexes_variables=indexes_variables,
            **kwargs,
        )

        self.current_num_constraint = -1
        self.list_cstr = []
        self.cstr_names = set()
        self._skipped_count = 0

    def add_constant(self, value: float):
        """
        Add a constant to the constraint.
        """
        self.list_cstr[self.current_num_constraint]["constant"] += value

    def add_var(self, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def check_current_constraint(self):
        """
        Check if the current constraint is valid.
        """
        if self.current_num_constraint == -1:
            raise ValueError("No current constraint. Please create a new one.")
        else:
            name = self.list_cstr[self.current_num_constraint]["name"]
            # print("Validating current constraint name : ", name)
            if self.list_cstr[self.current_num_constraint]["bound_type"] is None:
                raise ValueError(
                    "No bound type for the current constraint. Please set a bound type."
                )
            if self.list_cstr[self.current_num_constraint]["lb"] is None:
                raise ValueError(
                    "No lower bound for the current constraint. Please set a lower bound."
                )
            if self.list_cstr[self.current_num_constraint]["ub"] is None:
                raise ValueError(
                    "No upper bound for the current constraint. Please set an upper bound."
                )
            
            if self.list_cstr[self.current_num_constraint]["num_matrix"].size == 0:
                raise ValueError(
                    "No variable num_matrix for the current constraint. Please set a variable."
                )
            if self.list_cstr[self.current_num_constraint]["i"].size == 0:
                raise ValueError(
                    "No variable i for the current constraint. Please set a variable."
                )
            if self.list_cstr[self.current_num_constraint]["j"].size == 0:
                raise ValueError(
                    "No variable j for the current constraint. Please set a variable."
                )
            exists, num_matrix, i, j = exists_two_similar_pairs_in_three_lists(
                self.list_cstr[self.current_num_constraint]["num_matrix"],
                self.list_cstr[self.current_num_constraint]["i"],
                self.list_cstr[self.current_num_constraint]["j"],
            )
            if exists:
                raise ValueError(
                    f"Two similar pairs in the current constraint {name}: with the matrix n°{num_matrix} and indexes i={i} and j={j} \n \n {self.list_cstr[self.current_num_constraint]}"
                )
            diff = deduct_two_lists(
                self.list_cstr[self.current_num_constraint]["j"],
                self.list_cstr[self.current_num_constraint]["i"],
            )
            if any(el > 0 for el in diff):
                index = next(
                    (ind for ind, el in enumerate(diff) if el > 0),
                    None,
                )
                i = self.list_cstr[self.current_num_constraint]["i"][index]
                j = self.list_cstr[self.current_num_constraint]["j"][index]
                raise ValueError(
                    f"Indexes i and j are not sorted in the current constraint {name} : i = {i} and j = {j} \n"
                )

            logger_mosek.info(
                f"Current constraint {self.current_num_constraint} is valid."
            )
            if any(
                el == 0 for el in self.list_cstr[self.current_num_constraint]["value"]
            ):
                raise ValueError(
                    f"Zero value in the current constraint {name} : {self.list_cstr[self.current_num_constraint]}"
                )
            if not (
                (
                    self.list_cstr[self.current_num_constraint]["i"].size
                    == self.list_cstr[self.current_num_constraint]["j"].size
                )
                and (
                    self.list_cstr[self.current_num_constraint]["num_matrix"].size
                    == self.list_cstr[self.current_num_constraint]["value"].size
                )
                and (
                    self.list_cstr[self.current_num_constraint]["i"].size
                    == self.list_cstr[self.current_num_constraint]["value"].size
                )
            ):
                raise ValueError(
                    f"Size mismatch in the current constraint {name} : {self.list_cstr[self.current_num_constraint]}"
                )

    def print_current_constraint(self):
        """
        Print the current constraint.
        """
        if self.current_num_constraint == -1:
            raise ValueError("No current constraint. Please create a new one.")
        else:
            name = self.list_cstr[self.current_num_constraint]["name"]
            elements = self.list_cstr[self.current_num_constraint]["elements"]
            ub = self.list_cstr[self.current_num_constraint]["ub"]
            lb = self.list_cstr[self.current_num_constraint]["lb"]
            bound_type = self.list_cstr[self.current_num_constraint]["bound_type"]
            # print(
            #     f"Current constraint {name} : {elements};     bound : {bound_type}; ub : {ub}; lb : {lb}\n"
            # )

    def add_bound(self, bound_type: mosek.boundkey, bound: float):
        """
        Add a bound to the constraint.

        Parameters
        ----------
        bound_type: str
            The type of the bound (lower or upper).
        **kwargs: Dict
            The keyword arguments for the bound : lb, ub
        """
        self.list_cstr[self.current_num_constraint]["bound_type"] = bound_type
        if bound_type == mosek.boundkey.fx:
            self.list_cstr[self.current_num_constraint]["lb"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
            self.list_cstr[self.current_num_constraint]["ub"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
        elif bound_type == mosek.boundkey.up:
            self.list_cstr[self.current_num_constraint]["lb"] = -infinity
            self.list_cstr[self.current_num_constraint]["ub"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
        elif bound_type == mosek.boundkey.lo:
            self.list_cstr[self.current_num_constraint]["lb"] = (
                bound - self.list_cstr[self.current_num_constraint]["constant"]
            )
            self.list_cstr[self.current_num_constraint]["ub"] = infinity
        self.formate_cstr()

    def formate_cstr(self):
        """
        Format the constraint to be added to the task : adds values of parameters for the same variables.
        """
        # if self.verbose:
        #     print("STUDY : Formatting constraint", self.current_num_constraint)

        i, j, num_matrix, val = self.list_cstr[self.current_num_constraint][
            "elements"
        ].decode_key_vec()

        # Après agrégation (decode_key_vec somme les contributions par clé (i,j,num_matrix)),
        # une entrée peut légitimement retomber exactement à zéro : deux contributions
        # indépendantes à la même position qui s'annulent numériquement (observé par ex.
        # dans z_j2_beta_j2_less_than_zj sur un petit réseau — cf. task-dynamic-conic-bundle.md,
        # investigation "Bug B"). Un coefficient nul est mathématiquement inoffensif (le terme
        # ne contribue simplement pas), donc on le filtre ici plutôt que de laisser
        # check_current_constraint() lever une erreur sur un cas parfaitement valide.
        nonzero = val != 0
        if not nonzero.all():
            i, j, num_matrix, val = i[nonzero], j[nonzero], num_matrix[nonzero], val[nonzero]

        self.list_cstr[self.current_num_constraint]["i"] = i
        self.list_cstr[self.current_num_constraint]["j"] = j
        self.list_cstr[self.current_num_constraint]["num_matrix"] = num_matrix
        self.list_cstr[self.current_num_constraint]["value"] = val

        name = self.list_cstr[self.current_num_constraint]["name"]

        if self.verbose:
            pass

        # print(f"Formatted constraint {name} ")
        # print("i : ", i)
        # print("j : ", j)
        # print("num matrix : ", num_matrix)
        # print("value : ", val)
      
            
        # i_ = np.array(self.list_cstr[self.current_num_constraint]["i"])
        # j_ = np.array(self.list_cstr[self.current_num_constraint]["j"])
        # num_matrix = np.array(self.list_cstr[self.current_num_constraint]["num_matrix"])
        # val = np.array(self.list_cstr[self.current_num_constraint]["value"])

        # i = np.where(i_ > j_, i_, j_)
        # j = np.where(i_ > j_, j_, i_)
        # keys = np.stack((i, j, num_matrix), axis=-1)
        # unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        # values = np.bincount(inverse, weights=val)
        # i, j, num_matrix = unique_keys.T

        # self.list_cstr[self.current_num_constraint]["i"] = i
        # self.list_cstr[self.current_num_constraint]["j"] = j
        # self.list_cstr[self.current_num_constraint]["num_matrix"] = num_matrix
        # self.list_cstr[self.current_num_constraint]["value"] = values
        # print(
        #     f"Formatted constraint {self.current_num_constraint} : {self.list_cstr[self.current_num_constraint]}"
        # )

    def __str__(self):
        """
        String representation of the Constraints class.
        """
        line = "Constraints : \n"
        for i in range(len(self.list_cstr)):
            line += f"Constraint {i} : {self.list_cstr[i]}\n \n"

        return line

    def new_constraint(self, name: str, label: str = "to_change"):
        """
        Create a new constraint.
        """
        if self.current_num_constraint != -1:
            self.check_current_constraint()

        assert label in ["to_change", "same_for_data"], (
            "Label must be either 'to_change' or 'same_for_data'. "
            f"Got {label} instead."
        )

        # print("Creating new constraint : ", name)
        logger_mosek.info("Creating new constraint")
        if name in self.cstr_names:
            existing = next(c for c in self.list_cstr if c["name"] == name)
            if existing["label"] != "same_for_data":
                raise ValueError(
                    f"Duplicate constraint name '{name}' for a 'to_change' constraint — "
                    f"likely a naming bug in the constraint loop."
                )
            logger_mosek.debug(
                f"CONSTRAINT CALLBACK : Constraint '{name}' already exists (same_for_data). Skipping."
            )
            self._skipped_count += 1
            return True
        # else :
        #     print(f"CALLBACK Adding constraint {name}.")

        self.current_num_constraint += 1
        self.cstr_names.add(name)

        self.list_cstr.append(
            {
                "name": name,
                "elements": ElementsinConstraintsObjectives(
                    self.indexes_variables.max_index,
                ),
                "constant": 0.0,
                "lb": None,
                "ub": None,
                "bound_type": None,
                "dual_value": None,
                "label": label,
                "is_quadratic": False,
                "not_in_miqcr": False,
                "role": ConstraintRole.HARD,
            }
        )
        # print(f"Creating new constraint {self.current_num_constraint} : {name}")
        return False

    def mark_current_not_in_miqcr(self):
        """Marque la contrainte courante comme redondante pour MIQCR (déjà générée en interne)."""
        self.list_cstr[self.current_num_constraint]["not_in_miqcr"] = True

    def mark_current_dualizable(self):
        """Marque la contrainte courante comme éligible à la dualisation par la
        dynamic conic bundle method (cf. src/dynamic_conic_bundle/). N'a aucun effet
        sur un run SDP classique : la contrainte reste poussée dans le task MOSEK
        par add_to_task() tant qu'aucun DynamicConicBundleSolver ne la désactive
        explicitement via resolve_dualized().
        """
        self.list_cstr[self.current_num_constraint]["role"] = ConstraintRole.DUALIZABLE

    def first_term_equal_zero(self, num_matrices):
        """
        Set the first term of the constraint to zero.

        Parameters
        ----------
        num_matrices: int
            The number of matrices.
        """
        logger_mosek.info("Setting the first term of the matrix to zero")

        for num_matrix in range(num_matrices):
            name_matrix = self.indexes_matrices.get_name_matrix(num_matrix)
            if self.new_constraint(
                f"First term equal to zero of matrix {name_matrix}",
                label="same_for_data",
            ):
                continue
            self.mark_current_not_in_miqcr()
            # print(f"Adding constraint matrix {name_matrix}[1]=0")
            self.list_cstr[self.current_num_constraint]["elements"].add(
                i=0, j=0, num_matrix=num_matrix, value=1.0
            )
            self.add_bound(
                bound_type=mosek.boundkey.fx,
                bound=1.0,
            )

    def end_constraints(self):
        logger_mosek.info("Ending constraints")
        self.current_num_constraint += 1
        print(
            "CALLBACK : Ending constraints. Total number of constraints : ",
            len(self.list_cstr),
        )

    def get_histogram_of_coefficients(self):
        print("Getting histogram of coefficients...")
        histogram_coeff = {}
        min_coeff = infinity
        max_coeff = -infinity
        sum_coeff = 0.0

        histogram_bound = {}
        min_bound = infinity
        max_bound = -infinity
        sum_bound = 0

        close_to_zero_total_coeff = infinity
        close_to_zero_total_bound = infinity

        comparaison_by_constraints = []

        for cstr in self.list_cstr:

            if cstr["bound_type"] is not None:
                if cstr["bound_type"] == mosek.boundkey.fx:
                    bound = cstr["lb"]
                elif cstr["bound_type"] == mosek.boundkey.up:
                    bound = cstr["ub"]
                elif cstr["bound_type"] == mosek.boundkey.lo:
                    bound = cstr["lb"]
                if bound < min_bound:
                    min_bound = bound
                if bound > max_bound:
                    max_bound = bound
                if abs(bound) < close_to_zero_total_bound and abs(bound) > 1e-25:
                    close_to_zero_total_bound = abs(bound)
                sum_bound += bound
                if bound in histogram_bound:
                    histogram_bound[bound] += 1
                else:
                    histogram_bound[bound] = 1

            greater_coeff = 0
            smaller_coef = infinity
            for value in cstr["value"]:
                if value < min_coeff:
                    min_coeff = value
                if value > max_coeff:
                    max_coeff = value
                if abs(value) < close_to_zero_total_coeff and abs(value) > 1e-25:
                    close_to_zero_total_coeff = abs(value)
                if abs(value) < smaller_coef and abs(value) > 1e-25:
                    smaller_coef = abs(value)
                if abs(value) > greater_coeff:
                    greater_coeff = abs(value)
                sum_coeff += value
                if value in histogram_coeff:
                    histogram_coeff[value] += 1
                else:
                    histogram_coeff[value] = 1

            comparaison_by_constraints.append(
                {
                    "greater_coeff": greater_coeff,
                    "smaller_coef": smaller_coef,
                    "bound": abs(bound) if cstr["bound_type"] is not None else None,
                }
            )

        return (
            histogram_coeff,
            min_coeff,
            max_coeff,
            sum_coeff
            / sum(len(self.list_cstr[i]["value"]) for i in range(len(self.list_cstr))),
            close_to_zero_total_coeff,
            histogram_bound,
            min_bound,
            max_bound,
            sum_bound / len(self.list_cstr),
            close_to_zero_total_bound,
            comparaison_by_constraints,
        )


    def get_histogram_of_coefficients_name_constraint(self, name_constraint : str = "ReLU Relaxed"):
        print("Getting histogram of coefficients...")
        self.coefficient_values = {k : [] for k in range(self.K+1)}
        decimals_list = []
        equals_zero = 0
        
        for cstr in self.list_cstr:
            if "ReLU Relaxed" in cstr["name"]:
             
                for k in range(self.K+1):
                    if f"Layer {k}" in cstr["name"]:
                        self.coefficient_values[k].extend([float(val) for val in cstr["value"]])
                        for val in cstr["value"]:
                           
                           
                            d = decimal.Decimal(str(val))
                            if '.' in str(d):
                                nb_decimals = len(str(d).split('.')[1])
                                decimals_list.append(nb_decimals)

                            if abs(val) < 1e-6:

                                equals_zero += 1
                        break

    def reinitialize(self, verbose: bool):
        """
        Reinitialize the constraints.
        """
        self.verbose = verbose
        logger_mosek.info("Reinitializing constraints")
        same_for_data = 0
        to_change = 0
        for cst in self.list_cstr:
            if cst["label"] == "same_for_data":
                same_for_data += 1
            elif cst["label"] == "to_change":
                to_change += 1
            else:
                if verbose:
                    print(
                        "CALLBACK CONSTRAINT : a constraint with label : ", cst["label"]
                    )
        if verbose:
            print(
                f"CALLBACK CONSTRAINT  BEFORE FILTERING : same_for_data = {same_for_data};  to_change = {to_change}"
            )
        self.list_cstr = list(
            filter(lambda d: d["label"] == "same_for_data", self.list_cstr)
        )
        self.cstr_names = set(d["name"] for d in self.list_cstr if d["name"])

        # self.list_cstr = []
        if len(self.list_cstr) > 0:
            nb_relus = 0
            nb_rlt = 0
            others = 0
            for cst in self.list_cstr:
                if "ReLU" in cst["name"]:
                    nb_relus += 1
                elif "McCormick" in cst["name"]:
                    nb_rlt += 1
                else:
                    others += 1
            if verbose:
                print(
                    f"CALLBACK CONSTRAINT after filter : nb_relus = {nb_relus},  nb_rlt = {nb_rlt}, others : {others}"
                )

        if verbose:
            print(
                f"CALLBACK : Nombre de contraintes apres filter : ", len(self.list_cstr)
            )
        # print("CALLBACK constraints after filter : ", [cst["name"] for cst in self.list_cstr])
        # print("CALLBACK cstr_names after filter : ", self.cstr_names)

        self.current_num_constraint = len(self.list_cstr) - 1
