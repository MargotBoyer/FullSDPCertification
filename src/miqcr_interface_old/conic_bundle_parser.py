import os
import sys
from pydantic import ValidationError
import yaml

from fastsdp_tools.yaml_config import FullCertificationConfig
from fastsdp_tools.utils import get_project_path
from .constraint import LinearEquality, LinearInequality, QuadEquality, QuadInequality
from .objective import Objective
from networks import ReLUNN
from typing import List

from solve.generic_solver import CertificationProblemOneData


class ConicBundleParser(CertificationProblemOneData):
    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)
        print("kwargs dans conic bundle : ", kwargs)
        self.infos = kwargs
        self.which_McCormick = kwargs.get("McCormick", "none")
        print("Stable inactives neurons : ", self.stable_inactives_neurons)
        print("network : ", self.network)

    def create_model(self, **kwargs):
        kwargs = self.infos
        self.conic_bundle = None
        self.create_file()

        self.total_number_variables = 0
        self.integer_number_variables = 0

        self.Linear_equality = LinearEquality(
            file=self.file,
            K=self.K,
            n=self.n,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )
        self.Linear_inequality = LinearInequality(
            file=self.file,
            K=self.K,
            n=self.n,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )
        self.Quad_equality = QuadEquality(
            file=self.file,
            K=self.K,
            n=self.n,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )
        self.Quad_inequality = QuadInequality(
            file=self.file,
            K=self.K,
            n=self.n,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )
        self.Objective = Objective(
            file=self.file,
            K=self.K,
            n=self.n,
            stable_inactives_neurons=self.stable_inactives_neurons,
            **kwargs,
        )

        self.current_index_binary = True
        self.variables = []

    @staticmethod
    def parse_yaml_conic_bundle(yaml_file):
        with open(yaml_file, "r") as f:
            raw_config = yaml.safe_load(f)

        try:
            validated_config = FullCertificationConfig(**raw_config)
        except ValidationError as e:
            print(f"Erreur de validation du fichier YAML :\n{e}")
            raise

        return dict(
            filename=validated_config.conic_solver.filename,
            McCormick=validated_config.conic_solver.McCormick,
        )

    @classmethod
    def from_yaml(cls, yaml_file, **kwargs):
        param_conic_bundle = cls.parse_yaml_conic_bundle(yaml_file)
        params = cls.parse_yaml(yaml_file)
        print("params in conic bundle : ", params)
        print("param_conic_bundle in conic bundle : ", param_conic_bundle)
        print("kwargs : ", kwargs)
        return cls(**params, **param_conic_bundle, **kwargs)

    def add_variable(self, **kwargs):
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
        if kwargs.get("binary", False) is True:
            if not self.current_index_binary:
                raise ValueError("Binary variables must be added first.")
            self.variables.append({"lb": 0, "ub": 1, "binary": True})
            self.integer_number_variables += 1
        else:
            self.current_index_binary = False
            if kwargs.get("lb", None) is None or kwargs.get("ub", None) is None:
                raise ValueError("Non binary variables must have bounds.")
            self.variables.append(
                {"lb": kwargs.get("lb"), "ub": kwargs.get("ub"), "binary": False}
            )
        self.total_number_variables += 1

    def bounds_to_file(self):
        """
        Write the bounds to the file.
        """
        self.file.write("u\n")
        line_u = ""
        for line in self.variables:
            line_u += f"{line['ub']} "
        self.file.write(line_u[:-1] + "\n")

        self.file.write("l\n")
        line_l = ""
        for line in self.variables:
            line_l += f"{line['lb']} "
        self.file.write(line_l[:-1] + "\n")

    def create_file(self):
        print("filename dans create file: ", self.filename)
        self.filename = (
            "conic_bundle_"
            + self.data_modele
            + "_"
            + self.__class__.__name__.replace("Parser", "")
        )
        if self.__class__.__name__ == "LanParser":
            self.filename = self.filename + "_target=" + str(self.ytarget)

        self.filename = self.filename + "_McCormick=" + self.which_McCormick
        self.file = open(
            get_project_path(
                ("results/conic_bundle/" + self.filename + ".txt").replace("\\", "/")
            ),
            "w",
        )

    def close_file(self):
        self.file.close()

    def initialize(self):
        self.file.write(f"{self.total_number_variables} ")
        self.file.write(f"{self.integer_number_variables} ")
        self.file.write(f"{self.Linear_equality.total_number_constr} ")
        self.file.write(f"{self.Linear_inequality.total_number_constr} ")
        self.file.write(f"{self.Quad_equality.total_number_constr} ")
        self.file.write(f"{self.Quad_inequality.total_number_constr} ")
        self.file.write("\n")

    def write_file(self):
        try:
            self.initialize()
            self.bounds_to_file()
            self.Objective.to_file()
            self.Linear_equality.to_file()
            self.Linear_inequality.to_file()
            self.Quad_equality.to_file()
            self.Quad_inequality.to_file()
        finally:
            self.close_file()

    def to_file(self):

        print("class name : ", self.__class__.__name__)
        if "Targeted" in self.__class__.__name__:
            print("Targeted parser ")
            print("self.ytargets : ", self.ytargets)
            for ytarget in self.ytargets:
                self.ytarget = ytarget
                print("ytarget dans to file : ", self.ytarget)
                self.create_model()
                self.add_model()
                self.write_file()
        else:
            print("not Lan parser ")
            self.create_model()
            self.add_model()
            self.write_file()

    def add_model(self):
        raise NotImplementedError(
            "The method add_model is not implemented in the ConicBundleParser class."
        )
