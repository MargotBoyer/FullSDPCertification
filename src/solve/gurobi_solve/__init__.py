import os
import logging
from .gurobi_generic_solver import GurobiSolver
from .quadmodels.TargetedQuad import TargetedQuad
from .quadmodels.UntargetedQuad import UntargetedQuad
from .quadmodels.Mzbar_quad import MzbarQuad
from .lpmodels.LP_attack import ClassicLP
from .lpmodels.LP_layer_bound import LPBoundLayer
from fastsdp_tools.utils import get_project_path


logger_gurobi = logging.getLogger("Gurobi_logger")
logger_gurobi.setLevel(logging.DEBUG)
handler = logging.FileHandler(get_project_path("results/Gurobi_logger.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger_gurobi.addHandler(handler)
logger_gurobi.disabled = True


__all__ = ["GurobiSolver", "TargetedQuad", "UntargetedQuad", "MzbarQuad", "ClassicLP", "LPBoundLayer"]
