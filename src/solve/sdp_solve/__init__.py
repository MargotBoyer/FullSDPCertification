import logging
from .sdp_generic_solver import *
from .get_variables import *
from .SDPmodels.Targeted_SDP import TargetedSDP
from .SDPmodels.Untargeted_SDP import UntargetedSDP
from .SDPmodels.Mzbar import MzbarSDP
from .SDPmodels.SDP_attack import SDP_attack
import os
from fastsdp_tools.utils import get_project_path
from .handler.variables_call import LayersValues
from .run_benchmark import concat_dataframes_with_missing_columns


logger_mosek = logging.getLogger("Mosek_logger")
# WARNING (pas DEBUG) : logger_mosek.info()/.debug() sont appelés ~44500 fois chacun
# pendant la construction des contraintes (une fois par contrainte, cf. new_constraint/
# check_current_constraint dans handler/constraints.py) -- FileHandler.emit() flush le
# disque à CHAQUE appel (comportement par défaut de StreamHandler), ce qui mesurait
# ~1.4s de temps propre (LogRecord + flush) sur le prétraitement mnist-9x100 (cf.
# investigation temps de processing hors résolution SDP). Les messages INFO/DEBUG
# perdus n'apportaient pas de diagnostic utile (aucun contexte dans le message) ; les
# warnings/erreurs restent journalisés normalement.
logger_mosek.setLevel(logging.WARNING)
logger_mosek.propagate = False
handler = logging.FileHandler(get_project_path("results/Mosek_logger.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger_mosek.addHandler(handler)


__all__ = ["SDPSolver", "TargetedSDP", "UntargetedSDP", "MzbarSDP", "LayersValues", "SDP_attack", "concat_dataframes_with_missing_columns"]
