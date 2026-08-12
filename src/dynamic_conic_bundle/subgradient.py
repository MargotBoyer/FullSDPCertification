"""
Calcul du sous-gradient de la fonction duale lagrangienne h(theta), vectorisé.

Pour chaque contrainte dualisée r (normalisée en <A_r, X> <= rhs_r par
MosekClassicHandler.setup_dualization, cf. handler/mosek_classic/handler_classic.py) :

    g_r = <A_r, X*> - rhs_r

où X* est la solution primale renvoyée par resolve_dualized(). g_r > 0 signifie que
la contrainte relâchée est violée par X* : le pas de sous-gradient doit augmenter
theta_r (cf. proximal_master.py). `theta` (notation main.pdf, Algorithme 2) — sans
rapport avec les `alpha` d'alpha-CROWN (calcul de bornes).

Les contraintes RLT/McCormick (McCormick_inter_layers) sont creuses (quelques
non-nuls par contrainte) : ce module exploite cette parcimonie via les triplets
(num_matrix, i, j, value) déjà stockés dans list_cstr, au lieu de matérialiser une
matrice dense (n+1)x(n+1) par contrainte comme le fait l'ancien prototype
src/miqcr_bridge/lagrangian_cb.py::_aq_inner_product.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_subgradients(
    dualized_constraints_data: List[dict],
    X: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """
    g_r = <A_r, X*> - rhs_r pour chaque contrainte dualisée.

    Parameters
    ----------
    dualized_constraints_data : List[dict]
        Sortie de SDPSolver.get_dualized_constraints_data() : un dict par contrainte
        dualisée avec les clés "name", "num_matrix", "i", "j", "value" (triplets
        creux, déjà signés/normalisés en <=), "rhs".
    X : Dict[int, np.ndarray]
        Matrices primales reconstruites, indexées par num_matrix (sortie de
        SDPSolver.resolve_dualized()["X"]).

    Returns
    -------
    Dict[str, float]
        {constraint_name: g_r}, un sous-gradient par contrainte dualisée.
    """
    if not dualized_constraints_data:
        return {}

    names = [d["name"] for d in dualized_constraints_data]
    n_cstr = len(names)

    owner_parts, num_matrix_parts, i_parts, j_parts, val_parts = [], [], [], [], []
    for r, d in enumerate(dualized_constraints_data):
        n_terms = len(d["i"])
        if n_terms == 0:
            continue
        owner_parts.append(np.full(n_terms, r, dtype=np.int64))
        num_matrix_parts.append(np.asarray(d["num_matrix"], dtype=np.int64))
        i_parts.append(np.asarray(d["i"], dtype=np.int64))
        j_parts.append(np.asarray(d["j"], dtype=np.int64))
        val_parts.append(np.asarray(d["value"], dtype=np.float64))

    rhs = np.array([d["rhs"] for d in dualized_constraints_data], dtype=np.float64)

    if not owner_parts:
        return {name: -float(rhs[r]) for r, name in enumerate(names)}

    owner = np.concatenate(owner_parts)
    num_matrix = np.concatenate(num_matrix_parts)
    ii = np.concatenate(i_parts)
    jj = np.concatenate(j_parts)
    val = np.concatenate(val_parts)

    # Convention "dividing_non_diag=True" : seules les entrées i>=j sont stockées, et
    # les hors-diagonale sont déjà divisées par 2 (cf. add_dict_quad_to_elements).
    # Tr(A_r X) = sum_{i=j} val*X[i,i] + sum_{i>j} 2*val*X[i,j].
    factor = np.where(ii == jj, 1.0, 2.0)

    x_vals = np.empty(val.shape[0], dtype=np.float64)
    for nm in np.unique(num_matrix):
        mask = num_matrix == nm
        x_vals[mask] = X[int(nm)][ii[mask], jj[mask]]

    contrib = factor * val * x_vals
    trace = np.bincount(owner, weights=contrib, minlength=n_cstr)

    g = trace - rhs
    return {name: float(g[r]) for r, name in enumerate(names)}
