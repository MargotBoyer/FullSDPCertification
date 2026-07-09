"""
Extraction des matrices MIQCR depuis un handler Mosek après construction du modèle.

Flux attendu :
    solver = TargetedSDP(...)          # construit le Solver
    solver.handler.initiate_env()
    solver.add_objective()
    solver.add_constraints(cuts)
    solver.handler.Constraints.end_constraints()

    data = extract_miqcr_data(solver.handler)
    result = run_miqcr_sdp_phase(data)   # → betas
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import mosek


# ---------------------------------------------------------------------------
# Structures de données publiques
# ---------------------------------------------------------------------------

@dataclass
class MiqcrData:
    """Paramètres du MIQP continu à passer à la librairie C MIQCR."""
    n: int                    # nombre de variables
    nb_int: int               # entières (0 = problème continu)
    m: int                    # nb contraintes linéaires égalité
    p: int                    # nb contraintes linéaires inégalité
    mq: int                   # nb contraintes quadratiques égalité
    pq: int                   # nb contraintes quadratiques inégalité
    Q: np.ndarray             # (n, n) matrice hessienne de l'objectif (symétrique)
    c: np.ndarray             # (n,)   partie linéaire de l'objectif
    u: np.ndarray             # (n,)   bornes supérieures
    l: np.ndarray             # (n,)   bornes inférieures
    A: np.ndarray             # (m, n) égalités linéaires  A x = b
    b: np.ndarray             # (m,)
    D: np.ndarray             # (p, n) inégalités linéaires D x ≤ e
    e: np.ndarray             # (p,)
    Aq: np.ndarray            # (mq, n+1, n+1) quadratiques x^T Aq_k x = bq_k (homogénéisé)
    bq: np.ndarray            # (mq,)
    Dq: np.ndarray            # (pq, n+1, n+1) quadratiques x^T Dq_k x ≤ eq_k
    eq: np.ndarray            # (pq,)
    cons: float = 0.0         # constante de l'objectif
    flat_order: list = field(default_factory=list)  # (layer, neuron) → flat_idx


@dataclass
class MiqcrResult:
    """Résultat retourné par la phase SDP de MIQCR après convergence Conic Bundle."""
    beta: np.ndarray           # (n, n) multiplicateurs finaux β_{ij}
    alphaq: np.ndarray         # (mq,)  duaux des contraintes Aq
    alphabisq: np.ndarray      # (pq,)  duaux des contraintes Dq
    sol_sdp: float             # valeur objectif SDP à convergence


# ---------------------------------------------------------------------------
# Construction de la bijection (num_matrix, local_idx) ↔ flat_idx
# ---------------------------------------------------------------------------

def _build_var_map(indexes) -> Tuple[List, Dict, np.ndarray, np.ndarray]:
    """
    Construit l'ordre canonique plat des variables SDP et la table de correspondance.

    Retourne
    --------
    flat_order : list of (layer, neuron)
        Liste ordonnée des variables actives (couche 0 → K, neurones actifs).
    var_map : dict  (num_matrix: int, local_idx: int) → flat_idx
        Pour chaque représentation matrice/indice, l'index plat MIQCR.
    u_flat, l_flat : np.ndarray  shape (n_vars,)
        Bornes dans l'ordre plat.
    """
    K = indexes.K
    n = indexes.n

    # --- Ordre canonique : couches 0..K, neurones actifs seulement ---
    flat_order = []
    last_layer = K if indexes.LAST_LAYER else K - 1

    for layer in range(last_layer + 1):
        for j in range(n[layer]):
            # Exclure neurones inactifs stables
            if (layer, j) in indexes.stable_inactives_neurons:
                continue
            # Exclure neurones actifs stables (sauf avant-dernière couche gardée)
            if (layer, j) in indexes.stable_actives_neurons:
                if not (indexes.keep_penultimate_actives and layer == K - 1):
                    continue
            # Couche d'entrée : exclure les neurones prunés
            if layer == 0 and hasattr(indexes, 'pruned_input_neurons'):
                if j in indexes.pruned_input_neurons:
                    continue
            # Couche de sortie (LAST_LAYER) : garder seulement ytrue et ytargets
            if layer == K and indexes.LAST_LAYER:
                if j != indexes.ytrue and j not in indexes.ytargets:
                    continue
            flat_order.append((layer, j))

    n_vars = len(flat_order)
    flat_idx_of = {(layer, j): fi for fi, (layer, j) in enumerate(flat_order)}

    # --- Table (num_matrix, local_idx) → flat_idx ---
    var_map: Dict[Tuple[int, int], int] = {}

    for (layer, j), fi in flat_idx_of.items():
        # Une variable peut avoir jusqu'à 2 représentations (front/back)
        for fom in (True, False):
            try:
                mat = indexes.index_matrix_z(layer, front_of_matrix=fom)
                loc = indexes.index_variable_z(layer, j, front_of_matrix=fom)
                var_map[(mat, loc)] = fi
            except (ValueError, AssertionError):
                pass

    # --- Bornes ---
    # L et U sont indexés par (layer)[neuron] dans le handler
    # On les récupère via indexes (pas directement disponibles ici)
    # → ces tableaux sont remplis dans extract_miqcr_data()
    return flat_order, flat_idx_of, var_map, n_vars


# ---------------------------------------------------------------------------
# Conversion d'une contrainte Mosek → entrée MIQCR
# ---------------------------------------------------------------------------

def _is_linear(i_arr) -> bool:
    """Vraie si tous les termes quadratiques ont i=0 (termes linéaires uniquement)."""
    return all(int(x) == 0 for x in i_arr)


def _add_quad_entry(mat, flat_i: int, flat_j: int, val: float, sign: float = 1.0):
    """Ajoute val (avec signe) à la position (flat_i, flat_j) du tenseur MIQCR 3D."""
    fi, fj = flat_i + 1, flat_j + 1  # +1 pour l'index homogénéisé (0 = constante)
    if flat_i == flat_j:
        mat[fi, fj] += sign * val
    else:
        mat[fi, fj] += sign * val / 2
        mat[fj, fi] += sign * val / 2


def _add_lin_entry(mat, flat_j: int, val: float, sign: float = 1.0):
    """Ajoute un terme linéaire dans la matrice homogénéisée (ligne/col 0)."""
    fj = flat_j + 1
    mat[0, fj] += sign * val / 2
    mat[fj, 0] += sign * val / 2


def _build_cstr_matrix(
    n_vars: int,
    nm_arr, i_arr, j_arr, v_arr,
    constant: float,
    var_map: Dict,
    sign: float = 1.0,
) -> np.ndarray:
    """
    Construit la matrice (n+1)×(n+1) d'une contrainte quadratique MIQCR.

    sign = +1 pour ≤ / = ,  sign = -1 pour ≥ (négatif → flip en ≤ -lb)
    """
    M = np.zeros((n_vars + 1, n_vars + 1))
    M[0, 0] = sign * constant

    for k in range(len(nm_arr)):
        nm = int(nm_arr[k])
        i  = int(i_arr[k])
        j  = int(j_arr[k])
        v  = float(v_arr[k])

        if i == 0:
            # Terme linéaire : X[nm][0, j] = z_j
            fj = var_map.get((nm, j))
            if fj is None:
                continue
            _add_lin_entry(M, fj, v, sign)
        else:
            # Terme quadratique : X[nm][i, j] = z_i * z_j
            fi = var_map.get((nm, i))
            fj = var_map.get((nm, j))
            if fi is None or fj is None:
                continue
            _add_quad_entry(M, fi, fj, v, sign)

    return M


def _build_lin_row(
    n_vars: int,
    nm_arr, i_arr, j_arr, v_arr,
    var_map: Dict,
    sign: float = 1.0,
) -> np.ndarray:
    """Construit une ligne (n,) pour une contrainte linéaire."""
    row = np.zeros(n_vars)
    for k in range(len(nm_arr)):
        nm = int(nm_arr[k])
        j  = int(j_arr[k])   # i est toujours 0 ici
        v  = float(v_arr[k])
        fj = var_map.get((nm, j))
        if fj is not None:
            row[fj] += sign * v
    return row


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def extract_miqcr_data(handler) -> MiqcrData:
    """
    Extrait les matrices MIQCR depuis un handler Mosek après construction du modèle.

    Pré-requis :
        handler.Objective a été rempli via add_objective()
        handler.Constraints.list_cstr a été rempli via add_constraints()
        handler.Constraints.end_constraints() a été appelé

    Paramètres
    ----------
    handler : MosekClassicHandler (ou compatible)
        Le handler dont on extrait le modèle.

    Retourne
    --------
    MiqcrData avec toutes les matrices prêtes à passer au C.
    """
    indexes = handler.indexes_matrices
    flat_order, flat_idx_of, var_map, n_vars = _build_var_map(indexes)

    # --- Bornes u, l ---
    u_flat = np.zeros(n_vars)
    l_flat = np.zeros(n_vars)
    L = handler.Constraints.L
    U = handler.Constraints.U
    for (layer, j), fi in flat_idx_of.items():
        u_flat[fi] = float(U[layer][j])
        l_flat[fi] = float(L[layer][j])

    # --- Objectif : Q (n×n) et c (n,) ---
    handler.Objective.format_obj()
    Q = np.zeros((n_vars, n_vars))
    c = np.zeros(n_vars)
    cons_obj = float(handler.Objective.constant)

    nm_obj = handler.Objective.num_matrix
    i_obj  = handler.Objective.i
    j_obj  = handler.Objective.j
    v_obj  = handler.Objective.value

    for k in range(len(nm_obj)):
        nm = int(nm_obj[k])
        i  = int(i_obj[k])
        j  = int(j_obj[k])
        v  = float(v_obj[k])

        if i == 0:
            fj = var_map.get((nm, j))
            if fj is not None:
                c[fj] += v
        else:
            fi = var_map.get((nm, i))
            fj = var_map.get((nm, j))
            if fi is not None and fj is not None:
                if fi == fj:
                    Q[fi, fj] += v
                else:
                    Q[fi, fj] += v / 2
                    Q[fj, fi] += v / 2

    # --- Contraintes ---
    lin_eq_rows, lin_eq_rhs   = [], []
    lin_ineq_rows, lin_ineq_rhs = [], []
    quad_eq_mats, quad_eq_rhs  = [], []
    quad_ineq_mats, quad_ineq_rhs = [], []

    for cstr in handler.Constraints.list_cstr:
        nm_arr = cstr["num_matrix"]
        i_arr  = cstr["i"]
        j_arr  = cstr["j"]
        v_arr  = cstr["value"]
        constant   = float(cstr["constant"])
        bound_type = cstr["bound_type"]
        lb = float(cstr["lb"]) if cstr["lb"] is not None else None
        ub = float(cstr["ub"]) if cstr["ub"] is not None else None

        linear = _is_linear(i_arr)

        if linear:
            row = _build_lin_row(n_vars, nm_arr, i_arr, j_arr, v_arr, var_map)

            if bound_type == mosek.boundkey.fx:
                lin_eq_rows.append(row)
                lin_eq_rhs.append(lb - constant)

            elif bound_type == mosek.boundkey.up:
                lin_ineq_rows.append(row)
                lin_ineq_rhs.append(ub - constant)

            elif bound_type == mosek.boundkey.lo:
                # sum ≥ lb → -sum ≤ -(lb - constant)
                lin_ineq_rows.append(-row)
                lin_ineq_rhs.append(constant - lb)

            elif bound_type == mosek.boundkey.ra:
                # lb ≤ sum ≤ ub → deux inégalités
                lin_ineq_rows.append(row)
                lin_ineq_rhs.append(ub - constant)
                lin_ineq_rows.append(-row)
                lin_ineq_rhs.append(constant - lb)

        else:
            if bound_type == mosek.boundkey.fx:
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr,
                                        constant, var_map, sign=1.0)
                quad_eq_mats.append(M)
                quad_eq_rhs.append(lb)

            elif bound_type == mosek.boundkey.up:
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr,
                                        constant, var_map, sign=1.0)
                quad_ineq_mats.append(M)
                quad_ineq_rhs.append(ub)

            elif bound_type == mosek.boundkey.lo:
                # sum ≥ lb → -sum ≤ -lb  (on négative la matrice et le constant)
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr,
                                        constant, var_map, sign=-1.0)
                quad_ineq_mats.append(M)
                quad_ineq_rhs.append(-lb)

            elif bound_type == mosek.boundkey.ra:
                M_up = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr,
                                           constant, var_map, sign=1.0)
                quad_ineq_mats.append(M_up)
                quad_ineq_rhs.append(ub)
                M_lo = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr,
                                           constant, var_map, sign=-1.0)
                quad_ineq_mats.append(M_lo)
                quad_ineq_rhs.append(-lb)

    # --- Assemblage final ---
    m  = len(lin_eq_rows)
    p  = len(lin_ineq_rows)
    mq = len(quad_eq_mats)
    pq = len(quad_ineq_mats)

    A = np.array(lin_eq_rows,   dtype=np.float64) if m  > 0 else np.zeros((0, n_vars))
    b = np.array(lin_eq_rhs,    dtype=np.float64) if m  > 0 else np.zeros(0)
    D = np.array(lin_ineq_rows, dtype=np.float64) if p  > 0 else np.zeros((0, n_vars))
    e = np.array(lin_ineq_rhs,  dtype=np.float64) if p  > 0 else np.zeros(0)

    Aq = np.stack(quad_eq_mats,   axis=0) if mq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    bq = np.array(quad_eq_rhs,    dtype=np.float64) if mq > 0 else np.zeros(0)
    Dq = np.stack(quad_ineq_mats, axis=0) if pq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    eq = np.array(quad_ineq_rhs,  dtype=np.float64) if pq > 0 else np.zeros(0)

    return MiqcrData(
        n=n_vars,
        nb_int=0,
        m=m, p=p, mq=mq, pq=pq,
        Q=Q, c=c,
        u=u_flat, l=l_flat,
        A=A, b=b,
        D=D, e=e,
        Aq=Aq, bq=bq,
        Dq=Dq, eq=eq,
        cons=cons_obj,
        flat_order=flat_order,
    )
