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

Hypothèse : pas de décomposition chordale (CHORDAL_DECOMPOSITION=False),
donc une seule matrice SDP (index 0). La variable à la position k dans la
matrice (1-based) a l'index plat k-1 (0-based) dans les vecteurs MIQCR.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

try:
    import mosek
except ImportError:
    mosek = None  # mosek not installed; extract_miqcr_data will not be available


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
# Conversion d'une contrainte Mosek → entrée MIQCR
#
# Les indices i, j issus du handler mosek_classic sont déjà homogénéisés
# (1-based : 0 = coordonnée homogène, 1..n = variables). Les valeurs
# off-diagonales sont déjà divisées par 2 par add_dict_quad_to_elements
# et add_dict_linear_to_elements (dividing_non_diag=True). On les place
# directement dans la matrice MIQCR sans conversion supplémentaire.
# ---------------------------------------------------------------------------

def _add_quad_entry(mat, i: int, j: int, val: float, sign: float = 1.0):
    """i, j : indices homogénéisés 1-based. val : déjà /2 pour i≠j (mosek_classic)."""
    mat[i, j] += sign * val
    if i != j:
        mat[j, i] += sign * val


def _add_lin_entry(mat, i: int, val: float, sign: float = 1.0):
    """i : indice homogénéisé 1-based. val : déjà /2 (mosek_classic)."""
    mat[0, i] += sign * val
    mat[i, 0] += sign * val


def _build_cstr_matrix(
    n_vars: int,
    nm_arr, i_arr, j_arr, v_arr,
    constant: float,
    sign: float = 1.0,
) -> np.ndarray:
    """
    Construit la matrice (n+1)×(n+1) d'une contrainte quadratique MIQCR.

    Convention de stockage dans list_cstr (mosek_classic) :
      - termes linéaires  : i = index_variable (≥1, homogénéisé), j = 0
      - termes quadratiques : i ≥ j ≥ 1 (homogénéisés), valeurs déjà /2 pour i≠j
      - entrée constante    : i = 0, j = 0  (ignorée ici)

    sign = +1 pour ≤ / = ,  sign = -1 pour ≥ (négatif → flip en ≤ -lb)
    """
    M = np.zeros((n_vars + 1, n_vars + 1))
    M[0, 0] = sign * constant

    for k in range(len(nm_arr)):
        i = int(i_arr[k])
        j = int(j_arr[k])
        v = float(v_arr[k])

        if i == 0:
            # (0, j) : constant si j=0, linéaire si j>0
            if j > 0:
                _add_lin_entry(M, j, v, sign)
        elif j == 0:
            # Terme linéaire stocké en (i, 0) : z_i * 1
            _add_lin_entry(M, i, v, sign)
        else:
            # Terme quadratique : z_i * z_j
            _add_quad_entry(M, i, j, v, sign)

    return M


def _build_lin_row(
    n_vars: int,
    nm_arr, i_arr, j_arr, v_arr,
    sign: float = 1.0,
) -> np.ndarray:
    """Construit une ligne (n,) pour une contrainte linéaire.
    Convertit les indices homogénéisés 1-based en indices 0-based du vecteur."""
    row = np.zeros(n_vars)
    for k in range(len(nm_arr)):
        i = int(i_arr[k])
        j = int(j_arr[k])
        v = float(v_arr[k])
        if j == 0 and i > 0:
            row[i - 1] += sign * v
        elif i == 0 and j > 0:
            row[j - 1] += sign * v
    return row


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def extract_miqcr_data(handler, max_quad_constraints: int = None) -> MiqcrData:
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
        Doit avoir CHORDAL_DECOMPOSITION=False (une seule matrice SDP).
    max_quad_constraints : int | None
        Nombre maximum de contraintes quadratiques (mq + pq) à passer à MIQCR.
        None → auto-limite à ~1 Go de matrices.
        0   → aucune contrainte quadratique.

    Retourne
    --------
    MiqcrData avec toutes les matrices prêtes à passer au C.
    """
    from solve.sdp_solve.handler.mosek_classic import MosekClassicHandler
    assert isinstance(handler, MosekClassicHandler), (
        "extract_miqcr_data requiert le solveur 'mosek_classic' "
        "(YAML : solver: \"mosek_classic\"). "
        f"Handler reçu : {type(handler).__name__}"
    )

    indexes = handler.indexes_matrices

    # n_vars = taille de la matrice SDP unique - 1 (coordonnée homogène)
    n_vars = indexes.get_shape_matrix(0) - 1

    # flat_order : liste ordonnée des variables actives (pour sérialisation)
    flat_order = _build_flat_order(indexes)

    # --- Estimation mémoire et calcul de la limite ---
    list_cstr = handler.Constraints.list_cstr
    n_quad_total = sum(1 for c in list_cstr if c["is_quadratic"])
    bytes_per_mat = (n_vars + 1) ** 2 * 8
    print("bytes_per_mat =", bytes_per_mat)
    mem_est_gb = n_quad_total * bytes_per_mat / 1e9
    print("max_quad_constraints =", max_quad_constraints)

    print(f"[extract_miqcr_data] n_vars={n_vars} | {len(list_cstr)} contraintes "
          f"({n_quad_total} quadratiques)")
    print(f"[extract_miqcr_data] Mémoire dense estimée si toutes quadratiques : "
          f"{mem_est_gb:.1f} Go")

    if max_quad_constraints is None:
        budget_bytes = 1e9  # 1 Go
        print(" int(budget_bytes / bytes_per_mat)", int(budget_bytes / bytes_per_mat))
        max_quad_constraints = max(0, int(budget_bytes / bytes_per_mat))

        if max_quad_constraints < n_quad_total:
            print(f"[extract_miqcr_data] Limite auto à {max_quad_constraints} contraintes "
                  f"quadratiques (budget 1 Go). Passer max_quad_constraints= pour changer.")

    if max_quad_constraints < n_quad_total:
        print(f"[extract_miqcr_data] ATTENTION : {n_quad_total - max_quad_constraints} "
              f"contraintes quadratiques ignorées.")

    print("[extract_miqcr_data] Construction des matrices MIQCR... avec max_quad_constraints =", max_quad_constraints)

    def _arr_info(arr):
        if arr.size == 0:
            return "(vide)"
        nnz = int(np.count_nonzero(arr))
        return (f"shape={arr.shape}  nnz={nnz}/{arr.size}"
                f"  min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")

    # --- Bornes u, l ---
    # La variable à la position k dans la matrice (1-based) a l'index plat k-1.
    # On itère les variables actives et on utilise index_variable_z pour obtenir k.
    u_flat = np.zeros(n_vars)
    l_flat = np.zeros(n_vars)
    L = handler.Constraints.L
    U = handler.Constraints.U
    _fill_bounds(indexes, u_flat, l_flat, U, L)

    print(f"[MIQCR] u  : {_arr_info(u_flat)}")
    print(f"[MIQCR] l  : {_arr_info(l_flat)}")

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
        i = int(i_obj[k])
        j = int(j_obj[k])
        v = float(v_obj[k])
        print(f"[MIQCR] Objectif : k={k}  i={i}  j={j}  v={v:.4g}")

        # i, j : indices homogénéisés 1-based (0 = constante)
        if i == 0:
            if j > 0:
                print("c[j-1] :", j - 1)
                c[j - 1] += v
        elif j == 0:
            print("c[i-1] :", i - 1)
            c[i - 1] += v
        else:
            fi, fj = i - 1, j - 1
            print("fi : ", fi, "  fj :", fj)
            if fi == fj:
                Q[fi, fj] += v
            else:
                Q[fi, fj] += v   # déjà /2 par add_dict_quad_to_elements
                Q[fj, fi] += v

    print(f"[MIQCR] Q  : {_arr_info(Q)}  (constante obj={cons_obj:.4g})")
    print("[MIQCR Q  = ", Q)
    print(f"[MIQCR] c  : {_arr_info(c)}")
    print("[MIQCR c  = ", c)

    # --- Contraintes ---
    lin_eq_rows, lin_eq_rhs   = [], []
    lin_ineq_rows, lin_ineq_rhs = [], []
    quad_eq_mats, quad_eq_rhs  = [], []
    quad_ineq_mats, quad_ineq_rhs = [], []

    n_quad_kept = 0

    for cstr in list_cstr:
        if cstr.get("not_in_miqcr", False):
            continue
        print("\n [extract_miqcr_data] = ", cstr['name'])
        nm_arr = cstr["num_matrix"]
        i_arr  = cstr["i"]
        j_arr  = cstr["j"]
        v_arr  = cstr["value"]
        constant   = float(cstr["constant"])
        bound_type = cstr["bound_type"]
        lb = float(cstr["lb"]) if cstr["lb"] is not None else None
        ub = float(cstr["ub"]) if cstr["ub"] is not None else None

        linear = not cstr["is_quadratic"]
        print(f"[extract_miqcr_data] Contrainte {'linéaire' if linear else 'quadratique'} ")

        if linear:
            row = _build_lin_row(n_vars, nm_arr, i_arr, j_arr, v_arr)
            print('[extract_miqcr_data] row : ', row)

            if bound_type == mosek.boundkey.fx:
                print('[extract_miqcr_data] Contrainte linéaire d\'égalité')
                lin_eq_rows.append(row)
                lin_eq_rhs.append(lb - constant)

            elif bound_type == mosek.boundkey.up:
                print('[extract_miqcr_data] Contrainte linéaire d\'inégalité')
                lin_ineq_rows.append(row)
                lin_ineq_rhs.append(ub - constant)

            elif bound_type == mosek.boundkey.lo:
                print('[extract_miqcr_data] Contrainte linéaire d\'inégalité (borne inférieure)')
                lin_ineq_rows.append(-row)
                lin_ineq_rhs.append(constant - lb)

            elif bound_type == mosek.boundkey.ra:
                print('[extract_miqcr_data] Contrainte linéaire double borne')
                lin_ineq_rows.append(row)
                lin_ineq_rhs.append(ub - constant)
                lin_ineq_rows.append(-row)
                lin_ineq_rhs.append(constant - lb)

        else:
            if n_quad_kept >= max_quad_constraints:
                continue
            n_quad_kept += 1

            if bound_type == mosek.boundkey.fx:
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr, constant, sign=1.0)
                quad_eq_mats.append(M)
                quad_eq_rhs.append(lb)

            elif bound_type == mosek.boundkey.up:
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr, constant, sign=1.0)
                quad_ineq_mats.append(M)
                quad_ineq_rhs.append(ub)

            elif bound_type == mosek.boundkey.lo:
                M = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr, constant, sign=-1.0)
                quad_ineq_mats.append(M)
                quad_ineq_rhs.append(-lb)

            elif bound_type == mosek.boundkey.ra:
                M_up = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr, constant, sign=1.0)
                quad_ineq_mats.append(M_up)
                quad_ineq_rhs.append(ub)
                M_lo = _build_cstr_matrix(n_vars, nm_arr, i_arr, j_arr, v_arr, constant, sign=-1.0)
                quad_ineq_mats.append(M_lo)
                quad_ineq_rhs.append(-lb)
                n_quad_kept += 1
            else:
                raise ValueError(f"Type de borne inconnu : {bound_type}")

    # --- Assemblage final ---
    m  = len(lin_eq_rows)
    p  = len(lin_ineq_rows)
    mq = len(quad_eq_mats)
    pq = len(quad_ineq_mats)

    print(f"[extract_miqcr_data] Assemblage : m={m} lin_eq | p={p} lin_ineq | "
          f"mq={mq} quad_eq | pq={pq} quad_ineq")

    A = np.array(lin_eq_rows,   dtype=np.float64) if m  > 0 else np.zeros((0, n_vars))
    b = np.array(lin_eq_rhs,    dtype=np.float64) if m  > 0 else np.zeros(0)
    D = np.array(lin_ineq_rows, dtype=np.float64) if p  > 0 else np.zeros((0, n_vars))
    e = np.array(lin_ineq_rhs,  dtype=np.float64) if p  > 0 else np.zeros(0)

    Aq = np.stack(quad_eq_mats,   axis=0) if mq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    bq = np.array(quad_eq_rhs,    dtype=np.float64) if mq > 0 else np.zeros(0)
    Dq = np.stack(quad_ineq_mats, axis=0) if pq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    eq = np.array(quad_ineq_rhs,  dtype=np.float64) if pq > 0 else np.zeros(0)

    print(f"[MIQCR] A  : {_arr_info(A)}")
    print(f"[MIQCR] b  : {_arr_info(b)}")
    print(f"[MIQCR] D  : {_arr_info(D)}")
    print(f"[MIQCR] e  : {_arr_info(e)}")
    print(f"[MIQCR] Aq : {_arr_info(Aq)}")
    print(f"[MIQCR] bq : {_arr_info(bq)}")
    print(f"[MIQCR] Dq : {_arr_info(Dq)}")
    print(f"[MIQCR] eq : {_arr_info(eq)}")

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


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _build_flat_order(indexes) -> list:
    """Retourne la liste ordonnée (layer, neuron) des variables actives."""
    K = indexes.K
    n = indexes.n
    last_layer = K if indexes.LAST_LAYER else K - 1
    flat_order = []
    for layer in range(last_layer + 1):
        for j in range(n[layer]):
            if (layer, j) in indexes.stable_inactives_neurons:
                continue
            if (layer, j) in indexes.stable_actives_neurons:
                if not (indexes.keep_penultimate_actives and layer == K - 1):
                    continue
            if layer == 0 and hasattr(indexes, 'pruned_input_neurons'):
                if j in indexes.pruned_input_neurons:
                    continue
            if layer == K and indexes.LAST_LAYER:
                if j != indexes.ytrue and j not in indexes.ytargets:
                    continue
            flat_order.append((layer, j))
    return flat_order


def _fill_bounds(indexes, u_flat: np.ndarray, l_flat: np.ndarray, U, L) -> None:
    """Remplit les tableaux de bornes via index_variable_z."""
    K = indexes.K
    n = indexes.n
    last_layer = K if indexes.LAST_LAYER else K - 1
    for layer in range(last_layer + 1):
        for j in range(n[layer]):
            if (layer, j) in indexes.stable_inactives_neurons:
                continue
            if (layer, j) in indexes.stable_actives_neurons:
                if not (indexes.keep_penultimate_actives and layer == K - 1):
                    continue
            if layer == 0 and hasattr(indexes, 'pruned_input_neurons'):
                if j in indexes.pruned_input_neurons:
                    continue
            if layer == K and indexes.LAST_LAYER:
                if j != indexes.ytrue and j not in indexes.ytargets:
                    continue
            try:
                fi = indexes.index_variable_z(layer, j, front_of_matrix=False) - 1
                u_flat[fi] = float(U[layer][j])
                l_flat[fi] = float(L[layer][j])
            except (ValueError, AssertionError):
                pass
