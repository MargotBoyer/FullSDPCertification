"""
Interface ctypes entre Python et la librairie C MIQCR.

La shared library `libmiqcr_pyapi.so` est compilée depuis
`Miqcr-1.0_triang_sbb_gurobiter_QCP5/src/miqcr_pyapi.c`.

Flux d'appel :
    data   = extract_miqcr_data(handler)
    result = run_miqcr_sdp_phase(data)
    # result.beta[i,j] = β_{ij} après convergence Conic Bundle
"""
from __future__ import annotations

import ctypes
import os
import numpy as np

from .sdp_data import MiqcrData, MiqcrResult

# Chemin par défaut de la shared library (modifiable via variable d'env)
_DEFAULT_SO = os.environ.get(
    "MIQCR_PYAPI_SO",
    os.path.join(os.path.dirname(__file__), "libmiqcr_pyapi.so"),
)

_lib_cache: ctypes.CDLL | None = None


def _load_lib(so_path: str | None = None) -> ctypes.CDLL:
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    path = os.path.realpath(so_path or _DEFAULT_SO)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"libmiqcr_pyapi.so introuvable : {path}\n"
            "Compilez-la avec : cd Miqcr-1.0_.../src && make -f Makefile.pyapi"
        )
    _lib_cache = ctypes.CDLL(path)
    _setup_signatures(_lib_cache)
    return _lib_cache


# Type ctypes du callback SDP — doit correspondre exactement à sdp_solver_cb_t dans le C.
# Signature :
#   void cb(int n, int mq, int pq,
#           double *q_beta, double *c_beta, double l_beta,
#           double *x_out, double *beta_diag_out,
#           double *alphaq_out, double *alphabisq_out, double *sol_sdp_out)
_dbl_p = ctypes.POINTER(ctypes.c_double)
SDP_SOLVER_CB_TYPE = ctypes.CFUNCTYPE(
    None,                    # retour : void
    ctypes.c_int,            # n
    ctypes.c_int,            # mq
    ctypes.c_int,            # pq
    _dbl_p,                  # q_beta  [n*n]
    _dbl_p,                  # c_beta  [n]
    ctypes.c_double,         # l_beta
    _dbl_p,                  # x_out   [(n+1)(n+2)/2]
    _dbl_p,                  # beta_diag_out [n]
    _dbl_p,                  # alphaq_out    [mq]
    _dbl_p,                  # alphabisq_out [pq]
    _dbl_p,                  # sol_sdp_out   [1]
)

# Référence globale pour empêcher le garbage collector de libérer le callback
_registered_cb_ref = None

# Compteur d'itérations CB côté Python (incrémenté dans _c_callback à chaque appel)
_nb_iter_cb: int = 0


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Déclare les types d'arguments et de retour des fonctions C exportées."""
    dbl_p = ctypes.POINTER(ctypes.c_double)
    int_p = ctypes.POINTER(ctypes.c_int)

    # void miqcr_populate_qp(
    #     int n, int nb_int, int m, int p, int mq, int pq,
    #     double *Q,   double *c,   double *u,   double *l,
    #     double *A,   double *b,
    #     double *D,   double *e,
    #     double *Aq,  double *bq,
    #     double *Dq,  double *eq,
    #     double cons
    # )
    lib.miqcr_populate_qp.restype  = None
    lib.miqcr_populate_qp.argtypes = [
        ctypes.c_int, ctypes.c_int,        # n, nb_int
        ctypes.c_int, ctypes.c_int,        # m, p
        ctypes.c_int, ctypes.c_int,        # mq, pq
        dbl_p, dbl_p, dbl_p, dbl_p,       # Q, c, u, l
        dbl_p, dbl_p,                      # A, b
        dbl_p, dbl_p,                      # D, e
        dbl_p, dbl_p,                      # Aq, bq
        dbl_p, dbl_p,                      # Dq, eq
        ctypes.c_double,                   # cons
    ]

    # void miqcr_compute_betas(
    #     double *out_beta,       // n*n (row-major)
    #     double *out_alphaq,     // mq
    #     double *out_alphabisq,  // pq
    #     double *out_sol_sdp     // scalaire [1]
    # )
    lib.miqcr_compute_betas.restype  = None
    lib.miqcr_compute_betas.argtypes = [dbl_p, dbl_p, dbl_p, dbl_p]

    # void miqcr_set_param(const char *name, double value)
    lib.miqcr_set_param.restype  = None
    lib.miqcr_set_param.argtypes = [ctypes.c_char_p, ctypes.c_double]

    # void miqcr_register_sdp_solver(sdp_solver_cb_t cb)
    lib.miqcr_register_sdp_solver.restype  = None
    lib.miqcr_register_sdp_solver.argtypes = [SDP_SOLVER_CB_TYPE]

    # int miqcr_get_nb_iter(void)
    lib.miqcr_get_nb_iter.restype  = ctypes.c_int
    lib.miqcr_get_nb_iter.argtypes = []

    # void miqcr_get_final_x(double *out_x)
    # out_x : buffer de taille (n+1)*(n+2)/2, triangle inférieur row-major de X*
    lib.miqcr_get_final_x.restype  = None
    lib.miqcr_get_final_x.argtypes = [dbl_p]

    # double miqcr_get_true_obj(void)
    # Objectif vrai (non pénalisé) évalué sur X* via make_matrix_with_vector interne.
    lib.miqcr_get_true_obj.restype  = ctypes.c_double
    lib.miqcr_get_true_obj.argtypes = []


def _np_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    """Retourne un pointeur C vers le buffer numpy (double, C-contiguous)."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def set_param(name: str, value: float, so_path: str | None = None) -> None:
    """
    Surcharge un paramètre MIQCR (FACTOR, EPS_TERM_CB, NB_MAX_ITER, …).

    Exemples :
        set_param("FACTOR", 0.1)
        set_param("EPS_TERM_CB", 1e-3)
    """
    lib = _load_lib(so_path)
    lib.miqcr_set_param(name.encode(), ctypes.c_double(value))


def run_miqcr_sdp_phase(
    data: MiqcrData,
    so_path: str | None = None,
) -> MiqcrResult:
    """
    Passe les matrices à MIQCR et exécute la phase SDP (Conic Bundle).

    Paramètres
    ----------
    data : MiqcrData
        Paramètres du MIQP extraits via `extract_miqcr_data`.
    so_path : str | None
        Chemin optionnel vers la shared library (sinon valeur par défaut ou
        variable d'environnement MIQCR_PYAPI_SO).

    Retourne
    --------
    MiqcrResult
        beta (n×n), alphaq (mq,), alphabisq (pq,), sol_sdp (float).
    """
    lib = _load_lib(so_path)
    n, mq, pq = data.n, data.mq, data.pq

    print(f"[run_miqcr_sdp_phase] Démarrage — n={n} nb_int={data.nb_int} "
          f"m={data.m} p={data.p} mq={mq} pq={pq} cons={data.cons:.4g}")

    # Mise en forme C-contiguous (row-major) des tableaux numpy
    print("[run_miqcr_sdp_phase] Conversion des tableaux numpy en C-contiguous...")
    Q  = np.ascontiguousarray(data.Q,  dtype=np.float64)
    c  = np.ascontiguousarray(data.c,  dtype=np.float64)
    u  = np.ascontiguousarray(data.u,  dtype=np.float64)
    l  = np.ascontiguousarray(data.l,  dtype=np.float64)
    A  = np.ascontiguousarray(data.A,  dtype=np.float64) if data.m  > 0 else np.zeros((0, n))
    b  = np.ascontiguousarray(data.b,  dtype=np.float64) if data.m  > 0 else np.zeros(0)
    D  = np.ascontiguousarray(data.D,  dtype=np.float64) if data.p  > 0 else np.zeros((0, n))
    e  = np.ascontiguousarray(data.e,  dtype=np.float64) if data.p  > 0 else np.zeros(0)
    Aq = np.ascontiguousarray(data.Aq, dtype=np.float64) if mq > 0 else np.zeros((0, n + 1, n + 1))
    bq = np.ascontiguousarray(data.bq, dtype=np.float64) if mq > 0 else np.zeros(0)
    Dq = np.ascontiguousarray(data.Dq, dtype=np.float64) if pq > 0 else np.zeros((0, n + 1, n + 1))
    eq = np.ascontiguousarray(data.eq, dtype=np.float64) if pq > 0 else np.zeros(0)
    print(f"[run_miqcr_sdp_phase] Tableaux prêts : "
          f"Q{Q.shape} c{c.shape} u{u.shape} l{l.shape} "
          f"A{A.shape} D{D.shape} Aq{Aq.shape} Dq{Dq.shape}")

    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    print("[run_miqcr_sdp_phase] Appel miqcr_populate_qp (transfert vers la lib C)...")
    lib.miqcr_populate_qp(
        ctypes.c_int(n),
        ctypes.c_int(data.nb_int),
        ctypes.c_int(data.m),
        ctypes.c_int(data.p),
        ctypes.c_int(mq),
        ctypes.c_int(pq),
        ptr(Q),  ptr(c),  ptr(u),  ptr(l),
        ptr(A),  ptr(b),
        ptr(D),  ptr(e),
        ptr(Aq), ptr(bq),
        ptr(Dq), ptr(eq),
        ctypes.c_double(data.cons),
    )
    print("[run_miqcr_sdp_phase] miqcr_populate_qp terminé.")

    out_beta      = np.zeros(n * n,  dtype=np.float64)
    out_alphaq    = np.zeros(max(mq, 1), dtype=np.float64)
    out_alphabisq = np.zeros(max(pq, 1), dtype=np.float64)
    out_sol_sdp   = np.zeros(1,      dtype=np.float64)

    global _nb_iter_cb
    _nb_iter_cb = 0
    print("[run_miqcr_sdp_phase] Appel miqcr_compute_betas (Conic Bundle)...")
    lib.miqcr_compute_betas(
        ptr(out_beta),
        ptr(out_alphaq),
        ptr(out_alphabisq),
        ptr(out_sol_sdp),
    )
    # MIQCR stocke sol_sdp = -LB (convention signe inversé : objectif négativé en interne).
    # On nie pour obtenir la vraie borne inférieure effective sur le problème original.
    lb = -float(out_sol_sdp[0])
    print(f"[run_miqcr_sdp_phase] miqcr_compute_betas terminé — sol_sdp (borne effective) = {lb:.6g}")

    # Objectif vrai (non pénalisé) évalué sur X*, idem convention signe inversé.
    true_obj = -lib.miqcr_get_true_obj()
    print(f"[run_miqcr_sdp_phase] true_obj_sdp (objectif non pénalisé sur X*) = {true_obj:.6g}")

    # Priorité au compteur C-side (via --wrap=eval_fun_mixed, marche avec ou
    # sans callback Python). Fallback sur le compteur Python si C retourne 0.
    nb_iter = lib.miqcr_get_nb_iter()
    if nb_iter == 0:
        nb_iter = _nb_iter_cb
    result = MiqcrResult(
        beta=out_beta.reshape(n, n),
        alphaq=out_alphaq[:mq],
        alphabisq=out_alphabisq[:pq],
        sol_sdp=lb,
        true_obj_sdp=true_obj,
        nb_iter_cb=nb_iter,
    )
    print(f"[run_miqcr_sdp_phase] beta : max={np.abs(result.beta).max():.4g} "
          f"nz={np.count_nonzero(result.beta)}/{n*n}")
    print(f"[run_miqcr_sdp_phase] itérations CB : {nb_iter}")
    return result


def register_sdp_solver(python_solver_fn, so_path: str | None = None) -> None:
    """
    Enregistre une fonction Python comme solveur SDP pour la Conic Bundle MIQCR.

    À appeler après miqcr_populate_qp (via run_miqcr_sdp_phase) et avant
    miqcr_compute_betas. Une fois enregistrée, la fonction est appelée à chaque
    itération CB à la place de Mosek.

    Paramètres de python_solver_fn
    --------------------------------
    La fonction doit avoir la signature :

        def solve(n, mq, pq, q_beta, c_beta, l_beta,
                  x_out, beta_diag_out, alphaq_out, alphabisq_out, sol_sdp_out):
            ...

    où tous les tableaux sont des numpy arrays (vues sur la mémoire C, modifiables
    sur place) :
        q_beta        : (n*n,)           — partie quadratique de C_beta
        c_beta        : (n,)             — partie linéaire de C_beta
        l_beta        : float            — constante scalaire
        x_out         : ((n+1)(n+2)/2,) — à remplir : triangle inf. de X
        beta_diag_out : (n,)             — à remplir : duaux diagonaux
        alphaq_out    : (mq,)            — à remplir : duaux Aq
        alphabisq_out : (pq,)            — à remplir : duaux Dq
        sol_sdp_out   : (1,)             — à remplir : valeur objective

    Passer python_solver_fn=None pour revenir à Mosek.
    """
    global _registered_cb_ref
    lib = _load_lib(so_path)

    if python_solver_fn is None:
        _registered_cb_ref = None
        lib.miqcr_register_sdp_solver(SDP_SOLVER_CB_TYPE(0))
        return

    def _c_callback(n, mq, pq,
                    q_beta_p, c_beta_p, l_beta,
                    x_out_p, beta_diag_p, alphaq_p, alphabisq_p, sol_p):
        global _nb_iter_cb
        _nb_iter_cb += 1
        # Convertit les pointeurs C en vues numpy (zero-copy)
        len_x = (n + 1) * (n + 2) // 2

        q_beta        = np.frombuffer((ctypes.c_double * (n * n))    .from_address(ctypes.addressof(q_beta_p.contents)),  dtype=np.float64)
        c_beta        = np.frombuffer((ctypes.c_double * n)           .from_address(ctypes.addressof(c_beta_p.contents)),  dtype=np.float64)
        x_out         = np.frombuffer((ctypes.c_double * len_x)       .from_address(ctypes.addressof(x_out_p.contents)),   dtype=np.float64)
        beta_diag_out = np.frombuffer((ctypes.c_double * n)           .from_address(ctypes.addressof(beta_diag_p.contents)), dtype=np.float64)
        alphaq_out    = np.frombuffer((ctypes.c_double * max(mq, 1)) .from_address(ctypes.addressof(alphaq_p.contents)),   dtype=np.float64)
        alphabisq_out = np.frombuffer((ctypes.c_double * max(pq, 1)) .from_address(ctypes.addressof(alphabisq_p.contents)), dtype=np.float64)
        sol_sdp_out   = np.frombuffer((ctypes.c_double * 1)           .from_address(ctypes.addressof(sol_p.contents)),      dtype=np.float64)

        python_solver_fn(n, mq, pq,
                         q_beta, c_beta, float(l_beta),
                         x_out, beta_diag_out, alphaq_out, alphabisq_out, sol_sdp_out)

    _registered_cb_ref = SDP_SOLVER_CB_TYPE(_c_callback)
    lib.miqcr_register_sdp_solver(_registered_cb_ref)
