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
    os.path.join(
        os.path.dirname(__file__),
        "../../Miqcr-1.0_triang_sbb_gurobiter_QCP5/src/libmiqcr_pyapi.so",
    ),
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

    # Mise en forme C-contiguous (row-major) des tableaux numpy
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

    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

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

    out_beta      = np.zeros(n * n,  dtype=np.float64)
    out_alphaq    = np.zeros(max(mq, 1), dtype=np.float64)
    out_alphabisq = np.zeros(max(pq, 1), dtype=np.float64)
    out_sol_sdp   = np.zeros(1,      dtype=np.float64)

    lib.miqcr_compute_betas(
        ptr(out_beta),
        ptr(out_alphaq),
        ptr(out_alphabisq),
        ptr(out_sol_sdp),
    )

    return MiqcrResult(
        beta=out_beta.reshape(n, n),
        alphaq=out_alphaq[:mq],
        alphabisq=out_alphabisq[:pq],
        sol_sdp=float(out_sol_sdp[0]),
    )
