"""
Pont ctypes direct vers la vraie librairie ConicBundle (Helmberg/Kiwiel), via son
interface C generique cb_cinterface.h -- PAS via l'API MIQP-specifique de MIQCR
(miqcr_pyapi.c), qui dualise les McCormick internes de MIQCR et non nos contraintes
RLT (cf. Miqcr-1.0.../CLAUDE.md, section 5 : run_sdp_solver_mixed_mosek dualise
uniquement beta_1..4 sur les bornes de boite, Aq/Dq restent des contraintes dures).

libcb_native.so est construite en liant statiquement Miqcr-1.0.../ConicBundle/lib/libcb.a
(qui contient deja les symboles C cb_construct_problem, cb_do_descent_step, etc. --
CBSolver.o dans l'archive) :
    g++ -shared -fPIC -o libcb_native.so -Wl,--whole-archive libcb.a -Wl,--no-whole-archive -lstdc++ -lm

ConicBundle MINIMISE f_0(y)+...+f_k(y) (convention standard bundle method). Notre
probleme veut MAXIMISER h(theta) = min_{X faisable, RLT relaxees} <C,X> (Lagrangien
dual, theta_r >= 0 pour contraintes <=, libre pour contraintes d'egalite). On
minimise donc f(theta) = -h(theta), sous-gradient = -g(theta) avec
g_r(theta) = <A_r,X*> - rhs_r (formule standard, deja utilisee/validee dans
lagrangian_cb.run_lagrangian_cb et coherente avec handler.resolve_dualized).

L'oracle s'appuie exclusivement sur handler.resolve_dualized(theta) (deja verifie
byte-exact au sens theta=0, cf. investigation native_conic_bundle.py) -- aucune
reconstruction de matrices Aq/Dq/q_beta/c_beta n'est necessaire ici, contrairement
au pont MIQCR-specifique.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

_DEFAULT_SO = os.environ.get(
    "CB_NATIVE_SO",
    os.path.join(os.path.dirname(__file__), "libcb_native.so"),
)

_lib_cache: Optional[ctypes.CDLL] = None

_dbl_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)

# int (*cb_functionp)(void* function_key, double* arg, double relprec, int max_subg,
#                      double* objective_value, int* n_subgrads, double* subg_values,
#                      double* subgradients, double* primal)
CB_FUNCTION_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    _dbl_p,
    ctypes.c_double,
    ctypes.c_int,
    _dbl_p,
    _int_p,
    _dbl_p,
    _dbl_p,
    _dbl_p,
)


def _build_lib_via_make(expected_path: str) -> None:
    """Construit libcb_native.so via le Makefile de ce dossier si elle est absente
    (ex. après un clone frais sans le binaire). Best-effort : n'échoue jamais
    bruyamment ici, _load_lib retente l'existence du fichier et lève une erreur
    claire si le build a échoué (make/g++ absent, libcb.a manquant, etc.)."""
    module_dir = os.path.dirname(__file__)
    print(f"[cb_wrapper] {expected_path} introuvable — tentative de build via "
          f"`make -C {module_dir}`...")
    try:
        subprocess.run(["make", "-C", module_dir], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[cb_wrapper] échec du build automatique : {e}")


def _load_lib(so_path: Optional[str] = None) -> ctypes.CDLL:
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    path = os.path.realpath(so_path or _DEFAULT_SO)
    if not os.path.exists(path):
        _build_lib_via_make(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"libcb_native.so introuvable après tentative de build : {path}\n"
            f"Construire manuellement : make -C {os.path.dirname(__file__)}"
        )
    lib = ctypes.CDLL(path)

    lib.cb_construct_problem.restype = ctypes.c_void_p
    lib.cb_construct_problem.argtypes = [ctypes.c_int]

    lib.cb_destruct_problem.restype = ctypes.c_int
    lib.cb_destruct_problem.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    lib.cb_init_problem.restype = ctypes.c_int
    lib.cb_init_problem.argtypes = [ctypes.c_void_p, ctypes.c_int, _dbl_p, _dbl_p]

    lib.cb_add_function.restype = ctypes.c_int
    lib.cb_add_function.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, CB_FUNCTION_TYPE, ctypes.c_void_p, ctypes.c_int,
    ]

    lib.cb_do_descent_step.restype = ctypes.c_int
    lib.cb_do_descent_step.argtypes = [ctypes.c_void_p]

    lib.cb_termination_code.restype = ctypes.c_int
    lib.cb_termination_code.argtypes = [ctypes.c_void_p]

    lib.cb_print_termination_code.restype = ctypes.c_int
    lib.cb_print_termination_code.argtypes = [ctypes.c_void_p]

    lib.cb_get_objval.restype = ctypes.c_double
    lib.cb_get_objval.argtypes = [ctypes.c_void_p]

    lib.cb_get_center.restype = ctypes.c_int
    lib.cb_get_center.argtypes = [ctypes.c_void_p, _dbl_p]

    lib.cb_get_sgnorm.restype = ctypes.c_double
    lib.cb_get_sgnorm.argtypes = [ctypes.c_void_p]

    lib.cb_set_term_relprec.restype = ctypes.c_int
    lib.cb_set_term_relprec.argtypes = [ctypes.c_void_p, ctypes.c_double]

    lib.cb_set_eval_limit.restype = None
    lib.cb_set_eval_limit.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.cb_set_print_level.restype = None
    lib.cb_set_print_level.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.cb_get_minus_infinity.restype = ctypes.c_double
    lib.cb_get_minus_infinity.argtypes = []

    lib.cb_get_plus_infinity.restype = ctypes.c_double
    lib.cb_get_plus_infinity.argtypes = []

    lib.cb_get_dim.restype = ctypes.c_int
    lib.cb_get_dim.argtypes = [ctypes.c_void_p]

    lib.cb_set_active_bounds_fixing.restype = None
    lib.cb_set_active_bounds_fixing.argtypes = [ctypes.c_void_p, ctypes.c_int]

    _lib_cache = lib
    return lib


def _dualized_inner_product(d: dict, X_by_matrix: Dict[int, np.ndarray]) -> float:
    """<A_r, X*> pour la contrainte dualisee `d` (triplet num_matrix/i/j/value, deja
    signe/halve selon la convention mosek_classic -- meme format que
    handler.resolve_dualized / _dualization_triplet_for)."""
    total = 0.0
    nm, ii, jj, vv = d["num_matrix"], d["i"], d["j"], d["value"]
    for k in range(len(ii)):
        num_matrix, i, j, v = int(nm[k]), int(ii[k]), int(jj[k]), float(vv[k])
        X = X_by_matrix[num_matrix]
        if i == j:
            total += v * X[i, i]
        else:
            total += 2.0 * v * X[i, j]
    return total


def solve_native_conicbundle_dual(
    handler,
    dualizable_names: List[str],
    max_iter: int = 500,
    term_relprec: float = 1e-7,
    eval_limit: int = -1,
    print_level: int = 0,
    so_path: Optional[str] = None,
    log_every: int = 0,
) -> Tuple[float, Dict[str, float], int]:
    """
    Maximise h(theta) = min_{X faisable, contraintes `dualizable_names` relaxees}
    <C,X> - theta.rhs  sur theta_r >= 0 (>=0 pour <=/>=, libre pour ==), en pilotant
    la vraie ConicBundle (Helmberg/Kiwiel) via handler.resolve_dualized comme oracle.

    handler doit deja avoir appele setup_dualization(dualizable_names) -- ou cette
    fonction l'appelle elle-meme si besoin (idempotent).

    Retourne (h_best, theta_best, n_descent_steps).
    """
    lib = _load_lib(so_path)

    handler.setup_dualization(dualizable_names)
    dualized = handler.get_dualized_constraints_data()
    m = len(dualized)
    assert m == len(dualizable_names), (
        f"get_dualized_constraints_data() a retourne {m} entrees, "
        f"attendu {len(dualizable_names)} (dualizable_names)"
    )
    names = [d["name"] for d in dualized]

    minus_inf = lib.cb_get_minus_infinity()
    plus_inf = lib.cb_get_plus_infinity()
    lowerb = np.array(
        [minus_inf if d["free_sign"] else 0.0 for d in dualized], dtype=np.float64
    )
    upperb = np.full(m, plus_inf, dtype=np.float64)

    best = {"h": -np.inf, "theta": None}
    state = {"n_calls": 0}

    def oracle(function_key, arg_p, relprec, max_subg,
               objective_value_p, n_subgrads_p, subg_values_p, subgradients_p, primal_p):
        state["n_calls"] += 1
        theta_arr = np.ctypeslib.as_array(arg_p, shape=(m,))
        theta = {names[r]: float(theta_arr[r]) for r in range(m)}

        res = handler.resolve_dualized(theta)
        if res["lb_value"] is None:
            # Echec MOSEK : pas d'hyperplan valide a renvoyer -> echec de l'oracle.
            return 1

        h_val = res["lb_value"]
        if h_val > best["h"]:
            best["h"] = h_val
            best["theta"] = dict(theta)

        g = np.array([_dualized_inner_product(d, res["X"]) - d["rhs"] for d in dualized])

        if log_every and state["n_calls"] % log_every == 0:
            print(f"  [cb_native oracle #{state['n_calls']}] h={h_val:.6f} best={best['h']:.6f}",
                  flush=True)

        objective_value_p[0] = -h_val
        n_subgrads_p[0] = 1
        subg_values_p[0] = -h_val
        for r in range(m):
            subgradients_p[r] = -g[r]
        return 0

    oracle_cb = CB_FUNCTION_TYPE(oracle)

    p = lib.cb_construct_problem(0)
    if not p:
        raise RuntimeError("cb_construct_problem a echoue")
    try:
        rc = lib.cb_init_problem(p, m, lowerb.ctypes.data_as(_dbl_p), upperb.ctypes.data_as(_dbl_p))
        assert rc == 0, f"cb_init_problem: rc={rc}"

        rc = lib.cb_add_function(p, ctypes.c_void_p(1), oracle_cb, None, 0)
        assert rc == 0, f"cb_add_function: rc={rc}"

        lib.cb_set_term_relprec(p, term_relprec)
        lib.cb_set_eval_limit(p, eval_limit)
        lib.cb_set_print_level(p, print_level)
        # Recommande par cb_cinterface.h pour la relaxation lagrangienne : fixe
        # temporairement les theta_r dont le multiplicateur de borne est fort (ici,
        # la plupart des 450 contraintes RLT dualisees sont probablement inactives,
        # theta_r=0 -- accelere la convergence en reduisant la dimension effective
        # du sous-probleme quadratique interne.
        lib.cb_set_active_bounds_fixing(p, 1)

        n_iter = 0
        while lib.cb_termination_code(p) == 0 and n_iter < max_iter:
            rc = lib.cb_do_descent_step(p)
            n_iter += 1
            if rc != 0:
                break

        center = np.zeros(m, dtype=np.float64)
        lib.cb_get_center(p, center.ctypes.data_as(_dbl_p))
        f_center = lib.cb_get_objval(p)
        h_center = -f_center
        theta_center = {names[r]: float(center[r]) for r in range(m)}

        if print_level >= 0:
            code = lib.cb_termination_code(p)
            print(f"[cb_native] n_iter={n_iter} n_oracle_calls={state['n_calls']} "
                  f"termination_code={code} h_center={h_center:.6f} best_h={best['h']:.6f}")
    finally:
        p_holder = ctypes.c_void_p(p)
        lib.cb_destruct_problem(ctypes.byref(p_holder))

    # h_center (dernier centre ConicBundle) est la sortie standard, mais best["h"]
    # (max sur tous les points evalues, y compris les null steps) ne peut jamais
    # etre pire par construction de h -- on renvoie le meilleur des deux.
    if best["h"] > h_center:
        return best["h"], best["theta"], n_iter
    return h_center, theta_center, n_iter
