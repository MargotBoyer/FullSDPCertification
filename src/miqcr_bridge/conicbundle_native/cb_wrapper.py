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
from typing import Dict, List, Optional, Tuple, Union

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

    lib.cb_get_last_weight.restype = ctypes.c_double
    lib.cb_get_last_weight.argtypes = [ctypes.c_void_p]

    lib.cb_get_candidate_value.restype = ctypes.c_double
    lib.cb_get_candidate_value.argtypes = [ctypes.c_void_p]

    lib.cb_get_bundle_values.restype = ctypes.c_int
    lib.cb_get_bundle_values.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _int_p, _int_p]

    lib.cb_set_max_bundlesize.restype = ctypes.c_int
    lib.cb_set_max_bundlesize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.cb_set_max_new_subgradients.restype = ctypes.c_int
    lib.cb_set_max_new_subgradients.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.cb_set_next_weight.restype = ctypes.c_int
    lib.cb_set_next_weight.argtypes = [ctypes.c_void_p, ctypes.c_double]

    lib.cb_set_min_weight.restype = ctypes.c_int
    lib.cb_set_min_weight.argtypes = [ctypes.c_void_p, ctypes.c_double]

    lib.cb_set_max_weight.restype = ctypes.c_int
    lib.cb_set_max_weight.argtypes = [ctypes.c_void_p, ctypes.c_double]

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
    active_bounds_fixing: bool = True,
    max_bundle_size: Optional[int] = None,
    factor: Optional[float] = None,
    max_new_subgradients: Optional[int] = None,
    initial_weight: Optional[Union[float, str]] = "auto",
    print_png: bool = False,
    png_save_path: Optional[str] = None,
    png_title: Optional[str] = None,
    stop_when_positive: bool = True,
    positivity_threshold: float = 1e-6,
) -> Tuple[float, Dict[str, float], int, Optional[str]]:
    """
    Maximise h(theta) = min_{X faisable, contraintes `dualizable_names` relaxees}
    <C,X> - theta.rhs  sur theta_r >= 0 (>=0 pour <=/>=, libre pour ==), en pilotant
    la vraie ConicBundle (Helmberg/Kiwiel) via handler.resolve_dualized comme oracle.

    handler doit deja avoir appele setup_dualization(dualizable_names) -- ou cette
    fonction l'appelle elle-meme si besoin (idempotent).

    Si print_png=True (et png_save_path fourni), enregistre a chaque tour de la
    boucle de descente (n_iter, borne par max_iter) : h(theta) au dernier point
    evalue (cb_get_candidate_value), le meilleur h trouve jusque-la (toutes
    evaluations oracle confondues, y compris les pas nuls), le poids proximal u
    courant (cb_get_last_weight) et la taille du bundle (cb_get_bundle_values), puis
    ecrit le meme PNG a 4 sous-graphes que le moteur "python" (dynamic_conic_bundle
    /plotting.py::plot_run_diagnostics). Un "vrai pas" (pas de descente serieux, par
    opposition a un pas nul) est detecte en comparant candidate_value a objval juste
    apres l'appel a cb_do_descent_step -- cf. doc de cb_get_candidate_value dans
    cb_cinterface.h : "If this last evaluation led to a descent step, then it is the
    same value as in get_objval()".

    max_bundle_size fixe cb_set_max_bundlesize (defaut interne de la librairie = 50,
    constante codee en dur dans FunctionProblem::FunctionProblem, funproblem.cxx --
    non calibree sur m ou dim(theta)) ; max_new_subgradients fixe
    cb_set_max_new_subgradients (defaut interne = 5). None conserve le defaut de la
    librairie pour le parametre concerne.

    factor : equivalent du parametre FACTOR de la librairie MIQCR d'origine
    (Miqcr-1.0_.../src/parameters.h -- "Proportion of the considered constraints
    into the SDP solver"), transpose ici a nos contraintes DUALIZABLE (RLT) plutot
    qu'aux McCormick internes de MIQCR. Ignore si max_bundle_size est deja fourni
    explicitement (max_bundle_size a toujours priorite). Sinon, si factor n'est pas
    None, effective_max_bundle_size = max(1, round(factor * m)) ou m = nb de
    contraintes DUALIZABLE effectivement dualisees (== len(dualizable_names)) --
    permet de faire suivre le plafond du bundle a la taille du probleme au lieu
    d'un entier fixe, comme le fait FACTOR cote MIQCR pour son propre working-set
    de McCormick. None (defaut) laisse max_bundle_size/le defaut interne inchanges.

    initial_weight fixe cb_set_next_weight avant le premier cb_do_descent_step.
    SANS cet appel, le premier pas utilise le poids proximal choisi par
    l'heuristique automatique de la librairie -- observe empiriquement bien trop
    petit sur ce projet (chordal + RLT dualisees) : le tout premier pas quitte
    theta=0 (deja tres proche de l'optimum, cf. oracle #1) et atterrit sur un
    point catastrophique (h chute de +7.9 a -24.5, cf. run.log), et le budget
    d'evaluations restant est entierement consomme a re-converger vers la region
    de depart sans jamais la depasser.

    "auto" (defaut) calibre le poids sur la vraie geometrie du probleme : un
    appel oracle supplementaire est fait a theta=0 (cout ~1 resolution MOSEK)
    pour mesurer ||g(0)|| (norme du sous-gradient au centre initial), et
    initial_weight = ||g(0)|| est utilise -- meme principe que _calibrate_u
    dans dynamic_conic_bundle/solver.py (moteur "python"). ATTENTION : une
    valeur fixe naive comme 1.0 (suggestion generique de cb_cinterface.h,
    "1 is frequently a reasonable choice if the automatic default heuristic
    performs poorly") s'est averee CONTRE-PRODUCTIVE ici -- elle a produit un
    premier pas encore pire que le defaut automatique (h chute a -256 au lieu
    de -24.5), car u=1/step_size et le sous-gradient a une norme largement
    superieure a 1 sur ce probleme (grandeurs RLT non normalisees) ; une valeur
    fixe ne peut pas s'adapter d'un echantillon a l'autre. Un float fixe reste
    accepte pour des tests manuels ; None desactive l'appel (comportement
    historique, heuristique automatique de la librairie, elle-meme sous-calibree
    ici).

    stop_when_positive : si True (defaut), arrete la boucle de descente des qu'un
    point oracle deja evalue (best["h"], mis a jour a CHAQUE appel oracle, y compris
    les pas nuls internes a un cb_do_descent_step -- pas seulement les pas serieux)
    depasse positivity_threshold. Par dualite faible, un seul h(theta) valide > 0
    certifie deja la robustesse : inutile de continuer a affiner theta. Dans ce cas
    stop_reason="reached_positivity" est renvoye (4e element du tuple), sinon None.

    Retourne (h_best, theta_best, n_descent_steps, stop_reason).
    """
    lib = _load_lib(so_path)

    handler.setup_dualization(dualizable_names)
    dualized = handler.get_dualized_constraints_data()
    m = len(dualized)
    assert m == len(dualizable_names), (
        f"get_dualized_constraints_data() a retourne {m} entrees, "
        f"attendu {len(dualizable_names)} (dualizable_names)"
    )

    effective_max_bundle_size = max_bundle_size
    if effective_max_bundle_size is None and factor is not None:
        effective_max_bundle_size = max(1, round(factor * m))
        if print_level >= 0:
            print(f"[cb_wrapper] factor={factor} x m={m} contraintes dualisables "
                  f"=> max_bundle_size={effective_max_bundle_size}")
    names = [d["name"] for d in dualized]

    minus_inf = lib.cb_get_minus_infinity()
    plus_inf = lib.cb_get_plus_infinity()
    lowerb = np.array(
        [minus_inf if d["free_sign"] else 0.0 for d in dualized], dtype=np.float64
    )
    upperb = np.full(m, plus_inf, dtype=np.float64)

    best = {"h": -np.inf, "theta": None}
    state = {"n_calls": 0}

    bundlesize_c = ctypes.c_int(0)
    new_subgrads_c = ctypes.c_int(0)
    h_history: List[float] = []
    best_h_history: List[float] = []
    u_history: List[float] = []
    bundle_size_history: List[int] = []
    n_serious_history: List[int] = []
    n_serious_box = {"n": 0}

    effective_initial_weight: Optional[float] = None
    if initial_weight == "auto":
        theta0 = {name: 0.0 for name in names}
        res0 = handler.resolve_dualized(theta0)
        if res0["lb_value"] is not None:
            h0 = res0["lb_value"]
            g0 = np.array([_dualized_inner_product(d, res0["X"]) - d["rhs"] for d in dualized])
            sgnorm0 = float(np.linalg.norm(g0))
            effective_initial_weight = sgnorm0 if sgnorm0 > 0.0 else None
            best["h"], best["theta"] = h0, dict(theta0)
            state["n_calls"] += 1
            if log_every:
                print(f"  [cb_native calibration] h(0)={h0:.6f} ||g(0)||={sgnorm0:.6f} "
                      f"-> initial_weight={effective_initial_weight}", flush=True)
    elif initial_weight is not None:
        effective_initial_weight = float(initial_weight)

    stop_reason: Optional[str] = None
    if stop_when_positive and best["h"] > positivity_threshold:
        # Deja certifie au point de calibration theta=0 (initial_weight="auto") --
        # aucun besoin de construire/lancer la vraie ConicBundle.
        stop_reason = "reached_positivity"
        if print_level >= 0:
            print(f"[cb_native] h(0)={best['h']:.6f} > positivity_threshold={positivity_threshold} "
                  "-- arret immediat (stop_when_positive).")
        return best["h"], best["theta"], 0, stop_reason

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

        if print_png:
            # Un point par appel oracle reel (pas par pas de descente externe) :
            # un seul cb_do_descent_step peut declencher de tres nombreux pas nuls
            # (donc appels oracle) avant de rendre la main -- echantillonner au
            # niveau de la boucle externe ne donnait alors qu'un seul point (droite
            # invisible sur le graphique) meme apres des milliers de resolutions
            # MOSEK internes.
            h_history.append(h_val)
            best_h_history.append(best["h"])
            u_history.append(lib.cb_get_last_weight(p))
            lib.cb_get_bundle_values(p, function_key, ctypes.byref(bundlesize_c),
                                      ctypes.byref(new_subgrads_c))
            bundle_size_history.append(bundlesize_c.value)
            n_serious_history.append(n_serious_box["n"])

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

        function_key = ctypes.c_void_p(1)
        rc = lib.cb_add_function(p, function_key, oracle_cb, None, 0)
        assert rc == 0, f"cb_add_function: rc={rc}"

        if effective_max_bundle_size is not None:
            rc = lib.cb_set_max_bundlesize(p, function_key, effective_max_bundle_size)
            assert rc == 0, f"cb_set_max_bundlesize: rc={rc}"
        if max_new_subgradients is not None:
            rc = lib.cb_set_max_new_subgradients(p, function_key, max_new_subgradients)
            assert rc == 0, f"cb_set_max_new_subgradients: rc={rc}"

        lib.cb_set_term_relprec(p, term_relprec)
        lib.cb_set_eval_limit(p, eval_limit)
        lib.cb_set_print_level(p, print_level)
        # Recommande par cb_cinterface.h pour la relaxation lagrangienne : fixe
        # temporairement les theta_r dont le multiplicateur de borne est fort (ici,
        # la plupart des 450 contraintes RLT dualisees sont probablement inactives,
        # theta_r=0 -- accelere la convergence en reduisant la dimension effective
        # du sous-probleme quadratique interne. ATTENTION : "no convergence theory"
        # (cb_cinterface.h) -- suspect n#1 d'une fausse convergence (termination_code=1
        # avec violations KKT franches) observee en chordal, cf. task-dynamic-conic-bundle.md.
        lib.cb_set_active_bounds_fixing(p, 1 if active_bounds_fixing else 0)

        if effective_initial_weight is not None:
            rc = lib.cb_set_next_weight(p, effective_initial_weight)
            assert rc == 0, f"cb_set_next_weight: rc={rc}"

        n_iter = 0
        while lib.cb_termination_code(p) == 0 and n_iter < max_iter:
            rc = lib.cb_do_descent_step(p)
            n_iter += 1
            if rc != 0:
                break

            if print_png and h_history:
                # Le dernier point echantillonne dans oracle() pour cet appel a
                # cb_do_descent_step correspond-il a un vrai pas (nouveau centre)
                # ou a un pas nul ? cf. cb_cinterface.h / doc de cb_get_candidate_value.
                cand_h = -lib.cb_get_candidate_value(p)
                obj_h = -lib.cb_get_objval(p)
                if abs(cand_h - obj_h) <= 1e-12 * (abs(obj_h) + 1.0):
                    n_serious_box["n"] += 1
                    n_serious_history[-1] = n_serious_box["n"]

            if stop_when_positive and best["h"] > positivity_threshold:
                # best["h"] est mis a jour a chaque appel oracle (y compris les pas
                # nuls internes a cb_do_descent_step) -- un seul point valide > seuil
                # suffit (dualite faible), inutile de laisser le bundle continuer.
                stop_reason = "reached_positivity"
                break

        center = np.zeros(m, dtype=np.float64)
        lib.cb_get_center(p, center.ctypes.data_as(_dbl_p))
        f_center = lib.cb_get_objval(p)
        h_center = -f_center
        theta_center = {names[r]: float(center[r]) for r in range(m)}

        if print_level >= 0:
            code = lib.cb_termination_code(p)
            print(f"[cb_native] n_iter={n_iter} n_oracle_calls={state['n_calls']} "
                  f"termination_code={code} h_center={h_center:.6f} best_h={best['h']:.6f} "
                  f"stop_reason={stop_reason}")

        if print_png and png_save_path and h_history:
            # Try/except dedie : un echec de plot ne doit pas faire perdre le
            # resultat deja calcule (meme convention que _solve_conic_bundle_python).
            try:
                from dynamic_conic_bundle.plotting import plot_run_diagnostics

                plot_run_diagnostics(
                    h_history, u_history, bundle_size_history, png_save_path,
                    title=png_title, best_h_history=best_h_history,
                    n_serious_history=n_serious_history,
                )
            except Exception as e:
                print(f"[cb_native] echec de l'ecriture du PNG de diagnostic : {e}")
    finally:
        p_holder = ctypes.c_void_p(p)
        lib.cb_destruct_problem(ctypes.byref(p_holder))

    # h_center (dernier centre ConicBundle) est la sortie standard, mais best["h"]
    # (max sur tous les points evalues, y compris les null steps) ne peut jamais
    # etre pire par construction de h -- on renvoie le meilleur des deux.
    if best["h"] > h_center:
        return best["h"], best["theta"], n_iter, stop_reason
    return h_center, theta_center, n_iter, stop_reason
