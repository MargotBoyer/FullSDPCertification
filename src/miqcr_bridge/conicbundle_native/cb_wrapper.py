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

import csv
import ctypes
import os
import subprocess
import time
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

# cb_subgextp (extension de sous-gradient via agregation primale) : implemente
# et valide fonctionnellement (solve_native_conicbundle_dynamic), mais retire
# suite a un segfault natif deterministe observe a l'echelle de cette librairie
# sur un primal tres large (cf. commentaire dans la boucle de maj du pool de
# solve_native_conicbundle_dynamic pour le detail de l'investigation).


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

    # Utilisees par solve_native_conicbundle_dynamic (mode dynamique, cf. MIQCR
    # solver_sdp_mixed.c::run_conic_bundle_mixed) : redimensionnent le probleme
    # ConicBundle EN PLACE (sans jamais le detruire/reconstruire), contrairement au
    # mode dynamique du moteur "python" qui redemarre un round de bundle a chaque
    # mise a jour du pool.
    lib.cb_append_variables.restype = ctypes.c_int
    lib.cb_append_variables.argtypes = [ctypes.c_void_p, ctypes.c_int, _dbl_p, _dbl_p]

    lib.cb_reassign_variables.restype = ctypes.c_int
    lib.cb_reassign_variables.argtypes = [ctypes.c_void_p, ctypes.c_int, _int_p]

    lib.cb_reinit_function_model.restype = ctypes.c_int
    lib.cb_reinit_function_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.cb_clear_fail_counts.restype = None
    lib.cb_clear_fail_counts.argtypes = [ctypes.c_void_p]

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
    bundle_size_factor: Optional[float] = None,
    max_new_subgradients: Optional[int] = None,
    initial_weight: Optional[Union[float, str]] = "auto",
    print_png: bool = False,
    png_save_path: Optional[str] = None,
    png_title: Optional[str] = None,
    png_every: int = 10,
    stop_when_positive: bool = True,
    positivity_threshold: float = 1e-6,
    max_subg_by_point: int = 10,
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
    ecrit un PNG a sous-graphes (dynamic_conic_bundle/plotting.py::plot_run_diagnostics)
    avec, en plus de h(theta)/meilleur h/u/taille du bundle/nb vrais pas : la norme du
    sous-gradient (cb_get_sgnorm), le nb de sous-gradients epsilon renvoyes par appel
    (cf. max_subg_by_point), l'ecart relatif detecte par la sonde de coin, la distance
    entre le point d'essai et le centre courant, et le temps par appel oracle. Un
    "vrai pas" (pas de descente serieux, par opposition a un pas nul) est detecte en
    comparant candidate_value a objval juste apres l'appel a cb_do_descent_step -- cf.
    doc de cb_get_candidate_value dans cb_cinterface.h : "If this last evaluation led
    to a descent step, then it is the same value as in get_objval()".

    En plus du PNG final, ecrit a CHAQUE appel oracle une ligne dans un CSV compagnon
    (meme chemin que png_save_path, extension .csv) -- survit a un kill/timeout du
    process, contrairement au PNG qui n'etait ecrit qu'une fois a la toute fin avant
    ce parametre. png_every (defaut 10) regenere aussi le PNG tous les png_every
    appels oracle (0 = uniquement a la toute fin, comportement historique) : permet de
    suivre un run long sans attendre sa fin.

    max_bundle_size fixe cb_set_max_bundlesize (defaut interne de la librairie = 50,
    constante codee en dur dans FunctionProblem::FunctionProblem, funproblem.cxx --
    non calibree sur m ou dim(theta)) ; max_new_subgradients fixe
    cb_set_max_new_subgradients (defaut interne = 1, PAS 5 -- verifie dans
    FunctionBundleParameters::FunctionBundleParameters(), funproblem.hxx :
    "n_new_subgradients=1"). Avec ce defaut, max_subg recu par l'oracle() vaut
    toujours 1, donc k_max=min(max_subg,max_subg_by_point) ne peut jamais depasser 1
    -- la sonde de coin (rel_disagreement) ne s'execute JAMAIS et l'enrichissement
    multi-sous-gradients (max_subg_by_point) est inerte tant que
    max_new_subgradients n'est pas explicitement fixe a >=2. Confirme empiriquement :
    disagreement_history est reste NaN sur la totalite des appels oracle observes
    (data_index=59 target=8 et data_index=23 target=7). None conserve le defaut de
    la librairie pour le parametre concerne (donc PAS de sonde de coin par defaut).

    bundle_size_factor : anciennement nomme `factor` (renomme pour ne plus etre
    confondu avec le vrai parametre FACTOR de MIQCR -- cf. `factor` ci-dessous, qui
    a un role different). Ignore si max_bundle_size est deja fourni explicitement
    (max_bundle_size a toujours priorite). Sinon, si bundle_size_factor n'est pas
    None, effective_max_bundle_size = max(1, round(bundle_size_factor * m)) ou m =
    nb de contraintes DUALIZABLE effectivement dualisees (== len(dualizable_names))
    -- permet de faire suivre le plafond du bundle (cb_set_max_bundlesize, nb de
    COUPES conservees en memoire) a la taille du probleme au lieu d'un entier fixe.
    N'affecte PAS le nombre de contraintes reellement dualisees (m reste fixe, voir
    `factor`). None (defaut) laisse max_bundle_size/le defaut interne inchanges.

    factor : equivalent du VRAI parametre FACTOR de la librairie MIQCR d'origine
    (Miqcr-1.0_.../src/parameters.h -- "Proportion of the considered constraints
    into the SDP solver" ; psdp->nb_cont = FACTOR * psdp->length, cf. CLAUDE.md
    MIQCR section 7). Determine la proportion des contraintes DUALIZABLE candidates
    qui sont EFFECTIVEMENT dualisees (recoivent un theta_r) -- les autres sont
    totalement absentes du modele (ni dures, ni penalisees), exactement comme chez
    MIQCR. Applique par l'appelant (certification_problem.py) avant meme d'appeler
    cette fonction : dualizable_names est deja filtre en amont, `factor` n'est donc
    pas un parametre de cette fonction (uniquement documente ici par souci de
    cohesion avec bundle_size_factor). None (defaut) = tout le pool candidat est
    dualise (comportement historique).

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

    max_subg_by_point : nombre max de sous-gradients epsilon renvoyes par appel oracle
    QUAND un "coin" (degenerescence du X* optimal) est detecte a ce point -- cf.
    l'investigation "ReLU_linear/triangularization + RLT + ReLU_quad" (task-dynamic-conic-bundle.md) :
    une fois trop de contraintes dualisees simultanement, le sous-probleme MOSEK residuel
    admet plusieurs X* optimaux tres differents, et le sous-gradient extrait d'un seul X*
    n'est representatif que d'UNE direction -- la vraie derivee directionnelle de h est un
    max sur TOUT le sous-differentiel (l'enveloppe convexe des sous-gradients obtenables en
    balayant les X* optimaux), pas un seul point. cb_cinterface.h supporte nativement le
    renvoi de plusieurs sous-gradients "epsilon" par appel oracle (max_subg en entree, jamais
    exploite avant ce parametre -- l'oracle renvoyait toujours n_subgrads=1).

    Detection du "coin" : une sonde bon marche (1 resolution supplementaire a theta perturbe
    de facon infinitesimale) compare son sous-gradient a celui du point principal. Si l'ecart
    relatif depasse un seuil (5%), le point est considere degenere : on enrichit alors avec
    max_subg_by_point-2 sous-gradients supplementaires (perturbations aleatoires independantes
    de theta), chacun via une resolution MOSEK complete -- cout proportionnel a
    max_subg_by_point UNIQUEMENT sur les points detectes degeneres (cout d'1 resolution
    supplementaire partout ailleurs, pour la sonde). Chaque sous-gradient supplementaire est
    exact (pas approxime) : obtenu par resolve_dualized(theta') a un theta' reellement proche
    de theta, sa coupe affine est valide globalement (inegalite de sur-gradient standard pour
    h concave), evaluee au point theta demande par la librairie (cf. subg_values dans
    cb_cinterface.h : "store for each epsilon subgradient the value at the argument").
    max_subg_by_point=1 desactive l'enrichissement (comportement historique).

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
    if effective_max_bundle_size is None and bundle_size_factor is not None:
        effective_max_bundle_size = max(1, round(bundle_size_factor * m))
        if print_level >= 0:
            print(f"[cb_wrapper] bundle_size_factor={bundle_size_factor} x m={m} "
                  f"contraintes dualisables => max_bundle_size={effective_max_bundle_size}")
    names = [d["name"] for d in dualized]

    minus_inf = lib.cb_get_minus_infinity()
    plus_inf = lib.cb_get_plus_infinity()
    lowerb = np.array(
        [minus_inf if d["free_sign"] else 0.0 for d in dualized], dtype=np.float64
    )
    upperb = np.full(m, plus_inf, dtype=np.float64)

    best = {"h": -np.inf, "theta": None}
    state = {"n_calls": 0}
    rng = np.random.default_rng()

    bundlesize_c = ctypes.c_int(0)
    new_subgrads_c = ctypes.c_int(0)
    h_history: List[float] = []
    best_h_history: List[float] = []
    u_history: List[float] = []
    bundle_size_history: List[int] = []
    n_serious_history: List[int] = []
    n_serious_box = {"n": 0}
    sgnorm_history: List[float] = []
    new_subgrads_history: List[int] = []
    disagreement_history: List[float] = []
    step_norm_history: List[float] = []
    time_history: List[float] = []
    pretreatment_history: List[float] = []
    solve_time_history: List[float] = []

    csv_file = None
    csv_writer = None
    csv_save_path = None
    if print_png and png_save_path:
        csv_save_path = os.path.splitext(png_save_path)[0] + ".csv"
        csv_file = open(csv_save_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "oracle_call", "h", "best_h", "u", "bundle_size", "new_subgrads",
            "sgnorm", "step_norm", "disagreement", "time_s",
            "pretreatment_s", "mosek_solve_s",
        ])

    def _write_csv_row():
        if csv_writer is None:
            return
        csv_writer.writerow([
            state["n_calls"], h_history[-1], best_h_history[-1], u_history[-1],
            bundle_size_history[-1], new_subgrads_history[-1], sgnorm_history[-1],
            step_norm_history[-1], disagreement_history[-1], time_history[-1],
            pretreatment_history[-1], solve_time_history[-1],
        ])
        csv_file.flush()

    def _refresh_png():
        if not (print_png and png_save_path and h_history):
            return
        try:
            from dynamic_conic_bundle.plotting import plot_run_diagnostics

            plot_run_diagnostics(
                h_history, u_history, bundle_size_history, png_save_path,
                title=png_title, best_h_history=best_h_history,
                n_serious_history=n_serious_history, sgnorm_history=sgnorm_history,
                new_subgrads_history=new_subgrads_history,
                disagreement_history=disagreement_history,
                step_norm_history=step_norm_history, time_history=time_history,
                pretreatment_history=pretreatment_history,
                solve_time_history=solve_time_history,
            )
        except Exception as e:
            print(f"[cb_native] echec de l'ecriture du PNG de diagnostic : {e}")

    effective_initial_weight: Optional[float] = None
    if initial_weight == "auto":
        # theta=0 : dict vide, pas {name: 0.0 pour tout name} -- resolve_dualized
        # traite tout nom absent comme theta_r=0 (meme semantique), et sa boucle
        # creuse (cf. handler_classic.resolve_dualized) itere sur theta.items(),
        # donc un dict vide est le moyen le plus direct de coder theta=0 partout.
        theta0 = {}
        t_calib0 = time.perf_counter()
        res0 = handler.resolve_dualized(theta0)
        calib_elapsed = time.perf_counter() - t_calib0
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
            if print_png:
                # Point de calibration (theta=0) : logué même si stop_when_positive
                # court-circuite tout le reste (cas frequent -- beaucoup de targets
                # "faciles" sont deja certifies a theta=0), pour ne jamais renvoyer
                # un CSV/PNG vide.
                h_history.append(h0)
                best_h_history.append(best["h"])
                u_history.append(effective_initial_weight if effective_initial_weight is not None else float("nan"))
                bundle_size_history.append(0)
                n_serious_history.append(0)
                sgnorm_history.append(sgnorm0)
                new_subgrads_history.append(1)
                step_norm_history.append(0.0)
                disagreement_history.append(float("nan"))
                time_history.append(calib_elapsed)
                pretreatment_history.append(res0.get("pretreatment_time", float("nan")))
                solve_time_history.append(res0.get("solve_time", float("nan")))
                _write_csv_row()
        elif print_level >= 0:
            # resolve_dualized(theta=0) a echoue (status MOSEK non optimal, le plus
            # souvent un timeout contre solver_time_limit) -- signale immediatement
            # plutot que de laisser echouer silencieusement la construction du
            # probleme ConicBundle (qui produira alors un h_center non initialise,
            # cf. le garde-fou best["h"]==-inf en fin de fonction).
            print(f"[cb_native] ECHEC de la calibration : resolve_dualized(theta=0) "
                  f"n'a pas renvoye de solution valide (status={res0['status']}).")
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
        _refresh_png()
        if csv_file is not None:
            csv_file.close()
        return best["h"], best["theta"], 0, stop_reason

    def oracle(function_key, arg_p, relprec, max_subg,
               objective_value_p, n_subgrads_p, subg_values_p, subgradients_p, primal_p):
        state["n_calls"] += 1
        theta_arr = np.ctypeslib.as_array(arg_p, shape=(m,))
        # Ne garde que les composantes non nulles : active_bounds_fixing fixe la
        # plupart des theta_r a 0, donc theta_arr est generalement tres creux (m peut
        # depasser 100k). resolve_dualized() itere sur theta.items() (cf. sa boucle
        # creuse, handler_classic.py) -- un dict dense {name: 0.0 pour tout name}
        # coutait O(m) Python pur a CHAQUE appel oracle meme quand presque tout est
        # nul. np.nonzero fait le scan des m elements en C (numpy), pas en boucle
        # Python -- seule la construction du dict reste au niveau Python, et elle ne
        # porte plus que sur les indices non nuls. Semantique inchangee : tout nom
        # absent reste implicitement theta_r=0.
        nz_idx = np.nonzero(theta_arr)[0]
        theta = {names[r]: float(theta_arr[r]) for r in nz_idx}

        t_call0 = time.perf_counter()
        res = handler.resolve_dualized(theta)
        call_elapsed = time.perf_counter() - t_call0
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
            sgnorm_history.append(lib.cb_get_sgnorm(p))
            new_subgrads_history.append(new_subgrads_c.value)
            center_now = np.zeros(m, dtype=np.float64)
            lib.cb_get_center(p, center_now.ctypes.data_as(_dbl_p))
            step_norm_history.append(float(np.linalg.norm(theta_arr - center_now)))
            disagreement_history.append(float("nan"))  # patché ci-dessous si la sonde de coin tourne
            time_history.append(call_elapsed)
            # pretreatment_time : cote Python (construction de l'objectif dualise,
            # putbarcblocktriplet) avant task.optimize() ; solve_time : task.optimize()
            # seul. Les deux sont renvoyes par resolve_dualized (handler_classic.py) --
            # permet de voir, appel par appel, la part MOSEK vs la part Python de
            # call_elapsed (qui inclut aussi l'extraction du statut/X apres le solve).
            pretreatment_history.append(res.get("pretreatment_time", float("nan")))
            solve_time_history.append(res.get("solve_time", float("nan")))

        objective_value_p[0] = -h_val

        # --- Enrichissement multi-sous-gradients si on detecte un "coin" ---
        # subgrads : liste de (h_i, g_i, theta_i) -- theta_i le point ou (h_i,g_i) est exact.
        subgrads = [(h_val, g, theta_arr.copy())]
        k_max = min(max_subg, max_subg_by_point)
        if k_max > 1:
            probe_delta = 1e-6 * (1.0 + np.abs(theta_arr))
            probe_theta_arr = theta_arr + probe_delta * rng.choice([-1.0, 1.0], size=m)
            probe_theta = {names[r]: float(probe_theta_arr[r]) for r in range(m)}
            probe_res = handler.resolve_dualized(probe_theta)
            if probe_res["lb_value"] is not None:
                probe_g = np.array([_dualized_inner_product(d, probe_res["X"]) - d["rhs"] for d in dualized])
                rel_disagreement = np.linalg.norm(probe_g - g) / (np.linalg.norm(g) + 1e-12)
                subgrads.append((probe_res["lb_value"], probe_g, probe_theta_arr))
                if print_png and disagreement_history:
                    disagreement_history[-1] = float(rel_disagreement)

                if rel_disagreement > 0.05:
                    # Coin detecte (le sous-gradient a saute pour un pas infinitesimal) --
                    # on enrichit avec des points supplementaires perturbes independamment.
                    if log_every:
                        print(f"  [cb_native oracle #{state['n_calls']}] coin detecte "
                              f"(ecart sous-gradient={rel_disagreement:.3f}) -- enrichissement "
                              f"a {k_max} sous-gradients", flush=True)
                    for _ in range(k_max - 2):
                        extra_delta = 1e-6 * (1.0 + np.abs(theta_arr))
                        extra_theta_arr = theta_arr + extra_delta * rng.uniform(-1.0, 1.0, size=m)
                        extra_theta = {names[r]: float(extra_theta_arr[r]) for r in range(m)}
                        extra_res = handler.resolve_dualized(extra_theta)
                        if extra_res["lb_value"] is None:
                            continue
                        extra_g = np.array([_dualized_inner_product(d, extra_res["X"]) - d["rhs"] for d in dualized])
                        subgrads.append((extra_res["lb_value"], extra_g, extra_theta_arr))

        n_subgrads_p[0] = len(subgrads)
        for i, (h_i, g_i, theta_i_arr) in enumerate(subgrads):
            # Valeur de la coupe affine (h_i, g_i) -- exacte en theta_i -- evaluee au point
            # theta effectivement demande par la librairie (arg_p), cf. cb_cinterface.h.
            diff = theta_arr - theta_i_arr
            cut_value_at_theta = h_i + float(np.dot(g_i, diff))
            if i > 0:
                # Garde-fou numerique : par concavite de h, toute coupe issue d'un AUTRE
                # point theta_i doit rester une borne SUPERIEURE sur h(theta_arr), donc
                # cut_value_at_theta >= h_val (h_val = valeur du point principal, calculee
                # directement). Cf. funproblem.cxx:176-179 (FunctionProblem::eval_function) :
                # la librairie exige cand_ub_fun_val >= max(cand_subg_valvec), et rejette
                # l'appel entier (rc=1) sinon -- observe concretement sur data_index=23
                # target=7 (max_new_subgradients=2) : la sonde de coin, evaluee a un theta
                # infinitesimalement perturbe, donne un cut_value_at_theta inferieur a
                # h_val de ~1.8e-5 (residu numerique inevitable entre deux resolutions
                # MOSEK independantes, pas une vraie violation de concavite), ce qui a fait
                # echouer cb_do_descent_step() des le tout premier appel reel. Le clip ne
                # perd aucune information utile : la DIRECTION du sous-gradient (g_i) reste
                # intacte, seule sa valeur extrapolee au point de requete est corrigee.
                cut_value_at_theta = max(cut_value_at_theta, h_val)
            subg_values_p[i] = -cut_value_at_theta
            for r in range(m):
                subgradients_p[i * m + r] = -g_i[r]

        _write_csv_row()
        if png_every and state["n_calls"] % png_every == 0:
            _refresh_png()
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
        any_successful_descent_step = False
        while lib.cb_termination_code(p) == 0 and n_iter < max_iter:
            rc = lib.cb_do_descent_step(p)
            n_iter += 1
            if rc == 0:
                any_successful_descent_step = True
            if rc != 0:
                # rc != 0 signale une VRAIE erreur interne ConicBundle (distincte d'un
                # arret propre par critere de terminaison, qui se traduit par
                # cb_termination_code(p) != 0 avec rc == 0 -- cf. cb_cinterface.h,
                # doc de cb_do_descent_step : "0 on success, != 0 otherwise"). Vu sans
                # ce log jusqu'ici -- cf. data_index=59 target=8, rc jamais imprime,
                # termination_code=0 pourtant sorti de la boucle des n_iter=1.
                if print_level >= 0:
                    print(f"[cb_native] cb_do_descent_step a echoue : rc={rc} "
                          f"(n_iter={n_iter} n_oracle_calls={state['n_calls']})")
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

        # Rafraichissement final : garantit que le PNG reflete le tout dernier point
        # meme si state["n_calls"] % png_every != 0 au moment du dernier appel oracle.
        _refresh_png()
    finally:
        p_holder = ctypes.c_void_p(p)
        lib.cb_destruct_problem(ctypes.byref(p_holder))
        if csv_file is not None:
            csv_file.close()

    if best["h"] == -np.inf:
        # AUCUNE evaluation d'oracle n'a jamais reussi (ni la calibration, ni le
        # tout premier appel pilote par la vraie ConicBundle) -- typiquement
        # resolve_dualized(theta=0) qui time-out contre solver_time_limit (cf.
        # handler_classic.resolve_dualized : renvoie lb_value=None des que
        # task.getsolsta() n'est pas optimal/prim_and_dual_feas, sans lever
        # d'exception). Dans ce cas h_center/theta_center ne sont que l'etat
        # interne NON INITIALISE de la librairie (observe : h_center~0.0 avec le
        # message natif "MatrixFCBSolver::do_descent_step(): setting default
        # starting point failed") -- PAS un resultat valide. Le renvoyer comme un
        # succes serait un FAUX POSITIF silencieux (observe concretement sur
        # data_index=23 target=7 : optimal_value=-0.0, is_robust=True, alors
        # qu'aucune resolution MOSEK n'a reellement abouti). On renvoie donc un
        # echec explicite plutot que h_center, cf. convention deja utilisee par
        # l'appelant (_solve_conic_bundle_native : lb=None + status distinctif).
        if print_level >= 0:
            print(f"[cb_native] ECHEC : aucune evaluation d'oracle n'a reussi "
                  f"(n_oracle_calls={state['n_calls']}) -- h_center est l'etat non "
                  f"initialise de la librairie, pas un resultat valide.")
        return None, None, n_iter, stop_reason or "oracle_failed"

    if not any_successful_descent_step:
        # best["h"] est valide (calibration ou au moins un appel oracle reussi),
        # mais AUCUN cb_do_descent_step() n'a jamais renvoye rc==0 -- h_center est
        # alors encore l'etat interne NON INITIALISE de la librairie (meme garde-fou
        # que ci-dessus, mais ici best["h"] n'est PAS -inf donc ne le declenchait pas).
        # Observe concretement : data_index=23 target=7 avec max_new_subgradients=2,
        # la sonde de coin detecte un vrai coin (ecart=0.412) des le 1er appel reel,
        # mais soumettre 2 sous-gradients au tout premier appel (avant qu'un point de
        # depart valide soit etabli) fait echouer cb_do_descent_step des le n_iter=1
        # ("setting default starting point failed") -- h_center=-0.000000 est du
        # garbage numeriquement PROCHE de 0, qui l'emportait a tort sur le vrai
        # best["h"]=-0.074033 dans la comparaison max() ci-dessous (FAUX POSITIF :
        # optimal_value=-0.0, is_robust=True alors que rien n'a reellement converge).
        # On ignore donc h_center entierement tant qu'aucun pas n'a reussi.
        if print_level >= 0:
            print(f"[cb_native] Aucun cb_do_descent_step() reussi -- h_center ignore "
                  f"(etat non initialise), renvoi de best_h={best['h']:.6f} seul.")
        return best["h"], best["theta"], n_iter, stop_reason

    # h_center (dernier centre ConicBundle) est la sortie standard, mais best["h"]
    # (max sur tous les points evalues, y compris les null steps) ne peut jamais
    # etre pire par construction de h -- on renvoie le meilleur des deux.
    if best["h"] > h_center:
        return best["h"], best["theta"], n_iter, stop_reason
    return h_center, theta_center, n_iter, stop_reason


def solve_native_conicbundle_dynamic(
    handler,
    all_dualizable_names: List[str],
    add_batch_size: int = 50,
    theta_drop_tol: float = 1e-8,
    max_pool_updates: int = 100,
    max_iter: int = 500,
    term_relprec: float = 1e-7,
    eval_limit: int = -1,
    print_level: int = 0,
    so_path: Optional[str] = None,
    log_every: int = 0,
    active_bounds_fixing: bool = True,
    max_bundle_size: Optional[int] = None,
    bundle_size_factor: Optional[float] = None,
    initial_weight: Optional[Union[float, str]] = "auto",
    print_png: bool = False,
    png_save_path: Optional[str] = None,
    png_title: Optional[str] = None,
    png_every: int = 10,
    stop_when_positive: bool = True,
    positivity_threshold: float = 1e-6,
) -> Tuple[float, Dict, int, Optional[str]]:
    """
    ATTENTION -- limite de fiabilite connue (investigation data_index=23,
    mnist-9x100/UntargetedSDP, cf. task-dynamic-conic-bundle.md) : resolve_dualized()
    reutilise la MEME tache MOSEK pour toutes les resolutions successives d'un run
    (calibration + chaque appel oracle). Sur au moins un cas severement mal
    conditionne, resoudre repetement des objectifs differents sur cette tache
    reutilisee a produit une DERIVE confirmee : le MEME theta=0, resolu une
    premiere fois (calibration, ~-1.9029) puis une seconde fois apres seulement
    UNE resolution intermediaire a theta non-nul (meme tres petit, 5 composantes a
    0.01), redonne une valeur sensiblement differente (~-1.73), et apres plusieurs
    resolutions intermediaires, une valeur totalement incoherente (+4.54,
    superieure a la vraie borne). Confirme PAS du au hot-start MOSEK
    (intpnt_hotstart/intpnt_starting_point restent a leurs defauts, jamais
    modifies), PAS au cache de solution (deletesolution() avant optimize() ne
    change rien), PAS au threading (reproduit identique avec num_threads=1).
    Confirme, en revanche, que des process FRAIS independants (nouvelle tache a
    chaque lancement) redonnent systematiquement ~-1.9029 a chaque fois (observe
    sur 10+ lancements separes cette session). La cause exacte cote MOSEK n'est
    pas identifiee (probable degenerescence/mauvais conditionnement numerique du
    SDP autour de ce theta, amplifie par la reutilisation de la tache) -- mais
    l'effet est reproductible et confirme. CONFIRME SPECIFIQUE aux problemes mal
    conditionnes : le meme test de repetition (resolve_dualized({}) -> probe a
    theta non-nul -> resolve_dualized({}) a nouveau) sur blob_nn_4x10 (petit
    modele, bien conditionne) redonne une valeur EXACTEMENT identique (bit a bit)
    avant/apres -- aucune derive detectee sur un probleme bien conditionne, meme
    apres des centaines de resolutions successives sur la meme tache au cours de
    cette session. Implication : best_h (le max sur tous les points oracle, y
    compris d'eventuelles derives) peut etre un FAUX POSITIF de certification
    specifiquement sur les problemes mal conditionnes -- se mefier d'un best_h
    positif atteint apres plusieurs dizaines d'appels oracle sur la meme tache
    SANS confirmation independante, UNIQUEMENT quand le probleme sous-jacent est
    deja connu pour etre numeriquement difficile (ex. convergence tres lente ou
    stall du solveur classique sur la meme instance).

    Mode dynamique (main.pdf 5.4.2) transpose depuis le mecanisme reel de MIQCR
    (Miqcr-1.0_.../src/solver_sdp_mixed.c::run_conic_bundle_mixed, cf. son propre
    CLAUDE.md section 7) -- PAS le mode dynamique du moteur "python"
    (dynamic_conic_bundle/solver.py), qui redemarre un round de bundle proximal a
    chaque mise a jour du pool. Ici, la dimension du probleme ConicBundle est
    modifiee EN PLACE via cb_append_variables()/cb_reassign_variables(), sans
    jamais detruire/reconstruire le bundle -- le bundle accumule (coupes, poids u)
    est preserve d'une mise a jour du pool a l'autre, exactement comme chez MIQCR.

    all_dualizable_names : pool CANDIDAT complet (deja filtre par `factor` cote
    appelant si besoin, cf. certification_problem.py) -- toutes ces contraintes
    voient leur borne MOSEK relachee des l'appel a setup_dualization() (comme en
    mode statique), mais seul un SOUS-ENSEMBLE recoit effectivement un slot theta_r
    dans le probleme ConicBundle a un instant donne (les autres contribuent 0 a
    l'objectif, exactement absentes du modele -- meme semantique que MIQCR : ni
    dures, ni penalisees).

    Deroulement :
      1. Calibration a theta={} (rien d'actif) -- evalue aussi la violation de
         TOUS les candidats sur ce X initial, sert d'amorce du pool actif (les
         add_batch_size plus violes), a defaut de rien de viole on prend les
         add_batch_size premiers du pool candidat (garantit une dimension >=1).
      2. cb_construct_problem/cb_init_problem avec cette dimension initiale.
      3. Boucle : cb_do_descent_step(), puis a chaque pas :
         - retire du pool actif les contraintes dont |theta_r| < theta_drop_tol
           (cf. cb_get_center) via cb_reassign_variables -- critere sur la
           magnitude de theta plutot que sur le slack primal de MIQCR
           (cb_get_approximate_slacks) : on a deja theta directement, pas besoin
           d'une extraction supplementaire, et c'est le meme critere que le mode
           dynamique du moteur "python" (theta_drop_tol).
         - ajoute les add_batch_size candidats INACTIFS les plus violes (evalues
           sur le dernier X primal connu) via cb_append_variables.
         - s'arrete (pool convergé) si rien n'a ete ajoute ni retire ce tour-ci.
      max_pool_updates borne le nombre de mises a jour du pool (independant de
      max_iter, qui borne les pas de descente ConicBundle) -- pas de limite propre
      sinon si le pool oscille sans jamais se stabiliser.

    N'implemente PAS l'enrichissement multi-sous-gradients de la sonde de coin
    (max_subg_by_point) -- un seul sous-gradient par appel oracle, comme MIQCR.
    Peut etre ajoute plus tard si besoin, en reprenant le meme mecanisme que
    solve_native_conicbundle_dual.

    Retourne (h_best, theta_best, n_iter, stop_reason) -- meme contrat que
    solve_native_conicbundle_dual.
    """
    lib = _load_lib(so_path)

    handler.setup_dualization(all_dualizable_names)
    dualized = handler.get_dualized_constraints_data()
    by_name = {d["name"]: d for d in dualized}
    n_total = len(all_dualizable_names)
    assert n_total == len(dualized), (
        f"get_dualized_constraints_data() a retourne {len(dualized)} entrees, "
        f"attendu {n_total} (all_dualizable_names)"
    )

    minus_inf = lib.cb_get_minus_infinity()
    plus_inf = lib.cb_get_plus_infinity()

    best = {"h": -np.inf, "theta": None}
    state = {"n_calls": 0, "last_X": None}
    rng = np.random.default_rng()  # non utilise pour l'instant (pas d'enrichissement), garde pour coherence future

    h_history: List[float] = []
    best_h_history: List[float] = []
    u_history: List[float] = []
    bundle_size_history: List[int] = []
    pool_size_history: List[int] = []
    n_serious_history: List[int] = []
    n_serious_box = {"n": 0}
    time_history: List[float] = []
    pretreatment_history: List[float] = []
    solve_time_history: List[float] = []

    bundlesize_c = ctypes.c_int(0)
    new_subgrads_c = ctypes.c_int(0)

    csv_file = None
    csv_writer = None
    if print_png and png_save_path:
        csv_save_path = os.path.splitext(png_save_path)[0] + ".csv"
        csv_file = open(csv_save_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "oracle_call", "h", "best_h", "u", "bundle_size", "pool_size",
            "time_s", "pretreatment_s", "mosek_solve_s",
        ])

    def _write_csv_row():
        if csv_writer is None:
            return
        csv_writer.writerow([
            state["n_calls"], h_history[-1], best_h_history[-1], u_history[-1],
            bundle_size_history[-1], pool_size_history[-1],
            time_history[-1], pretreatment_history[-1], solve_time_history[-1],
        ])
        csv_file.flush()

    def _refresh_png():
        if not (print_png and png_save_path and h_history):
            return
        try:
            from dynamic_conic_bundle.plotting import plot_run_diagnostics

            plot_run_diagnostics(
                h_history, u_history, bundle_size_history, png_save_path,
                title=png_title, best_h_history=best_h_history,
                n_serious_history=n_serious_history,
                time_history=time_history,
                pretreatment_history=pretreatment_history,
                solve_time_history=solve_time_history,
                # pool_size_history reutilise le slot step_norm (meme unite "petit entier
                # croissant", pas de panneau dedie pour l'instant) -- cf. TODO ci-dessous.
                step_norm_history=[float(x) for x in pool_size_history],
            )
        except Exception as e:
            print(f"[cb_native] echec de l'ecriture du PNG de diagnostic : {e}")

    # --- Calibration (theta={}) + amorce du pool actif ---
    theta0: Dict[str, float] = {}
    t_calib0 = time.perf_counter()
    res0 = handler.resolve_dualized(theta0)
    calib_elapsed = time.perf_counter() - t_calib0

    active_names: List[str] = []
    effective_initial_weight: Optional[float] = None

    if res0["lb_value"] is not None:
        h0 = res0["lb_value"]
        X0 = res0["X"]
        state["last_X"] = X0
        best["h"], best["theta"] = h0, {}
        state["n_calls"] += 1

        viol0 = {
            name: _dualized_inner_product(by_name[name], X0) - by_name[name]["rhs"]
            for name in all_dualizable_names
        }
        seed = sorted((n for n, g in viol0.items() if g > 0), key=lambda n: -viol0[n])[:add_batch_size]
        if not seed:
            # Rien de viole a theta=0 (cas frequent -- cf. stop_when_positive juste
            # apres) : amorce quand meme avec les premiers candidats du pool pour
            # garantir une dimension ConicBundle >= 1.
            seed = all_dualizable_names[: max(1, min(add_batch_size, n_total))]
        active_names = list(seed)

        if initial_weight == "auto":
            g0_active = np.array([viol0[n] for n in active_names])
            sgnorm0 = float(np.linalg.norm(g0_active))
            effective_initial_weight = sgnorm0 if sgnorm0 > 0.0 else None
        elif initial_weight is not None:
            effective_initial_weight = float(initial_weight)

        if log_every:
            print(f"  [cb_native dynamic calibration] h(0)={h0:.6f} "
                  f"pool_initial={len(active_names)}/{n_total}", flush=True)

        if print_png:
            h_history.append(h0)
            best_h_history.append(best["h"])
            u_history.append(effective_initial_weight if effective_initial_weight is not None else float("nan"))
            bundle_size_history.append(0)
            pool_size_history.append(len(active_names))
            n_serious_history.append(0)
            time_history.append(calib_elapsed)
            pretreatment_history.append(res0.get("pretreatment_time", float("nan")))
            solve_time_history.append(res0.get("solve_time", float("nan")))
            _write_csv_row()
    elif print_level >= 0:
        print(f"[cb_native dynamic] ECHEC de la calibration : resolve_dualized(theta={{}}) "
              f"n'a pas renvoye de solution valide (status={res0['status']}).")

    stop_reason: Optional[str] = None
    if stop_when_positive and best["h"] > positivity_threshold:
        stop_reason = "reached_positivity"
        if print_level >= 0:
            print(f"[cb_native dynamic] h(0)={best['h']:.6f} > positivity_threshold="
                  f"{positivity_threshold} -- arret immediat (stop_when_positive).")
        _refresh_png()
        if csv_file is not None:
            csv_file.close()
        return best["h"], best["theta"], 0, stop_reason

    if not active_names:
        if print_level >= 0:
            print("[cb_native dynamic] ECHEC : aucun pool initial disponible "
                  "(calibration echouee et pool candidat vide).")
        if csv_file is not None:
            csv_file.close()
        return None, None, 0, stop_reason or "oracle_failed"

    # NOTE : agregation primale (cb_subgextp/primaldim) testee et retiree --
    # implementation correcte (callback d'extension verifie fonctionnel sur
    # plusieurs rounds, cf. investigation data_index=59 UntargetedSDP avec
    # dualize=[RLT, ReLU_quad]) mais provoque un SEGFAULT natif, deterministe
    # mais non localise a un round fixe (deplace de round 3 a round 5 en
    # plafonnant max_bundle_size, donc pas une simple histoire de croissance
    # non bornee du bundle), quelque part dans ConicBundle ou MOSEK apres un
    # nombre cumule d'appels oracle sur ce probleme tres mal conditionne et au
    # primal enorme (395366 valeurs, l'aplatissement dense de tous les blocs
    # Pk). Cause racine non identifiee (tierce librairie C, crash non
    # rattrapable cote Python) -- cf. task-dynamic-conic-bundle.md pour le
    # detail de l'investigation. On reste donc SANS extension de sous-gradient :
    # cb_add_function() recoit subgext=None, primaldim=0 (comme avant cette
    # investigation), et cb_reinit_function_model() est appele a chaque maj du
    # pool (cf. plus bas) pour eviter le theta-NaN observe sans ce filet.

    def oracle(function_key, arg_p, relprec, max_subg,
               objective_value_p, n_subgrads_p, subg_values_p, subgradients_p, primal_p):
        state["n_calls"] += 1
        m_now = len(active_names)
        theta_arr = np.ctypeslib.as_array(arg_p, shape=(m_now,))
        if not np.all(np.isfinite(theta_arr)):
            # Observe concretement juste apres une mise a jour du pool
            # (cb_reassign_variables + cb_append_variables) : ConicBundle propose
            # parfois un theta contenant des NaN pour plusieurs dimensions a la fois
            # -- fragilite numerique interne a la librairie sur les dimensions tout
            # juste modifiees, pas une donnee de contrainte corrompue (verifie : les
            # d["value"] correspondants sont des flottants normaux, seul t=nan).
            # Echec propre immediat, sans meme tenter resolve_dualized -- moins cher
            # que de laisser MOSEK rejeter le coefficient resultant (cf. le garde-fou
            # exception ci-dessous, qui reste utile pour d'autres cas).
            if print_level >= 0:
                n_nan = int(np.sum(~np.isfinite(theta_arr)))
                print(f"[cb_native dynamic] theta non fini recu de la librairie "
                      f"({n_nan}/{m_now} composantes) -- echec oracle immediat "
                      f"(n_oracle_calls={state['n_calls']}, pool={m_now}/{n_total}).")
            return 1
        nz_idx = np.nonzero(theta_arr)[0]
        theta = {active_names[r]: float(theta_arr[r]) for r in nz_idx}

        t_call0 = time.perf_counter()
        try:
            res = handler.resolve_dualized(theta)
        except Exception as e:
            # Une exception ici (ex. mosek.Error si theta_r extreme produit un
            # coefficient NaN/Inf dans putbarcblocktriplet) ne doit JAMAIS traverser
            # la frontiere du callback ctypes sans controle -- observe concretement
            # en mode dynamique (nouvelle dimension juste ajoutee par
            # cb_append_variables, premier essai a un theta_r mal calibre pour
            # cette dimension) : "Exception ignored on calling ctypes callback
            # function" laisse le code de retour dans un etat indefini cote C, ce
            # qui a corrompu tout le reste du run. On traite cette exception comme
            # un echec normal de l'oracle (meme convention que lb_value is None).
            if print_level >= 0:
                print(f"[cb_native dynamic] resolve_dualized a leve une exception "
                      f"(n_oracle_calls={state['n_calls']}, pool={m_now}/{n_total}) : {e}")
            return 1
        call_elapsed = time.perf_counter() - t_call0
        if res["lb_value"] is None:
            return 1

        h_val = res["lb_value"]
        if h_val > best["h"]:
            best["h"] = h_val
            best["theta"] = dict(theta)
        state["last_X"] = res["X"]

        g = np.array([
            _dualized_inner_product(by_name[n], res["X"]) - by_name[n]["rhs"]
            for n in active_names
        ])

        if log_every and state["n_calls"] % log_every == 0:
            print(f"  [cb_native dynamic oracle #{state['n_calls']}] h={h_val:.6f} "
                  f"best={best['h']:.6f} pool={m_now}/{n_total}", flush=True)

        if print_png:
            h_history.append(h_val)
            best_h_history.append(best["h"])
            u_history.append(lib.cb_get_last_weight(p))
            lib.cb_get_bundle_values(p, function_key, ctypes.byref(bundlesize_c),
                                      ctypes.byref(new_subgrads_c))
            bundle_size_history.append(bundlesize_c.value)
            pool_size_history.append(m_now)
            n_serious_history.append(n_serious_box["n"])
            time_history.append(call_elapsed)
            pretreatment_history.append(res.get("pretreatment_time", float("nan")))
            solve_time_history.append(res.get("solve_time", float("nan")))
            _write_csv_row()
            if png_every and state["n_calls"] % png_every == 0:
                _refresh_png()

        objective_value_p[0] = -h_val
        n_subgrads_p[0] = 1
        subg_values_p[0] = -h_val
        for r in range(m_now):
            subgradients_p[r] = -g[r]
        return 0

    oracle_cb = CB_FUNCTION_TYPE(oracle)

    m_init = len(active_names)
    lowerb = np.array(
        [minus_inf if by_name[n]["free_sign"] else 0.0 for n in active_names], dtype=np.float64
    )
    upperb = np.full(m_init, plus_inf, dtype=np.float64)

    effective_max_bundle_size = max_bundle_size
    if effective_max_bundle_size is None and bundle_size_factor is not None:
        # Base sur n_total (l'univers candidat), pas sur la dimension active
        # courante -- coherent avec le mode statique (cf. solve_native_conicbundle_dual).
        effective_max_bundle_size = max(1, round(bundle_size_factor * n_total))
        if print_level >= 0:
            print(f"[cb_wrapper dynamic] bundle_size_factor={bundle_size_factor} x "
                  f"n_total={n_total} => max_bundle_size={effective_max_bundle_size}")

    p = lib.cb_construct_problem(0)
    if not p:
        raise RuntimeError("cb_construct_problem a echoue")
    try:
        rc = lib.cb_init_problem(p, m_init, lowerb.ctypes.data_as(_dbl_p), upperb.ctypes.data_as(_dbl_p))
        assert rc == 0, f"cb_init_problem: rc={rc}"

        function_key = ctypes.c_void_p(1)
        rc = lib.cb_add_function(p, function_key, oracle_cb, None, 0)
        assert rc == 0, f"cb_add_function: rc={rc}"

        if effective_max_bundle_size is not None:
            rc = lib.cb_set_max_bundlesize(p, function_key, effective_max_bundle_size)
            assert rc == 0, f"cb_set_max_bundlesize: rc={rc}"

        lib.cb_set_term_relprec(p, term_relprec)
        lib.cb_set_eval_limit(p, eval_limit)
        lib.cb_set_print_level(p, print_level)
        lib.cb_set_active_bounds_fixing(p, 1 if active_bounds_fixing else 0)

        if effective_initial_weight is not None:
            rc = lib.cb_set_next_weight(p, effective_initial_weight)
            assert rc == 0, f"cb_set_next_weight: rc={rc}"

        n_iter = 0
        n_pool_updates = 0
        any_successful_descent_step = False
        # Historique des etats de pool deja visites (anti-cycle, cf. plus bas) --
        # l'amorce initiale compte comme le premier etat visite.
        seen_pools = {frozenset(active_names)}
        # IMPORTANT : ne JAMAIS gater l'appel a cb_do_descent_step() sur
        # cb_termination_code(p). termination_code reflete le "terminate" interne de
        # BundleSolver (bundle.cxx), remis a 0 UNIQUEMENT au debut de inner_loop()
        # (donc UNIQUEMENT au prochain cb_do_descent_step() lui-meme) -- aucun des
        # appels de mise a jour du pool (cb_append_variables/cb_reassign_variables/
        # cb_reinit_function_model/cb_clear_fail_counts, teste et confirme sans
        # effet) ne le reinitialise. Avec `while cb_termination_code(p)==0 and ...`
        # comme garde de boucle (suivant naivement le pattern documente dans
        # cb_cinterface.h, qui suppose une dimension FIXE), un premier
        # cb_do_descent_step() concluant "precision relative atteinte" (code=1) pour
        # le pool initial empechait TOUT appel suivant, meme juste apres avoir
        # agrandi le pool -- observe concretement : n_iter=1, best_h strictement egal
        # a la valeur de calibration, le pool etendu a 50/1302 jamais explore.
        # Le fix : la boucle externe ne s'arrete que sur n_iter>=max_iter, un echec
        # (rc!=0) ou -- seul cas ou termination_code fait foi -- quand le pool n'a
        # PAS change ce tour-ci ET que termination_code!=0 (double confirmation :
        # converge sur CE pool, et rien de plus a y ajouter/retirer).
        while n_iter < max_iter:
            rc = lib.cb_do_descent_step(p)
            n_iter += 1
            if rc == 0:
                any_successful_descent_step = True
            if rc != 0:
                if print_level >= 0:
                    print(f"[cb_native dynamic] cb_do_descent_step a echoue : rc={rc} "
                          f"(n_iter={n_iter} n_oracle_calls={state['n_calls']})")
                break

            if print_png and h_history:
                cand_h = -lib.cb_get_candidate_value(p)
                obj_h = -lib.cb_get_objval(p)
                if abs(cand_h - obj_h) <= 1e-12 * (abs(obj_h) + 1.0):
                    n_serious_box["n"] += 1
                    n_serious_history[-1] = n_serious_box["n"]

            if stop_when_positive and best["h"] > positivity_threshold:
                stop_reason = "reached_positivity"
                break

            term_code = lib.cb_termination_code(p)

            if n_pool_updates >= max_pool_updates:
                if term_code != 0:
                    if print_level >= 0:
                        print(f"[cb_native dynamic] max_pool_updates={max_pool_updates} "
                              f"atteint et termination_code={term_code} -- arret "
                              f"(pool fige a {len(active_names)}/{n_total}).")
                    break
                if print_level >= 0:
                    print(f"[cb_native dynamic] max_pool_updates={max_pool_updates} atteint "
                          f"-- pool fige a {len(active_names)}/{n_total}, la boucle de "
                          f"descente continue sans plus modifier le pool.")
                continue

            # --- Mise a jour du pool (equivalent MIQCR, sans redemarrer le bundle) ---
            m_now = len(active_names)
            center = np.zeros(m_now, dtype=np.float64)
            rc_c = lib.cb_get_center(p, center.ctypes.data_as(_dbl_p))
            if rc_c != 0:
                break  # pas de centre disponible (ne devrait pas arriver si rc==0 ci-dessus)

            keep_mask = np.abs(center) >= theta_drop_tol
            dropped_names = [active_names[i] for i in range(m_now) if not keep_mask[i]]

            X_last = state["last_X"]
            most_violated: List[str] = []
            if X_last is not None:
                active_set = set(active_names)
                inactive_names = [n for n in all_dualizable_names if n not in active_set]
                if inactive_names:
                    viol = {
                        n: _dualized_inner_product(by_name[n], X_last) - by_name[n]["rhs"]
                        for n in inactive_names
                    }
                    most_violated = sorted(
                        (n for n, g in viol.items() if g > 0), key=lambda n: -viol[n]
                    )[:add_batch_size]

            if not dropped_names and not most_violated:
                # Pool stable (rien a ajouter/retirer) -- n'arreter que si CE pool
                # (final, puisqu'il ne changera plus) est LUI-MEME converge
                # (term_code!=0). S'il ne l'est pas encore, il reste du progres a
                # faire dessus : continuer plutot que de sortir prematurement.
                if term_code != 0:
                    if print_level >= 0:
                        print(f"[cb_native dynamic] pool converge (rien a ajouter/retirer) "
                              f"et termination_code={term_code} a n_iter={n_iter}, "
                              f"pool={len(active_names)}/{n_total}.")
                    break
                continue

            # Point mort : rien a ajouter (most_violated vide) ET le retrait
            # voudrait vider TOUT le pool courant (len(dropped_names)==m_now, donc
            # n_keep_old==0 plus bas) -- le garde-fou dimension-0 empechera ce
            # retrait, et comme rien d'autre ne change (aucun ajout, active_names
            # identique), le round SUIVANT redetectera EXACTEMENT la meme situation
            # -- boucle infinie non productive si on continue. Observe concretement
            # (data_index=23) : 19 rounds identiques a bruler tout le budget
            # max_pool_updates pour rien. Comme le X* courant satisfait deja TOUTES
            # les contraintes RLT candidates (rien de viole) et qu'aucune des
            # contraintes actives n'est plus necessaire (theta_r->0 pour toutes),
            # c'est une vraie convergence : la famille RLT seule n'apporte plus
            # rien de plus dans cette region -- pas une erreur, juste la limite de
            # ce qu'on peut esperer de ces coupes ici. Arret propre plutot que
            # d'epuiser max_pool_updates en boucle.
            if not most_violated and len(dropped_names) == m_now:
                if print_level >= 0:
                    print(f"[cb_native dynamic] point mort : aucune contrainte RLT "
                          f"candidate violee et les {m_now} contraintes actives ne "
                          f"sont plus necessaires (theta_r->0) -- rien ne peut plus "
                          f"changer, arret a n_iter={n_iter}, "
                          f"pool={len(active_names)}/{n_total}.")
                break

            # Anti-cycle : certaines contraintes ont un dual quasi-nul (theta_r->0,
            # donc droppees) alors que leur retrait change immediatement X* au point
            # de les re-violer (le critere de drop, sur theta, et le critere d'ajout,
            # sur la violation primale de X*, peuvent se contredire des le round
            # suivant) -- observe concretement (data_index=59, UntargetedSDP) : un
            # cycle a 2 etats, +50 -11 puis +11 -50 en alternance, jamais stable,
            # best_h fige sur toute la sequence (aucun progres), jusqu'a epuiser
            # max_pool_updates. Si le pool resultant de CE round a deja ete visite
            # precedemment, on est en train de boucler -- arret propre plutot que de
            # bruler le budget en repetant indefiniment le meme cycle.
            prospective_pool = frozenset(
                active_names[i] for i in range(m_now) if keep_mask[i]
            ) | frozenset(most_violated)
            if prospective_pool in seen_pools:
                if print_level >= 0:
                    print(f"[cb_native dynamic] cycle detecte : le pool resultant de "
                          f"ce round (+{len(most_violated)} -{len(dropped_names)}) a "
                          f"deja ete visite -- arret pour eviter une oscillation "
                          f"infinie, n_iter={n_iter}, pool avant maj="
                          f"{len(active_names)}/{n_total}.")
                break
            seen_pools.add(prospective_pool)

            n_pool_updates += 1

            # IMPORTANT (x2) :
            # 1. Ajouter AVANT de retirer, jamais l'inverse. Observe concretement
            #    (core dump, mnist-9x100/UntargetedSDP data_index=23) :
            #    cb_reassign_variables() vers une dimension 0 (tout le pool actif
            #    tombe sous theta_drop_tol simultanement) suivi de cb_append_variables()
            #    pour la faire remonter fait planter ConicBundle -- la librairie ne
            #    supporte visiblement pas de passer par une dimension nulle. En
            #    ajoutant d'abord (la dimension ne peut que croitre depuis un etat non
            #    nul, toujours sur), le retrait qui suit n'a plus jamais besoin de
            #    viser 0 tant qu'au moins une contrainte est ajoutee ou survit.
            # 2. Mettre a jour `active_names` AVANT (pas apres) chaque appel C, pas
            #    apres. Observe concretement : cb_append_variables() declenche en
            #    interne un recompute_center() (MatFCBSolver.cxx, si un centre est
            #    deja disponible) qui RAPPELLE notre oracle() de facon SYNCHRONE,
            #    DURANT l'appel C lui-meme (visible dans les logs : "avant
            #    cb_append_variables" / "oracle #8 ..." / "apres cb_append_variables").
            #    Si `active_names` n'est mis a jour qu'apres coup, l'oracle lit alors
            #    m_now=ancienne dimension alors que la librairie l'appelle deja dans
            #    la NOUVELLE dimension (etendue) -- theta_arr est lu tronque (pas
            #    hors-bornes, juste incomplet) ET subgradients_p n'est ecrit que pour
            #    les anciennes composantes, laissant les nouvelles a du contenu
            #    memoire NON INITIALISE (garbage). Resultat concret observe : un
            #    h aberrant (+4.54 au lieu de ~-1.90, le vrai centre) enregistre a
            #    tort comme best_h -- FAUX POSITIF de certification (is_robust=True
            #    base sur une evaluation corrompue, pas un vrai resultat).
            if most_violated:
                n_append = len(most_violated)
                lowerb_new = np.array(
                    [minus_inf if by_name[n]["free_sign"] else 0.0 for n in most_violated],
                    dtype=np.float64,
                )
                active_names.extend(most_violated)
                rc_a = lib.cb_append_variables(
                    p, n_append, lowerb_new.ctypes.data_as(_dbl_p), None
                )
                assert rc_a == 0, f"cb_append_variables: rc={rc_a}"

            if dropped_names:
                # keep_mask/m_now se referent a l'etat AVANT l'ajout ci-dessus -- les
                # variables tout juste ajoutees (en dernieres positions, cf. doc
                # cb_append_variables "always in last positions") sont TOUJOURS
                # gardees ici, jamais visees par le retrait de ce round.
                n_keep_old = int(np.sum(keep_mask))
                n_keep_total = n_keep_old + len(most_violated)
                if n_keep_total == 0:
                    # Tout le pool actif (avant ajout) est sous theta_drop_tol ET
                    # rien n'a ete ajoute -- retirer viserait une dimension 0, meme
                    # probleme que ci-dessus. On laisse le pool actuel intact ce
                    # round-ci plutot que de crasher ; il sera reevalue au round
                    # suivant avec un nouveau centre.
                    if print_level >= 0:
                        print(f"[cb_native dynamic] retrait ignore ce round (viserait "
                              f"une dimension 0) -- pool laisse intact a {m_now}.")
                else:
                    keep_idx = np.array(
                        [i for i in range(m_now) if keep_mask[i]]
                        + list(range(m_now, m_now + len(most_violated))),
                        dtype=np.int32,
                    )
                    # Mise a jour de active_names AVANT l'appel C (meme raison que pour
                    # cb_append_variables ci-dessus) -- active_names[0:m_now] est encore
                    # le prefixe pre-ajout (extend() n'a touche que la fin), donc cette
                    # reconstruction est correcte meme apres l'extension deja faite.
                    active_names[:] = (
                        [active_names[i] for i in range(m_now) if keep_mask[i]] + most_violated
                    )
                    rc_r = lib.cb_reassign_variables(
                        p, len(keep_idx), keep_idx.ctypes.data_as(_int_p)
                    )
                    assert rc_r == 0, f"cb_reassign_variables: rc={rc_r}"

            if dropped_names or most_violated:
                # Force le recalcul du modele de fonction plutot que de compter sur
                # l'hypothese "valeurs a zero => rien a recalculer" documentee pour
                # cb_append_variables/cb_reassign_variables : nos variables retirees
                # sont seulement PROCHES de zero (|theta_r| < theta_drop_tol), pas
                # EXACTEMENT zero -- observe concretement : des theta contenant des
                # NaN sur plusieurs composantes a la fois juste apres une mise a jour
                # du pool (cf. investigation blob_4x10), plausible violation de cette
                # hypothese laissant le modele de bundle dans un etat incoherent.
                # Un vrai fix (cb_subgextp + primaldim>0, extension du sous-gradient
                # sans tout effacer) a ete implemente et VALIDE fonctionnellement
                # (callback d'extension correct, verifie sur plusieurs rounds), mais
                # provoque un SEGFAULT natif deterministe apres un nombre cumule
                # d'appels oracle sur ce probleme tres mal conditionne et au primal
                # enorme (395366 valeurs) -- cf. investigation data_index=59
                # UntargetedSDP, dualize=[RLT, ReLU_quad]. Cause racine non
                # identifiee (tierce librairie C, non rattrapable cote Python) --
                # on revient donc a ce reinit systematique (sans cb_subgextp), plus
                # lent/sous-optimal (le bundle est perdu a chaque maj du pool) mais
                # sans risque de crash silencieux.
                rc_reinit = lib.cb_reinit_function_model(p, function_key)
                if rc_reinit != 0 and print_level >= 0:
                    print(f"[cb_native dynamic] cb_reinit_function_model a echoue : "
                          f"rc={rc_reinit} (round {n_pool_updates})")

                # Reinitialise les compteurs d'echec numeriques (cb_cinterface.h :
                # "may be useful if this caused premature termination" -- fail counts
                # sur echecs numeriques/modele, PAS le flag "precision relative
                # atteinte", teste et confirme sans effet sur celui-ci). Gardee par
                # precaution pour les echecs numeriques genuins ; le vrai fix pour la
                # terminaison prematuree constatee (n_iter=1, pool jamais explore
                # apres agrandissement) est structurel -- cf. le commentaire au debut
                # de la boucle while externe (ne plus gater cb_do_descent_step() sur
                # cb_termination_code()).
                lib.cb_clear_fail_counts(p)

            if print_level >= 0:
                print(f"[cb_native dynamic] round {n_pool_updates}: "
                      f"+{len(most_violated)} -{len(dropped_names)} "
                      f"(pool={len(active_names)}/{n_total})")

        center = np.zeros(len(active_names), dtype=np.float64)
        lib.cb_get_center(p, center.ctypes.data_as(_dbl_p))
        f_center = lib.cb_get_objval(p)
        h_center = -f_center
        theta_center = {active_names[r]: float(center[r]) for r in range(len(active_names))}

        if print_level >= 0:
            code = lib.cb_termination_code(p)
            print(f"[cb_native dynamic] n_iter={n_iter} n_oracle_calls={state['n_calls']} "
                  f"n_pool_updates={n_pool_updates} pool_final={len(active_names)}/{n_total} "
                  f"termination_code={code} h_center={h_center:.6f} best_h={best['h']:.6f} "
                  f"stop_reason={stop_reason}")

        _refresh_png()
    finally:
        p_holder = ctypes.c_void_p(p)
        lib.cb_destruct_problem(ctypes.byref(p_holder))
        if csv_file is not None:
            csv_file.close()

    if best["h"] == -np.inf:
        return None, None, n_iter, stop_reason or "oracle_failed"

    if not any_successful_descent_step:
        return best["h"], best["theta"], n_iter, stop_reason

    if best["h"] > h_center:
        return best["h"], best["theta"], n_iter, stop_reason
    return h_center, theta_center, n_iter, stop_reason
