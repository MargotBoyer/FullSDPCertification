"""
DynamicConicBundleSolver — pilote la conic bundle method (main.pdf, section 5.4) en
appelant SDPSolver en boucle.

Statique (5.4.1, Algorithme 2) : le pool de contraintes dualisées (celles taguées
ConstraintRole.DUALIZABLE, cf. handler/constraints.py) est fixé dès le départ et ne
change plus — une seule boucle proximal-bundle jusqu'à convergence (voir
proximal_master.py pour la convention de signe miroir utilisée ici et la
justification du test sérieux/nul).

Dynamique (5.4.2) : démarre avec un pool de contraintes dualisées vide (round 0 =
SDP dur classique, sans aucune relaxation lagrangienne), puis alterne :
  1. une boucle proximal-bundle à pool fixe (comme le mode statique) jusqu'à
     convergence de ce round,
  2. une mise à jour du pool : ajout des `add_batch_size` contraintes DUALIZABLE pas
     encore actives les plus violées (plus grand sous-gradient positif au point
     courant), retrait de celles dont |theta_r| < theta_drop_tol.
Le bundle est conservé (avec padding sur les nouveaux noms, cf. ProximalBundle)
d'un round à l'autre pour ne pas repartir de zéro. Le nombre de rounds est borné
par `max_rounds` (pas de limite propre sinon — trouvé en pratique lors des premiers
tests à grande échelle, cf. task-dynamic-conic-bundle.md) ; `max_iter` ne borne que
la boucle interne à pool fixe, par round.

N'appelle jamais MOSEK directement : uniquement via les méthodes publiques de
SDPSolver (build_model, setup_dualization, resolve_dualized, teardown_dualization,
get_dualized_constraints_data, get_constraint_dualization_data,
get_dualizable_constraint_names) — cf. contrainte non-négociable de
task-dynamic-conic-bundle.md.

TODO Phase 3 : CHORDAL_DECOMPOSITION=True (plusieurs blocs P_k) n'est pas supporté
(cf. l'assertion dans MosekClassicHandler.setup_dualization) — l'interaction entre la
décomposition chordale et la dualisation des contraintes RLT/McCormick inter-blocs
reste à concevoir.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from .proximal_master import (
    ProximalBundle,
    serious_null_step_test,
    solve_master_problem,
)
from .subgradient import compute_subgradients

logger_cb = logging.getLogger("Dynamic_conic_bundle_logger")


class DynamicConicBundleSolver:
    def __init__(
        self,
        sdp_solver,
        cuts: Dict,
        dynamic: bool = False,
        max_iter: int = 200,
        C1: float = 1e-4,
        C2: float = 0.1,
        proximal_u_init: float = 1.0,
        u_min: float = 1e-4,
        u_max: float = 1e6,
        u_max_factor: float = 1e3,
        max_bundle_size: Optional[int] = None,
        theta_drop_tol: float = 1e-8,
        add_batch_size: int = 50,
        max_rounds: int = 100,
        log_theta_every_n_iter: int = 1,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        sdp_solver : SDPSolver
            Instance déjà construite (network, epsilon, x, ytrue, ... déjà passés au
            constructeur), pas encore résolue. build_model() y sera appelé.
        cuts : Dict
            Ensemble de coupes à construire (comme pour SDPSolver.run_optimization),
            doit inclure "RLT" pour que des contraintes DUALIZABLE existent.
        dynamic : bool
            False = algorithme statique (5.4.1, pool fixe) ; True = dynamique (5.4.2).
        max_iter : int
            Nombre max d'itérations proximal-bundle *par round*.
        C1, C2 : float
            Cf. proximal_master.serious_null_step_test.
        proximal_u_init : float
            Valeur de u avant la toute première calibration (cf. _calibrate_u) — n'est
            en pratique quasiment jamais utilisée telle quelle : dès le premier
            sous-gradient calculé (avant le premier appel au master problem), u est
            recalibré sur ||g_center|| (comme hkweight.cxx::init() dans ConicBundle),
            car une constante fixe est mal calée par rapport à l'échelle réelle des
            sous-gradients du problème (diagnostiqué le 2026-08-13, cf.
            task-dynamic-conic-bundle.md — un u=1.0 fixe provoquait un survol massif
            dès la 1ère itération sur blob_4x10). Ne sert de valeur réelle que si le
            tout premier sous-gradient est quasi nul (norme < 1e-10). Ensuite mis à
            jour adaptativement pendant la boucle (cf. _update_u).
        u_min, u_max : float
            Bornes absolues (indépendantes de l'échelle du problème) de l'ajustement
            adaptatif de u.
        u_max_factor : float
            Plafond *relatif* à l'échelle calibrée (cf. _calibrate_u) réellement
            utilisé en pratique : `min(u_max, u_max_factor * u_calibré)`. Nécessaire
            car `u_max` absolu (1e6 par défaut) est totalement déconnecté de
            l'échelle réelle du problème — diagnostiqué le 2026-08-13 sur
            data_index=2/blob_4x10 (cf. task-dynamic-conic-bundle.md, PNG de
            diagnostic) : u grimpait à 1e6 en ~25 itérations puis restait bloqué là
            pour les ~475 itérations restantes (aucun pas sérieux ne pouvait plus
            jamais se produire, le pas `theta_new-theta_center ≈ g/u` devenant
            négligeable — cercle vicieux, u ne peut redescendre qu'après des pas
            sérieux). Or `theta*` (vrai optimum, vérifié via les duaux MOSEK) avait
            des composantes jusqu'à 47 pour un `u` calibré initial de l'ordre de 17 —
            un facteur 1e4-1e5 entre `u_max` absolu et l'échelle calibrée garantissait
            le blocage. `u_max_factor=1e3` (défaut) laisse 3 ordres de grandeur de
            marge au-dessus de l'échelle calibrée, largement suffisant pour un
            problème plus difficile, sans dériver vers des pas numériquement nuls.
        max_bundle_size : Optional[int]
            Nombre max de coupes conservées dans le bundle (FIFO) — cf.
            ProximalBundle.max_size. None (défaut) = pas de cap : le nombre de
            coupes est de toute façon borné par max_iter (une coupe par
            itération), donc un cap fixe ne fait qu'ajouter un plafond
            artificiel, potentiellement bien plus restrictif que dim(theta)
            (= nb de contraintes dualisées) et source d'arrêt prématuré du
            bundle (modèle sous-déterminé — cf. task-dynamic-conic-bundle.md,
            "Analyse statique vs dynamique"). Fixer un entier reste utile pour
            borner le coût du master problem QP à grande échelle.
        theta_drop_tol : float
            Mode dynamique seulement : retire une contrainte active si |theta_r| en
            dessous de ce seuil au round suivant.
        add_batch_size : int
            Mode dynamique seulement : nombre de contraintes ajoutées par round.
        max_rounds : int
            Mode dynamique seulement : nombre max de rounds (ajout/retrait de
            contraintes) — sans cette borne, la boucle externe de _solve_dynamic
            n'a aucune limite propre (contrairement à max_iter qui ne borne que la
            boucle interne, par round) : avec un grand nombre de contraintes
            dualisables (ex. RLT sur un réseau réaliste) ou en cas d'oscillation
            ajout/retrait, le nombre de rounds pouvait être illimité. Si atteint,
            le résultat retourné reste une borne inférieure valide (juste pas
            garantie convergée sur l'ensemble des contraintes dualisables).
        log_theta_every_n_iter : int
            Fréquence (en itérations) d'enregistrement dans theta_history (0 = jamais).
        """
        self.sdp_solver = sdp_solver
        self.cuts = cuts
        self.dynamic = dynamic
        self.max_iter = max_iter
        self.C1 = C1
        self.C2 = C2
        self.u = proximal_u_init
        self.u_min = u_min
        self.u_max = u_max
        self.u_max_factor = u_max_factor
        self._u_calibration_base: Optional[float] = None
        self.max_bundle_size = max_bundle_size
        self.theta_drop_tol = theta_drop_tol
        self.add_batch_size = add_batch_size
        self.max_rounds = max_rounds
        self.log_theta_every_n_iter = log_theta_every_n_iter
        self.verbose = verbose
        self._regime = 0
        self._u_calibrated = False

        self.theta_history: List[Dict[str, float]] = []
        self.lb_history: List[float] = []
        self.u_history: List[float] = []
        self.bundle_size_history: List[int] = []
        self.n_iter = 0

    def get_current_thetas(self) -> Dict[str, float]:
        """Derniers multiplicateurs {constraint_name: theta_r} enregistrés."""
        return self.theta_history[-1] if self.theta_history else {}

    def _log_iteration(self, theta: Dict[str, float], h_value: float, bundle_size: int):
        self.lb_history.append(h_value)
        self.u_history.append(self.u)
        self.bundle_size_history.append(bundle_size)
        if self.log_theta_every_n_iter and self.n_iter % self.log_theta_every_n_iter == 0:
            self.theta_history.append(dict(theta))
        if self.verbose:
            logger_cb.info(
                "iter %d: h(theta)=%.6f  u=%.4g  bundle_size=%d", self.n_iter, h_value, self.u, bundle_size
            )

    def save_diagnostics_png(self, save_path: str, title: Optional[str] = None):
        """Écrit le PNG de diagnostic (h(theta)/u/taille du bundle vs itération) du run
        courant — cf. plotting.py. À appeler après solve()."""
        from .plotting import plot_run_diagnostics

        plot_run_diagnostics(
            self.lb_history, self.u_history, self.bundle_size_history, save_path, title=title
        )

    def _calibrate_u(self, g_center: Dict[str, float]):
        """Calibre `u` sur l'échelle réelle du problème via la norme du sous-gradient,
        au lieu de s'en tenir à `proximal_u_init` (une constante arbitraire) pour toute
        la durée du run.

        Porté de `hkweight.cxx::init()`/`prob_changed()` (ConicBundle, Helmberg-Kiwiel —
        cf. task-dynamic-conic-bundle.md, "Étude de la librairie MIQCR/ConicBundle") :
        `weight = max(||subgradient||, plancher)` à l'initialisation, puis
        `weight = max(weight, ||subgradient||)` (ne peut que grandir) à chaque
        changement de dimension de `theta` (mode dynamique : le pool de contraintes
        dualisées grossit d'un round à l'autre, cf. `prob_changed`, appelée quand "des
        variables ont été ajoutées ou supprimées").

        Diagnostiqué nécessaire (2026-08-13) après avoir constaté que `data_index=2,
        target=0` (blob_4x10) reste bloqué sur `theta=0` même après le fix de
        `_update_u` : le tout premier sous-gradient a des composantes jusqu'à 2.06 sur
        450 dimensions, donc `||g_center||` est largement supérieur à
        `proximal_u_init=1.0` — le tout premier pas d'essai (`theta_new ≈ g_center/u`)
        est démesurément grand (`s_predicted=26.4` prédit pour `h_center=-3.33`),
        provoquant un survol quasi certain dès la 1ère itération, que la règle de
        croissance de `u` ensuite corrige beaucoup trop lentement pour rattraper sur le
        budget `max_iter` disponible.
        """
        norm = math.sqrt(sum(v * v for v in g_center.values()))
        if norm < 1e-10:
            return
        if not self._u_calibrated:
            self.u = max(self.u_min, norm)
            self._u_calibrated = True
            self._u_calibration_base = self.u
        else:
            self._u_calibration_base = max(self._u_calibration_base, norm)
            self.u = min(self._effective_u_max(), max(self.u, norm))

    def _effective_u_max(self) -> float:
        """Plafond de `u` réellement appliqué par `_update_u` : `u_max` absolu, ou le
        plafond relatif `u_max_factor * u_calibré` si plus restrictif — cf. docstring
        du paramètre `u_max_factor` de `__init__`."""
        if self._u_calibration_base is None:
            return self.u_max
        return min(self.u_max, self.u_max_factor * self._u_calibration_base)

    def _update_u(self, is_serious_step: bool, predicted_increase: float, actual_increase: float):
        """Mise à jour adaptative du paramètre proximal u (absente d'Algorithme 2 tel
        qu'écrit — u y est un paramètre fixe donné en entrée — mais nécessaire en
        pratique : un u constant mal calé convergeait très lentement sur un réseau
        plus gros que blob_1x2, cf. task-dynamic-conic-bundle.md).

        Portage adapté de la règle de référence de ConicBundle (Helmberg-Kiwiel,
        `hkweight.cxx::descent_update`/`nullstep_update` — cf. task-dynamic-conic-bundle.md,
        "Bug E" et "Étude de la librairie MIQCR/ConicBundle"), après diagnostic d'un bug de
        convergence prématurée causé par l'ancienne règle (1 seul pas nul doublait u
        immédiatement, 3 pas sérieux *consécutifs* requis pour le diviser par 2 — un
        ratchet presque irréversible vers u_max en quelques pas nuls : observé sur
        data_index=93/blob_4x10, 0/57 pas sérieux, u saturé dès l'itération ~21).

        La vraie règle HK est volontairement asymétrique dans l'autre sens : facile de
        faire baisser u (une formule adaptative s'applique dès le 2e pas sérieux
        consécutif, pas besoin d'une série de 3), difficile de le faire monter (il faut
        au moins 4 pas nuls consécutifs, et la hausse est plafonnée à x10 par mise à
        jour plutôt qu'un doublement systématique). `self._regime` remplace l'ancien
        compteur unidirectionnel : signé, positif = nb de pas sérieux consécutifs,
        négatif = nb de pas nuls consécutifs, remis à ±1 dès que u change réellement
        (comme `iweight` dans hkweight.cxx).

        La formule de décroissance `u_new = 2*u*(1 - actual/predicted)` est reprise
        telle quelle de `descent_update` (mathématiquement bien fondée : ĥ est un
        majorant de h, donc predicted_increase >= actual_increase >= C2*predicted_increase
        > 0 dès qu'un pas est sérieux, ce qui garantit u_new dans [0, 2u)). Le point de
        bascule (où u_new == u) est le ratio de référence _U_SHRINK_REF=0.5 — repris du
        défaut `mR=0.5` de `BundleHKweight`, volontairement DÉCOUPLÉ de C2 (seuil de
        classification pas sérieux/nul, souvent << 0.5 dans nos configs, ex. 0.1) :
        réutiliser C2 comme point de bascule ferait *grossir* u sur des pas tout juste
        sérieux (ratio proche de C2), ce qui serait contraire à l'intention. `lin_approx`
        (le signal de qualité de modèle utilisé côté HK pour décider la hausse) n'est
        pas repris ici — jugé portable mais pas nécessaire pour ce fix, cf.
        task-dynamic-conic-bundle.md.

        Sonde périodique (2026-08-13, "Bug F" — cf. task-dynamic-conic-bundle.md) :
        la règle ci-dessus ne peut faire redescendre u qu'après un pas sérieux — mais
        si u devient assez grand pour que TOUS les pas soient nuls (le pas
        `theta_new-theta_center ≈ g/u` devient négligeable), aucun pas sérieux ne peut
        plus jamais se produire, donc u ne peut plus jamais redescendre : un
        verrouillage définitif, observé empiriquement sur data_index=2/blob_4x10 (u
        grimpe au plafond en ~25 itérations puis y reste bloqué pour les ~475
        itérations restantes du budget, `h(theta)` parfaitement plat) — persiste
        même après avoir abaissé le plafond via `u_max_factor`, seul le palier
        atteint change, pas le verrouillage lui-même. Au-delà de
        `_U_RESET_STREAK` pas nuls consécutifs, on force donc une redescente de u
        vers l'échelle calibrée initiale (`_u_calibration_base`) pour retester un pas
        plus grand plutôt que de rester figé — ce n'est pas dans hkweight.cxx (leur
        architecture à coupe agrégée unique + `active_bounds_fixing` gère
        différemment ce cas), c'est une addition pragmatique pour notre bundle complet
        résolu via cvxpy."""
        _U_SHRINK_REF = 0.5  # mR de hkweight.cxx (BundleHKweight::BundleHKweight(Real mRin=.5))
        _U_GROWTH_STREAK = 4  # nb de pas nuls consécutifs requis avant de faire grossir u (iweight<-3 côté HK)
        _U_RESET_STREAK = 20  # nb de pas nuls consécutifs au-delà duquel on force une redescente (sonde)

        oldweight = self.u
        if is_serious_step:
            if self._regime > 0:
                ratio = actual_increase / predicted_increase
                # factor(ratio=_U_SHRINK_REF)=1 (u inchangé), factor(ratio=1)=2*_U_SHRINK_REF-1
                # (=0 pour _U_SHRINK_REF=0.5, exactement la formule 2*(1-ratio) de hkweight.cxx) ;
                # max(u_min,...) protège contre un factor négatif si ratio>1 (bruit numérique,
                # ne devrait pas arriver puisque predicted_increase >= actual_increase par
                # construction — ĥ majorant de h).
                factor = 1.0 + 2.0 * (_U_SHRINK_REF - ratio)
                self.u = max(self.u_min, self.u * factor)
            self._regime = self._regime + 1 if self._regime >= 0 else 1
            if self.u < oldweight:
                self._regime = 1
        else:
            if self._u_calibration_base is not None and -self._regime > _U_RESET_STREAK:
                self.u = self._u_calibration_base
                self._regime = -1
                logger_cb.info(
                    "u bloqué depuis %d pas nuls consécutifs (aucun pas sérieux possible pour le "
                    "faire redescendre) — sonde : redescente forcée vers l'échelle calibrée u=%.4g.",
                    _U_RESET_STREAK, self.u,
                )
                return
            if -self._regime > _U_GROWTH_STREAK:
                self.u = min(self._effective_u_max(), 10.0 * self.u)
            self._regime = self._regime - 1 if self._regime <= 0 else -1
            if self.u > oldweight:
                self._regime = -1

    def solve(self) -> float:
        """Construit le modèle et lance l'algorithme statique ou dynamique.
        Retourne la borne inférieure certifiée finale."""
        self.sdp_solver.build_model(self.cuts)
        all_dualizable = self.sdp_solver.get_dualizable_constraint_names()

        if not all_dualizable:
            logger_cb.warning(
                "Aucune contrainte DUALIZABLE trouvée (cuts=%s) — la dynamic conic "
                "bundle method n'a rien à dualiser, résultat identique à un SDP classique.",
                self.cuts,
            )

        if not self.dynamic:
            lb, _ = self._solve_static(all_dualizable)
        else:
            lb, _ = self._solve_dynamic(all_dualizable)

        self.sdp_solver.teardown_dualization()
        return lb

    # ------------------------------------------------------------------
    # Round proximal-bundle à pool de contraintes dualisées fixe (coeur commun aux
    # deux modes — Algorithme 2, lignes 1-16).
    # ------------------------------------------------------------------

    def _run_bundle_round(
        self,
        active_names: List[str],
        theta_init: Dict[str, float],
        warm_bundle: Optional[ProximalBundle] = None,
    ):
        """Exécute la boucle proximal-bundle (Algorithme 2) jusqu'à convergence (au
        sens de C1) ou max_iter, à pool `active_names` fixe. Retourne
        (theta_center, h_center, X_center, bundle)."""
        self.sdp_solver.setup_dualization(active_names)

        bundle = warm_bundle if warm_bundle is not None else ProximalBundle(names=list(active_names), max_size=self.max_bundle_size)
        bundle.names = list(active_names)  # le pool a pu grandir depuis warm_bundle

        theta = dict(theta_init)
        for name in active_names:
            theta.setdefault(name, 0.0)

        result = self.sdp_solver.resolve_dualized(theta)
        assert result["lb_value"] is not None, (
            f"resolve_dualized a échoué au point initial theta=0 (status={result['status']}) "
            "— le sous-problème SDP sans les contraintes dualisées devrait toujours être faisable "
            "(cf. bloc impératif, task-dynamic-conic-bundle.md)."
        )
        h_center = result["lb_value"]
        X_center = result["X"]
        g_center = compute_subgradients(self.sdp_solver.get_dualized_constraints_data(), X_center)
        bundle.add_cut(point=theta, value=h_center, subgradient=g_center)
        self._calibrate_u(g_center)
        self._log_iteration(theta, h_center, len(bundle.cuts))

        if not active_names:
            # Rien à dualiser (mode dynamique, round 0) : theta est de dimension 0,
            # aucun master problem à résoudre — h_center est déjà la valeur du round
            # (SDP classique, RLT encore toutes en dur puisqu'aucune n'a été retirée
            # du task par setup_dualization([])).
            return theta, h_center, X_center, bundle

        for _ in range(self.max_iter):
            theta_new, s_predicted, u_used = solve_master_problem(bundle, theta_center=theta, u=self.u)
            if u_used != self.u:
                # solve_master_problem a dû réessayer avec un u plus grand (le problème
                # était numériquement instable au u courant, cf. proximal_master.py) —
                # on adopte ce u pour la suite plutôt que de retomber dans le même
                # problème à l'itération suivante.
                logger_cb.warning("u ajusté de %.4g à %.4g (instabilité numérique du master problem).", self.u, u_used)
                self.u = u_used
            result_new = self.sdp_solver.resolve_dualized(theta_new)
            if result_new["lb_value"] is None:
                logger_cb.warning(
                    "resolve_dualized non optimal au point d'essai (status=%s) — arrêt du round.",
                    result_new["status"],
                )
                break

            h_new = result_new["lb_value"]
            X_new = result_new["X"]
            g_new = compute_subgradients(self.sdp_solver.get_dualized_constraints_data(), X_new)
            bundle.add_cut(point=theta_new, value=h_new, subgradient=g_new)

            self.n_iter += 1
            predicted_increase = s_predicted - h_center
            actual_increase = h_new - h_center
            test = serious_null_step_test(actual_increase, predicted_increase, self.C1, self.C2)
            self._log_iteration(theta_new, h_new, len(bundle.cuts))

            if test.is_serious_step or test.should_stop:
                theta, h_center, X_center = theta_new, h_new, X_new

            if test.should_stop:
                break

            self._update_u(test.is_serious_step, predicted_increase, actual_increase)

        return theta, h_center, X_center, bundle

    # ------------------------------------------------------------------
    # Mode statique (5.4.1)
    # ------------------------------------------------------------------

    def _solve_static(self, all_dualizable: List[str]):
        theta0 = {name: 0.0 for name in all_dualizable}
        theta, h_center, X_center, _ = self._run_bundle_round(all_dualizable, theta0)
        return h_center, X_center

    # ------------------------------------------------------------------
    # Mode dynamique (5.4.2)
    # ------------------------------------------------------------------

    def _solve_dynamic(self, all_dualizable: List[str]):
        active: List[str] = []
        inactive: List[str] = list(all_dualizable)
        theta: Dict[str, float] = {}
        bundle: Optional[ProximalBundle] = None
        h_center = None
        X_center = None
        n_rounds = 0

        while True:
            if n_rounds >= self.max_rounds:
                logger_cb.warning(
                    "max_rounds=%d atteint — arrêt du mode dynamique (le résultat reste "
                    "une borne inférieure valide sur le pool courant, %d/%d contraintes "
                    "dualisées, mais n'est pas garanti convergé sur l'ensemble de R).",
                    self.max_rounds, len(active), len(all_dualizable),
                )
                break
            n_rounds += 1
            theta, h_center, X_center, bundle = self._run_bundle_round(active, theta, bundle)

            if not inactive:
                logger_cb.info("Round %d: plus aucune contrainte à ajouter, arrêt.", n_rounds)
                break

            # --- Sélection des contraintes les plus violées parmi celles inactives ---
            # Évaluées sur X_center (solution primale du dernier point accepté), sans
            # nouvel appel MOSEK : lecture pure des triplets déjà en mémoire.
            candidate_data = self.sdp_solver.get_constraint_dualization_data(inactive)
            violation = compute_subgradients(candidate_data, X_center)
            most_violated = sorted(
                (name for name, g in violation.items() if g > 0),
                key=lambda name: -violation[name],
            )[: self.add_batch_size]

            # --- Retrait des contraintes actives avec theta ~ 0 ---
            dropped = [
                name for name in active
                if abs(theta.get(name, 0.0)) < self.theta_drop_tol
            ]

            if not most_violated and not dropped:
                logger_cb.info(
                    "Round %d: aucune contrainte violée à ajouter et aucun theta~0 à "
                    "retirer, convergence du pool — arrêt.", n_rounds
                )
                break

            for name in dropped:
                active.remove(name)
                theta.pop(name, None)
                inactive.append(name)
            for name in most_violated:
                inactive.remove(name)
                active.append(name)
                theta[name] = 0.0

            logger_cb.info(
                "Round %d: +%d contraintes ajoutées, -%d retirées (|pool|=%d/%d).",
                n_rounds, len(most_violated), len(dropped), len(active), len(all_dualizable),
            )

        return h_center, X_center
