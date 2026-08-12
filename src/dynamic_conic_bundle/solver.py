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
        max_bundle_size: int = 50,
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
            Paramètre proximal u initial (Algorithme 2, ligne 5). Mis à jour
            adaptativement pendant la boucle (cf. _update_u) — ce n'est donc qu'un
            point de départ, pas une constante figée pour tout le run.
        u_min, u_max : float
            Bornes de l'ajustement adaptatif de u.
        max_bundle_size : int
            Nombre max de coupes conservées dans le bundle (FIFO) — cf.
            ProximalBundle.max_size, évite que le master problem devienne de plus
            en plus coûteux à mesure que les itérations s'accumulent.
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
        self.max_bundle_size = max_bundle_size
        self.theta_drop_tol = theta_drop_tol
        self.add_batch_size = add_batch_size
        self.max_rounds = max_rounds
        self.log_theta_every_n_iter = log_theta_every_n_iter
        self.verbose = verbose
        self._consecutive_serious = 0

        self.theta_history: List[Dict[str, float]] = []
        self.lb_history: List[float] = []
        self.n_iter = 0

    def get_current_thetas(self) -> Dict[str, float]:
        """Derniers multiplicateurs {constraint_name: theta_r} enregistrés."""
        return self.theta_history[-1] if self.theta_history else {}

    def _log_iteration(self, theta: Dict[str, float], h_value: float):
        self.lb_history.append(h_value)
        if self.log_theta_every_n_iter and self.n_iter % self.log_theta_every_n_iter == 0:
            self.theta_history.append(dict(theta))
        if self.verbose:
            logger_cb.info("iter %d: h(theta)=%.6f  u=%.4g", self.n_iter, h_value, self.u)

    def _update_u(self, is_serious_step: bool):
        """Mise à jour adaptative du paramètre proximal u (absente d'Algorithme 2 tel
        qu'écrit — u y est un paramètre fixe donné en entrée — mais nécessaire en
        pratique : un u constant mal calé convergeait très lentement sur un réseau
        plus gros que blob_1x2, cf. task-dynamic-conic-bundle.md). Règle standard des
        méthodes de bundle proximal (Kiwiel) : un pas nul signifie que le modèle ĥ
        était trop optimiste localement -> on resserre le rayon de confiance (u plus
        grand) ; plusieurs pas sérieux consécutifs signifient qu'on peut avancer plus
        vite -> on relâche le rayon (u plus petit)."""
        if is_serious_step:
            self._consecutive_serious += 1
            if self._consecutive_serious >= 3:
                self.u = max(self.u_min, self.u / 2.0)
                self._consecutive_serious = 0
        else:
            self.u = min(self.u_max, self.u * 2.0)
            self._consecutive_serious = 0

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
        self._log_iteration(theta, h_center)

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
            self._log_iteration(theta_new, h_new)

            if test.is_serious_step or test.should_stop:
                theta, h_center, X_center = theta_new, h_new, X_new

            if test.should_stop:
                break

            self._update_u(test.is_serious_step)

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
