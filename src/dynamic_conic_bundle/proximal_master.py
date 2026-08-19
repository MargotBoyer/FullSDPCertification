"""
Master problem proximal du conic bundle method (main.pdf, section 5.4.1, Algorithme 2).

Convention de signe : Algorithme 2 est écrit pour un SDPCB de la forme
`max <Q,P>` s.c. `<Qr,P>-dr <= 0` dualisées par theta>=0, et minimise le dual
convexe `h(theta) = max_P(...)` (theta>=0). Le code de ce dépôt (SDPSolver,
handler MOSEK) résout au contraire une **minimisation** `min <C,X>` (cf.
CLAUDE.md, sections 4.2.1/4.2.2 de main.pdf — c'est la convention globale du
papier, 5.4.1 utilise localement Q=-C). On dualise donc ici la version miroir :

    h(theta) = min_{P admissible non dualisé} <C,P> + sum_r theta_r * g_r(P),  theta>=0

`h` est une borne inférieure valide de la vraie valeur optimale pour tout theta>=0
(argument de dualité lagrangienne standard, indépendant du sens min/max — la preuve
de la Proposition 7 du papier sur le sous-gradient reste valable telle quelle : voir
dynamic_conic_bundle/subgradient.py). h est concave (min d'affines en theta) : on la
**maximise** sur theta>=0 (au lieu de minimiser un h convexe côté papier) pour obtenir
la meilleure borne inférieure — c'est le miroir exact d'Algorithme 2, pas une
transcription littérale.

`theta` (pas `alpha`) : notation gardée identique à celle du papier (main.pdf,
Algorithme 2). Ne pas confondre avec les `alpha`/`alpha-CROWN` du calcul de bornes
(bounds_crown.py, bounds_method="alpha-CROWN") — concept sans rapport.

Complétion du test sérieux/nul (lignes 7-13 d'Algorithme 2) : tel qu'imprimé dans le
PDF, les branches `> C2` et `else` sont identiques (theta_hat <- theta_{k+1} dans les
deux cas) — C2 ("not sufficient criteria") n'a donc aucun effet fonctionnel dans le
texte. Complété ici avec un test sérieux/nul standard des méthodes de bundle proximal
(Kiwiel) en utilisant C1/C2 dans le rôle suggéré par leur nom : C1 = tolérance d'arrêt,
C2 = seuil de progression suffisante pour accepter un pas sérieux (repère explicitement
comme une complétion, validée avec l'utilisateur — voir task-dynamic-conic-bundle.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class BundleCut:
    """Une coupe (linéarisation) du bundle : h(theta) <= value + <subgradient, theta - point>
    (majorant, valide car h est concave — cf. Proposition 7 du papier)."""

    point: Dict[str, float]
    value: float
    subgradient: Dict[str, float]


@dataclass
class ProximalBundle:
    """État du bundle proximal pour un ensemble de contraintes dualisées donné.

    names fixe l'ordre des coordonnées de theta pour ce round (mode dynamique : un
    nouveau ProximalBundle est recréé à chaque round avec le pool de contraintes
    dualisées mis à jour — cf. solver.py). Les coupes plus anciennes dont le nom
    n'apparaît pas encore dans `names` n'existent pas ici : elles sont implicitement
    à theta_r=0 pour les noms absents (contrainte pas encore dualisée à ce moment-là),
    ce qui est mathématiquement correct (theta_r=0 <=> la contrainte ne contribue pas
    à l'objectif pénalisé, exactement comme si elle n'existait pas).

    max_size borne le nombre de coupes conservées (FIFO, la plus ancienne tombe en
    premier). None (défaut) = pas de cap : le nombre de coupes ne peut de toute
    façon jamais dépasser max_iter (une coupe ajoutée par itération, cf.
    _run_bundle_round dans solver.py), donc un cap fixe est un plafond
    artificiel qui n'apporte rien niveau mémoire/temps mais peut rendre le
    modèle du bundle sous-déterminé quand dim(theta) (= nb de contraintes
    dualisées) dépasse le cap — diagnostiqué comme cause d'arrêt prématuré du
    mode statique sur blob_4x10 (dim(theta) ~ 400-960, cap à 50) : cf.
    task-dynamic-conic-bundle.md, section "Analyse statique vs dynamique". Un
    entier explicite reste utile pour borner le coût du master problem QP à
    grande échelle (dim(theta) et max_iter tous deux élevés).
    """

    names: List[str]
    cuts: List[BundleCut] = field(default_factory=list)
    max_size: Optional[int] = None

    def add_cut(self, point: Dict[str, float], value: float, subgradient: Dict[str, float]):
        self.cuts.append(BundleCut(point=dict(point), value=value, subgradient=dict(subgradient)))
        if self.max_size is not None and len(self.cuts) > self.max_size:
            self.cuts.pop(0)

    def evaluate_model(self, theta: Dict[str, float]) -> float:
        """ĥ(theta) = min_l [value_l + <g_l, theta - point_l>] — majorant de h (h concave)."""
        theta_vec = np.array([theta.get(n, 0.0) for n in self.names])
        best = np.inf
        for cut in self.cuts:
            point_vec = np.array([cut.point.get(n, 0.0) for n in self.names])
            g_vec = np.array([cut.subgradient.get(n, 0.0) for n in self.names])
            best = min(best, cut.value + float(np.dot(g_vec, theta_vec - point_vec)))
        return best


def solve_master_problem(
    bundle: ProximalBundle,
    theta_center: Dict[str, float],
    u: float,
    max_retries: int = 5,
) -> "tuple[Dict[str, float], float, float]":
    """
    Résout : max_{theta>=0, s} s - (u/2)*||theta - theta_center||^2
             s.t. s <= value_l + <g_l, theta - point_l>   pour chaque coupe l du bundle

    Retourne (theta maximiseur, s* = ĥ(theta) au maximiseur — la valeur *prédite* par
    le modèle du bundle, pas la vraie h(theta) : voir serious_null_step_test pour
    pourquoi cette distinction est nécessaire —, u effectivement utilisé). Requiert
    cvxpy (dépendance optionnelle du projet, comme pour solver="cvxpy" dans
    SDPSolverConfig).

    Ce problème est mathématiquement toujours borné pour u>0 (quadratique strictement
    concave + contraintes linéaires ⟹ optimum unique) : un statut cvxpy "unbounded"
    ou "infeasible" ne peut donc être qu'un artefact numérique — typiquement u devenu
    trop petit (après plusieurs pas nuls consécutifs qui l'ont divisé par 2, jusqu'à
    u_min) par rapport à la magnitude des sous-gradients du bundle, ce qui rend le
    terme quadratique numériquement négligeable pour le solveur QP par défaut de
    cvxpy (observé en pratique : crash à u proche de u_min sur un réseau plus gros
    que blob_1x2, cf. task-dynamic-conic-bundle.md). On corrige en réessayant avec un
    u multiplié par 10 à chaque tentative — le u effectivement utilisé est renvoyé
    pour que l'appelant (DynamicConicBundleSolver) adopte ce u plus grand pour la
    suite, plutôt que de retomber dans le même problème à l'itération suivante.
    """
    if cp is None:
        raise ImportError(
            "solve_master_problem nécessite cvxpy (pip install cvxpy) — "
            "même dépendance optionnelle que solver='cvxpy' dans SDPSolverConfig."
        )
    names = bundle.names
    n = len(names)
    theta_center_vec = np.array([theta_center.get(name, 0.0) for name in names])

    u_try = u
    last_status = None
    for attempt in range(max_retries):
        theta = cp.Variable(n)
        s = cp.Variable()

        constraints = [theta >= 0]
        for cut in bundle.cuts:
            point_vec = np.array([cut.point.get(name, 0.0) for name in names])
            g_vec = np.array([cut.subgradient.get(name, 0.0) for name in names])
            # value_l + <g_l, theta - point_l> = (value_l - <g_l, point_l>) + <g_l, theta>
            intercept = cut.value - float(np.dot(g_vec, point_vec))
            constraints.append(s <= intercept + g_vec @ theta)

        objective = cp.Maximize(s - (u_try / 2.0) * cp.sum_squares(theta - theta_center_vec))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if theta.value is not None:
            theta_star = {name: max(0.0, float(v)) for name, v in zip(names, theta.value)}
            return theta_star, float(s.value), u_try

        last_status = problem.status
        u_try *= 10.0

    raise RuntimeError(
        f"Master problem proximal non résolu après {max_retries} tentatives "
        f"(dernier statut={last_status}, u initial={u}, u final essayé={u_try})."
    )


@dataclass
class SeriousNullStepResult:
    is_serious_step: bool
    should_stop: bool
    predicted_increase: float
    actual_increase: float


def serious_null_step_test(
    actual_increase: float,
    predicted_increase: float,
    C1: float,
    C2: float,
) -> SeriousNullStepResult:
    """
    Complétion (validée utilisateur) du test des lignes 7-13 d'Algorithme 2, réécrite
    en test prédit/réel standard des méthodes de bundle proximal (Kiwiel) après
    diagnostic d'un faux positif de convergence (cf. task-dynamic-conic-bundle.md) :
    comparer seulement `h_new - h_center` (valeurs réelles) à un seuil fixe C1 se
    fait tromper quand le pas devient minuscule (u très grand après une série de
    pas nuls) — `h_new` colle alors artificiellement à `h_center` sans qu'on soit
    près de l'optimum. Le signal correct est la progression *prédite* par le modèle
    ĥ du bundle (`predicted_increase = ĥ(theta_new) - h_center`, renvoyée par
    solve_master_problem) : si le modèle lui-même ne promet plus de progrès, on est
    au voisinage d'un optimum du modèle courant — condition nécessaire à l'arrêt,
    indépendante de la taille du pas réellement pris.

    - predicted_increase <= C1                        -> arrêt (le modèle ne peut
      plus promettre de progrès significatif)
    - actual_increase >= C2 * predicted_increase       -> pas sérieux (le centre
      proximal avance à theta_new ; C2 = fraction minimale du progrès promis à
      réaliser réellement, ~0 < C2 < 1)
    - sinon                                            -> pas nul (le centre reste,
      le bundle est juste enrichi)
    """
    if predicted_increase <= C1:
        return SeriousNullStepResult(
            is_serious_step=False, should_stop=True,
            predicted_increase=predicted_increase, actual_increase=actual_increase,
        )
    if actual_increase >= C2 * predicted_increase:
        return SeriousNullStepResult(
            is_serious_step=True, should_stop=False,
            predicted_increase=predicted_increase, actual_increase=actual_increase,
        )
    return SeriousNullStepResult(
        is_serious_step=False, should_stop=False,
        predicted_increase=predicted_increase, actual_increase=actual_increase,
    )
