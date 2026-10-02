"""
Diagnostic PNG pour un run de conic bundle method (main.pdf, section 5.4 ;
task-dynamic-conic-bundle.md), commun aux deux moteurs (`engine="python"`,
DynamicConicBundleSolver.save_diagnostics_png, et `engine="conicbundle_native"`,
miqcr_bridge.conicbundle_native.cb_wrapper.solve_native_conicbundle_dual). Courbes
empilées vs itération : h(theta) (progression de la borne, avec en tireté le
meilleur h trouvé jusque-là si fourni), u (paramètre proximal / poids du terme
quadratique, échelle log — plage typiquement large, cf. "Bug E"), taille du bundle
(nombre de coupes conservées), et optionnellement le nombre cumulé de "vrais" pas
de descente (pas sérieux, par opposition aux pas nuls qui n'avancent pas le centre
mais enrichissent le bundle).

Activé via `print_png: true` dans le bloc yaml `dynamic_conic_bundle`
(DynamicConicBundleConfig) — désactivé par défaut (coût I/O/matplotlib non
négligeable sur un run à des centaines d'échantillons).
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib
# Backend non-interactif (savefig uniquement, jamais show()) -- requis sur un
# noeud de calcul sans display (ex. Jean-Zay) : sans ce .use() explicite,
# matplotlib peut tenter un backend GUI (Tk/Qt) et planter en l'absence de
# DISPLAY. Doit etre appele avant le premier `import matplotlib.pyplot`.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run_diagnostics(
    lb_history: List[float],
    u_history: List[float],
    bundle_size_history: List[int],
    save_path: str,
    title: Optional[str] = None,
    best_h_history: Optional[List[float]] = None,
    n_serious_history: Optional[List[int]] = None,
    sgnorm_history: Optional[List[float]] = None,
    new_subgrads_history: Optional[List[int]] = None,
    disagreement_history: Optional[List[float]] = None,
    step_norm_history: Optional[List[float]] = None,
    time_history: Optional[List[float]] = None,
    pretreatment_history: Optional[List[float]] = None,
    solve_time_history: Optional[List[float]] = None,
):
    """Sauvegarde un PNG à sous-graphes empilés vs itération (un point = un appel
    oracle). Panneaux de base : h(theta) (+ meilleur h trouvé jusque-là si
    `best_h_history` est fourni), u (échelle log), taille du bundle. Panneaux
    optionnels, un par historique fourni (None = omis) :
      - n_serious_history : nombre cumulé de vrais pas de descente
      - sgnorm_history : ||g(theta)|| au point évalué (cb_get_sgnorm)
      - new_subgrads_history : nb de sous-gradients epsilon renvoyés par cet appel
        oracle (>1 = enrichissement "coin" déclenché, cf. max_subg_by_point)
      - disagreement_history : écart relatif sous-gradient détecté par la sonde de
        coin (NaN quand la sonde n'a pas tourné pour cet appel -- ex. max_subg_by_point=1)
      - step_norm_history : ||theta_essai - centre courant|| (rayon d'exploration)
      - time_history : temps (s) total de l'appel oracle (resolve_dualized complet :
        pretreatment + solve MOSEK + extraction du statut/X)
      - pretreatment_history : temps (s) cote Python de resolve_dualized avant
        task.optimize() (construction de l'objectif dualisé, putbarcblocktriplet)
      - solve_time_history : temps (s) de task.optimize() seul (résolution SDP pure)
    Tous les historiques fournis doivent avoir la même longueur que lb_history (une
    entrée par appel oracle loggé)."""
    n = len(lb_history)
    iterations = list(range(n))

    optional_panels = [
        (n_serious_history, "nb vrais pas", "tab:purple", False),
        (sgnorm_history, "||g(theta)||", "tab:brown", True),
        (new_subgrads_history, "nb sous-gradients", "tab:pink", False),
        (disagreement_history, "écart sonde coin", "tab:olive", False),
        (step_norm_history, "||pas d'essai||", "tab:cyan", False),
        (time_history, "temps oracle total (s)", "tab:gray", False),
        (pretreatment_history, "pretraitement Python (s)", "tab:red", False),
        (solve_time_history, "solve MOSEK pur (s)", "tab:blue", False),
    ]
    active_optional = [p for p in optional_panels if p[0] is not None]

    n_panels = 3 + len(active_optional)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)

    axes[0].plot(iterations, lb_history, color="tab:blue", label="h(theta)")
    if best_h_history is not None:
        axes[0].plot(iterations, best_h_history, color="tab:red", linestyle="--",
                     label="meilleur h trouvé")
        axes[0].legend(loc="lower right", fontsize="small")
    axes[0].set_ylabel("h(theta)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, u_history, color="tab:orange")
    axes[1].set_ylabel("u")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, bundle_size_history, color="tab:green")
    axes[2].set_ylabel("taille du bundle")
    axes[2].grid(True, alpha=0.3)

    for idx, (history, ylabel, color, log_scale) in enumerate(active_optional):
        ax = axes[3 + idx]
        ax.plot(iterations, history, color=color)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("appel oracle")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
