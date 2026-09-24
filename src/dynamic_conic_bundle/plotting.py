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

import matplotlib.pyplot as plt


def plot_run_diagnostics(
    lb_history: List[float],
    u_history: List[float],
    bundle_size_history: List[int],
    save_path: str,
    title: Optional[str] = None,
    best_h_history: Optional[List[float]] = None,
    n_serious_history: Optional[List[int]] = None,
):
    """Sauvegarde un PNG à sous-graphes empilés vs itération : h(theta) (+ meilleur h
    trouvé jusque-là si `best_h_history` est fourni), u, taille du bundle, et
    (optionnel) nombre cumulé de vrais pas de descente si `n_serious_history` est
    fourni. Tous les historiques fournis doivent avoir la même longueur (une entrée
    par itération loggée)."""
    n = len(lb_history)
    iterations = list(range(n))

    n_panels = 3 + (1 if n_serious_history is not None else 0)
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

    if n_serious_history is not None:
        axes[3].plot(iterations, n_serious_history, color="tab:purple")
        axes[3].set_ylabel("nb vrais pas")
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel("itération")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
