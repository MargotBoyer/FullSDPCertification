"""
Diagnostic PNG pour un run de la dynamic conic bundle method (main.pdf, section 5.4 ;
task-dynamic-conic-bundle.md). Un PNG par résolution (data_index/target), 3 courbes
empilées vs itération : h(theta) (progression de la borne), u (paramètre proximal,
échelle log — plage typiquement large, cf. "Bug E"), taille du bundle (nombre de
coupes conservées, cf. ProximalBundle.max_size).

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
):
    """Sauvegarde un PNG à 3 sous-graphes (h(theta), u, taille du bundle) vs itération.
    Les trois historiques doivent avoir la même longueur (une entrée par itération
    loggée, cf. DynamicConicBundleSolver._log_iteration)."""
    n = len(lb_history)
    iterations = list(range(n))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(iterations, lb_history, color="tab:blue")
    axes[0].set_ylabel("h(theta)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, u_history, color="tab:orange")
    axes[1].set_ylabel("u")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, bundle_size_history, color="tab:green")
    axes[2].set_ylabel("taille du bundle")
    axes[2].set_xlabel("itération")
    axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
