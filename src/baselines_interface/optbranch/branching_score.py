"""Branching-coordinate selection (Theorem 20, Anderson/Ma/Li/Sojoudi,
"Towards Optimal Branching for Neural Network Robustness Certification", JMLR 2025).

Theorem 20 picks, among the *input* coordinates, the one with the largest
max(|l_k|, |u_k|) -- intuitively, the loosest box bound dominates the SDP
relaxation's worst-case error (Theorem 19), so tightening it first gives the
biggest worst-case improvement. The paper derives this for a single hidden
layer and the input set X, but remarks at the end of section 4.3.2 that the
score only depends on the bounds of the layer being branched on -- so it
extends unchanged to any neuron of any layer, not just the input. That
multi-layer extension is exactly what this function implements: since this
project's solver already takes arbitrary per-neuron L[k][i]/U[k][i] overrides
(see node_solver.py), branching on a hidden-layer neuron is no harder than
branching on an input coordinate.
"""
import numpy as np


def select_branching_neuron(L, U, layers=None):
    """Return (k, i, score) for the unstable neuron (L[k][i] < 0 < U[k][i])
    with the largest score = max(|L[k][i]|, |U[k][i]|), or None if no
    unstable neuron is found among the considered layers.

    L, U: per-layer bound lists (bounds.py convention: L[0]/U[0] = input box,
    L[k]/U[k] for k>=1 = pre-activation bounds of layer k; length K+1).
    layers: layer indices to consider (default: 0..K-1, i.e. the input and
    every hidden pre-activation layer, but never the output/logit layer K --
    that layer is the certification objective itself, not a network
    constraint to tighten).
    """
    K = len(L) - 1
    if layers is None:
        layers = range(0, K)

    best = None
    for k in layers:
        Lk = np.asarray(L[k], dtype=np.float64)
        Uk = np.asarray(U[k], dtype=np.float64)
        unstable = np.where((Lk < 0) & (Uk > 0))[0]
        if unstable.size == 0:
            continue
        scores = np.maximum(np.abs(Lk[unstable]), np.abs(Uk[unstable]))
        j = int(np.argmax(scores))
        score = float(scores[j])
        if best is None or score > best[2]:
            best = (k, int(unstable[j]), score)
    return best
