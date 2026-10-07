"""Closed-form negative eigenpair of the ReLU complementarity quadratic form Q1_{i,j}
from Lan, Bruckner & Lomuscio, "A Semidefinite Relaxation Based Branch-and-Bound Method
for Tight Neural Network Verification", AAAI 2023 (Proposition 1 / eq. 12-15b).

For ReLU neuron j of layer i+1 (0-indexed layers, i in [0, K-1], j in [0, n[i+1])), the
complementarity constraint x_{i+1,j} * (x_{i+1,j} - (W_i x_i + b_i)_j) = 0 has quadratic
part (ignoring the linear -b_i[j]*x_{i+1,j} term, which only affects the constant/linear
side of the constraint, not Q1) equal to, over the stacked vector (x_i, x_{i+1,j}):

    Q1_{i,j} = [[0_{n_i x n_i}, z], [z^T, 1]],   z = -W_i[j, :] / 2

This is an arrowhead matrix with a zero block on the diagonal, so (unlike the general
case) its spectrum is known in closed form: n_i - 1 zero eigenvalues, and exactly one
negative / one positive eigenvalue solving the 2x2 reduced system [[0, ||z||], [||z||, 1]]
-- no generic eigensolver needed.

Pruning (this project's LayerSDP, like Batten et al. 2021, removes stable neurons from
the SDP): a stable-INACTIVE neuron (k, a) has z_{k,a} === 0 identically (guaranteed by
its bounds, not an approximation) -- every term `-W_i[j,a] * z_{i,a}` in the
complementarity constraint is exactly zero regardless of W_i[j,a], so zeroing column a
of W_i[j,:] before computing z/lambda_neg/v is an EXACT substitution, not a relaxation.
This is strictly better than dropping the whole cut (the previous approach): it keeps
the cut for every ReLU neuron whose OUTPUT is not itself pruned, just over a smaller
(cheaper) support. A stable-ACTIVE neuron, by contrast, is not substitutable this
simply (it decomposes into a linear combination of earlier unstable neurons) --
handled transparently elsewhere already, by the existing
handler.variables_call.VariablesCall.verify_variable_z/equivalent_neurons machinery, so
it needs no special-casing here: referencing it in add_quad_variable/add_linear_variable
works as-is.

The eigenpair depends on W_i and, now, on which neurons of layer i and i+1 this sample's
bounds have pruned -- so unlike a pure function of the network, it must be recomputed per
sample (cheap: a few ms for an entire 9x100-sized network, see sdp_bab/README.md).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenCut:
    """Negative-eigenvalue direction for ReLU neuron j of layer i+1 (pre-activation
    x_{i+1,j} = ReLU((W_i x_i + b_i)_j)).

    v_xi: coefficients of x_i (length n_i) in the eigenvector -- zero at any pruned
    (stable-inactive) coordinate of layer i, by construction (see module docstring).
    v_out: coefficient of x_{i+1,j} in the eigenvector.
    lambda_neg: the (negative) eigenvalue itself -- kept for diagnostics, unused in the cut.
    """
    layer: int      # i : the pre-ReLU layer (x_i)
    neuron: int     # j : neuron index in layer i+1
    v_xi: np.ndarray
    v_out: float
    lambda_neg: float


def compute_eigencuts(W, stable_inactives_neurons=None, pruned_input_neurons=None,
                       input_in_variables=True):
    """W: list of per-layer weight matrices, PyTorch convention (W[k] shape (n_out, n_in)),
    e.g. from networks.network.ReLUNN.extract_weights().

    stable_inactives_neurons: set of (layer, neuron) pairs this sample's bounds have
    pruned as identically zero (SDPSolver.stable_inactives_neurons after bounds are
    computed -- e.g. a freshly-built LayerSDPBaBNode, or the lightweight
    bounds.check_stability_neurons probe branch_and_bound.py uses). None/empty means no
    pruning assumed (every coordinate kept) -- matches the unpruned case exactly.

    pruned_input_neurons / input_in_variables: layer-0 (input) pruning is a different
    mechanism (INPUT_IN_VARIABLES < 1 substitutes a nonzero bound constant, not 0) --
    not substituted here (would need to shift the cut's constant term, not just drop a
    variable); a cut referencing such a removed input neuron, or the whole input layer
    when input_in_variables=False, is dropped entirely instead (always valid, just
    loses tightening for that neuron).

    Returns a flat list of EigenCut: one per ReLU neuron whose OUTPUT (layer i+1) is not
    itself pruned (a pruned-inactive output neuron's complementarity constraint is
    trivially 0=0, nothing to tighten) and whose incoming row isn't entirely zeroed out.
    """
    stable_inactives_neurons = stable_inactives_neurons or set()
    pruned_input_neurons = pruned_input_neurons or set()

    cuts = []
    for i, Wi in enumerate(W[:-1]):  # layer K-1 (logits) has no ReLU -> no Q1_{K-1,j}
        Wi = np.asarray(Wi, dtype=np.float64)
        n_out, n_in = Wi.shape

        if i == 0 and not input_in_variables:
            continue  # whole input layer removed from the SDP -- no x_i to build a cut on
        pruned_cols = (
            [a for a in range(n_in) if a in pruned_input_neurons] if i == 0
            else [a for a in range(n_in) if (i, a) in stable_inactives_neurons]
        )

        for j in range(n_out):
            if (i + 1, j) in stable_inactives_neurons:
                continue  # output identically 0: no complementarity constraint to tighten
            row = Wi[j, :]
            if pruned_cols:
                row = row.copy()
                row[pruned_cols] = 0.0  # exact substitution z=0, see module docstring
            z = -row / 2.0
            norm_z = np.linalg.norm(z)
            if norm_z == 0.0:
                # No (remaining) incoming weight: Q1_{i,j} = diag(0,...,0,1), PSD already.
                continue
            lambda_neg = (1.0 - np.sqrt(1.0 + 4.0 * norm_z ** 2)) / 2.0
            # Eigenvector in the 2D invariant subspace span{z/||z||, e_last}, unnormalized
            # as (z, lambda_neg) -- see derivation in the module docstring.
            v = np.concatenate([z, [lambda_neg]])
            v /= np.linalg.norm(v)
            cuts.append(EigenCut(
                layer=i, neuron=j, v_xi=v[:-1], v_out=float(v[-1]), lambda_neg=lambda_neg,
            ))
    return cuts
