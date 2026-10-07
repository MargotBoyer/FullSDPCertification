"""Builds the flat-numpy QCQP data for one (sample, target) BM-r instance -- the Python
equivalent of BM-r's utils/get_arguments.m + utils/check_inputs.m, fed by this project's
own network loading and bounds (bounds.py), not BM-r's own (trivial [0,1]x[0,inf) or
LP-based get_bound_*.m) bound machinery -- same "our bounds, their relaxation/solver"
comparison convention as every other baseline adapter in this project (see
src/baselines_interface/sdp_fo/README.md).

Mirrors Chiu & Zhang (arXiv:2211.17244) bm_linf.m/bm_l2.m exactly, in their own
notation translated to 0-indexed Python:

  ell          : number of ReLU hidden layers = K - 1 (the network's LAST weight
                 matrix, producing logits, is excluded from "Weight" -- its effect is
                 folded into the objective, exactly like objective_Lan.py's c/c_0).
  x_1..x_ell   : post-activation vectors of the ell hidden layers (x_1 has no ReLU
                 applied to it in the formulation below -- it's the raw, bounded input).
  ni, nu, nx   : input dimension, sum of hidden-layer sizes, nx = ni + nu (total
                 variables -- NOT including logits, which never enter the QCQP as
                 variables, only via Cost/Offset on x_ell).
  img, sel     : index ranges into the stacked x vector for the input block / the
                 concatenated hidden-layer block, respectively (Python slices, not
                 1-indexed Matlab ranges).

Objective: min Cost . x_ell + Offset, Cost = W_last[ytrue,:] - W_last[ytarget,:],
Offset = b_last[ytrue] - b_last[ytarget] -- minimize; a positive optimum certifies
robustness, matching this project's own sign convention (results.csv's optimal_value).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class BMProblem:
    norm: str                 # "Linf" or "L2"
    ni: int                   # input dimension
    nu: int                   # sum of hidden-layer sizes (excludes input, excludes logits)
    nx: int                   # ni + nu
    img: slice                 # x[img] = input block
    sel: slice                 # x[sel] = concatenated hidden-layer block
    layer_sizes: list          # [n_1, ..., n_ell] hidden-layer sizes
    Wmat: sp.csr_matrix        # (nu, nx) block-diagonal hidden-layer weights, zero-padded
    bvec: np.ndarray           # (nu,) concatenated hidden-layer biases
    Smat: sp.csr_matrix        # (nu, nx) selection matrix: Smat @ x = x[sel]
    c: np.ndarray              # (nx,) objective vector (zero outside the x_ell block)
    c0: float                  # objective constant (Offset)
    LB: np.ndarray             # (nx,) box lower bounds (input: our L[0]; hidden: max(L_k,0))
    UB: np.ndarray             # (nx,) box upper bounds (input: our U[0]; hidden: max(U_k,0))
    # L2 only: single joint ball constraint ||x[img] - image_center||^2 <= radius^2.
    image_center: np.ndarray = None
    radius: float = None
    # Linf only: per-coordinate ball (x[img]_i - center_i)^2 <= bound_i^2, i.e. just the
    # box [LB[img], UB[img]] re-expressed as a quadratic diag constraint (see bm_constraints.py).


def build_problem(W, b, L, U, x, epsilon, ytrue, ytarget, norm="Linf"):
    """W, b: full K-length weight/bias lists (PyTorch convention, W[k] shape
    (n_out, n_in)), e.g. from networks.network.ReLUNN.extract_weights().
    L, U: this project's precomputed bounds (bounds.py), length K+1, L[0]/U[0] the
    input ball, L[k]/U[k] for k>=1 PRE-activation bounds (CLAUDE.md convention).
    x: the center image, length n_0. epsilon: perturbation radius (same units as the
    network's input, consistent with L[0]/U[0] -- no separate radius-vs-bounds
    reconciliation needed, L[0]/U[0] already encode it).
    """
    assert norm in ("Linf", "L2")
    K = len(W)
    ell = K - 1  # hidden layers only; the last weight matrix is folded into Cost/Offset

    layer_sizes = [int(W[k].shape[0]) for k in range(ell)]
    ni = int(W[0].shape[1])
    nu = int(sum(layer_sizes))
    nx = ni + nu
    img = slice(0, ni)
    sel = slice(ni, nx)

    # Wmat: (nu, nx) block-diagonal of hidden-layer weights, zero-padded on the right
    # (the logit layer's weights never multiply any x -- nothing maps *into* a
    # nonexistent "x_{ell+1}"). Wmat @ x = pre-activations of x_2..x_ell stacked with
    # a leading zero block for x_1 itself (x_1 has no "pre-activation": it's raw input).
    blocks = [sp.csr_matrix(np.asarray(W[k], dtype=np.float64)) for k in range(ell)]
    Wmat = sp.block_diag(blocks, format="csr")  # shape (nu, nx - layer_sizes[-1])
    # Pad on the right with zero columns for x_ell itself (nothing in Wmat maps *from*
    # the last hidden layer forward -- its only further use is in Cost, not Wmat).
    Wmat = sp.hstack([Wmat, sp.csr_matrix((nu, layer_sizes[-1]))], format="csr")
    bvec = np.concatenate([np.asarray(b[k], dtype=np.float64).reshape(-1) for k in range(ell)])

    # Smat: (nu, nx) selects x[sel] (the hidden-layer block) out of the full x vector.
    rows = np.arange(nu)
    cols = ni + np.arange(nu)
    Smat = sp.csr_matrix((np.ones(nu), (rows, cols)), shape=(nu, nx))

    c = np.zeros(nx)
    last_layer_start = ni + sum(layer_sizes[:-1])
    Wlast = np.asarray(W[-1], dtype=np.float64)
    blast = np.asarray(b[-1], dtype=np.float64).reshape(-1)
    c[last_layer_start:nx] = Wlast[ytrue, :] - Wlast[ytarget, :]
    c0 = float(blast[ytrue] - blast[ytarget])

    L0, U0 = np.asarray(L[0], dtype=np.float64), np.asarray(U[0], dtype=np.float64)
    LB = np.concatenate([L0] + [np.maximum(np.asarray(L[k + 1], dtype=np.float64), 0.0)
                                 for k in range(ell)])
    UB = np.concatenate([U0] + [np.maximum(np.asarray(U[k + 1], dtype=np.float64), 0.0)
                                 for k in range(ell)])

    prob = BMProblem(norm=norm, ni=ni, nu=nu, nx=nx, img=img, sel=sel,
                      layer_sizes=layer_sizes, Wmat=Wmat, bvec=bvec, Smat=Smat,
                      c=c, c0=c0, LB=LB, UB=UB)
    if norm == "L2":
        prob.image_center = np.asarray(x, dtype=np.float64).reshape(-1)
        prob.radius = float(epsilon)
    return prob
