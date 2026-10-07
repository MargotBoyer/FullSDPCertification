"""The dual slack matrix S(y,z) (bm_linf.m's S_mat/S_mv/Syz_mv) and the valid lower
bound formula -- the actual payoff of the Burer-Monteiro approach: if S(y,z)'s
smallest eigenvalue is >= 0 at a KKT point of the rank-r factorized NLP, that point is
PROVABLY the global optimum of the full (un-factored) SDP relaxation, not just a local
optimum of the NLP (Burer-Monteiro second-order sufficiency theory).

Derivation (re-derived directly from this package's own, already finite-difference
-verified, bm_constraints.constraints() -- not a line-by-line port of bm_linf.m's
"generic A,B,Afun,Bfun" abstraction layer, whose own definition isn't fully shown in
the source; re-deriving from our own verified g/h formulas is lower risk and was
cross-checked to reproduce bm_linf.m's S_mat structure exactly):

For the Lagrangian of the (un-factored) QCQP, L(x) = c'x - lambda_lin'(A_lin x - b_lin)
- lambda_g'g(x) - lambda_h'h(x), split into a quadratic part (matrix S_half, with
bm_linf.m's own S = 2*S_half, hence the "0.5*" factors throughout) and the rest:

  g_i(x) = (x_i - center_i)^2 - bound_i^2          (Linf: one row per input neuron i)
  g(x)   = sum_i (x_i - center_i)^2 - radius^2      (L2: single joint constraint;
           lambda_g is then a scalar, broadcast uniformly over every img row below --
           the only place the two norms need different handling in this file)
    -> quadratic coefficient on x_i: 1 (so Hessian entry 2*lambda_g once weighted)
    -> x-independent part of d g_i/d x_i: -2*center_i

  h_j(x) = x_{sel_j}^2 - x_{sel_j}*(Wmat x)_j - x_{sel_j}*bvec_j
    -> quadratic part: x_{sel_j}^2 - x_{sel_j}*(Wmat x)_j
       (Hessian: 2 at (sel_j,sel_j), -Wmat[j,a] at (sel_j,a) and (a,sel_j))
    -> x-independent part of d h_j/d x_{sel_j}: -bvec_j

  A_lin x <= b_lin: purely linear, contributes only to the border via -A_lin'@lambda_lin.

Stationarity (d/dx[Lagrangian] = 0) gives s (bm_linf.m: s = 0.5*(c - A'y - B'z)):
  s = 0.5*c - 0.5*A_lin'@lambda_lin + center_broadcast*lambda_g + 0.5*bvec*lambda_h
(the x-independent gradient terms above enter with a flipped sign relative to the
bullet list because they are being *subtracted* from the stationarity equation.)
"""
import numpy as np
import scipy.sparse.linalg as spla

from bm_constraints import _linear_bound


def _ball_center(prob):
    return (prob.UB[prob.img] + prob.LB[prob.img]) / 2.0 if prob.norm == "Linf" else prob.image_center


def _ball_rhs(prob):
    """b_nonlin: the x-independent constant of g(x) <= 0, i.e. g(x) = (linear-in-x
    terms) - b_nonlin -- center^2 - bound^2 per neuron (Linf) or ||image||^2 -
    radius^2 (L2, scalar, matching prob.radius / prob.image_center from problem.py)."""
    if prob.norm == "Linf":
        center = _ball_center(prob)
        bound = (prob.UB[prob.img] - prob.LB[prob.img]) / 2.0
        return center ** 2 - bound ** 2
    return np.array([np.sum(prob.image_center ** 2) - prob.radius ** 2])


def _broadcast_lambda_g(lambda_g, prob):
    return np.broadcast_to(np.atleast_1d(lambda_g), (prob.ni,)) if prob.norm == "L2" else lambda_g


def s_vector(lambda_lin, lambda_g, lambda_h, prob):
    """The nx-length border of S(y,z) (its [0, 1:] block)."""
    A_lin, _ = _linear_bound(prob)
    lambda_g_img = _broadcast_lambda_g(lambda_g, prob)

    s = 0.5 * prob.c - 0.5 * (A_lin.T @ lambda_lin)
    s[prob.img] += _ball_center(prob) * lambda_g_img
    s[prob.sel] += 0.5 * prob.bvec * lambda_h
    return s


def S_mv(v, lambda_g, lambda_h, prob):
    """(Hessian of the g/h part of the Lagrangian) @ v -- matches bm_linf.m's S_mv
    exactly (the "0.5*" scaling to build the actual S(y,z) block happens separately,
    in Syz_mv_factory below, exactly where bm_linf.m applies it too). v has length nx;
    returns length nx.

    Caught by test_bm_r_dual3.py-style finite-difference check (not the simpler
    near-converged-multipliers check tried first, which passed by accident -- near-zero
    multipliers make almost any formula's error negligible; see bm_r/README.md): an
    earlier version of this function was missing the factor of 2 on both diagonal
    terms below (2*lambda_g*v[img], and the "2*" inside lambda_h*(2*v[sel]-Wv)) --
    present in bm_linf.m's S_mv but silently dropped when re-deriving this from
    scratch instead of transcribing the Matlab line directly."""
    Wmat_coo = prob.Wmat.tocoo()
    w_rows, w_cols, w_data = Wmat_coo.row, Wmat_coo.col, Wmat_coo.data

    Wv = np.zeros(prob.nu)
    np.add.at(Wv, w_rows, w_data * v[w_cols])

    Sv = np.zeros(prob.nx)
    lambda_g_img = _broadcast_lambda_g(lambda_g, prob)
    Sv[prob.img] = 2.0 * lambda_g_img * v[prob.img]
    Sv[prob.sel] = lambda_h * (2.0 * v[prob.sel] - Wv)

    # transpose contribution: -W^T @ diag(lambda_h) acting on the sel-block of v also
    # feeds back into whichever rows Wmat's columns span (img, or an earlier sel block).
    WTv = np.zeros(prob.nx)
    np.add.at(WTv, w_cols, w_data * (lambda_h[w_rows] * v[prob.sel][w_rows]))
    Sv -= WTv
    return Sv


def Syz_mv_factory(s, s0, lambda_g, lambda_h, prob):
    """Matrix-free (nx+1)-dim linear operator for S(y,z) = [[s0, s'], [s, 0.5*S]], for
    scipy.sparse.linalg.eigsh (Lanczos) -- matches bm_linf.m's Syz_mv."""
    nxp1 = prob.nx + 1

    def matvec(v):
        out = np.empty(nxp1)
        out[0] = s0 * v[0] + s @ v[1:]
        out[1:] = v[0] * s + 0.5 * S_mv(v[1:], lambda_g, lambda_h, prob)
        return out

    return spla.LinearOperator((nxp1, nxp1), matvec=matvec, dtype=np.float64)


def smallest_eigenvalue(s, s0, lambda_g, lambda_h, prob, n_iter=500, seed=0):
    """scipy's eigsh (Lanczos, matrix-free) smallest eigenvalue of S(y,z) -- matches
    bm_linf.m's eigs(@Syz_mv, nx+1, 1, 'smallestreal', ...). Falls back to a dense
    eigendecomposition (this project's instances stay small enough for this to be an
    acceptable fallback, unlike BM-r's own MNIST-scale examples) if Lanczos fails to
    converge, matching bm_linf.m's own eigflag fallback to eig(full(S))."""
    op = Syz_mv_factory(s, s0, lambda_g, lambda_h, prob)
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=prob.nx + 1)
    try:
        vals = spla.eigsh(op, k=1, which="SA", v0=v0, maxiter=n_iter, return_eigenvectors=False)
        return float(vals[0])
    except spla.ArpackNoConvergence:
        dense = np.array([op.matvec(e) for e in np.eye(prob.nx + 1)])
        return float(np.min(np.linalg.eigvalsh(0.5 * (dense + dense.T))))


def lower_bound(u, lambda_lin, lambda_g, lambda_h, prob):
    """Valid lower bound on the true QCQP optimum, regardless of solver tolerance
    (matches bm_linf.m's final `lb = Offset + c'*u - epsilon_gap -
    epsilon_feas*(1+sum(uVopt(:).^2))` -- note this function returns epsilon_gap /
    epsilon_feas / eigval only; riemannian_staircase.py combines them with c0 and
    ||U||^2, which it already has from the solve, into the final lb).

    epsilon_gap literally ports bm_linf.m's `abs(c'*u - b'*y - d'*z + s0)` with
    b = [-b_lin; b_nonlin], d = 0 (zeros(nu) in bm_linf.m -- h has no "d" offset,
    only g does, via b_nonlin), y = [lambda_lin; lambda_g], z = -lambda_h (d'z term
    therefore vanishes regardless of z, kept here only for literal correctness)."""
    s = s_vector(lambda_lin, lambda_g, lambda_h, prob)
    s0 = -s @ u
    eigval = smallest_eigenvalue(s, s0, lambda_g, lambda_h, prob)

    _, b_lin = _linear_bound(prob)
    b_nonlin = _ball_rhs(prob)
    b_dot_y = -b_lin @ lambda_lin + b_nonlin @ np.atleast_1d(lambda_g)
    epsilon_gap = abs(prob.c @ u - b_dot_y + s0)
    epsilon_feas = max(0.0, -eigval)
    return epsilon_gap, epsilon_feas, eigval
