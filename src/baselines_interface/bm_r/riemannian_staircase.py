"""The outer loop of BM-r (bm_linf.m/bm_l2.m's main body, after mycon/myobj/S_mat):
solve the rank-r factorized NLP, check the dual certificate (dual_certificate.py), and
either stop (certificate holds: the current point is PROVABLY the SDP relaxation's
global optimum) or escape along the certificate's negative-eigenvalue direction and
retry at rank r+1, up to MaxSearchRank.

IMPORTANT, read before using optimal_value -- see bm_r/README.md "Known issue: the
certificate does not reliably close" for the full writeup: the dual certificate (the
whole point of the Burer-Monteiro approach -- a PROOF that the point found is the SDP
relaxation's global optimum, not just a local NLP optimum) has NOT been gotten to
close reliably with this IPOPT port, despite several fixes (exact Hessian, dual
warm-start, barrier-parameter tuning) that measurably helped without fully resolving
it. What IS solidly verified: the PRIMAL value (c.u + c0) empirically matches this
project's own SDP-IP (MOSEK) solve on the one real network tested so far, to 6 decimal
places. Per explicit user decision, solve_bm_r therefore returns the raw primal value
as `optimal_value` (status reported separately), NOT the certified lower bound -- which
remains available as `certified_lower_bound` for anyone who wants the
(currently quite loose) rigorously-valid number instead.

Warm start: bm_linf.m seeds u0 with a PGD attack's result (pgd_linf.m/pgd_l2.m), then
forward-propagates it through the network. This port uses the perturbation ball's
center (x itself) forward-propagated instead -- still a genuinely FEASIBLE starting
point (correctness doesn't depend on which one), just not adversarially-informed.
"""
import numpy as np

from bm_objective import unpack, pack
from bm_constraints import constraint_sizes
from ipopt_backend import solve_rank_r
from dual_certificate import lower_bound, s_vector, Syz_mv_factory


def feasible_warm_start(prob, r, x0_input=None, v_noise_scale=1e-10, seed=0):
    """Forward-propagates x0_input (default: the ball center) through the network
    (post_k = ReLU(W_k @ post_{k-1} + b_k)) to build an exactly-feasible U[:, 0], with
    tiny random noise on the extra r-1 rank columns (matches bm_linf.m's
    `V0 = 1e-10*randn(nx, StartSearchRank-1)`)."""
    if x0_input is None:
        x0_input = (prob.LB[prob.img] + prob.UB[prob.img]) / 2.0
    u = np.zeros(prob.nx)
    u[prob.img] = x0_input
    row_off = 0
    for size in prob.layer_sizes:
        rows = slice(row_off, row_off + size)
        pre = np.asarray(prob.Wmat[rows, :] @ u).reshape(-1) + prob.bvec[rows]
        u[prob.ni + row_off: prob.ni + row_off + size] = np.maximum(pre, 0.0)
        row_off += size

    U0 = np.zeros((prob.nx, r))
    U0[:, 0] = u
    rng = np.random.default_rng(seed)
    if r > 1:
        U0[:, 1:] = v_noise_scale * rng.normal(size=(prob.nx, r - 1))
    return pack(U0)


def _escape_eigvec(s, s0, lambda_g, lambda_h, prob, seed):
    op = Syz_mv_factory(s, s0, lambda_g, lambda_h, prob)
    import scipy.sparse.linalg as spla
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=prob.nx + 1)
    try:
        _, eigvecs = spla.eigsh(op, k=1, which="SA", v0=v0, maxiter=500)
        return eigvecs[:, 0]
    except spla.ArpackNoConvergence:
        dense = np.array([op.matvec(e) for e in np.eye(prob.nx + 1)])
        _, v = np.linalg.eigh(0.5 * (dense + dense.T))
        return v[:, 0]


def solve_bm_r(prob, max_search_rank_extra=6, max_iter=1000, tol=1e-7,
               feas_tol=1e-6, x0_input=None, verbose=False):
    """Returns a dict:
      optimal_value: the raw primal objective (c.u + c0) at the last rank tried --
        same sign convention as results.csv (>=0 means certified robust IF status is
        "certified"; for "uncertified_max_rank" this is an empirically-observed value
        matching the SDP relaxation in testing, but NOT backed by the dual proof --
        see the module docstring and bm_r/README.md before treating it as a certified
        bound).
      certified_lower_bound: the rigorously-valid but (currently) loose Prop. 3.1 bound.
      status: "certified" (dual certificate closed) / "uncertified_max_rank" (ran out
        of rank budget without closing) / "solve_failed".
      rank, epsilon_gap, epsilon_feas, eigval: diagnostics.
    """
    ell = len(prob.layer_sizes)
    start_rank = ell + 1
    max_rank = ell + max_search_rank_extra

    variables = feasible_warm_start(prob, start_rank, x0_input)
    n_lin, n_quad, n_eq = constraint_sizes(prob)
    mult_g0 = mult_x_L0 = mult_x_U0 = None  # no dual warm start for the first (lowest-rank) solve

    for r in range(start_rank, max_rank + 1):
        v_opt, obj_val, info, ok = solve_rank_r(
            prob, r, variables, max_iter=max_iter, tol=tol,
            mult_g0=mult_g0, mult_x_L0=mult_x_L0, mult_x_U0=mult_x_U0)
        if not ok:
            return {"optimal_value": float("nan"), "certified_lower_bound": float("nan"),
                    "status": "solve_failed", "rank": r, "ipopt_status": info["status_msg"]}

        U = unpack(v_opt, prob.nx, r)
        u = U[:, 0]
        primal_value = float(prob.c @ u + prob.c0)
        mult_g = info["mult_g"]
        lambda_lin = mult_g[:n_lin]
        lambda_g = mult_g[n_lin:n_lin + n_quad]
        lambda_h = mult_g[n_lin + n_quad:]

        epsilon_gap, epsilon_feas, eigval = lower_bound(u, lambda_lin, lambda_g, lambda_h, prob)
        certified_lb = prob.c0 + prob.c @ u - epsilon_gap - epsilon_feas * (1.0 + float(np.sum(U ** 2)))

        if verbose:
            v_norm = float(np.linalg.norm(U[:, 1:])) if r > 1 else 0.0
            print(f"[bm_r] rank={r} status={info['status']} primal={primal_value:.6f} "
                  f"eigval={eigval:.4e} epsilon_feas={epsilon_feas:.4e} "
                  f"certified_lb={certified_lb:.6f} ||V||={v_norm:.4e}")

        if epsilon_feas <= feas_tol:
            return {"optimal_value": primal_value, "certified_lower_bound": certified_lb,
                    "status": "certified", "rank": r, "epsilon_gap": epsilon_gap,
                    "epsilon_feas": epsilon_feas, "eigval": eigval}

        if r == max_rank:
            return {"optimal_value": primal_value, "certified_lower_bound": certified_lb,
                    "status": "uncertified_max_rank", "rank": r, "epsilon_gap": epsilon_gap,
                    "epsilon_feas": epsilon_feas, "eigval": eigval}

        # Escape direction: smallest eigenvector of S(y,z) (nx+1-dim: corner + border).
        s = s_vector(lambda_lin, lambda_g, lambda_h, prob)
        s0 = -s @ u
        eigvec = _escape_eigvec(s, s0, lambda_g, lambda_h, prob, seed=r)

        escape_dir = eigvec[0] * u + eigvec[1:]
        new_col = 0.01 * escape_dir
        U_new = np.hstack([U, new_col.reshape(-1, 1)])
        variables = pack(U_new)
        mult_g0 = info["mult_g"]  # same length every rank (constraint count is rank-independent)
        mult_x_L0, mult_x_U0 = info["mult_x_L"], info["mult_x_U"]  # padded for new vars in solve_rank_r

    return {"optimal_value": float("nan"), "certified_lower_bound": float("nan"),
            "status": "solve_failed", "rank": max_rank}
