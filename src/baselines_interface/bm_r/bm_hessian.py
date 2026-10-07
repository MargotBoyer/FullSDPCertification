"""Exact Lagrangian Hessian (bm_linf.m/bm_l2.m's myhess/myhess_mv), fed to IPOPT so its
dual multipliers are precise enough for the certificate in dual_certificate.py to
actually close (confirmed necessary: L-BFGS's approximate Hessian left the primal
objective exactly matching the true SDP relaxation value, but the certificate stuck at
a clearly-nonzero eigenvalue regardless of solver tolerance -- see bm_r/README.md).

The objective and the linear (ReLU-inequality) constraints are linear in x, so only g
(ball) and h (complementarity) contribute curvature -- IDENTICAL structure to
dual_certificate.S_mv (same lambda_g/lambda_h, same arrowhead-like pattern via Wmat's
own nonzero positions), reused directly here rather than re-derived: the Hessian is
exactly S_mv applied independently to EACH of the r columns (bm_linf.m: `H =
kron(speye(r), H)`, i.e. r identical copies of the same nx x nx block, matching the
factorization's block-diagonal structure across columns).

IPOPT wants only the LOWER TRIANGULAR nonzero entries (row >= col) of obj_factor*Hess(f)
+ sum(lagrange_i * Hess(constraint_i)) -- since f is linear (zero Hessian) and the lin
block is linear too, obj_factor is irrelevant here; only `lagrange`'s g/h portion
matters, read off in the same [lin, quad, eq] order constraints() uses.
"""
import numpy as np

from bm_constraints import constraint_sizes
from bm_objective import unpack


_hess_pattern_cache = {}


def hessian_structure(prob, r):
    """Value-independent (rows, cols), lower-triangular (row >= col), built the same
    way as bm_constraints.jacobian_structure -- from Wmat's own fixed nonzero pattern,
    never from evaluating anything. sel_j = ni+j is always > any w_col (Wmat's row j
    only ever references an earlier, hence smaller-index, layer -- see
    eigencuts.py-style note in dual_certificate.py), so the off-diagonal term's
    lower-triangular form is always (sel_j, w_col), never the reverse.
    """
    key = (id(prob), r)
    if key in _hess_pattern_cache:
        return _hess_pattern_cache[key]["rows"], _hess_pattern_cache[key]["cols"]

    Wmat_coo = prob.Wmat.tocoo()
    w_rows, w_cols, w_data = Wmat_coo.row, Wmat_coo.col, Wmat_coo.data
    sel_rows = prob.ni + w_rows  # sel_j for each Wmat nonzero's row j

    img_idx = np.arange(prob.ni)
    sel_idx = prob.ni + np.arange(prob.nu)

    rows, cols = [], []
    for col in range(r):
        offset = col * prob.nx
        rows.append(offset + img_idx)
        cols.append(offset + img_idx)
        rows.append(offset + sel_idx)
        cols.append(offset + sel_idx)
        rows.append(offset + sel_rows)
        cols.append(offset + w_cols)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    _hess_pattern_cache[key] = dict(rows=rows, cols=cols, w_rows=w_rows, w_cols=w_cols,
                                     w_data=w_data, sel_rows=sel_rows)
    return rows, cols


def hessian_values(lagrange, prob, r):
    """Values in EXACTLY the order hessian_structure built its (rows, cols) in.
    `lagrange`: the full per-constraint multiplier vector, same [lin, quad, eq] order
    as constraints() -- only the quad/eq slice is used (lin's Hessian is zero)."""
    key = (id(prob), r)
    hessian_structure(prob, r)  # ensure cache populated
    pat = _hess_pattern_cache[key]

    n_lin, n_quad, n_eq = constraint_sizes(prob)
    lambda_g = lagrange[n_lin:n_lin + n_quad]
    lambda_h = lagrange[n_lin + n_quad:]
    lambda_g_img = (np.broadcast_to(np.atleast_1d(lambda_g), (prob.ni,))
                     if prob.norm == "L2" else lambda_g)

    vals = []
    for _ in range(r):
        vals.append(2.0 * lambda_g_img)
        vals.append(2.0 * lambda_h)
        vals.append(-lambda_h[pat["w_rows"]] * pat["w_data"])
    return np.concatenate(vals)
