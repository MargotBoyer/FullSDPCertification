"""Constraints of the rank-r Burer-Monteiro factorized problem (bm_linf.m/bm_l2.m's
mycon), vectorized over numpy/scipy.sparse -- no per-neuron Python loops.

Three constraint families, concatenated in this fixed order:
  1. linear_ineq  (2*nu rows): ReLU inequality side, A_lin @ U[:,0] <= b_lin
       A_lin = [Wmat - Smat; -Smat], i.e. "z_{k+1} >= W z_k + b" and "z_{k+1} >= 0".
       Only acts on column 0 (the actual point) -- zero Jacobian on V's columns.
  2. quad_ineq    (ni rows for Linf, 1 row for L2): the perturbation ball, as a
       quadratic diag constraint on P[x_1,x_1] (see problem.py / module docstring in
       eigencuts.py-style derivation): (x_1 - center)(x_1 - center)^T lifted to the
       rank-r factorization x_1 x_1^T + V_1 V_1^T.
  3. equality     (nu rows): ReLU complementarity, diag(P[z,z]) - W P[z,z'] - b P[z] = 0,
       lifted the same way.

Box bounds (input: this project's L[0]/U[0]; hidden: max(L_k,0)/max(U_k,0), both from
problem.py) are NOT included here -- they're native variable bounds in ipopt_backend.py
(cheaper for IPOPT than general constraints), applied to column 0 only.
"""
import numpy as np
import scipy.sparse as sp

from bm_objective import unpack


def _pre_post(U, prob):
    pre = np.asarray(prob.Wmat @ U)
    pre = pre.copy()
    pre[:, 0] += prob.bvec
    post = U[prob.sel, :]
    return pre, post


def _linear_bound(prob):
    A_lin = sp.vstack([prob.Wmat - prob.Smat, -prob.Smat], format="csr")
    b_lin = np.concatenate([-prob.bvec, np.zeros(prob.nu)])
    return A_lin, b_lin


def constraint_sizes(prob):
    n_lin = 2 * prob.nu
    n_quad = prob.ni if prob.norm == "Linf" else 1
    n_eq = prob.nu
    return n_lin, n_quad, n_eq


def constraints_bounds(prob):
    """Returns (cl, cu) for the full concatenated constraint vector. All three
    families are returned by constraints() already shifted to a "<= 0" / "== 0" form
    (lin = A_lin@u - b_lin, quad = ... - bound^2, eq = ... already centered at 0) --
    so cu is 0 throughout, never b_lin (previously double-subtracted b_lin: once
    inside constraints()'s own `lin` formula, again here -- caught by a feasibility
    sanity check: a forward-propagated, exactly-feasible point should show exactly
    zero violation, and it didn't until this was fixed)."""
    n_lin, n_quad, n_eq = constraint_sizes(prob)
    cl = np.concatenate([-np.inf * np.ones(n_lin), -np.inf * np.ones(n_quad), np.zeros(n_eq)])
    cu = np.zeros(n_lin + n_quad + n_eq)
    return cl, cu


def _ball_err(U, prob):
    """err[:, col] = U[img, col] - center, with center subtracted only from col 0."""
    err = U[prob.img, :].copy()
    if prob.norm == "Linf":
        center = (prob.UB[prob.img] + prob.LB[prob.img]) / 2.0
    else:
        center = prob.image_center
    err[:, 0] -= center
    return err


def constraints(variables, prob, r):
    U = unpack(variables, prob.nx, r)
    A_lin, b_lin = _linear_bound(prob)
    lin = A_lin @ U[:, 0] - b_lin

    err = _ball_err(U, prob)
    if prob.norm == "Linf":
        bound = (prob.UB[prob.img] - prob.LB[prob.img]) / 2.0
        quad = np.sum(err ** 2, axis=1) - bound ** 2
    else:
        quad = np.array([np.sum(err ** 2) - prob.radius ** 2])

    pre, post = _pre_post(U, prob)
    eq = np.sum(post * (post - pre), axis=1)

    return np.concatenate([lin, quad, eq])


def constraints_jacobian_dense(variables, prob, r):
    """Dense (n_constraints, nx*r) Jacobian -- used by the finite-difference check and
    as a simple, obviously-correct reference; jacobian_structure/jacobian_values (same
    values, fully analytic sparsity) are what ipopt_backend.py actually uses."""
    rows, cols = jacobian_structure(prob, r)
    vals = jacobian_values(variables, prob, r)
    n_lin, n_quad, n_eq = constraint_sizes(prob)
    dense = np.zeros((n_lin + n_quad + n_eq, prob.nx * r))
    dense[rows, cols] = vals
    return dense


_structure_cache = {}


def jacobian_structure(prob, r):
    """Value-INDEPENDENT (rows, cols) of every Jacobian entry, built purely from
    Wmat/Smat/A_lin's own fixed nonzero positions -- never by evaluating the Jacobian
    at some probe point and reading off scipy.sparse's inferred structure from the
    result. That approach (tried first) breaks the moment any factor in a sparse
    product is transiently exactly 0, which happens routinely here (a ReLU neuron's
    `post` value legitimately hits exactly 0 at many feasible/optimal points, including
    IPOPT's actual iterates, not just at contrived test points) -- IPOPT requires
    jacobian() to return exactly as many values as jacobianstructure() declared, every
    single call, so the structure must be computed without ever looking at values.

    w_rows/w_cols (Wmat's own fixed nonzero pattern) and s_rows/s_cols (Smat's, a fixed
    bijection row j <-> col ni+j) never coincide (see module note below `jacobian_values`
    for why), so the two eq-block pieces are concatenated, never summed -- no risk of
    scipy's duplicate-index summation silently changing nnz either.
    """
    key = (id(prob), r)
    if key in _structure_cache:
        return _structure_cache[key]["rows"], _structure_cache[key]["cols"]

    n_lin, n_quad, n_eq = constraint_sizes(prob)
    A_lin, _ = _linear_bound(prob)
    A_lin_coo = A_lin.tocoo()
    Wmat_coo = prob.Wmat.tocoo()
    w_rows, w_cols, w_data = Wmat_coo.row, Wmat_coo.col, Wmat_coo.data
    s_rows = np.arange(prob.nu)
    s_cols = prob.ni + np.arange(prob.nu)
    quad_row_out = np.arange(prob.ni) if prob.norm == "Linf" else np.zeros(prob.ni, dtype=int)

    row_offset_quad = n_lin
    row_offset_eq = n_lin + n_quad

    rows = [A_lin_coo.row]
    cols = [A_lin_coo.col]  # column-0 block only -> flat col offset 0
    for col in range(r):
        offset = col * prob.nx
        rows.append(row_offset_quad + quad_row_out)
        cols.append(offset + np.arange(prob.ni))
        rows.append(row_offset_eq + w_rows)
        cols.append(offset + w_cols)
        rows.append(row_offset_eq + s_rows)
        cols.append(offset + s_cols)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    _structure_cache[key] = dict(rows=rows, cols=cols, A_lin_data=A_lin_coo.data,
                                  w_rows=w_rows, w_cols=w_cols, w_data=w_data,
                                  s_rows=s_rows, s_cols=s_cols)
    return rows, cols


def jacobian_values(variables, prob, r):
    """Values in EXACTLY the order jacobian_structure built its (rows, cols) in."""
    jacobian_structure(prob, r)  # ensure cache populated
    s = _structure_cache[(id(prob), r)]
    U = unpack(variables, prob.nx, r)
    err = _ball_err(U, prob)
    pre, post = _pre_post(U, prob)

    vals = [s["A_lin_data"]]
    for col in range(r):
        vals.append(2.0 * err[:, col])
        vals.append(-post[s["w_rows"], col] * s["w_data"])
        vals.append(2.0 * post[:, col] - pre[:, col])
    return np.concatenate(vals)
