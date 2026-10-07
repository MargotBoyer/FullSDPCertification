"""The only file that imports cyipopt -- wraps bm_objective/bm_constraints/bm_hessian
into cyipopt's Problem callback interface (bm_linf.m's call to knitro_nlp, with IPOPT
standing in for Knitro -- MOSEK can't play that role: this is a genuinely nonconvex
NLP, not a conic problem). Uses the EXACT Hessian (bm_hessian.py), not L-BFGS: an
L-BFGS-only version was tried first and found to leave dual_certificate.py's
eigenvalue test stuck at a clearly nonzero value regardless of IPOPT's convergence
tolerance (1e-7 through 1e-12 made no difference), even though the primal objective
already matched the true SDP relaxation value exactly -- i.e. the *point* was right but
the *multipliers* weren't precise enough for the second-order certificate, consistent
with the paper's Theorem 4.3 (zero duality gap) requiring a certifiably first-order
optimal point in the KKT sense, not just a converged primal. See bm_r/README.md.
"""
import cyipopt
import numpy as np

from bm_objective import objective, gradient, unpack, pack
from bm_constraints import constraints, constraints_bounds, jacobian_structure, jacobian_values
from bm_hessian import hessian_structure, hessian_values


class BMNLP:
    """cyipopt.Problem's required callback object for one fixed rank r."""

    def __init__(self, prob, r):
        self.prob = prob
        self.r = r

    def objective(self, variables):
        return objective(variables, self.prob, self.r)

    def gradient(self, variables):
        return gradient(variables, self.prob, self.r)

    def constraints(self, variables):
        return constraints(variables, self.prob, self.r)

    def jacobian(self, variables):
        return jacobian_values(variables, self.prob, self.r)

    def jacobianstructure(self):
        return jacobian_structure(self.prob, self.r)

    def hessian(self, variables, lagrange, obj_factor):
        # obj_factor is irrelevant: the objective is linear (zero Hessian), and so are
        # the linear (ReLU-inequality) constraints -- only lagrange's g/h slice (see
        # bm_hessian.hessian_values) contributes any curvature at all.
        return hessian_values(lagrange, self.prob, self.r)

    def hessianstructure(self):
        return hessian_structure(self.prob, self.r)


def variable_bounds(prob, r):
    """Box bounds on the full (nx*r) flat vector: prob.LB/UB on column 0 (the point
    itself), unbounded on the extra rank columns V (matches bm_linf.m/bm_l2.m: no
    Params.LB/UB applied to V, only to x{1})."""
    lb = np.full(prob.nx * r, -1e20)
    ub = np.full(prob.nx * r, 1e20)
    lb[:prob.nx] = prob.LB
    ub[:prob.nx] = prob.UB
    return lb, ub


def solve_rank_r(prob, r, variables0, max_iter=1000, tol=1e-7, print_level=0,
                  mult_g0=None, mult_x_L0=None, mult_x_U0=None):
    """Solves the rank-r factorized problem from a warm start. Returns
    (variables_opt, obj_value, info, status_ok).

    mult_g0/mult_x_L0/mult_x_U0: optional dual warm start (Algorithm 1's "initialize...
    with the old multipliers y, z"), typically the PREVIOUS rank's info['mult_g'] /
    ['mult_x_L'] / ['mult_x_U'] -- mult_g0 is reused as-is (the number of constraints
    never changes across ranks), mult_x_L0/mult_x_U0 are padded with zeros for the new
    rank's extra (unbounded, hence inactive-bound) variables. Needed in practice, not
    just in principle: without this, IPOPT recomputes its own initial multiplier guess
    from scratch every rank regardless of how good the primal warm start is, and
    reliably converges back to the SAME dual point -- which looked, before this was
    added, exactly like a stuck/non-functional Riemannian staircase (constant eigval
    across ranks, see bm_r/README.md)."""
    cl, cu = constraints_bounds(prob)
    lb, ub = variable_bounds(prob, r)
    n_vars = prob.nx * r
    n_cons = len(cl)

    nlp = cyipopt.Problem(
        n=n_vars, m=n_cons, problem_obj=BMNLP(prob, r),
        lb=lb, ub=ub, cl=cl, cu=cu,
    )
    nlp.add_option("max_iter", max_iter)
    nlp.add_option("tol", tol)
    nlp.add_option("print_level", print_level)

    solve_kwargs = {}
    if mult_g0 is not None:
        nlp.add_option("warm_start_init_point", "yes")
        nlp.add_option("bound_push", 1e-6)
        nlp.add_option("bound_frac", 1e-6)
        nlp.add_option("mu_init", 1e-8)
        nlp.add_option("mu_strategy", "monotone")
        zl0 = np.zeros(n_vars)
        zu0 = np.zeros(n_vars)
        zl0[:len(mult_x_L0)] = mult_x_L0
        zu0[:len(mult_x_U0)] = mult_x_U0
        solve_kwargs = dict(lagrange=mult_g0, zl=zl0, zu=zu0)

    variables_opt, info = nlp.solve(variables0, **solve_kwargs)
    status_ok = info["status"] == 0
    # cyipopt returns one multiplier per constraint (info['mult_g']) and per bound
    # (info['mult_x_L']/['mult_x_U']) -- dual_certificate.py needs both halves to form
    # S(y,z), since the ReLU-inequality/ball/complementarity constraints AND the box
    # bounds on x_1 (folded into Params.LB/UB in the original, kept as native variable
    # bounds here) jointly define the Lagrangian used to build the slack matrix.
    return variables_opt, info["obj_val"], info, status_ok
