"""
Lagrangian Conic Bundle for SDP certification.

Maximizes the Lagrangian dual:
    L(β) = min_{X⪰0, all non-Aq constraints} <C_0 + Σ β_k Aq_k, X>

Strategy
--------
We use the handler's Mosek task (already fully built by build_and_extract) which
contains ALL constraints:
  - Linear ReLU bounds (D matrix)
  - McCormick quadratic bounds   [not_in_miqcr=True  → kept as hard constraints]
  - X[0,0]=1                     [not_in_miqcr=True  → kept as hard constraint]
  - ReLU complementarity (Aq)    [not_in_miqcr=False → DEACTIVATED for the CB]

Only the ReLU complementarity constraints are dualized (added as β-weighted
objective terms). All other constraints stay hard, including the McCormick bounds
that ensure the SDP remains bounded for any β.

Mosek lower-triangular halved convention (dividing_non_diag=True):
    Off-diagonal (i>j): stored v → Mosek contribution 2·v·X[i,j]

data.Aq[k] storage:
    data.Aq[k, i, j] = Aq_true[i,j]/2  for i≠j  (both sides stored, full matrix)
    data.Aq[k, i, i] = Aq_true[i,i]

Subgradient:
    <Aq_true_k, X> = 2·Σ_{i,j} data.Aq[k,i,j]·X[i,j] − Σ_i data.Aq[k,i,i]·X[i,i]
"""
from __future__ import annotations

import numpy as np
import mosek

from .sdp_data import MiqcrData, MiqcrResult


def _set_objective(task, nm_base, i_base, j_base, v_base, aq_lt, beta, mq):
    """Replace the bar-objective of *task* with C_0 + Σ β_k Aq_k.

    putbarcblocktriplet rejects duplicate (barvar, i, j) entries, so we
    aggregate into a dict keyed by (barvar, i, j) before the call.
    """
    coeff: dict[tuple, float] = {}
    for k in range(len(nm_base)):
        key = (int(nm_base[k]), int(i_base[k]), int(j_base[k]))
        coeff[key] = coeff.get(key, 0.0) + float(v_base[k])

    for k in range(mq):
        bk = float(beta[k])
        if abs(bk) < 1e-16:
            continue
        ar_i, ar_j, ar_v = aq_lt[k]
        for idx in range(len(ar_i)):
            key = (0, int(ar_i[idx]), int(ar_j[idx]))
            coeff[key] = coeff.get(key, 0.0) + bk * float(ar_v[idx])

    if not coeff:
        return

    nm_all, i_all, j_all, v_all = zip(*[(nm, i, j, v) for (nm, i, j), v in coeff.items()])
    task.putbarcblocktriplet(
        np.array(nm_all, dtype=np.int32),
        np.array(i_all,  dtype=np.int32),
        np.array(j_all,  dtype=np.int32),
        np.array(v_all,  dtype=np.float64),
    )


def _extract_X(task, n1: int) -> np.ndarray:
    """Extract the full symmetric solution matrix X.

    getbarxj returns lower-triangular entries column by column, which matches
    np.triu_indices row-major upper-triangular order for symmetric matrices.
    """
    flat = np.array(task.getbarxj(mosek.soltype.itr, 0))
    X = np.zeros((n1, n1))
    tri = np.triu_indices(n1)
    X[tri] = flat
    X = X + X.T - np.diag(np.diag(X))
    return X


def _aq_inner_product(Aq_stored: np.ndarray, X: np.ndarray) -> float:
    """Tr(barA · X) = D + 2·OD_half, matching the Mosek constraint Tr(barA·X)=bq.

    data.Aq is the full symmetric matrix with halved off-diagonal entries
    (stored at both (i,j) and (j,i)). The Mosek barA uses the lower triangle,
    so the correct inner product uses only the lower triangle of Aq_stored:
        Tr(barA·X) = Σ_i Aq[i,i]·X[i,i] + 2·Σ_{i>j} Aq[i,j]·X[i,j]
    """
    D    = float(np.dot(np.diag(Aq_stored), np.diag(X)))
    OD   = float(np.sum(np.tril(Aq_stored, -1) * X))  # Σ_{i>j} Aq[i,j]·X[i,j]
    return D + 2.0 * OD


def run_lagrangian_cb(
    handler,
    data: MiqcrData,
    max_iter: int = 100,
    tol: float = 1e-5,
    step_alpha: float = 1.0,
) -> MiqcrResult:
    """Conic Bundle on the handler's Mosek task with Aq ReLU constraints deactivated.

    The Lagrangian dual is:
        L(β) = min_{feasible X without Aq} <C_0 + Σ β_k Aq_k, X> + cons − β·bq

    We maximise L(β) over β using the subgradient method.

    Parameters
    ----------
    handler : MosekClassicHandler
        Must have been fully built by build_and_extract (initialize_constraints +
        add_to_task called).
    data : MiqcrData
        Problem data.  data.Aq[k] are the ReLU complementarity matrices.
    max_iter : int
        Maximum number of CB iterations.
    tol : float
        Stop when subgradient norm < tol.
    step_alpha : float
        Step-size numerator: t_k = step_alpha / (‖g_k‖ · √(k+1)).
    """
    n, mq, n1 = data.n, data.mq, data.n + 1
    task = handler.task

    # --- Base objective arrays ---
    handler.Objective.format_obj()
    nm_base = np.asarray(handler.Objective.num_matrix, dtype=np.int32)
    i_base  = np.asarray(handler.Objective.i,          dtype=np.int32)
    j_base  = np.asarray(handler.Objective.j,          dtype=np.int32)
    v_base  = np.asarray(handler.Objective.value,      dtype=np.float64)
    cons    = float(handler.Objective.constant)

    # --- Precompute lower-triangular (i≥j) entries of each Aq[k] ---
    aq_lt: list[tuple] = []
    for k in range(mq):
        rows, cols, vals = [], [], []
        for i in range(n1):
            for j in range(i + 1):
                v = data.Aq[k, i, j]
                if abs(v) > 1e-15:
                    rows.append(i); cols.append(j); vals.append(v)
        aq_lt.append((
            np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(vals, dtype=np.float64),
        ))

    # --- Identify ReLU complementarity constraint indices in the Mosek task ---
    # These are is_quadratic=True AND not_in_miqcr=False.
    # McCormick bounds (not_in_miqcr=True) stay hard → SDP bounded for any β.
    aq_indices = [
        i for i, c in enumerate(handler.Constraints.list_cstr)
        if c.get("is_quadratic", False) and not c.get("not_in_miqcr", False)
    ]
    assert len(aq_indices) == mq, (
        f"Expected {mq} Aq constraints in list_cstr, found {len(aq_indices)}"
    )

    # Save original bounds and deactivate Aq constraints
    aq_saved = []
    for idx in aq_indices:
        c = handler.Constraints.list_cstr[idx]
        aq_saved.append((c["bound_type"], c["lb"], c["ub"]))
        task.putconbound(idx, mosek.boundkey.fr, -1e30, 1e30)

    beta      = np.zeros(mq)
    best_lb   = -np.inf
    best_beta = np.zeros(mq)
    best_X    = None
    nb_iter   = 0

    try:
        for iteration in range(max_iter):
            nb_iter = iteration + 1

            # Set penalised objective C_β = C_0 + Σ β_k Aq_k
            _set_objective(task, nm_base, i_base, j_base, v_base,
                           aq_lt, beta, mq)

            task.optimize()
            solsta = task.getsolsta(mosek.soltype.itr)
            if solsta not in (mosek.solsta.optimal,
                              mosek.solsta.prim_and_dual_feas):
                print(f"  [CB iter {iteration}] Non-optimal: {solsta}")
                break

            # L(β) = <C_β, X*> + cons − β·bq
            pobj  = task.getprimalobj(mosek.soltype.itr)
            lb    = pobj + cons - float(np.dot(beta, data.bq))
            X_val = _extract_X(task, n1)

            if lb > best_lb:
                best_lb   = lb
                best_beta = beta.copy()
                best_X    = X_val.copy()

            if mq == 0:
                break

            # Subgradients: g_k = <Aq_true_k, X*> − bq_k
            g = np.array([
                _aq_inner_product(data.Aq[k], X_val) - float(data.bq[k])
                for k in range(mq)
            ])
            g_norm = float(np.linalg.norm(g))

            print(f"  [CB {iteration:3d}] lb={lb:+.6f}  best={best_lb:+.6f}  "
                  f"‖g‖={g_norm:.4f}  ‖β‖={np.linalg.norm(beta):.4f}")

            if g_norm < tol:
                print(f"  [CB] Converged (‖g‖={g_norm:.2e}).")
                break

            # Subgradient ascent step
            t    = step_alpha / (g_norm * np.sqrt(iteration + 1))
            beta = beta + t * g

    finally:
        # Restore Aq constraints and base objective
        for k, idx in enumerate(aq_indices):
            bk, lb_orig, ub_orig = aq_saved[k]
            task.putconbound(idx, bk, lb_orig, ub_orig)
        _set_objective(task, nm_base, i_base, j_base, v_base, [], np.zeros(0), 0)

    # True (un-penalised) objective at the best iterate: <C_0, X*> + cons
    if best_X is not None:
        true_obj = cons
        for k in range(len(nm_base)):
            i, j, v = int(i_base[k]), int(j_base[k]), float(v_base[k])
            if i == j:
                true_obj += v * best_X[i, i]
            else:
                true_obj += 2.0 * v * best_X[i, j]
    else:
        true_obj = best_lb

    print(f"  [CB] Done. best_lb={best_lb:.6f}  true_obj={true_obj:.6f}  iters={nb_iter}")

    return MiqcrResult(
        beta=best_beta.reshape(1, mq) if mq > 0 else np.zeros((n, n)),
        alphaq=best_beta,
        alphabisq=np.zeros(data.pq),
        sol_sdp=best_lb,
        true_obj_sdp=true_obj,
        nb_iter_cb=nb_iter,
    )
