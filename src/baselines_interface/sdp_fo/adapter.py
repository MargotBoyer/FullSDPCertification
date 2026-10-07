"""Bridge to the SDP-FO baseline: Dathathri et al. 2020, "Enabling certification of
verification-agnostic networks via memory-efficient semidefinite programming",
implemented in google-deepmind/jax_verify (jax_verify/extensions/sdp_verify).

Must run in the dedicated `baselines` conda env (jax pinned <0.6 for compatibility
with this archived codebase) -- see baselines/README.md.
"""
import time

import jax.numpy as jnp
import numpy as np
import optax

from jax_verify.extensions.sdp_verify import problem as sdpfo_problem
from jax_verify.extensions.sdp_verify import sdp_verify as sdpfo_solver
from jax_verify.extensions.sdp_verify import utils as sdpfo_utils


def build_params(W, b):
    """W, b: per-layer weights/biases, PyTorch convention (W[k] shape (n_out, n_in)).
    jax_verify's predict_mlp uses inputs @ W + b, i.e. W shape (n_in, n_out): transpose."""
    return [(jnp.asarray(np.transpose(Wk)), jnp.asarray(bk)) for Wk, bk in zip(W, b)]


def build_bounds(L, U):
    """L, U: per-layer bounds, length K+1 (FastSDPCertification convention: L[0]/U[0] is
    the input ball, L[k]/U[k] for k>=1 are PRE-activation bounds). Mirrors the IntBound
    layout produced by jax_verify.extensions.sdp_verify.utils.boundprop, so our own
    precomputed bounds (bounds.py / bounds_crown.py) substitute exactly for jax_verify's
    internal boundprop -- this is what makes the comparison to TargetedSDP/UntargetedSDP
    apples-to-apples (same bounds, different SDP relaxation/solve method)."""
    bounds = [sdpfo_utils.IntBound(
        lb=jnp.asarray([L[0]]), ub=jnp.asarray([U[0]]), lb_pre=None, ub_pre=None)]
    for Lk, Uk in zip(L[1:], U[1:]):
        lb_pre, ub_pre = jnp.asarray([Lk]), jnp.asarray([Uk])
        bounds.append(sdpfo_utils.IntBound(
            lb=jnp.maximum(lb_pre, 0.0), ub=jnp.maximum(ub_pre, 0.0),
            lb_pre=lb_pre, ub_pre=ub_pre))
    return bounds


def solve_sdp_fo(W, b, L, U, label, target_label, input_bounds=(0.0, 1.0),
                  **solver_kwargs):
    """Runs SDP-FO on one (sample, target_label) certification instance.

    Returns a dict with `optimal_value` in the same sign convention as results.csv
    (min_j z^y - z^j over the perturbation ball; >=0 means certified robust).
    """
    params = build_params(W, b)
    bounds = build_bounds(L, U)
    verif_instance = sdpfo_utils.make_relu_robust_verif_instance(
        params, bounds=bounds, target_label=target_label, label=label,
        input_bounds=input_bounds)
    sdp_instance = sdpfo_problem.make_sdp_verif_instance(verif_instance)

    # jax_verify.solve_sdp_dual_simple defaults to optax.adam(1e3) (sic) when `opt`
    # is not passed, which diverges -- always supply a sane learning rate.
    solver_kwargs.setdefault("opt", optax.adam(1e-3))

    t0 = time.time()
    verified_ub, info = sdpfo_solver.solve_sdp_dual_simple(sdp_instance, **solver_kwargs)
    elapsed = time.time() - t0

    return {
        "optimal_value": -float(verified_ub),
        "time": elapsed,
        "status": "optimal",
        "label": label,
        "target_label": target_label,
        "best_train_loss": float(info.get("best_train_loss", float("nan"))),
    }
