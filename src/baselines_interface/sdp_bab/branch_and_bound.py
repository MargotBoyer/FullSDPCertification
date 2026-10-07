"""Algorithm 1 (Lan, Bruckner & Lomuscio, AAAI'23): Layer SDP-based B&B.

Orchestration only -- every SDP solve reuses sdp_bab_model.LayerSDPBaBNode (itself a
thin, additive subclass of TargetedSDP), so this module touches no file under
solve/sdp_solve/ directly.
"""
import heapq
import itertools
import time
import types

import numpy as np

from bounds import check_stability_neurons

from .eigencuts import compute_eigencuts
from .sdp_bab_model import DEFAULT_KWARGS, LayerSDPBaBNode


def _stable_inactives_neurons(K, n, L, U):
    """Lightweight probe: computes self.stable_inactives_neurons the exact way
    LayerSDPBaBNode's own super().__init__() will (same function, same pruning config,
    via DEFAULT_KWARGS), without building a full model/MOSEK task just to read it off.
    Reused for every node of this sample's B&B tree (L, U -- hence this set -- don't
    change across nodes, only eigencuts_bounds does), so eigencuts.compute_eigencuts's
    zero-substitution only needs to run once per sample, not once per node."""
    probe = types.SimpleNamespace(
        K=K, n=n, L=L, U=U,
        keep_penultimate_actives=DEFAULT_KWARGS["keep_penultimate_actives"],
        pruned_input_neurons=set(), ultimate_layer_use_active_neurons=100000,
    )
    check_stability_neurons(
        probe,
        use_active_neurons=DEFAULT_KWARGS["use_active_neurons"],
        use_inactive_neurons=DEFAULT_KWARGS["use_inactive_neurons"],
    )
    return probe.stable_inactives_neurons


def _initial_t_bounds(eigencuts, L, U):
    """Box bounds [l, u] on t_{i,j} = v_xi . x_i + v_out * x_{i+1,j} from the network's
    precomputed bounds (bounds.py convention: L[k]/U[k] are pre-activation for k>=1).
    Plain interval arithmetic on a linear form -- cheap, exact given L/U."""
    bounds = {}
    for cut in eigencuts:
        i, j = cut.layer, cut.neuron
        v_xi, v_out = cut.v_xi, cut.v_out
        Li, Ui = np.asarray(L[i], dtype=np.float64), np.asarray(U[i], dtype=np.float64)
        pos = v_xi >= 0
        l = float(np.sum(np.where(pos, v_xi * Li, v_xi * Ui)))
        u = float(np.sum(np.where(pos, v_xi * Ui, v_xi * Li)))
        lo_out, up_out = max(float(L[i + 1][j]), 0.0), max(float(U[i + 1][j]), 0.0)
        if v_out >= 0:
            l += v_out * lo_out
            u += v_out * up_out
        else:
            l += v_out * up_out
            u += v_out * lo_out
        bounds[(i, j)] = (l, u)
    return bounds


def _solve_and_score(eigencuts, eigencuts_bounds, common_kwargs, verbose):
    """One B&B node: solve the layer SDP (16) with this node's branch, then evaluate
    every active eigencut's (t, s) at the solution to get the violation distances
    D^k = {d_{i,j} = |s*_{i,j} - (t*_{i,j})^2|} used to pick the next branching variable
    (Algorithm 1, line 3/12). Returns (gamma, distances) or (None, None) if infeasible."""
    model = LayerSDPBaBNode(eigencuts=eigencuts, eigencuts_bounds=eigencuts_bounds, **common_kwargs)
    gamma, X = model.solve_node(verbose=verbose)
    if gamma is None or X is None:
        return None, None
    distances = {}
    for cut in model.eigencuts:  # already restricted to this sample's unpruned neurons
        key = (cut.layer, cut.neuron)
        if key not in model.eigencuts_bounds:
            continue
        _, _, d = model.eval_eigencut_expression_at_solution(cut, X)
        distances[key] = d
    return gamma, distances


def solve_sdp_bab(network, epsilon, norm, x, ytrue, ytarget, L, U, data_index=None,
                   dataset_name=None, network_name=None, folder_name=None,
                   max_time=3600.0, max_nodes=None, verbose=False):
    """Returns a dict: optimal_value (the certified lower bound gamma_bab, same sign
    convention as results.csv), status, n_nodes, time.

    Config fixed to match the paper's base LayerSDP relaxation (Theorem 3's starting
    point): CHORDAL_DECOMPOSITION=True, cuts=["triangularization"], full quadratic
    inter-matrix consistency (LayerSDPBaBNode sets this). max_time: wall-clock budget
    for the whole tree (default 1h, per-run override via max_time=).
    """
    t_start = time.time()
    common_kwargs = dict(
        network=network, epsilon=epsilon, norm=norm, x=x, ytrue=ytrue, ytarget=ytarget,
        L=L, U=U, data_index=data_index, dataset_name=dataset_name,
        network_name=network_name, folder_name=folder_name,
    )

    # Cheap short-circuit: the vanilla LayerSDP relaxation alone (full_quadratic_
    # consistency, no eigenvector cuts) is far cheaper to build than the full cut set
    # (eigencuts.py's O(n_i^2) dense blocks -- see README's "Known scaling limit").
    # By monotonicity (CLAUDE.md: adding valid cuts never decreases the optimum), if
    # this alone already certifies (gamma > 0), the eigenvector-tightened bound can
    # only agree -- skip paying for it.
    gamma_vanilla, _ = _solve_and_score([], {}, common_kwargs, verbose)
    if gamma_vanilla is not None and gamma_vanilla > 0:
        return {"optimal_value": gamma_vanilla, "status": "optimal", "n_nodes": 1,
                "time": time.time() - t_start}

    W, _ = network.extract_weights()
    # Once per sample (not per node: L, U -- hence the pruning and the eigencuts it
    # allows -- are fixed for the whole tree, only eigencuts_bounds changes per node).
    stable_inactives = _stable_inactives_neurons(network.K, network.n, L, U)
    eigencuts = compute_eigencuts(W, stable_inactives_neurons=stable_inactives)

    root_bounds = _initial_t_bounds(eigencuts, L, U)
    gamma0, dist0 = _solve_and_score(eigencuts, root_bounds, common_kwargs, verbose)
    if gamma0 is None:
        return {"optimal_value": float("nan"), "status": "infeasible", "n_nodes": 0,
                "time": time.time() - t_start}

    gamma_bab = gamma0
    if not dist0:
        # No negative-eigenvalue neuron in the whole network: the layer SDP (7) is
        # already exact for this instance (Theorem 1's condition holds trivially).
        return {"optimal_value": gamma_bab, "status": "optimal", "n_nodes": 1,
                "time": time.time() - t_start}

    counter = itertools.count(1)
    heap = [(gamma0, 0, root_bounds, dist0)]
    n_nodes, status = 1, "exhausted"

    while heap:
        if time.time() - t_start > max_time:
            status = "timeout"
            break
        if max_nodes is not None and n_nodes >= max_nodes:
            status = "max_nodes"
            break

        gamma_k, _, bounds_k, dist_k = heapq.heappop(heap)
        gamma_bab = gamma_k
        if gamma_bab > 0:
            status = "optimal"
            break
        if not dist_k:
            status = "optimal"
            break

        branch_key = max(dist_k, key=dist_k.get)
        l, u = bounds_k[branch_key]
        mid = (l + u) / 2.0

        for new_bounds in (
            {**bounds_k, branch_key: (l, mid)},
            {**bounds_k, branch_key: (mid, u)},
        ):
            gamma_c, dist_c = _solve_and_score(eigencuts, new_bounds, common_kwargs, verbose)
            n_nodes += 1
            if gamma_c is None:
                continue  # infeasible child -> pruned, per Algorithm 1 lines 16-23
            heapq.heappush(heap, (gamma_c, next(counter), new_bounds, dist_c))
    else:
        status = "exhausted"

    if not heap and status == "exhausted":
        status = "optimal"  # queue emptied without an early stop: gamma_bab is exact

    return {"optimal_value": gamma_bab, "status": status, "n_nodes": n_nodes,
            "time": time.time() - t_start}
