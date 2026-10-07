"""Serialization of a (network, bounds, sample) BM-r instance across the certif <->
baselines conda env boundary -- same two-env split and reason as
src/baselines_interface/sdp_fo/instance_io.py: the certif env (torch/auto_LiRPA/bounds.py)
writes the instance after computing W, b, L, U the usual way; the baselines env
(cyipopt) reads it back without needing torch installed. No cyipopt/torch import here
either way -- pure numpy, importable from both envs.
"""
import numpy as np


def save_instance(path, W, b, L, U, x, epsilon, ytrue, ytarget, norm="Linf"):
    """W, b: per-layer weights/biases (PyTorch convention, W[k] shape (n_out, n_in)),
    length K (includes the logit layer -- problem.build_problem excludes it itself).
    L, U: per-layer bounds, length K+1 (L[0]/U[0] = input ball, L[k]/U[k] k>=1 =
    pre-activation, FastSDPCertification's bounds.py convention)."""
    K = len(W)
    assert len(b) == K and len(L) == K + 1 and len(U) == K + 1
    data = {"K": K, "ytrue": int(ytrue), "ytarget": int(ytarget), "epsilon": float(epsilon),
            "norm": norm, "x": np.asarray(x, dtype=np.float64)}
    for k in range(K):
        data[f"W_{k}"] = np.asarray(W[k], dtype=np.float64)
        data[f"b_{k}"] = np.asarray(b[k], dtype=np.float64)
    for k in range(K + 1):
        data[f"L_{k}"] = np.asarray(L[k], dtype=np.float64)
        data[f"U_{k}"] = np.asarray(U[k], dtype=np.float64)
    np.savez(path, **data)


def load_instance(path):
    npz = np.load(path, allow_pickle=False)
    K = int(npz["K"])
    W = [npz[f"W_{k}"] for k in range(K)]
    b = [npz[f"b_{k}"] for k in range(K)]
    L = [npz[f"L_{k}"] for k in range(K + 1)]
    U = [npz[f"U_{k}"] for k in range(K + 1)]
    x = npz["x"]
    epsilon = float(npz["epsilon"])
    ytrue = int(npz["ytrue"])
    ytarget = int(npz["ytarget"])
    norm = str(npz["norm"])
    return W, b, L, U, x, epsilon, ytrue, ytarget, norm
