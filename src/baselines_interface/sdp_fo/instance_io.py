"""Serialization of a (network, bounds, sample) certification instance to/from disk.

Runs on either side of the certif <-> baselines conda env boundary: the certif env
(torch/auto_LiRPA/MOSEK) writes the instance after computing W, b, L, U the usual way
(networks/network.py, bounds.py); the baselines env (jax/jax_verify) reads it back
without needing torch/auto_LiRPA installed.
"""
import numpy as np


def save_instance(path, W, b, L, U, label, target_label, input_bounds=(0.0, 1.0)):
    """W, b: per-layer weights/biases (PyTorch convention, W[k] shape (n_out, n_in)).
    L, U: per-layer bounds, length K+1 (L[0]/U[0] = input ball, L[k]/U[k] k>=1 = pre-activation)."""
    K = len(W)
    assert len(b) == K and len(L) == K + 1 and len(U) == K + 1
    data = {"K": K, "label": label, "target_label": target_label,
            "input_bounds_lo": input_bounds[0], "input_bounds_hi": input_bounds[1]}
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
    label = int(npz["label"])
    target_label = int(npz["target_label"])
    input_bounds = (float(npz["input_bounds_lo"]), float(npz["input_bounds_hi"]))
    return W, b, L, U, label, target_label, input_bounds
