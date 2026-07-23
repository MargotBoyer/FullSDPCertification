#!/usr/bin/env python3
"""
Test d'intégration : vérifie que le callback Python est bien appelé
par la Conic Bundle MIQCR à chaque itération SDP.

Lancer depuis la racine du projet :
    python3 scripts/test_miqcr_callback.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.miqcr_interface.sdp_data import MiqcrData
from src.miqcr_interface.miqcr_wrapper import run_miqcr_sdp_phase, register_sdp_solver


def make_test_data(n=2):
    return MiqcrData(
        n=n, nb_int=0, m=0, p=0, mq=0, pq=0,
        Q=np.eye(n), c=np.zeros(n),
        u=np.ones(n), l=np.zeros(n),
        A=np.zeros((0, n)), b=np.zeros(0),
        D=np.zeros((0, n)), e=np.zeros(0),
        Aq=np.zeros((0, n+1, n+1)), bq=np.zeros(0),
        Dq=np.zeros((0, n+1, n+1)), eq=np.zeros(0),
        cons=0.0,
    )


def trivial_sdp_solver(n, mq, pq, q_beta, c_beta, l_beta,
                        x_out, beta_diag, alphaq, alphabisq, sol_out):
    """Solveur factice : X = I (solution primale triviale)."""
    x_out[:] = 0.0
    x_out[0] = 1.0          # X_00 = 1  (terme d'homogénéisation)
    sz = (n + 1) * (n + 2) // 2
    # X_ii = 1 pour i=1..n  (diagonale)
    idx = 0
    for i in range(n + 1):
        if i > 0:
            x_out[idx] = 1.0
        idx += n + 1 - i
    beta_diag[:] = 0.0
    sol_out[0] = float(l_beta)


def run():
    print("=" * 60)
    print("Test : callback Python dans la Conic Bundle MIQCR")
    print("=" * 60)

    data = make_test_data(n=2)
    call_log = []

    def logging_solver(n, mq, pq, q_beta, c_beta, l_beta,
                       x_out, beta_diag, alphaq, alphabisq, sol_out):
        call_log.append({"iter": len(call_log) + 1, "l_beta": float(l_beta)})
        print(f"  [SDP iter {len(call_log)}]  l_beta = {l_beta:.6f}", flush=True)
        trivial_sdp_solver(n, mq, pq, q_beta, c_beta, l_beta,
                           x_out, beta_diag, alphaq, alphabisq, sol_out)

    register_sdp_solver(logging_solver)
    print()

    result = run_miqcr_sdp_phase(data)

    print()
    print("-" * 60)
    print(f"Callback appelé   : {len(call_log)} fois")
    print(f"sol_sdp           : {result.sol_sdp:.6f}")
    print(f"beta (2x2)        :\n{result.beta}")
    print("-" * 60)

    # Vérifications minimales
    assert len(call_log) > 0, "ECHEC : le callback n'a jamais été appelé"
    print("\nOK — le callback Python est bien branché sur la Conic Bundle.")


if __name__ == "__main__":
    run()
