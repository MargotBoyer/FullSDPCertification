"""
Tests des invariants mathématiques de FastSDPCertification.

Invariants vérifiés (voir CLAUDE.md §"Invariants théoriques") :
  1. Monotonicity des coupes (UntargetedSDP)  : val(∅) ≤ val(triang) ≤ val(triang + RLT)
  2. Monotonicity des coupes (TargetedSDP) : val(∅) ≤ val(RLT)
     (TargetedSDP inclut toujours la triangularisation — les coupes s'ajoutent par-dessus)
  3. SDPu ≤ SDPt : val(UntargetedSDP, triang) ≤ val(TargetedSDP_j) pour chaque target j
  4. SDPu ≤ min_j SDPt : val(UntargetedSDP) ≤ min_j val(TargetedSDP_j)
  5. Relaxation chordale : val(TargetedSDP, CHORDAL_DECOMPOSITION=True) ≤ val(TargetedSDP, CHORDAL_DECOMPOSITION=False)

Réseau de test : blob_4x10 — K=5, n=[2,10,10,10,10,3], epsilon=0.5, Linf.
Bornes pré-calculées (alpha-CROWN) chargées depuis data/bounds/blob_4x10-0.5_linf.csv.

Usage :
    conda activate certif
    pytest tests/test_sdp_invariants.py -v
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
# conftest.py insère SRC_PATH en premier ; on garde la ligne pour exécution directe
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pandas as pd
import pytest
import torch

from networks.network import ReLUNN
from solve.sdp_solve.SDPmodels.Targeted_SDP import TargetedSDP
from solve.sdp_solve.SDPmodels.Untargeted_SDP import UntargetedSDP
from fastsdp_tools import get_project_path

# Tolérance numérique : les SDP peuvent avoir un gap primal-dual résiduel
ATOL = 1e-3

DATA_INDEX = 0
EPSILON = 0.5
NORM = "Linf"

NETWORK_PATH = "data/models/blob_adv_blob_nn_4x10.pt"
DATASET_PATH = "data/datasets/blob_dataset.pth"
BOUNDS_PATH = "data/bounds/blob_4x10-0.5_linf.csv"


# ──────────────────────────────────────────────────────────────────
# Chargement du réseau, du dataset, et des bornes pré-calculées
# ──────────────────────────────────────────────────────────────────

def _load_network() -> ReLUNN:
    return ReLUNN.from_pth(get_project_path(NETWORK_PATH), bb_beta_crown=False)


def _load_sample(data_index: int):
    raw = torch.load(get_project_path(DATASET_PATH), weights_only=False)
    dataset = raw["dataset"]
    x, label = dataset[data_index]
    x_t = x.view(-1).float()
    ytrue = int(label.item())
    return x_t, ytrue


def _load_bounds(network: ReLUNN, data_index: int):
    """Charge L, U depuis le CSV de bornes pré-calculées."""
    df = pd.read_csv(get_project_path(BOUNDS_PATH))
    row = df[df["data_index"] == data_index].iloc[0]
    K = network.K
    n = network.n
    L = []
    U = []
    for k in range(K + 1):
        L.append([float(row[f"LB_Layer_{k}_Neuron_{j}"]) for j in range(n[k])])
        U.append([float(row[f"UB_Layer_{k}_Neuron_{j}"]) for j in range(n[k])])
    return L, U


# ──────────────────────────────────────────────────────────────────
# Fixture partagée (module-scope = une seule instance pour tous les tests)
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    """
    Retourne un dict avec toutes les données de l'instance de certification :
    network, x, ytrue, ytargets, L, U, epsilon, folder.
    """
    net = _load_network()
    x_t, ytrue = _load_sample(DATA_INDEX)
    L, U = _load_bounds(net, DATA_INDEX)

    n_out = net.n[net.K]
    ytargets = [j for j in range(n_out) if j != ytrue]

    assert ytargets, "Tous les logits sont dominés — choisir un autre data_index."

    has_unstable = any(
        L[k][j] < 0 < U[k][j]
        for k in range(1, net.K)
        for j in range(net.n[k])
    )
    assert has_unstable, "Aucun neurone instable — le SDP ne sera pas informatif."

    folder = str(tmp_path_factory.mktemp("sdp_invariants"))
    return dict(
        network=net,
        x=x_t,
        ytrue=ytrue,
        ytargets=ytargets,
        L=L,
        U=U,
        epsilon=EPSILON,
        folder=folder,
    )


# ──────────────────────────────────────────────────────────────────
# Fixture autouse : weights_nn.txt écrit dans tmp_path (pas dans cwd)
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def chdir_tmp(tmp_path):
    """Redirige l'écriture de weights_nn.txt vers un répertoire temporaire."""
    orig = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(orig)


# ──────────────────────────────────────────────────────────────────
# Helpers : construction et résolution des modèles
# ──────────────────────────────────────────────────────────────────

def _base_kwargs(inst: dict) -> dict:
    return dict(
        network=inst["network"],
        epsilon=inst["epsilon"],
        x=inst["x"],
        ytrue=inst["ytrue"],
        L=inst["L"],
        U=inst["U"],
        RLT_props=[0.2],
        use_active_neurons=False,
        use_inactive_neurons=False,
        keep_penultimate_actives=True,
        norm=NORM,
        bounds_method="IBP",
        network_name="blob_4x10",
        dataset_name="blob",
        data_index=DATA_INDEX,
        INPUT_IN_VARIABLES=True,
    )


def _lan(inst: dict, ytarget: int, cuts: list, CHORDAL_DECOMPOSITION: bool = True, tag: str = "") -> TargetedSDP:
    folder = os.path.join(inst["folder"], "Targeted" + tag)
    return TargetedSDP(
        ytarget=ytarget,
        cuts=cuts,
        CHORDAL_DECOMPOSITION=CHORDAL_DECOMPOSITION,
        folder_name=folder,
        **_base_kwargs(inst),
    )


def _md(inst: dict, cuts: list, tag: str = "") -> UntargetedSDP:
    folder = os.path.join(inst["folder"], "md" + tag)
    return UntargetedSDP(
        cuts=cuts,
        CHORDAL_DECOMPOSITION=True,
        folder_name=folder,
        **_base_kwargs(inst),
    )


def _solve(solver) -> float | None:
    """
    Lance solve() et renvoie l'optimal_value (float) ou None si le solveur
    n'a pas atteint un statut optimal.
    """
    solver.solve()
    df = solver.benchmark_dataframe
    if df is None or "optimal_value" not in df.columns:
        return None
    val = df["optimal_value"].iloc[-1]
    return float(val) if val is not None else None


def _require(val: float | None, label: str) -> float:
    """Skip le test si val est None (problème numérique ou non-optimal)."""
    if val is None:
        pytest.skip(f"{label} : solveur non-optimal, invariant non vérifiable")
    return val


# ──────────────────────────────────────────────────────────────────
# 1. Monotonicity des coupes — UntargetedSDP
# ──────────────────────────────────────────────────────────────────

class TestMonotoniciteUntargetedSDP:
    """
    Ajouter des coupes ne peut qu'augmenter (ou laisser égale) la valeur optimale.
    Invariant théorique : violation = bug certain (CLAUDE.md).
    """

    def test_triangularization_ameliore_baseline(self, instance):
        v0 = _require(_solve(_md(instance, cuts=[], tag="_mono0")), "UntargetedSDP(∅)")
        v1 = _require(_solve(_md(instance, cuts=["triangularization"], tag="_mono1")), "UntargetedSDP(triang)")
        assert v1 >= v0 - ATOL, (
            f"Monotonicity violée : UntargetedSDP(triang)={v1:.6f} < UntargetedSDP(∅)={v0:.6f}"
        )

    def test_rlt_ameliore_triangularization(self, instance):
        v1 = _require(_solve(_md(instance, cuts=["triangularization"], tag="_rlt1")), "UntargetedSDP(triang)")
        v2 = _require(_solve(_md(instance, cuts=["triangularization", "RLT"], tag="_rlt2")), "UntargetedSDP(triang+RLT)")
        assert v2 >= v1 - ATOL, (
            f"Monotonicity violée : UntargetedSDP(triang+RLT)={v2:.6f} < UntargetedSDP(triang)={v1:.6f}"
        )


# ──────────────────────────────────────────────────────────────────
# 2. Monotonicity des coupes — TargetedSDP
# ──────────────────────────────────────────────────────────────────

class TestMonotoniciteTargetedSDP:
    """
    TargetedSDP inclut toujours la triangularisation.
    On vérifie que l'ajout de RLT améliore (ou laisse égale) la valeur.
    """

    def test_rlt_ameliore_baseline(self, instance):
        j = instance["ytargets"][0]
        v0 = _require(_solve(_lan(instance, j, cuts=[], tag="_rlt0")), "TargetedSDP(∅)")
        v1 = _require(_solve(_lan(instance, j, cuts=["RLT"], tag="_rlt1")), "TargetedSDP(RLT)")
        assert v1 >= v0 - ATOL, (
            f"Monotonicity violée : TargetedSDP(RLT)={v1:.6f} < TargetedSDP(∅)={v0:.6f}"
        )

    def test_rlt_ameliore_tous_les_targets(self, instance):
        """La monotonicity tient pour tous les targets, pas seulement le premier."""
        for j in instance["ytargets"]:
            v0 = _solve(_lan(instance, j, cuts=[], tag=f"_j{j}_rlt0"))
            v1 = _solve(_lan(instance, j, cuts=["RLT"], tag=f"_j{j}_rlt1"))
            if v0 is None or v1 is None:
                continue
            assert v1 >= v0 - ATOL, (
                f"Monotonicity violée pour j={j} : TargetedSDP(RLT)={v1:.6f} < TargetedSDP(∅)={v0:.6f}"
            )


# ──────────────────────────────────────────────────────────────────
# 3. SDPu ≤ SDPt — invariant central du papier
# ──────────────────────────────────────────────────────────────────

class TestSDPuLeqSDPt:
    """
    val(SDPu) ≤ val(SDPt^j) pour tout j.
    SDPU est une relaxation de SDPT^j (son ensemble admissible est plus grand).
    Invariant théorique : violation = bug certain (CLAUDE.md).
    """

    def test_sdpu_leq_sdpt_chaque_target(self, instance):
        """UntargetedSDP(triang) ≤ TargetedSDP_j pour chaque target j individuellement."""
        v_u = _require(
            _solve(_md(instance, cuts=["triangularization"], tag="_uleqt_u")),
            "UntargetedSDP(triang)"
        )
        any_optimal = False
        for j in instance["ytargets"]:
            v_t = _solve(_lan(instance, j, cuts=[], tag=f"_uleqt_t{j}"))
            if v_t is None:
                continue
            any_optimal = True
            assert v_u <= v_t + ATOL, (
                f"SDPu ≤ SDPt violé pour j={j} : "
                f"val(SDPu)={v_u:.6f} > val(SDPt[j={j}])={v_t:.6f}"
            )
        if not any_optimal:
            pytest.skip("Aucun TargetedSDP n'a atteint l'optimalité")

    def test_sdpu_leq_min_sdpt(self, instance):
        """val(SDPu) ≤ min_j val(SDPt^j)."""
        v_u = _require(
            _solve(_md(instance, cuts=["triangularization"], tag="_min_u")),
            "UntargetedSDP(triang)"
        )
        vt_vals = [
            v
            for j in instance["ytargets"]
            if (v := _solve(_lan(instance, j, cuts=[], tag=f"_min_t{j}"))) is not None
        ]
        if not vt_vals:
            pytest.skip("Aucun TargetedSDP n'a atteint l'optimalité")
        min_vt = min(vt_vals)
        assert v_u <= min_vt + ATOL, (
            f"SDPu ≤ min SDPt violé : "
            f"val(SDPu)={v_u:.6f} > min_j val(SDPt)={min_vt:.6f}"
        )

    def test_sdpu_sans_coupes_leq_sdpt_sans_coupes(self, instance):
        """
        Sans coupes supplémentaires :
        val(UntargetedSDP(∅)) ≤ val(TargetedSDP_j(∅)) pour tout j.
        UntargetedSDP(∅) n'a pas de triangularisation explicite, TargetedSDP(∅) l'a toujours —
        l'inégalité tient a fortiori.
        """
        v_u = _require(
            _solve(_md(instance, cuts=[], tag="_nocutu")),
            "UntargetedSDP(∅)"
        )
        for j in instance["ytargets"]:
            v_t = _solve(_lan(instance, j, cuts=[], tag=f"_nocut_t{j}"))
            if v_t is None:
                continue
            assert v_u <= v_t + ATOL, (
                f"SDPu(∅) ≤ SDPt(∅) violé pour j={j} : "
                f"val(UntargetedSDP)={v_u:.6f} > val(TargetedSDP[j={j}])={v_t:.6f}"
            )


# ──────────────────────────────────────────────────────────────────
# 4. Relaxation chordale
# ──────────────────────────────────────────────────────────────────

class TestRelaxationChordale:
    """
    val(SDPt chordale) ≤ val(SDPt classique).
    La décomposition chordale avec contraintes linéaires seulement est une relaxation
    de la formulation classique (une matrice unique). CLAUDE.md §"Invariants théoriques".
    """

    def test_chordal_leq_classique_par_target(self, instance):
        for j in instance["ytargets"]:
            v_classic = _solve(_lan(instance, j, cuts=[], CHORDAL_DECOMPOSITION=False, tag=f"_chor_classic{j}"))
            v_chordal = _solve(_lan(instance, j, cuts=[], CHORDAL_DECOMPOSITION=True, tag=f"_chor_chordal{j}"))
            if v_classic is None or v_chordal is None:
                continue
            assert v_chordal <= v_classic + ATOL, (
                f"Relaxation chordale violée pour j={j} : "
                f"val(chordale)={v_chordal:.6f} > val(classique)={v_classic:.6f}"
            )
