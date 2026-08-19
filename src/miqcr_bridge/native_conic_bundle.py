"""
OBSOLETE / MAUVAIS LEVIER — laissé en place comme trace documentee d'une
impasse, ne pas reutiliser ni brancher. Voir task-dynamic-conic-bundle.md,
section "Pivot : abandon du bundle from-scratch, pont direct vers ConicBundle"
(2026-08-18) pour le diagnostic complet : la Conic Bundle pilotee par
miqcr_pyapi.c ne dualise QUE les McCormick internes de MIQCR (beta_1..4,
derivees des bornes de boite u_i/l_i) -- Aq/Dq (ou l'on pousse nos coupes RLT
ci-dessous) restent des contraintes DURES dans chaque sous-probleme SDP,
quel que soit beta. Le pont ci-dessous calcule donc h(theta) pour une
relaxation differente de celle de dynamic_conic_bundle (McCormick dualise,
RLT dure), non comparable par dualite faible -- confirme empiriquement
(convergence stable a -3.304521 sur data_index=2/target=0, alors qu'une
valeur -3.267045 -- provenant de dynamic_conic_bundle -- est independamment
verifiee valide via resolve_dualized, ce qui est impossible si ce pont
calculait la meme fonction concave).

Le bon pont est src/miqcr_bridge/conicbundle_native/ (cb_wrapper.py) : lien
direct vers la vraie librairie ConicBundle (cb_cinterface.h), oracle base sur
handler.resolve_dualized -- pas de dependance a la structure MIQP/McCormick
de MIQCR.

---

Pont entre dynamic_conic_bundle (dualisation des coupes RLT, cf. main.pdf 5.4)
et la vraie Conic Bundle de MIQCR (src/miqcr_bridge/miqcr_pyapi.c), au lieu du
bundle proximal Python "from scratch" de src/dynamic_conic_bundle/.

Réutilise l'infrastructure déjà validée par dynamic_conic_bundle :
    handler.setup_dualization(dualizable_names)   # désactive les contraintes RLT
    handler.get_dualized_constraints_data()        # triplets signés + rhs (forme <=)

et l'infrastructure déjà validée par miqcr_bridge :
    sdp_data.extract_miqcr_data / _build_cstr_matrix / _build_flat_order / _fill_bounds
    miqcr_wrapper.run_miqcr_sdp_phase / register_sdp_solver
    lagrangian_cb._aq_inner_product

Principe : à chaque itération, la Conic Bundle C calcule elle-même l'objectif
pénalisé C_beta = Q + Σ beta_k * Aq_k (resp. Dq_k) et nous le donne déjà combiné
(q_beta, c_beta, l_beta) — on n'a donc jamais besoin de connaître beta
explicitement. Le callback se contente de :
    1. pousser (q_beta, c_beta) comme objectif du task MOSEK déjà construit par
       le handler (contraintes RLT désactivées, tout le reste — McCormick, ReLU,
       triangularisation, bornes — reste dur, exactement comme resolve_dualized) ;
    2. résoudre ;
    3. renvoyer X* (via getbarxj, même convention que le format MIQCR) et les
       "duaux" <Aq_k, X*> / <Dq_k, X*> (= sous-gradients) attendus par la lib C.

Convention de signe/homogénéisation : identique à handler_classic.resolve_dualized
(valeurs déjà /2 hors-diagonale, cf. dividing_non_diag=True), donc le corner (0,0)
de l'objectif ne code PAS la constante — celle-ci reste un scalaire Python ajouté
après coup, exactement comme _cb_base_obj_constant dans resolve_dualized.
"""
from __future__ import annotations

import numpy as np
import mosek

from . import sdp_data as _sdp_data
from .sdp_data import MiqcrData, MiqcrResult
from .lagrangian_cb import _aq_inner_product
from .miqcr_wrapper import run_miqcr_sdp_phase, register_sdp_solver


def build_miqcr_data_for_dualization(handler, dualizable_names: list[str]) -> MiqcrData:
    """
    Construit un MiqcrData où seules les contraintes DUALIZABLE nommées dans
    `dualizable_names` (typiquement les coupes RLT, cf. mark_current_dualizable())
    apparaissent en Aq/Dq. Toutes les autres contraintes du handler (McCormick,
    ReLU, triangularisation, bornes, ...) restent dures — elles sont déjà présentes
    dans handler.task et ne sont PAS repassées à MIQCR : le callback SDP les
    respecte automatiquement puisqu'il résout directement handler.task.

    Appelle handler.setup_dualization(dualizable_names), qui désactive ces
    contraintes dans le task (mosek.boundkey.fr) — doit être fait une seule fois
    avant run_native_conic_bundle().
    """
    handler.setup_dualization(dualizable_names)

    indexes = handler.indexes_matrices
    n_vars = indexes.get_shape_matrix(0) - 1
    flat_order = _sdp_data._build_flat_order(indexes)

    u_flat = np.zeros(n_vars)
    l_flat = np.zeros(n_vars)
    _sdp_data._fill_bounds(indexes, u_flat, l_flat, handler.Constraints.U, handler.Constraints.L)

    nm_base, i_base, j_base, v_base = handler._cb_base_obj
    Q = np.zeros((n_vars, n_vars))
    c = np.zeros(n_vars)
    for k in range(len(nm_base)):
        i, j, v = int(i_base[k]), int(j_base[k]), float(v_base[k])
        if i == 0:
            if j > 0:
                c[j - 1] += 2 * v
        elif j == 0:
            c[i - 1] += 2 * v
        else:
            fi, fj = i - 1, j - 1
            Q[fi, fj] += v
            if fi != fj:
                Q[fj, fi] += v
    cons_obj = float(handler._cb_base_obj_constant)

    aq_mats, bq_rhs = [], []
    dq_mats, eq_rhs = [], []
    for d in handler.get_dualized_constraints_data():
        M = _sdp_data._build_cstr_matrix(
            n_vars, d["num_matrix"], d["i"], d["j"], d["value"], constant=0.0, sign=1.0
        )
        if d["free_sign"]:
            aq_mats.append(M)
            bq_rhs.append(d["rhs"])
        else:
            dq_mats.append(M)
            eq_rhs.append(d["rhs"])

    mq, pq = len(aq_mats), len(dq_mats)
    Aq = np.stack(aq_mats, axis=0) if mq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    bq = np.array(bq_rhs, dtype=np.float64) if mq > 0 else np.zeros(0)
    Dq = np.stack(dq_mats, axis=0) if pq > 0 else np.zeros((0, n_vars + 1, n_vars + 1))
    eq = np.array(eq_rhs, dtype=np.float64) if pq > 0 else np.zeros(0)

    return MiqcrData(
        n=n_vars, nb_int=0,
        m=0, p=0, mq=mq, pq=pq,
        Q=Q, c=c,
        u=u_flat, l=l_flat,
        A=np.zeros((0, n_vars)), b=np.zeros(0),
        D=np.zeros((0, n_vars)), e=np.zeros(0),
        Aq=Aq, bq=bq, Dq=Dq, eq=eq,
        cons=cons_obj,
        flat_order=flat_order,
    )


def _set_task_objective_from_qcl(task, q_beta_mat: np.ndarray, c_beta: np.ndarray, n_vars: int) -> None:
    """Pousse (q_beta, c_beta) comme objectif du bloc bar 0. N'inclut PAS la
    constante l_beta (cf. docstring module) : le corner (0,0) reste inutilisé,
    exactement comme _cb_base_obj / resolve_dualized."""
    nm, ii, jj, vv = [], [], [], []
    for k in range(n_vars):
        v = float(c_beta[k]) / 2.0
        if v != 0.0:
            nm.append(0); ii.append(k + 1); jj.append(0); vv.append(v)
    for i in range(n_vars):
        row = q_beta_mat[i]
        for j in range(i + 1):
            v = float(row[j])
            if v != 0.0:
                nm.append(0); ii.append(i + 1); jj.append(j + 1); vv.append(v)
    if not nm:
        # Objectif nul : MOSEK exige quand même un appel pour purger l'ancien objectif.
        task.putbarcblocktriplet(
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64),
        )
        return
    task.putbarcblocktriplet(
        np.array(nm, dtype=np.int32), np.array(ii, dtype=np.int32),
        np.array(jj, dtype=np.int32), np.array(vv, dtype=np.float64),
    )


def make_sdp_callback(handler, data: MiqcrData, log_every: int = 0):
    """Construit le callback conforme à register_sdp_solver, qui résout le
    sous-problème SDP du handler (contraintes RLT désactivées par
    setup_dualization, tout le reste dur) avec l'objectif pénalisé fourni par
    la Conic Bundle MIQCR à chaque itération."""
    task = handler.task
    n_vars = data.n
    call_count = [0]

    def callback(n, mq, pq, q_beta, c_beta, l_beta,
                 x_out, beta_diag_out, alphaq_out, alphabisq_out, sol_sdp_out):
        call_count[0] += 1
        # MIQCR delivers (q_beta, c_beta, l_beta) already negated internally
        # (confirmé empiriquement : l_beta == -cons_true à beta=0, cf.
        # miqcr_wrapper.run_miqcr_sdp_phase qui négative sol_sdp en sortie de la
        # même façon). On repasse en convention "vraie" pour résoudre le bon
        # sous-problème, puis on renégative sol_sdp_out/alphaq_out/alphabisq_out
        # en sortie pour rester dans la convention interne attendue par la lib C.
        q_beta_mat = -np.asarray(q_beta).reshape(n_vars, n_vars)
        c_beta_true = -np.asarray(c_beta)
        l_beta_true = -float(l_beta)
        _set_task_objective_from_qcl(task, q_beta_mat, c_beta_true, n_vars)

        task.optimize()
        status = task.getsolsta(mosek.soltype.itr)
        if status not in (mosek.solsta.optimal, mosek.solsta.prim_and_dual_feas):
            raise RuntimeError(f"[native_conic_bundle] MOSEK non-optimal au round {call_count[0]}: {status}")

        primal_obj = task.getprimalobj(mosek.soltype.itr)
        flat = np.array(task.getbarxj(mosek.soltype.itr, 0))
        x_out[:] = flat

        n1 = n_vars + 1
        X = np.zeros((n1, n1))
        tri = np.triu_indices(n1)
        X[tri] = flat
        X = X + X.T - np.diag(np.diag(X))

        # Sous-gradient standard de la relaxation lagrangienne : g_r = <A_r,X*> - rhs_r
        # (formule deja utilisee et validee dans lagrangian_cb.run_lagrangian_cb).
        for k in range(mq):
            alphaq_out[k] = -(_aq_inner_product(data.Aq[k], X) - float(data.bq[k]))
        for k in range(pq):
            alphabisq_out[k] = -(_aq_inner_product(data.Dq[k], X) - float(data.eq[k]))
        beta_diag_out[:] = 0.0
        h_true = primal_obj + l_beta_true
        sol_sdp_out[0] = -h_true

        if log_every and call_count[0] % log_every == 0:
            print(f"  [native CB #{call_count[0]}] primal_obj={primal_obj:.6f} "
                  f"l_beta_true={l_beta_true:.6f} h_true={h_true:.6f}")

    return callback, call_count


def run_native_conic_bundle(
    handler,
    dualizable_names: list[str],
    so_path: str | None = None,
    log_every: int = 0,
) -> MiqcrResult:
    """
    Point d'entrée : remplace src.dynamic_conic_bundle.DynamicConicBundleSolver
    par la Conic Bundle native de MIQCR sur le même sous-problème (mêmes
    contraintes RLT dualisées, même task MOSEK sous-jacent).

    Prérequis : handler entièrement construit (build_and_extract), backend
    mosek_classic, CHORDAL_DECOMPOSITION=False (cf. setup_dualization).
    """
    data = build_miqcr_data_for_dualization(handler, dualizable_names)
    callback, call_count = make_sdp_callback(handler, data, log_every=log_every)

    register_sdp_solver(callback, so_path=so_path)
    try:
        result = run_miqcr_sdp_phase(data, so_path=so_path)
    finally:
        register_sdp_solver(None, so_path=so_path)

    print(f"[native_conic_bundle] {call_count[0]} appels SDP réels (MOSEK), "
          f"nb_iter_cb rapporté={result.nb_iter_cb}")
    return result
