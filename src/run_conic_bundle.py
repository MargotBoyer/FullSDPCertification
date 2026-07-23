#!/usr/bin/env python3
"""
Lance la phase Conic Bundle MIQCR sur une instance de certification.

Usage :
    python scripts/run_conic_bundle.py <config>  [--samples N] [--start I]

Exemples :
    python scripts/run_conic_bundle.py mnist-6x100
    python scripts/run_conic_bundle.py mnist-6x100 --samples 3
    python scripts/run_conic_bundle.py mnist-6x100 --start 5 --samples 2

<config> est le nom (sans extension) d'un fichier YAML dans config/.
"""
import argparse
import datetime
import os
import sys
import time

# Ajoute src/ au path pour importer les modules du projet
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from fastsdp_tools import get_project_path, FullCertificationConfig
from data import load_dataset
from networks import ReLUNN
import solve

from miqcr_interface import extract_miqcr_data, run_miqcr_sdp_phase, register_sdp_solver
from miqcr_interface.sdp_data import MiqcrData


# ---------------------------------------------------------------------------
# Sauvegarde des matrices MIQCR
# ---------------------------------------------------------------------------

def save_miqcr_data(data: MiqcrData, folder: str, sample_index: int) -> None:
    """Sauvegarde toutes les matrices MiqcrData dans un sous-dossier par échantillon."""
    sample_dir = os.path.join(folder, f"sample_{sample_index}")
    os.makedirs(sample_dir, exist_ok=True)

    np.savez(
        os.path.join(sample_dir, "miqcr_matrices.npz"),
        A=data.A,
        b=data.b,
        D=data.D,
        e=data.e,
        Aq=data.Aq,
        bq=data.bq,
        Dq=data.Dq,
        eq=data.eq,
        Q=data.Q,
        c=data.c,
        u=data.u,
        l=data.l,
    )

    scalars = {
        "n": data.n,
        "nb_int": data.nb_int,
        "m": data.m,
        "p": data.p,
        "mq": data.mq,
        "pq": data.pq,
        "cons": data.cons,
    }
    import json
    with open(os.path.join(sample_dir, "miqcr_scalars.json"), "w") as f:
        json.dump(scalars, f, indent=2)

    flat_order_serializable = [[int(layer), int(j)] for (layer, j) in data.flat_order]
    with open(os.path.join(sample_dir, "flat_order.json"), "w") as f:
        json.dump(flat_order_serializable, f)

    print(f"    Matrices MIQCR sauvegardées → {sample_dir}")


def print_miqcr_data(data: MiqcrData) -> None:
    """Affiche un résumé lisible des matrices MIQCR dans le terminal."""
    sep = "-" * 60
    print(sep)
    print("MIQCR DATA")
    print(sep)
    print(f"  n={data.n}  nb_int={data.nb_int}  cons={data.cons:.6g}")
    print(f"  Contraintes linéaires   : {data.m} égalités, {data.p} inégalités")
    print(f"  Contraintes quadratiques: {data.mq} égalités, {data.pq} inégalités")

    def _array_summary(name, arr):
        if arr is None or arr.size == 0:
            print(f"  {name}: (vide)")
            return
        print(f"  {name}: shape={arr.shape}  min={arr.min():.4g}  max={arr.max():.4g}"
              f"  nnz={np.count_nonzero(arr)}")

    _array_summary("Q  (hessienne objectif)", data.Q)
    _array_summary("c  (linéaire objectif) ", data.c)
    _array_summary("u  (bornes sup)        ", data.u)
    _array_summary("l  (bornes inf)        ", data.l)
    _array_summary("A  (égalités lin)      ", data.A)
    _array_summary("b  (rhs égalités lin)  ", data.b)
    _array_summary("D  (inégalités lin)    ", data.D)
    _array_summary("e  (rhs inégalités lin)", data.e)
    _array_summary("Aq (égalités quad)     ", data.Aq)
    _array_summary("bq (rhs égalités quad) ", data.bq)
    _array_summary("Dq (inégalités quad)   ", data.Dq)
    _array_summary("eq (rhs inégal. quad)  ", data.eq)
    print(sep)


# ---------------------------------------------------------------------------
# Callback SDP Python (placeholder — à remplacer par un vrai solveur)
# ---------------------------------------------------------------------------

def trivial_sdp_callback(n, mq, pq, q_beta, c_beta, l_beta,
                         x_out, beta_diag, alphaq, alphabisq, sol_out):
    """Solution triviale X = I : utilisée pour valider le pipeline."""
    x_out[:] = 0.0
    x_out[0] = 1.0
    idx = 0
    for i in range(n + 1):
        if i > 0:
            x_out[idx] = 1.0
        idx += n + 1 - i
    beta_diag[:] = 0.0
    sol_out[0] = float(l_beta)


# ---------------------------------------------------------------------------
# Construction du problème et extraction des données MIQCR
# ---------------------------------------------------------------------------

def build_and_extract(model_instance, cuts, max_quad_constraints: int = 1e5):
    """
    Reproduit le début de run_optimization() jusqu'à add_constraints(),
    puis extrait les matrices MIQCR depuis le handler (sans appeler Mosek).

    max_quad_constraints : budget de contraintes quadratiques denses passées à MIQCR.
        0 = aucune (utile pour valider le pipeline avec le callback trivial).
        Augmenter avec précaution : chaque matrice pèse (n+1)² × 8 octets.

    Retourne un MiqcrData prêt pour run_miqcr_sdp_phase().
    """
    handler = model_instance.handler
    assert not handler.CHORDAL_DECOMPOSITION, (
        "MIQCR Conic Bundle requiert CHORDAL_DECOMPOSITION=False : "
        "l'extraction suppose une seule matrice SDP globale."
    )
    assert not ("RLT" in model_instance.cuts), (
        "MIQCR Conic Bundle traite lui-même les contraintes RLT."
    )
    handler.initiate_env(verbose=False)
    model_instance.add_objective()
    handler.initialize_variables()
    model_instance.adapt_number_RLT()
    model_instance.add_constraints(cuts)     # appelle end_constraints() en fin
    return extract_miqcr_data(handler, max_quad_constraints=max_quad_constraints)


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",
                        help="Nom du fichier YAML dans config/ (sans extension)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Nombre maximum d'échantillons à traiter")
    parser.add_argument("--start", type=int, default=0,
                        help="Index du premier échantillon à traiter (défaut : 0)")
    parser.add_argument("--max-quad", type=int, default=1e5,
                        help="Nombre max de contraintes quadratiques denses passées à MIQCR "
                             "(0 = aucune, pour valider le pipeline ; augmenter avec précaution)")
    args = parser.parse_args()

    yaml_file = f"{args.config}.yaml"
    print("STUDY yaml_file : ", yaml_file)
    config_path = get_project_path(f"config/{yaml_file}")
    print("STUDY config_path : ", config_path)
    if not os.path.exists(config_path):
        sys.exit(f"Fichier de configuration introuvable : {config_path}")

    print(f"Chargement de la configuration : {config_path}")
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    validated = FullCertificationConfig(**raw_config)
    epsilon   = raw_config["input_ball"]["epsilon"]
    norm      = raw_config["input_ball"]["norm"]

    network_path = get_project_path(raw_config["network"]["path"])
    print("STUDY network_path : ", network_path)
    network = ReLUNN.from_pth(
        network_path, bb_beta_crown=False
    )

    dataset = load_dataset(config_path)
    print(f"Réseau : {validated.network.name} | ε={epsilon} | {len(dataset)} échantillons")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    # Enregistre le callback Python une seule fois (réutilisé à chaque itération)
    register_sdp_solver(trivial_sdp_callback)

    title = f"{validated.network.name}-{epsilon}"
    launch_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M_%Ss")
    run_name = f"{launch_date}_MIQCR_conic_bundle"
    results_dir = get_project_path(f"results/benchmark/{title}/{run_name}")
    os.makedirs(results_dir, exist_ok=True)

    # Copie du fichier de config dans le dossier de résultats
    import shutil
    shutil.copyfile(config_path, os.path.join(results_dir, yaml_file))
    print(f"Résultats sauvegardés dans : {results_dir}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []

    for solver_config in validated.models:
        model_class = getattr(solve, solver_config.certification_model_type)
        print(f"\n=== Modèle : {solver_config.certification_model_type} ===")

        bounds_csv = None
        if solver_config.bounds_method == "from_file":
            bounds_path = solver_config.bounds_file
            if not os.path.isabs(bounds_path):
                bounds_path = get_project_path(bounds_path)
            bounds_csv = pd.read_csv(bounds_path)

        n_processed = 0
        for i, (x, ytrue) in enumerate(dataloader):
            if i < args.start:
                continue
            if args.samples is not None and n_processed >= args.samples:
                break

            x = x.view(-1).to(device)
            network = network.to(device)
            print("x device : ", x.device)
            print("network device : ", next(network.parameters()).device)
            y_pred = network.label(x)
            if y_pred != ytrue.item():
                print(f"  [#{i}] label={ytrue.item()} → mal classifié, ignoré.")
                continue

            label_val = ytrue.item() if hasattr(ytrue, "item") else int(ytrue)
            print(f"\n  [#{i}] label={label_val} cible={y_pred}")

            if bounds_csv is not None:
                K = network.K
                L = [[float(bounds_csv[bounds_csv["data_index"] == i]
                            [f"LB_Layer_{k}_Neuron_{j}"].iloc[0])
                      for j in range(network.n[k])]
                     for k in range(K + 1)]
                U = [[float(bounds_csv[bounds_csv["data_index"] == i]
                            [f"UB_Layer_{k}_Neuron_{j}"].iloc[0])
                      for j in range(network.n[k])]
                     for k in range(K + 1)]
                solver_config.L = L
                solver_config.U = U

            dict_infos = dict(solver_config)
            dict_infos.pop("certification_model_type")

            model_instance = model_class(
                network=network,
                epsilon=epsilon,
                norm=norm,
                x=x,
                ytrue=y_pred,
                data_index=i,
                dataset_name=validated.data.name,
                network_name=validated.network.name,
                folder_name=f"results/benchmark/{title}/conic_bundle",
                **dict_infos,
            )

            # Choisit le premier jeu de cuts (le seul si all_combinations_cuts=False)
            cuts_to_test = model_instance.cuts_to_test
            cuts = cuts_to_test[0] if cuts_to_test else []

            # RLT_prop doit être initialisé avant adapt_number_RLT (comme dans solve())
            model_instance.RLT_prop = model_instance.RLT_props[0]

            t0 = time.time()
            data = build_and_extract(model_instance, cuts,
                                     max_quad_constraints=args.max_quad)
            t_extract = time.time() - t0

            print(f"    Extraction MIQCR : n={data.n} m={data.m} p={data.p} "
                  f"mq={data.mq} pq={data.pq}  ({t_extract:.2f}s)")

            save_miqcr_data(data, results_dir, i)
            print_miqcr_data(data)

            t1 = time.time()
            result = run_miqcr_sdp_phase(data)
            t_cb = time.time() - t1

            print(f"    Conic Bundle terminée en {t_cb:.2f}s | sol_sdp={result.sol_sdp:.6f}")
            print(f"    beta max={np.abs(result.beta).max():.4f}")

            results.append({
                "data_index": i,
                "label": ytrue.item(),
                "n_vars": data.n,
                "sol_sdp": result.sol_sdp,
                "t_extract_s": round(t_extract, 3),
                "t_cb_s": round(t_cb, 3),
            })
            n_processed += 1

    print("\n" + "=" * 60)
    print("Résumé :")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    results_csv = os.path.join(results_dir, "results.csv")
    df.to_csv(results_csv, index=False)
    print(f"Résumé sauvegardé → {results_csv}")


if __name__ == "__main__":
    main()
