"""CLI entry point for the SDP-BaB baseline (Lan, Bruckner & Lomuscio, AAAI'23).

Reuses Certification_Problem.load_from_yaml (unchanged) for network/dataset/epsilon
loading and the bounds_file CSV convention, exactly as certification_problem.py's main
loop does -- only the per-(sample, target) solve step differs (branch_and_bound instead
of a plain TargetedSDP/UntargetedSDP instantiation).

Usage:
    conda activate certif   # same env as the rest of the project -- no jax/MATLAB here
    python src/baselines_interface/sdp_bab/run_sdp_bab.py mnist-9x100 --title SDP-BaB-test \
        --indices 0 1 2 --max_time 3600
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from certification_problem import Certification_Problem
from fastsdp_tools import get_project_path
from baselines_interface.sdp_bab.branch_and_bound import solve_sdp_bab


def run(yaml_file, title_run, max_time=3600.0, max_nodes=None, include_indices=None, verbose=False):
    problem = Certification_Problem.load_from_yaml(yaml_file)
    solver_config = problem.models[0]
    assert solver_config.bounds_method == "from_file", \
        "run_sdp_bab.py currently only supports bounds_method: from_file (same convention as the other baselines)."
    bounds_path = solver_config.bounds_file
    if not os.path.isabs(bounds_path):
        bounds_path = get_project_path(bounds_path)
    bounds_csv = pd.read_csv(bounds_path)

    results_dir = get_project_path(f"results/baselines/sdp_bab/{problem.title}/{title_run}")
    nodes_dir = os.path.join(results_dir, "_nodes")  # per-node MOSEK solves -- not the run's own results.csv
    os.makedirs(nodes_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "results.csv")

    dataloader = DataLoader(problem.dataset, batch_size=1, shuffle=False)
    rows = []

    for i, (x, ytrue) in enumerate(dataloader):
        if include_indices is not None and i not in include_indices:
            continue

        x = x.view(-1).to(next(problem.network.parameters()).device)
        y_pred = problem.network.label(x)
        if y_pred != ytrue.item():
            print(f"Skipping sample {i} (misclassified).")
            continue

        L = [[float(bounds_csv[bounds_csv["data_index"] == i][f"LB_Layer_{k}_Neuron_{j}"].iloc[0])
              for j in range(problem.network.n[k])] for k in range(problem.network.K + 1)]
        U = [[float(bounds_csv[bounds_csv["data_index"] == i][f"UB_Layer_{k}_Neuron_{j}"].iloc[0])
              for j in range(problem.network.n[k])] for k in range(problem.network.K + 1)]

        ytargets = [t for t in range(problem.network.n[problem.network.K]) if t != int(y_pred)]
        for ytarget in ytargets:
            print(f"[sdp_bab] data_index={i} ytarget={ytarget} ...")
            result = solve_sdp_bab(
                network=problem.network, epsilon=problem.epsilon, norm=problem.norm,
                x=x, ytrue=int(y_pred), ytarget=ytarget, L=L, U=U,
                data_index=i, dataset_name=problem.dataset_name, network_name=problem.network_name,
                folder_name=nodes_dir,
                max_time=max_time, max_nodes=max_nodes, verbose=verbose,
            )
            row = {
                "network": problem.network_name, "model": "SDP-BaB", "dataset": problem.dataset_name,
                "data_index": i, "label": int(y_pred), "target": ytarget, "epsilon": problem.epsilon,
                **result,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(f"[sdp_bab] -> {result}")
            if result["optimal_value"] is not None and result["optimal_value"] <= 0:
                print(f"[sdp_bab] data_index={i}: NOT certified robust against target {ytarget} "
                      f"(optimal_value={result['optimal_value']:.4g}) -- stopping target loop for this sample.")
                break

    print(f"Done. Results written to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", help="e.g. blob4x10.yaml or mnist-9x100.yaml (config/ basename)")
    parser.add_argument("--title", default=None, help="run title (default: timestamp)")
    parser.add_argument("--max_time", type=float, default=3600.0, help="B&B wall-clock budget per (sample, target), seconds")
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="dataset indices to process (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    title = args.title or time.strftime("%Y_%m_%d_%Hh%Mm%Ss_SDP-BaB")
    run(args.yaml_file, title, max_time=args.max_time, max_nodes=args.max_nodes,
        include_indices=set(args.indices) if args.indices else None, verbose=args.verbose)
