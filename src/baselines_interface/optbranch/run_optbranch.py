"""CLI entry point for the OptBranch baseline (Anderson, Ma, Li & Sojoudi,
"Towards Optimal Branching for Neural Network Robustness Certification", JMLR 2025).

Reuses Certification_Problem.load_from_yaml (unchanged) for network/dataset/epsilon
loading and the bounds_file CSV convention, exactly as certification_problem.py's main
loop and the sdp_bab baseline do -- runs entirely in the certif env (no cross-env
boundary: unlike sdp_fo/bm_r, OptBranch needs no third-party code, just this project's
own solve/sdp_solve called in a loop with shrinking L/U, see node_solver.py).

Usage:
    conda activate certif
    python src/baselines_interface/optbranch/run_optbranch.py mnist-9x100 --title OptBranch-test \
        --indices 0 1 2 --max_depth 20 --max_time 3600
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
from baselines_interface.optbranch.branch_and_bound import solve_optbranch


def run(yaml_file, title_run, max_depth=20, max_nodes=None, max_time=3600.0,
        include_indices=None, verbose=False):
    problem = Certification_Problem.load_from_yaml(yaml_file)
    solver_config = problem.models[0]
    assert solver_config.bounds_method == "from_file", \
        "run_optbranch.py currently only supports bounds_method: from_file (same convention as the other baselines)."
    bounds_path = solver_config.bounds_file
    if not os.path.isabs(bounds_path):
        bounds_path = get_project_path(bounds_path)
    bounds_csv = pd.read_csv(bounds_path)

    results_dir = get_project_path(f"results/baselines/optbranch/{problem.title}/{title_run}")
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
            print(f"[optbranch] data_index={i} ytarget={ytarget} ...")
            result = solve_optbranch(
                network=problem.network, epsilon=problem.epsilon, norm=problem.norm,
                x=x, ytrue=int(y_pred), ytarget=ytarget, L=L, U=U,
                data_index=i, dataset_name=problem.dataset_name, network_name=problem.network_name,
                folder_name=nodes_dir,
                max_depth=max_depth, max_nodes=max_nodes, max_time=max_time, verbose=verbose,
            )
            row = {
                "network": problem.network_name, "model": "OptBranch", "dataset": problem.dataset_name,
                "data_index": i, "label": int(y_pred), "target": ytarget, "epsilon": problem.epsilon,
                **result,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
            print(f"[optbranch] -> {result}")
            if result["optimal_value"] is not None and result["optimal_value"] <= 0:
                print(f"[optbranch] data_index={i}: NOT certified robust against target {ytarget} "
                      f"(optimal_value={result['optimal_value']:.4g}) -- stopping target loop for this sample.")
                break

    print(f"Done. Results written to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", help="e.g. blob4x10.yaml or mnist-9x100.yaml (config/ basename)")
    parser.add_argument("--title", default=None, help="run title (default: timestamp)")
    parser.add_argument("--max_depth", type=int, default=20, help="DFS depth budget per (sample, target)")
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--max_time", type=float, default=3600.0, help="wall-clock budget per (sample, target), seconds")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="dataset indices to process (default: all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    title = args.title or time.strftime("%Y_%m_%d_%Hh%Mm%Ss_OptBranch")
    run(args.yaml_file, title, max_depth=args.max_depth, max_nodes=args.max_nodes, max_time=args.max_time,
        include_indices=set(args.indices) if args.indices else None, verbose=args.verbose)
