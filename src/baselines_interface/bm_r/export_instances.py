"""certif-env half of the BM-r pipeline: yaml-driven, mirrors run_sdp_bab.py's loading
(Certification_Problem.load_from_yaml, bounds_file CSV convention, misclassification
skip) -- writes one instance_io npz per (sample, target) into a directory. The
baselines-env half (run_bm_r.py) consumes that directory; this script never imports
cyipopt, run_bm_r.py never imports torch.

Usage (certif env):
    python src/baselines_interface/bm_r/export_instances.py blob_1x2.yaml \
        --out_dir /path/to/instances --indices 0 1 2 --norm Linf
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from certification_problem import Certification_Problem
from fastsdp_tools import get_project_path
from bounds import compute_bounds_data_new
from instance_io import save_instance


def run(yaml_file, out_dir, norm="Linf", include_indices=None):
    problem = Certification_Problem.load_from_yaml(yaml_file)
    solver_config = problem.models[0]
    use_from_file = solver_config.bounds_method == "from_file"
    if use_from_file:
        bounds_path = solver_config.bounds_file
        if not os.path.isabs(bounds_path):
            bounds_path = get_project_path(bounds_path)
        bounds_csv = pd.read_csv(bounds_path)
    else:
        print(f"bounds_method={solver_config.bounds_method!r} (not from_file): "
              f"computing bounds per-sample via bounds.compute_bounds_data_new instead "
              f"of reading a precomputed bounds_file -- matches what this yaml's own "
              f"pipeline would do, just recomputed here rather than cached.")

    os.makedirs(out_dir, exist_ok=True)
    W, b = problem.network.extract_weights()
    dataloader = DataLoader(problem.dataset, batch_size=1, shuffle=False)
    written = []

    for i, (x, ytrue) in enumerate(dataloader):
        if include_indices is not None and i not in include_indices:
            continue
        x = x.view(-1).to(next(problem.network.parameters()).device)
        y_pred = problem.network.label(x)
        if y_pred != ytrue.item():
            print(f"Skipping sample {i} (misclassified).")
            continue

        if use_from_file:
            L = [[float(bounds_csv[bounds_csv["data_index"] == i][f"LB_Layer_{k}_Neuron_{j}"].iloc[0])
                  for j in range(problem.network.n[k])] for k in range(problem.network.K + 1)]
            U = [[float(bounds_csv[bounds_csv["data_index"] == i][f"UB_Layer_{k}_Neuron_{j}"].iloc[0])
                  for j in range(problem.network.n[k])] for k in range(problem.network.K + 1)]
        else:
            L, U = compute_bounds_data_new(problem.network, x, problem.epsilon, problem.network.n,
                                            problem.network.K, method=solver_config.bounds_method, norm=norm)
        x_np = x.detach().cpu().numpy()

        for ytarget in range(problem.network.n[problem.network.K]):
            if ytarget == int(y_pred):
                continue
            path = os.path.join(out_dir, f"idx{i}_target{ytarget}.npz")
            save_instance(path, W, b, L, U, x_np, problem.epsilon, int(y_pred), ytarget, norm=norm)
            written.append(path)

    print(f"Wrote {len(written)} instances to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--norm", default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()
    run(args.yaml_file, args.out_dir, norm=args.norm,
        include_indices=set(args.indices) if args.indices else None)
