"""baselines-env half of the BM-r pipeline: consumes the npz instances
export_instances.py (certif env) wrote, runs solve_bm_r on each, writes a
results.csv-compatible file under results/baselines/bm_r/.

IMPORTANT: optimal_value is the raw primal value, not a certified bound -- see
bm_r/README.md "Known issue: the certificate does not reliably close" and this
module's own docstring in riemannian_staircase.py before using it as anything other
than an (empirically SDP-IP-matching, but unproven) point of comparison. Check the
`status` column: "certified" rows carry the paper's actual global-optimality
guarantee; "uncertified_max_rank" rows do not.

Usage (baselines env):
    conda activate baselines
    python src/baselines_interface/bm_r/run_bm_r.py /path/to/instances --title bm-r-test
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

from instance_io import load_instance
from problem import build_problem
from riemannian_staircase import solve_bm_r


def run(instances_dir, title_run, max_search_rank_extra=6, max_iter=1000, verbose=False):
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "results", "baselines", "bm_r", title_run,
    )
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "results.csv")

    rows = []
    for path in sorted(glob.glob(os.path.join(instances_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[bm_r] {name} ...")
        W, b, L, U, x, epsilon, ytrue, ytarget, norm = load_instance(path)
        prob = build_problem(W, b, L, U, x, epsilon, ytrue, ytarget, norm=norm)

        t0 = time.time()
        result = solve_bm_r(prob, max_search_rank_extra=max_search_rank_extra,
                             max_iter=max_iter, verbose=verbose)
        result["time"] = time.time() - t0
        result["instance"] = name
        result["label"] = ytrue
        result["target"] = ytarget
        result["epsilon"] = epsilon
        result["norm"] = norm
        rows.append(result)
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        print(f"[bm_r] -> {result}")

    print(f"Done. Results written to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instances_dir", help="directory of .npz instances from export_instances.py")
    parser.add_argument("--title", default=None)
    parser.add_argument("--max_search_rank_extra", type=int, default=6)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    title = args.title or time.strftime("%Y_%m_%d_%Hh%Mm%Ss_BM-r")
    run(args.instances_dir, title, max_search_rank_extra=args.max_search_rank_extra,
        max_iter=args.max_iter, verbose=args.verbose)
