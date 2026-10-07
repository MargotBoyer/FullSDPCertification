"""CLI entry point for the SDP-FO baseline. Run with the `baselines` conda env:

    conda run -n baselines python src/baselines_interface/sdp_fo/run_sdp_fo.py \
        --instance path/to/instance.npz --num_steps 2000 --out path/to/result.json
"""
import argparse
import json

from instance_io import load_instance
from adapter import solve_sdp_fo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, help="npz file written by instance_io.save_instance")
    parser.add_argument("--out", required=True, help="json file to write the result dict to")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    W, b, L, U, label, target_label, input_bounds = load_instance(args.instance)
    result = solve_sdp_fo(
        W, b, L, U, label, target_label, input_bounds=input_bounds,
        num_steps=args.num_steps, eval_every=args.eval_every, verbose=args.verbose)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
