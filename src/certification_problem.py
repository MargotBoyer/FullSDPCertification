import logging
import re
import numpy as np
import yaml
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from solve.generic_solver import Solver
from solve import LayersValues
import solve
from fastsdp_tools import FullCertificationConfig
from pydantic import BaseModel
import pandas as pd
import datetime
import shutil
import argparse
import multiprocessing as mp
from adversarial_attacks import PGDAttack
 
from fastsdp_tools import create_folder_benchmark, get_project_path
from fastsdp_tools.resume_utils import find_run_yaml, find_processed_indices, load_existing_results, log_run_history
from solve.sdp_solve import concat_dataframes_with_missing_columns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from networks import ReLUNN
from data import load_dataset

from fastsdp_tools import get_project_path

logger = logging.getLogger(__name__)

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_strategy_from_dict(strategy_dict: dict):
    """
    Convertit un dictionnaire de stratégie du format YAML au format attendu par le solveur
    (le dictionnaire de clés doit utiliser des tuples).
    """
    if not strategy_dict:
        return {}
        
    bound_strategy = {
        tuple(product_data['key']): product_data['type']
        for product_data in strategy_dict.values()
    }
    return bound_strategy


class Certification_Problem:
    def __init__(
        self,
        network: ReLUNN,
        epsilon: float,
        norm: str,
        dataset: TensorDataset,
        **kwargs,
    ):
        """Initialize the certification problem.
        Args:
            network (models.ReLUNN): The neural network model.
            epsilon (float): The perturbation bound.
            dataset (TensorDataset): The dataset for certification.
        """
        print("Initializing Certification Problem ...")
        self.network = network.to(device_ if kwargs.get("use_cuda", True) else "cpu")
        self.epsilon = epsilon
        self.norm = norm
        self.dataset = dataset
        self.models = kwargs.get("models", [])
        print("Models in certification problem:", self.models)

        self.network_name = kwargs.get("network_name", network.name)
        self.dataset_name = kwargs.get("dataset_name")
        self.yaml_file = kwargs.get("yaml_file", None)
        self._config_path = kwargs.get("config_path", None)
        self.divide_run = kwargs.get("divide_run", 1)
        print("Data name in certification problem:", self.dataset_name)

        print("dataset in certification problem:", self.dataset)

        self.title = f"{self.network_name}-{self.epsilon}"
        os.makedirs(get_project_path(f"results/benchmark/{self.title}"), exist_ok=True)

        self.benchmark = None

        print("Certification Problem initialized !")

    @classmethod
    def load_from_yaml(cls, yaml_file, config_path: str = None):
        """
        Load the certification problem from a YAML file.
        Args:
            yaml_file (str): Basename of the YAML file (e.g. "mnist-9x100.yaml").
            config_path (str): Full path to the yaml file. When provided (worker mode),
                               this overrides the default config/ lookup so workers always
                               use the config snapshot saved by the orchestrator.
        Returns:
            Certification_Problem: An instance of the Certification_Problem class.
        """
        actual_path = config_path if config_path else get_project_path(f"config/{yaml_file}")
        print(f"Loading certification problem from {actual_path} ...")

        print("Loading dataset ...")
        dataset = load_dataset(actual_path)
        if dataset is not None:
            print("Dataset loaded successfully.")
        else:
            print("Failed to load dataset.")
            return None
        print("Loading epsilon ...")
        with open(actual_path, "r") as file:
            config = yaml.safe_load(file)
            print("CONFIG  inf CERTIFICATION PROBLEM:     ", config)
            epsilon = config["input_ball"]["epsilon"]
            norm = config["input_ball"]["norm"]
            print(f"Epsilon: {epsilon}, Norm: {norm}")

        path_network = config["network"]["path"]
        logger.debug("path network : ", path_network)
        network = ReLUNN.from_pth(get_project_path(path_network), bb_beta_crown=False)

        if network is not None:
            print("Network loaded successfully.")
        else:
            print("Failed to load network.")
            return None
        validated_config = FullCertificationConfig(**config)
        print("Data name from config:", validated_config.data.name)
        return cls(
            network,
            epsilon,
            norm,
            dataset,
            models=validated_config.models,
            network_name=validated_config.network.name,
            dataset_name=validated_config.data.name,
            yaml_file=yaml_file,
            config_path=actual_path,
            divide_run=validated_config.divide_run,
        )

    def __str__(self):
        """
        String representation of the certification problem.
        """
        return f"Certification Problem with epsilon: {self.epsilon}, dataset size: {len(self.dataset)}"

    def run(self, solver_config: BaseModel, title_run: str = "", start: int = None, end: int = None, skip_indices: set = None, skip_pairs: set = None, include_indices: set = None) -> None:
        """
        Run the certification problem.

        If start/end are provided, only samples with raw dataset index in [start, end) are processed.
        If include_indices is provided, only samples whose index is in the set are processed.
        """
        model_class = getattr(solve, solver_config.certification_model_type)
        print(
            f"Running certification with solver: {solver_config.certification_model_type}"
        )

        print("SOLVER CONFIG:", solver_config)
      
      
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        stable_actives_study = pd.DataFrame(
            columns=[
                "label",
                "data_index",
                "Number_actives_stable",
                "Number_inactives_stable",
                "Number_targets",
            ]
        )
        width_model_study = pd.DataFrame()
        coefficient_values = {k : [] for k in range(1, self.network.K + 1)}

        if solver_config.bounds_method == "from_file":
                bounds_path = solver_config.bounds_file
                if not os.path.isabs(bounds_path):
                    bounds_path = get_project_path(bounds_path)
                bounds_csv = pd.read_csv(bounds_path)

        for i, (x, ytrue) in enumerate(dataloader):
            if start is not None and i < start:
                continue
            if end is not None and i >= end:
                break
            if include_indices is not None and i not in include_indices:
                continue
            if skip_indices and i in skip_indices:
                print(f"Skipping sample {i} (already processed).")
                continue
            # if (i) % 10 != 0:
            #     print(
            #         f"Skipping sample {i + 1} with label {ytrue.item()} as it is not a multiple of 10."
            #     )
            #     continue
            # if ytrue.item() != 1:
            #     print(
            #         f"Skipping sample {i + 1} with label {ytrue.item()} as it is not a multiple of 10."
            #     )
            #     continue
            # assert ytrue == y, "ytrue should match the label y"

            # if i > 100 :
            #     #     f"Stopping after 25 samples. Current sample index: {i}. You can change this limit in the code."
            #     # )
            #     #print("Skipping data sample ", i + 1, "for testing purposes.")
            #     continue

            print("i : ", i)

            x = x.view(-1)  # Ensure x is a 2D tensor
            print("x  shape after view:", x.shape)
            print(
                f"STUDY : Running certification for sample {i + 1} of label {ytrue.item()}"
            )
            # print("Network device : ", self.network.device)
            print("x device : ", x.device)
            x = x.to(next(self.network.parameters()).device)
            #torch.save(x, f"tensor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            print("x device : ", x.device)
            y_pred =self.network.label(x)
            logger.debug("ytrue:", ytrue)

            if y_pred != ytrue.item() :
                print(
                    f"Skipping sample {i + 1} with label {ytrue.item()} as it is misclassified by the network."
                )
                continue

            if solver_config.bounds_method == "from_file":
                L = [[float(bounds_csv[bounds_csv["data_index"] == i][f"LB_Layer_{k}_Neuron_{j}"].iloc[0])
                 for j in range(self.network.n[k])]
                for k in range(self.network.K + 1)]
                U = [[float(bounds_csv[bounds_csv["data_index"] == i][f"UB_Layer_{k}_Neuron_{j}"].iloc[0])
                        for j in range(self.network.n[k])]
                        for k in range(self.network.K + 1)]
                solver_config.L = L
                solver_config.U = U
                print("Bounds loaded from file : ", solver_config.bounds_file)

            dict_infos = dict(solver_config)
            dict_infos.pop("certification_model_type")
            logger.debug("dict_infos:", dict_infos)
   
            # --- Traitement de la stratégie de bornes personnalisée ---
            if solver_config.relu_relaxation_type == 'custom':
                if not solver_config.bound_strategy:
                    raise ValueError("relu_relaxation_type='custom' mais aucun dictionnaire 'bound_strategy' n'a été fourni.")
                
                # Le dictionnaire est déjà dans solver_config, il faut juste le formater
                dict_infos['bound_strategy'] = format_strategy_from_dict(solver_config.bound_strategy)
   
            # model_bounds = solve.LPBoundLayer(
            #     network=self.network,
            #     epsilon=self.epsilon,
            #     norm=self.norm,
            #     x=x,
            #     ytrue=y_pred.item(),
            #     data_index=i,
            #     dataset_name=self.dataset_name,
            #     network_name=self.network_name,
            #     folder_name=f"results/benchmark/{self.title}/{title_run}",
            #     use_active_neurons = True,
            #     use_inactive_neurons = True,
            #     bounds_method = "IBP"
            # )
            
            # model_bounds.solve()

            # L = model_bounds.L
            # U = model_bounds.U
            # print("Recuperation de L :", L)
            # print("Recuperation de U : ", U)
            try:
                model_instance = model_class(
                    network=self.network,
                    epsilon=self.epsilon,
                    norm=self.norm,
                    x=x,
                    ytrue=y_pred,
                    # L = L,
                    # U = U,
                    data_index=i,
                    dataset_name=self.dataset_name,
                    network_name=self.network_name,
                    folder_name=f"results/benchmark/{self.title}/{title_run}",
                    **dict_infos,
                )
                logger.debug("Model instance created")
                nb_actives = len(model_instance.stable_actives_neurons)
                nb_inactives = len(model_instance.stable_inactives_neurons)
                nb_targets = len(model_instance.ytargets)
                logger.debug("number of targets : ", nb_targets)
            except Exception as e:
                import traceback
                logger.debug("ERROR : Error while creating model instance:", e)
                traceback.print_exc()
                nb_actives = -1
                nb_inactives = -1
                nb_targets = 0
                continue

            output_bounds_U = model_instance.U[self.network.K]
            output_bounds_L = model_instance.L[self.network.K]

            logger.debug("output_bounds_U:", output_bounds_U)
            logger.debug("output_bounds_L:", output_bounds_L)

            model_instance.solve(verbose=True, only_bounds=False, skip_pairs=skip_pairs)
            logger.debug("Model instance solved")
            logger.debug("model_instance.benchmark_dataframe :", model_instance.benchmark_dataframe)
            
            # for k in range(1, self.network.K + 1):
            #     if k not in coefficient_values:
            #         coefficient_values[k] = []
            #     coefficient_values[k].extend(model_instance.handler.Constraints.coefficient_values[k])
            #print("STUDY COEFF after run: coefficient values for each layer: {}".format(coefficient_values))
            if model_instance.benchmark_dataframe is not None:
                self.benchmark = concat_dataframes_with_missing_columns(
                    self.benchmark, model_instance.benchmark_dataframe
                )
                self.benchmark.to_csv(
                    get_project_path(
                        f"results/benchmark/{self.title}/{title_run}/results.csv"
                    ),
                    index=False,
                )
            dict_stability = {
                            "label": [ytrue],
                            "data_index": [i],
                            "Number_actives_stable": [nb_actives],
                            "Number_inactives_stable": [nb_inactives],
                            "Number_targets": [nb_targets],
                        }
            for k in range(1, self.network.K + 1):
                nb_stable_actives_layer_k = len([(n, j) for (n, j) in model_instance.stable_actives_neurons if n == k])
                nb_stable_inactives_layer_k = len([(n, j) for (n, j) in model_instance.stable_inactives_neurons if n == k])
                dict_stability[f"Stable_Actives_Layer_{k}"] = nb_stable_actives_layer_k
                dict_stability[f"Stable_Inactives_Layer_{k}"] = nb_stable_inactives_layer_k
                print(
                    f"STUDY : Layer {k} - Stable actives neurons: {nb_stable_actives_layer_k} - Stable inactives neurons: {nb_stable_inactives_layer_k}"
                )
            for k in range(self.network.K + 1):
                for j in range(self.network.n[k]):
                    dict_stability[f"LB_Layer_{k}_Neuron_{j}"] = model_instance.L[k][j]
                    dict_stability[f"UB_Layer_{k}_Neuron_{j}"] = model_instance.U[k][j]

            stable_actives_study = pd.concat(
                [
                    stable_actives_study,
                    pd.DataFrame(
                       dict_stability
                    ),
                ],
                ignore_index=True,
            )

            stable_actives_study.to_csv(
                get_project_path(
                    f"results/benchmark/{self.title}/{title_run}/stable_actives_study.csv"
                )
            )

        for k in range(1, self.network.K + 1):
            if coefficient_values[k] == []:
                print(f"No coefficient values collected for layer {k}, skipping histogram.")
                continue
            # Histogram with all values
            plt.hist(coefficient_values[k], bins=50)
            plt.title("Histogram of coefficients in ReLU Relaxed Layer {}".format(k))
            plt.xlabel("Coefficient value")
            plt.ylabel("Frequency")
            plt.savefig(
                get_project_path(
                    f"results/benchmark/{self.title}/{title_run}/histogram_coefficients_Layer_{k}.png"
                )
            )
            plt.close()

            # Histogram with 95% of values (centered around median)
            median = np.median(coefficient_values[k])
            p95_distance = np.percentile(np.abs(np.array(coefficient_values[k]) - median), 95)
            filtered_values_95 = [v for v in coefficient_values[k] if abs(v - median) <= p95_distance]
            plt.hist(filtered_values_95, bins=50)
            plt.title("Histogram of coefficients in ReLU Relaxed 95% Layer {}".format(k))
            plt.xlabel("Coefficient value")
            plt.ylabel("Frequency")
            plt.savefig(
                get_project_path(
                    f"results/benchmark/{self.title}/{title_run}/histogram_coefficients_95_Layer_{k}.png"
                )
            )
            plt.close()

            # Histogram with 75% of values (centered around median)
            p75_distance = np.percentile(np.abs(np.array(coefficient_values[k]) - median), 75)
            filtered_values_75 = [v for v in coefficient_values[k] if abs(v - median) <= p75_distance]
            plt.hist(filtered_values_75, bins=50)
            plt.title("Histogram of coefficients in ReLU Relaxed 75% Layer {}".format(k))
            plt.xlabel("Coefficient value")
            plt.ylabel("Frequency")
            plt.savefig(
                get_project_path(
                    f"results/benchmark/{self.title}/{title_run}/histogram_coefficients_75_Layer_{k}.png"
                )
            )
            plt.close()

    def solve(self, title_run: str = "", start: int = None, end: int = None, skip_indices: set = None, skip_pairs: set = None, resume: bool = False, include_indices: set = None) -> None:
        print("Starting certification problem solving ...", flush=True)
        print("self.models:", self.models, flush=True)

        if not resume:
            self.benchmark = pd.DataFrame()

        if not os.path.exists(
            get_project_path(f"results/benchmark/{self.title}/{title_run}")
        ):
            os.makedirs(
                get_project_path(f"results/benchmark/{self.title}/{title_run}"),
                exist_ok=True,
            )

        if self.yaml_file is not None:
            src = self._config_path or get_project_path(f"config/{self.yaml_file}")
            dst = get_project_path(
                f"results/benchmark/{self.title}/{title_run}/{self.yaml_file}"
            )
            if not (os.path.exists(dst) and os.path.samefile(src, dst)):
                shutil.copyfile(src, dst)

        for i, model_config in enumerate(self.models):

            print("Solving with model:", model_config.certification_model_type, flush=True)
            print("model dict :", model_config, flush=True)
            self.run(model_config, title_run, start=start, end=end, skip_indices=skip_indices, skip_pairs=skip_pairs, include_indices=include_indices)



class _Tee:
    """Write to multiple streams simultaneously (e.g. stdout + log file)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()


def _run_orchestrator(certif_problem, network, title_run_full):
    """Fork N subprocess (one per chunk) and wait for them to complete."""
    import subprocess

    parent_dir = get_project_path(
        f"results/benchmark/{certif_problem.title}/{title_run_full}"
    )
    os.makedirs(parent_dir, exist_ok=True)

    if certif_problem.yaml_file is not None:
        shutil.copyfile(
            get_project_path(f"config/{certif_problem.yaml_file}"),
            os.path.join(parent_dir, certif_problem.yaml_file),
        )

    num_samples = len(certif_problem.dataset)
    n_chunks = certif_problem.divide_run
    chunk_size = -(-num_samples // n_chunks)  # ceil division

    print(f"Orchestrator: {n_chunks} chunks of ~{chunk_size} samples from {num_samples} total")

    saved_config = os.path.join(parent_dir, certif_problem.yaml_file) if certif_problem.yaml_file else None

    processes = []
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, num_samples)
        if chunk_start >= num_samples:
            break
        cmd = [
            sys.executable, os.path.abspath(__file__),
            network, title_run_full,
            "--start", str(chunk_start),
            "--end", str(chunk_end),
        ]
        if saved_config and os.path.exists(saved_config):
            cmd += ["--config", saved_config]
        print(f"  → chunk {chunk_idx}: samples [{chunk_start}, {chunk_end})")
        processes.append(subprocess.Popen(cmd))

    print(f"Waiting for {len(processes)} subprocesses…")
    exit_codes = [p.wait() for p in processes]
    print(f"All chunks done. Exit codes: {exit_codes}")


def main(network: str, title_run: str, start: int = None, end: int = None, config_path: str = None, include_indices: set = None):
    yaml_file = f"{network}.yaml"
    certif_problem = Certification_Problem.load_from_yaml(yaml_file, config_path=config_path)

    is_worker = start is not None and end is not None

    if is_worker:
        # The orchestrator (or SLURM launcher) already set the date prefix; use as-is.
        title_run_full = title_run
        title_run_for_solve = f"{title_run_full}/part_{start}_{end}"
    else:
        launch_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M_%Ss")
        title_run_full = f"{launch_date}_{title_run}"
        title_run_for_solve = title_run_full

        if certif_problem.divide_run > 1:
            parent_dir = Path(get_project_path(f"results/benchmark/{certif_problem.title}/{title_run_full}"))
            parent_dir.mkdir(parents=True, exist_ok=True)
            start_time = datetime.datetime.now()
            try:
                _run_orchestrator(certif_problem, network, title_run_full)
            finally:
                _done, _pairs = find_processed_indices(parent_dir)
                log_run_history(parent_dir, "initial", start_time, _done | {idx for idx, _ in _pairs})
            return

    results_dir = get_project_path(
        f"results/benchmark/{certif_problem.title}/{title_run_for_solve}"
    )
    os.makedirs(results_dir, exist_ok=True)

    if not is_worker:
        start_time = datetime.datetime.now()

    if is_worker and certif_problem.yaml_file is not None:
        parent_yaml = get_project_path(
            f"results/benchmark/{certif_problem.title}/{title_run_full}/{certif_problem.yaml_file}"
        )
        if not os.path.exists(parent_yaml):
            shutil.copyfile(
                get_project_path(f"config/{certif_problem.yaml_file}"),
                parent_yaml,
            )

    log_path = os.path.join(results_dir, "run.log")

    try:
        with open(log_path, "w") as log_file:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(original_stdout, log_file)
            sys.stderr = _Tee(original_stderr, log_file)
            try:
                certif_problem.solve(title_run_for_solve, start=start, end=end, include_indices=include_indices)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        print(f"Log saved → {log_path}")
    finally:
        if not is_worker:
            _done, _pairs = find_processed_indices(Path(results_dir))
            log_run_history(Path(results_dir), "initial", start_time, _done | {idx for idx, _ in _pairs})


def main_resume(run_folder: str):
    """Resume a partially completed run by skipping already-processed samples.

    If the folder name matches part_X_Y (a divided-run chunk), only samples
    in [X, Y) are processed (start=X, end=Y), mirroring the original split.
    """
    run_folder = Path(run_folder).resolve()
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder not found: {run_folder}")

    yaml_path = find_run_yaml(run_folder)
    network = yaml_path.stem

    skip_indices, skip_pairs = find_processed_indices(run_folder)
    # data_indices with partial TargetedSDP results (some targets done, not all)
    partial_indices = {idx for idx, _ in skip_pairs} - skip_indices
    print(f"Already fully processed: {len(skip_indices)} samples — {sorted(skip_indices)}")
    print(f"Partially processed (TargetedSDP): {len(partial_indices)} samples — {sorted(partial_indices)}")
    print(f"Done (data_index, target) pairs: {len(skip_pairs)}")

    existing_results = load_existing_results(run_folder)
    print(f"Loaded {len(existing_results)} existing result rows.")

    certif_problem = Certification_Problem.load_from_yaml(f"{network}.yaml", config_path=str(yaml_path))
    certif_problem.benchmark = existing_results

    results_base = get_project_path(f"results/benchmark/{certif_problem.title}")
    title_run = str(run_folder.relative_to(results_base))

    # Restore start/end range when resuming a divided-run chunk (folder = part_X_Y)
    start, end = None, None
    m = re.match(r"part_(\d+)_(\d+)$", run_folder.name)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        print(f"Chunk resume detected: samples [{start}, {end})")

    start_time = datetime.datetime.now()

    log_path = run_folder / "resume.log"
    try:
        with open(log_path, "a") as log_file:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(original_stdout, log_file)
            sys.stderr = _Tee(original_stderr, log_file)
            try:
                certif_problem.solve(title_run, skip_indices=skip_indices, skip_pairs=skip_pairs,
                                     resume=True, start=start, end=end)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        print(f"Resume log saved → {log_path}")
    finally:
        new_fully_done, new_pairs = find_processed_indices(run_folder)
        new_indices = (new_fully_done - skip_indices) | {idx for idx, _ in new_pairs - skip_pairs}
        log_run_history(run_folder, "resume", start_time, new_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Network Parser")

    parser.add_argument("network", type=str, nargs="?", help="Network to test", default=None)
    parser.add_argument("title_run", type=str, nargs="?", help="Description-run", default="")
    parser.add_argument("--start", type=int, default=None,
                        help="Worker mode: start index (inclusive) for sample slice")
    parser.add_argument("--end", type=int, default=None,
                        help="Worker mode: end index (exclusive) for sample slice")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume an existing run: path to the run folder")
    parser.add_argument("--config", type=str, default=None,
                        help="Worker mode: full path to yaml config (overrides config/ lookup)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Explicit list of data_indexes to process (e.g. --indices 3 7 42)")
    args = parser.parse_args()

    print("Number of CPU : ", mp.cpu_count())

    if args.resume:
        main_resume(args.resume)
    else:
        if args.network is None:
            parser.error("network is required when not using --resume")
        include_indices = set(args.indices) if args.indices else None
        main(network=args.network, title_run=args.title_run, start=args.start, end=args.end,
             config_path=args.config, include_indices=include_indices)

    
