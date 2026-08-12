from adv_train import evaluate_robust, load_adversarial_training_config, evaluate_clean
import data
from network import ReLUNN
import argparse
from torch.utils.data import DataLoader
from fastsdp_tools import get_project_path
import torch
from train import evaluate
from adv_train import evaluate_robust

device = "cuda:0"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Networks")

    parser.add_argument("network", type=str, help="Network to test", default="mnist-6x100")
    args = parser.parse_args()

    yaml_file = f"{args.network}.yaml"  # "mnist_one_data_benchmark.yaml"

    network = ReLUNN.from_yaml(get_project_path(f"config/{yaml_file}"))
    network = network.to(device)

    config = load_adversarial_training_config(get_project_path(f"config/networks/{args.network}.yaml"))
    # robust_to_test_dataset = torch.load(
    #     get_project_path(config["evaluate_robustness_path"]), weights_only=False
    # )["dataset"]
    robust_to_test_dataset = torch.load("/share/homes/boyerma/FastSDPCertification/data/datasets/blob_dataset.pth", weights_only = False)["dataset"]
  
    # test_dataset = torch.load(get_project_path(config["test_path"]), weights_only=False)["dataset"]

    dataloader = DataLoader(
        robust_to_test_dataset,
        batch_size=100,
        shuffle=False,  # Pas besoin de mélanger pour test
        num_workers=2,
        pin_memory=True,
    )
    robust_dataloader = DataLoader(
        robust_to_test_dataset,
        batch_size=1,
        shuffle=False,  # Pas besoin de mélanger pour test
        num_workers=2,
        pin_memory=True,
    )

    # pgd_robust = evaluate_robust(
    #                 network,
    #                 dataloader,
    #                 "cuda:0",
    #                 {
    #                     "eps": 0.015,
    #                     "alpha": 0.01,
    #                     "steps": 40,
    #                     "random_start": True,
    #                     "norm": "inf",
    #                 },
    #             )

    acc = evaluate(
        network,
        dataloader,
    )
    rob_acc = evaluate_robust(
        network,
        robust_dataloader,
        device,
        {
            "eps": 0.05,
            "alpha": 0.005,
            "steps": 40,
            "random_start": True,
            "norm": "inf",
        },
    )
    print("accuracy: ", acc)
    print("Robust accuracy : ", rob_acc)
