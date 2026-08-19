import logging
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from network import ReLUNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import sys
import os
import argparse
from typing import List
from adversarial_attacks import PGDAttack, LPAttack, SDPAttack, CrownIBP_Attack
from bounds import compute_bounds_data
from adv_train import (
    complex_adversarial_training_loop,
    load_adversarial_training_config,
)
from adv_train import evaluate_clean
from data import (
    analyze_class_distribution,
    # EMNISTBalancedCSV,
    # emnist_fix_orientation,
    # ShiftLabels,
)


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from fastsdp_tools.utils import str_to_list
from fastsdp_tools import get_project_path, Adversarial_Network_Training
from fastsdp_tools.cuda_probe import patch_torch_cuda_is_available
import data

logger = logging.getLogger(__name__)

device = torch.device("cuda" if patch_torch_cuda_is_available() else "cpu")


def train(model, trainloader, testloader, num_epochs=100, lr=1e-2):
    """
    Trains the model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    print("Training model ...")

    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss/len(trainloader)}")
            evaluate(model, testloader)


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")
    return 100 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Network Parser")

    parser.add_argument("data_modele", type=str, help="Dataset name", default="mnist")

    parser.add_argument(
        "network",
        type=str,
        help="Type of adversarial attack to use during training",
        default="PGD",
    )

    parser.add_argument(
        "adv_train",
        type=int,
        help="Defines if the model is trained with adversarial examples",
        default=True,
    )

    args = parser.parse_args()

    config = load_adversarial_training_config(f"config/networks/{args.network}.yaml")
    train_dataset = torch.load(get_project_path(config["train_path"]), weights_only=False)["dataset"]

    # print("Len of train_dataset: ", len(train_dataset))
    # print("train dataset: ", train_dataset)

    # analyze_class_distribution(
    #     train_dataset, dataset_name=f"Train Dataset {args.data_modele}"
    # )

    test_dataset = torch.load(get_project_path(config["test_path"]), weights_only=False)["dataset"]
    # print("Len of test_dataset: ", len(test_dataset))

    # analyze_class_distribution(
    #     test_dataset, dataset_name=f"Test Dataset {args.data_modele}"
    # )
    # print("")
    robust_to_test_dataset = torch.load(
        get_project_path(config["evaluate_robustness_path"]), weights_only=False
    )["dataset"]

    logger.debug("len =", len(train_dataset))
    logger.debug("dataset[0] = %s", train_dataset[0])

    attack = config["adversarial_attack"]

    print("CONFIG : ", config)
    epsilon_to_test = config.get("epsilon_test", config["epsilon"])

    batch_size = config["batch_size"]

    if args.data_modele == "blob":
        yaml_file = "config/blob.yaml"
        loaded_data = data.load_dataset(yaml_file)
        print("Loaded data: ", loaded_data)
        train_dataset = loaded_data
        test_dataset = loaded_data

    elif args.data_modele == "moon":
        print("args data modele == MOON...")
        yaml_file = "config/moon_one_data_benchmark.yaml"
        loaded_data = data.load_dataset(yaml_file)
        print("Loaded data: ", loaded_data)
        train_dataset = loaded_data
        test_dataset = loaded_data

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Parallélisation
        pin_memory=True,  # Plus rapide pour GPU
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Pas besoin de mélanger pour test
        num_workers=2,
        pin_memory=True,
    )
    robust_datas_loader = DataLoader(
        robust_to_test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Pas besoin de mélanger pour test
        num_workers=2,
        pin_memory=True,
    )

    n = config["n"]
    K = config["K"]

    logger.debug("n : ", n)
    logger.debug("K : ", K)

    model = ReLUNN(
        K,
        n,
        W=None,
        b=None,
        dropout_prob=config.get("dropout", 0.0),
        name=config["name_network"],
    )

    if args.adv_train == 1:
        print("Training with adversarial examples...")
        # simple_adversarial_training_loop(
        #     model,
        #     dataloader,
        #     dataloader,
        #     device,
        #     num_epochs=100,
        #     lr=0.001,
        #     eps=0.3,
        #     adversarial_attack=args.attack,
        # )
        results = complex_adversarial_training_loop(
            model=model,
            trainloader=train_loader,
            testloader=test_loader,
            robust_datas_loader=robust_datas_loader,
            device=device,
            project_name=config["name_network"],
            experiment_name=f"{args.data_modele}_{attack}_"
            + datetime.datetime.now().strftime("%m_%d_%Hh%M_%Ss"),
            log_frequency=10,
            use_wandb=False,
            yaml_file=f"networks/{args.network}.yaml",
            epsilon_to_test=epsilon_to_test,
            **config,
        )
    else:
        train(
            model,
            train_loader,
            test_loader,
            num_epochs=config["num_epochs"],
            lr=config["lr"],
        )

    state_dict = model.state_dict()
    print(type(state_dict))  # Cela doit afficher <class 'collections.OrderedDict'>
    print(state_dict)  # Affiche le contenu du state_dict

    name_network = config["name_network"]
    
    # print("STUDY K at the end of training : ", model.K)
    # print("STUDY n at the end of training  : ", model.n)
    # for k in range(K):
    #     print(
    #         f" STUDY LAYER = {k}"
    #         + f"len(b[{k}]) = {len(model.b[k])}"
    #         + f"len(W[{k}]) = {len(model.W[k])}"
    #         + f"len(W[{k}][0]) = {len(model.W[k][0])}"
    #     )
    if args.adv_train:
        print("SAVE ADV : ", f"data/models/{args.data_modele}_adv_{name_network}.pt")

        torch.save(state_dict, f"data/models/{args.data_modele}_adv_{name_network}.pt")
    else:
        print("SAVE ADV : ", f"data/models/{args.data_modele}_nn.pt")
        torch.save(state_dict, f"data/models/{args.data_modele}_nn.pt")
