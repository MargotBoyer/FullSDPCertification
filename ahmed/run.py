# experiments/run_bound_optimization.py

import os
import sys
import time
import json
import yaml  # Importer yaml pour lire les stratégies
import argparse # Pour passer le nom du fichier de stratégie en argument

# --- Configuration du chemin pour importer vos modules ---
# Adaptez ce chemin si nécessaire
# En supposant que ce script est dans FastSDPCertification/ahmed/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.solve.sdp_solve.SDPmodels import LanSDP
from src.networks import ReLUNN
from src.fastsdp_tools.utils import get_project_path


def run_single_certification(network_path, epsilon, ytrue, ytarget, bound_strategy):
    """
    Lance une seule certification SDP avec une stratégie de bornes donnée.
    """
    try:
        # Créez une instance du solveur en passant la stratégie
        solver = LanSDP(
            network=ReLUNN.from_pth(network_path),
            epsilon=epsilon,
            ytrue=ytrue,
            ytargets=[ytarget],
            bound_strategy=bound_strategy,  # <-- C'est ici que la recette est passée !
            # Ajoutez d'autres paramètres nécessaires à votre solveur
            MATRIX_BY_LAYERS=True, 
            LAST_LAYER=False,
            verbose=False # Mettre à True pour plus de détails pendant la résolution
        )
        
        start_time = time.time()
        objective_value = solver.solve()
        solve_time = time.time() - start_time
        
        print(f"  Résultat: {objective_value:.6f} (temps: {solve_time:.2f}s)")
        return objective_value, solve_time

    except Exception as e:
        print(f"  ERREUR lors de la résolution : {e}")
        return None, None


def load_strategy_from_yaml(filepath):
    """
    Charge une stratégie depuis un fichier YAML et la convertit au format attendu
    (le dictionnaire de clés doit utiliser des tuples).
    """
    with open(filepath, 'r') as f:
        strategy_from_yaml = yaml.safe_load(f)
    # Convertir les listes de clés en tuples pour qu'elles soient utilisables
    bound_strategy = {
        tuple(product_data['key']): product_data['type']
        for product_data in strategy_from_yaml.values()
    }
    return bound_strategy


def run_experiment_with_strategy(network_path, epsilon, ytrue, ytarget, strategy_path):
    """
    Teste une seule stratégie de bornes définie dans un fichier YAML et sauvegarde le résultat.
    """
    print(f"--- Lancement de l'expérience avec la stratégie : {strategy_path} ---")
    # Étape 1: Charger la stratégie depuis le fichier YAML
    try:
        strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
        bound_strategy = load_strategy_from_yaml(strategy_path)
        print(f"Stratégie '{strategy_name}' chargée avec succès.")
    except Exception as e:
        print(f"ERREUR: Impossible de charger le fichier de stratégie '{strategy_path}'. Détails: {e}")
        return

    # Étape 2: Créer un dossier pour sauvegarder le résultat
    experiment_timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = get_project_path(f"experiments/{strategy_name}_{experiment_timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Les résultats seront sauvegardés dans : {results_dir}")

    # Étape 3: Lancer la certification
    print(f"\n--- Test de la stratégie : '{strategy_name}' ---")
    objective, solve_time = run_single_certification(network_path, epsilon, ytrue, ytarget, bound_strategy)
    # Étape 4: Sauvegarder le résultat
    result_data = {
        "strategy_name": strategy_name,
        "objective_value": objective,
        "solve_time_seconds": solve_time,
        "network_path": network_path,
        "epsilon": epsilon,
        "ytrue": ytrue,
        "ytarget": ytarget,
        "bound_strategy": {str(k): v for k, v in bound_strategy.items()}
    }
    
    result_filepath = os.path.join(results_dir, f"result.json")
    with open(result_filepath, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"  Résultat sauvegardé dans : {result_filepath}")
    print("\n--- Expérimentation terminée ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer une certification SDP avec une stratégie de bornes spécifique.")
    parser.add_argument("strategy_file", type=str, help="Chemin vers le fichier de stratégie YAML.")
    args = parser.parse_args()
    
    # --- PARAMÈTRES DE VOTRE EXPÉRIENCE ---
    # Adaptez ces valeurs à votre cas d'usage
    NETWORK_PATH = get_project_path("data/models/mnist_adv_6x100.pt")
    EPSILON = 0.01
    YTRUE = 0
    YTARGET = 1
    
    run_experiment_with_strategy(NETWORK_PATH, EPSILON, YTRUE, YTARGET, args.strategy_file)
