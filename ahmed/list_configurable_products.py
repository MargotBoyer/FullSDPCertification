import os
import sys
import yaml

# --- Configuration du chemin pour importer vos modules ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.solve.sdp_solve.SDPmodels import LanSDP
from src.networks import ReLUNN
from src.fastsdp_tools.utils import get_project_path

def get_all_cross_products(solver_instance):
    """
    Parcourt le réseau pour identifier tous les produits croisés
    qui nécessiteront une relaxation.
    """
    cross_products = []
    # On parcourt les couches où une relaxation peut se produire
    for k in range(1, solver_instance.K):
        # Pour chaque neurone de la couche k
        for j in range(solver_instance.n[k]):
            # On vérifie s'il a des prédécesseurs stables actifs
            has_stable_actives = any((k - 1, i) in solver_instance.stable_actives_neurons for i in range(solver_instance.n[k - 1]))
            if not has_stable_actives:
                continue

            # Si oui, on récupère les termes de la décomposition
            for i in range(solver_instance.n[k - 1]):
                if (k - 1, i) in solver_instance.stable_actives_neurons:
                    decomposed_terms, _ = solver_instance.layers_values.get_equivalent_values(k - 1, i)
                    for (l1, n1), _ in decomposed_terms.items():
                        # On vérifie que ce produit n'est pas déjà dans une matrice SDP
                        groups_l1 = {g for g, _ in solver_instance.handler.indexes_matrices._layer_to_groups.get(l1, [])}
                        groups_k = {g for g, _ in solver_instance.handler.indexes_matrices._layer_to_groups.get(k, [])}
                        if not (groups_l1 & groups_k):
                             # C'est un produit croisé à relaxer !
                            key = (k - 1, i, l1, n1)
                            if key not in cross_products:
                                cross_products.append(key)
    return cross_products

def main():
    """
    Script principal pour lister les produits croisés configurables et générer un
    fichier de stratégie YAML modèle.
    """
    # --- PARAMÈTRES DE L'INSTANCE À ANALYSER ---
    NETWORK_PATH = get_project_path("data/models/blob_adv_blob_nn_4x10.pt") # Adaptez ce chemin
    X_PATH = "data/samples/blob/blob_samples.pth" # Adaptez ce chemin
    EPSILON = 0.01
    YTRUE = 0
    YTARGET = 1

    print("--- Identification des produits croisés configurables ---")

    # Instancier le solveur pour analyser la structure du problème
    base_solver_instance = LanSDP(network=ReLUNN.from_pth(NETWORK_PATH), x=X_PATH, epsilon=EPSILON, ytrue=YTRUE, ytargets=[YTARGET], MATRIX_BY_LAYERS=True, LAST_LAYER=False, verbose=False)
    all_products = get_all_cross_products(base_solver_instance)

    if not all_products:
        print("Aucun produit croisé nécessitant une relaxation n'a été trouvé.")
        return

    print(f"\nTrouvé {len(all_products)} produits croisés configurables.")

    # Créer un dictionnaire de stratégie modèle (par défaut 'composed')
    strategy_template = {
        f"product_{i}": {'key': list(product), 'type': 'composed'}
        for i, product in enumerate(all_products)
    }

    # Créer le bloc de texte YAML à copier
    yaml_output_dict = {'bound_strategy': strategy_template}
    yaml_string = yaml.dump(yaml_output_dict, default_flow_style=False, sort_keys=False, indent=2)

    # Sauvegarder ce bloc dans un fichier pour le copier facilement
    output_path = get_project_path("experiments/strategy_block_to_copy.yaml")
    with open(output_path, 'w') as f:
        f.write("# Copiez ce bloc dans votre fichier de configuration principal (ex: blob_4x10.yaml)\n")
        f.write("# et placez-le au même niveau que 'certification_model_name'.\n\n")
        f.write(yaml_string)
    print(f"\nUn bloc de configuration a été généré dans 'experiments/strategy_block_to_copy.yaml'.")
    print("Ouvrez ce fichier et copiez son contenu dans votre fichier de configuration principal (ex: config/blob_4x10.yaml).")

if __name__ == "__main__":
    main()