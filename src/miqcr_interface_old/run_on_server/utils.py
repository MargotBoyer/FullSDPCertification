import os
from fastsdp_tools import get_project_path

# Configuration des paramètres
factors = [0, 0.01, 0.1, 1]
targets_mnist = [0, 1, 3, 4, 5, 6, 7, 8, 9]
targets_blob = [0, 1]
targets_moon = [0]

data_modeles = ["blob"]
certification_problems = [
    "Targeted",
    "Mzbar",
    "Md",
]

# Chemin local où les logs seront stockés
local_logs_dir = get_project_path(
    "results/conic_bundle/logs"
)  # Répertoire local pour stocker les logs

# Intervalle de synchronisation des logs (en secondes)
sync_interval = 60  # Synchronisation toutes les 60 secondes

# Variable globale pour stocker les informations de log
all_logs = []


def give_certification_problem_parser_file(
    certification_problem: str,
    data_modele: str,
    target: int = None,
    McCormick: str = "none",
):
    """
    Fonction pour donner le nom du fichier de certification en fonction du problème
    """
    assert data_modele in ["blob", "moon", "mnist"]
    assert McCormick in ["none", "all", "inter_layers"]
    if certification_problem == "Targeted":
        assert target is not None
        return f"conic_bundle_files/conic_bundle_{data_modele}_Lan_target={target}_McCormick={McCormick}.txt"
    elif certification_problem == "Mzbar":
        return f"conic_bundle_files/conic_bundle_{data_modele}_Mzbar_McCormick={McCormick}.txt"
    elif certification_problem == "Md":
        return f"conic_bundle_files/conic_bundle_{data_modele}_Md_McCormick={McCormick}.txt"
    else:
        raise ValueError("Problème de certification non reconnu.")


def give_certification_problem_log_file(
    certification_problem: str,
    data_modele: str,
    factor: float,
    triangle=float,
    target: int = None,
    McCormick: str = "none",
):
    assert data_modele in ["blob", "moon", "mnist"]
    assert McCormick in ["none", "all", "inter_layers"]
    if certification_problem == "Targeted":
        assert target is not None
        return f"conic_bundle_files/logfiles/log_cb_{data_modele}_Lan_factor={factor}_triangle={triangle}_target={target}_McCormick={McCormick}.log"
    elif certification_problem == "Mzbar":
        return f"conic_bundle_files/logfiles/log_cb_{data_modele}_Mzbar_factor={factor}_triangle={triangle}_McCormick={McCormick}.log"
    elif certification_problem == "Md":
        return f"conic_bundle_files/logfiles/log_cb_{data_modele}_Md_factor={factor}_triangle={triangle}_McCormick={McCormick}.log"
    else:
        raise ValueError("Problème de certification non reconnu.")


def give_session_screen_name(
    certification_problem: str,
    data_modele: str,
    factor: float,
    triangle: float,
    target: int = None,
    McCormick: str = "none",
):
    assert data_modele in ["blob", "moon", "mnist"]
    assert McCormick in ["none", "all", "inter_layers"]
    if certification_problem == "Targeted":
        assert target is not None
        return f"log_cb_{data_modele}_Lan_factor={factor}_triangle={triangle}_target={target}_McCormick={McCormick}"
    elif certification_problem == "Mzbar":
        return f"log_cb_{data_modele}_Mzbar_factor={factor}_triangle={triangle}_McCormick={McCormick}"
    elif certification_problem == "Md":
        return f"log_cb_{data_modele}_Md_factor={factor}_triangle={triangle}_McCormick={McCormick}"
    else:
        raise ValueError("Problème de certification non reconnu.")


# Configuration SSH
ssh_host = "gpu6"  # Le nom ou l'adresse IP du serveur distant
ssh_username = None  # Sera demandé si None
ssh_password = None  # Sera demandé si None
ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")  # Chemin vers votre clé SSH

# Chemin distant où les commandes seront exécutées
remote_working_dir = (
    "~/Miqcr-1.0_Robustesse/src/"  # Remplacez par votre dossier sur le serveur distant
)
