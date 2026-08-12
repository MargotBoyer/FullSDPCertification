#!/usr/bin/env python3
import subprocess
import time
import os
import paramiko
import getpass
import threading
import datetime
from pathlib import Path
import sys
import argparse
import re

from utils import (
    ssh_host,
    ssh_username,
    ssh_password,
    ssh_key_path,
    remote_working_dir,
    factors,
    targets_mnist,
    targets_blob,
    targets_moon,
    certification_problems,
    data_modeles,
    local_logs_dir,
    all_logs,
    give_certification_problem_parser_file,
    give_certification_problem_log_file,
    give_session_screen_name,
)

from screen_utils import modify_element_in_file
from conic_bundle.run_on_server.screen_utils import (
    creer_session_screen,
    detacher_session_screen,
    lister_sessions_screen,
    entrer_dans_session_screen,
    executer_commande_dans_screen,
)

from fastsdp_tools import parse_string_list, parse_float_list


def se_connecter_ssh(hostname="cedric2-ro"):
    """
    Établit une connexion SSH vers le serveur spécifié.

    Args:
        hostname: Nom d'hôte ou adresse IP du serveur (défaut: cedric2-ro)

    Returns:
        client: Objet client SSH connecté ou None en cas d'erreur
    """
    # Créer le client SSH
    client = paramiko.SSHClient()

    # Ajouter automatiquement les clés d'hôte inconnues
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"Connexion à {hostname}...")

        # Utiliser la configuration SSH existante (comme si vous tapiez 'ssh cedric2-ro')
        # Pas besoin de spécifier username/password car votre config SSH s'en charge
        client.connect(hostname=hostname)

        print(f"Connexion établie avec succès à {hostname}")
        return client

    except paramiko.AuthenticationException:
        print("Erreur d'authentification.")
        return None
    except paramiko.SSHException as e:
        print(f"Erreur SSH: {str(e)}")
        return None
    except Exception as e:
        print(f"Erreur lors de la connexion: {str(e)}")
        return None


def lire_fichier_distant(client, chemin_fichier=None):
    """
    Lit le contenu d'un fichier sur le serveur distant.

    Args:
        client: Objet client SSH connecté
        chemin_fichier: Chemin du fichier à lire sur le serveur distant

    Returns:
        contenu: Contenu du fichier ou None en cas d'erreur
    """
    if client is None:
        print("Impossible de lire le fichier: pas de connexion SSH.")
        return None

    if chemin_fichier is None:
        chemin_fichier = input(
            "Entrez le chemin du fichier à lire sur le serveur distant: "
        )

    try:
        # Créer une session SFTP
        sftp = client.open_sftp()

        try:
            # Vérifier si le fichier existe
            sftp.stat(chemin_fichier)

            # Ouvrir et lire le fichier
            with sftp.open(chemin_fichier, "r") as fichier:
                contenu = fichier.read()

                # Convertir en texte si c'est des bytes
                if isinstance(contenu, bytes):
                    contenu = contenu.decode("utf-8")

                print(f"\nContenu du fichier {chemin_fichier}:\n")
                print(contenu)

                # Si vous voulez aussi sauvegarder le fichier localement
                nom_fichier_local = os.path.basename(chemin_fichier)
                sauvegarder = input(
                    f"\nVoulez-vous sauvegarder le fichier localement sous '{nom_fichier_local}'? (o/n): "
                )

                if sauvegarder.lower() == "o":
                    with open(nom_fichier_local, "w") as f_local:
                        f_local.write(contenu)
                    print(f"Fichier sauvegardé avec succès sous '{nom_fichier_local}'")

                return contenu

        except FileNotFoundError:
            print(f"Erreur: Le fichier '{chemin_fichier}' n'existe pas sur le serveur.")
            return None
        except PermissionError:
            print(f"Erreur: Pas les permissions pour lire '{chemin_fichier}'.")
            return None
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {str(e)}")
            return None
        finally:
            sftp.close()

    except Exception as e:
        print(f"Erreur lors de l'ouverture de la session SFTP: {str(e)}")
        return None


def analyser_log_sdp(contenu_log):
    """
    Analyse le contenu d'un log de résolution SDP pour extraire les valeurs importantes.

    Args:
        contenu_log: Contenu du fichier log

    Returns:
        dict: Dictionnaire contenant les valeurs extraites
    """
    resultats = {}

    # Rechercher la valeur objective
    match_obj = re.search(
        r"Objective function value sdp\s*:\s*(-?\d+\.?\d*)", contenu_log
    )
    if match_obj:
        resultats["valeur_objective"] = float(match_obj.group(1))
        print(f"Valeur de la fonction objective SDP: {resultats['valeur_objective']}")
    else:
        print("Valeur de la fonction objective non trouvée dans le log")

    # Rechercher le temps d'exécution final
    # On cherche la dernière ligne contenant _endit pour obtenir le temps final
    match_time = re.findall(r"((\d{2}:\d{2}:\d{2}\.\d{2}).*_endit.*)", contenu_log)
    if match_time:
        # Prendre la dernière occurrence
        last_match = match_time[-1]
        resultats["temps_execution"] = last_match[
            1
        ]  # Le groupe 1 contient juste le temps
        print(f"Temps d'exécution final du solveur SDP: {resultats['temps_execution']}")
    else:
        print("Temps d'exécution non trouvé dans le log")

    return resultats


def ecrire_fichier_distant(client, chemin_fichier=None, contenu=None):
    """
    Crée ou écrit dans un fichier sur le serveur distant.

    Args:
        client: Objet client SSH connecté
        chemin_fichier: Chemin du fichier à écrire sur le serveur distant
        contenu: Contenu à écrire dans le fichier

    Returns:
        bool: True si l'opération a réussi, False sinon
    """
    if client is None:
        print("Impossible d'écrire le fichier: pas de connexion SSH.")
        return False

    if chemin_fichier is None:
        chemin_fichier = input(
            "Entrez le chemin du fichier à créer/modifier sur le serveur distant: "
        )

    if contenu is None:
        print(
            "Entrez le contenu du fichier (terminez par Ctrl+D sur une nouvelle ligne):"
        )
        contenu_lines = []
        try:
            while True:
                line = input()
                contenu_lines.append(line)
        except EOFError:
            pass
        contenu = "\n".join(contenu_lines)

    try:
        # Créer une session SFTP
        sftp = client.open_sftp()

        try:
            # Ouvrir le fichier en mode écriture (écrase le contenu existant)
            with sftp.open(chemin_fichier, "w") as fichier:
                fichier.write(contenu)

            print(
                f"Fichier '{chemin_fichier}' créé/modifié avec succès sur le serveur distant."
            )
            return True

        except PermissionError:
            print(f"Erreur: Pas les permissions pour écrire dans '{chemin_fichier}'.")
            return False
        except Exception as e:
            print(f"Erreur lors de l'écriture du fichier: {str(e)}")
            return False
        finally:
            sftp.close()

    except Exception as e:
        print(f"Erreur lors de l'ouverture de la session SFTP: {str(e)}")
        return False


def cree_toute_execution_modele(
    client,
    data_modele,
    certification_problem,
    factor: float,
    triangle: float,
    McCormick: str,
    target: int = None,
):
    assert data_modele in ["blob", "moon", "mnist"]
    assert certification_problem in ["Targeted", "Mzbar", "Md"]
    assert factor in factors

    if certification_problem == "Targeted":
        assert target is not None
        chemin_fichier = give_certification_problem_parser_file(
            certification_problem, data_modele, target, McCormick
        )
        logfile = give_certification_problem_log_file(
            certification_problem, data_modele, factor, triangle, target, McCormick
        )
        session_screen_name = give_session_screen_name(
            certification_problem, data_modele, factor, triangle, target, McCormick
        )
    elif certification_problem == "Mzbar" or certification_problem == "Md":
        chemin_fichier = give_certification_problem_parser_file(
            certification_problem, data_modele, McCormick
        )
        logfile = give_certification_problem_log_file(
            certification_problem, data_modele, factor, triangle, McCormick
        )
        session_screen_name = give_session_screen_name(
            certification_problem, data_modele, factor, triangle, McCormick
        )

    ecrire_fichier_distant(client, chemin_fichier=logfile, contenu="Contenu de test\n")

    time.sleep(1)
    creer_session_screen(client, nom_session=session_screen_name, nom_log=logfile)
    time.sleep(10)  # Attendre un peu pour s'assurer que la session est créée

    command = "cd Miqcr-1.0_Robustesse/src"
    executer_commande_dans_screen(
        client, nom_session=session_screen_name, commande=command
    )
    time.sleep(5)
    modify_element_in_file(
        client,
        "Miqcr-1.0_Robustesse/src/param.smiqp",
        new_factor_value=factor,
        new_triangle_value=triangle,
    )
    time.sleep(5)  # Attendre un peu pour s'assurer que la commande est exécutée
    command_smiqp = f"./smiqp ../../{chemin_fichier}"
    executer_commande_dans_screen(
        client, nom_session=session_screen_name, commande=command_smiqp
    )
    # contenu = lire_fichier_distant(client, logfile)
    # analyser_log_sdp(contenu)
    detacher_session_screen(client, nom_session=session_screen_name)
    lister_sessions_screen(client)


def main():
    # Analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Lire un fichier distant sur un serveur SSH"
    )
    # parser.add_argument("--host", default="cedric2-ro", help="Nom d'hôte ou adresse IP")
    # parser.add_argument(
    #     "--file", help="Chemin du fichier à lire sur le serveur distant"
    # )
    parser.add_argument(
        "--factor",
        type=parse_float_list,
        default=[0.0],
        help="Liste de valeurs du facteur, séparées par des virgules (défaut: 0.0). Ex: 0.5,1.0,1.5",
    )
    parser.add_argument(
        "--triangle",
        type=parse_float_list,
        default=[0.0],
        help="Liste de valeurs du triangle, séparées par des virgules (défaut: 0.0). Ex: 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--McCormick",
        type=parse_string_list,
        default=["none"],
        help="Liste des types de coupes McCormick, séparées par des virgules (défaut: 'all'). Ex: 'all,none,some'",
    )

    args = parser.parse_args()

    # Se connecter au serveur
    client = se_connecter_ssh(hostname="cedric2-ro")

    if client:
        factors = args.factor
        triangles = args.triangle
        McCormicks = args.McCormick

        try:
            for data_modele in ["moon", "mnist"]:
                for certification_problem in ["Targeted", "Md", "Mzbar"]:
                    for factor in factors:
                        for triangle in triangles:
                            for McCormick in McCormicks:

                                if certification_problem == "Targeted":
                                    if data_modele == "blob":
                                        targets = targets_blob
                                    elif data_modele == "moon":
                                        targets = targets_moon
                                    elif data_modele == "mnist":
                                        targets = targets_mnist
                                    for target in targets:
                                        cree_toute_execution_modele(
                                            client,
                                            data_modele=data_modele,
                                            certification_problem="Targeted",
                                            factor=factor,
                                            triangle=triangle,
                                            McCormick=McCormick,
                                            target=target,
                                        )

                                        if factor > 0.1 and data_modele == "mnist":
                                            time.sleep(300)
                                    time.sleep(60)
                                else:
                                    cree_toute_execution_modele(
                                        client,
                                        data_modele=data_modele,
                                        certification_problem=certification_problem,
                                        triangle=triangle,
                                        factor=factor,
                                        McCormick=McCormick,
                                    )
                                    time.sleep(60)
                                    if factor > 0.1 and data_modele == "mnist":
                                        time.sleep(300)

        finally:
            # Fermer la connexions
            client.close()
            print("Connexion fermée.")


if __name__ == "__main__":
    main()
