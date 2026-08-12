def entrer_dans_session_screen(client, nom_session="ma_session"):
    """
    Rentre dans une session screen distante via une commande interactive.

    Args:
        client: Objet client SSH connecté
        nom_session: Nom de la session screen à rejoindre
    """
    if client is None:
        print("Pas de connexion SSH.")
        return

    try:
        # Préparer une session interactive
        channel = client.invoke_shell()
        commande = f"screen -dr {nom_session}\n"
        channel.send(commande)

        print(
            f"Connexion à la session screen '{nom_session}'... (Ctrl+A, D pour détacher)"
        )

        # Relais entre le terminal local et la session distante
        import select
        import sys

        while True:
            # Vérifie si une entrée utilisateur est disponible
            rlist, _, _ = select.select([channel, sys.stdin], [], [])
            if channel in rlist:
                try:
                    output = channel.recv(1024).decode()
                    if not output:
                        break
                    sys.stdout.write(output)
                    sys.stdout.flush()
                except Exception:
                    break
            if sys.stdin in rlist:
                cmd = sys.stdin.read(1)
                if not cmd:
                    break
                channel.send(cmd)

    except Exception as e:
        print(f"Erreur lors de la connexion à la session screen: {str(e)}")


def creer_session_screen(client, nom_session="ma_session", nom_log="log_session.log"):
    """
    Crée une session screen sur le serveur distant avec logging.

    Args:
        client: Objet client SSH connecté
        nom_session: Nom de la session screen
        nom_log: Nom du fichier de log

    Returns:
        True si la session a été créée avec succès, False sinon.
    """
    if client is None:
        print("Pas de connexion SSH.")
        return False

    commande = f'screen -L -Logfile "{nom_log}" -S "{nom_session}"'
    # commande = f'screen -dm -L -Logfile "{nom_log}" -S "{nom_session}" bash -c "echo SCREEN_STARTED; exec bash"'

    try:
        transport = client.get_transport()
        channel = transport.open_session()

        # Lancer la commande screen en mode interactif
        channel.get_pty()
        channel.exec_command(commande)

        print(f"Session screen '{nom_session}' créée avec le log '{nom_log}'")
        return True

    except Exception as e:
        print(f"Erreur lors de la création de la session screen: {str(e)}")
        return False


def detacher_session_screen(client, nom_session="ma_session"):
    """
    Détache une session screen existante sur le serveur distant.

    Args:
        client: Objet client SSH connecté
        nom_session: Nom de la session screen à détacher

    Returns:
        True si la session a été détachée avec succès, False sinon.
    """
    if client is None:
        print("Pas de connexion SSH.")
        return False

    # La commande 'screen -S <nom> -d' détache la session
    commande = f"screen -S {nom_session} -d"

    try:
        stdin, stdout, stderr = client.exec_command(commande)
        erreur = stderr.read().decode("utf-8").strip()
        if erreur:
            print(f"Erreur: {erreur}")
            return False

        print(f"Session screen '{nom_session}' détachée avec succès.")
        return True

    except Exception as e:
        print(f"Erreur lors du détachement de la session screen: {str(e)}")
        return False


def lister_sessions_screen(client):
    """
    Liste toutes les sessions screen actives sur le serveur distant.

    Args:
        client: Objet client SSH connecté

    Returns:
        Une liste des noms de sessions actives, ou une liste vide s'il n'y en a pas.
    """
    if client is None:
        print("Pas de connexion SSH.")
        return []

    try:
        stdin, stdout, stderr = client.exec_command("screen -ls")
        sortie = stdout.read().decode("utf-8")
        erreurs = stderr.read().decode("utf-8")

        if erreurs:
            print(f"Erreur: {erreurs.strip()}")
            return []

        print("\nSessions screen actives :")
        print(sortie)

        # Extraire les noms des sessions à partir de la sortie
        sessions = []
        for ligne in sortie.splitlines():
            if "\t" in ligne:
                partie = ligne.strip().split("\t")[1]  # format: ID.NOM
                if "." in partie:
                    sessions.append(partie.split(".", 1)[1])

        return sessions

    except Exception as e:
        print(f"Erreur lors de la récupération des sessions screen: {str(e)}")
        return []


def executer_commande_dans_screen(client, nom_session, commande):
    """
    Envoie une commande à exécuter dans une session screen détachée.

    Args:
        client: Objet client SSH connecté
        nom_session: Nom de la session screen
        commande: La commande shell à exécuter (ex: 'ls -l /tmp')
    """
    if client is None:
        print("Pas de connexion SSH.")
        return False

    # Assurez-vous que la commande se termine par un retour à la ligne
    commande_screen = f"screen -S {nom_session} -X stuff '{commande}\\n'"
    # commande_screen = commande

    try:
        stdin, stdout, stderr = client.exec_command(commande_screen)
        erreurs = stderr.read().decode().strip()
        if erreurs:
            print(f"Erreur screen: {erreurs}")
            return False

        print(f"Commande envoyée à la session screen '{nom_session}': {commande}")
        return True

    except Exception as e:
        print(f"Erreur lors de l'exécution dans la session screen: {str(e)}")
        return False


def replace_line_new_value(match, value, i, line, lines, name_parameter):
    """
    Remplace une ligne dans un fichier avec une nouvelle valeur.

    Args:
        match: Objet de correspondance de regex
        value: Nouvelle valeur à insérer

    Returns:
        str: Ligne modifiée
    """
    # Récupérer l'ancienne valeur pour l'afficher
    ancienne_valeur = match.group(2)
    # Remplacer l'ancienne valeur par la nouvelle
    prefix = match.group(1)  # "FACTOR " avec les espaces
    nouvelle_ligne = f"{prefix}{value}"
    lines[i] = line.replace(match.group(0), nouvelle_ligne)

    print(
        f"Valeur {name_parameter} modifiée sur le serveur: {ancienne_valeur} -> {value}"
    )


def modify_element_in_file(
    client,
    chemin_fichier_distant: str,
    new_factor_value: float = 0,
    new_triangle_value: float = 0,
):
    """
    Modifie la valeur FACTOR dans un fichier sur le serveur distant.

    Args:
        client: Objet client SSH connecté
        chemin_fichier_distant: Chemin du fichier sur le serveur distant
        new_factor_value: Nouvelle valeur pour FACTOR

    Returns:
        bool: True si l'opération a réussi, False sinon
    """
    if client is None:
        print("Impossible de modifier le fichier: pas de connexion SSH.")
        return False

    try:
        # Créer une session SFTP
        sftp = client.open_sftp()

        try:
            # Créer un nom de fichier temporaire local
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file_path = temp_file.name
            temp_file.close()

            # Télécharger le fichier distant
            sftp.get(chemin_fichier_distant, temp_file_path)

            # Modifier le fichier localement
            factor_modifie = False
            triangle_modifie = False
            with open(temp_file_path, "r") as f:
                lignes = f.readlines()

            import re

            for i, ligne in enumerate(lignes):
                # Recherche un motif "FACTOR" suivi d'un nombre flottant
                match_factor = re.search(r"(FACTOR\s+)(-?\d+\.?\d*)", ligne)
                match_triangle = re.search(r"(TRIANGLE\s+)(-?\d+\.?\d*)", ligne)
                if match_factor:
                    replace_line_new_value(
                        match_factor,
                        new_factor_value,
                        i,
                        ligne,
                        lignes,
                        name_parameter="FACTOR",
                    )
                    factor_modifie = True

                elif match_triangle:
                    replace_line_new_value(
                        match_triangle,
                        new_triangle_value,
                        i,
                        ligne,
                        lignes,
                        name_parameter="TRIANGLE",
                    )
                    triangle_modifie = True

                if factor_modifie and triangle_modifie:
                    break

            if not factor_modifie:
                print(
                    f"Aucune ligne commençant par 'FACTOR' suivie d'un nombre n'a été trouvée dans {chemin_fichier_distant}"
                )
                return False

            # Écrire les modifications dans le fichier temporaire
            with open(temp_file_path, "w") as f:
                f.writelines(lignes)

            # Renvoyer le fichier modifié vers le serveur
            sftp.put(temp_file_path, chemin_fichier_distant)

            # Supprimer le fichier temporaire
            import os

            os.unlink(temp_file_path)

            print(f"Fichier distant '{chemin_fichier_distant}' modifié avec succès !")
            return True

        except FileNotFoundError:
            print(
                f"Erreur: Le fichier '{chemin_fichier_distant}' n'existe pas sur le serveur."
            )
            return False
        except PermissionError:
            print(
                f"Erreur: Pas les permissions pour modifier '{chemin_fichier_distant}'."
            )
            return False
        except Exception as e:
            print(f"Erreur lors de la modification du fichier distant: {str(e)}")
            return False
        finally:
            sftp.close()

    except Exception as e:
        print(f"Erreur lors de l'ouverture de la session SFTP: {str(e)}")
        return False
