import re
import pandas as pd
from pandas.api.types import CategoricalDtype
from typing import List

from fastsdp_tools import add_row_from_dict, get_project_path


def extract_log_values(log_file_path):
    """
    Extrait plusieurs valeurs d'un fichier log:
    - La valeur de la fonction objective SDP
    - Le temps de calcul initial
    - Le temps d'exécution SDP (pre-processing)

    Args:
        log_file_path (str): Chemin vers le fichier log

    Returns:
        tuple: (sdp_value, initial_time, sdp_time) ou None pour chaque valeur non trouvée
    """
    try:
        with open(log_file_path, "r") as file:
            content = file.read()

            # Extraction de la valeur objective SDP
            sdp_pattern = r"Objective function value sdp : ([-+]?\d*\.?\d+)"
            # sdp_pattern = r"Primal.  obj: ([-+]?\d*\.?\d+)"  #ATTENTION : à mulitplier par -1 et faire attention à l'écriture sous format 1.51e-7
            sdp_match = re.search(sdp_pattern, content)
            sdp_value = float(sdp_match.group(1)) if sdp_match else None

            # Extraction du temps de calcul initial
            initial_time_pattern = r"Time for computing initial upper bound: (\d+)"
            initial_time_match = re.search(initial_time_pattern, content)
            initial_time = (
                int(initial_time_match.group(1)) if initial_time_match else None
            )

            # Extraction du temps d'exécution SDP (pre-processing)
            # sdp_time_pattern = r"Pre-processing time \(sdp\): (\d+)"
            sdp_time_pattern = r"Optimizer terminated. Time: (\d+)"
            sdp_time_match = re.search(sdp_time_pattern, content)
            sdp_time = int(sdp_time_match.group(1)) if sdp_time_match else None

            return sdp_value, initial_time, sdp_time

    except FileNotFoundError:
        print(f"Erreur: Le fichier '{log_file_path}' n'a pas été trouvé.")
        return None, None, None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return None, None, None


def print_values(sdp_value, initial_time, sdp_time):
    """
    Affiche les valeurs extraites.

    Args:
        sdp_value (float): Valeur de la fonction objective SDP
        initial_time (int): Temps de calcul initial
        sdp_time (int): Temps d'exécution SDP (pre-processing)
    """
    # Affichage des résultats
    if sdp_value is not None:
        print(f"La valeur de la fonction objective SDP est: {sdp_value}")
    else:
        print("La valeur de la fonction objective SDP n'a pas été trouvée.")

    if initial_time is not None:
        print(f"Le temps de calcul initial est: {initial_time}")
    else:
        print("Le temps de calcul initial n'a pas été trouvé.")

    if sdp_time is not None:
        print(f"Le temps d'exécution SDP (pre-processing) est: {sdp_time}")
    else:
        print("Le temps d'exécution SDP n'a pas été trouvé.")


def check_values(sdp_value, sdp_time):
    """
    Vérifie si les valeurs extraites sont valides.

    Args:
        sdp_value (float): Valeur de la fonction objective SDP
        initial_time (int): Temps de calcul initial
        sdp_time (int): Temps d'exécution SDP (pre-processing)

    Returns:
        bool: True si toutes les valeurs sont valides, False sinon
    """
    return all(value is not None for value in [sdp_value, sdp_time])


def create_dataframe_results_cb(
    data_modeles, certification_problems, factors, triangles, McCormicks
):

    benchmark_cb = pd.DataFrame(
        columns=[
            "data_modele",
            "certification_problem",
            "target",
            "factor",
            "triangle",
            "sdp_value",
            "initial_time",
            "sdp_time",
        ]
    )
    for data_modele in data_modeles:
        for factor in factors:
            if data_modele == "mnist" and factor > 0.1:
                print("factor > 0.1 for mnist, skipping")
                continue
            for triangle in triangles:
                for certification_problem in certification_problems:
                    for McCormick in McCormicks:
                        assert McCormick in ["none", "all"]
                        if certification_problem == "Targeted":
                            assert data_modele == "mnist" or data_modele == "moon"
                            for target in (
                                [0, 1, 3, 4, 5, 6, 7, 8, 9]
                                if data_modele == "mnist"
                                else [0]
                            ):
                                log_file = get_project_path(
                                    f"results/conic_bundle/logs/{data_modele}/{certification_problem}/log_cb_{data_modele}_{certification_problem}_factor={factor}_triangle={triangle}_target={target}_McCormick={McCormick}.log"
                                )
                                sdp_value, initial_time, sdp_time = extract_log_values(
                                    log_file
                                )
                                dic = {
                                    "data_modele": data_modele,
                                    "certification_problem": certification_problem,
                                    "target": target,
                                    "factor": factor,
                                    "triangle": triangle,
                                    "McCormick": McCormick,
                                    "sdp_value": sdp_value,
                                    "initial_time": initial_time,
                                    "sdp_time": sdp_time,
                                }
                                print_values(sdp_value, initial_time, sdp_time)
                                if check_values(sdp_value, sdp_time):
                                    benchmark_cb = add_row_from_dict(benchmark_cb, dic)
                                print("dataframe : ", benchmark_cb)
                        else:
                            log_file = get_project_path(
                                f"results/conic_bundle/logs/{data_modele}/{certification_problem}/log_cb_{data_modele}_{certification_problem}_factor={factor}_triangle={triangle}_McCormick={McCormick}.log"
                            )
                            sdp_value, initial_time, sdp_time = extract_log_values(
                                log_file
                            )
                            dic = {
                                "data_modele": data_modele,
                                "certification_problem": certification_problem,
                                "target": None,
                                "factor": factor,
                                "triangle": triangle,
                                "McCormick": McCormick,
                                "sdp_value": sdp_value,
                                "initial_time": initial_time,
                                "sdp_time": sdp_time,
                            }
                            # print_values(sdp_value, initial_time, sdp_time)
                            if check_values(sdp_value, sdp_time):
                                benchmark_cb = add_row_from_dict(benchmark_cb, dic)
                            print("dataframe : ", benchmark_cb)
    return benchmark_cb


def create_overleaf_table_cb_lan(
    benchmark_cb,
    data_modele,
    certification_problem,
    folder_name,
    name_file: str = "cb_table.tex",
    target: int = None,
):
    assert certification_problem == "Targeted"
    df_data_modele_certification_problem = benchmark_cb[
        (benchmark_cb["data_modele"] == data_modele)
        & (benchmark_cb["certification_problem"] == certification_problem)
    ]
    print("\n \n \n \n \n \n \n create overleaf table cb lan : target : ", target)
    print("data modele : ", data_modele)
    if target is not None:
        targets = [target]
    else:
        targets = df_data_modele_certification_problem["target"].unique()
    with open(
        get_project_path(f"{folder_name}/{name_file}"), "a", encoding="utf-8"
    ) as f:

        for target in targets:

            print("target : ", target)
            df_data_modele_certification_problem_target = (
                df_data_modele_certification_problem[
                    df_data_modele_certification_problem["target"] == target
                ].sort_values(by=["factor", "triangle"])
            )
            print("df target : ", df_data_modele_certification_problem_target)

            f.write("\\begin{tabular}{l|lllccc|}\n")
            f.write("\\toprule\n")
            f.write(
                "\\multicolumn{7}{c|}{\\textbf{\Large {"
                + data_modele
                + "} - \Large "
                + certification_problem
                + "}} \\\\\n"
            )
            f.write("\\hline\n")
            f.write("Target & Factor & Tri & McCor & Init & Tps & Val \\\\\n")
            f.write("\\hline\n")

            for index, row in df_data_modele_certification_problem_target.iterrows():
                print("row : ", row)
                f.write(
                    f"{row['target']} & {row['factor']} & {row['triangle']} & {row['McCormick']} & {row['initial_time']} & {row['sdp_time']} & {round(row['sdp_value'],3)} \\\\\n"
                )

                f.write("\\hline\n")

            f.write("\\end{tabular}\n")
            f.write("\\caption{Conic bundle solver performances}\n")


def create_overleaf_table_cb_md_mzbar(
    benchmark_cb,
    data_modele,
    certification_problem,
    folder_name,
    name_file: str = "cb_table.tex",
):
    """
    Crée un tableau pour Overleaf à partir du fichier CSV.
    """
    assert certification_problem in ["Md", "Mzbar"]
    df_data_modele_certification_problem = benchmark_cb[
        (benchmark_cb["data_modele"] == data_modele)
        & (benchmark_cb["certification_problem"] == certification_problem)
    ]
    with open(
        get_project_path(f"{folder_name}/{name_file}"), "a", encoding="utf-8"
    ) as f:

        f.write("\\begin{tabular}{|lllccc|}\n")
        f.write("\\toprule\n")
        f.write(
            "\\multicolumn{6}{c|}{\\textbf{\Large {"
            + data_modele
            + "} - \Large "
            + certification_problem
            + "}} \\\\\n"
        )
        f.write("\\hline\n")
        f.write("Factor & Tri & McCor & Init & Tps & Val \\\\\n")
        f.write("\\hline\n")

        for index, row in df_data_modele_certification_problem.iterrows():
            f.write(
                f"{row['factor']} & {row['triangle']} & {row['McCormick']} & {row['initial_time']} & {row['sdp_time']} & {round(row['sdp_value'],3)} \\\\\n"
            )
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\caption{Conic bundle solver performances}\n")


def create_overleaf_table_cb(
    benchmark_cb,
    data_modele,
    certification_problem,
    folder_name,
    name_file: str = "cb_table.tex",
    target: int = None,
):
    """
    Crée un tableau pour Overleaf à partir du fichier CSV.
    """
    if certification_problem == "Targeted":
        create_overleaf_table_cb_lan(
            benchmark_cb,
            data_modele,
            certification_problem,
            folder_name,
            name_file,
            target,
        )
    else:
        create_overleaf_table_cb_md_mzbar(
            benchmark_cb, data_modele, certification_problem, folder_name, name_file
        )


def apply(data_modeles: List[str]):

    certification_problems = ["Targeted", "Md", "Mzbar"]
    factors = [0.0, 0.1]
    triangles = [0.0, 0.1]

    benchmark_cb = create_dataframe_results_cb(
        data_modeles,
        certification_problems,
        factors,
        triangles,
        McCormicks=["none", "all"],
    )

    create_overleaf_table_cb(
        benchmark_cb,
        "mnist",
        "Targeted",
        "results/conic_bundle/",
        name_file="cb_table_mnist_Lan.tex",
    )
    create_overleaf_table_cb(
        benchmark_cb,
        "mnist",
        "Md",
        "results/conic_bundle/",
        name_file="cb_table_mnist_Md.tex",
    )
    create_overleaf_table_cb(
        benchmark_cb,
        "mnist",
        "Mzbar",
        "results/conic_bundle/",
        name_file="cb_table_mnist_Mzbar.tex",
    )

    create_overleaf_table_cb(
        benchmark_cb,
        "moon",
        "Targeted",
        "results/conic_bundle/",
        name_file="cb_table_moon_Lan.tex",
    )
    create_overleaf_table_cb(
        benchmark_cb,
        "moon",
        "Md",
        "results/conic_bundle/",
        name_file="cb_table_moon_Md_hey.tex",
    )
    create_overleaf_table_cb(
        benchmark_cb,
        "moon",
        "Mzbar",
        "results/conic_bundle/",
        name_file="cb_table_moon_Mzbar.tex",
    )


if __name__ == "__main__":

    data_modeles = ["mnist", "moon"]
    apply(data_modeles)
