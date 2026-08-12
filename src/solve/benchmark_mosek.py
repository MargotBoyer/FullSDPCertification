# from fastsdp_tools import (
#     FullCertificationConfig,
#     create_folder_benchmark,
#     create_subfolder_benchmark,
#     get_project_path,
# )

# import yaml
# from networks import ReLUNN
# from fastsdp_tools import get_project_path
# import datetime
# import os
# import solve
# from pydantic import ValidationError
# from typing import List

# import pandas as pd
# from pandas.api.types import CategoricalDtype





# # **************************************************************************************************************
# # **************************************************************************************************************
# # **************************************************************************************************************
# # **************************************************************************************************************
# def get_values_from_yaml(yaml_file):
#     """
#     Catch values for the certification problem fro the yaml file
#     """
#     with open(yaml_file, "r") as f:
#         raw_config = yaml.safe_load(f)
#     try:
#         validated_config = FullCertificationConfig(**raw_config)
#     except ValidationError as e:
#         print(f"Erreur de validation du fichier YAML :\n{e}")
#         print("raw config:", raw_config)
#         raise

#     instance = dict(
#         data_modele=validated_config.data.name,
#         network=ReLUNN.from_yaml(yaml_file),
#         epsilon=validated_config.certification_problem.epsilon,
#         x=validated_config.data.x,
#         ytrue=validated_config.data.y,
#         ytarget=validated_config.data.ytarget,
#         bounds_method=validated_config.data.bounds_method,
#         L=validated_config.data.L,
#         U=validated_config.data.U,
#         cuts=validated_config.certification_problem.cuts,
#         all_combinations_cuts=validated_config.certification_problem.all_combinations_cuts,
#         RLT_prop=validated_config.certification_problem.RLT_prop,
#     )

#     data = [(instance["x"], instance["ytrue"])]
#     certification_models = validated_config.certification_problem.models
#     data_modele = validated_config.data.name
#     epsilon = validated_config.certification_problem.epsilon

#     return data, certification_models, data_modele, epsilon, instance




# def run_benchmark_sdp(
#     folder_name: str,
#     data_modeles: List[str] = ["mnist", "moon"],
#     return_values: bool = False,
# ):
#     print("\n \n folder name dans sdp : ", folder_name)
#     instances_values = {}
#     benchmark_mosek = pd.DataFrame()

#     for data_modele in data_modeles:
#         yaml_file = get_project_path(f"config/{data_modele}_one_data_benchmark.yaml")
#         data, certification_models, data_modele, epsilon, instance = (
#             get_values_from_yaml(yaml_file)
#         )

#         for model_name in certification_models:
#             print("\n \n \n \n \n \n \n \n ")
#             # try:
#             # Récupère la classe par son nom depuis le module
#             for CHORDAL_DECOMPOSITION in [True, False]:
#                 model_class = getattr(solve, model_name)
#                 model_instance = model_class.from_yaml(
#                     yaml_file,
#                     CHORDAL_DECOMPOSITION=CHORDAL_DECOMPOSITION,
#                     folder_name=os.path.join(folder_name, data_modele),
#                 )
#                 model_instance.solve()

#                 if "SDP" in model_name:
#                     # print(" \n \n Model instance values : ", model_instance.current_matrices_variables)
#                     instances_values.update(
#                         {
#                             model_name: model_instance.handler.indexes_matrixes.current_matrices_variables
#                         }
#                     )
#                     print(
#                         " \n \n Model benchmark values : ",
#                         model_instance.benchmark_dataframe,
#                     )
#                     benchmark_mosek = concat_dataframes_with_missing_columns(
#                         benchmark_mosek, model_instance.benchmark_dataframe
#                     )

#             # except AttributeError:
#             #     print(f"Modèle non trouvé: {model_name}")
#     try:
#         benchmark_mosek = replace_none_with_false(benchmark_mosek, "Tij")
#     except:
#         pass
#     try:
#         benchmark_mosek = replace_none_with_false(benchmark_mosek, "RLT")
#     except:
#         pass
#     try:
#         benchmark_mosek = replace_none_with_false(benchmark_mosek, "triangularization")
#     except:
#         pass

#     try:
#         benchmark_mosek = replace_none_with_false(benchmark_mosek, "allMC")
#     except:
#         pass

#     if return_values:
#         return benchmark_mosek, instances_values
#     benchmark_mosek.to_csv(
#         get_project_path(os.path.join(folder_name, "results.csv")),
#         index=False,
#     )




# def create_overleaf_table_mosek_md_mzbar(
#     benchmark_mosek: pd.DataFrame,
#     data_modele: str,
#     certification_problem: str,
#     folder_name: str,
#     name_file: str = "mosek_table.tex",
# ):
#     """
#     Crée un tableau pour Overleaf à partir du fichier CSV.
#     """
#     assert certification_problem != "Targeted"

#     ordre_cuts = [
#         "$\\emptyset$",
#         "Tij",
#         "tri",
#         "RLT",
#         "allMC",
#         "Tij, tri",
#         "Tij, RLT",
#         "tri, RLT",
#         "Tij, allMC",
#         "tri, allMC",
#         "RLT, allMC",
#         "Tij, tri, RLT",
#         "Tij, tri, allMC",
#         "Tij, RLT, allMC",
#         "tri, RLT, allMC",
#         "Tij, tri, RLT, allMC",
#     ]
#     cut_type = CategoricalDtype(categories=ordre_cuts, ordered=True)

#     df_data_modele_certification_problem = benchmark_mosek[
#         (benchmark_mosek["dataset"] == data_modele)
#         & (benchmark_mosek["model"] == certification_problem + "SDP")
#     ].copy()
#     df_data_modele_certification_problem["cuts"] = (
#         df_data_modele_certification_problem.apply(lambda row: check_cuts(row), axis=1)
#     )
#     df_data_modele_certification_problem["cuts"] = df_data_modele_certification_problem[
#         "cuts"
#     ].astype(cut_type)

#     df_data_modele_certification_problem = (
#         df_data_modele_certification_problem.sort_values(by="cuts", ascending=True)
#     )

#     with open(
#         get_project_path(f"{folder_name}/{name_file}"),
#         "a",
#         encoding="utf-8",
#     ) as f:

#         f.write("\\begin{tabular}{|lllccc|}\n")
#         f.write("\\toprule\n")
#         f.write(
#             "\\multicolumn{4}{c|}{\\textbf{\Large {"
#             + data_modele
#             + "} - \\Large "
#             + certification_problem
#             + "}} \\\\\n"
#         )
#         f.write("\\hline\n")
#         f.write("Cuts & Div & Tps & Val \\\\\n")
#         f.write("\\hline\n")

#         for index, row in df_data_modele_certification_problem.iterrows():
#             cuts_str = df_data_modele_certification_problem.at[index, "cuts"]
#             div = "Y" if row["CHORDAL_DECOMPOSITION"] else "$\\emptyset$"
#             opt = row["optimal_value"]
#             f.write(
#                 f"{cuts_str} & {div} & {round(row['time'])} & {round(row['optimal_value'],3)} \\\\\n"
#             )

#         f.write("\\hline\n")

#         f.write("\\end{tabular}\n")
#         f.write("\\caption{Direct Mosek performances}\n")


# def create_overleaf_table_mosek_lan(
#     benchmark_mosek: pd.DataFrame,
#     data_modele: str,
#     certification_problem: str,
#     folder_name: str,
#     name_file: str = "mosek_table.tex",
#     target: int = None,
# ):
#     """
#     Crée un tableau pour Overleaf à partir du fichier CSV.
#     """
#     assert certification_problem == "Targeted"
#     ordre_cuts = [
#         "$\\emptyset$",
#         "tri",
#         "RLT",
#         "allMC",
#         "tri, RLT",
#         "tri, allMC",
#         "RLT, allMC",
#     ]
#     cut_type = CategoricalDtype(categories=ordre_cuts, ordered=True)

#     df_data_modele_certification_problem = benchmark_mosek[
#         (benchmark_mosek["dataset"] == data_modele)
#         & (benchmark_mosek["model"] == certification_problem + "SDP")
#     ].copy()
#     df_data_modele_certification_problem["cuts"] = (
#         df_data_modele_certification_problem.apply(lambda row: check_cuts(row), axis=1)
#     )
#     df_data_modele_certification_problem["cuts"] = df_data_modele_certification_problem[
#         "cuts"
#     ].astype(cut_type)

#     df_data_modele_certification_problem = (
#         df_data_modele_certification_problem.sort_values(
#             by=["target", "cuts"], ascending=True
#         )
#     )

#     if target is not None:
#         targets = [target]
#     else:
#         targets = df_data_modele_certification_problem["target"].unique()
#     with open(
#         get_project_path(f"{folder_name}/{name_file}"),
#         "a",
#         encoding="utf-8",
#     ) as f:

#         for target in targets:
#             df_data_modele_certification_problem_target = (
#                 df_data_modele_certification_problem[
#                     df_data_modele_certification_problem["target"] == target
#                 ]
#             )
#             f.write("\\begin{tabular}{l|lllccc|}\n")
#             f.write("\\toprule\n")
#             f.write(
#                 "\\multicolumn{5}{c|}{\\textbf{\Large {"
#                 + data_modele
#                 + "} - \\Large "
#                 + certification_problem
#                 + "}} \\\\\n"
#             )
#             f.write("\\hline\n")
#             f.write("target & Cuts & Div & Tps & Val \\\\\n")
#             f.write("\\hline\n")

#             for index, row in df_data_modele_certification_problem_target.iterrows():
#                 cuts_str = df_data_modele_certification_problem.at[index, "cuts"]
#                 div = "Y" if row["CHORDAL_DECOMPOSITION"] else "$\\emptyset$"

#                 if certification_problem == "Targeted":
#                     f.write(
#                         f"{int(row['target'])} & {cuts_str} & {div} & {round(row['time'])} & {round(row['optimal_value'],3)} \\\\\n"
#                     )
#                 else:
#                     f.write(
#                         f"{cuts_str} & {div} & {round(row['time'])} & {round(row['optimal_value'],3)} \\\\\n"
#                     )

#             f.write("\\hline\n")

#             f.write("\\end{tabular}\n")
#             f.write("\\caption{Direct Mosek performances}\n")


# def create_overleaf_table_mosek(
#     benchmark_mosek: pd.DataFrame,
#     data_modele: str,
#     certification_problem: str,
#     folder_name: str,
#     name_file: str = "mosek_table.tex",
#     target: int = None,
# ):
#     if certification_problem == "Targeted":
#         create_overleaf_table_mosek_lan(
#             benchmark_mosek,
#             data_modele,
#             certification_problem,
#             folder_name,
#             name_file=name_file,
#             target=target,
#         )
#     else:
#         create_overleaf_table_mosek_md_mzbar(
#             benchmark_mosek,
#             data_modele,
#             certification_problem,
#             folder_name,
#             name_file=name_file,
#         )


# def apply():
#     folder_name_ = "{:%d_%b_%y_%Hh_%M}".format(datetime.datetime.now())

#     folder_name = f"results/benchmark/{folder_name_}"

#     benchmark_mosek = run_benchmark_sdp(
#         folder_name=folder_name,
#         data_modeles=["moon", "mnist"],
#         return_values=True,
#     )
#     benchmark_mosek_dataframe = benchmark_mosek[0]
#     benchmark_mosek_dataframe = add_cuts_to_dataframe(benchmark_mosek_dataframe)
#     benchmark_mosek_dataframe.to_csv(
#         get_project_path(os.path.join(folder_name, "results.csv")),
#         index=False,
#     )
#     print("benchmark_mosek : ", benchmark_mosek)
#     print("benchmark_mosek 0 : ", benchmark_mosek[0])
#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "mnist",
#         "Md",
#         folder_name,
#         name_file="mosek_table_mnist_md.tex",
#     )
#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "mnist",
#         "Targeted",
#         folder_name,
#         name_file="mosek_table_mnist_lan.tex",
#     )
#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "mnist",
#         "Mzbar",
#         folder_name,
#         name_file="mosek_table_mnist_mzbar.tex",
#     )

#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "moon",
#         "Md",
#         folder_name,
#         name_file="mosek_table_moon_md.tex",
#     )
#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "moon",
#         "Targeted",
#         folder_name,
#         name_file="mosek_table_moon_lan.tex",
#     )
#     create_overleaf_table_mosek(
#         benchmark_mosek_dataframe,
#         "moon",
#         "Mzbar",
#         folder_name,
#         name_file="mosek_table_moon_mzbar.tex",
#     )


# if __name__ == "__main__":
#     apply()
