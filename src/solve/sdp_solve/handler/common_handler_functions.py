from fastsdp_tools import (
    get_m_indexes_of_higher_values_in_list,
    get_project_path, 
    create_folder, 
    add_row_from_dict
)
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
from solve.sdp_solve.run_benchmark import (
    compute_cuts_str,
    print_dual_variable_to_file_for_cb_solver,
    print_solution_to_file_for_cb_solver
)


logger_mosek = logging.getLogger("Mosek_logger")


def initialize_variables(self):
    """
    Add variables to the task.

    Itère sur self.indexes_matrices.layer_groups pour créer une matrice SDP
    par groupe. Le dernier groupe inclut les variables beta/zbar si BETAS_Z=True.
    """
    logger_mosek.info("Initializing variables...")
    print("Initializing variables...")

    layer_groups = self.indexes_matrices.layer_groups
    indexes = self.indexes_matrices  # objet fusionné (alias aussi indexes_variables)

    def group_name(group):
        return "_".join(map(str, group))

    def plain_dim(group):
        """Dimension d'une matrice z pure pour un groupe donné."""
        return 1 + sum(indexes._n_vars_in_layer(l) for l in group)

    if self.BETAS_Z:
        logger_mosek.info("Model with betaz variables")

        # Tous les groupes sauf le dernier : matrices z pures
        for group in layer_groups[:-1]:
            self.add_matrix_variable(
                name=f"z_layers_{group_name(group)}",
                dim=plain_dim(group),
            )

        # Dernier groupe : z + zbar (optionnel) + betas
        last_group = layer_groups[-1]
        base_dim = indexes._offset_end_of_last_group()
        n_betas = len(indexes.ytargets)

        if self.ZBAR:
            logger_mosek.info("Model with zbar")
            self.add_matrix_variable(
                name=f"z_layers_{group_name(last_group)}_zbar_betas",
                dim=base_dim + 1 + n_betas,
            )
        else:
            logger_mosek.info("Model without zbar")
            self.add_matrix_variable(
                name=f"z_layers_{group_name(last_group)}_betas",
                dim=base_dim + n_betas,
            )

    else:
        # Tous les groupes : matrices z pures
        for group in layer_groups:
            self.add_matrix_variable(
                name=f"z_layers_{group_name(group)}",
                dim=plain_dim(group),
            )

        if self.BETAS:
            self.add_vector_variable(name="betas", dim=self.n[self.K])
    
    assert len(self.indexes_matrices.current_matrices_variables) == self.indexes_matrices.nb_matrices, (                                                                                    
        f"Nombre de matrices incohérent : {len(self.indexes_matrices.current_matrices_variables)} créées "                                                                                  
        f"mais {self.indexes_matrices.nb_matrices} attendues "                                                                                                                              
        f"(BETAS_Z={self.BETAS_Z}, MATRIX_BY_LAYERS={self.MATRIX_BY_LAYERS}, "
        f"LAST_LAYER={self.LAST_LAYER}, BETAS={self.BETAS}, K={self.K})"                                                                                                                    
    ) 


def print_index_variables_matrices(self):
    """
    String representation of the class.
    """
    line = ""

    for layer in range(self.K + 1 if self.LAST_LAYER else self.K):
        line += f"\n Layer {layer} : \n"

        for j in range(self.n[layer]):
            line += f"      Neuron {j} : \n"

            if (layer, j) in self.stable_inactives_neurons:
                line += "           is inactive \n"
                continue
            if (layer < self.K - 1 and not self.LAST_LAYER) or (
                layer < self.K and self.LAST_LAYER
            ):

                ind_matrix_front = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )

                ind_col_front = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=True
                )
                line += f"          front : index = {ind_matrix_front}, i = {ind_col_front} \n"
            if layer > 0:
                ind_col_back = self.indexes_variables._get_variable_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                ind_matrix_back = self.indexes_matrices._get_matrix_index(
                    "z", layer=layer, neuron=j, front_of_matrix=False
                )
                line += f"          back  : index = {ind_matrix_back}, i = {ind_col_back} \n"

    if self.ZBAR:
        line += "\n  Zbar : \n"
        ind_matrix = self.indexes_matrices._get_matrix_index("zbar")
        ind_col = self.indexes_variables._get_variable_index("zbar")
        line += f"          index = {ind_matrix}, i = {ind_col} \n"

    if self.BETAS:
        line += "\n  Betas : \n"
        for class_label in self.ytargets:
            if class_label == self.ytrue:
                continue
            line += f"      Class {class_label} : \n"
            ind_matrix = self.indexes_matrices._get_matrix_index(
                "beta", class_label=class_label
            )
            ind_col = self.indexes_variables._get_variable_index(
                "beta", class_label=class_label
            )
            line += f"          index = {ind_matrix}, i = {ind_col} \n"

    print(line)
    return line


def num_matrices_variables(self):
    """
    Return the number of matrices.
    """
    return len(self.indexes_matrices.current_matrices_variables)

def print_num_variables(self):
    num_variables = 0
    for i in range(self.num_matrices_variables()):
        dim = self.indexes_matrices.current_matrices_variables[i]["dim"]
        num_variables += (dim+1) * (dim+1)
    print(f"CALLBACK num variables : {num_variables}")
    return num_variables


def save_matrix_png(self, mat, name_solution, cuts: List):
    """
    Save a 2D matrix as an image with values rounded to two decimal places.
    """
    cuts_str = compute_cuts_str(cuts)
    max_width = 200

    mat = np.array(mat)
    n_rows, n_cols = mat.shape
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if n_rows < max_width:

        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(self.name, fontsize=14, pad=20)

        table_data = [
            [f"{mat[i, j]:.2f}" for j in range(n_cols)] for i in range(n_rows)
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(n_cols)))

        plt.savefig(
            os.path.join(
                model_dir,
                f"{name_solution}_{cuts_str}_solution.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def save_matrix_csv(self, mat, name_solution, cuts: List):
    """
    Save a the solution matrix as a CSV file with values rounded to five decimal places.
    """

    cuts_str = compute_cuts_str(cuts)

    model_dir = get_project_path(f"{self.folder_name}/{self.name}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    mat = pd.DataFrame(np.round(mat, decimals=5))

    mat.to_csv(
        os.path.join(model_dir, f"{name_solution}_{cuts_str}.csv"),
        index=False,
    )
    plt.close()





class Matrices_Solutions:
    def __init__(self):
        self._data = {}

    def add_value(self, cuts, value):
        """
        Ajoute une value à une configuration existante

        Args:
            cuts: Liste ou ensemble des coupes actives
            value: value à ajouter
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(
                f"Ajout de la configuration {key} non presente avec la value de dim {value.shape}"
            )
            self._data[key] = value
            print("Configuration ajoutée.")

        else:
            raise ValueError(
                f"Configuration {key} déjà présente avec la value de dim {self._data[key].shape}"
            )

        return self

    def configurations_disponibles(self):
        """Retourne toutes les configurations de coupes enregistrées"""
        return [set(config) for config in self._data.keys()]

    def __getitem__(self, cuts):
        """Permet d'utiliser l'opérateur [] pour accéder aux configurations"""
        """
        Récupère les values pour une combinaison de coupes actives

        Args:
            cuts: Liste ou ensemble des coupes actives

        Returns:
            Liste de values associées ou liste vide si combinaison non trouvée
        """
        key = frozenset(cuts)
        if key not in self._data:
            print(f"Configuration {key} non trouvée")
            raise ValueError(
                f"Configuration {key} non trouvée dans les configurations disponibles"
            )
        return self._data.get(key)

    def __contains__(self, cuts):
        """Permet d'utiliser l'opérateur 'in' pour vérifier si une configuration existe"""
        return frozenset(cuts) in self._data


def get_matrices_variables(self, cuts: List):
    """
    Get the matrices variables of the optimization problem.
    """
    if self.current_matrices_variables is None:
        raise ValueError(
            "No matrices variables found. Please initialize the variables first."
        )
    matrices = []
    for ind_solution in range(len(self.current_matrices_variables)):
        name_solution = self.current_matrices_variables[ind_solution]["name"]
        mat = self.current_matrices_variables[ind_solution]["value"][cuts]
        matrices.append(mat)
    return matrices


def diagnose_infeasibility(self, gap_threshold: float = 0.01, max_constraints_for_rank: int = 5000):
    """
    Diagnostic de la condition de Slater : appelé quand le gap primal-dual est persistant.

    Étape 1 : valeurs propres de chaque matrice PSD (détecte X ≈ 0 ou rang déficient).
    Étape 2 : rang de la matrice des contraintes A_i (détecte redondances linéaires).
    """
    logger_mosek.warning("=== DIAGNOSTIC INFAISABILITÉ (condition de Slater) ===")

    # --- Étape 1 : spectre de chaque matrice PSD ---
    for ind, mat_info in enumerate(self.indexes_matrices.current_matrices_variables):
        name = mat_info["name"]
        dim  = mat_info["dim"]
        try:
            X = self.get_solution(ind_solution=ind, name_solution=name, dim=dim)
            eigvals = np.linalg.eigvalsh(X)
            effective_rank = int(np.sum(eigvals > 1e-8))
            min_eig = float(eigvals[0])
            max_eig = float(eigvals[-1])
            msg = (
                f"  Matrice '{name}' (dim={dim}) : "
                f"rang_effectif={effective_rank}/{dim}, "
                f"λ_min={min_eig:.3e}, λ_max={max_eig:.3e}"
            )
            logger_mosek.warning(msg)
            print(msg)
            if effective_rank < dim:
                warn = f"    ⚠ Rang déficient ({effective_rank}<{dim}) → X sur la frontière du cône, Slater non satisfaite"
                logger_mosek.warning(warn)
                print(warn)
        except Exception as e:
            logger_mosek.warning(f"  Impossible d'analyser '{name}' : {e}")

    # --- Étape 2 : identification des contraintes redondantes par QR avec pivot ---
    import re
    from scipy.linalg import qr as scipy_qr

    list_cstr = self.Constraints.list_cstr
    n_cstr = len(list_cstr)
    if n_cstr == 0:
        logger_mosek.warning("  Aucune contrainte trouvée.")
        return

    sample = list_cstr[:max_constraints_for_rank]
    truncated = n_cstr > max_constraints_for_rank

    # Construire la matrice A : une ligne par contrainte, une colonne par entrée PSD unique (nm, i, j)
    col_index: dict = {}
    col_counter = 0
    rows = []
    for cstr in sample:
        row: dict = {}
        for k in range(len(cstr["num_matrix"])):
            nm, ci, cj, cv = cstr["num_matrix"][k], cstr["i"][k], cstr["j"][k], cstr["value"][k]
            key = (nm, min(ci, cj), max(ci, cj))
            if key not in col_index:
                col_index[key] = col_counter
                col_counter += 1
            col = col_index[key]
            row[col] = row.get(col, 0.0) + cv
        rows.append(row)

    n_rows, n_cols = len(rows), col_counter
    if n_cols == 0 or n_rows == 0:
        logger_mosek.warning("  Matrice des contraintes vide — impossible de calculer le rang.")
        return

    # Garde mémoire : matrice dense n_rows × n_cols peut dépasser plusieurs GB pour grandes matrices PSD.
    # Seuil : 2 GB (float64 = 8 octets).
    max_dense_elements = 250_000_000  # 2 GB
    if n_rows * n_cols > max_dense_elements:
        skip_msg = (
            f"CALLBACK  Étape 2 ignorée : matrice {n_rows} × {n_cols} trop grande "
            f"({n_rows * n_cols * 8 / 1e9:.1f} GB). Augmenter max_constraints_for_rank "
            f"ou réduire la taille du problème pour activer l'analyse de rang."
        )
        logger_mosek.warning(skip_msg)
        print(skip_msg)
        return

    A = np.zeros((n_rows, n_cols))
    for r, row in enumerate(rows):
        for c, v in row.items():
            A[r, c] = v

    # Normalisation par ligne : élimine les artefacts d'échelle avant le QR
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    A_normalized = A / row_norms

    # QR avec pivotage sur A_normalized^T : identifie les lignes redondantes
    _, R, P = scipy_qr(A_normalized.T, pivoting=True)
    diag = np.abs(np.diag(R))

    # Distribution des valeurs singulières : distingue vraie redondance vs artefact numérique
    if diag[0] > 0:
        diag_rel = diag / diag[0]  # normalisé par la plus grande valeur
        rank_strict  = int(np.sum(diag_rel > 1e-8))   # seuil strict  → redondance certaine
        rank_loose   = int(np.sum(diag_rel > 1e-4))   # seuil lâche   → redondance probable
        # Percentiles pour visualiser la distribution
        pct = np.percentile(diag_rel, [25, 50, 75, 90, 99])
        dist_msg = (
            f"CALLBACK  Distribution diag(R)/max : "
            f"p25={pct[0]:.2e}  p50={pct[1]:.2e}  p75={pct[2]:.2e}  "
            f"p90={pct[3]:.2e}  p99={pct[4]:.2e}"
        )
        logger_mosek.warning(dist_msg)
        print(dist_msg)
        gap_msg = (
            f"CALLBACK  Rang (seuil 1e-8) = {rank_strict}  |  "
            f"Rang (seuil 1e-4) = {rank_loose}  "
            f"→ {'saut net = vraie redondance' if rank_strict != rank_loose else 'décroissance graduelle = probablement numérique'}"
        )
        logger_mosek.warning(gap_msg)
        print(gap_msg)
        rank = rank_strict
    else:
        rank = 0

    label = " (tronqué)" if truncated else ""
    msg = f"CALLBACK  Matrice des contraintes : {n_rows}{label} × {n_cols} → rang={rank}"
    logger_mosek.warning(msg)
    print(msg)

    if rank < n_rows:
        redundant_indices = P[rank:]

        def cstr_type(name: str) -> str:
            match = re.match(r'^(.*?)(?:_\d|$)', name)
            return match.group(1) if match else name

        counts: dict = {}
        for idx in redundant_indices:
            t = cstr_type(sample[idx]["name"])
            counts[t] = counts.get(t, 0) + 1

        warn = f"CALLBACK  ⚠ {len(redundant_indices)} contrainte(s) redondante(s) par type :"
        logger_mosek.warning(warn)
        print(warn)
        for t, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            line = f"CALLBACK      {cnt:4d}  {t}"
            logger_mosek.warning(line)
            print(line)

    logger_mosek.warning("=== FIN DIAGNOSTIC ===")


def compute_solutions(self, cuts: List, print_sol: bool = False):
    """
    Get the solutions and dual variables of the optimization problem.
    """
    cuts_str = compute_cuts_str(cuts)
    if print_sol:
        file_cb = open(
            get_project_path(f"{self.folder_name}/{self.name}/results_{cuts_str}.txt"),
            "w",
        )
        file_cb.write("Primal Solutions \n")
    for ind_solution in range(len(self.indexes_matrices.current_matrices_variables)):
        print(f"CALLBACK : Computing solution for matrix {ind_solution} with cuts {cuts_str}...")
        name_solution = self.indexes_matrices.current_matrices_variables[ind_solution][
            "name"
        ]
        dim = self.indexes_matrices.current_matrices_variables[ind_solution]["dim"]

        sol = self.get_solution(
            ind_solution=ind_solution, name_solution=name_solution, dim=dim
        )

        self.indexes_matrices.current_matrices_variables[ind_solution][
            "value"
        ].add_value(cuts, sol)

        # if verbose:
        #     logger_mosek.debug(
        #         f"Solution for {name_solution} of dimension {dim}: {mat}"
        #     )
        print(f"CALLBACK : Solution for {name_solution} of dimension {dim} computed.")
        #self.save_matrix_png(sol, name_solution=name_solution, cuts=cuts)
        #self.save_matrix_csv(sol, name_solution=name_solution, cuts=cuts)
        print(f"CALLBACK : Solution for {name_solution} of dimension {dim} saved as PNG and CSV.")

        if print_sol:
            print_solution_to_file_for_cb_solver(
                sol,
                index_matrix=ind_solution,
                dim=dim,
                file_cb=file_cb,
            )

    if print_sol:
        file_cb.write("Dual Solutions \n")
        self.get_dual_variables()
        print_dual_variable_to_file_for_cb_solver(
            list_cstr=self.Constraints.list_cstr, file_cb=file_cb
        )
        file_cb.close()

    if self.BETAS and self.indexes_matrices.BETAS_Z:
        save_beta_values(self, cuts)


def save_beta_values(self, cuts: List):
    """
    Extrait les valeurs β_j de la matrice solution et les sauvegarde dans betas_{cuts_str}.txt.

    La valeur de β_j est X[0, index_variable_beta(j)] dans la dernière matrice PSD
    (ligne 0 = variable constante 1, donc X[0, i] = β_j · 1 = β_j).
    """
    cuts_str = compute_cuts_str(cuts)
    model_dir = get_project_path(f"{self.folder_name}/{self.name}")
    os.makedirs(model_dir, exist_ok=True)

    # Trouver la matrice qui contient les betas (la dernière, dont le nom contient "betas")
    beta_mat_info = next(
        (m for m in self.indexes_matrices.current_matrices_variables if "betas" in m["name"]),
        None,
    )
    if beta_mat_info is None:
        logger_mosek.warning("save_beta_values : aucune matrice avec betas trouvée.")
        return

    ind = self.indexes_matrices.current_matrices_variables.index(beta_mat_info)
    X = beta_mat_info["value"][cuts]  # matrice déjà calculée par compute_solutions

    lines = []
    for class_label in self.indexes_matrices.ytargets:
        if class_label == self.indexes_matrices.ytrue:
            continue
        try:
            idx = self.indexes_matrices.index_variable_beta(class_label)
            beta_val = float(X[0, idx])
            lines.append(f"beta_{class_label} = {beta_val:.8f}")
        except Exception as e:
            lines.append(f"beta_{class_label} = ERROR ({e})")

    out_path = os.path.join(model_dir, f"betas_{cuts_str}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"CALLBACK : Valeurs beta sauvegardées dans {out_path}")
