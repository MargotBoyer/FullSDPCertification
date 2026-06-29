from typing import List, Union

from .variable_elements import big_M_cst


def build_layer_groups_with_stride(K: int, s: int = 1, LAST_LAYER: bool = False, INPUT_IN_VARIABLES: bool = True) -> List[List[int]]:
    """Construit des cliques de taille s+1 avec overlap d'une couche."""
    start = 0 if INPUT_IN_VARIABLES else 1
    last = K if LAST_LAYER else K - 1
    groups = []
    k = start
    while k < last:
        group = list(range(k, min(k + s + 1, last + 1)))
        groups.append(group)
        k += s  # overlap d'une couche entre groupes
    return groups


def resolve_layer_groups(
    MATRIX_BY_LAYERS: Union[bool, List[List[int]]],
    K: int,
    LAST_LAYER: bool = False,
    INPUT_IN_VARIABLES: bool = True,
    chordal_stride: int = 1,
) -> List[List[int]]:
    """
    Résout MATRIX_BY_LAYERS en liste canonique de groupes de couches.

    Exemples (INPUT_IN_VARIABLES=True) :
      True,  K=5 → [[0,1],[1,2],[2,3],[3,4]]
      False, K=5 → [[0,1,2,3,4]]
      [[0,1],[1,2,3,4],[4,5]] → tel quel (avec K=5, LAST_LAYER=True)

    Exemples (INPUT_IN_VARIABLES=False) :
      True,  K=5 → [[1,2],[2,3],[3,4]]
      False, K=5 → [[1,2,3,4]]
      Les groupes explicites doivent commencer à 1 et non 0.
    """
    print("LAST_LAYER in resolve_layer_groups : ", LAST_LAYER)
    print("K : ", K)
    last = K if LAST_LAYER else K - 1
    start = 0 if INPUT_IN_VARIABLES else 1
    print("MATRIX_BY_LAYERS  in resolve layer groups: ", MATRIX_BY_LAYERS)
    print("INPUT_IN_VARIABLES in resolve_layer_groups : ", INPUT_IN_VARIABLES)

    if isinstance(MATRIX_BY_LAYERS, bool):
        if MATRIX_BY_LAYERS:
            return build_layer_groups_with_stride(K, chordal_stride, LAST_LAYER, INPUT_IN_VARIABLES)
        else:
            return [list(range(start, last + 1))]
    else:
        assert MATRIX_BY_LAYERS[0][0] == start, (
            f"First group must start at layer {start} "
            f"(INPUT_IN_VARIABLES={INPUT_IN_VARIABLES})"
        )
        print("MATRIX_LAYERS[-1][-1] : ", MATRIX_BY_LAYERS[-1][-1])
        print("last : ", last)
        assert MATRIX_BY_LAYERS[-1][-1] == last, (
            f"Last group must end at layer {last}, got {MATRIX_BY_LAYERS[-1][-1]}"
        )
        for i in range(len(MATRIX_BY_LAYERS) - 1):
            assert MATRIX_BY_LAYERS[i][-1] == MATRIX_BY_LAYERS[i + 1][0], (
                f"Groups {MATRIX_BY_LAYERS[i]} and {MATRIX_BY_LAYERS[i + 1]} "
                f"must share exactly one boundary layer"
            )
        return MATRIX_BY_LAYERS


class Indexes_Mosek_Solver:
    """
    Classe unifiée gérant l'indexation des matrices SDP et des variables pour MOSEK.

    Regroupe l'ancienne Indexes_Matrixes_for_Mosek_Solver et
    Indexes_Variables_for_Mosek_Solver en un seul objet cohérent autour
    de la notion de layer_groups (décomposition chordale).
    """

    def __init__(
        self,
        K: int,
        n: List[int],
        MATRIX_BY_LAYERS: Union[bool, List[List[int]]] = False,
        LAST_LAYER: bool = False,
        BETAS: bool = False,
        BETAS_Z: bool = False,
        ZBAR: bool = False,
        INPUT_IN_VARIABLES: bool = True,
        chordal_stride: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        K : int
            Nombre de couches du réseau.
        n : List[int]
            Nombre de neurones par couche (longueur K+1).
        MATRIX_BY_LAYERS : Union[bool, List[List[int]]]
            True  → décomposition chordale standard (paires consécutives).
            False → matrice unique.
            List  → groupes explicites, ex. [[0,1],[1,2,3,4],[4,5]].
        LAST_LAYER : bool
            Si True, la dernière couche (logits) est incluse comme variable.
        BETAS : bool
            Active les variables beta (formulation untargeted).
        BETAS_Z : bool
            Si True, les betas sont embarquées dans la dernière matrice z.
        ZBAR : bool
            Active la variable zbar.
        INPUT_IN_VARIABLES : bool
            Si False, z_0 est retiré des variables SDP ; le premier bloc commence
            à la couche 1. La boule L∞ est alors implicitement absorbée dans les
            bornes pré-calculées L_1, U_1.
        **kwargs
            ytrue, ytargets, stable_inactives_neurons, stable_actives_neurons,
            keep_penultimate_actives.
        """
        self.n = n
        self.K = K
        self.MATRIX_BY_LAYERS = MATRIX_BY_LAYERS  # conservé pour référence
        self.LAST_LAYER = LAST_LAYER
        self.BETAS = BETAS
        self.BETAS_Z = BETAS_Z
        self.ZBAR = ZBAR
        self.INPUT_IN_VARIABLES = INPUT_IN_VARIABLES

        self.ytrue = kwargs.get("ytrue")
        self.ytargets = kwargs.get("ytargets")
        self.stable_inactives_neurons = kwargs.get("stable_inactives_neurons")
        self.stable_actives_neurons = kwargs.get("stable_actives_neurons")
        self.keep_penultimate_actives = kwargs.get("keep_penultimate_actives", False)

        self.layer_groups = resolve_layer_groups(MATRIX_BY_LAYERS, K, LAST_LAYER, INPUT_IN_VARIABLES, chordal_stride)
        self._layer_to_groups = self._build_layer_to_groups()
        self._pruned_adv_before = self._build_pruned_adv_before()


        self.check_conformity()

        self.current_matrices_variables = []
        self.count_nb_matrices()
        self.count_max_indexes()

    # ------------------------------------------------------------------
    # Construction interne
    # ------------------------------------------------------------------

    def _build_layer_to_groups(self) -> dict:
        """
        Mapping layer → liste de (group_idx, position_in_group).
        Une couche frontière partagée entre deux groupes apparaît deux fois.
        """
        mapping = {}
        for group_idx, group in enumerate(self.layer_groups):
            for pos, layer in enumerate(group):
                mapping.setdefault(layer, []).append((group_idx, pos))
        return mapping

    def _build_pruned_adv_before(self) -> dict:
        """
        Précalcule, pour chaque classe c, le nombre de classes adversariales prunées
        strictement avant c (i.e. ni ytrue, ni dans ytargets, et < c).
        """
        if self.ytargets is None or self.ytrue is None:
            return {}
        count = 0
        result = {}
        for c in range(self.n[self.K]):
            result[c] = count
            if c != self.ytrue and c not in self.ytargets:
                count += 1
        return result

    # ------------------------------------------------------------------
    # Comptage et vérification
    # ------------------------------------------------------------------

    def count_nb_matrices(self):
        """Nombre de matrices SDP basé sur le nombre de groupes."""
        self.nb_matrices = len(self.layer_groups)
        if not self.BETAS_Z and self.BETAS:
            self.nb_matrices += 1  # matrice dédiée aux betas
        assert self.nb_matrices < big_M_cst, (
            f"nb_matrices={self.nb_matrices} >= big_M_cst={big_M_cst}: "
            f"the hash-key encoding in variable_elements.py would overflow silently. "
            f"Increase big_M_cst in variable_elements.py before proceeding."
        )

    def count_max_indexes(self):
        """
        Borne supérieure conservative sur l'index maximal de variable.
        Aucun pruning n'est soustrait ici (marge +1000 ajoutée).
        """
        max_index = 0
        for group_idx, group in enumerate(self.layer_groups):
            size = 1
            for layer in group:
                if layer == self.K and self.LAST_LAYER:
                    # Couche de sortie : seulement ytargets + ytrue
                    size += len(self.ytargets) + 1
                else:
                    size += self.n[layer]
            if group_idx == len(self.layer_groups) - 1 and self.BETAS_Z:
                if self.ZBAR:
                    size += 1
                size += len(self.ytargets)
            max_index = max(max_index, size)

        if not self.BETAS_Z and self.BETAS:
            max_index = max(max_index, len(self.ytargets) + 1)

        self.max_index = max_index + 1000

    def check_conformity(self):
        """
        Vérifie la cohérence de la structure : K, n, et présence de neurones
        instables dans chaque groupe.
        """
        assert self.K == len(self.n) - 1
        if len(self.layer_groups) > 1:
            for group_idx, group in enumerate(self.layer_groups):
                unstable_count = sum(
                    1
                    for layer in group
                    for j in range(self.n[layer])
                    if (layer, j) not in self.stable_inactives_neurons
                    and (layer, j) not in self.stable_actives_neurons
                )
                print(
                    f"Taille de la matrice pour le groupe {group_idx} {group} : "
                    f"{unstable_count}"
                )
                assert unstable_count > 0, (
                    f"Group {group} has no unstable neurons — a special treatment is needed."
                )

        if any(
            all(
                (layer, j) in self.stable_inactives_neurons
                for j in range(self.n[layer])
            )
            for layer in range(1, self.K)
        ):
            raise ValueError(
                "There are layers with only inactive neurons : "
                "a special treatment is needed : the output is fixed."
            )

    # ------------------------------------------------------------------
    # Helpers pruning
    # ------------------------------------------------------------------

    def get_number_pruned_neurons_on_layer(self, layer: int, neuron: int = None) -> int:
        """
        Nombre de neurones prunés sur une couche jusqu'au neurone `neuron`.
        Si neuron=None, compte tous les neurones prunés de la couche.
        """
        if (
            self.stable_inactives_neurons is None
            and self.stable_actives_neurons is None
        ):
            return 0

        if neuron is None:
            neuron = self.n[layer]

        if (layer == self.K - 1) and self.keep_penultimate_actives:
            return len(
                [
                    (k, j)
                    for k, j in self.stable_inactives_neurons
                    if k == layer and j < neuron
                ]
            )
        else:
            return len(
                [
                    (k, j)
                    for k, j in (
                        self.stable_inactives_neurons + self.stable_actives_neurons
                    )
                    if k == layer and j < neuron
                ]
            )

    def get_number_pruned_neurons_before_layer(
        self, layer: int, neuron: int = None
    ) -> int:
        """
        Neurones prunés strictement avant `layer`, plus ceux sur `layer` jusqu'à `neuron`.
        """
        if self.stable_inactives_neurons is None:
            return 0
        if neuron is None:
            neuron = self.n[layer]
        return (
            sum(self.get_number_pruned_neurons_on_layer(k) for k in range(layer))
            + self.get_number_pruned_neurons_on_layer(layer, neuron)
        )

    def get_number_pruned_adversarial_targets_before_target(self, ytarget) -> int:
        """
        Nombre de classes non-ytarget, non-ytrue strictement avant `ytarget`.
        Utilisé pour ajuster l'index des variables z_K quand LAST_LAYER=True.
        """
        return self._pruned_adv_before[ytarget]

    # ------------------------------------------------------------------
    # Helpers internes pour l'indexation
    # ------------------------------------------------------------------

    def _n_vars_in_layer(self, layer: int) -> int:
        """
        Nombre de variables z pour une couche donnée, après pruning.

        Pour la couche K (LAST_LAYER=True) : ytargets + ytrue sont les seules
        variables, soit len(ytargets) + 1 moins les neurones stables éventuels.
        """
        if layer == self.K and self.LAST_LAYER:
            return (
                len(self.ytargets)
                + 1  # +1 pour ytrue
                - self.get_number_pruned_neurons_on_layer(layer)
            )
        return self.n[layer] - self.get_number_pruned_neurons_on_layer(layer)

    def _offset_end_of_last_group(self) -> int:
        """
        1 + nombre de variables z dans le dernier groupe.
        Sert de base pour positionner beta et zbar dans la matrice.
        """
        return 1 + sum(self._n_vars_in_layer(l) for l in self.layer_groups[-1])

    # ------------------------------------------------------------------
    # Indexation des matrices
    # ------------------------------------------------------------------

    def is_in_matrix_with_betas(self, layer: int) -> bool:
        """True si `layer` appartient au dernier groupe (qui contient les betas quand BETAS_Z=True)."""
        if self.BETAS_Z:
            return layer in self.layer_groups[-1]
        return False

    def index_matrix_z(self, layer: int, front_of_matrix: bool) -> int:
        """
        Index de la matrice SDP pour la variable z_{layer}.

        front_of_matrix=True  → cherche un groupe où layer n'est pas en dernière position.
        front_of_matrix=False → cherche un groupe où layer est en dernière position.
        """
        if layer not in self._layer_to_groups:
            raise ValueError(f"Layer {layer} not found in any group.")

        entries = self._layer_to_groups[layer]

        if front_of_matrix:
            for group_idx, pos_in_group in entries:
                if pos_in_group < len(self.layer_groups[group_idx]) - 1:
                    return group_idx
            raise ValueError(
                f"Layer {layer} cannot be front_of_matrix=True: "
                f"it is always the last element of its group(s)."
            )
        else:
            for group_idx, pos_in_group in entries:
                if pos_in_group == len(self.layer_groups[group_idx]) - 1:
                    return group_idx
            raise ValueError(
                f"Layer {layer} cannot be front_of_matrix=False: "
                f"it is never the last element of any group."
            )

    def index_matrix_beta(self) -> int:
        """Index de la matrice contenant les variables beta."""
        assert self.BETAS
        if self.BETAS_Z:
            return len(self.layer_groups) - 1  # dernier groupe z
        else:
            return len(self.layer_groups)  # matrice dédiée après les z

    def index_matrix_zbar(self) -> int:
        """Index de la matrice contenant la variable zbar."""
        assert self.ZBAR
        assert self.BETAS_Z
        return len(self.layer_groups) - 1  # même matrice que les betas

    def _get_matrix_index(self, var_type: str, is_first: bool = None, **kwargs) -> int:
        """
        Dispatcher : retourne l'index de matrice pour une variable de type donné.

        Parameters
        ----------
        var_type : str
            'z', 'beta' ou 'zbar'.
        is_first : bool or None
            None pour les variables linéaires, True/False pour les paires quadratiques.
        """
        suffix = "" if is_first is None else ("1" if is_first else "2")
        front_of_matrix = kwargs.get(f"front_of_matrix{suffix}", None)

        if var_type == "z":
            layer_key = f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            layer = kwargs.get(layer_key)
            if layer is None:
                raise ValueError(
                    f"Layer required for z variable ({'first' if is_first else 'second'})"
                )
            if front_of_matrix is None:
                last_layer = self.layer_groups[-1][-1]
                front_of_matrix = (layer != last_layer)
            return self.index_matrix_z(layer, front_of_matrix)

        elif var_type == "beta":
            class_label_key = (
                f"class_label{suffix}" if f"class_label{suffix}" in kwargs else "class_label"
            )
            if kwargs.get(class_label_key) is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_matrix_beta()

        elif var_type == "zbar":
            return self.index_matrix_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def get_shape_matrix(self, num_matrix: int):
        """Dimension de la matrice num_matrix."""
        return self.current_matrices_variables[num_matrix]["dim"]

    def get_name_matrix(self, num_matrix: int):
        """Nom de la matrice num_matrix."""
        return self.current_matrices_variables[num_matrix]["name"]

    # ------------------------------------------------------------------
    # Indexation des variables
    # ------------------------------------------------------------------

    def index_variable_z(self, layer: int, neuron: int, front_of_matrix: bool) -> int:
        """
        Index de z_{layer, neuron} dans sa matrice SDP.

        La matrice est identifiée par front_of_matrix (cf. index_matrix_z).
        La position dans la matrice [1, z_{g_0}, z_{g_1}, ..., z_{g_m}] est :
          1 + Σ n_vars_in_layer(g_p) pour p < pos + offset neuron dans la couche.
        """
        if (layer == self.K and not self.LAST_LAYER) or layer < 0 or layer > self.K:
            raise ValueError(f"Layer index {layer} out of range.")
        if (layer, neuron) in self.stable_inactives_neurons:
            raise ValueError(
                f"Neuron {neuron} in layer {layer} is inactive and has no z variable."
            )

        matrix_idx = self.index_matrix_z(layer, front_of_matrix)
        group = self.layer_groups[matrix_idx]
        pos = group.index(layer)

        offset = 1 + sum(self._n_vars_in_layer(group[p]) for p in range(pos))

        if layer == self.K and self.LAST_LAYER:
            offset += (
                neuron
                - self.get_number_pruned_adversarial_targets_before_target(neuron)
                - self.get_number_pruned_neurons_on_layer(layer, neuron)
            )
        else:
            offset += neuron - self.get_number_pruned_neurons_on_layer(layer, neuron)

        return offset

    def ind_label_beta(self, class_label: int) -> int:
        """Position 0-based de beta_{class_label} dans la liste des classes cibles actives."""
        if not self.BETAS:                                                                                                                                                                                               
          raise ValueError("Beta variables are not enabled.")                                                                                                                                                          
        if class_label == self.ytrue:                                                                                                                                                                                    
          raise ValueError("The true class label has no beta variable.")                                                                                                                                               
        if class_label not in self.ytargets:                                                                                                                                                                             
          raise ValueError(
              f"Class {class_label} is not an active target (pruned or invalid)."
          ) 
        pruned_before = self.get_number_pruned_adversarial_targets_before_target(class_label)
        if class_label < self.ytrue:
            return class_label - pruned_before
        else:
            return class_label - 1 - pruned_before

    def index_variable_beta(self, class_label: int) -> int:
        """
        Index de beta_{class_label} dans sa matrice.

        BETAS_Z=True  → embarquée dans le dernier groupe z, après zbar si ZBAR.
        BETAS_Z=False → matrice dédiée, position 1-based.
        """
        assert self.BETAS
        if self.BETAS_Z:
            base = self._offset_end_of_last_group()
            if self.ZBAR:
                base += 1
            return base + self.ind_label_beta(class_label)
        else:
            return 1 + self.ind_label_beta(class_label)

    def index_variable_zbar(self) -> int:
        """Index de zbar dans la dernière matrice z (juste après les variables z, avant les betas)."""
        assert self.ZBAR
        assert self.BETAS_Z
        return self._offset_end_of_last_group()

    def _get_variable_index(self, var_type: str, is_first: bool = None, **kwargs) -> int:
        """
        Dispatcher : retourne l'index de variable pour un type donné.

        Parameters
        ----------
        var_type : str
            'z', 'beta' ou 'zbar'.
        is_first : bool or None
            None pour les variables linéaires, True/False pour les paires quadratiques.
        """
        suffix = "" if is_first is None else ("1" if is_first else "2")
        front_of_matrix = kwargs.get(f"front_of_matrix{suffix}", None)

        if var_type == "z":
            layer_key = f"layer{suffix}" if f"layer{suffix}" in kwargs else "layer"
            neuron_key = f"neuron{suffix}" if f"neuron{suffix}" in kwargs else "neuron"
            layer = kwargs.get(layer_key)
            neuron = kwargs.get(neuron_key)
            if layer is None or neuron is None:
                raise ValueError(
                    f"Layer and neuron required for z variable ({'first' if is_first else 'second'})"
                )
            if front_of_matrix is None:
                last_layer = self.layer_groups[-1][-1]
                front_of_matrix = (layer != last_layer)
            return self.index_variable_z(layer, neuron, front_of_matrix)

        elif var_type == "beta":
            class_label_key = (
                f"class_label{suffix}" if f"class_label{suffix}" in kwargs else "class_label"
            )
            class_label = kwargs.get(class_label_key)
            if class_label is None:
                raise ValueError(
                    f"Class label required for beta variable ({'first' if is_first else 'second'})"
                )
            return self.index_variable_beta(class_label)

        elif var_type == "zbar":
            return self.index_variable_zbar()

        else:
            raise ValueError(f"Unknown variable type: {var_type}")
