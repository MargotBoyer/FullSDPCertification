import mosek
import numpy as np

from .certification_problem_constraints_bounds import McCormick_inter_layers
from fastsdp_tools import get_m_indexes_of_higher_values_in_list, infinity
from ..handler.constraints import ConstraintRole
from ..handler.variable_elements import _get_linear_indices_from_key_py


# ********************************* RLT Lan Constraints *********************************
def add_RLT_constraints(self, p: float = 0.5):
    """
    Add the RLT constraints to the task.
    """
    nb_rlt = 0
    
    print("Adding RLT constraint")
    start_k = 2 if not self.INPUT_IN_VARIABLES else 1
    for k in range(start_k, self.K + 1 if self.LAST_LAYER else self.K):
        nb_cstr = int(p * self.n[k - 1])
        
        indexes_pruned = [
            j
            for j in range(self.n[k - 1])
            if (k - 1, j) in self.stable_inactives_neurons
            or (k - 1, j) in self.stable_actives_neurons
            or (k == 1 and j in self.pruned_input_neurons)
        ]
        if k == self.K and self.LAST_LAYER:
            neurons_next = list(set([self.ytrue]).union(self.ytargets))
        else:
            neurons_next = [j for j in range(self.n[k]) 
                            if (k, j) not in self.stable_actives_neurons 
                            and (k, j) not in self.stable_inactives_neurons]
        print("Indexes pruned for layer", k, ":", indexes_pruned)
        study_ablation =  True
        for neuron_next in neurons_next:
            if (k, neuron_next) in self.stable_inactives_neurons:
                
                continue
            if (k, neuron_next) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                
                continue
            neurons_with_great_weights = get_m_indexes_of_higher_values_in_list(
                np.abs(self.W[k - 1][neuron_next]), nb_cstr, indexes_pruned
            )
            # if study_ablation : 
            #     print(f"STUDY ABLATION for layer {k}: number of neurons selected : ", len(neurons_with_great_weights))
            #     study_ablation = False
            for neuron_prev in neurons_with_great_weights:

                assert (k - 1, neuron_prev) not in self.stable_inactives_neurons and (
                    k - 1,
                    neuron_prev,
                ) not in self.stable_actives_neurons
                # print(
                #     "Adding RLT constraint for neuron_prev:",
                #     neuron_prev,
                #     "neuron_next:",
                #     neuron_next,
                # )
                # assert self.U[k - 1][neuron_prev] > 0
                # assert self.U[k][neuron_next] > 0
                self.McCormick_inter_layers(k, neuron_prev, neuron_next)
                nb_rlt+=1
    print(f"CALLBACK : RLT = {nb_rlt}")


# ********************************* RLT Lan Constraints (compacte, conic bundle) ***
def _add_McCormick_compact(self, k: int, neuron_prev: int, neuron_next: int) -> bool:
    """
    Chemin rapide pour les 3 sous-contraintes McCormick (12b, 12c, 12c-bis) de la
    paire RLT (k-1, neuron_prev)-(k, neuron_next). Reutilise exactement les memes
    fonctions de lookup deja validees (verify_variable_z, equivalent_neurons.
    get_equivalent/get_constant) que le chemin classique -- seule la construction
    finale change : les tableaux num_matrix/i/j/value/bound sont ecrits directement
    dans list_cstr, sans passer par new_constraint()/add_quad_variable()/add_var()
    (qui creent un numba.typed.Dict PAR contrainte -- cout fixe disproportionne
    pour des contraintes a 1-3 termes, cf. investigation temps de processing hors
    resolution SDP, task-dynamic-conic-bundle.md).

    Ne s'applique QUE si les deux neurones sont genuinement instables, sans
    substitution (pas de neurone stable actif exprime comme combinaison lineaire --
    cf. CLAUDE.md, "Pruning des neurones stables actifs"). Retombe sur
    McCormick_inter_layers() (chemin lent mais garanti correct) au moindre doute :
    retourne False sans rien ajouter si une hypothese du cas rapide n'est pas
    verifiee, l'appelant doit alors utiliser le chemin classique pour ce triple.
    """
    C = self.handler.Constraints

    dec1, fom1 = C.verify_variable_z(k, neuron_next, False)
    dec2, fom2 = C.verify_variable_z(k - 1, neuron_prev, True)
    dict1 = C.equivalent_neurons.get_equivalent(
        layer=k, neuron=neuron_next, front_of_matrix=fom1,
        decomposed_in_front_and_back_matrix=dec1,
    )
    dict2 = C.equivalent_neurons.get_equivalent(
        layer=k - 1, neuron=neuron_prev, front_of_matrix=fom2,
        decomposed_in_front_and_back_matrix=dec2,
    )
    const1 = C.equivalent_neurons.get_constant(k, neuron_next)
    const2 = C.equivalent_neurons.get_constant(k - 1, neuron_prev)
    if len(dict1) != 1 or len(dict2) != 1 or const1 != 0.0 or const2 != 0.0:
        return False  # neurone substitue (ex. stable actif) -- chemin lent requis

    key1 = next(iter(dict1.keys()))
    key2 = next(iter(dict2.keys()))
    w1 = dict1[key1]
    w2 = dict2[key2]
    if w1 != 1.0 or w2 != 1.0:
        return False  # poids non trivial -- ne devrait pas arriver, securite

    i1, num_matrix1 = _get_linear_indices_from_key_py(key1)
    i2, num_matrix2 = _get_linear_indices_from_key_py(key2)
    if num_matrix1 != num_matrix2 or i1 == i2:
        # num_matrix different : ne devrait pas arriver pour un groupement [k,k+1]
        # standard (cf. investigation dualisation, aucune contrainte RLT ne
        # traverse 2 blocs) -- securite. i1 == i2 : les deux termes lineaires
        # tomberaient sur la meme position (i,0), il faudrait les fusionner --
        # cas non gere par le chemin rapide, on retombe sur le chemin lent.
        return False

    i_a, i_b = (i1, i2) if i1 > i2 else (i2, i1)
    value_quad = 1.0 if i_a == i_b else 0.5

    L = C.L
    U_above_zero = C.U_above_zero
    if L[k - 1][neuron_prev] < 0 and (k - 1) > 0:
        lb_prev = 0.0
    else:
        lb_prev = L[k - 1][neuron_prev]
    if L[k][neuron_next] < 0 and k < self.K:
        lb_next = 0.0
    else:
        lb_next = L[k][neuron_next]
    U_prev = U_above_zero[k - 1][neuron_prev]
    U_next = U_above_zero[k][neuron_next]

    # Memes 3 sous-contraintes et memes formules que McCormick_inter_layers
    # (12b, 12c, 12c second part) : (suffixe, bound_type, coeff sur z_next, coeff sur z_prev, bound).
    subs = (
        ("12b", mosek.boundkey.lo, -lb_prev, -lb_next, -lb_prev * lb_next),
        ("12c", mosek.boundkey.up, -U_prev, -lb_next, -U_prev * lb_next),
        ("12c2", mosek.boundkey.up, -lb_prev, -U_next, -lb_prev * U_next),
    )
    for suffix, bound_type, coeff_next, coeff_prev, bound in subs:
        name = (
            f"McCormick - Layer {k - 1}, neuron {neuron_prev}    ; "
            f"Layer {k - 1 + 1}, neuron {neuron_next}  - {suffix} (RLT, compact)"
        )
        # Meme convention que add_dict_linear_to_elements (dividing_non_diag=True,
        # i!=0 toujours vrai ici car l'indice 0 est reserve a l'homogeneisation).
        lin_next = coeff_next / 2.0
        lin_prev = coeff_prev / 2.0

        num_matrix_arr = [num_matrix1]
        i_arr = [i_a]
        j_arr = [i_b]
        val_arr = [value_quad]
        if lin_next != 0.0:
            num_matrix_arr.append(num_matrix1)
            i_arr.append(i1)
            j_arr.append(0)
            val_arr.append(lin_next)
        if lin_prev != 0.0:
            num_matrix_arr.append(num_matrix1)
            i_arr.append(i2)
            j_arr.append(0)
            val_arr.append(lin_prev)

        if bound_type == mosek.boundkey.lo:
            lb, ub = bound, infinity
        else:
            lb, ub = -infinity, bound
        # constant=0 (pas de substitution ici) donc lb/ub finaux = bound tel quel,
        # cf. add_bound() (bound - constant).

        C.current_num_constraint += 1
        C.list_cstr.append({
            "name": name,
            "elements": None,  # tableaux deja materialises ci-dessous, pas besoin de decode_key_vec()
            "constant": 0.0,
            "lb": lb,
            "ub": ub,
            "bound_type": bound_type,
            "dual_value": None,
            "label": "to_change",
            "is_quadratic": True,
            "not_in_miqcr": False,
            "role": ConstraintRole.HARD,
            "dualizable_family": None,
            "num_matrix": np.array(num_matrix_arr, dtype=np.int32),
            "i": np.array(i_arr, dtype=np.int32),
            "j": np.array(j_arr, dtype=np.int32),
            "value": np.array(val_arr, dtype=np.float64),
        })
        C.cstr_names.add(name)
        C.mark_current_dualizable("RLT")
    return True


def add_RLT_constraints_compact(self, p: float = 0.5):
    """
    Version compacte de add_RLT_constraints -- meme boucle/heuristique de selection
    (identique bit a bit a add_RLT_constraints), mais chaque triple RLT passe par
    _add_McCormick_compact() (chemin rapide) et ne retombe sur McCormick_inter_layers()
    (chemin classique) que si le cas rapide ne s'applique pas.

    N'est utilisee que si use_compact_add_rlt=true dans le yaml (pense pour
    engine="conicbundle_native", ou generer 100% des contraintes RLT candidates --
    RLT_props: 1. -- pour laisser le bundle choisir lesquelles dualiser reellement
    -- devient couteux avec add_RLT_constraints classique). Le chemin classique
    (add_RLT_constraints) reste inchange et est celui utilise par defaut.
    """
    nb_rlt = 0
    nb_fallback = 0

    print("Adding RLT constraint (compact)")
    start_k = 2 if not self.INPUT_IN_VARIABLES else 1
    for k in range(start_k, self.K + 1 if self.LAST_LAYER else self.K):
        nb_cstr = int(p * self.n[k - 1])

        indexes_pruned = [
            j
            for j in range(self.n[k - 1])
            if (k - 1, j) in self.stable_inactives_neurons
            or (k - 1, j) in self.stable_actives_neurons
            or (k == 1 and j in self.pruned_input_neurons)
        ]
        if k == self.K and self.LAST_LAYER:
            neurons_next = list(set([self.ytrue]).union(self.ytargets))
        else:
            neurons_next = [j for j in range(self.n[k])
                            if (k, j) not in self.stable_actives_neurons
                            and (k, j) not in self.stable_inactives_neurons]
        print("Indexes pruned for layer", k, ":", indexes_pruned)
        for neuron_next in neurons_next:
            if (k, neuron_next) in self.stable_inactives_neurons:
                continue
            if (k, neuron_next) in self.stable_actives_neurons and (
                not self.keep_penultimate_actives or k != self.K - 1
            ):
                continue
            neurons_with_great_weights = get_m_indexes_of_higher_values_in_list(
                np.abs(self.W[k - 1][neuron_next]), nb_cstr, indexes_pruned
            )
            for neuron_prev in neurons_with_great_weights:
                assert (k - 1, neuron_prev) not in self.stable_inactives_neurons and (
                    k - 1,
                    neuron_prev,
                ) not in self.stable_actives_neurons
                added = self._add_McCormick_compact(k, neuron_prev, neuron_next)
                if not added:
                    self.McCormick_inter_layers(k, neuron_prev, neuron_next)
                    nb_fallback += 1
                nb_rlt += 1
    print(f"CALLBACK : RLT (compact) = {nb_rlt} (dont {nb_fallback} en repli chemin lent)")