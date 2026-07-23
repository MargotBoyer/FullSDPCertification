"""
Test unitaire pour _build_pruned_adv_before et get_number_pruned_adversarial_targets_before_target.

Cas testé : n[K]=10, ytrue=3, ytargets=[0, 2, 4, 9]
  Classes prunées (ni ytrue, ni ytargets) : 1, 5, 6, 7, 8

Résultats attendus pour get_number_pruned_adversarial_targets_before_target(c) :
  c=0 → 0   (rien avant)
  c=1 → 0   (1 est lui-même prunée, mais on compte AVANT lui)
  c=2 → 1   (classe 1 prunée avant)
  c=3 → 1   (ytrue, 1 prunée avant)
  c=4 → 1   (classe 1 prunée avant ; ytrue=3 ne compte pas)
  c=5 → 1   (seule classe 1 avant, ytrue=3 ne compte pas)
  c=9 → 5   (classes 1,5,6,7,8 prunées avant)
"""
import sys
sys.path.insert(0, "src/solve/mosek_solve/handler")

from indexes import Indexes_Mosek_Solver

K = 3
n = [4, 5, 5, 10]   # réseau jouet : 4 entrées, 2 couches cachées de 5, 10 sorties
ytrue = 3
ytargets = [0, 2, 4, 9]

# Tous les neurones des couches 1 et 2 sont instables (aucun pruning)
stable_inactives = []
stable_actives = []

idx = Indexes_Mosek_Solver(
    K=K,
    n=n,
    CHORDAL_DECOMPOSITION=True,
    LAST_LAYER=False,
    BETAS=True,
    BETAS_Z=False,
    ytrue=ytrue,
    ytargets=ytargets,
    stable_inactives_neurons=stable_inactives,
    stable_actives_neurons=stable_actives,
    keep_penultimate_actives=False,
)

expected = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5}

print("Test get_number_pruned_adversarial_targets_before_target :")
all_ok = True
for c, exp in expected.items():
    got = idx.get_number_pruned_adversarial_targets_before_target(c)
    status = "OK" if got == exp else f"FAIL (attendu {exp}, obtenu {got})"
    print(f"  c={c} → {got}  {status}")
    if got != exp:
        all_ok = False

print()
print("Test ind_label_beta (positions 0-based des betas) :")
# ytargets=[0,2,4,9] → indices attendus 0,1,2,3
beta_expected = {0: 0, 2: 1, 4: 2, 9: 3}
for c, exp in beta_expected.items():
    got = idx.ind_label_beta(c)
    status = "OK" if got == exp else f"FAIL (attendu {exp}, obtenu {got})"
    print(f"  beta[{c}] → {got}  {status}")
    if got != exp:
        all_ok = False

print()
print("Test index_variable_beta (positions dans la matrice dédiée, 1-based) :")
for c, exp in beta_expected.items():
    got = idx.index_variable_beta(c)
    status = "OK" if got == exp + 1 else f"FAIL (attendu {exp+1}, obtenu {got})"
    print(f"  index_variable_beta({c}) → {got}  {status}")
    if got != exp + 1:
        all_ok = False

print()
print("RÉSULTAT :", "TOUS LES TESTS PASSENT" if all_ok else "ÉCHEC(S) DÉTECTÉ(S)")
sys.exit(0 if all_ok else 1)
