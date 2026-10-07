# Baselines de comparaison

Implémentations/adaptateurs pour des méthodes de certification SDP publiées par d'autres
équipes, utilisées comme points de comparaison pour SDPT/SDPU. Convention (cohérente avec
le reste du projet) :

- Code source tiers **vendorisé tel quel** ici (`baselines/<nom>/`) quand il n'existe pas de
  package installable (ex. MATLAB). Code tiers **installé via pip** (pas vendorisé) quand
  c'est un vrai package Python, même ancien/archivé.
- Le **pont** vers notre pipeline (chargement réseau/bornes, export au format attendu par le
  baseline, parsing des résultats au format `results.csv`) vit dans
  `src/baselines_interface/<nom>/`, côté FastSDPCertification.

## Statut par papier

| Baseline | Papier | Code officiel | Statut |
|---|---|---|---|
| **SDP-FO** | Dathathri et al., *Enabling certification of verification-agnostic networks via memory-efficient semidefinite programming*, NeurIPS 2020 | [google-deepmind/jax_verify](https://github.com/google-deepmind/jax_verify) (archivé, `jax_verify/extensions/sdp_verify/`) | ✅ Adaptateur fonctionnel — voir `src/baselines_interface/sdp_fo/README.md` |
| **BM-r** | Chiu & Zhang, *Tight Certification of Adversarially Trained Neural Networks via Nonconvex Low-Rank Semidefinite Relaxations*, ICML 2023 | [Hong-Ming/BM-r](https://github.com/Hong-Ming/BM-r) (MATLAB + solveur Knitro) | 🚧 Réimplémenté en Python (`src/baselines_interface/bm_r/`) avec IPOPT (env dédié `baselines`, voir ci-dessous) à la place de MATLAB+Knitro (indisponibles ici). Valeur primale validée exacte contre `SDP-IP` (MOSEK) sur `blob_1x2` — **mais le certificat de second ordre (preuve d'optimalité globale, qui est le cœur de la méthode) ne se ferme pas de façon fiable avec IPOPT** ; `optimal_value` rapportée est donc la valeur primale brute, pas une borne certifiée — écart explicite documenté dans `src/baselines_interface/bm_r/README.md` ("Known issue"). |
| **SDP-BaB** | Lan, Brückner & Lomuscio, *A Semidefinite Relaxation Based Branch-and-Bound Method for Tight Neural Network Verification*, AAAI 2023 | Aucun dépôt public trouvé | 🚧 Réimplémenté (`src/baselines_interface/sdp_bab/`), tourne dans l'env `certif` (pas de MATLAB/Knitro nécessaire — contrairement à BM-r, c'est un problème NLP générique mais ici résolu via une relaxation SDP convexe, donc MOSEK suffit). Validé sur un réseau jouet (`blob_1x2`) ; pas encore testé sur un réseau réaliste — voir `src/baselines_interface/sdp_bab/README.md` pour le détail et les limites connues. |

## SDP-FO : environnement dédié

SDP-FO dépend de `jax_verify`, dont le code (2023, jamais mis à jour) utilise des API JAX
supprimées depuis `jax>=0.6`. Pour ne pas polluer les dépendances du projet principal
(torch/MOSEK/CVXPY dans l'env conda `certif`), il vit dans un env conda **dédié** `baselines` :

```bash
conda create -n baselines python=3.10
conda activate baselines
pip install "jax[cpu]<0.6" "jaxlib<0.6" dm-haiku optax dm-tree ml_collections cvxpy absl-py numpy typing_extensions
pip install "einshape @ git+https://github.com/deepmind/einshape.git"
pip install --no-deps "git+https://github.com/google-deepmind/jax_verify.git"
```

> ⚠️ Sur cette machine, `.venv/bin` est prioritaire dans le `PATH` même après `conda activate` :
> toujours vérifier `which python` (ou appeler l'interpréteur de l'env par son chemin absolu,
> `.../miniconda3/envs/baselines/bin/python`) avant d'installer quoi que ce soit, sous peine
> d'installer silencieusement dans le `.venv` du projet au lieu de l'env conda voulu.

Cet env est volontairement séparé de `certif` : le pont (`src/baselines_interface/sdp_fo/`)
communique par fichiers (`.npz` en entrée, `.json` en sortie) plutôt que par import direct,
exactement comme MOSEK est isolé par fork dans `sdp_generic_solver.py`.

## BM-r : même env `baselines` (dépendance supplémentaire)

BM-r n'a pas besoin de jax, mais a besoin de `cyipopt` (solveur NLP, remplace Knitro) et
`pandas` (écriture de `results.csv`) — installés dans le même env `baselines` que SDP-FO
(pas de conflit avec le pin `jax<0.6` constaté) :

```bash
conda install -n baselines -c conda-forge cyipopt
/path/to/envs/baselines/bin/python -m pip install pandas
```

Même pont par fichiers que SDP-FO (`src/baselines_interface/bm_r/instance_io.py`), même
avertissement sur le `PATH` (`.venv/bin` prioritaire même après `conda activate`).
