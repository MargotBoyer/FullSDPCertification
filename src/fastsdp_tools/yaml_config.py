from pydantic import BaseModel, validator, model_validator
from typing import List, Optional, Any, Union
from pathlib import Path
import yaml
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset


from .utils import get_project_path


class MiniDataset(Dataset):
    def __init__(self, x, y):
        self.data = [
            [(x.squeeze(0), y.squeeze(0))]
        ]  # enlever batch dim (1, 784) → (784)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# On veut simuler : dataloader → (label y, list_x) où list_x = [(x, ytrue), ...]
class GroupedByLabelDataset:
    def __init__(self, label_to_data, ytrue):
        self.label_to_data = label_to_data
        self.ytrue = ytrue.squeeze(0)  # (1,) → scalaire

    def __iter__(self):
        for label, xs in self.label_to_data.items():
            list_x = [(x, self.ytrue) for x in xs]
            yield label, list_x

    def __len__(self):
        return len(self.label_to_data)


class DataConfig(BaseModel):
    name: str
    y: int
    x: Union[Any, str]

    @validator("x", pre=True)  # pre = True donc s'exécute avant la validation du type
    def validate_before_x(
        cls, x, values
    ):  # Here values has all the already validated values by order of assignment
        if isinstance(x, (str, Path)):
            path = Path(get_project_path(x.replace("\\", "/")))
            if not path.exists():
                raise ValueError(f"Dataset file not found: {path}")
            try:
                examples = torch.load(path, weights_only=False)
            except TypeError:
                examples = torch.load(path)

            y = values.get("y")
            if y in examples.keys():
                assert examples[y][1] == y

                transform = transforms.ToTensor()
                return transform(examples[y][0].numpy()).view(-1).tolist()
            else:
                raise FileNotFoundError(
                    f"Not example found with label {y} in path : {path}"
                )
        else:
            return x  # x already defined explicitely

    ytarget: Optional[int] = None

   
    @model_validator(mode="after")
    def create_dataset(self) -> "DataConfig":
        # Créer une map par label
        print("ytrue in Data Config:", self.y)
        ytrue = torch.tensor([self.y], dtype=torch.int64).unsqueeze(0)
        print("ytrue in Data Config apres tensor operator : ", ytrue)
        x_tensor = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        label_to_data = {int(ytrue.item()): [x_tensor.squeeze(0)]}

        self.dataset = GroupedByLabelDataset(
            label_to_data=label_to_data,
            ytrue=ytrue,
        )

        return self


class InputBallConfig(BaseModel):
    norm: str
    @validator("norm")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["Linf", "L2", "L1"]:
            raise ValueError(
                f"Input ball norm {v} must be one of 'Linf', 'L2', or 'L1'."
            )
        return v
    epsilon: float

class DatasetConfig(BaseModel):
    name: str
    path: str

    num_classes: int
    num_samples: int


class DynamicConicBundleConfig(BaseModel):
    """Configuration de la conic bundle method (main.pdf, section 5.4). Ne prend
    effet que si un modèle SDPSolverConfig référence ce bloc via
    `dynamic_conic_bundle:`. Nécessite solver="mosek_classic" (seul backend
    supporté). CHORDAL_DECOMPOSITION=False (bloc unique) et =True (grouping
    standard [k,k+1]) sont supportés ; les regroupements custom
    (List[List[int]]) ne le sont pas encore (TODO Phase D, task-dynamic-conic-bundle.md)."""

    enabled: bool = True
    dualize: List[str] = ["RLT"]  # Familles de coupes à dualiser (doivent être dans `cuts` et taguées ConstraintRole.DUALIZABLE)

    engine: str = "conicbundle_native"  # "python" (bundle proximal from-scratch, src/dynamic_conic_bundle/) | "conicbundle_native" (pont direct vers la vraie librairie ConicBundle, src/miqcr_bridge/conicbundle_native/ — cf. task-dynamic-conic-bundle.md "Pivot" : converge plus vite et plus précisément, recommandé par défaut à terme mais pas encore le défaut pour ne pas casser les runs existants)

    @validator("engine")
    def validate_engine(cls, v):
        if v not in ["python", "conicbundle_native"]:
            raise ValueError(
                f"engine '{v}' inconnu — doit être 'python' (bundle from-scratch) ou "
                "'conicbundle_native' (pont direct ConicBundle, cf. task-dynamic-conic-bundle.md)."
            )
        return v

    # Paramètres spécifiques à engine="conicbundle_native" (ignorés par engine="python") :
    term_relprec: float = 1.0e-7  # Critère d'arrêt de la vraie ConicBundle (cb_set_term_relprec) : arrête quand la progression prédite est sous term_relprec*(|objectif|+1)
    eval_limit: int = 5000  # Nb max d'appels à l'oracle (cb_set_eval_limit) — indépendant de max_iter qui ne borne que les pas de descente ; un pas de descente peut englober plusieurs pas nuls/appels oracle
    print_level: int = 0  # Verbosité interne de la vraie ConicBundle (cb_set_print_level). 0 = silencieux (hors résumé final), >=1 = trace native par itération.
    log_every: int = 0  # Fréquence (en appels oracle) du print Python de suivi (h courant / meilleur h, cf. cb_wrapper.solve_native_conicbundle_dual) — 0 = jamais. Sur un run long (des heures), mettre par ex. 10-50 pour avoir une trace de progression dans run.log.
    active_bounds_fixing: bool = True  # cb_set_active_bounds_fixing -- recommandé par cb_cinterface.h pour la relaxation lagrangienne mais documenté "no convergence theory". Écarté empiriquement comme cause des échecs "upper bound < lower bound" (résultats identiques avec false) -- gardé configurable pour référence.
    max_subg_by_point: int = 10  # Nb max de sous-gradients epsilon renvoyés par appel oracle QUAND un "coin" est détecté (dégénérescence du X* optimal MOSEK, cf. cb_wrapper.solve_native_conicbundle_dual) -- adresse directement les échecs "upper bound < lower bound" observés en dualisant beaucoup de contraintes (RLT+ReLU_quad+triangularization/ReLU_linear) : un seul sous-gradient n'est représentatif que d'une direction quand le sous-problème résiduel a plusieurs X* optimaux très différents. 1 = désactive l'enrichissement (comportement historique).

    @validator("bundle_size_factor")
    def validate_bundle_size_factor(cls, v):
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError(f"bundle_size_factor doit être dans ]0, 1] (proportion du pool dualisé), reçu {v}.")
        return v

    @validator("factor")
    def validate_factor(cls, v):
        if v is not None and not (0.0 < v <= 1.0):
            raise ValueError(f"factor doit être dans ]0, 1] (proportion des contraintes dualisables réellement dualisées), reçu {v}.")
        return v

    @validator("dualize")
    def validate_dualize(cls, v):
        for family in v:
            if family not in [
                "RLT", "ReLU_quad", "ReLU_linear", "triangularization",
                "McCormick_beta_z", "beta_logits_comparaison", "sum_beta_logits_equal_logit",
            ]:
                raise ValueError(
                    f"Famille de coupes '{family}' non dualisable — familles taguées "
                    "ConstraintRole.DUALIZABLE (cf. Constraints.mark_current_dualizable) : "
                    "'RLT' (McCormick_inter_layers), 'ReLU_quad' (l'équation quadratique "
                    "z_k*(z_k - W z_(k-1) - b_k) = 0), 'ReLU_linear' (z_k>=0 et "
                    "z_k>=W z_(k-1)+b_k, ReLU_constraint_Lan), 'triangularization' "
                    "(ReLU_triangularization), 'McCormick_beta_z' (beta_j*z_(layer,i), "
                    "modèles untargeted uniquement), 'beta_logits_comparaison' "
                    "(z_(K,j2)*beta_j2, untargeted) et 'sum_beta_logits_equal_logit' "
                    "(z_i - sum_j beta_j*z_i = 0, untargeted). quad_bounds, "
                    "first_term_equal_zero, discrete_betas, betai_betaj et la cohérence "
                    "chordale (CHORDAL_DECOMPOSITION_rec) restent toujours dures."
                )
        return v

    dynamic: bool = False  # False = algorithme statique (5.4.1, Algorithme 2, pool fixe) ; True = dynamique (5.4.2, ajout/retrait de contraintes). Supporté par les deux moteurs, avec des mécanismes DIFFÉRENTS : engine="python" (dynamic_conic_bundle/solver.py) redémarre un round de bundle proximal à chaque mise à jour du pool ; engine="conicbundle_native" (solve_native_conicbundle_dynamic, cb_wrapper.py) redimensionne le problème ConicBundle EN PLACE via cb_append_variables/cb_reassign_variables (transposé du mécanisme réel de MIQCR, solver_sdp_mixed.c), sans jamais détruire/reconstruire le bundle. Réutilise add_batch_size/theta_drop_tol/max_rounds dans les deux cas.
    max_iter: int = 200  # Nombre max d'itérations proximal-bundle par round
    C1: float = 1.0e-4  # Critère d'arrêt : arrête quand la progression *prédite* par le modèle du bundle (pas la progression réelle) tombe sous ce seuil
    C2: float = 0.1  # Ratio (dans (0,1)) de la progression prédite à réaliser réellement pour qu'un pas soit dit "sérieux" (voir proximal_master.serious_null_step_test)
    proximal_u_init: float = 1.0  # Paramètre proximal u (Algorithme 2, ligne 5)
    theta_drop_tol: float = 1.0e-3  # Mode dynamique seulement : retire une contrainte active si |theta_r| < ce seuil (theta = notation main.pdf, sans rapport avec alpha-CROWN)
    add_batch_size: int = 50  # Mode dynamique seulement : nb de contraintes les plus violées ajoutées par round
    max_pool_size: Optional[int] = None  # Mode dynamique seulement : plafond STRICT sur le nombre de contraintes simultanément dualisées dans l'objectif (dimension active de ConicBundle), découplé de l'univers candidat (`all_dualizable_names`/`factor`, qui peut rester bien plus grand). None = pas de plafond (comportement historique). Sans ce plafond, avec un univers candidat large (factor proche de 1), le pool actif peut grossir librement round après round (observé concrètement : 50 → 614+ contraintes, data_index=59 UntargetedSDP RLT factor=1) et finir par rendre le problème MOSEK numériquement instable (coefficients ~1e+203, rescode.err_sym_mat_huge) — pas un problème de taille de l'univers candidat en soi, mais du nombre de theta_r simultanément actifs dans l'objectif à un instant donné.
    max_rounds: int = 100  # Mode dynamique seulement : nb max de rounds (ajout/retrait) — pas de limite propre sinon, contrairement à max_iter qui ne borne que la boucle interne par round
    use_subgext: bool = False  # Mode dynamique (engine="conicbundle_native") seulement : active cb_subgextp (extension de sous-gradient via agrégation primale) avec une représentation CREUSE du primal (uniquement les entrées de matrice effectivement référencées par au moins une contrainte DUALIZABLE candidate, pas l'union dense de tous les blocs Pk). Sans ça, le bundle ConicBundle est perdu à chaque mise à jour du pool (cf. doc cb_cinterface.h : "the cutting model of the objective is lost at each addition of constraints" sans ce callback) — confirmé concrètement : bundle_size plafonné à 0-6 tout au long d'un run, convergence nettement plus lente que le moteur statique (qui accumule librement). Une 1re tentative avec un primal DENSE (~395k valeurs, toutes les entrées de tous les blocs) a provoqué un segfault natif déterministe à cette échelle — cette version creuse vise à l'éviter. False (défaut) = comportement historique inchangé, aucune régression pour les runs existants qui ne fixent pas ce paramètre.
    max_bundle_size: Optional[int] = None  # Nb max de coupes conservées dans le bundle. None = comportement par défaut de chaque moteur : pour engine="python", pas de cap (borné naturellement par max_iter) — recommandé, cf. task-dynamic-conic-bundle.md "Analyse statique vs dynamique" (cap trop petit face à dim(theta) = arrêt prématuré), éviction FIFO (cf. ProximalBundle) ; pour engine="conicbundle_native", défaut interne de la vraie librairie = 50 (constante codée en dur, FunctionProblem::FunctionProblem dans funproblem.cxx), éviction/agrégation interne à la librairie une fois le plafond atteint. Fixer un entier pour borner le coût du master problem QP à grande échelle (au prix d'une convergence potentiellement plus lente/moins précise si le cap est trop petit).
    bundle_size_factor: Optional[float] = None  # Anciennement nommé `factor` (renommé pour ne plus être confondu avec le vrai paramètre FACTOR de MIQCR, cf. `factor` ci-dessous). Ignoré si max_bundle_size est déjà fourni explicitement (max_bundle_size a toujours priorité). Sinon, si bundle_size_factor n'est pas None, max_bundle_size = max(1, round(bundle_size_factor * nb_contraintes_dualizable)), calculé une fois le modèle construit (les deux moteurs "python" et "conicbundle_native" le supportent). Plafonne le nombre de COUPES conservées en mémoire dans le bundle ConicBundle (cb_set_max_bundlesize) — n'affecte PAS le nombre de contraintes réellement dualisées (ça, c'est le rôle de `factor`). Doit être dans ]0, 1].
    factor: Optional[float] = None  # Équivalent du VRAI paramètre FACTOR de la librairie MIQCR d'origine (Miqcr-1.0_.../src/parameters.h, "Proportion of the considered constraints into the SDP solver" ; CLAUDE.md MIQCR section 7 : psdp->nb_cont = FACTOR * psdp->length). Détermine la proportion des contraintes DUALIZABLE candidates qui sont EFFECTIVEMENT dualisées (reçoivent un theta_r) dans la relaxation lagrangienne de l'objectif -- les autres sont totalement absentes du modèle (ni dures, ni pénalisées), exactement comme dans MIQCR. None (défaut) = toutes les contraintes DUALIZABLE candidates sont dualisées (comportement historique). Sélection déterminée par `factor_constraint_heuristic` ci-dessous. engine="conicbundle_native" uniquement pour l'instant. Doit être dans ]0, 1]. Pour dynamic=true, doit être None ou 1.0 (cf. validate_factor_requires_full_universe_in_dynamic) -- la troncature n'a de sens que pour le moteur statique, où chaque contrainte retenue devient une variable ConicBundle permanente.

    factor_constraint_heuristic: str = "positional"  # Méthode de sélection des n_keep=factor*m contraintes réellement dualisées parmi les m candidates DUALIZABLE (ignoré si factor est None). "positional" (défaut, comportement historique) : garde les n_keep premières dans l'ordre de get_dualizable_constraint_names -- BIAISÉ : cet ordre suit la construction des coupes, strictement couche par couche (cf. certification_problem_constraints_rlt.py), donc un factor petit ne retient en pratique QUE les premières couches (confirmé concrètement sur data_index=59 : factor=0.01 => 100% des 1808 contraintes retenues venaient de la couche 0, 0% des couches 1-8). "random_pure" : échantillon uniforme sans remise parmi les m candidates, sans égard à la couche -- lève le biais positionnel mais peut encore sous-représenter certaines couches par pur hasard si elles sont peu nombreuses. "random_layer" : regroupe d'abord les candidates par couche (bloc num_matrix de list_cstr, Phase A garantit qu'une contrainte DUALIZABLE ne référence jamais 2 couches à la fois), puis prélève un nombre égal au hasard dans chaque couche (n_keep // nb_couches), le reste (arrondi + couches trop petites) étant comblé par un tirage aléatoire parmi les contraintes non retenues de toutes les couches -- recommandé quand on veut explicitement garantir une couverture de toutes les couches avec un factor<1.
    @validator("factor_constraint_heuristic")
    def validate_factor_constraint_heuristic(cls, v):
        if v not in ("positional", "random_pure", "random_layer"):
            raise ValueError(
                f"factor_constraint_heuristic '{v}' inconnu -- doit être 'positional' "
                "(comportement historique, biaisé par couche), 'random_pure' (tirage "
                "uniforme sur tout l'univers candidat) ou 'random_layer' (tirage "
                "équilibré par couche)."
            )
        return v
    max_new_subgradients: Optional[int] = None  # engine="conicbundle_native" uniquement (ignoré par engine="python", qui ne renvoie jamais plus d'un sous-gradient par évaluation) : nb max de nouveaux sous-gradients epsilon renvoyés par appel oracle (cb_set_max_new_subgradients). None = défaut interne de la librairie = 1 (FunctionBundleParameters::FunctionBundleParameters(), funproblem.hxx — PAS 5 comme documenté précédemment). Avec ce défaut, la sonde de coin (cf. max_subg_by_point dans cb_wrapper.solve_native_conicbundle_dual) ne s'exécute JAMAIS : il faut explicitement fixer max_new_subgradients >= 2 pour l'activer.
    log_theta_every_n_iter: int = 1  # Fréquence (en itérations) d'enregistrement de theta_history (0 = jamais)
    print_png: bool = True  # Si true, écrit un PNG de diagnostic par résolution (h(theta)/u/taille du bundle vs itération, cf. dynamic_conic_bundle/plotting.py) dans le dossier du run. Activé par défaut (coût I/O/matplotlib jugé acceptable face à la valeur diagnostique, cf. analyse du blocage data_index=59/target=8) — mettre à false pour un run à grande échelle (des centaines d'échantillons) où ce coût par résolution n'est plus négligeable.
    png_every: int = 10  # engine="conicbundle_native" uniquement : régénère le PNG (+ écrit une ligne dans le CSV compagnon .csv, à chaque appel oracle) tous les png_every appels oracle, plutôt qu'une seule fois à la toute fin. Permet de suivre un run long en direct et de conserver un diagnostic si le process est tué/timeout avant la fin. 0 = uniquement le PNG final (comportement historique) ; le CSV compagnon est toujours écrit en direct dès que print_png=true.

    stop_when_positive: bool = True  # Arrête le bundle dès qu'une évaluation de h(theta) (n'importe quel point oracle, y compris pas nuls) dépasse positivity_threshold : une seule borne duale valide strictement positive suffit à certifier la robustesse (dualité faible), inutile de continuer à affiner theta. status="reached_positivity" dans results_conic_bundle.csv. Supporté par les deux moteurs ("python" et "conicbundle_native").
    positivity_threshold: float = 1.0e-6  # Seuil utilisé par stop_when_positive.

    sgnorm_term_tol: Optional[float] = None  # engine="conicbundle_native", dynamic=true uniquement : critère d'arrêt additionnel sur ||sous-gradient agrégé|| (cb_get_sgnorm), même rôle que EPS_TERM_CB dans la boucle externe de MIQCR (solver_sdp_mixed.c::run_conic_bundle_mixed : continue tant que cb_get_sgnorm(p) > EPS_TERM_CB). None (défaut) = désactivé, comportement historique inchangé. Vérifié après termination_code (reste la condition d'arrêt primaire), pas un remplacement — un filet de sécurité additionnel pour éviter de tourner sur un pool quasi plat. 0.1 est une valeur de départ raisonnable pour nos tests (cf. investigation oscillation data_index=59).

    @validator("positivity_threshold")
    def validate_positivity_threshold(cls, v):
        if v <= 0:
            raise ValueError(f"positivity_threshold doit être strictement positif, reçu {v}.")
        return v

    @validator("sgnorm_term_tol")
    def validate_sgnorm_term_tol(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"sgnorm_term_tol doit être strictement positif ou None, reçu {v}.")
        return v

    @model_validator(mode="after")
    def validate_factor_requires_full_universe_in_dynamic(self) -> "DynamicConicBundleConfig":
        """factor<1 tronque l'univers candidat POSITIONNELLEMENT (les n_keep premières
        contraintes dans l'ordre de get_dualizable_constraint_names, pas un échantillon
        représentatif) -- confirmé concrètement sur data_index=59 : factor=0.01 ne gardait
        que les RLT de la couche 0 (0% des couches 1-8). En mode dynamique, max_pool_size
        joue déjà le rôle que `factor` visait (plafonner le nombre de contraintes
        simultanément dualisées dans l'objectif, cf. incident err_sym_mat_huge) sans ce
        biais, puisque le pool actif est choisi par violation parmi TOUT l'univers
        candidat. factor reste nécessaire pour le moteur statique (où chaque candidat
        devient une variable ConicBundle permanente), donc cette contrainte ne s'applique
        qu'à dynamic=True."""
        if self.dynamic and self.factor not in (None, 1.0):
            raise ValueError(
                f"factor={self.factor} invalide avec dynamic=true : le moteur dynamique "
                "doit toujours partir de l'univers candidat complet (factor=1.0 ou None), "
                "sous peine de reproduire le biais positionnel observé sur data_index=59 "
                "(factor=0.01 => 100% des contraintes retenues venaient de la couche 0). "
                "Utiliser max_pool_size pour plafonner le nombre de contraintes "
                "simultanément dualisées dans l'objectif, et use_subgext=true pour "
                "conserver le bundle ConicBundle d'un round à l'autre."
            )
        return self


class SDPSolverConfig(BaseModel):
    certification_model_type: str

    @validator("certification_model_type")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["TargetedSDP", "UntargetedSDP", "MzbarSDP"]:
            raise ValueError(
                f"SDP model name {v} must be one of 'TargetedSDP', 'UntargetedSDP', or 'MzbarSDP'."
            )
        return v

    cuts: Optional[List[str]] = []
    @validator("cuts")
    def validate_cuts(cls, v, values):
        if v is None:
            return []
        for cut in v :
            if cut not in ["RLT", "triangularization", "McCormick_beta_z", "beta_logits_comparaison", 
                           "beta_logits_comparaison_big_M", "sum_beta_logits_equal_logit"]:
                raise ValueError(f"cut {cut} not valid.")
        return v

    all_combinations_cuts: Optional[bool] = False
    RLT_props: Optional[List[float]] = [0.0]

    @validator("RLT_props")
    def validate_rlt_prop(cls, v, values):
        if (
            "cuts" in values
            and values["cuts"] is not None
            and "RLT" in values["cuts"]
            and v is None
        ):
            raise ValueError("RLT cuts are required, but RLT_prop is None.")
        return v

    CHORDAL_DECOMPOSITION: Union[bool, List[List[int]]] = True
    @validator("CHORDAL_DECOMPOSITION", pre=True)
    def validate_and_normalize_CHORDAL_DECOMPOSITION(cls, v, values):
        """Normalise en List[List[int]] ou garde bool pour résolution tardive."""
        if isinstance(v, bool):
            return v  # résolution tardive quand K est connu
        if isinstance(v, list):
            # Validation : chaque groupe a ≥ 2 couches
            for group in v:
                assert len(group) >= 2, f"Group {group} must have at least 2 layers"
            # Validation : chevauchement exact entre groupes consécutifs
            for i in range(len(v) - 1):
                assert v[i][-1] == v[i+1][0], (
                    f"Groups {v[i]} and {v[i+1]} must share exactly one boundary layer"
                )
            return v
           
        raise ValueError(f"CHORDAL_DECOMPOSITION must be bool or List[List[int]], got {type(v)}")
    LAST_LAYER: bool = (
        False  # Whether to use the last layer of the network (logits) as variables
    )
    solver: str = "mosek_classic"  # Backend solver : "mosek_classic" | "mosek_fusion" | "cvxpy"
    @validator("solver")
    def validate_solver(cls, v, values):
        if v not in ["mosek_classic", "mosek_fusion", "cvxpy"]:
            raise ValueError(f"solver must be one of 'mosek_classic', 'mosek_fusion', 'cvxpy', got '{v}'")
        return v
    cp_solver: str = "MOSEK"  # CVXPY backend : "MOSEK", "SCS", "CLARABEL", "CVXOPT", ...
    cp_solver_kwargs: Optional[dict] = None  # Kwargs passés à cp.Problem.solve()
    use_callback: bool = False  # Whether to use the callback for MOSEK
    use_active_neurons: Optional[bool] = (
        False  # Whether to use active neurons in the certification problem as variables
    )
    ultimate_layer_use_active_neurons: Optional[int] = 1e5 # Whether to use active neurons in the ultimate layer in the certification problem as variables, if use_active_neurons is True. 0 = no ultimate layer active neurons, 1 = only ultimate layer active neurons, 2 = all active neurons
    use_inactive_neurons: Optional[bool] = (
        False  # Whether to use inactive neurons in the certification problem as variables
    )
    keep_penultimate_actives : Optional[bool] = False
    @validator("keep_penultimate_actives")
    def validate_keep_penultimate_actives(cls, v, values):
        if v is False and values.get("use_active_neurons"):
            raise ValueError("Withdraw of active neurons on penultimate layer incompatible with use_active_neurons = True")
        return v
    use_compact_add_rlt: bool = False  # Si true, utilise add_RLT_constraints_compact (SDPmodels/certification_problem_constraints_rlt.py) au lieu de add_RLT_constraints : construit directement les contraintes McCormick RLT dans list_cstr, sans passer par new_constraint()/add_quad_variable()/add_var() (un numba.typed.Dict par contrainte -- cout fixe disproportionné pour des contraintes à 1-3 termes). Retombe automatiquement sur le chemin classique pour tout triple nécessitant une substitution (neurone stable actif). Pensé pour engine="conicbundle_native" avec RLT_props=1. (générer 100% des RLT candidates et laisser le bundle choisir lesquelles dualiser) -- garder à false pour un run classique (résultat identique, juste plus lent à construire).
    bounds_file: Optional[str] = None
    L: Optional[List[float]] = None
    U: Optional[List[float]] = None
    bounds_method: str = "alpha-CROWN"  # Method to compute bounds, options: "IBP", "alpha-CROWN", "GREAT_BOUNDS", "from_file"
    bounds_n_runs: int = 1  # Number of independent alpha-CROWN runs; best-of-N is kept (max L, min U). Ignored for non-CROWN methods.
    write_model : Optional[bool] = False
    INPUT_IN_VARIABLES: Union[bool, float] = True  # If False/0.0, z_0 removed from SDP; if 0<p<1, keep top p*n_0 input neurons by W_1 column norm
    solver_time_limit: Optional[int] = 7200  # Time limit in seconds for MOSEK solver (None = no limit)
    dynamic_conic_bundle: Optional[DynamicConicBundleConfig] = None  # None = comportement classique inchangé (voir task-dynamic-conic-bundle.md)


class GurobiSolverConfig(BaseModel):
    certification_model_type: str

    @validator("certification_model_type")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["TargetedQuad", "UntargetedQuad", "MzbarQuad", "ClassicLP", "LPBoundLayer"]:
            raise ValueError(
                f"Model name {v} must be one of 'TargetedQuad', 'UntargetedQuad', 'MzbarQuad','ClassicLP', 'LPBoundLayer."
            )
        return v

    LAST_LAYER: bool = (
        False  # Whether to use the last layer of the network (logits) as variables
    )
    use_active_neurons: Optional[bool] = (
        False  # Whether to use active neurons in the certification problem as variables
    )
    use_inactive_neurons: Optional[bool] = (
        False  # Whether to use inactive neurons in the certification problem as variables
    )
    L: Optional[List[float]] = None
    U: Optional[List[float]] = None
    bounds_method: str = "alpha-CROWN"  # Method to compute bounds, options: "IBP", "alpha-CROWN", "GREAT_BOUNDS", "from_file"
    bounds_file: Optional[str] = None
    bounds_n_runs: int = 1  # Number of independent alpha-CROWN runs; best-of-N is kept (max L, min U). Ignored for non-CROWN methods.
    write_model : Optional[bool] = False
    INPUT_IN_VARIABLES: Union[bool, float] = True  # If False/0.0, z_0 removed from SDP; if 0<p<1, keep top p*n_0 input neurons by W_1 column norm
    solver_time_limit: Optional[int] = 7200  # Time limit in seconds for GUROBI solver (None = no limit)

class NetworkConfig(BaseModel):
    name: str
    path: str
    K: int
    n: List[int]
    dropout: Optional[float] = 0


class ConicBundleConfig(BaseModel):
    filename: str
    McCormick: Optional[str] = "none"


class FullCertificationConfig(BaseModel):
    input_ball: InputBallConfig
    data: Union[DataConfig, DatasetConfig]
    network: NetworkConfig
    models: Optional[List[Union[SDPSolverConfig, GurobiSolverConfig]]] = None
    divide_run: int = 1
    @validator('models')
    def process_bounds(cls, v, values):
        input_ball = values.get("input_ball")
        norm = input_ball.norm if input_ball is not None else None

        for model in v:
            if not(model.bounds_method in ["IBP", "alpha-CROWN", "GREAT_BOUNDS", "from_file"]):
                raise ValueError("Bounds method must be one of 'IBP', 'alpha-CROWN', 'GREAT_BOUNDS', or 'from_file'.")
            if model.bounds_n_runs < 1:
                raise ValueError("bounds_n_runs must be >= 1.")
            elif model.bounds_method == "from_file" and model.bounds_file is None:
                raise ValueError("Bounds file must be specified if bounds method is 'from_file'.")
            elif model.bounds_method == "from_file" and norm is not None:
                filename_lower = Path(model.bounds_file).name.lower()
                if norm.lower() == "l2" and "l2" not in filename_lower:
                    raise ValueError(
                        f"norm='L2' but bounds file '{model.bounds_file}' does not contain 'l2' in its name. "
                        f"The bounds must have been computed with the L2 norm."
                    )
                if norm.lower() == "linf" and "linf" not in filename_lower:
                    raise ValueError(
                        f"norm='Linf' but bounds file '{model.bounds_file}' does not contain 'linf' in its name. "
                        f"The bounds must have been computed with the Linf norm."
                    )
            elif model.bounds_method == "GREAT_BOUNDS":
                L = [[model.L[k]] * model.n[k] for k in range(model.K + 1)]
                U = [[model.U[k]] * model.n[k] for k in range(model.K + 1)]
                model.L = L
                model.U = U
            else:
                model.L = None
                model.U = None

        network = values.get("network")
        if network is not None :
            K = network.K
            for model in v:
                if isinstance(model,SDPSolverConfig) and isinstance(model.CHORDAL_DECOMPOSITION, list):
                    last_index = model.CHORDAL_DECOMPOSITION[-1][-1]
                    expected_last = 7 if model.LAST_LAYER else K - 1
                    if last_index != expected_last:
                        raise ValueError(
                            f"With LAST_LAYER={model.LAST_LAYER}, last element of CHORDAL_DECOMPOSITION "
                            f"must be {expected_last}, got {last_index}."
                        )
        return v
    conic_solver: Optional[ConicBundleConfig] = None
   



class Adversarial_Network_Training(BaseModel):
    data: str
    train_path: str
    test_path: str
    evaluate_robustness_path: str = None
    num_classes: int
    adversarial_attack: str
    batch_size: int
    num_epochs: int
    lr: float
    epsilon: float
    epsilon_test: Optional[float] = None
    n: List[int]
    K: int
    name_network: str
    compute_bounds_method: Optional[str] = None
    alpha: Optional[float] = None
    steps: Optional[int] = None
    random_start: Optional[bool] = None
    dropout: Optional[float] = 0
