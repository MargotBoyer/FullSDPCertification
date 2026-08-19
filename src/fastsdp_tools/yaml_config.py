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
    supporté en Phase 2) et CHORDAL_DECOMPOSITION=False (TODO Phase 3)."""

    enabled: bool = True
    dualize: List[str] = ["RLT"]  # Familles de coupes à dualiser (doivent être dans `cuts` et taguées ConstraintRole.DUALIZABLE)

    engine: str = "python"  # "python" (bundle proximal from-scratch, src/dynamic_conic_bundle/) | "conicbundle_native" (pont direct vers la vraie librairie ConicBundle, src/miqcr_bridge/conicbundle_native/ — cf. task-dynamic-conic-bundle.md "Pivot" : converge plus vite et plus précisément, recommandé par défaut à terme mais pas encore le défaut pour ne pas casser les runs existants)

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

    @validator("dualize")
    def validate_dualize(cls, v):
        for family in v:
            if family not in ["RLT"]:
                raise ValueError(
                    f"Famille de coupes '{family}' non dualisable en Phase 2 (seule 'RLT' est "
                    "taguée ConstraintRole.DUALIZABLE pour l'instant — voir tableau Phase 1 "
                    "de task-dynamic-conic-bundle.md pour les candidates Phase 3+)."
                )
        return v

    dynamic: bool = False  # False = algorithme statique (5.4.1, Algorithme 2, pool fixe) ; True = dynamique (5.4.2, ajout/retrait de contraintes)
    max_iter: int = 200  # Nombre max d'itérations proximal-bundle par round
    C1: float = 1.0e-4  # Critère d'arrêt : arrête quand la progression *prédite* par le modèle du bundle (pas la progression réelle) tombe sous ce seuil
    C2: float = 0.1  # Ratio (dans (0,1)) de la progression prédite à réaliser réellement pour qu'un pas soit dit "sérieux" (voir proximal_master.serious_null_step_test)
    proximal_u_init: float = 1.0  # Paramètre proximal u (Algorithme 2, ligne 5)
    theta_drop_tol: float = 1.0e-8  # Mode dynamique seulement : retire une contrainte active si |theta_r| < ce seuil (theta = notation main.pdf, sans rapport avec alpha-CROWN)
    add_batch_size: int = 50  # Mode dynamique seulement : nb de contraintes les plus violées ajoutées par round
    max_rounds: int = 100  # Mode dynamique seulement : nb max de rounds (ajout/retrait) — pas de limite propre sinon, contrairement à max_iter qui ne borne que la boucle interne par round
    max_bundle_size: Optional[int] = None  # Nb max de coupes conservées dans le bundle (FIFO). None = pas de cap (borné naturellement par max_iter) — recommandé, cf. task-dynamic-conic-bundle.md "Analyse statique vs dynamique" (cap trop petit face à dim(theta) = arrêt prématuré). Fixer un entier pour borner le coût du master problem QP à grande échelle.
    log_theta_every_n_iter: int = 1  # Fréquence (en itérations) d'enregistrement de theta_history (0 = jamais)
    print_png: bool = False  # Si true, écrit un PNG de diagnostic par résolution (h(theta)/u/taille du bundle vs itération, cf. dynamic_conic_bundle/plotting.py) dans le dossier du run. Désactivé par défaut (coût I/O/matplotlib non négligeable sur un run à des centaines d'échantillons).


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
