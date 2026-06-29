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

    relu_relaxation_type: str = 'all'
    bound_strategy: Optional[dict] = None # Stratégie de bornes directement dans le YAML

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

    MATRIX_BY_LAYERS: Union[bool, List[List[int]]] = True
    @validator("MATRIX_BY_LAYERS", pre=True)
    def validate_and_normalize_matrix_by_layers(cls, v, values):
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
           
        raise ValueError(f"MATRIX_BY_LAYERS must be bool or List[List[int]], got {type(v)}")
    LAST_LAYER: bool = (
        False  # Whether to use the last layer of the network (logits) as variables
    )
    l_max: Optional[int] = None
    chordal_stride: int = 1
    use_fusion: bool = False  # Whether to use the fusion API for MOSEK (deprecated: use solver="mosek_fusion")
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


class GurobiSolverConfig(BaseModel):
    certification_model_type: str

    @validator("certification_model_type")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["LanQuad", "MdQuad", "MzbarQuad", "ClassicLP", "LPBoundLayer"]:
            raise ValueError(
                f"Model name {v} must be one of 'LanQuad', 'MdQuad', 'MzbarQuad','ClassicLP', 'LPBoundLayer."
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
    bounds_method: str = "IBP"


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
            if hasattr(model, 'bounds_n_runs') and model.bounds_n_runs < 1:
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
        if network is not None:
            K = network.K
            for model in v:
                if isinstance(model.MATRIX_BY_LAYERS, list):
                    last_index = model.MATRIX_BY_LAYERS[-1][-1]
                    expected_last = 7 if model.LAST_LAYER else K - 1
                    if last_index != expected_last:
                        raise ValueError(
                            f"With LAST_LAYER={model.LAST_LAYER}, last element of MATRIX_BY_LAYERS "
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
