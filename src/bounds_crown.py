import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from fastsdp_tools.cuda_probe import patch_torch_cuda_is_available

device = torch.device("cuda" if patch_torch_cuda_is_available() else "cpu")

def compute_bounds_data_crown(
    self,
    method="alpha-beta-CROWN",
):
    """
    Compute pre-activation bounds using alpha-beta-CROWN (modern API)
    """

    network = self.network.to(device).eval()

    if not torch.is_tensor(self.x):
        self.x = torch.tensor(self.x, dtype=torch.float32)

    x = self.x.view(1, -1).to(device)
    zeros = torch.zeros_like(x)

    bounded_model = BoundedModule(
        network,
        zeros,
        bound_opts={"conv_mode": "patches"}
    ).to(device)

    if self.norm == "Linf":
        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
    elif self.norm == "L2":
        ptb = PerturbationLpNorm(norm=2, eps=self.epsilon)
    else:
        raise ValueError("Norm not supported")

    bounded_x = BoundedTensor(x, ptb)

    use_crown = method.lower() in ["alpha-beta-crown", "beta-crown", "crown-optimized"]

    if use_crown:
        lb, ub, aux = bounded_model.compute_bounds(
            x=(bounded_x,),
            method="CROWN-Optimized",
            return_A=True,
            bound_lower=True,
            bound_upper=True,
            options={
                "optimize_bound_args": {
                    "ob_lr": crown_lr,
                    "ob_iteration": crown_iters,
                    "ob_verbose": 0,
                }
            }
        )
    else:
        with torch.no_grad():
            lb, ub, aux = bounded_model.compute_bounds(
                x=(bounded_x,),
                method="IBP",
                return_A=True
            )

    intermediate_bounds = aux["intermediate_bounds"]

    # Filtrer uniquement les pré-activations (avant ReLU)
    preact_bounds = {}
    for name, (l, u) in intermediate_bounds.items():
        if "preact" in name.lower() or "relu" in name.lower():
            preact_bounds[name] = (l.detach().cpu(), u.detach().cpu())

    print("\n--- PRE-ACTIVATION BOUNDS ---")
    for name, (l, u) in preact_bounds.items():
        print(f"{name:30s} | min = {l.min():.4f}, max = {u.max():.4f}")

    return lb.detach().cpu(), ub.detach().cpu(), preact_bounds