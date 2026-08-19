
import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from fastsdp_tools.cuda_probe import patch_torch_cuda_is_available

device = torch.device("cuda" if patch_torch_cuda_is_available() else "cpu")

def compute_bounds_data_crown(
    self,
    method="alpha-beta-crown",
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
        # ÉTAPE 1 : Initialiser avec IBP pour avoir toutes les bornes
        print("Step 1: Computing initial bounds with IBP...")
        with torch.no_grad():
            lb_ibp, ub_ibp = bounded_model.compute_bounds(
                x=(bounded_x,),
                method="IBP",
                bound_lower=True,
                bound_upper=True,
            )
        
        # ÉTAPE 2 : Optimiser avec CROWN (raffine les bornes)
        print("Step 2: Optimizing bounds with CROWN...")
        lb, ub = bounded_model.compute_bounds(
            x=(bounded_x,),
            method="CROWN-Optimized",
            bound_lower=True,
            bound_upper=True,
        )
        
        # Vérifier si les bornes sont bien calculées
        if ub is None:
            print("Warning: CROWN didn't compute upper bounds, using IBP bounds")
            ub = ub_ibp
        if lb is None:
            print("Warning: CROWN didn't compute lower bounds, using IBP bounds")
            lb = lb_ibp
            
    else:
        with torch.no_grad():
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_x,),
                method="IBP",
                bound_lower=True,
                bound_upper=True,
            )

    # Extraire les bornes intermédiaires
    preact_bounds = {}
    
    print("\n--- Extracting intermediate bounds ---")
    
    k = 0
    L = []
    U = []
    for i, node in enumerate(bounded_model.nodes()):
        node_type = type(node).__name__
        print("Node type : ", node_type, "node : ", node)
        
        
        required_types = ['Input', 'Linear']
        assert k>0 or 'Input' in node_type, f"Expected Input node at index 0, got {node_type}"
        if any(el in node_type for el in required_types):
            
            if hasattr(node, 'lower') and hasattr(node, 'upper'):
                lower = node.lower
                upper = node.upper
                
                # Vérifier que les bornes existent
                if lower is not None and upper is not None:
                    lower_cpu = lower.squeeze().detach().cpu()
                    upper_cpu = upper.squeeze().detach().cpu()
                    # if k == 0:
                    #     lower_cpu = torch.clamp(lower_cpu, min=0)
                    preact_bounds[k] = (lower_cpu, upper_cpu)
                    print(f"✓ {node.name}: lower ∈ [{lower_cpu.min():.4f}, {lower_cpu.max():.4f}], upper ∈ [{upper_cpu.min():.4f}, {upper_cpu.max():.4f}]")
                    L.append(lower_cpu.tolist())
                    U.append(upper_cpu.tolist())
                
                elif lower is not None and upper is None:
                    print(f"✗ {node.name}: lower exists but upper is None")
                    # Essayer de calculer la borne supérieure manuellement
                    print(f"  Attempting to compute upper bound manually...")
                    
                    layer_relu_before = node.inputs[0]
                    layer_linear_before = layer_relu_before.inputs[0]
                    layer_before_upper = layer_linear_before.upper
                    layer_before_lower = layer_linear_before.lower  
                    device_ = layer_before_upper.device
                    dtype = layer_before_upper.dtype
                    
                    W_pos = torch.clamp(torch.tensor(self.W[k-1], device=device_, dtype=dtype), min=0)
                    W_neg = torch.clamp(torch.tensor(self.W[k-1], device=device_, dtype=dtype), max=0)
                    bias = torch.tensor(self.b[k-1], device=device_, dtype=dtype).unsqueeze(dim=1)
                    estimated_upper = W_pos @ torch.transpose(layer_before_upper, 0, 1) + W_neg @ torch.transpose(layer_before_lower, 0, 1) + bias

                    preact_bounds[k] = (
                        lower.squeeze().detach().cpu(),
                        estimated_upper.squeeze().detach().cpu()
                    )
                    L.append(lower.squeeze().detach().cpu().tolist())  # Attention au squeeze dans le cas d'une couche convolutionnelle
                    U.append(estimated_upper.squeeze().detach().cpu().tolist())

                elif lower is None and upper is not None:
                    print(f"✗ {node.name}: upper exists but lower is None")
                    
                else:
                    print(f"✗ {node.name}: both bounds are None")

        if 'Relu' in node_type or 'Input' in node_type:
            k += 1
            print(f"Relu layer/Input detected, moving to next layer index: {k}")


    # Vérification finale
    print(f"\n--- Summary ---")
    print(f"Total nodes with bounds: {len(preact_bounds)}")
    print(f"Output bounds: lb shape = {lb.shape}, ub shape = {ub.shape}")
    print(f"Output range: [{lb.min():.4f}, {ub.max():.4f}]")

    if preact_bounds:
        print("\n--- PRE-ACTIVATION BOUNDS ---")
        for k, (l, u) in preact_bounds.items():
            print(f"Layer {k:2d} | lower: [{l.min():.4f}, {l.max():.4f}], upper: [{u.min():.4f}, {u.max():.4f}]")
    else:
        print("\n⚠ Warning: No intermediate bounds extracted!")
        
    self.L = L
    self.U = U

    self.compute_bounds_time = None

    return