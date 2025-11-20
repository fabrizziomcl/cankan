import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN, create_dataset
from mlp import SimpleMLP
from utils import create_experiment_folder, plot_loss_evolution, plot_function_fit, plot_3d_surface_comparison, count_params

# Configuración Global
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

# --- DEFINICIÓN DE EXPERIMENTOS ---
experiments = [
    (
        "1_Additive_Simple_2D",
        lambda x: x[:, [0]]**2 + 0.5 * x[:, [1]],
        2,
        [-2, 2],
        [2, 1], 5, 3, 300
    ),
    (
        "2_Paper_Compositional",
        lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2),
        2,
        [-1, 1],
        [2, 1, 1], 10, 3, 300
    ),
    (
        "3_Multiplication_Structural",
        lambda x: x[:, [0]] * x[:, [1]],
        2,
        [-2, 2],
        [2, 2, 1], 10, 3, 300
    ),
    (
        "4_Additive_4D_Interpretable",
        lambda x: torch.exp(x[:, [0]]/2) + torch.sin(torch.pi * x[:, [1]]) + x[:, [2]]**2 - x[:, [3]],
        4,
        [-2, 2],
        [4, 1], 10, 3, 300
    ),
    (
        "5_Rosenbrock_Valley",
        # El mínimo global está en (1, 1).
        lambda x: (1 - x[:, [0]])**2 + 100 * (x[:, [1]] - x[:, [0]]**2)**2,
        2,
        [-2, 2],
        [2, 4, 1], 20, 3, 300
    )
]

# --- CONFIGURACIÓN OPTIMIZADORES ---
mlp_opts = ["Adam", "SGD", "LBFGS"]
kan_opts = ["Adam", "LBFGS"] 

def calculate_mlp_hidden_for_params(n_in, n_out, target_params, depth):
    best_h = 1
    min_diff = float('inf')
    for h in range(1, 5000):
        dummy = SimpleMLP(n_in, h, n_out, depth=depth)
        p = count_params(dummy)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params: break
    return best_h

def run_benchmark():
    for name, func, n_var, r_range, k_width, k_grid, k_k, steps in experiments:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTO: {name} (Vars: {n_var})")
        print(f"{'='*80}")
        
        path = create_experiment_folder(name)
        dataset = create_dataset(func, n_var=n_var, train_num=1000, test_num=1000, ranges=r_range)
        all_results = {}
        
        # ==========================================
        # 1. KAN
        # ==========================================
        ref_kan = KAN(width=k_width, grid=k_grid, k=k_k, seed=42)
        kan_params = count_params(ref_kan)
        print(f"  [KAN Info] Params: {kan_params} | Grid: {k_grid}")

        for opt in kan_opts:
            print(f"  -> Entrenando KAN ({opt})...")
            
            kan_model = KAN(width=k_width, grid=k_grid, k=k_k, seed=42)
            kan_model(dataset['train_input']) 
            
            # Ajuste de LR para Adam en KAN
            lr = 0.01 if opt == "Adam" else 1.0
            res = kan_model.fit(dataset, opt=opt, steps=steps, lamb=0.001, lr=lr)
            all_results[f"KAN_{opt}"] = res
            
            # Plots
            plot_function_fit(kan_model, dataset, f"KAN_{opt}", path)
            if n_var == 2:
                plot_3d_surface_comparison(kan_model, func, r_range, f"KAN_{opt}", path)
            
            if opt == "LBFGS":
                plt.figure(figsize=(10, 8))
                kan_model.plot(beta=10, scale=1.0)
                plt.title(f"KAN Structure: {name}")
                plt.savefig(f"{path}/plots/kan_structure.png", dpi=150)
                plt.close()

        # ==========================================
        # 2. MLPs
        # ==========================================
        mlp_depth = len(k_width) - 1 
        if mlp_depth < 2: mlp_depth = 2

        small_hidden = calculate_mlp_hidden_for_params(n_var, 1, kan_params, depth=mlp_depth)
        
        mlp_configs = [
            ("MLP_Large", 128),         
            ("MLP_Small", small_hidden)
        ]
        
        for m_type, h_dim in mlp_configs:
            for opt in mlp_opts:
                mlp = SimpleMLP(n_var, h_dim, 1, depth=mlp_depth)
                p_count = count_params(mlp)
                
                print(f"  -> Entrenando {m_type} ({opt})...")
                
                lr = 0.1 if opt == "SGD" else 0.01
                res = mlp.fit(dataset, opt=opt, steps=steps, lr=lr)
                all_results[f"{m_type}_{opt}"] = res
                
                # Plots
                plot_function_fit(mlp, dataset, f"{m_type}_{opt}", path)
                if n_var == 2:
                    plot_3d_surface_comparison(mlp, func, r_range, f"{m_type}_{opt}", path)

        # ==========================================
        # 3. Resultados Finales
        # ==========================================
        plot_loss_evolution(all_results, name, path)
        
        if "LBFGS" in kan_opts:
            try:
                print("  [Symbolic] KAN LBFGS...")
                lib = ['x', 'x^2', 'sin', 'cos', 'exp', 'log']
                kan_model.auto_symbolic(lib=lib)
                formula = kan_model.symbolic_formula()[0][0]
                print(f"  >>> Fórmula: {formula}")
                with open(f"{path}/formula.txt", "w") as f:
                    f.write(f"Discovered: {str(formula)}\n")
            except Exception as e:
                print(f"  (Skip symbolic: {e})")

if __name__ == "__main__":
    run_benchmark()