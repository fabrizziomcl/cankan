import os
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import numpy as np

def create_experiment_folder(exp_name):
    base_path = f"./experiments/{exp_name}"
    os.makedirs(f"{base_path}/plots", exist_ok=True)
    return base_path

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_loss_evolution(results_dict, exp_name, save_path):
    plt.figure(figsize=(12, 8))
    plt.yscale('log')
    
    colors = {
        'Adam': '#1f77b4',  # Azul
        'LBFGS': '#d62728', # Rojo
        'SGD': '#2ca02c'    # Verde
    }
    
    for name, res in results_dict.items():
        if 'KAN' in name:
            style = '-'
            width = 2.5
            alpha = 1.0
        elif 'MLP_Large' in name:
            style = '--'
            width = 1.5
            alpha = 0.7
        elif 'MLP_Small' in name:
            style = ':'
            width = 1.5
            alpha = 0.9
            
        opt_name = name.split('_')[-1]
        c = colors.get(opt_name, 'black')
        
        plt.plot(res['test_loss'], label=name, linestyle=style, color=c, linewidth=width, alpha=alpha)

    plt.title(f"Test Loss: {exp_name}", fontsize=14)
    plt.xlabel("Steps")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_path}/plots/loss_benchmark.png", dpi=150)
    plt.close()

def plot_function_fit(model, dataset, model_name, save_path):
    """Fit 1D (Line) o Parity Plot (>1D)"""
    x_test = dataset['test_input']
    y_true = dataset['test_label'].detach().numpy()
    
    with torch.no_grad():
        y_pred = model(x_test).detach().numpy()
    
    input_dim = x_test.shape[1]
    
    plt.figure(figsize=(8, 6))
    
    if input_dim == 1:
        # 1D: Curvas
        x_np = x_test.detach().numpy().flatten()
        sort_idx = np.argsort(x_np)
        plt.scatter(x_np[sort_idx], y_true[sort_idx], label="Ground Truth", color='gray', alpha=0.4, s=15)
        plt.plot(x_np[sort_idx], y_pred[sort_idx], label="Prediction", color='red', linewidth=2)
        plt.xlabel("Input X")
        plt.ylabel("Output Y")
    else:
        # >1D: Parity Plot
        plt.scatter(y_true, y_pred, alpha=0.5, c='blue', s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
    
    plt.title(f"Fit Quality: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/plots/fit_parity_{model_name}.png", dpi=100)
    plt.close()

def plot_3d_surface_comparison(model, func, ranges, model_name, save_path):
    """
    Genera una comparación 3D: Real vs Predicción vs Error.
    Solo funciona para funciones de 2 variables.
    """
    # Crear grid denso
    resolution = 50
    x1 = np.linspace(ranges[0], ranges[1], resolution)
    x2 = np.linspace(ranges[0], ranges[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Aplanar para pasar por el modelo
    grid_flat = np.vstack([X1.ravel(), X2.ravel()]).T
    grid_tensor = torch.tensor(grid_flat, dtype=torch.float64)
    
    # Calcular valores
    with torch.no_grad():
        Z_true = func(grid_tensor).numpy().reshape(X1.shape)
        Z_pred = model(grid_tensor).detach().numpy().reshape(X1.shape)
    
    Z_error = np.abs(Z_true - Z_pred)

    # Configurar figura
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"Surface Analysis: {model_name}", fontsize=16)

    # Plot 1: Ground Truth
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, Z_true, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.8)
    ax1.set_title("Ground Truth Function")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Plot 2: Prediction
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X1, X2, Z_pred, cmap=cm.plasma, linewidth=0, antialiased=False, alpha=0.8)
    ax2.set_title("Model Prediction")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # Plot 3: Error Heatmap (Vista cenital para ver dónde falla)
    ax3 = fig.add_subplot(1, 3, 3)
    # Usamos pcolormesh para mapa de calor 2D del error
    im = ax3.pcolormesh(X1, X2, Z_error, cmap='hot', shading='auto')
    ax3.set_title(f"Absolute Error (Max: {Z_error.max():.2e})")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    fig.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/plots/surface_3d_{model_name}.png", dpi=150)
    plt.close()