import torch
from torch.func import jacrev as _jacrev
from torch.func import grad, jacfwd, vmap


def analyze_gradient_basis(system, system_def, model, q, space, k=10):
    """
    Analyze and print gradient, model prediction, and PCA basis at current configuration.

    Args:
        system: Physics system
        system_def: System definition
        model: Trained gradient basis field model
        q: Current configuration
        space: Space parameter
        k: Number of basis vectors to analyze
    """
    dim = q.shape[0]

    # 1. Compute actual gradient
    q_grad = q.clone().requires_grad_(True)
    E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0), space.unsqueeze(0))[0]
    grad_full = torch.autograd.grad(E, q_grad)[0]  # (dim,)

    print("\n" + "=" * 60)
    print("GRADIENT BASIS ANALYSIS")
    print("=" * 60)

    # 2. Get model's predicted basis
    B_pred = model(q.unsqueeze(0)).squeeze(0)  # (dim, k)

    print(f"\nActual gradient norm: {grad_full.norm().item():.6f}")
    print(f"Gradient: {grad_full.detach().cpu().numpy()}")
    print(f"Basis: {B_pred.detach().cpu().numpy()}")

    cos_sim = (grad_full @ B_pred) / (grad_full.norm() * B_pred.norm())

    print(f"Cos Sim: {cos_sim}")
    print(f"Norm: {B_pred.norm(dim=0, keepdim=True)}")

    # 3. Project gradient onto predicted basis
    grad_coeffs_pred = B_pred.T @ grad_full  # (k,)
    grad_reconstructed_pred = B_pred @ grad_coeffs_pred  # (dim,)
    recon_error_pred = (grad_full - grad_reconstructed_pred).norm().item()

    print(f"\n--- MODEL PREDICTION ---")
    print(f"Predicted basis shape: {B_pred.shape}")
    print(f"Gradient coefficients in predicted basis: {grad_coeffs_pred.detach().cpu().numpy()}")
    print(f"Reconstruction error: {recon_error_pred:.6f}")
    print(f"Explained variance: {1 - (recon_error_pred / grad_full.norm().item()) ** 2:.4f}")

    # Check orthonormality
    G = B_pred.T @ B_pred
    ortho_error = (G - torch.eye(G.shape[0])).norm().item()
    print(f"Orthonormality error (||B^T B - I||): {ortho_error:.6e}")

    # 4. Compute PCA basis from local gradient samples
    # Sample gradients near current configuration
    n_samples = 1000
    perturbations = torch.randn(n_samples, dim) * 0.01
    q_samples = q.unsqueeze(0) + perturbations  # (n_samples, dim)

    # Compute gradients at samples
    gradients_list = []
    for i in range(n_samples):
        q_sample = q_samples[i].requires_grad_(True)
        E_sample = system.potential_energy_batch(system_def, q_sample.unsqueeze(0), space.unsqueeze(0))[0]
        grad_sample = torch.autograd.grad(E_sample, q_sample)[0]
        gradients_list.append(grad_sample)

    gradients_samples = torch.stack(gradients_list, dim=0)  # (n_samples, dim)

    # Compute PCA
    mean_grad = gradients_samples.mean(dim=0)
    centered = gradients_samples - mean_grad
    cov = (centered.T @ centered) / (n_samples - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    V_pca = eigenvectors[:, :k]  # (dim, k)

    print(f"\n--- PCA BASIS (from {n_samples} local samples) ---")
    print(f"Top {k} eigenvalues: {eigenvalues[:k].detach().cpu().numpy()}")
    print(f"Explained variance ratio: {(eigenvalues[:k].sum() / eigenvalues.sum()).item():.4f}")

    # Project gradient onto PCA basis
    grad_coeffs_pca = V_pca.T @ grad_full
    grad_reconstructed_pca = V_pca @ grad_coeffs_pca
    recon_error_pca = (grad_full - grad_reconstructed_pca).norm().item()

    print(f"Gradient coefficients in PCA basis: {grad_coeffs_pca.detach().cpu().numpy()}")
    print(f"Reconstruction error: {recon_error_pca:.6f}")

    # 5. Compare predicted basis to PCA basis
    # Subspace distance (order-invariant)
    P_pred = B_pred @ B_pred.T
    P_pca = V_pca @ V_pca.T
    subspace_distance = (P_pred - P_pca).norm().item()

    print(f"\n--- COMPARISON ---")
    print(f"Subspace distance (||P_pred - P_pca||): {subspace_distance:.6f}")

    # Grassmann distance
    M = B_pred.T @ V_pca
    grassmann_dist = torch.sqrt(k - (M ** 2).sum() + 1e-8).item()
    print(f"Grassmann distance: {grassmann_dist:.6f}")

    print("=" * 60 + "\n")

