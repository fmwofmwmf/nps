import torch
from integrators import *

def batch_gram_schmidt(V):
    """
    Differentiable, autograd-safe batch Gram-Schmidt.
    Args:
        V: (B, dim, k)
    Returns:
        U: (B, dim, k)
    """
    B, dim, k = V.shape
    U_list = []

    for i in range(k):
        u_i = V[:, :, i]  # (B, dim)
        for j, u_j in enumerate(U_list):
            proj = (u_i * u_j).sum(dim=1, keepdim=True) * u_j
            u_i = u_i - proj

        norm = torch.sqrt((u_i ** 2).sum(dim=1, keepdim=True) + 1e-8)
        u_i = u_i / norm
        U_list.append(u_i)

    # Stack along last dimension
    U = torch.stack(U_list, dim=2)  # (B, dim, k)
    return U

# Alternative: Use QR decomposition (more stable)
def batch_orthonormalize_qr(V):
    """
    More numerically stable orthonormalization using QR.

    Args:
        V: (B, dim, k) - batch of vectors

    Returns:
        Q: (B, dim, k) - orthonormal basis
    """
    B = V.shape[0]
    Q_list = []

    for i in range(B):
        Q, R = torch.linalg.qr(V[i])  # V[i] is (dim, k)
        Q_list.append(Q)

    return torch.stack(Q_list, dim=0)  # (B, dim, k)

def compare_integration_error(
    system,
    system_def,
    q,
    basis,
    space,
    n_samples=32,
    vel_scale=1e-2,
):
    state_to_system = lambda system_def, z, space: z

    model = lambda q: basis

    errors = []
    mags = []
    in_basis_ratios = []

    for _ in range(n_samples):

        q_dot = vel_scale * torch.randn_like(q)

        q_full, q_dot_full, _ = latent_step_newton_batch(
            system,
            system_def,
            state_to_system,
            q,
            q_dot,
            space,
        )

        q_red, q_dot_red, _ = gradient_basis_step_newton_new(
            system,
            system_def,
            model,
            q,
            q_dot,
            space,
        )

        delta_full = q_full - q
        delta_red = q_red - q

        errors.append((q_full - q_red).norm())
        mags.append(delta_full.norm())

        proj = basis @ (basis.T @ delta_full)

        in_basis = proj.norm() / (delta_full.norm() + 1e-8)
        in_basis_ratios.append(in_basis)

    errors = torch.stack(errors)
    mags = torch.stack(mags)
    in_basis_ratios = torch.stack(in_basis_ratios)

    print(f"Mean error: {errors.mean()}")
    print(f"Mag: {mags.mean()}")
    print(f"Avg error: {errors.mean() / mags.mean() * 100:.2f}%")
    print(f"Update in basis: {in_basis_ratios.mean()}")

def compare_integration_error_single(
        system,
        system_def,
        q,
        basis,
        space,
        q_dot,
):
    """
    Compare integration error for a single given velocity.
    Also prints residuals from the Newton solver.

    Returns:
        error, mag, in_basis, residual_full, residual_red
    """

    state_to_system = lambda system_def, z, space: z
    model = lambda q: basis

    # -------------------------
    # Full step (with residual)
    # -------------------------
    q_full, q_dot_full, res_full = latent_step_newton_batch(
        system,
        system_def,
        state_to_system,
        q,
        q_dot,
        space,
    )

    # -------------------------
    # Reduced step (with residual)
    # -------------------------
    q_red, q_dot_red, res_red = gradient_basis_step_newton_new(
        system,
        system_def,
        model,
        q,
        q_dot,
        space,
    )

    # -------------------------
    # Integration error
    # -------------------------
    delta_full = q_full - q
    delta_red = q_red - q

    error = (delta_full - delta_red).norm()
    mag = delta_full.norm()

    proj = basis @ (basis.T @ delta_full)
    in_basis = proj.norm() / (mag + 1e-8)

    # -------------------------
    # Print debug info
    # -------------------------
    print(f"Reduced sim error: {error}")
    print(f"Mag (full space): {mag}")
    print(f"Relative error: {error / (mag + 1e-8) * 100:.2f}")
    print(f"% update in basis: {in_basis * 100:.2f}")

    delta_unit = delta_full / (delta_full.norm() + 1e-12)  # unit vector
    basis_optimal = add_basis(delta_unit.unsqueeze(1), get_gradient(system, system_def, q, space))   # (dim, 1)

    model_opt = lambda q: basis_optimal
    q_red_opt, _, res_opt = gradient_basis_step_newton_new(
        system,
        system_def,
        model_opt,
        q,
        q_dot,
        space,
    )
    delta_red_opt = q_red_opt - q
    error_opt = (delta_full - delta_red_opt).norm()
    proj_opt = basis_optimal @ (basis_optimal.T @ delta_full)
    in_basis_opt = proj_opt.norm() / (mag + 1e-8)

    print("\nReduced basis from full step:")
    print(f"Error: {error_opt}")
    print(f"Relative error (%): {error_opt / (mag + 1e-8) * 100:.2f}")

    print("\nResiduals from Newton solver:")
    print(f"Full residual:    {res_full:.6e}")
    print(f"Reduced residual: {res_red:.6e}")
    print(f"Full Reduced residual: {res_opt:.6e}")

    return error / (mag + 1e-8) * 100

def compare_integration_error_basic(system, system_def, q, q_dot, q1, q_dot1, basis, space):
    state_to_system = lambda system_def, z, space: z  # identity mapping for full space

    # Run full Newton step
    q_full, q_dot_full, _ = latent_step_newton(
        system,
        system_def,
        state_to_system,
        q,
        q_dot,
        space,
    )

    # Compute errors against reference
    pos_error_abs = (q_full - q1).norm()
    vel_error_abs = (q_dot_full - q_dot1).norm()

    # Compute update magnitudes for relative error
    delta_q = (q_full - q)
    delta_q_dot = (q_dot_full - q_dot)

    pos_error_pct = pos_error_abs / (delta_q.norm() + 1e-12) * 100
    vel_error_pct = vel_error_abs / (delta_q_dot.norm() + 1e-12) * 100

    proj_q = basis @ (basis.T @ delta_q)
    proj_q_dot = basis @ (basis.T @ delta_q_dot)

    pos_in_basis_pct = proj_q.norm().item() / (delta_q.norm().item() + 1e-12) * 100
    vel_in_basis_pct = proj_q_dot.norm().item() / (delta_q_dot.norm().item() + 1e-12) * 100

    return pos_error_abs.item(), vel_error_abs.item(), pos_error_pct.item(), vel_error_pct.item(), pos_in_basis_pct, vel_in_basis_pct

def gradient_pca(system, system_def, q, k, space, add_gradient=True):
    dim = q.shape[0]
    M_phys = system.physical_mass_matrix(system_def, q)

    q_grad = q.clone().requires_grad_(True)
    E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0),
                                      space.unsqueeze(0))[0]

    # Compute gradient and Hessian
    grad = torch.autograd.grad(E, q_grad, create_graph=True)[0]

    # Hessian of potential energy
    H = torch.zeros(dim, dim, device=q.device, dtype=q.dtype)
    for i in range(dim):
        H[i] = torch.autograd.grad(grad[i], q_grad, retain_graph=True)[0]

    # Eigendecomposition
    L = torch.linalg.cholesky(M_phys)
    L_inv = torch.linalg.inv(L)
    H_transformed = L_inv @ H @ L_inv.T
    eigenvalues, eigenvectors_w = torch.linalg.eigh(H_transformed)

    U = L_inv.T @ eigenvectors_w
    # eigenvalues, eigenvectors = torch.linalg.eigh(H)
    U = U[:, :k]

    if add_gradient:
        U = torch.cat((U, grad.unsqueeze(1)), dim=1)
    U, _ = torch.linalg.qr(U)

    return U
def local_pca_one_step_random(
    system,
    system_def,
    q0,
    space,
    num_samples=32,
    vel_scale=1e-2,
    k=2
):
    state_to_system = lambda _, z, __: z

    def single_step(z_dot):
        z = q0
        z_new, _, __ = latent_step_newton_batch(
            system,
            system_def,
            state_to_system,
            z,
            z_dot,
            space,
            max_iters=1
        )
        return z_new - q0

    # Sample all velocities at once
    z_dots = vel_scale * torch.nn.functional.normalize(torch.randn(num_samples, *q0.shape, device=q0.device),dim=1)
    # z_dots = vel_scale * torch.randn(num_samples, *q0.shape, device=q0.device)
    # Vectorize over velocity dimension
    offsets = torch.vmap(single_step)(z_dots)

    # -------------------------------------------------
    # PCA via SVD
    # -------------------------------------------------
    offsets = offsets

    U, S, Vh = torch.linalg.svd(offsets, full_matrices=False)

    basis = Vh[:k].T  # (dim, k)

    return basis

def add_basis(basis, gradient):
    grad_col = gradient.unsqueeze(1)  # (dim, 1)
    new_basis = torch.cat([basis, grad_col], dim=1)  # (dim, k+1)

    new_basis_batch = new_basis.unsqueeze(0)  # (1, dim, k+1)

    ortho_batch = batch_gram_schmidt(new_basis_batch)  # (1, dim, k+1)

    return ortho_batch.squeeze(0)  # (dim, k+1)

def get_gradient(
    system,
    system_def,
    q0,
    space,
):
    q_grad = q0.clone().requires_grad_(True)
    E = system.potential_energy_batch(system_def, q_grad.unsqueeze(0),
                                      space.unsqueeze(0))[0]

    # Compute gradient and Hessian
    grad = torch.autograd.grad(E, q_grad, create_graph=True)[0]

    return grad