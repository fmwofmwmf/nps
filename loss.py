import torch
from torch.func import vmap, grad
def landscape_loss(model, system, system_def, z_batch, q_batch, shape, t_schedule, device):
    # lip
    weight_landscape_reg = 5e-1
    B, d = z_batch.shape

    # ----------------------------
    # Random probe vector
    # ----------------------------
    v = torch.randn(d, device=device)  # [d]
    v = v / v.norm()

    # expand to batch dimension for vmap
    v_exp = v.unsqueeze(0).expand(B, -1)  # [B, d]

    # ----------------------------
    # Define energy and HVP functions
    # ----------------------------
    def energy_single(z, shape):
        # z: [d], shape: [s]
        input_vec = torch.cat([z, shape], dim=-1).unsqueeze(0)  # [1, d+s]
        q = model(input_vec, t_schedule=t_schedule)
        E = system.potential_energy_batch(system_def, q, shape.unsqueeze(0))
        return E.squeeze()

    grad_energy = grad(energy_single)

    def hvp_single(z, shape, v):
        # returns H(z) @ v
        return grad(lambda zz: grad_energy(zz, shape).dot(v))(z)

    # ----------------------------
    # Compute HVPs for all batch samples
    # ----------------------------
    Hv = vmap(hvp_single, randomness="different")(z_batch, shape, v_exp)  # [B, d]

    # ----------------------------
    # Pairwise Hessian distance estimate
    # ----------------------------
    Hv1 = Hv.unsqueeze(1)  # [B, 1, d]
    Hv2 = Hv.unsqueeze(0)  # [1, B, d]
    hessians_sqdist_mat = ((Hv1 - Hv2) ** 2).sum(dim=-1)  # [B, B]

    # ----------------------------
    # Pairwise latent distances
    # ----------------------------
    Z1 = z_batch.unsqueeze(1)  # [B, 1, d]
    Z2 = z_batch.unsqueeze(0)  # [1, B, d]
    latents_sqdist_mat = ((Z1 - Z2) ** 2).sum(dim=-1).clamp(min=1e-5)  # [B, B]

    # ----------------------------
    # Landscape loss
    # ----------------------------
    sqslope_mat = hessians_sqdist_mat / latents_sqdist_mat
    mask = ~torch.eye(B, dtype=torch.bool, device=device)
    return sqslope_mat[mask].mean()

def curvature_loss(model, system, system_def, z_batch, q_batch, shape, t_schedule, device):
    epsilon = 1e-3
    delta = 1e-6

    v = torch.randn_like(z_batch)
    v = torch.clamp(v, -1.0, 1.0)

    g = torch.autograd.grad(
        q_batch.sum(),
        z_batch,
        create_graph=True
    )[0]

    g_eps = torch.autograd.grad(
        model(z_batch + epsilon * v).sum(),
        z_batch,
        create_graph=True
    )[0]

    Hv = (g_eps - g) / epsilon

    numer = Hv.pow(2).sum(dim=1)
    denom = g.pow(2).sum(dim=1) + delta

    return (numer / denom).mean()

def loss_multiscale_curvature(
    model, system, system_def,
    z_batch, q_batch, shape,
    t_schedule, device
):
    """
    Penalizes curvature across multiple spatial scales.
    Prevents hiding curvature in sharp cliffs.
    """

    epsilons = [1e-3, 3e-3, 1e-2]
    B = z_batch.shape[0]

    # Base gradient
    g0 = torch.autograd.grad(
        q_batch.sum(),
        z_batch,
        create_graph=True
    )[0]

    # Random normalized directions
    v = torch.randn_like(z_batch)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    loss = 0.0
    for eps in epsilons:
        z_pert = z_batch + eps * v
        q_pert = model(z_pert)

        g_eps = torch.autograd.grad(
            q_pert.sum(),
            z_batch,
            create_graph=True
        )[0]

        Hv = (g_eps - g0) / eps
        loss = loss + Hv.pow(2).sum(dim=1).mean()

    loss = loss / len(epsilons)
    return loss


def loss_gradient_smoothness(
    model, system, system_def,
    z_batch, q_batch, shape,
    t_schedule, device
):
    """
    Directly penalizes sharp gradient jumps.
    Stronger and more stable than Hessian penalties.
    """

    delta = 5e-3

    dz = torch.randn_like(z_batch)
    dz = dz / (dz.norm(dim=1, keepdim=True) + 1e-8)
    dz = delta * dz

    g1 = torch.autograd.grad(
        q_batch.sum(),
        z_batch,
        create_graph=True
    )[0]

    q2 = model(z_batch + dz)
    g2 = torch.autograd.grad(
        q2.sum(),
        z_batch,
        create_graph=True
    )[0]

    loss = (g2 - g1).pow(2).sum(dim=1).mean()
    return loss


def loss_gradient_lipschitz(
    model, system, system_def,
    z_batch, q_batch, shape,
    t_schedule, device
):
    """
    Enforces bounded forces.
    Prevents infinite cliffs and Newton blowups.
    """

    L = 10.0  # max allowed gradient norm

    g = torch.autograd.grad(
        q_batch.sum(),
        z_batch,
        create_graph=True
    )[0]

    grad_norm = g.norm(dim=1)
    loss = torch.relu(grad_norm - L).pow(2).mean()

    return loss

def orthonormality_loss(B_batch):
    """
    Soft constraint: penalize deviation from orthonormality.

    For orthonormal basis: B^T @ B = I_k

    Args:
        B_batch: (B, dim, k) - batch of bases

    Returns:
        loss: scalar
    """
    # Compute Gram matrix: B^T @ B
    G = torch.bmm(B_batch.transpose(1, 2), B_batch)  # (B, k, k)

    # Target: identity matrix
    k = B_batch.shape[2]
    I_k = torch.eye(k, device=B_batch.device, dtype=B_batch.dtype)
    I_k = I_k.unsqueeze(0).expand(B_batch.shape[0], k, k)  # (B, k, k)

    # Penalize deviation
    diff = G - I_k  # (B, k, k)
    loss = (diff ** 2).sum(dim=(1, 2)).mean()  # Mean over batch

    return loss


def subspace_distance_loss(B_pred, B_target):
    """
    Order-invariant subspace distance using projection matrices.

    Distance between two k-dimensional subspaces is measured by
    the Frobenius norm of the difference of their projection matrices.

    This is invariant to:
    - Ordering of basis vectors
    - Rotation within the subspace
    - Sign flips of basis vectors

    Args:
        B_pred: (B, dim, k) - predicted bases
        B_target: (B, dim, k) - target PCA bases

    Returns:
        loss: scalar
    """
    if B_pred.dim() != 3 or B_target.dim() != 3:
        raise ValueError(
            f"Expected tensors of shape (B, dim, k), "
            f"got {B_pred.shape} and {B_target.shape}"
        )

    Bp, dim_p, k_p = B_pred.shape
    Bt, dim_t, k_t = B_target.shape

    if Bp != Bt or dim_p != dim_t:
        raise ValueError(
            f"Batch or ambient dimension mismatch: "
            f"{B_pred.shape} vs {B_target.shape}"
        )

    if k_p != k_t:
        raise ValueError(
            f"Subspace dimension mismatch: "
            f"pred k={k_p}, target k={k_t}"
        )

    # Compute projection matrices P = B @ B^T
    # P projects onto the subspace spanned by columns of B

    P_pred = torch.bmm(B_pred, B_pred.transpose(1, 2))  # (B, dim, dim)
    P_target = torch.bmm(B_target, B_target.transpose(1, 2))  # (B, dim, dim)

    # Frobenius norm of difference
    diff = P_pred - P_target  # (B, dim, dim)
    loss = (diff ** 2).sum(dim=(1, 2)) # Mean over batch

    return loss


def eigenvalue_weighted_subspace_loss(B_pred, B_target, eigenvalues, eigenvectors):
    """
    Subspace distance loss weighted by eigenvalues.

    Penalizes errors more heavily when they point toward stiff directions.

    Args:
        B_pred: (B, dim, k) - predicted bases
        B_target: (B, dim, k) - target PCA bases
        eigenvalues: (B, dim) - eigenvalues of Hessian (sorted ascending)
        eigenvectors: (B, dim, dim) - eigenvectors of Hessian

    Returns:
        loss: (B,) - weighted loss per sample
    """
    B, dim, k = B_pred.shape

    # Compute projection matrices
    P_pred = torch.bmm(B_pred, B_pred.transpose(1, 2))  # (B, dim, dim)
    P_target = torch.bmm(B_target, B_target.transpose(1, 2))  # (B, dim, dim)

    # Difference in projections (error directions)
    diff = P_pred - P_target  # (B, dim, dim)

    # Transform difference to eigenvector basis
    # V^T @ diff @ V gives the error in the eigenvector coordinate system
    diff_eigen = torch.bmm(eigenvectors.transpose(1, 2),
                           torch.bmm(diff, eigenvectors))  # (B, dim, dim)

    # Weight each eigendirection by its eigenvalue
    # Errors in stiff directions (high eigenvalue) are penalized more
    Lambda = torch.diag_embed(eigenvalues)  # (B, dim, dim)

    # Weighted error in eigenspace: Lambda @ diff_eigen
    # We want || Lambda^(1/2) @ diff_eigen ||_F^2
    weighted_diff_eigen = torch.diag_embed(torch.sqrt(torch.clamp(eigenvalues, min=0.0) + 1e-12)) @ diff_eigen

    loss = (weighted_diff_eigen ** 2).sum(dim=(1, 2))  # (B,)

    return loss
def fake_gradient_pca(system, system_def, q, k, space, add_gradient=True):
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

    if add_gradient:
        U = torch.cat((U[:, :k], grad.unsqueeze(1)), dim=1)
    U, _ = torch.linalg.qr(U)

    return U

import torch
from torch.func import hessian, vmap


def _energy_single(system, system_def, q_i, space_i):
    """
    Helper for vmap:
    computes scalar energy for a single sample.
    """
    return system.potential_energy_batch(
        system_def,
        q_i.unsqueeze(0),
        space_i.unsqueeze(0)
    )[0]


def fake_gradient_pca_batched_vmap(system, system_def, q, k, space):
    """
    Batched fake-gradient PCA using vmap + hessian.

    Inputs:
        q      : (B, dim)
        space  : (B, ...)
    Returns:
        U      : (B, dim, k)
    """

    # -------------------------------------------------
    # 1️⃣ Mass matrices (batched)
    # -------------------------------------------------
    M_phys = system.physical_mass_matrix(system_def, q)  # (B, dim, dim)

    # -------------------------------------------------
    # 2️⃣ Energy via vmap
    # -------------------------------------------------
    energies = vmap(_energy_single, in_dims=(None, None, 0, 0))(
        system, system_def, q, space
    )  # (B,)

    # -------------------------------------------------
    # 3️⃣ Gradient via autograd
    # -------------------------------------------------
    q_grad = q.clone().requires_grad_(True)

    energies2 = system.potential_energy_batch(
        system_def,
        q_grad,
        space
    )  # (B,)

    grad = torch.autograd.grad(
        energies2.sum(),
        q_grad,
        create_graph=True
    )[0]  # (B, dim)

    # -------------------------------------------------
    # 4️⃣ Hessian via vmap(hessian)
    # -------------------------------------------------
    def hess_fn(q_i, space_i):
        return hessian(lambda x: _energy_single(system, system_def, x, space_i))(q_i)

    H = vmap(hess_fn)(q, space)  # (B, dim, dim)

    # -------------------------------------------------
    # 5️⃣ Mass-whiten Hessian
    # -------------------------------------------------
    L = torch.linalg.cholesky(M_phys)
    L_inv = torch.linalg.inv(L)

    H_transformed = L_inv @ H @ L_inv.transpose(-1, -2)

    # -------------------------------------------------
    # 6️⃣ Eigen decomposition
    # -------------------------------------------------
    eigenvalues, eigenvectors_w = torch.linalg.eigh(H_transformed)

    eigenvectors = L_inv.transpose(-1, -2) @ eigenvectors_w

    # -------------------------------------------------
    # 7️⃣ Build subspace
    # -------------------------------------------------
    smallest_modes = eigenvectors[:, :, :k - 1]  # (B, dim, k-1)

    grad_vec = grad.unsqueeze(-1)               # (B, dim, 1)

    U = torch.cat((smallest_modes, grad_vec), dim=-1)  # (B, dim, k)

    # -------------------------------------------------
    # 8️⃣ QR (optional but good for orthonormality)
    # -------------------------------------------------
    U, _ = torch.linalg.qr(U)

    return U
