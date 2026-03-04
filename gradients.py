import torch
import torch.nn as nn

class SmoothGradientBasisField(nn.Module):
    """
    Learn a smooth field of orthonormal bases over configuration space.
    Input: q (configuration) ∈ ℝ^dim
    Output: B(q) ∈ ℝ^{dim × k} - orthonormal basis at q
    """

    def __init__(self, dim, k, hidden_layers, hidden_width):
        super().__init__()

        self.dim = dim  # Full configuration space dimension
        self.k = k  # Reduced basis dimension

        layers = [nn.Linear(dim, hidden_width), nn.Softplus()]

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Softplus())

        self.encoder = nn.Sequential(*layers)

        # Output k vectors in R^dim (not yet orthonormal)
        self.basis_head = nn.Linear(hidden_width, k * dim)

    def forward(self, q_batch):
        """
        Args:
            q_batch: (B, dim) or (dim,) - batch of configurations or single configuration

        Returns:
            B_batch: (B, dim, k) or (dim, k) - batch of orthonormal bases or single basis
        """
        # Handle single input (not batched)
        single_input = False
        if q_batch.dim() == 1:
            q_batch = q_batch.unsqueeze(0)  # (dim,) -> (1, dim)
            single_input = True

        B = q_batch.shape[0]

        features = self.encoder(q_batch)

        basis_raw = self.basis_head(features)  # (B, k*dim)
        basis_raw = basis_raw.reshape(B, self.k, self.dim)  # (B, k, dim)

        # Transpose to (B, dim, k) for column vectors
        basis_raw = basis_raw.transpose(1, 2)  # (B, dim, k)
        B_ortho = batch_gram_schmidt(basis_raw)

        # Remove batch dimension if input was single
        if single_input:
            B_ortho = B_ortho.squeeze(0)  # (1, dim, k) -> (dim, k)

        return B_ortho

    def inference(self, q, g):

        features = self.encoder(q.unsqueeze(0))  # (1, F)
        basis_raw = self.basis_head(features)  # (1, (k-1)*dim)

        basis_raw = basis_raw.reshape(1, self.k, self.dim)
        basis_raw = basis_raw.transpose(1, 2)  # (1, dim, k-1)

        grad_vec = g.unsqueeze(0).unsqueeze(2)  # (1, dim, 1)

        basis = torch.cat([basis_raw, grad_vec], dim=2)  # (1, dim, k)

        B_ortho = batch_gram_schmidt(basis)  # (1, dim, k)

        return B_ortho.squeeze(0)  # (dim, k)

    def inference_recompute(self, system, system_def, q, space):
        """
        q: (dim,)  -- never batched

        Returns:
            (dim, k) orthonormal basis
        """

        # -------------------------------------------------
        # 1️⃣ Compute gradient
        # -------------------------------------------------
        q_grad = q.clone().requires_grad_(True)

        E = system.potential_energy_batch(
            system_def,
            q_grad.unsqueeze(0),
            space.unsqueeze(0)
        )[0]

        grad = torch.autograd.grad(E, q_grad, create_graph=True)[0]  # (dim,)

        features = self.encoder(q.unsqueeze(0))  # (1, F)
        basis_raw = self.basis_head(features)  # (1, (k-1)*dim)

        basis_raw = basis_raw.reshape(1, self.k, self.dim)
        basis_raw = basis_raw.transpose(1, 2)  # (1, dim, k-1)

        grad_vec = grad.unsqueeze(0).unsqueeze(2)  # (1, dim, 1)

        basis = torch.cat([basis_raw, grad_vec], dim=2)  # (1, dim, k)

        B_ortho = batch_gram_schmidt(basis)  # (1, dim, k)

        return B_ortho.squeeze(0)  # (dim, k)


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