import torch
import torch.nn as nn
from gradient_helpers import *
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


