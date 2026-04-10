import torch
import torch.nn as nn
from gradient_helpers import *

class SmoothGradientBasisField(nn.Module):
    def __init__(self, dim, k, hidden_layers, hidden_width):
        super().__init__()

        self.dim = dim
        self.k = k

        layers = [nn.Linear(dim, hidden_width), nn.Softplus()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Softplus())
        self.encoder = nn.Sequential(*layers)

        self.basis_head = nn.Linear(hidden_width, k * dim)

        # Fixed periodic mask: True if that dimension is periodic
        self.register_buffer("periodic_mask", torch.zeros(dim, dtype=torch.bool), persistent=False)
        self.register_buffer("periods", torch.ones(dim), persistent=False)  # period values

        # Example: every 3rd coordinate (0-based indexing) is periodic with 2pi
        self.periodic_mask[2::3] = True
        self.periods[self.periodic_mask] = 2 * torch.pi

    def _apply_periodic_mask(self, q):
        single = False
        if q.dim() == 1:
            q = q.unsqueeze(0)
            single = True

        q_out = q#.clone()

        mask = self.periodic_mask
        if mask.any():
            q_out[:, mask] = torch.remainder(q[:, mask], self.periods[mask])

        if single:
            q_out = q_out.squeeze(0)
        return q_out

    def forward(self, q_batch):
        # Apply periodic mask first
        q_batch = self._apply_periodic_mask(q_batch)

        single_input = False
        if q_batch.dim() == 1:
            q_batch = q_batch.unsqueeze(0)
            single_input = True

        B = q_batch.shape[0]
        features = self.encoder(q_batch)
        basis_raw = self.basis_head(features)
        basis_raw = basis_raw.reshape(B, self.k, self.dim).transpose(1, 2)
        B_ortho = batch_gram_schmidt(basis_raw)

        if single_input:
            B_ortho = B_ortho.squeeze(0)
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



