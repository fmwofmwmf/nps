"""Small signed-distance network used by packaged physical-AE ROMs."""

import numpy as np
import torch
import torch.nn as nn


class SDFNet(nn.Module):
    """Latent coordinate to signed distance, with in-model normalization."""

    def __init__(self, d, width=128, layers=4, center=None, scale=1.0):
        super().__init__()
        self.d = int(d)
        widths = [self.d] + [int(width)] * int(layers) + [1]
        modules = []
        for index in range(len(widths) - 1):
            modules.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                modules.append(nn.Softplus(beta=100))
        self.net = nn.Sequential(*modules)
        self.register_buffer(
            "center",
            torch.zeros(self.d) if center is None
            else torch.as_tensor(center, dtype=torch.get_default_dtype()),
        )
        self.register_buffer("iscale", torch.tensor(1.0 / float(scale)))

    def forward(self, u):
        return self.net((u - self.center) * self.iscale).squeeze(-1)

    def grad(self, u, create_graph=False):
        u = u.detach().requires_grad_(True)
        with torch.enable_grad():
            value = self.forward(u).sum()
            (gradient,) = torch.autograd.grad(value, u, create_graph=create_graph)
        return gradient

    @property
    def _dtype(self):
        return self.iscale.dtype

    @torch.no_grad()
    def value_np(self, u_np):
        value = torch.as_tensor(np.atleast_1d(u_np), dtype=self._dtype)
        return float(self.forward(value))

    def grad_np(self, u_np):
        value = torch.as_tensor(np.atleast_1d(u_np), dtype=self._dtype)
        return self.grad(value).detach().cpu().numpy().astype(np.float64)
