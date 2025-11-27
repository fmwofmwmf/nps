import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

import Args


def str_to_act(s: str):
    d = {
        "ReLU": F.relu,
        "LeakyReLU": F.leaky_relu,
        "ELU": F.elu,
        "Cos": torch.cos,
    }

    if s not in d:
        raise ValueError(f"Unrecognized activation {s}. Should be one of {list(d.keys())}")
    return d[s]


def model_spec_from_args(args: Args.Args, in_dim, out_dim):
    spec_dict = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "model_type": args.model_type,
    }

    if spec_dict["model_type"] in ["SubspaceMLP"]:
        spec_dict["activation"] = args.activation
        spec_dict["MLP_hidden_layers"] = args.MLP_hidden_layers
        spec_dict["MLP_hidden_layer_width"] = args.MLP_hidden_layer_width
    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    return spec_dict


def create_model(spec_dict, base_output=None, device="cpu"):
    model_type = spec_dict["model_type"]

    if model_type == "SubspaceMLP":
        model = SubspaceMLP(spec_dict, base_output=base_output).to(device)
    else:
        raise ValueError(f"Unrecognized model_type {model_type}")

    print(f"\n== Created network ({model_type}):")
    print(model)
    return model


class NestedDropout(nn.Module):
    """
    Nested Dropout layer.
    Args:
        p (float): parameter of geometric distribution (probability of stopping).
                   Expected number of kept units = 1/p.
        dim (int): feature dimension (e.g., hidden size)
    """

    def __init__(self, dim, p=0.1):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        # sample cutoff k from geometric distribution
        # k ∈ {1, 2, ..., dim}
        geometric = torch.distributions.Geometric(probs=self.p)
        k = int(torch.clamp(geometric.sample(), 1, self.dim))

        # build mask [1, 1, ..., 1, 0, 0]
        mask = x.new_zeros(self.dim)
        mask[:k] = 1

        # reshape for broadcasting over batch
        mask = mask.view(1, -1)

        return x * mask

class VariationalNestedDropout(nn.Module):
    """
    Jacobian-safe Variational Nested Dropout.

    - dim: latent dimension
    - temperature: Gumbel-sigmoid temperature for stochastic masks during training
    - eps: minimum mask floor to avoid zero gradients (keeps Jacobian alive)
    - stochastic: whether to use Gumbel noise in training (set False for deterministic training)
    """
    def __init__(self, dim, temperature=0.1, eps=1e-3, stochastic=True):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.eps = float(eps)
        self.stochastic = stochastic

        # learnable per-dimension logits (shared base across batch)
        self.alpha = nn.Parameter(torch.zeros(dim))

    def _sample_s(self, batch_size):
        """
        Sample per-dim base keep-probabilities s_i in (0,1).
        We will build monotonic mask m by cumulative product: m_i = cumprod(s)_i,
        which is smooth and non-increasing when s_i in (0,1].
        """
        base_logits = self.alpha.unsqueeze(0).expand(batch_size, -1)  # [B, D]

        if self.training and self.stochastic:
            # Gumbel noise for relaxed sampling
            # sample Gumbel(0,1)
            u = torch.rand_like(base_logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            logits = base_logits + g
            s = torch.sigmoid(logits / max(self.temperature, 1e-8))
        else:
            # deterministic: just sigmoid of alpha
            s = torch.sigmoid(base_logits)

        # ensure s in (eps, 1], but keep smoothness
        s = s * (1.0 - self.eps) + self.eps
        return s  # [B, D]

    def forward(self, z):
        """
        z: [D] or [B, D]
        returns: (z_masked, mask)
            z_masked: same shape as z
            mask: same shape as z
        """
        # Handle non-batched input: [D]
        single = False
        if z.dim() == 1:
            z = z.unsqueeze(0)  # → [1, D]
            single = True

        if z.dim() != 2 or z.size(1) != self.dim:
            raise ValueError(f"Expected z shape [B, {self.dim}] or [{self.dim}], got {tuple(z.shape)}")

        B = z.size(0)

        # Sample base keep probabilities
        s = self._sample_s(B)  # [B, D]

        # Monotonic mask
        m = torch.cumprod(s, dim=1)  # [B, D]

        # Safety floor
        m = m * (1.0 - self.eps) + self.eps

        # Apply mask
        z_masked = z * m  # [B, D]

        # If single vector input, squeeze back
        if single:
            return z_masked.squeeze(0), m.squeeze(0)

        return z_masked, m

class SubspaceMLP(nn.Module):
    def __init__(self, spec_dict, base_output=None):
        super().__init__()

        self.activation = str_to_act(spec_dict["activation"])

        layers = []
        in_dim = spec_dict["in_dim"] + 1
        for i in range(spec_dict["MLP_hidden_layers"]):
            is_last = (i + 1 == spec_dict["MLP_hidden_layers"])
            out_dim = spec_dict["out_dim"] if is_last else spec_dict["MLP_hidden_layer_width"]
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.linear_layers = nn.ModuleList(layers)

        if base_output is None:
            self.base_output = nn.Parameter(torch.empty(spec_dict["out_dim"]).uniform_(-1., 1.))
        else:
            self.base_output = nn.Parameter(base_output.clone().detach())

    @torch.compile()
    def L2(self):
        return sum((p ** 2).sum() for p in self.parameters())

    def forward(self, z, t_schedule=1.0):
        for i, layer in enumerate(self.linear_layers):
            z = layer(z)
            if i < len(self.linear_layers) - 1:
                z = self.activation(z)
        # print(self.base_output.shape, z.shape)
        z = self.base_output + t_schedule * z
        # if self.training:
        #     return z, m
        return z
