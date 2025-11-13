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


class SubspaceMLP(nn.Module):
    def __init__(self, spec_dict, base_output=None):
        super().__init__()

        self.activation = str_to_act(spec_dict["activation"])
        layers = []
        in_dim = spec_dict["in_dim"]

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

    def forward(self, z, t_schedule=1.0):
        for i, layer in enumerate(self.linear_layers):
            z = layer(z)
            if i < len(self.linear_layers) - 1:
                z = self.activation(z)

        z = self.base_output + t_schedule * z
        return z
