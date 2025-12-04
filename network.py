import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

def mse_loss(y_pred, y, x=None):
    # use tanh to weight near 0 values more
    n = 2
    sigma = 0.5
    weights = torch.exp(-y**n / sigma**n)
    loss_dict = {'mse': torch.mean(weights * (y_pred - y)**2)}
    return loss_dict

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))


def build_criterion(config):
    # check if loss is specified in config
    if 'loss' in config:
        loss_config = config['loss']
        if loss_config['otype'] == 'mse':
            return mse_loss
    return mse_loss


class SubspaceMLP(nn.Module):
    """
    Standard PyTorch MLP for subspace learning.
    Supports variable t_schedule parameter for curriculum learning.
    Mirrors MLP class structure exactly.
    """
    
    def __init__(self, config: Dict[str, Any], base_output: Optional[torch.Tensor] = None):
        super().__init__()
        
        # Extract dimensions from config
        in_features = config.get('in_dim', config.get('n_input_dims'))
        out_features = config.get('out_dim', config.get('n_output_dims'))
        
        if in_features is None or out_features is None:
            raise ValueError("Config must specify 'in_dim'/'n_input_dims' and 'out_dim'/'n_output_dims'")
        
        # Extract network parameters from nested 'network' key or directly
        if 'network' in config:
            network_config = config['network']
            num_hidden_layers = network_config.get('n_hidden_layers', 5)
            hidden_features = network_config.get('n_neurons', 128)
            nonlinearity = network_config.get('activation', 'ReLU')
            output_nonlinearity = network_config.get('output_activation', 'None')
        else:
            num_hidden_layers = config.get('n_hidden_layers', config.get('MLP_hidden_layers', 5))
            hidden_features = config.get('n_neurons', config.get('MLP_hidden_layer_width', 128))
            nonlinearity = config.get('activation', 'ReLU')
            output_nonlinearity = None
        
        self.first_layer_init = None
        
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'Sine': (Sine(), sine_init, first_layer_sine_init),
                         'ReLU': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'ELU': (nn.ELU(inplace=True), init_weights_elu, None)}
        
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        
        self.weight_init = nl_weight_init
        
        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])
        
        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if output_nonlinearity and output_nonlinearity != 'None':
            self.net.append(nls_and_inits[output_nonlinearity][0])
        
        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        
        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)
        
        # Base output (reference configuration)
        if base_output is None:
            self.base_output = nn.Parameter(torch.zeros(out_features))
        else:
            self.base_output = nn.Parameter(base_output.clone().detach())
    
    def forward(self, z, t_schedule=1.0):
        """
        Args:
            z: (B, in_dim) input latent + shape parameters
            t_schedule: float in [0, 1] for curriculum learning
        
        Returns:
            (B, out_dim) output configuration
        """
        return self.base_output + t_schedule * self.net(z)


def make_network(config: Dict[str, Any], base_output: Optional[torch.Tensor] = None, device='cuda') -> nn.Module:
    """
    Factory function to create subspace networks from config.
    
    Args:
        config: Configuration dict with 'network' section
        base_output: Optional reference configuration tensor
        device: Device to place model on
    
    Returns:
        Neural network module (SubspaceMLP)
    """
    # Determine network type from config
    if 'network' in config:
        network_type = config['network'].get('otype', 'SubspaceMLP')
    else:
        network_type = config.get('otype', 'SubspaceMLP')
    
    # Create SubspaceMLP
    if network_type != 'SubspaceMLP' and 'MLP' not in network_type:
        print(f"Warning: Network type '{network_type}' not supported, using SubspaceMLP")
    
    model = SubspaceMLP(config, base_output=base_output)
    model = model.to(device)
    print(f"\n== Created network ({network_type}):")
    print(model)
    return model
