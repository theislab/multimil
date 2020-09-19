import torch
import os
from .mlp import MLP
from .multiscae import MultiScAE
from .multiscvae import MultiScVAE
from .multisccae import MultiScCAE
from .multisccvae import MultiScCVAE


def create_model(config, device='cpu'):
    model_name = config['name']
    model = model_factory(model_name, device, config['params'])
    return model


def model_factory(model_name, device, params):
    if model_name == 'mlp':
        model = MLP(**params, device=device)
    elif model_name == 'multiscae':
        model = MultiScAE(**params, device=device)
    elif model_name == 'multiscvae':
        model = MultiScVAE(**params, device=device)
    elif model_name == 'multisccae':
        model = MultiScCAE(**params, device=device)
    elif model_name == 'multisccvae':
        model = MultiScCVAE(**params, device=device)
    else:
        raise NotImplementedError(f'No model {model_name} is implemented.')
    return model


def load_model(config, model_path, device='cpu'):
    model = create_model(config, device)
    model.load_state_dict(torch.load(model_path))
    return model