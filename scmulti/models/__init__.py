import torch
import os
from .mlp import MLP
from .multivae import MultiVAE
from .multiscae import MultiScAE
from .multiscvae import MultiScVAE
from .multisccae import MultiScCAE
from .multisccvae import MultiScCVAE


def create_model(name, params):
    model = model_factory(name, params)
    return model


def model_factory(name, params):
    if name == 'MLP':
        model = MLP(**params)
    elif name == 'MultiVAE':
        model = MultiVAE(**params)
    elif name == 'MultiScAE':
        model = MultiScAE(**params)
    elif name == 'MultiScVAE':
        model = MultiScVAE(**params)
    elif name == 'MultiScCAE':
        model = MultiScCAE(**params)
    elif name == 'MultiScCVAE':
        model = MultiScCVAE(**params)
    else:
        raise NotImplementedError(f'No model {name} is implemented.')
    return model


def load_model(config, model_path, device='cpu'):  # TODO refactor
    model = create_model(config, device)
    model.load_state_dict(torch.load(model_path))
    return model