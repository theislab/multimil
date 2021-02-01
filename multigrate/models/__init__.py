import torch
import os
from .mlp import MLP
from .mlp_decoder import MLP_decoder
from .multivae import MultiVAE


def create_model(name, params):
    model = model_factory(name, params)
    return model


def model_factory(name, params):
    if name == 'MLP':
        model = MLP(**params)
    elif name == 'MultiVAE':
        model = MultiVAE(**params)
    else:
        raise NotImplementedError(f'No model {name} is implemented.')
    return model


def load_model(config, model_path, device='cpu'):  # TODO refactor
    model = create_model(config, device)
    model.load_state_dict(torch.load(model_path))
    return model
