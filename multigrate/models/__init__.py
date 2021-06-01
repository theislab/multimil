import torch
import os
from .mlp import MLP
from .mlp_decoder import MLP_decoder
from .multivae import MultiVAE
from .multivae_smaller import MultiVAE_smaller
from .multivae_all_mod import MultiVAE_all_mod
from .multivae_void import MultiVAE_void
from .multivae_poe import MultiVAE_PoE
from .multivae_poe_small import MultiVAE_PoE_small
from .multivae_poe_cond import MultiVAE_PoE_cond
from .multivae_poe_small_mse import MultiVAE_PoE_small_mse

def create_model(name, params):
    model = model_factory(name, params)
    return model

def model_factory(name, params):
    if name == 'MLP':
        model = MLP(**params)
    elif name == 'MultiVAE':
        model = MultiVAE(**params)
    elif name == 'MultiVAE_smaller':
        model = MultiVAE_smaller(**params)
    elif name == 'MultiVAE_all_mod':
        model = MultiVAE_all_mod(**params)
    elif name == 'MultiVAE_void':
        model = MultiVAE_void(**params)
    elif name == 'MultiVAE_PoE':
        model = MultiVAE_PoE(**params)
    elif name == 'MultiVAE_PoE_cond':
        model = MultiVAE_PoE_cond(**params)
    elif name == 'MultiVAE_PoE_small':
        model = MultiVAE_PoE_small(**params)
    elif name == 'MultiVAE_PoE_small_mse':
        model = MultiVAE_PoE_small_mse(**params)
    else:
        raise NotImplementedError(f'No model {name} is implemented.')
    return model

def load_model(config, model_path, device='cpu'):  # TODO refactor
    model = create_model(config, device)
    model.load_state_dict(torch.load(model_path))
    return model
