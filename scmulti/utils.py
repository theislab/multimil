import json
import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd
import os
from matplotlib import pyplot as plt
from torch import optim


def parse_config_file(path):
    with open(path) as json_file:
        return json.load(json_file)


def create_optimizer(model_params, config):
    if config['name'] == 'adam':
        optimizer = optim.Adam(model_params, lr=config['lr'])
    else:
        raise NotImplementedError(f'No optimizer with name {config["name"]} is implemented.')
    return optimizer


    

def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    return adata