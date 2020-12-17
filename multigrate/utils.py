import json
import scanpy as sc
import numpy as np
from scipy import sparse
import pandas as pd
import os
from matplotlib import pyplot as plt
from torch import optim


def parse_config_file(path):
    # TODO remove this function
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


def split_adatas(adatas, names, pair_groups, pair_split, pair_group_masks=None, shuffle_unpaired=True):
    if pair_group_masks is None:
        pair_group_masks = {}

    splitted_adatas = []
    splitted_names = []
    splitted_pair_groups = []

    for adata_set, name_set, pair_group_set in zip(adatas, names, pair_groups):
        splitted_adatas.append([])
        splitted_names.append([])
        splitted_pair_groups.append([])

        for adata, name, pair_group in zip(adata_set, name_set, pair_group_set):
            if pair_group in pair_group_masks:
                pair_mask = pair_group_masks[pair_group]
            else:
                pair_mask = np.zeros(len(adata), dtype=np.bool)
                pair_size = int(len(adata) * pair_split)
                pair_mask[:pair_size] = 1
                np.random.shuffle(pair_mask)
                if pair_group is not None:
                    pair_group_masks[pair_group] = pair_mask 

            paired_adata = adata[pair_mask]
            unpaired_adata = adata[~pair_mask]
            if shuffle_unpaired:
                unpair_mask = ~pair_mask
                np.random.shuffle(unpair_mask)
                unpaired_adata = adata[unpair_mask]

            if len(paired_adata) > 0:
                splitted_adatas[-1].append(paired_adata)
                splitted_names[-1].append(f'{name}-paired')
                splitted_pair_groups[-1].append(pair_group)
            if len(unpaired_adata) > 0:
                splitted_adatas[-1].append(unpaired_adata)
                splitted_names[-1].append(f'{name}-unpaired')
                splitted_pair_groups[-1].append(None)
    
    return splitted_adatas, splitted_names, splitted_pair_groups, pair_group_masks