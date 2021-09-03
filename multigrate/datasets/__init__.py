import torch
import numpy as np
import os
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
import anndata
from sklearn.model_selection import train_test_split
from .scmultidataset import SingleCellMultiDatasetBuilder
from .scdataset import SingleCellDataset
from .scdatasetMIL import SingleCellDatasetMIL

def load_dataset(config, device='cpu'):
    # load configs
    batch_size = config['batch_size']
    seed = config['seed']
    val_split = config['val_split']
    pair_split = config['pair-split']

    # join the datasets into one multi-dataset object
    dataset_builder = SingleCellMultiDatasetBuilder(val_split, pair_split, device, seed)
    for dataset_config in config['datasets']:
        h5ad_path = dataset_config.get('h5ad-path', None)
        pair_group = dataset_config.get('pair-group', None)
        dataset_builder.add_dataset(dataset_config['name'], h5ad_path, pair_group)
    train_dataset, test_dataset = dataset_builder.build()

    # create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
