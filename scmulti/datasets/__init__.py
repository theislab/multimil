import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from .scdataset import SingleCellDataset


def load_dataset(config, device='cpu'):
    # load configs
    batch_size = config['batch_size']
    seed = config['seed']
    val_split = config['val_split']

    # create two DataLoaders for each dataset (train, val)
    train_loaders, val_loaders = [], []
    for dataset_config in config['datasets']:
        dataset = SingleCellDataset(dataset_config['root-dir'], dataset_config['h5ad-filename'], device)  # load the dataset
        train, val = train_val_split(dataset, val_split, seed)
        train_loader = DataLoader(train, batch_size=batch_size)
        val_loader = DataLoader(val, batch_size=batch_size)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    return train_loaders, val_loaders


def train_val_split(dataset, val_split=0.2, seed=42):
    val_size = int(val_split * len(dataset))

    np.random.seed(seed)
    perm = np.random.permutation(len(dataset))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
