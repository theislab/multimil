import torch
import anndata
import numpy as np
from scipy import sparse
import os


class SingleCellDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, h5ad_filename='train.h5ad', device='cpu'):
        self.adata = self.load_h5ad(os.path.join(root_dir, h5ad_filename))
        self.x = self.create_tensor(self.adata.X)
        self.x = self.x.to(device)
    
    def load_h5ad(self, path):
        return anndata.read_h5ad(path)
    
    def get_adata(self):
        return self.adata
    
    def create_tensor(self, x):
        if sparse.issparse(x):
            x = x.todense()
            return torch.FloatTensor(x)
        else:
            return torch.FloatTensor(x)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        return self.x[idx]