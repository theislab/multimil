import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import torch

# def remove_sparsity(adata):
#     if sparse.issparse(adata.X):
#         adata.X = adata.X.A
#     return adata

def get_split_idx(class_label):
    labels = set(list(class_label))

    idx = []
    for label in labels:
        idx.append(np.where(class_label == label)[0][-1]+1)

    idx.sort()
    idx = idx[:-1]

    return idx

# taken from scIB
# on 11 November 2021
# https://github.com/theislab/scib/blob/985d8155391fdfbddec024de428308b5a57ee280/scib/metrics/utils.py

# checker functions for data sanity
def check_adata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')

def check_batch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')

def create_df(pred, columns=None, index=None):

    if isinstance(pred, dict):
        for key in pred.keys():
            pred[key] = torch.cat(pred[key]).squeeze().cpu().numpy()
    else:
        pred = torch.cat(pred).squeeze().cpu().numpy()

    df = pd.DataFrame(pred)
    print('creating df....')
    print(df)
    print(columns)
    if index is not None:
        df.index = index
    if columns is not None:
        df.columns = columns
    return df
