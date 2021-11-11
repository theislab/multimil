import numpy as np
import scanpy as sc

def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    return adata

def get_split_idx(class_label):
    labels = set(list(class_label))

    idx = []
    for label in labels:
        idx.append(np.where(class_label == label)[0][-1]+1)

    idx.sort()
    idx = idx[:-1]

    return idx
