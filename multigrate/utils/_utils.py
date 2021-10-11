import scanpy as sc

def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    return adata
