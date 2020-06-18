import json
import scanpy as sc
import numpy as np
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


def plot_latent(adatas, zs, save_dir, prefix='val-'):
    # plot a UMAP for each dataset
    for i, adata in enumerate(adatas):
        latent_adata = sc.AnnData(zs[i])
        latent_adata.obs = adata.obs.copy(deep=True)
        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        sc.pl.umap(latent_adata, color=['cell_type'])
        plt.savefig(os.path.join(save_dir, f'{prefix}umap-latent-{i}.png'), dpi=200, bbox_inches='tight')
    
    # plot a UMAP for integrated datasets
    latent_all_adata = np.concatenate(zs, axis=0)
    latent_all_adata = sc.AnnData(latent_all_adata)
    obss = []
    for i, adata in enumerate(adatas):
        obs = adata.obs.copy(deep=True)
        obs['modal'] = f'modal-{i}'
        obss.append(obs)
    obss = pd.concat(obss)
    latent_all_adata.obs = obss

    sc.pp.neighbors(latent_all_adata)
    sc.tl.umap(latent_all_adata)
    sc.pl.umap(latent_all_adata, color=['modal', 'cell_type'], ncols=1)
    plt.savefig(os.path.join(save_dir, f'{prefix}umap-all-latents.png'), dpi=200, bbox_inches='tight')
    