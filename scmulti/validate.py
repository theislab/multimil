import argparse
import numpy as np
import pandas as pd
import time
import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import scanpy as sc
from .utils import parse_config_file, split_adatas
from . import metrics
from .datasets import load_dataset
from .models import create_model
import torch


def validate(experiment_name, output_dir, config):
    # load experiment configurations
    experiment_config = config['experiment']
    random_seed = experiment_config['seed']
    pair_split = experiment_config.get('pair-split', None)
    # torch.manual_seed(random_seed)

    # load adatas
    model_params = config['model']['params']
    adatas = []
    for adata_set in model_params['adatas']:
        adatas.append([])
        for adata_path in adata_set:
            adata = sc.read_h5ad(adata_path)
            adatas[-1].append(adata)
    model_params['adatas'] = adatas

    # recover the paired/unpaired splitting done by the training script
    if pair_split is not None:
        pair_group_masks = torch.load(os.path.join(output_dir, 'pair-group-masks.pt'))
        adatas, names, pair_groups, _ = split_adatas(adatas, model_params['names'], model_params['pair_groups'], pair_split, pair_group_masks, shuffle_unpaired=False)
        model_params['adatas'] = adatas
        model_params['names'] = names
        model_params['pair_groups'] = pair_groups

    # load the model
    model = create_model(config['model']['name'], model_params)
    model.load(os.path.join(output_dir, 'last-model.pt'))

    # validate the model
    with torch.no_grad():
        # predict the shared latent space
        z = model.predict(adatas, names, batch_size=config['model']['train']['batch_size'])

        # plot the unintegrated latents 
        sc.pp.neighbors(z)
        sc.tl.umap(z)
        sc.pl.umap(z, color=['modality', 'cell_type'], ncols=1)
        plt.savefig(os.path.join(output_dir, f'umap-z.png'), dpi=200, bbox_inches='tight')

        # calculate metrics and save them
        sc.pp.pca(z)
        mtrcs = metrics.scibmetrics.metrics(
            z, z,
            batch_key='modality',
            label_key='cell_type',
            hvg_score_=False,
            nmi_=True,
            ari_=True,
            silhouette_=True,
        )
        json.dump(mtrcs.to_dict()[0], open(os.path.join(output_dir, 'metrics.json'), 'w'), indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))
    experiment_name = os.path.basename(args.root_dir)

    validate(experiment_name, args.root_dir, config)