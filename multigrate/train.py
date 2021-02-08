import argparse
import numpy as np
import time
import os
import json
from itertools import cycle
import scanpy as sc
from .utils import parse_config_file, split_adatas
from .datasets import load_dataset
from .models import create_model
import torch


def train(experiment_name, output_dir, **config):
    # load experiment configurations
    experiment_config = config['experiment']
    random_seed = experiment_config['seed']
    pair_split = experiment_config.get('pair-split', None)
    # torch.manual_seed(random_seed)

    # configure output directory to save logs and results
    os.makedirs(output_dir, exist_ok=True)
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)

    # load adatas
    model_params = config['model']['params']
    adatas = []
    for adata_set in model_params['adatas']:
        adatas.append([])
        for adata_path in adata_set:
            adata = sc.read_h5ad(adata_path)
            adatas[-1].append(adata)
    model_params['adatas'] = adatas

    # hold out a portion of the paired datasets as unpaired
    # this option can be disabled in the config by setting "pair_split" to None (or completely ignoring setting it)
    if pair_split is not None:
        adatas, names, pair_groups, pair_group_masks = split_adatas(adatas, model_params['names'], model_params['pair_groups'], pair_split, shuffle_unpaired=True)
        model_params['adatas'] = adatas
        model_params['names'] = names
        model_params['pair_groups'] = pair_groups

    # create the model to be trained
    model = create_model(config['model']['name'], model_params)
    
    # train
    model.train(**config['model']['train'])

    # save the last state of the model
    model.save(output_dir)
    torch.save(pair_group_masks, os.path.join(output_dir, 'pair-group-masks.pt'))

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(args.config_file)
    experiment_name = os.path.splitext(os.path.basename(args.config_file))[0]

    train(experiment_name, args.output_dir, **config)
