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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from .train import train
from .validate import validate
from functools import partial
from copy import deepcopy


def objective(params, base_experiment_name, output_dir, pair_split, save_losses, config):
    experiment_name = f'{base_experiment_name}_kl({params["kl_coef"]:.5f})_integ({params["integ_coef"]:.5f})'
    output_dir = os.path.join(output_dir, experiment_name)

    config = deepcopy(config)
    model_params = config['model']['params']
    for p in ['kl_coef', 'integ_coef']:
        model_params[p] = params[p]
    train(experiment_name, output_dir, **config)

    config = parse_config_file(os.path.join(output_dir, 'config.json'))
    validate(base_experiment_name, output_dir, config)

    metrics = parse_config_file(os.path.join(output_dir, 'metrics.json'))
    mean_integ_metrics = np.mean([metrics[i] for i in ['ASW_label/batch', 'graph_conn']]) #'PCR_batch',
    mean_bio_metrics = np.mean([metrics[i] for i in ['ASW_label', 'NMI_cluster/label', 'ARI_cluster/label']]) # 'isolated_label_silhouette'

    return {
        'loss': -(0.4*mean_integ_metrics + 0.6*mean_bio_metrics),
        'status': STATUS_OK,
        'eval_time': time.time()
    }


def hyper_optimize(base_experiment_name, output_dir, config, max_evals, kl_coefs_range, integ_coefs_range, cycle_coefs_range, pair_split, save_losses):
    # define the search space
    space = hp.choice('model_params', [{
        'kl_coef': hp.choice('kl_coef', kl_coefs_list),
        'integ_coef': hp.choice('integ_coef', integ_coefs_list)
    }])

    trials = Trials()
    fmin_objective = partial(objective, base_experiment_name=base_experiment_name, output_dir=output_dir, pair_split=pair_split, save_losses=save_losses, config=config)
    best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description='Perform hyper-parameter optimization.')
    parser.add_argument('--base-config-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--kl-coefs-list', nargs='+', type=float, required=True)
    parser.add_argument('--integ-coefs-list', nargs='+', type=float, required=True)
    #parser.add_argument('--cycle-coefs-range', nargs=2, type=float, required=True)
    #parser.add_argument('--pair-split', type=float, required=True)
    parser.add_argument('--max-evals', type=int, default=100)
    parser.add_argument('--save-losses', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(args.base_config_file)
    base_experiment_name = os.path.splitext(os.path.basename(args.base_config_file))[0]

    best = hyper_optimize(
        base_experiment_name,
        args.output_dir,
        config,
        args.max_evals,
        args.kl_coefs_range,
        args.integ_coefs_range,
        #args.cycle_coefs_range,
        #args.pair_split,
        args.save_losses
    )
    print(best)
