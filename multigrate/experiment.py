import argparse
import numpy as np
import time
import os
import json
import copy
import subprocess
import ast
from .utils import parse_config_file


def create_experiment_files(base_config,
                            base_experiment_name,
                            configs_dir,
                            recon_coefs=[1],
                            kl_coefs=[1e-5, 1e-3, 1e-1],
                            integ_coefs=[0, 1],
                            cycle_coefs=[0, 1],
                            adversarials=[False, True],
                            adv_iters=5,
                            pair_splits=[1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0]):

    configs = []
    for recon_coef in recon_coefs:
        for kl_coef in kl_coefs:
            for integ_coef in integ_coefs:
                for cycle_coef in cycle_coefs:
                    for adversarial in adversarials:
                        for pair_split in pair_splits:
                            if not validate_params(recon_coef,
                                                    kl_coef,
                                                    integ_coef,
                                                    cycle_coef,
                                                    adversarial,
                                                    pair_split):
                                continue
                            config = create_config(base_config,
                                                    recon_coef,
                                                    kl_coef,
                                                    integ_coef,
                                                    cycle_coef,
                                                    adversarial,
                                                    adv_iters,
                                                    pair_split)
                            configs.append(config)
    return configs


def validate_params(recon_coef,
                    kl_coef,
                    integ_coef,
                    cycle_coef,
                    adversarial,
                    pair_split):
    # pair split should be in interval [0 1]
    if not 0 <= pair_split <= 1:
        return False
    # in adversarial setting, there should be integration
    # since without integration coefficient, adversarial and non-adversarial settings are the same
    if adversarial and integ_coef <= 0:
        return False
    return True


def create_config(base_config,
                  recon_coef=0,
                  kl_coef=0,
                  integ_coef=0,
                  cycle_coef=0,
                  adversarial=False,
                  adv_iters=5,
                  pair_split=0):
    config = copy.deepcopy(base_config)

    model_params = config['model']['params'] 
    model_params['recon_coef'] = recon_coef
    model_params['kl_coef'] = kl_coef
    model_params['integ_coef'] = integ_coef
    model_params['cycle_coef'] = cycle_coef
    model_params['adversarial'] = adversarial

    train_config = config['model']['train']
    train_config['adv_iters'] = adv_iters if adversarial else 0

    experiment_config = config['experiment']
    experiment_config['pair-split'] = pair_split

    return config


def save_configs(configs, base_name, dir):
    config_paths = []
    for config in configs:
        model_params = config['model']['params'] 
        experiment_config = config['experiment']
        experiment_name = f'{base_name}_' + \
                        f'recon{model_params["recon_coef"]}_' + \
                        f'kl{model_params["kl_coef"]}_' + \
                        f'integ{model_params["integ_coef"]}_' + \
                        f'cycle{model_params["cycle_coef"]}_' + \
                        f'adver{int(model_params["adversarial"])}_' + \
                        f'pair{experiment_config["pair-split"]}'

        path = os.path.join(dir, f'{experiment_name}.json')
        json.dump(config, open(path, 'w'), indent=2)
        config_paths.append(path)
    return config_paths


def submit_experiments(config_paths, output_dir):
    scmulti_path = os.path.dirname(os.path.realpath(__file__))
    for config_file in config_paths:
        subprocess.run(['sbatch', f'{scmulti_path}/strain-val.sh', config_file, output_dir])


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--base-config-file', type=str, required=True)
    parser.add_argument('--configs-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--recon-coefs', nargs='+', type=float, required=True)
    parser.add_argument('--kl-coefs', nargs='+', type=float, required=True)
    parser.add_argument('--integ-coefs', nargs='+', type=float, required=True)
    parser.add_argument('--cycle-coefs', nargs='+', type=float, required=True)
    parser.add_argument('--adversarials', nargs='+', type=ast.literal_eval, required=True)
    parser.add_argument('--adv-iters', type=int, required=True)
    parser.add_argument('--pair-splits', nargs='+', type=float, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base_config = parse_config_file(args.base_config_file)
    base_experiment_name = os.path.splitext(os.path.basename(args.base_config_file))[0]
    os.makedirs(args.configs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    configs = create_experiment_files(base_config,
                                      base_experiment_name,
                                      args.configs_dir,
                                      args.recon_coefs,
                                      args.kl_coefs,
                                      args.integ_coefs,
                                      args.cycle_coefs,
                                      args.adversarials,
                                      args.adv_iters,
                                      args.pair_splits)
    reply = input(f'{len(configs)} config files are created. Save and run them? [Y/n]').lower().strip()
    if reply == '' or reply == 'y':
        config_paths = save_configs(configs, base_experiment_name, args.configs_dir)
        submit_experiments(config_paths, args.output_dir)
    elif reply == 'n':
        pass
    else:
        print(f'Expected Y or N (either lower or uppercase) but recieved {reply}')