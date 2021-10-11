import sys
import torch
import time
import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from torch import nn
from collections import defaultdict, Counter
from operator import itemgetter, attrgetter
from torch.nn import functional as F
from itertools import cycle, zip_longest, groupby
from ..nn import *
from ..module import MultiVAETorch
from ..distributions import *
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import DataSplitter
from typing import List, Optional, Union

class MultiVAE:
    def __init__(
        self,
        adatas,
        condition=None,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        hiddens=[],
        losses=[],
        output_activations=[],
        shared_hiddens=[],
        dropout=0.2,
        device=None,
        loss_coefs=[],
        theta=None,
    ):
        # configure to CUDA if is available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if len(adatas) != len(hiddens):
            if len(hiddens) == 0:
                hiddens = [[] for _ in adatas]
            else:
                raise ValueError(f'adatas and hiddens must be the same length. len(adatas) = {len(adatas)} != {len(hiddens)} = len(hiddens)')

        if len(adatas) != len(output_activations):
            if len(output_activations) == 0:
                output_activations = ['linear' for _ in adatas] #or leaky relu?
            else:
                raise ValueError(f'adatas and output_activations must be the same length. len(adatas) = {len(adatas)} != {len(output_activations)} = len(output_activations)')

        if normalization not in ['layer', 'batch', None]:
            raise ValueError(f'Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        self.adatas = adatas
        self._train_history = defaultdict(list)
        self._val_history = defaultdict(list)
        self.condition = condition
        self.hiddens = hiddens
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.shared_hiddens = shared_hiddens
        self.dropout = dropout
        self.output_activations = output_activations
        self.input_dims = [modality_adatas[0].shape[1] if len(modality_adatas) > 0 else 0 for modality_adatas in adatas]  # the feature set size of each modality
        self.n_modality = len(self.input_dims)
        self.normalization = normalization # need for architecture surgery

        if len(losses) == 0:
            self.losses = ['mse']*self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(f'adatas and losses arguments must be the same length or losses has to be []. len(adatas) = {len(adatas)} != {len(losses)} = len(losses)')

        # don't need this actually
        # why not?
        if len(loss_coefs) == 0:
            self.loss_coefs = [1.0]*self.n_modality
        elif len(loss_coefs) == self.n_modality:
            self.loss_coefs = loss_coefs
        else:
            raise ValueError(f'adatas and loss_coefs arguments must be the same length or loss_coefs has to be []. len(adatas) = {len(adatas)} != {len(loss_coefs)} = len(loss_coefs)')

        self.loss_coef_dict = {}
        for i, loss in enumerate(self.losses):
            self.loss_coef_dict[loss] = self.loss_coefs[i]

        #self.batch_labels = [list(range(len(modality_adatas))) for modality_adatas in adatas]

        self.n_batch_labels = [0]*self.n_modality

        if condition:
            self.n_batch_labels = [len(modality_adatas) for modality_adatas in adatas]

        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        self.theta = theta
        if self.theta == None:
            for i, loss in enumerate(losses):
                if loss in ["nb", "zinb"]:
                    self.theta = torch.nn.Parameter(torch.randn(self.input_dims[i], max(self.n_batch_labels[i], 1))).to(self.device).detach().requires_grad_(True)

        # need for surgery TODO check
        # self.mod_dec_dim = h_dim
        # create modules
        self.encoders = [MLP(x_dim + self.n_batch_labels[i], z_dim, hs, output_activation='leakyrelu',
                             dropout=dropout, norm=normalization, regularize_last_layer=True) if x_dim > 0 else None for i, (x_dim, hs) in enumerate(zip(self.input_dims, hiddens))]
        self.decoders = [MLP_decoder(h_dim + self.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=dropout, norm=normalization, loss=loss) if x_dim > 0 else None for i, (x_dim, hs, out_act, loss) in enumerate(zip(self.input_dims, hiddens, output_activations, self.losses))]
        self.shared_decoder = MLP(z_dim + self.n_modality, h_dim, shared_hiddens[::-1], output_activation='leakyrelu',
                                  dropout=dropout, norm=normalization, regularize_last_layer=True)

        self.mu = MLP(z_dim, z_dim)
        self.logvar = MLP(z_dim, z_dim)

        self.module = MultiVAETorch(self.encoders, self.decoders, self.shared_decoder,
                                   self.mu, self.logvar, self.device, self.condition, self.n_batch_labels)

    @property
    def history(self):
        # TODO: check if all the same length, if not take the smallest or sth like that
        return pd.DataFrame(self._val_history)

    def reset_history(self):
        self._train_history = defaultdict(list)
        self._val_history = defaultdict(list)

    def print_progress_train(self, n_iters, end=''):
        current_iter = self._train_history['iteration'][-1]
        msg_train = 'iter={:d}/{:d}, loss={:.4f}, recon={:.4f}, kl={:.4f}, integ={:.4f}, cycle={:.4f}'.format(
            current_iter+1,
            n_iters,
            self._train_history['loss'][-1],
            self._train_history['recon'][-1],
            self._train_history['kl'][-1],
            self._train_history['integ'][-1],
            self._train_history['cycle'][-1]
        )
        self._print_progress_bar(current_iter+1, n_iters, prefix='', suffix=msg_train, end=end)

    def print_progress_val(self, n_iters, time_, end='\n'):
        current_iter = self._val_history['iteration'][-1]
        msg_train = 'iter={:d}/{:d}, time={:.2f}(s)' \
            ', loss={:.4f}, recon={:.4f}, kl={:.4f}, integ={:.4f}, cycle={:.4f}' \
            ', val_loss={:.4f}, val_recon={:.4f}, val_kl={:.4f}, val_integ={:.4f}, val_cycle={:.4f}'.format(
            current_iter+1,
            n_iters,
            time_,
            self._val_history['train_loss'][-1],
            self._val_history['train_recon'][-1],
            self._val_history['train_kl'][-1],
            self._val_history['train_integ'][-1],
            self._val_history['train_cycle'][-1],
            self._val_history['val_loss'][-1],
            self._val_history['val_recon'][-1],
            self._val_history['val_kl'][-1],
            self._val_history['val_integ'][-1],
            self._val_history['val_cycle'][-1]
        )
        self._print_progress_bar(current_iter+1, n_iters, prefix='', suffix=msg_train, end=end)

    def _print_progress_bar(
        self,
        iteration,
        total,
        prefix = '',
        suffix = '',
        decimals = 1,
        length = 20,
        fill = '█',
        end=''
    ):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_len = int(length * iteration // total)
        bar = fill * filled_len + '-' * (length - filled_len)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
        sys.stdout.write(end)
        sys.stdout.flush()

    # TODO
    def impute(
        self,
        adatas,
        names,
        pair_groups,
        target_modality,
        batch_labels,
        target_pair,
        modality_key='modality',
        celltype_key='cell_type',
        layers=[],
        batch_size=64,
    ):
        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        #TODO redo prep pair stuff in case pair names are different

        adatas = self.reshape_adatas(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        zs = []
        with torch.no_grad():
            self.module.eval()

            for datas in zip_longest(*dataloaders):
                datas = [data for data in datas if data is not None]
                xs = [data[0].to(self.device) for data in datas]
                names = [data[1] for data in datas]
                modalities = [data[2] for data in datas]
                pair_groups = [data[3] for data in datas]
                celltypes = [data[4] for data in datas]
                indices = [data[5] for data in datas]
                batch_labels = [data[-1] for data in datas]

                group_indices = {}
                for i, pair in enumerate(pair_groups):
                    group_indices[pair] = group_indices.get(pair, []) + [i]

                # TODO: deal with batches
                for x, pair, mod, batch in zip(xs, pair_groups, modalities, batch_labels):
                    # get imputed modality
                    xij = self.impute_batch(x, pair, mod, batch, target_pair, target_modality)

                    z = sc.AnnData(xij.detach().cpu().numpy())
                    modalities = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(modalities)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    zs.append(z)

        return sc.AnnData.concatenate(*zs)

    def impute_batch(self, x, pair, mod, batch, target_pair, target_modality):
        zi = self.module.to_latent(x, mod, batch)
        zij = self.module.convert(zi, pair, source_pair=True, dest=target_modality, dest_pair=False)

        # assume data is paired for the decoder
        hs = self.module.z_to_h(zij)
        hs, pair_groups = self.module.decode_pairs([hs], [target_pair])
        index_of_the_modality = np.where(np.array(self.modalities_per_group[target_pair]) == target_modality)[0][0]
        xij = self.module.decode_from_shared(hs[index_of_the_modality], target_modality, target_pair, batch)

        return xij

    def get_latent_representation(
        self,
        batch_size=64
    ):
        dataloaders, _ = self.create_dataloaders(batch_size, validation_size=0)

        groups = []
        zs = []

        with torch.no_grad():
            self.module.eval()

            i = 0
            for datas in zip_longest(*dataloaders):
                #if i > 2:
                #    break
                datas = [data for data in datas if data] # to get rid of None's when a dataloader runs out
                xs = [data['X'].to(self.device) for data in datas]
                groups = [int(data['cat_covs'][:, 0][0]) for data in datas]
                modalities = [int(data['cat_covs'][:, 1][0]) for data in datas]
                batch_labels = [int(data['batch_indices'][0][0]) for data in datas]
                size_factors = [data['cont_covs'].squeeze() for data in datas]

                # forward propagation
                out = self.module.inference(xs, modalities, groups, batch_labels)
                zs_joint = out[1]

                if len(zs) == 0:
                    zs = [[] for _ in set(groups)]

                for i, z in enumerate(zs_joint):
                    zs[i] += [z.cpu()]

                i += 1

        for i, z in enumerate(zs):
            zs[i] = torch.cat(z).numpy()

        for modality_adatas in self.adatas:
            for adata in modality_adatas:
                group = adata.obs['group'][0]
                adata.obsm['latent'] = zs[group]

    def train(
        self,
        n_iters=10000,
        batch_size=64,
        lr=3e-4,
        recon_coef = 1,
        kl_coef = 1e-5,
        integ_coef = 1e-2,
        cycle_coef = 0,
        kl_anneal_iters=None,
        validation_size=0.1,
        validate_every=None,
        verbose=1,
        kernel_type='gaussian',
        print_losses=False
    ):
        # configure training parameters
        print_every = n_iters // 50 if n_iters >= 50 else n_iters

        if not kl_anneal_iters:
            kl_anneal_iters = max(n_iters // 3, 1) # to avoid division by 0

        if not validate_every:
            validate_every = max(n_iters // 10, 1)

        # create data loaders
        train_dataloaders, val_dataloaders = self.create_dataloaders(batch_size, validation_size)

        # create optimizer
        params = self.module.get_params()
        if self.theta is not None:
                params.extend([self.theta])
        optimizer_ae = torch.optim.Adam(params, lr)

        # the training loop
        epoch_time = 0  # epoch is the time between two consequtive validations
        self.module.train()
        for iteration, datas in enumerate(cycle(zip(*train_dataloaders))):
            tik = time.time()
            if iteration >= n_iters:
                break

            xs = [data['X'].to(self.device) for data in datas]
            groups = [int(data['cat_covs'][:, 0][0]) for data in datas]
            modalities = [int(data['cat_covs'][:, 1][0]) for data in datas]
            batch_labels = [int(data['batch_indices'][0][0]) for data in datas]
            size_factors = [data['cont_covs'].squeeze() for data in datas]

            number_of_batches = len(datas) # need for later logging

            rs, zs, mus, logvars = self.module(xs, modalities, groups, batch_labels)

            losses = [self.losses[mod] for mod in modalities]

            recon_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)
            kl_loss = self.calc_kl_loss(mus, logvars)
            integ_loss = self.calc_integ_loss(zs, kernel_type)
            if cycle_coef == 0:
                cycle_loss = 0
            else:
                cycle_loss = self.calc_cycle_loss(xs, zs, groups, modalities, batch_labels, new_pair_groups, losses)

            kl_coef = self.kl_anneal(iteration, kl_anneal_iters, kl_coef)  # KL annealing

            loss_ae = recon_coef * recon_loss + \
                      kl_coef * kl_loss + \
                      integ_coef * integ_loss + \
                      cycle_coef * cycle_loss

            # AE backpropagation
            optimizer_ae.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()

            # update progress bar
            if iteration % print_every == 0:
                # as function
                self._train_history['iteration'].append(iteration)
                self._train_history['loss'].append(loss_ae.detach().cpu().item())
                self._train_history['recon'].append(recon_loss.detach().cpu().item())
                for mod_loss, name in zip([mse_loss, nb_loss, zinb_loss, bce_loss], ['mse', 'nb', 'zinb', 'bce']):
                    name = 'recon_' + name
                    self._train_history[name].append(mod_loss.detach().cpu().item() if mod_loss !=0 else 0)
                self._train_history['kl'].append(kl_loss.detach().cpu().item())
                self._train_history['integ'].append(integ_loss.detach().cpu().item() if integ_loss != 0 else 0)
                self._train_history['cycle'].append(cycle_loss.detach().cpu().item() if cycle_loss != 0 else 0)

                if verbose >= 2:
                    self.print_progress_train(n_iters)

            # add this iteration to the epoch time
            epoch_time += time.time() - tik

            # validate
            # check how exactly this is calculated, prob need to change to just for the last iteration?
            if iteration > 0 and iteration % validate_every == 0 or iteration == n_iters - 1:
                # add average train losses of the elapsed epoch to the validation history
                self._val_history['iteration'].append(iteration)
                self._val_history['train_loss'].append(np.mean(self._train_history['loss'][-(validate_every//print_every):]) / number_of_batches)
                self._val_history['train_recon'].append(np.mean(self._train_history['recon'][-(validate_every//print_every):]) / number_of_batches)
                for mod_loss, name in zip([mse_loss, nb_loss, zinb_loss, bce_loss], ['mse', 'nb', 'zinb', 'bce']):
                    name_train = 'recon_' + name
                    name = 'train_recon_' + name
                    self._val_history[name].append(np.mean(self._train_history[name_train][-(validate_every//print_every):]) / number_of_batches)
                self._val_history['train_kl'].append(np.mean(self._train_history['kl'][-(validate_every//print_every):]) / number_of_batches)
                self._val_history['train_integ'].append(np.mean(self._train_history['integ'][-(validate_every//print_every):]) / number_of_batches)
                self._val_history['train_cycle'].append(np.mean(self._train_history['cycle'][-(validate_every//print_every):]) / number_of_batches)

                self.module.eval()
                self.validate(val_dataloaders, n_iters, recon_coef, kl_coef, integ_coef, cycle_coef, epoch_time, verbose, kernel_type)
                self.module.train()
                epoch_time = 0  # reset epoch time

    def validate(
        self,
        val_dataloaders,
        n_iters,
        recon_coef,
        kl_coef,
        integ_coef,
        cycle_coef,
        train_time=None,
        verbose=1,
        kernel_type='gaussian'
    ):
        tik = time.time()

        # we want mean losses of all validation batches
        recon_loss = 0
        mse_l, nb_l, zinb_l, bce_l = 0, 0, 0, 0
        kl_loss = 0
        integ_loss = 0
        cycle_loss = 0

        val_n_iters = 0

        for datas in zip_longest(*val_dataloaders):
            datas = [data for data in datas if data is not None]
            val_n_iters += len(datas)
            xs = [data['X'].to(self.device) for data in datas]
            groups = [int(data['cat_covs'][:, 0][0]) for data in datas]
            modalities = [int(data['cat_covs'][:, 1][0]) for data in datas]
            batch_labels = [int(data['batch_indices'][0][0]) for data in datas]
            size_factors = [data['cont_covs'].squeeze() for data in datas]

            # forward propagation
            rs, zs, mus, logvars = self.module(xs, modalities, groups, batch_labels)

            losses = [self.losses[mod] for mod in modalities]

            # calculate the losses
            r_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)
            recon_loss += r_loss
            mse_l += mse_loss
            nb_l += nb_loss
            zinb_l += zinb_loss
            bce_l += bce_loss
            kl_loss += self.calc_kl_loss(mus, logvars)
            integ_loss += self.calc_integ_loss(zs, kernel_type)
            if cycle_coef > 0:
                cycle_loss += self.calc_cycle_loss(xs, zs, groups, modalities, batch_labels, losses)

        # calculate overal losses
        loss_ae = recon_coef * recon_loss + \
                  kl_coef * kl_loss + \
                  integ_coef * integ_loss + \
                  cycle_coef * cycle_loss

        # logging
        self._val_history['val_loss'].append(loss_ae.detach().cpu().item() / val_n_iters)
        self._val_history['val_recon'].append(recon_loss.detach().cpu().item() / val_n_iters)

        for mod_loss, name in zip([mse_l, nb_l, zinb_l, bce_l], ['mse', 'nb', 'zinb', 'bce']):
            if mod_loss != 0:
                name = 'val_recon_' + name
                self._val_history[name].append(mod_loss.detach().cpu().item() / val_n_iters)

        self._val_history['val_kl'].append(kl_loss.detach().cpu().item() / val_n_iters)
        if integ_loss != 0:
            self._val_history['val_integ'].append(integ_loss.detach().cpu().item() / val_n_iters)
        else:
            self._val_history['val_integ'].append(0)
        if cycle_loss != 0:
            self._val_history['val_cycle'].append(cycle_loss.detach().cpu().item() / val_n_iters)
        else:
            self._val_history['val_cycle'].append(0)

        val_time = time.time() - tik

        if verbose == 1:
            self.print_progress_val(n_iters, train_time + val_time, end='')
        elif verbose >= 2:
            self.print_progress_val(n_iters, train_time + val_time, end='\n')

    def calc_recon_loss(self, xs, rs, losses, batch_labels, size_factors):
        loss = 0
        mse_loss = 0
        nb_loss = 0
        zinb_loss = 0
        bce_loss = 0
        for x, r, loss_type, batch, size_factor in zip(xs, rs, losses, batch_labels, size_factors):
            if loss_type == 'mse':
                mse_loss = self.loss_coef_dict['mse']*nn.MSELoss()(r, x)
                loss += mse_loss
            elif loss_type == 'nb':
                dec_mean = r
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1)).to(self.device)
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[batch] if self.condition else self.theta.T[0]
                dispersion = torch.exp(dispersion)
                nb_loss = self.loss_coef_dict['nb']*NB()(x, dec_mean, dispersion)
                loss -= nb_loss
            elif loss_type == 'zinb':
                dec_mean, dec_dropout = r
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1))
                dispersion = self.theta.T[batch] if self.condition else self.theta.T[0]
                dispersion = torch.exp(dispersion)
                zinb_loss = self.loss_coef_dict['zinb']*ZINB()(x, dec_mean, dispersion, dec_dropout)
                loss -= zinb_loss
            elif loss_type == 'bce':
                bce_loss = self.loss_coef_dict['bce']*nn.BCELoss()(r, x)
                loss += bce_loss

        return loss, mse_loss, -nb_loss, -zinb_loss, bce_loss

    def calc_kl_loss(self, mus, logvars):
        return sum([KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])

    def calc_integ_loss(self, zs, kernel_type):
        loss = 0
        for i, zi in enumerate(zs):
            for j, zj in enumerate(zs):
                if i == j:  # do not integrate one dataset with itself
                    continue
                loss += MMD(kernel_type=kernel_type)(zi, zj)
        return loss

    def calc_cycle_loss(self, xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses):
        loss = 0
        for i, (xi, pgi, modi, batchi) in enumerate(zip(xs, pair_groups, modalities, batch_labels)):
            for j, (pgj, modj, batchj, lossj) in enumerate(zip(pair_groups, modalities, batch_labels, losses)):
                if i == j:  # do not make a dataset cycle consistent with itself
                    continue
                idx = np.argwhere(np.array(new_pair_groups) == pgi)[0][0]
                zij = self.module.convert(zs[idx], pgi, source_pair=True, dest=modj, dest_pair=False)
                rij = self.module.decode(zij, modj, batchj)
                if lossj in ['zinb']:
                    rij = rij[0]
                ziji = self.module.convert(self.module.to_latent(rij, modj, batchj), modj, source_pair=False, dest=modi, dest_pair=False)
                xiji = self.module.decode(ziji, modi, batchi)
                loss += nn.MSELoss()(xiji, xi)
        return loss

    def kl_anneal(self, iteration, anneal_iters, kl_coef):
        kl_coef = min(
            kl_coef,
            (iteration / anneal_iters) * kl_coef
        )
        return kl_coef

    def save(self, path):
        torch.save({
            'state_dict' : self.module.state_dict(),
        }, os.path.join(path, 'last-model.pt'), pickle_protocol=4)
        pd.DataFrame(self._val_history).to_csv(os.path.join(path, 'history.csv'))

    def load(self, path):
        model_file = torch.load(os.path.join(path, 'last-model.pt'), map_location=self.device)
        self.module.load_state_dict(model_file['state_dict'])
        self._val_history = pd.read_csv(os.path.join(path, 'history.csv'), index_col=0)

    def setup_anndata(
            adatas: List[List[ad.AnnData]],
            groups: List[List[Union[int, str]]],
            batch_keys: Optional[List[List[str]]]=None,
            layers: Optional[List[List[str]]]=None
            ):

        #TODO: split by batch and return new adatas
        batches = [list(range(len(modality_adatas))) for modality_adatas in adatas]

        if not layers:
            layers = [[None]*len(modality_adatas) for modality_adatas in adatas]

        for modality, (modality_adatas, modality_batches, modality_layers, modality_groups)  in enumerate(zip(adatas, batches, layers, groups)):
            for adata, batch, layer, group in zip(modality_adatas, modality_batches, modality_layers, modality_groups):
                adata.obs['group'] = group
                adata.obs['modality'] = modality
                adata.obs['batch_key'] = batch
                adata.obs['size_factors'] = adata.layers[layer].sum(1).T.tolist()[0] if layer else adata.X.sum(1).T.tolist()[0] # need for NB/ZINB calculations
                _setup_anndata(
                    adata,
                    batch_key='batch_key',
                    layer=layer,
                    continuous_covariate_keys=['size_factors'],
                    categorical_covariate_keys=['group', 'modality']
                )
                adata.obsm['_scvi_extra_categoricals']['modality'] = adata.obs.modality # hack fix, otherwise all modalities are set to 0
                adata.obsm['_scvi_extra_categoricals']['group'] = adata.obs.group
                adata.obs['_scvi_batch'] = adata.obs['batch_key']

    # TODO
    def plot_losses(
        self,
        recon=True,
        kl=True,
        integ=True,
        cycle=False
    ):
        pass

    def create_dataloaders(self, batch_size, validation_size):
        train_dataloaders, val_dataloaders = [], []

        # first sort adatas by group, then by modality
        groups = [adata.obs['group'][0] for modality_adatas in self.adatas for adata in modality_adatas]
        self.sorting = np.argsort(groups, kind='stable')

        adatas = [adata for modality_adatas in self.adatas for adata in modality_adatas]

        adatas = [adatas[i] for i in self.sorting]

        for adata in adatas:
            splitter = DataSplitter(
                adata,
                train_size = 1 - validation_size,
                validation_size = validation_size,
                batch_size = batch_size
            )
            splitter.setup()
            train_dataloaders.append(splitter.train_dataloader())
            val_dataloaders.append(splitter.val_dataloader())

        return train_dataloaders, val_dataloaders
