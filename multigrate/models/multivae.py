
import sys
import numpy as np
import pandas as pd
import time
import os
from collections import defaultdict, Counter
from operator import itemgetter, attrgetter
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from itertools import cycle, zip_longest, groupby
from ..datasets import SingleCellDataset
from .mlp import MLP
from .mlp_decoder import MLP_decoder
from .losses import MMD, KLD, NB, ZINB

class MultiVAETorch(nn.Module):
    def __init__(
        self,
        encoders,
        decoders,
        shared_encoder,
        shared_decoder,
        mu,
        logvar,
        modality_vectors,
        device='cpu',
        condition=None,
        n_batch_labels=None,
        paired_dict={},
        modalities_per_group={},
        paired_networks_per_modality_pairs={}
    ):
        super().__init__()

        self.encoders = encoders
        self.decoders = decoders
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.mu = mu
        self.logvar = logvar
        self.modality_vectors = modality_vectors
        self.device = device
        self.condition = condition
        self.n_modality = len(self.encoders)
        self.n_batch_labels = n_batch_labels
        self.paired_dict = paired_dict
        self.modalities_per_group = modalities_per_group
        self.paired_networks_per_modality_pairs = paired_networks_per_modality_pairs

        # register sub-modules
        for i, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(f'encoder_{i}', enc)
            self.add_module(f'decoder_{i}', dec)

        self = self.to(device)

    def get_params(self):
        params = []
        for enc in self.encoders:
            params.extend(list(enc.parameters()))
        for dec in self.decoders:
            params.extend(list(dec.parameters()))

        params.extend(list(self.shared_encoder.parameters()))
        params.extend(list(self.shared_decoder.parameters()))
        params.extend(list(self.mu.parameters()))
        params.extend(list(self.logvar.parameters()))
        params.extend(list(self.modality_vectors.parameters()))
        return params

    def to_shared_dim(self, x, i, batch_label):
        if self.condition:
            x = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in x])
        return self.x_to_h(x, i)

    def encode_pairs(self, hs, pair_groups):
        hs_concat = []
        new_pair_groups = []
        current = 0
        for pair, group in groupby(pair_groups):
            group_size = len(list(group))
            hs_group = hs[current:current+group_size]
            j = 0
            hs_group_concat = []
            for n in range(self.n_modality):
                if n in self.modalities_per_group[pair]:
                    hs_group_concat.append(hs_group[j])
                    j += 1
                else:
                    hs_group_concat.append(torch.zeros_like(hs[current])) # doesn't matter which one
            hs_group_concat = torch.cat(hs_group_concat, dim=-1)
            hs_concat.append(self.shared_encoder(hs_group_concat))
            new_pair_groups.append(pair)
            current += group_size

        return hs_concat, new_pair_groups

    def decode_pairs(self, hs_concat, concat_pair_groups):
        hs = []
        pair_groups = []
        for h, pair in zip(hs_concat, concat_pair_groups):
            hs_pair = self.shared_decoder(h)

            n_dim = len(hs_pair[0]) // self.n_modality
            hs_pair = torch.split(hs_pair, n_dim, dim=1)
            hs_pair_filtered = [] # leave only necessary modalities
            for mod in self.modalities_per_group[pair]:
                hs_pair_filtered.append(hs_pair[mod])

            hs.extend(hs_pair_filtered)
            pair_groups.extend([pair]*len(self.modalities_per_group[pair]))

        return hs, pair_groups

    def encode_shared(self, h):
        z = self.h_to_z(h)
        return self.bottleneck(z)

    def encode(self, x, i, batch_label):
        # add batch labels
        if self.condition:
            x = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in x])
        h = self.x_to_h(x, i)

        # depricated
        # add batch labels and modality labels to hidden representation
        # if self.condition == '1':
        #    h = torch.stack([torch.cat((cell, self.modal_vector(i)[0])) for cell in h])
        z = self.h_to_z(h)
        return self.bottleneck(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_from_shared(self, h, i, pair_group, batch_label):
        if self.condition:
            h = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in h])
        x = self.h_to_x(h, i)
        return x

    def decode(self, z, i, batch_label):
        # depricated
        # if self.condition == '1':
        #    z = torch.stack([torch.cat((cell, self.modal_vector(i)[0])) for cell in z])
        h = self.z_to_h(z)
        if self.condition:
            h = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in h])
        x = self.h_to_x(h, i)
        return x

    def to_latent(self, x, i, batch_label):
        z, _, _ = self.encode(x, i, batch_label)
        return z

    def x_to_h(self, x, i):
        return self.encoders[i](x)

    def h_to_z(self, h):
        z = self.shared_encoder(h)
        return z

    def z_to_h(self, z):
        h = self.shared_decoder(z)
        return h

    def h_to_x(self, h, i):
        x = self.decoders[i](h)
        return x

    def forward(self, xs, modalities, pair_groups, batch_labels):
        # encode
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        hs_concat, concat_pair_groups = self.encode_pairs(hs, pair_groups)
        #zs = [self.encode_shared(h) for h in hs_concat]
        zs = [self.bottleneck(h) for h in hs_concat]
        # bottleneck
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        # decode
        #hs_concat = [self.z_to_h(z) for z in zs]
        hs, pair_groups = self.decode_pairs(zs, concat_pair_groups)
        # change the order of data back to the original as encode_pairs changes it
        rs = [self.decode_from_shared(h, mod, pair_group, batch_label) for h, mod, pair_group, batch_label in zip(hs, modalities, pair_groups, batch_labels)]
        return rs, zs, mus, logvars, concat_pair_groups

    def modal_vector(self, i):
        return F.one_hot(torch.tensor([i]).long(), self.n_modality).float().to(self.device)

    def batch_vector(self, i, modality):
        return F.one_hot(torch.tensor([i]).long(), self.n_batch_labels[modality]).float().to(self.device)

    # depricated
    def test(self, *xs):
        outputs, loss, losses = self.forward(*xs)
        return loss, losses

    # check if need to distinguish between pairs not pairs, i don't think we need any more
    def convert(self, z, source, source_pair, dest=None, dest_pair=None):

        v = torch.zeros(1, self.n_modality).to(self.device)
        if not source_pair:
            v -= self.modal_vector(source)
            #print(self.modal_vector(source))
        else:
            for mod in self.modalities_per_group[source]:
                v -= self.modal_vector(mod)
                #print(self.modal_vector(mod))
        #print(v)

        if dest is not None:
            #print('TO...')
            if not dest_pair:
                v += self.modal_vector(dest)
                #print(self.modal_vector(dest))
            else:
                for mod in self.modalities_per_group[dest]:
                    v += self.modal_vector(mod)
                    #print(self.modal_vector(mod))

        return z + v @ self.modality_vectors.weight

    def integrate(self, xs, mods, pair_groups, batch_labels, j=None):

        # encode
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, mods, batch_labels)]
        hs_concat, concat_pair_groups = self.encode_pairs(hs, pair_groups)
        #zs = [self.encode_shared(h) for h in hs_concat]
        zs = [self.bottleneck(h) for h in hs_concat]
        zs = [z[0] for z in zs]

        # hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, mods, batch_labels)]
        # hs_concat, pair_groups = self.encode_pairs(hs, pair_groups)
        # zs = [self.encode_shared(h) for h in hs_concat]
        # zs = [z[0] for z in zs]
        for i, zi in enumerate(zs):
            zs[i] = self.convert(zs[i], concat_pair_groups[i], source_pair=True, dest=j)
        return zs, concat_pair_groups

class MultiVAE:
    def __init__(
        self,
        adatas,
        names,
        pair_groups,
        condition=None,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        hiddens=[],
        losses=[],
        output_activations=[],
        shared_hiddens=[],
        recon_coef=1,
        kl_coef=1e-4,
        integ_coef=1e-1,
        cycle_coef=1e-2,
        dropout=0.2,
        device=None,
        loss_coefs=[],
        layers=[],
        theta=None,
    ):
        # configure to CUDA if is available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # assertions
        if len(adatas) != len(names):
            raise ValueError(f'adatas and names arguments must be the same length. len(adatas) = {len(adatas)} != {len(names)} = len(names)')
        if len(adatas) != len(pair_groups):
            raise ValueError(f'adatas and pair_groups arguments must be the same length. len(adatas) = {len(adatas)} != {len(pair_groups)} = len(pair_groups)')
        if len(adatas) != len(hiddens):
            raise ValueError(f'adatas and hiddens arguments must be the same length. len(adatas) = {len(adatas)} != {len(hiddens)} = len(hiddens)')
        if len(adatas) != len(output_activations):
            raise ValueError(f'adatas and output_activations arguments must be the same length. len(adatas) = {len(adatas)} != {len(output_activations)} = len(output_activations)')
        # TODO: do some assertions for other parameters

        self._train_history = defaultdict(list)
        self._val_history = defaultdict(list)
        self.recon_coef = recon_coef
        self.kl_coef = kl_coef
        self.integ_coef = integ_coef
        self.cycle_coef = cycle_coef
        self.condition = condition
        self.hiddens = hiddens
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.shared_hiddens = shared_hiddens
        self.dropout = dropout
        self.output_activations = output_activations
        self.pair_groups = pair_groups
        self.x_dims = [modality_adatas[0].shape[1] if len(modality_adatas) > 0 else 0 for modality_adatas in adatas]  # the feature set size of each modality
        self.n_modality = len(self.x_dims)

        pair_count = self.prep_paired_groups(pair_groups)

        if normalization not in ['layer', 'batch', None]:
            raise ValueError(f'normalization has to be one of ["layer", "batch"]')

        # need for surgery
        self.normalization = normalization

        if len(losses) == 0:
            self.losses = ['mse']*self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(f'adatas and losses arguments must be the same length or losses has to be []. len(adatas) = {len(adatas)} != {len(losses)} = len(losses)')

        # don't need this actually
        if len(loss_coefs) == 0:
            self.loss_coefs = [1.0]*self.n_modality
        elif len(loss_coefs) == self.n_modality:
            self.loss_coefs = loss_coefs
        else:
            raise ValueError(f'adatas and loss_coefs arguments must be the same length or loss_coefs has to be []. len(adatas) = {len(adatas)} != {len(loss_coefs)} = len(loss_coefs)')

        self.loss_coef_dict = {}
        for i, loss in enumerate(self.losses):
            self.loss_coef_dict[loss] = self.loss_coefs[i]

        self.batch_labels = [list(range(len(modality_adatas))) for modality_adatas in adatas]

        self.n_batch_labels = [0]*self.n_modality

        if condition:
            self.n_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
        # depricated
        # elif condition == '1':
        #    self.n_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
        #    n_mod_labels = self.n_modality

        if len(layers) == 0:
            self.layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]
        elif len(layers) == self.n_modality:
            self.layers = layers
        else:
            raise ValueError(f'adatas and layers arguments must be the same shape or layers has to be []. len(adatas) = {len(adatas)} != {len(layers)} = len(layers)')

        self.adatas = self.reshape_adatas(adatas, names, self.layers, pair_groups, self.batch_labels)
        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        self.theta = theta
        if self.theta == None:
            for i, loss in enumerate(losses):
                if loss in ["nb", "zinb"]:
                    self.theta = torch.nn.Parameter(torch.randn(self.x_dims[i], max(self.n_batch_labels[i], 1))).to(self.device).detach().requires_grad_(True)
        # create modules
        self.encoders = [MLP(x_dim + self.n_batch_labels[i], h_dim, hs, output_activation='leakyrelu',
                             dropout=dropout, norm=normalization, regularize_last_layer=True) if x_dim > 0 else None for i, (x_dim, hs) in enumerate(zip(self.x_dims, hiddens))]
        self.decoders = [MLP_decoder(h_dim + self.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=dropout, norm=normalization, loss=loss) if x_dim > 0 else None for i, (x_dim, hs, out_act, loss) in enumerate(zip(self.x_dims, hiddens, output_activations, self.losses))]
        self.shared_encoder = MLP(h_dim*self.n_modality, z_dim, shared_hiddens, output_activation='leakyrelu',
                                  dropout=dropout, norm=normalization, regularize_last_layer=True)
        self.shared_decoder = MLP(z_dim, h_dim*self.n_modality, shared_hiddens[::-1], output_activation='leakyrelu',
                                  dropout=dropout, norm=normalization, regularize_last_layer=True)

        self.mu = MLP(z_dim, z_dim)
        self.logvar = MLP(z_dim, z_dim)
        self.modality_vecs = nn.Embedding(self.n_modality, z_dim)

        self.model = MultiVAETorch(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    @property
    def history(self):
        # TODO: check if all the same length, if not take the smallest or sth like that
        return pd.DataFrame(self._val_history)

    def reset_history(self):
        self._train_history = defaultdict(list)
        self._val_history = defaultdict(list)

    def reshape_adatas(self, adatas, names, layers, pair_groups=None, batch_labels=None):
        # TODO: check if names are unique?
        if pair_groups is None:
            pair_groups = names  # dummy pair_groups
        # TODO: this should never happen
        if batch_labels is None:
            batch_labels = names
        reshaped_adatas = {}
        for modality, (adata_set, name_set, layer_set, pair_group_set, batch_label_set) in enumerate(zip(adatas, names, layers, pair_groups, batch_labels)):
            for adata, name, layer, pair_group, batch_label in zip(adata_set, name_set, layer_set, pair_group_set, batch_label_set):
                reshaped_adatas[name] = {
                    'adata': adata,
                    'layer': layer,
                    'modality': modality,
                    'pair_group': pair_group,
                    'batch_label': batch_label
                }
        return reshaped_adatas

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

        #pair_count = self.prep_paired_groups(pair_groups)

        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        #self.model.paired_dict = self.pair_groups_dict
        #self.model.unique_pairs_of_modalities = self.unique_pairs_of_modalities
        #self.model.modalities_per_group = self.modalities_per_group
        #self.model.paired_networks_per_modality_pairs = self.paired_networks_per_modality_pairs

        adatas = self.reshape_adatas(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        zs = []
        with torch.no_grad():
            self.model.eval()

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
                    zi = self.model.to_latent(x, mod, batch)
                    zij = self.model.convert(zi, pair, source_pair=True, dest=target_modality, dest_pair=False)

                    # assume data is paired for the decoder
                    hs = self.model.z_to_h(zij)
                    hs, pair_groups = self.model.decode_pairs([hs], [target_pair])
                    index_of_the_modality = np.where(np.array(self.modalities_per_group[target_pair]) == target_modality)[0][0]
                    xij = self.model.decode_from_shared(hs[index_of_the_modality], target_modality, target_pair, batch)

                    z = sc.AnnData(xij.detach().cpu().numpy())
                    modalities = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(modalities)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    zs.append(z)

        return sc.AnnData.concatenate(*zs)

    def test(self,
        adatas,
        names,
        pair_groups,
        modality_key='modality',
        celltype_key='cell_type',
        batch_size=64,
        batch_labels=None,
        #correct='all', #'missing'
        layers=[]
    ):
        print('old :(')
        if not batch_labels:
            batch_labels = self.batch_labels

        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        pair_count = self.prep_paired_groups(pair_groups)

        # TODO: check if need unique_pairs_of_modalities
        self.model.paired_dict = self.pair_groups_dict
        #self.model.unique_pairs_of_modalities = self.unique_pairs_of_modalities
        self.model.modalities_per_group = self.modalities_per_group
        self.model.paired_networks_per_modality_pairs = self.paired_networks_per_modality_pairs

        adatas = self.reshape_adatas(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        ad_all, ad_missing, ad_zs_latent, ad_hs_concat = [], [], [], []

        with torch.no_grad():
            self.model.eval()

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

                hs = [self.model.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
                hs_concat, new_pair_groups = self.model.encode_pairs(hs, pair_groups)
                #zs = [self.model.encode_shared(h) for h in hs_concat]
                zs = [self.model.bottleneck(h) for h in hs_concat]
                zs_latent = [z[0] for z in zs]
                zs_corrected_all = zs_latent.copy()
                zs_corrected_to_missing = zs_latent.copy()

                # todo matrix form
                for i, zi in enumerate(zs_latent):
                    zs_corrected_all[i] = self.model.convert(zs_corrected_all[i], new_pair_groups[i], source_pair=True, dest=None)

                for i, zi in enumerate(zs_latent):
                    zs_corrected_to_missing[i] = self.model.convert(zs_corrected_to_missing[i], new_pair_groups[i], source_pair=True, dest=0)

                #zs_pred, pair_groups = self.model.integrate(xs, modalities, pair_groups, batch_labels)

                for i, (z_corrected_all, z_latent, z_corrected_to_missing, pair) in enumerate(zip(zs_corrected_all, zs_latent, zs_corrected_to_missing, new_pair_groups)):
                    z = sc.AnnData(z_corrected_all.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['study'] = pair
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_all.append(z)

                    z = sc.AnnData(z_corrected_to_missing.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['study'] = pair
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_missing.append(z)

                    z = sc.AnnData(z_latent.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['study'] = pair
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_zs_latent.append(z)

                    # z = sc.AnnData(h.detach().cpu().numpy())
                    # modalities = np.array(names)[group_indices[pair], ]
                    # z.obs['modality'] = '-'.join(modalities)
                    # z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    # z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    # ad_hs_concat.append(z)

                for h, pg, mod, cell_type, idx in zip(hs, pair_groups, modalities, celltypes, indices):
                    z = sc.AnnData(h.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['study'] = pg
                    z.obs['barcode'] = idx
                    z.obs[celltype_key] = cell_type
                    ad_hs_concat.append(z)

        return sc.AnnData.concatenate(*ad_all), sc.AnnData.concatenate(*ad_missing), sc.AnnData.concatenate(*ad_zs_latent), sc.AnnData.concatenate(*ad_hs_concat)

    def predict(
        self,
        adatas,
        names,
        pair_groups,
        modality_key='modality',
        celltype_key='cell_type',
        batch_size=64,
        batch_labels=None,
        layers=[],
    ):
        if not batch_labels:
            batch_labels = self.batch_labels

        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        pair_count = self.prep_paired_groups(pair_groups)

        # TODO: check if need unique_pairs_of_modalities
        self.model.paired_dict = self.pair_groups_dict
        #self.model.unique_pairs_of_modalities = self.unique_pairs_of_modalities
        self.model.modalities_per_group = self.modalities_per_group
        self.model.paired_networks_per_modality_pairs = self.paired_networks_per_modality_pairs

        adatas = self.reshape_adatas(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        zs = []
        with torch.no_grad():
            self.model.eval()

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

                # forward propagation
                zs_pred, pair_groups = self.model.integrate(xs, modalities, pair_groups, batch_labels)

                for i, (z, pair) in enumerate(zip(zs_pred, pair_groups)):
                    z = sc.AnnData(z.detach().cpu().numpy())
                    modalities = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(modalities)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    zs.append(z)

        return sc.AnnData.concatenate(*zs)

    def train(
        self,
        n_iters=10000,
        batch_size=64,
        lr=3e-4,
        kl_anneal_iters=3000,
        val_split=0.1,
        modality_key='modality',
        celltype_key='cell_type',
        validate_every=1000,
        verbose=1,
        correct='all', # 'missing', 'none'
        version='1', # '1' or '2'
        kernel_type='gaussian',
        print_losses=False
    ):
        # configure training parameters
        print_every = n_iters // 50
        # create data loaders
        train_datasets, val_datasets = self.make_datasets(self.adatas, val_split, modality_key, celltype_key, batch_size)
        train_dataloaders = [d.loader for d in train_datasets]
        val_dataloaders = [d.loader for d in val_datasets]

        # create optimizers
        params = self.model.get_params()
        if self.theta is not None:
            params.extend([self.theta])
        optimizer_ae = torch.optim.Adam(params, lr)

        # the training loop
        epoch_time = 0  # epoch is the time between two consequtive validations
        self.model.train()
        for iteration, datas in enumerate(cycle(zip(*train_dataloaders))):
            tik = time.time()
            if iteration >= n_iters:
                break

            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]
            size_factors = [data[-2] for data in datas]

            # forward propagation
            # TODO fix
            out = self.model(xs, modalities, pair_groups, batch_labels, size_factors)
            #    out = self.model(xs, modalities, pair_groups, batch_labels)

            if len(out) == 5:
                rs, zs, mus, logvars, new_pair_groups = out
            elif len(out) == 9:
                rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors = out
            else:
                print('sth\'s wrong')
            losses = [self.losses[mod] for mod in modalities]

            recon_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)
            kl_loss = self.calc_kl_loss(mus, logvars)
            integ_loss = self.calc_integ_loss(zs, new_pair_groups, correct, kernel_type, version)
            if self.cycle_coef > 0:
                cycle_loss = self.calc_cycle_loss(xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses)
            else:
                cycle_loss = 0
            kl_coef = self.kl_anneal(iteration, kl_anneal_iters)  # KL annealing
            loss_ae = self.recon_coef * recon_loss + \
                      kl_coef * kl_loss + \
                      self.integ_coef * integ_loss + \
                      self.cycle_coef * cycle_loss

            # AE backpropagation
            optimizer_ae.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()

            # update progress bar
            if iteration % print_every == 0:
                self._train_history['iteration'].append(iteration)
                self._train_history['loss'].append(loss_ae.detach().cpu().item())
                self._train_history['recon'].append(recon_loss.detach().cpu().item())
                #self._train_history['recon_mse'].append(mse_loss.detach().cpu().item())
                self._train_history['recon_nb'].append(nb_loss.detach().cpu().item())
                #self._train_history['recon_zinb'].append(zinb_loss.detach().cpu().item())
                #self._train_history['recon_bce'].append(bce_loss.detach().cpu().item())
                self._train_history['kl'].append(kl_loss.detach().cpu().item())
                self._train_history['integ'].append(integ_loss.detach().cpu().item() if integ_loss != 0 else 0)
                self._train_history['cycle'].append(cycle_loss.detach().cpu().item() if cycle_loss != 0 else 0)
                self._train_history['norm_mod_vector0'].append(torch.linalg.norm(self.model.modality_vectors.weight[0]).detach().cpu().item())
                self._train_history['norm_mod_vector1'].append(torch.linalg.norm(self.model.modality_vectors.weight[1]).detach().cpu().item())
                #self._train_history['norm_mod_vector2'].append(torch.linalg.norm(self.model.modality_vectors.weight[2]).detach().cpu().item())
                if verbose >= 2:
                    self.print_progress_train(n_iters)

            # add this iteration to the epoch time
            epoch_time += time.time() - tik

            # validate
            if iteration > 0 and iteration % validate_every == 0 or iteration == n_iters - 1:
                # add average train losses of the elapsed epoch to the validation history
                self._val_history['iteration'].append(iteration)
                self._val_history['train_loss'].append(np.mean(self._train_history['loss'][-(validate_every//print_every):]))
                self._val_history['train_recon'].append(np.mean(self._train_history['recon'][-(validate_every//print_every):]))
                #self._val_history['train_recon_mse'].append(np.mean(self._train_history['recon_mse'][-(validate_every//print_every):]))
                self._val_history['train_recon_nb'].append(np.mean(self._train_history['recon_nb'][-(validate_every//print_every):]))
                #self._val_history['train_recon_zinb'].append(np.mean(self._train_history['recon_zinb'][-(validate_every//print_every):]))
                #self._val_history['train_recon_bce'].append(np.mean(self._train_history['recon_bce'][-(validate_every//print_every):]))
                self._val_history['train_kl'].append(np.mean(self._train_history['kl'][-(validate_every//print_every):]))
                self._val_history['train_integ'].append(np.mean(self._train_history['integ'][-(validate_every//print_every):]))
                self._val_history['train_cycle'].append(np.mean(self._train_history['cycle'][-(validate_every//print_every):]))
                self._val_history['norm_mod_vector0'].append(torch.linalg.norm(self.model.modality_vectors.weight[0]).detach().cpu().item())
                self._val_history['norm_mod_vector1'].append(torch.linalg.norm(self.model.modality_vectors.weight[1]).detach().cpu().item())
                #self._val_history['norm_mod_vector2'].append(torch.linalg.norm(self.model.modality_vectors.weight[2]).detach().cpu().item())

                self.model.eval()
                self.validate(val_dataloaders, n_iters, epoch_time, kl_coef=kl_coef, verbose=verbose, correct=correct, kernel_type=kernel_type, version=version)
                self.model.train()
                epoch_time = 0  # reset epoch time

    def validate(self, val_dataloaders, n_iters, train_time=None, kl_coef=None, verbose=1, correct='all', kernel_type='gaussian', version='1'):
        tik = time.time()
        val_n_iters = max([len(loader) for loader in val_dataloaders])
        if kl_coef is None:
            kl_coef = self.kl_coef

        # we want mean losses of all validation batches
        recon_loss = 0
        mse_l, nb_l, zinb_l, bce_l = 0, 0, 0, 0
        kl_loss = 0
        integ_loss = 0
        cycle_loss = 0
        for iteration, datas in enumerate(cycle(zip(*val_dataloaders))):
            # iterate until all of the dataloaders run out of data
            if iteration >= val_n_iters:
                break

            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]
            size_factors = [data[-2] for data in datas]

            losses = [self.losses[mod] for mod in modalities]

            # forward propagation
            # TODO fix
            rs, zs, mus, logvars, new_pair_groups, modalities, batch_labels, xs, size_factors = self.model(xs, modalities, pair_groups, batch_labels, size_factors)

            # calculate the losses
            r_loss, mse_loss, nb_loss, zinb_loss, bce_loss = self.calc_recon_loss(xs, rs, losses, batch_labels, size_factors)
            recon_loss += r_loss
            mse_l += mse_loss
            nb_l += nb_loss
            zinb_l += zinb_loss
            bce_l += bce_loss
            kl_loss += self.calc_kl_loss(mus, logvars)
            integ_loss += self.calc_integ_loss(zs, new_pair_groups, correct, kernel_type, version)
            if self.cycle_coef > 0:
                cycle_loss += self.calc_cycle_loss(xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses)

        # calculate overal losses
        loss_ae = self.recon_coef * recon_loss + \
                  kl_coef * kl_loss + \
                  self.integ_coef * integ_loss + \
                  self.cycle_coef * cycle_loss

        # logging
        self._val_history['val_loss'].append(loss_ae.detach().cpu().item() / val_n_iters)
        self._val_history['val_recon'].append(recon_loss.detach().cpu().item() / val_n_iters)
        #self._val_history['val_recon_mse'].append(mse_l.detach().cpu().item() / val_n_iters)
        self._val_history['val_recon_nb'].append(nb_l.detach().cpu().item() / val_n_iters)
        #self._val_history['val_recon_zinb'].append(zinb_l.detach().cpu().item() / val_n_iters)
        #self._val_history['val_recon_bce'].append(bce_l.detach().cpu().item() / val_n_iters)
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

        # print(losses)
        # print(batch_labels)
        # print(len(size_factors))
        # print(len(xs))
        # print(len(rs))

        for x, r, loss_type, batch, size_factor in zip(xs, rs, losses, batch_labels, size_factors):
            if loss_type == 'mse':
                mse_loss = self.loss_coef_dict['mse']*nn.MSELoss()(r, x)
                loss += mse_loss
                #print(f'MSE loss = {mse_loss}')
            elif loss_type == 'nb':
                dec_mean = r
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1)).to(self.device)
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[batch] if self.condition else self.theta.T[0]
                dispersion = torch.exp(dispersion)
                nb_loss = self.loss_coef_dict['nb']*NB()(x, dec_mean, dispersion)
                #print(f'NB loss = {-nb_loss}')
                loss -= nb_loss
            elif loss_type == 'zinb':
                dec_mean, dec_dropout = r
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1))
                dispersion = self.theta.T[batch] if self.condition else self.theta.T[0]
                dispersion = torch.exp(dispersion)
                zinb_loss = self.loss_coef_dict['zinb']*ZINB()(x, dec_mean, dispersion, dec_dropout)
                loss -= zinb_loss
                #print(f'NB loss = {-zinb_loss}')
            elif loss_type == 'bce':
                bce_loss = self.loss_coef_dict['bce']*nn.BCELoss()(r, x)
                loss += bce_loss
                #print(f'BCE loss = {bce_loss}')

        return loss, mse_loss, -nb_loss, -zinb_loss, bce_loss

    def calc_kl_loss(self, mus, logvars):
        return sum([KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])

    def calc_integ_loss(self, zs, pair_groups, correct, kernel_type, version):
        loss = 0
        for i, (zi, pgi) in enumerate(zip(zs, pair_groups)):
            for j, (zj, pgj) in enumerate(zip(zs, pair_groups)):
                if i == j:  # do not integrate one dataset with itself
                    continue

                if correct == 'all':
                    zij = self.model.convert(zi, pgi, source_pair=True, dest=None)
                elif correct == 'missing':
                    zij = self.model.convert(zi, pgi, source_pair=True, dest=pgj, dest_pair=True)
                elif correct == 'none':
                    zij = zi
                #zj = zj.detach()

                loss += MMD(kernel_type=kernel_type)(zij, zj)
                #loss += MMD()(zi, zj)
        return loss

    def calc_cycle_loss(self, xs, zs, pair_groups, modalities, batch_labels, new_pair_groups, losses):
        loss = 0
        for i, (xi, pgi, modi, batchi) in enumerate(zip(xs, pair_groups, modalities, batch_labels)):
            for j, (pgj, modj, batchj, lossj) in enumerate(zip(pair_groups, modalities, batch_labels, losses)):
                if i == j:  # do not make a dataset cycle consistent with itself
                    continue
                idx = np.argwhere(np.array(new_pair_groups) == pgi)[0][0]
                zij = self.model.convert(zs[idx], pgi, source_pair=True, dest=modj, dest_pair=False)
                rij = self.model.decode(zij, modj, batchj)
                if lossj in ['zinb']:
                    rij = rij[0]
                ziji = self.model.convert(self.model.to_latent(rij, modj, batchj), modj, source_pair=False, dest=modi, dest_pair=False)
                xiji = self.model.decode(ziji, modi, batchi)
                loss += nn.MSELoss()(xiji, xi)
        return loss

    def kl_anneal(self, iteration, anneal_iters):
        kl_coef = min(
            self.kl_coef,
            (iteration / anneal_iters) * self.kl_coef
        )
        return kl_coef

    def make_datasets(self, adatas, val_split, modality_key, celltype_key, batch_size):
        train_datasets, val_datasets = [], []
        pair_group_train_masks = {}
        for name in adatas:
            adata = adatas[name]['adata']
            modality = adatas[name][modality_key]
            pair_group = adatas[name]['pair_group']
            batch_label = adatas[name]['batch_label']
            layer = adatas[name]['layer']
            if pair_group in pair_group_train_masks:
                train_mask = pair_group_train_masks[pair_group]
            else:
                train_mask = np.zeros(len(adata), dtype=np.bool)
                train_size = int(len(adata) * (1 - val_split))
                train_mask[:train_size] = 1
                np.random.shuffle(train_mask)
                if pair_group is not None:
                    pair_group_train_masks[pair_group] = train_mask

            train_adata = adata[train_mask]
            val_adata = adata[~train_mask]

            train_dataset = SingleCellDataset(train_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label, layer=layer)
            val_dataset = SingleCellDataset(val_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label, layer=layer)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        train_datasets = sorted(train_datasets, key=attrgetter('pair_group', 'modality'))
        val_datasets = sorted(val_datasets, key=attrgetter('pair_group', 'modality'))

        return train_datasets, val_datasets

    def prep_paired_groups(self, pair_groups):
        # prepare dictionary with paired groups
        pair_groups_flat = [group for modal_pairs in pair_groups for group in modal_pairs]
        self.pair_counts = Counter(pair_groups_flat)
        pair_count = [k for k, v in self.pair_counts.items() if v > 1]
        self.pair_groups_dict = {v: i for i, v in enumerate(pair_count)}

        # modalities per paired group
        self.modalities_per_group = {}
        for pair in set(pair_groups_flat):
            self.modalities_per_group[pair] = []
            for i, group in enumerate(pair_groups):
                if pair in group:
                    self.modalities_per_group[pair].append(i)

        # unique modality pairs
        list_of_pairs = [val for val in self.modalities_per_group.values()]
        unique_list_of_pairs = [list(x) for x in set(tuple(x) for x in list_of_pairs)]
        self.unique_pairs_of_modalities = {}
        for i, pair_of_modalities in enumerate(unique_list_of_pairs):
            self.unique_pairs_of_modalities[i] = pair_of_modalities

        # which paired encoder/decoder to use for each pair
        self.paired_networks_per_modality_pairs = {}
        for index in self.unique_pairs_of_modalities:
            for pair in self.modalities_per_group:
                if self.unique_pairs_of_modalities[index] == self.modalities_per_group[pair]:
                    self.paired_networks_per_modality_pairs[pair] = index

        return pair_count

    def save(self, path):
        torch.save({
            'state_dict' : self.model.state_dict(),
        }, os.path.join(path, 'last-model.pt'), pickle_protocol=4)
        pd.DataFrame(self._val_history).to_csv(os.path.join(path, 'history.csv'))

    def load(self, path):
        model_file = torch.load(os.path.join(path, 'last-model.pt'), map_location=self.device)
        self.model.load_state_dict(model_file['state_dict'])
        self._val_history = pd.read_csv(os.path.join(path, 'history.csv'), index_col=0)
