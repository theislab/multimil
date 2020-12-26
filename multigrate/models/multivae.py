import sys
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import scanpy as sc
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from itertools import cycle
from ..datasets import SingleCellDataset
from .mlp import MLP
from .losses import MMD, KLD


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
        adversarial_discriminator,
        device='cpu',
        condition=None,
        n_batch_labels=None
    ):
        super().__init__()

        self.encoders = encoders
        self.decoders = decoders
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.mu = mu
        self.logvar = logvar
        self.modality_vectors = modality_vectors
        self.adv_disc = adversarial_discriminator
        self.device = device
        self.condition = condition
        self.n_modality = len(self.encoders)
        self.n_batch_labels = n_batch_labels

        # register sub-modules
        for i, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(f'encoder_{i}', enc)
            self.add_module(f'decoder_{i}', dec)

        self = self.to(device)

    def get_nonadversarial_params(self):
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

    def get_adversarial_params(self):
        params = list(self.adv_disc.parameters())
        return params

    def encode(self, x, i, batch_label):
        # add batch labels
        if self.condition:
            x = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in x])
        h = self.x_to_h(x, i)
        # add batch labels and modality labels to hidden representation
        if self.condition == '1':
            h = torch.stack([torch.cat((cell, self.modal_vector(i)[0])) for cell in h])
        z = self.h_to_z(h, i)
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

    def decode(self, z, i, batch_label):
        if self.condition == '1':
            z = torch.stack([torch.cat((cell, self.modal_vector(i)[0])) for cell in z])
        h = self.z_to_h(z, i)
        if self.condition:
            h = torch.stack([torch.cat((cell, self.batch_vector(batch_label, i)[0])) for cell in h])
        x = self.h_to_x(h, i)
        return x

    def to_latent(self, x, i, batch_label):
        z, _, _ = self.encode(x, i, batch_label)
        return z

    def x_to_h(self, x, i):
        return self.encoders[i](x)

    def h_to_z(self, h, i):
        z = self.shared_encoder(h)
        return z

    def z_to_h(self, z, i):
        h = self.shared_decoder(z)
        return h

    def h_to_x(self, h, i):
        x = self.decoders[i](h)
        return x

    def adversarial_loss(self, z, y):
        y = torch.ones(z.size(0)).long().to(self.device) * y
        y_pred = self.adversarial_discriminator(z)
        return nn.CrossEntropyLoss()(y_pred, y)

    def forward(self, xs, modalities, batch_labels):
        zs = [self.encode(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        rs = [self.decode(z, mod, batch_label) for z, mod, batch_label in zip(zs, modalities, batch_labels)]
        return rs, zs, mus, logvars

    def calc_adv_loss(self, zs):
        zs = [self.convert(z, i) for i, z in enumerate(zs)]  # remove the modality informations from the Zs
        loss = sum([self.adversarial_loss(z, i) for i, z in enumerate(zs)])
        return self.adversarial * loss, {'adver': loss}

    def modal_vector(self, i):
        return F.one_hot(torch.tensor([i]).long(), self.n_modality).float().to(self.device)

    def batch_vector(self, i, modality):
        return F.one_hot(torch.tensor([i]).long(), self.n_batch_labels[modality]).float().to(self.device)

    def test(self, *xs):
        outputs, loss, losses = self.forward(*xs)
        return loss, losses

    def convert(self, z, i, j=None):
        v = -self.modal_vector(i)
        if j is not None:
            v += self.modal_vector(j)
        return z + v @ self.modality_vectors.weight

    def integrate(self, x, i, batch_label, j=None):
        zi = self.to_latent(x, i, batch_label)
        zij = self.convert(zi, i, j)
        return zij


class MultiVAE:
    def __init__(
        self,
        adatas,
        names,
        pair_groups,
        condition = None,
        z_dim=10,
        h_dim=32,
        hiddens=[],
        output_activations=[],
        shared_hiddens=[],
        adver_hiddens=[],
        recon_coef=1,
        kl_coef=1e-5,
        integ_coef=1e-1,
        cycle_coef=1e-2,
        adversarial=True,
        dropout=0.2,
        device=None,
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
        self.adversarial = adversarial
        self.condition = condition
        self.hiddens = hiddens
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.shared_hiddens = shared_hiddens
        self.adver_hiddens = adver_hiddens
        self.dropout = dropout
        self.output_activations = output_activations
        self.pair_groups = pair_groups

        # reshape hiddens into a dict for easier use in the following

        # TODO: change this hack with else 0
        self.x_dims = [modality_adatas[0].shape[1] if len(modality_adatas) > 0 else 0 for modality_adatas in adatas]  # the feature set size of each modality
        self.n_modality = len(self.x_dims)

        self.batch_labels = [list(range(len(modality_adatas))) for modality_adatas in adatas]
        self.adatas = self.reshape_adatas(adatas, names, pair_groups, self.batch_labels)

        self.n_batch_labels = [0]*self.n_modality
        n_mod_labels = 0
        if condition == '0':
            self.n_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
        elif condition == '1':
            self.n_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
            n_mod_labels = self.n_modality

        # create modules
        self.encoders = [MLP(x_dim + self.n_batch_labels[i], h_dim, hs, output_activation='leakyrelu',
                             dropout=dropout, batch_norm=True, regularize_last_layer=True) if x_dim > 0 else None for i, (x_dim, hs) in enumerate(zip(self.x_dims, hiddens))]
        self.decoders = [MLP(h_dim + self.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=dropout, batch_norm=True) if x_dim > 0 else None for i, (x_dim, hs, out_act) in enumerate(zip(self.x_dims, hiddens, output_activations))]
        self.shared_encoder = MLP(h_dim + n_mod_labels, z_dim, shared_hiddens, output_activation='leakyrelu',
                                  dropout=dropout, batch_norm=True, regularize_last_layer=True)
        self.shared_decoder = MLP(z_dim + n_mod_labels, h_dim, shared_hiddens[::-1], output_activation='leakyrelu',
                                  dropout=dropout, batch_norm=True, regularize_last_layer=True)
        self.mu = MLP(z_dim, z_dim)
        self.logvar = MLP(z_dim, z_dim)
        self.modality_vecs = nn.Embedding(self.n_modality, z_dim)
        self.adv_disc = MLP(z_dim, self.n_modality, adver_hiddens, dropout=dropout, batch_norm=True)

        self.model = MultiVAETorch(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.adv_disc, self.device, self.condition, self.n_batch_labels)

    @property
    def history(self):
        return pd.DataFrame(self._val_history)

    def reset_history(self):
        self._train_history = defaultdict(list)
        self._val_history = defaultdict(list)

    def reshape_adatas(self, adatas, names, pair_groups=None, batch_labels=None):
        if pair_groups is None:
            pair_groups = names  # dummy pair_groups
            # TODO: refactor this hack
        if batch_labels is None:
            batch_labels = names
        reshaped_adatas = {}
        for modality, (adata_set, name_set, pair_group_set, batch_label_set) in enumerate(zip(adatas, names, pair_groups, batch_labels)):
            # TODO: if sets are not lists, convert them to lists
            for adata, name, pair_group, batch_label in zip(adata_set, name_set, pair_group_set, batch_label_set):
                reshaped_adatas[name] = {
                    'adata': adata,
                    'modality': modality,
                    'pair_group': pair_group,
                    'batch_label': batch_label
                }
        return reshaped_adatas

    def print_progress_train(self, n_iters):
        current_iter = self._train_history['iteration'][-1]
        msg_train = 'iter={:d}/{:d}, loss={:.4f}, recon={:.4f}, kl={:.4f}, integ={:.4f}'.format(
            current_iter+1,
            n_iters,
            self._train_history['loss'][-1],
            self._train_history['recon'][-1],
            self._train_history['kl'][-1],
            self._train_history['integ'][-1]
        )
        self._print_progress_bar(current_iter+1, n_iters, prefix='', suffix=msg_train)

    def print_progress_val(self, n_iters, time_):
        current_iter = self._val_history['iteration'][-1]
        msg_train = 'iter={:d}/{:d}, time={:.2f}(s)' \
            ', loss={:.4f}, recon={:.4f}, kl={:.4f}, integ={:.4f}' \
            ', val_loss={:.4f}, val_recon={:.4f}, val_kl={:.4f}, val_integ={:.4f}'.format(
            current_iter+1,
            n_iters,
            time_,
            self._val_history['train_loss'][-1],
            self._val_history['train_recon'][-1],
            self._val_history['train_kl'][-1],
            self._val_history['train_integ'][-1],
            self._val_history['val_loss'][-1],
            self._val_history['val_recon'][-1],
            self._val_history['val_kl'][-1],
            self._val_history['val_integ'][-1]
        )
        self._print_progress_bar(current_iter+1, n_iters, prefix='', suffix=msg_train, end='\n')

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

    def predict(
        self,
        adatas,
        names,
        celltype_key='cell_type',
        batch_size=64,
        batch_labels=None
    ):
        if not batch_labels:
            batch_labels = self.batch_labels
        adatas = self.reshape_adatas(adatas, names, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]
        zs = []
        for loader in dataloaders:
            z = []
            celltypes = []
            for x, name, modality, _, celltype, batch_label in loader:
                x = x.to(self.device)
                z_pred = self.model.integrate(x, modality, batch_label)
                z.append(z_pred)
                celltypes.extend(celltype)
            z = torch.cat(z, dim=0)
            z = sc.AnnData(z.detach().cpu().numpy())
            z.obs['modality'] = name
            z.obs[celltype_key] = celltypes
            zs.append(z)

        obs_names = []
        for name in adatas.keys():
            obs_names.append(list(adatas[name]['adata'].obs_names))
        obs_names = [obs for sublist in obs_names for obs in sublist]
        res = sc.AnnData.concatenate(*zs)
        res.obs_names = obs_names
        return res

    def train(
        self,
        n_iters=10000,
        batch_size=64,
        lr=3e-4,
        kl_anneal_iters=3000,
        val_split=0.1,
        adv_iters=0,
        celltype_key='cell_type',
        validate_every=1000,
    ):
        # configure training parameters
        print_every = n_iters // 50

        # create data loaders
        train_datasets, val_datasets = self.make_datasets(self.adatas, val_split, celltype_key, batch_size)
        train_dataloaders = [d.loader for d in train_datasets]
        val_dataloaders = [d.loader for d in val_datasets]

        # create optimizers
        optimizer_ae = torch.optim.Adam(self.model.get_nonadversarial_params(), lr)
        optimizer_adv = torch.optim.Adam(self.model.get_adversarial_params(), lr)

        # the training loop
        epoch_time = 0  # epoch is the time between two consequtive validations
        self.model.train()
        for iteration, datas in enumerate(cycle(zip(*train_dataloaders))):
            tik = time.time()
            if iteration >= n_iters:
                break

            # TODO: refactor datas to be like (xs, modalities, pair_groups)
            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]

            # forward propagation
            rs, zs, mus, logvars = self.model.forward(xs, modalities, batch_labels)

            # calculate the losses
            recon_loss = self.calc_recon_loss(xs, rs)
            kl_loss = self.calc_kl_loss(mus, logvars)
            integ_loss = self.calc_integ_loss(zs, modalities, pair_groups)
            kl_coef = self.kl_anneal(iteration, kl_anneal_iters)  # KL annealing
            loss_ae = self.recon_coef * recon_loss + \
                      kl_coef * kl_loss + \
                      self.integ_coef * integ_loss
            loss_adv = -integ_loss

            # AE backpropagation
            optimizer_ae.zero_grad()
            optimizer_adv.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()

            # adversarial discriminator backpropagation
            for _ in range(adv_iters):
                optimizer_ae.zero_grad()
                optimizer_adv.zero_grad()
                loss_adv.backward()
                optimizer_adv.step()

            # TODO: fix this, if there is one one dataset then integration loss is 0
            # update progress bar
            if iteration % print_every == 0:
                self._train_history['iteration'].append(iteration)
                self._train_history['loss'].append(loss_ae.detach().cpu().item())
                self._train_history['recon'].append(recon_loss.detach().cpu().item())
                self._train_history['kl'].append(kl_loss.detach().cpu().item())
                if integ_loss != 0:
                    self._train_history['integ'].append(integ_loss.detach().cpu().item())
                else:
                    self._train_history['integ'] = [-1]
                self.print_progress_train(n_iters)

            # add this iteration to the epoch time
            epoch_time += time.time() - tik

            # validate
            if iteration > 0 and iteration % validate_every == 0 or iteration == n_iters - 1:
                # add average train losses of the elapsed epoch to the validation history
                self._val_history['iteration'].append(iteration)
                self._val_history['train_loss'].append(np.mean(self._train_history['loss'][-(validate_every//print_every):]))
                self._val_history['train_recon'].append(np.mean(self._train_history['recon'][-(validate_every//print_every):]))
                self._val_history['train_kl'].append(np.mean(self._train_history['kl'][-(validate_every//print_every):]))
                self._val_history['train_integ'].append(np.mean(self._train_history['integ'][-(validate_every//print_every):]))

                self.model.eval()
                self.validate(val_dataloaders, n_iters, epoch_time, kl_coef=kl_coef)
                self.model.train()
                epoch_time = 0  # reset epoch time

    def validate(self, val_dataloaders, n_iters, train_time=None, kl_coef=None):
        tik = time.time()
        val_n_iters = max([len(loader) for loader in val_dataloaders])
        if kl_coef is None:
            kl_coef = self.kl_coef

        # we want mean losses of all validation batches
        recon_loss = 0
        kl_loss = 0
        integ_loss = 0
        for iteration, datas in enumerate(cycle(zip(*val_dataloaders))):
            # iterate until all of the dataloaders run out of data
            if iteration >= val_n_iters:
                break

            # TODO: refactor datas to be like (xs, modalities, pair_groups)
            xs = [data[0].to(self.device) for data in datas]
            modalities = [data[2] for data in datas]
            pair_groups = [data[3] for data in datas]
            batch_labels = [data[-1] for data in datas]

            # forward propagation
            rs, zs, mus, logvars = self.model.forward(xs, modalities, batch_labels)

            # calculate the losses
            recon_loss += self.calc_recon_loss(xs, rs)
            kl_loss += self.calc_kl_loss(mus, logvars)
            integ_loss += self.calc_integ_loss(zs, modalities, pair_groups)

        # calculate overal losses
        loss_ae = self.recon_coef * recon_loss + \
                  kl_coef * kl_loss + \
                  self.integ_coef * integ_loss
        loss_adv = -integ_loss

        # logging
        self._val_history['val_loss'].append(loss_ae.detach().cpu().item() / val_n_iters)
        self._val_history['val_recon'].append(recon_loss.detach().cpu().item() / val_n_iters)
        self._val_history['val_kl'].append(kl_loss.detach().cpu().item() / val_n_iters)
        if integ_loss != 0:
            self._val_history['val_integ'].append(integ_loss.detach().cpu().item() / val_n_iters)
        else:
            self._val_history['val_integ'] = [-1]

        val_time = time.time() - tik
        self.print_progress_val(n_iters, train_time + val_time)

    def calc_recon_loss(self, xs, rs):
        return sum([nn.MSELoss()(r, x) for x, r in zip(xs, rs)])

    def calc_kl_loss(self, mus, logvars):
        return sum([KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])

    def calc_integ_loss(self, zs, modalities, pair_groups):
        loss = 0
        for i, (zi, modi, pgi) in enumerate(zip(zs, modalities, pair_groups)):
            for j, (zj, modj, pgj) in enumerate(zip(zs, modalities, pair_groups)):
                if i == j:  # do not integrate one dataset with itself
                    continue
                zij = self.model.convert(zi, modi, modj)
                if pgi is not None and pgi == pgj:  # paired loss
                    loss += nn.MSELoss()(zij, zj)
                else:  # unpaired loss
                    loss += MMD()(zij, zj)
        return loss

    def kl_anneal(self, iteration, anneal_iters):
        kl_coef = min(
            self.kl_coef,
            (iteration / anneal_iters) * self.kl_coef
        )
        return kl_coef

    def make_datasets(self, adatas, val_split, celltype_key, batch_size):
        train_datasets, val_datasets = [], []
        pair_groups_train_indices = {}
        for name in adatas:
            adata = adatas[name]['adata']
            modality = adatas[name]['modality']
            pair_group = adatas[name]['pair_group']
            batch_label = adatas[name]['batch_label']
            if pair_group in pair_groups_train_indices:
                train_indices = pair_groups_train_indices[pair_group]
            else:
                train_indices = np.arange(len(adata))
                np.random.shuffle(train_indices)
                train_size = int(len(adata) * (1 - val_split))
                train_indices = train_indices[:train_size]
                if pair_group is not None:
                    pair_groups_train_indices[pair_group] = train_indices

            train_adata = adata[train_indices]
            val_adata = adata[~train_indices]
            train_dataset = SingleCellDataset(train_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label)
            val_dataset = SingleCellDataset(val_adata, name, modality, pair_group, celltype_key, batch_size, batch_label=batch_label)
            train_datasets.insert(modality, train_dataset)
            val_datasets.insert(modality, val_dataset)

        return train_datasets, val_datasets
