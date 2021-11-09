
import sys
import time
import os
import torch

import numpy as np
import pandas as pd
import scanpy as sc

from torch import nn
from torch.nn import functional as F
from itertools import groupby
from collections import defaultdict, Counter
from ..distributions import *
from ..nn import *
from operator import itemgetter, attrgetter
from scvi.module.base import BaseModuleClass, LossRecorder
from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

class MultiVAETorch(BaseModuleClass):
    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=10,
        kernel_type='not gaussian',
        loss_coefs=[],
        num_groups=1,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
    ):
        super().__init__()

        self.input_dims = modality_lengths
        self.condition_encoders = condition_encoders
        self.condition_decoders = condition_decoders
        self.input_dims = modality_lengths
        self.n_modality = len(self.input_dims)
        self.kernel_type = kernel_type

        # TODO fix
        if len(losses) == 0:
            self.losses = ['mse']*self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(f'losses has to be the same length as the number of modalities. number of modalities = {self.n_modality} != {len(losses)} = len(losses)')

        self.loss_coefs = {'recon': 1,
                          'kl': 1e-6,
                          'integ': 1e-2,
                          'cycle': 0,
                          'nb': 1,
                          'zinb': 1,
                          'mse': 1,
                          'bce': 1}
        self.loss_coefs.update(loss_coefs)

        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        # TODO: add check for multiple nb/zinb losses given
        self.theta = None
        for i, loss in enumerate(losses):
            if loss in ["nb", "zinb"]:
                self.theta = torch.nn.Parameter(torch.randn(self.input_dims[i], num_groups))#.to(device).detach().requires_grad_(True)
                break

        self.shared_decoder = CondMLP(z_dim + self.n_modality, h_dim, dropout_rate=dropout, normalization=normalization)
        cond_dim_enc = cond_dim*(len(cat_covariate_dims) + len(cont_covariate_dims)) if self.condition_encoders else 0
        self.encoders = [CondMLP(x_dim + cond_dim_enc, z_dim, embed_dim=0, dropout_rate=dropout, normalization=normalization) for x_dim in self.input_dims]
        cond_dim_dec = cond_dim*(len(cat_covariate_dims) + len(cont_covariate_dims)) if self.condition_decoders else 0
        self.decoders = [Decoder(h_dim + cond_dim_dec, x_dim, embed_dim=0, dropout_rate=dropout, normalization=normalization, loss=loss) for x_dim, loss in zip(self.input_dims, self.losses)]

        self.mus = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]
        self.logvars = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]

        self.cat_covariate_embeddings = [nn.Embedding(dim, cond_dim) for dim in cat_covariate_dims]
        self.cont_covariate_embeddings = [nn.Linear(dim, cond_dim) for dim in cont_covariate_dims] # dim is always 1 here

        # register sub-modules
        for i, (enc, dec, mu, logvar) in enumerate(zip(self.encoders, self.decoders, self.mus, self.logvars)):
            self.add_module(f'encoder_{i}', enc)
            self.add_module(f'decoder_{i}', dec)
            self.add_module(f'mu_{i}', mu)
            self.add_module(f'logvar_{i}', logvar)

        for i, emb in enumerate(self.cat_covariate_embeddings):
            self.add_module(f'cat_covariate_embedding_{i}', emb)

        for i, emb in enumerate(self.cont_covariate_embeddings):
            self.add_module(f'cont_covariate_embedding_{i}', emb)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, z, i):
        mu = self.mus[i](z)
        logvar = self.logvars[i](z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def x_to_h(self, x, i):
        return self.encoders[i](x)

    def h_to_x(self, h, i):
        x = self.decoders[i](h)
        return x

    def z_to_h(self, z, mod):
        z = torch.stack([torch.cat((cell, self.modal_vector(mod)[0])) for cell in z])
        h = self.shared_decoder(z)
        return h

    def modal_vector(self, i):
        return F.one_hot(torch.tensor([i]).long(), self.n_modality).float().to(self.device)

    def product_of_experts(self, mus, logvars, masks):
        vars = torch.exp(logvars)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars.shape[-1])
        mus_joint = torch.sum(mus * masks / vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint) # batch size
        logvars_joint += torch.sum(masks / vars, dim=1)
        logvars_joint = 1.0 / logvars_joint # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, cat_covs=cat_covs, cont_covs=cont_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs['z_joint']

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return dict(z_joint=z_joint, cat_covs=cat_covs, cont_covs=cont_covs)

    def inference(self, x, cat_covs, cont_covs, masks=None):
        # cat_covs or cont_covs can be longer than xs in case of MIL and class labels or size_factors,
        # but it's taken care of later by using zip

        # split x into modality xs
        if torch.is_tensor(x):
            xs = torch.split(x, self.input_dims, dim=-1) # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        else:
            xs = x

        if masks is None:
            masks = [x.sum(dim=1) > 0 for x in xs] # list of masks per modality
            masks = torch.stack(masks, dim=1)

        if self.condition_encoders:

            cat_embedds = [cat_covariate_embedding(covariate.long()) for covariate, cat_covariate_embedding in zip(cat_covs.T, self.cat_covariate_embeddings)]
            cont_embedds = [cont_covariate_embedding(torch.log1p(covariate.unsqueeze(-1))) for covariate, cont_covariate_embedding in zip(cont_covs.T, self.cont_covariate_embeddings)]

            if len(cat_embedds) > 0:
                cat_embedds = torch.cat(cat_embedds, dim=-1)
            else:
                cat_embedds = torch.Tensor()

            if len(cont_embedds) > 0:
                cont_embedds = torch.cat(cont_embedds, dim=-1)
            else:
                cont_embedds = torch.Tensor()

            xs = [torch.cat([x, cat_embedds, cont_embedds], dim=-1) for x in xs] # concat embedding to each modality x along the feature axis

        hs = [self.x_to_h(x, mod) for mod, x in enumerate(xs)]
        out = [self.bottleneck(h, mod) for mod, h in enumerate(hs)]
        mus = [mod_out[1] for mod_out in out]
        mu = torch.stack(mus, dim=1)
        logvars = [mod_out[2] for mod_out in out]
        logvar = torch.stack(logvars, dim=1)
        mu_joint, logvar_joint = self.product_of_experts(mu, logvar, masks)
        z_joint = self.reparameterize(mu_joint, logvar_joint)
        # drop mus and logvars according to masks for kl calculation
        # TODO here or in loss calculation? check multiVI
        # return mus+mus_joint
        return dict(z_joint=z_joint, mu=mu_joint, logvar=logvar_joint)

    def generative(self, z_joint, cat_covs, cont_covs):
        mod_vecs = self.modal_vector(list(range(self.n_modality))) # shape 1 x n_mod x n_mod
        z_joint = z_joint.unsqueeze(1).repeat(1, self.n_modality, 1)
        mod_vecs = mod_vecs.repeat(z_joint.shape[0], 1, 1) # shape batch_size x n_mod x n_mod

        z_joint = torch.cat([z_joint, mod_vecs], dim=-1) # shape batch_size x n_mod x latent_dim+n_mod
        z = self.shared_decoder(z_joint)
        zs = torch.split(z, 1, dim=1)

        if self.condition_decoders:

            cat_embedds = [cat_covariate_embedding(covariate.long()) for covariate, cat_covariate_embedding in zip(cat_covs.T, self.cat_covariate_embeddings)]
            cont_embedds = [cont_covariate_embedding(torch.log1p(covariate.unsqueeze(-1))) for covariate, cont_covariate_embedding in zip(cont_covs.T, self.cont_covariate_embeddings)]

            if len(cat_embedds) > 0:
                cat_embedds = torch.cat(cat_embedds, dim=-1)
            else:
                cat_embedds = torch.Tensor()

            if len(cont_embedds) > 0:
                cont_embedds = torch.cat(cont_embedds, dim=-1)
            else:
                cont_embedds = torch.Tensor()

            zs = [torch.cat([z.squeeze(1), cat_embedds, cont_embedds], dim=-1) for z in zs] # concat embedding to each modality x along the feature axis

        rs = [self.h_to_x(z, mod) for mod, z in enumerate(zs)]
        return dict(rs=rs)

    def loss(self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0
    ):

        x = tensors[_CONSTANTS.X_KEY]
        group = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, -1] # always last
        size_factor = tensors.get(_CONSTANTS.CONT_COVS_KEY)[:, -1] # always last
        rs = generative_outputs['rs']
        mu = inference_outputs['mu']
        logvar = inference_outputs['logvar']
        z_joint = inference_outputs['z_joint']

        xs = torch.split(x, self.input_dims, dim=-1) # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        masks = [x.sum(dim=1) > 0 for x in xs]

        recon_loss = self.calc_recon_loss(xs, rs, self.losses, group, size_factor, self.loss_coefs, masks)
        kl_loss = kl(Normal(mu, torch.sqrt(torch.exp(logvar))), Normal(0, 1)).sum(dim=1)
        integ_loss = 0 if self.loss_coefs['integ'] == 0 else self.calc_integ_loss(z_joint, group)
        cycle_loss = 0 if self.loss_coefs['cycle'] == 0 else self.calc_cycle_loss(xs, z_joint, group, masks, self.losses, size_factor, self.loss_coefs)

        loss = torch.mean(self.loss_coefs['recon'] * recon_loss  + self.loss_coefs['kl'] * kl_loss + self.loss_coefs['integ'] * integ_loss + self.loss_coefs['cycle'] * cycle_loss)
        reconst_losses = dict(
            recon_loss = recon_loss
        )

        return LossRecorder(loss, reconst_losses, self.loss_coefs['kl'] * kl_loss, kl_global=torch.tensor(0.0), integ_loss=integ_loss, cycle_loss=cycle_loss)

    #TODO ??
    @torch.no_grad()
    def sample(self, tensors):
        with torch.no_grad():
            _, generative_outputs, = self.forward(
                tensors,
                compute_loss=False
            )

        return generative_outputs['rs']

    def calc_recon_loss(self, xs, rs, losses, group, size_factor, loss_coefs, masks):
        loss = []
        condition = self.condition_encoders or self.condition_decoders
        for i, (x, r, loss_type) in enumerate(zip(xs, rs, losses)):
            if len(r) != 2 and len(r.shape) == 3:
                r = r.squeeze()
            if loss_type == 'mse':
                mse_loss = loss_coefs['mse']*torch.sum(nn.MSELoss(reduction='none')(r, x), dim=-1)
                loss.append(mse_loss)
            elif loss_type == 'nb':
                dec_mean = r
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1)).to(self.device)
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()] if condition else self.theta.T[0].unsqueeze(0).repeat(group.shape[0], 1)
                dispersion = torch.exp(dispersion)
                nb_loss = torch.sum(NegativeBinomial(mu=dec_mean, theta=dispersion).log_prob(x), dim=-1)
                nb_loss = loss_coefs['nb']*nb_loss
                loss.append(-nb_loss)
            elif loss_type == 'zinb':
                dec_mean, dec_dropout = r
                dec_mean = dec_mean.squeeze()
                dec_dropout = dec_dropout.squeeze()
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1)).to(self.device)
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()] if condition else self.theta.T[0].unsqueeze(0).repeat(group.shape[0], 1)
                dispersion = torch.exp(dispersion)
                zinb_loss = torch.sum(ZeroInflatedNegativeBinomial(mu=dec_mean, theta=dispersion, zi_logits=dec_dropout).log_prob(x), dim=-1)
                zinb_loss = loss_coefs['zinb']*zinb_loss
                loss.append(-zinb_loss)
            elif loss_type == 'bce':
                bce_loss = loss_coefs['bce']*torch.sum(torch.nn.BCELoss(reduction='none')(r, x), dim=-1)
                loss.append(bce_loss)

        return torch.sum(torch.stack(loss, dim=-1)*torch.stack(masks, dim=-1), dim=1)

    def calc_integ_loss(self, z, group):
        loss = 0
        zs = []

        for g in set(list(group.squeeze().numpy())):
            idx = (group == g).nonzero(as_tuple=True)[0]
            zs.append(z[idx])

        for i, zi in enumerate(zs):
            for j, zj in enumerate(zs):
                if i == j:  # do not integrate one dataset with itself
                    continue
                loss += MMD(kernel_type=self.kernel_type)(zi, zj)
        return loss

    def calc_cycle_loss(self, xs, z, cat_covs, cont_covs, masks, losses, size_factor, loss_coefs):

        generative_outputs = self.generative(z, cat_covs, cont_covs)
        rs = generative_outputs['rs']
        for i, r in enumerate(rs):
            if len(r) == 2: # hack for zinb
                rs[i] = r[0]
            rs[i] = rs[i].squeeze()

        masks_stacked = torch.stack(masks, dim=1)
        complement_masks = torch.logical_not(masks_stacked)

        inference_outputs = self.inference(rs, cat_covs, cont_covs, complement_masks)
        z_joint = inference_outputs['z_joint']
        # generate again
        generative_outputs = self.generative(z_joint, cat_covs, cont_covs)
        rs = generative_outputs['rs']

        group = cat_covs[:, -1]

        return self.calc_recon_loss(xs, rs, losses, group, size_factor, loss_coefs, masks)
