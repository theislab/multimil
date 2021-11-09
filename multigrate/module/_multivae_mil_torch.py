import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from operator import attrgetter
from itertools import cycle, zip_longest, groupby
from ..nn import *

from scvi.module.base import BaseModuleClass, LossRecorder
from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from ._multivae_torch import MultiVAETorch
from ..utils._utils import get_split_idx

class Aggregator(nn.Module):
    def __init__(self,
                z_dim=None,
                scoring='sum',
                attn_dim=32 # D
                ):
        super().__init__()

        self.scoring = scoring

        if self.scoring == 'attn':
            self.attn_dim = attn_dim # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            self.attention = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Tanh(),
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == 'gated_attn':
            self.attn_dim = attn_dim
            self.attention_V = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Tanh()
            )

            self.attention_U = nn.Sequential(
                nn.Linear(z_dim, self.attn_dim),
                nn.Sigmoid()
            )

            self.attention_weights = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, x):
        if self.scoring == 'sum':
            return torch.sum(x, dim=0) # z_dim
        elif self.scoring == 'attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            self.A = self.attention(x)  # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N
            if len(x.shape) == 3:
                return torch.bmm(self.A, x).squeeze(dim=1) # z_dim
            elif len(x.shape) == 2:
                return torch.mm(self.A, x).squeeze()

        elif self.scoring == 'gated_attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # NxD
            A_U = self.attention_U(x)  # NxD
            self.A = self.attention_weights(A_V * A_U) # element wise multiplication # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N
            if len(x.shape) == 3:
                return torch.bmm(self.A, x).squeeze() # z_dim
            elif len(x.shape) == 2:
                return torch.mm(self.A, x).squeeze()  # z_dim

class MultiVAETorch_MIL(BaseModuleClass):
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
        # mil specific
        num_classes=None,
        scoring='attn',
        attn_dim=32,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
        class_layers=1,
        class_layer_size=128,
        class_loss_coef=1.0
    ):
        super().__init__()

        self.vae = MultiVAETorch(
            modality_lengths=modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            h_dim=h_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=num_groups,
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
        )

        self.class_loss_coef = class_loss_coef

        self.cond_dim = cond_dim
        self.cell_level_aggregator = nn.Sequential(
                            CondMLP(
                                z_dim,
                                cond_dim,
                                embed_dim=0,
                                n_layers=class_layers,
                                n_hidden=class_layer_size
                            ),
                            Aggregator(cond_dim, scoring, attn_dim=attn_dim)
                        )
        self.classifier = nn.Sequential(
                            CondMLP(
                                cond_dim,
                                cond_dim,
                                embed_dim=0,
                                n_layers=class_layers,
                                n_hidden=class_layer_size
                            ),
                            Aggregator(cond_dim, scoring, attn_dim=attn_dim),
                            nn.Linear(cond_dim, num_classes)
                        )

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
        # vae part
        inference_outputs = self.vae.inference(x, cat_covs, cont_covs)
        z_joint = inference_outputs['z_joint']


        # MIL part
        class_label = cat_covs[:, -1].cpu().detach().numpy() # always last
        idx = get_split_idx(class_label)

        cat_embedds = torch.cat([cat_covariate_embedding(covariate.long()) for covariate, cat_covariate_embedding in zip(cat_covs.T, self.vae.cat_covariate_embeddings)], dim=-1)
        cont_embedds = torch.cat([cont_covariate_embedding(torch.log1p(covariate.unsqueeze(-1))) for covariate, cont_covariate_embedding in zip(cont_covs.T, self.vae.cont_covariate_embeddings)], dim=-1)

        cov_embedds = torch.cat([cat_embedds, cont_embedds], dim=-1)

        cov_embedds = torch.tensor_split(cov_embedds, idx)
        cov_embedds = [embed[0] for embed in cov_embedds]
        cov_embedds = torch.stack(cov_embedds, dim=0)

        zs = torch.tensor_split(z_joint, idx, dim=0)
        zs = torch.stack(zs, dim=0)
        zs = self.cell_level_aggregator(zs)

        aggr_bag_level = torch.cat([zs, cov_embedds], dim=-1)
        aggr_bag_level = torch.split(aggr_bag_level, self.cond_dim, dim=-1)
        aggr_bag_level = torch.stack(aggr_bag_level, dim=1) # num of bags in batch x num of cat covs + num of cont covs + 1 (molecular information) x cond_dim
        prediction = self.classifier(aggr_bag_level) # num of bags in batch x num of classes

        inference_outputs.update({'prediction': prediction})
        return inference_outputs # z_joint, mu, logvar, prediction

    def generative(self, z_joint, cat_covs, cont_covs):
        return self.vae.generative(z_joint, cat_covs, cont_covs)

    def loss(self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0
    ):
        x = tensors[_CONSTANTS.X_KEY]
        group = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, -2] # always second to last
        size_factor = tensors.get(_CONSTANTS.CONT_COVS_KEY)[:, -1] # always last
        class_label = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, -1] # always last

        rs = generative_outputs['rs']
        mu = inference_outputs['mu']
        logvar = inference_outputs['logvar']
        z_joint = inference_outputs['z_joint']
        prediction = inference_outputs['prediction']

        xs = torch.split(x, self.vae.input_dims, dim=-1) # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        masks = [x.sum(dim=1) > 0 for x in xs]

        recon_loss = self.vae.calc_recon_loss(xs, rs, self.vae.losses, group, size_factor, self.vae.loss_coefs, masks)
        kl_loss = kl(Normal(mu, torch.sqrt(torch.exp(logvar))), Normal(0, 1)).sum(dim=1)
        integ_loss = 0 if self.vae.loss_coefs['integ'] == 0 else self.vae.calc_integ_loss(z_joint, group)
        cycle_loss = 0 if self.vae.loss_coefs['cycle'] == 0 else self.vae.calc_cycle_loss(xs, z_joint, group, masks, self.vae.losses, size_factor, self.vae.loss_coefs)

        # MIL classification loss
        idx = get_split_idx(class_label.cpu().detach().numpy())
        class_label = torch.tensor_split(class_label, idx, dim=0)
        class_label = [torch.Tensor([labels[0]]).long() for labels in class_label]
        class_label = torch.cat(class_label, dim=0)

        classification_loss = F.cross_entropy(prediction, class_label) # assume same in the batch

        loss = torch.mean(self.vae.loss_coefs['recon'] * recon_loss
            + self.vae.loss_coefs['kl'] * kl_loss
            + self.vae.loss_coefs['integ'] * integ_loss
            + self.vae.loss_coefs['cycle'] * cycle_loss
            + self.class_loss_coef * classification_loss
            )

        reconst_losses = dict(
            recon_loss = recon_loss
        )

        # TODO record additional losses
        return LossRecorder(loss, reconst_losses, self.vae.loss_coefs['kl'] * kl_loss, kl_global=torch.tensor(0.0), integ_loss=integ_loss, cycle_loss=cycle_loss)

    #TODO ??
    @torch.no_grad()
    def sample(self, tensors):
        with torch.no_grad():
            _, generative_outputs, = self.forward(
                tensors,
                compute_loss=False
            )

        return generative_outputs['rs']
