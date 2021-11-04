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
            self.A = torch.transpose(self.A, 1, 0)  # 1xN
            self.A = F.softmax(self.A, dim=1)  # softmax over N
            return torch.mm(self.A, x).squeeze() # z_dim

        elif self.scoring == 'gated_attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # NxD
            A_U = self.attention_U(x)  # NxD
            self.A = self.attention_weights(A_V * A_U) # element wise multiplication # Nx1
            self.A = torch.transpose(self.A, 1, 0)  # 1xN
            self.A = F.softmax(self.A, dim=1)  # softmax over N
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
        covariate_embed_dim=10,
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
        )

        self.cat_covariate_embeddings = [nn.Embedding(dim, covariate_embed_dim) for dim in cat_covariate_dims]
        self.cont_covariate_embeddings = [nn.Linear(dim, covariate_embed_dim) for dim in cont_covariate_dims]

        self.class_loss_coef = class_loss_coef

        for i, emb in enumerate(self.cat_covariate_embeddings):
            self.add_module(f'cat_covariate_embedding_{i}', emb)

        for i, emb in enumerate(self.cont_covariate_embeddings):
            self.add_module(f'cont_covariate_embedding_{i}', emb)

        n_cat_covariates = len(cat_covariate_dims)
        n_cont_covariates = len(cont_covariate_dims)

        mil_dim = covariate_embed_dim

        self.cell_level_aggregator = nn.Sequential(
                            CondMLP(
                                z_dim,
                                mil_dim,
                                embed_dim=0,
                                n_layers=class_layers,
                                n_hidden=class_layer_size
                            ),
                            Aggregator(mil_dim, scoring, attn_dim=attn_dim)
                        )
        self.classifier = nn.Sequential(
                            CondMLP(
                                mil_dim,
                                mil_dim,
                                embed_dim=0,
                                n_layers=class_layers,
                                n_hidden=class_layer_size
                            ),
                            Aggregator(mil_dim, scoring, attn_dim=attn_dim),
                            nn.Linear(mil_dim, num_classes)
                        )


    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        group = tensors[_CONSTANTS.BATCH_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, group=group, cat_covs=cat_covs, cont_covs=cont_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs['z_joint']
        group = tensors[_CONSTANTS.BATCH_KEY]
        return dict(z_joint=z_joint, group=group)

    def inference(self, x, group, cat_covs, cont_covs, masks=None):
        # vae part
        inference_outputs = self.vae.inference(x, group)
        z_joint = inference_outputs['z_joint']
        # MIL part
        z_aggr = self.cell_level_aggregator(z_joint)

        # size factor and class label are included here but taking care of it later
        cat_covs = torch.split(cat_covs[0], 1, dim=-1) # list of tensors shape 1 x 1, assume all same in the batch
        cont_covs = torch.split(cont_covs[0], 1, dim=-1)

        cat_covs_embeds = [cat_covariate_embedding(covariate.squeeze().long()) for covariate, cat_covariate_embedding in zip(cat_covs, self.cat_covariate_embeddings)]
        cont_covs_embeds = [cont_covariate_embedding(covariate) for covariate, cont_covariate_embedding in zip(cat_covs, self.cont_covariate_embeddings)]

        cat_covs_embeds = torch.cat(cat_covs_embeds, dim=-1)
        cont_covs_embeds = torch.cat(cont_covs_embeds, dim=-1)
        aggr_bag_level = torch.stack([z_aggr, cat_covs_embeds, cont_covs_embeds], dim=0)
        prediction = self.classifier(aggr_bag_level)

        inference_outputs.update({'prediction': prediction})
        return inference_outputs # z_joint, mu, logvar, prediction

    def generative(self, z_joint, group):
        return self.vae.generative(z_joint, group)

    def loss(self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0
    ):
        x = tensors[_CONSTANTS.X_KEY]
        group = tensors[_CONSTANTS.BATCH_KEY]
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
        classification_loss = F.cross_entropy(prediction.unsqueeze(0), torch.Tensor([class_label[0]]).long()) # assume same in the batch

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

    #
    # def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
    #     hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
    #     zs = [self.bottleneck(h) for h in hs]
    #     mus = [z[1] for z in zs]
    #     logvars = [z[2] for z in zs]
    #     zs = [z[0] for z in zs]
    #     mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
    #     zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
    #     out = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)
    #     zs = out[0]
    #     hs_dec = [self.z_to_h(z, mod) for z, mod in zip(zs, modalities)]
    #     rs = [self.decode_from_shared(h, mod, pair_group, batch_label) for h, mod, pair_group, batch_label in zip(hs_dec, modalities, pair_groups, batch_labels)]
    #     # classify
    #     predicted_scores = [self.classifier(z_joint) for z_joint in zs_joint]
    #     #if len(predicted_scores[0]) == 2:
    # #        predicted_scores = [predicted_scores[i][0] for i in range(len(predicted_scores))]
    #     return rs, zs, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors, predicted_scores
