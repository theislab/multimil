import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from operator import attrgetter
from itertools import cycle, zip_longest, groupby
from ..nn import *

from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from ._multivae_torch import MultiVAETorch
from ..utils._utils import get_split_idx

class Aggregator(nn.Module):
    def __init__(self,
                n_input=None,
                scoring='sum',
                attn_dim=16, # D
                patient_batch_size=None,
                scale=False,
                attention_dropout=True,
                dropout=0.2,
                n_layers_mlp_attn=1,
                n_hidden_mlp_attn=16,
                ):
        super().__init__()

        self.scoring = scoring
        self.patient_batch_size = patient_batch_size
        self.scale = scale

        if self.scoring == 'attn':
            self.attn_dim = attn_dim # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            self.attention = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout) if attention_dropout else None,
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == 'gated_attn':
            self.attn_dim = attn_dim
            self.attention_V = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout) if attention_dropout else None,
            )

            self.attention_U = nn.Sequential(
                #orthogonal(nn.Linear(z_dim, self.attn_dim)),
                nn.Linear(n_input, self.attn_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout) if attention_dropout else None,
            )

            self.attention_weights = nn.Linear(self.attn_dim, 1, bias=False)

        elif self.scoring == 'MLP':

            if n_layers_mlp_attn == 1:
                self.attention = nn.Linear(n_input, 1)
            else:
                self.attention = nn.Sequential(
                                MLP(
                                    n_input,
                                    n_hidden_mlp_attn,
                                    n_layers=n_layers_mlp_attn-1,
                                    n_hidden=n_hidden_mlp_attn,
                                    dropout_rate=dropout,
                                ),
                                nn.Linear(n_hidden_mlp_attn, 1)
                            )

            

    def forward(self, x):
        if self.scoring == 'sum':
            return torch.sum(x, dim=0) # z_dim

        elif self.scoring == 'attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            self.A = self.attention(x)  # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N

        elif self.scoring == 'gated_attn':
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # NxD
            A_U = self.attention_U(x)  # NxD
            self.A = self.attention_weights(A_V * A_U) # element wise multiplication # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N

        elif self.scoring == 'MLP':
            self.A = self.attention(x) # N
            self.A = torch.transpose(self.A, -1, -2)
            self.A = F.softmax(self.A, dim=-1)

        if self.scale:
            self.A = self.A * self.A.shape[-1] / self.patient_batch_size

        return torch.bmm(self.A, x).squeeze(dim=1) # z_dim

class MultiVAETorch_MIL(BaseModuleClass):
    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        z_dim=16,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=16,
        kernel_type='gaussian',
        loss_coefs=[],
        num_groups=1,
        integrate_on_idx=None,
        n_layers_encoders = [],
        n_layers_decoders = [],
        n_layers_shared_decoder: int = 1,
        n_hidden_encoders = [],
        n_hidden_decoders = [],
        n_hidden_shared_decoder: int = 32,
        patient_idx=None,
        num_classes=None,
        scoring='gated_attn',
        attn_dim=16,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
        n_layers_cell_aggregator=1,
        n_layers_cov_aggregator=1,
        n_layers_classifier=1,
        n_layers_mlp_attn=1,
        n_hidden_cell_aggregator=16,
        n_hidden_cov_aggregator=16,
        n_hidden_classifier=16,
        n_hidden_mlp_attn=16,
        class_loss_coef=1.0,
        reg_coef=1, 
        add_patient_to_classifier=False,
        hierarchical_attn=True,
        patient_batch_size=128,
        regularize_cell_attn=False,
        regularize_cov_attn=False,
        regularize_vae=False,
        attention_dropout=True,
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
            integrate_on_idx=integrate_on_idx,
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_shared_decoder=n_layers_shared_decoder,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_shared_decoder=n_hidden_shared_decoder,
        )

        self.integrate_on_idx = integrate_on_idx
        self.class_loss_coef = class_loss_coef
        self.reg_coef = reg_coef
        self.add_patient_to_classifier = add_patient_to_classifier
        self.patient_idx = patient_idx
        self.hierarchical_attn = hierarchical_attn
        self.patient_batch_size = patient_batch_size
        self.regularize_cell_attn = regularize_cell_attn
        self.regularize_cov_attn = regularize_cov_attn
        self.regularize_vae = regularize_vae

        self.cond_dim = cond_dim
        self.cell_level_aggregator = nn.Sequential(
                            MLP(
                                z_dim,
                                cond_dim,
                                n_layers=n_layers_cell_aggregator,
                                n_hidden=n_hidden_cell_aggregator,
                                dropout_rate=dropout,
                            ),
                            Aggregator(
                                n_input=cond_dim, 
                                scoring=scoring, 
                                attn_dim=attn_dim, 
                                patient_batch_size=patient_batch_size, 
                                scale=True,
                                attention_dropout=attention_dropout,
                                dropout=dropout,
                                n_layers_mlp_attn=n_layers_mlp_attn,
                                n_hidden_mlp_attn=n_hidden_mlp_attn,
                            )
                        )
        if hierarchical_attn:
            self.cov_level_aggregator = nn.Sequential(
                                MLP(
                                   cond_dim,
                                   cond_dim,
                                   n_layers=n_layers_cov_aggregator,
                                   n_hidden=n_hidden_cov_aggregator,
                                   dropout_rate=dropout,
                                ),
                                Aggregator(
                                    n_input=cond_dim, 
                                    scoring=scoring, 
                                    attn_dim=attn_dim,
                                    attention_dropout=True,
                                    dropout=dropout,
                                    n_layers_mlp_attn=n_layers_mlp_attn,
                                    n_hidden_mlp_attn=n_hidden_mlp_attn,
                                )
                            )

        if n_layers_classifier == 1:
            self.classifier = nn.Linear(cond_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                                MLP(
                                    cond_dim,
                                    n_hidden_classifier,
                                    n_layers=n_layers_classifier-1,
                                    n_hidden=n_hidden_classifier,
                                    dropout_rate=dropout,
                                ),
                                nn.Linear(n_hidden_classifier, num_classes)
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

    @auto_move_data
    def inference(self, x, cat_covs, cont_covs):
        # vae part
        inference_outputs = self.vae.inference(x, cat_covs, cont_covs) # cat_covs is 1 longer than needed because of class label but it's taken care of in zip
        z_joint = inference_outputs['z_joint']

        # MIL part
        batch_size = x.shape[0]

        idx = list(range(self.patient_batch_size,  batch_size, self.patient_batch_size)) # or depending on model.train() and model.eval() ???
        if batch_size % self.patient_batch_size != 0: # can only happen during inference for last batches for each patient
            idx = []

        zs = torch.tensor_split(z_joint, idx, dim=0)
        zs = torch.stack(zs, dim=0)
        zs = self.cell_level_aggregator(zs) # num of bags in batch x cond_dim

        if self.hierarchical_attn:
            add_covariate = lambda i: self.add_patient_to_classifier or (not self.add_patient_to_classifier and i != self.patient_idx)
            if len(self.vae.cat_covariate_embeddings) > 0:
                cat_embedds = torch.cat([cat_covariate_embedding(covariate.long()) for i, (covariate, cat_covariate_embedding) in enumerate(zip(cat_covs.T, self.vae.cat_covariate_embeddings)) if add_covariate(i)], dim=-1)
            else:
                cat_embedds = torch.Tensor().to(self.device) # so cat works later

            if len(self.vae.cont_covariate_embeddings) > 0:
                cont_embedds = torch.cat([cont_covariate_embedding(torch.log1p(covariate.unsqueeze(-1))) for covariate, cont_covariate_embedding in zip(cont_covs.T, self.vae.cont_covariate_embeddings)], dim=-1)
            else:
                cont_embedds = torch.Tensor().to(self.device) # so cat works later

            cov_embedds = torch.cat([cat_embedds, cont_embedds], dim=-1)

            cov_embedds = torch.tensor_split(cov_embedds, idx)
            cov_embedds = [embed[0] for embed in cov_embedds]
            cov_embedds = torch.stack(cov_embedds, dim=0)

            aggr_bag_level = torch.cat([zs, cov_embedds], dim=-1)
            aggr_bag_level = torch.split(aggr_bag_level, self.cond_dim, dim=-1)
            aggr_bag_level = torch.stack(aggr_bag_level, dim=1) # num of bags in batch x num of cat covs + num of cont covs + 1 (molecular information) x cond_dim
            aggr_bag_level = self.cov_level_aggregator(aggr_bag_level)
            prediction = self.classifier(aggr_bag_level) # num of bags in batch x num of classes
        else:
            prediction = self.classifier(zs)

        inference_outputs.update({'prediction': prediction})
        return inference_outputs # z_joint, mu, logvar, prediction

    @auto_move_data
    def generative(self, z_joint, cat_covs, cont_covs):
        return self.vae.generative(z_joint, cat_covs, cont_covs)

    def orthogonal_regularization(self, weights, axis=0):
        loss = torch.tensor(0.0).to(self.device)
        for weight in weights:
            if axis == 1:
                weight = weight.T 
            dim = weight.shape[1]
            loss += torch.sqrt(torch.sum((torch.matmul(weight.T, weight) - torch.eye(dim).to(self.device)).pow(2)))
        return loss

    def loss(self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0
    ):
        x = tensors[_CONSTANTS.X_KEY]
        if self.integrate_on_idx:
            integrate_on = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, self.integrate_on_idx]
        else:
            integrate_on = torch.zeros(x.shape[0], 1).to(self.device)

        size_factor = tensors.get(_CONSTANTS.CONT_COVS_KEY)[:, -1] # always last
        class_label = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, -1] # always last

        rs = generative_outputs['rs']
        mu = inference_outputs['mu']
        logvar = inference_outputs['logvar']
        z_joint = inference_outputs['z_joint']
        prediction = inference_outputs['prediction']

        xs = torch.split(x, self.vae.input_dims, dim=-1) # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        masks = [x.sum(dim=1) > 0 for x in xs]

        recon_loss = self.vae.calc_recon_loss(xs, rs, self.vae.losses, integrate_on, size_factor, self.vae.loss_coefs, masks)
        kl_loss = kl(Normal(mu, torch.sqrt(torch.exp(logvar))), Normal(0, 1)).sum(dim=1)
        integ_loss = torch.tensor(0.0) if self.vae.loss_coefs['integ'] == 0 else self.vae.calc_integ_loss(z_joint, integrate_on)
        cycle_loss = torch.tensor(0.0) if self.vae.loss_coefs['cycle'] == 0 else self.vae.calc_cycle_loss(xs, z_joint, integrate_on, masks, self.vae.losses, size_factor, self.vae.loss_coefs)

        # MIL classification loss
        batch_size = x.shape[0]
        idx = list(range(self.patient_batch_size,  batch_size, self.patient_batch_size)) # or depending on model.train() and model.eval() ???
        if batch_size % self.patient_batch_size != 0: # can only happen during inference for last batches for each patient
            idx = []

        class_label = torch.tensor_split(class_label, idx, dim=0)
        class_label = [torch.Tensor([labels[0]]).long().to(self.device) for labels in class_label]
        class_label = torch.cat(class_label, dim=0)

        classification_loss = F.cross_entropy(prediction, class_label) # assume same in the batch
        accuracy = torch.sum(torch.eq(torch.argmax(prediction, dim=-1), class_label)) / class_label.shape[0]

        # what to regularize:
        weights = []
        if self.regularize_cov_attn:
            weights.append(self.cov_level_aggregator[1].attention_U[0].weight)
            weights.append(self.cov_level_aggregator[1].attention_V[0].weight)
        if self.regularize_cell_attn:
            weights.append(self.cell_level_aggregator[1].attention_U[0].weight)
            weights.append(self.cell_level_aggregator[1].attention_V[0].weight)

        # TODO: fix if other layers
        if self.regularize_vae:
            weights.append(self.vae.shared_decoder.encoder.fc_layers[0][0].weight)
            weights.append(self.vae.encoder_0.encoder.fc_layers[0][0].weight)
            weights.append(self.vae.encoder_1.encoder.fc_layers[0][0].weight)
            weights.append(self.vae.decoder_0.decoder.encoder.fc_layers[0][0].weight)
            weights.append(self.vae.decoder_1.decoder.encoder.fc_layers[0][0].weight)

        reg_loss = self.orthogonal_regularization(weights)

        loss = torch.mean(self.vae.loss_coefs['recon'] * recon_loss
            + self.vae.loss_coefs['kl'] * kl_loss
            + self.vae.loss_coefs['integ'] * integ_loss
            + self.vae.loss_coefs['cycle'] * cycle_loss
            + self.class_loss_coef * classification_loss
            + self.reg_coef * reg_loss
            )

        reconst_losses = dict(
            recon_loss = recon_loss
        )

        return LossRecorder(loss, reconst_losses, self.vae.loss_coefs['kl'] * kl_loss, kl_global=torch.tensor(0.0), integ_loss=integ_loss, cycle_loss=cycle_loss, class_loss=classification_loss, accuracy=accuracy, reg_loss=reg_loss)

    #TODO ??
    @torch.no_grad()
    def sample(self, tensors):
        with torch.no_grad():
            _, generative_outputs, = self.forward(
                tensors,
                compute_loss=False
            )

        return generative_outputs['rs']
