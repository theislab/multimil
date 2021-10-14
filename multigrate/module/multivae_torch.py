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
from operator import itemgetter, attrgetter
from scvi.module.base import BaseModuleClass

# TODO: inherit from BaseModuleClass
class MultiVAETorch(nn.Module):
    def __init__(
        self,
        encoders,
        decoders,
        shared_decoder,
        mu,
        logvar,
        device='cpu',
        condition=None,
        cond_embedding=None
    ):
        super().__init__()

        self.encoders = encoders
        self.decoders = decoders
        self.shared_decoder = shared_decoder
        self.mu = mu
        self.logvar = logvar
        self.device = device
        self.condition = condition
        self.cond_embedding = cond_embedding
        self.n_modality = len(self.encoders)

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

        params.extend(list(self.shared_decoder.parameters()))
        params.extend(list(self.mu.parameters()))
        params.extend(list(self.logvar.parameters()))
        if self.condition:
            params.extend(list(self.cond_embedding.parameters()))
        return params

    def to_shared_dim(self, x, mod, group):
        if self.condition:
            cond_emb_vec = self.cond_embedding(torch.tensor([group]))
            x = torch.cat([x, cond_emb_vec.repeat(x.shape[0], 1)], dim=-1)
        return self.x_to_h(x, mod)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_from_shared(self, h, mod, group):
        if self.condition:
            cond_emb_vec = self.cond_embedding(torch.tensor([group]))
            h = torch.cat([h, cond_emb_vec.repeat(h.shape[0], 1)], dim=-1)
        x = self.h_to_x(h, mod)
        return x

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

    def product_of_experts(self, mus, logvars, groups):
        """
           This PoE function was adjusted from:
           Title: mvTCR
           Date: 9th October 2021
           Availability: https://github.com/SchubertLab/mvTCR/blob/387789f774d19eff2778b39b3e4ad3758edf5d0a/tcr_embedding/models/poe.py
        """
        mus_joint = []
        logvars_joint = []
        current = 0

        for pair, group in groupby(groups):
            group_size = len(list(group))

            if group_size == 1:
                current += 1
                continue
		    # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
            logvar_joint = 1.0
            # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, where mu_prior = 0.0
            mu_joint = 0.0

            for i in range(current, current+group_size):
                logvar_joint += 1.0 / torch.exp(logvars[i]) # sum up all inverse vars, logvars first needs to be converted to var, last 1.0 is coming from the prior
                mu_joint += mus[i] * (1.0 / torch.exp(logvars[i]))

            current += group_size
            logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar
            mu_joint = mu_joint * torch.exp(logvar_joint)

            mus_joint.append(mu_joint)
            logvars_joint.append(logvar_joint)

        return mus_joint, logvars_joint

    def prep_latent(self, zs, zs_joint, groups):
        zs_new = []
        current_joint = 0 # index for joint zs
        current = 0
        for pair, group in groupby(groups):
            group_size = len(list(group))
            if group_size == 1:
                zs_new.append(zs[current])
            else:
                zs_new.extend([zs_joint[current_joint]]*group_size)
                current_joint += 1

            current += group_size
        return zs_new

    def forward(self, xs, modalities, groups):
        # get latent
        zs, zs_joint, mus, logvars = self.inference(xs, modalities, groups)
        # prepare to decode
        zs = self.prep_latent(zs, zs_joint, groups)
        # decode
        hs_dec = [self.z_to_h(z, mod) for z, mod in zip(zs, modalities)]
        rs = [self.decode_from_shared(h, mod, group) for h, mod, group in zip(hs_dec, modalities, groups)]
        return rs, zs, mus, logvars

    def _get_inference_input():
        pass

    def _get_generative_input():
        pass

    # TODO: adjust to scvi framework
    def inference(self, xs, modalities, groups):
        hs = [self.to_shared_dim(x, mod, group) for x, mod, group in zip(xs, modalities, groups)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint = self.product_of_experts(mus, logvars, groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        return zs, zs_joint, mus+mus_joint, logvars+logvars_joint

    def generative():
        pass

    def loss():
        pass

    def sample():
        pass
