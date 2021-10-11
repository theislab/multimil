import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from operator import attrgetter
from itertools import cycle, zip_longest, groupby
from scipy import spatial
from ..nn import MLP

from .multivae_torch import MultiVAETorch

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

class MultiVAETorch_MIL(MultiVAETorch):
    def __init__(self,
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
        paired_networks_per_modality_pairs={},
        num_classes=None,
        scoring='attn',
        classifier_hiddens=[],
        normalization='layer',
        dropout=None,
        attn_dim=32
    ):
        super().__init__(
            encoders,
            decoders,
            shared_encoder,
            shared_decoder,
            mu,
            logvar,
            modality_vectors,
            device,
            condition,
            n_batch_labels,
            paired_dict,
            modalities_per_group,
            paired_networks_per_modality_pairs
        )

        z_dim = self.modality_vectors.weight.shape[1]

        if len(classifier_hiddens) > 0:
            #classifier_hiddens.extend([classifier_hiddens[-1]]) # hack to make work with existing MLP module
            mil_dim = classifier_hiddens[-1]
        else:
            mil_dim = z_dim

        self.classifier = nn.Sequential(
                            MLP(z_dim, mil_dim, classifier_hiddens, output_activation='leakyrelu', # [:-1]
                                  dropout=dropout, norm=normalization, last_layer=False, regularize_last_layer=True),
                            Aggregator(mil_dim, scoring, attn_dim=attn_dim),
                            nn.Linear(mil_dim, num_classes)
                            )

        self = self.to(device)

    def get_params(self):
        params = super().get_params()
        params.extend(list(self.classifier.parameters()))
        return params

    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        out = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)
        zs = out[0]
        hs_dec = [self.z_to_h(z, mod) for z, mod in zip(zs, modalities)]
        rs = [self.decode_from_shared(h, mod, pair_group, batch_label) for h, mod, pair_group, batch_label in zip(hs_dec, modalities, pair_groups, batch_labels)]
        # classify
        predicted_scores = [self.classifier(z_joint) for z_joint in zs_joint]
        #if len(predicted_scores[0]) == 2:
    #        predicted_scores = [predicted_scores[i][0] for i in range(len(predicted_scores))]
        return rs, zs, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors, predicted_scores
