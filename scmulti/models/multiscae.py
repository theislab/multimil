import torch
from torch import nn
from torch.nn import functional as F
from .mlp import MLP
from .losses import MMD


class MultiScAE(nn.Module):
    def __init__(self, x_dims,
                 z_dim=10,
                 h_dim=32,
                 hiddens=[],
                 shared_hiddens=[],
                 recon_coef=1,
                 cross_coef=1,
                 integ_coef=1,
                 cycle_coef=1,
                 dropout=0.2,
                 pair_groups=[],
                 shared_encoder_output_activation='linear',
                 regularize_shared_encoder_last_layer=False,
                 device='cpu'):

        super(MultiScAE, self).__init__()

        # save model parameters
        self.n_modal = len(x_dims)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.recon_coef = recon_coef
        self.cross_coef = self.cross_coef_init = cross_coef
        self.integ_coef = self.integ_coef_init = integ_coef
        self.cycle_coef = self.cycle_coef_init = cycle_coef
        self.pair_groups = pair_groups
        self.device = device

        # TODO: do some assertions for the model parameters

        # create sub-modules
        self.encoders = [MLP(x_dim, h_dim, hiddens, output_activation='leakyrelu',
                             dropout=dropout, batch_norm=True, regularize_last_layer=True) for x_dim in x_dims]
        self.decoders = [MLP(h_dim, x_dim, hiddens[::-1], dropout=dropout, batch_norm=True) for x_dim in x_dims]
        self.shared_encoder = MLP(h_dim, z_dim, shared_hiddens, output_activation=shared_encoder_output_activation,
                                  dropout=dropout, batch_norm=True, regularize_last_layer=regularize_shared_encoder_last_layer)
        self.shared_decoder = MLP(z_dim, h_dim, shared_hiddens[::-1], output_activation='leakyrelu',
                                  dropout=dropout, batch_norm=True, regularize_last_layer=True)
        self.modality = nn.Embedding(self.n_modal, z_dim)

        # register sub-modules
        for i, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(f'encoder-{i}', enc)
            self.add_module(f'decoder-{i}', dec)

        self = self.to(device)
    
    def warmup_mode(self, on=True):
        self.cross_coef = self.cross_coef_init * (not on)
        self.integ_coef = self.integ_coef_init * (not on)
        self.cycle_coef = self.cycle_coef_init * (not on)

    def encode(self, x, i):
        h = self.x_to_h(x, i)
        z = self.h_to_z(h, i)
        return z 

    def decode(self, z, i):
        h = self.z_to_h(z, i)
        x = self.h_to_x(h, i)
        return x
    
    def to_latent(self, x, i):
        return self.encode(x, i)
    
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
    
    def forward(self, xs, pair_masks):
        # encoder and decoder
        zs = [self.encode(x, i) for i, x in enumerate(xs)]
        rs = [self.decode(z, i) for i, z in enumerate(zs)]

        self.loss, losses = self.calc_loss(xs, rs, zs, pair_masks)
        
        return rs, self.loss, losses

    def calc_loss(self, xs, rs, zs, pair_masks):
        # reconstruction loss for each modality, seaprately
        recon_loss = sum([nn.MSELoss()(r, x) for x, r in zip(xs, rs)])

        # losses between modalities
        cross_loss = 0
        integ_loss = 0
        cycle_loss = 0
        for i, (xi, zi, pmi) in enumerate(zip(xs, zs, pair_masks)):
            for j, (xj, zj, pmj) in enumerate(zip(xs, zs, pair_masks)):
                if i == j:
                    continue
                zij = self.convert(zi, i, j)
                rij = self.decode(zij, j)
                ziji = self.convert(self.to_latent(rij, j), j, i)

                cycle_loss += nn.MSELoss()(zi, ziji)

                if self.pair_groups[i] is not None and self.pair_groups[i] == self.pair_groups[j]:
                    xj_paired, xj_unpaired = xj[pmj == 1], xj[pmj == 0]
                    zj_paired, zj_unpaired = zj[pmj == 1], zj[pmj == 0]
                    zij_paired, zij_unpaired = zij[pmi == 1], zij[pmi == 0]
                    rij_paired, rij_unpaired = rij[pmi == 1], rij[pmi == 0]

                    # unpaired losses
                    if len(zij_unpaired) > 0 and len(zj_unpaired) > 0:
                        integ_loss += MMD()(zij_unpaired, zj_unpaired)
                    if len(rij_unpaired) > 0 and len(xj_unpaired) > 0:
                        cross_loss += MMD()(rij_unpaired, xj_unpaired)

                    # paired losses
                    if len(zij_paired) > 0 and len(zj_paired) > 0:
                        integ_loss += nn.MSELoss()(zij_paired, zj_paired)
                    if len(rij_paired) > 0 and len(xj_paired) > 0:
                        cross_loss += nn.MSELoss()(rij_paired, xj_paired)
                else:
                    cross_loss += MMD()(rij, xj)
                    integ_loss += MMD()(zij, zj)

        
        return self.recon_coef * recon_loss + \
               self.cross_coef * cross_loss + \
               self.integ_coef * integ_loss + \
               self.cycle_coef * cycle_loss, {
                   'recon': recon_loss,
                   'cross': cross_loss,
                   'integ': integ_loss,
                   'cycle': cycle_loss
                }
    
    def modal_vector(self, i):
        return F.one_hot(torch.tensor([i]).long(), self.n_modal).float().to(self.device)

    def backward(self):
        self.loss.backward()
    
    def test(self, *xs):
        outputs, loss, losses = self.forward(*xs)
        return loss, losses

    def convert(self, z, i, j=None):
        """This function converts vector z from modality i to modality j

        Args:
            z (torch.tensor): latent vector in modality i
            i (int): id of the source modality
            j (int): id of the destination modality

        Returns:
            torch.tensor: the converted latent vector
        """
        v = -self.modal_vector(i)
        if j is not None:
            v += self.modal_vector(j)
        return z + v @ self.modality.weight

    def integrate(self, x, i, j=None):
        zi = self.to_latent(x, i)
        zij = self.convert(zi, i, j)
        return zij