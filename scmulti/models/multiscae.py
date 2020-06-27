import torch
from torch import nn
from .mlp import MLP
from .losses import MMD


class MultiScAE(nn.Module):
    def __init__(self, x_dims,
                 z_dim=10,
                 h_dim=32,
                 hiddens=[],
                 shared_hiddens=[],
                 paired=False,
                 recon_coef=1,
                 cross_coef=1,
                 dropout=0.2,
                 device='cpu'):

        super(MultiScAE, self).__init__()

        # save model parameters
        self.n_modal = len(x_dims)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.paired = paired
        self.recon_coef = recon_coef
        self.cross_coef = cross_coef
        self.device = device

        # create sub-modules
        self.encoders = [MLP(x_dim, h_dim, hiddens, output_activation='leakyrelu',
                             dropout=dropout, batch_norm=True, regularize_last_layer=True) for x_dim in x_dims]
        self.decoders = [MLP(h_dim, x_dim, hiddens[::-1], dropout=dropout, batch_norm=True) for x_dim in x_dims]
        self.shared_encoder = MLP(h_dim, z_dim, shared_hiddens, output_activation='leakyrelu',
                                  dropout=dropout, batch_norm=True, regularize_last_layer=True)
        self.shared_decoder = MLP(z_dim, h_dim, shared_hiddens[::-1], output_activation='leakyrelu',
                                  dropout=dropout, batch_norm=True, regularize_last_layer=True)
        self.modality = nn.Embedding(self.n_modal, z_dim)

        # register sub-modules
        for i, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(f'encoder-{i}', enc)
            self.add_module(f'decoder-{i}', dec)

        self = self.to(device)

    def encode(self, x, i):
        h = self.encoders[i](x)
        z = self.shared_encoder(h)
        return z 
    
    def to_latent(self, x, i):
        return self.encode(x, i)

    def decode(self, z, i):
        h = self.shared_decoder(z)
        x = self.decoders[i](h)
        return x

    def forward(self, *xs):
        zs = [self.encode(x, i) for i, x in enumerate(xs)]
        rs = [self.decode(z, i) for i, z in enumerate(zs)]

        self.loss, losses = self.calc_loss(xs, rs, zs)
        
        return rs, self.loss, losses

    def calc_loss(self, xs, rs, zs):
        recon_loss = sum([nn.MSELoss()(r, x) for x, r in zip(xs, rs)])

        cross_loss = 0
        for i, zi in enumerate(zs):
            vi = self.modal_vector(i)
            for j, xj in enumerate(xs):
                if i == j:
                    continue
                vj = self.modal_vector(j)
                rij = self.decode(self.convert(zi, vj - vi), j)
                if self.paired:
                    cross_loss += nn.MSELoss()(rij, xj)
                else:
                    cross_loss += MMD()(rij, xj)

        return self.recon_coef * recon_loss + self.cross_coef * cross_loss, {
            'recon': recon_loss,
            'cross': cross_loss
        }
    
    def modal_vector(self, i):
        return nn.functional.one_hot(torch.tensor([i]).long(), self.n_modal).float().to(self.device)

    def backward(self):
        self.loss.backward()
    
    def test(self, *xs):
        outputs, loss, losses = self.forward(*xs)
        return loss, losses

    def convert(self, z, v):
        return z + v @ self.modality.weight

    def integrate(self, *xs, center=None):
        zs = []
        for i, x in enumerate(xs):
            z = self.to_latent(x, i)
            v = self.modal_vector(i)
            z = self.convert(z, -v)
            if center is not None:
                vc = self.modal_vector(center)
                z = self.convert(z, vc)
            zs.append(z)
        return zs