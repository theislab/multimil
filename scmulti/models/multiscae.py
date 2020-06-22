import torch
from torch import nn
from .mlp import MLP
from .losses import MMD


class MultiScAE(nn.Module):
    def __init__(self, x_dims, z_dim=10, hiddens=[], paired=False, recon_coef=1, cross_coef=1, device='cpu'):
        super(MultiScAE, self).__init__()

        # save model parameters
        self.n_modal = len(x_dims)
        self.paired = paired
        self.z_dim = z_dim
        self.recon_coef = recon_coef
        self.cross_coef = cross_coef * paired
        self.device = device

        # create encoders and decoders
        self.encoders = [MLP(x_dim, z_dim, hiddens, dropout=0.5, batch_norm=True) for x_dim in x_dims]
        self.decoders = [MLP(z_dim, x_dim, hiddens[::-1], dropout=0.5, batch_norm=True) for x_dim in x_dims]
        self.modality = nn.Embedding(self.n_modal, z_dim)

        # register sub-modules
        for i, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            self.add_module(f'encoder-{i}', enc)
            self.add_module(f'decoder-{i}', dec)
        self = self.to(device)

    def encode(self, x, i):
        return self.encoders[i](x)

    def to_latent(self, x, i):
        z = self.encode(x, i)
        return z

    def decode(self, z, i):
        return self.decoders[i](z)

    def forward(self, *xs):
        zs = [self.encode(x, i) for i, x in enumerate(xs)]
        rs = [self.decode(z, i) for i, z in enumerate(zs)]

        self.loss = self.calc_loss(xs, rs, zs)
        
        return rs, self.loss

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

        return self.recon_coef * recon_loss + self.cross_coef * cross_loss
    
    def modal_vector(self, i):
        return nn.functional.one_hot(torch.tensor([i]).long(), self.n_modal).float().to(self.device)

    def backward(self):
        self.loss.backward()
    
    def test(self, *xs):
        outputs, loss = self.forward(*xs)
        return loss

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