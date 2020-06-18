import torch
from torch import nn
from .mlp import MLP
from .losses import KLD, MMD


class MultiScVAE(nn.Module):
    def __init__(self, x_dims, z_dim=10, hiddens=[], paired=False, recon_coef=1, kl_coef=1, cross_coef=1, device='cpu'):
        super(MultiScVAE, self).__init__()

        # save model parameters
        self.n_modal = len(x_dims)
        self.paired = paired
        self.z_dim = z_dim
        self.recon_coef = recon_coef
        self.kl_coef = kl_coef
        self.cross_coef = cross_coef * paired
        self.device = device

        # create encoders and decoders
        self.encoders = [MLP(x_dim, z_dim, hiddens) for x_dim in x_dims]
        self.decoders = [MLP(z_dim, x_dim, hiddens[::-1]) for x_dim in x_dims]
        self.mus = [nn.Linear(z_dim, z_dim) for i in range(self.n_modal)]
        self.logvars = [nn.Linear(z_dim, z_dim) for i in range(self.n_modal)]
        self.modality = nn.Embedding(self.n_modal, z_dim)

        # register sub-modules
        for i, (enc, dec, mu, logvar) in enumerate(zip(self.encoders, self.decoders, self.mus, self.logvars)):
            self.add_module(f'encoder-{i}', enc)
            self.add_module(f'decoder-{i}', dec)
            self.add_module(f'mu-{i}', mu)
            self.add_module(f'logvar-{i}', logvar)
        self = self.to(device)

    def encode(self, x, i):
        return self.encoders[i](x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def bottleneck(self, h, i):
        mu = self.mus[i](h)
        logvar = self.logvars[i](h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def to_latent(self, x, i):
        h = self.encode(x, i)
        z, _, _ = self.bottleneck(h, i)
        return z

    def decode(self, z, i):
        return self.decoders[i](z)

    def forward(self, *xs):
        hs = [self.encode(x, i) for i, x in enumerate(xs)]
        bs = [self.bottleneck(h, i) for i, h in enumerate(hs)]
        zs = [b[0] for b in bs]
        mus = [b[1] for b in bs]
        logvars = [b[2] for b in bs]
        rs = [self.decode(z, i) for i, z in enumerate(zs)]

        self.loss = self.calc_loss(xs, rs, zs, mus, logvars)
        
        return (rs, mus, logvars), self.loss

    def calc_loss(self, xs, rs, zs, mus, logvars):
        recon_loss = sum([self.recon_coef * nn.MSELoss()(r, x) for x, r in zip(xs, rs)])
        kl_loss = sum([self.kl_coef * KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])

        cross_loss = 0
        for i, (xi, zi) in enumerate(zip(xs, zs)):
            vi = self.modal_vector(i)
            for j, (xj, zj) in enumerate(zip(xs, zs)):
                if i == j:
                    continue
                vj = self.modal_vector(j)
                rij = self.decode(self.convert(zi, vj - vi), j)
                if self.paired:
                    cross_loss += self.cross_coef * nn.MSELoss()(rij, xj)
                else:
                    cross_loss += self.cross_coef * MMD()(rij, xj)

        return recon_loss + kl_loss + cross_loss
    
    def modal_vector(self, i):
        return nn.functional.one_hot(torch.tensor([i]).long(), self.n_modal).float().to(self.device)

    def backward(self):
        self.loss.backward()
    
    def test(self, *xs):
        outputs, loss = self.forward(*xs)
        return loss

    def convert(self, z, v):
        return z + v @ self.modality.weight

    def integrate(self, *xs):
        zs = []
        for i, x in enumerate(xs):
            z = self.to_latent(x, i)
            v = self.modal_vector(i)
            zs.append(self.convert(z, -v))
        return zs