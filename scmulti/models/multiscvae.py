import torch
from torch import nn
from .mlp import MLP
from .multiscae import MultiScAE
from .losses import KLD, MMD


class MultiScVAE(MultiScAE):
    def __init__(self,
                 z_dim=20,
                 kl_coef=1,
                 device='cpu',
                 **kwargs):

        super(MultiScVAE, self).__init__(z_dim=z_dim, device=device, shared_encoder_output_activation='leakyrelu',
                                         regularize_shared_encoder_last_layer=True, **kwargs)

        # save model parameters
        self.kl_coef_init = self.kl_coef = kl_coef

        # create sub-modules
        self.mu = MLP(self.z_dim, self.z_dim)
        self.logvar = MLP(self.z_dim, self.z_dim)

        self = self.to(device)

    def kl_anneal(self, epoch, anneal_limit):
        self.kl_coef = min(self.kl_coef_init, epoch / anneal_limit * self.kl_coef_init)
        self.kl_coef = max(0, self.kl_coef)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def bottleneck(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x, i):
        z = super(MultiScVAE, self).encode(x, i)
        return self.bottleneck(z)

    def to_latent(self, x, i):
        z, _, _ = self.encode(x, i)
        return z
    
    def forward(self, *xs):
        rs, zs, mus, logvars = [], [], [], []
        for i, x in enumerate(xs):
            z, mu, logvar = self.encode(x, i)
            r = self.decode(z, i)
            rs.append(r)
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        self.loss, losses = self.calc_loss(xs, rs, zs, mus, logvars)

        return (rs, mus, logvars), self.loss, losses

    def calc_loss(self, xs, rs, zs, mus, logvars):
        recon_cross_loss, losses = super(MultiScVAE, self).calc_loss(xs, rs, zs)
        kl_loss = sum([KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])
        losses['kl'] = kl_loss
        return recon_cross_loss + self.kl_coef * kl_loss, losses