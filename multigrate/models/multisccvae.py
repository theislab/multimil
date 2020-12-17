import torch
from torch import nn
from .mlp import MLP
from .multisccae import MultiScCAE
from .losses import KLD, MMD


class MultiScCVAE(MultiScCAE):
    def __init__(self,
                 z_dim=20,
                 kl_coef=1,
                 device='cpu',
                 **kwargs):

        super(MultiScCVAE, self).__init__(z_dim=z_dim, device=device, shared_encoder_output_activation='leakyrelu',
                                         regularize_shared_encoder_last_layer=True, **kwargs)

        # save model parameters
        self.kl_coef_init = self.kl_coef = kl_coef

        # create sub-modules
        self.mu = MLP(self.z_dim, self.z_dim)
        self.logvar = MLP(self.z_dim, self.z_dim)

        self = self.to(device)
    
    def get_nonadversarial_params(self):
        params = super(MultiScCVAE, self).get_nonadversarial_params()
        params.extend(list(self.mu.parameters()))
        params.extend(list(self.logvar.parameters()))
        return params

    def kl_anneal(self, epoch, anneal_limit):
        self.kl_coef = min(self.kl_coef_init, epoch / anneal_limit * self.kl_coef_init)
        self.kl_coef = max(0, self.kl_coef)
    
    def warmup_mode(self, on=True):
        super(MultiScCVAE, self).warmup_mode(on)
        self.kl_coef = self.kl_coef_init * (not on)

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
        z = super(MultiScCVAE, self).encode(x, i)
        return self.bottleneck(z)

    def to_latent(self, x, i):
        z, _, _ = self.encode(x, i)
        return z
    
    def forward(self, xs, pair_masks):
        # encoder and decode
        zs = [self.encode(x, i) for i, x in enumerate(xs)]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        rs = [self.decode(z, i) for i, z in enumerate(zs)]

        self.loss, losses = self.calc_loss(xs, rs, zs, pair_masks, mus, logvars)
        self.adv_loss, adv_losses = self.calc_adv_loss(zs)

        return (rs, mus, logvars), self.loss - self.adv_loss, {**losses, **adv_losses}

    def calc_loss(self, xs, rs, zs, pair_masks, mus, logvars):
        loss, losses = super(MultiScCVAE, self).calc_loss(xs, rs, zs, pair_masks)
        kl_loss = sum([KLD()(mu, logvar) for mu, logvar in zip(mus, logvars)])
        losses['kl'] = kl_loss
        return loss + self.kl_coef * kl_loss, losses