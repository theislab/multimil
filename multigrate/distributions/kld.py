import torch

class KLD(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(f'Reduction method {self.reduction} is not implemented.')
        self.reduction = reduction

    def forward(self, mu, logvar):
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
        if self.reduction == 'mean':
            kl = torch.mean(kl)
        elif self.reduction == 'sum':
            kl = torch.sum(kl)

        return kl
