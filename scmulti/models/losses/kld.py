import torch


class KLD(torch.nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
