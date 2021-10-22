import torch

class NB(torch.nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(f'Reduction method {self.reduction} is not implemented.')
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor):
        """
            This negative binomial function was taken from:
            Title: scvi-tools
            Authors: Romain Lopez <romain_lopez@gmail.com>,
                    Adam Gayoso <adamgayoso@berkeley.edu>,
                    Galen Xing <gx2113@columbia.edu>
            Date: 16th November 2020
            Code version: 0.8.1
            Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

            Computes negative binomial loss.
            Parameters
            ----------
            x: torch.Tensor
                Torch Tensor of ground truth data.
            mu: torch.Tensor
                Torch Tensor of means of the negative binomial (has to be positive support).
            theta: torch.Tensor
                Torch Tensor of inverse dispersion parameter (has to be positive support).
            eps: Float
                numerical stability constant.

            Returns
            -------
            If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
        """
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))

        log_theta_mu_eps = torch.log(theta + mu + self.eps)
        res = (
            theta * (torch.log(theta + self.eps) - log_theta_mu_eps)
            + x * (torch.log(mu + self.eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )

        if self.reduction == 'mean':
            res = torch.mean(res, dim=-1)
        elif self.reduction == 'sum':
            res = torch.sum(res, dim=-1)

        return res
