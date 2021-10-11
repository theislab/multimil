import torch
import torch.nn.functional as F

class ZINB(torch.nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor):
        """
           This zero-inflated negative binomial function was taken from:
           Title: scvi-tools
           Authors: Romain Lopez <romain_lopez@gmail.com>,
                    Adam Gayoso <adamgayoso@berkeley.edu>,
                    Galen Xing <gx2113@columbia.edu>
           Date: 16th November 2020
           Code version: 0.8.1
           Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py
           Computes zero inflated negative binomial loss.
           Parameters
           ----------
           x: torch.Tensor
                Torch Tensor of ground truth data.
           mu: torch.Tensor
                Torch Tensor of means of the negative binomial (has to be positive support).
           theta: torch.Tensor
                Torch Tensor of inverses dispersion parameter (has to be positive support).
           pi: torch.Tensor
                Torch Tensor of logits of the dropout parameter (real support)
           eps: Float
                numerical stability constant.
           Returns
           -------
           If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
        """
        # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
        if theta.ndimension() == 1:
            theta = theta.view(
                1, theta.size(0)
            )  # In this case, we reshape theta for broadcasting

        softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
        log_theta_eps = torch.log(theta + self.eps)
        log_theta_mu_eps = torch.log(theta + mu + self.eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < self.eps).type(torch.float32), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + self.eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > self.eps).type(torch.float32), case_non_zero)

        res = mul_case_zero + mul_case_non_zero

        if self.reduction == 'mean':
            res = torch.mean(res)
        elif self.reduction == 'sum':
            res = torch.sum(res)
        else:
            raise NotImplementedError(f'Reduction method {self.reduction} is not implemented.')

        return res
