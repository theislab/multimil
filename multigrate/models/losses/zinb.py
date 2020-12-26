import torch
import torch.nn.functional as F

class ZINB(torch.nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super(ZINB, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor):

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
            raise NotImplementedError(f'reduction method {self.reduction} is not implemented.')

        return res
