from typing import List, Optional

import torch


class MMD(torch.nn.Module):
    """Maximum mean discrepancy.

    :param kernel_type:
        Indicates if to use Gaussian kernel. One of
        * ``'gaussian'`` - use Gaussian kernel
        * ``'not gaussian'`` - do not use Gaussian kernel
    """

    def __init__(self, kernel_type="gaussian"):
        super().__init__()
        self.kernel_type = kernel_type
        # TODO: add check for gaussian kernel that shapes are same

    def gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Apply Guassian kernel.

        :param x:
            Tensor from the first distribution
        :param y:
            Tensor from the second distribution
        :param gamma:
            List of gamma parameters
        """
        if gamma is None:
            gamma = [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                100,
                1e3,
                1e4,
                1e5,
                1e6,
            ]

        D = torch.cdist(x, y).pow(2)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K / len(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Adapted from
        Title: scarches
        Date: 9th Octover 2021
        Code version: 0.4.0
        Availability: https://github.com/theislab/scarches/blob/63a7c2b35a01e55fe7e1dd871add459a86cd27fb/scarches/models/trvae/losses.py
        Citation: Gretton, Arthur, et al. "A Kernel Two-Sample Test", 2012.

        :param x:
            Tensor with shape ``(batch_size, z_dim)``
        :param y:
            Tensor with shape ``(batch_size, z_dim)``
        :returns:
            MMD between ``x`` and ``y``
        """
        # in case there is only one sample in a batch belonging to one of the groups, then skip the batch
        if len(x) == 1 or len(y) == 1:
            return torch.tensor(0.0)

        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
