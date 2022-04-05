import torch
from torch import nn
from scvi.nn import FCLayers

from typing import Callable, Iterable, List, Optional

class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str = 'layer'
    ):
        super().__init__()
        use_layer_norm = False
        use_batch_norm = True
        if normalization == 'layer':
            use_layer_norm = True
            use_batch_norm = False
        elif normalization == 'none':
            use_batch_norm = False

        self.mlp = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            activation_fn=nn.LeakyReLU
        )

    def forward(
        self,
        x
    ):
        return self.mlp(x)

class Decoder(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            normalization: str = 'layer',
            loss = 'mse'
    ):
        super().__init__()

        if loss not in ['mse', 'nb', 'zinb', 'bce']:
            raise NotImplementedError(f'Loss function {loss} is not implemented.')
        else:
            self.loss = loss

        self.decoder = MLP(n_input, n_hidden, n_layers, n_hidden, dropout_rate, normalization) #embed_dim,

        if loss == 'mse':
            self.recon_decoder = nn.Linear(n_hidden, n_output)
        elif loss == 'nb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        elif loss == 'zinb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(n_hidden, n_output)

        elif loss == 'bce':
            self.recon_decoder = FCLayers(
                n_in=n_hidden,
                n_out=n_output,
                n_layers=0,
                dropout_rate=0,
                use_layer_norm=False,
                use_batch_norm=False,
                activation_fn=nn.Sigmoid
            )

    def forward(self, x):
        x = self.decoder(x)
        if self.loss == 'mse' or self.loss == 'bce':
            return self.recon_decoder(x)
        elif self.loss == 'nb':
            return self.mean_decoder(x)
        elif self.loss == 'zinb':
            return self.mean_decoder(x), self.dropout_decoder(x)

class GeneralizedSigmoid(nn.Module):
    """ Adapted from
        Title: CPA (c) Facebook, Inc.
        Date: 26.01.2022
        Link to the used code:
        https://github.com/facebookresearch/CPA/blob/382ff641c588820a453d801e5d0e5bb56642f282/compert/model.py#L109
    """
    """
    Sigmoid, log-sigmoid or linear functions for encoding continuous covariates.
    """

    def __init__(self, dim, nonlin='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super().__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(
            torch.ones(1, dim),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, dim),
            requires_grad=True
        )

    def forward(self, x):
        if self.nonlin == 'logsigm':
            return (torch.log1p(x) * self.beta + self.bias).sigmoid()
        elif self.nonlin == 'sigm':
            return (x * self.beta + self.bias).sigmoid()
        else:
            return x

    def one_cov(self, x, i):
        if self.nonlin == 'logsigm':
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid()
        elif self.nonlin == 'sigm':
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid()
        else:
            return x
