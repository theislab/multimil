import torch
from torch import nn
from scvi.nn import FCLayers

from typing import Callable, Iterable, List, Optional

class CondMLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        embed_dim: int = 0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str = 'layer'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        use_layer_norm = False
        use_batch_norm = True
        if normalization == 'layer':
            use_layer_norm = True
            use_batch_norm = False
        elif normalization == 'none':
            use_batch_norm = False

        self.encoder = FCLayers(
            n_in=n_input+embed_dim,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            activation_fn=nn.LeakyReLU
        )

        if embed_dim > 0:
            self.condition_layer = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x
    ):
        if self.embed_dim > 0:
            expr, cond = torch.split(
                x,
                [x.shape[1] - self.embed_dim, self.embed_dim],
                dim=1
            )
            x = torch.cat([expr, self.condition_layer(cond)], dim=1)
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_output: int,
            embed_dim: int = 0,
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

        self.decoder = CondMLP(n_input, n_hidden, embed_dim, n_layers-1, n_hidden, dropout_rate, normalization)

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
