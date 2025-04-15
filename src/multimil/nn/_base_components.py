from typing import Literal

import torch
from scvi.nn import FCLayers
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """A helper class to build blocks of fully-connected, normalization, dropout and activation layers.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Number of output features.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units.
    dropout_rate
        Dropout rate.
    normalization
        Type of normalization to use. Can be one of ["layer", "batch", "none"].
    activation
        Activation function to use.

    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str = "layer",
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        use_layer_norm = False
        use_batch_norm = True
        if normalization == "layer":
            use_layer_norm = True
            use_batch_norm = False
        elif normalization == "none":
            use_batch_norm = False

        self.mlp = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            activation_fn=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(n_input,)``.

        Returns
        -------
        Tensor of values with shape ``(n_output,)``.
        """
        return self.mlp(x)


class Decoder(nn.Module):
    """A helper class to build custom decoders depending on which loss was passed.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Number of output features.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units.
    dropout_rate
        Dropout rate.
    normalization
        Type of normalization to use. Can be one of ["layer", "batch", "none"].
    activation
        Activation function to use.
    loss
        Loss function to use. Can be one of ["mse", "nb", "zinb", "bce"].
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        normalization: str = "layer",
        activation=nn.LeakyReLU,
        loss="mse",
    ):
        super().__init__()

        if loss not in ["mse", "nb", "zinb", "bce"]:
            raise NotImplementedError(f"Loss function {loss} is not implemented.")
        else:
            self.loss = loss

        self.decoder = MLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            normalization=normalization,
            activation=activation,
        )
        if loss == "mse":
            self.recon_decoder = nn.Linear(n_hidden, n_output)
        elif loss == "nb":
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))

        elif loss == "zinb":
            self.mean_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(n_hidden, n_output)

        elif loss == "bce":
            self.recon_decoder = FCLayers(
                n_in=n_hidden,
                n_out=n_output,
                n_layers=0,
                dropout_rate=0,
                use_layer_norm=False,
                use_batch_norm=False,
                activation_fn=nn.Sigmoid,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(n_input,)``.

        Returns
        -------
        Tensor of values with shape ``(n_output,)``.
        """
        x = self.decoder(x)
        if self.loss == "mse" or self.loss == "bce":
            return self.recon_decoder(x)
        elif self.loss == "nb":
            return self.mean_decoder(x)
        elif self.loss == "zinb":
            return self.mean_decoder(x), self.dropout_decoder(x)


class GeneralizedSigmoid(nn.Module):
    """Sigmoid, log-sigmoid or linear functions for encoding continuous covariates.

    Adapted from
    Title: CPA (c) Facebook, Inc.
    Date: 26.01.2022
    Link to the used code:
    https://github.com/facebookresearch/CPA/blob/382ff641c588820a453d801e5d0e5bb56642f282/compert/model.py#L109

    Parameters
    ----------
    dim
        Number of input features.
    nonlin
        Type of non-linearity to use. Can be one of ["logsigm", "sigm"]. Default is "logsigm".
    """

    def __init__(self, dim, nonlin: Literal["logsigm", "sigm"] | None = "logsigm"):
        super().__init__()
        self.nonlin = nonlin
        self.beta = torch.nn.Parameter(torch.ones(1, dim), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, dim), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values.

        Returns
        -------
        Tensor of values with the same shape as ``x``.
        """
        if self.nonlin == "logsigm":
            return (torch.log1p(x) * self.beta + self.bias).sigmoid()
        elif self.nonlin == "sigm":
            return (x * self.beta + self.bias).sigmoid()
        else:
            return x


class Aggregator(nn.Module):
    """A helper class to build custom aggregators depending on the scoring function passed.

    Parameters
    ----------
    n_input
        Number of input features.
    scoring
        Scoring function to use. Can be one of ["attn", "gated_attn", "mean", "max", "sum", "mlp"].
    attn_dim
        Dimension of the hidden attention layer.
    sample_batch_size
        Bag batch size.
    scale
        Whether to scale the attention weights.
    dropout
        Dropout rate.
    n_layers_mlp_attn
        Number of hidden layers in the MLP attention.
    n_hidden_mlp_attn
        Number of hidden units in the MLP attention.
    activation
        Activation function to use.
    """

    def __init__(
        self,
        n_input=None,
        scoring="gated_attn",
        attn_dim=16,  # D
        sample_batch_size=None,
        scale=False,
        dropout=0.2,
        n_layers_mlp_attn=1,
        n_hidden_mlp_attn=16,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.scoring = scoring
        self.patient_batch_size = sample_batch_size
        self.scale = scale

        if self.scoring == "attn":
            self.attn_dim = attn_dim  # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            self.attention = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == "gated_attn":
            self.attn_dim = attn_dim
            self.attention_V = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
            )

            self.attention_U = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Sigmoid(),
            )

            self.attention_weights = nn.Linear(self.attn_dim, 1, bias=False)

        elif self.scoring == "mlp":
            if n_layers_mlp_attn == 1:
                self.attention = nn.Linear(n_input, 1)
            else:
                self.attention = nn.Sequential(
                    MLP(
                        n_input,
                        n_hidden_mlp_attn,
                        n_layers=n_layers_mlp_attn - 1,
                        n_hidden=n_hidden_mlp_attn,
                        dropout_rate=dropout,
                        activation=activation,
                    ),
                    nn.Linear(n_hidden_mlp_attn, 1),
                )

    def forward(self, x) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(n_input,)``.

        Returns
        -------
        Tensor of pooled values.
        """
        # # TODO add mean and max pooling
        # if self.scoring == "attn":
        #     # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
        #     self.A = self.attention(x)  # Nx1
        #     self.A = torch.transpose(self.A, -1, -2)  # 1xN
        #     self.A = F.softmax(self.A, dim=-1)  # softmax over N

        # elif self.scoring == "gated_attn":
        #     # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
        #     A_V = self.attention_V(x)  # NxD
        #     A_U = self.attention_U(x)  # NxD
        #     self.A = self.attention_weights(A_V * A_U)  # element wise multiplication # Nx1
        #     self.A = torch.transpose(self.A, -1, -2)  # 1xN
        #     self.A = F.softmax(self.A, dim=-1)  # softmax over N

        # elif self.scoring == "mlp":
        #     self.A = self.attention(x)  # N
        #     self.A = torch.transpose(self.A, -1, -2)
        #     self.A = F.softmax(self.A, dim=-1)

        # else:
        #     raise NotImplementedError(
        #         f'scoring = {self.scoring} is not implemented. Has to be one of ["attn", "gated_attn", "mlp"].'
        #     )

        # if self.scale:
        #     self.A = self.A * self.A.shape[-1] / self.patient_batch_size

        # return torch.bmm(self.A, x).squeeze(dim=1)
        # Apply different pooling strategies based on the scoring method
        if self.scoring in ["attn", "gated_attn", "mlp", "sum", "mean", "max"]:
            if self.scoring == "attn":
                # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
                A = self.attention(x)  # (batch_size, N, 1)
                A = A.transpose(1, 2)  # (batch_size, 1, N)
                A = F.softmax(A, dim=-1)
            elif self.scoring == "gated_attn":
                # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
                A_V = self.attention_V(x)  # (batch_size, N, attn_dim)
                A_U = self.attention_U(x)  # (batch_size, N, attn_dim)
                A = self.attention_weights(A_V * A_U)  # (batch_size, N, 1)
                A = A.transpose(1, 2)  # (batch_size, 1, N)
                A = F.softmax(A, dim=-1)
            elif self.scoring == "mlp":
                A = self.attention(x)  # (batch_size, N, 1)
                A = A.transpose(1, 2)  # (batch_size, 1, N)
                A = F.softmax(A, dim=-1)

            elif self.scoring == "sum":
                return torch.sum(x, dim=1)  # (batch_size, n_input)
            elif self.scoring == "mean":
                return torch.mean(x, dim=1)  # (batch_size, n_input)
            elif self.scoring == "max":
                return torch.max(x, dim=1).values  # (batch_size, n_input)
            else:
                raise NotImplementedError(
                    f'scoring = {self.scoring} is not implemented. Has to be one of ["attn", "gated_attn", "mlp", "sum", "mean", "max"].'
                )
            if self.scale:
                if self.patient_batch_size is None:
                    raise ValueError("patient_batch_size must be set when scale is True.")
                self.A = A * A.shape[-1] / self.patient_batch_size

            pooled = torch.bmm(A, x).squeeze(dim=1)  # (batch_size, n_input)
            return pooled
