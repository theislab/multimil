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


class Aggregator(nn.Module):
    """A helper class to build custom aggregators depending on the scoring function passed.

    Parameters
    ----------
    n_input
        Number of input features.
    scoring
        Scoring function to use. Can be one of ["attn", "gated_attn", "mean", "max", "sum"].
    attn_dim
        Dimension of the hidden attention layer.
    sample_batch_size
        Bag batch size.
    scale
        Whether to scale the attention weights.
    dropout
        Dropout rate.
    activation
        Activation function to use.
    """

    def __init__(
        self,
        n_input: int | None = None,
        scoring="gated_attn",
        attn_dim=16,  # D
        sample_batch_size=None,
        scale=False,
        dropout=0.2,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.scoring = scoring
        self.patient_batch_size = sample_batch_size
        self.scale = scale

        if self.scoring == "attn":
            if n_input is None:
                raise ValueError("n_input must be provided for attn scoring")
            self.attn_dim = attn_dim  # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            self.attention = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == "gated_attn":
            if n_input is None:
                raise ValueError("n_input must be provided for gated_attn scoring")
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

    def forward(self, x) -> torch.Tensor:
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            Tensor of values with shape ``(batch_size, N, n_input)``.

        Returns
        -------
        Tensor of pooled values with shape ``(batch_size, n_input)``.
        """
        if self.scoring == "attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A = self.attention(x)  # (batch_size, N, 1)
            A = A.transpose(1, 2)  # (batch_size, 1, N)
            self.A = F.softmax(A, dim=-1)
        elif self.scoring == "gated_attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # (batch_size, N, attn_dim)
            A_U = self.attention_U(x)  # (batch_size, N, attn_dim)
            A = self.attention_weights(A_V * A_U)  # (batch_size, N, 1)
            A = A.transpose(1, 2)  # (batch_size, 1, N)
            self.A = F.softmax(A, dim=-1)
        elif self.scoring == "sum":
            return torch.sum(x, dim=1)  # (batch_size, n_input)
        elif self.scoring == "mean":
            return torch.mean(x, dim=1)  # (batch_size, n_input)
        elif self.scoring == "max":
            return torch.max(x, dim=1).values  # (batch_size, n_input)
        else:
            raise NotImplementedError(
                f'scoring = {self.scoring} is not implemented. Has to be one of ["attn", "gated_attn", "sum", "mean", "max"].'
            )

        if self.scale:
            if self.patient_batch_size is None:
                raise ValueError("patient_batch_size must be set when scale is True.")
            self.A = self.A * self.A.shape[-1] / self.patient_batch_size

        pooled = torch.bmm(self.A, x).squeeze(dim=1)  # (batch_size, n_input)
        return pooled
