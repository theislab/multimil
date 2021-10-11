from torch import nn
from .mlp import MLP

class MLP_decoder(MLP):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 hiddens=[],
                 output_activation='linear',
                 dropout=None,
                 norm='layer',
                 regularize_last_layer=False,
                 loss='mse',
                 device='cpu'):

        if loss not in ['mse', 'nb', 'zinb', 'bce']:
            raise NotImplementedError(f'Loss function {loss} is not implemented.')
        else:
            self.loss = loss

        super().__init__(n_inputs, n_outputs, hiddens, output_activation, dropout, norm, regularize_last_layer, last_layer=False)

        n_inputs_last_layer = n_inputs if len(hiddens) == 0 else hiddens[-1]
        norm_last_layer = norm if regularize_last_layer else 'no_reg'
        
        if loss == 'mse':
            self.recon_decoder = self._fc(n_inputs_last_layer, n_outputs, activation=output_activation,
                                                   dropout=dropout if regularize_last_layer else None,
                                                   norm=norm_last_layer)

        elif loss == 'nb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_inputs_last_layer, n_outputs), nn.Softmax(dim=-1))

        elif loss == 'zinb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_inputs_last_layer, n_outputs), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(n_inputs_last_layer, n_outputs)

        elif loss == 'bce':
            self.recon_decoder = self._fc(n_inputs_last_layer, n_outputs, activation='sigmoid',
                                                   dropout=dropout if regularize_last_layer else None,
                                                   norm=norm_last_layer)

        self = self.to(device)

    def _activation(self, name='leakyrelu'):
        if name in ['relu', 'leakyrelu', 'sigmoid']:
            return super()._activation(name)
        elif name == 'softmax':
            return nn.Softmax()
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    def forward(self, x):
        x = self.network(x)
        if self.loss == 'mse' or self.loss == 'bce':
            return self.recon_decoder(x)
        elif self.loss == 'nb':
            return self.mean_decoder(x)
        elif self.loss == 'zinb':
            return self.mean_decoder(x), self.dropout_decoder(x)
