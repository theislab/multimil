from torch import nn

class MLP_decoder(nn.Module):
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
        super(MLP_decoder, self).__init__()

        if loss not in ['mse', 'nb', 'zinb', 'bce']:
            raise NotImplementedError(f'Loss function {loss} is not implemented.')
        else:
            self.loss = loss
        # add hidden layers architecture
        layers = []
        norm_last_layer = norm if regularize_last_layer else 'no_reg'

        if len(hiddens) > 0:

            layers.append(self._fc(n_inputs, hiddens[0], activation='leakyrelu', dropout=dropout, norm=norm))  # first layer
            for l in range(1, len(hiddens)):  # inner layers
                layers.append(self._fc(hiddens[l-1], hiddens[l], activation='leakyrelu', dropout=dropout, norm=norm))

        # add last layer
        n_inputs_last_layer = n_inputs if len(hiddens) == 0 else hiddens[-1]

        if loss == 'mse':
            self.recon_decoder = self._fc(n_inputs_last_layer, n_outputs, activation=output_activation,
                                                   dropout=dropout if regularize_last_layer else None,
                                                   norm=norm_last_layer)

        elif loss == 'nb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_inputs_last_layer, n_outputs), nn.Softmax(dim=-1))

        elif loss == 'zinb':
            self.mean_decoder = nn.Sequential(nn.Linear(n_inputs_last_layer, n_outputs), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(n_inputs_last_layer, n_outputs)

        # TODO: check babel
        elif loss == 'bce':
            self.recon_decoder = self._fc(n_inputs_last_layer, n_outputs, activation='sigmoid',
                                                   dropout=dropout if regularize_last_layer else None,
                                                   norm=norm_last_layer)

        self.network = nn.Sequential(*layers)
        self = self.to(device)

    def _fc(self, n_inputs, n_outputs, activation='leakyrelu', dropout=None, norm='layer'):
        bias = norm != 'batch'
        layers = [nn.Linear(n_inputs, n_outputs, bias=bias)]
        if norm == 'batch':
            layers.append(nn.BatchNorm1d(n_outputs))
        elif norm == 'layer':
            layers.append(nn.LayerNorm(n_outputs, elementwise_affine=False))
        if activation != 'linear':
            layers.append(self._activation(activation))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _activation(self, name='leakyrelu'):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'softmax':
            return nn.Softmax()
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    def forward(self, x):
        if self.loss == 'mse' or self.loss == 'bce':
            return self.recon_decoder(self.network(x))
        elif self.loss == 'nb':
            return self.mean_decoder(self.network(x))
        elif self.loss == 'zinb':
            return self.mean_decoder(self.network(x)), self.dropout_decoder(self.network(x))

    # TODO make work with new losses
    def through(self, x):
        outputs = []
        for layer in self.network:
            x = layer(x)
            outputs.append(x)
        return outputs
