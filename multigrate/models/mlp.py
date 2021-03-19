from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 hiddens=[],
                 output_activation='linear',
                 dropout=None,
                 norm='layer',
                 regularize_last_layer=False,
                 device='cpu'):

        super(MLP, self).__init__()

        # create network architecture
        layers = []
        norm_last_layer = norm if regularize_last_layer else 'no_reg'

        if hiddens == []:  # no hidden layers
            layers.append(self._fc(n_inputs, n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   norm=norm_last_layer))
        else:
            layers.append(self._fc(n_inputs, hiddens[0], activation='leakyrelu', dropout=dropout, norm=norm))  # first layer
            for l in range(1, len(hiddens)):  # inner layers
                layers.append(self._fc(hiddens[l-1], hiddens[l], activation='leakyrelu', dropout=dropout, norm=norm))

            layers.append(self._fc(hiddens[-1], n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   norm=norm_last_layer))

        self.network = nn.Sequential(*layers)
        self = self.to(device)

    def _fc(self, n_inputs, n_outputs, activation='leakyrelu', dropout=None, norm='layer'):
        layers = [nn.Linear(n_inputs, n_outputs, bias=True if norm=='layer' else False)]
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
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    def forward(self, x):
        return self.network(x)

    def through(self, x):
        outputs = []
        for layer in self.network:
            x = layer(x)
            outputs.append(x)
        return outputs
