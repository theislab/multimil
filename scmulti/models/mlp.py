from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=[], output_relu=False, dropout=None, batch_norm=True, regularize_last_layer=False,
                 device='cpu'):
        super(MLP, self).__init__()

        # create network architecture
        layers = []
        if hiddens == []:  # no hidden layers
            layers += self._fc(n_inputs, n_outputs, activation='leakyrelu', dropout=None, batch_norm=False)
        else:
            layers += self._fc(n_inputs, hiddens[0], activation='leakyrelu', dropout=dropout, batch_norm=batch_norm)  # first layer
            for l in range(1, len(hiddens)):  # inner layers
                layers += self._fc(hiddens[l-1], hiddens[l], activation='leakyrelu', dropout=dropout, batch_norm=batch_norm)
            layers += self._fc(hiddens[-1], n_outputs, activation='relu' if output_relu else 'linear',
                               dropout=dropout if regularize_last_layer else None,
                               batch_norm=regularize_last_layer)  # last layer

        self.network = nn.Sequential(*layers)
        self = self.to(device)

    def _fc(self, n_inputs, n_outputs, activation='leakyrelu', dropout=None, batch_norm=True):
        layers = [nn.Linear(n_inputs, n_outputs)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        if activation != 'linear':
            layers.append(self._activation(activation))
        return layers
    
    def _activation(self, name='leakyrelu'):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    
    def forward(self, x):
        return self.network(x)