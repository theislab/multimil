from torch import nn


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hiddens=[], output_relu=False, device='cpu'):
        super(MLP, self).__init__()

        # create network architecture
        layers = []
        if hiddens == []:  # no hidden layers
            layers.append(nn.Linear(n_inputs, n_outputs))
        else:
            layers.append(nn.Linear(n_inputs, hiddens[0]))  # first layer
            layers.append(nn.ReLU())
            for l in range(1, len(hiddens)):  # inner layers
                layers.append(nn.Linear(hiddens[l-1], hiddens[l]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hiddens[-1], n_outputs))  # last layer

        if output_relu:
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        self = self.to(device)

    def forward(self, x):
        return self.network(x)