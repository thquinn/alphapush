import torch
import torch.nn as nn

class AlphaPushNetwork(nn.Module):
    def __init__(self):
        super(AlphaPushNetwork, self).__init__()
        # input vector: see PFState.to_tensor
        input_size = 160
        # output vector:
        #   [1]: value: 1 if the current player is winning, -1 if the current player is losing
        #   [26]: (policy) place a piece here
        #   [26*26]: (policy) move a piece from space to space
        #   [26*4]: (policy) from space A, push in direction B
        output_size = 1 + 26 + 26*26 + 26*4 # 807
        hidden_layers = [input_size, 256, 104, 104, 104, 256]
        self.model = nn.Sequential(
            *[
                item for i in range(len(hidden_layers) - 1)
                for item in (
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_layers[i + 1]),
                )
            ],
            nn.Linear(hidden_layers[-1], output_size),
        )
        self.value_activation = nn.Tanh()
    
    def forward(self, x):
        features = self.model(x)
        value = self.value_activation(features[0:1])
        policy = features[1:]
        return torch.cat((value, policy))

class NullNet:
    def forward(self, _):
        return None
    def eval(self):
        pass