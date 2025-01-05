import torch
import torch.nn as nn

class AlphaPushNetwork(nn.Module):
    def __init__(self):
        super(AlphaPushNetwork, self).__init__()
        # output vector:
        #   [1]: value
        #   [26]: (policy) place a piece here
        #   [26*26]: (policy) move a piece from space to space
        #   [26*4]: (policy) from space A, push in direction B
        output_size = 1 + 26 + 26*26 + 26*4 # 807
        input_size = 160
        hidden_layers = [256, 104, 104, 104, 256]
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            *[
                item for i in range(len(hidden_layers) - 1)
                for item in (
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                    nn.ReLU()
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