import torch
import torch.nn as nn

class NullNet:
    def forward(self, _):
        return None
    def eval(self):
        pass