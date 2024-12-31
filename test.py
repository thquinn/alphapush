import glob
import random

import torch
from pushfight import PFPiece, PFState
from training import eval_match

old_net = torch.load('model_750K_v001.pt', weights_only=False).cpu().eval()
new_net = torch.load('model_1700K_v003.pt', weights_only=False).cpu().eval()

eval_match(old_net, new_net, evals_per_position=2048, verbose=True)