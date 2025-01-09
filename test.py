import math
import random
import torch
import torch.nn as nn
from mcts import MCTS
from network import NullNet
from pushfight import PFDirection, PFMove, PFPiece, PFState
from test_null_training import NullTrainingNetwork
from training import eval_match

def versus():
    old_net = NullNet()
    new_net = torch.load('model_270K_v000.pt', weights_only=False).cpu()
    new_net.eval()
    eval_match(old_net, new_net, evals_per_position=2048, verbose=True)

def mate_in_three():
    net = torch.load('model_1M_v009.pt', weights_only=False).cpu().eval()

    state = PFState()
    state.move(PFMove.place(5))
    for i in range(9, 16):
        state.move(PFMove.place(i))
    state.move(PFMove.place(18))
    state.move(PFMove.place(25))
    print(state)

    net.eval()
    mcts = MCTS(state, net.forward(state.to_tensor()))
    for j in range(3):
        for i in range(4000): # best so far: 1-2K
            mcts.select_and_expand()
            mcts.receive_network_output(net.forward(mcts.get_current_state_tensor()))
        mcts.advance_root(temperature=0, print_depth=1)
        print(mcts.root.state)

def debug_state():
    # net = NullNet()
    net = torch.load('model_270K_v000.pt', weights_only=False).cpu().eval()

    # state = PFState()
    # state = PFState.construct('wWWw.......W..bB.BBb......', True, 2, -1) # win-in-two that NullNet is missing @512, catching @2048. 270Kv0 catches it at 512!
    state = PFState.construct('.....W.w...bWW.Bw.bB....B.', True, 2, 19) # tricky win-in-three

    print(state)
    mcts = MCTS(state, net.forward(state.to_tensor().unsqueeze(0)))
    for i in range(2048):
        mcts.select_and_expand()
        mcts.receive_network_output(net.forward(mcts.get_current_state_tensor().unsqueeze(0)))
    mcts.advance_root(temperature=0, print_depth=3)

if __name__ == '__main__':
    debug_state()