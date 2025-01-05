import torch
from mcts import MCTS
from pushfight import PFDirection, PFMove, PFPiece, PFState
from training import eval_match

class NullNet:
    def forward(self, _):
        return torch.zeros(807)

def versus():
    old_net = torch.load('model_320K_v001.pt', weights_only=False).cpu().eval()
    new_net = torch.load('model_320K_bad.pt', weights_only=False).cpu().eval()
    eval_match(old_net, new_net, evals_per_position=1024, verbose=True)

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
        mcts.advance_root(temperature=0, debug_print=True)
        print(mcts.root.state)

def debug_state():
    net = torch.load('model_320K_bad.pt', weights_only=False).cpu().eval()
    #state = PFState.construct('.....Bww..b...BWB..W....Wb', False, 1, 15)
    #state = PFState.construct('..w.....WWB..Wb.bwB..B....', True, 2, 21)
    state = PFState.construct('..b.....BBW..Bw.wbW..W....', False, 2, 21)
    print(state)
    print(state.to_tensor())
    mcts = MCTS(state, net.forward(state.to_tensor()))
    for i in range(1024):
        mcts.select_and_expand()
        mcts.receive_network_output(net.forward(mcts.get_current_state_tensor()))
    mcts.advance_root(temperature=0, debug_print=True)

debug_state()