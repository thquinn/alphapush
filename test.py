import torch
from mcts import MCTS
from network import NullNet
from pushfight import PFDirection, PFMove, PFPiece, PFState
from training import eval_match

def versus():
    old_net = NullNet() # torch.load('model_320K_v001.pt', weights_only=False).cpu().eval()
    new_net = torch.load('model_320K_v007.pt', weights_only=False).cpu().eval()
    eval_match(old_net, new_net, evals_per_position=512, verbose=True)

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
    net = torch.load('model_320K_v007.pt', weights_only=False).cpu().eval()

    # state = PFState()
    # state = PFState.construct('.....Bww..b...BWB..W....Wb', False, 1, 15) # mate in two
    # state = PFState.construct('..w.....WWB..Wb.bwB..B....', True, 2, 21) # lose in one (White)
    # state = PFState.construct('..b.....BBW..Bw.wbW..W....', False, 2, 21) # lose in one (Black)
    # state = PFState.construct('.W...WWw.w.....b...BbB...B', True, 2, -1)
    # state = PFState.construct('.W...WWw.w.....b...Bb...BB', True, 2, -1)
    # state = PFState.construct('......BBbB..b..w...Ww...WW', True, 2, 6)
    # state = PFState.construct('...Www....WW..Bb.B..b....B', True, 1, -1)
    state = PFState.construct('......w..wWWW.............', False, -1, -1)

    print(state)
    mcts = MCTS(state, net.forward(state.to_tensor()))
    for i in range(4096):
        mcts.select_and_expand()
        mcts.receive_network_output(net.forward(mcts.get_current_state_tensor()))
    mcts.advance_root(temperature=0, print_depth=3)

debug_state()