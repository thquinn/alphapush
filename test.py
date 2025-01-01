import torch
from mcts import MCTS
from pushfight import PFMove, PFPiece, PFState
from training import eval_match

def versus():
    old_net = torch.load('model_1700K_v003.pt', weights_only=False).cpu().eval()
    new_net = torch.load('model_1M_v009.pt', weights_only=False).cpu().eval()
    eval_match(old_net, new_net, evals_per_position=20000, verbose=True)

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
        for i in range(4000):
            mcts.select_and_expand()
            mcts.receive_network_output(net.forward(mcts.get_current_state_tensor()))
        mcts.advance_root(temperature=0, debug_print=True)
        print(mcts.root.state)

versus()