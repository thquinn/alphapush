import math
import random
import torch
import torch.nn as nn
from mcts import MCTS
from network import NullNet
from pushfight import PFDirection, PFMove, PFPiece, PFState
from test_null_training import NullTrainingNetwork, NullTrainingNetworkV2, NullTrainingNetworkV3
from training import eval_match

def versus():
    # old_net = NullNet()
    old_net = torch.load('model_270K_v006.pt', weights_only=False).cpu().eval()
    new_net = torch.load('model_270K_v007.pt', weights_only=False).cpu().eval()
    eval_match(old_net, new_net, evals_per_position=2048, verbose=True)

@torch.no_grad()
def versus_human(model_name = '270K_v006', human_plays_white=True, evals_per_position=20000, temperature=0, state=PFState()):
    print(f'Starting human vs. AI game. Human is playing the {'white' if human_plays_white else 'black'} pieces. Loading {model_name}...')
    net = torch.load(f'model_{model_name}.pt', weights_only=False).cpu()
    illegal_move = False
    while state.winner == PFPiece.Empty:
        if not illegal_move:
            print(state)
        human_turn = human_plays_white == state.white_to_move
        if human_turn:
            move_string = input('Enter your move: ')
            move = PFMove.parse(move_string)
            move = move if move else f'"{move_string}"'
            if move not in state.moves:
                if not illegal_move:
                    print(f'Can\'t play {move}. Legal moves are {', '.join(map(str, state.moves))}.')
                illegal_move = True
                continue
            illegal_move = False
            state.move(move)
            print()
        else:
            root_output = net.forward(state.to_tensor())
            if root_output is not None:
                root_output = root_output.numpy()[0]
            mcts = MCTS(PFState.copy(state), root_output)
            moves = []
            while not human_turn and state.winner == PFPiece.Empty:
                print(f'{model_name} running {evals_per_position} evals...')
                mcts.run_with_net(net, evals_per_position, advance=False)
                move = mcts.advance_root(temperature=temperature, print_depth=1, top_n=3)
                state.move(move)
                human_turn = human_plays_white == state.white_to_move
                moves.append(move)
            print(f'{model_name} played {', '.join(map(str, moves))}.')
            print()
    print(state)
    print(f'Game over: {'human' if human_plays_white == state.white_to_move else 'AI'} wins!')

@torch.no_grad()
def debug_state():
    # net = NullNet()
    net = torch.load('model_270K_v006.pt', weights_only=False).cpu().eval()
    
    state = PFState()
    # state = PFState.construct('......W..WwwW.............', False, -1, -1) # 'blackstandard': example before black's first placement
    # state = PFState.construct('wWWw.......W..bB.BBb......', True, 2, -1) # 'firstwin': win-in-two that NullNet is missing @512, catching @2048. 270Kv0 catches it at 80!
    # state = PFState.construct('.....W.w...bWW.Bw.bB....B.', True, 2, 19) # 'antipolicy': win-in-three. Null spots it @24K 270Kv0-2 @32K, v3-6@24K
    # state = PFState.construct('.......W.wb.WB.B...wB.Wb..', False, 1, 22) # 'horizon5': 270Kv3-5 saves @100K, v6@64K
    # state = PFState.construct('...bwWBw....B..WW..B..b...', False, 2, 16) # 'goodenough': NullNet catches win-in-three @256K, 270Kv3-4 misses @200K+
    # state = PFState.construct('......Ww.W...bBbB.W.w.B...', True, 0, 16) # 'lockin': v3-6 dislikes the win-in-five until 4K
    # state = PFState.construct('......wWwW.WB.....BBb.b...', True, 2, 12) # 'squeeze': v3-6 saves @64K
    # state = PFState.construct('.w.b..Bw.WW..B..B..Wb.....', False, 2, 9) # 'unsurething': v4 misses the win-in-three @100K, v5 spots @64K, v6@32K
    state = PFState.construct('....Ww.w.WW.....B.B.......', False, -1, -1) # 'misplaced': v3-6 like the killer Place@17 @2K
    # state = PFState.construct('...b...Bw..BW..WW..wb.B...', True, 0, 11) # 'regression': v3 saves @512, v4-6@16K
    # state = PFState.construct('.w......wWWW.BBbb..B......', True, 2, -1) # v4-5 blunders to Black's win-in-three @64K, v6 spots it from 64K all the way to 0 (!!)
    # state = PFState.construct('.....WwWw..WbBBbB.........', True, 2, 16) # 'squeeze2': v6 saves @32K
    # state = PFState.construct('...W...B..Bw.b.Ww..bW.B...', True, 2, 7) # 'singularity': this is a win-in-seven, v6 misses @100K
    # state = PFState.construct('.......B.W..w.WWB.Bwb.b...', False, 0, 15) # 'lockin2': win-in-five, v6 needs 8K
    # state = PFState.construct('...wbwBW..B.b.W.B..W......', False, 1, 7) # 'cliffhanger': v5-6 can't see the winning Push@6.Up @64K
    state.move(PFMove.place(17))
    state.move(PFMove.place(14))
    state.move(PFMove.place(15))
    state.move(PFMove.move(9, 13))

    print(state)
    root_output = net.forward(state.to_tensor())
    if root_output is not None:
        root_output = root_output[0].numpy()
        print(f'Network value estimate: {root_output[0]:.2f}')
    mcts = MCTS(state, root_output)
    mcts.run_with_net(net, min_evals=2048, temperature=0, print_depth=3, top_n=3)

if __name__ == '__main__':
    debug_state()